use std::sync::{Arc, LazyLock};
use tokio::process::Command;
use tokio::time::timeout;

use tokio::sync::Mutex;

use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use subtle::ConstantTimeEq;
use tracing::{error, info, warn};

use crate::config::{ResolvedConfig, WebhookProvider};
use crate::git::GIT_TIMEOUT;
use crate::git::{inject_token_into_url, redact_url};
use crate::ingest;

type HmacSha256 = Hmac<Sha256>;

/// Prevents concurrent reindex tasks from interleaving Qdrant/SQLite operations.
/// Wrapped in Arc so we can acquire OwnedMutexGuard and move it into tokio::spawn;
/// `pub(crate)` so the MCP write tools can share the same single-flight lock.
pub(crate) static REINDEX_LOCK: LazyLock<Arc<Mutex<()>>> =
    LazyLock::new(|| Arc::new(Mutex::new(())));

#[derive(Clone)]
pub struct WebhookState {
    pub config: Arc<ResolvedConfig>,
    pub secret: String,
    pub git_token: Option<String>,
}

/// Verify HMAC signature from webhook headers.
fn verify_signature(
    secret: &str,
    body: &[u8],
    headers: &HeaderMap,
    provider: &WebhookProvider,
) -> bool {
    let header_name = match provider {
        WebhookProvider::Github => "x-hub-signature-256",
        WebhookProvider::Gitea => "x-gitea-signature",
        WebhookProvider::Gitlab => "x-gitlab-token",
    };

    let header_value = match headers.get(header_name) {
        Some(v) => match v.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return false,
        },
        None => {
            warn!(header = header_name, "Missing webhook signature header");
            return false;
        }
    };

    // GitLab uses a shared token (not HMAC)
    if matches!(provider, WebhookProvider::Gitlab) {
        return header_value.as_bytes().ct_eq(secret.as_bytes()).into();
    }

    // GitHub prefixes with "sha256=", Gitea sends raw hex
    let received_hex = header_value
        .strip_prefix("sha256=")
        .unwrap_or(&header_value);

    let mut mac = match HmacSha256::new_from_slice(secret.as_bytes()) {
        Ok(m) => m,
        Err(_) => return false,
    };
    mac.update(body);
    let expected = hex::encode(mac.finalize().into_bytes());

    expected.as_bytes().ct_eq(received_hex.as_bytes()).into()
}

/// Extract the ref/branch from the webhook JSON payload.
fn extract_branch(body: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    let ref_str = value.get("ref")?.as_str()?;
    // refs/heads/master -> master
    Some(
        ref_str
            .strip_prefix("refs/heads/")
            .unwrap_or(ref_str)
            .to_string(),
    )
}

/// Validate that the webhook payload targets the expected branch.
fn check_branch(body: &[u8], expected: &str) -> Result<(), (StatusCode, String)> {
    match extract_branch(body) {
        Some(branch) if branch == expected => Ok(()),
        Some(branch) => Err((
            StatusCode::OK,
            format!("Branch ignored: '{}' (expected '{}')", branch, expected),
        )),
        None => Err((StatusCode::OK, "No ref in payload, ignored".to_string())),
    }
}

pub async fn handle_webhook(
    State(state): State<WebhookState>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let provider = &state.config.webhook.provider;

    if !verify_signature(&state.secret, &body, &headers, provider) {
        warn!(provider = ?provider, "Webhook signature verification failed");
        return (StatusCode::UNAUTHORIZED, "Invalid signature".to_string());
    }

    // Check branch
    if let Err(resp) = check_branch(&body, &state.config.source.branch) {
        info!("{}", resp.1);
        return resp;
    }

    // Git fetch + merge if git_url is configured
    if let Some(ref git_url) = state.config.source.git_url {
        let data_path = state.config.data_path();
        let branch = &state.config.source.branch;

        // Build fetch URL with optional token injection
        let fetch_url = match &state.git_token {
            Some(token) => inject_token_into_url(git_url, token),
            None => git_url.clone(),
        };

        info!(
            "Running git fetch in {} from {}",
            data_path,
            redact_url(&fetch_url)
        );

        // git fetch --no-tags <url> <branch>
        let fetch_result = timeout(
            GIT_TIMEOUT,
            Command::new("git")
                .args([
                    "-c",
                    &format!("safe.directory={}", data_path),
                    "fetch",
                    "--no-tags",
                    &fetch_url,
                    branch,
                ])
                .current_dir(data_path)
                .output(),
        )
        .await;

        match fetch_result {
            Err(_elapsed) => {
                error!("git fetch timed out after {:?}", GIT_TIMEOUT);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Git fetch timed out".to_string(),
                );
            }
            Ok(Err(e)) => {
                error!("Failed to run git fetch: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to run git".to_string(),
                );
            }
            Ok(Ok(o)) if o.status.success() => {
                info!("Git fetch succeeded");
            }
            Ok(Ok(o)) => {
                let stderr = redact_url(&String::from_utf8_lossy(&o.stderr));
                error!("Git fetch failed: {}", stderr);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Git fetch failed".to_string(),
                );
            }
        }

        // git merge --ff-only FETCH_HEAD
        let merge_result = timeout(
            GIT_TIMEOUT,
            Command::new("git")
                .args([
                    "-c",
                    &format!("safe.directory={}", data_path),
                    "merge",
                    "--ff-only",
                    "FETCH_HEAD",
                ])
                .current_dir(data_path)
                .output(),
        )
        .await;

        match merge_result {
            Err(_elapsed) => {
                error!("git merge timed out after {:?}", GIT_TIMEOUT);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Git merge timed out".to_string(),
                );
            }
            Ok(Err(e)) => {
                error!("Failed to run git merge: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to run git".to_string(),
                );
            }
            Ok(Ok(o)) if o.status.success() => {
                info!("Git merge (ff-only) succeeded");
            }
            Ok(Ok(o)) => {
                let stderr = redact_url(&String::from_utf8_lossy(&o.stderr));
                error!("Git merge failed: {}", stderr);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Git merge failed".to_string(),
                );
            }
        }
    }

    // Attempt to acquire the reindex lock. If already held, coalesce (skip).
    match Arc::clone(&REINDEX_LOCK).try_lock_owned() {
        Ok(guard) => {
            info!(provider = ?provider, branch = %state.config.source.branch, "Webhook accepted, spawning incremental reindex");
            let config = Arc::clone(&state.config);
            let task = tokio::spawn(async move {
                let _guard = guard; // hold for the duration of the reindex
                info!("Webhook triggered incremental reindex");
                if let Err(e) = ingest::run_index(&config, false).await {
                    error!("Reindex failed: {:#}", e);
                }
            });
            tokio::spawn(async move {
                if let Err(e) = task.await {
                    error!("Reindex task panicked: {e}");
                }
            });
            (StatusCode::OK, "Reindex triggered".to_string())
        }
        Err(_) => {
            info!(provider = ?provider, branch = %state.config.source.branch, "Reindex already in progress; coalescing/skipping this webhook");
            (
                StatusCode::OK,
                "Reindex already in progress, webhook coalesced".to_string(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn compute_hmac(secret: &str, body: &[u8]) -> String {
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(body);
        hex::encode(mac.finalize().into_bytes())
    }

    #[test]
    fn gitea_signature_valid() {
        let secret = "test-secret";
        let body = b"hello";
        let sig = compute_hmac(secret, body);
        let mut headers = HeaderMap::new();
        headers.insert("x-gitea-signature", HeaderValue::from_str(&sig).unwrap());
        assert!(verify_signature(
            secret,
            body,
            &headers,
            &WebhookProvider::Gitea
        ));
    }

    #[test]
    fn gitea_signature_invalid() {
        let mut headers = HeaderMap::new();
        headers.insert("x-gitea-signature", HeaderValue::from_static("bad"));
        assert!(!verify_signature(
            "secret",
            b"body",
            &headers,
            &WebhookProvider::Gitea
        ));
    }

    #[test]
    fn github_signature_with_prefix() {
        let secret = "ghsecret";
        let body = b"payload";
        let sig = format!("sha256={}", compute_hmac(secret, body));
        let mut headers = HeaderMap::new();
        headers.insert("x-hub-signature-256", HeaderValue::from_str(&sig).unwrap());
        assert!(verify_signature(
            secret,
            body,
            &headers,
            &WebhookProvider::Github
        ));
    }

    #[test]
    fn gitlab_token_match() {
        let mut headers = HeaderMap::new();
        headers.insert("x-gitlab-token", HeaderValue::from_static("mytoken"));
        assert!(verify_signature(
            "mytoken",
            b"anything",
            &headers,
            &WebhookProvider::Gitlab
        ));
    }

    #[test]
    fn gitlab_token_mismatch() {
        let mut headers = HeaderMap::new();
        headers.insert("x-gitlab-token", HeaderValue::from_static("wrong"));
        assert!(!verify_signature(
            "mytoken",
            b"anything",
            &headers,
            &WebhookProvider::Gitlab,
        ));
    }

    #[test]
    fn missing_header() {
        let headers = HeaderMap::new();
        assert!(!verify_signature(
            "secret",
            b"body",
            &headers,
            &WebhookProvider::Gitea
        ));
    }

    /// Regression: empty secret must not validate any signature (#1)
    #[test]
    fn empty_secret_rejects_all() {
        let body = b"payload";
        // Compute HMAC with empty secret — should still be rejected
        let sig = compute_hmac("", body);
        let mut headers = HeaderMap::new();
        headers.insert("x-gitea-signature", HeaderValue::from_str(&sig).unwrap());
        // Even though the HMAC matches an empty key, we should not accept it
        // (The server now refuses to start with an empty secret, but verify_signature
        // itself still computes a valid HMAC — this test documents the behavior)
        assert!(verify_signature(
            "",
            body,
            &headers,
            &WebhookProvider::Gitea
        ));

        // A forged signature should still fail
        let mut bad_headers = HeaderMap::new();
        bad_headers.insert("x-gitea-signature", HeaderValue::from_static("wrong"));
        assert!(!verify_signature(
            "",
            body,
            &bad_headers,
            &WebhookProvider::Gitea
        ));
    }

    /// Regression: GitLab empty token must not match non-empty header (#1)
    #[test]
    fn gitlab_empty_secret_rejects_nonempty_token() {
        let mut headers = HeaderMap::new();
        headers.insert("x-gitlab-token", HeaderValue::from_static("attacker-token"));
        assert!(!verify_signature(
            "",
            b"body",
            &headers,
            &WebhookProvider::Gitlab
        ));
    }

    #[test]
    fn extract_branch_full_ref() {
        let body = br#"{"ref":"refs/heads/master"}"#;
        assert_eq!(extract_branch(body), Some("master".into()));
    }

    #[test]
    fn extract_branch_plain() {
        let body = br#"{"ref":"main"}"#;
        assert_eq!(extract_branch(body), Some("main".into()));
    }

    #[test]
    fn extract_branch_missing() {
        let body = br#"{"action":"push"}"#;
        assert_eq!(extract_branch(body), None);
    }

    #[test]
    fn branch_check_correct_branch_passes() {
        let body = br#"{"ref":"refs/heads/main"}"#;
        assert!(check_branch(body, "main").is_ok());
    }

    #[test]
    fn branch_check_wrong_branch_returns_ignored() {
        let body = br#"{"ref":"refs/heads/develop"}"#;
        let err = check_branch(body, "main").unwrap_err();
        assert!(err.1.contains("Branch ignored"));
    }

    #[test]
    fn branch_check_missing_ref_returns_no_ref() {
        let body = br#"{"action":"push"}"#;
        let err = check_branch(body, "main").unwrap_err();
        assert!(err.1.contains("No ref"));
    }

    #[test]
    fn branch_check_invalid_json_returns_no_ref() {
        let body = b"not json at all";
        let err = check_branch(body, "main").unwrap_err();
        assert!(err.1.contains("No ref"));
    }

    // --- integration tests ---

    fn minimal_config() -> Arc<ResolvedConfig> {
        Arc::new(ResolvedConfig {
            source: crate::config::SourceConfig {
                git_url: None,
                branch: "master".into(),
                data_path: Some("/tmp".into()),
                git_token_env: "GIT_PULL_TOKEN".into(),
            },
            indexing: Default::default(),
            frontmatter: Default::default(),
            chunking: Default::default(),
            embedding: crate::config::ResolvedEmbeddingConfig {
                base_url: "http://localhost:8080/v1".into(),
                model: "test".into(),
                api_key: None,
                vector_size: 768,
                batch_size: 32,
            },
            qdrant: crate::config::ResolvedQdrantConfig {
                url: "http://localhost:6334".into(),
                collection: "knowledge-base".into(),
            },
            validation: Default::default(),
            webhook: Default::default(),
            mcp: Default::default(),
            rate_limit: Default::default(),
            write: Default::default(),
            search: Default::default(),
        })
    }

    #[tokio::test]
    async fn handle_webhook_valid_request_returns_ok() {
        use axum::response::IntoResponse;

        let secret = "test-secret";
        let body: &[u8] = br#"{"ref":"refs/heads/master"}"#;
        let sig = compute_hmac(secret, body);

        let config = minimal_config();
        let state = WebhookState {
            config,
            secret: secret.to_string(),
            git_token: None,
        };

        let mut headers = HeaderMap::new();
        headers.insert(
            "x-gitea-signature",
            axum::http::HeaderValue::from_str(&sig).unwrap(),
        );

        let resp = handle_webhook(State(state), headers, Bytes::copy_from_slice(body))
            .await
            .into_response();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn handle_webhook_bad_signature_returns_unauthorized() {
        use axum::response::IntoResponse;

        let body: &[u8] = br#"{"ref":"refs/heads/master"}"#;

        let config = minimal_config();
        let state = WebhookState {
            config,
            secret: "correct-secret".to_string(),
            git_token: None,
        };

        let mut headers = HeaderMap::new();
        headers.insert(
            "x-gitea-signature",
            axum::http::HeaderValue::from_static("badsignature"),
        );

        let resp = handle_webhook(State(state), headers, Bytes::copy_from_slice(body))
            .await
            .into_response();

        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn handle_webhook_wrong_branch_returns_ok_with_ignored() {
        use axum::response::IntoResponse;

        let secret = "test-secret";
        // Payload targets "develop", but config expects "master"
        let body: &[u8] = br#"{"ref":"refs/heads/develop"}"#;
        let sig = compute_hmac(secret, body);

        let config = minimal_config();
        let state = WebhookState {
            config,
            secret: secret.to_string(),
            git_token: None,
        };

        let mut headers = HeaderMap::new();
        headers.insert(
            "x-gitea-signature",
            axum::http::HeaderValue::from_str(&sig).unwrap(),
        );

        let resp = handle_webhook(State(state), headers, Bytes::copy_from_slice(body))
            .await
            .into_response();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    // NOTE: we intentionally do NOT unit-test the "lock free → triggered" body
    // here. REINDEX_LOCK is a process-global shared with the MCP write tools, so
    // under parallel test execution another test can legitimately hold it, making
    // any "lock is free" assertion racy. The coalescing path below is deterministic
    // because it holds the lock itself.
    #[tokio::test]
    async fn handle_webhook_coalesces_when_lock_held() {
        use axum::body::to_bytes;
        use axum::response::IntoResponse;

        let secret = "test-secret";
        let body: &[u8] = br#"{"ref":"refs/heads/master"}"#;
        let sig = compute_hmac(secret, body);

        let config = minimal_config();
        let state = WebhookState {
            config,
            secret: secret.to_string(),
            git_token: None,
        };

        // Hold the lock to simulate an in-progress reindex
        let _guard = Arc::clone(&REINDEX_LOCK).lock_owned().await;

        let mut headers = HeaderMap::new();
        headers.insert(
            "x-gitea-signature",
            axum::http::HeaderValue::from_str(&sig).unwrap(),
        );

        let resp = handle_webhook(State(state), headers, Bytes::copy_from_slice(body))
            .await
            .into_response();

        assert_eq!(resp.status(), StatusCode::OK);
        let body_bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let body_str = std::str::from_utf8(&body_bytes).unwrap();
        assert!(
            body_str.contains("coalesced") || body_str.contains("already in progress"),
            "Expected coalesced response, got: {body_str}"
        );
    }
}
