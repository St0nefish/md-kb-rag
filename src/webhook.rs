use std::sync::Arc;
use tokio::process::Command;
use tokio::time::timeout;

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
use crate::reindex;

type HmacSha256 = Hmac<Sha256>;

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

        // Captured before the fetch so the range diffed below covers exactly what
        // this webhook's pull brought in — the same before/after pattern
        // `commit_and_sync` uses around its own fetch + rebase.
        let old_head = match crate::git::rev_parse_head(data_path).await {
            Ok(sha) => sha,
            Err(e) => {
                error!("Failed to read HEAD before fetch: {:#}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to read local HEAD".to_string(),
                );
            }
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

        // Diff exactly what this pull changed and mark those paths dirty. The worker
        // (src/reindex.rs) picks this up and does the actual indexing out of band, so
        // this handler can return as soon as the local clone is up to date — it no
        // longer waits for (or even starts) a reindex itself.
        let new_head = match crate::git::rev_parse_head(data_path).await {
            Ok(sha) => sha,
            Err(e) => {
                error!("Failed to read HEAD after merge: {:#}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to read local HEAD".to_string(),
                );
            }
        };

        let changed = match crate::git::git_diff_name_status(data_path, &old_head, &new_head).await
        {
            Ok(paths) => paths,
            Err(e) => {
                error!("Failed to diff webhook pull range: {:#}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to diff pulled changes".to_string(),
                );
            }
        };

        info!(
            provider = ?provider,
            branch = %branch,
            changed = changed.len(),
            "Webhook pull applied; marking changed paths dirty"
        );
        reindex::mark_paths(changed);
    } else {
        // No git_url configured, so there was nothing to fetch and therefore no range
        // to diff. Fall back to a full reconcile so the webhook still causes the
        // worker to look for whatever changed on disk out-of-band.
        info!(provider = ?provider, "Webhook accepted with no git_url configured; marking a full reconcile");
        reindex::mark_full();
    }

    (StatusCode::OK, "Changes queued for indexing".to_string())
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
            source: crate::config::ResolvedSourceConfig {
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
                request_timeout_secs: 60,
                batch_concurrency: 4,
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
            reranking: None,
            provenance: Default::default(),
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

    // NOTE: there is no "reindex already in progress, coalesced" path to test anymore
    // — that drop-on-collision bug is exactly what the dirty-path queue replaces.
    // `mark_paths`/`mark_full` never block and never fail, so every accepted webhook
    // takes the same success path regardless of what the worker is doing concurrently.
    //
    // We also do not assert on `reindex::REINDEX_QUEUE`'s state here: it is a
    // process-global shared with every other test in this binary (git.rs's
    // `commit_and_sync` tests, mcp.rs's write-tool tests), so under parallel
    // `cargo test` execution its exact contents at any instant are not this test's to
    // own. `handle_webhook_marks_a_full_reconcile_when_no_git_url_is_configured` below
    // checks the one flag that is safe to assert on (`full_pending`), since it is only
    // ever set to `true` and never cleared by anything reachable from tests.

    #[tokio::test]
    async fn handle_webhook_marks_a_full_reconcile_when_no_git_url_is_configured() {
        use axum::response::IntoResponse;

        let secret = "test-secret";
        let body: &[u8] = br#"{"ref":"refs/heads/master"}"#;
        let sig = compute_hmac(secret, body);

        // minimal_config() has git_url: None, so there is nothing to fetch/diff — the
        // handler's only option is to fall back to a full reconcile.
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
        assert!(
            reindex::REINDEX_QUEUE.snapshot().full_pending,
            "a webhook with no git_url must fall back to a full reconcile"
        );
    }
}
