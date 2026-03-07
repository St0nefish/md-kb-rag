use std::process::Command;
use std::sync::Arc;

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

use crate::config::Config;
use crate::ingest;

type HmacSha256 = Hmac<Sha256>;

#[derive(Clone)]
pub struct WebhookState {
    pub config: Arc<Config>,
    pub secret: String,
}

/// Verify HMAC signature from webhook headers.
fn verify_signature(secret: &str, body: &[u8], headers: &HeaderMap, provider: &str) -> bool {
    let header_name = match provider {
        "github" => "x-hub-signature-256",
        "gitea" => "x-gitea-signature",
        "gitlab" => "x-gitlab-token",
        _ => {
            warn!("Unknown webhook provider: {}", provider);
            return false;
        }
    };

    let header_value = match headers.get(header_name) {
        Some(v) => match v.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return false,
        },
        None => {
            warn!("Missing webhook signature header: {}", header_name);
            return false;
        }
    };

    // GitLab uses a shared token (not HMAC)
    if provider == "gitlab" {
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

pub async fn handle_webhook(
    State(state): State<WebhookState>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let provider = &state.config.webhook.provider;

    if !verify_signature(&state.secret, &body, &headers, provider) {
        warn!("Webhook signature verification failed");
        return (StatusCode::UNAUTHORIZED, "Invalid signature".to_string());
    }

    // Check branch
    if let Some(branch) = extract_branch(&body) {
        if branch != state.config.source.branch {
            info!(
                "Ignoring webhook for branch '{}' (expected '{}')",
                branch, state.config.source.branch
            );
            return (StatusCode::OK, "Branch ignored".to_string());
        }
    }

    // Git pull if git_url is configured
    if state.config.source.git_url.is_some() {
        let data_path = state.config.data_path();
        info!("Running git pull in {}", data_path);
        let output = Command::new("git")
            .args(["pull", "--ff-only"])
            .current_dir(data_path)
            .output();

        match output {
            Ok(o) if o.status.success() => {
                info!("Git pull succeeded");
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                error!("Git pull failed: {}", stderr);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Git pull failed".to_string(),
                );
            }
            Err(e) => {
                error!("Failed to run git: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to run git".to_string(),
                );
            }
        }
    }

    // Trigger incremental reindex
    let config = Arc::clone(&state.config);
    tokio::spawn(async move {
        info!("Webhook triggered incremental reindex");
        if let Err(e) = ingest::run_index(&config, false).await {
            error!("Reindex failed: {:#}", e);
        }
    });

    (StatusCode::OK, "Reindex triggered".to_string())
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
        assert!(verify_signature(secret, body, &headers, "gitea"));
    }

    #[test]
    fn gitea_signature_invalid() {
        let mut headers = HeaderMap::new();
        headers.insert("x-gitea-signature", HeaderValue::from_static("bad"));
        assert!(!verify_signature("secret", b"body", &headers, "gitea"));
    }

    #[test]
    fn github_signature_with_prefix() {
        let secret = "ghsecret";
        let body = b"payload";
        let sig = format!("sha256={}", compute_hmac(secret, body));
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-hub-signature-256",
            HeaderValue::from_str(&sig).unwrap(),
        );
        assert!(verify_signature(secret, body, &headers, "github"));
    }

    #[test]
    fn gitlab_token_match() {
        let mut headers = HeaderMap::new();
        headers.insert("x-gitlab-token", HeaderValue::from_static("mytoken"));
        assert!(verify_signature("mytoken", b"anything", &headers, "gitlab"));
    }

    #[test]
    fn gitlab_token_mismatch() {
        let mut headers = HeaderMap::new();
        headers.insert("x-gitlab-token", HeaderValue::from_static("wrong"));
        assert!(!verify_signature("mytoken", b"anything", &headers, "gitlab"));
    }

    #[test]
    fn missing_header() {
        let headers = HeaderMap::new();
        assert!(!verify_signature("secret", b"body", &headers, "gitea"));
    }

    #[test]
    fn unknown_provider() {
        let headers = HeaderMap::new();
        assert!(!verify_signature("secret", b"body", &headers, "bitbucket"));
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
}
