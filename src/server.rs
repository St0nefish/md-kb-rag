use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::Response,
};
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use subtle::ConstantTimeEq;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::config::Config;
use crate::embed::EmbedClient;
use crate::mcp::KbSearchServer;
use crate::qdrant::QdrantStore;
use crate::webhook::{self, WebhookState};

#[derive(Clone)]
struct HealthState {
    qdrant: Arc<QdrantStore>,
    embed: Arc<EmbedClient>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub qdrant: ComponentHealth,
    pub embeddings: ComponentHealth,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ComponentHealth {
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

async fn health_handler(State(state): State<HealthState>) -> (StatusCode, Json<HealthResponse>) {
    let (qdrant_result, embed_result) =
        tokio::join!(state.qdrant.health_check(), state.embed.health_check());

    let qdrant = match &qdrant_result {
        Ok(()) => ComponentHealth {
            status: "ok".into(),
            error: None,
        },
        Err(e) => {
            warn!("qdrant health check failed: {e:#}");
            ComponentHealth {
                status: "unavailable".into(),
                error: None,
            }
        }
    };

    let embeddings = match &embed_result {
        Ok(()) => ComponentHealth {
            status: "ok".into(),
            error: None,
        },
        Err(e) => {
            warn!("embeddings health check failed: {e:#}");
            ComponentHealth {
                status: "unavailable".into(),
                error: None,
            }
        }
    };

    let all_ok = qdrant_result.is_ok() && embed_result.is_ok();
    let status = if all_ok { "healthy" } else { "degraded" };
    let code = if all_ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (
        code,
        Json(HealthResponse {
            status: status.into(),
            qdrant,
            embeddings,
        }),
    )
}

#[derive(Clone)]
struct AuthState {
    bearer_token: Option<String>,
}

async fn bearer_auth(
    State(auth): State<AuthState>,
    headers: HeaderMap,
    request: axum::extract::Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let Some(ref expected_token) = auth.bearer_token else {
        return Ok(next.run(request).await);
    };

    let auth_header = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    let token = auth_header.strip_prefix("Bearer ").unwrap_or("");

    if token.as_bytes().ct_eq(expected_token.as_bytes()).into() {
        Ok(next.run(request).await)
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

pub async fn run_server(config: Config) -> Result<()> {
    let config = Arc::new(config);

    // Set up shared services
    let embed_client = Arc::new(EmbedClient::new(&config.embedding));
    let qdrant = Arc::new(QdrantStore::new(&config.qdrant).context("Failed to connect to Qdrant")?);

    // Ensure collection exists
    qdrant
        .ensure_collection(
            &config.qdrant.collection,
            config.embedding.vector_size,
            &config.effective_indexed_fields(),
        )
        .await
        .context("Failed to ensure Qdrant collection")?;

    // MCP service
    let collection = config.qdrant.collection.clone();
    let data_path = std::path::PathBuf::from(config.data_path());
    let include_patterns = config.indexing.include.clone();
    let embed_for_mcp = Arc::clone(&embed_client);
    let qdrant_for_mcp = Arc::clone(&qdrant);

    let ct = CancellationToken::new();

    let mcp_service = StreamableHttpService::new(
        move || {
            Ok(KbSearchServer::new(
                Arc::clone(&embed_for_mcp),
                Arc::clone(&qdrant_for_mcp),
                collection.clone(),
                data_path.clone(),
                &include_patterns,
            ))
        },
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig {
            cancellation_token: ct.child_token(),
            ..Default::default()
        },
    );

    // Bearer token for MCP auth
    let bearer_token = match std::env::var(&config.mcp.bearer_token_env) {
        Ok(val) if !val.is_empty() => Some(val),
        _ => {
            if !config.mcp.allow_unauthenticated {
                anyhow::bail!(
                    "Environment variable '{}' is not set or empty. \
                     Set it to a bearer token, or set mcp.allow_unauthenticated: true \
                     in config.yaml to explicitly opt out of authentication.",
                    config.mcp.bearer_token_env
                );
            }
            warn!(
                "Environment variable '{}' is not set or empty — MCP endpoints will have no auth \
                 (allow_unauthenticated is enabled)",
                config.mcp.bearer_token_env
            );
            None
        }
    };
    let auth_state = AuthState { bearer_token };

    // Webhook state — optional, skip if secret is unset/empty
    let webhook_secret = std::env::var(&config.webhook.secret_env)
        .ok()
        .filter(|s| !s.is_empty());

    // Build router
    let mcp_router = Router::new().nest_service("/mcp", mcp_service).route_layer(
        middleware::from_fn_with_state(auth_state.clone(), bearer_auth),
    );

    let health_state = HealthState {
        qdrant: Arc::clone(&qdrant),
        embed: Arc::clone(&embed_client),
    };

    let mut app = Router::new()
        .route(
            "/health",
            axum::routing::get(health_handler).with_state(health_state),
        )
        .merge(mcp_router);

    if let Some(secret) = webhook_secret {
        let webhook_state = WebhookState {
            config: Arc::clone(&config),
            secret,
        };
        let webhook_router = Router::new()
            .route(
                "/hooks/reindex",
                axum::routing::post(webhook::handle_webhook),
            )
            .layer(DefaultBodyLimit::max(1024 * 1024)) // 1 MB
            .with_state(webhook_state);
        app = app.merge(webhook_router);
        info!("  Webhook endpoint: /hooks/reindex");
    } else {
        warn!(
            "Environment variable '{}' is not set or empty — webhook endpoint disabled",
            config.webhook.secret_env
        );
    }

    let mcp_port = config.mcp.port;
    let bind_addr = format!("0.0.0.0:{}", mcp_port);
    info!("Starting server on {}", bind_addr);
    info!("  MCP endpoint: /mcp");

    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .context("Failed to bind server address")?;

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let mut sigterm =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("failed to register SIGTERM handler");
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {},
                _ = sigterm.recv() => {},
            }
            info!("Shutting down server");
            ct.cancel();
        })
        .await
        .context("Server error")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request, routing::get};
    use tower::ServiceExt;

    fn test_app(token: Option<String>) -> Router {
        let auth_state = AuthState {
            bearer_token: token,
        };
        Router::new()
            .route("/test", get(|| async { "ok" }))
            .route_layer(middleware::from_fn_with_state(auth_state, bearer_auth))
    }

    #[tokio::test]
    async fn no_auth_configured_allows_all() {
        let app = test_app(None);
        let req = Request::builder().uri("/test").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn valid_bearer_token_allowed() {
        let app = test_app(Some("secret-token".to_string()));
        let req = Request::builder()
            .uri("/test")
            .header("authorization", "Bearer secret-token")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn invalid_bearer_token_rejected() {
        let app = test_app(Some("secret-token".to_string()));
        let req = Request::builder()
            .uri("/test")
            .header("authorization", "Bearer wrong-token")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn missing_auth_header_rejected() {
        let app = test_app(Some("secret-token".to_string()));
        let req = Request::builder().uri("/test").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn malformed_auth_header_rejected() {
        let app = test_app(Some("secret-token".to_string()));
        let req = Request::builder()
            .uri("/test")
            .header("authorization", "Basic c2VjcmV0LXRva2Vu")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }
}
