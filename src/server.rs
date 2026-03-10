use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

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
use std::sync::RwLock;
use subtle::ConstantTimeEq;
use tokio_util::sync::CancellationToken;
use tower_governor::{
    GovernorLayer, governor::GovernorConfigBuilder, key_extractor::SmartIpKeyExtractor,
};
use tracing::{debug, info, warn};

use crate::config::ResolvedConfig;
use crate::embed::EmbedClient;
use crate::mcp::{self, KbSearchServer};
use crate::qdrant::QdrantStore;
use crate::webhook::{self, WebhookState};

#[derive(Clone)]
struct HealthState {
    qdrant: Arc<QdrantStore>,
    embed: Arc<EmbedClient>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OverallStatus {
    Healthy,
    Degraded,
}

impl std::fmt::Display for OverallStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded => write!(f, "degraded"),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComponentStatus {
    Ok,
    Unavailable,
}

impl std::fmt::Display for ComponentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ok => write!(f, "ok"),
            Self::Unavailable => write!(f, "unavailable"),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct HealthResponse {
    pub status: OverallStatus,
    pub qdrant: ComponentHealth,
    pub embeddings: ComponentHealth,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ComponentHealth {
    pub status: ComponentStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

async fn health_handler(State(state): State<HealthState>) -> (StatusCode, Json<HealthResponse>) {
    let (qdrant_result, embed_result) =
        tokio::join!(state.qdrant.health_check(), state.embed.health_check());

    let qdrant = match &qdrant_result {
        Ok(()) => ComponentHealth {
            status: ComponentStatus::Ok,
            error: None,
        },
        Err(e) => {
            warn!("qdrant health check failed: {e:#}");
            ComponentHealth {
                status: ComponentStatus::Unavailable,
                error: None,
            }
        }
    };

    let embeddings = match &embed_result {
        Ok(()) => ComponentHealth {
            status: ComponentStatus::Ok,
            error: None,
        },
        Err(e) => {
            warn!("embeddings health check failed: {e:#}");
            ComponentHealth {
                status: ComponentStatus::Unavailable,
                error: None,
            }
        }
    };

    let all_ok = qdrant_result.is_ok() && embed_result.is_ok();
    let status = if all_ok {
        OverallStatus::Healthy
    } else {
        OverallStatus::Degraded
    };
    let code = if all_ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (
        code,
        Json(HealthResponse {
            status,
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

/// Build MCP server instructions by combining config narrative with
/// dynamically discovered filter values from Qdrant.
async fn build_instructions(
    base: &str,
    qdrant: &QdrantStore,
    collection: &str,
    indexed_fields: &[String],
) -> String {
    let mut instructions = base.to_string();

    for field in indexed_fields {
        if field == "file_path" {
            continue;
        }
        match qdrant.fetch_facet_values(collection, field, 50).await {
            Ok(values) if !values.is_empty() => {
                instructions.push_str(&format!("\nAvailable {field}: {}", values.join(", ")));
            }
            _ => {}
        }
    }

    instructions
}

pub async fn run_server(config: ResolvedConfig) -> Result<()> {
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

    // Build dynamic MCP instructions
    let base_instructions = config
        .mcp
        .instructions
        .as_deref()
        .unwrap_or(mcp::DEFAULT_INSTRUCTIONS);
    let indexed_fields = config.effective_indexed_fields();
    let initial_instructions = build_instructions(
        base_instructions,
        &qdrant,
        &config.qdrant.collection,
        &indexed_fields,
    )
    .await;
    let shared_instructions = Arc::new(RwLock::new(initial_instructions));

    // Spawn metadata refresh task
    let refresh_instructions = Arc::clone(&shared_instructions);
    let refresh_qdrant = Arc::clone(&qdrant);
    let refresh_collection = config.qdrant.collection.clone();
    let refresh_base = base_instructions.to_string();
    let refresh_fields = indexed_fields.clone();
    let refresh_secs = config.mcp.metadata_refresh_secs;

    let ct = CancellationToken::new();
    let refresh_ct = ct.child_token();

    tokio::spawn(async move {
        loop {
            tokio::select! {
                () = tokio::time::sleep(Duration::from_secs(refresh_secs)) => {}
                () = refresh_ct.cancelled() => {
                    break;
                }
            }
            let updated = build_instructions(
                &refresh_base,
                &refresh_qdrant,
                &refresh_collection,
                &refresh_fields,
            )
            .await;
            *refresh_instructions.write().unwrap() = updated;
            debug!("Refreshed MCP instructions metadata");
        }
    });

    // MCP service
    let collection = config.qdrant.collection.clone();
    let data_path = std::path::PathBuf::from(config.data_path());
    let include_patterns = config.indexing.include.clone();
    let embed_for_mcp = Arc::clone(&embed_client);
    let qdrant_for_mcp = Arc::clone(&qdrant);

    let mcp_service = StreamableHttpService::new(
        move || {
            KbSearchServer::new(
                Arc::clone(&embed_for_mcp),
                Arc::clone(&qdrant_for_mcp),
                collection.clone(),
                data_path.clone(),
                &include_patterns,
                Arc::clone(&shared_instructions),
            )
            .map_err(std::io::Error::other)
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

    // Rate limiting (per-IP via SmartIpKeyExtractor for proxy-aware extraction)
    let governor_conf = Arc::new(
        GovernorConfigBuilder::default()
            .per_second(config.rate_limit.per_second)
            .burst_size(config.rate_limit.burst_size)
            .key_extractor(SmartIpKeyExtractor)
            .use_headers()
            .finish()
            .unwrap(),
    );

    let governor_limiter = governor_conf.limiter().clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        loop {
            interval.tick().await;
            governor_limiter.retain_recent();
        }
    });

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
        .merge(mcp_router)
        .layer(GovernorLayer::new(Arc::clone(&governor_conf)));

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

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        let sigterm_result =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate());
        match sigterm_result {
            Ok(mut sigterm) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {},
                    _ = sigterm.recv() => {},
                }
            }
            Err(e) => {
                warn!("Failed to register SIGTERM handler: {e}, falling back to ctrl-c only");
                let _ = tokio::signal::ctrl_c().await;
            }
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

    fn rate_limited_app(burst_size: u32) -> Router {
        let governor_conf = Arc::new(
            GovernorConfigBuilder::default()
                .per_second(1)
                .burst_size(burst_size)
                .key_extractor(SmartIpKeyExtractor)
                .finish()
                .unwrap(),
        );
        Router::new()
            .route("/test", get(|| async { "ok" }))
            .layer(GovernorLayer::new(governor_conf))
    }

    #[tokio::test]
    async fn rate_limit_allows_burst() {
        let app = rate_limited_app(3);
        for _ in 0..3 {
            let req = Request::builder()
                .uri("/test")
                .header("x-forwarded-for", "1.2.3.4")
                .body(Body::empty())
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }
    }

    #[tokio::test]
    async fn rate_limit_rejects_over_burst() {
        let app = rate_limited_app(2);
        // Exhaust the burst
        for _ in 0..2 {
            let req = Request::builder()
                .uri("/test")
                .header("x-forwarded-for", "5.6.7.8")
                .body(Body::empty())
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }
        // Next request should be rate limited (429)
        let req = Request::builder()
            .uri("/test")
            .header("x-forwarded-for", "5.6.7.8")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn rate_limit_is_per_ip() {
        let app = rate_limited_app(1);
        // First IP uses its burst
        let req = Request::builder()
            .uri("/test")
            .header("x-forwarded-for", "10.0.0.1")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // First IP is now limited
        let req = Request::builder()
            .uri("/test")
            .header("x-forwarded-for", "10.0.0.1")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);

        // Second IP still has its own burst
        let req = Request::builder()
            .uri("/test")
            .header("x-forwarded-for", "10.0.0.2")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
