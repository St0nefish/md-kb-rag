//! The knowledge-base web UI: a Cytoscape.js graph browser served straight from this
//! binary (`assets/ui/`, embedded via `include_str!` — no filesystem reads at
//! request time, no external network fetches, no new server dependency).
//!
//! Deliberately unauthenticated (see the feature plan): every route in
//! [`ui_router`] is mounted with no bearer-auth layer, the same posture `/health`
//! already has. The deployment sits behind Authentik via Traefik; this binary does
//! not gate these routes itself.
//!
//! [`UiState`] mirrors `server::StatusState` and `mcp::KbSearchServer`'s own
//! fields/`deps()` helper — same retrieval plumbing (`retrieval::search`,
//! `retrieval::get_document`), reused rather than duplicated, just wired to axum
//! instead of rmcp.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path as AxumPath, Query, State},
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::get,
};
use globset::{Glob, GlobSet, GlobSetBuilder};
use serde::{Deserialize, Serialize};
use tracing::error;

use crate::config::{ResolvedConfig, SharedConfig};
use crate::embed::{EmbedClient, QueryEmbedder};
use crate::qdrant::{QdrantStore, RetrievalStore, SearchResult};
use crate::rerank::RerankClient;
use crate::retrieval::{self, RetrievalDeps, SearchFilters, SearchOptions};
use crate::schema::{self, SharedSchemaCache};
use crate::state::{DocumentSummary, StateDb};
use crate::write::{self, WriteDeps, WriteError, WriteOutcome, WriteRequest, WriteSuccess};

/// Cap on a search query's length — the same value `mcp::MAX_QUERY_LEN` uses,
/// redeclared here because that constant is private to `mcp.rs` and this route is
/// reachable with no authentication at all.
const MAX_SEARCH_QUERY_LEN: usize = 4096;
/// Cap on a single `domain`/`type` filter value's length, mirroring
/// `mcp::MAX_FILTER_STR_LEN` (also private to that module).
const MAX_SEARCH_FILTER_LEN: usize = 256;
/// Cap on how many comma-separated tags a single search request may filter on.
const MAX_SEARCH_TAGS: usize = 20;
/// Cap on the `content` field of a `POST /api/doc/{*path}` body, mirroring
/// `mcp::MAX_CONTENT_LEN` (private to that module) — this is a content-shape
/// guard distinct from the router's 10 MB `DefaultBodyLimit`, which only bounds
/// the raw request body.
const MAX_WRITE_CONTENT_LEN: usize = 512 * 1024; // 512 KB
/// The `DefaultBodyLimit` applied to the `/api/doc/{*path}` write routes, matching
/// `mcp_router`'s limit in `server.rs` (see the API contract's fixed spec).
const MAX_WRITE_BODY_BYTES: usize = 10 * 1024 * 1024; // 10 MB
/// Cap on fields echoed back per scope by `/api/schema`, mirroring
/// `mcp::MAX_REPORTED_FIELDS` (private to that module) — schema files sync from a
/// git remote, so field count is attacker-controlled.
const MAX_SCHEMA_FIELDS: usize = 500;
/// Cap on permitted values echoed back per field, mirroring
/// `mcp::MAX_REPORTED_VALUES`.
const MAX_SCHEMA_VALUES: usize = 200;
/// Fixed cytoscape node diameter. The API contract only specifies a constant value;
/// a computed sizing scheme (e.g. by backlink count) can be added later without
/// changing the wire shape.
const NODE_SIZE: u32 = 30;
/// Fallback node color for a document with no `type` frontmatter.
const DEFAULT_NODE_COLOR: &str = "#94a3b8";

/// Shared state for every route in [`ui_router`].
///
/// Fields mirror `server::StatusState` and `mcp::KbSearchServer`: a live
/// [`SharedConfig`] handle (fetched fresh per request via [`UiState::config`], so a
/// `POST /admin/reload` is observed immediately, same as the MCP tools), concrete
/// `QdrantStore`/`EmbedClient` handles (not trait objects — the retrieval functions
/// this delegates to are generic over the trait, so tests exercise that generic
/// core directly with fakes rather than needing a mockable `UiState`), and a
/// lazily-opened state DB handle.
#[derive(Clone)]
pub struct UiState {
    config: SharedConfig,
    qdrant: Arc<QdrantStore>,
    embed_client: Arc<EmbedClient>,
    collection: String,
    canonical_data_path: PathBuf,
    include_patterns: Arc<GlobSet>,
    rerank_client: Option<Arc<RerankClient>>,
    schema_cache: SharedSchemaCache,
    /// Opened on first use. Callers MUST pass the same `Arc` instance already used
    /// for `server::StatusState` in `run_server` — a second lazily-opened pool onto
    /// the same SQLite file is pointless duplication, not extra safety.
    state_db: Arc<tokio::sync::OnceCell<StateDb>>,
}

/// Build an include `GlobSet` for the UI's document-serving routes.
///
/// Mirrors `mcp::build_include_globset` (private to that module, hence this small,
/// deliberate duplication rather than a shared helper) — same `**/*.md` fallback
/// when no patterns are valid.
fn build_include_globset(patterns: &[String]) -> GlobSet {
    let (mut builder, valid_count) = crate::ingest::parse_globs(patterns);
    if valid_count == 0 {
        builder.add(Glob::new("**/*.md").unwrap());
    }
    builder.build().unwrap_or_else(|_| {
        let mut fallback = GlobSetBuilder::new();
        fallback.add(Glob::new("**/*.md").unwrap());
        fallback
            .build()
            .expect("hardcoded fallback glob '**/*.md' must compile")
    })
}

impl UiState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: SharedConfig,
        qdrant: Arc<QdrantStore>,
        embed_client: Arc<EmbedClient>,
        collection: String,
        canonical_data_path: PathBuf,
        include_patterns: &[String],
        rerank_client: Option<Arc<RerankClient>>,
        schema_cache: SharedSchemaCache,
        state_db: Arc<tokio::sync::OnceCell<StateDb>>,
    ) -> Self {
        Self {
            config,
            qdrant,
            embed_client,
            collection,
            canonical_data_path,
            include_patterns: Arc::new(build_include_globset(include_patterns)),
            rerank_client,
            schema_cache,
            state_db,
        }
    }

    /// A fresh snapshot of the live config, same convention as
    /// `KbSearchServer::config` — every handler fetches its own snapshot rather
    /// than one captured at construction, so a `POST /admin/reload` is observed
    /// starting with the very next request.
    fn config(&self) -> Arc<ResolvedConfig> {
        crate::config::load_shared_config(&self.config)
    }

    /// The document metadata index, opened on first use.
    async fn state_db(&self) -> anyhow::Result<&StateDb> {
        self.state_db
            .get_or_try_init(|| async {
                let path = self.config().state_db_path();
                StateDb::new(Path::new(&path)).await
            })
            .await
    }

    /// Build a `RetrievalDeps` bundle from this state's fields, same shape
    /// `KbSearchServer::deps` builds.
    fn deps(&self) -> RetrievalDeps<'_, EmbedClient, QdrantStore> {
        RetrievalDeps {
            embed_client: &self.embed_client,
            qdrant: &self.qdrant,
            collection: &self.collection,
            data_path: &self.canonical_data_path,
            include_patterns: &self.include_patterns,
            reranker: self
                .rerank_client
                .as_ref()
                .map(|c| c.as_ref() as &(dyn crate::rerank::Reranker + Send + Sync)),
        }
    }

    /// Build a `write::WriteDeps` bundle from this state's fields and a config
    /// snapshot, same shape `KbSearchServer::write_document`/`delete_document`
    /// build for the MCP tool surface. `token` must outlive the returned value —
    /// callers read it from the environment once per request and hold it in a
    /// local binding (see the handlers below), same convention `mcp.rs` uses.
    fn write_deps<'a>(
        &'a self,
        config: &'a ResolvedConfig,
        token: &'a Option<String>,
    ) -> WriteDeps<'a, EmbedClient, QdrantStore> {
        WriteDeps {
            retrieval: self.deps(),
            canonical_data_path: &self.canonical_data_path,
            schema_cache: &self.schema_cache,
            validation: &config.validation,
            prepend_description: config.chunking.prepend_description,
            dedup_enabled: config.write.dedup_enabled,
            dedup_threshold: config.write.dedup_threshold,
            git_url: config.source.git_url.as_deref(),
            branch: &config.source.branch,
            token: token.as_deref(),
            commit_author_name: &config.write.commit_author_name,
            commit_author_email: &config.write.commit_author_email,
        }
    }

    /// The git-pull token for this KB, read fresh from the environment per
    /// request — same convention `KbSearchServer::write_document`/`delete_document`
    /// use, so a token rotated via env var takes effect on the next request with
    /// no restart.
    fn git_token(&self, config: &ResolvedConfig) -> Option<String> {
        std::env::var(&config.source.git_token_env)
            .ok()
            .filter(|s| !s.is_empty())
    }
}

// ---------------------------------------------------------------------------
// Static shell + assets
// ---------------------------------------------------------------------------

async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../assets/ui/index.html"))
}

async fn viz_css_handler() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        include_str!("../assets/ui/viz.css"),
    )
}

async fn viz_js_handler() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        include_str!("../assets/ui/viz.js"),
    )
}

async fn cytoscape_handler() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        include_str!("../assets/ui/vendor/cytoscape.min.js"),
    )
}

async fn marked_handler() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        include_str!("../assets/ui/vendor/marked.min.js"),
    )
}

async fn dompurify_handler() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        include_str!("../assets/ui/vendor/dompurify.min.js"),
    )
}

async fn edit_js_handler() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        include_str!("../assets/ui/edit.js"),
    )
}

// ---------------------------------------------------------------------------
// GET /api/graph
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, PartialEq)]
struct GraphNodeData {
    id: String,
    label: String,
    #[serde(rename = "type")]
    r#type: Option<String>,
    domain: String,
    tags: Vec<String>,
    status: Option<String>,
    description: Option<String>,
    color: String,
    size: u32,
    mtime: i64,
}

#[derive(Debug, Serialize, PartialEq)]
struct GraphNode {
    data: GraphNodeData,
}

#[derive(Debug, Serialize, PartialEq)]
struct GraphEdgeData {
    id: String,
    source: String,
    target: String,
    kind: String,
    /// Present only on semantic edges — markdown edges carry no score.
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<f64>,
}

#[derive(Debug, Serialize, PartialEq)]
struct GraphEdge {
    data: GraphEdgeData,
}

#[derive(Debug, Serialize)]
struct GraphResponse {
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    types: Vec<String>,
    palette: BTreeMap<String, String>,
}

/// Deterministic FNV-1a hash. NOT `std::collections`'s default hasher (SipHash with
/// a random per-process seed) — the palette must assign the same color to the same
/// type across restarts and across replicas, not just within one process.
fn fnv1a(s: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for b in s.bytes() {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

/// A stable, deterministic color for an arbitrary `type` value.
///
/// The OKF original (`generator.py`) hard-codes a 3-entry palette map, which only
/// works for a small, closed vocabulary. This knowledge base's `type` vocabulary is
/// open-ended and schema-declared per scope (see `schema.rs`), so rather than a
/// fixed table this hashes the type name to a hue and renders a fixed-saturation,
/// fixed-lightness HSL color: same input, same output, forever, with no table to
/// keep in sync as new types appear.
fn stable_color_for_type(type_name: &str) -> String {
    let hue = (fnv1a(type_name) % 360) as f64;
    hsl_to_hex(hue, 0.55, 0.55)
}

/// Minimal HSL -> `#rrggbb` conversion (h in `[0, 360)`, s/l in `[0.0, 1.0]`).
fn hsl_to_hex(h: f64, s: f64, l: f64) -> String {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = h / 60.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r1, g1, b1) = match hp as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    let to_byte = |v: f64| ((v + m) * 255.0).round().clamp(0.0, 255.0) as u8;
    format!("#{:02x}{:02x}{:02x}", to_byte(r1), to_byte(g1), to_byte(b1))
}

/// Build the `/api/graph` response from raw rows. Pure and separate from the
/// handler so it is unit-testable without a database.
///
/// Edges are dropped when either endpoint is not among `summaries`' paths — the
/// contract only requires dropping a dangling *target* (a renamed/removed file
/// whose link row hasn't been cleaned up yet), but a dangling *source* would be
/// just as broken to hand to Cytoscape, so both ends are checked.
fn build_graph_response(
    summaries: &[DocumentSummary],
    links: &[(String, String, String, Option<f64>)],
) -> GraphResponse {
    let node_ids: HashSet<&str> = summaries.iter().map(|s| s.file_path.as_str()).collect();

    let mut types: BTreeSet<String> = BTreeSet::new();
    let mut nodes = Vec::with_capacity(summaries.len());

    for s in summaries {
        let obj = s.frontmatter.as_object();
        let r#type = obj
            .and_then(|o| o.get("type"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let status = obj
            .and_then(|o| o.get("status"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let domain = obj
            .and_then(|o| o.get("domain"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let tags = obj
            .and_then(|o| o.get("tags"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| t.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        if let Some(t) = &r#type {
            types.insert(t.clone());
        }

        let label = s.title.clone().unwrap_or_else(|| {
            Path::new(&s.file_path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&s.file_path)
                .to_string()
        });
        let color = r#type
            .as_deref()
            .map(stable_color_for_type)
            .unwrap_or_else(|| DEFAULT_NODE_COLOR.to_string());

        nodes.push(GraphNode {
            data: GraphNodeData {
                id: s.file_path.clone(),
                label,
                r#type,
                domain,
                tags,
                status,
                description: s.description.clone(),
                color,
                size: NODE_SIZE,
                mtime: s.mtime,
            },
        });
    }

    let palette: BTreeMap<String, String> = types
        .iter()
        .map(|t| (t.clone(), stable_color_for_type(t)))
        .collect();

    let edges: Vec<GraphEdge> = links
        .iter()
        .filter(|(source, target, _, _)| {
            node_ids.contains(source.as_str()) && node_ids.contains(target.as_str())
        })
        .map(|(source, target, kind, score)| GraphEdge {
            data: GraphEdgeData {
                id: format!("{source}__{target}__{kind}"),
                source: source.clone(),
                target: target.clone(),
                kind: kind.clone(),
                score: *score,
            },
        })
        .collect();

    GraphResponse {
        nodes,
        edges,
        types: types.into_iter().collect(),
        palette,
    }
}

async fn graph_handler(State(state): State<UiState>) -> Response {
    let db = match state.state_db().await {
        Ok(db) => db,
        Err(e) => {
            error!("graph: metadata index unavailable: {:#}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("document index unavailable: {e:#}")})),
            )
                .into_response();
        }
    };

    let summaries = match db.all_document_summaries().await {
        Ok(s) => s,
        Err(e) => {
            error!("graph: failed to list documents: {:#}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("{e:#}")})),
            )
                .into_response();
        }
    };
    let links = match db.all_links().await {
        Ok(l) => l,
        Err(e) => {
            error!("graph: failed to list links: {:#}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("{e:#}")})),
            )
                .into_response();
        }
    };

    Json(build_graph_response(&summaries, &links)).into_response()
}

// ---------------------------------------------------------------------------
// GET /api/search
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct SearchQueryParams {
    #[serde(default)]
    q: Option<String>,
    #[serde(default)]
    limit: Option<u64>,
    #[serde(default)]
    domain: Option<String>,
    #[serde(rename = "type", default)]
    r#type: Option<String>,
    #[serde(default)]
    tags: Option<String>,
}

#[derive(Debug, Serialize, PartialEq)]
struct ApiSearchResult {
    score: f32,
    file_path: String,
    title: String,
    text: String,
    line_start: Option<i64>,
    line_end: Option<i64>,
}

fn to_api_result(r: &SearchResult, data_path: &Path) -> ApiSearchResult {
    let file_path_raw = r
        .payload
        .get("file_path")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    ApiSearchResult {
        score: r.score,
        file_path: retrieval::relative_to_data(file_path_raw, data_path),
        title: r
            .payload
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("(untitled)")
            .to_string(),
        text: r
            .payload
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        line_start: r.payload.get("line_start").and_then(|v| v.as_i64()),
        line_end: r.payload.get("line_end").and_then(|v| v.as_i64()),
    }
}

/// Core search, generic over the retrieval traits so it is unit-testable with
/// fakes — independent of `UiState`'s concrete `QdrantStore`/`EmbedClient` fields,
/// same reasoning `retrieval::search` itself is tested against mocks in
/// `retrieval.rs`.
async fn run_search<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &RetrievalDeps<'_, E, Q>,
    query: &str,
    filters: &SearchFilters,
    opts: &SearchOptions,
) -> Result<Vec<ApiSearchResult>, retrieval::SearchError> {
    let results = retrieval::search(deps, query, filters, opts).await?;
    Ok(results
        .iter()
        .map(|r| to_api_result(r, deps.data_path))
        .collect())
}

fn search_error_response(err: &retrieval::SearchError) -> (StatusCode, serde_json::Value) {
    match err {
        retrieval::SearchError::Embed(e) => {
            error!("web search: embedding failed: {:#}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                serde_json::json!({"error": "failed to generate query embedding"}),
            )
        }
        retrieval::SearchError::Search(e) => {
            error!("web search: qdrant search failed: {:#}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                serde_json::json!({"error": "search query failed"}),
            )
        }
    }
}

async fn search_handler(
    State(state): State<UiState>,
    Query(params): Query<SearchQueryParams>,
) -> Response {
    let q = params.q.as_deref().map(str::trim).unwrap_or("");
    if q.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "missing or empty 'q' parameter"})),
        )
            .into_response();
    }
    if q.len() > MAX_SEARCH_QUERY_LEN {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("q exceeds maximum length of {MAX_SEARCH_QUERY_LEN} characters")
            })),
        )
            .into_response();
    }
    if let Some(d) = &params.domain
        && d.len() > MAX_SEARCH_FILTER_LEN
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("domain exceeds maximum length of {MAX_SEARCH_FILTER_LEN} characters")
            })),
        )
            .into_response();
    }
    if let Some(t) = &params.r#type
        && t.len() > MAX_SEARCH_FILTER_LEN
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("type exceeds maximum length of {MAX_SEARCH_FILTER_LEN} characters")
            })),
        )
            .into_response();
    }

    let tags: Option<Vec<String>> = params.tags.as_deref().map(|s| {
        s.split(',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .take(MAX_SEARCH_TAGS)
            .map(str::to_string)
            .collect()
    });

    let config = state.config();
    let limit = params
        .limit
        .unwrap_or(config.search.default_limit)
        .min(config.search.max_limit)
        .max(1);

    let filters = SearchFilters {
        domain: params.domain,
        r#type: params.r#type,
        tags,
    };
    // Same defaults `KbSearchServer::search` resolves from config — explain and
    // the mtime range filters are not exposed by this endpoint's query params.
    let opts = SearchOptions {
        limit,
        min_score: config.search.min_score,
        hybrid: config.search.hybrid,
        rrf_candidates: config.search.rrf_candidates as u64,
        explain: false,
        modified_after: None,
        modified_before: None,
        rerank_candidate_limit: config.reranking.as_ref().map(|r| r.candidate_limit as u64),
        diversity_max_per_document: config.search.diversity_max_per_document,
    };

    match run_search(&state.deps(), q, &filters, &opts).await {
        Ok(results) => (
            StatusCode::OK,
            Json(serde_json::json!({"results": results})),
        )
            .into_response(),
        Err(e) => {
            let (status, body) = search_error_response(&e);
            (status, Json(body)).into_response()
        }
    }
}

// ---------------------------------------------------------------------------
// GET /api/doc/{*path}
// ---------------------------------------------------------------------------

fn get_doc_error_response(
    err: &retrieval::GetDocumentError,
    raw: &str,
) -> (StatusCode, serde_json::Value) {
    match err {
        retrieval::GetDocumentError::Outside => (
            StatusCode::FORBIDDEN,
            serde_json::json!({"error": "path is outside the data directory"}),
        ),
        retrieval::GetDocumentError::NotPermitted => (
            StatusCode::FORBIDDEN,
            serde_json::json!({"error": "file type not permitted"}),
        ),
        retrieval::GetDocumentError::NotFound { suggestions } => (
            StatusCode::NOT_FOUND,
            serde_json::json!({
                "error": format!("document not found: '{raw}'"),
                "suggestions": suggestions,
            }),
        ),
        retrieval::GetDocumentError::Ambiguous { matches } => (
            StatusCode::NOT_FOUND,
            serde_json::json!({
                "error": format!("ambiguous path '{raw}'"),
                "matches": matches,
            }),
        ),
        retrieval::GetDocumentError::Io(msg) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            serde_json::json!({"error": msg}),
        ),
    }
}

async fn get_doc_handler(
    State(state): State<UiState>,
    AxumPath(raw_path): AxumPath<String>,
) -> Response {
    let index = match state.state_db().await {
        Ok(db) => db,
        Err(e) => {
            error!("get_doc: metadata index unavailable: {:#}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("document index unavailable: {e:#}")})),
            )
                .into_response();
        }
    };

    match retrieval::get_document(&state.deps(), index, &raw_path).await {
        Ok(doc) => {
            let content_hash = crate::ingest::compute_hash_from_bytes(doc.content.as_bytes());
            let rel = retrieval::relative_to_data(
                &doc.path.to_string_lossy(),
                &state.canonical_data_path,
            );
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "path": rel,
                    "content": doc.content,
                    "content_hash": content_hash,
                })),
            )
                .into_response()
        }
        Err(e) => {
            let (status, body) = get_doc_error_response(&e, &raw_path);
            (status, Json(body)).into_response()
        }
    }
}

// ---------------------------------------------------------------------------
// GET /api/schema/{*path}
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, PartialEq)]
struct SchemaFieldEntry {
    field: String,
    #[serde(rename = "type")]
    r#type: Option<String>,
    required: bool,
    indexed: bool,
    values: Option<Vec<String>>,
    default: Option<serde_json::Value>,
    open: bool,
    declared_in: Option<String>,
}

#[derive(Debug, Serialize, PartialEq)]
struct SchemaResponse {
    frozen: bool,
    fields: Vec<SchemaFieldEntry>,
}

/// Normalize a caller-supplied path into a safe, KB-relative path.
///
/// Mirrors `mcp::normalize_scope_path`'s traversal rejection (private to that
/// module, hence redeclared here): this is a read-only cache lookup, not a
/// filesystem access, but a `..` component would still resolve the schema for the
/// wrong scope.
fn normalize_ui_path(raw: &str) -> Result<PathBuf, String> {
    let trimmed = raw.trim().trim_start_matches("./").trim_matches('/');
    if trimmed.is_empty() {
        return Ok(PathBuf::new());
    }
    let path = PathBuf::from(trimmed);
    if path
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err("path must not contain '..'".to_string());
    }
    Ok(path)
}

async fn get_schema_handler(
    State(state): State<UiState>,
    AxumPath(raw_path): AxumPath<String>,
) -> Response {
    let rel = match normalize_ui_path(&raw_path) {
        Ok(r) => r,
        Err(msg) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": msg})),
            )
                .into_response();
        }
    };

    let schemas = schema::load_shared(&state.schema_cache);
    // A document path resolves via its parent directory; a directory reference
    // resolves to itself — append a sentinel filename so `resolve_for`'s internal
    // `.parent()` call lands on `rel`, same trick `mcp::get_schema` uses.
    let lookup = if raw_path.ends_with(".md") {
        rel.clone()
    } else {
        rel.join("_")
    };

    let resolved = schemas.resolve_for(&lookup);
    let frozen = schemas.is_frozen(&lookup).is_some();

    let fields: Vec<SchemaFieldEntry> = resolved
        .fields
        .iter()
        .take(MAX_SCHEMA_FIELDS)
        .map(|(field, def)| SchemaFieldEntry {
            field: field.clone(),
            r#type: def.ty.map(|t| format!("{t:?}").to_lowercase()),
            required: def.required,
            indexed: def.indexed,
            values: def
                .values
                .as_ref()
                .map(|v| v.iter().take(MAX_SCHEMA_VALUES).cloned().collect()),
            default: def.default.clone(),
            open: def.open,
            declared_in: resolved.origin.get(field).cloned(),
        })
        .collect();

    Json(SchemaResponse { frozen, fields }).into_response()
}

// ---------------------------------------------------------------------------
// POST/DELETE /api/doc/{*path}
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct WriteDocBody {
    content: String,
    #[serde(default)]
    commit_message: Option<String>,
    #[serde(default)]
    create: bool,
    #[serde(default)]
    expected_hash: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeleteDocBody {
    #[serde(default)]
    commit_message: Option<String>,
}

fn bad_request(msg: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error": msg.into()})),
    )
        .into_response()
}

fn forbidden(msg: impl Into<String>) -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(serde_json::json!({"error": msg.into()})),
    )
        .into_response()
}

/// Map a `retrieval::ResolveErr` produced while resolving a write/delete target
/// onto an HTTP response. Mirrors the MCP write tools' own handling of this exact
/// enum (`edit_document`/`delete_document` in `mcp.rs`) so the web UI enforces
/// the same eligibility rules through a different transport: `NotPermitted`
/// (a path the `indexing.include` globset would not index — e.g. anything
/// outside `**/*.md`, including files under `.git/`) and `Outside` (path
/// traversal/symlink escape) are both hard failures with no side effect, not
/// something `write::write_document`/`write::delete_document` are asked to
/// adjudicate — those two functions only re-verify traversal safety, not file
/// -type eligibility, so this check MUST happen here before either is called.
///
/// `Other`'s message (built in `retrieval::resolve_within_data`) embeds the
/// canonicalized absolute filesystem path — fine for the MCP surface, which is
/// trusted and where that path is diagnostically useful, but this route is
/// reachable with no authentication at all (see the module doc comment), so a
/// caller must never learn the container's data-path layout from a 500 body.
/// Log the detailed message server-side and return a generic one instead —
/// unlike the MCP adapter, this is NOT a message-parity surface.
fn resolve_write_target_error_response(err: retrieval::ResolveErr) -> Response {
    match err {
        retrieval::ResolveErr::NotFound => write_error_response(&WriteError::NotFound),
        retrieval::ResolveErr::Outside => forbidden("path is outside the data directory"),
        retrieval::ResolveErr::NotPermitted => forbidden("file type not permitted"),
        retrieval::ResolveErr::Other(msg) => {
            error!("resolve_within_data: {msg}");
            write_error_response(&WriteError::Io {
                msg: "internal error resolving path".to_string(),
            })
        }
    }
}

/// Map a successful `write::write_document`/`write::delete_document` result onto
/// the fixed HTTP contract's 200 body: `{"outcome", "sha", "rebased_paths"}`.
fn write_success_response(success: WriteSuccess) -> Response {
    let outcome = match success.outcome {
        WriteOutcome::Synced => "synced",
        WriteOutcome::CommittedPendingSync => "committed_pending_sync",
    };
    let rebased_paths: Vec<String> = success
        .rebased_paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "outcome": outcome,
            "sha": success.sha,
            "rebased_paths": rebased_paths,
        })),
    )
        .into_response()
}

/// Map a `write::write_document`/`write::delete_document` failure onto the fixed
/// HTTP contract's status codes:
///
/// - 422 `{"outcome": "failed_no_change", "field_errors": [...]}}` — frontmatter
///   validation failed (and `Frozen`, which is the same "nothing was written,
///   here's why" shape one level up: the schema governing the write couldn't
///   even be resolved).
/// - 409 — `expected_hash` mismatch, create-on-existing, edit/delete-on-missing,
///   or a near-duplicate document blocking a create (`DedupHit`).
/// - 400 — an unsafe path or a commit message git/the log would mangle.
/// - 500 `{"outcome": "failed_inconsistent_state"}` — the pre-commit rollback
///   itself failed; filesystem and git state may disagree.
/// - 500 `{"outcome": "failed_no_change"}` — any other pre-commit failure
///   (a clean rollback, or an I/O error before git was ever touched).
fn write_error_response(err: &WriteError) -> Response {
    match err {
        WriteError::Frozen { reason } => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "outcome": "failed_no_change",
                "error": format!(
                    "the schema governing this directory is invalid: {reason}"
                ),
            })),
        )
            .into_response(),
        WriteError::Validation { result } => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "outcome": "failed_no_change",
                "field_errors": result.field_errors,
            })),
        )
            .into_response(),
        WriteError::DedupHit {
            duplicate_of,
            similarity,
            threshold,
        } => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "outcome": "failed_no_change",
                "error": "a similar document already exists",
                "duplicate_of": duplicate_of,
                "similarity": similarity,
                "threshold": threshold,
            })),
        )
            .into_response(),
        WriteError::InvalidCommitMessage { reason } => bad_request(reason.clone()),
        WriteError::UnsafePath { msg } => bad_request(msg.clone()),
        // `msg` embeds a server-side absolute filesystem path (a
        // `resolve_safe_write_path` canonicalize failure) — see
        // `WriteError::Internal`'s doc comment. This route is reachable with no
        // authentication at all (see this module's doc comment), so log the
        // detail server-side and return the same generic message
        // `resolve_write_target_error_response`'s `Other` arm uses for the
        // analogous failure in `retrieval::resolve_within_data`, rather than
        // relay `msg` verbatim like the MCP adapters do (that surface is
        // trusted, and this text is not, by design of `WriteError::Internal`'s
        // split from `UnsafePath` — see F4/G2).
        WriteError::Internal { msg } => {
            error!("write pipeline: {msg}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "outcome": "failed_no_change",
                    "error": "internal error resolving path",
                })),
            )
                .into_response()
        }
        WriteError::AlreadyExists => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "outcome": "failed_no_change",
                "error": "document already exists",
            })),
        )
            .into_response(),
        WriteError::NotFound => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "outcome": "failed_no_change",
                "error": "document does not exist",
            })),
        )
            .into_response(),
        WriteError::StaleHash { expected, actual } => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "outcome": "failed_no_change",
                "error": "document has changed since it was read",
                "expected_hash": expected,
                "actual_hash": actual,
            })),
        )
            .into_response(),
        WriteError::PreCommitFailed {
            rolled_back: true,
            msg,
        } => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "outcome": "failed_no_change",
                "error": msg,
            })),
        )
            .into_response(),
        WriteError::PreCommitFailed {
            rolled_back: false,
            msg,
        } => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "outcome": "failed_inconsistent_state",
                "error": msg,
            })),
        )
            .into_response(),
        WriteError::Io { msg } => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "outcome": "failed_no_change",
                "error": msg,
            })),
        )
            .into_response(),
    }
}

/// `POST /api/doc/{*path}`: create (`create: true`) or full-replace-edit
/// (`create: false`) a document. Thin adapter over `write::write_document` — see
/// the API contract in the feature plan for the exact request/response shapes.
async fn post_doc_handler(
    State(state): State<UiState>,
    AxumPath(raw_path): AxumPath<String>,
    Json(body): Json<WriteDocBody>,
) -> Response {
    let rel_path = retrieval::kb_root_relative(raw_path.trim()).to_string();
    if rel_path.is_empty() {
        return bad_request("path parameter is empty");
    }
    if body.content.len() > MAX_WRITE_CONTENT_LEN {
        return bad_request(format!(
            "content is too large ({} bytes); maximum is {} bytes",
            body.content.len(),
            MAX_WRITE_CONTENT_LEN
        ));
    }

    // Include-pattern eligibility guard, same rule the MCP write tools enforce
    // (`create_document`'s explicit `include_patterns.is_match` check /
    // `edit_document`'s `resolve_within_data` call) — `write::write_document`
    // itself only re-verifies path *traversal* safety immediately before each
    // filesystem action, never file-*type* eligibility. Without this check here,
    // an unauthenticated `create: false` edit could overwrite the bytes of any
    // existing path under the KB checkout the include globset would never index
    // — including files under `.git/` (a hostile `.git/config` is a well-known
    // local code-execution primitive) — before `commit_and_sync` ever runs `git
    // add`/`commit` against that same repo.
    //
    // For an edit, this doubles as reading the current on-disk content: it is
    // both the base the diff/`expected_hash` check compares against and (for a
    // create) simply empty, since there is nothing on disk yet.
    let (old_content, rel_path) = if body.create {
        if !state.include_patterns.is_match(&rel_path) {
            return forbidden("file type not permitted");
        }
        (String::new(), rel_path)
    } else {
        let canonical = match retrieval::resolve_within_data(
            &rel_path,
            &state.canonical_data_path,
            &state.include_patterns,
        ) {
            Ok(c) => c,
            Err(e) => return resolve_write_target_error_response(e),
        };
        let rel_path = canonical
            .strip_prefix(&state.canonical_data_path)
            .unwrap_or(&canonical)
            .to_string_lossy()
            .into_owned();
        match tokio::fs::read_to_string(&canonical).await {
            Ok(s) => (s, rel_path),
            Err(e) => {
                error!("post_doc: failed to read '{}': {}", canonical.display(), e);
                return write_error_response(&WriteError::Io { msg: e.to_string() });
            }
        }
    };

    let config = state.config();
    let token = state.git_token(&config);
    let deps = state.write_deps(&config, &token);

    let req = WriteRequest {
        rel_path: &rel_path,
        old_content: &old_content,
        new_content: &body.content,
        is_create: body.create,
        message: body.commit_message.as_deref(),
        default_verb: if body.create { "add" } else { "update" },
        // No HTTP-level dedup bypass in the fixed API contract — the create-path
        // dedup gate (when `write.dedup_enabled`) always applies.
        force_new: None,
        operation: if body.create {
            "create_document (web ui)"
        } else {
            "edit_document (web ui, full replace)"
        },
        expected_hash: body.expected_hash.as_deref(),
    };

    match write::write_document(&deps, req).await {
        Ok(success) => write_success_response(success),
        Err(err) => write_error_response(&err),
    }
}

/// `DELETE /api/doc/{*path}`: thin adapter over `write::delete_document`.
async fn delete_doc_handler(
    State(state): State<UiState>,
    AxumPath(raw_path): AxumPath<String>,
    Json(body): Json<DeleteDocBody>,
) -> Response {
    let rel_path = retrieval::kb_root_relative(raw_path.trim()).to_string();
    if rel_path.is_empty() {
        return bad_request("path parameter is empty");
    }

    // Same include-pattern eligibility guard as `post_doc_handler`'s edit
    // branch — `write::delete_document` only re-verifies traversal safety, never
    // file-type eligibility, so this MUST happen here: without it, an
    // unauthenticated `DELETE /api/doc/.git/config` (or `.gitignore`,
    // `.kb-schema.yaml`, or any other tracked/untracked path the include
    // globset would never index) would delete and commit/push the deletion of
    // that file with no content required at all.
    let canonical = match retrieval::resolve_within_data(
        &rel_path,
        &state.canonical_data_path,
        &state.include_patterns,
    ) {
        Ok(c) => c,
        Err(e) => return resolve_write_target_error_response(e),
    };
    let rel_path = canonical
        .strip_prefix(&state.canonical_data_path)
        .unwrap_or(&canonical)
        .to_string_lossy()
        .into_owned();

    let config = state.config();
    let token = state.git_token(&config);
    let deps = state.write_deps(&config, &token);

    match write::delete_document(&deps, &rel_path, body.commit_message.as_deref()).await {
        Ok(success) => write_success_response(success),
        Err(err) => write_error_response(&err),
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Assemble every UI route. Merged into the main app in `server::run_server`
/// BEFORE the `GovernorLayer` wrap, so rate limiting applies to it like every
/// other route — see that function's comments at the merge site.
pub fn ui_router(state: UiState) -> Router {
    // `/api/doc/{*path}` carries all three methods (read, write, delete) on one
    // route so a single `DefaultBodyLimit` layer — matching `mcp_router`'s 10 MB
    // limit in `server.rs` — governs the two methods that accept a body, without
    // widening the limit for every other route in this router (which keeps
    // axum's un-overridden default for everything else).
    let doc_router = Router::new()
        .route(
            "/api/doc/{*path}",
            get(get_doc_handler)
                .post(post_doc_handler)
                .delete(delete_doc_handler),
        )
        .layer(DefaultBodyLimit::max(MAX_WRITE_BODY_BYTES));

    Router::new()
        .route("/", get(index_handler))
        .route("/assets/viz.css", get(viz_css_handler))
        .route("/assets/viz.js", get(viz_js_handler))
        .route("/assets/cytoscape.min.js", get(cytoscape_handler))
        .route("/assets/marked.min.js", get(marked_handler))
        .route("/assets/dompurify.min.js", get(dompurify_handler))
        .route("/assets/edit.js", get(edit_js_handler))
        .route("/api/graph", get(graph_handler))
        .route("/api/search", get(search_handler))
        .route("/api/schema/{*path}", get(get_schema_handler))
        .merge(doc_router)
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use std::collections::HashMap;
    use std::sync::RwLock;
    use tower::ServiceExt;

    // ------------------------------------------------------------------
    // Test fixtures
    // ------------------------------------------------------------------

    fn test_config(data_path: &Path) -> Arc<ResolvedConfig> {
        Arc::new(ResolvedConfig {
            source: crate::config::ResolvedSourceConfig {
                git_url: None,
                branch: "master".into(),
                data_path: Some(data_path.to_string_lossy().into_owned()),
                git_token_env: "GIT_PULL_TOKEN".into(),
            },
            indexing: Default::default(),
            frontmatter: Default::default(),
            chunking: Default::default(),
            embedding: crate::config::ResolvedEmbeddingConfig {
                base_url: "http://127.0.0.1:1/v1".into(),
                model: "test".into(),
                api_key: None,
                vector_size: 8,
                batch_size: 1,
                request_timeout_secs: 5,
                batch_concurrency: 1,
            },
            qdrant: crate::config::ResolvedQdrantConfig {
                // Port 1 refuses immediately rather than hanging a test that
                // accidentally reaches the network.
                url: "http://127.0.0.1:1".into(),
                collection: "test-col".into(),
            },
            validation: Default::default(),
            webhook: Default::default(),
            mcp: Default::default(),
            rate_limit: Default::default(),
            write: Default::default(),
            search: Default::default(),
            reranking: None,
            ui: Default::default(),
            provenance: Default::default(),
        })
    }

    fn test_schema_cache(config: &ResolvedConfig) -> SharedSchemaCache {
        Arc::new(RwLock::new(Arc::new(
            schema::SchemaCache::from_config_only(&config.frontmatter),
        )))
    }

    fn test_state(canonical_data_path: &Path) -> UiState {
        let config = test_config(canonical_data_path);
        UiState::new(
            crate::config::shared_config(Arc::clone(&config)),
            Arc::new(QdrantStore::new(&config.qdrant).expect("client construction is lazy")),
            Arc::new(EmbedClient::new(&config.embedding)),
            config.qdrant.collection.clone(),
            canonical_data_path.to_path_buf(),
            &["**/*.md".to_string()],
            None,
            test_schema_cache(&config),
            Arc::new(tokio::sync::OnceCell::new()),
        )
    }

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    struct MockEmbedder(Vec<f32>);
    impl QueryEmbedder for MockEmbedder {
        async fn embed_query(&self, _q: &str) -> anyhow::Result<Vec<f32>> {
            Ok(self.0.clone())
        }
    }

    struct MockStore(Vec<SearchResult>);
    impl RetrievalStore for MockStore {
        async fn search(
            &self,
            _collection: &str,
            _vector: Vec<f32>,
            _filters: HashMap<String, serde_json::Value>,
            _limit: u64,
        ) -> anyhow::Result<Vec<SearchResult>> {
            Ok(self.0.clone())
        }

        async fn hybrid_search(
            &self,
            _collection: &str,
            _dense: Vec<f32>,
            _sparse: (Vec<u32>, Vec<f32>),
            _filters: HashMap<String, serde_json::Value>,
            _limit: u64,
            _rrf_candidates: u64,
            _explain: bool,
        ) -> anyhow::Result<Vec<SearchResult>> {
            Ok(self.0.clone())
        }
    }

    fn make_md_globset() -> GlobSet {
        let mut builder = GlobSetBuilder::new();
        builder.add(Glob::new("**/*.md").unwrap());
        builder.build().unwrap()
    }

    fn default_opts() -> SearchOptions {
        SearchOptions {
            limit: 10,
            min_score: None,
            hybrid: false,
            rrf_candidates: 50,
            explain: false,
            modified_after: None,
            modified_before: None,
            rerank_candidate_limit: None,
            diversity_max_per_document: None,
        }
    }

    // ------------------------------------------------------------------
    // Palette
    // ------------------------------------------------------------------

    #[test]
    fn stable_color_for_type_is_deterministic_and_distinguishes_types() {
        assert_eq!(
            stable_color_for_type("guide"),
            stable_color_for_type("guide")
        );
        assert_ne!(
            stable_color_for_type("guide"),
            stable_color_for_type("reference")
        );
        assert!(stable_color_for_type("guide").starts_with('#'));
        assert_eq!(stable_color_for_type("guide").len(), 7);
    }

    // ------------------------------------------------------------------
    // build_graph_response (pure)
    // ------------------------------------------------------------------

    #[test]
    fn build_graph_response_empty_input_yields_empty_arrays() {
        let resp = build_graph_response(&[], &[]);
        assert!(resp.nodes.is_empty());
        assert!(resp.edges.is_empty());
        assert!(resp.types.is_empty());
        assert!(resp.palette.is_empty());
    }

    #[test]
    fn build_graph_response_drops_edges_with_a_dangling_endpoint() {
        let summaries = vec![
            DocumentSummary {
                file_path: "a.md".into(),
                title: Some("A".into()),
                description: None,
                mtime: 1,
                indexed_at: "now".into(),
                frontmatter: serde_json::json!({"type": "guide"}),
            },
            DocumentSummary {
                file_path: "b.md".into(),
                title: None,
                description: None,
                mtime: 2,
                indexed_at: "now".into(),
                frontmatter: serde_json::json!({}),
            },
        ];
        let links = vec![
            (
                "a.md".to_string(),
                "b.md".to_string(),
                "markdown".to_string(),
                None,
            ),
            (
                "a.md".to_string(),
                "missing.md".to_string(),
                "semantic".to_string(),
                Some(0.9),
            ),
        ];

        let resp = build_graph_response(&summaries, &links);
        assert_eq!(resp.nodes.len(), 2);
        assert_eq!(
            resp.edges.len(),
            1,
            "the edge targeting a non-existent node must be dropped: {:?}",
            resp.edges
        );
        assert_eq!(resp.edges[0].data.target, "b.md");
        assert_eq!(resp.edges[0].data.kind, "markdown");
        assert_eq!(resp.edges[0].data.id, "a.md__b.md__markdown");

        // Node b has no title, so the label falls back to the filename.
        let node_b = resp.nodes.iter().find(|n| n.data.id == "b.md").unwrap();
        assert_eq!(node_b.data.label, "b.md");
        assert_eq!(node_b.data.r#type, None);
        assert_eq!(node_b.data.color, DEFAULT_NODE_COLOR);

        assert_eq!(resp.types, vec!["guide".to_string()]);
        assert!(resp.palette.contains_key("guide"));
    }

    #[test]
    fn build_graph_response_gives_distinct_ids_when_a_pair_has_both_edge_kinds() {
        // A linked pair of docs that also happen to be each other's nearest
        // semantic neighbor produces a markdown link AND a semantic link over
        // the exact same (source, target) pair. Edge ids must stay distinct
        // or downstream Cytoscape.js construction throws on the duplicate id.
        let summaries = vec![
            DocumentSummary {
                file_path: "a.md".into(),
                title: Some("A".into()),
                description: None,
                mtime: 1,
                indexed_at: "now".into(),
                frontmatter: serde_json::json!({}),
            },
            DocumentSummary {
                file_path: "b.md".into(),
                title: Some("B".into()),
                description: None,
                mtime: 2,
                indexed_at: "now".into(),
                frontmatter: serde_json::json!({}),
            },
        ];
        let links = vec![
            (
                "a.md".to_string(),
                "b.md".to_string(),
                "markdown".to_string(),
                None,
            ),
            (
                "a.md".to_string(),
                "b.md".to_string(),
                "semantic".to_string(),
                Some(0.95),
            ),
        ];

        let resp = build_graph_response(&summaries, &links);
        assert_eq!(resp.edges.len(), 2);
        let ids: std::collections::HashSet<&str> =
            resp.edges.iter().map(|e| e.data.id.as_str()).collect();
        assert_eq!(ids.len(), 2, "edge ids must be unique: {:?}", ids);
        assert!(ids.contains("a.md__b.md__markdown"));
        assert!(ids.contains("a.md__b.md__semantic"));
    }

    #[test]
    fn build_graph_response_degrades_non_object_frontmatter() {
        let summaries = vec![DocumentSummary {
            file_path: "weird.md".into(),
            title: None,
            description: None,
            mtime: 0,
            indexed_at: "now".into(),
            frontmatter: serde_json::Value::Null,
        }];
        let resp = build_graph_response(&summaries, &[]);
        assert_eq!(resp.nodes.len(), 1);
        assert_eq!(resp.nodes[0].data.r#type, None);
        assert_eq!(resp.nodes[0].data.tags, Vec::<String>::new());
        assert_eq!(resp.nodes[0].data.domain, "");
    }

    // ------------------------------------------------------------------
    // run_search (generic core, mocked)
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn run_search_happy_path_maps_payload_into_api_shape() {
        let mut payload = HashMap::new();
        payload.insert(
            "file_path".to_string(),
            serde_json::json!("/data/food/a.md"),
        );
        payload.insert("title".to_string(), serde_json::json!("A Doc"));
        payload.insert("text".to_string(), serde_json::json!("chunk text"));
        payload.insert("line_start".to_string(), serde_json::json!(1));
        payload.insert("line_end".to_string(), serde_json::json!(10));
        let result = SearchResult {
            score: 0.9,
            pre_rerank_score: None,
            dense_score: None,
            sparse_score: None,
            payload,
        };

        let embed = MockEmbedder(vec![0.1, 0.2]);
        let store = MockStore(vec![result]);
        let data_path = Path::new("/data");
        let include = make_md_globset();
        let deps = RetrievalDeps {
            embed_client: &embed,
            qdrant: &store,
            collection: "test",
            data_path,
            include_patterns: &include,
            reranker: None,
        };
        let filters = SearchFilters {
            domain: None,
            r#type: None,
            tags: None,
        };

        let results = run_search(&deps, "query", &filters, &default_opts())
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, "food/a.md");
        assert_eq!(results[0].title, "A Doc");
        assert_eq!(results[0].text, "chunk text");
        assert_eq!(results[0].line_start, Some(1));
        assert_eq!(results[0].line_end, Some(10));
        assert_eq!(results[0].score, 0.9);
    }

    #[tokio::test]
    async fn run_search_maps_missing_payload_fields_to_fallbacks() {
        let result = SearchResult {
            score: 0.5,
            pre_rerank_score: None,
            dense_score: None,
            sparse_score: None,
            payload: HashMap::new(),
        };
        let embed = MockEmbedder(vec![0.1]);
        let store = MockStore(vec![result]);
        let data_path = Path::new("/data");
        let include = make_md_globset();
        let deps = RetrievalDeps {
            embed_client: &embed,
            qdrant: &store,
            collection: "test",
            data_path,
            include_patterns: &include,
            reranker: None,
        };
        let filters = SearchFilters {
            domain: None,
            r#type: None,
            tags: None,
        };

        let results = run_search(&deps, "query", &filters, &default_opts())
            .await
            .unwrap();
        assert_eq!(results[0].title, "(untitled)");
        assert_eq!(results[0].text, "");
        assert_eq!(results[0].line_start, None);
    }

    // ------------------------------------------------------------------
    // get_doc error mapping (pure)
    // ------------------------------------------------------------------

    #[test]
    fn get_doc_error_response_maps_every_variant() {
        let (status, _) = get_doc_error_response(&retrieval::GetDocumentError::Outside, "x");
        assert_eq!(status, StatusCode::FORBIDDEN);

        let (status, _) = get_doc_error_response(&retrieval::GetDocumentError::NotPermitted, "x");
        assert_eq!(status, StatusCode::FORBIDDEN);

        let (status, body) = get_doc_error_response(
            &retrieval::GetDocumentError::NotFound {
                suggestions: vec!["a.md".into()],
            },
            "x",
        );
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["suggestions"][0], "a.md");

        let (status, body) = get_doc_error_response(
            &retrieval::GetDocumentError::Ambiguous {
                matches: vec!["a.md".into(), "b/a.md".into()],
            },
            "a.md",
        );
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["matches"].as_array().unwrap().len(), 2);

        let (status, body) =
            get_doc_error_response(&retrieval::GetDocumentError::Io("boom".into()), "x");
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body["error"], "boom");
    }

    // ------------------------------------------------------------------
    // normalize_ui_path (pure)
    // ------------------------------------------------------------------

    #[test]
    fn normalize_ui_path_rejects_parent_dir_traversal() {
        assert!(normalize_ui_path("../etc/passwd").is_err());
        assert!(normalize_ui_path("food/../../etc").is_err());
    }

    #[test]
    fn normalize_ui_path_accepts_plain_relative_paths() {
        assert_eq!(
            normalize_ui_path("food/recipes").unwrap(),
            PathBuf::from("food/recipes")
        );
        assert_eq!(normalize_ui_path("").unwrap(), PathBuf::new());
        assert_eq!(normalize_ui_path("/food/").unwrap(), PathBuf::from("food"));
    }

    // ------------------------------------------------------------------
    // Router-level (oneshot) tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn index_route_serves_html() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(&dir.path().canonicalize().unwrap());
        let app = ui_router(state);
        let req = Request::builder().uri("/").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .unwrap()
            .to_str()
            .unwrap();
        assert!(ct.starts_with("text/html"), "{ct}");
    }

    #[tokio::test]
    async fn asset_routes_serve_with_correct_content_type() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        for (path, expected_ct) in [
            ("/assets/viz.css", "text/css"),
            ("/assets/viz.js", "application/javascript"),
            ("/assets/cytoscape.min.js", "application/javascript"),
            ("/assets/marked.min.js", "application/javascript"),
            ("/assets/dompurify.min.js", "application/javascript"),
            ("/assets/edit.js", "application/javascript"),
        ] {
            let app = ui_router(test_state(&canonical));
            let req = Request::builder().uri(path).body(Body::empty()).unwrap();
            let resp = app.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "{path}");
            let ct = resp
                .headers()
                .get(header::CONTENT_TYPE)
                .unwrap()
                .to_str()
                .unwrap();
            assert!(ct.starts_with(expected_ct), "{path}: {ct}");
        }
    }

    #[tokio::test]
    async fn graph_handler_empty_db_returns_empty_arrays() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        StateDb::new(&canonical.join("state.db")).await.unwrap();

        let app = ui_router(test_state(&canonical));
        let req = Request::builder()
            .uri("/api/graph")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["nodes"], serde_json::json!([]));
        assert_eq!(json["edges"], serde_json::json!([]));
        assert_eq!(json["types"], serde_json::json!([]));
        assert_eq!(json["palette"], serde_json::json!({}));
    }

    #[tokio::test]
    async fn graph_handler_drops_dangling_edges_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        let db = StateDb::new(&canonical.join("state.db")).await.unwrap();

        let fm_a: HashMap<String, serde_json::Value> = [
            ("title".to_string(), serde_json::json!("Doc A")),
            ("type".to_string(), serde_json::json!("guide")),
        ]
        .into_iter()
        .collect();
        db.upsert_document_metadata("a.md", &fm_a, 100, "ha", 1)
            .await
            .unwrap();

        let fm_b: HashMap<String, serde_json::Value> =
            [("type".to_string(), serde_json::json!("reference"))]
                .into_iter()
                .collect();
        db.upsert_document_metadata("b.md", &fm_b, 200, "hb", 1)
            .await
            .unwrap();

        db.replace_links("a.md", "markdown", &[("b.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("a.md", "semantic", &[("missing.md".to_string(), Some(0.9))])
            .await
            .unwrap();

        let app = ui_router(test_state(&canonical));
        let req = Request::builder()
            .uri("/api/graph")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;

        assert_eq!(json["nodes"].as_array().unwrap().len(), 2);
        let edges = json["edges"].as_array().unwrap();
        assert_eq!(
            edges.len(),
            1,
            "the semantic edge to a nonexistent node must be dropped: {edges:?}"
        );
        assert_eq!(edges[0]["data"]["kind"], "markdown");
        assert_eq!(edges[0]["data"]["target"], "b.md");

        let types: Vec<&str> = json["types"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(types, vec!["guide", "reference"]);
        assert!(json["palette"]["guide"].as_str().unwrap().starts_with('#'));
    }

    #[tokio::test]
    async fn graph_handler_degrades_malformed_frontmatter_without_500() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        let db = StateDb::new(&canonical.join("state.db")).await.unwrap();
        sqlx::query(
            "INSERT INTO documents
                (file_path, title, description, frontmatter, mtime, content_hash, chunk_count, indexed_at)
             VALUES ('bad.md', NULL, NULL, 'not valid json{{', 0, 'h', 0, datetime('now'))",
        )
        .execute(db.pool_for_test())
        .await
        .unwrap();

        let app = ui_router(test_state(&canonical));
        let req = Request::builder()
            .uri("/api/graph")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        let nodes = json["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0]["data"]["id"], "bad.md");
        assert_eq!(nodes[0]["data"]["type"], serde_json::Value::Null);
    }

    #[tokio::test]
    async fn search_handler_rejects_missing_query() {
        let dir = tempfile::tempdir().unwrap();
        let app = ui_router(test_state(&dir.path().canonicalize().unwrap()));
        let req = Request::builder()
            .uri("/api/search")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn search_handler_rejects_blank_query() {
        let dir = tempfile::tempdir().unwrap();
        let app = ui_router(test_state(&dir.path().canonicalize().unwrap()));
        let req = Request::builder()
            .uri("/api/search?q=%20%20")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn get_doc_handler_returns_content_and_hash() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        std::fs::write(canonical.join("a.md"), "---\ntitle: A\n---\nbody").unwrap();

        let app = ui_router(test_state(&canonical));
        let req = Request::builder()
            .uri("/api/doc/a.md")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["path"], "a.md");
        assert!(json["content"].as_str().unwrap().contains("body"));
        assert!(!json["content_hash"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn get_doc_handler_not_found_reports_suggestions() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        std::fs::write(canonical.join("alpha.md"), "content").unwrap();
        let db = StateDb::new(&canonical.join("state.db")).await.unwrap();
        // The fuzzy fallback reads `all_paths()`, backed by the `documents` metadata
        // table — not `indexed_files` bookkeeping — so seed via
        // `upsert_document_metadata`, not `upsert`.
        db.upsert_document_metadata("alpha.md", &HashMap::new(), 1, "h", 1)
            .await
            .unwrap();

        let app = ui_router(test_state(&canonical));
        let req = Request::builder()
            .uri("/api/doc/alfa.md")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let json = body_json(resp).await;
        assert!(
            json["suggestions"]
                .as_array()
                .unwrap()
                .iter()
                .any(|s| s == "alpha.md"),
            "{json}"
        );
    }

    #[tokio::test]
    async fn schema_handler_returns_declared_fields() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        std::fs::write(
            canonical.join(".kb-schema.yaml"),
            "fields:\n  status:\n    values: [draft, active, archived]\n    required: true\n",
        )
        .unwrap();

        let config = test_config(&canonical);
        let schema_cache: SharedSchemaCache = Arc::new(RwLock::new(Arc::new(
            schema::SchemaCache::build(&canonical, &config.frontmatter),
        )));
        let state = UiState::new(
            crate::config::shared_config(Arc::clone(&config)),
            Arc::new(QdrantStore::new(&config.qdrant).unwrap()),
            Arc::new(EmbedClient::new(&config.embedding)),
            config.qdrant.collection.clone(),
            canonical.clone(),
            &["**/*.md".to_string()],
            None,
            schema_cache,
            Arc::new(tokio::sync::OnceCell::new()),
        );

        let app = ui_router(state);
        let req = Request::builder()
            .uri("/api/schema/food/a.md")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["frozen"], false);
        let fields = json["fields"].as_array().unwrap();
        let status_field = fields.iter().find(|f| f["field"] == "status").unwrap();
        assert_eq!(status_field["required"], true);
        assert_eq!(status_field["values"].as_array().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn schema_handler_bad_path_returns_400() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        let app = ui_router(test_state(&canonical));
        // `%2e%2e` decodes to `..` — a real traversal attempt, not a literal
        // two-character segment.
        let req = Request::builder()
            .uri("/api/schema/%2e%2e/escape")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ------------------------------------------------------------------
    // write_error_response — every `WriteError` variant maps to the fixed
    // contract's status code (tested directly on constructed values, per the
    // chunk's own test guidance, rather than forcing every failure mode through
    // a live git repo).
    // ------------------------------------------------------------------

    fn sample_field_error() -> crate::validate::FieldError {
        crate::validate::FieldError {
            field: "title".to_string(),
            rule: "required".to_string(),
            message: "title is required".to_string(),
            got: None,
            expected: None,
            schema_origin: None,
        }
    }

    #[tokio::test]
    async fn write_error_response_validation_is_422_with_field_errors() {
        let result = crate::validate::ValidationResult {
            file_path: "docs/x.md".into(),
            valid: false,
            errors: vec!["title is required".into()],
            field_errors: vec![sample_field_error()],
        };
        let resp = write_error_response(&WriteError::Validation { result });
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let json = body_json(resp).await;
        assert_eq!(json["outcome"], "failed_no_change");
        let errors = json["field_errors"].as_array().unwrap();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0]["field"], "title");
        assert_eq!(errors[0]["rule"], "required");
    }

    #[tokio::test]
    async fn write_error_response_frozen_is_422() {
        let resp = write_error_response(&WriteError::Frozen {
            reason: "invalid yaml".into(),
        });
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let json = body_json(resp).await;
        assert_eq!(json["outcome"], "failed_no_change");
    }

    #[tokio::test]
    async fn write_error_response_dedup_hit_is_409() {
        let resp = write_error_response(&WriteError::DedupHit {
            duplicate_of: "docs/existing.md".into(),
            similarity: 0.92,
            threshold: 0.85,
        });
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let json = body_json(resp).await;
        assert_eq!(json["duplicate_of"], "docs/existing.md");
    }

    #[tokio::test]
    async fn write_error_response_already_exists_is_409() {
        let resp = write_error_response(&WriteError::AlreadyExists);
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn write_error_response_not_found_is_409() {
        let resp = write_error_response(&WriteError::NotFound);
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn write_error_response_stale_hash_is_409() {
        let resp = write_error_response(&WriteError::StaleHash {
            expected: "aaa".into(),
            actual: "bbb".into(),
        });
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let json = body_json(resp).await;
        assert_eq!(json["expected_hash"], "aaa");
        assert_eq!(json["actual_hash"], "bbb");
    }

    #[tokio::test]
    async fn write_error_response_invalid_commit_message_is_400() {
        let resp = write_error_response(&WriteError::InvalidCommitMessage {
            reason: "must not contain newlines".into(),
        });
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn write_error_response_unsafe_path_is_400() {
        let resp = write_error_response(&WriteError::UnsafePath {
            msg: "path escapes the knowledge base root".into(),
        });
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn write_error_response_internal_hides_the_absolute_path() {
        // G2: a canonicalize-failure message embeds a server-side absolute
        // filesystem path — this route has no authentication, so the response
        // body must never contain it (mirrors the F4 test for
        // `resolve_within_data`'s analogous failure).
        let abs = "/data/kb/some/container/path";
        let resp = write_error_response(&WriteError::Internal {
            msg: format!(
                "Invalid path: cannot canonicalize data root '{abs}': No such file or directory"
            ),
        });
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = body_json(resp).await;
        let body_str = body.to_string();
        assert!(
            !body_str.contains(abs),
            "response body leaked an absolute path: {body_str}"
        );
        assert_eq!(body["error"], "internal error resolving path");
    }

    #[tokio::test]
    async fn write_error_response_precommit_rolled_back_is_500_failed_no_change() {
        let resp = write_error_response(&WriteError::PreCommitFailed {
            rolled_back: true,
            msg: "commit failed".into(),
        });
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let json = body_json(resp).await;
        assert_eq!(json["outcome"], "failed_no_change");
    }

    #[tokio::test]
    async fn write_error_response_precommit_not_rolled_back_is_500_inconsistent_state() {
        let resp = write_error_response(&WriteError::PreCommitFailed {
            rolled_back: false,
            msg: "commit failed AND rollback failed".into(),
        });
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let json = body_json(resp).await;
        assert_eq!(json["outcome"], "failed_inconsistent_state");
    }

    #[tokio::test]
    async fn write_error_response_io_is_500_failed_no_change() {
        let resp = write_error_response(&WriteError::Io {
            msg: "disk full".into(),
        });
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let json = body_json(resp).await;
        assert_eq!(json["outcome"], "failed_no_change");
    }

    #[test]
    fn write_success_response_reports_outcome_sha_and_rebased_paths() {
        let success = WriteSuccess {
            outcome: WriteOutcome::Synced,
            sha: "deadbeef".into(),
            rebased_paths: vec![PathBuf::from("other.md")],
            diff: "+line".into(),
            sync_failure_cause: None,
        };
        let resp = write_success_response(success);
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ------------------------------------------------------------------
    // Router-level write/delete tests (git-backed, mirroring write.rs's own
    // harness pattern)
    // ------------------------------------------------------------------

    fn git_backed_ui_state(work: &tempfile::TempDir) -> UiState {
        let mut config = crate::mcp::make_test_resolved_config(work.path());
        // Bypass the dedup gate: it would otherwise call out to a (nonexistent)
        // embedding service before the write ever reaches the commit.
        Arc::get_mut(&mut config).unwrap().write.dedup_enabled = false;
        UiState::new(
            crate::config::shared_config(Arc::clone(&config)),
            Arc::new(QdrantStore::new(&config.qdrant).expect("client construction is lazy")),
            Arc::new(EmbedClient::new(&config.embedding)),
            config.qdrant.collection.clone(),
            work.path().canonicalize().unwrap(),
            &["**/*.md".to_string()],
            None,
            test_schema_cache(&config),
            Arc::new(tokio::sync::OnceCell::new()),
        )
    }

    fn post_doc_request(path: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(format!("/api/doc/{path}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn post_doc_create_then_edit_round_trip() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let state = git_backed_ui_state(&work);

        // Create.
        let app = ui_router(state.clone());
        let req = post_doc_request(
            "new.md",
            serde_json::json!({
                "content": "---\ntitle: New\n---\n\n# Body\n",
                "commit_message": "docs: add new.md",
                "create": true,
            }),
        );
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "create must succeed");
        let json = body_json(resp).await;
        assert_eq!(json["outcome"], "synced");
        assert!(!json["sha"].as_str().unwrap().is_empty());
        assert!(json["rebased_paths"].as_array().unwrap().is_empty());
        assert!(work.path().join("new.md").exists());

        // Re-create over the same path must fail as a conflict, not silently
        // succeed — proves `create: true` really is wired to the create path.
        let app = ui_router(state.clone());
        let req = post_doc_request(
            "new.md",
            serde_json::json!({
                "content": "---\ntitle: Again\n---\n\n# Body\n",
                "create": true,
            }),
        );
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);

        // Edit (create: false) must succeed and change the file on disk — proves
        // the create/edit flag actually selects a different code path.
        let app = ui_router(state.clone());
        let req = post_doc_request(
            "new.md",
            serde_json::json!({
                "content": "---\ntitle: Updated\n---\n\n# New body\n",
                "commit_message": "docs: update new.md",
                "create": false,
            }),
        );
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "edit must succeed");
        let json = body_json(resp).await;
        assert_eq!(json["outcome"], "synced");
        assert_eq!(
            std::fs::read_to_string(work.path().join("new.md")).unwrap(),
            "---\ntitle: Updated\n---\n\n# New body\n"
        );

        // Editing a path that was never created reports the edit-on-missing
        // conflict, not a generic 500.
        let app = ui_router(state.clone());
        let req = post_doc_request(
            "never-existed.md",
            serde_json::json!({
                "content": "---\ntitle: X\n---\n\n# Body\n",
                "create": false,
            }),
        );
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn post_doc_stale_expected_hash_is_409() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::write(
            work.path().join("edit-me.md"),
            "---\ntitle: Old\n---\n\n# Old\n",
        )
        .unwrap();
        std::process::Command::new("git")
            .args(["add", "--", "edit-me.md"])
            .current_dir(work.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args([
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "commit",
                "-m",
                "add edit-me.md",
            ])
            .current_dir(work.path())
            .output()
            .unwrap();

        let state = git_backed_ui_state(&work);
        let app = ui_router(state);
        let req = post_doc_request(
            "edit-me.md",
            serde_json::json!({
                "content": "---\ntitle: New\n---\n\n# New\n",
                "create": false,
                "expected_hash": "not-the-real-hash",
            }),
        );
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let json = body_json(resp).await;
        assert_eq!(json["outcome"], "failed_no_change");
        assert!(json["actual_hash"].as_str().is_some());
    }

    #[tokio::test]
    async fn post_doc_missing_content_field_is_rejected_before_reaching_the_write_pipeline() {
        // A body missing the required `content` field never reaches
        // `post_doc_handler` at all — axum's `Json<T>` extractor rejects it first.
        // That rejection is itself a 422 (a well-formed JSON document that doesn't
        // deserialize into `WriteDocBody`), which is a distinct failure mode from
        // this handler's own `422 {"outcome", "field_errors"}` shape for a
        // frontmatter validation failure — this test only pins down that a
        // malformed request body cannot slip through to write a file.
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        let app = ui_router(test_state(&canonical));
        let req = Request::builder()
            .method("POST")
            .uri("/api/doc/x.md")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::json!({"create": true}).to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert!(!canonical.join("x.md").exists());
    }

    // ------------------------------------------------------------------
    // Include-pattern eligibility guard — regression coverage for the
    // unauthenticated arbitrary-path write/delete finding: every write/delete
    // target must be rejected up front when it does not match the configured
    // `indexing.include` globset, the same rule the MCP write tools enforce.
    // `write::write_document`/`write::delete_document` only re-verify path
    // *traversal* safety, never file-*type* eligibility, so this MUST be
    // checked in the handler before either is ever called.
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn post_doc_create_rejects_a_path_the_include_globset_does_not_match() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        let app = ui_router(test_state(&canonical));
        let req = post_doc_request(
            "notes.txt",
            serde_json::json!({
                "content": "not markdown",
                "create": true,
            }),
        );
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert!(!canonical.join("notes.txt").exists());
    }

    #[tokio::test]
    async fn post_doc_edit_rejects_a_dotgit_target_without_touching_it() {
        // The concrete exploit this guards against: an unauthenticated
        // `create: false` edit of `.git/config` would otherwise overwrite that
        // file's bytes directly via `tokio::fs::write`, before
        // `commit_and_sync` ever runs its own `git add`/`commit` sequence
        // against the same repo — a well-known local code-execution primitive
        // (hostile `core.fsmonitor`/`core.sshCommand`), reachable with no
        // content validation at all since `.git/` files aren't governed by any
        // frontmatter schema.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let git_config_path = work.path().join(".git").join("config");
        let original = std::fs::read_to_string(&git_config_path).unwrap();

        let state = git_backed_ui_state(&work);
        let app = ui_router(state);
        let req = post_doc_request(
            ".git/config",
            serde_json::json!({
                "content": "[core]\n\tsshCommand = \"evil\"\n",
                "create": false,
            }),
        );
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert_eq!(
            std::fs::read_to_string(&git_config_path).unwrap(),
            original,
            ".git/config must be byte-for-byte unchanged"
        );
    }

    #[tokio::test]
    async fn post_doc_create_rejects_a_dotgit_target() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");

        let state = git_backed_ui_state(&work);
        let app = ui_router(state);
        let req = post_doc_request(
            ".git/hooks/pre-commit",
            serde_json::json!({
                "content": "#!/bin/sh\necho pwned\n",
                "create": true,
            }),
        );
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert!(!work.path().join(".git/hooks/pre-commit").exists());
    }

    #[tokio::test]
    async fn post_doc_create_rejects_a_traversal_path_with_no_validation_attempt() {
        // G1: `GlobSet::is_match` (the include-pattern check `post_doc_handler`'s
        // create branch runs) accepts `..` segments as plain characters —
        // `**/*.md` matches `../escape.md` — so the create branch has no
        // traversal check of its own; it relies entirely on
        // `write::write_document` rejecting the path before validation (and
        // before any configured `validation.lint_command` execs against it).
        // Covered here via the percent-encoded form, which axum's path
        // extractor decodes back to a literal `..` component before this
        // handler ever sees it.
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        let app = ui_router(test_state(&canonical));
        let req = post_doc_request(
            "..%2Fescape.md",
            serde_json::json!({
                "content": "---\ntitle: T\n---\n\n# Body\n",
                "create": true,
            }),
        );
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert!(!dir.path().parent().unwrap().join("escape.md").exists());
    }

    #[tokio::test]
    async fn post_doc_create_rejects_a_raw_traversal_path() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        let app = ui_router(test_state(&canonical));
        let req = post_doc_request(
            "../escape.md",
            serde_json::json!({
                "content": "---\ntitle: T\n---\n\n# Body\n",
                "create": true,
            }),
        );
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert!(!dir.path().parent().unwrap().join("escape.md").exists());
    }

    #[tokio::test]
    async fn post_doc_resolve_error_hides_absolute_path_in_response_body() {
        // Concrete trigger: `existing.md/x` treats an existing *file* as a path
        // component, so `canonicalize()` fails with `NotADirectory` — not
        // `NotFound` or `PermissionDenied`, so `resolve_within_data` returns
        // `ResolveErr::Other`, whose message embeds the canonicalized absolute
        // path (e.g. `/tmp/.../existing.md/x`). The response body must not leak
        // that path or the data-directory prefix (F4).
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        std::fs::write(canonical.join("existing.md"), "hello").unwrap();

        let app = ui_router(test_state(&canonical));
        let req = post_doc_request(
            "existing.md/x",
            serde_json::json!({
                "content": "new content",
                "create": false,
            }),
        );
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = body_json(resp).await;
        let body_str = body.to_string();
        assert!(
            !body_str.contains(canonical.to_str().unwrap()),
            "response body leaked the data directory path: {body_str}"
        );
        assert!(
            !body_str.contains("existing.md/x"),
            "response body should not embed the resolved absolute path: {body_str}"
        );
        assert_eq!(body["error"], "internal error resolving path");
    }

    #[tokio::test]
    async fn delete_doc_resolve_error_hides_absolute_path_in_response_body() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        std::fs::write(canonical.join("existing.md"), "hello").unwrap();

        let app = ui_router(test_state(&canonical));
        let req = Request::builder()
            .method("DELETE")
            .uri("/api/doc/existing.md/x")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::json!({}).to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = body_json(resp).await;
        let body_str = body.to_string();
        assert!(
            !body_str.contains(canonical.to_str().unwrap()),
            "response body leaked the data directory path: {body_str}"
        );
        assert_eq!(body["error"], "internal error resolving path");
    }

    #[tokio::test]
    async fn delete_doc_rejects_a_dotgit_target_without_touching_it() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let git_config_path = work.path().join(".git").join("config");
        assert!(git_config_path.exists());

        let state = git_backed_ui_state(&work);
        let app = ui_router(state);
        let req = Request::builder()
            .method("DELETE")
            .uri("/api/doc/.git/config")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::json!({}).to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert!(git_config_path.exists(), ".git/config must not be deleted");
    }

    #[tokio::test]
    async fn delete_doc_rejects_non_markdown_target() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        std::fs::write(canonical.join("notes.txt"), "hello").unwrap();

        let app = ui_router(test_state(&canonical));
        let req = Request::builder()
            .method("DELETE")
            .uri("/api/doc/notes.txt")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::json!({}).to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert!(canonical.join("notes.txt").exists());
    }

    #[tokio::test]
    async fn delete_doc_round_trip_and_missing_reports_conflict() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::write(
            work.path().join("doomed.md"),
            "---\ntitle: D\n---\n\n# Body\n",
        )
        .unwrap();
        std::process::Command::new("git")
            .args(["add", "--", "doomed.md"])
            .current_dir(work.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args([
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "commit",
                "-m",
                "add doomed.md",
            ])
            .current_dir(work.path())
            .output()
            .unwrap();

        let state = git_backed_ui_state(&work);

        let app = ui_router(state.clone());
        let req = Request::builder()
            .method("DELETE")
            .uri("/api/doc/doomed.md")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::json!({"commit_message": "docs: delete doomed.md"}).to_string(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["outcome"], "synced");
        assert!(!work.path().join("doomed.md").exists());

        // Deleting it again reports the missing-document conflict.
        let app = ui_router(state);
        let req = Request::builder()
            .method("DELETE")
            .uri("/api/doc/doomed.md")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::json!({}).to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn doc_write_routes_enforce_the_10mb_body_limit() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        let app = ui_router(test_state(&canonical));

        let oversized = vec![b'x'; MAX_WRITE_BODY_BYTES + 1];
        let req = Request::builder()
            .method("POST")
            .uri("/api/doc/big.md")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(oversized))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn get_doc_route_is_unaffected_by_the_write_body_limit() {
        // Sanity check that merging the write routes' `DefaultBodyLimit` layer
        // onto the shared `/api/doc/{*path}` path didn't disturb the existing GET
        // handler wired to the same route template.
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        std::fs::write(canonical.join("a.md"), "---\ntitle: A\n---\nbody").unwrap();
        let app = ui_router(test_state(&canonical));
        let req = Request::builder()
            .uri("/api/doc/a.md")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
