use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, RwLock};
use tracing::{info, warn};

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub source: SourceConfig,
    #[serde(default)]
    pub indexing: IndexingConfig,
    #[serde(default)]
    pub frontmatter: FrontmatterConfig,
    #[serde(default)]
    pub chunking: ChunkingConfig,
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub validation: ValidationConfig,
    #[serde(default)]
    pub webhook: WebhookConfig,
    #[serde(default)]
    pub mcp: McpConfig,
    #[serde(default)]
    pub rate_limit: RateLimitConfig,
    #[serde(default)]
    pub write: WriteConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub reranking: RerankingConfig,
    #[serde(default)]
    pub ui: UiConfig,
}

/// `source` — YAML side. Every setting is either a secret name-indirection field
/// (`git_token_env`, unaffected by the ENV/YAML split since it never held a secret
/// value itself) or moved to ENV-only. `git_url`, `branch`, and `data_path` used to
/// live here with YAML defaults; they are bootstrap bindings ("what repo, what
/// path") that cannot change without a restart, so they now come exclusively from
/// `GIT_URL` / `GIT_BRANCH` / `DATA_PATH` — see [`ResolvedSourceConfig`]. Setting any
/// of them here now fails loudly via `deny_unknown_fields` rather than being
/// silently ignored.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceConfig {
    /// Name of the env var containing the personal access token for git fetch
    /// over HTTPS.
    #[serde(default = "default_git_token_env")]
    pub git_token_env: String,
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self {
            git_token_env: default_git_token_env(),
        }
    }
}

/// `source` — resolved side. `git_url`/`branch`/`data_path` are read once from
/// `GIT_URL` / `GIT_BRANCH` / `DATA_PATH` in [`Config::resolve`] and carried here;
/// `git_token_env` just passes through from [`SourceConfig`] unchanged (it is a
/// secret name-indirection field, read lazily at each use site via
/// `std::env::var(&config.source.git_token_env)`, not a value itself).
#[derive(Debug, Clone)]
pub struct ResolvedSourceConfig {
    pub git_url: Option<String>,
    pub branch: String,
    /// Path to the knowledge base root (defaults to /data in Docker)
    pub data_path: Option<String>,
    pub git_token_env: String,
}

impl Default for ResolvedSourceConfig {
    fn default() -> Self {
        Self {
            git_url: None,
            branch: default_branch(),
            data_path: default_data_path(),
            git_token_env: default_git_token_env(),
        }
    }
}

impl ResolvedSourceConfig {
    /// True when no git remote is configured (`GIT_URL` unset).
    ///
    /// This is a legitimate, deliberate configuration — a bind-mount-only
    /// deployment provides the knowledge base directly and never wants a clone or
    /// a webhook pull. But `git_url` being `None` is ALSO exactly what a
    /// deployment produces if `GIT_URL` is dropped by accident during a config
    /// migration: the server starts fine, keeps serving whatever is already at
    /// the data path, and every fetch/merge call that gates on `git_url` being
    /// `Some` becomes a permanent, silent no-op — no error, anywhere. Pulled out
    /// as its own predicate (rather than inlined at the log call site) so the
    /// condition can be unit-tested without capturing tracing output.
    pub fn git_integration_disabled(&self) -> bool {
        self.git_url.is_none()
    }
}

fn default_branch() -> String {
    "master".into()
}

fn default_git_token_env() -> String {
    "GIT_PULL_TOKEN".into()
}

fn default_data_path() -> Option<String> {
    Some("/data".into())
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IndexingConfig {
    #[serde(default = "default_include")]
    pub include: Vec<String>,
    #[serde(default = "default_exclude")]
    pub exclude: Vec<String>,
    #[serde(default = "default_exclude_files")]
    pub exclude_files: Vec<String>,
    /// How often (in seconds) the background reindex worker runs a full reconcile
    /// sweep (`ingest::scan_for_dirty`), independent of anything a write, webhook, or
    /// startup explicitly marked dirty.
    ///
    /// This is a backstop for LOST events — the process died mid-run, a webhook was
    /// never delivered, Qdrant was unavailable when a write tried to mark its path —
    /// not the indexing interval. The worker is event-driven and wakes on `Notify`
    /// immediately when a write commits or a webhook lands, so ordinary post-write
    /// index latency is near-instant and does not depend on this value at all. This
    /// interval only bounds how long a *lost* event can go unnoticed before the next
    /// full sweep rediscovers it, which is why an interval measured in minutes (not
    /// seconds) is the right cost/benefit trade at this project's scale: a full sweep
    /// pages through `indexed_files` and stats each file it already knows about
    /// (`StateDb::fetch_indexed_files_page`), not full content hashing, so its cost
    /// grows with corpus size, not with how often it runs.
    #[serde(default = "default_reconcile_interval_secs")]
    pub reconcile_interval_secs: u64,
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            include: default_include(),
            exclude: default_exclude(),
            exclude_files: default_exclude_files(),
            reconcile_interval_secs: default_reconcile_interval_secs(),
        }
    }
}

fn default_include() -> Vec<String> {
    vec!["**/*.md".into()]
}

fn default_exclude() -> Vec<String> {
    vec![
        ".git/**".into(),
        ".claude/**".into(),
        ".tools/**".into(),
        "node_modules/**".into(),
    ]
}

fn default_exclude_files() -> Vec<String> {
    vec!["CLAUDE.md".into(), "README.md".into()]
}

fn default_reconcile_interval_secs() -> u64 {
    600
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct FrontmatterConfig {
    #[serde(default)]
    pub required: Vec<String>,
    #[serde(default)]
    pub indexed_fields: Vec<String>,
    #[serde(default)]
    pub defaults: HashMap<String, String>,
    /// Maps a frontmatter field name to its closed set of allowed values.
    /// If a field is present in a document but its value is not in the set,
    /// validation fails. Absent fields are not checked (use `required` for that).
    #[serde(default)]
    pub allowed: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkingConfig {
    #[serde(default = "default_max_chunk_size")]
    pub max_chunk_size: usize,
    /// Target chunk size — accumulate markdown sections up to this size.
    /// Defaults to max_chunk_size (i.e. fill chunks as much as possible).
    #[serde(default = "default_target_chunk_size")]
    pub target_chunk_size: Option<usize>,
    #[serde(default = "default_true")]
    pub prepend_description: bool,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: default_max_chunk_size(),
            target_chunk_size: default_target_chunk_size(),
            prepend_description: true,
        }
    }
}

impl ChunkingConfig {
    pub fn target(&self) -> usize {
        self.target_chunk_size.unwrap_or(self.max_chunk_size)
    }
}

fn default_max_chunk_size() -> usize {
    1500
}

fn default_target_chunk_size() -> Option<usize> {
    Some(1000)
}

/// `embedding` — YAML side. `base_url`, `model`, `vector_size` are connection wiring
/// and model identity: they cannot change without a restart (the embedding client
/// and the Qdrant collection's vector size are both fixed at startup), so they are
/// ENV-only (`EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` / `EMBEDDING_VECTOR_SIZE`) and
/// do not appear here — see [`ResolvedEmbeddingConfig`]. The API key is a secret, so
/// it follows the same name-indirection pattern as `source.git_token_env`:
/// `api_key_env` names the env var, never the key itself. `batch_size`,
/// `request_timeout_secs`, and `batch_concurrency` are pure tuning knobs and stay
/// YAML-only with no env override at all.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingConfig {
    /// Name of the env var containing the API key for the embedding provider.
    /// Required for OpenAI, hosted Ollama, or any authenticated embedding service.
    /// Leave the env var unset for local/unauthenticated servers (e.g. bundled
    /// llama.cpp) — an unset var is not an error, it just means no key is sent.
    #[serde(default = "default_embedding_api_key_env")]
    pub api_key_env: String,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Wall-clock timeout for a single embedding HTTP request. A tuning knob, not a
    /// startup-required connection setting, so (per project convention) this is
    /// YAML-only with no env var override.
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
    /// Max number of embedding batches sent concurrently. A tuning knob — YAML-only,
    /// no env var override.
    #[serde(default = "default_batch_concurrency")]
    pub batch_concurrency: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            api_key_env: default_embedding_api_key_env(),
            batch_size: default_batch_size(),
            request_timeout_secs: default_request_timeout_secs(),
            batch_concurrency: default_batch_concurrency(),
        }
    }
}

fn default_embedding_api_key_env() -> String {
    "EMBEDDING_API_KEY".into()
}

fn default_vector_size() -> u64 {
    768
}

fn default_batch_size() -> usize {
    32
}

fn default_request_timeout_secs() -> u64 {
    // Embedding a batch of 32 chunks legitimately takes several seconds on the
    // deployed CPU-only embeddings service, so this needs to be generous rather
    // than a typical short API timeout (compare rerank.rs's 10s for a single
    // lightweight rerank call). A hung connection that never errors is still
    // caught once this elapses, at which point it becomes a retryable error via
    // the exponential backoff in embed.rs instead of blocking forever.
    60
}

fn default_batch_concurrency() -> usize {
    // The deployed kb-embeddings service (llama.cpp) runs with `--parallel 2`, so
    // concurrency beyond 2 in-flight batches mostly queues server-side today rather
    // than speeding anything up. 4 is a compromise: it doesn't leave throughput on
    // the table if the service later moves to GPU with more parallel slots, and it
    // costs nothing extra when the server is the bottleneck since the excess
    // requests just queue rather than error.
    4
}

/// `reranking` — YAML side. `enabled` and `candidate_limit` are pure tuning knobs
/// and stay YAML-only with NO env override — `RERANKING_ENABLED` /
/// `RERANKING_CANDIDATE_LIMIT` used to silently override these (the incident this
/// migration fixes: a deployed `RERANKING_CANDIDATE_LIMIT` env var made a YAML
/// change a no-op), so both env vars are now recognized-but-unhonored (see
/// [`DEPRECATED_ENV_VARS`]). `base_url`/`model` are connection wiring and model
/// identity — ENV-only, same reasoning as `embedding.base_url`/`model` — so they
/// live on [`ResolvedRerankingConfig`] instead. `api_key_env` follows the same
/// secret name-indirection pattern as `embedding.api_key_env`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RerankingConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Name of the env var containing the API key for the reranking provider.
    /// Same semantics as `embedding.api_key_env`: unset means no key is sent.
    #[serde(default = "default_reranking_api_key_env")]
    pub api_key_env: String,
    #[serde(default = "default_reranking_candidate_limit")]
    pub candidate_limit: usize,
}

impl Default for RerankingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            api_key_env: default_reranking_api_key_env(),
            candidate_limit: default_reranking_candidate_limit(),
        }
    }
}

fn default_reranking_api_key_env() -> String {
    "RERANKING_API_KEY".into()
}

fn default_reranking_candidate_limit() -> usize {
    50
}

/// Resolved reranking config — only present when reranking is enabled and all required fields are set.
#[derive(Debug, Clone)]
pub struct ResolvedRerankingConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub candidate_limit: usize,
    /// Per-document byte budget for the rerank request, **derived** from
    /// `chunking.max_chunk_size` — deliberately not a knob of its own. The
    /// reranker is sent chunk text, so the largest document it can be asked to
    /// score is exactly what the chunker is configured to emit; one number
    /// governs both ends and they cannot drift apart.
    ///
    /// This exists because the reranker rejects the *whole request* when any
    /// single document exceeds its physical batch size (llama.cpp's
    /// `--ubatch-size`, 512 tokens by default), which cost every candidate its
    /// reranking over one long chunk (#128). Note this is a byte budget, not a
    /// token count — see `truncate_for_rerank` in `rerank.rs`.
    pub max_document_bytes: usize,
}

fn default_collection() -> String {
    "knowledge-base".into()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub strict: bool,
    pub lint_command: Option<Vec<String>>,
    /// Wall-clock timeout for a single `lint_command` invocation. Validation runs
    /// inside `ingest::index_paths`, which is driven one file at a time by the
    /// single background reindex worker (`reindex::run_worker`) — an operator-
    /// configured lint script that hangs (waiting on stdin, a network call with no
    /// timeout of its own, a misconfigured binary) would otherwise stall that
    /// worker forever with no crash and no log line, silently taking indexing down
    /// for the whole KB (#146). Same rationale as `embedding.request_timeout_secs`,
    /// but a much shorter default: lint commands are typically fast static checks
    /// (markdownlint, a small custom script) with nothing like the embedding
    /// service's legitimate multi-second batch latency, so 30s is already generous
    /// for real work while still catching a hang promptly. YAML-only, no env
    /// override — a pure tuning knob like its `validation.*` neighbours.
    #[serde(default = "default_lint_timeout_secs")]
    pub lint_timeout_secs: u64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            strict: false,
            lint_command: None,
            lint_timeout_secs: default_lint_timeout_secs(),
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_lint_timeout_secs() -> u64 {
    30
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum WebhookProvider {
    #[default]
    Gitea,
    Github,
    Gitlab,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WebhookConfig {
    #[serde(default = "default_webhook_secret_env")]
    pub secret_env: String,
    #[serde(default)]
    pub provider: WebhookProvider,
}

impl Default for WebhookConfig {
    fn default() -> Self {
        Self {
            secret_env: "WEBHOOK_SECRET".into(),
            provider: WebhookProvider::default(),
        }
    }
}

fn default_webhook_secret_env() -> String {
    "WEBHOOK_SECRET".into()
}

/// `mcp` — YAML side. `port` is a bootstrap binding (the process cannot change
/// which port it is listening on without a restart), so it moved to `MCP_PORT` —
/// see [`ResolvedMcpConfig`]. Everything else here is genuine runtime/tuning
/// behaviour and stays YAML-only.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct McpConfig {
    #[serde(default = "default_bearer_token_env")]
    pub bearer_token_env: String,
    #[serde(default)]
    pub allow_unauthenticated: bool,
    /// Custom narrative for MCP server instructions.
    ///
    /// Deprecated: replaced by a `server.md` file under `mcp.extensions_path`
    /// in the served knowledge base, which is editable through `write_document`
    /// without a restart. Still honored when set — logged once at startup as
    /// deprecated — but a `server.md` extension file, if present, wins.
    pub instructions: Option<String>,
    /// How often (in seconds) to refresh discovered metadata from Qdrant.
    #[serde(default = "default_metadata_refresh_secs")]
    pub metadata_refresh_secs: u64,
    /// Hostnames accepted in the inbound `Host` header. Entries may include a
    /// port, e.g. `["kb.example.com", "kb.example.com:8001"]`.
    ///
    /// Empty (the default) disables the check and accepts any `Host`. Most
    /// deployments want that: the check guards against DNS rebinding, where a
    /// browser is tricked into treating this server as same-origin, and a
    /// bearer-authenticated server already refuses those requests at auth. It
    /// earns its keep mainly when `allow_unauthenticated` is set.
    ///
    /// Behind a reverse proxy, list the public hostname clients use rather than
    /// the container's address — the proxy forwards the original `Host`.
    #[serde(default)]
    pub allowed_hosts: Vec<String>,
    /// Where per-KB MCP tool/server description extensions live: `server.md`
    /// and `tools/<tool>.md`, appended to the compiled description after any
    /// config-derived sentences — see `descriptions.rs`.
    ///
    /// A RELATIVE value (the default, `"meta/mcp"`) resolves against
    /// `source.data_path`, the served knowledge base root — which is what
    /// makes these files editable through `write_document`. An ABSOLUTE value
    /// is accepted but is not reachable through `write_document`; this is
    /// logged once at startup. Empty disables extension loading entirely.
    #[serde(default = "default_extensions_path")]
    pub extensions_path: String,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            bearer_token_env: default_bearer_token_env(),
            allow_unauthenticated: false,
            instructions: None,
            metadata_refresh_secs: default_metadata_refresh_secs(),
            allowed_hosts: Vec::new(),
            extensions_path: default_extensions_path(),
        }
    }
}

/// `mcp` — resolved side. `port` is read once from `MCP_PORT` in
/// [`Config::resolve`]; every other field passes through from [`McpConfig`]
/// unchanged.
#[derive(Debug, Clone)]
pub struct ResolvedMcpConfig {
    pub port: u16,
    pub bearer_token_env: String,
    pub allow_unauthenticated: bool,
    pub instructions: Option<String>,
    pub metadata_refresh_secs: u64,
    pub allowed_hosts: Vec<String>,
    pub extensions_path: String,
}

impl Default for ResolvedMcpConfig {
    fn default() -> Self {
        Self {
            port: default_mcp_port(),
            bearer_token_env: default_bearer_token_env(),
            allow_unauthenticated: false,
            instructions: None,
            metadata_refresh_secs: default_metadata_refresh_secs(),
            allowed_hosts: Vec::new(),
            extensions_path: default_extensions_path(),
        }
    }
}

fn default_metadata_refresh_secs() -> u64 {
    300
}

fn default_extensions_path() -> String {
    "meta/mcp".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RateLimitConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_rate_limit_per_second")]
    pub per_second: u64,
    #[serde(default = "default_rate_limit_burst_size")]
    pub burst_size: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            per_second: default_rate_limit_per_second(),
            burst_size: default_rate_limit_burst_size(),
        }
    }
}

fn default_rate_limit_per_second() -> u64 {
    20
}

fn default_rate_limit_burst_size() -> u32 {
    50
}

fn default_mcp_port() -> u16 {
    8001
}

/// Configuration for write-tool behaviour (create_document / edit_document).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WriteConfig {
    /// If true, creating a new document runs a similarity check against the
    /// existing collection and refuses if a near-duplicate exists.
    #[serde(default = "default_dedup_enabled")]
    pub dedup_enabled: bool,
    /// Cosine similarity at or above which a new document is treated as a
    /// duplicate and the write is refused (unless `force_new = true`).
    ///
    /// Always compared against a *dense* cosine score: the dedup search is
    /// pinned to dense-only with the reranker detached (see
    /// `mcp::dedup_search_opts`), so this value is independent of
    /// `search.hybrid` — whose RRF scores top out near 0.03 — and of
    /// `reranking.enabled`, whose relevance scores are not similarities.
    #[serde(default = "default_dedup_threshold")]
    pub dedup_threshold: f32,
    /// Git author name used for commits made by the write tools.
    /// Distinguishes tool-authored commits from hand-made commits.
    #[serde(default = "default_commit_author_name")]
    pub commit_author_name: String,
    /// Git author email used for commits made by the write tools.
    /// Distinguishes tool-authored commits from hand-made commits.
    #[serde(default = "default_commit_author_email")]
    pub commit_author_email: String,
}

fn default_dedup_enabled() -> bool {
    true
}

fn default_dedup_threshold() -> f32 {
    0.80
}

fn default_commit_author_name() -> String {
    "md-kb-rag".to_string()
}

fn default_commit_author_email() -> String {
    "md-kb-rag@localhost".to_string()
}

impl Default for WriteConfig {
    fn default() -> Self {
        Self {
            dedup_enabled: default_dedup_enabled(),
            dedup_threshold: default_dedup_threshold(),
            commit_author_name: default_commit_author_name(),
            commit_author_email: default_commit_author_email(),
        }
    }
}

fn default_bearer_token_env() -> String {
    "MCP_BEARER_TOKEN".into()
}

/// Configuration for retrieval (search) behaviour.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchConfig {
    /// Enable hybrid sparse+dense retrieval with RRF fusion. When `false`, search
    /// uses the dense vector only (legacy behaviour). Toggling this does NOT require
    /// a reindex — both named vectors are always stored.
    #[serde(default = "default_true")]
    pub hybrid: bool,
    /// Number of candidates fetched from each arm (dense + sparse) before RRF fusion.
    /// Higher values improve recall at the cost of latency.
    #[serde(default = "default_rrf_candidates")]
    pub rrf_candidates: usize,
    /// Enable exact-phrase matching for double-quoted spans in a query (e.g.
    /// `"node:ares"`). Adds a phrase-filtered prefetch arm fused via the same RRF
    /// as dense/sparse, independent of `hybrid`. Requires a Qdrant server new
    /// enough to support `phrase_matching` text indexes — `ensure_collection`
    /// degrades gracefully (logs and disables phrase matching for the process)
    /// when an older server rejects it, so this never fails startup. When
    /// `false`, quoted text is treated as literal characters and no phrase index
    /// is created.
    #[serde(default = "default_true")]
    pub phrase: bool,
    /// Global minimum relevance score floor. Results below this threshold are
    /// dropped before returning. `None` (the default) disables the floor.
    /// Note: RRF scores are ~0.01–0.03 — set accordingly when hybrid is true.
    #[serde(default)]
    pub min_score: Option<f32>,
    /// Per-document result diversity: the maximum number of chunks from a single
    /// document allowed to occupy the final result set (see `retrieval::search`'s
    /// diversity-cap comment for the full funnel-placement reasoning). Chunks
    /// beyond the cap are dropped in favor of the next-best chunk from a
    /// different document, so this bounds monopolization without discarding a
    /// document's top N chunks when several genuinely are the best answer.
    ///
    /// `None` disables diversity entirely, restoring historical behaviour (a
    /// single document may fill every result slot) — same `Option` convention
    /// as `min_score` above: set explicit `null` in YAML to opt out. Defaults to
    /// `Some(3)` rather than `None`: issue #86 established that one prolific
    /// document silently crowding out the rest of the corpus is a real,
    /// observed failure mode, so diversity ships on by default with a value
    /// conservative enough (3 of a default 10-result page) to preserve
    /// legitimate multi-chunk relevance without obviously degrading results
    /// that were never a monoculture in the first place.
    #[serde(default = "default_diversity_max_per_document")]
    pub diversity_max_per_document: Option<usize>,
    /// Results returned when a caller does not specify `limit`.
    #[serde(default = "default_search_limit")]
    pub default_limit: u64,
    /// Ceiling on `limit`. A larger request is clamped to this rather than
    /// rejected, so a caller asking for more simply gets the maximum.
    ///
    /// This bounds MCP response size, not retrieval cost: candidate depth is
    /// governed by `rrf_candidates` and `reranking.candidate_limit`, which are
    /// unaffected by how many of those candidates are ultimately returned.
    /// Raising it makes responses larger, not searches slower.
    #[serde(default = "default_max_search_limit")]
    pub max_limit: u64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            hybrid: true,
            rrf_candidates: default_rrf_candidates(),
            phrase: default_true(),
            min_score: None,
            diversity_max_per_document: default_diversity_max_per_document(),
            default_limit: default_search_limit(),
            max_limit: default_max_search_limit(),
        }
    }
}

fn default_rrf_candidates() -> usize {
    50
}

fn default_diversity_max_per_document() -> Option<usize> {
    Some(3)
}

fn default_search_limit() -> u64 {
    10
}

fn default_max_search_limit() -> u64 {
    50
}

/// `ui` — pure YAML tuning knobs for the knowledge-base web UI (issue #53). No
/// secrets, no bootstrap wiring, so unlike `source`/`embedding` there is no
/// env-only split and no `Resolved*` counterpart: the parsed struct is copied
/// straight onto `ResolvedConfig`, same as `rate_limit` and `search`.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct UiConfig {
    #[serde(default)]
    pub semantic_edges: SemanticEdgesConfig,
}

/// Precomputed semantic (kNN) graph edges shown alongside markdown-link edges
/// in the web UI's graph view. Off by default: computing them costs a Qdrant
/// `recommend` query per indexed document on every run.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticEdgesConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Number of nearest-neighbor edges to keep per document (after dedup and
    /// the `min_score` filter), highest score first.
    #[serde(default = "default_semantic_edges_k")]
    pub k: u64,
    /// Minimum cosine similarity for a neighbor to be kept as an edge.
    #[serde(default = "default_semantic_edges_min_score")]
    pub min_score: f32,
}

impl Default for SemanticEdgesConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            k: default_semantic_edges_k(),
            min_score: default_semantic_edges_min_score(),
        }
    }
}

fn default_semantic_edges_k() -> u64 {
    5
}

fn default_semantic_edges_min_score() -> f32 {
    0.6
}

/// Resolved embedding config — all required fields are guaranteed present.
#[derive(Debug, Clone)]
pub struct ResolvedEmbeddingConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub vector_size: u64,
    pub batch_size: usize,
    pub request_timeout_secs: u64,
    pub batch_concurrency: usize,
}

/// Resolved Qdrant config — `url` is guaranteed present.
#[derive(Debug, Clone)]
pub struct ResolvedQdrantConfig {
    pub url: String,
    pub collection: String,
}

/// Where a single resolved setting's value came from. Never carries a secret
/// VALUE — for the `*_env` indirection fields (`embedding.api_key`,
/// `reranking.api_key`) this names the env var that was read, not its contents.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum SettingSource {
    /// Read from the named environment variable.
    Env { var: String },
    /// Read from the YAML config file (the setting's top-level section was present
    /// in the parsed document).
    Yaml,
    /// Neither ENV nor the YAML file supplied a value; the built-in default applies
    /// (or, for an optional secret like an API key, none was configured at all).
    Default,
}

impl SettingSource {
    /// Human-readable form for logs and `/status`, e.g. `"env EMBEDDING_BASE_URL"`.
    pub fn describe(&self) -> String {
        match self {
            SettingSource::Env { var } => format!("env {var}"),
            SettingSource::Yaml => "yaml".to_string(),
            SettingSource::Default => "default".to_string(),
        }
    }
}

/// Provenance of every resolved setting, keyed by dotted config path (e.g.
/// `"embedding.base_url"`). Built once in [`Config::resolve`] and carried on
/// [`ResolvedConfig`] so the startup log and `/status` can both report it without
/// re-deriving anything — this is what would have made `RERANKING_CANDIDATE_LIMIT`
/// silently overriding YAML obvious immediately instead of requiring a source read.
///
/// YAML-only settings are tracked at *section* granularity: a setting reports
/// `Yaml` if its top-level YAML section (e.g. `chunking:`) was present in the
/// parsed document, even if that specific leaf relied on its own default within
/// the section. Full leaf-level precision would need a second walk of the parsed
/// document per field; section granularity answers the practically useful question
/// — "is my config.yaml being read at all for this area" — far more cheaply.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct ConfigProvenance(pub BTreeMap<String, SettingSource>);

impl ConfigProvenance {
    /// Log one INFO line summarizing every setting's source. Never logs a secret
    /// VALUE — only which source (env var name, yaml, or default) supplied it.
    pub fn log(&self) {
        use std::fmt::Write as _;
        let mut lines = String::from("Configuration provenance (source of each resolved setting):");
        for (name, source) in &self.0 {
            let _ = write!(lines, "\n  {name}: {}", source.describe());
        }
        info!("{lines}");
    }
}

/// Env var names this project used to honor as overrides and no longer does. A
/// deployment that still sets one of these silently stops applying it — this list
/// is the safety net: [`Config::resolve`] warns once at startup for each one found
/// set, naming it explicitly, rather than leaving the drift to be discovered by
/// reading source.
///
/// `RERANKING_ENABLED` and `RERANKING_CANDIDATE_LIMIT` are pure tuning knobs that
/// moved to YAML-only (`reranking.enabled` / `reranking.candidate_limit`) — this is
/// exactly the incident that motivated this list: a deployed
/// `RERANKING_CANDIDATE_LIMIT` silently overrode a YAML change until someone read
/// the source to find out why.
pub const DEPRECATED_ENV_VARS: &[&str] = &["RERANKING_ENABLED", "RERANKING_CANDIDATE_LIMIT"];

/// Which of [`DEPRECATED_ENV_VARS`] are currently set in the process environment.
/// Pure function (no logging) so tests can assert on detection without capturing
/// tracing output.
fn deprecated_env_vars_present() -> Vec<&'static str> {
    DEPRECATED_ENV_VARS
        .iter()
        .copied()
        .filter(|name| std::env::var(name).is_ok())
        .collect()
}

/// Dotted setting name → the top-level YAML section it lives under, for every
/// YAML-only setting. Drives [`ConfigProvenance`]'s `Yaml`/`Default` split for
/// these fields — see that type's doc comment for the section-granularity caveat.
const YAML_ONLY_SETTINGS: &[(&str, &str)] = &[
    ("source.git_token_env", "source"),
    ("indexing.include", "indexing"),
    ("indexing.exclude", "indexing"),
    ("indexing.exclude_files", "indexing"),
    ("indexing.reconcile_interval_secs", "indexing"),
    ("frontmatter.required", "frontmatter"),
    ("frontmatter.indexed_fields", "frontmatter"),
    ("frontmatter.defaults", "frontmatter"),
    ("frontmatter.allowed", "frontmatter"),
    ("chunking.max_chunk_size", "chunking"),
    ("chunking.target_chunk_size", "chunking"),
    ("chunking.prepend_description", "chunking"),
    ("embedding.api_key_env", "embedding"),
    ("embedding.batch_size", "embedding"),
    ("embedding.request_timeout_secs", "embedding"),
    ("embedding.batch_concurrency", "embedding"),
    ("validation.enabled", "validation"),
    ("validation.strict", "validation"),
    ("validation.lint_command", "validation"),
    ("validation.lint_timeout_secs", "validation"),
    ("webhook.secret_env", "webhook"),
    ("webhook.provider", "webhook"),
    ("mcp.bearer_token_env", "mcp"),
    ("mcp.allow_unauthenticated", "mcp"),
    ("mcp.instructions", "mcp"),
    ("mcp.metadata_refresh_secs", "mcp"),
    ("mcp.allowed_hosts", "mcp"),
    ("mcp.extensions_path", "mcp"),
    ("rate_limit.enabled", "rate_limit"),
    ("rate_limit.per_second", "rate_limit"),
    ("rate_limit.burst_size", "rate_limit"),
    ("write.dedup_enabled", "write"),
    ("write.dedup_threshold", "write"),
    ("write.commit_author_name", "write"),
    ("write.commit_author_email", "write"),
    ("search.hybrid", "search"),
    ("search.rrf_candidates", "search"),
    ("search.phrase", "search"),
    ("search.min_score", "search"),
    ("search.diversity_max_per_document", "search"),
    ("search.default_limit", "search"),
    ("search.max_limit", "search"),
    ("reranking.enabled", "reranking"),
    ("reranking.candidate_limit", "reranking"),
    ("reranking.api_key_env", "reranking"),
    ("ui.semantic_edges.enabled", "ui"),
    ("ui.semantic_edges.k", "ui"),
    ("ui.semantic_edges.min_score", "ui"),
];

/// Every top-level section name [`YAML_ONLY_SETTINGS`] can point at. Used to filter
/// a parsed document's top-level keys down to ones this project actually resolves,
/// so an already-rejected (`deny_unknown_fields`) or renamed section cannot end up
/// looking like a recognized one.
const KNOWN_SECTIONS: &[&str] = &[
    "source",
    "indexing",
    "frontmatter",
    "chunking",
    "embedding",
    "validation",
    "webhook",
    "mcp",
    "rate_limit",
    "write",
    "search",
    "reranking",
    "ui",
];

/// Top-level YAML section keys actually present in `content`, intersected with
/// [`KNOWN_SECTIONS`]. Best-effort: if `content` fails to parse as generic YAML
/// (should not happen — `Config::load` only calls this after already deserializing
/// the same content into a typed `Config`), an empty set is returned, which just
/// makes every YAML-only setting report `Default` provenance rather than failing
/// the whole load over an observability nicety.
fn yaml_top_level_sections(content: &str) -> HashSet<&'static str> {
    let Ok(value) = serde_yaml_ng::from_str::<serde_yaml_ng::Value>(content) else {
        return HashSet::new();
    };
    let Some(mapping) = value.as_mapping() else {
        return HashSet::new();
    };
    KNOWN_SECTIONS
        .iter()
        .copied()
        .filter(|section| mapping.contains_key(serde_yaml_ng::Value::String((*section).into())))
        .collect()
}

/// Fully resolved configuration — all required fields validated and present.
#[derive(Debug, Clone)]
pub struct ResolvedConfig {
    pub source: ResolvedSourceConfig,
    pub indexing: IndexingConfig,
    pub frontmatter: FrontmatterConfig,
    pub chunking: ChunkingConfig,
    pub embedding: ResolvedEmbeddingConfig,
    pub qdrant: ResolvedQdrantConfig,
    pub validation: ValidationConfig,
    pub webhook: WebhookConfig,
    pub mcp: ResolvedMcpConfig,
    pub rate_limit: RateLimitConfig,
    pub write: WriteConfig,
    pub search: SearchConfig,
    pub reranking: Option<ResolvedRerankingConfig>,
    pub ui: UiConfig,
    /// Where every resolved setting's value came from — see [`ConfigProvenance`].
    pub provenance: ConfigProvenance,
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<ResolvedConfig> {
        let (config, present_sections) = if path.exists() {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read config file '{}'", path.display()))?;
            let config: Config = serde_yaml_ng::from_str(&content).with_context(|| {
                format!(
                    "Failed to parse config file '{}'. \
                     If you see 'unknown field', compare your config against \
                     config.example.yaml — fields may have been added or removed.",
                    path.display()
                )
            })?;
            let sections = yaml_top_level_sections(&content);
            (config, sections)
        } else {
            warn!("Config file '{}' not found, using defaults", path.display());
            (Config::default(), HashSet::new())
        };
        config.resolve_inner(&present_sections)
    }

    /// Apply env var overrides and validate required fields.
    ///
    /// Thin wrapper around [`Self::resolve_inner`] with no section-presence
    /// context, so YAML-only settings conservatively report `Default` provenance.
    /// Real provenance precision comes from [`Self::load`], which does track
    /// section presence; this entry point exists for tests that only have a typed
    /// `Config` (no source text) to work with — [`Self::load`] is the only
    /// non-test caller, and it calls `resolve_inner` directly.
    #[cfg(test)]
    fn resolve(self) -> anyhow::Result<ResolvedConfig> {
        self.resolve_inner(&HashSet::new())
    }

    fn resolve_inner(
        self,
        present_sections: &HashSet<&'static str>,
    ) -> anyhow::Result<ResolvedConfig> {
        // Safety net for the migration: a deployment that still sets one of these
        // silently stops applying it, since ENV/YAML are now mutually exclusive per
        // setting and these two were dropped rather than migrated. Named explicitly
        // rather than discovered by reading source, which is the whole point.
        for name in deprecated_env_vars_present() {
            warn!(
                "Environment variable {name} is set but no longer honored (it moved to \
                 YAML-only config — see deploy/config.example.yaml). It is currently a \
                 silent no-op; remove it from your deployment."
            );
        }

        let mut provenance: BTreeMap<&'static str, SettingSource> = BTreeMap::new();

        // ── ENV-only settings ───────────────────────────────────────────────────
        // Connection wiring, model identity, and bootstrap bindings each have
        // exactly one legal source: the environment. None of these fields exist on
        // the YAML-deserializable structs any more, so a leftover YAML key for one
        // of them is already caught loudly by `deny_unknown_fields`; what's left
        // here is reading each var (falling back to its built-in default when
        // optional and unset) and recording where the value came from.

        let embedding_base_url = std::env::var("EMBEDDING_BASE_URL").ok();
        if embedding_base_url.is_some() {
            provenance.insert(
                "embedding.base_url",
                SettingSource::Env {
                    var: "EMBEDDING_BASE_URL".into(),
                },
            );
        }

        let embedding_model = std::env::var("EMBEDDING_MODEL").ok();
        if embedding_model.is_some() {
            provenance.insert(
                "embedding.model",
                SettingSource::Env {
                    var: "EMBEDDING_MODEL".into(),
                },
            );
        }

        let embedding_vector_size = match std::env::var("EMBEDDING_VECTOR_SIZE") {
            Ok(val) => {
                provenance.insert(
                    "embedding.vector_size",
                    SettingSource::Env {
                        var: "EMBEDDING_VECTOR_SIZE".into(),
                    },
                );
                val.parse()
                    .map_err(|_| anyhow::anyhow!("EMBEDDING_VECTOR_SIZE must be a valid integer"))?
            }
            Err(_) => {
                provenance.insert("embedding.vector_size", SettingSource::Default);
                default_vector_size()
            }
        };

        let embedding_api_key_env = self.embedding.api_key_env.clone();
        let embedding_api_key = std::env::var(&embedding_api_key_env).ok();
        provenance.insert(
            "embedding.api_key",
            match &embedding_api_key {
                Some(_) => SettingSource::Env {
                    var: embedding_api_key_env,
                },
                None => SettingSource::Default,
            },
        );

        let qdrant_url = std::env::var("QDRANT_URL").ok();
        if qdrant_url.is_some() {
            provenance.insert(
                "qdrant.url",
                SettingSource::Env {
                    var: "QDRANT_URL".into(),
                },
            );
        }

        let qdrant_collection = match std::env::var("QDRANT_COLLECTION") {
            Ok(val) => {
                provenance.insert(
                    "qdrant.collection",
                    SettingSource::Env {
                        var: "QDRANT_COLLECTION".into(),
                    },
                );
                val
            }
            Err(_) => {
                provenance.insert("qdrant.collection", SettingSource::Default);
                default_collection()
            }
        };

        let reranking_base_url = std::env::var("RERANKING_BASE_URL").ok();
        if reranking_base_url.is_some() {
            provenance.insert(
                "reranking.base_url",
                SettingSource::Env {
                    var: "RERANKING_BASE_URL".into(),
                },
            );
        }

        let reranking_model = std::env::var("RERANKING_MODEL").ok();
        if reranking_model.is_some() {
            provenance.insert(
                "reranking.model",
                SettingSource::Env {
                    var: "RERANKING_MODEL".into(),
                },
            );
        }

        let reranking_api_key_env = self.reranking.api_key_env.clone();
        let reranking_api_key = std::env::var(&reranking_api_key_env).ok();
        provenance.insert(
            "reranking.api_key",
            match &reranking_api_key {
                Some(_) => SettingSource::Env {
                    var: reranking_api_key_env,
                },
                None => SettingSource::Default,
            },
        );

        let source_git_url = std::env::var("GIT_URL").ok();
        provenance.insert(
            "source.git_url",
            match &source_git_url {
                Some(_) => SettingSource::Env {
                    var: "GIT_URL".into(),
                },
                None => SettingSource::Default,
            },
        );

        let source_branch = match std::env::var("GIT_BRANCH") {
            Ok(val) => {
                provenance.insert(
                    "source.branch",
                    SettingSource::Env {
                        var: "GIT_BRANCH".into(),
                    },
                );
                val
            }
            Err(_) => {
                provenance.insert("source.branch", SettingSource::Default);
                default_branch()
            }
        };

        let source_data_path = match std::env::var("DATA_PATH") {
            Ok(val) => {
                provenance.insert(
                    "source.data_path",
                    SettingSource::Env {
                        var: "DATA_PATH".into(),
                    },
                );
                Some(val)
            }
            Err(_) => {
                provenance.insert("source.data_path", SettingSource::Default);
                default_data_path()
            }
        };

        let mcp_port = match std::env::var("MCP_PORT") {
            Ok(val) => {
                provenance.insert(
                    "mcp.port",
                    SettingSource::Env {
                        var: "MCP_PORT".into(),
                    },
                );
                val.parse().map_err(|_| {
                    anyhow::anyhow!("MCP_PORT must be a valid port number (0-65535)")
                })?
            }
            Err(_) => {
                provenance.insert("mcp.port", SettingSource::Default);
                default_mcp_port()
            }
        };

        // ── YAML-only settings ──────────────────────────────────────────────────
        // See `YAML_ONLY_SETTINGS`'s doc comment for the section-granularity caveat.
        for (name, section) in YAML_ONLY_SETTINGS {
            let source = if present_sections.contains(section) {
                SettingSource::Yaml
            } else {
                SettingSource::Default
            };
            provenance.insert(*name, source);
        }

        // Validate chunk size config
        if let Some(target) = self.chunking.target_chunk_size
            && target > self.chunking.max_chunk_size
        {
            anyhow::bail!(
                "chunking.target_chunk_size ({}) must be <= chunking.max_chunk_size ({})",
                target,
                self.chunking.max_chunk_size
            );
        }

        // Validate lower bounds
        if embedding_vector_size == 0 {
            anyhow::bail!("embedding.vector_size must be >= 1");
        }
        if self.embedding.batch_size == 0 {
            anyhow::bail!("embedding.batch_size must be >= 1");
        }
        if self.embedding.request_timeout_secs == 0 {
            anyhow::bail!("embedding.request_timeout_secs must be >= 1");
        }
        if self.validation.lint_timeout_secs == 0 {
            anyhow::bail!("validation.lint_timeout_secs must be >= 1");
        }
        if self.embedding.batch_concurrency == 0 {
            anyhow::bail!("embedding.batch_concurrency must be >= 1");
        }
        if self.chunking.max_chunk_size == 0 {
            anyhow::bail!("chunking.max_chunk_size must be >= 1");
        }
        if self.rate_limit.per_second == 0 {
            anyhow::bail!("rate_limit.per_second must be >= 1");
        }
        if self.rate_limit.burst_size == 0 {
            anyhow::bail!("rate_limit.burst_size must be >= 1");
        }
        if !(0.0..=1.0).contains(&self.write.dedup_threshold) {
            anyhow::bail!("write.dedup_threshold must be between 0.0 and 1.0");
        }
        if self.search.rrf_candidates == 0 {
            anyhow::bail!("search.rrf_candidates must be >= 1");
        }
        if self.search.max_limit == 0 {
            anyhow::bail!("search.max_limit must be >= 1");
        }
        if self.search.default_limit == 0 {
            anyhow::bail!("search.default_limit must be >= 1");
        }
        // Caught here rather than silently clamped: a default above the ceiling
        // means every caller who omits `limit` gets the ceiling, so the default
        // they configured never applies and nothing says so.
        if self.search.default_limit > self.search.max_limit {
            anyhow::bail!(
                "search.default_limit ({}) must not exceed search.max_limit ({})",
                self.search.default_limit,
                self.search.max_limit
            );
        }
        if self.search.diversity_max_per_document == Some(0) {
            anyhow::bail!(
                "search.diversity_max_per_document must be >= 1, or null to disable diversity"
            );
        }
        if self.mcp.metadata_refresh_secs < 10 {
            anyhow::bail!("mcp.metadata_refresh_secs must be >= 10");
        }
        if self.indexing.reconcile_interval_secs == 0 {
            anyhow::bail!("indexing.reconcile_interval_secs must be >= 1");
        }
        if self.ui.semantic_edges.k == 0 {
            anyhow::bail!("ui.semantic_edges.k must be >= 1");
        }
        if !(0.0..=1.0).contains(&self.ui.semantic_edges.min_score) {
            anyhow::bail!("ui.semantic_edges.min_score must be between 0.0 and 1.0");
        }

        // Validate required env vars — named all at once, not one at a time, so a
        // fresh deployment finds every missing var on the first failed start
        // instead of fixing them one restart at a time.
        let mut missing = Vec::new();
        if embedding_base_url.is_none() {
            missing.push("EMBEDDING_BASE_URL");
        }
        if embedding_model.is_none() {
            missing.push("EMBEDDING_MODEL");
        }
        if qdrant_url.is_none() {
            missing.push("QDRANT_URL");
        }
        if self.reranking.enabled {
            if reranking_base_url.is_none() {
                missing.push("RERANKING_BASE_URL");
            }
            if reranking_model.is_none() {
                missing.push("RERANKING_MODEL");
            }
        }
        if !missing.is_empty() {
            anyhow::bail!(
                "Missing required environment variable(s):\n  - {}",
                missing.join("\n  - ")
            );
        }

        // SAFETY: all fields referenced below were checked for None above; bail!
        // prevents reaching here with any of them absent. We use ok_or_else rather
        // than unwrap so the compiler enforces the invariant — if the check block
        // above is ever refactored, this produces a proper error instead of a panic.
        let embedding_base_url = embedding_base_url.ok_or_else(|| {
            anyhow::anyhow!(
                "EMBEDDING_BASE_URL must be set (internal error: missing after validation)"
            )
        })?;
        let embedding_model = embedding_model.ok_or_else(|| {
            anyhow::anyhow!(
                "EMBEDDING_MODEL must be set (internal error: missing after validation)"
            )
        })?;
        let qdrant_url = qdrant_url.ok_or_else(|| {
            anyhow::anyhow!("QDRANT_URL must be set (internal error: missing after validation)")
        })?;

        // Read before the struct literal below moves `self.chunking` into it.
        // The reranker's per-document budget is derived from this, not configured
        // separately — see `ResolvedRerankingConfig::max_document_bytes`.
        let max_chunk_size = self.chunking.max_chunk_size;

        Ok(ResolvedConfig {
            source: ResolvedSourceConfig {
                git_url: source_git_url,
                branch: source_branch,
                data_path: source_data_path,
                git_token_env: self.source.git_token_env,
            },
            indexing: self.indexing,
            frontmatter: self.frontmatter,
            chunking: self.chunking,
            embedding: ResolvedEmbeddingConfig {
                base_url: embedding_base_url,
                model: embedding_model,
                api_key: embedding_api_key,
                vector_size: embedding_vector_size,
                batch_size: self.embedding.batch_size,
                request_timeout_secs: self.embedding.request_timeout_secs,
                batch_concurrency: self.embedding.batch_concurrency,
            },
            qdrant: ResolvedQdrantConfig {
                url: qdrant_url,
                collection: qdrant_collection,
            },
            validation: self.validation,
            webhook: self.webhook,
            mcp: ResolvedMcpConfig {
                port: mcp_port,
                bearer_token_env: self.mcp.bearer_token_env,
                allow_unauthenticated: self.mcp.allow_unauthenticated,
                instructions: self.mcp.instructions,
                metadata_refresh_secs: self.mcp.metadata_refresh_secs,
                allowed_hosts: self.mcp.allowed_hosts,
                extensions_path: self.mcp.extensions_path,
            },
            rate_limit: self.rate_limit,
            write: self.write,
            search: self.search,
            reranking: if self.reranking.enabled {
                Some(ResolvedRerankingConfig {
                    base_url: reranking_base_url.unwrap(),
                    model: reranking_model.unwrap(),
                    api_key: reranking_api_key,
                    candidate_limit: self.reranking.candidate_limit,
                    max_document_bytes: max_chunk_size,
                })
            } else {
                None
            },
            ui: self.ui,
            provenance: ConfigProvenance(
                provenance
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
            ),
        })
    }
}

impl ResolvedConfig {
    /// Resolve the data path (source.data_path, or /data as default)
    pub fn data_path(&self) -> &str {
        self.source.data_path.as_deref().unwrap_or("/data")
    }

    /// [`Self::data_path`], canonicalized — falling back to the configured path
    /// unresolved (with a warning) if canonicalization fails, e.g. a fresh clone
    /// whose directory does not exist yet.
    ///
    /// Every consumer of the schema tree (the shared `SchemaCache`, the write
    /// tools' `resolve_safe_write_path`, `ingest`'s own rel-key derivation) needs
    /// to agree on the SAME base path, or a relative path computed against one
    /// silently fails to match a lookup against the other.
    pub fn canonical_data_path(&self) -> std::path::PathBuf {
        let configured = std::path::PathBuf::from(self.data_path());
        configured.canonicalize().unwrap_or_else(|e| {
            warn!(
                "Could not canonicalize data_path '{}': {} — using configured path as-is",
                configured.display(),
                e
            );
            configured
        })
    }

    /// Derive the state DB path from data_path: `{data_path}/state.db`
    pub fn state_db_path(&self) -> String {
        format!("{}/state.db", self.data_path())
    }

    /// Returns the full set of fields to keyword-index in Qdrant.
    ///
    /// Always includes `"file_path"` (required for `delete_by_file` and
    /// filtered searches), in addition to any user-configured
    /// `frontmatter.indexed_fields`.
    pub fn effective_indexed_fields(&self) -> Vec<String> {
        let mut fields = self.frontmatter.indexed_fields.clone();
        if !fields.iter().any(|f| f == "file_path") {
            fields.push("file_path".to_string());
        }
        fields
    }

    /// Log a startup-time note when git integration is off — see
    /// [`ResolvedSourceConfig::git_integration_disabled`] for why this is worth
    /// surfacing even though it is not an error: a deliberate bind-mount-only
    /// deployment and a migration that dropped `GIT_URL` by accident look
    /// IDENTICAL from the process's point of view (no error, no crash, just a
    /// server that quietly never fetches again), so this is the only place either
    /// case becomes visible. Deliberately worded as informational rather than
    /// alarming — a bind-mount deployment reading this at every startup should
    /// not read it as something broken.
    pub fn log_git_integration_status(&self) {
        if self.source.git_integration_disabled() {
            warn!(
                "Git integration is disabled (GIT_URL is not set): the server will only \
                 ever serve whatever is already present at '{}' — no clone-on-empty, no \
                 webhook-driven pulls. This is expected for a bind-mount-only deployment; \
                 if you meant to track a git remote, set GIT_URL.",
                self.data_path()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Live config handle — backs `POST /admin/reload` (src/reload.rs)
// ---------------------------------------------------------------------------

/// Live handle to the process's resolved configuration, following the same
/// `RwLock<Arc<_>>` idiom as `schema::SharedSchemaCache`: a reader takes the lock,
/// clones the `Arc`, and drops the guard immediately (see [`load_shared_config`]) —
/// a handful of atomic operations, cheap enough for a read-mostly value that a
/// lock-free-swap crate is not justified. `reload::reload_config` is the only
/// writer; every other holder just reads whatever snapshot is current.
///
/// Not every consumer of `ResolvedConfig` holds this type. Plenty of settings are
/// baked into a value or service built once at server startup — a `reqwest::Client`
/// timeout, a compiled `GlobSet`, a `GovernorLayer` — and stay that way for the rest
/// of the process's life by construction, restart or not. `SharedConfig` is only for
/// the subset of consumers that genuinely re-read the config on every use (an MCP
/// tool call, a webhook request, the reindex worker's next drain, a periodic
/// refresh's next tick); see `reload::diff`'s classification table for exactly which
/// setting falls into which category and the file:line evidence behind each one.
pub type SharedConfig = Arc<RwLock<Arc<ResolvedConfig>>>;

/// Wrap an already-resolved config as a fresh `SharedConfig` handle. Used by server
/// startup (building the live view from its initial `Config::load`) and by tests
/// (building one from a one-off `ResolvedConfig`) — anywhere a caller has an owned
/// snapshot and needs a handle other code can later swap.
pub fn shared_config(config: Arc<ResolvedConfig>) -> SharedConfig {
    Arc::new(RwLock::new(config))
}

/// Clone the current config out of `shared`. Cheap: a lock acquisition plus an `Arc`
/// clone, with the guard dropped before returning — never held across an `.await`.
/// Mirrors `schema::load_shared`.
///
/// A poisoned lock (a reader or writer panicked while holding it) is recovered
/// rather than propagated, the same policy already applied to the schema cache and
/// the MCP instructions lock: a panic in one caller must not brick every subsequent
/// config read for the rest of the process's life.
pub fn load_shared_config(shared: &SharedConfig) -> Arc<ResolvedConfig> {
    shared
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// Swap a freshly validated `ResolvedConfig` into `shared`, replacing whatever was
/// there. Mirrors `schema::store_shared`.
///
/// Callers of [`load_shared_config`] that are already mid-request hold their own
/// `Arc` clone and are unaffected by a swap landing underneath them — they simply
/// finish using the snapshot they already took, and the next `load_shared_config`
/// call sees the new one. This is what makes the swap atomic from every reader's
/// point of view: there is no instant at which a caller can observe a
/// partially-applied config.
pub fn store_shared_config(shared: &SharedConfig, new: ResolvedConfig) {
    let new = Arc::new(new);
    match shared.write() {
        Ok(mut guard) => *guard = new,
        Err(poisoned) => *poisoned.into_inner() = new,
    }
}

/// Test-only env var helpers shared across modules whose tests exercise
/// `Config::load`/`resolve` (this module and `reload.rs`). Deliberately `pub(crate)`
/// rather than private-to-`mod tests`: a SEPARATE, module-local `Mutex` per test file
/// would not actually serialize anything, since `cargo test` runs every test in this
/// crate in one multi-threaded binary — two different mutexes guarding the same
/// process-global env vars race exactly as if there were no lock at all. Every test
/// anywhere in the crate that sets `EMBEDDING_BASE_URL`/`EMBEDDING_MODEL`/
/// `QDRANT_URL` must go through this single `ENV_MUTEX`.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Mutex;

    /// Mutex to serialize tests that modify environment variables.
    pub(crate) static ENV_MUTEX: Mutex<()> = Mutex::new(());

    /// The three env vars every successful `resolve()` needs at minimum
    /// (reranking is disabled by default, so its vars are not required).
    pub(crate) fn set_required_env() {
        // SAFETY: caller holds ENV_MUTEX
        unsafe {
            std::env::set_var("EMBEDDING_BASE_URL", "http://test:8080/v1");
            std::env::set_var("EMBEDDING_MODEL", "test-model");
            std::env::set_var("QDRANT_URL", "http://test:6334");
        }
    }

    pub(crate) fn clear_required_env() {
        // SAFETY: caller holds ENV_MUTEX
        unsafe {
            std::env::remove_var("EMBEDDING_BASE_URL");
            std::env::remove_var("EMBEDDING_MODEL");
            std::env::remove_var("QDRANT_URL");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{ENV_MUTEX, clear_required_env, set_required_env};
    use super::*;

    impl Config {
        /// Deserialize + resolve (requires env vars for the required fields)
        fn from_str(yaml: &str) -> anyhow::Result<ResolvedConfig> {
            let config: Config = serde_yaml_ng::from_str(yaml)?;
            config.resolve()
        }

        /// Deserialize only — no env var resolution or validation
        fn from_str_raw(yaml: &str) -> anyhow::Result<Self> {
            Ok(serde_yaml_ng::from_str(yaml)?)
        }
    }

    /// A YAML doc using only settings that are still YAML-settable post-migration.
    const MINIMAL_CONFIG: &str = r#"
indexing:
  include: ["**/*.md"]
frontmatter:
  required: [title, description]
chunking:
  max_chunk_size: 1000
"#;

    #[test]
    fn parse_minimal_config() {
        let cfg = Config::from_str_raw(MINIMAL_CONFIG).unwrap();
        assert_eq!(cfg.source.git_token_env, "GIT_PULL_TOKEN");
        assert_eq!(cfg.embedding.batch_size, 32);
        assert_eq!(cfg.chunking.max_chunk_size, 1000);
        assert!(cfg.validation.enabled);
        assert!(!cfg.validation.strict);
    }

    #[test]
    fn parse_full_config() {
        let yaml = r#"
source:
  git_token_env: "MY_GIT_TOKEN"
indexing:
  include: ["**/*.md"]
  exclude: [".git/**"]
  exclude_files: ["README.md"]
frontmatter:
  required: [title]
  indexed_fields: [type, domain]
  defaults:
    status: "draft"
chunking:
  max_chunk_size: 2000
  prepend_description: true
embedding:
  api_key_env: "MY_EMBEDDING_KEY"
  batch_size: 16
validation:
  enabled: false
  strict: true
webhook:
  secret_env: "MY_SECRET"
  provider: "github"
mcp:
  bearer_token_env: "MY_TOKEN"
"#;
        let cfg = Config::from_str_raw(yaml).unwrap();
        assert_eq!(cfg.source.git_token_env, "MY_GIT_TOKEN");
        assert_eq!(cfg.embedding.api_key_env, "MY_EMBEDDING_KEY");
        assert_eq!(cfg.embedding.batch_size, 16);
        assert!(!cfg.validation.enabled);
        assert!(cfg.validation.strict);
        assert_eq!(cfg.webhook.provider, WebhookProvider::Github);
        assert_eq!(cfg.frontmatter.defaults.get("status").unwrap(), "draft");
    }

    #[test]
    fn default_data_path() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert_eq!(cfg.data_path(), "/data");
        clear_required_env();
    }

    #[test]
    fn empty_yaml_deserializes_to_defaults() {
        let cfg = Config::from_str_raw("{}").unwrap();
        assert_eq!(cfg.source.git_token_env, "GIT_PULL_TOKEN");
        assert_eq!(cfg.indexing.include, vec!["**/*.md"]);
        assert_eq!(
            cfg.indexing.exclude,
            vec![".git/**", ".claude/**", ".tools/**", "node_modules/**"]
        );
        assert_eq!(cfg.indexing.exclude_files, vec!["CLAUDE.md", "README.md"]);
        assert_eq!(cfg.indexing.reconcile_interval_secs, 600);
        assert!(cfg.frontmatter.required.is_empty());
        assert_eq!(cfg.chunking.max_chunk_size, 1500);
        assert_eq!(cfg.chunking.target_chunk_size, Some(1000));
        assert!(cfg.chunking.prepend_description);
        assert_eq!(cfg.embedding.batch_size, 32);
        assert_eq!(cfg.embedding.api_key_env, "EMBEDDING_API_KEY");
        assert!(cfg.validation.enabled);
        assert_eq!(cfg.rate_limit.per_second, 20);
        assert_eq!(cfg.rate_limit.burst_size, 50);
    }

    #[test]
    fn env_only_settings_are_read_from_env() {
        let _lock = ENV_MUTEX.lock().unwrap();

        // SAFETY: serialized by ENV_MUTEX
        unsafe {
            std::env::set_var("EMBEDDING_BASE_URL", "http://env-embed:9090/v1");
            std::env::set_var("EMBEDDING_MODEL", "env-model");
            std::env::set_var("QDRANT_URL", "http://env-qdrant:6334");
        }

        let cfg = Config::from_str_raw(MINIMAL_CONFIG)
            .unwrap()
            .resolve()
            .unwrap();

        assert_eq!(cfg.embedding.base_url, "http://env-embed:9090/v1");
        assert_eq!(cfg.embedding.model, "env-model");
        assert_eq!(cfg.qdrant.url, "http://env-qdrant:6334");

        unsafe {
            std::env::remove_var("EMBEDDING_BASE_URL");
            std::env::remove_var("EMBEDDING_MODEL");
            std::env::remove_var("QDRANT_URL");
        }
    }

    #[test]
    fn missing_required_env_produces_clear_error_naming_all_of_them() {
        let _lock = ENV_MUTEX.lock().unwrap();
        clear_required_env();

        let result = Config::from_str_raw("{}").unwrap().resolve();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("EMBEDDING_BASE_URL"),
            "error should mention EMBEDDING_BASE_URL: {err}"
        );
        assert!(
            err.contains("EMBEDDING_MODEL"),
            "error should mention EMBEDDING_MODEL: {err}"
        );
        assert!(
            err.contains("QDRANT_URL"),
            "error should mention QDRANT_URL: {err}"
        );
    }

    #[test]
    fn missing_reranking_env_is_only_required_when_enabled() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        unsafe {
            std::env::remove_var("RERANKING_BASE_URL");
            std::env::remove_var("RERANKING_MODEL");
        }

        // Disabled: resolves fine even with no reranking env vars set.
        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert!(cfg.reranking.is_none());

        // Enabled: now both become required, and both missing ones are named.
        let yaml = format!("{MINIMAL_CONFIG}\nreranking:\n  enabled: true\n");
        let result = Config::from_str_raw(&yaml).unwrap().resolve();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("RERANKING_BASE_URL"), "{err}");
        assert!(err.contains("RERANKING_MODEL"), "{err}");

        clear_required_env();
    }

    #[test]
    fn load_missing_file_returns_defaults() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();

        let cfg = Config::load(Path::new("/nonexistent/config.yaml")).unwrap();
        assert_eq!(cfg.source.branch, "master");
        assert_eq!(cfg.chunking.max_chunk_size, 1500);
        assert_eq!(cfg.qdrant.collection, "knowledge-base");

        clear_required_env();
    }

    #[test]
    fn env_var_vector_size_override() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        unsafe {
            std::env::set_var("EMBEDDING_VECTOR_SIZE", "1024");
        }

        let cfg = Config::from_str_raw("{}").unwrap().resolve().unwrap();
        assert_eq!(cfg.embedding.vector_size, 1024);

        unsafe {
            std::env::remove_var("EMBEDDING_VECTOR_SIZE");
        }
        clear_required_env();
    }

    #[test]
    fn vector_size_defaults_when_env_unset() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        unsafe {
            std::env::remove_var("EMBEDDING_VECTOR_SIZE");
        }

        let cfg = Config::from_str_raw("{}").unwrap().resolve().unwrap();
        assert_eq!(cfg.embedding.vector_size, 768);

        clear_required_env();
    }

    #[test]
    fn env_var_vector_size_invalid() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        unsafe {
            std::env::set_var("EMBEDDING_VECTOR_SIZE", "not-a-number");
        }

        let result = Config::from_str_raw("{}").unwrap().resolve();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("EMBEDDING_VECTOR_SIZE"),
            "error should mention EMBEDDING_VECTOR_SIZE"
        );

        unsafe {
            std::env::remove_var("EMBEDDING_VECTOR_SIZE");
        }
        clear_required_env();
    }

    #[test]
    fn mcp_port_defaults_when_env_unset() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        unsafe {
            std::env::remove_var("MCP_PORT");
        }
        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert_eq!(cfg.mcp.port, 8001);
        clear_required_env();
    }

    #[test]
    fn mcp_port_read_from_env() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        unsafe {
            std::env::set_var("MCP_PORT", "9002");
        }
        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert_eq!(cfg.mcp.port, 9002);
        unsafe {
            std::env::remove_var("MCP_PORT");
        }
        clear_required_env();
    }

    #[test]
    fn mcp_port_invalid_is_rejected() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        unsafe {
            std::env::set_var("MCP_PORT", "not-a-port");
        }
        let result = Config::from_str(MINIMAL_CONFIG);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("MCP_PORT"));
        unsafe {
            std::env::remove_var("MCP_PORT");
        }
        clear_required_env();
    }

    #[test]
    fn source_bootstrap_bindings_read_from_env() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        unsafe {
            std::env::set_var("GIT_URL", "https://example.com/repo.git");
            std::env::set_var("GIT_BRANCH", "main");
            std::env::set_var("DATA_PATH", "/custom/path");
            std::env::set_var("QDRANT_COLLECTION", "my-kb");
        }

        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert_eq!(
            cfg.source.git_url.as_deref(),
            Some("https://example.com/repo.git")
        );
        assert_eq!(cfg.source.branch, "main");
        assert_eq!(cfg.source.data_path.as_deref(), Some("/custom/path"));
        assert_eq!(cfg.qdrant.collection, "my-kb");
        assert_eq!(cfg.state_db_path(), "/custom/path/state.db");

        unsafe {
            std::env::remove_var("GIT_URL");
            std::env::remove_var("GIT_BRANCH");
            std::env::remove_var("DATA_PATH");
            std::env::remove_var("QDRANT_COLLECTION");
        }
        clear_required_env();
    }

    #[test]
    fn source_bootstrap_bindings_default_when_env_unset() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        unsafe {
            std::env::remove_var("GIT_URL");
            std::env::remove_var("GIT_BRANCH");
            std::env::remove_var("DATA_PATH");
            std::env::remove_var("QDRANT_COLLECTION");
        }

        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert_eq!(cfg.source.git_url, None);
        assert_eq!(cfg.source.branch, "master");
        assert_eq!(cfg.source.data_path.as_deref(), Some("/data"));
        assert_eq!(cfg.qdrant.collection, "knowledge-base");

        clear_required_env();
    }

    #[test]
    fn git_integration_disabled_when_git_url_absent() {
        let source = ResolvedSourceConfig {
            git_url: None,
            ..Default::default()
        };
        assert!(source.git_integration_disabled());
    }

    #[test]
    fn git_integration_not_disabled_when_git_url_present() {
        let source = ResolvedSourceConfig {
            git_url: Some("https://example.com/repo.git".into()),
            ..Default::default()
        };
        assert!(!source.git_integration_disabled());
    }

    #[test]
    fn git_integration_disabled_via_full_resolve_when_env_unset() {
        // End-to-end through Config::resolve, not just the predicate in isolation
        // — confirms GIT_URL unset really does propagate to the disabled state.
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        unsafe {
            std::env::remove_var("GIT_URL");
        }

        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert!(cfg.source.git_integration_disabled());

        unsafe {
            std::env::set_var("GIT_URL", "https://example.com/repo.git");
        }
        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert!(!cfg.source.git_integration_disabled());

        unsafe {
            std::env::remove_var("GIT_URL");
        }
        clear_required_env();
    }

    #[test]
    fn resolved_config_usable_without_raw_config() {
        // Construct ResolvedConfig directly — proves no Option unwrapping needed at use sites.
        let cfg = ResolvedConfig {
            source: ResolvedSourceConfig::default(),
            indexing: IndexingConfig::default(),
            frontmatter: FrontmatterConfig::default(),
            chunking: ChunkingConfig::default(),
            embedding: ResolvedEmbeddingConfig {
                base_url: "http://embed:8080/v1".into(),
                model: "test-model".into(),
                api_key: None,
                vector_size: 768,
                batch_size: 32,
                request_timeout_secs: 60,
                batch_concurrency: 4,
            },
            qdrant: ResolvedQdrantConfig {
                url: "http://qdrant:6334".into(),
                collection: "test-collection".into(),
            },
            validation: ValidationConfig::default(),
            webhook: WebhookConfig::default(),
            mcp: ResolvedMcpConfig::default(),
            rate_limit: RateLimitConfig::default(),
            write: WriteConfig::default(),
            search: SearchConfig::default(),
            reranking: None,
            ui: UiConfig::default(),
            provenance: ConfigProvenance::default(),
        };

        // All fields are directly accessible — no unwrap, no panic path.
        assert_eq!(cfg.embedding.base_url, "http://embed:8080/v1");
        assert_eq!(cfg.embedding.model, "test-model");
        assert_eq!(cfg.qdrant.url, "http://qdrant:6334");
        assert_eq!(cfg.qdrant.collection, "test-collection");
        assert_eq!(cfg.data_path(), "/data");
        assert_eq!(cfg.state_db_path(), "/data/state.db");
    }

    #[test]
    fn load_returns_resolved_config() {
        let _lock = ENV_MUTEX.lock().unwrap();
        unsafe {
            std::env::set_var("EMBEDDING_BASE_URL", "http://load-test:8080/v1");
            std::env::set_var("EMBEDDING_MODEL", "load-model");
            std::env::set_var("QDRANT_URL", "http://load-qdrant:6334");
        }

        let cfg = Config::load(Path::new("/nonexistent/config.yaml")).unwrap();

        assert_eq!(cfg.embedding.base_url, "http://load-test:8080/v1");
        assert_eq!(cfg.embedding.model, "load-model");
        assert_eq!(cfg.qdrant.url, "http://load-qdrant:6334");

        unsafe {
            std::env::remove_var("EMBEDDING_BASE_URL");
            std::env::remove_var("EMBEDDING_MODEL");
            std::env::remove_var("QDRANT_URL");
        }
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let yaml = r#"
source:
  git_token_env: "X"
unknown_top_level: true
"#;
        let result: Result<Config, _> = serde_yaml_ng::from_str(yaml);
        assert!(
            result.is_err(),
            "top-level unknown field should be rejected"
        );
    }

    #[test]
    fn unknown_fields_in_nested_struct_are_rejected() {
        let yaml = r#"
mcp:
  bearer_token_env: "X"
  unknown_nested: "oops"
"#;
        let result: Result<Config, _> = serde_yaml_ng::from_str(yaml);
        assert!(result.is_err(), "nested unknown field should be rejected");
    }

    #[test]
    fn removed_bootstrap_fields_are_rejected_in_yaml() {
        // source.git_url / branch / data_path, embedding.base_url / model /
        // vector_size, and qdrant.url / collection all moved to ENV-only. A
        // deployed config.yaml that still sets any of them must fail loudly at
        // parse time rather than have the setting silently stop applying.
        for yaml in [
            "source:\n  git_url: \"https://example.com/repo.git\"\n",
            "source:\n  branch: \"main\"\n",
            "source:\n  data_path: \"/data\"\n",
            "embedding:\n  base_url: \"http://embed:8080/v1\"\n",
            "embedding:\n  model: \"nomic\"\n",
            "embedding:\n  vector_size: 768\n",
            "embedding:\n  api_key: \"sk-leaked\"\n",
            "reranking:\n  base_url: \"http://rerank:8081/v1\"\n",
            "reranking:\n  model: \"reranker\"\n",
            "reranking:\n  api_key: \"sk-leaked\"\n",
            "mcp:\n  port: 9002\n",
        ] {
            let result: Result<Config, _> = serde_yaml_ng::from_str(yaml);
            assert!(
                result.is_err(),
                "expected '{yaml}' to be rejected as an unknown field"
            );
        }
    }

    #[test]
    fn state_db_path_derived_from_data_path() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        unsafe {
            std::env::set_var("DATA_PATH", "/custom/path");
        }

        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert_eq!(cfg.state_db_path(), "/custom/path/state.db");

        unsafe {
            std::env::remove_var("DATA_PATH");
        }
        clear_required_env();
    }

    #[test]
    fn state_db_path_uses_default_data_path() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert_eq!(cfg.state_db_path(), "/data/state.db");
        clear_required_env();
    }

    #[test]
    fn mcp_config_defaults_for_new_fields() {
        let cfg = Config::from_str_raw("{}").unwrap();
        assert!(cfg.mcp.instructions.is_none());
        assert_eq!(cfg.mcp.metadata_refresh_secs, 300);
        assert_eq!(cfg.mcp.extensions_path, "meta/mcp");
    }

    #[test]
    fn mcp_config_custom_instructions() {
        let yaml = r#"
mcp:
  instructions: "My custom KB description."
  metadata_refresh_secs: 60
  extensions_path: "meta/mcp-ext"
"#;
        let cfg = Config::from_str_raw(yaml).unwrap();
        assert_eq!(
            cfg.mcp.instructions.as_deref(),
            Some("My custom KB description.")
        );
        assert_eq!(cfg.mcp.metadata_refresh_secs, 60);
        assert_eq!(cfg.mcp.extensions_path, "meta/mcp-ext");
    }

    #[test]
    fn mcp_extensions_path_empty_string_parses_fine() {
        // Empty disables extension loading entirely (descriptions::resolve_extensions_dir);
        // this only checks that config parsing itself accepts it.
        let yaml = "mcp:\n  extensions_path: \"\"\n";
        let cfg = Config::from_str_raw(yaml).unwrap();
        assert_eq!(cfg.mcp.extensions_path, "");
    }

    #[test]
    fn mcp_extensions_path_passes_through_to_resolved_config() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        let yaml = "mcp:\n  extensions_path: \"custom/ext\"\n";
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.yaml");
        std::fs::write(&path, yaml).unwrap();
        let resolved = Config::load(&path).unwrap();
        assert_eq!(resolved.mcp.extensions_path, "custom/ext");
        clear_required_env();
    }

    #[test]
    fn example_config_deserializes() {
        let yaml = include_str!("../deploy/config.example.yaml");
        let cfg: Config = serde_yaml_ng::from_str(yaml).expect("config.example.yaml should parse");
        // Spot-check a few values to catch drift between example and struct
        assert_eq!(cfg.chunking.max_chunk_size, 1500);
        assert_eq!(cfg.chunking.target_chunk_size, Some(1000));
        assert!(cfg.chunking.prepend_description);
        assert_eq!(cfg.embedding.batch_size, 32);
        assert!(cfg.validation.enabled);
        assert!(!cfg.validation.strict);
        assert_eq!(cfg.webhook.provider, WebhookProvider::Gitea);
        // Verify new write identity fields round-trip from the example config
        assert_eq!(cfg.write.commit_author_name, "md-kb-rag");
        assert_eq!(cfg.write.commit_author_email, "md-kb-rag@localhost");
        // Verify search section round-trips from the example config
        assert!(cfg.search.hybrid);
        assert_eq!(cfg.search.rrf_candidates, 50);
        assert!(cfg.search.phrase);
        assert_eq!(cfg.search.diversity_max_per_document, Some(3));
    }

    #[test]
    fn search_config_defaults() {
        // A config with no `search` section gets hybrid=true, rrf_candidates=50,
        // phrase=true, diversity_max_per_document=Some(3) — issue #86's
        // default-on diversity cap.
        let cfg = Config::from_str_raw("{}").unwrap();
        assert!(cfg.search.hybrid);
        assert_eq!(cfg.search.rrf_candidates, 50);
        assert!(cfg.search.phrase);
        assert_eq!(cfg.search.diversity_max_per_document, Some(3));
    }

    #[test]
    fn search_config_phrase_explicit_false() {
        let yaml = "search:\n  phrase: false\n";
        let cfg = Config::from_str_raw(yaml).unwrap();
        assert!(!cfg.search.phrase);
        // Untouched siblings keep their own defaults.
        assert!(cfg.search.hybrid);
    }

    #[test]
    fn search_config_diversity_explicit_null_disables() {
        // The same Option convention as search.min_score: an explicit `null`
        // overrides the non-None default and disables the feature.
        let yaml = "search:\n  diversity_max_per_document: null\n";
        let cfg = Config::from_str_raw(yaml).unwrap();
        assert_eq!(cfg.search.diversity_max_per_document, None);
    }

    #[test]
    fn search_config_diversity_custom_value() {
        let yaml = "search:\n  diversity_max_per_document: 7\n";
        let cfg = Config::from_str_raw(yaml).unwrap();
        assert_eq!(cfg.search.diversity_max_per_document, Some(7));
    }

    #[test]
    fn search_config_diversity_zero_rejected_at_resolve() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        let yaml = "search:\n  diversity_max_per_document: 0\n";
        let err = Config::from_str(yaml).unwrap_err();
        assert!(
            err.to_string()
                .contains("search.diversity_max_per_document must be >= 1"),
            "expected the diversity validation message, got: {err}"
        );
        clear_required_env();
    }

    #[test]
    fn search_limits_default_to_the_historical_hardcoded_values() {
        let cfg = Config::from_str_raw("").unwrap();
        assert_eq!(cfg.search.default_limit, 10);
        assert_eq!(cfg.search.max_limit, 50);
    }

    #[test]
    fn search_limits_round_trip_from_yaml() {
        let yaml = "search:\n  default_limit: 25\n  max_limit: 200\n";
        let cfg = Config::from_str_raw(yaml).unwrap();
        assert_eq!(cfg.search.default_limit, 25);
        assert_eq!(cfg.search.max_limit, 200);
    }

    #[test]
    fn search_default_limit_above_max_limit_is_rejected_at_resolve() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        // Rejected rather than silently clamped: a default above the ceiling means
        // every caller omitting `limit` gets the ceiling, so the configured default
        // never applies and nothing would say so.
        let yaml = "search:\n  default_limit: 100\n  max_limit: 50\n";
        let err = Config::from_str(yaml).unwrap_err();
        assert!(
            err.to_string()
                .contains("search.default_limit (100) must not exceed search.max_limit (50)"),
            "expected the default-exceeds-max message, got: {err}"
        );
        clear_required_env();
    }

    #[test]
    fn search_zero_limits_are_rejected_at_resolve() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        let err = Config::from_str("search:\n  max_limit: 0\n").unwrap_err();
        assert!(
            err.to_string().contains("search.max_limit must be >= 1"),
            "expected the max_limit validation message, got: {err}"
        );
        let err = Config::from_str("search:\n  default_limit: 0\n").unwrap_err();
        assert!(
            err.to_string()
                .contains("search.default_limit must be >= 1"),
            "expected the default_limit validation message, got: {err}"
        );
        clear_required_env();
    }

    #[test]
    fn search_config_custom_values() {
        let yaml = r#"
search:
  hybrid: false
  rrf_candidates: 100
"#;
        let cfg = Config::from_str_raw(yaml).unwrap();
        assert!(!cfg.search.hybrid);
        assert_eq!(cfg.search.rrf_candidates, 100);
    }

    #[test]
    fn search_config_partial_uses_defaults_for_missing() {
        // Only hybrid specified — rrf_candidates falls back to the default.
        let yaml = r#"
search:
  hybrid: false
"#;
        let cfg = Config::from_str_raw(yaml).unwrap();
        assert!(!cfg.search.hybrid);
        assert_eq!(cfg.search.rrf_candidates, 50);
    }

    #[test]
    fn write_config_commit_author_defaults() {
        // Config without a write section should still produce the default bot identity.
        let cfg = Config::from_str_raw("{}").unwrap();
        assert_eq!(cfg.write.commit_author_name, "md-kb-rag");
        assert_eq!(cfg.write.commit_author_email, "md-kb-rag@localhost");
    }

    #[test]
    fn write_config_without_commit_author_fields_still_loads() {
        // A config that has a write section but omits the new identity fields must
        // still deserialize successfully (serde defaults fill them in).
        let yaml = r#"
write:
  dedup_enabled: false
  dedup_threshold: 0.90
"#;
        let cfg = Config::from_str_raw(yaml).unwrap();
        assert!(!cfg.write.dedup_enabled);
        assert_eq!(cfg.write.commit_author_name, "md-kb-rag");
        assert_eq!(cfg.write.commit_author_email, "md-kb-rag@localhost");
    }

    #[test]
    fn write_config_custom_commit_author() {
        let yaml = r#"
write:
  commit_author_name: "kb-bot"
  commit_author_email: "kb-bot@example.com"
"#;
        let cfg = Config::from_str_raw(yaml).unwrap();
        assert_eq!(cfg.write.commit_author_name, "kb-bot");
        assert_eq!(cfg.write.commit_author_email, "kb-bot@example.com");
    }

    #[test]
    fn invalid_provider_rejected_at_parse_time() {
        let yaml = r#"
webhook:
  provider: "bitbucket"
"#;
        let result: Result<Config, _> = serde_yaml_ng::from_str(yaml);
        assert!(
            result.is_err(),
            "unknown provider should be rejected at parse time"
        );
    }

    #[test]
    fn target_exceeds_max_is_rejected() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();

        let yaml = r#"
chunking:
  target_chunk_size: 1500
  max_chunk_size: 1000
"#;
        let result = Config::from_str_raw(yaml).unwrap().resolve();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("target_chunk_size"),
            "error should mention target_chunk_size: {err}"
        );
        assert!(
            err.contains("max_chunk_size"),
            "error should mention max_chunk_size: {err}"
        );

        clear_required_env();
    }

    #[test]
    fn zero_batch_size_is_rejected() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();

        let yaml = r#"
embedding:
  batch_size: 0
"#;
        let result = Config::from_str_raw(yaml).unwrap().resolve();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("batch_size"),
            "error should mention batch_size: {err}"
        );

        clear_required_env();
    }

    #[test]
    fn zero_lint_timeout_secs_is_rejected() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();

        let yaml = r#"
validation:
  lint_timeout_secs: 0
"#;
        let result = Config::from_str_raw(yaml).unwrap().resolve();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("lint_timeout_secs"),
            "error should mention lint_timeout_secs: {err}"
        );

        clear_required_env();
    }

    #[test]
    fn zero_max_chunk_size_is_rejected() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();

        let yaml = r#"
chunking:
  max_chunk_size: 0
"#;
        let result = Config::from_str_raw(yaml).unwrap().resolve();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("max_chunk_size"),
            "error should mention max_chunk_size: {err}"
        );

        clear_required_env();
    }

    #[test]
    fn zero_vector_size_is_rejected() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        unsafe {
            std::env::set_var("EMBEDDING_VECTOR_SIZE", "0");
        }

        let result = Config::from_str_raw("{}").unwrap().resolve();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("vector_size"),
            "error should mention vector_size: {err}"
        );

        unsafe {
            std::env::remove_var("EMBEDDING_VECTOR_SIZE");
        }
        clear_required_env();
    }

    #[test]
    fn effective_indexed_fields_always_includes_file_path() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();

        // When indexed_fields is empty, file_path is injected.
        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert!(cfg.frontmatter.indexed_fields.is_empty());
        let fields = cfg.effective_indexed_fields();
        assert!(
            fields.contains(&"file_path".to_string()),
            "effective_indexed_fields must include file_path"
        );

        clear_required_env();
    }

    #[test]
    fn unknown_field_gives_helpful_error() {
        let dir = std::env::temp_dir().join("md-kb-rag-test-unknown-field");
        std::fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.yaml");
        std::fs::write(&config_path, "chunking:\n  strategy: \"markdown\"\n").unwrap();

        let result = Config::load(&config_path);
        assert!(result.is_err());
        let err = format!("{:#}", result.unwrap_err());
        assert!(
            err.contains("config.example.yaml"),
            "error should mention config.example.yaml: {err}"
        );
        assert!(
            err.contains("unknown field"),
            "error should mention 'unknown field': {err}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn effective_indexed_fields_no_duplicate_file_path() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();

        // When indexed_fields already contains file_path, it should not be duplicated.
        let yaml = r#"
indexing:
  include: ["**/*.md"]
frontmatter:
  required: [title]
  indexed_fields: [file_path, domain]
chunking:
  max_chunk_size: 1000
"#;
        let cfg = Config::from_str(yaml).unwrap();
        let fields = cfg.effective_indexed_fields();
        let count = fields.iter().filter(|f| f.as_str() == "file_path").count();
        assert_eq!(count, 1, "file_path should appear exactly once");
        assert!(fields.contains(&"domain".to_string()));

        clear_required_env();
    }

    #[test]
    fn zero_per_second_is_rejected() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();

        let yaml = r#"
rate_limit:
  per_second: 0
"#;
        let result = Config::from_str_raw(yaml).unwrap().resolve();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("rate_limit.per_second"),
            "error should mention rate_limit.per_second: {err}"
        );

        clear_required_env();
    }

    #[test]
    fn reconcile_interval_secs_custom_value() {
        let yaml = r#"
indexing:
  reconcile_interval_secs: 120
"#;
        let cfg = Config::from_str_raw(yaml).unwrap();
        assert_eq!(cfg.indexing.reconcile_interval_secs, 120);
    }

    #[test]
    fn zero_reconcile_interval_secs_is_rejected() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();

        let yaml = r#"
indexing:
  reconcile_interval_secs: 0
"#;
        let result = Config::from_str_raw(yaml).unwrap().resolve();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("reconcile_interval_secs"),
            "error should mention reconcile_interval_secs: {err}"
        );

        clear_required_env();
    }

    #[test]
    fn zero_burst_size_is_rejected() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();

        let yaml = r#"
rate_limit:
  burst_size: 0
"#;
        let result = Config::from_str_raw(yaml).unwrap().resolve();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("rate_limit.burst_size"),
            "error should mention rate_limit.burst_size: {err}"
        );

        clear_required_env();
    }

    #[test]
    fn low_metadata_refresh_secs_is_rejected() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();

        let yaml = r#"
mcp:
  metadata_refresh_secs: 5
"#;
        let result = Config::from_str_raw(yaml).unwrap().resolve();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mcp.metadata_refresh_secs"),
            "error should mention mcp.metadata_refresh_secs: {err}"
        );

        clear_required_env();
    }

    #[test]
    fn api_key_env_absent_defaults_to_none() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        unsafe {
            std::env::remove_var("EMBEDDING_API_KEY");
        }

        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert!(
            cfg.embedding.api_key.is_none(),
            "api_key should be None when the named env var is unset"
        );

        clear_required_env();
    }

    #[test]
    fn api_key_read_from_default_named_env_var() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        unsafe {
            std::env::set_var("EMBEDDING_API_KEY", "sk-from-env");
        }

        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert_eq!(cfg.embedding.api_key.as_deref(), Some("sk-from-env"));

        unsafe {
            std::env::remove_var("EMBEDDING_API_KEY");
        }
        clear_required_env();
    }

    #[test]
    fn api_key_env_indirection_uses_custom_var_name() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        unsafe {
            std::env::set_var("MY_CUSTOM_EMBED_KEY", "sk-custom");
        }

        let yaml =
            format!("{MINIMAL_CONFIG}\nembedding:\n  api_key_env: \"MY_CUSTOM_EMBED_KEY\"\n");
        let cfg = Config::from_str(&yaml).unwrap();
        assert_eq!(cfg.embedding.api_key.as_deref(), Some("sk-custom"));

        unsafe {
            std::env::remove_var("MY_CUSTOM_EMBED_KEY");
        }
        clear_required_env();
    }

    #[test]
    fn reranking_config_defaults() {
        let cfg = Config::from_str_raw("{}").unwrap();
        assert!(!cfg.reranking.enabled);
        assert_eq!(cfg.reranking.candidate_limit, 50);
        assert_eq!(cfg.reranking.api_key_env, "RERANKING_API_KEY");
    }

    #[test]
    fn reranking_config_full() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        unsafe {
            std::env::set_var("RERANKING_BASE_URL", "http://reranker:8081/v1");
            std::env::set_var("RERANKING_MODEL", "reranker");
            std::env::set_var("RERANKING_API_KEY", "sk-rerank");
        }

        let yaml = r#"
reranking:
  enabled: true
  candidate_limit: 100
"#;
        let cfg = Config::from_str(yaml).unwrap();
        let reranking = cfg.reranking.expect("reranking should be resolved");
        assert_eq!(reranking.base_url, "http://reranker:8081/v1");
        assert_eq!(reranking.model, "reranker");
        assert_eq!(reranking.api_key.as_deref(), Some("sk-rerank"));
        assert_eq!(reranking.candidate_limit, 100);

        unsafe {
            std::env::remove_var("RERANKING_BASE_URL");
            std::env::remove_var("RERANKING_MODEL");
            std::env::remove_var("RERANKING_API_KEY");
        }
        clear_required_env();
    }

    #[test]
    fn resolved_reranking_is_none_when_disabled() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert!(cfg.reranking.is_none());
        clear_required_env();
    }

    // ── ui.semantic_edges ────────────────────────────────────────────────────────

    #[test]
    fn ui_semantic_edges_config_defaults() {
        // A config with no `ui` section at all: off by default, k=5, min_score=0.6.
        let cfg = Config::from_str_raw("{}").unwrap();
        assert!(!cfg.ui.semantic_edges.enabled);
        assert_eq!(cfg.ui.semantic_edges.k, 5);
        assert_eq!(cfg.ui.semantic_edges.min_score, 0.6);
    }

    #[test]
    fn ui_semantic_edges_config_round_trips_from_yaml() {
        let yaml = r#"
ui:
  semantic_edges:
    enabled: true
    k: 8
    min_score: 0.75
"#;
        let cfg = Config::from_str_raw(yaml).unwrap();
        assert!(cfg.ui.semantic_edges.enabled);
        assert_eq!(cfg.ui.semantic_edges.k, 8);
        assert_eq!(cfg.ui.semantic_edges.min_score, 0.75);
    }

    #[test]
    fn ui_semantic_edges_config_partial_uses_defaults_for_missing() {
        // Only `enabled` specified — k and min_score fall back to their defaults.
        let yaml = "ui:\n  semantic_edges:\n    enabled: true\n";
        let cfg = Config::from_str_raw(yaml).unwrap();
        assert!(cfg.ui.semantic_edges.enabled);
        assert_eq!(cfg.ui.semantic_edges.k, 5);
        assert_eq!(cfg.ui.semantic_edges.min_score, 0.6);
    }

    #[test]
    fn ui_semantic_edges_resolves_end_to_end() {
        // Full round trip through `resolve()`, not just deserialization — proves
        // the section survives into `ResolvedConfig` (the `ui: self.ui` copy).
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        let yaml = "ui:\n  semantic_edges:\n    enabled: true\n    k: 3\n    min_score: 0.9\n";
        let cfg = Config::from_str(yaml).unwrap();
        assert!(cfg.ui.semantic_edges.enabled);
        assert_eq!(cfg.ui.semantic_edges.k, 3);
        assert_eq!(cfg.ui.semantic_edges.min_score, 0.9);
        clear_required_env();
    }

    #[test]
    fn ui_semantic_edges_unknown_field_is_rejected() {
        let yaml = "ui:\n  semantic_edges:\n    bogus: true\n";
        let result: Result<Config, _> = serde_yaml_ng::from_str(yaml);
        assert!(
            result.is_err(),
            "unknown field under ui.semantic_edges should be rejected"
        );

        let yaml_top = "ui:\n  bogus: true\n";
        let result: Result<Config, _> = serde_yaml_ng::from_str(yaml_top);
        assert!(result.is_err(), "unknown field under ui should be rejected");
    }

    #[test]
    fn ui_semantic_edges_zero_k_is_rejected_at_resolve() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        let err = Config::from_str("ui:\n  semantic_edges:\n    k: 0\n").unwrap_err();
        assert!(
            err.to_string().contains("ui.semantic_edges.k must be >= 1"),
            "expected the k validation message, got: {err}"
        );
        clear_required_env();
    }

    #[test]
    fn ui_semantic_edges_out_of_range_min_score_is_rejected_at_resolve() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        let err = Config::from_str("ui:\n  semantic_edges:\n    min_score: 1.5\n").unwrap_err();
        assert!(
            err.to_string()
                .contains("ui.semantic_edges.min_score must be between 0.0 and 1.0"),
            "expected the min_score validation message, got: {err}"
        );
        let err = Config::from_str("ui:\n  semantic_edges:\n    min_score: -0.1\n").unwrap_err();
        assert!(
            err.to_string()
                .contains("ui.semantic_edges.min_score must be between 0.0 and 1.0"),
            "expected the min_score validation message, got: {err}"
        );
        clear_required_env();
    }

    #[test]
    fn ui_semantic_edges_provenance_reports_default_when_absent() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        let cfg = Config::load(Path::new("/nonexistent/config.yaml")).unwrap();
        assert_eq!(
            cfg.provenance.0.get("ui.semantic_edges.enabled"),
            Some(&SettingSource::Default)
        );
        clear_required_env();
    }

    #[test]
    fn ui_semantic_edges_provenance_reports_yaml_when_section_present_in_loaded_file() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();

        let dir = std::env::temp_dir().join("md-kb-rag-test-provenance-ui");
        std::fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.yaml");
        std::fs::write(&config_path, "ui:\n  semantic_edges:\n    enabled: true\n").unwrap();

        let cfg = Config::load(&config_path).unwrap();
        assert_eq!(
            cfg.provenance.0.get("ui.semantic_edges.enabled"),
            Some(&SettingSource::Yaml)
        );

        std::fs::remove_dir_all(&dir).ok();
        clear_required_env();
    }

    // ── Migration-specific coverage ─────────────────────────────────────────────
    // The three behaviors this migration is required to prove: a removed env
    // override no longer takes effect, a recognized-but-unhonored var triggers the
    // deprecation warning path, and fail-fast names every missing required var at
    // once (already covered above by
    // `missing_required_env_produces_clear_error_naming_all_of_them`, which now
    // asserts on env var names instead of dotted YAML paths).

    #[test]
    fn removed_reranking_env_overrides_no_longer_apply() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        unsafe {
            std::env::set_var("RERANKING_BASE_URL", "http://reranker:8081/v1");
            std::env::set_var("RERANKING_MODEL", "reranker");
            // Both of these used to override the YAML value; now they must be
            // fully inert. RERANKING_ENABLED=false previously would have disabled
            // reranking despite `enabled: true` in YAML — here it does nothing.
            std::env::set_var("RERANKING_ENABLED", "false");
            std::env::set_var("RERANKING_CANDIDATE_LIMIT", "999");
        }

        let yaml = r#"
reranking:
  enabled: true
  candidate_limit: 25
"#;
        let cfg = Config::from_str(yaml).unwrap();
        let reranking = cfg
            .reranking
            .expect("YAML enabled=true must win — RERANKING_ENABLED=false must be ignored");
        assert_eq!(
            reranking.candidate_limit, 25,
            "YAML candidate_limit must win — RERANKING_CANDIDATE_LIMIT must be ignored"
        );

        unsafe {
            std::env::remove_var("RERANKING_BASE_URL");
            std::env::remove_var("RERANKING_MODEL");
            std::env::remove_var("RERANKING_ENABLED");
            std::env::remove_var("RERANKING_CANDIDATE_LIMIT");
        }
        clear_required_env();
    }

    #[test]
    fn deprecated_env_var_is_detected_when_set() {
        let _lock = ENV_MUTEX.lock().unwrap();
        unsafe {
            std::env::remove_var("RERANKING_ENABLED");
            std::env::remove_var("RERANKING_CANDIDATE_LIMIT");
        }
        assert!(deprecated_env_vars_present().is_empty());

        unsafe {
            std::env::set_var("RERANKING_CANDIDATE_LIMIT", "999");
        }
        let present = deprecated_env_vars_present();
        assert_eq!(present, vec!["RERANKING_CANDIDATE_LIMIT"]);

        unsafe {
            std::env::set_var("RERANKING_ENABLED", "false");
        }
        let present = deprecated_env_vars_present();
        assert_eq!(
            present,
            vec!["RERANKING_ENABLED", "RERANKING_CANDIDATE_LIMIT"]
        );

        unsafe {
            std::env::remove_var("RERANKING_ENABLED");
            std::env::remove_var("RERANKING_CANDIDATE_LIMIT");
        }
    }

    // ── Provenance ───────────────────────────────────────────────────────────

    #[test]
    fn provenance_reports_env_source_with_var_name() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();

        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert_eq!(
            cfg.provenance.0.get("embedding.base_url"),
            Some(&SettingSource::Env {
                var: "EMBEDDING_BASE_URL".into()
            })
        );
        assert_eq!(
            cfg.provenance.0.get("qdrant.url"),
            Some(&SettingSource::Env {
                var: "QDRANT_URL".into()
            })
        );

        clear_required_env();
    }

    #[test]
    fn provenance_reports_default_for_unset_env_only_settings() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        unsafe {
            std::env::remove_var("MCP_PORT");
            std::env::remove_var("DATA_PATH");
        }

        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert_eq!(
            cfg.provenance.0.get("mcp.port"),
            Some(&SettingSource::Default)
        );
        assert_eq!(
            cfg.provenance.0.get("source.data_path"),
            Some(&SettingSource::Default)
        );

        clear_required_env();
    }

    #[test]
    fn provenance_never_carries_a_secret_value() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        unsafe {
            std::env::set_var("EMBEDDING_API_KEY", "sk-super-secret-value");
        }

        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        let source = cfg.provenance.0.get("embedding.api_key").unwrap();
        let described = source.describe();
        assert!(!described.contains("sk-super-secret-value"), "{described}");
        assert_eq!(described, "env EMBEDDING_API_KEY");

        unsafe {
            std::env::remove_var("EMBEDDING_API_KEY");
        }
        clear_required_env();
    }

    #[test]
    fn provenance_reports_yaml_for_a_present_section() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();

        // MINIMAL_CONFIG sets `chunking:` and `frontmatter:` but not `search:`.
        let cfg = Config::load(Path::new("/nonexistent/config.yaml")).unwrap();
        // No file at all: every YAML-only setting must report Default.
        assert_eq!(
            cfg.provenance.0.get("search.hybrid"),
            Some(&SettingSource::Default)
        );

        clear_required_env();
    }

    #[test]
    fn provenance_reports_yaml_when_section_present_in_loaded_file() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();

        let dir = std::env::temp_dir().join("md-kb-rag-test-provenance-yaml");
        std::fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.yaml");
        std::fs::write(&config_path, "search:\n  hybrid: false\n").unwrap();

        let cfg = Config::load(&config_path).unwrap();
        assert_eq!(
            cfg.provenance.0.get("search.hybrid"),
            Some(&SettingSource::Yaml)
        );
        assert_eq!(
            cfg.provenance.0.get("chunking.max_chunk_size"),
            Some(&SettingSource::Default),
            "a section absent from the file must still report Default"
        );

        std::fs::remove_dir_all(&dir).ok();
        clear_required_env();
    }

    /// Recursively flattens a `serde_yaml_ng::Value` tree into dot-joined leaf
    /// paths, mirroring `YAML_ONLY_SETTINGS`'s naming convention (e.g.
    /// `"ui.semantic_edges.k"`). A non-empty mapping is a sub-struct and is
    /// descended into; anything else — including an EMPTY mapping, which is how
    /// a `HashMap`-typed field (`frontmatter.defaults`/`allowed`) serializes at
    /// its default value — is treated as a leaf, since no real config sub-struct
    /// in this file has zero fields.
    fn collect_leaf_paths(value: &serde_yaml_ng::Value, prefix: String, out: &mut HashSet<String>) {
        match value.as_mapping() {
            Some(map) if !map.is_empty() => {
                for (k, v) in map {
                    let key = k.as_str().expect("Config keys are always strings");
                    let path = if prefix.is_empty() {
                        key.to_string()
                    } else {
                        format!("{prefix}.{key}")
                    };
                    collect_leaf_paths(v, path, out);
                }
            }
            _ => {
                out.insert(prefix);
            }
        }
    }

    #[test]
    fn yaml_only_settings_matches_every_config_struct_field() {
        // Regression test for #144: `YAML_ONLY_SETTINGS` is a hand-maintained
        // second source of truth parallel to the actual `Config` struct fields,
        // and it drifted exactly this way once already (`search.default_limit`/
        // `search.max_limit` were added to `SearchConfig` — and validated at
        // `resolve()` — without ever being added here, so `ConfigProvenance` had
        // no entry for either and silently omitted them from `/status` and the
        // startup log).
        //
        // Rather than hand-list every field a second time in the test too (the
        // same drift-prone pattern, just moved), this derives the real leaf-field
        // set straight from `Config`'s own `Default` impl via serialization, so
        // ANY future field added to ANY YAML-deserializable config struct without
        // a matching `YAML_ONLY_SETTINGS` entry fails this test — and any entry
        // left behind after a field is renamed or removed fails it too.
        let value = serde_yaml_ng::to_value(Config::default())
            .expect("every Config field type must round-trip through serde_yaml_ng");

        let mut leaves = HashSet::new();
        collect_leaf_paths(&value, String::new(), &mut leaves);

        let documented: HashSet<String> = YAML_ONLY_SETTINGS
            .iter()
            .map(|(name, _)| name.to_string())
            .collect();

        let undocumented: Vec<&String> = leaves.difference(&documented).collect();
        assert!(
            undocumented.is_empty(),
            "Config field(s) present on the struct but missing from YAML_ONLY_SETTINGS — \
             config provenance will silently omit them (see #144): {undocumented:?}"
        );

        let stale: Vec<&String> = documented.difference(&leaves).collect();
        assert!(
            stale.is_empty(),
            "YAML_ONLY_SETTINGS entr(y/ies) with no matching Config field — renamed or \
             removed without updating the table: {stale:?}"
        );
    }
}
