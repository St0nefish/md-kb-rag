//! Runtime config reload: `POST /admin/reload` re-reads and re-validates
//! `config.yaml` from disk and swaps it into the process without a restart.
//!
//! The ENV/YAML partition (`config.rs`) makes the SCOPE of a reload well-defined by
//! construction: every setting has exactly one legal source — ENV for anything
//! startup-only, YAML for everything else — so "reload" always means exactly one
//! thing: re-run [`Config::load`] on the same file and swap the result in. There is
//! no separate, hand-maintained list of "which fields reload is allowed to touch".
//!
//! What this module DOES hand-maintain is [`diff`]'s table of what happens to each
//! YAML setting once it actually changes. Some are read fresh on every use and take
//! effect immediately ([`ReloadEffect::Applied`]). Some only matter to *future*
//! indexing and leave existing Qdrant points inconsistent with the new setting until
//! a real reindex ([`ReloadEffect::ReindexRequired`]). The rest are baked into a
//! value or service built once at server startup — a `reqwest::Client` timeout, a
//! compiled `GlobSet`, a `GovernorLayer` — and stay exactly as they were until the
//! process restarts ([`ReloadEffect::RestartRequired`]). That table is unavoidable
//! bookkeeping, not a shortcut: reporting a setting as "applied" when the code that
//! matters never re-reads it would be a silent lie — the same class of failure as
//! the deployed `RERANKING_CANDIDATE_LIMIT` env var that motivated the ENV/YAML
//! partition in the first place (see `config::DEPRECATED_ENV_VARS`).

use std::path::Path;

use crate::config::{Config, ResolvedConfig, SharedConfig};

/// Where a changed setting's effect becomes visible, if at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReloadEffect {
    /// The code path(s) that use this setting read it fresh from the live
    /// [`SharedConfig`] — the very next read after the swap observes the new value.
    Applied,
    /// Baked into a value or service built once at server startup. The swap updates
    /// the snapshot every OTHER reader sees, but this particular consumer keeps
    /// whatever was true when it was constructed; only a restart changes what it
    /// actually does.
    RestartRequired,
    /// Read fresh by the indexer on its next run, but the change only reaches
    /// documents indexed AFTER that point — existing Qdrant points keep the chunk
    /// boundaries (or text basis) the OLD setting produced, so the corpus is
    /// inconsistent with the new config until `md-kb-rag index --full` rewrites it.
    ReindexRequired,
}

/// One setting whose resolved value changed across a reload.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SettingChange {
    /// Dotted config path, e.g. `"chunking.max_chunk_size"`. A few settings have
    /// more than one consumer with DIFFERENT reload behavior — `indexing.include`
    /// filters both the indexer (which re-reads it) and an MCP glob compiled once
    /// at startup (which does not). Those appear as two separate entries, the
    /// consumer named in parentheses.
    pub setting: String,
    pub old: String,
    pub new: String,
    /// One-sentence, caller-facing reason for the classification. See this module's
    /// doc comment and [`diff`]'s inline comments for the full reasoning and
    /// file:line evidence behind each one.
    pub note: String,
}

/// Every setting a reload changed, bucketed by [`ReloadEffect`]. Only settings whose
/// resolved value actually differs appear here — an unchanged setting has nothing to
/// report and cannot be misreported.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ReloadReport {
    pub applied: Vec<SettingChange>,
    pub restart_required: Vec<SettingChange>,
    pub reindex_required: Vec<SettingChange>,
}

impl ReloadReport {
    fn record(
        &mut self,
        effect: ReloadEffect,
        setting: &str,
        old: String,
        new: String,
        note: &str,
    ) {
        if old == new {
            return;
        }
        let change = SettingChange {
            setting: setting.to_string(),
            old,
            new,
            note: note.to_string(),
        };
        match effect {
            ReloadEffect::Applied => self.applied.push(change),
            ReloadEffect::RestartRequired => self.restart_required.push(change),
            ReloadEffect::ReindexRequired => self.reindex_required.push(change),
        }
    }

    /// True when nothing observable changed at all.
    pub fn is_empty(&self) -> bool {
        self.applied.is_empty()
            && self.restart_required.is_empty()
            && self.reindex_required.is_empty()
    }
}

/// Debug-format a value for the report's `old`/`new` fields. None of the settings
/// compared in [`diff`] are secrets — secret VALUES live only behind the `*_env`
/// name-indirection fields (`embedding.api_key_env`, `reranking.api_key_env`,
/// `source.git_token_env`, `webhook.secret_env`, `mcp.bearer_token_env`), and this
/// module never diffs the looked-up secret itself, only the env var NAME for the
/// one indirection field (`source.git_token_env`) that has a runtime consumer worth
/// reporting on — see the comment on the `embedding`/`reranking` sections below for
/// why the other two are not diffed at all.
fn d<T: std::fmt::Debug>(v: &T) -> String {
    format!("{v:?}")
}

/// Compare every YAML-reloadable setting between `old` and `new`, recording each one
/// that changed with its classification. This is the "hard part" this feature exists
/// for: a setting only belongs in [`ReloadEffect::Applied`] if some code path
/// genuinely re-reads the live [`SharedConfig`] (or an equivalent per-run/per-request
/// snapshot) rather than a value captured once at startup.
///
/// `embedding.api_key_env` and `reranking.api_key_env` are deliberately absent from
/// this diff: `ResolvedConfig` only retains the env var's looked-up VALUE
/// (`ResolvedEmbeddingConfig::api_key` / `ResolvedRerankingConfig::api_key`), not the
/// var's NAME — the name is consumed once inside `Config::resolve_inner` and
/// discarded. Both are restart-required regardless (`EmbedClient`/`RerankClient`
/// bake the resolved key in at construction — `embed.rs`, `rerank.rs`), this just
/// means a reload has nothing to diff for that specific setting changing.
///
/// `ResolvedRerankingConfig::max_document_bytes` is likewise absent, and is not a
/// YAML setting at all: it is derived from `chunking.max_chunk_size` in
/// `Config::resolve_inner` and baked into `RerankClient` at construction. It rides
/// on the classification `chunking.max_chunk_size` already has (reindex-required),
/// so a change to that number reaches the reranker only on restart — the same
/// lifetime as every other `RerankClient` field.
pub fn diff(old: &ResolvedConfig, new: &ResolvedConfig) -> ReloadReport {
    let mut r = ReloadReport::default();

    // ── source ───────────────────────────────────────────────────────────────
    // `source.git_token_env` names the env var used to authenticate git operations.
    // It has two independent consumers with different lifetimes: the MCP write
    // tools resolve it fresh on every call, but the webhook/startup-clone path
    // resolves it once and keeps the result for the process's life.
    if old.source.git_token_env != new.source.git_token_env {
        r.record(
            ReloadEffect::Applied,
            "source.git_token_env (MCP write-tool commits)",
            d(&old.source.git_token_env),
            d(&new.source.git_token_env),
            "create_document/edit_document/delete_document resolve this fresh from \
             the live config on every call (mcp.rs write_document/delete_document) \
             before reading the named env var — the very next write picks up a new \
             var name.",
        );
        r.record(
            ReloadEffect::RestartRequired,
            "source.git_token_env (webhook pull / startup clone)",
            d(&old.source.git_token_env),
            d(&new.source.git_token_env),
            "the token is resolved once at server startup (server.rs run_server) and \
             stored by value on WebhookState and passed to the initial \
             git::ensure_repo call; neither re-reads the env var name afterward.",
        );
    }

    // ── indexing ─────────────────────────────────────────────────────────────
    if old.indexing.include != new.indexing.include {
        r.record(
            ReloadEffect::Applied,
            "indexing.include (scan / reconcile filtering)",
            d(&old.indexing.include),
            d(&new.indexing.include),
            "ingest::discover_relative reads config.indexing fresh on every indexing \
             run (ingest.rs); the reindex worker loads a fresh config snapshot before \
             each drain, and a successful reload queues an immediate full reconcile \
             so this is picked up right away rather than on the next periodic sweep.",
        );
        r.record(
            ReloadEffect::RestartRequired,
            "indexing.include (MCP get_document path filter)",
            d(&old.indexing.include),
            d(&new.indexing.include),
            "KbSearchServer compiles this into a GlobSet once at construction \
             (mcp.rs build_include_globset, invoked from KbSearchServer::new); \
             get_document keeps filtering against the old patterns until restart.",
        );
    }
    if old.indexing.exclude != new.indexing.exclude {
        r.record(
            ReloadEffect::Applied,
            "indexing.exclude",
            d(&old.indexing.exclude),
            d(&new.indexing.exclude),
            "read fresh per indexing run by ingest::discover_files (ingest.rs); \
             unlike indexing.include, no MCP path filter bakes this in.",
        );
    }
    if old.indexing.exclude_files != new.indexing.exclude_files {
        r.record(
            ReloadEffect::Applied,
            "indexing.exclude_files",
            d(&old.indexing.exclude_files),
            d(&new.indexing.exclude_files),
            "read fresh per indexing run by ingest::discover_files (ingest.rs).",
        );
    }
    if old.indexing.reconcile_interval_secs != new.indexing.reconcile_interval_secs {
        r.record(
            ReloadEffect::Applied,
            "indexing.reconcile_interval_secs",
            d(&old.indexing.reconcile_interval_secs),
            d(&new.indexing.reconcile_interval_secs),
            "the periodic reconcile task (server.rs run_server) reads this fresh from \
             the live config at the top of every iteration to schedule its next \
             sleep — takes effect starting with the next sleep it schedules.",
        );
    }

    // ── frontmatter ──────────────────────────────────────────────────────────
    // Feeds the implicit root schema (`ResolvedSchema::from_config`), which is
    // rebuilt by the reindex worker before the next unit that touches a schema file
    // or a full reconcile, and synchronously by update_schema's own rebuild. A
    // successful reload always queues a full reconcile, so these reach the shared
    // schema cache immediately rather than waiting for the periodic sweep.
    if old.frontmatter.required != new.frontmatter.required {
        r.record(
            ReloadEffect::Applied,
            "frontmatter.required",
            d(&old.frontmatter.required),
            d(&new.frontmatter.required),
            "feeds the implicit root schema, rebuilt by the reindex worker (a \
             reload queues an immediate full reconcile) and by update_schema's own \
             synchronous rebuild.",
        );
    }
    if old.frontmatter.indexed_fields != new.frontmatter.indexed_fields {
        r.record(
            ReloadEffect::Applied,
            "frontmatter.indexed_fields",
            d(&old.frontmatter.indexed_fields),
            d(&new.frontmatter.indexed_fields),
            "index_paths re-ensures Qdrant payload indexes for \
             effective_indexed_fields on every run (ingest.rs), so a newly indexed \
             field gets its payload index on the next run; existing documents' \
             document_fields projection for that field backfills only when they are \
             next reindexed, or immediately via `md-kb-rag reproject-fields`.",
        );
    }
    if old.frontmatter.defaults != new.frontmatter.defaults {
        r.record(
            ReloadEffect::Applied,
            "frontmatter.defaults",
            d(&old.frontmatter.defaults),
            d(&new.frontmatter.defaults),
            "applied by validate::apply_defaults during validation, against the \
             schema rebuilt from the live config on the next indexing run or write.",
        );
    }
    if old.frontmatter.allowed != new.frontmatter.allowed {
        r.record(
            ReloadEffect::Applied,
            "frontmatter.allowed",
            d(&old.frontmatter.allowed),
            d(&new.frontmatter.allowed),
            "feeds the implicit root schema's value checks, rebuilt the same way as \
             frontmatter.required above.",
        );
    }

    // ── chunking ─────────────────────────────────────────────────────────────
    // The indexer reads these fresh (same fresh-config path as indexing.*/
    // frontmatter.* above), but the EFFECT only reaches documents chunked after the
    // change — this is the reindex case the task spec calls out by name.
    if old.chunking.max_chunk_size != new.chunking.max_chunk_size {
        r.record(
            ReloadEffect::ReindexRequired,
            "chunking.max_chunk_size",
            d(&old.chunking.max_chunk_size),
            d(&new.chunking.max_chunk_size),
            "read fresh by the chunker on the next indexing run, but only for \
             documents that run touches — existing Qdrant chunks keep the old \
             boundaries. Run `md-kb-rag index --full` for a consistent corpus.",
        );
    }
    if old.chunking.target_chunk_size != new.chunking.target_chunk_size {
        r.record(
            ReloadEffect::ReindexRequired,
            "chunking.target_chunk_size",
            d(&old.chunking.target_chunk_size),
            d(&new.chunking.target_chunk_size),
            "same as chunking.max_chunk_size: applies to future chunking only. Run \
             `md-kb-rag index --full` for a consistent corpus.",
        );
    }
    if old.chunking.prepend_description != new.chunking.prepend_description {
        r.record(
            ReloadEffect::ReindexRequired,
            "chunking.prepend_description",
            d(&old.chunking.prepend_description),
            d(&new.chunking.prepend_description),
            "changes the text every future chunk embeds (and, incidentally, the \
             create_document dedup query text — mcp.rs build_dedup_query); existing \
             chunks were embedded on the old basis. Run `md-kb-rag index --full` for \
             a consistent corpus.",
        );
    }
    if old.chunking.prepend_heading_path != new.chunking.prepend_heading_path {
        r.record(
            ReloadEffect::ReindexRequired,
            "chunking.prepend_heading_path",
            d(&old.chunking.prepend_heading_path),
            d(&new.chunking.prepend_heading_path),
            "changes the text every future chunk embeds — a chunk's heading-ancestry \
             breadcrumb is prepended before embedding, so existing chunks were embedded \
             without it (or with it). Same basis-change as prepend_description above, \
             including the dedup query text. Run `md-kb-rag index --full` for a \
             consistent corpus.",
        );
    }

    // ── embedding ────────────────────────────────────────────────────────────
    // EmbedClient is built exactly once, at server startup, from a `&ResolvedEmbeddingConfig`
    // snapshot — every field below is captured by value or baked into a
    // `reqwest::Client` at that point (embed.rs EmbedClient::new).
    if old.embedding.batch_size != new.embedding.batch_size {
        r.record(
            ReloadEffect::RestartRequired,
            "embedding.batch_size",
            d(&old.embedding.batch_size),
            d(&new.embedding.batch_size),
            "EmbedClient captures this by value at construction (embed.rs \
             EmbedClient::new); it is built once at server startup and never \
             rebuilt.",
        );
    }
    if old.embedding.request_timeout_secs != new.embedding.request_timeout_secs {
        r.record(
            ReloadEffect::RestartRequired,
            "embedding.request_timeout_secs",
            d(&old.embedding.request_timeout_secs),
            d(&new.embedding.request_timeout_secs),
            "baked into the reqwest::Client's timeout when EmbedClient is \
             constructed (embed.rs); a reqwest::Client's timeout cannot be changed \
             after the client is built.",
        );
    }
    if old.embedding.batch_concurrency != new.embedding.batch_concurrency {
        r.record(
            ReloadEffect::RestartRequired,
            "embedding.batch_concurrency",
            d(&old.embedding.batch_concurrency),
            d(&new.embedding.batch_concurrency),
            "EmbedClient captures this by value at construction (embed.rs \
             EmbedClient::new).",
        );
    }

    // ── validation ───────────────────────────────────────────────────────────
    if old.validation.enabled != new.validation.enabled {
        r.record(
            ReloadEffect::Applied,
            "validation.enabled",
            d(&old.validation.enabled),
            d(&new.validation.enabled),
            "read fresh per indexing run (ingest.rs process_file); a reload queues \
             an immediate full reconcile.",
        );
    }
    if old.validation.strict != new.validation.strict {
        r.record(
            ReloadEffect::Applied,
            "validation.strict",
            d(&old.validation.strict),
            d(&new.validation.strict),
            "read fresh per indexing run (ingest.rs process_file).",
        );
    }
    if old.validation.lint_command != new.validation.lint_command {
        r.record(
            ReloadEffect::Applied,
            "validation.lint_command",
            d(&old.validation.lint_command),
            d(&new.validation.lint_command),
            "read fresh per file validated (validate.rs), invoked from the next \
             indexing run or MCP write.",
        );
    }
    if old.validation.lint_timeout_secs != new.validation.lint_timeout_secs {
        r.record(
            ReloadEffect::Applied,
            "validation.lint_timeout_secs",
            d(&old.validation.lint_timeout_secs),
            d(&new.validation.lint_timeout_secs),
            "read fresh per file validated, the same call site as \
             validation.lint_command above (validate.rs) — bounds how long a \
             configured lint command is allowed to run before that file's \
             validation fails with a timeout.",
        );
    }

    // ── webhook ──────────────────────────────────────────────────────────────
    if old.webhook.secret_env != new.webhook.secret_env {
        r.record(
            ReloadEffect::RestartRequired,
            "webhook.secret_env",
            d(&old.webhook.secret_env),
            d(&new.webhook.secret_env),
            "the secret is resolved once at server startup (server.rs run_server) \
             and stored by value on WebhookState.secret; whether /hooks/reindex even \
             exists is also decided at startup from this same lookup.",
        );
    }
    if old.webhook.provider != new.webhook.provider {
        r.record(
            ReloadEffect::Applied,
            "webhook.provider",
            d(&old.webhook.provider),
            d(&new.webhook.provider),
            "read fresh from the live config on every webhook request (webhook.rs \
             handle_webhook) to pick the signature header and verification scheme.",
        );
    }

    // ── mcp ──────────────────────────────────────────────────────────────────
    if old.mcp.bearer_token_env != new.mcp.bearer_token_env {
        r.record(
            ReloadEffect::RestartRequired,
            "mcp.bearer_token_env",
            d(&old.mcp.bearer_token_env),
            d(&new.mcp.bearer_token_env),
            "the bearer token is resolved once at server startup and stored by \
             value on AuthState (server.rs run_server) — security-critical, \
             deliberately not made live.",
        );
    }
    if old.mcp.allow_unauthenticated != new.mcp.allow_unauthenticated {
        r.record(
            ReloadEffect::RestartRequired,
            "mcp.allow_unauthenticated",
            d(&old.mcp.allow_unauthenticated),
            d(&new.mcp.allow_unauthenticated),
            "gates how AuthState is built at startup (server.rs run_server) — \
             security-critical, deliberately not made live.",
        );
    }
    if old.mcp.instructions != new.mcp.instructions {
        r.record(
            ReloadEffect::Applied,
            "mcp.instructions",
            d(&old.mcp.instructions),
            d(&new.mcp.instructions),
            "the metadata-refresh timer (server.rs run_server) reads this fresh from \
             the live config every tick; takes effect on the next tick, within \
             mcp.metadata_refresh_secs seconds.",
        );
    }
    if old.mcp.metadata_refresh_secs != new.mcp.metadata_refresh_secs {
        r.record(
            ReloadEffect::Applied,
            "mcp.metadata_refresh_secs",
            d(&old.mcp.metadata_refresh_secs),
            d(&new.mcp.metadata_refresh_secs),
            "the same refresh timer reads this fresh at the top of every iteration \
             to schedule its next sleep.",
        );
    }
    if old.mcp.allowed_hosts != new.mcp.allowed_hosts {
        r.record(
            ReloadEffect::RestartRequired,
            "mcp.allowed_hosts",
            d(&old.mcp.allowed_hosts),
            d(&new.mcp.allowed_hosts),
            "baked into the StreamableHttpServerConfig when the MCP service is \
             built (server.rs mcp_transport_config, invoked once from run_server).",
        );
    }
    if old.mcp.extensions_path != new.mcp.extensions_path {
        r.record(
            ReloadEffect::Applied,
            "mcp.extensions_path",
            d(&old.mcp.extensions_path),
            d(&new.mcp.extensions_path),
            "the metadata-refresh timer (server.rs run_server) re-resolves this fresh \
             from the live config every tick and recomposes both the server \
             instructions and every tool's description overlay from it; takes effect \
             on the next tick, within mcp.metadata_refresh_secs seconds.",
        );
    }

    // ── rate_limit ───────────────────────────────────────────────────────────
    // The whole section is restart-required: whether GovernorLayer is even added to
    // the router, and every knob on it, is decided once when the router is built.
    if old.rate_limit.enabled != new.rate_limit.enabled {
        r.record(
            ReloadEffect::RestartRequired,
            "rate_limit.enabled",
            d(&old.rate_limit.enabled),
            d(&new.rate_limit.enabled),
            "decides whether GovernorLayer is added to the router at all (server.rs \
             run_server); the router is built once.",
        );
    }
    if old.rate_limit.per_second != new.rate_limit.per_second {
        r.record(
            ReloadEffect::RestartRequired,
            "rate_limit.per_second",
            d(&old.rate_limit.per_second),
            d(&new.rate_limit.per_second),
            "baked into the GovernorConfigBuilder when the rate limiter is built \
             once at startup (server.rs run_server).",
        );
    }
    if old.rate_limit.burst_size != new.rate_limit.burst_size {
        r.record(
            ReloadEffect::RestartRequired,
            "rate_limit.burst_size",
            d(&old.rate_limit.burst_size),
            d(&new.rate_limit.burst_size),
            "baked into the GovernorConfigBuilder when the rate limiter is built \
             once at startup (server.rs run_server).",
        );
    }

    // ── write ────────────────────────────────────────────────────────────────
    if old.write.dedup_enabled != new.write.dedup_enabled {
        r.record(
            ReloadEffect::Applied,
            "write.dedup_enabled",
            d(&old.write.dedup_enabled),
            d(&new.write.dedup_enabled),
            "read fresh from the live config on every create_document call (mcp.rs \
             write_document).",
        );
    }
    if old.write.dedup_threshold != new.write.dedup_threshold {
        r.record(
            ReloadEffect::Applied,
            "write.dedup_threshold",
            d(&old.write.dedup_threshold),
            d(&new.write.dedup_threshold),
            "read fresh from the live config on every create_document call (mcp.rs \
             write_document).",
        );
    }
    if old.write.commit_author_name != new.write.commit_author_name {
        r.record(
            ReloadEffect::Applied,
            "write.commit_author_name",
            d(&old.write.commit_author_name),
            d(&new.write.commit_author_name),
            "read fresh from the live config on every write-tool commit (mcp.rs).",
        );
    }
    if old.write.commit_author_email != new.write.commit_author_email {
        r.record(
            ReloadEffect::Applied,
            "write.commit_author_email",
            d(&old.write.commit_author_email),
            d(&new.write.commit_author_email),
            "read fresh from the live config on every write-tool commit (mcp.rs).",
        );
    }

    // ── search ───────────────────────────────────────────────────────────────
    if old.search.phrase != new.search.phrase {
        r.record(
            ReloadEffect::Applied,
            "search.phrase",
            d(&old.search.phrase),
            d(&new.search.phrase),
            "read fresh from the live config on every search call (mcp.rs/web.rs, \
             gated by phrase_matching_available) and by the metadata-refresh \
             timer (server.rs compose_server_instructions/compose_tool_overlay) \
             that recomposes the server/tool description overlay so it never \
             advertises quoted-phrase support the config just turned off.",
        );
    }
    if old.search.hybrid != new.search.hybrid {
        r.record(
            ReloadEffect::Applied,
            "search.hybrid",
            d(&old.search.hybrid),
            d(&new.search.hybrid),
            "read fresh from the live config on every search call (mcp.rs search).",
        );
    }
    if old.search.rrf_candidates != new.search.rrf_candidates {
        r.record(
            ReloadEffect::Applied,
            "search.rrf_candidates",
            d(&old.search.rrf_candidates),
            d(&new.search.rrf_candidates),
            "read fresh from the live config on every search call (mcp.rs search).",
        );
    }
    if old.search.min_score != new.search.min_score {
        r.record(
            ReloadEffect::Applied,
            "search.min_score",
            d(&old.search.min_score),
            d(&new.search.min_score),
            "read fresh from the live config on every search call (mcp.rs search) \
             as the default when the caller does not pass their own min_score.",
        );
    }
    if old.search.diversity_max_per_document != new.search.diversity_max_per_document {
        r.record(
            ReloadEffect::Applied,
            "search.diversity_max_per_document",
            d(&old.search.diversity_max_per_document),
            d(&new.search.diversity_max_per_document),
            "read fresh from the live config on every search call (mcp.rs search, \
             retrieval::search's per-document diversity cap).",
        );
    }
    if old.search.default_limit != new.search.default_limit {
        r.record(
            ReloadEffect::Applied,
            "search.default_limit",
            d(&old.search.default_limit),
            d(&new.search.default_limit),
            "read fresh from the live config on every search call (mcp.rs \
             resolve_limit) when the caller omits limit.",
        );
    }
    if old.search.max_limit != new.search.max_limit {
        r.record(
            ReloadEffect::Applied,
            "search.max_limit",
            d(&old.search.max_limit),
            d(&new.search.max_limit),
            "read fresh from the live config on every search call (mcp.rs \
             resolve_limit) as the ceiling a requested limit is clamped to.",
        );
    }

    // ── reranking ────────────────────────────────────────────────────────────
    if old.reranking.is_some() != new.reranking.is_some() {
        r.record(
            ReloadEffect::RestartRequired,
            "reranking.enabled",
            d(&old.reranking.is_some()),
            d(&new.reranking.is_some()),
            "gates whether KbSearchServer even holds a RerankClient at all \
             (server.rs run_server); the Option is decided once at construction — a \
             reload cannot add or remove the client.",
        );
    }
    if let (Some(old_r), Some(new_r)) = (&old.reranking, &new.reranking)
        && old_r.candidate_limit != new_r.candidate_limit
    {
        r.record(
            ReloadEffect::Applied,
            "reranking.candidate_limit",
            d(&old_r.candidate_limit),
            d(&new_r.candidate_limit),
            "read fresh from the live config on every search call (mcp.rs search) — \
             but only takes effect when reranking was already enabled at startup; a \
             server that started with it disabled has no RerankClient to hand this \
             to regardless.",
        );
    }

    // ── ui.semantic_edges ────────────────────────────────────────────────────
    // #226: previously missing from this table entirely — every change below it
    // reported nothing on reload, which reads as "did not take effect" even
    // though the config genuinely is re-read on the next indexing run.
    //
    // Classified ReindexRequired, NOT Applied, despite the values being read
    // fresh: `ingest::index_paths` passes `&config.ui.semantic_edges` fresh to
    // `update_semantic_edges` on every run, but that call only covers `pending`
    // — the files THAT RUN actually re-chunked/re-embedded (ingest.rs, right
    // after `upsert_pending`). A reload's automatic full reconcile
    // (`queue.mark_full()`, unconditional — see this function's doc comment)
    // still goes through the worker's ordinary `scan_for_dirty`, which skips any
    // file whose content hash is unchanged; only `md-kb-rag index --full`
    // (`force = true`) bypasses that skip and reprocesses everything. So exactly
    // like `chunking.*` above: flipping `ui.semantic_edges.enabled` on does not
    // retroactively populate semantic edges for the existing corpus, and
    // flipping it off does not retroactively remove already-computed ones —
    // both only apply to documents a future run actually touches, same caveat,
    // same fix (`md-kb-rag index --full`).
    if old.ui.semantic_edges.enabled != new.ui.semantic_edges.enabled {
        r.record(
            ReloadEffect::ReindexRequired,
            "ui.semantic_edges.enabled",
            d(&old.ui.semantic_edges.enabled),
            d(&new.ui.semantic_edges.enabled),
            "read fresh by the indexer on the next run (ingest.rs \
             update_semantic_edges), but only for documents that run re-embeds — \
             existing (or missing) semantic edges in the graph view are \
             unchanged otherwise. Run `md-kb-rag index --full` for a consistent \
             corpus.",
        );
    }
    if old.ui.semantic_edges.k != new.ui.semantic_edges.k {
        r.record(
            ReloadEffect::ReindexRequired,
            "ui.semantic_edges.k",
            d(&old.ui.semantic_edges.k),
            d(&new.ui.semantic_edges.k),
            "same as ui.semantic_edges.enabled: applies to future re-embeds only. \
             Run `md-kb-rag index --full` for a consistent corpus.",
        );
    }
    if old.ui.semantic_edges.min_score != new.ui.semantic_edges.min_score {
        r.record(
            ReloadEffect::ReindexRequired,
            "ui.semantic_edges.min_score",
            d(&old.ui.semantic_edges.min_score),
            d(&new.ui.semantic_edges.min_score),
            "same as ui.semantic_edges.enabled: applies to future re-embeds only. \
             Run `md-kb-rag index --full` for a consistent corpus.",
        );
    }

    r
}

/// Re-read and re-validate `path`, then atomically swap the result into `shared`.
///
/// On ANY parse or validation error, `shared` is left completely untouched: this
/// calls the exact same [`Config::load`] the startup path uses (same `bail!`
/// checks, same missing-env-var reporting), so a malformed file fails exactly the
/// way it would have failed a restart — the only difference is this returns the
/// error to an HTTP caller instead of exiting the process.
///
/// After a successful swap, an immediate full reconcile is queued
/// (`reindex::mark_full`) so every indexing-observing setting that changed —
/// `indexing.include`/`exclude`, `frontmatter.*`, `validation.*` — takes effect on
/// the reindex worker's very next wake rather than waiting for the periodic sweep
/// (`indexing.reconcile_interval_secs`, several minutes by default). This is queued
/// unconditionally, not only when [`diff`] says something indexing-related changed:
/// `scan_for_dirty` is deliberately cheap — it pages `indexed_files` and stats what
/// it already knows about rather than hashing file content, see its doc comment —
/// so the cost of one extra sweep is far smaller than the risk of this function's
/// classification missing a case and silently leaving a change unpicked-up.
pub fn reload_config(
    path: &Path,
    shared: &SharedConfig,
    queue: &crate::reindex::ReindexQueue,
) -> anyhow::Result<ReloadReport> {
    let new = Config::load(path)?;
    let old = crate::config::load_shared_config(shared);
    let report = diff(&old, &new);
    crate::config::store_shared_config(shared, new);
    queue.mark_full();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ResolvedRerankingConfig;
    // Reused rather than duplicated: a second, module-local `Mutex` guarding the
    // same process-global env vars would not actually serialize anything against
    // config.rs's own env-var tests, since `cargo test` runs every test in this
    // crate in one multi-threaded binary. See `config::test_support`'s doc comment.
    use crate::config::test_support::{ENV_MUTEX, clear_required_env, set_required_env};
    use std::io::Write as _;

    /// A YAML doc using only settings that are still YAML-settable (see
    /// config.rs's `MINIMAL_CONFIG`), plus the env vars `Config::load` requires.
    fn write_config(dir: &Path, yaml: &str) -> std::path::PathBuf {
        let path = dir.join("config.yaml");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(yaml.as_bytes()).unwrap();
        path
    }

    fn base_config() -> ResolvedConfig {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();
        let tmp = tempfile::tempdir().unwrap();
        let path = write_config(tmp.path(), "chunking:\n  max_chunk_size: 1000\n");
        let cfg = Config::load(&path).unwrap();
        clear_required_env();
        cfg
    }

    #[test]
    fn unchanged_config_produces_an_empty_report() {
        let cfg = base_config();
        let report = diff(&cfg, &cfg);
        assert!(report.is_empty());
    }

    #[test]
    fn a_dynamic_setting_is_reported_as_applied() {
        let mut old = base_config();
        let mut new = base_config();
        old.search.hybrid = true;
        new.search.hybrid = false;

        let report = diff(&old, &new);
        assert_eq!(report.applied.len(), 1);
        assert_eq!(report.applied[0].setting, "search.hybrid");
        assert_eq!(report.applied[0].old, "true");
        assert_eq!(report.applied[0].new, "false");
        assert!(report.restart_required.is_empty());
        assert!(report.reindex_required.is_empty());
    }

    #[test]
    fn a_restart_required_setting_is_never_reported_as_applied() {
        let mut old = base_config();
        let mut new = base_config();
        old.rate_limit.per_second = 20;
        new.rate_limit.per_second = 5;

        let report = diff(&old, &new);
        assert!(
            report.applied.is_empty(),
            "rate_limit.per_second must never be claimed as applied: {:?}",
            report.applied
        );
        assert_eq!(report.restart_required.len(), 1);
        assert_eq!(report.restart_required[0].setting, "rate_limit.per_second");
    }

    #[test]
    fn a_chunking_setting_is_reported_as_reindex_required_not_applied() {
        let mut old = base_config();
        let mut new = base_config();
        old.chunking.max_chunk_size = 1000;
        new.chunking.max_chunk_size = 2000;

        let report = diff(&old, &new);
        assert!(report.applied.is_empty());
        assert!(report.restart_required.is_empty());
        assert_eq!(report.reindex_required.len(), 1);
        assert_eq!(
            report.reindex_required[0].setting,
            "chunking.max_chunk_size"
        );
    }

    #[test]
    fn indexing_include_reports_both_its_dynamic_and_static_consumers() {
        let mut old = base_config();
        let mut new = base_config();
        old.indexing.include = vec!["**/*.md".into()];
        new.indexing.include = vec!["**/*.mdx".into()];

        let report = diff(&old, &new);
        assert_eq!(report.applied.len(), 1);
        assert!(
            report.applied[0]
                .setting
                .contains("scan / reconcile filtering")
        );
        assert_eq!(report.restart_required.len(), 1);
        assert!(
            report.restart_required[0]
                .setting
                .contains("MCP get_document path filter")
        );
    }

    #[test]
    fn reranking_candidate_limit_only_diffed_when_both_sides_are_enabled() {
        let mut old = base_config();
        let mut new = base_config();
        // Disabled on both sides: nothing to compare, nothing reported.
        let report = diff(&old, &new);
        assert!(report.is_empty());

        let rr = |limit: usize| {
            Some(ResolvedRerankingConfig {
                base_url: "http://localhost:9000".into(),
                model: "test".into(),
                api_key: None,
                candidate_limit: limit,
                max_document_bytes: 1500,
            })
        };
        old.reranking = rr(50);
        new.reranking = rr(100);
        let report = diff(&old, &new);
        assert_eq!(report.applied.len(), 1);
        assert_eq!(report.applied[0].setting, "reranking.candidate_limit");
    }

    #[test]
    fn reload_rejects_an_invalid_config_and_leaves_the_running_config_untouched() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();

        let tmp = tempfile::tempdir().unwrap();
        let good_path = write_config(tmp.path(), "chunking:\n  max_chunk_size: 1234\n");
        let running = Config::load(&good_path).unwrap();
        let shared = crate::config::shared_config(std::sync::Arc::new(running));

        // target_chunk_size > max_chunk_size fails the same bail! the startup path runs.
        let bad_path = write_config(
            tmp.path(),
            "chunking:\n  max_chunk_size: 100\n  target_chunk_size: 500\n",
        );
        let queue = crate::reindex::ReindexQueue::new();
        let result = reload_config(&bad_path, &shared, &queue);
        assert!(result.is_err());

        let still_running = crate::config::load_shared_config(&shared);
        assert_eq!(
            still_running.chunking.max_chunk_size, 1234,
            "a failed reload must leave the running config completely untouched"
        );

        clear_required_env();
    }

    #[test]
    fn reload_swaps_in_a_valid_config_and_reports_the_diff() {
        let _lock = ENV_MUTEX.lock().unwrap();
        set_required_env();

        let tmp = tempfile::tempdir().unwrap();
        let path = write_config(tmp.path(), "search:\n  hybrid: true\n");
        let running = Config::load(&path).unwrap();
        let shared = crate::config::shared_config(std::sync::Arc::new(running));

        write_config(tmp.path(), "search:\n  hybrid: false\n");
        let queue = crate::reindex::ReindexQueue::new();
        let report = reload_config(&path, &shared, &queue).unwrap();

        assert_eq!(report.applied.len(), 1);
        assert_eq!(report.applied[0].setting, "search.hybrid");
        let live = crate::config::load_shared_config(&shared);
        assert!(!live.search.hybrid, "the swap must actually take effect");

        clear_required_env();
    }

    // --- #226: ui.* reporting -------------------------------------------------

    #[test]
    fn ui_semantic_edges_changes_are_reported_as_reindex_required() {
        // Regression test for #226: before this change, `diff()` had no entries
        // at all for `ui.*` — a changed `ui.semantic_edges.*` setting produced an
        // EMPTY report despite genuinely taking effect (on the next indexing run
        // that touches a document), which reads to an operator as "did not
        // apply" when it did. This must fail against the pre-#226 `diff()` (no
        // ui.semantic_edges branch at all, so `report.is_empty()`) and pass now
        // that the branch exists — and it must land under `reindex_required`,
        // not `applied`: see the `ui.semantic_edges` section's comment in
        // `diff()` for why (only future re-embeds pick it up, exactly like
        // `chunking.*`).
        let mut old = base_config();
        let mut new = base_config();
        old.ui.semantic_edges.enabled = false;
        new.ui.semantic_edges.enabled = true;
        old.ui.semantic_edges.k = 5;
        new.ui.semantic_edges.k = 10;
        old.ui.semantic_edges.min_score = 0.6;
        new.ui.semantic_edges.min_score = 0.8;

        let report = diff(&old, &new);
        assert!(
            report.applied.is_empty(),
            "ui.semantic_edges.* must never be claimed as applied: {:?}",
            report.applied
        );
        assert!(report.restart_required.is_empty());
        assert_eq!(report.reindex_required.len(), 3);
        let settings: std::collections::BTreeSet<&str> = report
            .reindex_required
            .iter()
            .map(|c| c.setting.as_str())
            .collect();
        assert_eq!(
            settings,
            std::collections::BTreeSet::from([
                "ui.semantic_edges.enabled",
                "ui.semantic_edges.k",
                "ui.semantic_edges.min_score",
            ])
        );
    }

    #[test]
    fn search_phrase_is_reported_as_applied() {
        // Also previously missing from `diff()` entirely, discovered by the
        // #226 drift test below rather than named in the issue itself — the
        // same drift class, just a different field.
        let mut old = base_config();
        let mut new = base_config();
        old.search.phrase = true;
        new.search.phrase = false;

        let report = diff(&old, &new);
        assert_eq!(report.applied.len(), 1);
        assert_eq!(report.applied[0].setting, "search.phrase");
        assert!(report.restart_required.is_empty());
        assert!(report.reindex_required.is_empty());
    }

    #[test]
    fn validation_lint_timeout_secs_is_reported_as_applied() {
        // Same story as `search_phrase_is_reported_as_applied` above.
        let mut old = base_config();
        let mut new = base_config();
        old.validation.lint_timeout_secs = 30;
        new.validation.lint_timeout_secs = 60;

        let report = diff(&old, &new);
        assert_eq!(report.applied.len(), 1);
        assert_eq!(report.applied[0].setting, "validation.lint_timeout_secs");
    }

    /// Every dotted setting [`diff`] actually compares — one entry per
    /// underlying `Config`/`ResolvedConfig` field, not per "(consumer)"-suffixed
    /// display variant (`source.git_token_env` and `indexing.include` each
    /// produce TWO [`SettingChange`] rows from ONE `if old.X != new.X` check, for
    /// their two independently-lifetimed consumers — see `diff`'s own comments
    /// on those sections). This is [`diff`]'s counterpart to `config.rs`'s
    /// `YAML_ONLY_SETTINGS`: a hand-maintained list documenting what real code
    /// covers, checked below against the same programmatically-derived ground
    /// truth (`config::config_setting_leaf_paths`) #144's test already
    /// established — see that test for why deriving the ground truth beats
    /// hand-listing it a second time.
    ///
    /// Two settings are deliberately absent, matching `diff`'s own doc comment
    /// on why: `embedding.api_key_env` and `reranking.api_key_env` have no
    /// `ResolvedConfig` field to diff at all — `Config::resolve_inner` looks the
    /// env var up once and keeps only the resolved secret VALUE
    /// (`ResolvedEmbeddingConfig::api_key`/`ResolvedRerankingConfig::api_key`),
    /// discarding the var's name — so there is nothing here for `diff` to
    /// compare regardless of classification.
    const RELOAD_DIFF_SETTINGS: &[&str] = &[
        "source.git_token_env",
        "indexing.include",
        "indexing.exclude",
        "indexing.exclude_files",
        "indexing.reconcile_interval_secs",
        "frontmatter.required",
        "frontmatter.indexed_fields",
        "frontmatter.defaults",
        "frontmatter.allowed",
        "chunking.max_chunk_size",
        "chunking.target_chunk_size",
        "chunking.prepend_description",
        "chunking.prepend_heading_path",
        "embedding.batch_size",
        "embedding.request_timeout_secs",
        "embedding.batch_concurrency",
        "validation.enabled",
        "validation.strict",
        "validation.lint_command",
        "validation.lint_timeout_secs",
        "webhook.secret_env",
        "webhook.provider",
        "mcp.bearer_token_env",
        "mcp.allow_unauthenticated",
        "mcp.instructions",
        "mcp.metadata_refresh_secs",
        "mcp.allowed_hosts",
        "mcp.extensions_path",
        "rate_limit.enabled",
        "rate_limit.per_second",
        "rate_limit.burst_size",
        "write.dedup_enabled",
        "write.dedup_threshold",
        "write.commit_author_name",
        "write.commit_author_email",
        "search.hybrid",
        "search.rrf_candidates",
        "search.phrase",
        "search.min_score",
        "search.diversity_max_per_document",
        "search.default_limit",
        "search.max_limit",
        "reranking.enabled",
        "reranking.candidate_limit",
        "ui.semantic_edges.enabled",
        "ui.semantic_edges.k",
        "ui.semantic_edges.min_score",
    ];

    /// Dotted [`Config`] leaf paths with no `ResolvedConfig` field to diff — see
    /// [`RELOAD_DIFF_SETTINGS`]'s doc comment for why these two, specifically,
    /// are excluded rather than covered.
    const RELOAD_DIFF_EXCLUDED: &[&str] = &["embedding.api_key_env", "reranking.api_key_env"];

    #[test]
    fn reload_diff_settings_matches_every_yaml_reloadable_config_field() {
        // Bidirectional drift test for #226, the same technique
        // `yaml_only_settings_matches_every_config_struct_field` (config.rs,
        // #144) already uses: derive the real leaf-field set from `Config`'s own
        // `Default` impl rather than hand-listing it a second time, so a future
        // YAML-settable field added anywhere without a matching `diff()` branch
        // (and a matching `RELOAD_DIFF_SETTINGS` entry) fails this test — and a
        // `RELOAD_DIFF_SETTINGS` entry left behind after a field is renamed or
        // removed fails it too. `config_setting_leaf_paths` is `pub(crate)`
        // specifically so this module can share the one derivation rather than
        // re-implementing the YAML-value flattening a second time.
        let leaves = crate::config::config_setting_leaf_paths();

        let covered: std::collections::HashSet<String> = RELOAD_DIFF_SETTINGS
            .iter()
            .chain(RELOAD_DIFF_EXCLUDED)
            .map(|s| s.to_string())
            .collect();

        let uncovered: Vec<&String> = leaves.difference(&covered).collect();
        assert!(
            uncovered.is_empty(),
            "Config field(s) with no diff() coverage and no RELOAD_DIFF_EXCLUDED \
             entry — POST /admin/reload will silently omit them from every \
             report (see #226): {uncovered:?}"
        );

        let documented: std::collections::HashSet<String> =
            RELOAD_DIFF_SETTINGS.iter().map(|s| s.to_string()).collect();
        let stale: Vec<&String> = documented.difference(&leaves).collect();
        assert!(
            stale.is_empty(),
            "RELOAD_DIFF_SETTINGS entr(y/ies) with no matching Config field — \
             renamed or removed without updating this table: {stale:?}"
        );

        let excluded: std::collections::HashSet<String> =
            RELOAD_DIFF_EXCLUDED.iter().map(|s| s.to_string()).collect();
        let stale_excluded: Vec<&String> = excluded.difference(&leaves).collect();
        assert!(
            stale_excluded.is_empty(),
            "RELOAD_DIFF_EXCLUDED entr(y/ies) with no matching Config field — \
             renamed or removed without updating this table: {stale_excluded:?}"
        );
    }
}
