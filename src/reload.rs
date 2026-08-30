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

/// One YAML-reloadable [`ResolvedConfig`] leaf field: how to read its value out of a
/// snapshot, and every consumer that cares when it changes.
///
/// This is #241's fix. Before this type existed, `diff()` was ~40 hand-written
/// `if old.X != new.X { r.record(...) }` blocks, and a SEPARATE hand-maintained list
/// (`RELOAD_DIFF_SETTINGS`, in the test module below) documented which settings that
/// code was supposed to cover — checked only against `Config`'s own field set, never
/// against `diff()` itself. A contributor could add a field to that list and to
/// `Config` and satisfy every test without ever writing the corresponding branch
/// here, and the reload response would then silently omit that setting when it
/// changed — the exact bug #226 was filed for, reintroduced with a green suite.
///
/// [`DIFF_TABLE`] collapses the list and the comparison into one structure: `get` IS
/// the code that reads the field (there is no separate branch to forget to write),
/// and `consumers` is the classification data the old per-field blocks hard-coded
/// inline. `diff()` below just walks the table. The remaining pair —
/// [`DIFF_TABLE`]'s `path`s against `Config`'s real leaf-field set
/// (`config::config_setting_leaf_paths`) — is still test-enforced (see
/// `reload_diff_settings_matches_every_yaml_reloadable_config_field` below), the same
/// way it always was; what changed is that there is no longer a THIRD structure (the
/// hand-written branches) that list can silently drift from.
struct DiffField {
    /// Dotted `Config`/`ResolvedConfig` leaf path, e.g. `"chunking.max_chunk_size"`.
    /// Matches [`config::config_setting_leaf_paths`]'s naming exactly — this is what
    /// the bidirectional drift test below compares against that ground truth. Read
    /// only by that test (`#[cfg(test)]`), never by `diff()` itself — `consumers`
    /// already carries the caller-facing `setting` name(s) `diff()` reports, so
    /// `path` exists purely as the stable join key between this table and
    /// `config_setting_leaf_paths()`'s ground truth.
    #[allow(dead_code, reason = "read only by the #[cfg(test)] drift test below")]
    path: &'static str,
    /// Read this field's value out of a `ResolvedConfig` snapshot, `Debug`-formatted
    /// (via [`d`]) the same way every hand-written branch used to format it.
    ///
    /// Returns `None` when the field does not apply to this snapshot at all — not
    /// "unchanged", but "there is nothing here to compare". The one field that needs
    /// this is `reranking.candidate_limit`: it lives on `ResolvedRerankingConfig`,
    /// which only exists when `old.reranking`/`new.reranking` is `Some` (reranking
    /// was enabled at startup). `None` on either side means the comparison is
    /// skipped entirely, reproducing the old `if let (Some(old_r), Some(new_r)) = ..`
    /// guard exactly: a reranking on/off transition is reported once, by the
    /// `reranking.enabled` field below, never twice.
    get: fn(&ResolvedConfig) -> Option<String>,
    /// Every consumer of this field, each with its own classification and
    /// caller-facing note. Most fields have exactly one. `source.git_token_env` and
    /// `indexing.include` have two: a genuinely independent second code path with a
    /// DIFFERENT reload lifetime for the SAME underlying setting — see their entries
    /// in [`DIFF_TABLE`] for what each path is. Driving both off one `get` call is
    /// what makes it impossible for the two rows to disagree about what value
    /// actually changed.
    consumers: &'static [ConsumerEntry],
}

/// One reported consumer of a [`DiffField`] — the classification and caller-facing
/// explanation `SettingChange::note` carries. See [`SettingChange::note`]'s doc
/// comment for why the explanation matters and must not be lost by a generic diff.
struct ConsumerEntry {
    effect: ReloadEffect,
    /// Display name for [`SettingChange::setting`]. Equal to the owning
    /// [`DiffField::path`] for a single-consumer field; a two-consumer field
    /// disambiguates with a parenthetical, e.g.
    /// `"indexing.include (MCP get_document path filter)"` — unchanged from what
    /// `diff()`'s hand-written blocks always reported.
    setting: &'static str,
    note: &'static str,
}

/// Every YAML-reloadable [`ResolvedConfig`] leaf `diff()` compares, in the same
/// section order the original hand-written `diff()` used (source, indexing,
/// frontmatter, chunking, embedding, validation, webhook, mcp, rate_limit, write,
/// search, reranking, ui.semantic_edges) purely so a diff against this file's
/// history reads as a mechanical transform rather than a reshuffle.
///
/// `embedding.api_key_env` and `reranking.api_key_env` are deliberately absent:
/// `ResolvedConfig` only retains the env var's looked-up VALUE
/// (`ResolvedEmbeddingConfig::api_key` / `ResolvedRerankingConfig::api_key`), not the
/// var's NAME — the name is consumed once inside `Config::resolve_inner` and
/// discarded. Both are restart-required regardless (`EmbedClient`/`RerankClient`
/// bake the resolved key in at construction — `embed.rs`, `rerank.rs`), this just
/// means there is nothing on `ResolvedConfig` for a `get` fn here to read. See
/// `RELOAD_DIFF_EXCLUDED` in the test module for how these two stay accounted for
/// without a `DiffField` entry.
///
/// `ResolvedRerankingConfig::max_document_bytes` is likewise absent, and is not a
/// YAML setting at all: it is derived from `chunking.max_chunk_size` in
/// `Config::resolve_inner` and baked into `RerankClient` at construction. It rides
/// on the classification `chunking.max_chunk_size` already has (reindex-required),
/// so a change to that number reaches the reranker only on restart — the same
/// lifetime as every other `RerankClient` field.
const DIFF_TABLE: &[DiffField] = &[
    // ── source ───────────────────────────────────────────────────────────────
    // `source.git_token_env` names the env var used to authenticate git operations.
    // It has two independent consumers with different lifetimes: the MCP write
    // tools resolve it fresh on every call, but the webhook/startup-clone path
    // resolves it once and keeps the result for the process's life.
    DiffField {
        path: "source.git_token_env",
        get: |c| Some(d(&c.source.git_token_env)),
        consumers: &[
            ConsumerEntry {
                effect: ReloadEffect::Applied,
                setting: "source.git_token_env (MCP write-tool commits)",
                note: "create_document/edit_document/delete_document resolve this fresh from \
                       the live config on every call (mcp.rs write_document/delete_document) \
                       before reading the named env var — the very next write picks up a new \
                       var name.",
            },
            ConsumerEntry {
                effect: ReloadEffect::RestartRequired,
                setting: "source.git_token_env (webhook pull / startup clone)",
                note: "the token is resolved once at server startup (server.rs run_server) and \
                       stored by value on WebhookState and passed to the initial \
                       git::ensure_repo call; neither re-reads the env var name afterward.",
            },
        ],
    },
    // ── indexing ─────────────────────────────────────────────────────────────
    DiffField {
        path: "indexing.include",
        get: |c| Some(d(&c.indexing.include)),
        consumers: &[
            ConsumerEntry {
                effect: ReloadEffect::Applied,
                setting: "indexing.include (scan / reconcile filtering)",
                note: "ingest::discover_relative reads config.indexing fresh on every indexing \
                       run (ingest.rs); the reindex worker loads a fresh config snapshot before \
                       each drain, and a successful reload queues an immediate full reconcile \
                       so this is picked up right away rather than on the next periodic sweep.",
            },
            ConsumerEntry {
                effect: ReloadEffect::RestartRequired,
                setting: "indexing.include (MCP get_document path filter)",
                note: "KbSearchServer compiles this into a GlobSet once at construction \
                       (mcp.rs build_include_globset, invoked from KbSearchServer::new); \
                       get_document keeps filtering against the old patterns until restart.",
            },
        ],
    },
    DiffField {
        path: "indexing.exclude",
        get: |c| Some(d(&c.indexing.exclude)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "indexing.exclude",
            note: "read fresh per indexing run by ingest::discover_files (ingest.rs); \
                   unlike indexing.include, no MCP path filter bakes this in.",
        }],
    },
    DiffField {
        path: "indexing.exclude_files",
        get: |c| Some(d(&c.indexing.exclude_files)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "indexing.exclude_files",
            note: "read fresh per indexing run by ingest::discover_files (ingest.rs).",
        }],
    },
    DiffField {
        path: "indexing.reconcile_interval_secs",
        get: |c| Some(d(&c.indexing.reconcile_interval_secs)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "indexing.reconcile_interval_secs",
            note: "the periodic reconcile task (server.rs run_server) reads this fresh from \
                   the live config at the top of every iteration to schedule its next \
                   sleep — takes effect starting with the next sleep it schedules.",
        }],
    },
    // ── frontmatter ──────────────────────────────────────────────────────────
    // Feeds the implicit root schema (`ResolvedSchema::from_config`), which is
    // rebuilt by the reindex worker before the next unit that touches a schema file
    // or a full reconcile, and synchronously by update_schema's own rebuild. A
    // successful reload always queues a full reconcile, so these reach the shared
    // schema cache immediately rather than waiting for the periodic sweep.
    DiffField {
        path: "frontmatter.required",
        get: |c| Some(d(&c.frontmatter.required)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "frontmatter.required",
            note: "feeds the implicit root schema, rebuilt by the reindex worker (a \
                   reload queues an immediate full reconcile) and by update_schema's own \
                   synchronous rebuild.",
        }],
    },
    DiffField {
        path: "frontmatter.indexed_fields",
        get: |c| Some(d(&c.frontmatter.indexed_fields)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "frontmatter.indexed_fields",
            note: "index_paths re-ensures Qdrant payload indexes for \
                   effective_indexed_fields on every run (ingest.rs), so a newly indexed \
                   field gets its payload index on the next run; existing documents' \
                   document_fields projection for that field backfills only when they are \
                   next reindexed, or immediately via `md-kb-rag reproject-fields`.",
        }],
    },
    DiffField {
        path: "frontmatter.defaults",
        get: |c| Some(d(&c.frontmatter.defaults)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "frontmatter.defaults",
            note: "applied by validate::apply_defaults during validation, against the \
                   schema rebuilt from the live config on the next indexing run or write.",
        }],
    },
    DiffField {
        path: "frontmatter.allowed",
        get: |c| Some(d(&c.frontmatter.allowed)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "frontmatter.allowed",
            note: "feeds the implicit root schema's value checks, rebuilt the same way as \
                   frontmatter.required above.",
        }],
    },
    // ── chunking ─────────────────────────────────────────────────────────────
    // The indexer reads these fresh (same fresh-config path as indexing.*/
    // frontmatter.* above), but the EFFECT only reaches documents chunked after the
    // change — this is the reindex case the task spec calls out by name.
    DiffField {
        path: "chunking.max_chunk_size",
        get: |c| Some(d(&c.chunking.max_chunk_size)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::ReindexRequired,
            setting: "chunking.max_chunk_size",
            note: "read fresh by the chunker on the next indexing run, but only for \
                   documents that run touches — existing Qdrant chunks keep the old \
                   boundaries. Run `md-kb-rag index --full` for a consistent corpus.",
        }],
    },
    DiffField {
        path: "chunking.target_chunk_size",
        get: |c| Some(d(&c.chunking.target_chunk_size)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::ReindexRequired,
            setting: "chunking.target_chunk_size",
            note: "same as chunking.max_chunk_size: applies to future chunking only. Run \
                   `md-kb-rag index --full` for a consistent corpus.",
        }],
    },
    DiffField {
        path: "chunking.prepend_description",
        get: |c| Some(d(&c.chunking.prepend_description)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::ReindexRequired,
            setting: "chunking.prepend_description",
            note: "changes the text every future chunk embeds (and, incidentally, the \
                   create_document dedup query text — mcp.rs build_dedup_query); existing \
                   chunks were embedded on the old basis. Run `md-kb-rag index --full` for \
                   a consistent corpus.",
        }],
    },
    DiffField {
        path: "chunking.prepend_heading_path",
        get: |c| Some(d(&c.chunking.prepend_heading_path)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::ReindexRequired,
            setting: "chunking.prepend_heading_path",
            note: "changes the text every future chunk embeds — a chunk's heading-ancestry \
                   breadcrumb is prepended before embedding, so existing chunks were embedded \
                   on the other basis. Unlike prepend_description this does NOT affect the \
                   create_document dedup query (a document's first chunk never carries a \
                   breadcrumb — see write.rs build_dedup_query). Run `md-kb-rag index --full` \
                   for a consistent corpus.",
        }],
    },
    // ── embedding ────────────────────────────────────────────────────────────
    // EmbedClient is built exactly once, at server startup, from a `&ResolvedEmbeddingConfig`
    // snapshot — every field below is captured by value or baked into a
    // `reqwest::Client` at that point (embed.rs EmbedClient::new).
    DiffField {
        path: "embedding.batch_size",
        get: |c| Some(d(&c.embedding.batch_size)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::RestartRequired,
            setting: "embedding.batch_size",
            note: "EmbedClient captures this by value at construction (embed.rs \
                   EmbedClient::new); it is built once at server startup and never \
                   rebuilt.",
        }],
    },
    DiffField {
        path: "embedding.request_timeout_secs",
        get: |c| Some(d(&c.embedding.request_timeout_secs)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::RestartRequired,
            setting: "embedding.request_timeout_secs",
            note: "baked into the reqwest::Client's timeout when EmbedClient is \
                   constructed (embed.rs); a reqwest::Client's timeout cannot be changed \
                   after the client is built.",
        }],
    },
    DiffField {
        path: "embedding.batch_concurrency",
        get: |c| Some(d(&c.embedding.batch_concurrency)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::RestartRequired,
            setting: "embedding.batch_concurrency",
            note: "EmbedClient captures this by value at construction (embed.rs \
                   EmbedClient::new).",
        }],
    },
    // ── validation ───────────────────────────────────────────────────────────
    DiffField {
        path: "validation.enabled",
        get: |c| Some(d(&c.validation.enabled)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "validation.enabled",
            note: "read fresh per indexing run (ingest.rs process_file); a reload queues \
                   an immediate full reconcile.",
        }],
    },
    DiffField {
        path: "validation.strict",
        get: |c| Some(d(&c.validation.strict)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "validation.strict",
            note: "read fresh per indexing run (ingest.rs process_file).",
        }],
    },
    DiffField {
        path: "validation.lint_command",
        get: |c| Some(d(&c.validation.lint_command)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "validation.lint_command",
            note: "read fresh per file validated (validate.rs), invoked from the next \
                   indexing run or MCP write.",
        }],
    },
    DiffField {
        path: "validation.lint_timeout_secs",
        get: |c| Some(d(&c.validation.lint_timeout_secs)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "validation.lint_timeout_secs",
            note: "read fresh per file validated, the same call site as \
                   validation.lint_command above (validate.rs) — bounds how long a \
                   configured lint command is allowed to run before that file's \
                   validation fails with a timeout.",
        }],
    },
    // ── webhook ──────────────────────────────────────────────────────────────
    DiffField {
        path: "webhook.secret_env",
        get: |c| Some(d(&c.webhook.secret_env)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::RestartRequired,
            setting: "webhook.secret_env",
            note: "the secret is resolved once at server startup (server.rs run_server) \
                   and stored by value on WebhookState.secret; whether /hooks/reindex even \
                   exists is also decided at startup from this same lookup.",
        }],
    },
    DiffField {
        path: "webhook.provider",
        get: |c| Some(d(&c.webhook.provider)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "webhook.provider",
            note: "read fresh from the live config on every webhook request (webhook.rs \
                   handle_webhook) to pick the signature header and verification scheme.",
        }],
    },
    // ── mcp ──────────────────────────────────────────────────────────────────
    DiffField {
        path: "mcp.bearer_token_env",
        get: |c| Some(d(&c.mcp.bearer_token_env)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::RestartRequired,
            setting: "mcp.bearer_token_env",
            note: "the bearer token is resolved once at server startup and stored by \
                   value on AuthState (server.rs run_server) — security-critical, \
                   deliberately not made live.",
        }],
    },
    DiffField {
        path: "mcp.allow_unauthenticated",
        get: |c| Some(d(&c.mcp.allow_unauthenticated)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::RestartRequired,
            setting: "mcp.allow_unauthenticated",
            note: "gates how AuthState is built at startup (server.rs run_server) — \
                   security-critical, deliberately not made live.",
        }],
    },
    DiffField {
        path: "mcp.instructions",
        get: |c| Some(d(&c.mcp.instructions)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "mcp.instructions",
            note: "the metadata-refresh timer (server.rs run_server) reads this fresh from \
                   the live config every tick; takes effect on the next tick, within \
                   mcp.metadata_refresh_secs seconds.",
        }],
    },
    DiffField {
        path: "mcp.metadata_refresh_secs",
        get: |c| Some(d(&c.mcp.metadata_refresh_secs)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "mcp.metadata_refresh_secs",
            note: "the same refresh timer reads this fresh at the top of every iteration \
                   to schedule its next sleep.",
        }],
    },
    DiffField {
        path: "mcp.allowed_hosts",
        get: |c| Some(d(&c.mcp.allowed_hosts)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::RestartRequired,
            setting: "mcp.allowed_hosts",
            note: "baked into the StreamableHttpServerConfig when the MCP service is \
                   built (server.rs mcp_transport_config, invoked once from run_server).",
        }],
    },
    DiffField {
        path: "mcp.extensions_path",
        get: |c| Some(d(&c.mcp.extensions_path)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "mcp.extensions_path",
            note: "the metadata-refresh timer (server.rs run_server) re-resolves this fresh \
                   from the live config every tick and recomposes both the server \
                   instructions and every tool's description overlay from it; takes effect \
                   on the next tick, within mcp.metadata_refresh_secs seconds.",
        }],
    },
    // ── rate_limit ───────────────────────────────────────────────────────────
    // The whole section is restart-required: whether GovernorLayer is even added to
    // the router, and every knob on it, is decided once when the router is built.
    DiffField {
        path: "rate_limit.enabled",
        get: |c| Some(d(&c.rate_limit.enabled)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::RestartRequired,
            setting: "rate_limit.enabled",
            note: "decides whether GovernorLayer is added to the router at all (server.rs \
                   run_server); the router is built once.",
        }],
    },
    DiffField {
        path: "rate_limit.per_second",
        get: |c| Some(d(&c.rate_limit.per_second)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::RestartRequired,
            setting: "rate_limit.per_second",
            note: "baked into the GovernorConfigBuilder when the rate limiter is built \
                   once at startup (server.rs run_server).",
        }],
    },
    DiffField {
        path: "rate_limit.burst_size",
        get: |c| Some(d(&c.rate_limit.burst_size)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::RestartRequired,
            setting: "rate_limit.burst_size",
            note: "baked into the GovernorConfigBuilder when the rate limiter is built \
                   once at startup (server.rs run_server).",
        }],
    },
    // ── write ────────────────────────────────────────────────────────────────
    DiffField {
        path: "write.dedup_enabled",
        get: |c| Some(d(&c.write.dedup_enabled)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "write.dedup_enabled",
            note: "read fresh from the live config on every create_document call (mcp.rs \
                   write_document).",
        }],
    },
    DiffField {
        path: "write.dedup_threshold",
        get: |c| Some(d(&c.write.dedup_threshold)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "write.dedup_threshold",
            note: "read fresh from the live config on every create_document call (mcp.rs \
                   write_document).",
        }],
    },
    DiffField {
        path: "write.commit_author_name",
        get: |c| Some(d(&c.write.commit_author_name)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "write.commit_author_name",
            note: "read fresh from the live config on every write-tool commit (mcp.rs).",
        }],
    },
    DiffField {
        path: "write.commit_author_email",
        get: |c| Some(d(&c.write.commit_author_email)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "write.commit_author_email",
            note: "read fresh from the live config on every write-tool commit (mcp.rs).",
        }],
    },
    // ── search ───────────────────────────────────────────────────────────────
    DiffField {
        path: "search.phrase",
        get: |c| Some(d(&c.search.phrase)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "search.phrase",
            note: "read fresh from the live config on every search call (mcp.rs/web.rs, \
                   gated by phrase_matching_available) and by the metadata-refresh \
                   timer (server.rs compose_server_instructions/compose_tool_overlay) \
                   that recomposes the server/tool description overlay so it never \
                   advertises quoted-phrase support the config just turned off.",
        }],
    },
    DiffField {
        path: "search.hybrid",
        get: |c| Some(d(&c.search.hybrid)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "search.hybrid",
            note: "read fresh from the live config on every search call (mcp.rs search).",
        }],
    },
    DiffField {
        path: "search.rrf_candidates",
        get: |c| Some(d(&c.search.rrf_candidates)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "search.rrf_candidates",
            note: "read fresh from the live config on every search call (mcp.rs search).",
        }],
    },
    DiffField {
        path: "search.min_score",
        get: |c| Some(d(&c.search.min_score)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "search.min_score",
            note: "read fresh from the live config on every search call (mcp.rs search) \
                   as the default when the caller does not pass their own min_score.",
        }],
    },
    DiffField {
        path: "search.diversity_max_per_document",
        get: |c| Some(d(&c.search.diversity_max_per_document)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "search.diversity_max_per_document",
            note: "read fresh from the live config on every search call (mcp.rs search, \
                   retrieval::search's per-document diversity cap).",
        }],
    },
    DiffField {
        path: "search.default_limit",
        get: |c| Some(d(&c.search.default_limit)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "search.default_limit",
            note: "read fresh from the live config on every search call (mcp.rs \
                   resolve_limit) when the caller omits limit.",
        }],
    },
    DiffField {
        path: "search.max_limit",
        get: |c| Some(d(&c.search.max_limit)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "search.max_limit",
            note: "read fresh from the live config on every search call (mcp.rs \
                   resolve_limit) as the ceiling a requested limit is clamped to.",
        }],
    },
    // ── reranking ────────────────────────────────────────────────────────────
    DiffField {
        path: "reranking.enabled",
        get: |c| Some(d(&c.reranking.is_some())),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::RestartRequired,
            setting: "reranking.enabled",
            note: "gates whether KbSearchServer even holds a RerankClient at all \
                   (server.rs run_server); the Option is decided once at construction — a \
                   reload cannot add or remove the client.",
        }],
    },
    DiffField {
        path: "reranking.candidate_limit",
        // `None` on either side (reranking disabled in that snapshot) skips the
        // comparison entirely — see this field's doc comment on `DiffField::get`.
        get: |c| c.reranking.as_ref().map(|r| d(&r.candidate_limit)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::Applied,
            setting: "reranking.candidate_limit",
            note: "read fresh from the live config on every search call (mcp.rs search) — \
                   but only takes effect when reranking was already enabled at startup; a \
                   server that started with it disabled has no RerankClient to hand this \
                   to regardless.",
        }],
    },
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
    // (`queue.mark_full()`, unconditional — see `reload_config`'s doc comment)
    // still goes through the worker's ordinary `scan_for_dirty`, which skips any
    // file whose content hash is unchanged; only `md-kb-rag index --full`
    // (`force = true`) bypasses that skip and reprocesses everything. So exactly
    // like `chunking.*` above: flipping `ui.semantic_edges.enabled` on does not
    // retroactively populate semantic edges for the existing corpus, and
    // flipping it off does not retroactively remove already-computed ones —
    // both only apply to documents a future run actually touches, same caveat,
    // same fix (`md-kb-rag index --full`).
    DiffField {
        path: "ui.semantic_edges.enabled",
        get: |c| Some(d(&c.ui.semantic_edges.enabled)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::ReindexRequired,
            setting: "ui.semantic_edges.enabled",
            note: "read fresh by the indexer on the next run (ingest.rs \
                   update_semantic_edges), but only for documents that run re-embeds — \
                   existing (or missing) semantic edges in the graph view are \
                   unchanged otherwise. Run `md-kb-rag index --full` for a consistent \
                   corpus.",
        }],
    },
    DiffField {
        path: "ui.semantic_edges.k",
        get: |c| Some(d(&c.ui.semantic_edges.k)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::ReindexRequired,
            setting: "ui.semantic_edges.k",
            note: "same as ui.semantic_edges.enabled: applies to future re-embeds only. \
                   Run `md-kb-rag index --full` for a consistent corpus.",
        }],
    },
    DiffField {
        path: "ui.semantic_edges.min_score",
        get: |c| Some(d(&c.ui.semantic_edges.min_score)),
        consumers: &[ConsumerEntry {
            effect: ReloadEffect::ReindexRequired,
            setting: "ui.semantic_edges.min_score",
            note: "same as ui.semantic_edges.enabled: applies to future re-embeds only. \
                   Run `md-kb-rag index --full` for a consistent corpus.",
        }],
    },
];

/// Compare every YAML-reloadable setting between `old` and `new`, recording each one
/// that changed with its classification. This is the "hard part" this feature exists
/// for: a setting only belongs in [`ReloadEffect::Applied`] if some code path
/// genuinely re-reads the live [`SharedConfig`] (or an equivalent per-run/per-request
/// snapshot) rather than a value captured once at startup — see [`DIFF_TABLE`]'s doc
/// comment for how that table keeps this function from silently missing one.
pub fn diff(old: &ResolvedConfig, new: &ResolvedConfig) -> ReloadReport {
    let mut r = ReloadReport::default();

    for field in DIFF_TABLE {
        // `None` means "not applicable to this snapshot" (see `DiffField::get`),
        // not "no change" — skip rather than compare a placeholder.
        let (Some(old_v), Some(new_v)) = ((field.get)(old), (field.get)(new)) else {
            continue;
        };
        if old_v == new_v {
            continue;
        }
        for consumer in field.consumers {
            r.record(
                consumer.effect,
                consumer.setting,
                old_v.clone(),
                new_v.clone(),
                consumer.note,
            );
        }
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

    /// Dotted [`Config`] leaf paths with no `ResolvedConfig` field to diff — see
    /// [`DIFF_TABLE`]'s doc comment for why these two, specifically, are excluded
    /// rather than covered by a `DiffField`.
    const RELOAD_DIFF_EXCLUDED: &[&str] = &["embedding.api_key_env", "reranking.api_key_env"];

    #[test]
    fn reload_diff_settings_matches_every_yaml_reloadable_config_field() {
        // Bidirectional drift test for #226 (and, since #241, the ONLY hand-listed
        // structure left to drift): derive the real leaf-field set from `Config`'s
        // own `Default` impl, the same technique
        // `yaml_only_settings_matches_every_config_struct_field` (config.rs, #144)
        // already uses, and compare it against `DIFF_TABLE`'s `path`s directly.
        //
        // Before #241, this compared against a SEPARATE hand-maintained list
        // (`RELOAD_DIFF_SETTINGS`) that merely had to mention a setting's name —
        // nothing here checked that `diff()` itself actually held a matching
        // `if old.X != new.X` branch, so the list and the code could silently
        // diverge (the exact bug #226 was filed for, reintroduced with a green
        // suite). Reading `DIFF_TABLE.path` instead of a parallel list closes that
        // gap: `DIFF_TABLE` entries ARE what `diff()` runs, so this test now
        // verifies the real comparison logic exists for every leaf field, not just
        // that someone remembered to write its name down twice.
        //
        // `config_setting_leaf_paths` is `pub(crate)` specifically so this module
        // can share the one derivation rather than re-implementing the YAML-value
        // flattening a second time.
        let leaves = crate::config::config_setting_leaf_paths();

        let covered: std::collections::HashSet<String> = DIFF_TABLE
            .iter()
            .map(|f| f.path.to_string())
            .chain(RELOAD_DIFF_EXCLUDED.iter().map(|s| s.to_string()))
            .collect();

        let uncovered: Vec<&String> = leaves.difference(&covered).collect();
        assert!(
            uncovered.is_empty(),
            "Config field(s) with no diff() coverage and no RELOAD_DIFF_EXCLUDED \
             entry — POST /admin/reload will silently omit them from every \
             report (see #226): {uncovered:?}"
        );

        let documented: std::collections::HashSet<String> =
            DIFF_TABLE.iter().map(|f| f.path.to_string()).collect();
        let stale: Vec<&String> = documented.difference(&leaves).collect();
        assert!(
            stale.is_empty(),
            "DIFF_TABLE entr(y/ies) with no matching Config field — renamed or \
             removed without updating this table: {stale:?}"
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
