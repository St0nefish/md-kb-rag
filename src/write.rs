//! Transport-agnostic core of the write tools: validation, the create-path dedup
//! gate, filesystem write/removal, git commit+sync, and the pre-commit /
//! post-commit rollback state machine.
//!
//! Extracted out of `mcp.rs` so both the MCP tool surface (`create_document`,
//! `edit_document`, `delete_document`) and the HTTP UI (`web.rs`, a later chunk)
//! drive the exact same commit/rollback logic instead of maintaining two copies of
//! it. `mcp.rs`'s tool methods are thin adapters: they do transport-specific path
//! resolution (fuzzy basename matching, the `expected_hash` stale-read guard ahead
//! of a surgical edit) and map [`WriteSuccess`]/[`WriteError`] back onto the exact
//! `CallToolResult`/`McpError` shapes their existing tests pin down.
//!
//! A note on `WriteError::PostCommitPending`, which does not exist here: the plan
//! this module was built from listed it as a `WriteError` variant, but a
//! "committed locally, push still pending" write is not a failure — every existing
//! caller (and the fixed HTTP contract this module was built against) reports it
//! as a 200/`Ok` result, just like a fully synced write, so it is modeled as
//! `WriteSuccess { outcome: WriteOutcome::CommittedPendingSync, .. }` instead. The
//! extra `sync_failure_cause` field carries the push-failure detail a
//! human-readable summary needs, which the fixed `WriteSuccess{outcome, sha,
//! rebased_paths, diff}` shape had no room for otherwise.

use std::path::{Path, PathBuf};

use tracing::{error, warn};

use crate::config::ValidationConfig;
use crate::embed::QueryEmbedder;
use crate::git;
use crate::qdrant::RetrievalStore;
use crate::retrieval::{RetrievalDeps, SearchFilters};
use crate::schema::SharedSchemaCache;
use crate::validate::{self, ValidationResult};

// ---------------------------------------------------------------------------
// Dedup gate (create-path near-duplicate refusal)
// ---------------------------------------------------------------------------

/// Maximum number of characters from the new document's content used to build
/// the dedup query text. Keeps the embedding request within typical token limits
/// (most embedding models cap at ~512–8192 tokens; 2000 chars ≈ 400–500 tokens).
const DEDUP_QUERY_CHAR_LIMIT: usize = 2000;

/// A near-duplicate found during dedup gate evaluation.
#[derive(Debug, Clone)]
pub struct DuplicateHit {
    pub file_path: String,
    pub score: f32,
}

/// Pure decision function: given the closest match from Qdrant (if any) and a
/// threshold, decide whether to refuse the write. Returns `Some(DuplicateHit)`
/// when the write should be blocked, `None` when it is safe to proceed.
///
/// This is factored out so it can be unit-tested without a live Qdrant/embedder.
pub fn dedup_verdict(top: Option<(String, f32)>, threshold: f32) -> Option<DuplicateHit> {
    match top {
        Some((path, score)) if score >= threshold => Some(DuplicateHit {
            file_path: path,
            score,
        }),
        _ => None,
    }
}

/// Build the dedup query text on the same textual basis the indexer uses, then
/// truncate it to `DEDUP_QUERY_CHAR_LIMIT`.
///
/// This must stay aligned with `chunk::chunk_markdown`: every indexed chunk is
/// prefixed with its document's own `description` when
/// `chunking.prepend_description` is set, so a dedup query built without that
/// prefix would be compared against a different textual basis than the
/// candidates it is scored against.
pub(crate) fn build_dedup_query(
    body: &str,
    description: Option<&str>,
    prepend_description: bool,
) -> String {
    let assembled = match (prepend_description, description) {
        (true, Some(desc)) => format!("{}\n\n{}", desc, body),
        _ => body.to_string(),
    };
    assembled.chars().take(DEDUP_QUERY_CHAR_LIMIT).collect()
}

/// Search options for the dedup gate.
///
/// Deliberately pinned to dense-only rather than inheriting `search.hybrid`, so
/// the returned score is a cosine similarity comparable to
/// `write.dedup_threshold`. Hybrid RRF scores top out around 0.03 — against a
/// cosine threshold like the 0.80 default the gate could never fire — and a
/// cross-encoder relevance score is not a similarity at all, so reranking is
/// also kept out of this path (see the `reranker: None` at the call site).
pub(crate) fn dedup_search_opts() -> crate::retrieval::SearchOptions {
    crate::retrieval::SearchOptions {
        limit: 1,
        min_score: None,
        hybrid: false,
        // Unused in the dense-only path, which performs no RRF fusion.
        rrf_candidates: 0,
        explain: false,
        modified_after: None,
        modified_before: None,
        rerank_candidate_limit: None,
        // The dedup gate wants the single closest existing chunk, full stop — not
        // a diversified page of results (limit: 1 above makes a per-document cap
        // moot anyway, but this keeps intent explicit rather than accidental).
        diversity_max_per_document: None,
    }
}

// ---------------------------------------------------------------------------
// Commit message helpers
// ---------------------------------------------------------------------------

/// Above this length (or containing a newline) a caller-supplied commit message
/// would confuse `git log`/`git commit -m`, so it is rejected up front.
const MAX_COMMIT_MESSAGE_LEN: usize = 1000;

/// Reject a caller-supplied commit message that git or the log would mangle.
fn validate_commit_message(message: Option<&str>) -> Result<(), WriteError> {
    let Some(msg) = message else {
        return Ok(());
    };
    if msg.contains('\n') {
        return Err(WriteError::InvalidCommitMessage {
            reason: "commit message must not contain newlines".to_string(),
        });
    }
    if msg.len() > MAX_COMMIT_MESSAGE_LEN {
        return Err(WriteError::InvalidCommitMessage {
            reason: format!(
                "commit message too long ({} chars); maximum is {}",
                msg.len(),
                MAX_COMMIT_MESSAGE_LEN
            ),
        });
    }
    Ok(())
}

/// Build a commit message with git trailers identifying the tool and operation.
///
/// The resulting message has the form:
/// ```text
/// <subject line>
///
/// Tool: md-kb-rag
/// Operation: <operation>
/// ```
///
/// `user_subject` is the caller-supplied commit message (if any). When absent,
/// `default_subject` is used. The trailer block is always appended after a blank line.
pub fn build_commit_message(
    user_subject: Option<&str>,
    default_subject: &str,
    operation: &str,
) -> String {
    let subject = user_subject.unwrap_or(default_subject);
    format!("{}\n\nTool: md-kb-rag\nOperation: {}", subject, operation)
}

/// Render a unified diff between `old` and `new` content, labelled with
/// `a/<relpath>` and `b/<relpath>`. Returns an empty string if there is no
/// diff (shouldn't happen for a real change).
pub fn render_unified_diff(old: &str, new: &str, relpath: &str) -> String {
    use similar::TextDiff;
    let diff = TextDiff::from_lines(old, new);
    diff.unified_diff()
        .context_radius(3)
        .header(&format!("a/{relpath}"), &format!("b/{relpath}"))
        .to_string()
}

// ---------------------------------------------------------------------------
// Dependencies, request/response/error shapes
// ---------------------------------------------------------------------------

/// Dependencies needed to run the write pipeline, independent of transport.
///
/// Mirrors what `KbSearchServer` currently pulls from `self`/its config snapshot
/// for `write_document`/`delete_document`. `retrieval` is reused wholesale for the
/// create-path dedup gate — see `crate::retrieval::RetrievalDeps`.
pub struct WriteDeps<'a, E: QueryEmbedder, Q: RetrievalStore> {
    pub retrieval: RetrievalDeps<'a, E, Q>,
    /// Canonicalized knowledge-base root. Every filesystem action re-resolves
    /// against this immediately before it happens (see `resolve_safe_write_path`),
    /// closing the TOCTOU window a slow schema/dedup lookup would otherwise leave
    /// open.
    pub canonical_data_path: &'a Path,
    pub schema_cache: &'a SharedSchemaCache,
    pub validation: &'a ValidationConfig,
    /// Mirrors `chunking.prepend_description` — the dedup query must be built on
    /// the same textual basis the indexer embeds.
    pub prepend_description: bool,
    pub dedup_enabled: bool,
    pub dedup_threshold: f32,
    pub git_url: Option<&'a str>,
    pub branch: &'a str,
    pub token: Option<&'a str>,
    pub commit_author_name: &'a str,
    pub commit_author_email: &'a str,
}

/// A create or edit request against the write pipeline.
pub struct WriteRequest<'a> {
    /// Repo-relative path, already resolved and validated by the caller.
    pub rel_path: &'a str,
    /// Existing file bytes (empty string for a create).
    pub old_content: &'a str,
    /// The content to write, already computed by the caller (e.g. after applying
    /// a surgical old_string/new_string replacement).
    pub new_content: &'a str,
    pub is_create: bool,
    pub message: Option<&'a str>,
    /// Verb for the default commit message, e.g. `"add"` or `"update"`.
    pub default_verb: &'a str,
    /// When `Some(true)`, bypasses the dedup gate on create paths.
    pub force_new: Option<bool>,
    /// Label for the `Operation:` git trailer, e.g. `"create_document"`.
    pub operation: &'a str,
    /// Optional stale-read guard: reject the write if this does not match the
    /// SHA-256 hex digest of `old_content`. `None` skips the check. Callers that
    /// already perform this check themselves against the same `old_content` (as
    /// `mcp.rs`'s `edit_document` does, ahead of applying a surgical replacement)
    /// may safely pass `None` here — re-checking the same in-memory content a
    /// second time can never disagree with the first check.
    pub expected_hash: Option<&'a str>,
}

/// The two outcomes a write can land on. Both are the tool's happy path from a
/// caller's point of view — `synced` fully so, `committed_pending_sync` with a
/// caveat — which is why both live under `WriteSuccess` rather than being split
/// across the `Ok`/`Err` boundary. See this module's doc comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteOutcome {
    /// Committed and pushed.
    Synced,
    /// Committed locally, but the remote push failed. NOT rolled back — the commit
    /// is real. Will sync on the next successful write, or on manual intervention.
    CommittedPendingSync,
}

/// A successful write or delete.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WriteSuccess {
    pub outcome: WriteOutcome,
    pub sha: String,
    pub rebased_paths: Vec<PathBuf>,
    /// Unified diff of the change (empty for a no-op, which should not happen for
    /// a real write).
    pub diff: String,
    /// Present only when `outcome == CommittedPendingSync`: the redacted,
    /// already-`{:#}`-formatted cause of the sync failure (fetch/rebase/push),
    /// for a human-readable summary. `None` on a fully synced write.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sync_failure_cause: Option<String>,
}

/// Every structured failure mode of the write pipeline. Callers map these onto
/// their own transport's error shape (`McpError` for MCP, an HTTP status + JSON
/// body for the web UI) — see this module's doc comment for why
/// `PostCommitPending` is not among them.
#[derive(Debug)]
pub enum WriteError {
    /// The schema governing `rel_path`'s directory failed to parse. `reason` is
    /// the parse-failure message from `SchemaCache::is_frozen`.
    Frozen { reason: String },
    /// Frontmatter validation failed. Carries the full structured result so a
    /// caller can report per-field errors, not just a flat message.
    Validation { result: ValidationResult },
    /// A near-duplicate document already exists (create-path dedup gate).
    DedupHit {
        duplicate_of: String,
        similarity: f32,
        threshold: f32,
    },
    /// The caller-supplied commit message would confuse git or the log.
    InvalidCommitMessage { reason: String },
    /// `rel_path` failed the path-safety check (traversal, absolute path escaping
    /// the root, or a symlinked ancestor pointing outside it). `msg` is always
    /// built from caller-supplied input only — safe to relay to an untrusted
    /// caller verbatim (see `WriteError::Internal` for the counterpart that is
    /// not).
    UnsafePath { msg: String },
    /// The path-safety check itself failed for a reason that has nothing to do
    /// with what the caller supplied — `resolve_safe_write_path`'s
    /// canonicalize-failure branches, which embed a server-side absolute
    /// filesystem path in their message (e.g. "cannot canonicalize data root
    /// '<abs>': <io error>"). Kept distinct from `UnsafePath` so a transport
    /// that must not leak container paths to an untrusted caller (`web.rs`) can
    /// map this to a generic message while `mcp.rs` — a trusted surface where
    /// this text is diagnostically useful and was already being returned
    /// before this variant existed — keeps emitting it verbatim.
    Internal { msg: String },
    /// Create was requested but `rel_path` already exists.
    AlreadyExists,
    /// Edit or delete was requested but `rel_path` does not exist.
    NotFound,
    /// `expected_hash` did not match the current content hash of `old_content`.
    StaleHash { expected: String, actual: String },
    /// `git add`/`git commit` failed (HEAD never moved). `rolled_back = true`
    /// means the working tree and git index were successfully restored to their
    /// pre-call state — safe to retry. `rolled_back = false` means the rollback
    /// attempt ITSELF also failed, leaving filesystem and git state inconsistent
    /// with each other and with HEAD — this needs operator attention, not a blind
    /// retry. `msg` is the formatted cause (or causes, for the `false` case).
    PreCommitFailed { rolled_back: bool, msg: String },
    /// Any other I/O or internal error (parent-directory creation, the write
    /// itself, reading a file, or validation blowing up rather than just failing).
    Io { msg: String },
}

// ---------------------------------------------------------------------------
// Shared path-safety helper
// ---------------------------------------------------------------------------

/// The two failure shapes [`resolve_safe_write_path_detailed`] can produce,
/// split by whether the message is safe to relay to an untrusted caller.
///
/// `resolve_safe_write_path` (the public, `String`-returning function every
/// existing caller outside this module already uses) flattens both variants
/// back into a plain string, so its behavior and exact message text are
/// unchanged by this split — it exists purely so `safe_write_path` (below),
/// the wrapper `write_document`/`delete_document` use internally, can pick the
/// right `WriteError` variant without re-deriving the classification by
/// sniffing message text (fragile — see `WriteError::Internal`'s doc comment
/// for why that classification exists at all).
enum PathSafetyError {
    /// Built entirely from the caller-supplied relative path and fixed text —
    /// safe to show the caller as-is.
    Rejected(String),
    /// Embeds a server-side absolute filesystem path (a canonicalize
    /// failure) — must not reach an untrusted caller verbatim.
    Internal(String),
}

impl PathSafetyError {
    fn into_message(self) -> String {
        match self {
            PathSafetyError::Rejected(msg) | PathSafetyError::Internal(msg) => msg,
        }
    }
}

/// Validate that `rel_path` is safe to write inside `data_root`, returning the
/// absolute target path on success or a classified error on failure — see
/// [`PathSafetyError`].
///
/// Checks performed (in order):
/// 1. Reject absolute paths.
/// 2. Reject any `..` component.
/// 3. Lexical `starts_with` check on the joined abs path.
/// 4. Canonicalize the deepest *existing* ancestor of the target; verify it
///    still `starts_with` the canonical data_root. This catches a symlinked
///    ancestor directory that resolves to a location outside data_root.
fn resolve_safe_write_path_detailed(
    data_root: &Path,
    rel_path: &str,
) -> Result<PathBuf, PathSafetyError> {
    // A leading `/` means "the knowledge-base root", not a filesystem path — callers
    // have no way to know where the KB lives inside the container, so treating `/x.md`
    // and `x.md` as the same location is the only reading that makes sense here.
    let rel_path = crate::retrieval::kb_root_relative(rel_path);
    let requested = Path::new(rel_path);
    if requested.is_absolute() {
        return Err(PathSafetyError::Rejected(
            "path must be relative to the knowledge base root".to_string(),
        ));
    }

    // 2. Reject any `..` component.
    for component in requested.components() {
        if component == std::path::Component::ParentDir {
            return Err(PathSafetyError::Rejected(
                "path must not contain '..' components".to_string(),
            ));
        }
    }

    let abs_path = data_root.join(rel_path);

    // 3. Lexical starts_with check.
    if !abs_path.starts_with(data_root) {
        return Err(PathSafetyError::Rejected(
            "path escapes the knowledge base root".to_string(),
        ));
    }

    // 4. Canonical-ancestor check: canonicalize data_root, then walk up from
    //    abs_path to find the deepest ancestor that actually exists on disk,
    //    canonicalize it, and confirm it still sits under canonical data_root.
    let canonical_root = data_root.canonicalize().map_err(|e| {
        PathSafetyError::Internal(format!(
            "cannot canonicalize data root '{}': {}",
            data_root.display(),
            e
        ))
    })?;

    // Walk from abs_path upward until we find an existing ancestor.
    let mut candidate = abs_path.as_path();
    let existing_ancestor = loop {
        if candidate.exists() {
            break candidate;
        }
        match candidate.parent() {
            Some(p) => candidate = p,
            None => {
                // No ancestor exists at all (shouldn't happen since data_root must exist).
                break data_root;
            }
        }
    };

    let canonical_ancestor = existing_ancestor.canonicalize().map_err(|e| {
        PathSafetyError::Internal(format!(
            "cannot canonicalize ancestor '{}': {}",
            existing_ancestor.display(),
            e
        ))
    })?;

    if !canonical_ancestor.starts_with(&canonical_root) {
        return Err(PathSafetyError::Rejected(
            "path escapes the knowledge base root (symlink detected)".to_string(),
        ));
    }

    Ok(abs_path)
}

/// Validate that `rel_path` is safe to write inside `data_root`, returning the
/// absolute target path on success or an error string on failure.
///
/// Moved here (from `mcp.rs`) so this core write module does not depend back on
/// the transport layer it is meant to be independent of — `mcp.rs`'s tool
/// methods call this via `crate::write::resolve_safe_write_path`.
///
/// A thin `String`-returning wrapper over [`resolve_safe_write_path_detailed`]:
/// every caller outside this module predates the `PathSafetyError` split and
/// expects a flat string (several of them — `mcp.rs`'s `create_document` and
/// `write_raw_file` — relay it directly to a trusted MCP client), so this
/// function's behavior and exact message text are unchanged by that split.
pub fn resolve_safe_write_path(data_root: &Path, rel_path: &str) -> Result<PathBuf, String> {
    resolve_safe_write_path_detailed(data_root, rel_path).map_err(PathSafetyError::into_message)
}

/// Resolve `rel_path` against the KB root via
/// [`resolve_safe_write_path_detailed`], mapping a caller-facing rejection onto
/// `WriteError::UnsafePath` and a canonicalize failure onto
/// `WriteError::Internal` (see that variant's doc comment for why the two must
/// not be conflated).
fn safe_write_path<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &WriteDeps<'_, E, Q>,
    rel_path: &str,
) -> Result<PathBuf, WriteError> {
    resolve_safe_write_path_detailed(deps.canonical_data_path, rel_path).map_err(|e| match e {
        PathSafetyError::Rejected(msg) => WriteError::UnsafePath {
            msg: format!("Invalid path: {}", msg),
        },
        PathSafetyError::Internal(msg) => WriteError::Internal {
            msg: format!("Invalid path: {}", msg),
        },
    })
}

/// Reject `rel_path` if it does not match `include_patterns` — a path the
/// indexer would never pick up (e.g. anything outside `**/*.md`, including
/// files under `.git/`) must never be written or deleted through this
/// pipeline, no matter which caller reaches it.
///
/// Before this existed, every transport enforced this itself: MCP's
/// `create_document` checked `include_patterns.is_match` directly, and
/// `edit_document`/`delete_document` got it for free from
/// `retrieval::resolve_within_data`'s `NotPermitted` arm. That meant a new
/// caller (the HTTP UI's write routes, briefly) had to remember to re-derive
/// the same check itself, and a caller that forgot it entirely would let a
/// write reach `commit_and_sync` for a path the indexer would never clean up
/// or even see — including files under `.git/`, where a hostile write is a
/// well-known local code-execution primitive. Enforcing it once, here, means
/// no caller can drop it by omission.
///
/// Reuses `WriteError::UnsafePath` rather than adding a new variant so every
/// existing caller's (necessarily exhaustive) match over `WriteError` keeps
/// compiling unchanged — see `mcp.rs`'s and `web.rs`'s error-mapping functions.
///
/// `pub(crate)` (rather than folded entirely into `check_include_pattern`
/// below) so `mcp.rs`'s `create_document` adapter can run this exact check —
/// same message text — ahead of its own `exists()` pre-check, restoring the
/// pre-refactor error priority when a path both exists on disk and fails this
/// check. See that call site's comment.
pub(crate) fn check_include_pattern_against(
    include_patterns: &globset::GlobSet,
    rel_path: &str,
) -> Result<(), WriteError> {
    let normalized = crate::retrieval::kb_root_relative(rel_path);
    if include_patterns.is_match(normalized) {
        return Ok(());
    }
    Err(WriteError::UnsafePath {
        msg: format!(
            "path '{}' does not match any indexable include pattern \
             (e.g. must be a markdown file under an included path)",
            rel_path
        ),
    })
}

fn check_include_pattern<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &WriteDeps<'_, E, Q>,
    rel_path: &str,
) -> Result<(), WriteError> {
    check_include_pattern_against(deps.retrieval.include_patterns, rel_path)
}

// ---------------------------------------------------------------------------
// create_document / edit_document core
// ---------------------------------------------------------------------------

/// Shared pipeline for a create or edit write.
///
/// Handles the stale-read guard, schema-frozen check, validation, optional dedup
/// gating (create only), filesystem write, git commit, reindex queuing, and diff
/// output. Callers are responsible for resolving `req.rel_path` and computing
/// `req.new_content` (e.g. applying a surgical old_string/new_string replacement)
/// before calling this.
///
/// The absolute path is deliberately re-resolved from `rel_path` immediately
/// before each filesystem action rather than computed once up front, so a path
/// validated earlier in this call cannot go stale across the awaits in between
/// (schema load, an embedding call, a Qdrant dedup query).
pub async fn write_document<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &WriteDeps<'_, E, Q>,
    req: WriteRequest<'_>,
) -> Result<WriteSuccess, WriteError> {
    let WriteRequest {
        rel_path,
        old_content,
        new_content,
        is_create,
        message,
        default_verb,
        force_new,
        operation,
        expected_hash,
    } = req;

    // 0. Include-pattern eligibility guard: reject paths the indexer would not
    //    pick up, before anything else runs. See `check_include_pattern`'s doc
    //    comment for why this must live here rather than in each caller.
    check_include_pattern(deps, rel_path)?;

    // 0.5. Early path-safety check: reject traversal (or any other
    // `resolve_safe_write_path` rejection) before ANY further processing,
    // and in particular before `validate::validate_content` below, which —
    // when `validation.lint_command` is configured — execs the configured
    // lint program with the raw, caller-supplied `rel_path` as an argument
    // and echoes its output back in the 422 body. `GlobSet::is_match` (the
    // include-pattern check just above) accepts `..` segments as ordinary
    // path characters — `**/*.md` matches `../../etc/cron.d/x.md` — so
    // without this, a path that fails the traversal check could still reach
    // the lint command first. The resolved `PathBuf` is intentionally
    // discarded here: it is re-resolved again immediately before each
    // filesystem mutation below (see the doc comment on that pattern), so
    // reusing this one would not shrink the TOCTOU window any further, only
    // remove one of the re-checks.
    safe_write_path(deps, rel_path)?;

    // 1. Optional stale-read guard.
    if let Some(expected) = expected_hash {
        let actual = crate::ingest::compute_hash_from_bytes(old_content.as_bytes());
        if !expected.trim().eq_ignore_ascii_case(&actual) {
            return Err(WriteError::StaleHash {
                expected: expected.trim().to_string(),
                actual,
            });
        }
    }

    // 2. Schema-frozen guard, then validate new_content.
    //
    // The schema is resolved from the TARGET path's directory, so writing into a
    // subdirectory is governed by that folder's rules regardless of where the
    // caller has been reading. This reads the shared, caller-owned cache rather
    // than rebuilding it — see `KbSearchServer::schema_cache`'s doc comment for
    // why that is safe to read without staleness after `update_schema`.
    let schemas = crate::schema::load_shared(deps.schema_cache);
    if let Some(reason) = schemas.is_frozen(Path::new(rel_path)) {
        return Err(WriteError::Frozen {
            reason: reason.to_string(),
        });
    }
    let schema = schemas.resolve_for(Path::new(rel_path));

    let (validation_result, validated) =
        validate::validate_content(Path::new(rel_path), new_content, schema, deps.validation)
            .await
            .map_err(|e| {
                error!("Validation error for '{}': {:#}", rel_path, e);
                WriteError::Io {
                    msg: format!("Failed to validate content: {}", e),
                }
            })?;

    if !validation_result.valid {
        return Err(WriteError::Validation {
            result: validation_result,
        });
    }

    // 3. Dedup gate: on create paths, check for near-duplicate existing documents.
    //    Gate runs only when: this is a create (not edit), dedup is enabled in
    //    config, and the caller has not set force_new = true.
    if is_create && deps.dedup_enabled && !matches!(force_new, Some(true)) {
        // Reuse the body already parsed during validation above rather than
        // re-deriving it here: that keeps the dedup query on exactly the
        // frontmatter-stripped basis the indexer embeds.
        let query_text = validated
            .as_ref()
            .map(|v| {
                let description = v.frontmatter.get("description").and_then(|d| d.as_str());
                build_dedup_query(&v.body, description, deps.prepend_description)
            })
            .unwrap_or_default();

        if query_text.trim().is_empty() {
            warn!(
                "Dedup gate skipped for '{}': no body text to compare",
                rel_path
            );
        } else {
            let empty_filters = SearchFilters {
                domain: None,
                r#type: None,
                tags: None,
            };
            // Detach the reranker: `dedup_threshold` is a cosine similarity, and a
            // cross-encoder relevance score is not comparable to it.
            let dedup_deps = RetrievalDeps {
                embed_client: deps.retrieval.embed_client,
                qdrant: deps.retrieval.qdrant,
                collection: deps.retrieval.collection,
                data_path: deps.retrieval.data_path,
                include_patterns: deps.retrieval.include_patterns,
                reranker: None,
            };
            match crate::retrieval::search(
                &dedup_deps,
                &query_text,
                &empty_filters,
                &dedup_search_opts(),
            )
            .await
            {
                Ok(results) => {
                    let top = results.into_iter().next().map(|r| {
                        let path = r
                            .payload
                            .get("file_path")
                            .and_then(|v| v.as_str())
                            .map(|p| {
                                crate::retrieval::relative_to_data(p, deps.canonical_data_path)
                            })
                            .unwrap_or_default();
                        (path, r.score)
                    });
                    if let Some((path, score)) = top.as_ref() {
                        tracing::debug!(
                            "Dedup gate for '{}': nearest '{}' at dense cosine {:.4} \
                             (threshold {:.2})",
                            rel_path,
                            path,
                            score,
                            deps.dedup_threshold
                        );
                    }
                    if let Some(hit) = dedup_verdict(top, deps.dedup_threshold) {
                        return Err(WriteError::DedupHit {
                            duplicate_of: hit.file_path,
                            similarity: hit.score,
                            threshold: deps.dedup_threshold,
                        });
                    }
                }
                Err(e) => {
                    warn!(
                        "Dedup search failed for '{}' (proceeding with write): {:#?}",
                        rel_path, e
                    );
                }
            }
        }
    }

    // 4. Validate the commit message BEFORE touching the filesystem. Rejecting it
    //    afterwards would leave the file written but never committed, and the
    //    index purge that follows a successful commit would never run.
    validate_commit_message(message)?;

    // Resolve fresh before creating directories too. The caller's resolution (if
    // any) happened before schema validation and a Qdrant dedup query — a wide
    // window in which a concurrent git sync could swap a component for a symlink,
    // which would otherwise let create_dir_all materialize real directories
    // outside the KB.
    let abs_path = safe_write_path(deps, rel_path)?;

    if let Some(parent) = abs_path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|e| {
            error!(
                "Failed to create parent directories for '{}': {}",
                abs_path.display(),
                e
            );
            WriteError::Io {
                msg: format!("Failed to create parent directories: {}", e),
            }
        })?;
    }

    // Re-verify immediately before writing. The initial resolution could only
    // canonicalize ancestors that existed at the time, and the work between then
    // and here — schema validation, an embedding call, a Qdrant dedup query — is a
    // wide window in which a concurrent git sync could swap a path component for a
    // symlink. Checking afterwards would only report an escape that already
    // happened; the freshly-verified path is what we write to.
    let abs_path = safe_write_path(deps, rel_path)?;

    if is_create {
        use tokio::io::AsyncWriteExt as _;
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&abs_path)
            .await
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists {
                    WriteError::AlreadyExists
                } else {
                    error!("Failed to create file '{}': {}", abs_path.display(), e);
                    WriteError::Io {
                        msg: format!("Failed to create file: {}", e),
                    }
                }
            })?;
        file.write_all(new_content.as_bytes()).await.map_err(|e| {
            error!("Failed to write file '{}': {}", abs_path.display(), e);
            WriteError::Io {
                msg: format!("Failed to write file: {}", e),
            }
        })?;
    } else {
        if !abs_path.exists() {
            return Err(WriteError::NotFound);
        }
        tokio::fs::write(&abs_path, new_content.as_bytes())
            .await
            .map_err(|e| {
                error!("Failed to write file '{}': {}", abs_path.display(), e);
                WriteError::Io {
                    msg: format!("Failed to write file: {}", e),
                }
            })?;
    }

    let commit_message = build_commit_message(
        message,
        &format!("docs: {} {}", default_verb, rel_path),
        operation,
    );

    let data_path_str = deps.canonical_data_path.to_str().unwrap_or_default();

    // `commit_and_sync` distinguishes WHERE it failed — see `git::CommitSyncError`
    // — and the two phases demand opposite handling. A `PreCommit` failure means
    // HEAD never moved, so the file write above (already on disk, not yet
    // committed) is rolled back and reported as "nothing changed". A `PostCommit`
    // failure means the commit is a real, durable part of local history — rolling
    // it back here would silently undo work that genuinely happened, so it is
    // left alone and reported as "committed, sync pending" instead.
    let commit_outcome = match git::commit_and_sync(
        deps.git_url,
        deps.branch,
        data_path_str,
        deps.token,
        rel_path,
        &commit_message,
        deps.commit_author_name,
        deps.commit_author_email,
    )
    .await
    {
        Ok(outcome) => outcome,

        Err(git::CommitSyncError::PreCommit(source)) => {
            error!(
                "commit_and_sync pre-commit failure for '{}', rolling back: {:#}",
                rel_path, source
            );

            // For a create, there is no HEAD content to restore to — undo the
            // filesystem write directly and unstage whatever `git add` staged.
            // For an edit, HEAD already has the previous content, so restore it
            // (this also un-stages any partial `git add`, in one step).
            let rollback = if is_create {
                match tokio::fs::remove_file(&abs_path).await {
                    Ok(()) => git::unstage(data_path_str, rel_path).await,
                    Err(e) => Err(anyhow::Error::new(e)
                        .context("Failed to remove newly-written file during rollback")),
                }
            } else {
                git::restore_from_head(data_path_str, rel_path).await
            };

            return match rollback {
                Ok(()) => Err(WriteError::PreCommitFailed {
                    rolled_back: true,
                    msg: format!("{:#}", source),
                }),
                // The rollback ITSELF failed — a third, worse state than either of
                // the above. The file may now be gone/changed on disk with no
                // corresponding commit, or the index may not match HEAD.
                Err(rollback_err) => {
                    error!(
                        "Rollback FAILED after a pre-commit git failure for '{}': {:#}. \
                         Original cause: {:#}. Filesystem and git state may now be \
                         inconsistent.",
                        rel_path, rollback_err, source
                    );
                    Err(WriteError::PreCommitFailed {
                        rolled_back: false,
                        msg: format!(
                            "Commit cause: {:#}. Rollback cause: {:#}",
                            source, rollback_err
                        ),
                    })
                }
            };
        }

        Err(git::CommitSyncError::PostCommit { sha, source }) => {
            warn!(
                "commit_and_sync post-commit (sync) failure for '{}', commit {} stands \
                 uncorrected: {:#}",
                rel_path, sha, source
            );

            // The local file already reflects the new content regardless of push
            // status, so the local index should too. `rebased_paths` is empty
            // here — the rebase never ran (fetch/rebase/push all happen after the
            // commit, so any of them failing means we never got as far as a
            // trustworthy rebase diff).
            crate::reindex::mark_paths(std::iter::once(PathBuf::from(rel_path)));

            return Ok(WriteSuccess {
                outcome: WriteOutcome::CommittedPendingSync,
                sha,
                rebased_paths: Vec::new(),
                diff: render_unified_diff(old_content, new_content, rel_path),
                sync_failure_cause: Some(format!("{:#}", source)),
            });
        }
    };

    // Mark this path — and anything the rebase pulled in from other commits —
    // dirty and return immediately. The reindex worker (src/reindex.rs) does the
    // actual chunk/embed/upsert work out of band; this call never blocks on it,
    // which is the whole point — embedding is far slower than a caller's request
    // timeout on a large document.
    crate::reindex::mark_paths(
        std::iter::once(PathBuf::from(rel_path))
            .chain(commit_outcome.rebased_paths.iter().cloned()),
    );

    Ok(WriteSuccess {
        outcome: WriteOutcome::Synced,
        sha: commit_outcome.sha,
        diff: render_unified_diff(old_content, new_content, rel_path),
        rebased_paths: commit_outcome.rebased_paths,
        sync_failure_cause: None,
    })
}

// ---------------------------------------------------------------------------
// delete_document core
// ---------------------------------------------------------------------------

/// Delete `rel_path`: read it (for the diff), remove it from disk, commit the
/// deletion to git, and queue reindex cleanup on success.
///
/// `rel_path` must already have been resolved by the caller against the KB root
/// (this function re-resolves and re-verifies it itself immediately before each
/// filesystem action — see `write_document`'s doc comment for why).
pub async fn delete_document<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &WriteDeps<'_, E, Q>,
    rel_path: &str,
    message: Option<&str>,
) -> Result<WriteSuccess, WriteError> {
    // Include-pattern eligibility guard, ahead of everything else — see
    // `check_include_pattern`'s doc comment for why this must live here rather
    // than in each caller.
    check_include_pattern(deps, rel_path)?;

    // Early path-safety check, ahead of everything else below — mirrors
    // `write_document`'s identical early check (see its comment for why this
    // must run before any further processing, not just before the filesystem
    // mutation). `delete_document` has no `validate_content` step to race
    // ahead of, but this keeps both halves of the pipeline consistent and
    // fails a traversal path before it can be reported by the wrong error
    // (`InvalidCommitMessage`) or after wasted work. The resolved `PathBuf` is
    // discarded — see that comment for why it is not reused.
    safe_write_path(deps, rel_path)?;

    // Validate the commit message BEFORE deleting anything. Rejecting it after
    // the removal would leave the file gone from disk but never committed, with
    // the Qdrant and state-DB purge — which only runs after a successful commit —
    // skipped too, so search would keep returning a document that no longer
    // exists.
    validate_commit_message(message)?;

    let abs_path = safe_write_path(deps, rel_path)?;
    if !abs_path.exists() {
        return Err(WriteError::NotFound);
    }

    let old_content = tokio::fs::read_to_string(&abs_path).await.map_err(|e| {
        error!("Failed to read '{}': {}", abs_path.display(), e);
        WriteError::Io {
            msg: format!("Failed to read file before deletion: {}", e),
        }
    })?;

    // Re-verify immediately before removing — see `write_document`'s doc comment
    // for why this is re-resolved rather than reusing the path from above.
    let abs_path = safe_write_path(deps, rel_path)?;
    tokio::fs::remove_file(&abs_path).await.map_err(|e| {
        error!("Failed to remove '{}': {}", abs_path.display(), e);
        WriteError::Io {
            msg: format!("Failed to remove file: {}", e),
        }
    })?;

    let commit_message = build_commit_message(
        message,
        &format!("docs: delete {}", rel_path),
        "delete_document",
    );

    let data_path_str = deps.canonical_data_path.to_str().unwrap_or_default();

    // `commit_and_sync` distinguishes WHERE it failed — see `git::CommitSyncError`
    // — which matters a great deal here, since the file is already gone from disk
    // by this point:
    //
    // - `PreCommit` (add/commit failed): HEAD never recorded the deletion, so the
    //   file's absence from disk is the ONLY trace of this call. Restore it from
    //   HEAD so the caller sees "nothing changed" and can safely retry.
    // - `PostCommit` (fetch/rebase/push failed): the deletion IS a real local
    //   commit — HEAD already reflects the file being gone. Restoring it here
    //   would resurrect a document that, as far as local git history is
    //   concerned, was legitimately deleted. Leave it deleted and report the sync
    //   as pending instead.
    let commit_outcome = match git::commit_and_sync(
        deps.git_url,
        deps.branch,
        data_path_str,
        deps.token,
        rel_path,
        &commit_message,
        deps.commit_author_name,
        deps.commit_author_email,
    )
    .await
    {
        Ok(outcome) => outcome,

        Err(git::CommitSyncError::PreCommit(source)) => {
            error!(
                "commit_and_sync pre-commit failure deleting '{}', restoring from HEAD: {:#}",
                rel_path, source
            );

            match git::restore_from_head(data_path_str, rel_path).await {
                Ok(()) => {
                    return Err(WriteError::PreCommitFailed {
                        rolled_back: true,
                        msg: format!("{:#}", source),
                    });
                }
                // The restore ITSELF failed — a third, worse state than either a
                // clean delete or a clean no-op: the file is gone from disk with
                // no corresponding commit.
                Err(restore_err) => {
                    error!(
                        "Restore FAILED after a pre-commit git failure deleting '{}': {:#}. \
                         Original cause: {:#}. The file is gone from disk and NOT \
                         committed — filesystem and git are now inconsistent.",
                        rel_path, restore_err, source
                    );
                    return Err(WriteError::PreCommitFailed {
                        rolled_back: false,
                        msg: format!(
                            "Commit cause: {:#}. Restore cause: {:#}",
                            source, restore_err
                        ),
                    });
                }
            }
        }

        Err(git::CommitSyncError::PostCommit { sha, source }) => {
            warn!(
                "commit_and_sync post-commit (sync) failure deleting '{}', deletion commit \
                 {} stands uncorrected: {:#}",
                rel_path, sha, source
            );

            // The file is already gone from local disk regardless of push status,
            // so the local index should reflect that regardless too.
            crate::reindex::mark_paths(std::iter::once(PathBuf::from(rel_path)));

            return Ok(WriteSuccess {
                outcome: WriteOutcome::CommittedPendingSync,
                sha,
                rebased_paths: Vec::new(),
                diff: render_unified_diff(&old_content, "", rel_path),
                sync_failure_cause: Some(format!("{:#}", source)),
            });
        }
    };

    // Mark this path — and anything the rebase pulled in — dirty and return
    // immediately. The worker's scoped indexer purges a path's Qdrant points and
    // state rows itself once it re-checks and finds the file gone (the
    // missing-file branch of `ingest::index_paths`), so there is no separate
    // purge to do here — this is "one reindex path" applied to deletes too, not a
    // special case.
    crate::reindex::mark_paths(
        std::iter::once(PathBuf::from(rel_path))
            .chain(commit_outcome.rebased_paths.iter().cloned()),
    );

    Ok(WriteSuccess {
        outcome: WriteOutcome::Synced,
        sha: commit_outcome.sha,
        diff: render_unified_diff(&old_content, "", rel_path),
        rebased_paths: commit_outcome.rebased_paths,
        sync_failure_cause: None,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ResolvedConfig;
    use crate::embed::EmbedClient;
    use crate::qdrant::QdrantStore;
    use std::sync::Arc;

    // -----------------------------------------------------------------------
    // dedup_verdict / build_dedup_query / dedup_search_opts — pure unit tests
    // (ported from mcp.rs; these are now write.rs's own logic)
    // -----------------------------------------------------------------------

    #[test]
    fn dedup_verdict_score_above_threshold_returns_hit() {
        let result = dedup_verdict(Some(("docs/existing.md".into(), 0.92)), 0.85);
        assert!(result.is_some());
        let hit = result.unwrap();
        assert_eq!(hit.file_path, "docs/existing.md");
        assert!((hit.score - 0.92).abs() < 1e-6);
    }

    #[test]
    fn dedup_verdict_no_results_allows() {
        assert!(dedup_verdict(None, 0.85).is_none());
    }

    #[test]
    fn build_dedup_query_prepends_description() {
        let q = build_dedup_query("Body text here.", Some("A short summary."), true);
        assert_eq!(q, "A short summary.\n\nBody text here.");
    }

    #[test]
    fn build_dedup_query_truncates_to_limit() {
        let long_body = "x".repeat(DEDUP_QUERY_CHAR_LIMIT * 2);
        let q = build_dedup_query(&long_body, None, false);
        assert_eq!(q.chars().count(), DEDUP_QUERY_CHAR_LIMIT);
    }

    #[test]
    fn dedup_search_opts_is_dense_only() {
        let opts = dedup_search_opts();
        assert!(!opts.hybrid);
        assert_eq!(opts.limit, 1);
        assert!(opts.min_score.is_none());
    }

    // -----------------------------------------------------------------------
    // resolve_safe_write_path unit tests (ported from mcp.rs; the function
    // itself moved here too — see this module's `resolve_safe_write_path`)
    // -----------------------------------------------------------------------

    #[test]
    fn safe_write_path_treats_a_leading_slash_as_the_kb_root() {
        // Callers cannot know where the KB lives inside the container, so `/x.md` and
        // `x.md` must address the same document. The escape checks still apply — this
        // resolves under the data root, it does not reach the real /etc.
        let tmp = tempfile::tempdir().unwrap();

        let rooted = resolve_safe_write_path(tmp.path(), "/notes/a.md").unwrap();
        let relative = resolve_safe_write_path(tmp.path(), "notes/a.md").unwrap();
        assert_eq!(rooted, relative);
        assert!(rooted.starts_with(tmp.path()));

        // And traversal is still rejected however it is spelled.
        assert!(resolve_safe_write_path(tmp.path(), "/../escape.md").is_err());
        assert!(resolve_safe_write_path(tmp.path(), "../escape.md").is_err());
    }

    #[test]
    fn safe_write_path_rejects_parent_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let result = resolve_safe_write_path(tmp.path(), "../../etc/shadow");
        assert!(result.is_err(), "parent-dir component must be rejected");
        let msg = result.unwrap_err();
        assert!(msg.contains(".."), "error should mention '..', got: {msg}");
    }

    #[test]
    fn safe_write_path_accepts_normal_nested_path() {
        let tmp = tempfile::tempdir().unwrap();
        // The file doesn't need to exist; the ancestor (tmp itself) does.
        let result = resolve_safe_write_path(tmp.path(), "subdir/docs/guide.md");
        assert!(
            result.is_ok(),
            "normal nested path should be accepted, got: {:?}",
            result
        );
        let abs = result.unwrap();
        assert!(
            abs.starts_with(tmp.path()),
            "returned path should be under data_root"
        );
    }

    #[test]
    fn safe_write_path_rejects_symlinked_ancestor_outside_root() {
        // Create two separate temp directories.
        let inside_tmp = tempfile::tempdir().unwrap();
        let outside_tmp = tempfile::tempdir().unwrap();

        // Create a subdirectory inside inside_tmp that is actually a symlink
        // pointing to outside_tmp.
        let escaped_dir = inside_tmp.path().join("escaped");
        std::os::unix::fs::symlink(outside_tmp.path(), &escaped_dir)
            .expect("failed to create symlink");

        // A path through the symlink: "escaped/secret.md"
        // Lexically this is under inside_tmp, but canonically it resolves outside.
        let result = resolve_safe_write_path(inside_tmp.path(), "escaped/secret.md");

        assert!(
            result.is_err(),
            "path through symlinked ancestor pointing outside root must be rejected"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("symlink") || msg.contains("escapes"),
            "error should mention symlink or escape, got: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // build_commit_message / render_unified_diff — pure unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn commit_message_has_trailers() {
        let msg = build_commit_message(None, "docs: add notes/guide.md", "create_document");
        assert!(msg.contains("Tool: md-kb-rag"));
        assert!(msg.contains("Operation: create_document"));
        assert!(msg.starts_with("docs: add notes/guide.md"));
    }

    #[test]
    fn commit_message_user_subject_overrides_default() {
        let msg = build_commit_message(Some("chore: x"), "docs: add y", "create_document");
        assert!(msg.starts_with("chore: x"));
    }

    #[test]
    fn diff_create_shows_all_additions() {
        let diff = render_unified_diff("", "line1\n", "docs/new.md");
        assert!(diff.contains("+line1"));
    }

    #[test]
    fn diff_delete_shows_all_removals() {
        let diff = render_unified_diff("line1\n", "", "docs/gone.md");
        assert!(diff.contains("-line1"));
    }

    // -----------------------------------------------------------------------
    // Test harness: a WriteDeps backed by a temp dir and (unreachable) real
    // EmbedClient/QdrantStore — matching the pattern `mcp.rs`'s own write tests
    // use, since dedup is disabled by default in `make_test_resolved_config`.
    // -----------------------------------------------------------------------

    fn test_embed_and_qdrant() -> (Arc<EmbedClient>, Arc<QdrantStore>) {
        let qdrant_config = crate::config::ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: "test".into(),
        };
        let qdrant = Arc::new(QdrantStore::new(&qdrant_config).unwrap());
        let embed_config = crate::config::ResolvedEmbeddingConfig {
            base_url: "http://localhost:8080/v1".into(),
            model: "test".into(),
            api_key: None,
            vector_size: 768,
            batch_size: 32,
            request_timeout_secs: 60,
            batch_concurrency: 4,
        };
        let embed = Arc::new(EmbedClient::new(&embed_config));
        (embed, qdrant)
    }

    /// Bundle owning everything `WriteDeps<'_, EmbedClient, QdrantStore>` borrows,
    /// so a test can build the deps and hold this alive for the call.
    struct Harness {
        embed: Arc<EmbedClient>,
        qdrant: Arc<QdrantStore>,
        canonical_data_path: PathBuf,
        include_patterns: globset::GlobSet,
        schema_cache: SharedSchemaCache,
        config: Arc<ResolvedConfig>,
        token: Option<String>,
    }

    impl Harness {
        fn new(tmp: &tempfile::TempDir, config: Arc<ResolvedConfig>) -> Self {
            let (embed, qdrant) = test_embed_and_qdrant();
            let canonical_data_path = tmp.path().canonicalize().unwrap();
            let mut builder = globset::GlobSetBuilder::new();
            builder.add(globset::Glob::new("**/*.md").unwrap());
            let schema_cache: SharedSchemaCache = Arc::new(std::sync::RwLock::new(Arc::new(
                crate::schema::SchemaCache::build(&canonical_data_path, &config.frontmatter),
            )));
            Harness {
                embed,
                qdrant,
                canonical_data_path,
                include_patterns: builder.build().unwrap(),
                schema_cache,
                config,
                token: None,
            }
        }

        fn deps(&self) -> WriteDeps<'_, EmbedClient, QdrantStore> {
            WriteDeps {
                retrieval: RetrievalDeps {
                    embed_client: &self.embed,
                    qdrant: &self.qdrant,
                    collection: &self.config.qdrant.collection,
                    data_path: &self.canonical_data_path,
                    include_patterns: &self.include_patterns,
                    reranker: None,
                },
                canonical_data_path: &self.canonical_data_path,
                schema_cache: &self.schema_cache,
                validation: &self.config.validation,
                prepend_description: self.config.chunking.prepend_description,
                dedup_enabled: self.config.write.dedup_enabled,
                dedup_threshold: self.config.write.dedup_threshold,
                git_url: self.config.source.git_url.as_deref(),
                branch: &self.config.source.branch,
                token: self.token.as_deref(),
                commit_author_name: &self.config.write.commit_author_name,
                commit_author_email: &self.config.write.commit_author_email,
            }
        }
    }

    fn make_req<'a>(rel_path: &'a str, new_content: &'a str, is_create: bool) -> WriteRequest<'a> {
        WriteRequest {
            rel_path,
            old_content: "",
            new_content,
            is_create,
            message: None,
            default_verb: if is_create { "add" } else { "update" },
            force_new: Some(true),
            operation: "test",
            expected_hash: None,
        }
    }

    // -----------------------------------------------------------------------
    // write_document core tests (ported from mcp.rs's write/delete suite,
    // exercised directly against `write::write_document`/`write::delete_document`
    // rather than through the MCP tool layer)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn stale_expected_hash_is_rejected_before_touching_the_file() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let stale_hash = crate::ingest::compute_hash_from_bytes(b"not the current content");
        let mut req = make_req("docs/edit-me.md", "---\ntitle: New\n---\n# New", false);
        req.old_content = "---\ntitle: Old\n---\n# Old";
        req.expected_hash = Some(&stale_hash);

        let err = write_document(&harness.deps(), req)
            .await
            .expect_err("stale hash must be rejected");
        match err {
            WriteError::StaleHash { expected, actual } => {
                assert_eq!(expected, stale_hash);
                assert_ne!(actual, stale_hash);
            }
            other => panic!("expected StaleHash, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn matching_expected_hash_proceeds_past_the_guard() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let old_content = "---\ntitle: Old\n---\n# Old";
        let correct_hash = crate::ingest::compute_hash_from_bytes(old_content.as_bytes());
        let mut req = make_req("docs/edit-me.md", "---\ntitle: New\n---\n# New", false);
        req.old_content = old_content;
        req.expected_hash = Some(&correct_hash);

        // No git repo under `tmp`, so this will fail later in the pipeline (at
        // the commit step, or NotFound since the file was never created on disk)
        // — the point of this test is only that it does NOT fail with StaleHash.
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(
            !matches!(err, WriteError::StaleHash { .. }),
            "a correct expected_hash must not be treated as stale, got {err:?}"
        );
    }

    #[tokio::test]
    async fn frozen_schema_rejects_the_write() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("notes");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(
            sub.join(crate::schema::SCHEMA_FILE_NAME),
            "not: [valid: yaml",
        )
        .unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let req = make_req("notes/new.md", "---\ntitle: T\n---\n# Body", true);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(matches!(err, WriteError::Frozen { .. }), "got {err:?}");
    }

    #[tokio::test]
    async fn validation_failure_carries_the_structured_result() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = crate::mcp::make_test_resolved_config(tmp.path());
        Arc::get_mut(&mut config).unwrap().frontmatter.required = vec!["title".into()];
        let harness = Harness::new(&tmp, config);

        let req = make_req(
            "guide/missing-title.md",
            "---\ntype: guide\n---\n# No title",
            true,
        );
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::Validation { result } => {
                assert!(!result.valid);
                assert!(result.field_errors.iter().any(|e| e.field == "title"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn create_on_existing_file_reports_already_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("existing.md"), "# Already here").unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let req = make_req("docs/existing.md", "---\ntitle: T\n---\n# New", true);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(matches!(err, WriteError::AlreadyExists), "got {err:?}");
    }

    // -----------------------------------------------------------------------
    // check_include_pattern: the eligibility guard must fire for create, edit,
    // and delete alike, regardless of which caller reaches this pipeline.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn create_rejects_path_outside_include_patterns() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        // Harness's include globset is `**/*.md` only (see `Harness::new`).
        let harness = Harness::new(&tmp, config);

        let req = make_req("notes.txt", "Some plain text", true);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::UnsafePath { msg } => {
                assert!(msg.contains("indexable include pattern"), "got: {msg}");
            }
            other => panic!("expected UnsafePath, got {other:?}"),
        }
        assert!(
            !tmp.path().join("notes.txt").exists(),
            "nothing should be written when the include-pattern guard rejects the create"
        );
    }

    #[tokio::test]
    async fn edit_rejects_path_outside_include_patterns() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("notes.txt"), "old text").unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let mut req = make_req("notes.txt", "new text", false);
        req.old_content = "old text";
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(matches!(err, WriteError::UnsafePath { .. }), "got {err:?}");
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("notes.txt")).unwrap(),
            "old text",
            "the existing file must be untouched when the include-pattern guard rejects the edit"
        );
    }

    #[tokio::test]
    async fn delete_rejects_path_outside_include_patterns() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("notes.txt"), "content").unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let err = delete_document(&harness.deps(), "notes.txt", None)
            .await
            .unwrap_err();
        assert!(matches!(err, WriteError::UnsafePath { .. }), "got {err:?}");
        assert!(
            tmp.path().join("notes.txt").exists(),
            "file must be untouched when the include-pattern guard rejects the delete"
        );
    }

    #[tokio::test]
    async fn edit_of_missing_file_reports_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let req = make_req("docs/nonexistent.md", "---\ntitle: T\n---\n# Body", false);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(matches!(err, WriteError::NotFound), "got {err:?}");
    }

    #[tokio::test]
    async fn delete_of_missing_file_reports_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let err = delete_document(&harness.deps(), "docs/nonexistent.md", None)
            .await
            .unwrap_err();
        assert!(matches!(err, WriteError::NotFound), "got {err:?}");
    }

    #[tokio::test]
    async fn invalid_commit_message_is_rejected_before_any_filesystem_change() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("del-me.md"), "# Content").unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let err = delete_document(&harness.deps(), "docs/del-me.md", Some("bad\nmessage"))
            .await
            .unwrap_err();
        assert!(
            matches!(err, WriteError::InvalidCommitMessage { .. }),
            "got {err:?}"
        );
        assert!(
            sub.join("del-me.md").exists(),
            "file must be untouched when the commit message is rejected up front"
        );
    }

    // -----------------------------------------------------------------------
    // Git-backed pre-commit / post-commit rollback tests (ported from mcp.rs)
    // -----------------------------------------------------------------------

    fn head_sha(work: &tempfile::TempDir) -> String {
        let out = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(work.path())
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn git_status(work: &tempfile::TempDir) -> String {
        let out = std::process::Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(work.path())
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn git_commit_all(work: &tempfile::TempDir, rel_path: &str, message: &str) {
        std::process::Command::new("git")
            .args(["add", "--", rel_path])
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
                message,
            ])
            .current_dir(work.path())
            .output()
            .unwrap();
    }

    /// See `mcp.rs`'s identical helper for why this is repo-local git CONFIG
    /// rather than a `.git/hooks/pre-commit` script.
    fn force_git_commit_to_fail(work: &tempfile::TempDir) {
        for args in [
            ["config", "commit.gpgsign", "true"],
            ["config", "user.signingkey", "nonexistent-bogus-key-id"],
        ] {
            std::process::Command::new("git")
                .args(args)
                .current_dir(work.path())
                .output()
                .unwrap();
        }
    }

    fn git_backed_harness(work: &tempfile::TempDir) -> Harness {
        let mut config = crate::mcp::make_test_resolved_config(work.path());
        // Bypass the dedup gate: it would otherwise call out to a (nonexistent)
        // embedding service before we ever reach the commit.
        Arc::get_mut(&mut config).unwrap().write.dedup_enabled = false;
        Harness::new(work, config)
    }

    #[tokio::test]
    async fn create_precommit_failure_removes_the_new_file_and_rolls_back_cleanly() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let harness = git_backed_harness(&work);

        let req = make_req(
            "docs/new.md",
            "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n",
            true,
        );
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(rolled_back),
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }
        assert!(!work.path().join("docs/new.md").exists());
        assert_eq!(head_before, head_sha(&work));
        assert_eq!(git_status(&work), "");
    }

    #[tokio::test]
    async fn edit_precommit_failure_restores_previous_content() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original =
            "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
        std::fs::write(work.path().join("edit-me.md"), original).unwrap();
        git_commit_all(&work, "edit-me.md", "add edit-me.md");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let harness = git_backed_harness(&work);

        let mut req = make_req(
            "edit-me.md",
            "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New body\n",
            false,
        );
        req.old_content = original;
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(rolled_back),
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }
        assert_eq!(
            std::fs::read_to_string(work.path().join("edit-me.md")).unwrap(),
            original
        );
        assert_eq!(head_before, head_sha(&work));
        assert_eq!(git_status(&work), "");
    }

    #[tokio::test]
    async fn create_synced_write_marks_the_path_dirty_and_returns_a_diff() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let harness = git_backed_harness(&work);

        let pending_before = crate::reindex::REINDEX_QUEUE.snapshot().pending_paths;

        // A path unique to this test: the global `REINDEX_QUEUE` is a process-wide
        // `HashSet<PathBuf>` shared with every other test in this binary (including
        // `mcp.rs`'s own `docs/queued.md`-named test), so reusing a path literal
        // used elsewhere could collide and silently fail to grow the pending count.
        let req = make_req(
            "docs/queued-write-core-test.md",
            "---\ntitle: Queued\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n",
            true,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();
        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(!success.sha.is_empty());
        assert!(success.diff.contains("+title: Queued"));

        let pending_after = crate::reindex::REINDEX_QUEUE.snapshot().pending_paths;
        assert!(pending_after > pending_before);
    }

    #[tokio::test]
    async fn delete_precommit_failure_restores_the_file() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original = "---\ntitle: D\n---\n\n# Body\n";
        std::fs::write(work.path().join("doomed.md"), original).unwrap();
        git_commit_all(&work, "doomed.md", "add doomed.md");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let harness = git_backed_harness(&work);

        let err = delete_document(&harness.deps(), "doomed.md", None)
            .await
            .unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(rolled_back),
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }
        assert!(work.path().join("doomed.md").exists());
        assert_eq!(
            std::fs::read_to_string(work.path().join("doomed.md")).unwrap(),
            original
        );
        assert_eq!(head_before, head_sha(&work));
        assert_eq!(git_status(&work), "");
    }

    #[tokio::test]
    async fn delete_with_no_git_repo_reports_unrecoverable_precommit_failure() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(
            sub.join("delete-me.md"),
            "---\ntitle: Delete Me\n---\n# Body",
        )
        .unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let err = delete_document(&harness.deps(), "docs/delete-me.md", None)
            .await
            .unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(!rolled_back),
            other => panic!("expected PreCommitFailed{{rolled_back: false}}, got {other:?}"),
        }
        // The restore could not put it back (there is no repo to restore from), so
        // the file really is gone — that IS the inconsistent state being reported.
        assert!(!sub.join("delete-me.md").exists());
    }

    #[tokio::test]
    async fn delete_postcommit_failure_leaves_the_commit_and_reports_pending_sync() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::write(
            work.path().join("doomed.md"),
            "---\ntitle: D\n---\n\n# Body\n",
        )
        .unwrap();
        git_commit_all(&work, "doomed.md", "add doomed.md");

        let mut config = crate::mcp::make_test_resolved_config(work.path());
        {
            let c = Arc::get_mut(&mut config).unwrap();
            c.write.dedup_enabled = false;
            c.source.git_url = Some("/nonexistent/path/to/repo.git".to_string());
        }
        let harness = Harness::new(&work, config);

        let success = delete_document(&harness.deps(), "doomed.md", None)
            .await
            .unwrap();
        assert_eq!(success.outcome, WriteOutcome::CommittedPendingSync);
        assert!(success.sync_failure_cause.is_some());
        assert!(!work.path().join("doomed.md").exists());
    }

    // -----------------------------------------------------------------------
    // G1 — the traversal check must run before validation (and, in particular,
    // before an exec of a configured `validation.lint_command` against the raw
    // caller-supplied path). See `write_document`'s "0.5" step comment.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn create_traversal_is_rejected_before_the_lint_command_ever_runs() {
        let tmp = tempfile::tempdir().unwrap();
        let marker = tmp.path().join("lint-invoked-marker");
        let mut config = crate::mcp::make_test_resolved_config(tmp.path());
        // A fake lint command that leaves undeniable evidence it ran. If the
        // traversal check did not run until after `validate::validate_content`
        // (the pre-fix ordering), this file would exist afterwards.
        Arc::get_mut(&mut config).unwrap().validation.lint_command = Some(vec![
            "sh".into(),
            "-c".into(),
            format!("touch '{}'", marker.display()),
        ]);
        let harness = Harness::new(&tmp, config);

        // Both spellings from the plan: `GlobSet::is_match` (the include-pattern
        // check that runs just before this one) accepts `..` segments as plain
        // characters, so `**/*.md` matches both of these — the traversal check
        // is the only thing standing between them and the lint command.
        for traversal_path in ["../escape.md", "a/../../escape.md"] {
            let req = make_req(traversal_path, "---\ntitle: T\n---\n# Body", true);
            let err = write_document(&harness.deps(), req).await.unwrap_err();
            assert!(
                matches!(err, WriteError::UnsafePath { .. }),
                "expected UnsafePath (rejected before validation) for '{traversal_path}', \
                 got {err:?}"
            );
            assert!(
                !marker.exists(),
                "lint command must not run for a traversal path '{traversal_path}' \
                 rejected before validation"
            );
        }
    }

    #[tokio::test]
    async fn delete_traversal_is_rejected_before_commit_message_validation() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        // An invalid commit message (a newline) would normally be reported as
        // `InvalidCommitMessage` — but a traversal path must be rejected ahead
        // of that check, mirroring `write_document`'s ordering.
        let err = delete_document(&harness.deps(), "../escape.md", Some("bad\nmessage"))
            .await
            .unwrap_err();
        assert!(
            matches!(err, WriteError::UnsafePath { .. }),
            "expected UnsafePath ahead of commit-message validation, got {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // G2 — a canonicalize failure inside the path-safety check must surface as
    // `WriteError::Internal`, never `WriteError::UnsafePath`, so a caller-safe
    // transport (`web.rs`) can tell the two apart. See `WriteError::Internal`'s
    // doc comment.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn missing_data_root_reports_internal_not_unsafe_path() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        // Remove the data root out from under the already-built deps: this
        // forces `resolve_safe_write_path`'s `data_root.canonicalize()` call to
        // fail. The resulting message is allowed to embed this absolute path
        // (this is `write.rs`'s own classification test, not a transport-facing
        // one) — the point is that it must be `Internal`, not `UnsafePath`.
        std::fs::remove_dir_all(tmp.path()).unwrap();

        let req = make_req("docs/new.md", "---\ntitle: T\n---\n# Body", true);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::Internal { msg } => {
                assert!(msg.contains("cannot canonicalize"), "got: {msg}");
            }
            other => panic!("expected Internal, got {other:?}"),
        }
    }
}
