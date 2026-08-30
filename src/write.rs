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

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use futures::stream::{self, StreamExt};
use tracing::{error, warn};

use crate::config::ValidationConfig;
use crate::embed::QueryEmbedder;
use crate::git;
use crate::qdrant::RetrievalStore;
use crate::retrieval::{RetrievalDeps, SearchFilters};
use crate::schema::{self, SharedSchemaCache};
use crate::state::StateDb;
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
///
/// It deliberately does NOT reproduce `chunking.prepend_heading_path`'s
/// breadcrumb, and that is safe for a non-obvious structural reason worth
/// stating: `chunk::annotate_heading_paths` walks sections with an ancestor
/// stack that starts empty for every document, and a chunk's heading path is
/// fixed from whichever section started it. A document's FIRST chunk is always
/// seeded from its first section, whose ancestor path is therefore always
/// empty — so chunk 0 never carries a breadcrumb, whatever the document's
/// heading structure. Since a create-path dedup query is doc-start text scored
/// against the corpus, matching chunk 0's basis is what matters.
///
/// If a future chunking change breaks that invariant — anything that can give
/// a document's first chunk a prefix this function does not build — the dedup
/// gate silently starts comparing unlike text and lets near-duplicates through
/// with no error. `mcp::tests::build_dedup_query_matches_chunk_prepend_format`
/// is the pin; keep it honest rather than adjusting it to match a regression.
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
        // Same reasoning as `hybrid` above: phrase matching adds a third fused
        // arm, which this dense-only comparison has no use for.
        phrase: false,
        explain: false,
        modified_after: None,
        modified_before: None,
        path_prefix: None,
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
    /// The dirty-path queue every successful write marks paths on before
    /// returning — see `reindex::ReindexQueue`. Unlike `state` below, this has
    /// no `None`/optional mode: a write that lands but never marks its path
    /// dirty is a correctness bug (the document silently never gets indexed),
    /// not a degraded-but-acceptable path, so every call site must supply a
    /// real queue. Borrowed the same way `schema_cache`/`retrieval.qdrant`/
    /// `retrieval.embed_client` are — the owning transport (`KbSearchServer`,
    /// `UiState`) holds an `Arc<ReindexQueue>` field and lends a plain
    /// reference in for the duration of one write.
    pub queue: &'a crate::reindex::ReindexQueue,
    /// The document metadata index. Three consumers, all going through
    /// `StateDb::links_targeting`'s reverse lookup ("what points at this
    /// path"): `write_document_move` and `move_directory` find documents whose
    /// body links to the move's SOURCE path so their link text can be
    /// rewritten in the same commit as the move, and `delete_document` warns
    /// about documents that link to the file being removed.
    ///
    /// `None` disables all three — the move/delete itself still happens, the
    /// reverse-link query is just skipped. For a move that means it
    /// leaves every other document's links exactly as they were (they still
    /// self-heal on that document's own next reindex, since `document_links`
    /// is rebuilt from each document's current on-disk body, not trusted as an
    /// authoritative index). This exists for callers/tests that have no
    /// `StateDb` handle at all — it is NOT a normal operating mode. Both real
    /// callers (`mcp.rs`'s `KbSearchServer`, `web.rs`'s `UiState`) have a
    /// `StateDb` available via their own `Arc<OnceCell<StateDb>>` and MUST
    /// pass `Some` here so production writes always rewrite incoming links.
    pub state: Option<&'a StateDb>,
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
    /// When `Some`, turns this call into a document MOVE instead of a create/edit:
    /// `rel_path` is the move's SOURCE and this is its DESTINATION. The caller is
    /// still responsible for computing `new_content` (this function never reads
    /// `rel_path`'s content itself, move or not) — typically the source's current
    /// content, possibly transformed.
    ///
    /// Both paths are subject to the same eligibility (include-pattern) and
    /// path-safety checks the non-move path applies to `rel_path`, and either
    /// directory being schema-frozen blocks the whole move. Frontmatter
    /// validation, however, runs against the DESTINATION's resolved schema, not
    /// the source's — that is the whole point of a move: the destination
    /// directory may enforce different frontmatter than the source did. The
    /// create-only dedup gate never runs for a move (it is not a create, and the
    /// document's own pre-move content would trivially self-match).
    ///
    /// `is_create` must be `false` whenever this is `Some` — a create has no
    /// source to move from, so combining the two is a caller bug, reported as
    /// `WriteError::Internal` rather than any user-facing variant. `expected_hash`
    /// IS applied to the move path: it guards against a stale read of the
    /// SOURCE, checked against `old_content` before anything touches the
    /// filesystem, with the same `WriteError::StaleHash` contract as the
    /// non-move path. `force_new` remains meaningless for a move and is simply
    /// ignored when `dest_path` is `Some` — there is no dedup gate to bypass.
    ///
    /// `None` (the default) is the existing create/edit behavior, byte-for-byte
    /// unchanged — this field did not exist before, so every existing caller gets
    /// `None` for free.
    pub dest_path: Option<&'a str>,
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
    /// Repo-relative paths of OTHER documents whose link text to the move's
    /// SOURCE was rewritten to point at its new location, and which rode along
    /// in the same commit. Always empty for a create/edit/delete, and for a
    /// move with `WriteDeps::state == None` or with no incoming links to
    /// rewrite. A move silently editing other documents without surfacing
    /// which ones is not acceptable — callers must report this list, not just
    /// the move's own source/destination.
    pub rewritten_paths: Vec<String>,
    /// (#229) Repo-relative paths of OTHER documents that still link to the
    /// document just DELETED, per `StateDb::links_targeting`'s reverse lookup.
    /// Always empty for a create/edit/move — only `delete_document` populates
    /// this, and only when `WriteDeps::state` is `Some` (see that field's doc
    /// comment) and at least one referencing document exists. `delete_document`
    /// does not refuse the delete or rewrite these documents — see its own
    /// comment for why a dangling link here is treated as self-healing, same
    /// as everywhere else in this pipeline — this field exists purely so a
    /// caller with no access to server logs (the `warn!` #181 added) can still
    /// learn what it warned about and decide whether follow-up work is needed.
    pub referencing_paths: Vec<String>,
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
// (#179) Content-mode helpers: frontmatter patch / append
//
// Both compute a full `new_content` string the same way `mcp.rs`'s
// `apply_surgical` already does for old_string/new_string — a pure function
// the caller (currently only `mcp.rs`'s `write_document_edit`) invokes BEFORE
// building a `WriteRequest`, not a new field on `WriteRequest` itself. That
// keeps `write_document`'s core pipeline (schema validation, the dedup gate,
// the commit, the pre-commit rollback) completely unaware that a patch or an
// append happened at all: by the time it sees `new_content`, a patch/append
// write looks exactly like a full-replace write of that same content, so it
// participates in schema validation, `expected_hash`, and rollback for free,
// with no special-casing anywhere in the pipeline below this point.
// ---------------------------------------------------------------------------

/// A single structured edit to a document's OWN frontmatter, applied by
/// [`apply_frontmatter_patch`].
///
/// Mirrors `schema::SchemaEdit`'s vocabulary
/// (`set_field`/`remove_field`/`add_values`/`remove_values`) — this
/// codebase's established idiom for "a structured edit instead of a
/// free-text patch" (see `update_schema`) — applied here to a document's own
/// frontmatter VALUES rather than a schema's field DECLARATIONS. `field` is a
/// dot-path, same convention `schema::get_by_dotpath`/`set_by_dotpath` and
/// `GetSchemaParams::fields`/`SearchFiltersInput` already use throughout this
/// codebase.
#[derive(Debug, Clone, PartialEq)]
pub enum FrontmatterEdit {
    /// Set (or create) a field to an exact value, replacing whatever was
    /// there. Always succeeds — mirrors `SchemaEdit::SetField`'s unconditional
    /// insert-or-replace.
    SetField {
        field: String,
        value: serde_json::Value,
    },
    /// Remove a field declaration. Errors if the field is not currently set —
    /// mirrors `SchemaEdit::RemoveField`'s identical refusal, so a caller
    /// cannot mistake "nothing to remove" for a silent no-op.
    RemoveField { field: String },
    /// Append values to a list field, creating it (as a new list) if absent,
    /// de-duplicated against what is already there. Errors if the field
    /// exists and is not a list.
    AddValues {
        field: String,
        values: Vec<serde_json::Value>,
    },
    /// Remove values from a list field. Errors if the field is absent or is
    /// not a list — mirrors `SchemaEdit::RemoveValues`'s identical refusal.
    RemoveValues {
        field: String,
        values: Vec<serde_json::Value>,
    },
}

/// Split `content` into `(frontmatter_block, body)` at the byte level,
/// preserving BOTH halves EXACTLY as they appear in `content` — every byte of
/// `content` is accounted for in exactly one of the two halves, concatenating
/// them losslessly reconstructs `content`.
///
/// This is deliberately NOT `validate::parse_frontmatter_raw` (the
/// `gray_matter`-backed parser used everywhere else in this codebase): that
/// function trims trailing whitespace from the body it returns, so its output
/// is not always a byte-exact suffix of the original content — fine for
/// validation (which only cares about the parsed VALUES), fatal here, where
/// [`apply_append`] depends on exactness to guarantee it can never write into
/// the frontmatter block at all, structurally, rather than by carefully
/// avoiding it.
///
/// Delimiter convention: `content` must begin with a `---` line (`\n`- or
/// `\r\n`-terminated); the block ends at the next line that is exactly `---`
/// (optionally `\r`-terminated), found by scanning line-by-line so a literal
/// `---` inside a YAML value (e.g. a horizontal-rule in a `description`
/// string) can never be mistaken for the closing delimiter. Returns `None`
/// when `content` has no opening delimiter, or an opening delimiter with no
/// matching close — both read as "no frontmatter block", the same as
/// `gray_matter` treats them.
fn split_frontmatter_bytes(content: &str) -> Option<(&str, &str)> {
    let after_open = content
        .strip_prefix("---\r\n")
        .or_else(|| content.strip_prefix("---\n"))?;

    let mut offset = 0usize;
    for line in after_open.split_inclusive('\n') {
        let trimmed = line.strip_suffix('\n').unwrap_or(line);
        let trimmed = trimmed.strip_suffix('\r').unwrap_or(trimmed);
        if trimmed == "---" {
            let close_end = offset + line.len();
            let fm_len = (content.len() - after_open.len()) + close_end;
            return Some((&content[..fm_len], &content[fm_len..]));
        }
        offset += line.len();
    }
    None
}

/// Remove the value at a dot-path, mirroring `schema::get_by_dotpath`'s own
/// traversal. Not in `schema.rs` alongside its `get`/`set` siblings because
/// this module's remit is deliberately kept to `write.rs`/`mcp.rs`/`web.rs`
/// (see this crate's write-pipeline module boundaries) — the two sibling
/// functions are schema-declaration helpers `schema.rs` owns for its own
/// `apply_defaults`; this one is document-frontmatter-only. Returns whether
/// anything was actually removed.
fn remove_by_dotpath(frontmatter: &mut HashMap<String, serde_json::Value>, path: &str) -> bool {
    let segments: Vec<&str> = path.split('.').collect();
    if segments.len() == 1 {
        return frontmatter.remove(segments[0]).is_some();
    }
    let Some(mut cursor) = frontmatter
        .get_mut(segments[0])
        .and_then(|v| v.as_object_mut())
    else {
        return false;
    };
    for segment in &segments[1..segments.len() - 1] {
        let Some(next) = cursor.get_mut(*segment).and_then(|v| v.as_object_mut()) else {
            return false;
        };
        cursor = next;
    }
    cursor.remove(segments[segments.len() - 1]).is_some()
}

/// Render a frontmatter map back to a `---`-delimited YAML block (including
/// both delimiters and the trailing newline after the closing one).
///
/// Keys are sorted (`BTreeMap`) before serializing — the exact same reasoning
/// as `SchemaFile::to_yaml`'s identical `BTreeMap` conversion: `HashMap`
/// iteration order is unspecified (and randomized per-process), so
/// serializing straight from it would reorder every field on every patch, for
/// no reason connected to what actually changed. Sorting trades that for a
/// deterministic, minimal diff — at the cost of NOT preserving whatever key
/// order (or comments) the document's own frontmatter happened to have,
/// exactly the same trade-off this codebase already made for
/// `.kb-schema.yaml` when `update_schema` rewrites one.
fn render_frontmatter_block(
    frontmatter: &HashMap<String, serde_json::Value>,
    newline: &str,
) -> Result<String, String> {
    let ordered: BTreeMap<&String, &serde_json::Value> = frontmatter.iter().collect();
    let yaml = serde_yaml_ng::to_string(&ordered)
        .map_err(|e| format!("failed to serialize frontmatter: {e}"))?;
    if newline == "\n" {
        return Ok(format!("---\n{yaml}---\n"));
    }
    // serde_yaml_ng always emits LF. Re-terminate every line with the
    // document's own ending so a patch does not silently convert a CRLF file
    // into a mixed-ending one — the bytes would all still be there, but every
    // subsequent diff of that document would show the whole frontmatter block
    // as changed.
    let converted: String = yaml
        .split_inclusive('\n')
        .map(|line| format!("{}{newline}", line.trim_end_matches('\n')))
        .collect();
    Ok(format!("---{newline}{converted}---{newline}"))
}

/// The line ending `content` uses, for round-tripping a rewrite through it.
///
/// Decided by the first ending actually present, not by a majority vote: a
/// document with mixed endings is already inconsistent, and picking its first
/// one at least keeps a rewrite from making the inconsistency worse. Content
/// with no newline at all gets LF, matching what this project writes by
/// default everywhere else.
fn detect_newline(content: &str) -> &'static str {
    match content.find('\n') {
        Some(i) if i > 0 && content.as_bytes()[i - 1] == b'\r' => "\r\n",
        _ => "\n",
    }
}

/// Apply a structured frontmatter patch to `old_content`, returning the full
/// new document content (frontmatter block + body, byte-identical body).
///
/// Parses the existing frontmatter via `validate::parse_frontmatter_raw` (the
/// same basis `write_document`'s own validation step re-parses immediately
/// afterward — see the doc comment on this module's content-mode-helpers
/// section for why that duplication is fine: this function's OUTPUT is just
/// ordinary `new_content`, re-validated from scratch like any other write),
/// applies each edit in order, then re-serializes and reattaches the ORIGINAL
/// body untouched — this function never reads, modifies, or even fully
/// re-parses the body, so it cannot corrupt it, structurally, not just by
/// convention.
///
/// Handles a document with no existing frontmatter block by creating one
/// (mirrors `SchemaEdit::AddValues`'s "creating the field if absent"): the
/// whole original `old_content` becomes the body, separated from the new
/// frontmatter block by a blank line (unless the body is empty, in which case
/// no trailing blank line is added either).
///
/// `expected_hash` (checked by the caller, both before this runs and again
/// under `GIT_LOCK` immediately before the write — see `write_document`'s
/// step 1 and its re-check) still guards the WHOLE file, unchanged, and that
/// is deliberate even though a patch only ever touches frontmatter: the body
/// reattached here is whatever `old_content` happened to contain, so a stale
/// read of the BODY must still be caught, or a patch computed against a
/// stale `old_content` could silently commit a stale body over a concurrent
/// body edit that landed in between. Since this function always derives the
/// body from the exact `old_content` the hash was checked against, the
/// existing whole-file guard already provides that protection for free — no
/// patch-specific handling is needed here or in `write_document`.
pub fn apply_frontmatter_patch(
    old_content: &str,
    edits: &[FrontmatterEdit],
) -> Result<String, String> {
    let (fm_block, body) = split_frontmatter_bytes(old_content).unwrap_or(("", old_content));
    let had_frontmatter = !fm_block.is_empty();

    let (mut frontmatter, _) = validate::parse_frontmatter_raw(old_content);

    for edit in edits {
        match edit {
            FrontmatterEdit::SetField { field, value } => {
                schema::set_by_dotpath(&mut frontmatter, field, value.clone());
            }
            FrontmatterEdit::RemoveField { field } => {
                if !remove_by_dotpath(&mut frontmatter, field) {
                    return Err(format!(
                        "field '{field}' is not set in this document's frontmatter"
                    ));
                }
            }
            FrontmatterEdit::AddValues { field, values } => {
                let mut existing: Vec<serde_json::Value> =
                    match schema::get_by_dotpath(&frontmatter, field) {
                        Some(serde_json::Value::Array(arr)) => arr.clone(),
                        Some(_) => {
                            return Err(format!(
                                "field '{field}' is not a list in this document's frontmatter"
                            ));
                        }
                        None => Vec::new(),
                    };
                for v in values {
                    if !existing.contains(v) {
                        existing.push(v.clone());
                    }
                }
                schema::set_by_dotpath(&mut frontmatter, field, serde_json::Value::Array(existing));
            }
            FrontmatterEdit::RemoveValues { field, values } => {
                let existing = match schema::get_by_dotpath(&frontmatter, field) {
                    Some(serde_json::Value::Array(arr)) => arr.clone(),
                    Some(_) => {
                        return Err(format!(
                            "field '{field}' is not a list in this document's frontmatter"
                        ));
                    }
                    None => {
                        return Err(format!(
                            "field '{field}' has no value list in this document's frontmatter"
                        ));
                    }
                };
                let filtered: Vec<serde_json::Value> = existing
                    .into_iter()
                    .filter(|v| !values.contains(v))
                    .collect();
                schema::set_by_dotpath(&mut frontmatter, field, serde_json::Value::Array(filtered));
            }
        }
    }

    let new_fm_block = render_frontmatter_block(&frontmatter, detect_newline(old_content))?;

    if had_frontmatter {
        Ok(format!("{new_fm_block}{body}"))
    } else if body.is_empty() {
        Ok(new_fm_block)
    } else {
        Ok(format!("{new_fm_block}\n{body}"))
    }
}

/// Append `text` to the end of `old_content`'s BODY — never past the
/// frontmatter block, structurally guaranteed by reusing
/// [`split_frontmatter_bytes`] rather than a substring/offset computed by
/// hand: whatever that function calls the frontmatter block is copied through
/// completely untouched, and `text` only ever lands inside whatever it calls
/// the body.
///
/// Exactly one newline separates existing body content from `text` — this
/// function does not fabricate blank-line spacing beyond that (a caller that
/// wants a blank line before its entry includes the leading newline in
/// `text` itself; see `write_document.md`), except for the one case where
/// there is no separator to reuse at all: a frontmatter block with an empty
/// body gets a single blank line before `text`, so the appended content does
/// not land glued to the closing `---`.
///
/// Handles a document with no frontmatter block (appends to the whole
/// content), an empty file (the result is just `text`), and a body with no
/// trailing newline (one is inserted before appending) — see this function's
/// tests for each case.
pub fn apply_append(old_content: &str, text: &str) -> String {
    let (fm_block, body) = split_frontmatter_bytes(old_content).unwrap_or(("", old_content));
    // Match the document's own line ending rather than always emitting LF —
    // otherwise the first append to a CRLF document glues an LF-terminated
    // block onto it and leaves the file with mixed endings.
    let nl = detect_newline(old_content);

    let mut new_body = body.to_string();
    if !new_body.is_empty() && !new_body.ends_with('\n') {
        new_body.push_str(nl);
    }
    if !fm_block.is_empty() && new_body.is_empty() {
        new_body.push_str(nl);
    }
    // Normalize the caller's text to the document's ending too: an agent
    // composing an append has no idea what the file on disk uses.
    let appended: String = text
        .trim_end_matches('\n')
        .trim_end_matches('\r')
        .split_inclusive('\n')
        .map(|line| {
            let stripped = line.trim_end_matches('\n').trim_end_matches('\r');
            if line.ends_with('\n') {
                format!("{stripped}{nl}")
            } else {
                stripped.to_string()
            }
        })
        .collect();
    new_body.push_str(&appended);
    new_body.push_str(nl);

    format!("{fm_block}{new_body}")
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
    // A move is a different enough shape at nearly every step (two paths through
    // eligibility/safety/frozen checks, validation against the DESTINATION's
    // schema rather than `rel_path`'s, a write-then-remove filesystem sequence, a
    // two-path rollback) that folding it into the branches below would make both
    // harder to follow — see `write_document_move`'s doc comment.
    if req.dest_path.is_some() {
        return write_document_move(deps, req).await;
    }

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
        dest_path: _,
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
            let empty_filters = SearchFilters::default();
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

    // 4.5. Acquire GIT_LOCK now and hold ONE guard across every remaining step —
    // the final on-disk write, the commit, and any rollback — rather than
    // acquiring it only just before the commit as this used to. That gap is
    // exactly what let #142 reopen the `expected_hash` stale-read guard: step 1
    // above checks `expected_hash` once, against the caller-supplied
    // `old_content`, but schema validation and (for a create) the dedup gate's
    // embedding+Qdrant round trip both run AFTER that check and can take a
    // while — long enough for a webhook merge, which independently needs this
    // same lock for its own fetch + `git merge --ff-only`, to change the file's
    // on-disk content in between with nothing to detect it. Acquiring the lock
    // here and re-verifying against LIVE content immediately before the write
    // below (rather than only re-checking the now-stale `old_content` again)
    // closes the window instead of just narrowing it: nothing else can touch
    // the working tree for the rest of this call.
    let git_lock = git::lock_git().await;

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

        // Re-verify the stale-read guard against the file's ACTUAL current
        // content, immediately before the overwrite — this is the re-check
        // #142 was filed over. Step 1's check ran against the caller-supplied
        // `old_content`, which can go stale in the window between then and
        // here (see the GIT_LOCK comment above); this one reads what is
        // really on disk right now, under the lock, right before it gets
        // clobbered. Skipped when the caller passed no `expected_hash`,
        // matching that earlier check's own opt-in contract.
        if let Some(expected) = expected_hash {
            let live_content = tokio::fs::read(&abs_path).await.map_err(|e| {
                error!(
                    "Failed to re-read '{}' for stale-hash re-check: {}",
                    abs_path.display(),
                    e
                );
                WriteError::Io {
                    msg: format!("Failed to read file for stale-hash re-check: {}", e),
                }
            })?;
            let actual = crate::ingest::compute_hash_from_bytes(&live_content);
            if !expected.trim().eq_ignore_ascii_case(&actual) {
                return Err(WriteError::StaleHash {
                    expected: expected.trim().to_string(),
                    actual,
                });
            }
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
    // `git_lock` was acquired back at step 4.5, before the stale-hash re-check
    // and the write itself, and is held continuously through here and any
    // rollback below — NOT re-acquired at this point as it used to be.
    // Releasing and re-acquiring in between would let another writer see — and,
    // since it stages its own path into the same index, commit — the
    // half-staged entry this call is about to undo (and would reopen exactly
    // the #142 race step 4.5 exists to close).
    let commit_outcome = match git::commit_and_sync(
        &git_lock,
        deps.git_url,
        deps.branch,
        data_path_str,
        deps.token,
        &[rel_path],
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
                    Ok(()) => git::unstage(&git_lock, data_path_str, rel_path).await,
                    Err(e) => Err(anyhow::Error::new(e)
                        .context("Failed to remove newly-written file during rollback")),
                }
            } else {
                git::restore_from_head(&git_lock, data_path_str, rel_path).await
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
            deps.queue
                .mark_paths(std::iter::once(PathBuf::from(rel_path)));

            return Ok(WriteSuccess {
                outcome: WriteOutcome::CommittedPendingSync,
                sha,
                rebased_paths: Vec::new(),
                diff: render_unified_diff(old_content, new_content, rel_path),
                sync_failure_cause: Some(format!("{:#}", source)),
                // Not a move — nothing else was rewritten.
                rewritten_paths: Vec::new(),
                // Not a delete — nothing else was checked for inbound links.
                referencing_paths: Vec::new(),
            });
        }
    };

    // Mark this path — and anything the rebase pulled in from other commits —
    // dirty and return immediately. The reindex worker (src/reindex.rs) does the
    // actual chunk/embed/upsert work out of band; this call never blocks on it,
    // which is the whole point — embedding is far slower than a caller's request
    // timeout on a large document.
    deps.queue.mark_paths(
        std::iter::once(PathBuf::from(rel_path))
            .chain(commit_outcome.rebased_paths.iter().cloned()),
    );

    Ok(WriteSuccess {
        outcome: WriteOutcome::Synced,
        sha: commit_outcome.sha,
        diff: render_unified_diff(old_content, new_content, rel_path),
        rebased_paths: commit_outcome.rebased_paths,
        sync_failure_cause: None,
        // Not a move — nothing else was rewritten.
        rewritten_paths: Vec::new(),
        // Not a delete — nothing else was checked for inbound links.
        referencing_paths: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// write_document_move: the MOVE branch of write_document (WriteRequest::dest_path)
// ---------------------------------------------------------------------------

/// The MOVE branch of `write_document`, split out because a move touches TWO
/// paths at every stage that the create/edit path only ever touches one:
/// eligibility, path-safety, schema-frozen, the filesystem mutation itself, the
/// commit, and — the part most worth keeping legible on its own — the rollback.
/// Interleaving that with the single-path create/edit logic above would have
/// made both harder to reason about; keeping it here means the non-move path
/// reads exactly as it did before this existed, and this function's rollback is
/// the only thing you need to hold in your head to convince yourself it is
/// correct.
///
/// Called only from `write_document` when `req.dest_path.is_some()`; implements
/// that field's contract in the order documented there.
async fn write_document_move<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &WriteDeps<'_, E, Q>,
    req: WriteRequest<'_>,
) -> Result<WriteSuccess, WriteError> {
    let WriteRequest {
        rel_path: source_rel,
        old_content,
        new_content,
        is_create,
        message,
        default_verb: _,
        force_new: _,
        operation,
        expected_hash,
        dest_path,
    } = req;
    let dest_rel = dest_path.expect("write_document_move called with req.dest_path == None");

    // 1. A create with a dest_path is a caller bug, not a runtime condition: a
    //    create has no prior file to move FROM. Reported as `Internal` — the same
    //    variant this pipeline uses for other "this should never happen from a
    //    well-behaved caller" states — rather than a user-facing variant.
    if is_create {
        return Err(WriteError::Internal {
            msg: "write_document called with is_create=true and dest_path set; a create \
                  cannot also be a move"
                .to_string(),
        });
    }

    // 2. Eligibility + path-safety for BOTH paths, before anything else — mirrors
    //    write_document's own "0" / "0.5" ordering (see its comments for why this
    //    must run before validation, and in particular before a configured
    //    validation.lint_command exec). A move that is safe to write TO but not
    //    safe to remove FROM (or vice versa) must be rejected before either side
    //    is touched. The resolved paths are discarded here, same as
    //    write_document's early check — each is re-resolved immediately before
    //    the filesystem action that uses it, below.
    check_include_pattern(deps, source_rel)?;
    check_include_pattern(deps, dest_rel)?;
    safe_write_path(deps, source_rel)?;
    safe_write_path(deps, dest_rel)?;

    // 3. Optional stale-read guard against the SOURCE, before anything touches
    //    the filesystem — same contract as write_document's step 1 (see that
    //    comment for the reasoning), just relocated ahead of the existence/
    //    collision checks below so a stale read never even gets to observe
    //    whether the destination is free. `old_content` is hashed exactly as
    //    the non-move path hashes it: callers supply the freshly-read on-disk
    //    SOURCE content, which is what makes the comparison meaningful.
    if let Some(expected) = expected_hash {
        let actual = crate::ingest::compute_hash_from_bytes(old_content.as_bytes());
        if !expected.trim().eq_ignore_ascii_case(&actual) {
            return Err(WriteError::StaleHash {
                expected: expected.trim().to_string(),
                actual,
            });
        }
    }

    // 4. Source must exist.
    let abs_source = safe_write_path(deps, source_rel)?;
    if !abs_source.exists() {
        return Err(WriteError::NotFound);
    }

    // 5. Destination must NOT already exist — a move never overwrites.
    let abs_dest = safe_write_path(deps, dest_rel)?;
    if abs_dest.exists() {
        return Err(WriteError::AlreadyExists);
    }

    // 6. Schema-frozen guard against BOTH paths: removing a file from a frozen
    //    directory mutates that directory's contents exactly as adding one does,
    //    so either side being frozen blocks the whole move.
    let schemas = crate::schema::load_shared(deps.schema_cache);
    if let Some(reason) = schemas.is_frozen(Path::new(source_rel)) {
        return Err(WriteError::Frozen {
            reason: reason.to_string(),
        });
    }
    if let Some(reason) = schemas.is_frozen(Path::new(dest_rel)) {
        return Err(WriteError::Frozen {
            reason: reason.to_string(),
        });
    }

    // 6.5. Outbound-link re-relativization: EVERY relative link inside the
    //    document being moved was authored against wherever it used to live
    //    (`source_rel`'s directory) — not just a link back to itself. Moving
    //    the document changes that base directory, so any such link whose
    //    text doesn't change would silently repoint at a different file once
    //    resolved from the new location (e.g. `../shared/doc.md` from
    //    `old/a.md` means `shared/doc.md`; the same text from `new/deep/a.md`
    //    means something else entirely). This is distinct from step 10.5
    //    below, which rewrites OTHER documents that link INTO this one — this
    //    step fixes the links this document itself contains, which point OUT.
    //
    //    `find_markdown_link_occurrences(new_content, source_rel)` resolves
    //    every occurrence against `source_rel` — the document's OLD
    //    location — which is the correct context regardless of what the link
    //    points at, because that's the directory the link text was actually
    //    written against. Each occurrence's true KB-root-relative target is
    //    therefore `o.resolved`, with one exception: a link whose target IS
    //    `source_rel` is a self-reference, and the document's true target is
    //    no longer `source_rel` — it's `dest_rel`, since that's where the
    //    document now lives. Every occurrence then gets re-relativized from
    //    `dest_rel` (the document's NEW location) to its own target, which is
    //    what keeps it resolving to the same file post-move.
    let content_to_write = rewrite_outbound_links(new_content, source_rel, dest_rel, |resolved| {
        (resolved == source_rel).then(|| dest_rel.to_string())
    });

    // 7. Frontmatter validation against the DESTINATION's resolved schema, NOT
    //    the source's. This is the whole point of a move: the destination
    //    directory may require different frontmatter than the source did.
    //    Validated against `content_to_write` (the self-link rewrite above, if
    //    any) since that is what actually lands at the destination — not the
    //    caller's original `new_content`.
    let schema = schemas.resolve_for(Path::new(dest_rel));
    let (validation_result, _validated) = validate::validate_content(
        Path::new(dest_rel),
        &content_to_write,
        schema,
        deps.validation,
    )
    .await
    .map_err(|e| {
        error!(
            "Validation error moving '{}' -> '{}': {:#}",
            source_rel, dest_rel, e
        );
        WriteError::Io {
            msg: format!("Failed to validate content: {}", e),
        }
    })?;

    if !validation_result.valid {
        return Err(WriteError::Validation {
            result: validation_result,
        });
    }

    // 8. No dedup gate: a move is not a create, and the document's own content
    //    would trivially self-match its pre-move copy anyway.

    // 9. Validate the commit message BEFORE touching the filesystem — same reason
    //    as write_document/delete_document: rejecting it after a mutation would
    //    leave that mutation uncommitted.
    validate_commit_message(message)?;

    // 9.5. Acquire GIT_LOCK now and hold ONE guard across every remaining
    //    mutation of the clone below — the destination write, the source
    //    removal, the referencing-document read-modify-write (step 10.5), the
    //    commit, and any rollback — rather than acquiring it only just before
    //    the commit as this used to. Two reasons, both load-bearing:
    //
    //    - Step 10.5 reads another document's CURRENT body, computes a
    //      rewrite, and writes it back with no hash/staleness check of its
    //      own (unlike a caller-driven edit, which can supply
    //      `expected_hash`). Left unlocked, a concurrent writer to that same
    //      referencing document can land its own write in the gap between
    //      this read and this write and be silently clobbered with no
    //      conflict reported to either side. Holding GIT_LOCK across the
    //      whole read-modify-write serializes it against every other
    //      operation that also mutates the clone through this lock,
    //      including another concurrent move.
    //    - The destination write and source removal immediately below are
    //      themselves mutations of the clone, and the SAME reasoning that
    //      motivates locking step 10.5 applies to them: leaving them outside
    //      the acquisition would still let a concurrent webhook merge or
    //      another write's commit interleave with an in-progress, not-yet-
    //      committed move, and would reintroduce exactly the kind of
    //      "acquire, release, re-acquire" gap CLAUDE.md's git-serialization
    //      section warns against. Pulling them in also matches this
    //      codebase's existing convention of one acquisition per logical
    //      mutating sequence (see `write_document`'s and `delete_document`'s
    //      identical single acquisition spanning their own commit+rollback).
    //      Nothing between here and the commit below performs a SECOND
    //      `lock_git()` call — every helper that used to acquire its own
    //      (the `cleanup_lock` below, formerly a fresh acquisition) now takes
    //      this same guard by reference instead, which is what keeps this
    //      non-reentrant mutex from deadlocking against itself.
    let git_lock = git::lock_git().await;

    // 9.6. Re-verify the stale-read guard against the SOURCE's live on-disk
    //    content, now that GIT_LOCK is held and nothing else can touch the
    //    clone for the rest of this call. Step 3's check ran before
    //    `validate::validate_content` above — which can exec an arbitrarily
    //    slow `validation.lint_command` — and before this lock acquisition,
    //    so a webhook merge (which independently needs this same lock for its
    //    own fetch + `git merge --ff-only`) could have changed the source's
    //    content in that window with nothing to detect it; re-checking the
    //    already-stale `old_content` a second time could never catch that.
    //    Reads `abs_source` resolved back at step 4 — nothing between there
    //    and here can have made it unsafe, since no filesystem mutation of
    //    the clone has happened yet. Skipped when the caller passed no
    //    `expected_hash`, matching step 3's own opt-in contract.
    if let Some(expected) = expected_hash {
        let live_source = tokio::fs::read(&abs_source).await.map_err(|e| {
            error!(
                "Failed to re-read source '{}' for stale-hash re-check while moving to '{}': {}",
                source_rel, dest_rel, e
            );
            WriteError::Io {
                msg: format!("Failed to read source file for stale-hash re-check: {}", e),
            }
        })?;
        let actual = crate::ingest::compute_hash_from_bytes(&live_source);
        if !expected.trim().eq_ignore_ascii_case(&actual) {
            return Err(WriteError::StaleHash {
                expected: expected.trim().to_string(),
                actual,
            });
        }
    }

    // 10. Filesystem: write the DESTINATION first (`create_new`, so this can never
    //    silently clobber a file that appeared between the check above and now),
    //    THEN remove the source. This order is load-bearing, not arbitrary:
    //    - If the destination write fails, nothing has happened yet — the source
    //      is exactly as it was.
    //    - If the SOURCE removal fails afterward, the content still exists (at
    //      the destination) — recoverable by deleting that destination copy and
    //      reporting failure, which is exactly what happens below.
    //    The reverse order (remove source, then write destination) has no such
    //    recovery: a crash or failure between the two would delete the document
    //    from disk with no copy anywhere, for real user content. Write-then-remove
    //    is the only ordering where every failure point still has a path back to
    //    "nothing lost".
    let abs_dest = safe_write_path(deps, dest_rel)?;
    if let Some(parent) = abs_dest.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|e| {
            error!(
                "Failed to create parent directories for '{}': {}",
                abs_dest.display(),
                e
            );
            WriteError::Io {
                msg: format!("Failed to create parent directories: {}", e),
            }
        })?;
    }

    {
        use tokio::io::AsyncWriteExt as _;
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&abs_dest)
            .await
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists {
                    WriteError::AlreadyExists
                } else {
                    error!("Failed to create file '{}': {}", abs_dest.display(), e);
                    WriteError::Io {
                        msg: format!("Failed to create file: {}", e),
                    }
                }
            })?;
        file.write_all(content_to_write.as_bytes())
            .await
            .map_err(|e| {
                error!("Failed to write file '{}': {}", abs_dest.display(), e);
                WriteError::Io {
                    msg: format!("Failed to write file: {}", e),
                }
            })?;
    }

    // Re-verify the source immediately before removing it — same TOCTOU
    // reasoning as write_document's own re-resolve-before-mutation pattern.
    let abs_source = safe_write_path(deps, source_rel)?;
    if let Err(e) = tokio::fs::remove_file(&abs_source).await {
        error!(
            "Failed to remove source '{}' while moving it to '{}'; deleting the destination \
             copy so the move leaves nothing behind: {}",
            source_rel, dest_rel, e
        );
        // The destination write above already landed on disk with no git change
        // yet to make it durable — undo it fully rather than leave the document
        // sitting at two paths at once.
        if let Err(cleanup_err) = tokio::fs::remove_file(&abs_dest).await {
            error!(
                "Failed to clean up destination '{}' after a failed source removal during a \
                 move: {}. The document now exists at BOTH '{}' and '{}' — this needs operator \
                 attention.",
                dest_rel, cleanup_err, source_rel, dest_rel
            );
        }
        return Err(WriteError::Io {
            msg: format!("Failed to remove source file during move: {}", e),
        });
    }

    let data_path_str = deps.canonical_data_path.to_str().unwrap_or_default();

    // 10.5. Rewrite OTHER documents whose body links to the SOURCE path, so
    //    those links keep resolving after the move — riding along in the SAME
    //    commit as the move itself (the path slice below includes every path
    //    rewritten here). Runs after the destination/source filesystem work
    //    above (the move is definitely proceeding by this point — every
    //    validation gate has already passed) and before anything touches git,
    //    so a failure partway through can still be undone by hand (see the
    //    write failure branch below) rather than racing a rollback against a
    //    half-committed state.
    //
    //    Frozen directories: deliberately NOT checked here. `schemas.is_frozen`
    //    guards frontmatter/content changes to a directory's document set — a
    //    link-text rewrite touches neither; it edits prose inside the Markdown
    //    body of a document that already exists and was already valid. Treating
    //    "the referencing document happens to live under a frozen directory" as
    //    a reason to fail the WHOLE MOVE would be surprising (the caller asked
    //    to move one document, not to write into the frozen one) and gains
    //    nothing safety-wise, so frozen referencing documents are rewritten
    //    exactly like any other.
    //
    //    Best-effort against the reverse-link index itself: if there is no
    //    `StateDb` (`deps.state == None` — see that field's doc comment) or the
    //    query fails outright, link rewriting is simply skipped; the move still
    //    proceeds. Every referencing document's `document_links` rows also
    //    self-heal on that document's own next reindex regardless of what
    //    happens here.
    let mut rewritten_paths: Vec<String> = Vec::new();
    if let Some(state) = deps.state {
        match state.links_targeting(source_rel, "markdown").await {
            Ok(referencing_paths) => {
                for ref_path in referencing_paths {
                    // The source linking to itself is the self-reference case
                    // handled above as part of `content_to_write` — it is not a
                    // separate file to rewrite, and must not be processed twice.
                    if ref_path == source_rel {
                        continue;
                    }

                    let abs_ref = match safe_write_path(deps, &ref_path) {
                        Ok(p) => p,
                        Err(_) => {
                            warn!(
                                "Skipping link rewrite in '{}' while moving '{}' -> '{}': the \
                                 path no longer resolves safely (stale document_links row?)",
                                ref_path, source_rel, dest_rel
                            );
                            continue;
                        }
                    };
                    let body = match tokio::fs::read_to_string(&abs_ref).await {
                        Ok(b) => b,
                        Err(e) => {
                            // Stale `document_links` row: the referencing document no
                            // longer exists on disk (or isn't readable). Not this
                            // move's problem to fix — skip it rather than fail the
                            // move over another document's already-broken state.
                            warn!(
                                "Skipping link rewrite in '{}' while moving '{}' -> '{}': \
                                 failed to read it, likely a stale document_links row: {}",
                                ref_path, source_rel, dest_rel, e
                            );
                            continue;
                        }
                    };
                    let occurrences: Vec<_> = find_all_markdown_link_occurrences(&body, &ref_path)
                        .into_iter()
                        .filter(|o| o.resolved.as_str() == source_rel)
                        .collect();
                    if occurrences.is_empty() {
                        // Stale row again: `document_links` says this document links
                        // to the source, but nothing in its CURRENT body actually
                        // resolves there anymore. Skip — writing it back unchanged
                        // would put a no-op entry in the move's commit.
                        continue;
                    }

                    let replacement = crate::ingest::relativize_md_path(&ref_path, dest_rel);
                    let new_body = apply_link_replacements(&body, &occurrences, &replacement);
                    if let Err(e) = tokio::fs::write(&abs_ref, new_body.as_bytes()).await {
                        error!(
                            "Failed to rewrite links into '{}' while moving '{}' -> '{}': {}. \
                             Undoing every filesystem change made for this move so far.",
                            ref_path, source_rel, dest_rel, e
                        );
                        // Nothing has touched git yet at this point (no `git add`, no
                        // commit) — every path involved is either still tracked at
                        // HEAD (the source, and every referencing document already
                        // rewritten this loop) or brand new and untracked (the
                        // destination), so unwinding by hand is safe: restore the
                        // tracked ones from HEAD, delete the untracked one. Reuses the
                        // `git_lock` acquired in step 9.5 above rather than acquiring a
                        // second guard — this non-reentrant mutex is already held for
                        // this entire sequence, and a fresh `lock_git()` call here would
                        // deadlock against it.
                        for done in &rewritten_paths {
                            if let Err(e) =
                                git::restore_from_head(&git_lock, data_path_str, done).await
                            {
                                error!(
                                    "Rollback: failed to restore rewritten referencing \
                                     document '{}': {:#}. This needs operator attention.",
                                    done, e
                                );
                            }
                        }
                        if let Err(e) =
                            git::restore_from_head(&git_lock, data_path_str, source_rel).await
                        {
                            error!(
                                "Rollback: failed to restore source '{}': {:#}. This needs \
                                 operator attention.",
                                source_rel, e
                            );
                        }
                        if let Err(e) = tokio::fs::remove_file(&abs_dest).await {
                            error!(
                                "Rollback: failed to remove destination '{}': {}. This needs \
                                 operator attention.",
                                dest_rel, e
                            );
                        }
                        return Err(WriteError::Io {
                            msg: format!("Failed to rewrite links in '{}': {}", ref_path, e),
                        });
                    }
                    rewritten_paths.push(ref_path);
                }
            }
            Err(e) => {
                warn!(
                    "Skipping incoming-link rewrite while moving '{}' -> '{}': the \
                     reverse-link query failed: {:#}",
                    source_rel, dest_rel, e
                );
            }
        }
    }

    let commit_message = build_commit_message(
        message,
        &format!("docs: move {} to {}", source_rel, dest_rel),
        operation,
    );

    // 11. Commit the move AND every rewritten referencing document as ONE
    //     atomic commit, under the SAME lock acquisition (step 9.5, above)
    //     already held across the destination write, source removal, and
    //     referencing-document rewrite — releasing it in between any of those
    //     and the commit would let another writer stage into (and, since it
    //     commits its own path, commit) the very half-staged state this call
    //     is about to undo. See write_document's identical comment for the
    //     full reasoning.
    //     Deduplicated defensively even though `links_targeting` already
    //     returns DISTINCT source paths and cannot return `source_rel`/
    //     `dest_rel` themselves (dest_rel is guaranteed not to have existed as
    //     a prior document, and source_rel is filtered out above).
    let mut commit_paths: Vec<&str> = vec![source_rel, dest_rel];
    for p in &rewritten_paths {
        if !commit_paths.contains(&p.as_str()) {
            commit_paths.push(p.as_str());
        }
    }

    let commit_outcome = match git::commit_and_sync(
        &git_lock,
        deps.git_url,
        deps.branch,
        data_path_str,
        deps.token,
        &commit_paths,
        &commit_message,
        deps.commit_author_name,
        deps.commit_author_email,
    )
    .await
    {
        Ok(outcome) => outcome,

        Err(git::CommitSyncError::PreCommit(source_err)) => {
            error!(
                "commit_and_sync pre-commit failure moving '{}' -> '{}', rolling back both \
                 halves and {} rewritten referencing document(s): {:#}",
                source_rel,
                dest_rel,
                rewritten_paths.len(),
                source_err
            );

            // 12. Roll back EVERY part of this move — both halves of the move
            //     itself, plus every referencing document rewritten above.
            //     Each is independent of the others and ALL of them always run
            //     unconditionally, so a failure in one never leaves a
            //     recoverable part undone:
            //     - the source was already tracked at HEAD before this call
            //       (this is a move, not a create), so `restore_from_head` puts
            //       its content — and un-stages whatever `git add`/removal
            //       staged for it — back in one step, exactly like
            //       write_document's edit rollback.
            //     - the destination has no HEAD content (it is new), so it is
            //       rolled back exactly like write_document's create rollback:
            //       remove the file, then unstage whatever `git add` staged
            //       for it.
            //     - every rewritten referencing document is, like the source,
            //       pre-existing and tracked at HEAD, so `restore_from_head`
            //       reverts its link-text edit the same way it reverts the
            //       source's content. Getting this third group right is the
            //       whole point of this rollback being careful: a bug here
            //       corrupts a user's OTHER, unrelated documents — not just
            //       the one being moved — so `rolled_back` is true only if
            //       ALL THREE groups succeed, not just the two the move
            //       itself touches.
            let source_restore = git::restore_from_head(&git_lock, data_path_str, source_rel).await;
            let dest_rollback = match tokio::fs::remove_file(&abs_dest).await {
                Ok(()) => git::unstage(&git_lock, data_path_str, dest_rel).await,
                Err(e) => Err(anyhow::Error::new(e)
                    .context("Failed to remove the new destination file during rollback")),
            };
            let mut rewrite_restore_failures: Vec<(String, anyhow::Error)> = Vec::new();
            for path in &rewritten_paths {
                if let Err(e) = git::restore_from_head(&git_lock, data_path_str, path).await {
                    rewrite_restore_failures.push((path.clone(), e));
                }
            }

            // All three groups must succeed for the rollback to be considered
            // clean — if any failed, filesystem and git state are inconsistent
            // with each other and with HEAD, which needs operator attention
            // rather than a blind retry (mirrors `PreCommitFailed::rolled_back`'s
            // contract on the non-move path).
            let rolled_back = source_restore.is_ok()
                && dest_rollback.is_ok()
                && rewrite_restore_failures.is_empty();
            if !rolled_back {
                error!(
                    "Rollback FAILED after a pre-commit git failure moving '{}' -> '{}'. Source \
                     restore: {:?}. Destination rollback: {:?}. Rewritten-document restore \
                     failures: {:?}. Original cause: {:#}. Filesystem and git state may now be \
                     inconsistent.",
                    source_rel,
                    dest_rel,
                    source_restore,
                    dest_rollback,
                    rewrite_restore_failures,
                    source_err
                );
            }

            let mut msg = format!("{:#}", source_err);
            if let Err(e) = &source_restore {
                msg.push_str(&format!(". Source restore cause: {:#}", e));
            }
            if let Err(e) = &dest_rollback {
                msg.push_str(&format!(". Destination rollback cause: {:#}", e));
            }
            for (path, e) in &rewrite_restore_failures {
                msg.push_str(&format!(". Restore of '{}' cause: {:#}", path, e));
            }

            return Err(WriteError::PreCommitFailed { rolled_back, msg });
        }

        Err(git::CommitSyncError::PostCommit {
            sha,
            source: source_err,
        }) => {
            warn!(
                "commit_and_sync post-commit (sync) failure moving '{}' -> '{}', commit {} \
                 stands uncorrected: {:#}",
                source_rel, dest_rel, sha, source_err
            );

            // 13. The commit is real and durable — both halves of the move, and
            //     every rewritten referencing document, already happened as far
            //     as local git history is concerned, so this is left alone (not
            //     rolled back) and reported as sync-pending, same as every other
            //     post-commit failure in this pipeline. `rebased_paths` is empty
            //     for the same reason as elsewhere: the rebase never ran.
            deps.queue.mark_paths(
                [PathBuf::from(source_rel), PathBuf::from(dest_rel)]
                    .into_iter()
                    .chain(rewritten_paths.iter().map(PathBuf::from)),
            );

            return Ok(WriteSuccess {
                outcome: WriteOutcome::CommittedPendingSync,
                sha,
                rebased_paths: Vec::new(),
                diff: render_unified_diff(old_content, &content_to_write, dest_rel),
                sync_failure_cause: Some(format!("{:#}", source_err)),
                rewritten_paths,
                // Not a delete — nothing else was checked for inbound links.
                referencing_paths: Vec::new(),
            });
        }
    };

    // 14. Mark the source, the destination, and every rewritten referencing
    //     document dirty — plus anything the rebase pulled in — in the SAME
    //     marking call. `ingest::index_paths` purges the now-missing source,
    //     indexes the new destination, and re-chunks/re-embeds each rewritten
    //     document (whose `document_links` rows self-heal from its new body in
    //     the same pass); all of them need to be in the same worklist for the
    //     worker to do that in one sweep.
    deps.queue.mark_paths(
        [PathBuf::from(source_rel), PathBuf::from(dest_rel)]
            .into_iter()
            .chain(rewritten_paths.iter().map(PathBuf::from))
            .chain(commit_outcome.rebased_paths.iter().cloned()),
    );

    Ok(WriteSuccess {
        outcome: WriteOutcome::Synced,
        sha: commit_outcome.sha,
        diff: render_unified_diff(old_content, &content_to_write, dest_rel),
        rebased_paths: commit_outcome.rebased_paths,
        sync_failure_cause: None,
        rewritten_paths,
        // Not a delete — nothing else was checked for inbound links.
        referencing_paths: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// Wiki pipe-alias link support (`[[target|Display Text]]`) — fix #131
// ---------------------------------------------------------------------------
//
// `ingest::resolve_link_target` deliberately REJECTS any wiki-style `[[...]]`
// target containing a `|` (see that function's doc comment and its
// `kind == RawLinkKind::Wiki && target.contains('|')` guard): the pipe-alias
// form is not parsed apart from its target at all today, so
// `ingest::extract_markdown_links`/`ingest::find_markdown_link_occurrences`
// never produce an occurrence for `[[old/path|Display Text]]` — the KB's link
// graph doesn't know the edge exists, and this module's move-time rewriter,
// which is built entirely on top of `find_markdown_link_occurrences`, never
// sees these links either. A move therefore leaves a pipe-alias link pointing
// at the moved document's old path with no error.
//
// A full fix belongs in `ingest.rs` (out of scope for this change): teach
// `resolve_link_target`/`scan_line_constructs` to split `[[target|alias]]`
// into target and alias instead of rejecting it outright, so
// `find_markdown_link_occurrences` returns an occurrence spanning just the
// target portion (alias preserved), and `extract_markdown_links` records the
// edge in `document_links` the same way it does for a bare `[[target]]`.
// Until then, the functions below duplicate just enough of
// `ingest::resolve_link_target`/`resolve_relative_md_path`'s judging rules
// (trim, strip a trailing `#fragment`, reject external/absolute targets,
// default a missing extension to `.md`, resolve `./`/`../` against the
// containing document's directory) to find and correctly resolve pipe-alias
// occurrences from THIS module, so at least everything the rewriter itself
// controls — a moved document's own outbound links (self-reference included,
// via `rewrite_outbound_links` below) and any referencing document that gets
// visited because `StateDb::links_targeting` already returns it (e.g. it also
// has a non-alias link to the same target) — gets a correctly rewritten
// pipe-alias link with its alias preserved exactly. What this CANNOT fix: a
// referencing document whose ONLY link to the moved target is a pipe-alias
// link is never visited at all, because `document_links` (built from
// `ingest::extract_markdown_links`) has no edge for it to be found by in the
// first place. That case needs the `ingest.rs` change described above.

/// Find every wiki pipe-alias link (`[[target|alias]]`) in `body` whose target
/// resolves against `source_rel_path`, shaped exactly like an
/// `ingest::LinkOccurrence` — `span` covers ONLY the target portion (not the
/// `|alias` suffix or the surrounding `]]`), so replacing it in place leaves
/// the alias untouched, and `resolved`/`raw` follow the same contract
/// `ingest::find_markdown_link_occurrences` uses. See this section's doc
/// comment for why this exists as a separate scan rather than a case
/// `ingest::find_markdown_link_occurrences` already covers.
///
/// Line-oriented fence tracking mirrors `ingest`'s private `scan_link_occurrences`
/// (`` ``` ``/`~~~` toggles a fence, and nothing inside one is scanned) so the two
/// scanners never disagree about what counts as "inside a fence".
fn find_pipe_alias_link_occurrences(
    body: &str,
    source_rel_path: &str,
) -> Vec<crate::ingest::LinkOccurrence> {
    let mut out = Vec::new();
    let mut in_fence = false;
    let mut offset = 0usize;

    for line in body.split_inclusive('\n') {
        let content = line.strip_suffix('\n').unwrap_or(line);
        let content = content.strip_suffix('\r').unwrap_or(content);

        let trimmed = content.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            offset += line.len();
            continue;
        }

        if !in_fence {
            for (span, raw_target) in scan_pipe_alias_wiki_links(content) {
                if let Some(resolved) = resolve_pipe_alias_wiki_target(&raw_target, source_rel_path)
                {
                    out.push(crate::ingest::LinkOccurrence {
                        span: offset + span.start..offset + span.end,
                        raw: raw_target,
                        resolved,
                    });
                }
            }
        }

        offset += line.len();
    }

    out
}

/// Find every `[[target|alias]]` construct on one already-fence-filtered line,
/// honoring inline code spans and images the same way `ingest`'s private
/// `scan_line_constructs` does for the syntaxes it recognizes (a
/// backtick-quoted `` `[[foo|bar]]` `` is literal text, not a link;
/// `![[foo|bar]]` is image-shaped and skipped). Returns the byte span of the
/// TARGET portion only (not the alias) alongside the raw target text, for
/// every bracketed pair whose content contains a `|` — a bare `[[target]]`
/// with no alias is left for `ingest::find_markdown_link_occurrences` to find,
/// since this function only exists to cover what that one cannot yet see.
fn scan_pipe_alias_wiki_links(line: &str) -> Vec<(std::ops::Range<usize>, String)> {
    let chars: Vec<char> = line.chars().collect();
    let mut byte_at: Vec<usize> = line.char_indices().map(|(b, _)| b).collect();
    byte_at.push(line.len());

    let mut out = Vec::new();
    let mut i = 0usize;
    let mut in_code = false;
    let mut code_run_len = 0usize;

    while i < chars.len() {
        if chars[i] == '`' {
            let start = i;
            while i < chars.len() && chars[i] == '`' {
                i += 1;
            }
            let run_len = i - start;
            if !in_code {
                in_code = true;
                code_run_len = run_len;
            } else if run_len == code_run_len {
                in_code = false;
            }
            continue;
        }

        if in_code {
            i += 1;
            continue;
        }

        // `![[...]]` is image-shaped, not a document link — skip past its own
        // `[[` so the branch below never treats it as a wiki link.
        if chars[i] == '!' && chars.get(i + 1) == Some(&'[') && chars.get(i + 2) == Some(&'[') {
            i += 2;
            continue;
        }

        if chars[i] == '['
            && chars.get(i + 1) == Some(&'[')
            && let Some(close) = find_double_bracket_close_for_pipe_alias(&chars, i)
        {
            let (content_start, content_end) = (i + 2, close);
            let inner: String = chars[content_start..content_end].iter().collect();
            if let Some(pipe_idx) = inner.find('|') {
                let raw_target = inner[..pipe_idx].to_string();
                // `pipe_idx` is a byte offset into `inner`, valid to slice on since
                // `|` is single-byte ASCII — counting chars up to it gives the char
                // offset needed to remap through `byte_at` for the line's real span.
                let target_char_len = inner[..pipe_idx].chars().count();
                let target_end_char = content_start + target_char_len;
                out.push((byte_at[content_start]..byte_at[target_end_char], raw_target));
            }
            i = close + 2;
            continue;
        }

        i += 1;
    }

    out
}

/// Find the index of the first `]` of a closing `]]` for a wiki link opened at
/// `chars[open] == chars[open + 1] == '['`. Same no-nested-brackets
/// simplification as `ingest`'s private `find_double_bracket_close`, which
/// this mirrors (that one is not `pub(crate)`, so it cannot be called
/// directly from here).
fn find_double_bracket_close_for_pipe_alias(chars: &[char], open: usize) -> Option<usize> {
    let mut j = open + 2;
    while j + 1 < chars.len() {
        if chars[j] == ']' && chars[j + 1] == ']' {
            return Some(j);
        }
        j += 1;
    }
    None
}

/// Resolve a pipe-alias wiki link's target exactly the way
/// `ingest::resolve_link_target` would for an ordinary `[[target]]` (wiki
/// kind): trim, strip a trailing `#fragment`, reject external/absolute
/// targets, default a missing extension to `.md`, then resolve `./`/`../`
/// against `source_rel_path`'s directory. Duplicated here only because
/// `ingest::resolve_link_target`/`resolve_relative_md_path` are private
/// `fn`s, not `pub(crate)` — this mirrors their judging rules, not a
/// divergent policy. See this section's doc comment for why this
/// duplication exists and what would let it be deleted.
fn resolve_pipe_alias_wiki_target(raw_target: &str, source_rel_path: &str) -> Option<String> {
    let target = raw_target.trim();
    let target = match target.find('#') {
        Some(idx) => &target[..idx],
        None => target,
    };
    if target.is_empty() {
        return None;
    }

    let lower = target.to_ascii_lowercase();
    if lower.starts_with("http://")
        || lower.starts_with("https://")
        || lower.starts_with("mailto:")
        || target.starts_with('/')
        || target.contains("://")
    {
        return None;
    }

    let with_ext: String;
    let target = if target.ends_with(".md") {
        target
    } else {
        with_ext = format!("{target}.md");
        with_ext.as_str()
    };

    resolve_relative_md_path_for_pipe_alias(target, source_rel_path)
}

/// Join `target` onto `source_rel_path`'s directory and normalize
/// component-by-component. Mirrors `ingest`'s private
/// `resolve_relative_md_path` byte-for-byte (see
/// `resolve_pipe_alias_wiki_target`'s doc comment for why it is duplicated
/// rather than called directly) — a `..` climbing above the knowledge-base
/// root rejects the whole target, same as there.
fn resolve_relative_md_path_for_pipe_alias(target: &str, source_rel_path: &str) -> Option<String> {
    let mut stack: Vec<&str> = source_rel_path
        .rsplit_once('/')
        .map(|(dir, _)| dir.split('/').filter(|c| !c.is_empty()).collect())
        .unwrap_or_default();

    for comp in target.split('/') {
        match comp {
            "" | "." => {}
            ".." => {
                stack.pop()?;
            }
            other => stack.push(other),
        }
    }

    if stack.is_empty() {
        None
    } else {
        Some(stack.join("/"))
    }
}

/// Every recognized outbound Markdown link occurrence in `body`, resolved
/// against `source_rel_path` — `ingest::find_markdown_link_occurrences` (every
/// syntax it recognizes) UNIONED with [`find_pipe_alias_link_occurrences`]
/// (the one syntax it doesn't yet: `[[target|alias]]` — fix #131). Both scans
/// produce the same `ingest::LinkOccurrence` shape, so every existing
/// consumer (`apply_link_replacements`/`apply_link_replacements_each`) handles
/// the merged list with no changes of its own.
fn find_all_markdown_link_occurrences(
    body: &str,
    source_rel_path: &str,
) -> Vec<crate::ingest::LinkOccurrence> {
    let mut occurrences = crate::ingest::find_markdown_link_occurrences(body, source_rel_path);
    occurrences.extend(find_pipe_alias_link_occurrences(body, source_rel_path));
    occurrences
}

/// Rewrite every outbound Markdown link occurrence in `content` — a document being
/// relocated from `old_rel` to `new_rel` — so each one keeps resolving to its
/// intended target once the document lives at its new location.
///
/// `find_markdown_link_occurrences(content, old_rel)` resolves every occurrence
/// against `old_rel` — the document's OLD location, which is the directory the link
/// text was actually authored against, regardless of what a given link points at.
/// For each occurrence, `translate(resolved)` decides that occurrence's TRUE target
/// after the move: `Some(new_target)` when the linked document is ITSELF moving in
/// lockstep with this one (its own new path), `None` when it is staying exactly
/// where it is (the target is unchanged; only the relative spelling needs to
/// change because the mover's own directory changed). Either way, the final
/// replacement text is `relativize_md_path(new_rel, true_target)`.
///
/// Shared by two callers with different `translate` closures:
/// - `write_document_move`, whose closure maps only a self-reference
///   (`resolved == old_rel`) to `new_rel` and nothing else — a single document has
///   no OTHER document moving alongside it.
/// - `move_directory`, whose closure is backed by the whole batch's old→new map, so
///   a link between two documents that are BOTH moving in the same directory move
///   keeps pointing at each other post-move (see that function's doc comment).
fn rewrite_outbound_links(
    content: &str,
    old_rel: &str,
    new_rel: &str,
    translate: impl Fn(&str) -> Option<String>,
) -> String {
    let occurrences = find_all_markdown_link_occurrences(content, old_rel);
    if occurrences.is_empty() {
        return content.to_string();
    }
    let replacements: Vec<(crate::ingest::LinkOccurrence, String)> = occurrences
        .into_iter()
        .map(|o| {
            let true_target = translate(&o.resolved).unwrap_or_else(|| o.resolved.clone());
            let replacement = crate::ingest::relativize_md_path(new_rel, &true_target);
            (o, replacement)
        })
        .collect();
    apply_link_replacements_each(content, &replacements)
}

/// Apply a PER-OCCURRENCE text replacement at each occurrence's own span,
/// back-to-front by span start so an earlier edit's byte-length change can
/// never invalidate a later span still waiting to be applied. Every
/// occurrence must have come from scanning `body` itself (e.g.
/// `ingest::find_markdown_link_occurrences`) — a span from a different string
/// is undefined behavior for `String::replace_range` (it may panic on a
/// non-char-boundary, or silently replace the wrong bytes).
///
/// This is the ONE place that owns the back-to-front span-ordering rule —
/// [`apply_link_replacements`] is a thin single-replacement wrapper over this
/// function rather than a second copy of the sort, so the two rewrite sites
/// below can never drift apart on ordering.
///
/// Used by [`rewrite_outbound_links`] (each link in a moved document's content
/// can resolve to a DIFFERENT target, so each occurrence needs its own
/// re-relativized replacement text) and directly by `move_directory`'s
/// outside-referencing-document rewrite, where one document can reference
/// SEVERAL different moved documents, each needing its own replacement text —
/// unlike `apply_link_replacements` below, where every occurrence shares one.
fn apply_link_replacements_each(
    body: &str,
    replacements: &[(crate::ingest::LinkOccurrence, String)],
) -> String {
    let mut spans: Vec<(std::ops::Range<usize>, &str)> = replacements
        .iter()
        .map(|(o, r)| (o.span.clone(), r.as_str()))
        .collect();
    // Sort back-to-front (descending by start) so replacing an earlier-in-text
    // span never shifts the byte offsets a later-in-iteration-but-earlier-in-text
    // span still needs.
    spans.sort_by_key(|(span, _)| std::cmp::Reverse(span.start));

    let mut out = body.to_string();
    for (span, replacement) in spans {
        out.replace_range(span, replacement);
    }
    out
}

/// Single-replacement convenience wrapper over
/// [`apply_link_replacements_each`], for the common case where every
/// occurrence gets the SAME replacement text.
///
/// Used by `write_document_move`'s referencing-document rewrite (step
/// 10.5): every occurrence found there resolves to the same moved source
/// path, so they all become the same relativized destination text.
fn apply_link_replacements(
    body: &str,
    occurrences: &[crate::ingest::LinkOccurrence],
    replacement: &str,
) -> String {
    let paired: Vec<(crate::ingest::LinkOccurrence, String)> = occurrences
        .iter()
        .cloned()
        .map(|o| (o, replacement.to_string()))
        .collect();
    apply_link_replacements_each(body, &paired)
}

// ---------------------------------------------------------------------------
// move_directory: atomic relocation of every document under a source prefix
// ---------------------------------------------------------------------------

/// A successful [`move_directory`] call.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DirectoryMoveSuccess {
    pub outcome: WriteOutcome,
    pub sha: String,
    pub rebased_paths: Vec<PathBuf>,
    /// `(old_rel, new_rel)` for every document AND schema file moved (a
    /// `.kb-schema.yaml` found under the source subtree moves along with the
    /// documents it governs — see `DirectoryMoveError::BrokenSchemaInSource`'s
    /// doc comment), sorted by `old_rel`.
    pub moved: Vec<(String, String)>,
    /// Documents OUTSIDE the moved subtree whose inline links were rewritten to
    /// point at a moved document's new location, and which rode along in the same
    /// commit. Never includes a document that was itself moved — those are
    /// reported in `moved` instead. Empty when `WriteDeps::state` is `None` or
    /// nothing outside the subtree referenced it.
    pub rewritten_paths: Vec<String>,
    /// Present only when `outcome == CommittedPendingSync`: see
    /// `WriteSuccess::sync_failure_cause`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sync_failure_cause: Option<String>,
}

/// Every structured failure mode of [`move_directory`]. Mirrors
/// [`WriteError`]'s split of caller-facing vs. operator-facing variants — see
/// that enum's doc comments for the reasoning behind each shape reused here.
#[derive(Debug)]
pub enum DirectoryMoveError {
    /// `source_dir` does not exist, is not a directory, or contains no document
    /// matching the configured include patterns — there is nothing to move.
    SourceEmpty {
        msg: String,
    },
    /// At least one file already lives under `dest_dir` — a directory move never
    /// merges into, or overwrites, an existing prefix.
    AlreadyExists,
    /// A `.kb-schema.yaml` somewhere under the source subtree, at `path`, failed
    /// to parse — `reason` is `SchemaCache::is_frozen`'s message for its
    /// governing directory.
    ///
    /// This mirrors `is_frozen`'s own "rules unreadable ⇒ don't touch" stance:
    /// moving documents governed by rules this process cannot read is exactly
    /// the situation where it cannot verify the move is safe, so it refuses
    /// outright rather than silently carrying the parse failure through into
    /// the destination. A schema file that DOES parse is not blocked — it moves
    /// with the documents it governs, and they are validated against the
    /// cascade that results (see [`SchemaCache::with_remapped_scopes`]).
    BrokenSchemaInSource {
        path: String,
        reason: String,
    },
    /// The schema governing a source or destination document's directory failed
    /// to parse (`SchemaCache::is_frozen`).
    Frozen {
        reason: String,
    },
    /// Frontmatter validation against the DESTINATION's schema cascade failed
    /// for one or more documents. Carries every failure, not just the first —
    /// the whole move is all-or-nothing, so a caller needs to know everything
    /// that would need fixing, not just whichever document happened to be
    /// checked first.
    Validation {
        failures: Vec<(String, ValidationResult)>,
        /// `(old_rel, new_rel)` for every `.kb-schema.yaml` this move is
        /// relocating — empty when the source subtree carries no schema file
        /// of its own. Non-empty means the destination cascade these failures
        /// were checked against is not just "whatever already governed the
        /// destination" but a genuinely NEW cascade, re-parented by this very
        /// move — see `SchemaCache::with_remapped_scopes`. Lets the caller-
        /// facing error name that explicitly instead of leaving a document
        /// that was valid moments ago looking like an unexplained failure.
        moved_schema_files: Vec<(String, String)>,
    },
    UnsafePath {
        msg: String,
    },
    Internal {
        msg: String,
    },
    InvalidCommitMessage {
        reason: String,
    },
    /// `git add`/`git commit` failed. See `WriteError::PreCommitFailed`'s doc
    /// comment for the `rolled_back` contract — identical here, just scaled to
    /// every path this move touched: `true` only if every document's source
    /// restore, every document's destination removal, and every rewritten
    /// referencing document's restore all succeeded.
    PreCommitFailed {
        rolled_back: bool,
        msg: String,
    },
    Io {
        msg: String,
    },
}

/// Maps [`safe_write_path`]/[`check_include_pattern`]/[`validate_commit_message`]
/// failures onto [`DirectoryMoveError`], so [`move_directory`] can reuse those
/// helpers with `?` instead of duplicating their logic. In practice only the
/// first four arms are ever produced by those three call sites — the fallback
/// exists purely to keep this conversion exhaustive against `WriteError`'s full
/// variant set, which those helpers' return types do not restrict.
impl From<WriteError> for DirectoryMoveError {
    fn from(err: WriteError) -> Self {
        match err {
            WriteError::UnsafePath { msg } => DirectoryMoveError::UnsafePath { msg },
            WriteError::Internal { msg } => DirectoryMoveError::Internal { msg },
            WriteError::InvalidCommitMessage { reason } => {
                DirectoryMoveError::InvalidCommitMessage { reason }
            }
            WriteError::Io { msg } => DirectoryMoveError::Io { msg },
            other => DirectoryMoveError::Internal {
                msg: format!("unexpected error surfaced in move_directory: {:?}", other),
            },
        }
    }
}

/// Recursively collect the KB-root-relative path of every regular file under
/// `abs_dir` (which must itself already exist as a directory), at any depth.
/// Symlinks are skipped — delegates the actual recursive walk to
/// `ingest::walk_dir_unfiltered`, the same walker `discover_files` runs (just in
/// its unfiltered mode), so a future fix to symlink-loop or entry-error
/// handling in one reaches both instead of only whichever one it landed in.
///
/// Unfiltered: returns every file, not just indexable documents. `move_directory`
/// uses this both for the source-subtree scan (filtered to indexable documents,
/// and checked for a stray `.kb-schema.yaml`, by the caller) and the
/// destination-prefix collision check (deliberately left UNFILTERED there, since
/// ANY file under the destination — indexable or not — means the prefix is not
/// free).
///
/// Synchronous (a plain recursive `std::fs` walk) — callers on the async path
/// must run this via [`walk_subtree_files_async`] instead of calling it
/// directly, so a large subtree scan runs off the tokio worker thread rather
/// than blocking every other task scheduled on it.
fn walk_subtree_files(canonical_data_path: &Path, abs_dir: &Path) -> std::io::Result<Vec<String>> {
    let files = crate::ingest::walk_dir_unfiltered(canonical_data_path, abs_dir)
        .map_err(|e| std::io::Error::other(format!("{e:#}")))?;
    let mut out: Vec<String> = files
        .into_iter()
        .map(|path| {
            path.strip_prefix(canonical_data_path)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/")
        })
        .collect();
    out.sort();
    Ok(out)
}

/// Off-thread wrapper around [`walk_subtree_files`] for callers inside async
/// `move_directory`: a large recursive subtree scan is exactly the kind of
/// blocking filesystem work `ingest.rs`'s own `discover_relative` already runs
/// via `spawn_blocking` (see its doc comment) rather than directly on a tokio
/// worker thread — done inline here it would stall every unrelated MCP/webhook
/// task scheduled on that same worker for the whole walk. Takes owned buffers
/// so the spawned closure needs nothing borrowed from the caller's stack
/// (`Path` args are cloned into `PathBuf`s before crossing into the blocking
/// closure) — in particular, this never captures the caller's held `GitLock`
/// guard, which must stay on the calling task.
async fn walk_subtree_files_async(
    canonical_data_path: &Path,
    abs_dir: &Path,
) -> std::io::Result<Vec<String>> {
    let canonical_data_path = canonical_data_path.to_path_buf();
    let abs_dir = abs_dir.to_path_buf();
    match tokio::task::spawn_blocking(move || walk_subtree_files(&canonical_data_path, &abs_dir))
        .await
    {
        Ok(result) => result,
        Err(e) => Err(std::io::Error::other(format!(
            "walk_subtree_files task panicked: {e}"
        ))),
    }
}

/// Best-effort, recursive, deepest-first removal of every now-empty directory
/// under (and including) `dir`. `std::fs::remove_dir` only ever succeeds on an
/// actually-empty directory, so this silently leaves anything non-empty (a
/// stray non-indexable file the include patterns never touched, e.g.) exactly
/// where it is — this is tidying up after a move, not a second guarantee
/// layered on top of `move_directory`'s own guards. A missing `dir` (already
/// gone, or never existed) is likewise a silent no-op.
///
/// Git does not track empty directories at all, so this has no bearing on
/// what gets committed — it exists purely so that after every document under
/// `source_dir` has been moved out, `source_dir` itself does not linger as an
/// empty husk on disk (and, symmetrically, so a rolled-back move's
/// now-empty destination directory does not linger either).
///
/// Synchronous, same reason as [`walk_subtree_files`] — async callers must go
/// through [`remove_empty_dirs_best_effort_async`].
fn remove_empty_dirs_best_effort(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.file_type().is_ok_and(|t| t.is_dir()) {
            remove_empty_dirs_best_effort(&entry.path());
        }
    }
    let _ = std::fs::remove_dir(dir);
}

/// Off-thread wrapper around [`remove_empty_dirs_best_effort`], same rationale
/// as [`walk_subtree_files_async`]: a deep, mostly-empty directory tree left
/// behind by a large move is still a synchronous recursive `std::fs` walk, and
/// every call site here runs while `move_directory` holds `GitLock` — this
/// takes an owned `PathBuf` rather than borrowing the caller's `&Path`, so
/// nothing from the caller's stack (least of all the lock guard, which callers
/// never pass in) crosses into the blocking closure. Best-effort by
/// construction already (the sync version swallows every error), so a panicked
/// blocking task is just logged, not propagated.
async fn remove_empty_dirs_best_effort_async(dir: &Path) {
    let dir = dir.to_path_buf();
    if let Err(e) = tokio::task::spawn_blocking(move || remove_empty_dirs_best_effort(&dir)).await {
        error!("remove_empty_dirs_best_effort task panicked: {e}");
    }
}

/// Undo every filesystem change [`move_directory`] has made SO FAR — before it
/// has touched git at all (no `commit_and_sync` call has run yet) — on a failure
/// partway through writing destinations, removing sources, or rewriting outside
/// referencing documents.
///
/// Safe to call unconditionally over the FULL `moves` list regardless of how far
/// phase 1/2 actually got: `git::restore_from_head` on a source whose worktree
/// content already matches HEAD (i.e. not yet removed) is a harmless no-op, so
/// passing every source rather than tracking exactly which ones were already
/// removed keeps this one function correct for every call site instead of
/// several slightly-different partial-rollback paths. `written_dest` and
/// `rewritten_refs`, by contrast, are exactly the paths actually mutated so far
/// — there is no such no-op equivalent for creating/removing a file that may or
/// may not exist, so those two stay precise per call site.
///
/// Takes the caller's already-held [`git::GitLock`] rather than acquiring its
/// own: `move_directory` now holds ONE guard across its entire mutating
/// sequence — every destination write, every source removal, every outside
/// referencing-document rewrite, the commit, and any rollback — for the same
/// data-loss reason `write_document_move` holds one across its equivalent
/// sequence (an unlocked read-modify-write of another document can be raced
/// and silently clobbered by a concurrent writer). This function runs from
/// failure paths INSIDE that sequence, so it must reuse the same guard rather
/// than call `git::lock_git()` itself — `GIT_LOCK` is a non-reentrant mutex,
/// and a second acquisition here while the first is still held by the caller
/// would deadlock the whole call chain against itself. (Previously this
/// acquired its own lock, on the reasoning that nothing had been staged or
/// committed yet at any call site; that remains true of the git plumbing, but
/// not of the filesystem mutations this function itself performs, which is
/// exactly the gap the data-loss finding this change fixes was about.)
/// Every individual restore/removal failure is logged, not propagated — the
/// caller has already committed to failing the whole move and just needs
/// everything recoverable put back, best effort.
async fn rollback_directory_move_filesystem(
    lock: &git::GitLock,
    data_path_str: &str,
    moves: &[(String, String)],
    abs_dest_dir: &Path,
    written_dest: &[PathBuf],
    rewritten_refs: &[String],
) {
    for (old_rel, _new_rel) in moves {
        if let Err(e) = git::restore_from_head(lock, data_path_str, old_rel).await {
            error!(
                "move_directory rollback: failed to restore source '{}': {:#}. This needs \
                 operator attention.",
                old_rel, e
            );
        }
    }
    for dest in written_dest {
        if let Err(e) = tokio::fs::remove_file(dest).await {
            error!(
                "move_directory rollback: failed to remove destination '{}': {}. This needs \
                 operator attention.",
                dest.display(),
                e
            );
        }
    }
    // Best-effort: tidy up any destination directory left empty by the removals
    // above, so a rolled-back move does not leave an empty destination prefix
    // behind — see `remove_empty_dirs_best_effort`'s doc comment.
    remove_empty_dirs_best_effort_async(abs_dest_dir).await;
    for ref_path in rewritten_refs {
        if let Err(e) = git::restore_from_head(lock, data_path_str, ref_path).await {
            error!(
                "move_directory rollback: failed to restore referencing document '{}': {:#}. \
                 This needs operator attention.",
                ref_path, e
            );
        }
    }
}

/// Relocate every document under `source_dir` to the same relative path under
/// `dest_dir`, as ONE atomic commit — the directory-move counterpart to
/// [`write_document_move`]'s single-document move, sharing its path-safety,
/// eligibility, schema-resolution, link-rewriting, and commit/rollback helpers
/// rather than duplicating them.
///
/// # Guards (all before any mutation)
/// 1. `source_dir` must exist and contain at least one document matching the
///    configured include patterns, or this is refused as
///    [`DirectoryMoveError::SourceEmpty`].
/// 2. No file may already live anywhere under `dest_dir`
///    ([`DirectoryMoveError::AlreadyExists`]) — a directory move never merges.
/// 3. Every `.kb-schema.yaml` under the source subtree must currently parse
///    ([`DirectoryMoveError::BrokenSchemaInSource`] otherwise — see that
///    variant's doc comment). One that does is not a blocker: it moves WITH the
///    documents it governs (as a raw copy — schema files are never frontmatter-
///    validated or link-rewritten), and every moved document is validated
///    against a cascade rebuilt with that schema file's governing directory
///    re-parented onto the destination (`SchemaCache::with_remapped_scopes`),
///    not against the live cache, which still reflects the OLD parentage until
///    the post-commit reindex rebuilds it. Relocating a schema file is a
///    genuine semantic change — a document valid under the source's cascade can
///    fail under the destination's — and that is exactly what guard 4 below
///    (`DirectoryMoveError::Validation`) exists to catch before anything moves.
/// 4. Neither the source nor destination subtree may be schema-frozen (checked
///    per document, via `SchemaCache::is_frozen`), and every moved document's
///    frontmatter must validate against the (possibly re-parented, per guard 3)
///    destination cascade.
/// 5. Every source and destination document path passes the same path-safety
///    ([`safe_write_path`]) and include-pattern eligibility
///    ([`check_include_pattern`]) checks a single-document write applies.
///
/// # Link rewriting
/// Every relative link inside a moved document was authored against wherever it
/// used to live. For each moved document, every link occurrence is resolved
/// against its OLD path ([`rewrite_outbound_links`]), then re-targeted:
/// - A link whose resolved target is ALSO moving (i.e. is another document under
///   `source_dir`) is translated to that target's own post-move path, then
///   re-relativized from the mover's NEW path — preserving the link's original
///   target now that both ends moved in lockstep. For a target moving in
///   lockstep with the source this usually reproduces identical text; no rewrite
///   is emitted when it does (`rewrite_outbound_links` returns the original
///   content unchanged whenever there are no occurrences, and otherwise only
///   ever replaces the exact occurrence spans it found).
/// - A link whose resolved target is NOT moving keeps that exact target; only
///   its relative spelling is recomputed from the mover's new path.
/// - A link targeting the document itself maps to its own new path (the
///   self-reference case, same as `write_document_move`).
///
/// Separately, every document OUTSIDE the moved subtree that links INTO it
/// (`StateDb::links_targeting_many`, one batched query over every moved path,
/// filtered to sources outside `source_dir` — sources INSIDE it are the
/// outbound pass above, and are never processed twice) has its link text
/// rewritten to the moved document's new location,
/// riding along in the same commit. Same best-effort semantics as
/// `write_document_move`: with no `StateDb` (`WriteDeps::state == None`), this
/// step is skipped entirely and the move still proceeds.
///
/// # Commit / rollback / reindex
/// One `git::commit_and_sync` call over every old path, every new path, and
/// every rewritten outside-referencing document (deduped). On a pre-commit
/// failure, EVERY part of the move is rolled back under the SAME held
/// `GitLock`: every source restored from HEAD, every destination removed and
/// unstaged, every rewritten referencing document restored from HEAD — all
/// steps run unconditionally, and `rolled_back` is `true` only if every single
/// one of them succeeded. On success, every path is marked dirty in one
/// `reindex::mark_paths` call.
///
/// How many documents' `validate::validate_content` calls run at once (see the
/// body below): each may exec an external `lint_command` subprocess, so this
/// is a *process* concurrency bound, not just an async-task one. 8 is
/// deliberately conservative — in the same range as a typical machine's core
/// count — chosen to get most of the win over a fully serial loop (N times
/// fewer round trips through subprocess spawn/exit latency) without letting a
/// large subtree move fork hundreds of lint processes at once and risk
/// exhausting file descriptors or thrashing the host.
const DIRECTORY_MOVE_VALIDATION_CONCURRENCY: usize = 8;

pub async fn move_directory<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &WriteDeps<'_, E, Q>,
    source_dir: &str,
    dest_dir: &str,
    message: Option<&str>,
) -> Result<DirectoryMoveSuccess, DirectoryMoveError> {
    let source_dir = crate::retrieval::kb_root_relative(source_dir).trim_end_matches('/');
    let dest_dir = crate::retrieval::kb_root_relative(dest_dir).trim_end_matches('/');

    // Guard 5 (path safety) against the two prefixes themselves, ahead of
    // walking either one.
    let abs_source_dir = safe_write_path(deps, source_dir)?;
    let abs_dest_dir = safe_write_path(deps, dest_dir)?;

    if !abs_source_dir.is_dir() {
        return Err(DirectoryMoveError::SourceEmpty {
            msg: format!("source directory '{}' does not exist", source_dir),
        });
    }

    // Guard 1: walk the whole source subtree once, collecting both the indexable
    // documents (guard 1) and any `.kb-schema.yaml` living anywhere underneath —
    // one filesystem walk answers both. A schema file no longer blocks the move
    // (guard 3, checked further below, once `deps.schema_cache` is loaded) — it
    // travels WITH the subtree instead, see this function's doc comment.
    let source_files = walk_subtree_files_async(deps.canonical_data_path, &abs_source_dir)
        .await
        .map_err(|e| DirectoryMoveError::Io {
            msg: format!("Failed to scan source directory '{}': {}", source_dir, e),
        })?;

    let schema_files_in_source: Vec<String> = source_files
        .iter()
        .filter(|p| {
            Path::new(p)
                .file_name()
                .is_some_and(|n| n == crate::schema::SCHEMA_FILE_NAME)
        })
        .cloned()
        .collect();

    let documents: Vec<String> = source_files
        .into_iter()
        .filter(|p| deps.retrieval.include_patterns.is_match(p.as_str()))
        .collect();
    if documents.is_empty() {
        return Err(DirectoryMoveError::SourceEmpty {
            msg: format!(
                "source directory '{}' contains no indexable document",
                source_dir
            ),
        });
    }

    // Guard 2: the destination prefix must be completely free — ANY file
    // underneath it, indexable or not, is a collision.
    if abs_dest_dir.is_file() {
        return Err(DirectoryMoveError::AlreadyExists);
    }
    if abs_dest_dir.is_dir() {
        let dest_files = walk_subtree_files_async(deps.canonical_data_path, &abs_dest_dir)
            .await
            .map_err(|e| DirectoryMoveError::Io {
                msg: format!("Failed to scan destination directory '{}': {}", dest_dir, e),
            })?;
        if !dest_files.is_empty() {
            return Err(DirectoryMoveError::AlreadyExists);
        }
    }

    // Every document's new path, preserving its position under the source
    // subtree. `documents` is already sorted (`walk_subtree_files` sorts), so
    // `moves` is too.
    let source_prefix = format!("{}/", source_dir);
    let moves: Vec<(String, String)> = documents
        .iter()
        .map(|old_rel| {
            let suffix = old_rel
                .strip_prefix(&source_prefix)
                .expect("every scanned document falls under its own source prefix");
            (old_rel.clone(), format!("{}/{}", dest_dir, suffix))
        })
        .collect();
    let moving: HashMap<&str, &str> = moves
        .iter()
        .map(|(old, new)| (old.as_str(), new.as_str()))
        .collect();

    // Every schema file's new path, same prefix substitution as `moves` above.
    // These move as raw copies alongside the documents they govern: never
    // through frontmatter validation (a `.kb-schema.yaml` has no frontmatter of
    // its own) and never through link rewriting (nothing in one is a markdown
    // link). Deliberately excluded from `moving`/`moves` above — those drive the
    // markdown link-rewrite passes, which a schema file relocation has nothing
    // to do with.
    let schema_moves: Vec<(String, String)> = schema_files_in_source
        .iter()
        .map(|old_rel| {
            let suffix = old_rel
                .strip_prefix(&source_prefix)
                .expect("every scanned schema file falls under its own source prefix");
            (old_rel.clone(), format!("{}/{}", dest_dir, suffix))
        })
        .collect();
    // Every path this move touches, documents and schema files alike — used for
    // commit staging, dirty-marking, rollback, and the success report. `moves`/
    // `moving` above stay document-only: a schema file is never a markdown link
    // target and never runs through `validate::validate_content`.
    let mut all_moves: Vec<(String, String)> = moves
        .iter()
        .cloned()
        .chain(schema_moves.iter().cloned())
        .collect();
    // `moves` and `schema_moves` are each individually sorted by `old_rel`
    // (`walk_subtree_files` sorts), but the chained concatenation is not —
    // re-sort so `DirectoryMoveSuccess::moved`'s documented ordering holds.
    all_moves.sort_by(|a, b| a.0.cmp(&b.0));

    let schemas = crate::schema::load_shared(deps.schema_cache);

    // Guard 3: any `.kb-schema.yaml` under the source subtree must currently
    // parse, or this move is refused outright — see
    // `DirectoryMoveError::BrokenSchemaInSource`'s doc comment for why (mirrors
    // `SchemaCache::is_frozen`'s "rules unreadable ⇒ don't touch" stance).
    // Checked against the `schemas` snapshot just loaded above, the same
    // staleness tolerance every other check in this function already accepts
    // (`is_frozen` below, in particular).
    for schema_path in &schema_files_in_source {
        let governing_dir = Path::new(schema_path.as_str())
            .parent()
            .unwrap_or(Path::new(""));
        if let Some((_, reason)) = schemas
            .broken_scopes()
            .find(|(broken_dir, _)| broken_dir.as_path() == governing_dir)
        {
            return Err(DirectoryMoveError::BrokenSchemaInSource {
                path: schema_path.clone(),
                reason: reason.to_string(),
            });
        }
    }

    // A NEW, detached cache with every schema file under `source_dir` re-parented
    // onto `dest_dir` (see `SchemaCache::with_remapped_scopes`). Moved documents
    // are validated against THIS cache below, not the live one: if the subtree
    // carries its own schema file(s), the live cache still reflects the OLD
    // parentage until the post-commit reindex rebuilds it
    // (`reindex::unit_touches_schema`), so validating against it here would
    // silently ignore the exact re-parenting this move is about to cause. When
    // the subtree has no schema file of its own, `remap` never matches anything
    // actually present in `schemas.raw`... other than possibly one of its own
    // ancestors, which is intentional: an ancestor's schema is not itself under
    // `source_dir`, so it is never relocated, and `remapped_schemas` resolves
    // identically to `schemas` for every moved document in that case.
    let source_dir_path = Path::new(source_dir);
    let dest_dir_path = Path::new(dest_dir);
    let remapped_schemas = schemas.with_remapped_scopes(|dir| {
        if dir.starts_with(source_dir_path) {
            let suffix = dir.strip_prefix(source_dir_path).unwrap_or(Path::new(""));
            Some(dest_dir_path.join(suffix))
        } else {
            None
        }
    });

    // Guard 4 (frozen, per document) + guard 5 (eligibility/safety, per
    // document) + the outbound link rewrite. ALL before any mutation: every
    // document is only ever READ here, and every failure path below returns
    // before touching the filesystem. These checks are all cheap, in-memory or
    // single-file-read work, so they stay a plain serial loop — same
    // first-failure-wins behavior as before, e.g. `AlreadyExists`/`Frozen` on
    // whichever document trips it first. Only `validate::validate_content`
    // below (which may exec a `lint_command` subprocess) is expensive enough,
    // and independent enough per document, to run concurrently.
    // (old_rel, new_rel, content_to_write)
    let mut contents: Vec<(String, String, String)> = Vec::new();

    for (old_rel, new_rel) in &moves {
        check_include_pattern(deps, old_rel)?;
        check_include_pattern(deps, new_rel)?;
        let abs_source_doc = safe_write_path(deps, old_rel)?;
        let abs_dest_doc = safe_write_path(deps, new_rel)?;

        if let Some(reason) = schemas.is_frozen(Path::new(old_rel.as_str())) {
            return Err(DirectoryMoveError::Frozen {
                reason: reason.to_string(),
            });
        }
        if let Some(reason) = schemas.is_frozen(Path::new(new_rel.as_str())) {
            return Err(DirectoryMoveError::Frozen {
                reason: reason.to_string(),
            });
        }

        if abs_dest_doc.exists() {
            // Guard 2 already checked the whole prefix; this is a defensive
            // re-check against a TOCTOU race between that walk and here.
            return Err(DirectoryMoveError::AlreadyExists);
        }

        let old_content = tokio::fs::read_to_string(&abs_source_doc)
            .await
            .map_err(|e| DirectoryMoveError::Io {
                msg: format!("Failed to read '{}': {}", old_rel, e),
            })?;

        let content_to_write = rewrite_outbound_links(&old_content, old_rel, new_rel, |resolved| {
            moving.get(resolved).map(|new| new.to_string())
        });

        contents.push((old_rel.clone(), new_rel.clone(), content_to_write));
    }

    // Read every schema file's raw content too — same path-safety (guard 5) as
    // any other moved file, but deliberately NOT `check_include_pattern` (a
    // `.kb-schema.yaml` never matches the markdown include patterns, so that
    // check would always reject it) and no `rewrite_outbound_links` (schema
    // files hold no markdown links). These ride along in the same physical
    // write/remove phases as `contents` below, chained rather than merged into
    // it, so they never enter `validate::validate_content`.
    let mut schema_contents: Vec<(String, String, String)> = Vec::new();
    for (old_rel, new_rel) in &schema_moves {
        let abs_source_schema = safe_write_path(deps, old_rel)?;
        let _abs_dest_schema = safe_write_path(deps, new_rel)?;
        let raw = tokio::fs::read_to_string(&abs_source_schema)
            .await
            .map_err(|e| DirectoryMoveError::Io {
                msg: format!("Failed to read '{}': {}", old_rel, e),
            })?;
        schema_contents.push((old_rel.clone(), new_rel.clone(), raw));
    }

    // Destination-schema validation, run concurrently across every document
    // rather than one `validate::validate_content` await at a time — each call
    // may exec an external `lint_command` subprocess, so a 200-document
    // subtree previously paid 200x that subprocess's spawn/exit latency
    // serially. Bounded via `buffer_unordered`, not spawned one task per
    // document unbounded: an unbounded fan-out on a large subtree would fork
    // hundreds of lint subprocesses at once and risks exhausting file
    // descriptors or thrashing the machine. `DIRECTORY_MOVE_VALIDATION_CONCURRENCY`
    // documents `N`'s reasoning.
    //
    // Semantics are preserved exactly: this collects EVERY document's outcome
    // before deciding anything (`.collect::<Vec<_>>().await` drains the whole
    // bounded stream), so a document that fails validation can never be
    // reported as "the only failure" just because it finished first under
    // concurrency — same as the old serial loop reporting every failure
    // encountered before returning `Validation`. A genuine `validate_content`
    // error (as opposed to an ordinary "invalid" result) still aborts the move
    // exactly like the old loop's `?` did — the first one found after the
    // whole bounded batch settles, rather than mid-loop, since concurrent
    // tasks already in flight cannot be un-started once launched.
    // Built via a plain loop (not `Iterator::map`) so each future's captures are
    // inferred independently rather than through one `FnMut` closure signature
    // that has to hold for every item uniformly — the latter runs into a known
    // rustc limitation ("implementation of `FnOnce` is not general enough")
    // once the closure's return type borrows from both the loop item and an
    // outer variable (`remapped_schemas`) at once.
    //
    // Deliberately `remapped_schemas`, not the live `schemas` snapshot: every
    // moved document's frontmatter is checked against the cascade it will
    // ACTUALLY resolve to post-move, with any schema file in this subtree
    // already re-parented onto the destination — see `remapped_schemas`'s doc
    // comment above and `SchemaCache::with_remapped_scopes`.
    let mut validation_futures = Vec::with_capacity(contents.len());
    for (_old_rel, new_rel, content_to_write) in &contents {
        let schema = remapped_schemas.resolve_for(Path::new(new_rel.as_str()));
        validation_futures.push(async move {
            let outcome = validate::validate_content(
                Path::new(new_rel.as_str()),
                content_to_write,
                schema,
                deps.validation,
            )
            .await
            .map(|(validation_result, _validated)| validation_result);
            (new_rel.clone(), outcome)
        });
    }
    let validation_outcomes: Vec<(String, anyhow::Result<ValidationResult>)> =
        stream::iter(validation_futures)
            .buffer_unordered(DIRECTORY_MOVE_VALIDATION_CONCURRENCY)
            .collect()
            .await;

    let mut validation_failures: Vec<(String, ValidationResult)> = Vec::new();
    for (new_rel, outcome) in validation_outcomes {
        match outcome {
            Ok(validation_result) => {
                if !validation_result.valid {
                    validation_failures.push((new_rel, validation_result));
                }
            }
            Err(e) => {
                error!(
                    "Validation error moving into '{}' (source '{}' -> '{}'): {:#}",
                    new_rel, source_dir, dest_dir, e
                );
                return Err(DirectoryMoveError::Io {
                    msg: format!("Failed to validate content: {}", e),
                });
            }
        }
    }

    if !validation_failures.is_empty() {
        // Sort so the reported order is deterministic regardless of which
        // validation happened to finish first under `buffer_unordered` — the
        // old serial loop always reported failures in `moves` order (which is
        // sorted, per `walk_subtree_files`), and callers' error text
        // (`mcp::move_directory_error_to_mcp_error`) reads more like a stable
        // report when it stays that way.
        validation_failures.sort_by(|a, b| a.0.cmp(&b.0));
        return Err(DirectoryMoveError::Validation {
            failures: validation_failures,
            // Empty unless this subtree carries its own schema file(s) — lets
            // the caller-facing error explain WHY a document that was valid at
            // the source can fail here: the cascade it is being checked against
            // just re-parented, not merely relocated.
            moved_schema_files: schema_moves.clone(),
        });
    }

    validate_commit_message(message)?;

    let data_path_str = deps.canonical_data_path.to_str().unwrap_or_default();

    // Acquire GIT_LOCK now and hold ONE guard across every remaining mutation
    // of the clone below — phase 1 (destination writes), phase 2 (source
    // removals), phase 3 (outside referencing-document rewrites), the commit,
    // and any rollback — rather than only around the commit as this used to.
    // Same reasoning as `write_document_move`'s identical hoist: phase 3 is an
    // unlocked read-modify-write of documents OUTSIDE this move's own path
    // set, with no staleness check of its own, so a concurrent writer to one
    // of those documents can land its write in the gap between this read and
    // this write and be silently clobbered. Phases 1 and 2 are pulled in too
    // for the same reason `write_document_move` pulls in its own destination
    // write/source removal: they are themselves clone mutations, and holding
    // one guard across the whole sequence (rather than acquire/release/
    // re-acquire) is both this codebase's existing convention and what keeps
    // an in-progress, not-yet-committed move from interleaving with another
    // writer's commit or a concurrent webhook merge. Every helper reachable
    // from here that used to acquire its own `GitLock` —
    // `rollback_directory_move_filesystem`, called from every failure branch
    // in phases 1-3 — now takes this same guard by reference instead, which
    // is what keeps this non-reentrant mutex from deadlocking against itself.
    let git_lock = git::lock_git().await;

    // Filesystem mutation, phase 1: write every DESTINATION first (`create_new`,
    // same non-clobbering guarantee `write_document_move` relies on), before
    // touching a single source — the same write-then-remove ordering that
    // function uses, batched: if any destination write fails partway through,
    // no source has been touched at all, so recovery is just deleting whatever
    // destinations already landed. Chained with `schema_contents` so every
    // schema file under the subtree gets the same treatment as any other moved
    // file — `rollback_directory_move_filesystem` below is always handed
    // `all_moves` (documents AND schema files), never the document-only `moves`.
    let mut written_dest: Vec<PathBuf> = Vec::new();
    for (_old_rel, new_rel, content_to_write) in contents.iter().chain(schema_contents.iter()) {
        let abs_dest = match safe_write_path(deps, new_rel) {
            Ok(p) => p,
            Err(e) => {
                rollback_directory_move_filesystem(
                    &git_lock,
                    data_path_str,
                    &all_moves,
                    &abs_dest_dir,
                    &written_dest,
                    &[],
                )
                .await;
                return Err(e.into());
            }
        };
        if let Some(parent) = abs_dest.parent()
            && let Err(e) = tokio::fs::create_dir_all(parent).await
        {
            rollback_directory_move_filesystem(
                &git_lock,
                data_path_str,
                &all_moves,
                &abs_dest_dir,
                &written_dest,
                &[],
            )
            .await;
            return Err(DirectoryMoveError::Io {
                msg: format!(
                    "Failed to create parent directories for '{}': {}",
                    new_rel, e
                ),
            });
        }

        let write_outcome: std::io::Result<()> = async {
            use tokio::io::AsyncWriteExt as _;
            let mut file = tokio::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&abs_dest)
                .await?;
            file.write_all(content_to_write.as_bytes()).await
        }
        .await;

        match write_outcome {
            Ok(()) => written_dest.push(abs_dest),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                // TOCTOU collision: guard 2 above walked the destination prefix
                // and found it clear, but something has landed at this exact
                // computed destination since then. This is the same benign,
                // retryable race `write_document_move`'s equivalent
                // `create_new` open maps to `WriteError::AlreadyExists` — map
                // it identically here rather than letting it fall into the
                // generic `Io` arm below, which `move_directory_error_to_mcp_error`
                // reports as an opaque internal error instead of a clear
                // "already exists" the caller can act on.
                error!(
                    "Destination '{}' already exists (TOCTOU collision) while moving directory \
                     '{}' -> '{}'. Undoing every filesystem change made so far.",
                    new_rel, source_dir, dest_dir
                );
                rollback_directory_move_filesystem(
                    &git_lock,
                    data_path_str,
                    &all_moves,
                    &abs_dest_dir,
                    &written_dest,
                    &[],
                )
                .await;
                return Err(DirectoryMoveError::AlreadyExists);
            }
            Err(e) => {
                error!(
                    "Failed to write destination '{}' while moving directory '{}' -> '{}': {}. \
                     Undoing every filesystem change made so far.",
                    new_rel, source_dir, dest_dir, e
                );
                rollback_directory_move_filesystem(
                    &git_lock,
                    data_path_str,
                    &all_moves,
                    &abs_dest_dir,
                    &written_dest,
                    &[],
                )
                .await;
                return Err(DirectoryMoveError::Io {
                    msg: format!(
                        "Failed to write destination '{}' during directory move: {}",
                        new_rel, e
                    ),
                });
            }
        }
    }

    // Filesystem mutation, phase 2: remove every SOURCE, now that every
    // destination is confirmed written. A failure partway through is recovered
    // by restoring every source from HEAD and deleting every destination
    // written in phase 1 — nothing has touched git yet, so this is a pure
    // filesystem undo (see `rollback_directory_move_filesystem`'s doc comment
    // for why restoring the FULL source list, not just the ones already
    // removed, is safe). Chained with `schema_contents`, same reasoning as
    // phase 1 above.
    for (old_rel, _new_rel, _content_to_write) in contents.iter().chain(schema_contents.iter()) {
        let abs_source = match safe_write_path(deps, old_rel) {
            Ok(p) => p,
            Err(e) => {
                rollback_directory_move_filesystem(
                    &git_lock,
                    data_path_str,
                    &all_moves,
                    &abs_dest_dir,
                    &written_dest,
                    &[],
                )
                .await;
                return Err(e.into());
            }
        };
        if let Err(e) = tokio::fs::remove_file(&abs_source).await {
            error!(
                "Failed to remove source '{}' while moving directory '{}' -> '{}': {}. \
                 Restoring every source and deleting every written destination.",
                old_rel, source_dir, dest_dir, e
            );
            rollback_directory_move_filesystem(
                &git_lock,
                data_path_str,
                &all_moves,
                &abs_dest_dir,
                &written_dest,
                &[],
            )
            .await;
            return Err(DirectoryMoveError::Io {
                msg: format!(
                    "Failed to remove source '{}' during directory move: {}",
                    old_rel, e
                ),
            });
        }
    }

    // Every document under `source_dir` is gone; tidy up any subdirectory (and
    // `source_dir` itself) that removing them left empty, best-effort — see
    // `remove_empty_dirs_best_effort`'s doc comment. Git does not track empty
    // directories, so this has no bearing on the commit below; it just keeps
    // the old prefix from lingering as an empty husk on disk.
    remove_empty_dirs_best_effort_async(&abs_source_dir).await;

    // Phase 3: rewrite documents OUTSIDE the moved subtree that link INTO it, so
    // those links keep resolving after the move — riding along in the SAME
    // commit as the move itself. Sources INSIDE the subtree are handled by the
    // outbound pass above and must never be processed again here.
    //
    // One batched `links_targeting_many` call over every moved path rather than
    // a `links_targeting` call per document: for a large subtree that was
    // hundreds of independent SQLite round-trips in a plain for-loop. The
    // per-source aggregation below is unchanged — a referencing document that
    // links to several moved targets still gets exactly one `outside_refs`
    // entry, with every target it references collected onto it.
    let mut outside_refs: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    if let Some(state) = deps.state {
        let target_paths: Vec<String> = moves.iter().map(|(old_rel, _)| old_rel.clone()).collect();
        match state.links_targeting_many(&target_paths, "markdown").await {
            Ok(by_target) => {
                for (old_rel, new_rel) in &moves {
                    let Some(referencing_paths) = by_target.get(old_rel.as_str()) else {
                        continue;
                    };
                    for ref_path in referencing_paths {
                        if moving.contains_key(ref_path.as_str()) {
                            // Inside the subtree — handled by the outbound
                            // rewrite above; processing it again here would
                            // double-edit it.
                            continue;
                        }
                        outside_refs
                            .entry(ref_path.clone())
                            .or_default()
                            .push((old_rel.clone(), new_rel.clone()));
                    }
                }
            }
            Err(e) => {
                warn!(
                    "Skipping incoming-link rewrite for every moved document while moving \
                     directory '{}' -> '{}': the batched reverse-link query failed: {:#}",
                    source_dir, dest_dir, e
                );
            }
        }
    }

    let mut rewritten_paths: Vec<String> = Vec::new();
    for (ref_path, targets) in &outside_refs {
        let abs_ref = match safe_write_path(deps, ref_path) {
            Ok(p) => p,
            Err(_) => {
                warn!(
                    "Skipping link rewrite in '{}' while moving directory '{}' -> '{}': the \
                     path no longer resolves safely (stale document_links row?)",
                    ref_path, source_dir, dest_dir
                );
                continue;
            }
        };
        let body = match tokio::fs::read_to_string(&abs_ref).await {
            Ok(b) => b,
            Err(e) => {
                warn!(
                    "Skipping link rewrite in '{}' while moving directory '{}' -> '{}': failed \
                     to read it, likely a stale document_links row: {}",
                    ref_path, source_dir, dest_dir, e
                );
                continue;
            }
        };

        let mut replacements: Vec<(crate::ingest::LinkOccurrence, String)> = Vec::new();
        for (old_rel, new_rel) in targets {
            let replacement = crate::ingest::relativize_md_path(ref_path, new_rel);
            for occurrence in find_all_markdown_link_occurrences(&body, ref_path)
                .into_iter()
                .filter(|o| &o.resolved == old_rel)
            {
                replacements.push((occurrence, replacement.clone()));
            }
        }
        if replacements.is_empty() {
            // Stale document_links row(s): nothing in the CURRENT body actually
            // resolves to any moved target anymore.
            continue;
        }

        let new_body = apply_link_replacements_each(&body, &replacements);
        if let Err(e) = tokio::fs::write(&abs_ref, new_body.as_bytes()).await {
            error!(
                "Failed to rewrite links into '{}' while moving directory '{}' -> '{}': {}. \
                 Undoing every filesystem change made for this move so far.",
                ref_path, source_dir, dest_dir, e
            );
            rollback_directory_move_filesystem(
                &git_lock,
                data_path_str,
                &all_moves,
                &abs_dest_dir,
                &written_dest,
                &rewritten_paths,
            )
            .await;
            return Err(DirectoryMoveError::Io {
                msg: format!("Failed to rewrite links in '{}': {}", ref_path, e),
            });
        }
        rewritten_paths.push(ref_path.clone());
    }

    // Commit the move AND every rewritten referencing document as ONE atomic
    // commit, under the SAME lock acquisition (above) already held across
    // phases 1-3 — releasing it in between any of those and the commit would
    // let another writer stage into (and, since it commits its own path,
    // commit) the very half-staged state this call is about to undo. See
    // `write_document_move`'s identical comment for the full reasoning.
    let commit_message = build_commit_message(
        message,
        &format!("docs: move {} to {}", source_dir, dest_dir),
        "move_directory",
    );

    let mut commit_paths: Vec<&str> = Vec::new();
    for (old_rel, new_rel) in &all_moves {
        commit_paths.push(old_rel.as_str());
        commit_paths.push(new_rel.as_str());
    }
    for ref_path in &rewritten_paths {
        if !commit_paths.contains(&ref_path.as_str()) {
            commit_paths.push(ref_path.as_str());
        }
    }

    let commit_outcome = match git::commit_and_sync(
        &git_lock,
        deps.git_url,
        deps.branch,
        data_path_str,
        deps.token,
        &commit_paths,
        &commit_message,
        deps.commit_author_name,
        deps.commit_author_email,
    )
    .await
    {
        Ok(outcome) => outcome,

        Err(git::CommitSyncError::PreCommit(source_err)) => {
            error!(
                "commit_and_sync pre-commit failure moving directory '{}' -> '{}', rolling \
                 back {} document(s) and {} rewritten referencing document(s): {:#}",
                source_dir,
                dest_dir,
                all_moves.len(),
                rewritten_paths.len(),
                source_err
            );

            // Roll back EVERY part of this move — every source, every
            // destination, plus every referencing document rewritten above.
            // Each group is independent of the others and ALL of them always
            // run unconditionally, so a failure in one never leaves a
            // recoverable part undone. `rolled_back` is true only if every
            // single one of these succeeds.
            let mut rolled_back = true;
            for (old_rel, _new_rel) in &all_moves {
                if let Err(e) = git::restore_from_head(&git_lock, data_path_str, old_rel).await {
                    rolled_back = false;
                    error!(
                        "move_directory rollback: failed to restore source '{}': {:#}. This \
                         needs operator attention.",
                        old_rel, e
                    );
                }
            }
            for (_old_rel, new_rel) in &all_moves {
                let result = match safe_write_path(deps, new_rel) {
                    Ok(abs) => match tokio::fs::remove_file(&abs).await {
                        Ok(()) => git::unstage(&git_lock, data_path_str, new_rel).await,
                        Err(e) => Err(anyhow::Error::new(e)
                            .context("Failed to remove the new destination file during rollback")),
                    },
                    Err(e) => Err(anyhow::anyhow!(
                        "destination '{}' no longer resolves safely during rollback: {:?}",
                        new_rel,
                        e
                    )),
                };
                if let Err(e) = result {
                    rolled_back = false;
                    error!(
                        "move_directory rollback: failed to remove destination '{}': {:#}. \
                         This needs operator attention.",
                        new_rel, e
                    );
                }
            }
            // Best-effort: tidy up any destination directory left empty by the
            // removals above — see `remove_empty_dirs_best_effort`'s doc comment.
            remove_empty_dirs_best_effort_async(&abs_dest_dir).await;
            for ref_path in &rewritten_paths {
                if let Err(e) = git::restore_from_head(&git_lock, data_path_str, ref_path).await {
                    rolled_back = false;
                    error!(
                        "move_directory rollback: failed to restore referencing document \
                         '{}': {:#}. This needs operator attention.",
                        ref_path, e
                    );
                }
            }

            if !rolled_back {
                error!(
                    "Rollback FAILED after a pre-commit git failure moving directory '{}' -> \
                     '{}'. Filesystem and git state may now be inconsistent. Original cause: \
                     {:#}",
                    source_dir, dest_dir, source_err
                );
            }

            return Err(DirectoryMoveError::PreCommitFailed {
                rolled_back,
                msg: format!("{:#}", source_err),
            });
        }

        Err(git::CommitSyncError::PostCommit {
            sha,
            source: source_err,
        }) => {
            warn!(
                "commit_and_sync post-commit (sync) failure moving directory '{}' -> '{}', \
                 commit {} stands uncorrected: {:#}",
                source_dir, dest_dir, sha, source_err
            );

            deps.queue.mark_paths(
                all_moves
                    .iter()
                    .flat_map(|(o, n)| [PathBuf::from(o.clone()), PathBuf::from(n.clone())])
                    .chain(rewritten_paths.iter().map(PathBuf::from)),
            );

            return Ok(DirectoryMoveSuccess {
                outcome: WriteOutcome::CommittedPendingSync,
                sha,
                rebased_paths: Vec::new(),
                moved: all_moves,
                rewritten_paths,
                sync_failure_cause: Some(format!("{:#}", source_err)),
            });
        }
    };

    // Mark the source, the destination, and every rewritten referencing
    // document dirty — plus anything the rebase pulled in — in the SAME
    // marking call, same reasoning as `write_document_move`'s identical final
    // step. `all_moves` includes any relocated schema file alongside every
    // document, which is exactly what makes `reindex::unit_touches_schema`
    // force the shared `SchemaCache` to rebuild before this unit is next
    // indexed — see that function's doc comment; nothing further is needed
    // here for the post-commit self-correction this move depends on.
    deps.queue.mark_paths(
        all_moves
            .iter()
            .flat_map(|(o, n)| [PathBuf::from(o.clone()), PathBuf::from(n.clone())])
            .chain(rewritten_paths.iter().map(PathBuf::from))
            .chain(commit_outcome.rebased_paths.iter().cloned()),
    );

    Ok(DirectoryMoveSuccess {
        outcome: WriteOutcome::Synced,
        sha: commit_outcome.sha,
        rebased_paths: commit_outcome.rebased_paths,
        moved: all_moves,
        rewritten_paths,
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

    // Best-effort: warn if anything else in the KB still links to the document
    // about to be deleted (#181), and (#229) carry the same paths through to
    // `WriteSuccess::referencing_paths` — a `warn!` reaches an operator tailing
    // the server, not the caller (usually an agent with no log access), which
    // is the party actually deciding whether the delete was a good idea.
    // `StateDb::links_targeting` is the exact same reverse-link query
    // `write_document_move`'s step 10.5 already runs to find documents whose
    // body needs rewriting — reused here purely to look, not to touch anything.
    //
    // Deliberately WARN/report, not refuse: this codebase's established stance
    // on a stale/dangling link is "self-heal, don't block" — `write_document_move`
    // and `move_directory` both skip a referencing document outright rather
    // than fail the whole operation when a `document_links` row turns out to
    // be stale, and a referencing document's OWN next reindex rebuilds its
    // links from whatever its current on-disk body actually resolves to,
    // silently dropping the edge that no longer exists. A delete leaving a
    // dangling link behaves no differently from that already-accepted case.
    // Refusing outright would also need a `force` escape hatch threaded
    // through every caller's request shape — the MCP tool's parameter schema
    // and the HTTP API's request body — which is a cross-cutting change this
    // transport-agnostic pipeline should not decide unilaterally (see #229).
    // `deps.state` is `None` for callers with no `StateDb` wired up (see that
    // field's doc comment on `WriteDeps`); this is skipped silently in that
    // case, same as the move path's own reverse-link query — `referencing_paths`
    // then stays empty, indistinguishable from "checked, found nothing".
    let mut referencing_paths: Vec<String> = Vec::new();
    if let Some(state) = deps.state {
        match state.links_targeting(rel_path, "markdown").await {
            Ok(inbound) if !inbound.is_empty() => {
                warn!(
                    "Deleting '{}', which is still linked from {} other document(s): {}. \
                     This delete does not rewrite or remove those links — they will dangle \
                     until each referencing document's own next reindex drops the now-stale \
                     edge.",
                    rel_path,
                    inbound.len(),
                    inbound.join(", ")
                );
                referencing_paths = inbound;
            }
            Ok(_) => {}
            Err(e) => {
                warn!(
                    "Skipping inbound-link check before deleting '{}': the reverse-link \
                     query failed: {:#}",
                    rel_path, e
                );
            }
        }
    }

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
    // Held across the commit AND the restore below, for the same reason as on the
    // write path: a rollback that runs outside the acquisition that produced the
    // failure is racing every other writer.
    let git_lock = git::lock_git().await;

    let commit_outcome = match git::commit_and_sync(
        &git_lock,
        deps.git_url,
        deps.branch,
        data_path_str,
        deps.token,
        &[rel_path],
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

            match git::restore_from_head(&git_lock, data_path_str, rel_path).await {
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
            deps.queue
                .mark_paths(std::iter::once(PathBuf::from(rel_path)));

            return Ok(WriteSuccess {
                outcome: WriteOutcome::CommittedPendingSync,
                sha,
                rebased_paths: Vec::new(),
                diff: render_unified_diff(&old_content, "", rel_path),
                sync_failure_cause: Some(format!("{:#}", source)),
                // Deletes never rewrite other documents' links — a dangling
                // link to a deleted document self-heals to a dropped edge on
                // the referencing document's own next reindex, same as any
                // other stale `document_links` row.
                rewritten_paths: Vec::new(),
                referencing_paths,
            });
        }
    };

    // Mark this path — and anything the rebase pulled in — dirty and return
    // immediately. The worker's scoped indexer purges a path's Qdrant points and
    // state rows itself once it re-checks and finds the file gone (the
    // missing-file branch of `ingest::index_paths`), so there is no separate
    // purge to do here — this is "one reindex path" applied to deletes too, not a
    // special case.
    deps.queue.mark_paths(
        std::iter::once(PathBuf::from(rel_path))
            .chain(commit_outcome.rebased_paths.iter().cloned()),
    );

    Ok(WriteSuccess {
        outcome: WriteOutcome::Synced,
        sha: commit_outcome.sha,
        diff: render_unified_diff(&old_content, "", rel_path),
        rebased_paths: commit_outcome.rebased_paths,
        sync_failure_cause: None,
        rewritten_paths: Vec::new(),
        referencing_paths,
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
    // (#179) split_frontmatter_bytes / apply_frontmatter_patch / apply_append —
    // pure unit tests. These are the exact edge cases #179 calls out by name:
    // no frontmatter, an empty file, and no trailing newline.
    // -----------------------------------------------------------------------

    #[test]
    fn split_frontmatter_bytes_normal_document() {
        let content = "---\ntitle: X\n---\n\n# Body\ntext\n";
        let (fm, body) = split_frontmatter_bytes(content).unwrap();
        assert_eq!(fm, "---\ntitle: X\n---\n");
        assert_eq!(body, "\n# Body\ntext\n");
        assert_eq!(format!("{fm}{body}"), content, "split must be lossless");
    }

    #[test]
    fn split_frontmatter_bytes_no_trailing_newline_on_body() {
        let content = "---\ntitle: X\n---\n\n# Body\ntext";
        let (fm, body) = split_frontmatter_bytes(content).unwrap();
        assert_eq!(fm, "---\ntitle: X\n---\n");
        assert_eq!(body, "\n# Body\ntext");
    }

    #[test]
    fn split_frontmatter_bytes_no_frontmatter_returns_none() {
        assert!(split_frontmatter_bytes("# Just a doc\nbody text\n").is_none());
    }

    #[test]
    fn split_frontmatter_bytes_empty_content_returns_none() {
        assert!(split_frontmatter_bytes("").is_none());
    }

    #[test]
    fn split_frontmatter_bytes_unterminated_delimiter_returns_none() {
        // Opens with `---` but never closes — must not be mistaken for a
        // (frontmatter, "") split.
        assert!(split_frontmatter_bytes("---\ntitle: X\nno closing delimiter\n").is_none());
    }

    #[test]
    fn split_frontmatter_bytes_dashes_inside_a_value_are_not_the_closing_delimiter() {
        let content = "---\ndescription: a---b\n---\n\nBody\n";
        let (fm, body) = split_frontmatter_bytes(content).unwrap();
        assert_eq!(fm, "---\ndescription: a---b\n---\n");
        assert_eq!(body, "\nBody\n");
    }

    #[test]
    fn apply_frontmatter_patch_set_field_on_existing_document() {
        let old = "---\ntitle: X\nstatus: draft\n---\n\n# Body\n";
        let new = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::SetField {
                field: "status".into(),
                value: serde_json::json!("active"),
            }],
        )
        .unwrap();
        let (fm, body) = validate::parse_frontmatter_raw(&new);
        assert_eq!(fm.get("status").unwrap(), "active");
        assert_eq!(fm.get("title").unwrap(), "X");
        assert_eq!(body.trim(), "# Body");
    }

    #[test]
    fn apply_frontmatter_patch_never_touches_the_body() {
        let old = "---\ntitle: X\n---\n\n# Body\nwith --- a dash-line\nand text\n";
        let new = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::SetField {
                field: "title".into(),
                value: serde_json::json!("Y"),
            }],
        )
        .unwrap();
        assert!(new.ends_with("# Body\nwith --- a dash-line\nand text\n"));
    }

    #[test]
    fn apply_frontmatter_patch_creates_frontmatter_when_absent() {
        let old = "# Just a doc\nno frontmatter here\n";
        let new = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::SetField {
                field: "title".into(),
                value: serde_json::json!("New Title"),
            }],
        )
        .unwrap();
        let (fm, body) = validate::parse_frontmatter_raw(&new);
        assert_eq!(fm.get("title").unwrap(), "New Title");
        assert_eq!(body.trim(), "# Just a doc\nno frontmatter here");
    }

    #[test]
    fn apply_frontmatter_patch_on_empty_file_creates_frontmatter_only() {
        let new = apply_frontmatter_patch(
            "",
            &[FrontmatterEdit::SetField {
                field: "title".into(),
                value: serde_json::json!("T"),
            }],
        )
        .unwrap();
        assert!(new.starts_with("---\n"));
        assert!(new.trim_end().ends_with("---"));
    }

    #[test]
    fn apply_frontmatter_patch_remove_field_errors_when_absent() {
        let old = "---\ntitle: X\n---\n\nBody\n";
        let err = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::RemoveField {
                field: "nonexistent".into(),
            }],
        )
        .unwrap_err();
        assert!(err.contains("nonexistent"), "got: {err}");
    }

    #[test]
    fn apply_frontmatter_patch_remove_field_removes_when_present() {
        let old = "---\ntitle: X\nstatus: draft\n---\n\nBody\n";
        let new = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::RemoveField {
                field: "status".into(),
            }],
        )
        .unwrap();
        let (fm, _) = validate::parse_frontmatter_raw(&new);
        assert!(!fm.contains_key("status"));
        assert!(fm.contains_key("title"));
    }

    #[test]
    fn apply_frontmatter_patch_add_values_creates_and_dedupes() {
        let old = "---\ntitle: X\ntags: [a]\n---\n\nBody\n";
        let new = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::AddValues {
                field: "tags".into(),
                values: vec![serde_json::json!("a"), serde_json::json!("b")],
            }],
        )
        .unwrap();
        let (fm, _) = validate::parse_frontmatter_raw(&new);
        let tags: Vec<String> = fm
            .get("tags")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(tags, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn apply_frontmatter_patch_add_values_creates_field_when_absent() {
        let old = "---\ntitle: X\n---\n\nBody\n";
        let new = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::AddValues {
                field: "tags".into(),
                values: vec![serde_json::json!("new")],
            }],
        )
        .unwrap();
        let (fm, _) = validate::parse_frontmatter_raw(&new);
        assert_eq!(
            fm.get("tags").unwrap().as_array().unwrap(),
            &vec![serde_json::json!("new")]
        );
    }

    #[test]
    fn apply_frontmatter_patch_add_values_errors_on_non_list_field() {
        let old = "---\ntitle: X\n---\n\nBody\n";
        let err = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::AddValues {
                field: "title".into(),
                values: vec![serde_json::json!("x")],
            }],
        )
        .unwrap_err();
        assert!(err.contains("title"), "got: {err}");
    }

    #[test]
    fn apply_frontmatter_patch_remove_values_filters_the_list() {
        let old = "---\ntitle: X\ntags: [a, b, c]\n---\n\nBody\n";
        let new = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::RemoveValues {
                field: "tags".into(),
                values: vec![serde_json::json!("b")],
            }],
        )
        .unwrap();
        let (fm, _) = validate::parse_frontmatter_raw(&new);
        let tags: Vec<String> = fm
            .get("tags")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(tags, vec!["a".to_string(), "c".to_string()]);
    }

    #[test]
    fn apply_frontmatter_patch_remove_values_errors_when_field_absent() {
        let old = "---\ntitle: X\n---\n\nBody\n";
        let err = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::RemoveValues {
                field: "tags".into(),
                values: vec![serde_json::json!("a")],
            }],
        )
        .unwrap_err();
        assert!(err.contains("tags"), "got: {err}");
    }

    #[test]
    fn apply_frontmatter_patch_multiple_edits_apply_in_order() {
        let old = "---\ntitle: X\nstatus: draft\ntags: [a]\n---\n\nBody\n";
        let new = apply_frontmatter_patch(
            old,
            &[
                FrontmatterEdit::SetField {
                    field: "status".into(),
                    value: serde_json::json!("active"),
                },
                FrontmatterEdit::AddValues {
                    field: "tags".into(),
                    values: vec![serde_json::json!("b")],
                },
                FrontmatterEdit::RemoveField {
                    field: "title".into(),
                },
            ],
        )
        .unwrap();
        let (fm, _) = validate::parse_frontmatter_raw(&new);
        assert_eq!(fm.get("status").unwrap(), "active");
        assert!(!fm.contains_key("title"));
        assert_eq!(
            fm.get("tags").unwrap().as_array().unwrap().len(),
            2,
            "got: {:?}",
            fm.get("tags")
        );
    }

    #[test]
    fn apply_append_no_frontmatter() {
        let result = apply_append("existing line\n", "new entry");
        assert_eq!(result, "existing line\nnew entry\n");
    }

    /// A CRLF document must stay entirely CRLF after an append.
    ///
    /// Both halves matter: the separator this function inserts, and the
    /// caller's own text, which arrives LF-terminated because an agent
    /// composing an append has no idea what the file on disk uses. Getting
    /// either wrong leaves a file with mixed endings — no bytes lost, but
    /// every later diff of that document shows lines nobody edited.
    #[test]
    fn apply_append_preserves_crlf_line_endings() {
        let result = apply_append("# Body\r\nold line\r\n", "new entry");
        assert_eq!(result, "# Body\r\nold line\r\nnew entry\r\n");

        let multi = apply_append("# Body\r\n", "first\nsecond");
        assert_eq!(multi, "# Body\r\nfirst\r\nsecond\r\n");

        let no_trailing = apply_append("# Body\r\nold line", "new entry");
        assert_eq!(no_trailing, "# Body\r\nold line\r\nnew entry\r\n");
    }

    /// An LF document must not acquire CRLF from CRLF-terminated input text.
    #[test]
    fn apply_append_normalizes_crlf_input_into_an_lf_document() {
        let result = apply_append("# Body\nold line\n", "first\r\nsecond");
        assert_eq!(result, "# Body\nold line\nfirst\nsecond\n");
    }

    /// A frontmatter patch re-serializes the whole block, so it is the other
    /// place a CRLF document can silently become mixed — `serde_yaml_ng`
    /// always emits LF.
    #[test]
    fn apply_frontmatter_patch_preserves_crlf_line_endings() {
        let doc = "---\r\ntitle: X\r\nstatus: draft\r\n---\r\n\r\n# Body\r\ntext\r\n";
        let result = apply_frontmatter_patch(
            doc,
            &[FrontmatterEdit::SetField {
                field: "status".into(),
                value: serde_json::json!("active"),
            }],
        )
        .unwrap();
        // Deliberately not asserting the absence of "\n\r": two adjacent CRLF
        // endings contain that sequence at their boundary, so it says nothing.
        // The per-line check below is the real invariant.
        for line in result.split_inclusive('\n') {
            assert!(
                !line.ends_with('\n') || line.ends_with("\r\n"),
                "every terminated line must keep CRLF, got {line:?}"
            );
        }
        assert!(
            result.contains("# Body\r\ntext\r\n"),
            "body must survive byte-exact: {result:?}"
        );
    }

    #[test]
    fn apply_append_no_trailing_newline_on_existing_content() {
        let result = apply_append("existing line", "new entry");
        assert_eq!(result, "existing line\nnew entry\n");
    }

    #[test]
    fn apply_append_empty_file() {
        let result = apply_append("", "first entry");
        assert_eq!(result, "first entry\n");
    }

    #[test]
    fn apply_append_with_frontmatter_and_body() {
        let old = "---\ntitle: X\n---\n\n# Log\n- entry one\n";
        let result = apply_append(old, "- entry two");
        assert_eq!(
            result,
            "---\ntitle: X\n---\n\n# Log\n- entry one\n- entry two\n"
        );
    }

    #[test]
    fn apply_append_with_frontmatter_and_no_body_inserts_a_separator() {
        let old = "---\ntitle: X\n---\n";
        let result = apply_append(old, "first body line");
        assert_eq!(result, "---\ntitle: X\n---\n\nfirst body line\n");
    }

    #[test]
    fn apply_append_never_touches_the_frontmatter_block() {
        let old = "---\ntitle: X\ndescription: a---b\n---\n\nBody\n";
        let result = apply_append(old, "more");
        assert!(result.starts_with("---\ntitle: X\ndescription: a---b\n---\n"));
    }

    #[test]
    fn apply_append_strips_a_trailing_newline_the_caller_included() {
        // Whether the caller's `text` ends in `\n` or not, the result always
        // has exactly one trailing newline — never two.
        let a = apply_append("body\n", "entry\n");
        let b = apply_append("body\n", "entry");
        assert_eq!(a, b);
        assert!(a.ends_with("entry\n") && !a.ends_with("entry\n\n"));
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
    ///
    /// `state_db` is `None` by default — matching `WriteDeps::state`'s own
    /// "disabled unless explicitly wired up" semantics — so every existing test
    /// that never calls `with_state_db` keeps exercising the no-rewrite path.
    /// Tests that exercise link rewriting call `with_state_db` to open a real
    /// (temp-file-backed) `StateDb` and seed it via `StateDb::replace_links`.
    struct Harness {
        embed: Arc<EmbedClient>,
        qdrant: Arc<QdrantStore>,
        canonical_data_path: PathBuf,
        include_patterns: globset::GlobSet,
        schema_cache: SharedSchemaCache,
        config: Arc<ResolvedConfig>,
        token: Option<String>,
        state_db: Option<StateDb>,
        /// Fresh, private to this `Harness` instance — this is the whole point
        /// of `WriteDeps::queue` becoming an injected dependency: every test
        /// that builds its own `Harness` gets its own `ReindexQueue`, so a path
        /// literal used by another test's harness cannot collide with this
        /// one's, structurally rather than by convention. See
        /// `same_path_literal_in_two_independent_tests_a`/`_b` for the
        /// regression guard this makes possible.
        reindex_queue: crate::reindex::ReindexQueue,
        /// Keeps the state DB's backing temp directory alive for as long as the
        /// harness lives. Deliberately a SEPARATE temp dir from the KB root
        /// (`canonical_data_path`) — the state DB file must never sit inside the
        /// git working copy, or every git-backed test's `git status --porcelain
        /// == ""` assertion would start seeing it as an untracked file.
        _state_db_dir: Option<tempfile::TempDir>,
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
                state_db: None,
                reindex_queue: crate::reindex::ReindexQueue::new(),
                _state_db_dir: None,
            }
        }

        /// Open a real, temp-file-backed `StateDb` (in its own directory, NOT
        /// the KB root — see `_state_db_dir`'s doc comment) and attach it, so
        /// `deps()` passes `Some` for `WriteDeps::state` and
        /// `write_document_move` performs link rewriting. Returns `self` so
        /// callers can chain it onto `Harness::new(..)`.
        async fn with_state_db(mut self) -> Self {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("state.db");
            self.state_db = Some(StateDb::new(&db_path).await.unwrap());
            self._state_db_dir = Some(dir);
            self
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
                queue: &self.reindex_queue,
                state: self.state_db.as_ref(),
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
            dest_path: None,
        }
    }

    fn make_move_req<'a>(
        source_rel: &'a str,
        dest_rel: &'a str,
        old_content: &'a str,
        new_content: &'a str,
    ) -> WriteRequest<'a> {
        WriteRequest {
            rel_path: source_rel,
            old_content,
            new_content,
            is_create: false,
            message: None,
            default_verb: "update",
            force_new: Some(true),
            operation: "test",
            expected_hash: None,
            dest_path: Some(dest_rel),
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

    /// Like [`git_commit_all`], but stages and commits several paths in one
    /// commit — used by the `move_directory` tests below to seed a source
    /// subtree with more than one document without a separate commit per file.
    fn git_commit_paths(work: &tempfile::TempDir, rel_paths: &[&str], message: &str) {
        for rel_path in rel_paths {
            std::process::Command::new("git")
                .args(["add", "--", rel_path])
                .current_dir(work.path())
                .output()
                .unwrap();
        }
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

    // -----------------------------------------------------------------------
    // (#179) frontmatter_patch / append end-to-end through `write_document`.
    // These prove the design decision in this module's content-mode-helpers
    // section: `apply_frontmatter_patch`/`apply_append` only ever COMPUTE
    // `new_content` — by the time it reaches `write_document`, a patch/append
    // write is indistinguishable from an ordinary full-replace write of that
    // same content, so schema validation and the pre-commit rollback apply to
    // it with no special-casing.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn frontmatter_patch_synced_write_updates_only_frontmatter_end_to_end() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original = "---\ntitle: Log\nstatus: draft\n---\n\n# Body\nunchanged\n";
        std::fs::write(work.path().join("log.md"), original).unwrap();
        git_commit_all(&work, "log.md", "add log.md");

        let harness = git_backed_harness(&work);
        let new_content = apply_frontmatter_patch(
            original,
            &[FrontmatterEdit::SetField {
                field: "status".into(),
                value: serde_json::json!("active"),
            }],
        )
        .unwrap();

        let mut req = make_req("log.md", &new_content, false);
        req.old_content = original;
        let success = write_document(&harness.deps(), req).await.unwrap();
        assert_eq!(success.outcome, WriteOutcome::Synced);

        let on_disk = std::fs::read_to_string(work.path().join("log.md")).unwrap();
        let (fm, body) = validate::parse_frontmatter_raw(&on_disk);
        assert_eq!(fm.get("status").unwrap(), "active");
        assert_eq!(fm.get("title").unwrap(), "Log");
        assert_eq!(body.trim(), "# Body\nunchanged");

        crate::reindex::test_support::assert_marked_dirty(&harness.reindex_queue, &["log.md"]);
    }

    #[tokio::test]
    async fn frontmatter_patch_validation_failure_leaves_the_file_untouched() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original = "---\ntitle: Log\nstatus: draft\n---\n\n# Body\n";
        std::fs::write(work.path().join("log.md"), original).unwrap();
        git_commit_all(&work, "log.md", "add log.md");

        let mut config = crate::mcp::make_test_resolved_config(work.path());
        Arc::get_mut(&mut config).unwrap().frontmatter.required = vec!["title".into()];
        let harness = Harness::new(&work, config);

        // Patch removes the very field the schema requires — this must fail
        // schema validation exactly like any other write, not silently commit.
        let new_content = apply_frontmatter_patch(
            original,
            &[FrontmatterEdit::RemoveField {
                field: "title".into(),
            }],
        )
        .unwrap();

        let mut req = make_req("log.md", &new_content, false);
        req.old_content = original;
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::Validation { result } => {
                assert!(result.field_errors.iter().any(|e| e.field == "title"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
        assert_eq!(
            std::fs::read_to_string(work.path().join("log.md")).unwrap(),
            original,
            "a rejected patch must never touch the file on disk"
        );
    }

    #[tokio::test]
    async fn frontmatter_patch_precommit_failure_rolls_back_to_the_original_content() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original = "---\ntitle: Log\nstatus: draft\n---\n\n# Body\n";
        std::fs::write(work.path().join("log.md"), original).unwrap();
        git_commit_all(&work, "log.md", "add log.md");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let harness = git_backed_harness(&work);

        let new_content = apply_frontmatter_patch(
            original,
            &[FrontmatterEdit::SetField {
                field: "status".into(),
                value: serde_json::json!("active"),
            }],
        )
        .unwrap();

        let mut req = make_req("log.md", &new_content, false);
        req.old_content = original;
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(rolled_back),
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }
        assert_eq!(
            std::fs::read_to_string(work.path().join("log.md")).unwrap(),
            original,
            "a patch write's rollback must participate exactly like any other edit's"
        );
        assert_eq!(head_before, head_sha(&work));
        assert_eq!(git_status(&work), "");
    }

    #[tokio::test]
    async fn append_synced_write_adds_to_the_end_of_the_body_end_to_end() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original = "---\ntitle: Log\n---\n\n# Log\n- entry one\n";
        std::fs::write(work.path().join("log.md"), original).unwrap();
        git_commit_all(&work, "log.md", "add log.md");

        let harness = git_backed_harness(&work);
        let new_content = apply_append(original, "- entry two");

        let mut req = make_req("log.md", &new_content, false);
        req.old_content = original;
        let success = write_document(&harness.deps(), req).await.unwrap();
        assert_eq!(success.outcome, WriteOutcome::Synced);

        assert_eq!(
            std::fs::read_to_string(work.path().join("log.md")).unwrap(),
            "---\ntitle: Log\n---\n\n# Log\n- entry one\n- entry two\n"
        );
    }

    #[tokio::test]
    async fn append_precommit_failure_rolls_back_to_the_original_content() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original = "---\ntitle: Log\n---\n\n# Log\n- entry one\n";
        std::fs::write(work.path().join("log.md"), original).unwrap();
        git_commit_all(&work, "log.md", "add log.md");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let harness = git_backed_harness(&work);
        let new_content = apply_append(original, "- entry two");

        let mut req = make_req("log.md", &new_content, false);
        req.old_content = original;
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(rolled_back),
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }
        assert_eq!(
            std::fs::read_to_string(work.path().join("log.md")).unwrap(),
            original,
            "an append write's rollback must participate exactly like any other edit's"
        );
        assert_eq!(head_before, head_sha(&work));
        assert_eq!(git_status(&work), "");
    }

    /// #147: the "rollback ITSELF also failed" branch of `write_document`'s
    /// CREATE path (`rolled_back: false`) — previously exercised only by
    /// `delete_document`'s equivalent test. Mirrors that test's technique:
    /// point the harness at a plain temp directory with no `.git` at all, so
    /// `git add` fails at `commit_and_sync`'s very first step (a `PreCommit`
    /// failure, same as any other pre-commit failure), and the create
    /// rollback's SECOND step — `git::unstage`, which runs after
    /// `tokio::fs::remove_file` already succeeded — fails too, because there
    /// is no repository to run `git reset` against.
    #[tokio::test]
    async fn create_rollback_failure_with_no_git_repo_reports_rolled_back_false() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let req = make_req(
            "docs/new-no-repo.md",
            "---\ntitle: New\n---\n\n# Body\n",
            true,
        );
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(!rolled_back),
            other => panic!("expected PreCommitFailed{{rolled_back: false}}, got {other:?}"),
        }
        // `remove_file` (the first of the two rollback steps) succeeded — the
        // filesystem write really is undone. Only `git::unstage` (the second
        // step) failed, which is exactly what makes this the "rollback
        // itself also failed" case rather than a clean rollback.
        assert!(!tmp.path().join("docs/new-no-repo.md").exists());
    }

    /// #147: the same "rollback itself also failed" branch, but for
    /// `write_document`'s EDIT path, which rolls back via a single call to
    /// `git::restore_from_head` instead of create's two-step remove+unstage —
    /// a genuinely different call to fail, so the create test above does not
    /// cover it.
    #[tokio::test]
    async fn edit_rollback_failure_with_no_git_repo_reports_rolled_back_false() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(
            sub.join("edit-me-no-repo.md"),
            "---\ntitle: Old\n---\n# Old",
        )
        .unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let req = make_req(
            "docs/edit-me-no-repo.md",
            "---\ntitle: New\n---\n# New",
            false,
        );
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(!rolled_back),
            other => panic!("expected PreCommitFailed{{rolled_back: false}}, got {other:?}"),
        }
        // The overwrite already landed on disk (an edit writes in place, with
        // no separate "create the new file" step to undo) and `git restore`
        // cannot put the old content back with no HEAD to restore from — the
        // file is left holding the new, uncommitted content, which IS the
        // inconsistency `rolled_back: false` is reporting.
        assert_eq!(
            std::fs::read_to_string(sub.join("edit-me-no-repo.md")).unwrap(),
            "---\ntitle: New\n---\n# New"
        );
    }

    /// #142 regression: `expected_hash` must be re-verified against the file's
    /// LIVE on-disk content immediately before the overwrite, not just once,
    /// early, against the caller-supplied `old_content`. Simulates the failure
    /// scenario from the issue: a caller reads `original`, and by the time its
    /// write finally reaches the filesystem, something else (a webhook merge, in
    /// production) has already changed the working-tree content — here modeled
    /// directly as a second on-disk write between request construction and the
    /// call, with `expected_hash` still computed from the ORIGINAL content the
    /// caller actually read. Before the fix there is only the early check
    /// (which the caller's own stale `old_content` trivially satisfies), so the
    /// write silently clobbers the concurrent change; after the fix the
    /// re-check catches the live mismatch and the file is left untouched.
    #[tokio::test]
    async fn stale_hash_re_check_catches_a_change_made_after_the_first_check() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original =
            "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
        std::fs::write(work.path().join("edit-me.md"), original).unwrap();
        git_commit_all(&work, "edit-me.md", "add edit-me.md");
        let head_before = head_sha(&work);

        let harness = git_backed_harness(&work);

        // The hash the caller computed when it read `original` — still valid
        // against `old_content` below, which is why the FIRST check (step 1)
        // lets this through.
        let expected_hash = crate::ingest::compute_hash_from_bytes(original.as_bytes());

        // Something else changes the file on disk after the caller read
        // `original` but before this write reaches the filesystem — a webhook
        // merge landing mid-flight, in production.
        let concurrent = "---\ntitle: Concurrent\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Concurrent body\n";
        std::fs::write(work.path().join("edit-me.md"), concurrent).unwrap();

        let mut req = make_req(
            "edit-me.md",
            "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New body\n",
            false,
        );
        req.old_content = original;
        req.expected_hash = Some(&expected_hash);

        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::StaleHash { expected, actual } => {
                assert_eq!(expected, expected_hash);
                assert_eq!(
                    actual,
                    crate::ingest::compute_hash_from_bytes(concurrent.as_bytes()),
                    "the re-check must hash the LIVE on-disk content, not `old_content` again"
                );
            }
            other => panic!("expected StaleHash, got {other:?}"),
        }

        // The concurrent write must survive untouched — that's the whole point:
        // it must not be silently clobbered by the stale caller's edit.
        assert_eq!(
            std::fs::read_to_string(work.path().join("edit-me.md")).unwrap(),
            concurrent
        );
        assert_eq!(head_before, head_sha(&work), "no commit must be made");
    }

    /// Same #142 regression, for `write_document_move`'s SOURCE: the stale-read
    /// guard at step 3 runs before `validate::validate_content` (which can exec
    /// an arbitrarily slow `lint_command`) and before GIT_LOCK is acquired, so a
    /// concurrent change to the source's on-disk content in that window must
    /// still be caught before the move writes the destination or removes the
    /// source — not silently carried through as if the stale read were current.
    #[tokio::test]
    async fn move_stale_hash_re_check_catches_a_change_made_after_the_first_check() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let harness = git_backed_harness(&work);

        let expected_hash = crate::ingest::compute_hash_from_bytes(original.as_bytes());

        // Concurrent change to the SOURCE after the caller's read but before the
        // move's filesystem work runs — committed, same as a real webhook merge
        // would (see `webhook.rs`), so the working tree is clean afterward and
        // `git_status` below actually exercises the move's own rollback rather
        // than an artifact of this test's own setup.
        let concurrent =
            "---\ntitle: Concurrent\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Concurrent\n";
        std::fs::write(work.path().join("old/loc.md"), concurrent).unwrap();
        git_commit_all(&work, "old/loc.md", "concurrent change to old/loc.md");
        let head_before = head_sha(&work);

        let mut req = make_move_req("old/loc.md", "new/loc.md", original, original);
        req.expected_hash = Some(&expected_hash);

        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::StaleHash { expected, actual } => {
                assert_eq!(expected, expected_hash);
                assert_eq!(
                    actual,
                    crate::ingest::compute_hash_from_bytes(concurrent.as_bytes()),
                    "the re-check must hash the LIVE on-disk source, not `old_content` again"
                );
            }
            other => panic!("expected StaleHash, got {other:?}"),
        }

        assert_eq!(
            std::fs::read_to_string(work.path().join("old/loc.md")).unwrap(),
            concurrent,
            "source must be left exactly as the concurrent writer left it"
        );
        assert!(
            !work.path().join("new/loc.md").exists(),
            "destination must never be created when the re-check catches a stale source"
        );
        assert_eq!(head_before, head_sha(&work), "no commit must be made");
        assert_eq!(git_status(&work), "");
    }

    #[tokio::test]
    async fn create_synced_write_marks_the_path_dirty_and_returns_a_diff() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let harness = git_backed_harness(&work);

        // `harness.reindex_queue` is private to this `Harness` instance — no
        // other test's writes can land on it, so the path literal below needs
        // no cross-test uniqueness of its own (see `Harness::reindex_queue`'s
        // doc comment and the `same_path_literal_*` regression guard near the
        // bottom of this module).
        let req = make_req(
            "docs/queued-write-core-test.md",
            "---\ntitle: Queued\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n",
            true,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();
        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(!success.sha.is_empty());
        assert!(success.diff.contains("+title: Queued"));

        crate::reindex::test_support::assert_marked_dirty(
            &harness.reindex_queue,
            &["docs/queued-write-core-test.md"],
        );
    }

    // Regression guard for the original bug: before the queue became an
    // injected dependency, both tests below would have collided on
    // `crate::reindex::REINDEX_QUEUE` — a single process-wide `HashSet<PathBuf>`
    // shared by the whole test binary. Marking an already-pending path is a
    // no-op on a `HashSet`'s cardinality, so whichever of these two ran second
    // would silently fail to observe its own write having marked its path
    // dirty. That failure showed up only under `cargo test --
    // --test-threads=1`, where libtest's alphabetical run order made the
    // collision deterministic (a plain `cargo test` run could get lucky and
    // interleave them apart). Each test here now builds its own `Harness` —
    // and therefore its own private `ReindexQueue` (see
    // `Harness::reindex_queue`'s doc comment) — so using the IDENTICAL path
    // literal in both is not just safe, it is the point: this is the case that
    // used to fail and now provably does not.

    #[tokio::test]
    async fn same_path_literal_in_two_independent_tests_a() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let harness = git_backed_harness(&work);

        let req = make_req(
            "docs/same-literal-regression-guard.md",
            "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n",
            true,
        );
        write_document(&harness.deps(), req).await.unwrap();

        crate::reindex::test_support::assert_marked_dirty(
            &harness.reindex_queue,
            &["docs/same-literal-regression-guard.md"],
        );
    }

    #[tokio::test]
    async fn same_path_literal_in_two_independent_tests_b() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let harness = git_backed_harness(&work);

        let req = make_req(
            "docs/same-literal-regression-guard.md",
            "---\ntitle: B\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n",
            true,
        );
        write_document(&harness.deps(), req).await.unwrap();

        crate::reindex::test_support::assert_marked_dirty(
            &harness.reindex_queue,
            &["docs/same-literal-regression-guard.md"],
        );
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

        // #150: this is the ONLY trigger for the reindex worker to purge the
        // deleted document's Qdrant points and state rows (see
        // `ingest::index_paths`'s missing-file branch) — a regression that
        // dropped this call would leave the document searchable forever,
        // with every other assertion above still green.
        crate::reindex::test_support::assert_marked_dirty(&harness.reindex_queue, &["doomed.md"]);
    }

    /// #150: `delete_document`'s OTHER success path — commit AND push both
    /// succeed — was, per the issue, never exercised by any test at all
    /// (only the `CommittedPendingSync` branch above was reached, and even
    /// that omitted this assertion). Mirrors `create_synced_write_marks_the_path_dirty_and_returns_a_diff`'s
    /// pattern for the create path.
    #[tokio::test]
    async fn delete_synced_write_marks_the_path_dirty() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::write(
            work.path().join("doomed-synced.md"),
            "---\ntitle: D\n---\n\n# Body\n",
        )
        .unwrap();
        git_commit_all(&work, "doomed-synced.md", "add doomed-synced.md");

        let harness = git_backed_harness(&work);

        let success = delete_document(&harness.deps(), "doomed-synced.md", None)
            .await
            .unwrap();
        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(!work.path().join("doomed-synced.md").exists());

        crate::reindex::test_support::assert_marked_dirty(
            &harness.reindex_queue,
            &["doomed-synced.md"],
        );
    }

    // -----------------------------------------------------------------------
    // #181 / #229 — delete_document warns about inbound links instead of
    // silently orphaning them, AND (#229) surfaces the same referencing paths
    // on `WriteSuccess::referencing_paths` so a caller with no access to
    // server logs can see them too. `CapturedLogs` below is a minimal
    // `tracing_subscriber::fmt::MakeWriter` that redirects exactly the calls
    // made while its guard is alive into an in-memory buffer a test can
    // assert on — `tracing::subscriber::set_default`'s guard scopes it to
    // this one call, so it cannot leak into (or be polluted by) any other
    // test's logging.
    // -----------------------------------------------------------------------

    #[derive(Clone, Default)]
    struct CapturedLogs(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for CapturedLogs {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLogs {
        type Writer = Self;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl CapturedLogs {
        fn text(&self) -> String {
            String::from_utf8_lossy(&self.0.lock().unwrap()).into_owned()
        }
    }

    /// Scoped to `WARN` and above so that unrelated `info!`/`debug!` output
    /// elsewhere in the write pipeline (or in `git.rs`) can never show up in
    /// `CapturedLogs::text()` and be mistaken for the inbound-link warning
    /// this module's tests care about.
    fn capture_warnings() -> (CapturedLogs, tracing::subscriber::DefaultGuard) {
        let captured = CapturedLogs::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(captured.clone())
            .with_max_level(tracing::Level::WARN)
            .without_time()
            .with_target(false)
            .finish();
        let guard = tracing::subscriber::set_default(subscriber);
        (captured, guard)
    }

    #[tokio::test]
    async fn delete_warns_about_inbound_links_but_does_not_refuse() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::write(
            work.path().join("linked.md"),
            "---\ntitle: Linked\n---\n\n# Body\n",
        )
        .unwrap();
        git_commit_all(&work, "linked.md", "add linked.md");

        let harness = git_backed_harness_with_state_db(&work).await;
        // `referencer.md` need not exist on disk — the check queries the
        // reverse-link INDEX, not the filesystem, same as
        // `write_document_move`'s step 10.5.
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencer.md",
                "markdown",
                &[("linked.md".to_string(), None)],
            )
            .await
            .unwrap();

        let (captured, guard) = capture_warnings();
        let success = delete_document(&harness.deps(), "linked.md", None)
            .await
            .expect("an inbound link must WARN, not refuse the delete — see #181's PR notes");
        drop(guard);

        assert_eq!(
            success.outcome,
            WriteOutcome::Synced,
            "the delete itself must still succeed"
        );
        let log_text = captured.text();
        assert!(
            log_text.contains("referencer.md"),
            "expected a warning naming the referencing document, got log: {log_text:?}"
        );
        assert!(
            log_text.contains("linked.md"),
            "expected the warning to name the document being deleted too, got log: {log_text:?}"
        );
        assert_eq!(
            success.referencing_paths,
            vec!["referencer.md".to_string()],
            "#229: the same referencing path must also reach the caller via \
             WriteSuccess, not just the server log"
        );
    }

    #[tokio::test]
    async fn delete_with_no_inbound_links_logs_no_warning() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::write(
            work.path().join("unlinked.md"),
            "---\ntitle: Unlinked\n---\n\n# Body\n",
        )
        .unwrap();
        git_commit_all(&work, "unlinked.md", "add unlinked.md");

        // With a state DB wired up but no `document_links` row targeting this
        // document, the query must come back empty and stay silent.
        let harness = git_backed_harness_with_state_db(&work).await;

        let (captured, guard) = capture_warnings();
        let success = delete_document(&harness.deps(), "unlinked.md", None)
            .await
            .unwrap();
        drop(guard);

        assert!(
            captured.text().is_empty(),
            "no inbound links means no warning, got log: {:?}",
            captured.text()
        );
        assert!(
            success.referencing_paths.is_empty(),
            "#229: no inbound links means an empty referencing_paths too"
        );
    }

    #[tokio::test]
    async fn delete_with_no_state_db_skips_the_inbound_link_check_silently() {
        // `WriteDeps::state == None` (no `with_state_db`) must not be treated
        // as "querying failed" — it is a normal, documented degraded mode
        // (see that field's doc comment), so the delete must proceed exactly
        // as it always has, with no warning and no error.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::write(
            work.path().join("no-state-db.md"),
            "---\ntitle: No State DB\n---\n\n# Body\n",
        )
        .unwrap();
        git_commit_all(&work, "no-state-db.md", "add no-state-db.md");

        let harness = git_backed_harness(&work);

        let (captured, guard) = capture_warnings();
        let success = delete_document(&harness.deps(), "no-state-db.md", None)
            .await
            .unwrap();
        drop(guard);

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(captured.text().is_empty());
        assert!(
            success.referencing_paths.is_empty(),
            "#229: with no state DB wired up, referencing_paths must stay empty, \
             same as the warning it mirrors"
        );
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

    // -----------------------------------------------------------------------
    // WriteRequest::dest_path (document MOVE) — write_document_move
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn successful_move_relocates_the_file_in_one_commit_and_marks_both_paths_dirty() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc-move-core-test-src.md"), original).unwrap();
        git_commit_all(
            &work,
            "old/loc-move-core-test-src.md",
            "add old/loc-move-core-test-src.md",
        );
        let head_before = head_sha(&work);

        let harness = git_backed_harness(&work);

        let req = make_move_req(
            "old/loc-move-core-test-src.md",
            "new/loc-moved-write-core-test.md",
            original,
            original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(!success.sha.is_empty());
        assert_ne!(
            success.sha, head_before,
            "the move must produce a new commit"
        );
        assert!(
            !work.path().join("old/loc-move-core-test-src.md").exists(),
            "source must be gone after a successful move"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("new/loc-moved-write-core-test.md")).unwrap(),
            original
        );
        // The removal and the addition both landed in the single commit — nothing
        // left staged or dangling in the working tree afterward.
        assert_eq!(git_status(&work), "");

        crate::reindex::test_support::assert_marked_dirty(
            &harness.reindex_queue,
            &[
                "old/loc-move-core-test-src.md",
                "new/loc-moved-write-core-test.md",
            ],
        );
    }

    #[tokio::test]
    async fn move_with_a_stale_expected_hash_is_rejected_and_mutates_nothing() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");
        let head_before = head_sha(&work);

        let harness = git_backed_harness(&work);

        let stale_hash = crate::ingest::compute_hash_from_bytes(b"not the current content");
        let mut req = make_move_req("old/loc.md", "new/loc.md", original, original);
        req.expected_hash = Some(&stale_hash);

        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::StaleHash { expected, actual } => {
                assert_eq!(expected, stale_hash);
                assert_ne!(actual, stale_hash);
            }
            other => panic!("expected StaleHash, got {other:?}"),
        }
        assert!(
            work.path().join("old/loc.md").exists(),
            "source must be untouched when the expected_hash is stale"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("old/loc.md")).unwrap(),
            original
        );
        assert!(
            !work.path().join("new/loc.md").exists(),
            "destination must never be created when the expected_hash is stale"
        );
        assert_eq!(head_before, head_sha(&work), "no commit must be made");
        assert_eq!(git_status(&work), "");
    }

    #[tokio::test]
    async fn move_with_a_matching_expected_hash_proceeds_normally() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let harness = git_backed_harness(&work);

        let correct_hash = crate::ingest::compute_hash_from_bytes(original.as_bytes());
        let mut req = make_move_req(
            "old/loc.md",
            "new/loc-matching-hash-test.md",
            original,
            original,
        );
        req.expected_hash = Some(&correct_hash);

        let success = write_document(&harness.deps(), req).await.unwrap();
        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(!work.path().join("old/loc.md").exists());
        assert_eq!(
            std::fs::read_to_string(work.path().join("new/loc-matching-hash-test.md")).unwrap(),
            original
        );
    }

    #[tokio::test]
    async fn move_to_an_existing_destination_reports_already_exists_and_mutates_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let source_dir = tmp.path().join("docs");
        std::fs::create_dir_all(&source_dir).unwrap();
        let original = "---\ntitle: T\n---\n# Body";
        std::fs::write(source_dir.join("source.md"), original).unwrap();
        let dest_dir = tmp.path().join("other");
        std::fs::create_dir_all(&dest_dir).unwrap();
        std::fs::write(dest_dir.join("dest.md"), "# Already here").unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let req = make_move_req("docs/source.md", "other/dest.md", original, original);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(matches!(err, WriteError::AlreadyExists), "got {err:?}");
        assert_eq!(
            std::fs::read_to_string(source_dir.join("source.md")).unwrap(),
            original,
            "source must be untouched when the destination already exists"
        );
        assert_eq!(
            std::fs::read_to_string(dest_dir.join("dest.md")).unwrap(),
            "# Already here",
            "the pre-existing destination file must be untouched"
        );
    }

    #[tokio::test]
    async fn move_validates_against_the_destination_schema_not_the_sources() {
        let tmp = tempfile::tempdir().unwrap();
        let source_dir = tmp.path().join("loose");
        std::fs::create_dir_all(&source_dir).unwrap();
        // Valid under the (schema-less) source directory, but missing a field the
        // destination directory's schema requires.
        let content = "---\ntitle: T\n---\n# Body";
        std::fs::write(source_dir.join("source.md"), content).unwrap();

        let dest_dir = tmp.path().join("strict");
        std::fs::create_dir_all(&dest_dir).unwrap();
        std::fs::write(
            dest_dir.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  strict_field:\n    required: true\n",
        )
        .unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let req = make_move_req("loose/source.md", "strict/dest.md", content, content);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::Validation { result } => {
                assert!(!result.valid);
                assert!(
                    result
                        .field_errors
                        .iter()
                        .any(|e| e.field == "strict_field"),
                    "expected a strict_field error, got {:?}",
                    result.field_errors
                );
            }
            other => panic!("expected Validation, got {other:?}"),
        }
        assert!(
            source_dir.join("source.md").exists(),
            "source must be untouched when destination validation fails"
        );
        assert!(
            !dest_dir.join("dest.md").exists(),
            "nothing should be written to the destination when validation fails"
        );
    }

    #[tokio::test]
    async fn move_with_a_frozen_source_directory_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let source_dir = tmp.path().join("frozen-src");
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::write(
            source_dir.join(crate::schema::SCHEMA_FILE_NAME),
            "not: [valid: yaml",
        )
        .unwrap();
        let content = "---\ntitle: T\n---\n# Body";
        std::fs::write(source_dir.join("source.md"), content).unwrap();
        let dest_dir = tmp.path().join("dest-ok");
        std::fs::create_dir_all(&dest_dir).unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let req = make_move_req("frozen-src/source.md", "dest-ok/dest.md", content, content);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(matches!(err, WriteError::Frozen { .. }), "got {err:?}");
        assert!(source_dir.join("source.md").exists());
        assert!(!dest_dir.join("dest.md").exists());
    }

    #[tokio::test]
    async fn move_with_a_frozen_destination_directory_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let source_dir = tmp.path().join("source-ok");
        std::fs::create_dir_all(&source_dir).unwrap();
        let content = "---\ntitle: T\n---\n# Body";
        std::fs::write(source_dir.join("source.md"), content).unwrap();
        let dest_dir = tmp.path().join("frozen-dest");
        std::fs::create_dir_all(&dest_dir).unwrap();
        std::fs::write(
            dest_dir.join(crate::schema::SCHEMA_FILE_NAME),
            "not: [valid: yaml",
        )
        .unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let req = make_move_req(
            "source-ok/source.md",
            "frozen-dest/dest.md",
            content,
            content,
        );
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(matches!(err, WriteError::Frozen { .. }), "got {err:?}");
        assert!(source_dir.join("source.md").exists());
        assert!(!dest_dir.join("dest.md").exists());
    }

    #[tokio::test]
    async fn move_precommit_failure_rolls_back_both_halves() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let harness = git_backed_harness(&work);

        let req = make_move_req("old/loc.md", "new/loc.md", original, original);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(rolled_back),
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }
        assert!(
            work.path().join("old/loc.md").exists(),
            "source must be restored after a rolled-back move"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("old/loc.md")).unwrap(),
            original
        );
        assert!(
            !work.path().join("new/loc.md").exists(),
            "destination must be gone after a rolled-back move"
        );
        assert_eq!(head_before, head_sha(&work));
        assert_eq!(git_status(&work), "");
    }

    /// #147: `write_document_move`'s own "rollback itself also failed" branch
    /// (`rolled_back: false`) — same no-git-repo technique as the create/edit
    /// tests above and `delete_document`'s existing test. `rolled_back` here
    /// is `source_restore.is_ok() && dest_rollback.is_ok() &&
    /// rewrite_restore_failures.is_empty()`: with no `.git` at all, `git add`
    /// fails first (`PreCommit`), and then BOTH `source_restore`
    /// (`git::restore_from_head`) and the git half of `dest_rollback`
    /// (`git::unstage`, reached after its own `tokio::fs::remove_file` step
    /// already succeeded) fail too, since neither has a repository to run
    /// against — so `rolled_back` ends up `false` on two independent counts
    /// at once, not just one.
    #[tokio::test]
    async fn move_rollback_failure_with_no_git_repo_reports_rolled_back_false() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("old");
        std::fs::create_dir_all(&sub).unwrap();
        let source_original = "---\ntitle: Move Me\n---\n\n# Body\n";
        std::fs::write(sub.join("loc.md"), source_original).unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let req = make_move_req("old/loc.md", "new/loc.md", source_original, source_original);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(!rolled_back),
            other => panic!("expected PreCommitFailed{{rolled_back: false}}, got {other:?}"),
        }
        // The destination copy was written, then successfully removed during
        // rollback (`remove_file` needs no git repo) — but `restore_from_head`
        // on the source can't run with no HEAD to restore from, so the
        // source is left gone too. Both paths now missing IS the
        // inconsistency `rolled_back: false` reports.
        assert!(!tmp.path().join("new/loc.md").exists());
        assert!(!sub.join("loc.md").exists());
    }

    #[tokio::test]
    async fn move_with_a_content_change_writes_the_new_content_to_the_destination() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let original =
            "---\ntitle: Old Title\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
        std::fs::write(work.path().join("old/loc.md"), original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let harness = git_backed_harness(&work);

        let new_content =
            "---\ntitle: New Title\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New body\n";
        let req = make_move_req("old/loc.md", "new/loc.md", original, new_content);
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(!work.path().join("old/loc.md").exists());
        assert_eq!(
            std::fs::read_to_string(work.path().join("new/loc.md")).unwrap(),
            new_content,
            "the destination must hold the NEW content, not a copy of the source's old content"
        );
        assert!(success.diff.contains("+title: New Title"));
        assert!(success.diff.contains("-title: Old Title"));
    }

    // -----------------------------------------------------------------------
    // GIT_LOCK hoist regression (data-loss finding fix): `write_document_move`
    // must acquire `GIT_LOCK` before its destination write / source removal /
    // referencing-document rewrite, not just before the commit, and must hold
    // that ONE guard across all of it. Proven at runtime rather than merely
    // structurally: another holder of `GIT_LOCK` must observably block the
    // call, and releasing that holder must let it proceed to completion
    // without hanging (a hang would mean something reachable from this call
    // tried to reacquire the already-held, non-reentrant mutex).
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn write_document_move_blocks_while_git_lock_is_externally_held_then_completes() {
        let tmp = tempfile::tempdir().unwrap();
        let original = "---\ntitle: A\n---\n# A";
        std::fs::write(tmp.path().join("source-lockcheck.md"), original).unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        // Hold GIT_LOCK exactly as a concurrent writer, the webhook handler, or
        // the reindex worker would.
        let held = git::lock_git().await;

        let req = make_move_req(
            "source-lockcheck.md",
            "dest-lockcheck.md",
            original,
            original,
        );
        let deps = harness.deps();
        let move_fut = write_document(&deps, req);
        tokio::pin!(move_fut);
        let still_blocked =
            tokio::time::timeout(std::time::Duration::from_millis(200), &mut move_fut).await;
        assert!(
            still_blocked.is_err(),
            "write_document_move must block on GIT_LOCK (acquired ahead of the destination \
             write, per the finding's fix) while another holder has it"
        );

        drop(held);
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), move_fut)
            .await
            .expect(
                "write_document_move must proceed to completion once GIT_LOCK is released, not \
                 hang against its own held guard -- a hang here would mean this non-reentrant \
                 mutex is being acquired a second time somewhere in the call chain",
            );
        // Not git-backed, so the commit itself fails fast once the lock is free --
        // the point of this test is that nothing deadlocks, not the outcome.
        match result.unwrap_err() {
            WriteError::PreCommitFailed { .. } => {}
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Move + incoming-link rewrite (WriteDeps::state, `document_links` reverse
    // lookup, `ingest::find_markdown_link_occurrences`/`relativize_md_path`)
    // -----------------------------------------------------------------------

    /// A git-backed harness whose `WriteDeps::state` is wired up to a real,
    /// temp-file-backed `StateDb` (see `Harness::with_state_db`'s doc comment
    /// for why it lives outside the git working copy), so `write_document_move`
    /// actually performs link rewriting.
    async fn git_backed_harness_with_state_db(work: &tempfile::TempDir) -> Harness {
        git_backed_harness(work).with_state_db().await
    }

    #[tokio::test]
    async fn move_rewrites_a_referencing_documents_link_in_the_same_commit() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let referencing_original = "---\ntitle: Referencer\ndescription: d\ntype: guide\n\
             tags: [t]\n---\n\nSee [the moved doc](old/loc.md) for more.\n";
        std::fs::write(work.path().join("referencing.md"), referencing_original).unwrap();
        git_commit_all(&work, "referencing.md", "add referencing.md");
        let head_before = head_sha(&work);

        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencing.md",
                "markdown",
                &[("old/loc.md".to_string(), None)],
            )
            .await
            .unwrap();

        let req = make_move_req(
            "old/loc.md",
            "new/loc-rewrite-test1.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert_eq!(success.rewritten_paths, vec!["referencing.md".to_string()]);

        let referencing_after =
            std::fs::read_to_string(work.path().join("referencing.md")).unwrap();
        assert!(
            referencing_after.contains("[the moved doc](new/loc-rewrite-test1.md)"),
            "referencing document's link must point at the new location, got: {referencing_after}"
        );
        assert!(!referencing_after.contains("old/loc.md"));

        // Both halves of the move AND the referencing-document rewrite landed in
        // exactly ONE commit: the working tree is clean and HEAD moved exactly
        // once (`success.sha` is the only new commit, matching this test's own
        // assertions on the file contents above having already landed).
        assert_eq!(git_status(&work), "");
        assert_ne!(head_before, head_sha(&work));
        assert_eq!(success.sha, head_sha(&work));
    }

    #[tokio::test]
    async fn move_rewrite_relativizes_correctly_for_a_referencing_document_elsewhere() {
        // The case a naive string substitution gets wrong: the referencing
        // document lives in a THIRD directory, unrelated to both the source's
        // and the destination's, so the correct replacement text is neither the
        // raw destination path nor a copy of the old relative text — it must be
        // freshly computed from the referencing document's own location,
        // climbing up ("../../") before descending back down.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("docs/sub")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("docs/sub/loc.md"), source_original).unwrap();
        git_commit_all(&work, "docs/sub/loc.md", "add docs/sub/loc.md");

        std::fs::create_dir_all(work.path().join("other/deep")).unwrap();
        let referencing_original = "---\ntitle: Referencer\ndescription: d\ntype: guide\n\
             tags: [t]\n---\n\nSee [it](../../docs/sub/loc.md) for more.\n";
        std::fs::write(work.path().join("other/deep/ref.md"), referencing_original).unwrap();
        git_commit_all(&work, "other/deep/ref.md", "add other/deep/ref.md");

        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "other/deep/ref.md",
                "markdown",
                &[("docs/sub/loc.md".to_string(), None)],
            )
            .await
            .unwrap();

        let req = make_move_req(
            "docs/sub/loc.md",
            "archive/2024/loc.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert_eq!(
            success.rewritten_paths,
            vec!["other/deep/ref.md".to_string()]
        );

        let referencing_after =
            std::fs::read_to_string(work.path().join("other/deep/ref.md")).unwrap();
        assert!(
            referencing_after.contains("[it](../../archive/2024/loc.md)"),
            "expected a correctly relativized (climbing) link, got: {referencing_after}"
        );
    }

    #[tokio::test]
    async fn move_rewrite_skips_links_inside_fences_and_code_spans() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let referencing_original = "---\ntitle: Referencer\ndescription: d\ntype: guide\n\
             tags: [t]\n---\n\nSee [Real Link](old/loc.md) for docs.\n\n\
             ```md\n[Fenced](old/loc.md)\n```\n\n\
             Use `[Code](old/loc.md)` literally.\n";
        std::fs::write(work.path().join("referencing.md"), referencing_original).unwrap();
        git_commit_all(&work, "referencing.md", "add referencing.md");

        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencing.md",
                "markdown",
                &[("old/loc.md".to_string(), None)],
            )
            .await
            .unwrap();

        let req = make_move_req(
            "old/loc.md",
            "new/loc-rewrite-test3.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.rewritten_paths, vec!["referencing.md".to_string()]);
        let referencing_after =
            std::fs::read_to_string(work.path().join("referencing.md")).unwrap();
        assert!(
            referencing_after.contains("[Real Link](new/loc-rewrite-test3.md)"),
            "the real inline link must be rewritten, got: {referencing_after}"
        );
        assert!(
            referencing_after.contains("[Fenced](old/loc.md)"),
            "a link inside a fenced code block must NOT be rewritten, got: {referencing_after}"
        );
        assert!(
            referencing_after.contains("`[Code](old/loc.md)`"),
            "a link inside an inline code span must NOT be rewritten, got: {referencing_after}"
        );
    }

    // -----------------------------------------------------------------------
    // Wiki pipe-alias links `[[target|Display]]` — fix #131
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn move_rewrites_a_referencing_documents_pipe_alias_link_with_alias_preserved() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let referencing_original = "---\ntitle: Referencer\ndescription: d\ntype: guide\n\
             tags: [t]\n---\n\nSee [[old/loc.md|Display Text]] for more.\n";
        std::fs::write(work.path().join("referencing.md"), referencing_original).unwrap();
        git_commit_all(&work, "referencing.md", "add referencing.md");

        let harness = git_backed_harness_with_state_db(&work).await;
        // Seeded directly, matching every other incoming-link-rewrite test in this
        // file — `document_links` is populated by `ingest::extract_markdown_links`
        // in production, which does NOT yet recognize this syntax (fix #131 is
        // scoped to the rewriter; the extractor change is a separate, out-of-scope
        // fix). Seeding the edge here isolates what THIS module is responsible
        // for: once a referencing document is known/visited, its pipe-alias
        // occurrences must be found and correctly rewritten with the alias intact.
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencing.md",
                "markdown",
                &[("old/loc.md".to_string(), None)],
            )
            .await
            .unwrap();

        let req = make_move_req(
            "old/loc.md",
            "new/loc-alias-test1.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert_eq!(success.rewritten_paths, vec!["referencing.md".to_string()]);

        let referencing_after =
            std::fs::read_to_string(work.path().join("referencing.md")).unwrap();
        assert!(
            referencing_after.contains("[[new/loc-alias-test1.md|Display Text]]"),
            "the pipe-alias link's target must be rewritten to the new location while its \
             alias survives byte-identical, got: {referencing_after}"
        );
        assert!(!referencing_after.contains("old/loc.md"));
    }

    #[tokio::test]
    async fn move_rewrites_pipe_alias_link_whose_alias_contains_path_like_characters() {
        // The alias itself may contain `/` and look like a path — the rewriter
        // must split on the FIRST `|` only and never mistake alias text for
        // more of the target.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let referencing_original = "---\ntitle: Referencer\ndescription: d\ntype: guide\n\
             tags: [t]\n---\n\nSee [[old/loc.md|old/style/looking/alias]] for more.\n";
        std::fs::write(work.path().join("referencing.md"), referencing_original).unwrap();
        git_commit_all(&work, "referencing.md", "add referencing.md");

        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencing.md",
                "markdown",
                &[("old/loc.md".to_string(), None)],
            )
            .await
            .unwrap();

        let req = make_move_req(
            "old/loc.md",
            "new/loc-alias-test2.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.rewritten_paths, vec!["referencing.md".to_string()]);
        let referencing_after =
            std::fs::read_to_string(work.path().join("referencing.md")).unwrap();
        assert!(
            referencing_after.contains("[[new/loc-alias-test2.md|old/style/looking/alias]]"),
            "the path-like alias must survive byte-identical while only the target is \
             rewritten, got: {referencing_after}"
        );
        assert!(!referencing_after.contains("old/loc.md"));
    }

    #[tokio::test]
    async fn move_rewrite_skips_pipe_alias_links_inside_fences_and_code_spans() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let referencing_original = "---\ntitle: Referencer\ndescription: d\ntype: guide\n\
             tags: [t]\n---\n\nSee [[old/loc.md|Real Alias]] for docs.\n\n\
             ```md\n[[old/loc.md|Fenced Alias]]\n```\n\n\
             Use `[[old/loc.md|Code Alias]]` literally.\n";
        std::fs::write(work.path().join("referencing.md"), referencing_original).unwrap();
        git_commit_all(&work, "referencing.md", "add referencing.md");

        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencing.md",
                "markdown",
                &[("old/loc.md".to_string(), None)],
            )
            .await
            .unwrap();

        let req = make_move_req(
            "old/loc.md",
            "new/loc-alias-test3.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.rewritten_paths, vec!["referencing.md".to_string()]);
        let referencing_after =
            std::fs::read_to_string(work.path().join("referencing.md")).unwrap();
        assert!(
            referencing_after.contains("[[new/loc-alias-test3.md|Real Alias]]"),
            "the real pipe-alias link must be rewritten, got: {referencing_after}"
        );
        assert!(
            referencing_after.contains("[[old/loc.md|Fenced Alias]]"),
            "a pipe-alias link inside a fenced code block must NOT be rewritten, got: \
             {referencing_after}"
        );
        assert!(
            referencing_after.contains("`[[old/loc.md|Code Alias]]`"),
            "a pipe-alias link inside an inline code span must NOT be rewritten, got: \
             {referencing_after}"
        );
    }

    #[tokio::test]
    async fn move_rewrites_a_pipe_alias_self_reference_with_alias_preserved() {
        // The moved document's own outbound pipe-alias link to itself — exercised
        // through `rewrite_outbound_links`, which scans `new_content` directly and
        // has no dependency on the `document_links` reverse-lookup index at all,
        // unlike the incoming-link-rewrite tests above.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original = "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [[loc.md|Myself]] too.\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let harness = git_backed_harness(&work);

        let req = make_move_req(
            "old/loc.md",
            "new/loc-self-alias.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);

        let dest_content =
            std::fs::read_to_string(work.path().join("new/loc-self-alias.md")).unwrap();
        assert!(
            dest_content.contains("[[loc-self-alias.md|Myself]]"),
            "the self-referencing pipe-alias link must be rewritten relative to the \
             destination's own directory, with the alias preserved, got: {dest_content}"
        );
        assert!(!dest_content.contains("old/loc.md"));
    }

    #[tokio::test]
    async fn move_directory_rewrites_an_outside_referencing_documents_pipe_alias_link() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old6")).unwrap();
        let a = "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# A\n";
        std::fs::write(work.path().join("old6/a.md"), a).unwrap();
        git_commit_paths(&work, &["old6/a.md"], "add old6/a.md");

        let referencing = "---\ntitle: Referencer\ndescription: d\ntype: guide\ntags: [t]\n---\n\n\
                            See [[old6/a.md|A Doc]] for more.\n";
        std::fs::write(work.path().join("referencing6.md"), referencing).unwrap();
        git_commit_paths(&work, &["referencing6.md"], "add referencing6.md");

        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencing6.md",
                "markdown",
                &[("old6/a.md".to_string(), None)],
            )
            .await
            .unwrap();

        let success = move_directory(&harness.deps(), "old6", "new6", None)
            .await
            .unwrap();
        assert_eq!(success.moved.len(), 1);
        assert_eq!(success.rewritten_paths, vec!["referencing6.md".to_string()]);

        let ref_after = std::fs::read_to_string(work.path().join("referencing6.md")).unwrap();
        assert!(
            ref_after.contains("[[new6/a.md|A Doc]]"),
            "the outside document's pipe-alias link must resolve to the moved document's new \
             location with its alias preserved, got: {ref_after}"
        );
        assert!(!ref_after.contains("old6/a.md"));
    }

    #[tokio::test]
    async fn move_rewrites_a_self_reference_relative_to_the_destination() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original = "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [self](loc.md) too.\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        // No `with_state_db` — the self-reference rewrite is computed purely from
        // `new_content` and does not depend on the reverse-link index at all, so
        // this must work identically whether or not `WriteDeps::state` is set.
        let harness = git_backed_harness(&work);

        let req = make_move_req(
            "old/loc.md",
            "new/loc-self.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(
            success.rewritten_paths.is_empty(),
            "the moved document itself is not a separate 'referencing document'"
        );

        let dest_content = std::fs::read_to_string(work.path().join("new/loc-self.md")).unwrap();
        assert!(
            dest_content.contains("[self](loc-self.md)"),
            "the self-link must be rewritten relative to the destination's own directory, got: \
             {dest_content}"
        );
        assert!(!dest_content.contains("old/loc.md"));
    }

    /// Guards the self-reference filter (`o.resolved.as_str() == source_rel`,
    /// step 6.5 above) against being "fixed" to also match on raw link text
    /// (`|| o.raw == source_rel`).
    ///
    /// Markdown link targets in this codebase are ALWAYS resolved relative to
    /// the containing document's own directory — there is no root-relative
    /// form, not even via a leading `/` (`ingest::resolve_relative_md_path`
    /// treats it as an empty, no-op component). That means a document's raw
    /// link text can be textually identical to that same document's own
    /// repo-relative path while resolving somewhere else entirely. Here,
    /// `old/loc.md` contains a link literally written as `old/loc.md`; from
    /// inside `old/`, that resolves to `old/old/loc.md` — a different
    /// document — NOT to `old/loc.md` itself.
    ///
    /// The move DOES legitimately rewrite this link's spelling — every
    /// outbound link in the moved document gets re-relativized, because a
    /// relative link's meaning depends on the containing document's
    /// directory, and that directory just changed. "Unchanged text" was
    /// never the invariant. What must be preserved is the link's *resolved
    /// target*: comparing the self-reference filter on `o.resolved` (the
    /// actually-resolved target) correctly re-relativizes this link while
    /// keeping it pointed at `old/old/loc.md`. Broadening that comparison to
    /// `o.resolved.as_str() == source_rel || o.raw == source_rel` would
    /// corrupt it: the raw text matches `source_rel` by coincidence, so the
    /// rewrite would mistake it for a self-reference and repoint it at the
    /// moved document — a file it never referenced — while the real target,
    /// `old/old/loc.md`, is silently dropped. That broadened check only
    /// "works" by accident for documents living at the KB root, where raw
    /// text and resolved path happen to coincide; it is wrong everywhere
    /// else.
    #[tokio::test]
    async fn move_preserves_the_target_of_a_link_whose_raw_text_matches_the_source_path() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original = "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [not self](old/loc.md) too.\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        // No `with_state_db` — the self-reference rewrite is computed purely from
        // `new_content` and does not depend on the reverse-link index at all, so
        // this must work identically whether or not `WriteDeps::state` is set.
        let harness = git_backed_harness(&work);

        let req = make_move_req("old/loc.md", "new/loc.md", source_original, source_original);
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(
            success.rewritten_paths.is_empty(),
            "the moved document itself is not a separate 'referencing document'"
        );

        let dest_content = std::fs::read_to_string(work.path().join("new/loc.md")).unwrap();
        let occurrences =
            crate::ingest::find_markdown_link_occurrences(&dest_content, "new/loc.md");
        let not_self = occurrences
            .iter()
            .find(|o| o.raw.contains("loc.md"))
            .unwrap_or_else(|| panic!("expected a link to loc.md in: {dest_content}"));
        assert_eq!(
            not_self.resolved, "old/old/loc.md",
            "the link's raw text happens to equal the source path but must keep resolving \
             to old/old/loc.md, a different document, after the move — got raw text {:?} \
             in: {dest_content}",
            not_self.raw
        );
        assert_ne!(
            not_self.resolved, "new/loc.md",
            "the link must not be mistaken for a self-reference (matching on raw text \
             instead of resolved target) and hijacked onto the moved document — got: \
             {dest_content}"
        );
    }

    #[tokio::test]
    async fn move_rewrites_a_non_self_link_across_a_depth_change() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        std::fs::create_dir_all(work.path().join("shared")).unwrap();
        std::fs::write(
            work.path().join("shared/doc.md"),
            "---\ntitle: Shared\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Shared\n",
        )
        .unwrap();
        git_commit_all(&work, "shared/doc.md", "add shared/doc.md");

        let source_original = "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [shared](../shared/doc.md) too.\n";
        std::fs::write(work.path().join("old/a.md"), source_original).unwrap();
        git_commit_all(&work, "old/a.md", "add old/a.md");

        let harness = git_backed_harness(&work);

        let req = make_move_req(
            "old/a.md",
            "new/deep/a.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);

        let dest_content = std::fs::read_to_string(work.path().join("new/deep/a.md")).unwrap();
        // Before the fix, this link's raw text (`../shared/doc.md`) would be left
        // unchanged by the move and silently resolve to `new/shared/doc.md` — a
        // different, likely nonexistent, file — once read from the destination's
        // deeper directory. Assert the actual resolved target, not just a string
        // match, so this states the real invariant.
        let occurrences =
            crate::ingest::find_markdown_link_occurrences(&dest_content, "new/deep/a.md");
        assert_eq!(
            occurrences.len(),
            1,
            "expected exactly one outbound link, got: {dest_content}"
        );
        assert_eq!(
            occurrences[0].resolved, "shared/doc.md",
            "the link must still resolve to shared/doc.md after the move, got raw text {:?} \
             in: {dest_content}",
            occurrences[0].raw
        );
        assert!(
            dest_content.contains("[shared](../../shared/doc.md)"),
            "expected a correctly re-relativized (deeper-climbing) link, got: {dest_content}"
        );
    }

    #[tokio::test]
    async fn move_rewrites_a_wiki_link_and_a_reference_definition_across_a_depth_change() {
        // Companion to `move_rewrites_a_non_self_link_across_a_depth_change`, but for
        // the two syntaxes that function predates: a wiki-style `[[target]]` link and
        // a reference-style definition, both pointing at a document that stays put
        // while the mover's own depth changes. Both must have their relative
        // spelling recomputed exactly like an inline link does — this is the "new
        // syntax exercised across a move with a depth change" case called for
        // alongside the inline-only depth-change test above.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        std::fs::create_dir_all(work.path().join("shared")).unwrap();
        std::fs::write(
            work.path().join("shared/doc.md"),
            "---\ntitle: Shared\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Shared\n",
        )
        .unwrap();
        git_commit_all(&work, "shared/doc.md", "add shared/doc.md");

        let source_original = "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [[../shared/doc]] and [Shared][ref] too.\n\n[ref]: ../shared/doc.md\n";
        std::fs::write(work.path().join("old/a.md"), source_original).unwrap();
        git_commit_all(&work, "old/a.md", "add old/a.md");

        let harness = git_backed_harness(&work);

        let req = make_move_req(
            "old/a.md",
            "new/deep/a.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);

        let dest_content = std::fs::read_to_string(work.path().join("new/deep/a.md")).unwrap();
        let occurrences =
            crate::ingest::find_markdown_link_occurrences(&dest_content, "new/deep/a.md");
        assert_eq!(
            occurrences.len(),
            2,
            "expected exactly one wiki-link occurrence and one reference-definition \
             occurrence, got: {dest_content}"
        );
        assert!(
            occurrences.iter().all(|o| o.resolved == "shared/doc.md"),
            "both must still resolve to shared/doc.md after the move, got: {occurrences:?}"
        );

        // The wiki link's bracketed target is rewritten the same way an inline
        // link's parenthesized target is: `relativize_md_path` always emits the
        // full `.md`-suffixed path, so the extension the author omitted is now
        // explicit — the link still resolves identically either way.
        assert!(
            dest_content.contains("[[../../shared/doc.md]]"),
            "expected the wiki link's climb to deepen from ../ to ../../, got: {dest_content}"
        );
        // The reference DEFINITION is rewritten once...
        assert!(
            dest_content.contains("[ref]: ../../shared/doc.md"),
            "expected the reference definition's climb to deepen from ../ to ../../, got: \
             {dest_content}"
        );
        // ...and the use site is left completely untouched.
        assert!(
            dest_content.contains("[Shared][ref]"),
            "the reference use site's text must be untouched, got: {dest_content}"
        );
    }

    #[tokio::test]
    async fn move_rewrites_a_non_self_link_across_a_sibling_directory_change() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("teams/alpha")).unwrap();
        std::fs::create_dir_all(work.path().join("shared/inner")).unwrap();
        std::fs::write(
            work.path().join("shared/inner/target.md"),
            "---\ntitle: Target\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Target\n",
        )
        .unwrap();
        git_commit_all(
            &work,
            "shared/inner/target.md",
            "add shared/inner/target.md",
        );

        let source_original = "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [target](../../shared/inner/target.md) too.\n";
        std::fs::write(work.path().join("teams/alpha/doc.md"), source_original).unwrap();
        git_commit_all(&work, "teams/alpha/doc.md", "add teams/alpha/doc.md");

        let harness = git_backed_harness(&work);

        let req = make_move_req(
            "teams/alpha/doc.md",
            "shared/beta/doc.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);

        let dest_content = std::fs::read_to_string(work.path().join("shared/beta/doc.md")).unwrap();
        // Same directory DEPTH (2 components) on both sides, so this is testing
        // something `move_rewrites_a_non_self_link_across_a_depth_change` does not:
        // the destination now shares a top-level ancestor with the target it never
        // shared before, so the correct climb SHRINKS from 2 segments to 1, not
        // grows.
        let occurrences =
            crate::ingest::find_markdown_link_occurrences(&dest_content, "shared/beta/doc.md");
        assert_eq!(occurrences.len(), 1, "got: {dest_content}");
        assert_eq!(occurrences[0].resolved, "shared/inner/target.md");
        assert!(
            dest_content.contains("[target](../inner/target.md)"),
            "expected the climb to shorten from ../../ to ../, got: {dest_content}"
        );
    }

    #[tokio::test]
    async fn move_pure_rename_in_same_directory_leaves_non_self_links_byte_identical() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("docs")).unwrap();
        std::fs::create_dir_all(work.path().join("other")).unwrap();
        std::fs::write(
            work.path().join("other/x.md"),
            "---\ntitle: Other\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Other\n",
        )
        .unwrap();
        git_commit_all(&work, "other/x.md", "add other/x.md");

        let source_original = "---\ntitle: Rename Me\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [other](../other/x.md) too.\n";
        std::fs::write(work.path().join("docs/a.md"), source_original).unwrap();
        git_commit_all(&work, "docs/a.md", "add docs/a.md");

        let harness = git_backed_harness(&work);

        let req = make_move_req("docs/a.md", "docs/b.md", source_original, source_original);
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);

        let dest_content = std::fs::read_to_string(work.path().join("docs/b.md")).unwrap();
        assert_eq!(
            dest_content, source_original,
            "a pure rename within one directory must leave non-self outbound links \
             byte-identical — no spurious rewrite, no needless diff"
        );
    }

    #[tokio::test]
    async fn move_outbound_rewrite_skips_links_inside_fences_and_code_spans() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        std::fs::create_dir_all(work.path().join("shared")).unwrap();
        std::fs::write(
            work.path().join("shared/doc.md"),
            "---\ntitle: Shared\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Shared\n",
        )
        .unwrap();
        git_commit_all(&work, "shared/doc.md", "add shared/doc.md");

        let source_original = "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [Real Link](../shared/doc.md) for docs.\n\n\
             ```md\n[Fenced](../shared/doc.md)\n```\n\n\
             Use `[Code](../shared/doc.md)` literally.\n";
        std::fs::write(work.path().join("old/a.md"), source_original).unwrap();
        git_commit_all(&work, "old/a.md", "add old/a.md");

        let harness = git_backed_harness(&work);

        let req = make_move_req(
            "old/a.md",
            "new/deep/a.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);

        let dest_content = std::fs::read_to_string(work.path().join("new/deep/a.md")).unwrap();
        assert!(
            dest_content.contains("[Real Link](../../shared/doc.md)"),
            "the real inline link must be re-relativized, got: {dest_content}"
        );
        assert!(
            dest_content.contains("[Fenced](../shared/doc.md)"),
            "a link inside a fenced code block must NOT be rewritten, got: {dest_content}"
        );
        assert!(
            dest_content.contains("`[Code](../shared/doc.md)`"),
            "a link inside an inline code span must NOT be rewritten, got: {dest_content}"
        );
    }

    #[tokio::test]
    async fn move_with_a_stale_document_links_row_for_a_deleted_referencing_file_does_not_fail() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let harness = git_backed_harness_with_state_db(&work).await;
        // `ghost.md` was never written to disk — a stale row, as if the
        // referencing document had since been deleted without the index
        // catching up yet.
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links("ghost.md", "markdown", &[("old/loc.md".to_string(), None)])
            .await
            .unwrap();

        let req = make_move_req(
            "old/loc.md",
            "new/loc-rewrite-test5.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req)
            .await
            .expect("a stale document_links row must not fail the move");

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(
            success.rewritten_paths.is_empty(),
            "nothing was actually rewritten — the referencing file doesn't exist"
        );
        assert!(!work.path().join("old/loc.md").exists());
        assert!(work.path().join("new/loc-rewrite-test5.md").exists());
    }

    #[tokio::test]
    async fn move_precommit_failure_with_rewrites_restores_the_referencing_document_too() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let referencing_original = "---\ntitle: Referencer\ndescription: d\ntype: guide\n\
             tags: [t]\n---\n\nSee [the moved doc](old/loc.md) for more.\n";
        std::fs::write(work.path().join("referencing.md"), referencing_original).unwrap();
        git_commit_all(&work, "referencing.md", "add referencing.md");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencing.md",
                "markdown",
                &[("old/loc.md".to_string(), None)],
            )
            .await
            .unwrap();

        let req = make_move_req(
            "old/loc.md",
            "new/loc-rewrite-test6.md",
            source_original,
            source_original,
        );
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(rolled_back),
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }

        assert!(
            work.path().join("old/loc.md").exists(),
            "source must be restored after a rolled-back move"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("old/loc.md")).unwrap(),
            source_original
        );
        assert!(!work.path().join("new/loc-rewrite-test6.md").exists());

        let referencing_after =
            std::fs::read_to_string(work.path().join("referencing.md")).unwrap();
        assert_eq!(
            referencing_after, referencing_original,
            "the referencing document's link rewrite must be rolled back too, not just the move"
        );

        assert_eq!(head_before, head_sha(&work));
        assert_eq!(git_status(&work), "");
    }

    /// Covers write.rs's step-10.5 self-contained rollback (fires when a
    /// `tokio::fs::write` into a referencing document fails DURING the
    /// rewrite loop, before git is touched at all) — a distinct code path
    /// from `move_precommit_failure_with_rewrites_restores_the_referencing_document_too`
    /// above, which exercises the LATER rollback triggered by `git commit`
    /// itself failing (#145).
    ///
    /// Two referencing documents point at the source. `StateDb::links_targeting`
    /// returns sources `ORDER BY source_path`, so `ref-a.md` is rewritten
    /// FIRST and lands on disk successfully, then `ref-b.md`'s write is forced
    /// to fail (its permission bits stripped to read-only) — standing in for
    /// a permission race or a full disk hitting the SECOND of several
    /// referencing documents mid-loop, which is exactly the scenario the
    /// issue's "duplicated at both old and new locations, or another
    /// document's content corrupted" failure mode depends on. The rollback
    /// must undo THREE things, not just the failing write: the
    /// already-rewritten `ref-a.md`, the source removal, and the destination
    /// write — proving the loop's own by-hand unwind (not the later
    /// git-commit-triggered one) is correct.
    #[tokio::test]
    async fn move_link_rewrite_write_failure_rolls_back_the_move_and_every_already_rewritten_document()
     {
        use std::os::unix::fs::PermissionsExt;

        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let ref_a_original = "---\ntitle: Ref A\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [the moved doc](old/loc.md) for more.\n";
        let ref_b_original = "---\ntitle: Ref B\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [the moved doc](old/loc.md) too.\n";
        std::fs::write(work.path().join("ref-a.md"), ref_a_original).unwrap();
        std::fs::write(work.path().join("ref-b.md"), ref_b_original).unwrap();
        git_commit_paths(&work, &["ref-a.md", "ref-b.md"], "add referencing docs");
        let head_before = head_sha(&work);

        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links("ref-a.md", "markdown", &[("old/loc.md".to_string(), None)])
            .await
            .unwrap();
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links("ref-b.md", "markdown", &[("old/loc.md".to_string(), None)])
            .await
            .unwrap();

        // Read-only: `read_to_string` (earlier in the same loop iteration)
        // still succeeds, so the code genuinely reaches — and fails at — the
        // write this test targets, rather than taking the earlier "stale
        // row, skip" branch on a read failure. That read-succeeds/write-fails
        // asymmetry is why this uses a permission bit rather than, say,
        // replacing the file with a directory: EISDIR would fail the *read*
        // and route around the branch under test entirely.
        //
        // The catch is that DAC permission bits do not stop a privileged
        // writer. Root — or anything holding CAP_DAC_OVERRIDE — writes through
        // 0o444 unimpeded, the move then succeeds, and `unwrap_err()` below
        // panics on an `Ok`. This repo has already been bitten by exactly that:
        // see `git::tests::reject_pushes`, whose doc comment records a
        // read-only-remote test that passed locally and failed on the
        // self-hosted CI runner for this reason.
        //
        // There is no privilege-independent way to make one regular file
        // readable but not writable, so rather than assume the injection
        // worked, probe it: if the permission bit does not actually block a
        // write here, skip loudly instead of reporting a failure that says
        // nothing about the code under test.
        let ref_b_path = work.path().join("ref-b.md");
        let mut perms = std::fs::metadata(&ref_b_path).unwrap().permissions();
        perms.set_mode(0o444);
        std::fs::set_permissions(&ref_b_path, perms).unwrap();

        if std::fs::OpenOptions::new()
            .write(true)
            .open(&ref_b_path)
            .is_ok()
        {
            let mut perms = std::fs::metadata(&ref_b_path).unwrap().permissions();
            perms.set_mode(0o644);
            std::fs::set_permissions(&ref_b_path, perms).unwrap();
            eprintln!(
                "SKIP move_link_rewrite_write_failure_rolls_back_the_move_and_every_already_rewritten_document: \
                 this process can write through a 0o444 file (running as root or with \
                 CAP_DAC_OVERRIDE), so the write failure this test injects cannot be produced. \
                 Run as an unprivileged user to exercise the link-rewrite rollback."
            );
            return;
        }

        let req = make_move_req("old/loc.md", "new/loc.md", source_original, source_original);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(
            matches!(err, WriteError::Io { .. }),
            "expected Io (the write-failure branch, not a git failure), got {err:?}"
        );

        // Restore permissions before the assertions below (and the tempdir's
        // own cleanup) touch the file again.
        let mut perms = std::fs::metadata(&ref_b_path).unwrap().permissions();
        perms.set_mode(0o644);
        std::fs::set_permissions(&ref_b_path, perms).unwrap();

        assert!(
            !work.path().join("new/loc.md").exists(),
            "destination must be removed by the rollback"
        );
        assert!(
            work.path().join("old/loc.md").exists(),
            "source must be restored by the rollback"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("old/loc.md")).unwrap(),
            source_original
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("ref-a.md")).unwrap(),
            ref_a_original,
            "the first (already-rewritten) referencing document must be restored too, not \
             just the move itself"
        );

        // This rollback runs entirely before git is touched — no `git add`
        // ever ran, so there is nothing to unstage and HEAD never moved.
        assert_eq!(head_before, head_sha(&work));
        assert_eq!(git_status(&work), "");
    }

    #[tokio::test]
    async fn move_with_no_state_db_does_not_rewrite_a_genuinely_referencing_document() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let referencing_original = "---\ntitle: Referencer\ndescription: d\ntype: guide\n\
             tags: [t]\n---\n\nSee [the moved doc](old/loc.md) for more.\n";
        std::fs::write(work.path().join("referencing.md"), referencing_original).unwrap();
        git_commit_all(&work, "referencing.md", "add referencing.md");

        // Plain `git_backed_harness`, with no `with_state_db` — `WriteDeps::state`
        // is `None`, so this exercises the "existing non-move / no-DB tests still
        // pass" contract even though a real referencing document (and, unlike
        // `move_precommit_failure_with_rewrites_restores_the_referencing_document_too`,
        // no `document_links` row to find it by) exists on disk.
        let harness = git_backed_harness(&work);

        let req = make_move_req(
            "old/loc.md",
            "new/loc-rewrite-test7.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(success.rewritten_paths.is_empty());
        let referencing_after =
            std::fs::read_to_string(work.path().join("referencing.md")).unwrap();
        assert_eq!(
            referencing_after, referencing_original,
            "with no state DB wired up, the referencing document must be left untouched"
        );
    }

    // -----------------------------------------------------------------------
    // move_directory: atomic directory move
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn move_directory_relocates_every_document_in_one_commit() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old-dir/sub")).unwrap();
        let a = "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# A\n";
        let b = "---\ntitle: B\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# B\n";
        let c = "---\ntitle: C\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# C\n";
        std::fs::write(work.path().join("old-dir/a.md"), a).unwrap();
        std::fs::write(work.path().join("old-dir/b.md"), b).unwrap();
        std::fs::write(work.path().join("old-dir/sub/c.md"), c).unwrap();
        git_commit_paths(
            &work,
            &["old-dir/a.md", "old-dir/b.md", "old-dir/sub/c.md"],
            "add old-dir",
        );
        let head_before = head_sha(&work);

        let harness = git_backed_harness(&work);
        let success = move_directory(&harness.deps(), "old-dir", "new-dir-test1", None)
            .await
            .unwrap();

        assert_eq!(success.moved.len(), 3);
        assert_ne!(
            success.sha, head_before,
            "the move must produce a new commit"
        );
        assert!(
            !work.path().join("old-dir").exists(),
            "the old prefix must be gone entirely"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("new-dir-test1/a.md")).unwrap(),
            a
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("new-dir-test1/b.md")).unwrap(),
            b
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("new-dir-test1/sub/c.md")).unwrap(),
            c
        );
        // Every document plus both prefixes' worth of filesystem changes landed
        // in the single commit — nothing left staged or dangling.
        assert_eq!(git_status(&work), "");
    }

    #[tokio::test]
    async fn move_directory_preserves_a_link_between_two_documents_inside_the_moved_subtree() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old2/sub1")).unwrap();
        std::fs::create_dir_all(work.path().join("old2/sub2")).unwrap();
        let a = "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n\
                 See [b](../sub2/b.md) for more.\n";
        let b = "---\ntitle: B\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# B\n";
        std::fs::write(work.path().join("old2/sub1/a.md"), a).unwrap();
        std::fs::write(work.path().join("old2/sub2/b.md"), b).unwrap();
        git_commit_paths(&work, &["old2/sub1/a.md", "old2/sub2/b.md"], "add old2");

        let harness = git_backed_harness(&work);
        let success = move_directory(&harness.deps(), "old2", "moved2/deep/target", None)
            .await
            .unwrap();
        assert_eq!(success.moved.len(), 2);

        // Assert on the RESOLVED target, not the link text: a correct
        // lockstep-move rewrite usually reproduces identical text (both
        // documents moved by the same prefix change), so the meaningful check
        // is that the link still resolves to the MOVED copy of `b.md`, not a
        // stale path under the old, now-nonexistent `old2/`.
        let a_after =
            std::fs::read_to_string(work.path().join("moved2/deep/target/sub1/a.md")).unwrap();
        let occurrences =
            crate::ingest::find_markdown_link_occurrences(&a_after, "moved2/deep/target/sub1/a.md");
        assert_eq!(occurrences.len(), 1, "expected exactly one link occurrence");
        assert_eq!(occurrences[0].resolved, "moved2/deep/target/sub2/b.md");
    }

    #[tokio::test]
    async fn move_directory_preserves_a_link_from_inside_the_subtree_to_a_document_outside_it() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old3")).unwrap();
        std::fs::create_dir_all(work.path().join("shared")).unwrap();
        let a = "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n\
                 See [shared](../shared/target.md) for more.\n";
        let target =
            "---\ntitle: Target\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Target\n";
        std::fs::write(work.path().join("old3/a.md"), a).unwrap();
        std::fs::write(work.path().join("shared/target.md"), target).unwrap();
        git_commit_paths(
            &work,
            &["old3/a.md", "shared/target.md"],
            "add old3 and shared",
        );

        let harness = git_backed_harness(&work);
        // Move to a destination several levels DEEPER than the source — a naive
        // "leave outbound links untouched" implementation would break this link,
        // since reaching the untouched `shared/target.md` from the new, deeper
        // location requires more `../` climbs than the original text has.
        let success = move_directory(&harness.deps(), "old3", "moved3/deeper/still/here", None)
            .await
            .unwrap();
        assert_eq!(success.moved.len(), 1);

        let a_after =
            std::fs::read_to_string(work.path().join("moved3/deeper/still/here/a.md")).unwrap();
        let occurrences = crate::ingest::find_markdown_link_occurrences(
            &a_after,
            "moved3/deeper/still/here/a.md",
        );
        assert_eq!(occurrences.len(), 1, "expected exactly one link occurrence");
        assert_eq!(
            occurrences[0].resolved, "shared/target.md",
            "the link must still resolve to the same, unmoved outside document"
        );
        assert!(
            work.path().join("shared/target.md").exists(),
            "the outside document must never have moved"
        );
    }

    #[tokio::test]
    async fn move_directory_rewrites_an_outside_referencing_documents_link() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old4")).unwrap();
        let a = "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# A\n";
        std::fs::write(work.path().join("old4/a.md"), a).unwrap();
        git_commit_paths(&work, &["old4/a.md"], "add old4/a.md");

        let referencing = "---\ntitle: Referencer\ndescription: d\ntype: guide\ntags: [t]\n---\n\n\
                            See [a](old4/a.md) for more.\n";
        std::fs::write(work.path().join("referencing4.md"), referencing).unwrap();
        git_commit_paths(&work, &["referencing4.md"], "add referencing4.md");

        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencing4.md",
                "markdown",
                &[("old4/a.md".to_string(), None)],
            )
            .await
            .unwrap();

        let success = move_directory(&harness.deps(), "old4", "new4", None)
            .await
            .unwrap();
        assert_eq!(success.moved.len(), 1);
        assert_eq!(success.rewritten_paths, vec!["referencing4.md".to_string()]);

        let ref_after = std::fs::read_to_string(work.path().join("referencing4.md")).unwrap();
        let occurrences =
            crate::ingest::find_markdown_link_occurrences(&ref_after, "referencing4.md");
        assert_eq!(occurrences.len(), 1, "expected exactly one link occurrence");
        assert_eq!(
            occurrences[0].resolved, "new4/a.md",
            "the outside document's link must resolve to the moved document's new location"
        );
    }

    #[tokio::test]
    async fn move_directory_validation_failure_for_one_document_aborts_the_whole_move() {
        let tmp = tempfile::tempdir().unwrap();
        let source_dir = tmp.path().join("loose5");
        std::fs::create_dir_all(&source_dir).unwrap();
        // Valid under the (schema-less) source directory, but missing a field the
        // destination's schema requires.
        let a = "---\ntitle: A\n---\n# A";
        let b = "---\ntitle: B\n---\n# B";
        std::fs::write(source_dir.join("a.md"), a).unwrap();
        std::fs::write(source_dir.join("b.md"), b).unwrap();

        // The schema lives on the DESTINATION'S PARENT, not literally inside the
        // (as-yet-nonexistent, and so guard-2-empty) destination prefix itself —
        // `strict5/target` inherits it via the normal cascade.
        let dest_parent = tmp.path().join("strict5");
        std::fs::create_dir_all(&dest_parent).unwrap();
        std::fs::write(
            dest_parent.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  strict_field:\n    required: true\n",
        )
        .unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let err = move_directory(&harness.deps(), "loose5", "strict5/target", None)
            .await
            .unwrap_err();
        match err {
            DirectoryMoveError::Validation {
                failures,
                moved_schema_files,
            } => {
                assert_eq!(
                    failures.len(),
                    2,
                    "both documents are missing strict_field, expected {:?}",
                    failures
                );
                assert!(
                    moved_schema_files.is_empty(),
                    "no schema file is moving in this scenario"
                );
            }
            other => panic!("expected Validation, got {other:?}"),
        }
        assert!(
            source_dir.join("a.md").exists(),
            "nothing must be mutated when even one document fails validation"
        );
        assert!(source_dir.join("b.md").exists());
        assert!(!tmp.path().join("strict5/target").exists());
    }

    // -- move_directory: schema files travel with the subtree ---------------
    //
    // These replace the old, stricter behavior (a source subtree containing its
    // own `.kb-schema.yaml` was rejected outright — see git history for
    // `move_directory_with_a_schema_file_in_the_source_subtree_is_rejected`).
    // Lifting that restriction is the whole point of this suite: a schema file
    // now moves along with the documents it governs, re-parenting its cascade
    // onto the destination — see `SchemaCache::with_remapped_scopes`.

    #[tokio::test]
    async fn move_directory_relocates_a_subtree_including_its_own_schema_file() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let source_dir = work.path().join("old6/sub");
        std::fs::create_dir_all(&source_dir).unwrap();
        let content = "---\ntitle: A\n---\n# A";
        std::fs::write(source_dir.join("a.md"), content).unwrap();
        std::fs::write(
            source_dir.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  title:\n    required: true\n",
        )
        .unwrap();
        git_commit_paths(
            &work,
            &[
                "old6/sub/a.md",
                &format!("old6/sub/{}", crate::schema::SCHEMA_FILE_NAME),
            ],
            "add old6",
        );

        let harness = git_backed_harness(&work);

        let success = move_directory(&harness.deps(), "old6", "new6", None)
            .await
            .unwrap();

        assert!(
            !work.path().join("old6").exists(),
            "the whole old prefix, schema file included, must be gone"
        );
        assert!(work.path().join("new6/sub/a.md").exists());
        let moved_schema = work
            .path()
            .join("new6/sub")
            .join(crate::schema::SCHEMA_FILE_NAME);
        assert!(
            moved_schema.exists(),
            "the schema file must have moved along with the document it governs"
        );
        assert!(
            success
                .moved
                .contains(&("old6/sub/a.md".to_string(), "new6/sub/a.md".to_string())),
        );
        assert!(
            success.moved.contains(&(
                format!("old6/sub/{}", crate::schema::SCHEMA_FILE_NAME),
                format!("new6/sub/{}", crate::schema::SCHEMA_FILE_NAME),
            )),
            "the schema file's own relocation must be reported in `moved` too: {:?}",
            success.moved
        );
        assert_eq!(git_status(&work), "");

        // Post-move: rebuilding a real cache off disk must agree with what the
        // move validated against — proving the prediction was right, not merely
        // self-consistent.
        let rebuilt = crate::schema::SchemaCache::build(
            &work.path().canonicalize().unwrap(),
            &crate::config::FrontmatterConfig::default(),
        );
        assert!(
            rebuilt
                .resolve_for(Path::new("new6/sub/a.md"))
                .fields
                .get("title")
                .is_some_and(|f| f.required),
            "the rebuilt cache must show the relocated schema's rule in effect"
        );
    }

    #[tokio::test]
    async fn move_directory_schema_file_travels_into_a_stricter_destination_is_rejected() {
        // The crux case the old guard existed to prevent: the subtree's OWN
        // schema file travels with it, but the destination's ancestor declares
        // an ADDITIONAL required field the source's ancestor never did. Nothing
        // in the moved subtree satisfies it, so the move must fail — and fail
        // for exactly this reason, not some other validation quirk.
        let tmp = tempfile::tempdir().unwrap();
        let source_dir = tmp.path().join("src7/sub");
        std::fs::create_dir_all(&source_dir).unwrap();
        let content = "---\ntitle: A\n---\n# A";
        std::fs::write(source_dir.join("a.md"), content).unwrap();
        std::fs::write(
            source_dir.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  title:\n    required: true\n",
        )
        .unwrap();

        // The destination's PARENT declares a field the source's parent never
        // required.
        let dest_parent = tmp.path().join("dest7");
        std::fs::create_dir_all(&dest_parent).unwrap();
        std::fs::write(
            dest_parent.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  extra_required:\n    required: true\n",
        )
        .unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let err = move_directory(&harness.deps(), "src7", "dest7/target", None)
            .await
            .unwrap_err();
        match err {
            DirectoryMoveError::Validation {
                failures,
                moved_schema_files,
            } => {
                assert_eq!(failures.len(), 1);
                assert_eq!(failures[0].0, "dest7/target/sub/a.md");
                assert!(
                    failures[0]
                        .1
                        .errors
                        .iter()
                        .any(|e| e.contains("extra_required")),
                    "must name the field the destination newly requires: {:?}",
                    failures[0].1.errors
                );
                assert_eq!(
                    moved_schema_files,
                    vec![(
                        format!("src7/sub/{}", crate::schema::SCHEMA_FILE_NAME),
                        format!("dest7/target/sub/{}", crate::schema::SCHEMA_FILE_NAME),
                    )],
                    "the error must name the schema file that is relocating"
                );
            }
            other => panic!("expected Validation, got {other:?}"),
        }

        assert!(
            source_dir.join("a.md").exists(),
            "nothing must be mutated on a rejected move"
        );
        assert!(source_dir.join(crate::schema::SCHEMA_FILE_NAME).exists());
        assert!(!tmp.path().join("dest7/target").exists());
    }

    #[tokio::test]
    async fn move_directory_schema_file_travels_into_a_more_permissive_destination_succeeds() {
        // The mirror of the crux case: the destination's ancestor is more
        // permissive than the source's was, so documents that were valid stay
        // valid.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let source_parent = work.path().join("src8");
        std::fs::create_dir_all(&source_parent).unwrap();
        std::fs::write(
            source_parent.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  status:\n    required: true\n",
        )
        .unwrap();
        let source_dir = source_parent.join("sub");
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::write(
            source_dir.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  status:\n    type: enum\n    values: [draft, active]\n",
        )
        .unwrap();
        std::fs::write(source_dir.join("a.md"), "---\nstatus: draft\n---\n# A").unwrap();
        git_commit_paths(
            &work,
            &[
                &format!("src8/{}", crate::schema::SCHEMA_FILE_NAME),
                &format!("src8/sub/{}", crate::schema::SCHEMA_FILE_NAME),
                "src8/sub/a.md",
            ],
            "add src8",
        );

        // Destination has NO ancestor schema at all — strictly more permissive
        // than the source's parent, which required `status`.
        let harness = git_backed_harness(&work);

        let success = move_directory(&harness.deps(), "src8/sub", "dest8/sub", None)
            .await
            .unwrap();
        assert_eq!(success.moved.len(), 2, "the document and its schema file");
        assert!(work.path().join("dest8/sub/a.md").exists());
        assert_eq!(git_status(&work), "");
    }

    #[tokio::test]
    async fn move_directory_relocates_multiple_schema_files_at_different_depths() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let root = work.path().join("src9");
        std::fs::create_dir_all(root.join("mid/deep")).unwrap();
        std::fs::write(
            root.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  top:\n    required: true\n",
        )
        .unwrap();
        std::fs::write(
            root.join("mid").join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  mid_field:\n    required: true\n",
        )
        .unwrap();
        std::fs::write(
            root.join("mid/deep").join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  deep_field:\n    required: true\n",
        )
        .unwrap();
        std::fs::write(
            root.join("mid/deep/doc.md"),
            "---\ntop: t\nmid_field: m\ndeep_field: d\n---\n# D",
        )
        .unwrap();
        git_commit_paths(
            &work,
            &[
                &format!("src9/{}", crate::schema::SCHEMA_FILE_NAME),
                &format!("src9/mid/{}", crate::schema::SCHEMA_FILE_NAME),
                &format!("src9/mid/deep/{}", crate::schema::SCHEMA_FILE_NAME),
                "src9/mid/deep/doc.md",
            ],
            "add src9",
        );

        let harness = git_backed_harness(&work);

        let success = move_directory(&harness.deps(), "src9", "dest9", None)
            .await
            .unwrap();
        // 1 document + 3 schema files, at 3 different depths.
        assert_eq!(success.moved.len(), 4, "{:?}", success.moved);
        for suffix in ["".to_string(), "mid/".to_string(), "mid/deep/".to_string()] {
            assert!(
                work.path()
                    .join(format!(
                        "dest9/{}{}",
                        suffix,
                        crate::schema::SCHEMA_FILE_NAME
                    ))
                    .exists(),
                "schema file at depth '{}' must have relocated",
                suffix
            );
        }
        assert!(work.path().join("dest9/mid/deep/doc.md").exists());
        assert_eq!(git_status(&work), "");

        let rebuilt = crate::schema::SchemaCache::build(
            &work.path().canonicalize().unwrap(),
            &crate::config::FrontmatterConfig::default(),
        );
        let resolved = rebuilt.resolve_for(Path::new("dest9/mid/deep/doc.md"));
        assert!(resolved.fields["top"].required);
        assert!(resolved.fields["mid_field"].required);
        assert!(resolved.fields["deep_field"].required);
    }

    #[tokio::test]
    async fn move_directory_values_splicing_field_resolves_differently_under_destination_parent() {
        // A `$values`-splicing field must resolve against the DESTINATION
        // parent's set, not the source's — proving the remapped cache re-runs
        // the splice rather than reusing whatever the live cache had cached.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let source_parent = work.path().join("src10");
        std::fs::create_dir_all(&source_parent).unwrap();
        std::fs::write(
            source_parent.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  tags:\n    values: [source_tag]\n",
        )
        .unwrap();
        let source_dir = source_parent.join("sub");
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::write(
            source_dir.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  tags:\n    values: [$values, own_tag]\n",
        )
        .unwrap();
        std::fs::write(source_dir.join("a.md"), "---\ntags: [own_tag]\n---\n# A").unwrap();

        let dest_parent = work.path().join("dest10");
        std::fs::create_dir_all(&dest_parent).unwrap();
        std::fs::write(
            dest_parent.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  tags:\n    values: [dest_tag]\n",
        )
        .unwrap();
        git_commit_paths(
            &work,
            &[
                &format!("src10/{}", crate::schema::SCHEMA_FILE_NAME),
                &format!("src10/sub/{}", crate::schema::SCHEMA_FILE_NAME),
                "src10/sub/a.md",
                &format!("dest10/{}", crate::schema::SCHEMA_FILE_NAME),
            ],
            "add src10 and dest10",
        );

        let harness = git_backed_harness(&work);

        // `source_tag` is no longer permitted post-move (the source's parent
        // set is gone), but the document only ever used `own_tag`, which the
        // moved schema's own splice still contributes — so it stays valid.
        let success = move_directory(&harness.deps(), "src10/sub", "dest10/sub", None)
            .await
            .unwrap();
        assert_eq!(success.moved.len(), 2);
        assert_eq!(git_status(&work), "");

        let rebuilt = crate::schema::SchemaCache::build(
            &work.path().canonicalize().unwrap(),
            &crate::config::FrontmatterConfig::default(),
        );
        let resolved = rebuilt.resolve_for(Path::new("dest10/sub/a.md"));
        assert_eq!(
            resolved.fields["tags"].values,
            Some(vec!["dest_tag".to_string(), "own_tag".to_string()]),
            "the splice must resolve against the DESTINATION parent's set"
        );
    }

    #[tokio::test]
    async fn move_directory_with_an_unparseable_schema_file_in_source_is_blocked() {
        let tmp = tempfile::tempdir().unwrap();
        let source_dir = tmp.path().join("src11/sub");
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::write(source_dir.join("a.md"), "---\ntitle: A\n---\n# A").unwrap();
        std::fs::write(
            source_dir.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  tags:\n    values: [$oops]\n",
        )
        .unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let err = move_directory(&harness.deps(), "src11", "dest11", None)
            .await
            .unwrap_err();
        match err {
            DirectoryMoveError::BrokenSchemaInSource { path, reason } => {
                assert_eq!(
                    path,
                    format!("src11/sub/{}", crate::schema::SCHEMA_FILE_NAME)
                );
                assert!(!reason.is_empty());
            }
            other => panic!("expected BrokenSchemaInSource, got {other:?}"),
        }
        assert!(source_dir.join("a.md").exists());
        assert!(!tmp.path().join("dest11").exists());
    }

    #[tokio::test]
    async fn move_directory_to_an_occupied_destination_prefix_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let source_dir = tmp.path().join("old7");
        std::fs::create_dir_all(&source_dir).unwrap();
        let content = "---\ntitle: A\n---\n# A";
        std::fs::write(source_dir.join("a.md"), content).unwrap();

        let dest_dir = tmp.path().join("occupied7");
        std::fs::create_dir_all(&dest_dir).unwrap();
        std::fs::write(dest_dir.join("already-here.md"), "# Already here").unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let err = move_directory(&harness.deps(), "old7", "occupied7", None)
            .await
            .unwrap_err();
        assert!(
            matches!(err, DirectoryMoveError::AlreadyExists),
            "got {err:?}"
        );
        assert!(
            source_dir.join("a.md").exists(),
            "source must be untouched when the destination prefix is occupied"
        );
        assert_eq!(
            std::fs::read_to_string(dest_dir.join("already-here.md")).unwrap(),
            "# Already here",
            "the pre-existing destination content must be untouched"
        );
    }

    #[tokio::test]
    async fn move_directory_toctou_destination_collision_reports_already_exists_not_io() {
        // A collision that appears AFTER the batch pre-check (guard 2) and the
        // per-document defensive re-check, but before the per-document
        // `create_new` write in phase 1, must surface as `AlreadyExists` (a
        // benign, retryable race), not the generic `Io` arm. `Io` maps to
        // `McpError::internal_error`, which would misreport a completely
        // ordinary race as a server fault.
        //
        // Reproduced deterministically rather than via a hopeful thread race:
        // a configured `lint_command` makes `validate::validate_content`
        // `.await` a real subprocess (`sleep 0.3`) for the one document in
        // this move, which happens AFTER that document's own per-document
        // `exists()` check but BEFORE phase 1 ever runs. `tokio::join!` runs
        // that subprocess wait concurrently (same task, cooperative
        // scheduling) with a second future that creates the destination file
        // partway through the sleep -- landing squarely in the TOCTOU window
        // this fix closes. This also exercises `rollback_directory_move_filesystem`
        // for real with the lock-hoist fix in place: if that helper still
        // tried to acquire its own `GitLock` (the pre-fix behavior) instead of
        // reusing the one `move_directory` already holds by this point, this
        // test would hang until the outer `timeout` below fails it.
        let tmp = tempfile::tempdir().unwrap();
        let source_dir = tmp.path().join("old-toctou");
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::write(source_dir.join("a.md"), "---\ntitle: A\n---\n# A").unwrap();

        let mut config = crate::mcp::make_test_resolved_config(tmp.path());
        Arc::get_mut(&mut config).unwrap().validation.lint_command =
            Some(vec!["sh".into(), "-c".into(), "sleep 0.3".into()]);
        let harness = Harness::new(&tmp, config);

        let dest_dir = tmp.path().join("new-toctou");
        let dest_file = dest_dir.join("a.md");
        let collision_content = "collision, landed after the pre-check";

        let deps = harness.deps();
        let (move_result, _) = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            tokio::join!(
                move_directory(&deps, "old-toctou", "new-toctou", None),
                async {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    std::fs::create_dir_all(&dest_dir).unwrap();
                    std::fs::write(&dest_file, collision_content).unwrap();
                },
            )
        })
        .await
        .expect("move_directory must not hang");

        let err = move_result.unwrap_err();
        assert!(
            matches!(err, DirectoryMoveError::AlreadyExists),
            "a destination collision appearing after the pre-check must map to \
             AlreadyExists, not a generic Io error; got {err:?}"
        );
        assert!(
            source_dir.join("a.md").exists(),
            "the source must be untouched -- phase 1 never got far enough to remove it"
        );
        assert_eq!(
            std::fs::read_to_string(&dest_file).unwrap(),
            collision_content,
            "move_directory must never have touched the colliding file (create_new can't \
             overwrite it, and rollback only removes what IT wrote)"
        );
    }

    #[tokio::test]
    async fn move_directory_blocks_while_git_lock_is_externally_held_then_completes() {
        // GIT_LOCK hoist regression (data-loss finding fix): `move_directory`
        // must acquire `GIT_LOCK` before phase 1 (destination writes), not
        // just before the commit, and hold that ONE guard across phases 1-3,
        // the commit, and any rollback. Proven at runtime: another holder of
        // `GIT_LOCK` must observably block the call, and releasing that
        // holder must let it proceed to completion without hanging -- a hang
        // would mean something reachable from `move_directory` (e.g.
        // `rollback_directory_move_filesystem`) is trying to reacquire the
        // already-held, non-reentrant mutex instead of reusing it.
        let tmp = tempfile::tempdir().unwrap();
        let source_dir = tmp.path().join("old-lockcheck");
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::write(source_dir.join("a.md"), "---\ntitle: A\n---\n# A").unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let held = git::lock_git().await;

        let deps = harness.deps();
        let move_fut = move_directory(&deps, "old-lockcheck", "new-lockcheck", None);
        tokio::pin!(move_fut);
        let still_blocked =
            tokio::time::timeout(std::time::Duration::from_millis(200), &mut move_fut).await;
        assert!(
            still_blocked.is_err(),
            "move_directory must block on GIT_LOCK (acquired ahead of phase 1, per the \
             finding's fix) while another holder has it"
        );

        drop(held);
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), move_fut)
            .await
            .expect(
                "move_directory must proceed to completion once GIT_LOCK is released, not hang \
                 against its own held guard -- a hang here would mean this non-reentrant mutex \
                 is being acquired a second time somewhere in the call chain",
            );
        // Not git-backed, so the commit itself fails fast once the lock is free --
        // the point of this test is that nothing deadlocks, not the outcome.
        match result.unwrap_err() {
            DirectoryMoveError::PreCommitFailed { .. } => {}
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn move_directory_precommit_failure_rolls_back_every_document_and_referencing_document() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old8")).unwrap();
        let a = "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# A\n";
        let b = "---\ntitle: B\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# B\n";
        std::fs::write(work.path().join("old8/a.md"), a).unwrap();
        std::fs::write(work.path().join("old8/b.md"), b).unwrap();
        git_commit_paths(&work, &["old8/a.md", "old8/b.md"], "add old8");

        let referencing = "---\ntitle: Referencer\ndescription: d\ntype: guide\ntags: [t]\n---\n\n\
                            See [a](old8/a.md) for more.\n";
        std::fs::write(work.path().join("referencing8.md"), referencing).unwrap();
        git_commit_paths(&work, &["referencing8.md"], "add referencing8.md");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencing8.md",
                "markdown",
                &[("old8/a.md".to_string(), None)],
            )
            .await
            .unwrap();

        let err = move_directory(&harness.deps(), "old8", "new8", None)
            .await
            .unwrap_err();
        match err {
            DirectoryMoveError::PreCommitFailed { rolled_back, .. } => assert!(rolled_back),
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }

        assert_eq!(
            std::fs::read_to_string(work.path().join("old8/a.md")).unwrap(),
            a,
            "source a.md must be restored after a rolled-back directory move"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("old8/b.md")).unwrap(),
            b,
            "source b.md must be restored after a rolled-back directory move"
        );
        assert!(
            !work.path().join("new8").exists(),
            "destination prefix must be gone after a rolled-back directory move"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("referencing8.md")).unwrap(),
            referencing,
            "the referencing document's link rewrite must be rolled back too, not just the move"
        );
        assert_eq!(head_before, head_sha(&work));
        assert_eq!(git_status(&work), "");
    }

    /// #147: `move_directory`'s own "rollback itself also failed" branch
    /// (`rolled_back: false`) — the fourth and last of the four
    /// structurally-identical sites the issue names, and like the other
    /// three, exercised only by `delete_document`'s test before this. Same
    /// no-git-repo technique: `git add` fails first (`PreCommit`), and then
    /// `git::restore_from_head` on the moved source ALSO fails (no
    /// repository to restore from), which alone is enough to flip this
    /// move's `rolled_back` to `false` — `move_directory` ORs a failure in
    /// per-source, per-destination, and per-rewritten-document, and any
    /// single one failing is enough.
    #[tokio::test]
    async fn move_directory_rollback_failure_with_no_git_repo_reports_rolled_back_false() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("old-dir-no-repo");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("a.md"), "---\ntitle: A\n---\n\n# A\n").unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let err = move_directory(&harness.deps(), "old-dir-no-repo", "new-dir-no-repo", None)
            .await
            .unwrap_err();
        match err {
            DirectoryMoveError::PreCommitFailed { rolled_back, .. } => assert!(!rolled_back),
            other => panic!("expected PreCommitFailed{{rolled_back: false}}, got {other:?}"),
        }
        // The destination copy was written, then successfully removed during
        // rollback — but the source restore can't run with no HEAD to
        // restore from, so the source is left gone too, same inconsistency
        // as the single-document move's equivalent test above.
        assert!(!tmp.path().join("new-dir-no-repo/a.md").exists());
        assert!(!sub.join("a.md").exists());
    }
}
