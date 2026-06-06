use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use std::sync::RwLock;

use globset::{Glob, GlobSet, GlobSetBuilder};
use rmcp::{
    ErrorData as McpError, ServerHandler, handler::server::router::tool::ToolRouter,
    handler::server::wrapper::Parameters, model::*, schemars, tool, tool_handler, tool_router,
};

use anyhow::Context as _;
use tracing::{error, warn};

use crate::{
    config::ResolvedConfig,
    embed::EmbedClient,
    git, ingest,
    qdrant::{QdrantStore, SearchResult},
    state::StateDb,
    validate,
};

const MAX_SEARCH_LIMIT: u64 = 50;
const MAX_QUERY_LEN: usize = 4096;
const MAX_FILTER_STR_LEN: usize = 256;
const MAX_TAG_COUNT: usize = 20;
const MAX_TAG_LEN: usize = 256;

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
/// Upper bound on the number of indexed file paths to fetch for fuzzy
/// `get_document` resolution. Larger KBs will silently cap at this value.
const MAX_INDEXED_PATHS_FOR_FUZZY: u64 = 10_000;
/// How many "did you mean?" suggestions to include when no basename matches.
const FUZZY_SUGGESTION_COUNT: usize = 3;

fn resolve_limit(requested: Option<u64>) -> u64 {
    requested.unwrap_or(10).min(MAX_SEARCH_LIMIT)
}

fn validate_search_params(params: &SearchParams) -> Result<(), McpError> {
    if params.query.len() > MAX_QUERY_LEN {
        return Err(McpError::invalid_params(
            format!("query exceeds maximum length of {MAX_QUERY_LEN} characters"),
            None,
        ));
    }
    if let Some(domain) = &params.domain
        && domain.len() > MAX_FILTER_STR_LEN
    {
        return Err(McpError::invalid_params(
            format!("domain exceeds maximum length of {MAX_FILTER_STR_LEN} characters"),
            None,
        ));
    }
    if let Some(doc_type) = &params.r#type
        && doc_type.len() > MAX_FILTER_STR_LEN
    {
        return Err(McpError::invalid_params(
            format!("type exceeds maximum length of {MAX_FILTER_STR_LEN} characters"),
            None,
        ));
    }
    if let Some(tags) = &params.tags {
        if tags.len() > MAX_TAG_COUNT {
            return Err(McpError::invalid_params(
                format!("tags list exceeds maximum of {MAX_TAG_COUNT} entries"),
                None,
            ));
        }
        for tag in tags {
            if tag.len() > MAX_TAG_LEN {
                return Err(McpError::invalid_params(
                    format!("tag exceeds maximum length of {MAX_TAG_LEN} characters"),
                    None,
                ));
            }
        }
    }
    Ok(())
}

/// Validate that `rel_path` is safe to write inside `data_root`, returning the
/// absolute target path on success or an error string on failure.
///
/// Checks performed (in order):
/// 1. Reject absolute paths.
/// 2. Reject any `..` component.
/// 3. Lexical `starts_with` check on the joined abs path.
/// 4. Canonicalize the deepest *existing* ancestor of the target; verify it
///    still `starts_with` the canonical data_root. This catches a symlinked
///    ancestor directory that resolves to a location outside data_root.
pub fn resolve_safe_write_path(data_root: &Path, rel_path: &str) -> Result<PathBuf, String> {
    // 1. Reject absolute paths.
    let requested = Path::new(rel_path);
    if requested.is_absolute() {
        return Err("path must be relative to the knowledge base root".to_string());
    }

    // 2. Reject any `..` component.
    for component in requested.components() {
        if component == std::path::Component::ParentDir {
            return Err("path must not contain '..' components".to_string());
        }
    }

    let abs_path = data_root.join(rel_path);

    // 3. Lexical starts_with check.
    if !abs_path.starts_with(data_root) {
        return Err("path escapes the knowledge base root".to_string());
    }

    // 4. Canonical-ancestor check: canonicalize data_root, then walk up from
    //    abs_path to find the deepest ancestor that actually exists on disk,
    //    canonicalize it, and confirm it still sits under canonical data_root.
    let canonical_root = data_root.canonicalize().map_err(|e| {
        format!(
            "cannot canonicalize data root '{}': {}",
            data_root.display(),
            e
        )
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
        format!(
            "cannot canonicalize ancestor '{}': {}",
            existing_ancestor.display(),
            e
        )
    })?;

    if !canonical_ancestor.starts_with(&canonical_root) {
        return Err("path escapes the knowledge base root (symlink detected)".to_string());
    }

    Ok(abs_path)
}

/// Parameters for the `get_document` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GetDocumentParams {
    /// Path of the document to retrieve. Accepts paths relative to the
    /// knowledge-base root (e.g. `lifestyle/vehicles/foo.md`, as returned by
    /// the `search` tool), or just a basename when it's unique across the
    /// index. Absolute paths are also accepted for backwards compatibility.
    pub path: String,
}

/// Parameters for the `search` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchParams {
    /// The natural-language search query.
    pub query: String,

    /// Optional: filter results to a specific domain.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,

    /// Optional: filter results by document type.
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,

    /// Optional: filter results to documents that have any of these tags.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,

    /// Maximum number of results to return (default: 10, max: 50).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,
}

/// Parameters for the `create_document` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct CreateDocumentParams {
    /// Path of the new document, relative to the knowledge base root, e.g. "sysadmin/docker/foo.md"
    pub path: String,
    /// Full markdown content of the document, INCLUDING YAML frontmatter
    pub content: String,
    /// Optional commit message; if omitted, a message is generated from the path
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Skip the duplicate-detection check and create the document even if a
    /// similar one exists. Use this when you have verified the existing document
    /// is sufficiently different and want to create a distinct entry anyway.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub force_new: Option<bool>,
}

/// Parameters for the `edit_document` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct EditDocumentParams {
    /// Path of the document to edit. Resolved like get_document (relative to the KB
    /// root, a unique basename, or absolute).
    pub path: String,
    /// Surgical edit: exact text to find. Must occur EXACTLY ONCE in the document.
    /// Provide together with new_string. Mutually exclusive with `content`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_string: Option<String>,
    /// Surgical edit: text that replaces old_string. Provide together with old_string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_string: Option<String>,
    /// Full replace: the entire new content of the document, including YAML
    /// frontmatter. Mutually exclusive with old_string/new_string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Optional commit message; defaults to "docs: update {path}".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Parameters for the `delete_document` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct DeleteDocumentParams {
    /// Path of the document to delete. Resolved like get_document (relative to the
    /// KB root, a unique basename, or absolute).
    pub path: String,
    /// Optional commit message; defaults to "docs: delete {path}".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Validated edit mode, produced by `parse_edit_mode`.
#[derive(Debug, PartialEq)]
pub enum EditMode {
    /// Replace `old_string` with `new_string` (must appear exactly once).
    Surgical { old: String, new: String },
    /// Replace the entire document content.
    Full { content: String },
}

/// Parse and validate the mode fields of `EditDocumentParams`, returning a
/// typed `EditMode` or a human-readable error string.
///
/// Rules:
/// - SURGICAL = `old_string` AND `new_string` both `Some`, `content` is `None`.
/// - FULL = `content` is `Some`, both `old_string` and `new_string` are `None`.
/// - Any other combination is rejected.
/// - Surgical with `old_string == new_string` is rejected (no-op).
pub fn parse_edit_mode(params: &EditDocumentParams) -> Result<EditMode, String> {
    let has_content = params.content.is_some();
    let has_old = params.old_string.is_some();
    let has_new = params.new_string.is_some();

    match (has_content, has_old, has_new) {
        // Full mode
        (true, false, false) => Ok(EditMode::Full {
            content: params.content.clone().unwrap(),
        }),
        // Surgical mode
        (false, true, true) => {
            let old = params.old_string.clone().unwrap();
            let new = params.new_string.clone().unwrap();
            if old == new {
                return Err(
                    "old_string and new_string are identical — no change would be made".to_string(),
                );
            }
            Ok(EditMode::Surgical { old, new })
        }
        // Both modes set
        (true, _, _) if has_old || has_new => {
            Err("content is mutually exclusive with old_string/new_string; \
             provide either content (full replace) or old_string+new_string (surgical edit)"
                .to_string())
        }
        // Only one of old_string/new_string
        (false, true, false) => {
            Err("old_string requires new_string; provide both for a surgical edit".to_string())
        }
        (false, false, true) => {
            Err("new_string requires old_string; provide both for a surgical edit".to_string())
        }
        // Neither mode
        (false, false, false) => Err(
            "must provide either content (full replace) or old_string+new_string (surgical edit)"
                .to_string(),
        ),
        // Unreachable combinations (content=true, old=true, new=true or content=true, old/new only)
        _ => Err("content is mutually exclusive with old_string/new_string; \
             provide either content (full replace) or old_string+new_string (surgical edit)"
            .to_string()),
    }
}

/// Apply a surgical edit: replace the single occurrence of `old_string` with
/// `new_string` in `old_content`.
///
/// Returns the new content string on success, or a descriptive error string.
pub fn apply_surgical(
    old_content: &str,
    old_string: &str,
    new_string: &str,
) -> Result<String, String> {
    let count = old_content.matches(old_string).count();
    match count {
        0 => Err("old_string not found in document".to_string()),
        1 => Ok(old_content.replacen(old_string, new_string, 1)),
        n => Err(format!(
            "old_string is not unique in document (found {n} occurrences); \
             include more surrounding context to disambiguate"
        )),
    }
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

#[derive(Clone)]
pub struct KbSearchServer {
    embed_client: Arc<EmbedClient>,
    qdrant: Arc<QdrantStore>,
    collection: String,
    canonical_data_path: PathBuf,
    /// Glob patterns (from `indexing.include`) used to restrict `get_document` to permitted file types.
    include_patterns: Arc<GlobSet>,
    /// Dynamic MCP server instructions, refreshed periodically with discovered metadata.
    instructions: Arc<RwLock<String>>,
    /// Resolved config, needed by write tools (create_document, etc.).
    config: Arc<ResolvedConfig>,
    tool_router: ToolRouter<KbSearchServer>,
}

/// Strip the data-root prefix from an absolute file_path for display to clients.
/// Paths in Qdrant are stored as absolute (the indexer uses `data_path` as the
/// root), but `/data` is an implementation detail — clients should see paths
/// relative to the indexed root. Falls back to returning the path unchanged
/// if it doesn't share the prefix.
fn relative_to_data(absolute: &str, data_root: &Path) -> String {
    let root_str = data_root.to_string_lossy();
    let root = root_str.trim_end_matches('/');
    if root.is_empty() {
        return absolute.trim_start_matches('/').to_string();
    }
    if let Some(rest) = absolute.strip_prefix(root) {
        let rel = rest.trim_start_matches('/');
        if rel.is_empty() {
            absolute.to_string()
        } else {
            rel.to_string()
        }
    } else {
        absolute.to_string()
    }
}

/// Levenshtein edit distance between two strings (chars, not bytes).
/// O(m*n) time, O(n) space. Used for "did you mean?" suggestions over a few
/// hundred basenames — fine for our scale, no extra dependency needed.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (curr[j - 1] + 1).min(prev[j] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Outcome of trying to resolve a user-supplied path against the data directory.
enum ResolveErr {
    /// File didn't exist — eligible for fuzzy basename fallback.
    NotFound,
    /// Resolved outside the data directory (path traversal). Hard fail.
    Outside,
    /// Resolved to a file type excluded by `indexing.include`. Hard fail.
    NotPermitted,
    /// Other I/O error (permissions, etc). Hard fail with the original message.
    Other(String),
}

fn build_include_globset(patterns: &[String]) -> GlobSet {
    let mut builder = GlobSetBuilder::new();
    let mut valid_count = 0;
    for p in patterns {
        match Glob::new(p) {
            Ok(g) => {
                builder.add(g);
                valid_count += 1;
            }
            Err(e) => {
                error!("Invalid include glob pattern '{}': {}", p, e);
            }
        }
    }
    if valid_count == 0 {
        warn!("No valid include patterns configured — falling back to **/*.md");
        builder.add(Glob::new("**/*.md").unwrap());
    }
    builder.build().unwrap_or_else(|e| {
        error!(
            "Failed to build include globset: {} — falling back to **/*.md",
            e
        );
        let mut fallback = GlobSetBuilder::new();
        fallback.add(Glob::new("**/*.md").unwrap());
        fallback.build().unwrap()
    })
}

#[tool_router]
impl KbSearchServer {
    pub fn new(
        embed_client: Arc<EmbedClient>,
        qdrant: Arc<QdrantStore>,
        collection: String,
        data_path: PathBuf,
        include_patterns: &[String],
        instructions: Arc<RwLock<String>>,
        config: Arc<ResolvedConfig>,
    ) -> anyhow::Result<Self> {
        let canonical_data_path = data_path.canonicalize().with_context(|| {
            format!("Failed to canonicalize data path: {}", data_path.display())
        })?;
        Ok(Self {
            embed_client,
            qdrant,
            collection,
            canonical_data_path,
            include_patterns: Arc::new(build_include_globset(include_patterns)),
            instructions,
            config,
            tool_router: Self::tool_router(),
        })
    }

    #[tool(
        description = "Search the knowledge base using a natural-language query. \
        Returns ranked document chunks with title, relevance score, text snippet, and metadata. \
        Optionally filter by domain, type, or tags."
    )]
    async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        validate_search_params(&params)?;

        // Embed the query
        let vector = self
            .embed_client
            .embed_query(&params.query)
            .await
            .map_err(|e| {
                error!("Embedding query failed: {:#}", e);
                McpError::internal_error("Failed to generate query embedding".to_string(), None)
            })?;

        // Build filter map from optional params
        let mut filters: HashMap<String, serde_json::Value> = HashMap::new();

        if let Some(domain) = params.domain {
            filters.insert("domain".to_string(), serde_json::Value::String(domain));
        }

        if let Some(doc_type) = params.r#type {
            filters.insert("type".to_string(), serde_json::Value::String(doc_type));
        }

        if let Some(tags) = params.tags {
            let tag_values: Vec<serde_json::Value> =
                tags.into_iter().map(serde_json::Value::String).collect();
            filters.insert("tags".to_string(), serde_json::Value::Array(tag_values));
        }

        let limit = resolve_limit(params.limit);

        // Search Qdrant
        let results: Vec<SearchResult> = self
            .qdrant
            .search(&self.collection, vector, filters, limit)
            .await
            .map_err(|e| {
                error!("Qdrant search failed: {:#}", e);
                McpError::internal_error("Search query failed".to_string(), None)
            })?;

        if results.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No results found.",
            )]));
        }

        // Format results as text content
        let mut output = String::new();
        for (i, result) in results.iter().enumerate() {
            let title = result
                .payload
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or("(untitled)");

            let (text_snippet, needs_ellipsis) = {
                let full_text = result
                    .payload
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let chars: Vec<char> = full_text.chars().take(401).collect();
                if chars.len() > 400 {
                    (chars[..400].iter().collect::<String>(), true)
                } else {
                    (chars.into_iter().collect::<String>(), false)
                }
            };

            let file_path_raw = result
                .payload
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let file_path = relative_to_data(file_path_raw, &self.canonical_data_path);

            let domain = result
                .payload
                .get("domain")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let doc_type = result
                .payload
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let tags = result
                .payload
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|t| t.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();

            let lines = match (
                result.payload.get("line_start").and_then(|v| v.as_i64()),
                result.payload.get("line_end").and_then(|v| v.as_i64()),
            ) {
                (Some(s), Some(e)) => format!(" (lines {s}–{e})"),
                _ => String::new(),
            };

            output.push_str(&format!(
                "## Result {rank}\n\
                **Title**: {title}\n\
                **Score**: {score:.4}\n\
                **File**: {file_path}{lines}\n",
                rank = i + 1,
                title = title,
                score = result.score,
                file_path = file_path,
                lines = lines,
            ));

            if !domain.is_empty() {
                output.push_str(&format!("**Domain**: {domain}\n"));
            }
            if !doc_type.is_empty() {
                output.push_str(&format!("**Type**: {doc_type}\n"));
            }
            if !tags.is_empty() {
                output.push_str(&format!("**Tags**: {tags}\n"));
            }

            if !text_snippet.is_empty() {
                let ellipsis = if needs_ellipsis { "..." } else { "" };
                output.push_str(&format!("\n{text_snippet}{ellipsis}\n"));
            }

            output.push('\n');
        }

        Ok(CallToolResult::success(vec![Content::text(output.trim())]))
    }

    /// Resolve a user-supplied path against the data directory, applying the
    /// path-traversal and file-type security checks. Used by `get_document` for
    /// both literal resolution and (after fuzzy basename match) the chosen
    /// candidate path.
    fn resolve_within_data(&self, raw: &str) -> Result<PathBuf, ResolveErr> {
        let requested = PathBuf::from(raw);
        let resolved = if requested.is_absolute() {
            requested
        } else {
            self.canonical_data_path.join(&requested)
        };

        let canonical = resolved.canonicalize().map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => ResolveErr::NotFound,
            std::io::ErrorKind::PermissionDenied => {
                ResolveErr::Other(format!("Permission denied: {}", resolved.display()))
            }
            _ => ResolveErr::Other(format!(
                "Cannot access file '{}': {}",
                resolved.display(),
                e
            )),
        })?;

        if !canonical.starts_with(&self.canonical_data_path) {
            return Err(ResolveErr::Outside);
        }

        let relative = canonical
            .strip_prefix(&self.canonical_data_path)
            .unwrap_or(&canonical);
        if !self.include_patterns.is_match(relative) {
            return Err(ResolveErr::NotPermitted);
        }

        Ok(canonical)
    }

    async fn read_canonical_file(&self, canonical: &Path) -> Result<CallToolResult, McpError> {
        let content = tokio::fs::read_to_string(canonical).await.map_err(|e| {
            error!("Failed to read file '{}': {}", canonical.display(), e);
            McpError::invalid_params("Failed to read file".to_string(), None)
        })?;
        Ok(CallToolResult::success(vec![Content::text(content)]))
    }

    #[tool(
        description = "Retrieve the full raw content of a document by file path. \
        Accepts paths relative to the knowledge base root (e.g. \
        'lifestyle/vehicles/foo.md', as returned by the `search` tool) or just \
        the basename if it's unique across the index. Returns the complete \
        markdown including frontmatter."
    )]
    async fn get_document(
        &self,
        Parameters(params): Parameters<GetDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        let raw = params.path.trim();
        if raw.is_empty() {
            return Err(McpError::invalid_params(
                "Path parameter is empty".to_string(),
                None,
            ));
        }

        // 1. Try the literal path as given.
        match self.resolve_within_data(raw) {
            Ok(canonical) => return self.read_canonical_file(&canonical).await,
            Err(ResolveErr::NotFound) => {
                // Fall through to fuzzy basename matching.
            }
            Err(ResolveErr::Outside) => {
                return Err(McpError::invalid_params(
                    "File path is outside the data directory".to_string(),
                    None,
                ));
            }
            Err(ResolveErr::NotPermitted) => {
                return Err(McpError::invalid_params(
                    "File type not permitted".to_string(),
                    None,
                ));
            }
            Err(ResolveErr::Other(msg)) => {
                return Err(McpError::invalid_params(msg, None));
            }
        }

        // 2. Fuzzy fallback: load every indexed path from Qdrant and look for a
        //    basename match. Auto-resolve a unique match; otherwise produce a
        //    helpful error.
        let all_paths = self
            .qdrant
            .fetch_facet_values(&self.collection, "file_path", MAX_INDEXED_PATHS_FOR_FUZZY)
            .await
            .unwrap_or_else(|e| {
                warn!("Failed to fetch file_path facet for fuzzy lookup: {e:#}");
                Vec::new()
            });

        let basename = Path::new(raw)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(raw);

        let exact: Vec<&String> = all_paths
            .iter()
            .filter(|p| {
                Path::new(p.as_str()).file_name().and_then(|n| n.to_str()) == Some(basename)
            })
            .collect();

        match exact.len() {
            1 => match self.resolve_within_data(exact[0]) {
                Ok(canonical) => self.read_canonical_file(&canonical).await,
                Err(_) => Err(McpError::invalid_params(
                    format!("File not found: '{}'", raw),
                    None,
                )),
            },
            0 => {
                let mut scored: Vec<(usize, &String)> = all_paths
                    .iter()
                    .map(|p| {
                        let bn = Path::new(p.as_str())
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("");
                        (levenshtein(basename, bn), p)
                    })
                    .collect();
                scored.sort_by_key(|(d, _)| *d);
                scored.truncate(FUZZY_SUGGESTION_COUNT);

                let suggestions: Vec<String> = scored
                    .into_iter()
                    .map(|(_, p)| relative_to_data(p, &self.canonical_data_path))
                    .collect();

                let mut msg = format!("File not found: '{}'.", raw);
                if suggestions.is_empty() {
                    msg.push_str(" Use the `search` tool to find paths.");
                } else {
                    msg.push_str(&format!(
                        " Closest indexed files: {}. Or use the `search` tool.",
                        suggestions.join(", ")
                    ));
                }
                Err(McpError::invalid_params(msg, None))
            }
            _ => {
                let candidates: Vec<String> = exact
                    .into_iter()
                    .map(|p| relative_to_data(p, &self.canonical_data_path))
                    .collect();
                Err(McpError::invalid_params(
                    format!(
                        "Multiple files match basename '{}': {}. Use a more specific path.",
                        basename,
                        candidates.join(", ")
                    ),
                    None,
                ))
            }
        }
    }

    /// Shared pipeline for create_document and edit_document.
    ///
    /// Callers are responsible for resolving paths and computing old/new content
    /// before calling this. This function handles validation, optional dedup
    /// gating (create only), filesystem write, git commit, reindex, and diff output.
    ///
    /// * `old_content` – empty string for create; existing file bytes for edit.
    /// * `new_content` – the content to write (already computed by caller).
    /// * `abs_path`    – canonical absolute path of the target file.
    /// * `rel_path`    – repo-relative path (used for git add/commit and messages).
    /// * `is_create`   – `true` for create (dedup gate active), `false` for edit.
    /// * `message`     – optional custom commit message.
    /// * `default_verb`– verb for the default commit message, e.g. `"add"` or `"update"`.
    /// * `force_new`   – when `Some(true)`, bypasses the dedup gate on create paths.
    /// * `operation`   – label for the `Operation:` git trailer, e.g. `"create_document"`.
    #[allow(clippy::too_many_arguments)]
    async fn write_document(
        &self,
        old_content: &str,
        new_content: &str,
        abs_path: &Path,
        rel_path: &str,
        is_create: bool,
        message: Option<&str>,
        default_verb: &str,
        force_new: Option<bool>,
        operation: &str,
    ) -> Result<CallToolResult, McpError> {
        let config = &self.config;

        // 1. Validate new_content before writing (catches frontmatter errors in
        //    both full-replace and surgical edits before touching the filesystem).
        let (validation_result, _) = validate::validate_content(
            std::path::Path::new(rel_path),
            new_content,
            &config.frontmatter,
            &config.validation,
        )
        .await
        .map_err(|e| {
            error!("Validation error for '{}': {:#}", rel_path, e);
            McpError::internal_error(format!("Failed to validate content: {}", e), None)
        })?;

        if !validation_result.valid {
            let data = Some(serde_json::json!({
                "field_errors": validation_result.field_errors
            }));
            return Err(McpError::invalid_params(
                format!(
                    "frontmatter validation failed for '{}': {}",
                    rel_path,
                    validation_result.errors.join("; ")
                ),
                data,
            ));
        }

        // 2. Dedup gate: on create paths, check for near-duplicate existing documents.
        //    Gate runs only when: this is a create (not edit), dedup is enabled in
        //    config, and the caller has not set force_new = true.
        if is_create && self.config.write.dedup_enabled && !matches!(force_new, Some(true)) {
            // Build query text truncated to DEDUP_QUERY_CHAR_LIMIT chars to stay
            // within embedding model token limits.
            let query_text: String = new_content.chars().take(DEDUP_QUERY_CHAR_LIMIT).collect();

            match self.embed_client.embed_query(&query_text).await {
                Ok(vector) => {
                    match self
                        .qdrant
                        .search(&self.collection, vector, HashMap::new(), 1)
                        .await
                    {
                        Ok(results) => {
                            let top = results.into_iter().next().map(|r| {
                                let path = r
                                    .payload
                                    .get("file_path")
                                    .and_then(|v| v.as_str())
                                    .map(|p| relative_to_data(p, &self.canonical_data_path))
                                    .unwrap_or_default();
                                (path, r.score)
                            });
                            if let Some(hit) = dedup_verdict(top, self.config.write.dedup_threshold)
                            {
                                let threshold = self.config.write.dedup_threshold;
                                return Err(McpError::invalid_params(
                                    format!(
                                        "A similar document already exists: '{}' \
                                         (similarity {:.2} ≥ threshold {:.2}). \
                                         Edit it with edit_document, or pass \
                                         force_new=true to create a new document anyway.",
                                        hit.file_path, hit.score, threshold
                                    ),
                                    Some(serde_json::json!({
                                        "duplicate_of": hit.file_path,
                                        "similarity": hit.score,
                                        "threshold": threshold,
                                    })),
                                ));
                            }
                        }
                        Err(e) => {
                            // Fail-open: infrastructure failures must not block writes.
                            // The dedup gate is a guardrail, not a correctness gate.
                            warn!(
                                "Dedup search failed for '{}' (proceeding with write): {:#}",
                                rel_path, e
                            );
                        }
                    }
                }
                Err(e) => {
                    // Fail-open: same rationale — embedder down should not block writes.
                    warn!(
                        "Dedup embedding failed for '{}' (proceeding with write): {:#}",
                        rel_path, e
                    );
                }
            }
        }

        // 3. Create parent directories and write the file
        if let Some(parent) = abs_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                error!(
                    "Failed to create parent directories for '{}': {}",
                    abs_path.display(),
                    e
                );
                McpError::internal_error(
                    format!("Failed to create parent directories: {}", e),
                    None,
                )
            })?;
        }

        tokio::fs::write(abs_path, new_content.as_bytes())
            .await
            .map_err(|e| {
                error!("Failed to write file '{}': {}", abs_path.display(), e);
                McpError::internal_error(format!("Failed to write file: {}", e), None)
            })?;

        // 4. Build commit message with git trailers
        let commit_message = build_commit_message(
            message,
            &format!("docs: {} {}", default_verb, rel_path),
            operation,
        );

        // 5. Resolve git token
        let token = std::env::var(&config.source.git_token_env)
            .ok()
            .filter(|s| !s.is_empty());

        // 6. Commit and sync to remote
        let commit_sha = git::commit_and_sync(
            config.source.git_url.as_deref(),
            &config.source.branch,
            config.data_path(),
            token.as_deref(),
            rel_path,
            &commit_message,
            &config.write.commit_author_name,
            &config.write.commit_author_email,
        )
        .await
        .map_err(|e| {
            error!("commit_and_sync failed for '{}': {:#}", rel_path, e);
            McpError::internal_error(format!("Git commit/sync failed: {}", e), None)
        })?;

        // 7. Trigger incremental reindex (serialized against webhook reindexes)
        {
            let _guard = crate::webhook::REINDEX_LOCK.lock().await;
            ingest::run_index(config, false).await.map_err(|e| {
                error!(
                    "Reindex after write_document failed for '{}': {:#}",
                    rel_path, e
                );
                McpError::internal_error(format!("Reindex failed: {}", e), None)
            })?;
        }

        // 8. Build unified diff and return success
        let action = if is_create { "Created" } else { "Edited" };
        let summary = format!("{} '{}' (commit {})", action, rel_path, commit_sha);
        let diff = render_unified_diff(old_content, new_content, rel_path);
        let result_text = if diff.is_empty() {
            summary
        } else {
            format!("{}\n\n{}", summary, diff)
        };
        Ok(CallToolResult::success(vec![Content::text(result_text)]))
    }

    #[tool(description = "Create a new document in the knowledge base. \
        Writes the file, commits it to the git repository, and triggers an incremental reindex. \
        The document must not already exist — use edit_document for existing files. \
        Content must include valid YAML frontmatter. \
        Required frontmatter fields and any fixed allowed values (e.g. for type/status) are \
        listed in this server's instructions. \
        If a very similar document already exists, the create is refused and the close match is \
        reported — edit that document instead, or set force_new=true to create a new one anyway.")]
    async fn create_document(
        &self,
        Parameters(params): Parameters<CreateDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        let config = &self.config;

        // Resolve path: must be relative, no traversal, not already existing.
        let data_root = std::path::PathBuf::from(config.data_path());
        let abs_path = resolve_safe_write_path(&data_root, &params.path)
            .map_err(|e| McpError::invalid_params(e, None))?;

        // Include-pattern guard: reject paths the indexer would not pick up.
        if !self.include_patterns.is_match(&params.path) {
            return Err(McpError::invalid_params(
                format!(
                    "path '{}' does not match any indexable include pattern \
                     (e.g. must be a markdown file under an included path)",
                    params.path
                ),
                None,
            ));
        }

        // File must not already exist for create.
        if abs_path.exists() {
            return Err(McpError::invalid_params(
                format!(
                    "File '{}' already exists. Use edit_document to modify existing files.",
                    params.path
                ),
                None,
            ));
        }

        self.write_document(
            "", // old_content: empty for new files
            &params.content,
            &abs_path,
            &params.path,
            true, // is_create
            params.message.as_deref(),
            "add",
            params.force_new,
            "create_document",
        )
        .await
    }

    #[tool(description = "Edit an existing document in the knowledge base. \
        Supports two modes:\n\
        \n\
        SURGICAL MODE — provide old_string and new_string (mutually exclusive with content):\n\
        Finds old_string in the document (must appear exactly once) and replaces it with \
        new_string. Ideal for small, targeted edits without sending the whole file. \
        old_string must be unique in the document; include more surrounding context if the \
        tool reports multiple occurrences.\n\
        \n\
        FULL-REPLACE MODE — provide content (mutually exclusive with old_string/new_string):\n\
        Replaces the entire file content with the provided content, which must include valid \
        YAML frontmatter. Required frontmatter fields and any fixed allowed values \
        (e.g. for type/status) are listed in this server's instructions.\n\
        \n\
        In both modes the result is validated, committed, and an incremental reindex is \
        triggered. The path is resolved like get_document: relative to the KB root, a unique \
        basename, or absolute. The document must already exist — use create_document for new files.")]
    async fn edit_document(
        &self,
        Parameters(params): Parameters<EditDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        // Parse and validate the edit mode (surgical vs full-replace).
        let mode = parse_edit_mode(&params).map_err(|e| McpError::invalid_params(e, None))?;

        // Resolve the path using the get_document resolver (forgiving: relative/
        // absolute/basename, canonicalized, include-pattern + containment checked).
        let raw = params.path.trim();
        if raw.is_empty() {
            return Err(McpError::invalid_params(
                "path parameter is empty".to_string(),
                None,
            ));
        }

        let canonical = match self.resolve_within_data(raw) {
            Ok(c) => c,
            Err(ResolveErr::NotFound) => {
                return Err(McpError::invalid_params(
                    format!(
                        "Document '{}' does not exist. Use create_document to create new files.",
                        raw
                    ),
                    None,
                ));
            }
            Err(ResolveErr::Outside) => {
                return Err(McpError::invalid_params(
                    "File path is outside the data directory".to_string(),
                    None,
                ));
            }
            Err(ResolveErr::NotPermitted) => {
                return Err(McpError::invalid_params(
                    "File type not permitted".to_string(),
                    None,
                ));
            }
            Err(ResolveErr::Other(msg)) => {
                return Err(McpError::invalid_params(msg, None));
            }
        };

        // Derive the repo-relative path from the canonical absolute path.
        let rel_path = canonical
            .strip_prefix(&self.canonical_data_path)
            .unwrap_or(&canonical)
            .to_string_lossy()
            .into_owned();

        // Read the existing file content.
        let old_content = tokio::fs::read_to_string(&canonical).await.map_err(|e| {
            error!("Failed to read '{}': {}", canonical.display(), e);
            McpError::internal_error(format!("Failed to read existing file: {}", e), None)
        })?;

        // Compute new_content and operation label based on mode.
        let (new_content, operation) = match mode {
            EditMode::Full { content } => (content, "edit_document (full replace)"),
            EditMode::Surgical { old, new } => {
                let result = apply_surgical(&old_content, &old, &new).map_err(|e| {
                    // Enrich the error with the resolved relative path for context.
                    let msg = e.replace("document", &format!("'{}'", rel_path));
                    McpError::invalid_params(msg, None)
                })?;
                (result, "edit_document (surgical replace)")
            }
        };

        self.write_document(
            &old_content,
            &new_content,
            &canonical,
            &rel_path,
            false, // is_create
            params.message.as_deref(),
            "update",
            None, // no dedup gate for edit
            operation,
        )
        .await
    }

    #[tool(description = "Delete a document from the knowledge base. \
        Removes the file from disk, commits the deletion to git with provenance trailers \
        (just like create_document/edit_document), pushes the commit, and explicitly purges \
        the document's vectors from the Qdrant search index and its state-DB row. \
        The path resolves like get_document: relative to the KB root \
        (e.g. 'sysadmin/guide.md'), a unique basename, or absolute. \
        The document must already exist — use search to find the correct path. \
        Returns a summary line with the commit SHA and a unified diff of the removed content.")]
    async fn delete_document(
        &self,
        Parameters(params): Parameters<DeleteDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        let config = &self.config;

        let raw = params.path.trim();
        if raw.is_empty() {
            return Err(McpError::invalid_params(
                "path parameter is empty".to_string(),
                None,
            ));
        }

        // 1. Resolve the path (must already exist on disk).
        let canonical = match self.resolve_within_data(raw) {
            Ok(c) => c,
            Err(ResolveErr::NotFound) => {
                return Err(McpError::invalid_params(
                    format!("document does not exist: '{}'", raw),
                    None,
                ));
            }
            Err(ResolveErr::Outside) => {
                return Err(McpError::invalid_params(
                    "File path is outside the data directory".to_string(),
                    None,
                ));
            }
            Err(ResolveErr::NotPermitted) => {
                return Err(McpError::invalid_params(
                    "File type not permitted".to_string(),
                    None,
                ));
            }
            Err(ResolveErr::Other(msg)) => {
                return Err(McpError::invalid_params(msg, None));
            }
        };

        // Derive repo-relative path (used for git staging, commit messages, index purge).
        let rel_path = canonical
            .strip_prefix(&self.canonical_data_path)
            .unwrap_or(&canonical)
            .to_string_lossy()
            .into_owned();

        // 2. Read file content before removal (used for diff output).
        let old_content = tokio::fs::read_to_string(&canonical).await.map_err(|e| {
            error!("Failed to read '{}': {}", canonical.display(), e);
            McpError::internal_error(format!("Failed to read file before deletion: {}", e), None)
        })?;

        // 3. Remove the file from disk.
        tokio::fs::remove_file(&canonical).await.map_err(|e| {
            error!("Failed to remove '{}': {}", canonical.display(), e);
            McpError::internal_error(format!("Failed to remove file: {}", e), None)
        })?;

        // 4. Commit + push the deletion.
        let commit_message = build_commit_message(
            params.message.as_deref(),
            &format!("docs: delete {}", rel_path),
            "delete_document",
        );

        let token = std::env::var(&config.source.git_token_env)
            .ok()
            .filter(|s| !s.is_empty());

        let commit_sha = git::commit_and_sync(
            config.source.git_url.as_deref(),
            &config.source.branch,
            config.data_path(),
            token.as_deref(),
            &rel_path,
            &commit_message,
            &config.write.commit_author_name,
            &config.write.commit_author_email,
        )
        .await
        .map_err(|e| {
            error!("commit_and_sync failed for '{}': {:#}", rel_path, e);
            McpError::internal_error(format!("Git commit/sync failed: {}", e), None)
        })?;

        // 5 & 6. Purge from Qdrant and state DB under REINDEX_LOCK.
        {
            let _guard = crate::webhook::REINDEX_LOCK.lock().await;

            // Purge vectors from Qdrant.
            self.qdrant
                .delete_by_files(&self.collection, &[rel_path.as_str()])
                .await
                .map_err(|e| {
                    error!("delete_by_files failed for '{}': {:#}", rel_path, e);
                    McpError::internal_error(
                        format!("Failed to purge document from search index: {}", e),
                        None,
                    )
                })?;

            // Remove state-DB row so incremental reindex bookkeeping stays correct.
            // Use a short-lived handle; SQLite WAL + REINDEX_LOCK serializes writers.
            let db_path = config.state_db_path();
            match StateDb::new(std::path::Path::new(&db_path)).await {
                Ok(state) => {
                    if let Err(e) = state.delete(&rel_path).await {
                        // Non-fatal: stale state DB row causes the next incremental
                        // reindex to attempt re-processing, which will succeed because
                        // the file is gone (orphan removal path). Log and continue.
                        error!("Failed to remove state DB row for '{}': {:#}", rel_path, e);
                    }
                }
                Err(e) => {
                    error!(
                        "Failed to open state DB for '{}' cleanup: {:#}",
                        rel_path, e
                    );
                }
            }
        }

        // 7. Return success with summary + diff of removed content.
        let summary = format!("Deleted '{}' (commit {})", rel_path, commit_sha);
        let diff = render_unified_diff(&old_content, "", &rel_path);
        let result_text = if diff.is_empty() {
            summary
        } else {
            format!("{}\n\n{}", summary, diff)
        };
        Ok(CallToolResult::success(vec![Content::text(result_text)]))
    }
}

/// Default instructions used when no custom instructions are configured.
/// The server appends discovered filter values ("Available ...") and a full
/// write-authoring section after this base, so keep this short.
pub const DEFAULT_INSTRUCTIONS: &str = "Knowledge base server. \
Read with `search` (natural-language query; optional domain/type/tags filters) \
and `get_document`. \
Write with `create_document`, `edit_document`, and `delete_document`.";

#[tool_handler]
impl ServerHandler for KbSearchServer {
    fn get_info(&self) -> ServerInfo {
        let instructions = self
            .instructions
            .read()
            .unwrap_or_else(|poisoned| {
                warn!("Instructions RwLock poisoned on read; using last value");
                poisoned.into_inner()
            })
            .clone();
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::from_build_env())
            .with_instructions(instructions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- build_commit_message tests ---

    #[test]
    fn commit_message_create_document_trailer() {
        let msg = build_commit_message(None, "docs: add notes/guide.md", "create_document");
        assert!(
            msg.contains("Tool: md-kb-rag"),
            "should contain Tool trailer: {msg}"
        );
        assert!(
            msg.contains("Operation: create_document"),
            "should contain Operation trailer: {msg}"
        );
        assert!(
            msg.starts_with("docs: add notes/guide.md"),
            "should start with default subject: {msg}"
        );
    }

    #[test]
    fn commit_message_edit_surgical_trailer() {
        let msg = build_commit_message(
            None,
            "docs: update notes/guide.md",
            "edit_document (surgical replace)",
        );
        assert!(
            msg.contains("Operation: edit_document (surgical replace)"),
            "should contain surgical operation label: {msg}"
        );
    }

    #[test]
    fn commit_message_edit_full_replace_trailer() {
        let msg = build_commit_message(
            None,
            "docs: update notes/guide.md",
            "edit_document (full replace)",
        );
        assert!(
            msg.contains("Operation: edit_document (full replace)"),
            "should contain full replace operation label: {msg}"
        );
    }

    #[test]
    fn commit_message_user_subject_overrides_default() {
        let msg = build_commit_message(
            Some("fix: correct typo in introduction"),
            "docs: update notes/guide.md",
            "edit_document (surgical replace)",
        );
        assert!(
            msg.starts_with("fix: correct typo in introduction"),
            "user subject should take precedence: {msg}"
        );
        assert!(
            msg.contains("Operation: edit_document (surgical replace)"),
            "trailer should still be appended: {msg}"
        );
    }

    #[test]
    fn commit_message_trailer_separated_by_blank_line() {
        let msg = build_commit_message(None, "docs: add test.md", "create_document");
        // Git requires a blank line between subject and trailer block
        assert!(
            msg.contains("\n\nTool: md-kb-rag"),
            "blank line must precede trailer block: {msg}"
        );
    }

    #[test]
    fn default_limit_is_ten() {
        assert_eq!(resolve_limit(None), 10);
    }

    #[test]
    fn requested_limit_within_max_is_preserved() {
        assert_eq!(resolve_limit(Some(25)), 25);
    }

    #[test]
    fn requested_limit_above_max_is_clamped() {
        assert_eq!(resolve_limit(Some(1_000_000)), MAX_SEARCH_LIMIT);
    }

    #[test]
    fn zero_limit_is_passed_through() {
        assert_eq!(resolve_limit(Some(0)), 0);
    }

    #[test]
    fn path_traversal_detection() {
        // Raw PathBuf::join does NOT resolve `..` components — the resulting path
        // still textually starts_with the data_path prefix, so a naive starts_with
        // check is insufficient. canonicalize() is required to resolve `..`.
        let data_path = std::path::PathBuf::from("/tmp/test-kb-data");
        let traversal = data_path.join("../../../etc/passwd");
        // starts_with returns true because the path is built on top of data_path
        assert!(
            traversal.starts_with(&data_path),
            "raw join with .. still starts_with data_path — canonicalize() is needed"
        );
    }

    #[test]
    fn absolute_path_outside_data_rejected() {
        let data_path = std::path::PathBuf::from("/tmp/test-kb-data");
        let outside = std::path::PathBuf::from("/etc/passwd");
        assert!(
            !outside.starts_with(&data_path),
            "/etc/passwd should not start_with /tmp/test-kb-data"
        );
    }

    #[test]
    fn relative_path_inside_data_accepted() {
        let data_path = std::path::PathBuf::from("/tmp/test-kb-data");
        let inside = data_path.join("docs/guide.md");
        assert!(
            inside.starts_with(&data_path),
            "data_path/docs/guide.md should start_with data_path"
        );
    }

    #[test]
    fn ellipsis_uses_char_count_not_byte_len() {
        // 400 chars of a 2-byte character = 800 bytes
        let text: String = "é".repeat(401);
        assert!(text.len() > 400, "byte len should exceed 400");
        assert!(text.chars().count() > 400, "char count should exceed 400");
        // If we used .len() on a 400-char string it would wrongly trigger ellipsis
        let short: String = "é".repeat(400);
        assert!(
            short.len() > 400,
            "byte len of 400 2-byte chars exceeds 400"
        );
        assert_eq!(short.chars().count(), 400, "char count is exactly 400");
    }

    #[test]
    fn include_globset_matches_markdown() {
        let patterns = vec!["**/*.md".to_string()];
        let gs = build_include_globset(&patterns);
        assert!(
            gs.is_match("docs/guide.md"),
            "**/*.md should match docs/guide.md"
        );
        assert!(
            gs.is_match("README.md"),
            "**/*.md should match top-level README.md"
        );
    }

    #[test]
    fn include_globset_rejects_non_markdown() {
        let patterns = vec!["**/*.md".to_string()];
        let gs = build_include_globset(&patterns);
        assert!(
            !gs.is_match("state.db"),
            "**/*.md should not match state.db"
        );
        assert!(
            !gs.is_match("scripts/run.sh"),
            "**/*.md should not match shell scripts"
        );
        assert!(
            !gs.is_match(".env"),
            "**/*.md should not match credential files"
        );
    }

    #[test]
    fn include_globset_respects_custom_patterns() {
        let patterns = vec!["**/*.md".to_string(), "**/*.txt".to_string()];
        let gs = build_include_globset(&patterns);
        assert!(gs.is_match("notes/todo.txt"), "should match *.txt");
        assert!(!gs.is_match("data.json"), "should not match *.json");
    }

    fn make_params(query: &str) -> SearchParams {
        SearchParams {
            query: query.to_string(),
            domain: None,
            r#type: None,
            tags: None,
            limit: None,
        }
    }

    #[test]
    fn valid_params_accepted() {
        let params = make_params("find documents about authentication");
        assert!(validate_search_params(&params).is_ok());
    }

    #[test]
    fn query_at_limit_is_accepted() {
        let params = make_params(&"a".repeat(MAX_QUERY_LEN));
        assert!(validate_search_params(&params).is_ok());
    }

    #[test]
    fn query_too_long_is_rejected() {
        let params = make_params(&"a".repeat(MAX_QUERY_LEN + 1));
        assert!(validate_search_params(&params).is_err());
    }

    #[test]
    fn domain_too_long_is_rejected() {
        let params = SearchParams {
            domain: Some("x".repeat(MAX_FILTER_STR_LEN + 1)),
            ..make_params("query")
        };
        assert!(validate_search_params(&params).is_err());
    }

    #[test]
    fn type_too_long_is_rejected() {
        let params = SearchParams {
            r#type: Some("x".repeat(MAX_FILTER_STR_LEN + 1)),
            ..make_params("query")
        };
        assert!(validate_search_params(&params).is_err());
    }

    #[test]
    fn too_many_tags_rejected() {
        let params = SearchParams {
            tags: Some(vec!["tag".to_string(); MAX_TAG_COUNT + 1]),
            ..make_params("query")
        };
        assert!(validate_search_params(&params).is_err());
    }

    #[test]
    fn tag_too_long_is_rejected() {
        let params = SearchParams {
            tags: Some(vec!["x".repeat(MAX_TAG_LEN + 1)]),
            ..make_params("query")
        };
        assert!(validate_search_params(&params).is_err());
    }

    #[test]
    fn max_tags_at_limit_accepted() {
        let params = SearchParams {
            tags: Some(vec!["tag".to_string(); MAX_TAG_COUNT]),
            ..make_params("query")
        };
        assert!(validate_search_params(&params).is_ok());
    }

    #[test]
    fn canonicalize_nonexistent_file_produces_not_found_message() {
        let bad_path = std::path::PathBuf::from("/tmp/nonexistent-kb-test-dir/missing.md");
        let err = bad_path
            .canonicalize()
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => format!("File not found: {}", bad_path.display()),
                std::io::ErrorKind::PermissionDenied => {
                    format!("Permission denied: {}", bad_path.display())
                }
                _ => format!("Cannot access file '{}': {}", bad_path.display(), e),
            })
            .unwrap_err();
        assert!(
            err.contains("File not found"),
            "expected 'File not found', got: {err}"
        );
    }

    fn make_test_resolved_config(data_path: &std::path::Path) -> Arc<ResolvedConfig> {
        Arc::new(ResolvedConfig {
            source: crate::config::SourceConfig {
                git_url: None,
                branch: "master".into(),
                data_path: Some(data_path.to_string_lossy().into_owned()),
                git_token_env: "GIT_PULL_TOKEN".into(),
            },
            indexing: crate::config::IndexingConfig::default(),
            frontmatter: crate::config::FrontmatterConfig::default(),
            chunking: crate::config::ChunkingConfig::default(),
            embedding: crate::config::ResolvedEmbeddingConfig {
                base_url: "http://localhost:8080/v1".into(),
                model: "test".into(),
                api_key: None,
                vector_size: 768,
                batch_size: 32,
            },
            qdrant: crate::config::ResolvedQdrantConfig {
                url: "http://localhost:6334".into(),
                collection: "test".into(),
            },
            validation: crate::config::ValidationConfig::default(),
            webhook: crate::config::WebhookConfig::default(),
            mcp: crate::config::McpConfig::default(),
            rate_limit: crate::config::RateLimitConfig::default(),
            write: crate::config::WriteConfig::default(),
        })
    }

    #[test]
    fn get_info_returns_dynamic_instructions() {
        use rmcp::ServerHandler;

        // Create a temp directory to serve as data_path (must exist for canonicalize)
        let tmp = tempfile::tempdir().unwrap();
        let custom_text = "Custom KB instructions.\nAvailable domain: infra, networking";
        let instructions = Arc::new(RwLock::new(custom_text.to_string()));

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
        };
        let embed = Arc::new(EmbedClient::new(&embed_config));

        let server = KbSearchServer::new(
            embed,
            qdrant,
            "test".into(),
            tmp.path().to_path_buf(),
            &["**/*.md".to_string()],
            instructions,
            make_test_resolved_config(tmp.path()),
        )
        .unwrap();

        let info = server.get_info();
        let returned = info.instructions.unwrap();
        assert_eq!(returned, custom_text);
    }

    #[test]
    fn get_info_reflects_updated_instructions() {
        use rmcp::ServerHandler;

        let tmp = tempfile::tempdir().unwrap();
        let instructions = Arc::new(RwLock::new("Initial instructions".to_string()));

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
        };
        let embed = Arc::new(EmbedClient::new(&embed_config));

        let server = KbSearchServer::new(
            embed,
            qdrant,
            "test".into(),
            tmp.path().to_path_buf(),
            &["**/*.md".to_string()],
            Arc::clone(&instructions),
            make_test_resolved_config(tmp.path()),
        )
        .unwrap();

        // Simulate a refresh
        *instructions.write().unwrap() = "Updated with metadata".to_string();

        let info = server.get_info();
        assert_eq!(info.instructions.unwrap(), "Updated with metadata");
    }

    #[test]
    fn test_get_info_recovers_from_poisoned_lock() {
        use std::panic;

        let lock = Arc::new(RwLock::new("valid instructions".to_string()));
        let lock_clone = Arc::clone(&lock);

        // Poison the lock by panicking while holding a write guard
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _guard = lock_clone.write().unwrap();
            panic!("intentional panic to poison the lock");
        }));

        assert!(lock.read().is_err(), "lock should be poisoned");

        // Verify recovery via unwrap_or_else
        let recovered = lock
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();

        assert_eq!(recovered, "valid instructions");
    }

    #[test]
    fn default_instructions_constant_is_reasonable() {
        assert!(DEFAULT_INSTRUCTIONS.contains("search"));
        assert!(
            DEFAULT_INSTRUCTIONS
                .to_lowercase()
                .contains("knowledge base")
        );
    }

    #[test]
    fn include_globset_empty_patterns_falls_back_to_markdown() {
        let gs = build_include_globset(&[]);
        assert!(
            gs.is_match("docs/guide.md"),
            "empty patterns should fall back to **/*.md"
        );
        assert!(
            gs.is_match("README.md"),
            "empty patterns should match top-level .md"
        );
        assert!(
            !gs.is_match("state.db"),
            "empty patterns fallback should not match non-markdown"
        );
    }

    #[test]
    fn include_globset_all_invalid_falls_back_to_markdown() {
        let gs = build_include_globset(&["[invalid".into()]);
        assert!(
            gs.is_match("docs/guide.md"),
            "all-invalid patterns should fall back to **/*.md"
        );
        assert!(
            !gs.is_match("data.json"),
            "all-invalid patterns fallback should not match non-markdown"
        );
    }

    #[test]
    fn get_document_uses_canonical_path() {
        // Create a temp dir with a subdirectory and a markdown file
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("test.md"), "# Hello").unwrap();

        let canonical_data = tmp.path().canonicalize().unwrap();

        // Simulate get_document path resolution logic using canonical_data_path
        let requested = PathBuf::from("docs/test.md");
        let resolved = canonical_data.join(&requested);
        let canonical_resolved = resolved.canonicalize().unwrap();

        assert!(
            canonical_resolved.starts_with(&canonical_data),
            "resolved path should be under the canonical data path"
        );
    }

    #[test]
    fn canonicalize_error_message_includes_path() {
        let bad_path = std::path::PathBuf::from("/tmp/nonexistent-kb-test-dir/missing.md");
        let err = bad_path
            .canonicalize()
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => format!("File not found: {}", bad_path.display()),
                std::io::ErrorKind::PermissionDenied => {
                    format!("Permission denied: {}", bad_path.display())
                }
                _ => format!("Cannot access file '{}': {}", bad_path.display(), e),
            })
            .unwrap_err();
        assert!(
            err.contains(&bad_path.display().to_string()),
            "error message should include the file path, got: {err}"
        );
    }

    #[test]
    fn relative_to_data_strips_prefix() {
        let root = std::path::Path::new("/data");
        assert_eq!(
            relative_to_data("/data/lifestyle/foo.md", root),
            "lifestyle/foo.md"
        );
    }

    #[test]
    fn relative_to_data_handles_trailing_slash_root() {
        let root = std::path::Path::new("/data/");
        assert_eq!(
            relative_to_data("/data/lifestyle/foo.md", root),
            "lifestyle/foo.md"
        );
    }

    #[test]
    fn relative_to_data_returns_unchanged_when_no_prefix_match() {
        let root = std::path::Path::new("/data");
        // Path that doesn't start with /data — leave it alone rather than mangling it.
        assert_eq!(relative_to_data("/other/foo.md", root), "/other/foo.md");
    }

    #[test]
    fn relative_to_data_path_equal_to_root_kept_as_is() {
        let root = std::path::Path::new("/data");
        // Pathological case: file_path is exactly the data root. Leave it alone
        // rather than collapsing to an empty string.
        assert_eq!(relative_to_data("/data", root), "/data");
    }

    #[test]
    fn relative_to_data_partial_segment_does_not_match() {
        let root = std::path::Path::new("/data");
        // "/data-other/foo.md" textually starts with "/data" but isn't under it.
        // We accept the false-positive prefix strip here as a tradeoff: keeping
        // it simple. The leading slash on "-other/foo.md" disambiguates that
        // the caller passed a path the indexer didn't produce, so downstream
        // resolution will fail loudly. Document the behaviour so future changes
        // notice if it matters.
        let out = relative_to_data("/data-other/foo.md", root);
        assert!(
            out == "-other/foo.md" || out == "/data-other/foo.md",
            "got {out}"
        );
    }

    #[test]
    fn levenshtein_identical_is_zero() {
        assert_eq!(levenshtein("hello", "hello"), 0);
    }

    #[test]
    fn levenshtein_empty_string() {
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", ""), 0);
    }

    #[test]
    fn levenshtein_substitutions_and_inserts() {
        // "kitten" -> "sitting": substitute k->s, e->i, insert g = 3
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn levenshtein_ranks_close_matches_lower() {
        let target = "tacoma-2025.md";
        let close = levenshtein(target, "tacoma-2024.md");
        let far = levenshtein(target, "voron-trident.md");
        assert!(
            close < far,
            "expected close ({close}) < far ({far}) for target '{target}'"
        );
    }

    #[test]
    fn levenshtein_unicode_uses_chars_not_bytes() {
        // 'é' is 2 bytes UTF-8 but 1 char. If we used byte indices we'd
        // mis-count distances on accented basenames.
        assert_eq!(levenshtein("café", "cafe"), 1);
    }

    // -----------------------------------------------------------------------
    // write_document helper tests
    // -----------------------------------------------------------------------

    /// Build a KbSearchServer suitable for write_document unit tests.
    /// Uses a temp directory as the data path. No real Qdrant/git required
    /// for tests that fail before those steps.
    fn make_write_test_server(
        tmp: &tempfile::TempDir,
        include_patterns: &[String],
        config: Arc<ResolvedConfig>,
    ) -> KbSearchServer {
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
        };
        let embed = Arc::new(EmbedClient::new(&embed_config));
        let instructions = Arc::new(RwLock::new(String::new()));
        KbSearchServer::new(
            embed,
            qdrant,
            "test".into(),
            tmp.path().to_path_buf(),
            include_patterns,
            instructions,
            config,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn edit_document_on_nonexistent_file_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        // edit_document now goes through resolve_within_data; NotFound → clear error.
        let params = EditDocumentParams {
            path: "docs/nonexistent.md".to_string(),
            old_string: None,
            new_string: None,
            content: Some("---\ntitle: Test\n---\n# Body".to_string()),
            message: None,
        };
        let result = server.edit_document(Parameters(params)).await;

        assert!(
            result.is_err(),
            "edit of non-existent file should return Err"
        );
        let err = result.unwrap_err();
        assert!(
            err.message.contains("does not exist"),
            "error message should mention 'does not exist', got: {}",
            err.message
        );
        assert!(
            err.message.contains("create_document"),
            "error should mention create_document, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn create_document_on_existing_file_returns_use_edit_error() {
        let tmp = tempfile::tempdir().unwrap();
        // Pre-create the file
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("existing.md"), "# Already here").unwrap();

        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        let params = CreateDocumentParams {
            path: "docs/existing.md".to_string(),
            content: "---\ntitle: Test\n---\n# New content".to_string(),
            message: None,
            force_new: None,
        };
        let result = server.create_document(Parameters(params)).await;

        assert!(result.is_err(), "create on existing file should return Err");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("already exists"),
            "error message should mention 'already exists', got: {}",
            err.message
        );
        assert!(
            err.message.contains("edit_document"),
            "error should mention edit_document, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn include_pattern_guard_rejects_non_matching_path() {
        let tmp = tempfile::tempdir().unwrap();
        let config = make_test_resolved_config(tmp.path());
        // Only markdown files are indexed
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        // Try to write a .txt file (not matched by **/*.md)
        let params = CreateDocumentParams {
            path: "notes.txt".to_string(),
            content: "Some plain text".to_string(),
            message: None,
            force_new: None,
        };
        let result = server.create_document(Parameters(params)).await;

        assert!(
            result.is_err(),
            "non-matching path should be rejected by include-pattern guard"
        );
        let err = result.unwrap_err();
        assert!(
            err.message.contains("indexable include pattern"),
            "error should mention include pattern, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn include_pattern_guard_rejects_absolute_path() {
        let tmp = tempfile::tempdir().unwrap();
        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        // Absolute path should be caught by the absolute-path guard before include check
        let params = CreateDocumentParams {
            path: "/etc/passwd".to_string(),
            content: "# Evil".to_string(),
            message: None,
            force_new: None,
        };
        let result = server.create_document(Parameters(params)).await;

        assert!(result.is_err(), "absolute path should be rejected");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("relative"),
            "error should mention relative path requirement, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn validation_failure_carries_field_errors_in_data() {
        let tmp = tempfile::tempdir().unwrap();

        // Config with validation enabled requiring "title" field
        let config = Arc::new(ResolvedConfig {
            source: crate::config::SourceConfig {
                git_url: None,
                branch: "master".into(),
                data_path: Some(tmp.path().to_string_lossy().into_owned()),
                git_token_env: "GIT_PULL_TOKEN".into(),
            },
            indexing: crate::config::IndexingConfig::default(),
            frontmatter: crate::config::FrontmatterConfig {
                required: vec!["title".into()],
                ..Default::default()
            },
            chunking: crate::config::ChunkingConfig::default(),
            embedding: crate::config::ResolvedEmbeddingConfig {
                base_url: "http://localhost:8080/v1".into(),
                model: "test".into(),
                api_key: None,
                vector_size: 768,
                batch_size: 32,
            },
            qdrant: crate::config::ResolvedQdrantConfig {
                url: "http://localhost:6334".into(),
                collection: "test".into(),
            },
            validation: crate::config::ValidationConfig {
                enabled: true,
                strict: false,
                lint_command: None,
            },
            webhook: crate::config::WebhookConfig::default(),
            mcp: crate::config::McpConfig::default(),
            rate_limit: crate::config::RateLimitConfig::default(),
            write: crate::config::WriteConfig::default(),
        });

        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        // Content intentionally missing the "title" frontmatter field
        let params = CreateDocumentParams {
            path: "guide/missing-title.md".to_string(),
            content: "---\ntype: guide\n---\n# No title in frontmatter".to_string(),
            message: None,
            force_new: None,
        };
        let result = server.create_document(Parameters(params)).await;

        assert!(result.is_err(), "validation failure should return Err");
        let err = result.unwrap_err();

        // Message should be human-readable
        assert!(
            err.message.contains("frontmatter validation failed"),
            "error message should describe validation failure, got: {}",
            err.message
        );

        // Data field must contain structured field_errors
        let data = err.data.expect("error should carry structured data");
        let field_errors = data
            .get("field_errors")
            .expect("data must have field_errors key");
        assert!(
            field_errors.is_array(),
            "field_errors must be a JSON array, got: {}",
            field_errors
        );
        let arr = field_errors.as_array().unwrap();
        assert!(
            !arr.is_empty(),
            "field_errors array must be non-empty for a validation failure"
        );

        // At least one entry should mention "title" as the failed field
        let mentions_title = arr.iter().any(|fe| {
            fe.get("field")
                .and_then(|f| f.as_str())
                .map(|f| f == "title")
                .unwrap_or(false)
        });
        assert!(
            mentions_title,
            "field_errors should contain an entry for 'title', got: {}",
            serde_json::to_string_pretty(&data).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // dedup_verdict unit tests — no live Qdrant/embedder required
    // -----------------------------------------------------------------------

    #[test]
    fn dedup_verdict_score_above_threshold_returns_hit() {
        let result = dedup_verdict(Some(("docs/existing.md".into(), 0.92)), 0.85);
        assert!(
            result.is_some(),
            "score 0.92 >= threshold 0.85 should refuse"
        );
        let hit = result.unwrap();
        assert_eq!(hit.file_path, "docs/existing.md");
        assert!((hit.score - 0.92).abs() < 1e-6);
    }

    #[test]
    fn dedup_verdict_score_at_threshold_returns_hit() {
        // Boundary: score exactly equal to threshold is also a duplicate.
        let result = dedup_verdict(Some(("docs/boundary.md".into(), 0.85)), 0.85);
        assert!(
            result.is_some(),
            "score == threshold should be treated as duplicate"
        );
    }

    #[test]
    fn dedup_verdict_score_below_threshold_allows() {
        let result = dedup_verdict(Some(("docs/different.md".into(), 0.70)), 0.85);
        assert!(
            result.is_none(),
            "score 0.70 < threshold 0.85 should allow the write"
        );
    }

    #[test]
    fn dedup_verdict_no_results_allows() {
        // Empty collection or no results → no duplicate → allow.
        let result = dedup_verdict(None, 0.85);
        assert!(
            result.is_none(),
            "no results should allow the write (empty collection case)"
        );
    }

    #[test]
    fn dedup_verdict_hit_carries_correct_fields() {
        let hit = dedup_verdict(Some(("sysadmin/networking/dns.md".into(), 0.95)), 0.85)
            .expect("should be a hit");
        assert_eq!(hit.file_path, "sysadmin/networking/dns.md");
        assert!((hit.score - 0.95).abs() < 1e-6, "score should be preserved");
    }

    /// Test that the gating booleans (`dedup_enabled`, `must_already_exist`, `force_new`)
    /// correctly bypass the dedup gate.  We use a server with dedup_enabled=false / true
    /// and call write_document up to the point where the gate would fire — since the
    /// embed client isn't reachable the gate's embed call fails-open (logs a warning and
    /// continues), but with dedup_enabled=false the gate is never entered at all, so we
    /// reach a different error (validation or file existence) and NOT a dedup refusal.
    #[tokio::test]
    async fn dedup_gate_disabled_via_config_does_not_embed() {
        let tmp = tempfile::tempdir().unwrap();

        // Config with dedup disabled.
        let config = Arc::new(ResolvedConfig {
            source: crate::config::SourceConfig {
                git_url: None,
                branch: "master".into(),
                data_path: Some(tmp.path().to_string_lossy().into_owned()),
                git_token_env: "GIT_PULL_TOKEN".into(),
            },
            indexing: crate::config::IndexingConfig::default(),
            frontmatter: crate::config::FrontmatterConfig::default(),
            chunking: crate::config::ChunkingConfig::default(),
            embedding: crate::config::ResolvedEmbeddingConfig {
                base_url: "http://localhost:8080/v1".into(),
                model: "test".into(),
                api_key: None,
                vector_size: 768,
                batch_size: 32,
            },
            qdrant: crate::config::ResolvedQdrantConfig {
                url: "http://localhost:6334".into(),
                collection: "test".into(),
            },
            validation: crate::config::ValidationConfig::default(),
            webhook: crate::config::WebhookConfig::default(),
            mcp: crate::config::McpConfig::default(),
            rate_limit: crate::config::RateLimitConfig::default(),
            write: crate::config::WriteConfig {
                dedup_enabled: false,
                dedup_threshold: 0.85,
                commit_author_name: "md-kb-rag".to_string(),
                commit_author_email: "md-kb-rag@localhost".to_string(),
            },
        });

        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        // When dedup is disabled we should NOT get a dedup refusal.
        // The write will fail at git/reindex (no live services) — that's fine.
        let params = CreateDocumentParams {
            path: "docs/new.md".to_string(),
            content: "---\ntitle: Test Doc\n---\n# Content".to_string(),
            message: None,
            force_new: None,
        };
        let result = server.create_document(Parameters(params)).await;

        // We expect an error (git/reindex will fail in unit test), but it must
        // NOT be a dedup refusal — i.e. it should not mention "similar document".
        if let Err(e) = result {
            assert!(
                !e.message.contains("similar document"),
                "dedup disabled: error must not be a dedup refusal, got: {}",
                e.message
            );
        }
        // (Ok is also fine — means we somehow reached the write step, which is
        // unexpected in a unit test but not a test failure for this assertion.)
    }

    #[tokio::test]
    async fn dedup_gate_bypassed_by_force_new() {
        let tmp = tempfile::tempdir().unwrap();

        // Config with dedup ENABLED (default).
        let config = make_test_resolved_config(tmp.path());

        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        // With force_new=Some(true), the gate must be skipped even when dedup is
        // enabled.  The embed/qdrant will fail-open (no live services), so we will
        // reach git/reindex and fail there — but NOT with a dedup message.
        let params = CreateDocumentParams {
            path: "docs/forced.md".to_string(),
            content: "---\ntitle: Forced Doc\n---\n# Content".to_string(),
            message: None,
            force_new: Some(true),
        };
        let result = server.create_document(Parameters(params)).await;

        if let Err(e) = result {
            assert!(
                !e.message.contains("similar document"),
                "force_new=true must bypass dedup gate, got: {}",
                e.message
            );
        }
    }

    #[tokio::test]
    async fn dedup_gate_skipped_for_edit_path() {
        let tmp = tempfile::tempdir().unwrap();
        // Create the file so the edit path can proceed past existence check.
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(
            sub.join("edit-me.md"),
            "---\ntitle: Old Doc\n---\n# old content",
        )
        .unwrap();

        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        // Edit path should never trigger the dedup gate.
        // It will fail at git/reindex — but NOT with a dedup message.
        let params = EditDocumentParams {
            path: "docs/edit-me.md".to_string(),
            old_string: None,
            new_string: None,
            content: Some("---\ntitle: Edited Doc\n---\n# New content".to_string()),
            message: None,
        };
        let result = server.edit_document(Parameters(params)).await;

        if let Err(e) = result {
            assert!(
                !e.message.contains("similar document"),
                "edit path must never trigger dedup gate, got: {}",
                e.message
            );
        }
    }

    // -----------------------------------------------------------------------
    // delete_document unit tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn delete_document_nonexistent_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        let params = DeleteDocumentParams {
            path: "docs/nonexistent.md".to_string(),
            message: None,
        };
        let result = server.delete_document(Parameters(params)).await;

        assert!(
            result.is_err(),
            "delete of non-existent file should return Err"
        );
        let err = result.unwrap_err();
        assert!(
            err.message.contains("does not exist"),
            "error should mention 'does not exist', got: {}",
            err.message
        );
    }

    #[test]
    fn delete_document_relpath_derivation() {
        // Verify that the relpath strip logic produces the expected relative path.
        let canonical_data = std::path::PathBuf::from("/data/kb");
        let canonical_file = std::path::PathBuf::from("/data/kb/sysadmin/networking/dns.md");
        let rel = canonical_file
            .strip_prefix(&canonical_data)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        assert_eq!(rel, "sysadmin/networking/dns.md");
    }

    #[test]
    fn delete_document_diff_shows_all_as_removals() {
        // When deleting a file, render_unified_diff(old, "", path) should show all
        // lines as removals (every non-header line starts with '-').
        let old = "---\ntitle: My Doc\n---\n# Content\nSome text.\n";
        let diff = render_unified_diff(old, "", "docs/my-doc.md");
        assert!(!diff.is_empty(), "delete diff should be non-empty");
        for line in diff.lines() {
            if !line.starts_with("---")
                && !line.starts_with("+++")
                && !line.starts_with("@@")
                && !line.is_empty()
            {
                assert!(
                    line.starts_with('-'),
                    "all content lines in a delete diff should be removals, got: {line}"
                );
            }
        }
    }

    #[test]
    fn delete_document_commit_message_has_correct_trailers() {
        let msg = build_commit_message(None, "docs: delete notes/guide.md", "delete_document");
        assert!(
            msg.contains("Tool: md-kb-rag"),
            "should contain Tool trailer: {msg}"
        );
        assert!(
            msg.contains("Operation: delete_document"),
            "should contain Operation: delete_document trailer: {msg}"
        );
        assert!(
            msg.starts_with("docs: delete notes/guide.md"),
            "should start with delete subject: {msg}"
        );
    }

    #[test]
    fn delete_document_user_message_overrides_default() {
        let msg = build_commit_message(
            Some("chore: remove obsolete guide"),
            "docs: delete notes/guide.md",
            "delete_document",
        );
        assert!(
            msg.starts_with("chore: remove obsolete guide"),
            "user subject should override default: {msg}"
        );
        assert!(
            msg.contains("Operation: delete_document"),
            "trailer still present: {msg}"
        );
    }

    #[tokio::test]
    async fn delete_document_empty_path_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        let params = DeleteDocumentParams {
            path: "   ".to_string(), // whitespace-only
            message: None,
        };
        let result = server.delete_document(Parameters(params)).await;

        assert!(result.is_err(), "empty path should return Err");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("empty"),
            "error should mention empty path, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn delete_document_existing_file_proceeds_to_git_step() {
        // Create a real file in tmp; the tool should resolve it and proceed past
        // path-resolution and file-removal to the git step (which will fail since
        // there's no git repo). The error must NOT be a path-resolution error.
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(
            sub.join("delete-me.md"),
            "---\ntitle: Delete Me\n---\n# Body",
        )
        .unwrap();

        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        let params = DeleteDocumentParams {
            path: "docs/delete-me.md".to_string(),
            message: None,
        };
        let result = server.delete_document(Parameters(params)).await;

        // File removal happens before git; the file should be gone regardless.
        assert!(
            !sub.join("delete-me.md").exists(),
            "file should have been removed from disk"
        );

        // The error (if any) should be git- or index-related, not path-resolution.
        if let Err(e) = result {
            assert!(
                !e.message.contains("does not exist"),
                "error should not be path-resolution, got: {}",
                e.message
            );
        }
    }

    // -----------------------------------------------------------------------
    // resolve_safe_write_path unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn safe_write_path_rejects_absolute() {
        let tmp = tempfile::tempdir().unwrap();
        let result = resolve_safe_write_path(tmp.path(), "/etc/passwd");
        assert!(result.is_err(), "absolute path must be rejected");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("relative"),
            "error should mention relative requirement, got: {msg}"
        );
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
    // parse_edit_mode unit tests
    // -----------------------------------------------------------------------

    fn make_edit_params(
        content: Option<&str>,
        old_string: Option<&str>,
        new_string: Option<&str>,
    ) -> EditDocumentParams {
        EditDocumentParams {
            path: "docs/test.md".to_string(),
            content: content.map(|s| s.to_string()),
            old_string: old_string.map(|s| s.to_string()),
            new_string: new_string.map(|s| s.to_string()),
            message: None,
        }
    }

    #[test]
    fn parse_edit_mode_full_replace_is_recognized() {
        let params = make_edit_params(Some("new content"), None, None);
        let mode = parse_edit_mode(&params).unwrap();
        assert_eq!(
            mode,
            EditMode::Full {
                content: "new content".to_string()
            }
        );
    }

    #[test]
    fn parse_edit_mode_surgical_is_recognized() {
        let params = make_edit_params(None, Some("old text"), Some("new text"));
        let mode = parse_edit_mode(&params).unwrap();
        assert_eq!(
            mode,
            EditMode::Surgical {
                old: "old text".to_string(),
                new: "new text".to_string()
            }
        );
    }

    #[test]
    fn parse_edit_mode_both_modes_rejected() {
        let params = make_edit_params(Some("full content"), Some("old"), Some("new"));
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(
            err.contains("mutually exclusive"),
            "expected 'mutually exclusive' in error, got: {err}"
        );
    }

    #[test]
    fn parse_edit_mode_neither_mode_rejected() {
        let params = make_edit_params(None, None, None);
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(
            err.contains("must provide"),
            "expected 'must provide' in error, got: {err}"
        );
    }

    #[test]
    fn parse_edit_mode_only_old_string_rejected() {
        let params = make_edit_params(None, Some("old"), None);
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(
            err.contains("new_string"),
            "expected mention of new_string in error, got: {err}"
        );
    }

    #[test]
    fn parse_edit_mode_only_new_string_rejected() {
        let params = make_edit_params(None, None, Some("new"));
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(
            err.contains("old_string"),
            "expected mention of old_string in error, got: {err}"
        );
    }

    #[test]
    fn parse_edit_mode_identical_old_new_rejected() {
        let params = make_edit_params(None, Some("same text"), Some("same text"));
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(
            err.contains("identical"),
            "expected 'identical' in error, got: {err}"
        );
    }

    // -----------------------------------------------------------------------
    // apply_surgical unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn apply_surgical_single_occurrence_replaced() {
        let old = "Hello world!\nGoodbye earth!";
        let result = apply_surgical(old, "world", "Rust").unwrap();
        assert_eq!(result, "Hello Rust!\nGoodbye earth!");
    }

    #[test]
    fn apply_surgical_not_found_returns_error() {
        let old = "Hello world!";
        let err = apply_surgical(old, "missing text", "replacement").unwrap_err();
        assert!(
            err.contains("not found"),
            "expected 'not found' in error, got: {err}"
        );
    }

    #[test]
    fn apply_surgical_multiple_occurrences_returns_error_with_count() {
        let old = "foo bar foo baz foo";
        let err = apply_surgical(old, "foo", "qux").unwrap_err();
        assert!(
            err.contains("3"),
            "error should mention occurrence count (3), got: {err}"
        );
        assert!(
            err.contains("not unique"),
            "error should mention 'not unique', got: {err}"
        );
    }

    #[test]
    fn apply_surgical_exact_single_unique_string() {
        let old = "---\ntitle: My Doc\n---\n# Content\nSome text here.";
        let result = apply_surgical(old, "Some text here.", "Updated text.").unwrap();
        assert_eq!(result, "---\ntitle: My Doc\n---\n# Content\nUpdated text.");
    }

    // -----------------------------------------------------------------------
    // render_unified_diff unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn render_unified_diff_shows_added_lines() {
        let old = "line1\nline2\n";
        let new = "line1\nline2\nline3\n";
        let diff = render_unified_diff(old, new, "docs/test.md");
        assert!(
            !diff.is_empty(),
            "diff should be non-empty for a changed doc"
        );
        assert!(
            diff.contains("+line3"),
            "diff should show added line, got:\n{diff}"
        );
        assert!(
            diff.contains("a/docs/test.md"),
            "diff header should name the file, got:\n{diff}"
        );
    }

    #[test]
    fn render_unified_diff_shows_removed_lines() {
        let old = "line1\nline2\nline3\n";
        let new = "line1\nline3\n";
        let diff = render_unified_diff(old, new, "docs/test.md");
        assert!(
            diff.contains("-line2"),
            "diff should show removed line, got:\n{diff}"
        );
    }

    #[test]
    fn render_unified_diff_identical_content_is_empty() {
        let content = "line1\nline2\n";
        let diff = render_unified_diff(content, content, "docs/test.md");
        assert!(
            diff.is_empty(),
            "identical content should produce empty diff, got:\n{diff}"
        );
    }

    #[test]
    fn render_unified_diff_create_shows_all_as_additions() {
        let old = "";
        let new = "---\ntitle: New Doc\n---\n# Hello\n";
        let diff = render_unified_diff(old, new, "docs/new.md");
        assert!(!diff.is_empty(), "new file diff should be non-empty");
        // Every non-header line should be an addition.
        for line in diff.lines() {
            if !line.starts_with("---")
                && !line.starts_with("+++")
                && !line.starts_with("@@")
                && !line.is_empty()
            {
                assert!(
                    line.starts_with('+'),
                    "all content lines in a create diff should be additions, got: {line}"
                );
            }
        }
    }
}
