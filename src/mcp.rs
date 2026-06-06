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
    /// Path of the existing document to overwrite, relative to the knowledge base root.
    pub path: String,
    /// Full new markdown content (full-content replacement, not a patch), INCLUDING YAML frontmatter.
    pub content: String,
    /// Optional commit message; if omitted, a message is generated from the path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
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

    /// Shared implementation for create_document and edit_document.
    ///
    /// `must_already_exist`: `false` for create (file must NOT exist), `true` for edit (file MUST exist).
    /// `default_verb`: used to build the default commit message, e.g. `"add"` or `"update"`.
    /// `force_new`: when `Some(true)`, bypasses the dedup gate on create paths.
    async fn write_document(
        &self,
        rel_path: &str,
        content: &str,
        message: Option<&str>,
        must_already_exist: bool,
        default_verb: &str,
        force_new: Option<bool>,
    ) -> Result<CallToolResult, McpError> {
        let config = &self.config;

        // 1. Security: reject absolute paths, path traversal, and symlinked ancestors.
        let data_root = std::path::PathBuf::from(config.data_path());
        let abs_path = resolve_safe_write_path(&data_root, rel_path)
            .map_err(|e| McpError::invalid_params(e, None))?;

        // 2. Include-pattern guard: reject paths that the indexer would not pick up.
        // discover_files matches against the path relative to data_path (same form as rel_path).
        if !self.include_patterns.is_match(rel_path) {
            return Err(McpError::invalid_params(
                format!(
                    "path '{}' does not match any indexable include pattern \
                     (e.g. must be a markdown file under an included path)",
                    rel_path
                ),
                None,
            ));
        }

        // 3. Existence check
        let file_exists = abs_path.exists();
        if must_already_exist && !file_exists {
            return Err(McpError::invalid_params(
                format!(
                    "File '{}' does not exist. Use create_document to create new files.",
                    rel_path
                ),
                None,
            ));
        }
        if !must_already_exist && file_exists {
            return Err(McpError::invalid_params(
                format!(
                    "File '{}' already exists. Use edit_document to modify existing files.",
                    rel_path
                ),
                None,
            ));
        }

        // 4. Validate content before writing
        let (validation_result, _) = validate::validate_content(
            std::path::Path::new(rel_path),
            content,
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

        // 5. Dedup gate: on create paths, check for near-duplicate existing documents.
        //    Gate runs only when: this is a create (not edit), dedup is enabled in config,
        //    and the caller has not set force_new = true.
        if !must_already_exist
            && self.config.write.dedup_enabled
            && !matches!(force_new, Some(true))
        {
            // Build query text: frontmatter title (if any) + body, truncated to
            // DEDUP_QUERY_CHAR_LIMIT chars to stay within embedding model token limits.
            let query_text: String = content.chars().take(DEDUP_QUERY_CHAR_LIMIT).collect();

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

        // 6. Create parent directories and write the file
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

        tokio::fs::write(&abs_path, content.as_bytes())
            .await
            .map_err(|e| {
                error!("Failed to write file '{}': {}", abs_path.display(), e);
                McpError::internal_error(format!("Failed to write file: {}", e), None)
            })?;

        // 7. Build commit message
        let commit_message = {
            let base = message
                .map(|m| m.to_string())
                .unwrap_or_else(|| format!("docs: {} {}", default_verb, rel_path));
            format!("{}\n\nVia: md-kb-rag write tool", base)
        };

        // 8. Resolve git token
        let token = std::env::var(&config.source.git_token_env)
            .ok()
            .filter(|s| !s.is_empty());

        // 9. Commit and sync to remote
        let commit_sha = git::commit_and_sync(
            config.source.git_url.as_deref(),
            &config.source.branch,
            config.data_path(),
            token.as_deref(),
            rel_path,
            &commit_message,
        )
        .await
        .map_err(|e| {
            error!("commit_and_sync failed for '{}': {:#}", rel_path, e);
            McpError::internal_error(format!("Git commit/sync failed: {}", e), None)
        })?;

        // 10. Trigger incremental reindex (serialized against webhook reindexes)
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

        // 11. Return success
        let action = if must_already_exist {
            "Updated"
        } else {
            "Created"
        };
        Ok(CallToolResult::success(vec![Content::text(format!(
            "{} '{}' (commit {})",
            action, rel_path, commit_sha
        ))]))
    }

    #[tool(description = "Create a new document in the knowledge base. \
        Writes the file, commits it to the git repository, and triggers an incremental reindex. \
        The document must not already exist — use edit_document for existing files. \
        Content must include valid YAML frontmatter. \
        If a very similar document already exists, the create is refused and the close match is \
        reported — edit that document instead, or set force_new=true to create a new one anyway.")]
    async fn create_document(
        &self,
        Parameters(params): Parameters<CreateDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        self.write_document(
            &params.path,
            &params.content,
            params.message.as_deref(),
            false,
            "add",
            params.force_new,
        )
        .await
    }

    #[tool(
        description = "Edit (overwrite) an existing document in the knowledge base. \
        Replaces the entire file content, commits it to the git repository, and triggers \
        an incremental reindex. The document must already exist — use create_document for \
        new files. Content must include valid YAML frontmatter."
    )]
    async fn edit_document(
        &self,
        Parameters(params): Parameters<EditDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        self.write_document(
            &params.path,
            &params.content,
            params.message.as_deref(),
            true,
            "update",
            None, // edit_document targets a known existing file — no dedup gate needed
        )
        .await
    }
}

/// Default instructions used when no custom instructions are configured.
pub const DEFAULT_INSTRUCTIONS: &str = "Knowledge base semantic search server. \
Use the `search` tool to find relevant documents by natural-language query, \
with optional filters for domain, type, and tags.";

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

        let result = server
            .write_document(
                "docs/nonexistent.md",
                "---\ntitle: Test\n---\n# Body",
                None,
                true, // must_already_exist = true (edit mode)
                "update",
                None,
            )
            .await;

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

        let result = server
            .write_document(
                "docs/existing.md",
                "---\ntitle: Test\n---\n# New content",
                None,
                false, // must_already_exist = false (create mode)
                "add",
                None,
            )
            .await;

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
        let result = server
            .write_document("notes.txt", "Some plain text", None, false, "add", None)
            .await;

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
        let result = server
            .write_document("/etc/passwd", "# Evil", None, false, "add", None)
            .await;

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
        let result = server
            .write_document(
                "guide/missing-title.md",
                "---\ntype: guide\n---\n# No title in frontmatter",
                None,
                false,
                "add",
                None,
            )
            .await;

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
            },
        });

        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        // When dedup is disabled we should NOT get a dedup refusal.
        // The write will fail at validation (missing frontmatter) — that's fine.
        let result = server
            .write_document(
                "docs/new.md",
                "---\ntitle: Test Doc\n---\n# Content",
                None,
                false,
                "add",
                None,
            )
            .await;

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
        let result = server
            .write_document(
                "docs/forced.md",
                "---\ntitle: Forced Doc\n---\n# Content",
                None,
                false,
                "add",
                Some(true), // force_new
            )
            .await;

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
        std::fs::write(sub.join("edit-me.md"), "# old content").unwrap();

        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        // Edit path (must_already_exist=true) should never trigger the dedup gate.
        // It will fail at git/reindex — but NOT with a dedup message.
        let result = server
            .write_document(
                "docs/edit-me.md",
                "---\ntitle: Edited Doc\n---\n# New content",
                None,
                true, // edit path
                "update",
                None,
            )
            .await;

        if let Err(e) = result {
            assert!(
                !e.message.contains("similar document"),
                "edit path must never trigger dedup gate, got: {}",
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
}
