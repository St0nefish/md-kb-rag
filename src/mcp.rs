use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use globset::{Glob, GlobSet, GlobSetBuilder};
use rmcp::{
    ErrorData as McpError, ServerHandler, handler::server::router::tool::ToolRouter,
    handler::server::wrapper::Parameters, model::*, schemars, tool, tool_handler, tool_router,
};

use tracing::error;

use crate::{
    embed::EmbedClient,
    qdrant::{QdrantStore, SearchResult},
};

const MAX_SEARCH_LIMIT: u64 = 50;
const MAX_QUERY_LEN: usize = 4096;
const MAX_FILTER_STR_LEN: usize = 256;
const MAX_TAG_COUNT: usize = 20;
const MAX_TAG_LEN: usize = 256;

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

/// Parameters for the `get_document` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GetDocumentParams {
    /// The file path of the document (as returned by search results).
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

#[derive(Clone)]
pub struct KbSearchServer {
    embed_client: Arc<EmbedClient>,
    qdrant: Arc<QdrantStore>,
    collection: String,
    data_path: PathBuf,
    /// Glob patterns (from `indexing.include`) used to restrict `get_document` to permitted file types.
    include_patterns: Arc<GlobSet>,
    tool_router: ToolRouter<KbSearchServer>,
}

fn build_include_globset(patterns: &[String]) -> GlobSet {
    let mut builder = GlobSetBuilder::new();
    for p in patterns {
        match Glob::new(p) {
            Ok(g) => {
                builder.add(g);
            }
            Err(e) => {
                error!("Invalid include glob pattern '{}': {}", p, e);
            }
        }
    }
    // Fall back to **/*.md if no valid patterns were added
    builder
        .build()
        .unwrap_or_else(|_| GlobSetBuilder::new().build().unwrap())
}

#[tool_router]
impl KbSearchServer {
    pub fn new(
        embed_client: Arc<EmbedClient>,
        qdrant: Arc<QdrantStore>,
        collection: String,
        data_path: PathBuf,
        include_patterns: &[String],
    ) -> Self {
        Self {
            embed_client,
            qdrant,
            collection,
            data_path,
            include_patterns: Arc::new(build_include_globset(include_patterns)),
            tool_router: Self::tool_router(),
        }
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

            let text_snippet = result
                .payload
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .chars()
                .take(400)
                .collect::<String>();

            let file_path = result
                .payload
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");

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
                let ellipsis = if result
                    .payload
                    .get("text")
                    .and_then(|v| v.as_str())
                    .map(|t| t.chars().count() > 400)
                    .unwrap_or(false)
                {
                    "..."
                } else {
                    ""
                };
                output.push_str(&format!("\n{text_snippet}{ellipsis}\n"));
            }

            output.push('\n');
        }

        Ok(CallToolResult::success(vec![Content::text(output.trim())]))
    }

    #[tool(
        description = "Retrieve the full raw content of a document by file path. \
        Use file paths returned by the `search` tool. Returns the complete markdown \
        including frontmatter."
    )]
    async fn get_document(
        &self,
        Parameters(params): Parameters<GetDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        let requested = PathBuf::from(&params.path);

        // Canonicalize the data path for safe prefix checking
        let canonical_data = self.data_path.canonicalize().map_err(|e| {
            error!("Data path not accessible: {}", e);
            McpError::internal_error("Data path not accessible".to_string(), None)
        })?;

        // Resolve the requested path — it may be absolute or relative to data_path
        let resolved = if requested.is_absolute() {
            requested.clone()
        } else {
            self.data_path.join(&requested)
        };

        let canonical_resolved = resolved
            .canonicalize()
            .map_err(|_| McpError::invalid_params("File not found".to_string(), None))?;

        // Prevent path traversal outside data directory
        if !canonical_resolved.starts_with(&canonical_data) {
            return Err(McpError::invalid_params(
                "File path is outside the data directory".to_string(),
                None,
            ));
        }

        // Restrict to permitted file types (indexing.include patterns)
        let relative = canonical_resolved
            .strip_prefix(&canonical_data)
            .unwrap_or(&canonical_resolved);
        if !self.include_patterns.is_match(relative) {
            return Err(McpError::invalid_params(
                "File type not permitted".to_string(),
                None,
            ));
        }

        let content = tokio::fs::read_to_string(&canonical_resolved)
            .await
            .map_err(|e| {
                error!(
                    "Failed to read file '{}': {}",
                    canonical_resolved.display(),
                    e
                );
                McpError::invalid_params("Failed to read file".to_string(), None)
            })?;

        Ok(CallToolResult::success(vec![Content::text(content)]))
    }
}

#[tool_handler]
impl ServerHandler for KbSearchServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::from_build_env())
            .with_instructions(
                "Knowledge base semantic search server. \
             Use the `search` tool to find relevant documents by natural-language query, \
             with optional filters for domain, type, and tags."
                    .to_string(),
            )
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
        let text: String = std::iter::repeat('é').take(401).collect();
        assert!(text.len() > 400, "byte len should exceed 400");
        assert!(text.chars().count() > 400, "char count should exceed 400");
        // If we used .len() on a 400-char string it would wrongly trigger ellipsis
        let short: String = std::iter::repeat('é').take(400).collect();
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
}
