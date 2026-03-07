use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::router::tool::ToolRouter,
    handler::server::wrapper::Parameters,
    model::*,
    schemars, tool, tool_handler, tool_router,
};

use tracing::error;

use crate::{
    embed::EmbedClient,
    qdrant::{QdrantStore, SearchResult},
};

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

    /// Maximum number of results to return (default: 10).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,
}

#[derive(Clone)]
pub struct KbSearchServer {
    embed_client: Arc<EmbedClient>,
    qdrant: Arc<QdrantStore>,
    collection: String,
    data_path: PathBuf,
    tool_router: ToolRouter<KbSearchServer>,
}

#[tool_router]
impl KbSearchServer {
    pub fn new(
        embed_client: Arc<EmbedClient>,
        qdrant: Arc<QdrantStore>,
        collection: String,
        data_path: PathBuf,
    ) -> Self {
        Self {
            embed_client,
            qdrant,
            collection,
            data_path,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Search the knowledge base using a natural-language query. \
        Returns ranked document chunks with title, relevance score, text snippet, and metadata. \
        Optionally filter by domain, type, or tags.")]
    async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
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
            let tag_values: Vec<serde_json::Value> = tags
                .into_iter()
                .map(serde_json::Value::String)
                .collect();
            filters.insert("tags".to_string(), serde_json::Value::Array(tag_values));
        }

        let limit = params.limit.unwrap_or(10);

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
                    .map(|t| t.len() > 400)
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

    #[tool(description = "Retrieve the full raw content of a document by file path. \
        Use file paths returned by the `search` tool. Returns the complete markdown \
        including frontmatter.")]
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

        let canonical_resolved = resolved.canonicalize().map_err(|_| {
            McpError::invalid_params("File not found".to_string(), None)
        })?;

        // Prevent path traversal outside data directory
        if !canonical_resolved.starts_with(&canonical_data) {
            return Err(McpError::invalid_params(
                "File path is outside the data directory".to_string(),
                None,
            ));
        }

        let content = tokio::fs::read_to_string(&canonical_resolved)
            .await
            .map_err(|e| {
                error!("Failed to read file '{}': {}", canonical_resolved.display(), e);
                McpError::invalid_params("Failed to read file".to_string(), None)
            })?;

        Ok(CallToolResult::success(vec![Content::text(content)]))
    }
}

#[tool_handler]
impl ServerHandler for KbSearchServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .build(),
        )
        .with_server_info(Implementation::from_build_env())
        .with_instructions(
            "Knowledge base semantic search server. \
             Use the `search` tool to find relevant documents by natural-language query, \
             with optional filters for domain, type, and tags."
                .to_string(),
        )
    }
}
