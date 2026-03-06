use std::collections::HashMap;
use std::sync::Arc;

use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::router::tool::ToolRouter,
    handler::server::wrapper::Parameters,
    model::*,
    schemars, tool, tool_handler, tool_router,
};

use crate::{
    embed::EmbedClient,
    qdrant::{QdrantStore, SearchResult},
};

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
    tool_router: ToolRouter<KbSearchServer>,
}

#[tool_router]
impl KbSearchServer {
    pub fn new(
        embed_client: Arc<EmbedClient>,
        qdrant: Arc<QdrantStore>,
        collection: String,
    ) -> Self {
        Self {
            embed_client,
            qdrant,
            collection,
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
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

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
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

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

            output.push_str(&format!(
                "## Result {rank}\n\
                **Title**: {title}\n\
                **Score**: {score:.4}\n\
                **File**: {file_path}\n",
                rank = i + 1,
                title = title,
                score = result.score,
                file_path = file_path,
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
