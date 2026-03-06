use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub source: SourceConfig,
    pub indexing: IndexingConfig,
    pub frontmatter: FrontmatterConfig,
    pub chunking: ChunkingConfig,
    pub embedding: EmbeddingConfig,
    pub qdrant: QdrantConfig,
    #[serde(default)]
    pub validation: ValidationConfig,
    #[serde(default)]
    pub webhook: WebhookConfig,
    #[serde(default)]
    pub mcp: McpConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SourceConfig {
    pub git_url: Option<String>,
    #[serde(default = "default_branch")]
    pub branch: String,
    /// Path to the knowledge base root (defaults to /data in Docker)
    #[serde(default = "default_data_path")]
    pub data_path: Option<String>,
}

fn default_branch() -> String {
    "master".into()
}

fn default_data_path() -> Option<String> {
    Some("/data".into())
}

#[derive(Debug, Clone, Deserialize)]
pub struct IndexingConfig {
    #[serde(default = "default_include")]
    pub include: Vec<String>,
    #[serde(default)]
    pub exclude: Vec<String>,
    #[serde(default)]
    pub exclude_files: Vec<String>,
}

fn default_include() -> Vec<String> {
    vec!["**/*.md".into()]
}

#[derive(Debug, Clone, Deserialize)]
pub struct FrontmatterConfig {
    #[serde(default)]
    pub required: Vec<String>,
    #[serde(default)]
    pub indexed_fields: Vec<String>,
    #[serde(default)]
    pub defaults: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChunkingConfig {
    #[serde(default = "default_strategy")]
    pub strategy: String,
    #[serde(default = "default_max_chunk_size")]
    pub max_chunk_size: usize,
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
    #[serde(default)]
    pub prepend_description: bool,
}

fn default_strategy() -> String {
    "markdown".into()
}

fn default_max_chunk_size() -> usize {
    1500
}

fn default_chunk_overlap() -> usize {
    200
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingConfig {
    pub base_url: String,
    pub model: String,
    #[serde(default = "default_vector_size")]
    pub vector_size: u64,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

fn default_vector_size() -> u64 {
    768
}

fn default_batch_size() -> usize {
    32
}

#[derive(Debug, Clone, Deserialize)]
pub struct QdrantConfig {
    pub url: String,
    #[serde(default = "default_collection")]
    pub collection: String,
}

fn default_collection() -> String {
    "knowledge-base".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ValidationConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub strict: bool,
    pub lint_command: Option<String>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strict: false,
            lint_command: None,
        }
    }
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct WebhookConfig {
    #[serde(default = "default_webhook_port")]
    pub port: u16,
    #[serde(default = "default_webhook_secret_env")]
    pub secret_env: String,
    #[serde(default = "default_provider")]
    pub provider: String,
}

impl Default for WebhookConfig {
    fn default() -> Self {
        Self {
            port: 9000,
            secret_env: "WEBHOOK_SECRET".into(),
            provider: "gitea".into(),
        }
    }
}

fn default_webhook_port() -> u16 {
    9000
}

fn default_webhook_secret_env() -> String {
    "WEBHOOK_SECRET".into()
}

fn default_provider() -> String {
    "gitea".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct McpConfig {
    #[serde(default = "default_mcp_port")]
    pub port: u16,
    #[serde(default = "default_bearer_token_env")]
    pub bearer_token_env: String,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            port: 8001,
            bearer_token_env: "MCP_BEARER_TOKEN".into(),
        }
    }
}

fn default_mcp_port() -> u16 {
    8001
}

fn default_bearer_token_env() -> String {
    "MCP_BEARER_TOKEN".into()
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_yaml_ng::from_str(&content)?;
        Ok(config)
    }

    /// Resolve the data path (source.data_path, or /data as default)
    pub fn data_path(&self) -> &str {
        self.source.data_path.as_deref().unwrap_or("/data")
    }

    /// Parse config from a YAML string (for testing)
    pub fn from_str(yaml: &str) -> anyhow::Result<Self> {
        Ok(serde_yaml_ng::from_str(yaml)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_CONFIG: &str = r#"
source:
  git_url: "https://example.com/repo.git"
indexing:
  include: ["**/*.md"]
frontmatter:
  required: [title, description]
chunking:
  max_chunk_size: 1000
embedding:
  base_url: "http://localhost:8080/v1"
  model: "test-model"
qdrant:
  url: "http://localhost:6334"
"#;

    #[test]
    fn parse_minimal_config() {
        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert_eq!(cfg.source.branch, "master");
        assert_eq!(cfg.embedding.vector_size, 768);
        assert_eq!(cfg.embedding.batch_size, 32);
        assert_eq!(cfg.qdrant.collection, "knowledge-base");
        assert_eq!(cfg.chunking.max_chunk_size, 1000);
        assert_eq!(cfg.chunking.chunk_overlap, 200);
        assert!(cfg.validation.enabled);
        assert!(!cfg.validation.strict);
        assert_eq!(cfg.mcp.port, 8001);
        assert_eq!(cfg.webhook.port, 9000);
    }

    #[test]
    fn parse_full_config() {
        let yaml = r#"
source:
  git_url: "https://example.com/repo.git"
  branch: "main"
  data_path: "/custom/path"
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
  strategy: "markdown"
  max_chunk_size: 2000
  chunk_overlap: 100
  prepend_description: true
embedding:
  base_url: "http://embed:8080/v1"
  model: "nomic"
  vector_size: 512
  batch_size: 16
qdrant:
  url: "http://qdrant:6334"
  collection: "my-kb"
validation:
  enabled: false
  strict: true
webhook:
  port: 9001
  secret_env: "MY_SECRET"
  provider: "github"
mcp:
  port: 9002
  bearer_token_env: "MY_TOKEN"
"#;
        let cfg = Config::from_str(yaml).unwrap();
        assert_eq!(cfg.source.branch, "main");
        assert_eq!(cfg.data_path(), "/custom/path");
        assert_eq!(cfg.embedding.vector_size, 512);
        assert_eq!(cfg.qdrant.collection, "my-kb");
        assert!(!cfg.validation.enabled);
        assert!(cfg.validation.strict);
        assert_eq!(cfg.webhook.provider, "github");
        assert_eq!(cfg.frontmatter.defaults.get("status").unwrap(), "draft");
    }

    #[test]
    fn default_data_path() {
        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert_eq!(cfg.data_path(), "/data");
    }
}
