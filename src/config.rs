use anyhow::Context;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use tracing::warn;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub source: SourceConfig,
    #[serde(default)]
    pub indexing: IndexingConfig,
    #[serde(default)]
    pub frontmatter: FrontmatterConfig,
    #[serde(default)]
    pub chunking: ChunkingConfig,
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub qdrant: QdrantConfig,
    #[serde(default)]
    pub validation: ValidationConfig,
    #[serde(default)]
    pub webhook: WebhookConfig,
    #[serde(default)]
    pub mcp: McpConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceConfig {
    pub git_url: Option<String>,
    #[serde(default = "default_branch")]
    pub branch: String,
    /// Path to the knowledge base root (defaults to /data in Docker)
    #[serde(default = "default_data_path")]
    pub data_path: Option<String>,
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self {
            git_url: None,
            branch: default_branch(),
            data_path: default_data_path(),
        }
    }
}

fn default_branch() -> String {
    "master".into()
}

fn default_data_path() -> Option<String> {
    Some("/data".into())
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexingConfig {
    #[serde(default = "default_include")]
    pub include: Vec<String>,
    #[serde(default = "default_exclude")]
    pub exclude: Vec<String>,
    #[serde(default = "default_exclude_files")]
    pub exclude_files: Vec<String>,
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            include: default_include(),
            exclude: default_exclude(),
            exclude_files: default_exclude_files(),
        }
    }
}

fn default_include() -> Vec<String> {
    vec!["**/*.md".into()]
}

fn default_exclude() -> Vec<String> {
    vec![
        ".git/**".into(),
        ".claude/**".into(),
        ".tools/**".into(),
        "node_modules/**".into(),
    ]
}

fn default_exclude_files() -> Vec<String> {
    vec!["CLAUDE.md".into(), "README.md".into()]
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct FrontmatterConfig {
    #[serde(default)]
    pub required: Vec<String>,
    #[serde(default)]
    pub indexed_fields: Vec<String>,
    #[serde(default)]
    pub defaults: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkingConfig {
    #[serde(default = "default_max_chunk_size")]
    pub max_chunk_size: usize,
    /// Target chunk size — accumulate markdown sections up to this size.
    /// Defaults to max_chunk_size (i.e. fill chunks as much as possible).
    #[serde(default = "default_target_chunk_size")]
    pub target_chunk_size: Option<usize>,
    #[serde(default = "default_true")]
    pub prepend_description: bool,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: default_max_chunk_size(),
            target_chunk_size: default_target_chunk_size(),
            prepend_description: true,
        }
    }
}

impl ChunkingConfig {
    pub fn target(&self) -> usize {
        self.target_chunk_size.unwrap_or(self.max_chunk_size)
    }
}

fn default_max_chunk_size() -> usize {
    1500
}

fn default_target_chunk_size() -> Option<usize> {
    Some(1000)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingConfig {
    pub base_url: Option<String>,
    pub model: Option<String>,
    #[serde(default = "default_vector_size")]
    pub vector_size: u64,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            base_url: None,
            model: None,
            vector_size: default_vector_size(),
            batch_size: default_batch_size(),
        }
    }
}

fn default_vector_size() -> u64 {
    768
}

fn default_batch_size() -> usize {
    32
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QdrantConfig {
    pub url: Option<String>,
    #[serde(default = "default_collection")]
    pub collection: String,
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: None,
            collection: default_collection(),
        }
    }
}

fn default_collection() -> String {
    "knowledge-base".into()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub struct WebhookConfig {
    #[serde(default = "default_webhook_secret_env")]
    pub secret_env: String,
    #[serde(default = "default_provider")]
    pub provider: String,
}

impl Default for WebhookConfig {
    fn default() -> Self {
        Self {
            secret_env: "WEBHOOK_SECRET".into(),
            provider: "gitea".into(),
        }
    }
}

fn default_webhook_secret_env() -> String {
    "WEBHOOK_SECRET".into()
}

fn default_provider() -> String {
    "gitea".into()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
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
        let config = if path.exists() {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read config file '{}'", path.display()))?;
            serde_yaml_ng::from_str(&content)?
        } else {
            warn!("Config file '{}' not found, using defaults", path.display());
            Config::default()
        };
        config.resolve()
    }

    /// Apply env var overrides and validate required fields.
    fn resolve(mut self) -> anyhow::Result<Self> {
        // Env var overrides
        if let Ok(val) = std::env::var("EMBEDDING_BASE_URL") {
            self.embedding.base_url = Some(val);
        }
        if let Ok(val) = std::env::var("EMBEDDING_MODEL") {
            self.embedding.model = Some(val);
        }
        if let Ok(val) = std::env::var("EMBEDDING_VECTOR_SIZE") {
            self.embedding.vector_size = val
                .parse()
                .map_err(|_| anyhow::anyhow!("EMBEDDING_VECTOR_SIZE must be a valid integer"))?;
        }
        if let Ok(val) = std::env::var("QDRANT_URL") {
            self.qdrant.url = Some(val);
        }

        // Validate chunk size config
        if let Some(target) = self.chunking.target_chunk_size
            && target > self.chunking.max_chunk_size
        {
            anyhow::bail!(
                "chunking.target_chunk_size ({}) must be <= chunking.max_chunk_size ({})",
                target,
                self.chunking.max_chunk_size
            );
        }

        // Validate required fields
        let mut missing = Vec::new();
        if self.embedding.base_url.is_none() {
            missing
                .push("embedding.base_url (set EMBEDDING_BASE_URL or config embedding.base_url)");
        }
        if self.embedding.model.is_none() {
            missing.push("embedding.model (set EMBEDDING_MODEL or config embedding.model)");
        }
        if self.qdrant.url.is_none() {
            missing.push("qdrant.url (set QDRANT_URL or config qdrant.url)");
        }
        if !missing.is_empty() {
            anyhow::bail!(
                "Missing required configuration:\n  - {}",
                missing.join("\n  - ")
            );
        }

        Ok(self)
    }

    /// Resolve the data path (source.data_path, or /data as default)
    pub fn data_path(&self) -> &str {
        self.source.data_path.as_deref().unwrap_or("/data")
    }

    /// Derive the state DB path from data_path: `{data_path}/state.db`
    pub fn state_db_path(&self) -> String {
        format!("{}/state.db", self.data_path())
    }
}

impl EmbeddingConfig {
    /// Returns the base_url, panics if not resolved.
    pub fn base_url(&self) -> &str {
        self.base_url
            .as_deref()
            .expect("embedding.base_url must be set (call Config::resolve first)")
    }

    /// Returns the model, panics if not resolved.
    pub fn model(&self) -> &str {
        self.model
            .as_deref()
            .expect("embedding.model must be set (call Config::resolve first)")
    }
}

impl QdrantConfig {
    /// Returns the url, panics if not resolved.
    pub fn url(&self) -> &str {
        self.url
            .as_deref()
            .expect("qdrant.url must be set (call Config::resolve first)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Mutex to serialize tests that modify environment variables.
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    impl Config {
        /// Deserialize + resolve (requires env vars or config values for required fields)
        fn from_str(yaml: &str) -> anyhow::Result<Self> {
            let config: Config = serde_yaml_ng::from_str(yaml)?;
            config.resolve()
        }

        /// Deserialize only — no env var resolution or validation
        fn from_str_raw(yaml: &str) -> anyhow::Result<Self> {
            Ok(serde_yaml_ng::from_str(yaml)?)
        }
    }

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
        let cfg = Config::from_str_raw(MINIMAL_CONFIG).unwrap();
        assert_eq!(cfg.source.branch, "master");
        assert_eq!(cfg.embedding.vector_size, 768);
        assert_eq!(cfg.embedding.batch_size, 32);
        assert_eq!(cfg.qdrant.collection, "knowledge-base");
        assert_eq!(cfg.chunking.max_chunk_size, 1000);
        assert!(cfg.validation.enabled);
        assert!(!cfg.validation.strict);
        assert_eq!(cfg.mcp.port, 8001);
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
  max_chunk_size: 2000
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
  secret_env: "MY_SECRET"
  provider: "github"
mcp:
  port: 9002
  bearer_token_env: "MY_TOKEN"
"#;
        let cfg = Config::from_str_raw(yaml).unwrap();
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
        let cfg = Config::from_str_raw(MINIMAL_CONFIG).unwrap();
        assert_eq!(cfg.data_path(), "/data");
    }

    #[test]
    fn empty_yaml_deserializes_to_defaults() {
        let cfg = Config::from_str_raw("{}").unwrap();
        assert_eq!(cfg.source.branch, "master");
        assert_eq!(cfg.source.data_path.as_deref(), Some("/data"));
        assert_eq!(cfg.source.git_url, None);
        assert_eq!(cfg.indexing.include, vec!["**/*.md"]);
        assert_eq!(
            cfg.indexing.exclude,
            vec![".git/**", ".claude/**", ".tools/**", "node_modules/**"]
        );
        assert_eq!(cfg.indexing.exclude_files, vec!["CLAUDE.md", "README.md"]);
        assert!(cfg.frontmatter.required.is_empty());
        assert_eq!(cfg.chunking.max_chunk_size, 1500);
        assert_eq!(cfg.chunking.target_chunk_size, Some(1000));
        assert!(cfg.chunking.prepend_description);
        assert_eq!(cfg.embedding.vector_size, 768);
        assert_eq!(cfg.embedding.batch_size, 32);
        assert_eq!(cfg.embedding.base_url, None);
        assert_eq!(cfg.embedding.model, None);
        assert_eq!(cfg.qdrant.url, None);
        assert_eq!(cfg.qdrant.collection, "knowledge-base");
        assert!(cfg.validation.enabled);
        assert_eq!(cfg.mcp.port, 8001);
    }

    #[test]
    fn env_vars_override_config_values() {
        let _lock = ENV_MUTEX.lock().unwrap();

        // SAFETY: serialized by ENV_MUTEX
        unsafe {
            std::env::set_var("EMBEDDING_BASE_URL", "http://env-embed:9090/v1");
            std::env::set_var("EMBEDDING_MODEL", "env-model");
            std::env::set_var("QDRANT_URL", "http://env-qdrant:6334");
        }

        let cfg = Config::from_str_raw(MINIMAL_CONFIG)
            .unwrap()
            .resolve()
            .unwrap();

        assert_eq!(
            cfg.embedding.base_url.as_deref(),
            Some("http://env-embed:9090/v1")
        );
        assert_eq!(cfg.embedding.model.as_deref(), Some("env-model"));
        assert_eq!(cfg.qdrant.url.as_deref(), Some("http://env-qdrant:6334"));

        unsafe {
            std::env::remove_var("EMBEDDING_BASE_URL");
            std::env::remove_var("EMBEDDING_MODEL");
            std::env::remove_var("QDRANT_URL");
        }
    }

    #[test]
    fn missing_required_fields_produces_clear_error() {
        let _lock = ENV_MUTEX.lock().unwrap();

        // SAFETY: serialized by ENV_MUTEX
        unsafe {
            std::env::remove_var("EMBEDDING_BASE_URL");
            std::env::remove_var("EMBEDDING_MODEL");
            std::env::remove_var("QDRANT_URL");
        }

        let result = Config::from_str_raw("{}").unwrap().resolve();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("embedding.base_url"),
            "error should mention embedding.base_url: {err}"
        );
        assert!(
            err.contains("embedding.model"),
            "error should mention embedding.model: {err}"
        );
        assert!(
            err.contains("qdrant.url"),
            "error should mention qdrant.url: {err}"
        );
    }

    #[test]
    fn load_missing_file_returns_defaults() {
        let _lock = ENV_MUTEX.lock().unwrap();

        // Provide required env vars so resolve() succeeds
        unsafe {
            std::env::set_var("EMBEDDING_BASE_URL", "http://test:8080/v1");
            std::env::set_var("EMBEDDING_MODEL", "test-model");
            std::env::set_var("QDRANT_URL", "http://test:6334");
        }

        let cfg = Config::load(Path::new("/nonexistent/config.yaml")).unwrap();
        assert_eq!(cfg.source.branch, "master");
        assert_eq!(cfg.chunking.max_chunk_size, 1500);
        assert_eq!(cfg.qdrant.collection, "knowledge-base");

        unsafe {
            std::env::remove_var("EMBEDDING_BASE_URL");
            std::env::remove_var("EMBEDDING_MODEL");
            std::env::remove_var("QDRANT_URL");
        }
    }

    #[test]
    fn env_var_vector_size_override() {
        let _lock = ENV_MUTEX.lock().unwrap();

        unsafe {
            std::env::set_var("EMBEDDING_BASE_URL", "http://test:8080/v1");
            std::env::set_var("EMBEDDING_MODEL", "test-model");
            std::env::set_var("EMBEDDING_VECTOR_SIZE", "1024");
            std::env::set_var("QDRANT_URL", "http://test:6334");
        }

        let cfg = Config::from_str_raw("{}").unwrap().resolve().unwrap();
        assert_eq!(cfg.embedding.vector_size, 1024);

        unsafe {
            std::env::remove_var("EMBEDDING_BASE_URL");
            std::env::remove_var("EMBEDDING_MODEL");
            std::env::remove_var("EMBEDDING_VECTOR_SIZE");
            std::env::remove_var("QDRANT_URL");
        }
    }

    #[test]
    fn env_var_vector_size_invalid() {
        let _lock = ENV_MUTEX.lock().unwrap();

        unsafe {
            std::env::set_var("EMBEDDING_BASE_URL", "http://test:8080/v1");
            std::env::set_var("EMBEDDING_MODEL", "test-model");
            std::env::set_var("EMBEDDING_VECTOR_SIZE", "not-a-number");
            std::env::set_var("QDRANT_URL", "http://test:6334");
        }

        let result = Config::from_str_raw("{}").unwrap().resolve();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("EMBEDDING_VECTOR_SIZE"),
            "error should mention EMBEDDING_VECTOR_SIZE"
        );

        unsafe {
            std::env::remove_var("EMBEDDING_BASE_URL");
            std::env::remove_var("EMBEDDING_MODEL");
            std::env::remove_var("EMBEDDING_VECTOR_SIZE");
            std::env::remove_var("QDRANT_URL");
        }
    }

    #[test]
    fn partial_config_with_env_filling_gaps() {
        let _lock = ENV_MUTEX.lock().unwrap();

        // Config provides qdrant.url, env provides embedding fields
        unsafe {
            std::env::set_var("EMBEDDING_BASE_URL", "http://env:8080/v1");
            std::env::set_var("EMBEDDING_MODEL", "env-model");
            std::env::remove_var("QDRANT_URL");
        }

        let yaml = r#"
qdrant:
  url: "http://config-qdrant:6334"
chunking:
  max_chunk_size: 2000
"#;
        let cfg = Config::from_str_raw(yaml).unwrap().resolve().unwrap();
        assert_eq!(
            cfg.embedding.base_url.as_deref(),
            Some("http://env:8080/v1")
        );
        assert_eq!(cfg.embedding.model.as_deref(), Some("env-model"));
        assert_eq!(cfg.qdrant.url.as_deref(), Some("http://config-qdrant:6334"));
        assert_eq!(cfg.chunking.max_chunk_size, 2000);
        // Other fields should still be defaults
        assert_eq!(cfg.source.branch, "master");
        assert_eq!(cfg.qdrant.collection, "knowledge-base");

        unsafe {
            std::env::remove_var("EMBEDDING_BASE_URL");
            std::env::remove_var("EMBEDDING_MODEL");
        }
    }

    #[test]
    fn accessor_methods_work_after_resolve() {
        let _lock = ENV_MUTEX.lock().unwrap();

        unsafe {
            std::env::remove_var("EMBEDDING_BASE_URL");
            std::env::remove_var("EMBEDDING_MODEL");
            std::env::remove_var("QDRANT_URL");
        }

        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert_eq!(cfg.embedding.base_url(), "http://localhost:8080/v1");
        assert_eq!(cfg.embedding.model(), "test-model");
        assert_eq!(cfg.qdrant.url(), "http://localhost:6334");
    }

    #[test]
    #[should_panic(expected = "embedding.base_url must be set")]
    fn accessor_panics_without_resolve() {
        let cfg = Config::from_str_raw("{}").unwrap();
        let _ = cfg.embedding.base_url();
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let yaml = r#"
source:
  branch: "main"
unknown_top_level: true
embedding:
  base_url: "http://localhost:8080/v1"
  model: "test"
qdrant:
  url: "http://localhost:6334"
"#;
        let result: Result<Config, _> = serde_yaml_ng::from_str(yaml);
        assert!(
            result.is_err(),
            "top-level unknown field should be rejected"
        );
    }

    #[test]
    fn unknown_fields_in_nested_struct_are_rejected() {
        let yaml = r#"
source:
  branch: "main"
  unknown_nested: "oops"
"#;
        let result: Result<Config, _> = serde_yaml_ng::from_str(yaml);
        assert!(result.is_err(), "nested unknown field should be rejected");
    }

    #[test]
    fn state_db_path_derived_from_data_path() {
        let yaml = r#"
source:
  data_path: "/custom/path"
"#;
        let cfg = Config::from_str_raw(yaml).unwrap();
        assert_eq!(cfg.state_db_path(), "/custom/path/state.db");
    }

    #[test]
    fn state_db_path_uses_default_data_path() {
        let cfg = Config::from_str_raw("{}").unwrap();
        assert_eq!(cfg.state_db_path(), "/data/state.db");
    }

    #[test]
    fn example_config_deserializes() {
        let yaml = include_str!("../config.example.yaml");
        let cfg: Config = serde_yaml_ng::from_str(yaml).expect("config.example.yaml should parse");
        // Spot-check a few values to catch drift between example and struct
        assert_eq!(cfg.source.branch, "master");
        assert_eq!(cfg.chunking.max_chunk_size, 1500);
        assert_eq!(cfg.chunking.target_chunk_size, Some(1000));
        assert!(cfg.chunking.prepend_description);
        assert_eq!(cfg.embedding.vector_size, 768);
        assert_eq!(cfg.embedding.batch_size, 32);
        assert_eq!(cfg.qdrant.collection, "knowledge-base");
        assert!(cfg.validation.enabled);
        assert!(!cfg.validation.strict);
        assert_eq!(cfg.webhook.provider, "gitea");
        assert_eq!(cfg.mcp.port, 8001);
    }

    #[test]
    fn config_file_values_preserved_when_env_absent() {
        let _lock = ENV_MUTEX.lock().unwrap();

        unsafe {
            std::env::remove_var("EMBEDDING_BASE_URL");
            std::env::remove_var("EMBEDDING_MODEL");
            std::env::remove_var("EMBEDDING_VECTOR_SIZE");
            std::env::remove_var("QDRANT_URL");
        }

        let cfg = Config::from_str(MINIMAL_CONFIG).unwrap();
        assert_eq!(cfg.embedding.base_url(), "http://localhost:8080/v1");
        assert_eq!(cfg.embedding.model(), "test-model");
        assert_eq!(cfg.qdrant.url(), "http://localhost:6334");
        assert_eq!(cfg.embedding.vector_size, 768);
    }

    #[test]
    fn target_exceeds_max_is_rejected() {
        let _lock = ENV_MUTEX.lock().unwrap();

        unsafe {
            std::env::set_var("EMBEDDING_BASE_URL", "http://test:8080/v1");
            std::env::set_var("EMBEDDING_MODEL", "test-model");
            std::env::set_var("QDRANT_URL", "http://test:6334");
        }

        let yaml = r#"
chunking:
  target_chunk_size: 1500
  max_chunk_size: 1000
"#;
        let result = Config::from_str_raw(yaml).unwrap().resolve();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("target_chunk_size"),
            "error should mention target_chunk_size: {err}"
        );
        assert!(
            err.contains("max_chunk_size"),
            "error should mention max_chunk_size: {err}"
        );

        unsafe {
            std::env::remove_var("EMBEDDING_BASE_URL");
            std::env::remove_var("EMBEDDING_MODEL");
            std::env::remove_var("QDRANT_URL");
        }
    }
}
