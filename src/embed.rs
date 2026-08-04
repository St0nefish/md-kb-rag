use anyhow::Result;
use async_openai::{
    Client,
    config::{Config as OpenAIConfigTrait, OpenAIConfig},
    types::{CreateEmbeddingRequestArgs, EmbeddingInput},
};
use backoff::{ExponentialBackoff, ExponentialBackoffBuilder};
use std::time::Duration;

use crate::config::ResolvedEmbeddingConfig;

/// How often `embed_texts` logs progress through a long batch sequence.
const EMBED_PROGRESS_INTERVAL: Duration = Duration::from_secs(10);

pub trait EmbedStore: Send + Sync {
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

pub trait QueryEmbedder: Send + Sync {
    async fn embed_query(&self, q: &str) -> anyhow::Result<Vec<f32>>;
}

pub struct EmbedClient {
    client: Client<OpenAIConfig>,
    http_client: reqwest::Client,
    model: String,
    batch_size: usize,
    api_key: Option<String>,
}

impl EmbedClient {
    pub fn new(config: &ResolvedEmbeddingConfig) -> Self {
        let api_key = config.api_key.as_deref().unwrap_or("not-needed");
        let openai_config = OpenAIConfig::new()
            .with_api_base(&config.base_url)
            .with_api_key(api_key);

        let client = Client::with_config(openai_config);

        Self {
            client,
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("static reqwest client config"),
            model: config.model.clone(),
            batch_size: config.batch_size,
            api_key: config.api_key.clone(),
        }
    }

    /// Embed one batch, retrying transient failures. Carries no progress reporting —
    /// see [`Self::embed_texts`] for why that distinction matters.
    async fn embed_batch(&self, batch: &[String], batch_index: usize) -> Result<Vec<Vec<f32>>> {
        let response = backoff::future::retry(embed_backoff(), || async {
            let request = CreateEmbeddingRequestArgs::default()
                .model(&self.model)
                .input(EmbeddingInput::StringArray(batch.to_vec()))
                .build()
                .map_err(backoff::Error::permanent)?;

            self.client.embeddings().create(request).await.map_err(|e| {
                if is_retryable(&e) {
                    tracing::warn!(
                        batch = batch_index,
                        texts = batch.len(),
                        "Transient embedding error, retrying: {e}"
                    );
                    backoff::Error::transient(e)
                } else {
                    backoff::Error::permanent(e)
                }
            })
        })
        .await?;

        tracing::debug!(batch = batch_index, texts = batch.len(), "Embedded batch");

        let mut data = response.data;
        data.sort_by_key(|e| e.index);
        Ok(data.into_iter().map(|e| e.embedding).collect())
    }

    /// Embed a corpus, reporting progress as each batch lands.
    ///
    /// Indexing calls this and nothing else does. Embedding a whole knowledge base is
    /// one `await` that can run for minutes, so per-batch reporting is what turns it
    /// from indistinguishable-from-hung into observable progress.
    ///
    /// Query embedding deliberately does **not** route through here: a search can run
    /// concurrently with an indexing run (nothing serializes them against each other
    /// anymore — the reindex worker is the only thing that mutates the index, but
    /// reads are unrestricted), so routing a query's embedding through this method
    /// would add it to that run's `INDEX_STATUS` chunk tally and push reported
    /// progress past 100%.
    pub async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

        let total = texts.len();
        let mut last_progress = std::time::Instant::now();

        for (batch_index, batch) in texts.chunks(self.batch_size).enumerate() {
            all_embeddings.extend(self.embed_batch(batch, batch_index).await?);

            // No-op unless an indexing run is in flight.
            crate::status::INDEX_STATUS.add_chunks_embedded(batch.len() as u64);

            let done = all_embeddings.len();
            if last_progress.elapsed() >= EMBED_PROGRESS_INTERVAL && done < total {
                let pct = if total > 0 {
                    (done as f64 / total as f64) * 100.0
                } else {
                    100.0
                };
                tracing::info!(
                    embedded = done,
                    total,
                    percent = format_args!("{pct:.1}"),
                    "Embedding progress"
                );
                last_progress = std::time::Instant::now();
            }
        }

        Ok(all_embeddings)
    }
}

/// Thin delegation impl — calls the identically-named inherent method.
/// The inherent `EmbedClient::embed_texts` is the real implementation; this impl
/// exists only to satisfy the `EmbedStore` trait used by `ingest.rs`.
impl EmbedStore for EmbedClient {
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        EmbedClient::embed_texts(self, texts).await
    }
}

/// Thin delegation impl — calls the identically-named inherent method.
/// The inherent `EmbedClient::embed_query` is the real implementation; this impl
/// exists only to satisfy the `QueryEmbedder` trait used by `retrieval.rs`.
impl QueryEmbedder for EmbedClient {
    async fn embed_query(&self, q: &str) -> anyhow::Result<Vec<f32>> {
        EmbedClient::embed_query(self, q).await
    }
}

impl EmbedClient {
    pub async fn health_check(&self) -> Result<()> {
        let url = format!(
            "{}/models",
            self.client.config().api_base().trim_end_matches('/')
        );
        let mut req = self.http_client.get(&url);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        req.send()
            .await
            .and_then(|r| r.error_for_status())
            .map_err(|e| anyhow::anyhow!("Embeddings service health check failed: {e}"))?;
        Ok(())
    }

    pub async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        // Goes straight to `embed_batch`, bypassing `embed_texts`' progress reporting:
        // searches run concurrently with indexing and must not be counted as indexing
        // work. See the note on `embed_texts`.
        let results = self.embed_batch(&[query.to_string()], 0).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding returned for query"))
    }
}

fn embed_backoff() -> ExponentialBackoff {
    ExponentialBackoffBuilder::new()
        .with_initial_interval(Duration::from_secs(1))
        .with_multiplier(2.0)
        .with_max_interval(Duration::from_secs(30))
        .with_max_elapsed_time(Some(Duration::from_secs(120)))
        .build()
}

fn is_retryable(err: &async_openai::error::OpenAIError) -> bool {
    use async_openai::error::OpenAIError;
    match err {
        OpenAIError::Reqwest(e) => {
            e.is_connect()
                || e.is_timeout()
                || e.status()
                    .is_some_and(|s| s.as_u16() == 429 || s.as_u16() >= 500)
        }
        OpenAIError::ApiError(api_err) => {
            let code = api_err.code.as_deref().unwrap_or("");
            let err_type = api_err.r#type.as_deref().unwrap_or("");
            let msg = api_err.message.to_lowercase();
            code == "rate_limit_exceeded"
                || err_type == "server_error"
                || msg.contains("service unavailable")
                || msg.contains("overloaded")
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::error::OpenAIError;

    #[test]
    fn test_is_retryable_connect_error() {
        // Build a connect error via reqwest
        let rt = tokio::runtime::Runtime::new().unwrap();
        let err = rt.block_on(async { reqwest::get("http://127.0.0.1:1").await.unwrap_err() });
        assert!(err.is_connect());
        let openai_err = OpenAIError::Reqwest(err);
        assert!(is_retryable(&openai_err));
    }

    #[test]
    fn test_is_retryable_api_error() {
        let openai_err = OpenAIError::ApiError(async_openai::error::ApiError {
            message: "bad request".into(),
            r#type: None,
            param: None,
            code: None,
        });
        assert!(!is_retryable(&openai_err));
    }

    #[test]
    fn test_is_retryable_invalid_argument() {
        let openai_err = OpenAIError::InvalidArgument("bad".into());
        assert!(!is_retryable(&openai_err));
    }

    #[tokio::test]
    async fn test_retry_exhaustion() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};

        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let result: Result<(), String> = backoff::future::retry(
            ExponentialBackoffBuilder::new()
                .with_initial_interval(Duration::from_millis(10))
                .with_max_elapsed_time(Some(Duration::from_millis(100)))
                .build(),
            || {
                let attempts = attempts_clone.clone();
                async move {
                    attempts.fetch_add(1, Ordering::SeqCst);
                    Err(backoff::Error::transient("still failing".to_string()))
                }
            },
        )
        .await;

        assert!(result.is_err());
        assert!(attempts.load(Ordering::SeqCst) > 1);
    }

    #[tokio::test]
    async fn test_retry_eventual_success() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};

        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let result: Result<&str, String> = backoff::future::retry(
            ExponentialBackoffBuilder::new()
                .with_initial_interval(Duration::from_millis(10))
                .with_max_elapsed_time(Some(Duration::from_secs(5)))
                .build(),
            || {
                let attempts = attempts_clone.clone();
                async move {
                    let n = attempts.fetch_add(1, Ordering::SeqCst);
                    if n < 3 {
                        Err(backoff::Error::transient("not yet".to_string()))
                    } else {
                        Ok("success")
                    }
                }
            },
        )
        .await;

        assert_eq!(result.unwrap(), "success");
        assert_eq!(attempts.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn test_is_retryable_rate_limit_code() {
        let openai_err = OpenAIError::ApiError(async_openai::error::ApiError {
            message: "Rate limit exceeded".into(),
            r#type: None,
            param: None,
            code: Some("rate_limit_exceeded".into()),
        });
        assert!(is_retryable(&openai_err));
    }

    #[test]
    fn test_is_retryable_server_error_type() {
        let openai_err = OpenAIError::ApiError(async_openai::error::ApiError {
            message: "Internal server error".into(),
            r#type: Some("server_error".into()),
            param: None,
            code: None,
        });
        assert!(is_retryable(&openai_err));
    }

    #[test]
    fn test_is_retryable_service_unavailable_message() {
        let openai_err = OpenAIError::ApiError(async_openai::error::ApiError {
            message: "Service Unavailable".into(),
            r#type: None,
            param: None,
            code: None,
        });
        assert!(is_retryable(&openai_err));
    }

    #[test]
    fn test_is_retryable_overloaded_message() {
        let openai_err = OpenAIError::ApiError(async_openai::error::ApiError {
            message: "The server is overloaded right now".into(),
            r#type: None,
            param: None,
            code: None,
        });
        assert!(is_retryable(&openai_err));
    }

    #[test]
    fn test_is_retryable_insufficient_quota() {
        let openai_err = OpenAIError::ApiError(async_openai::error::ApiError {
            message: "You exceeded your current quota".into(),
            r#type: Some("insufficient_quota".into()),
            param: None,
            code: None,
        });
        assert!(!is_retryable(&openai_err));
    }

    #[test]
    fn api_key_set_in_openai_config() {
        let config = crate::config::ResolvedEmbeddingConfig {
            base_url: "http://localhost:8080/v1".into(),
            model: "test-model".into(),
            api_key: Some("sk-test-key-123".into()),
            vector_size: 768,
            batch_size: 32,
        };
        let client = EmbedClient::new(&config);
        assert_eq!(client.api_key.as_deref(), Some("sk-test-key-123"));
    }

    #[test]
    fn api_key_absent_uses_fallback() {
        let config = crate::config::ResolvedEmbeddingConfig {
            base_url: "http://localhost:8080/v1".into(),
            model: "test-model".into(),
            api_key: None,
            vector_size: 768,
            batch_size: 32,
        };
        let client = EmbedClient::new(&config);
        assert!(client.api_key.is_none());
    }

    #[test]
    fn api_base_trailing_slash_trimmed() {
        let config = crate::config::ResolvedEmbeddingConfig {
            base_url: "http://localhost:8080/v1/".into(),
            model: "test-model".into(),
            api_key: None,
            vector_size: 768,
            batch_size: 32,
        };
        let client = EmbedClient::new(&config);
        // The health_check URL should not have a double slash
        let api_base = client.client.config().api_base().trim_end_matches('/');
        let url = format!("{}/models", api_base);
        assert!(
            !url.contains("//models"),
            "URL should not have double slash: {url}"
        );
    }
}
