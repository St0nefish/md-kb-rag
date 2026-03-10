use anyhow::Result;
use async_openai::{
    Client,
    config::{Config as OpenAIConfigTrait, OpenAIConfig},
    types::{CreateEmbeddingRequestArgs, EmbeddingInput},
};
use backoff::{ExponentialBackoff, ExponentialBackoffBuilder};
use std::time::Duration;

use crate::config::ResolvedEmbeddingConfig;

pub trait EmbedStore: Send + Sync {
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

pub struct EmbedClient {
    client: Client<OpenAIConfig>,
    http_client: reqwest::Client,
    model: String,
    batch_size: usize,
}

impl EmbedClient {
    pub fn new(config: &ResolvedEmbeddingConfig) -> Self {
        let openai_config = OpenAIConfig::new()
            .with_api_base(&config.base_url)
            .with_api_key("not-needed");

        let client = Client::with_config(openai_config);

        Self {
            client,
            http_client: reqwest::Client::new(),
            model: config.model.clone(),
            batch_size: config.batch_size,
        }
    }

    pub async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

        for batch in texts.chunks(self.batch_size) {
            let response = backoff::future::retry(embed_backoff(), || async {
                let request = CreateEmbeddingRequestArgs::default()
                    .model(&self.model)
                    .input(EmbeddingInput::StringArray(batch.to_vec()))
                    .build()
                    .map_err(backoff::Error::permanent)?;

                self.client.embeddings().create(request).await.map_err(|e| {
                    if is_retryable(&e) {
                        tracing::warn!("Transient embedding error, retrying: {e}");
                        backoff::Error::transient(e)
                    } else {
                        backoff::Error::permanent(e)
                    }
                })
            })
            .await?;

            let mut data = response.data;
            data.sort_by_key(|e| e.index);

            for embedding in data {
                all_embeddings.push(embedding.embedding);
            }
        }

        Ok(all_embeddings)
    }
}

impl EmbedStore for EmbedClient {
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        EmbedClient::embed_texts(self, texts).await
    }
}

impl EmbedClient {
    pub async fn health_check(&self) -> Result<()> {
        let url = format!(
            "{}/models",
            self.client.config().api_base().trim_end_matches('/')
        );
        self.http_client
            .get(&url)
            .send()
            .await
            .and_then(|r| r.error_for_status())
            .map_err(|e| anyhow::anyhow!("Embeddings service health check failed: {e}"))?;
        Ok(())
    }

    pub async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let results = self.embed_texts(&[query.to_string()]).await?;
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
        OpenAIError::Reqwest(e) => e.is_connect() || e.is_timeout(),
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
    fn api_base_trailing_slash_trimmed() {
        let config = crate::config::ResolvedEmbeddingConfig {
            base_url: "http://localhost:8080/v1/".into(),
            model: "test-model".into(),
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
