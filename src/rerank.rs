use anyhow::Result;
use backoff::{ExponentialBackoff, ExponentialBackoffBuilder, future::retry};
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use crate::config::ResolvedRerankingConfig;

pub trait Reranker: Send + Sync {
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        documents: &'a [&'a str],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<RerankResult>>> + Send + 'a>>;
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f32,
}

pub struct RerankClient {
    http_client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
}

impl RerankClient {
    pub fn new(config: &ResolvedRerankingConfig) -> Self {
        Self {
            http_client: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .expect("static reqwest client config"),
            base_url: config.base_url.clone(),
            model: config.model.clone(),
            api_key: config.api_key.clone(),
        }
    }
}

impl Reranker for RerankClient {
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        documents: &'a [&'a str],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<RerankResult>>> + Send + 'a>> {
        Box::pin(async move {
            let body = serde_json::json!({
                "model": self.model,
                "query": query,
                "documents": documents,
            });

            let base_url = self.base_url.trim_end_matches('/').to_string();
            let api_key = self.api_key.clone();
            let http_client = self.http_client.clone();

            let results = retry(rerank_backoff(), || {
                let body = body.clone();
                let base_url = base_url.clone();
                let api_key = api_key.clone();
                let http_client = http_client.clone();
                async move {
                    let url = format!("{base_url}/rerank");
                    let mut req = http_client.post(&url).json(&body);
                    if let Some(ref key) = api_key {
                        req = req.bearer_auth(key);
                    }
                    let resp = req.send().await.map_err(|e| {
                        if is_retryable(&e) {
                            tracing::warn!("Transient rerank error, retrying: {e}");
                            backoff::Error::transient(anyhow::anyhow!(e))
                        } else {
                            backoff::Error::permanent(anyhow::anyhow!(e))
                        }
                    })?;

                    let status = resp.status();
                    if !status.is_success() {
                        let err = anyhow::anyhow!("Reranker returned status {status}");
                        if status.as_u16() == 429 || status.as_u16() >= 500 {
                            tracing::warn!("Transient rerank error {status}, retrying");
                            return Err(backoff::Error::transient(err));
                        }
                        return Err(backoff::Error::permanent(err));
                    }

                    #[derive(serde::Deserialize)]
                    struct RerankResponse {
                        results: Vec<RerankResult>,
                    }

                    let parsed: RerankResponse = resp
                        .json()
                        .await
                        .map_err(|e| backoff::Error::permanent(anyhow::anyhow!(e)))?;

                    Ok(parsed.results)
                }
            })
            .await?;

            Ok(results)
        })
    }
}

fn is_retryable(err: &reqwest::Error) -> bool {
    err.is_connect()
        || err.is_timeout()
        || err
            .status()
            .is_some_and(|s| s.as_u16() == 429 || s.as_u16() >= 500)
}

fn rerank_backoff() -> ExponentialBackoff {
    ExponentialBackoffBuilder::new()
        .with_initial_interval(Duration::from_secs(1))
        .with_multiplier(2.0)
        .with_max_interval(Duration::from_secs(30))
        .with_max_elapsed_time(Some(Duration::from_secs(120)))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn is_retryable_connect_error() {
        let err = reqwest::get("http://127.0.0.1:1").await.unwrap_err();
        assert!(err.is_connect());
        assert!(is_retryable(&err));
    }

    #[test]
    fn rerank_result_deserializes() {
        let val = serde_json::json!({"index": 2, "relevance_score": 0.95});
        let r: RerankResult = serde_json::from_value(val).unwrap();
        assert_eq!(r.index, 2);
        assert!((r.relevance_score - 0.95).abs() < 1e-5);
    }
}
