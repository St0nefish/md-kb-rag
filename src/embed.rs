use anyhow::Result;
use async_openai::{
    Client,
    config::{Config as OpenAIConfigTrait, OpenAIConfig},
    types::{CreateEmbeddingRequestArgs, EmbeddingInput},
};
use backoff::{ExponentialBackoff, ExponentialBackoffBuilder};
use std::future::Future;
use std::time::Duration;
use tokio::task::JoinSet;

use crate::config::ResolvedEmbeddingConfig;

/// How often `embed_texts` logs progress through a long batch sequence.
const EMBED_PROGRESS_INTERVAL: Duration = Duration::from_secs(10);

/// A completed batch task's result: (batch index, chunk count, embeddings-or-error).
/// The index is what `embed_batches_ordered` uses to write into the correct slot
/// regardless of completion order — see its doc comment.
type BatchOutcome = (usize, usize, Result<Vec<Vec<f32>>>);

pub trait EmbedStore: Send + Sync {
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

pub trait QueryEmbedder: Send + Sync {
    async fn embed_query(&self, q: &str) -> anyhow::Result<Vec<f32>>;
}

/// Cloning is cheap: `Client<OpenAIConfig>` and `reqwest::Client` both clone an
/// internal `Arc`, and the rest of the fields are small owned values. This makes
/// `EmbedClient` spawn-friendly — `embed_texts` clones it into each concurrent
/// batch task (see below) rather than needing `Arc<EmbedClient>` at every call site.
#[derive(Clone)]
pub struct EmbedClient {
    client: Client<OpenAIConfig>,
    http_client: reqwest::Client,
    model: String,
    batch_size: usize,
    batch_concurrency: usize,
    api_key: Option<String>,
}

impl EmbedClient {
    pub fn new(config: &ResolvedEmbeddingConfig) -> Self {
        let api_key = config.api_key.as_deref().unwrap_or("not-needed");
        let openai_config = OpenAIConfig::new()
            .with_api_base(&config.base_url)
            .with_api_key(api_key);

        // async-openai's `Client::with_config` builds its own internal
        // `reqwest::Client` with NO timeout — unlike `self.http_client` below (used
        // only by `health_check`), that internal client is what actually issues
        // embedding requests, so without attaching our own, a hung connection would
        // block forever instead of erroring. The existing exponential backoff in
        // `embed_backoff` only bounds requests that *fail*; it does nothing for a
        // request that never returns. `with_http_client` (a builder method on
        // `Client`, verified against async-openai 0.27.2's `client.rs`) is how a
        // custom `reqwest::Client` gets wired in. The default of 60s (vs.
        // rerank.rs's 10s for a single lightweight call) reflects that embedding a
        // full batch on the deployed CPU-only service legitimately takes several
        // seconds.
        let embed_http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .build()
            .expect("static reqwest client config");
        let client = Client::with_config(openai_config).with_http_client(embed_http_client);

        Self {
            client,
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("static reqwest client config"),
            model: config.model.clone(),
            batch_size: config.batch_size,
            batch_concurrency: config.batch_concurrency,
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
    ///
    /// Batches run concurrently, bounded by `self.batch_concurrency` — see
    /// [`embed_batches_ordered`] for how results stay positionally aligned with
    /// `texts` despite batches completing out of order.
    pub async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let total = texts.len();
        let batches: Vec<Vec<String>> = texts.chunks(self.batch_size).map(|b| b.to_vec()).collect();

        let mut done = 0usize;
        let mut last_progress = std::time::Instant::now();

        // Cloned once up front, then cloned again per spawned batch task inside
        // `embed_batches_ordered` — both clones are cheap (see the doc comment on
        // `EmbedClient`).
        let client = self.clone();
        embed_batches_ordered(
            batches,
            self.batch_concurrency,
            move |batch_index, batch| {
                let client = client.clone();
                async move { client.embed_batch(&batch, batch_index).await }
            },
            move |batch_len| {
                // No-op unless an indexing run is in flight.
                crate::status::INDEX_STATUS.add_chunks_embedded(batch_len as u64);

                done += batch_len;
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
            },
        )
        .await
    }
}

/// Runs `batches` through `embed_one`, keeping at most `concurrency` batches in
/// flight at a time, and reassembles the results in **input order** regardless of
/// which batch happens to complete first.
///
/// # Why this exists
///
/// The returned `Vec<Vec<f32>>` must correspond positionally to the flattened
/// input batches — chunk *N* must get embedding *N*, always. Under concurrency,
/// batches do not complete in the order they were spawned (a later batch can
/// finish before an earlier one, e.g. if the embedding server picks it up first).
/// If results were assembled in *completion* order instead of *input* order (e.g.
/// by naively `extend`-ing a `Vec` as each batch finishes, the way the old
/// strictly-sequential loop did), every chunk from that point on would silently
/// receive some other chunk's vector — no error, no panic, just permanently wrong
/// search results. `slots` below is indexed by batch position specifically to
/// make that failure mode structurally impossible: a batch can only ever write
/// into its own slot, no matter when it finishes.
///
/// Extracted as a free function (generic over `embed_one`) so this reassembly
/// logic can be exercised directly in tests with a fake embedder whose batches
/// complete out of order, without needing a real HTTP embedding service.
async fn embed_batches_ordered<F, Fut>(
    batches: Vec<Vec<String>>,
    concurrency: usize,
    embed_one: F,
    mut on_batch_done: impl FnMut(usize),
) -> Result<Vec<Vec<f32>>>
where
    F: Fn(usize, Vec<String>) -> Fut + Send + Sync + Clone + 'static,
    Fut: Future<Output = Result<Vec<Vec<f32>>>> + Send + 'static,
{
    let batch_count = batches.len();
    if batch_count == 0 {
        return Ok(Vec::new());
    }
    let concurrency = concurrency.max(1);

    // Written by batch index, never by completion order — see the function doc
    // comment above for why that's the load-bearing correctness property here.
    let mut slots: Vec<Option<Vec<Vec<f32>>>> = vec![None; batch_count];

    let mut set: JoinSet<BatchOutcome> = JoinSet::new();
    let mut queue = batches.into_iter().enumerate();

    // Prime the pipeline with up to `concurrency` in-flight batches. Each time one
    // completes below, its slot is filled and (if the queue isn't empty) the next
    // queued batch is spawned — so at most `concurrency` requests are ever
    // outstanding at once, no matter how fast or slow individual batches are.
    for (batch_index, batch) in queue.by_ref().take(concurrency) {
        let embed_one = embed_one.clone();
        set.spawn(async move {
            let batch_len = batch.len();
            (batch_index, batch_len, embed_one(batch_index, batch).await)
        });
    }

    while let Some(joined) = set.join_next().await {
        let (batch_index, batch_len, result) =
            joined.map_err(|e| anyhow::anyhow!("embedding task panicked: {e}"))?;
        // A batch error propagates immediately, same as the old sequential `?`.
        // Dropping `set` here (via early return) aborts any still-in-flight
        // sibling batches rather than letting them run to a result nobody reads.
        let embeddings = result?;
        slots[batch_index] = Some(embeddings);
        on_batch_done(batch_len);

        if let Some((next_index, next_batch)) = queue.next() {
            let embed_one = embed_one.clone();
            set.spawn(async move {
                let batch_len = next_batch.len();
                (
                    next_index,
                    batch_len,
                    embed_one(next_index, next_batch).await,
                )
            });
        }
    }

    let mut all_embeddings = Vec::with_capacity(batch_count);
    for slot in slots {
        // Every slot is guaranteed filled here: the loop above only exits once
        // `set` is empty, and it returns early (before this point) on any batch
        // error, so reaching this line means every spawned batch — i.e. every
        // slot — succeeded.
        all_embeddings
            .extend(slot.expect("all batch slots are filled when no batch returned an error"));
    }

    Ok(all_embeddings)
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
            request_timeout_secs: 60,
            batch_concurrency: 4,
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
            request_timeout_secs: 60,
            batch_concurrency: 4,
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
            request_timeout_secs: 60,
            batch_concurrency: 4,
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

    /// The regression test for the highest-risk part of concurrent batch
    /// embedding: if `embed_batches_ordered` assembled results by *completion*
    /// order instead of *batch index* (e.g. a naive `extend` as each batch
    /// finishes — which is exactly what the old strictly-sequential loop did,
    /// and which happens to still pass if batches merely complete in the order
    /// they were spawned), every chunk would silently get some other chunk's
    /// embedding.
    ///
    /// To actually exercise that failure mode, this deliberately makes batches
    /// complete in the REVERSE of spawn order (batch 0 sleeps longest, batch 2
    /// sleeps shortest) and encodes each batch's index into its fake
    /// "embedding" so misassembly is directly observable in the assertion.
    /// `start_paused = true` lets the sleeps resolve instantly in wall-clock
    /// terms while still forcing that completion ordering.
    #[tokio::test(start_paused = true)]
    async fn embed_batches_ordered_reassembles_reversed_completions() {
        let batches = vec![
            vec!["a".to_string()],
            vec!["b".to_string(), "b2".to_string()],
            vec!["c".to_string()],
        ];

        let embed_one = |batch_index: usize, batch: Vec<String>| async move {
            // Batch 0 finishes last, batch 2 finishes first: completion order is
            // the exact reverse of input/spawn order.
            let delay = Duration::from_secs(3 - batch_index as u64);
            tokio::time::sleep(delay).await;
            Ok(batch
                .into_iter()
                .map(|_| vec![batch_index as f32])
                .collect())
        };

        // All 3 batches fit within the concurrency budget, so all spawn at once
        // and race to completion in reverse order.
        let result = embed_batches_ordered(batches, 3, embed_one, |_| {})
            .await
            .unwrap();

        assert_eq!(
            result,
            vec![vec![0.0], vec![1.0], vec![1.0], vec![2.0]],
            "output must stay positionally aligned with input batch order, \
             not completion order"
        );
    }

    /// Confirms `embed_batches_ordered` actually bounds concurrency rather than
    /// firing every batch at once — the point of this change is to not overrun
    /// the embedding server's `--parallel` slot count.
    #[tokio::test(start_paused = true)]
    async fn embed_batches_ordered_bounds_concurrency() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_in_flight = Arc::new(AtomicUsize::new(0));

        let batches: Vec<Vec<String>> = (0..6).map(|i| vec![format!("t{i}")]).collect();

        let in_flight_for_closure = in_flight.clone();
        let max_in_flight_for_closure = max_in_flight.clone();
        let embed_one = move |batch_index: usize, batch: Vec<String>| {
            let in_flight = in_flight_for_closure.clone();
            let max_in_flight = max_in_flight_for_closure.clone();
            async move {
                let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                max_in_flight.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(10)).await;
                in_flight.fetch_sub(1, Ordering::SeqCst);
                Ok(batch
                    .into_iter()
                    .map(|_| vec![batch_index as f32])
                    .collect())
            }
        };

        let result = embed_batches_ordered(batches, 2, embed_one, |_| {})
            .await
            .unwrap();

        assert_eq!(
            result,
            (0..6).map(|i| vec![i as f32]).collect::<Vec<_>>(),
            "order must still be preserved under bounded concurrency"
        );
        assert!(
            max_in_flight.load(Ordering::SeqCst) <= 2,
            "concurrency exceeded the configured limit of 2: saw {} in flight",
            max_in_flight.load(Ordering::SeqCst)
        );
    }

    /// A permanent error from any batch must still propagate out of
    /// `embed_batches_ordered`, same as the old sequential `?` did.
    #[tokio::test(start_paused = true)]
    async fn embed_batches_ordered_propagates_batch_error() {
        let batches = vec![vec!["a".to_string()], vec!["b".to_string()]];

        let embed_one = |batch_index: usize, _batch: Vec<String>| async move {
            if batch_index == 1 {
                anyhow::bail!("simulated permanent failure");
            }
            Ok(vec![vec![0.0]])
        };

        let result = embed_batches_ordered(batches, 2, embed_one, |_| {}).await;
        assert!(result.is_err());
    }

    /// `on_batch_done` must fire once per batch with that batch's chunk count,
    /// regardless of completion order — this is what keeps `embed_texts`'s
    /// progress counter (and `INDEX_STATUS.add_chunks_embedded`) accurate under
    /// concurrency.
    #[tokio::test(start_paused = true)]
    async fn embed_batches_ordered_reports_each_batch_once() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let batches = vec![
            vec!["a".to_string(), "a2".to_string()],
            vec!["b".to_string()],
            vec!["c".to_string(), "c2".to_string(), "c3".to_string()],
        ];

        let embed_one = |batch_index: usize, batch: Vec<String>| async move {
            Ok(batch
                .into_iter()
                .map(|_| vec![batch_index as f32])
                .collect())
        };

        let total_reported = Arc::new(AtomicUsize::new(0));
        let total_reported_for_closure = total_reported.clone();
        embed_batches_ordered(batches, 2, embed_one, move |batch_len| {
            total_reported_for_closure.fetch_add(batch_len, Ordering::SeqCst);
        })
        .await
        .unwrap();

        assert_eq!(total_reported.load(Ordering::SeqCst), 6);
    }
}
