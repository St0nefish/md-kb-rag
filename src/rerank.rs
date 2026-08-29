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
    /// Per-document byte budget, derived from `chunking.max_chunk_size` — see
    /// [`ResolvedRerankingConfig::max_document_bytes`].
    max_document_bytes: usize,
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
            max_document_bytes: config.max_document_bytes,
        }
    }
}

/// Largest prefix of `doc` that fits in `budget` bytes without splitting a UTF-8
/// character. Returns `doc` untouched when it already fits, so the common case
/// costs nothing and allocates nothing — the result is a borrowed subslice.
///
/// The reranker rejects the entire request when any single document exceeds its
/// physical batch size, so one oversized chunk otherwise costs every candidate
/// its reranking (#128). Truncating degrades *that document's* score rather than
/// the whole query.
fn truncate_for_rerank(doc: &str, budget: usize) -> &str {
    if doc.len() <= budget {
        return doc;
    }
    // Walk back to a char boundary — at most 3 steps, since a UTF-8 sequence is
    // 4 bytes at most. `str::floor_char_boundary` is still unstable.
    let mut end = budget;
    while end > 0 && !doc.is_char_boundary(end) {
        end -= 1;
    }
    &doc[..end]
}

impl Reranker for RerankClient {
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        documents: &'a [&'a str],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<RerankResult>>> + Send + 'a>> {
        Box::pin(async move {
            // Bound every document to the chunking budget before it goes on the
            // wire. Count and order are preserved, so `RerankResult.index` still
            // maps back onto the caller's candidate list unchanged.
            let budget = self.max_document_bytes;
            let truncated: Vec<&str> = documents
                .iter()
                .map(|d| truncate_for_rerank(d, budget))
                .collect();
            let shortened = documents.iter().filter(|d| d.len() > budget).count();
            if shortened > 0 {
                let longest = documents.iter().map(|d| d.len()).max().unwrap_or(0);
                // Once per request, not once per document: visible without being
                // noisy, since a single oversized chunk used to fail silently.
                tracing::warn!(
                    "Truncated {shortened} of {} rerank documents to the {budget}-byte budget \
                     (longest was {longest} bytes). The budget follows chunking.max_chunk_size; \
                     documents exceed it when a description is prepended or a section cannot be \
                     split. Only the reranker's view is truncated — returned content is unaffected.",
                    truncated.len()
                );
            }

            let body = serde_json::json!({
                "model": self.model,
                "query": query,
                "documents": truncated,
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

/// Deliberately small. Reranking is an *enhancement* — its failure path (`Err` →
/// fused order, `retrieval::search`) is perfectly serviceable, so a long retry
/// budget buys nothing and costs a great deal: a permanent llama.cpp 500 is
/// classified transient like any other 5xx, and under the old 120s budget it
/// retried ~6 times over ~40s, turning a 0.9s failure into an MCP tool timeout
/// that made search look broken (#127).
///
/// 5s still gives a genuine blip 2-3 attempts (~0s, ~1s, ~3s) while a permanent
/// error costs about a second. We deliberately do *not* string-match the error
/// body to detect permanent 500s — that is brittle across server versions and
/// backends; bounding the budget fixes the symptom regardless of cause.
fn rerank_backoff() -> ExponentialBackoff {
    ExponentialBackoffBuilder::new()
        .with_initial_interval(Duration::from_secs(1))
        .with_multiplier(2.0)
        .with_max_interval(Duration::from_secs(2))
        .with_max_elapsed_time(Some(Duration::from_secs(5)))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

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

    // ── truncate_for_rerank ──────────────────────────────────────────────────

    #[test]
    fn under_budget_passes_through_untouched() {
        let doc = "short enough";
        let out = truncate_for_rerank(doc, 1500);
        assert_eq!(out, doc);
        // Borrowed, not rebuilt: the common path must not copy.
        assert!(std::ptr::eq(out, doc));
    }

    #[test]
    fn exactly_at_budget_is_not_truncated() {
        let doc = "abcde";
        assert_eq!(truncate_for_rerank(doc, 5), "abcde");
    }

    #[test]
    fn over_budget_is_cut_to_the_budget() {
        let doc = "abcdefghij";
        let out = truncate_for_rerank(doc, 4);
        assert_eq!(out, "abcd");
        assert!(out.len() <= 4);
    }

    #[test]
    fn multibyte_is_never_split_mid_character() {
        // Each 'é' is 2 bytes, each '日' is 3 — budgets below land mid-sequence.
        let accented = "ééééé"; // 10 bytes
        let out = truncate_for_rerank(accented, 5);
        assert_eq!(
            out, "éé",
            "must step back to a char boundary, not cut 'é' in half"
        );
        assert!(out.len() <= 5);

        let cjk = "日本語テキスト"; // 21 bytes
        for budget in 1..=21 {
            let out = truncate_for_rerank(cjk, budget);
            assert!(
                out.len() <= budget,
                "budget {budget} exceeded: {} bytes",
                out.len()
            );
            // A `&str` that survived slicing is valid UTF-8 by construction; the
            // real assertion is that the slice did not panic and stayed a prefix.
            assert!(cjk.starts_with(out));
        }
    }

    #[test]
    fn budget_smaller_than_the_first_character_yields_empty() {
        // 3-byte character, 2-byte budget: there is no non-empty prefix that fits.
        assert_eq!(truncate_for_rerank("日", 2), "");
    }

    // ── end-to-end against a fake reranker ───────────────────────────────────

    /// A throwaway HTTP server on loopback: reads each request, records its body,
    /// and replies with a canned response. Deliberately hand-rolled — the repo has
    /// no HTTP-mock dev-dependency, and this is ~40 lines against adding one.
    struct FakeReranker {
        base_url: String,
        bodies: Arc<Mutex<Vec<String>>>,
    }

    async fn spawn_fake_reranker(status_line: &'static str, payload: &'static str) -> FakeReranker {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let bodies = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::clone(&bodies);

        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let recorder = Arc::clone(&recorder);
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut tmp = [0u8; 4096];
                    // Read until end-of-headers.
                    let header_end = loop {
                        match sock.read(&mut tmp).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        }
                        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            break p;
                        }
                    };
                    let headers = String::from_utf8_lossy(&buf[..header_end]).to_lowercase();
                    let len = headers
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length:"))
                        .and_then(|v| v.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    let body_start = header_end + 4;
                    // Then read exactly the declared body.
                    while buf.len() < body_start + len {
                        match sock.read(&mut tmp).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        }
                    }
                    let end = (body_start + len).min(buf.len());
                    recorder
                        .lock()
                        .unwrap()
                        .push(String::from_utf8_lossy(&buf[body_start..end]).into_owned());

                    let resp = format!(
                        "{status_line}\r\ncontent-type: application/json\r\n\
                         content-length: {}\r\nconnection: close\r\n\r\n{payload}",
                        payload.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                    let _ = sock.flush().await;
                });
            }
        });

        FakeReranker {
            base_url: format!("http://{addr}"),
            bodies,
        }
    }

    fn client_for(base_url: &str, max_document_bytes: usize) -> RerankClient {
        RerankClient::new(&ResolvedRerankingConfig {
            base_url: base_url.to_string(),
            model: "test-reranker".into(),
            api_key: None,
            candidate_limit: 50,
            max_document_bytes,
        })
    }

    /// The #128 regression: one oversized document must not cost the request. It
    /// is truncated to the budget, every other document is sent untouched, and the
    /// document count is preserved so `RerankResult.index` still lines up.
    #[tokio::test]
    async fn oversized_documents_are_truncated_not_dropped() {
        let fake = spawn_fake_reranker(
            "HTTP/1.1 200 OK",
            r#"{"results":[{"index":0,"relevance_score":0.9},{"index":1,"relevance_score":0.8}]}"#,
        )
        .await;

        let long = "x".repeat(500);
        let docs = ["short doc", long.as_str()];
        let results = client_for(&fake.base_url, 32)
            .rerank("a query", &docs)
            .await
            .expect("one oversized document must not fail the request");
        assert_eq!(results.len(), 2);

        let sent = fake.bodies.lock().unwrap();
        assert_eq!(sent.len(), 1, "a 200 must not be retried");
        let body: serde_json::Value = serde_json::from_str(&sent[0]).unwrap();
        let documents = body["documents"].as_array().unwrap();
        assert_eq!(documents.len(), 2, "no document may be dropped");
        assert_eq!(
            documents[0].as_str().unwrap(),
            "short doc",
            "a document within budget goes over the wire byte-identical"
        );
        let oversized = documents[1].as_str().unwrap();
        assert_eq!(
            oversized.len(),
            32,
            "the long document is cut to the budget"
        );
        assert!(long.starts_with(oversized));
    }

    /// The #127 regression: a persistently-500ing reranker must fail fast enough
    /// to fall back to fused order well inside an MCP call, not retry for 120s.
    /// The bound is deliberately far looser than the 5s budget so a slow CI runner
    /// cannot flake it; the old 120s budget would blow through it regardless.
    #[tokio::test]
    async fn persistent_5xx_gives_up_within_the_retry_budget() {
        let fake =
            spawn_fake_reranker("HTTP/1.1 500 Internal Server Error", r#"{"error":"boom"}"#).await;

        let started = std::time::Instant::now();
        let out = client_for(&fake.base_url, 1500)
            .rerank("a query", &["doc"])
            .await;
        let elapsed = started.elapsed();

        assert!(out.is_err(), "a persistent 500 must surface as an error");
        assert!(
            elapsed < Duration::from_secs(15),
            "gave up after {elapsed:?}; the retry budget is supposed to be ~5s"
        );
        assert!(
            fake.bodies.lock().unwrap().len() > 1,
            "a 5xx is still classified transient, so it must be retried at least once"
        );
    }
}
