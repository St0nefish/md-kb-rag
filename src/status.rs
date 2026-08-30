//! Runtime observability for the indexing pipeline.
//!
//! `ingest::index_paths` computes everything worth knowing about a run — how many
//! paths it saw, how many it embedded, how many it refused — and used to log it and
//! throw it away. Answering "is an index running, and how did the last one go?" then
//! meant `docker exec` into the container and reading logs, which is the wrong shape
//! for a service whose whole job is keeping an index in sync.
//!
//! This module keeps that state in memory so `/status` and `/metrics` can answer in one
//! call. It is deliberately process-global: `index_paths` is the sole index mutator per
//! process (the reindex worker is the only thing that calls it in `serve` mode), and
//! nothing here is written from more than one call site the way `reindex::ReindexQueue`
//! used to be reached from every write tool, the webhook handler, and the worker at
//! once — see that module's doc comment on why an ambient global was the wrong shape
//! for something with that many independent writers, and why it is now an injected
//! `Arc<ReindexQueue>` instead.
//!
//! Nothing here is persisted. A restart resets the run history, which is correct — the
//! interesting question is almost always "what has this process done", and the durable
//! counts (documents, points) live in SQLite and Qdrant where they belong.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Wall-clock seconds since the epoch, saturating at 0 if the clock is before 1970.
fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Format epoch seconds as RFC 3339 for the JSON view.
///
/// `chrono` is built without the `clock` feature here, so `Utc::now()` is unavailable —
/// wall-clock time comes from `SystemTime` and is only *formatted* by chrono.
pub fn format_unix(ts: i64) -> String {
    chrono::DateTime::from_timestamp(ts, 0)
        .map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
        .unwrap_or_else(|| ts.to_string())
}

/// Whether the run rebuilds from scratch or only touches what changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    Full,
    Incremental,
}

impl RunMode {
    /// Every variant, so state-set metrics can emit a 0 for the inactive ones.
    pub const ALL: [Self; 2] = [Self::Full, Self::Incremental];

    pub fn from_full(full: bool) -> Self {
        if full { Self::Full } else { Self::Incremental }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Incremental => "incremental",
        }
    }
}

/// What caused this run to start. Distinguishing these is what makes "the webhook fired
/// but nothing happened" separable from "nobody ever told it to index".
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Trigger {
    /// `md-kb-rag index` from the command line.
    Cli,
    /// The initial full index after a fresh git clone, run synchronously before the
    /// server starts serving — the one case with no worker to hand work to yet.
    Startup,
    /// A push notification from the knowledge base's git host.
    ///
    /// No longer set by `webhook.rs` directly — the webhook handler marks the pushed
    /// commit range's paths dirty and returns; the reindex worker performs the actual
    /// run, tagged `Worker`. Kept as a variant for API stability and because tests
    /// still exercise the webhook's path-marking behavior under this name.
    Webhook,
    /// A `create_document` / `edit_document` / `delete_document` MCP call.
    ///
    /// Same caveat as `Webhook`: the write tools mark their paths dirty and return
    /// immediately rather than calling into the indexer, so `Worker` is what actually
    /// appears in `/status` for the resulting run.
    WriteTool,
    /// The background reindex worker, draining the dirty-path queue (`src/reindex.rs`).
    /// This is what performs almost every indexing run in a running server: writes,
    /// webhooks, and both the startup and periodic reconcile sweeps all funnel through
    /// here, which is why an MCP write call's latency is no longer coupled to embedding
    /// time — the worker does that work out of band.
    Worker,
}

impl Trigger {
    /// Every variant, so state-set metrics can emit a 0 for the inactive ones.
    pub const ALL: [Self; 5] = [
        Self::Cli,
        Self::Startup,
        Self::Webhook,
        Self::WriteTool,
        Self::Worker,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cli => "cli",
            Self::Startup => "startup",
            Self::Webhook => "webhook",
            Self::WriteTool => "write_tool",
            Self::Worker => "worker",
        }
    }
}

/// Coarse stage of an in-flight run, ordered as the pipeline executes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Opening the state DB, ensuring the collection, walking schemas.
    Starting,
    /// Walking the data directory for candidate files.
    Discovering,
    /// Reading, validating and chunking each discovered file.
    Scanning,
    /// Calling the embedding API and upserting points.
    Embedding,
    /// Repairing metadata for files whose content did not change.
    Backfilling,
    /// Deleting index entries for files no longer on disk.
    RemovingOrphans,
}

impl Phase {
    /// Every variant, so state-set metrics can emit a 0 for the inactive ones.
    pub const ALL: [Self; 6] = [
        Self::Starting,
        Self::Discovering,
        Self::Scanning,
        Self::Embedding,
        Self::Backfilling,
        Self::RemovingOrphans,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Discovering => "discovering",
            Self::Scanning => "scanning",
            Self::Embedding => "embedding",
            Self::Backfilling => "backfilling",
            Self::RemovingOrphans => "removing_orphans",
        }
    }
}

/// The per-run tallies `run_index` already reported to its summary log line.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RunCounters {
    pub discovered: u64,
    pub indexed: u64,
    pub skipped: u64,
    pub invalid: u64,
    pub empty: u64,
    pub read_errors: u64,
    pub metadata_backfilled: u64,
    pub frozen_by_broken_schema: u64,
    pub broken_schemas: u64,
    pub orphans_removed: u64,
    /// Files rejected this run by `validation.strict` — see `ingest::FileOutcome::Rejected`.
    /// Unlike every other counter here, a non-zero value is not fully explained by this
    /// one run: `INDEX_STATUS::strict_rejected_files` (populated alongside this counter)
    /// is the durable, per-path view that survives past this run's summary, because a
    /// rejected file's state row is never updated and so it resurfaces on every sweep.
    pub strict_rejected: u64,
}

impl RunCounters {
    /// Name/value pairs for the Prometheus encoder, so adding a counter above cannot
    /// silently fail to appear in `/metrics`.
    pub fn as_pairs(&self) -> [(&'static str, u64); 11] {
        [
            ("discovered", self.discovered),
            ("indexed", self.indexed),
            ("skipped", self.skipped),
            ("invalid", self.invalid),
            ("empty", self.empty),
            ("read_errors", self.read_errors),
            ("metadata_backfilled", self.metadata_backfilled),
            ("frozen_by_broken_schema", self.frozen_by_broken_schema),
            ("broken_schemas", self.broken_schemas),
            ("orphans_removed", self.orphans_removed),
            ("strict_rejected", self.strict_rejected),
        ]
    }
}

/// Snapshot of a run that is still going.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RunProgress {
    pub mode: RunMode,
    pub trigger: Trigger,
    pub phase: Phase,
    pub started_unix: i64,
    pub started_at: String,
    pub elapsed_secs: f64,
    /// Files scanned so far out of those discovered. Zero/zero before discovery finishes.
    pub files_done: u64,
    pub files_total: u64,
    /// Chunks embedded so far out of those queued. This is the number that was missing
    /// during a nine-minute silent re-embed of an entire knowledge base.
    pub chunks_embedded: u64,
    pub chunks_total: u64,
    /// Counters accumulated so far this run.
    pub counters: RunCounters,
}

/// Snapshot of the most recent run that finished, successfully or not.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LastRun {
    pub mode: RunMode,
    pub trigger: Trigger,
    pub success: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub started_unix: i64,
    pub started_at: String,
    pub finished_unix: i64,
    pub finished_at: String,
    pub duration_secs: f64,
    pub counters: RunCounters,
}

/// Whether a Qdrant payload index was successfully created.
///
/// Index creation failures are logged and tolerated so a bad index cannot stop the
/// server from booting, which means they were previously invisible the moment the log
/// line scrolled away — while filters on that field silently returned partial results.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PayloadIndexState {
    Ok,
    Failed { error: String },
}

/// A file currently rejected by `validation.strict`, and why.
///
/// Populated via [`IndexStatus::record_strict_rejection`] from `ingest::index_paths`
/// when `process_file` returns `FileOutcome::Rejected` for a path, and cleared for
/// that path when it later indexes cleanly (`FileOutcome::Ready`) or is purged as an
/// orphan (removed from disk). See that method's doc comment for why a healed path is
/// removed from the map entirely rather than flipped to some "ok" variant — unlike
/// [`PayloadIndexState`], there is no healthy state worth recording here, only an
/// absence of a defect.
///
/// This is deliberately per-run-instance state (reset on restart, like everything else
/// in this module), not a substitute for durability: what makes it trustworthy anyway
/// is that a rejected file's `indexed_files` row is never updated on rejection, so
/// `scan_for_dirty` re-presents it on every reconcile sweep — the map repopulates
/// within one sweep interval even after a restart wipes it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StrictRejection {
    pub reason: String,
}

/// Everything the status endpoints read, captured under one lock acquisition.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StatusSnapshot {
    /// True while a run is in flight. The single field that was impossible to obtain.
    pub indexing: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current: Option<RunProgress>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_run: Option<LastRun>,
    pub runs_total: u64,
    pub runs_failed: u64,
    /// When a run last completed *successfully*, distinct from when one last completed.
    /// This is the field worth alerting on: "no successful index in six hours" is a real
    /// problem, whereas "last run failed" may already have been retried.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_success_unix: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_success_at: Option<String>,
    pub payload_indexes: BTreeMap<String, PayloadIndexState>,
    /// Files currently rejected by `validation.strict`, keyed by repo-relative path.
    /// The map's cardinality is the `kb_strict_rejected_files` gauge — see
    /// [`StrictRejection`] for why an entry is removed on healing rather than marked ok.
    #[serde(default)]
    pub strict_rejected_files: BTreeMap<String, StrictRejection>,
}

#[derive(Debug)]
struct Current {
    /// Monotonic run identity. A guard only retires the run it was issued for, so a
    /// stale guard dropping late cannot cancel the run that superseded it.
    generation: u64,
    mode: RunMode,
    trigger: Trigger,
    phase: Phase,
    started: Instant,
    started_unix: i64,
    files_done: u64,
    files_total: u64,
    chunks_embedded: u64,
    chunks_total: u64,
    counters: RunCounters,
}

#[derive(Debug, Default)]
struct Inner {
    current: Option<Current>,
    last: Option<LastRun>,
    /// Incremented by every `begin`, so each run has a distinct identity.
    generation: u64,
    runs_total: u64,
    runs_failed: u64,
    last_success_unix: Option<i64>,
    payload_indexes: BTreeMap<String, PayloadIndexState>,
    strict_rejected_files: BTreeMap<String, StrictRejection>,
}

/// In-memory record of indexing activity for this process.
#[derive(Debug, Default)]
pub struct IndexStatus {
    inner: RwLock<Inner>,
}

impl IndexStatus {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run a closure against the inner state, recovering from a poisoned lock.
    ///
    /// A panic in one status update must never make the server permanently unable to
    /// report status — observability is the last thing that should take the process
    /// down with it.
    fn with<T>(&self, f: impl FnOnce(&mut Inner) -> T) -> T {
        let mut guard = self.inner.write().unwrap_or_else(|e| e.into_inner());
        f(&mut guard)
    }

    fn read<T>(&self, f: impl FnOnce(&Inner) -> T) -> T {
        let guard = self.inner.read().unwrap_or_else(|e| e.into_inner());
        f(&guard)
    }

    /// Mark a run as started, discarding any stale in-flight record.
    ///
    /// Returns a guard that retires the run when dropped. Holding the guard is what
    /// makes "in flight" honest: a run that panics or whose future is cancelled skips
    /// every explicit `finish` call in its path, and would otherwise stay marked
    /// in-flight until the process restarted — reporting "still working" for a run that
    /// died is worse than reporting nothing.
    #[must_use = "the returned guard retires the run when dropped; binding it to `_` \
                  ends the run immediately"]
    pub fn begin(&self, mode: RunMode, trigger: Trigger) -> RunGuard<'_> {
        let generation = self.with(|inner| {
            inner.generation = inner.generation.wrapping_add(1);
            let generation = inner.generation;
            inner.current = Some(Current {
                generation,
                mode,
                trigger,
                phase: Phase::Starting,
                started: Instant::now(),
                started_unix: unix_now(),
                files_done: 0,
                files_total: 0,
                chunks_embedded: 0,
                chunks_total: 0,
                counters: RunCounters::default(),
            });
            generation
        });

        RunGuard {
            status: self,
            generation,
            armed: true,
        }
    }

    pub fn set_phase(&self, phase: Phase) {
        self.with(|inner| {
            if let Some(cur) = inner.current.as_mut() {
                cur.phase = phase;
            }
        });
    }

    pub fn set_files_total(&self, total: u64) {
        self.with(|inner| {
            if let Some(cur) = inner.current.as_mut() {
                cur.files_total = total;
                cur.counters.discovered = total;
            }
        });
    }

    pub fn set_files_done(&self, done: u64) {
        self.with(|inner| {
            if let Some(cur) = inner.current.as_mut() {
                cur.files_done = done;
            }
        });
    }

    pub fn set_chunks_total(&self, total: u64) {
        self.with(|inner| {
            if let Some(cur) = inner.current.as_mut() {
                cur.chunks_total = total;
                cur.chunks_embedded = 0;
            }
        });
    }

    pub fn add_chunks_embedded(&self, n: u64) {
        self.with(|inner| {
            if let Some(cur) = inner.current.as_mut() {
                cur.chunks_embedded = cur.chunks_embedded.saturating_add(n);
            }
        });
    }

    /// Overwrite the in-flight counters with the run's authoritative tallies.
    pub fn set_counters(&self, counters: RunCounters) {
        self.with(|inner| {
            if let Some(cur) = inner.current.as_mut() {
                cur.counters = counters;
            }
        });
    }

    /// Retire the in-flight run into `last_run`, if it is still the run identified by
    /// `generation`.
    ///
    /// `error` is `None` for success. Reached only through [`RunGuard`]. Idempotent, and
    /// generation-checked: a guard whose run was already superseded by a later `begin`
    /// does nothing, so a late drop cannot retire someone else's run.
    fn finish_run(&self, generation: u64, error: Option<String>) {
        self.with(|inner| {
            if inner.current.as_ref().map(|c| c.generation) != Some(generation) {
                return;
            }
            let Some(cur) = inner.current.take() else {
                return;
            };
            let success = error.is_none();
            inner.runs_total = inner.runs_total.saturating_add(1);
            if !success {
                inner.runs_failed = inner.runs_failed.saturating_add(1);
            }
            let finished_unix = unix_now();
            if success {
                inner.last_success_unix = Some(finished_unix);
            }
            inner.last = Some(LastRun {
                mode: cur.mode,
                trigger: cur.trigger,
                success,
                error,
                started_unix: cur.started_unix,
                started_at: format_unix(cur.started_unix),
                finished_unix,
                finished_at: format_unix(finished_unix),
                duration_secs: cur.started.elapsed().as_secs_f64(),
                counters: cur.counters,
            });
        });
    }

    /// Record whether a Qdrant payload index is in place for `field`.
    pub fn record_payload_index(&self, field: &str, error: Option<String>) {
        self.with(|inner| {
            let state = match error {
                None => PayloadIndexState::Ok,
                Some(error) => PayloadIndexState::Failed { error },
            };
            inner.payload_indexes.insert(field.to_string(), state);
        });
    }

    /// Record (or clear) a strict-mode validation rejection for `path`.
    ///
    /// `Some(reason)` marks the path rejected — call this from `ingest::index_paths`
    /// exactly when `process_file` returns `FileOutcome::Rejected { reason }` for it.
    /// `None` clears it — call this when the same path later reaches
    /// `FileOutcome::Ready` (it validated cleanly this run) or is purged as an orphan
    /// (deleted from disk, so it can never resurface as dirty again). Mirrors
    /// [`Self::record_payload_index`]'s "record every outcome, including recovery" shape,
    /// with one difference: a healed path is removed from the map entirely rather than
    /// recorded as some `Ok` variant, because the map's cardinality IS the
    /// `kb_strict_rejected_files` gauge — a path with no current defect must not linger
    /// in it under any state, healthy or otherwise.
    pub fn record_strict_rejection(&self, path: &str, reason: Option<String>) {
        self.with(|inner| match reason {
            Some(reason) => {
                inner
                    .strict_rejected_files
                    .insert(path.to_string(), StrictRejection { reason });
            }
            None => {
                inner.strict_rejected_files.remove(path);
            }
        });
    }

    /// Whether the phrase-matching text index on the `text` payload field is known
    /// to be present and working, per the last `ensure_collection` outcome recorded
    /// via [`Self::record_payload_index`] for `"text"`.
    ///
    /// `false` until an outcome has been recorded at all (config disabled, an older
    /// Qdrant server rejected `phrase_matching`, or `ensure_collection` simply
    /// hasn't run yet) — the fail-safe default, since sending a phrase filter to a
    /// server that never confirmed support would error instead of gracefully
    /// degrading. A caller wanting "disabled for the process lifetime" semantics
    /// should read this once per request rather than caching it further; it only
    /// flips back to `true` if a later `ensure_collection` run succeeds, which is
    /// the same self-healing behavior every other payload index already has.
    pub fn phrase_matching_available(&self) -> bool {
        self.read(|inner| {
            matches!(
                inner.payload_indexes.get("text"),
                Some(PayloadIndexState::Ok)
            )
        })
    }

    pub fn snapshot(&self) -> StatusSnapshot {
        self.read(|inner| StatusSnapshot {
            indexing: inner.current.is_some(),
            current: inner.current.as_ref().map(|cur| RunProgress {
                mode: cur.mode,
                trigger: cur.trigger,
                phase: cur.phase,
                started_unix: cur.started_unix,
                started_at: format_unix(cur.started_unix),
                elapsed_secs: cur.started.elapsed().as_secs_f64(),
                files_done: cur.files_done,
                files_total: cur.files_total,
                chunks_embedded: cur.chunks_embedded,
                chunks_total: cur.chunks_total,
                counters: cur.counters.clone(),
            }),
            last_run: inner.last.clone(),
            runs_total: inner.runs_total,
            runs_failed: inner.runs_failed,
            last_success_unix: inner.last_success_unix,
            last_success_at: inner.last_success_unix.map(format_unix),
            payload_indexes: inner.payload_indexes.clone(),
            strict_rejected_files: inner.strict_rejected_files.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Retrieval-side operational metrics (#168)
// ---------------------------------------------------------------------------
//
// `/status`/`/metrics` are rich about indexing (everything above this point in the
// file) and had nothing at all about queries — the actual product, since this is a
// RAG server. This section is the recording surface for that: a query count, a
// zero-result count, rerank attempt/failure counts, and a per-stage latency
// histogram for embed/Qdrant/rerank. `render_prometheus` (server.rs, which owns the
// Prometheus encoder) is the consumer.
//
// This is deliberately NOT shaped like `IndexStatus` above, despite the issue
// asking for an "INDEX_STATUS-style" global: `record_payload_index` and friends
// run at most a few dozen times per process lifetime (once per field, inside
// `ensure_collection`), so a `RwLock<Inner>` write-lock per call is free. A query
// happens on EVERY search request — this is the genuine hot path retrieval.rs's
// `search` sits on — so recording a query outcome must never contend a lock
// against a concurrent request, and must not depend on (or block behind) anything
// `/status`'s 5s cache or single-flight collection does. Every field below is a
// bare atomic instead: `record_query` never blocks, and a reader takes a
// wait-free snapshot by loading each atomic once.
//
// #245 wired this into `retrieval.rs`: `search_paged` (which `search` wraps) calls
// `QUERY_METRICS.record_query(embed_ms as u64, search_ms as u64, rerank,
// results.len())` at every point it can produce a completed `Ok(SearchOutcome)` —
// not just the common path's final `debug!("search timing")` line, but also the
// `docs.is_empty()` early return (before a rerank is even attempted) and the
// reranker's own "response was entirely unusable, falling back to fused order"
// branch (an attempted-but-unusable rerank, recorded the same as the `Err(e)`
// arm: `rerank: Some((rerank_ms, false))`). Missing either of the latter two would
// undercount zero/low-result queries specifically whenever reranking is enabled —
// see that function's doc comment for the full accounting of its return points.
// `search_grouped` has no reranker at all, so it always passes `rerank: None`.

/// Upper bounds (inclusive), in milliseconds, for the buckets in every
/// [`LatencyHistogram`] below. This is exactly Prometheus's own client-library
/// default bucket set (`.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10` seconds)
/// scaled by 1000 so the hot-path recording side can stay in integer
/// milliseconds — `render_prometheus` converts back to seconds (the Prometheus
/// convention for a time unit) when it prints the `le` labels. Reusing the
/// well-known default, rather than hand-tuning per stage, is what makes these
/// three histograms directly comparable on one dashboard panel without a reader
/// having to first go look up what each one's buckets mean.
const LATENCY_BUCKETS_MS: [u64; 11] = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000];

/// A point-in-time read of a [`LatencyHistogram`], shaped exactly as
/// `render_prometheus` needs it: `bucket_counts[i]` is the CUMULATIVE count of
/// samples at-or-below `bucket_bounds_ms[i]` (Prometheus's `_bucket{le=".."}`
/// convention), and `bucket_counts` has one more entry than `bucket_bounds_ms` —
/// the trailing entry is the "+Inf" bucket, which always equals `count`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LatencyHistogramSnapshot {
    pub bucket_bounds_ms: Vec<u64>,
    pub bucket_counts: Vec<u64>,
    pub count: u64,
    pub sum_ms: u64,
}

/// A lock-free, Prometheus-shaped latency histogram: fixed buckets, atomic
/// counters, no registry. `record` touches exactly one bucket counter (the
/// smallest bound the sample fits under, or an implicit overflow bucket for
/// anything past the largest bound) plus `count`/`sum_ms` — three atomic
/// increments total, none of them a lock. The cumulative per-bound counts a
/// Prometheus histogram needs are computed in [`Self::snapshot`], at scrape
/// time, by prefix-summing the per-bucket counters — not maintained
/// incrementally, since that would mean every `record` touching every bucket
/// instead of just its own.
#[derive(Debug)]
struct LatencyHistogram {
    /// One counter per bound in [`LATENCY_BUCKETS_MS`], plus a trailing overflow
    /// ("+Inf") counter for samples past the largest bound. Non-cumulative: each
    /// slot counts only samples that landed in exactly that bucket.
    buckets: [AtomicU64; LATENCY_BUCKETS_MS.len() + 1],
    count: AtomicU64,
    sum_ms: AtomicU64,
}

impl LatencyHistogram {
    fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            count: AtomicU64::new(0),
            sum_ms: AtomicU64::new(0),
        }
    }

    // Called from `RetrievalMetrics::record_query` below, which `retrieval::search`
    // (and `search_grouped`) call on every completed query (#245) — plus this
    // module's own tests.
    fn record(&self, ms: u64) {
        let idx = LATENCY_BUCKETS_MS
            .iter()
            .position(|&bound| ms <= bound)
            .unwrap_or(LATENCY_BUCKETS_MS.len());
        self.buckets[idx].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_ms.fetch_add(ms, Ordering::Relaxed);
    }

    fn snapshot(&self) -> LatencyHistogramSnapshot {
        let mut bucket_counts = Vec::with_capacity(LATENCY_BUCKETS_MS.len() + 1);
        let mut running = 0u64;
        for slot in self.buckets.iter().take(LATENCY_BUCKETS_MS.len()) {
            running += slot.load(Ordering::Relaxed);
            bucket_counts.push(running);
        }
        // The overflow slot folds into the final "+Inf" entry, which is why this
        // total is expected to equal `count` below.
        running += self.buckets[LATENCY_BUCKETS_MS.len()].load(Ordering::Relaxed);
        bucket_counts.push(running);

        LatencyHistogramSnapshot {
            bucket_bounds_ms: LATENCY_BUCKETS_MS.to_vec(),
            bucket_counts,
            count: self.count.load(Ordering::Relaxed),
            sum_ms: self.sum_ms.load(Ordering::Relaxed),
        }
    }
}

/// A point-in-time read of [`RetrievalMetrics`]. See that type's doc comment for
/// what each counter means and why a rate (e.g. "zero-result rate") is reported
/// as two raw counters rather than a pre-divided fraction — same convention
/// `IndexStatus` already uses for `runs_total`/`runs_failed`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RetrievalMetricsSnapshot {
    pub queries_total: u64,
    pub zero_result_total: u64,
    pub rerank_attempted_total: u64,
    pub rerank_failed_total: u64,
    pub embed_latency_ms: LatencyHistogramSnapshot,
    pub qdrant_latency_ms: LatencyHistogramSnapshot,
    pub rerank_latency_ms: LatencyHistogramSnapshot,
}

/// Process-global retrieval (query-side) operational counters — the counterpart
/// to [`IndexStatus`] for the other half of this service. See this module's
/// "Retrieval-side operational metrics" section doc comment above for the full
/// rationale, including why this is built on bare atomics rather than
/// `IndexStatus`'s `RwLock<Inner>` shape.
#[derive(Debug)]
pub struct RetrievalMetrics {
    queries_total: AtomicU64,
    /// Count of queries in `record_query` calls where `result_count == 0` — the
    /// single best signal that retrieval is failing users, per #168.
    zero_result_total: AtomicU64,
    /// Queries where a reranker was actually invoked (`rerank: Some(..)` passed
    /// to `record_query`), regardless of outcome. Reranking is optional and off
    /// by default in the query path whenever `reranking:` is absent from
    /// `config.yaml` or a caller sends no candidates to rerank — those calls
    /// never touch this counter, so it only ever counts genuine attempts.
    rerank_attempted_total: AtomicU64,
    /// Subset of `rerank_attempted_total` where the reranker call failed and
    /// `retrieval::search` fell back to fused order. Reranking degrades silently
    /// by design (a warn! log and a fallback, never a request failure), so
    /// without this counter a reranker that has been down for a week produces no
    /// operator-visible signal at all.
    rerank_failed_total: AtomicU64,
    embed_latency_ms: LatencyHistogram,
    qdrant_latency_ms: LatencyHistogram,
    rerank_latency_ms: LatencyHistogram,
}

impl Default for RetrievalMetrics {
    fn default() -> Self {
        Self {
            queries_total: AtomicU64::new(0),
            zero_result_total: AtomicU64::new(0),
            rerank_attempted_total: AtomicU64::new(0),
            rerank_failed_total: AtomicU64::new(0),
            embed_latency_ms: LatencyHistogram::new(),
            qdrant_latency_ms: LatencyHistogram::new(),
            rerank_latency_ms: LatencyHistogram::new(),
        }
    }
}

impl RetrievalMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one completed retrieval call. Never blocks — see the module
    /// section doc comment on why that matters on this specific call path.
    ///
    /// `rerank` is `None` when no reranker was attempted for this call (no
    /// `RerankClient` configured, or the call returned zero candidates before a
    /// rerank was ever attempted) and `Some((rerank_ms, success))` whenever a
    /// rerank genuinely was attempted — pass `success: false` on a reranker
    /// error, not a skipped call, since that is the silent-degradation case this
    /// counter exists to surface.
    ///
    /// This is the recording half of #168's API. #245 wired it in:
    /// `retrieval::search_paged` (which `search` is a thin wrapper over) calls
    /// this at every point it can produce a completed `Ok(SearchOutcome)` —
    /// the common path at the bottom of the function, the `docs.is_empty()`
    /// early return, and the reranker's own "response was entirely unusable"
    /// fallback — so a query is counted exactly once regardless of which path
    /// it takes, including the zero-result paths that bypass the timing
    /// `debug!` this call sits next to. `search_grouped` calls it too, always
    /// with `rerank: None` (that path has no reranker).
    pub fn record_query(
        &self,
        embed_ms: u64,
        qdrant_ms: u64,
        rerank: Option<(u64, bool)>,
        result_count: usize,
    ) {
        self.queries_total.fetch_add(1, Ordering::Relaxed);
        if result_count == 0 {
            self.zero_result_total.fetch_add(1, Ordering::Relaxed);
        }
        self.embed_latency_ms.record(embed_ms);
        self.qdrant_latency_ms.record(qdrant_ms);
        if let Some((rerank_ms, success)) = rerank {
            self.rerank_attempted_total.fetch_add(1, Ordering::Relaxed);
            if !success {
                self.rerank_failed_total.fetch_add(1, Ordering::Relaxed);
            }
            self.rerank_latency_ms.record(rerank_ms);
        }
    }

    pub fn snapshot(&self) -> RetrievalMetricsSnapshot {
        RetrievalMetricsSnapshot {
            queries_total: self.queries_total.load(Ordering::Relaxed),
            zero_result_total: self.zero_result_total.load(Ordering::Relaxed),
            rerank_attempted_total: self.rerank_attempted_total.load(Ordering::Relaxed),
            rerank_failed_total: self.rerank_failed_total.load(Ordering::Relaxed),
            embed_latency_ms: self.embed_latency_ms.snapshot(),
            qdrant_latency_ms: self.qdrant_latency_ms.snapshot(),
            rerank_latency_ms: self.rerank_latency_ms.snapshot(),
        }
    }
}

/// Process-wide retrieval (query-side) metrics. See the section doc comment
/// above for why this is global and why it is safe to update from a hot path.
pub static QUERY_METRICS: LazyLock<RetrievalMetrics> = LazyLock::new(RetrievalMetrics::new);

/// Query-string keys whose values are scrubbed by [`redact_error`].
const SECRET_QUERY_KEYS: [&str; 6] = ["api-key", "api_key", "apikey", "key", "token", "password"];

/// Authorization scheme prefixes whose following token is scrubbed by [`redact_error`].
///
/// `key=value` matching misses the prose form an HTTP layer emits when it echoes a
/// request header back in an error.
const SECRET_AUTH_PREFIXES: [&str; 2] = ["bearer ", "basic "];

/// Scrub credentials out of an error message before it is served over HTTP.
///
/// Error text reaching `/status` comes from `anyhow`-wrapped Qdrant and sqlx failures,
/// and the Qdrant client renders its full connection URL on a transport error. There is
/// no separate `qdrant.api_key` setting, so the only way to authenticate against a
/// secured Qdrant is to embed the credential in `QDRANT_URL` — which put it one failed
/// connection away from the response body. Worse, a payload-index failure is recorded
/// once and served on every request for the life of the process, turning a momentary
/// blip into a standing leak.
///
/// [`crate::git::redact_url`] handles `scheme://token@host` userinfo, which is why git
/// errors were already safe; this additionally scrubs `?api-key=` style query
/// parameters, the form Qdrant Cloud uses.
pub fn redact_error(msg: &str) -> String {
    let mut out = crate::git::redact_url(msg);

    // Scrub `key=value` pairs wherever they appear, not only after a `?` — error text
    // wraps and reformats URLs, so anchoring on URL structure would miss cases.
    //
    // The search is ASCII-case-insensitive over the ORIGINAL bytes. Searching a
    // lowercased copy and reusing its offsets would be wrong, because lowercasing can
    // change a string's byte length: U+0130 'İ' (2 bytes) lowercases to two chars
    // (3 bytes), and U+212A 'K' (3 bytes) lowercases to 'k' (1 byte). One such
    // character before a credential shifts every later offset, so the redaction would
    // cut the wrong span and leave part of the secret behind. Every key is ASCII, and
    // an ASCII byte can never appear inside a multi-byte UTF-8 sequence, so a byte
    // match is always on a char boundary.
    let bytes = out.as_bytes().to_vec();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for key in SECRET_QUERY_KEYS {
        let needle = format!("{key}=");
        let needle = needle.as_bytes();
        if bytes.len() < needle.len() {
            continue;
        }
        for start in 0..=bytes.len() - needle.len() {
            if !bytes[start..start + needle.len()].eq_ignore_ascii_case(needle) {
                continue;
            }
            // Require a delimiter before the key so `monkey=` does not match `key=`.
            if start > 0 {
                let prev = bytes[start - 1];
                if prev.is_ascii_alphanumeric() || prev == b'_' || prev == b'-' {
                    continue;
                }
            }
            let value_start = start + needle.len();
            if value_start >= out.len() {
                continue;
            }
            let value_end = out[value_start..]
                .find(|c: char| c == '&' || c == '"' || c == '\'' || c.is_whitespace())
                .map(|i| value_start + i)
                .unwrap_or(out.len());
            if value_end > value_start {
                spans.push((value_start, value_end));
            }
        }
    }

    // Same ASCII-on-original-bytes scan for `Bearer <token>` / `Basic <token>`, which
    // no `key=value` pattern catches.
    for prefix in SECRET_AUTH_PREFIXES {
        let needle = prefix.as_bytes();
        if bytes.len() < needle.len() {
            continue;
        }
        for start in 0..=bytes.len() - needle.len() {
            if !bytes[start..start + needle.len()].eq_ignore_ascii_case(needle) {
                continue;
            }
            if start > 0 && bytes[start - 1].is_ascii_alphanumeric() {
                continue;
            }
            let value_start = start + needle.len();
            if value_start >= out.len() {
                continue;
            }
            let value_end = out[value_start..]
                .find(|c: char| c == '"' || c == '\'' || c == ',' || c.is_whitespace())
                .map(|i| value_start + i)
                .unwrap_or(out.len());
            if value_end > value_start {
                spans.push((value_start, value_end));
            }
        }
    }

    // Merge overlaps before rewriting. `password=key=x` matches twice with overlapping
    // spans; applying both would rewrite already-rewritten text.
    spans.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in spans {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }

    // Apply right-to-left so earlier offsets stay valid.
    for (start, end) in merged.into_iter().rev() {
        out.replace_range(start..end, "***");
    }

    out
}

/// Reason recorded when a run's guard drops without an explicit outcome.
pub const ABORTED_REASON: &str =
    "run ended without reporting an outcome (panicked, or its future was cancelled)";

/// Keeps a run marked in-flight for exactly as long as it is running.
///
/// Every path out of an indexing run — normal return, `?`, panic-unwind, future
/// cancellation — drops this guard, so the "indexing" flag cannot survive the work it
/// describes. `tokio::spawn` catches panics at the task boundary, which means a panicked
/// reindex leaves the process alive; without this guard it would also leave `/status`
/// claiming a run was still going, with `elapsed_secs` climbing forever.
#[must_use = "dropping the guard immediately ends the run it represents"]
pub struct RunGuard<'a> {
    status: &'a IndexStatus,
    generation: u64,
    armed: bool,
}

impl RunGuard<'_> {
    /// Retire the run with a known outcome. `None` means success.
    pub fn finish(mut self, error: Option<String>) {
        self.armed = false;
        self.status.finish_run(self.generation, error);
    }
}

impl Drop for RunGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.status
                .finish_run(self.generation, Some(ABORTED_REASON.to_string()));
        }
    }
}

/// Process-wide indexing status. See the module docs for why this is global.
pub static INDEX_STATUS: LazyLock<IndexStatus> = LazyLock::new(IndexStatus::new);

/// When this process started, for the `uptime` field.
///
/// Read once early in `main` so uptime counts from startup rather than from the first
/// `/status` request.
pub static PROCESS_START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Force [`PROCESS_START`] to initialize now.
pub fn init_process_start() {
    LazyLock::force(&PROCESS_START);
}

pub fn uptime_secs() -> f64 {
    PROCESS_START.elapsed().as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn begin_marks_indexing_and_finish_clears_it() {
        let s = IndexStatus::new();
        assert!(!s.snapshot().indexing);

        let run = s.begin(RunMode::Incremental, Trigger::Webhook);
        assert!(s.snapshot().indexing);
        let snap = s.snapshot();
        assert!(snap.indexing);
        let cur = snap.current.expect("current run");
        assert_eq!(cur.mode, RunMode::Incremental);
        assert_eq!(cur.trigger, Trigger::Webhook);
        assert_eq!(cur.phase, Phase::Starting);

        run.finish(None);
        assert!(!s.snapshot().indexing);
        let snap = s.snapshot();
        assert!(snap.current.is_none());
        assert_eq!(snap.runs_total, 1);
        assert_eq!(snap.runs_failed, 0);
        assert!(snap.last_run.expect("last run").success);
    }

    #[test]
    fn failed_run_records_error_and_increments_failure_count() {
        let s = IndexStatus::new();
        let run = s.begin(RunMode::Full, Trigger::Cli);
        run.finish(Some("qdrant unreachable".into()));

        let snap = s.snapshot();
        assert_eq!(snap.runs_total, 1);
        assert_eq!(snap.runs_failed, 1);
        let last = snap.last_run.expect("last run");
        assert!(!last.success);
        assert_eq!(last.error.as_deref(), Some("qdrant unreachable"));
        assert_eq!(last.mode, RunMode::Full);
    }

    #[test]
    fn phrase_matching_available_is_false_before_any_outcome_is_recorded() {
        // Fail-safe default: nothing has confirmed the phrase-matching text index
        // exists yet (config disabled, an older Qdrant server rejected it, or
        // `ensure_collection` just hasn't run), so a caller must not attempt a
        // phrase arm against a server that never confirmed support.
        let s = IndexStatus::new();
        assert!(!s.phrase_matching_available());
    }

    #[test]
    fn phrase_matching_available_true_after_a_successful_index_creation() {
        let s = IndexStatus::new();
        s.record_payload_index("text", None);
        assert!(s.phrase_matching_available());
    }

    #[test]
    fn phrase_matching_available_false_after_a_failed_index_creation() {
        // Models an older Qdrant server rejecting `phrase_matching` —
        // `ensure_collection` records the failure and must not fail startup over
        // it; this is what then keeps phrase matching disabled for the process.
        let s = IndexStatus::new();
        s.record_payload_index("text", Some("phrase_matching is not supported".into()));
        assert!(!s.phrase_matching_available());
    }

    #[test]
    fn phrase_matching_available_flips_back_on_a_later_success() {
        // Same self-healing behavior every other payload index already has: a
        // later successful `ensure_collection` run clears a previous failure
        // rather than looking permanently broken.
        let s = IndexStatus::new();
        s.record_payload_index("text", Some("old server".into()));
        assert!(!s.phrase_matching_available());
        s.record_payload_index("text", None);
        assert!(s.phrase_matching_available());
    }

    #[test]
    fn phrase_matching_available_is_unaffected_by_other_fields_failing() {
        let s = IndexStatus::new();
        s.record_payload_index("text", None);
        s.record_payload_index("tags", Some("wrong index type".into()));
        assert!(
            s.phrase_matching_available(),
            "an unrelated field's failure must not affect the text index's own status"
        );
    }

    #[test]
    fn progress_updates_are_visible_mid_run() {
        let s = IndexStatus::new();
        let _run = s.begin(RunMode::Incremental, Trigger::Startup);
        s.set_phase(Phase::Discovering);
        s.set_files_total(329);
        s.set_phase(Phase::Scanning);
        s.set_files_done(100);
        s.set_phase(Phase::Embedding);
        s.set_chunks_total(2400);
        s.add_chunks_embedded(64);
        s.add_chunks_embedded(64);

        let cur = s.snapshot().current.expect("current run");
        assert_eq!(cur.phase, Phase::Embedding);
        assert_eq!(cur.files_total, 329);
        assert_eq!(cur.files_done, 100);
        assert_eq!(cur.chunks_total, 2400);
        assert_eq!(cur.chunks_embedded, 128);
        // discovered is mirrored from files_total so a run that dies mid-flight still
        // reports how much work it had found.
        assert_eq!(cur.counters.discovered, 329);
    }

    #[test]
    fn set_chunks_total_resets_embedded_progress() {
        let s = IndexStatus::new();
        let _run = s.begin(RunMode::Incremental, Trigger::Cli);
        s.set_chunks_total(10);
        s.add_chunks_embedded(10);
        // A second embedding pass within the same run must not report 20/10.
        s.set_chunks_total(5);
        let cur = s.snapshot().current.expect("current run");
        assert_eq!(cur.chunks_embedded, 0);
        assert_eq!(cur.chunks_total, 5);
    }

    #[test]
    fn updates_without_an_active_run_are_ignored() {
        let s = IndexStatus::new();
        // These fire from the embed client, which also runs outside an indexing run.
        s.set_phase(Phase::Embedding);
        s.add_chunks_embedded(50);
        s.set_files_done(3);
        assert!(!s.snapshot().indexing);
        assert!(s.snapshot().current.is_none());
    }

    #[test]
    fn a_superseded_guard_cannot_retire_the_live_run() {
        let s = IndexStatus::new();
        let stale = s.begin(RunMode::Full, Trigger::Cli);
        let live = s.begin(RunMode::Incremental, Trigger::Webhook);

        // The stale guard reports its own run as aborted...
        stale.finish(Some("this should be ignored".into()));
        assert!(
            s.snapshot().indexing,
            "the live run must survive a superseded guard finishing"
        );
        assert_eq!(s.snapshot().runs_total, 0, "no run has actually retired");

        live.finish(None);
        let snap = s.snapshot();
        assert_eq!(snap.runs_total, 1);
        assert!(snap.last_run.expect("last run").success);
    }

    #[test]
    fn counters_survive_into_last_run() {
        let s = IndexStatus::new();
        let run = s.begin(RunMode::Incremental, Trigger::Webhook);
        s.set_counters(RunCounters {
            discovered: 329,
            indexed: 12,
            skipped: 317,
            invalid: 2,
            orphans_removed: 4,
            ..Default::default()
        });
        run.finish(None);

        let last = s.snapshot().last_run.expect("last run");
        assert_eq!(last.counters.discovered, 329);
        assert_eq!(last.counters.indexed, 12);
        assert_eq!(last.counters.skipped, 317);
        assert_eq!(last.counters.invalid, 2);
        assert_eq!(last.counters.orphans_removed, 4);
    }

    #[test]
    fn a_new_run_replaces_stale_in_flight_state() {
        let s = IndexStatus::new();
        let stale = s.begin(RunMode::Full, Trigger::Cli);
        s.set_files_total(10);
        // No finish() — simulate a run whose process-level bookkeeping was skipped.
        let _fresh = s.begin(RunMode::Incremental, Trigger::Webhook);
        // The superseded guard drops after its replacement already exists. Generation
        // checking is what stops its Drop from retiring the run that replaced it —
        // otherwise a late drop would report the live run as aborted.
        drop(stale);

        let cur = s.snapshot().current.expect("current run");
        assert_eq!(cur.mode, RunMode::Incremental);
        assert_eq!(cur.files_total, 0);
        assert!(s.snapshot().indexing);
    }

    #[test]
    fn payload_index_failures_are_recorded_and_overwritten() {
        let s = IndexStatus::new();
        s.record_payload_index("tags", None);
        s.record_payload_index("planning.effort", Some("timeout".into()));

        let snap = s.snapshot();
        assert_eq!(
            snap.payload_indexes.get("tags"),
            Some(&PayloadIndexState::Ok)
        );
        assert_eq!(
            snap.payload_indexes.get("planning.effort"),
            Some(&PayloadIndexState::Failed {
                error: "timeout".into()
            })
        );

        // A later successful run must clear the failure, not leave it stuck.
        s.record_payload_index("planning.effort", None);
        assert_eq!(
            s.snapshot().payload_indexes.get("planning.effort"),
            Some(&PayloadIndexState::Ok)
        );
    }

    #[test]
    fn counter_pairs_cover_every_field() {
        let c = RunCounters {
            discovered: 1,
            indexed: 2,
            skipped: 3,
            invalid: 4,
            empty: 5,
            read_errors: 6,
            metadata_backfilled: 7,
            frozen_by_broken_schema: 8,
            broken_schemas: 9,
            orphans_removed: 10,
            strict_rejected: 11,
        };
        let pairs = c.as_pairs();
        // Every field is distinct and non-zero above, so a missing or duplicated entry
        // in as_pairs() shows up as a sum mismatch.
        assert_eq!(pairs.iter().map(|(_, v)| v).sum::<u64>(), 66);
        let names: std::collections::BTreeSet<_> = pairs.iter().map(|(n, _)| *n).collect();
        assert_eq!(names.len(), 11);
    }

    #[test]
    fn strict_rejections_are_recorded_and_cleared_on_healing() {
        let s = IndexStatus::new();
        s.record_strict_rejection("bad.md", Some("missing 'description'".into()));
        s.record_strict_rejection("also-bad.md", Some("missing 'title'".into()));

        let snap = s.snapshot();
        assert_eq!(snap.strict_rejected_files.len(), 2);
        assert_eq!(
            snap.strict_rejected_files.get("bad.md"),
            Some(&StrictRejection {
                reason: "missing 'description'".into()
            })
        );

        // A later clean run (Ready) or an orphan purge clears the entry entirely —
        // unlike `record_payload_index`, there is no "ok" variant to flip to; a
        // healed path must not linger in the map under any state.
        s.record_strict_rejection("bad.md", None);
        let snap = s.snapshot();
        assert_eq!(snap.strict_rejected_files.len(), 1);
        assert!(!snap.strict_rejected_files.contains_key("bad.md"));
        assert!(snap.strict_rejected_files.contains_key("also-bad.md"));
    }

    #[test]
    fn dropping_the_guard_retires_the_run_as_aborted() {
        let s = IndexStatus::new();
        {
            let _run = s.begin(RunMode::Incremental, Trigger::Webhook);
            s.set_files_total(329);
            assert!(s.snapshot().indexing);
            // Guard dropped here without finish() — the shape of a panicked or
            // cancelled run. tokio::spawn catches the panic, so the process survives
            // and nothing else would ever clear the flag.
        }

        let snap = s.snapshot();
        assert!(
            !snap.indexing,
            "a run that died must not keep reporting as in flight"
        );
        let last = snap.last_run.expect("last run");
        assert!(!last.success);
        assert_eq!(last.error.as_deref(), Some(ABORTED_REASON));
        assert_eq!(snap.runs_failed, 1);
        // Work discovered before the abort is still reported.
        assert_eq!(last.counters.discovered, 329);
    }

    #[test]
    fn explicit_finish_wins_over_the_guard_drop() {
        let s = IndexStatus::new();
        {
            let run = s.begin(RunMode::Full, Trigger::Cli);
            run.finish(None);
        }
        // The guard's Drop must not turn a completed run into an aborted one, nor
        // double-count it.
        let snap = s.snapshot();
        assert_eq!(snap.runs_total, 1);
        assert_eq!(snap.runs_failed, 0);
        assert!(snap.last_run.expect("last run").success);
    }

    #[test]
    fn last_success_survives_a_later_failure() {
        let s = IndexStatus::new();
        s.begin(RunMode::Incremental, Trigger::Webhook).finish(None);
        let after_success = s.snapshot().last_success_unix.expect("success timestamp");

        s.begin(RunMode::Incremental, Trigger::Webhook)
            .finish(Some("embeddings unreachable".into()));

        let snap = s.snapshot();
        // The failed run becomes last_run, but the success timestamp must not be
        // clobbered — "when did this last actually work" is the alertable question.
        assert!(!snap.last_run.expect("last run").success);
        assert_eq!(snap.last_success_unix, Some(after_success));
        assert_eq!(snap.runs_total, 2);
        assert_eq!(snap.runs_failed, 1);
    }

    #[test]
    fn last_success_is_absent_until_a_run_succeeds() {
        let s = IndexStatus::new();
        assert!(s.snapshot().last_success_unix.is_none());
        s.begin(RunMode::Full, Trigger::Cli)
            .finish(Some("boom".into()));
        assert!(s.snapshot().last_success_unix.is_none());
    }

    #[test]
    fn redact_error_strips_url_userinfo() {
        let msg = "Failed to connect to http://apikey:secrettoken@127.0.0.1:6334/: transport error";
        let out = redact_error(msg);
        assert!(!out.contains("secrettoken"), "{out}");
        assert!(!out.contains("apikey:"), "{out}");
        assert!(out.contains("127.0.0.1:6334"), "host must survive: {out}");
    }

    #[test]
    fn redact_error_strips_api_key_query_parameters() {
        // The Qdrant Cloud auth pattern: there is no separate api_key config field, so
        // the credential can only live in QDRANT_URL.
        let msg = "error contacting https://xyz.cloud.qdrant.io:6334/?api-key=abc123def&foo=bar";
        let out = redact_error(msg);
        assert!(!out.contains("abc123def"), "{out}");
        assert!(out.contains("api-key=***"), "{out}");
        assert!(out.contains("foo=bar"), "unrelated params survive: {out}");
    }

    #[test]
    fn redact_error_covers_the_common_secret_key_spellings() {
        for key in ["api_key", "apikey", "token", "password", "key"] {
            let out = redact_error(&format!("boom {key}=s3cret trailing"));
            assert!(!out.contains("s3cret"), "{key} not redacted: {out}");
            assert!(out.contains("trailing"), "{key} over-consumed: {out}");
        }
    }

    #[test]
    fn redact_error_does_not_match_a_key_suffix() {
        // `monkey=` ends in `key=` but is not a credential.
        let out = redact_error("monkey=banana");
        assert_eq!(out, "monkey=banana");
    }

    #[test]
    fn redact_error_leaves_ordinary_messages_alone() {
        let msg = "no such table: documents";
        assert_eq!(redact_error(msg), msg);
    }

    #[test]
    fn redact_error_survives_length_changing_lowercase() {
        // U+212A KELVIN SIGN is 3 bytes and lowercases to 1; U+0130 is 2 bytes and
        // lowercases to 3. Searching a lowercased copy and reusing its byte offsets
        // would cut the wrong span here and leave part of the secret in the output.
        for prefix in ["\u{212A}", "\u{130}", "\u{1E9E}", "\u{212A}\u{130}"] {
            let out = redact_error(&format!("{prefix} token=sup3rs3cret rest"));
            assert!(
                !out.contains("sup3rs3cret"),
                "secret survived after prefix {prefix:?}: {out}"
            );
            assert!(out.contains("token=***"), "{out}");
            assert!(
                out.contains("rest"),
                "over-consumed after {prefix:?}: {out}"
            );
        }
    }

    #[test]
    fn redact_error_is_case_insensitive_on_the_key() {
        for key in ["TOKEN", "Api-Key", "PassWord"] {
            let out = redact_error(&format!("{key}=s3cret x"));
            assert!(!out.contains("s3cret"), "{key}: {out}");
        }
    }

    #[test]
    fn redact_error_strips_authorization_scheme_tokens() {
        // The prose form an HTTP layer emits when it echoes a request header back.
        let out = redact_error("request failed: Authorization: Bearer eyJhbGci.SECRET, retrying");
        assert!(!out.contains("eyJhbGci.SECRET"), "{out}");
        assert!(out.contains("Bearer ***"), "{out}");
        assert!(out.contains("retrying"), "over-consumed: {out}");

        let out = redact_error("Basic dXNlcjpwYXNz end");
        assert!(!out.contains("dXNlcjpwYXNz"), "{out}");
        assert!(out.contains("end"), "{out}");
    }

    #[test]
    fn redact_error_does_not_match_a_scheme_word_suffix() {
        // `Overbearing ` ends in neither prefix; `bearer` inside a longer word must not
        // trigger and eat the following text.
        let out = redact_error("unbearable latency observed");
        assert_eq!(out, "unbearable latency observed");
    }

    #[test]
    fn redact_error_merges_overlapping_matches() {
        // `password=` and `key=` both match, with overlapping spans. Rewriting both
        // independently would rewrite already-rewritten text.
        let out = redact_error("password=key=hunter2");
        assert!(!out.contains("hunter2"), "{out}");
        assert_eq!(out, "password=***");
    }

    #[test]
    fn redact_error_handles_multibyte_input() {
        // Byte offsets are computed on a lowercased copy; a value containing multi-byte
        // characters must not panic on a non-boundary slice.
        let out = redact_error("token=café&x=1 — naïve");
        assert!(!out.contains("café"), "{out}");
        assert!(out.contains("naïve"), "{out}");
    }

    #[test]
    fn format_unix_renders_rfc3339() {
        assert_eq!(format_unix(0), "1970-01-01T00:00:00Z");
        assert_eq!(format_unix(1_700_000_000), "2023-11-14T22:13:20Z");
    }

    // --- retrieval metrics (#168) ---------------------------------------------

    #[test]
    fn a_fresh_retrieval_metrics_reports_all_zeros() {
        let m = RetrievalMetrics::new();
        let snap = m.snapshot();
        assert_eq!(snap.queries_total, 0);
        assert_eq!(snap.zero_result_total, 0);
        assert_eq!(snap.rerank_attempted_total, 0);
        assert_eq!(snap.rerank_failed_total, 0);
        assert_eq!(snap.embed_latency_ms.count, 0);
        assert_eq!(snap.qdrant_latency_ms.count, 0);
        assert_eq!(snap.rerank_latency_ms.count, 0);
        // The "+Inf" trailing bucket is always present, even with zero samples.
        assert_eq!(
            snap.embed_latency_ms.bucket_counts.len(),
            snap.embed_latency_ms.bucket_bounds_ms.len() + 1
        );
    }

    #[test]
    fn record_query_counts_a_query_and_its_latencies() {
        let m = RetrievalMetrics::new();
        m.record_query(12, 8, None, 5);

        let snap = m.snapshot();
        assert_eq!(snap.queries_total, 1);
        assert_eq!(snap.zero_result_total, 0, "5 results is not zero-result");
        assert_eq!(snap.rerank_attempted_total, 0, "no reranker was passed");
        assert_eq!(snap.embed_latency_ms.count, 1);
        assert_eq!(snap.embed_latency_ms.sum_ms, 12);
        assert_eq!(snap.qdrant_latency_ms.count, 1);
        assert_eq!(snap.qdrant_latency_ms.sum_ms, 8);
        assert_eq!(
            snap.rerank_latency_ms.count, 0,
            "no rerank attempted, so its histogram must stay empty"
        );
    }

    #[test]
    fn record_query_with_zero_results_increments_the_zero_result_counter() {
        let m = RetrievalMetrics::new();
        m.record_query(10, 10, None, 0);
        assert_eq!(m.snapshot().zero_result_total, 1);

        // A later query with results must not retroactively "fix" the count —
        // it is a running total of how many queries came back empty, not a
        // current-state flag.
        m.record_query(10, 10, None, 3);
        let snap = m.snapshot();
        assert_eq!(snap.queries_total, 2);
        assert_eq!(snap.zero_result_total, 1);
    }

    #[test]
    fn record_query_tracks_a_successful_rerank_attempt() {
        let m = RetrievalMetrics::new();
        m.record_query(5, 5, Some((40, true)), 10);

        let snap = m.snapshot();
        assert_eq!(snap.rerank_attempted_total, 1);
        assert_eq!(
            snap.rerank_failed_total, 0,
            "a successful rerank must not count as a failure"
        );
        assert_eq!(snap.rerank_latency_ms.count, 1);
        assert_eq!(snap.rerank_latency_ms.sum_ms, 40);
    }

    #[test]
    fn record_query_tracks_a_failed_rerank_attempt() {
        // Models `retrieval::search`'s `Err(e)` arm on `reranker.rerank(..)`: it
        // still degrades to fused order rather than failing the request, which is
        // exactly the silent-failure mode #168 exists to surface — a failed
        // attempt still counts as an ATTEMPT (and still contributes a latency
        // sample: the reranker was reached and took time before it failed).
        let m = RetrievalMetrics::new();
        m.record_query(5, 5, Some((200, false)), 10);

        let snap = m.snapshot();
        assert_eq!(snap.rerank_attempted_total, 1);
        assert_eq!(snap.rerank_failed_total, 1);
        assert_eq!(snap.rerank_latency_ms.count, 1);
    }

    #[test]
    fn retrieval_metrics_snapshot_is_a_running_total_across_many_queries() {
        let m = RetrievalMetrics::new();
        for _ in 0..3 {
            m.record_query(10, 10, None, 1);
        }
        m.record_query(10, 10, Some((30, false)), 0);

        let snap = m.snapshot();
        assert_eq!(snap.queries_total, 4);
        assert_eq!(snap.zero_result_total, 1);
        assert_eq!(snap.rerank_attempted_total, 1);
        assert_eq!(snap.rerank_failed_total, 1);
        assert_eq!(snap.embed_latency_ms.count, 4);
    }

    #[test]
    fn latency_histogram_buckets_a_sample_into_the_smallest_fitting_bound() {
        let m = RetrievalMetrics::new();
        // Exactly on a bound: Prometheus buckets are `le` (inclusive), so a
        // 25ms sample belongs in the "25" bucket and every bucket above it, not
        // the next one up.
        m.record_query(25, 0, None, 1);
        let snap = m.snapshot();

        let idx_25 = snap
            .embed_latency_ms
            .bucket_bounds_ms
            .iter()
            .position(|&b| b == 25)
            .unwrap();
        let idx_10 = snap
            .embed_latency_ms
            .bucket_bounds_ms
            .iter()
            .position(|&b| b == 10)
            .unwrap();
        assert_eq!(
            snap.embed_latency_ms.bucket_counts[idx_25], 1,
            "a 25ms sample must land in the le=25 bucket"
        );
        assert_eq!(
            snap.embed_latency_ms.bucket_counts[idx_10], 0,
            "a 25ms sample must not count toward le=10"
        );
    }

    #[test]
    fn latency_histogram_buckets_are_cumulative() {
        let m = RetrievalMetrics::new();
        m.record_query(3, 0, None, 1); // falls in the first (le=5) bucket
        m.record_query(8, 0, None, 1); // falls in the le=10 bucket
        let snap = m.snapshot();

        // le=5 sees only the 3ms sample; le=10 and everything above must also
        // include it (cumulative), plus the 8ms sample.
        let at = |bound: u64| {
            let idx = snap
                .embed_latency_ms
                .bucket_bounds_ms
                .iter()
                .position(|&b| b == bound)
                .unwrap();
            snap.embed_latency_ms.bucket_counts[idx]
        };
        assert_eq!(at(5), 1);
        assert_eq!(at(10), 2);
        assert_eq!(at(10000), 2);
    }

    #[test]
    fn latency_histogram_overflow_bucket_catches_samples_past_the_largest_bound() {
        let m = RetrievalMetrics::new();
        m.record_query(999_999, 0, None, 1);
        let snap = m.snapshot();

        // Every explicit bound must show 0 — the sample is larger than all of
        // them — while the trailing "+Inf" entry (one past the last bound) must
        // still catch it, matching Prometheus's own overflow-bucket contract.
        for &count in
            &snap.embed_latency_ms.bucket_counts[..snap.embed_latency_ms.bucket_counts.len() - 1]
        {
            assert_eq!(count, 0);
        }
        assert_eq!(
            *snap.embed_latency_ms.bucket_counts.last().unwrap(),
            1,
            "an over-bound sample must still land in the +Inf bucket"
        );
        assert_eq!(snap.embed_latency_ms.count, 1);
    }

    #[test]
    fn the_process_global_query_metrics_starts_at_zero() {
        // A smoke test on the actual static, distinct from the `RetrievalMetrics::new()`
        // instances every other test above uses — those are deliberately private so
        // tests never observe another test's recordings through a shared global.
        // This only asserts the type is reachable and constructs cleanly; it does
        // NOT assert exact counts, since `cargo test` runs many tests in one process
        // and nothing else in this module writes to `QUERY_METRICS`, but a future
        // test that does would otherwise make this one order-dependent.
        let _ = QUERY_METRICS.snapshot();
    }
}
