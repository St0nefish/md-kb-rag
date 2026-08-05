//! The dirty-path queue and its single background worker.
//!
//! Before this module, every producer of an index change — the MCP write tools, the
//! webhook handler, and (indirectly) the CLI — called `ingest::run_index` itself,
//! inline, and waited for it to finish. That coupled an MCP tool call's latency to
//! embedding time (measured: ~0.34s/chunk on a CPU-only embeddings service, sequential
//! batches of 32 — a large document blows past most MCP clients' 60s timeout), and it
//! meant two producers racing for the same reindex serialized on
//! `webhook::REINDEX_LOCK` with the loser's work silently DROPPED rather than queued,
//! which is precisely the bug this module exists to fix.
//!
//! The fix is the architecture in this crate's design doc:
//!
//! ```text
//! Producers                                         Worker (single task)
//!   write tools    -> paths they wrote              drain queue
//!   webhook        -> paths from the pull's diff    run scoped index
//!   startup        -> full reconcile                if dirtied during run -> loop
//!   reconcile timer -> full reconcile                else sleep on Notify
//! ```
//!
//! The reconcile timer's period is `indexing.reconcile_interval_secs` (default 600s —
//! see that field's doc comment for why it's minutes, not the original design sketch's
//! "60s", and why the interval barely matters for ordinary indexing latency at all).
//!
//! Every producer just marks paths (or a full reconcile) dirty and returns
//! immediately — `mark_paths`/`mark_full` never await and never touch the index. The
//! single worker task spawned by [`run_worker`] is the ONLY thing that ever calls into
//! `ingest::index_paths` (directly, or via `ingest::scan_and_index` for a full
//! reconcile), which is itself the only function that mutates Qdrant or the state DB —
//! see its doc comment for that invariant.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use tokio::sync::Notify;
use tracing::warn;

use crate::config::{ResolvedConfig, SharedConfig};
use crate::schema::SharedSchemaCache;
use crate::status::Trigger;

/// Dirty state accumulated since the worker's last drain, behind a plain
/// (synchronous) `Mutex`. Every critical section here is a couple of `HashSet`
/// operations — never an `.await` — so a std `Mutex` is correct and cheaper than a
/// tokio one; nothing here can deadlock the executor.
struct QueueState {
    paths: HashSet<PathBuf>,
    /// Whether a full reconcile — `ingest::scan_and_index` re-walking and re-comparing
    /// the whole corpus rather than trusting an explicit path list — is pending.
    ///
    /// This is NOT the same thing as `ingest::index_paths`'s own `force` flag (the
    /// destructive `md-kb-rag index --full` drop-and-rebuild). This one only means "run
    /// the scanner first" — used by the startup catch-up and the periodic safety-net
    /// sweep, both of which still index through the ordinary scoped, non-destructive
    /// path once the scan produces a worklist. Nothing in this module ever sets
    /// `index_paths`' `force`; only the CLI's `--full` flag does that, synchronously,
    /// outside the queue entirely.
    full: bool,
}

/// The dirty-path queue plus the `Notify` the worker sleeps on.
pub struct ReindexQueue {
    state: Mutex<QueueState>,
    notify: Notify,
}

impl ReindexQueue {
    fn new() -> Self {
        Self {
            state: Mutex::new(QueueState {
                paths: HashSet::new(),
                full: false,
            }),
            notify: Notify::new(),
        }
    }

    /// Mark `paths` dirty and wake the worker. Returns immediately — this is the
    /// entire contract producers get: after this call, the path WILL be reindexed
    /// (eventually, possibly coalesced with other work), and the caller does not wait
    /// for that to happen.
    pub fn mark_paths(&self, paths: impl IntoIterator<Item = PathBuf>) {
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.paths.extend(paths);
        }
        self.notify.notify_one();
    }

    /// Mark a full reconcile pending and wake the worker.
    pub fn mark_full(&self) {
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.full = true;
        }
        self.notify.notify_one();
    }

    /// Atomically take everything dirty and reset to empty/false.
    ///
    /// This is the coalesce-don't-drop mechanism: the worker calls `drain` again
    /// immediately after finishing a run, and if anything was marked WHILE that run
    /// was in flight, `drain` returns it non-empty and the worker runs again before
    /// going back to sleep. A path is lost only if it is never marked at all — never
    /// because it was marked at an "unlucky" moment.
    fn drain(&self) -> (HashSet<PathBuf>, bool) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let paths = std::mem::take(&mut state.paths);
        let full = std::mem::replace(&mut state.full, false);
        (paths, full)
    }

    /// Current pending-work summary, for `/status` and `/metrics`. Read-only; never
    /// touches the index.
    pub fn snapshot(&self) -> QueueSnapshot {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        QueueSnapshot {
            pending_paths: state.paths.len(),
            full_pending: state.full,
        }
    }
}

/// Pending-work summary for `/status` and `/metrics`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct QueueSnapshot {
    pub pending_paths: usize,
    pub full_pending: bool,
}

/// The process-wide queue. Global for the same reason `status::INDEX_STATUS` and the
/// old `webhook::REINDEX_LOCK` were: there is exactly one worker per process, so
/// threading a handle through every producer (three MCP tools, the webhook handler,
/// the startup/timer tasks) would buy nothing over a `LazyLock`.
pub static REINDEX_QUEUE: LazyLock<ReindexQueue> = LazyLock::new(ReindexQueue::new);

/// Mark `paths` dirty on the process-wide queue. See [`ReindexQueue::mark_paths`].
pub fn mark_paths(paths: impl IntoIterator<Item = PathBuf>) {
    REINDEX_QUEUE.mark_paths(paths);
}

/// Mark a full reconcile pending on the process-wide queue. See [`ReindexQueue::mark_full`].
pub fn mark_full() {
    REINDEX_QUEUE.mark_full();
}

// ---------------------------------------------------------------------------
// Retry policy
// ---------------------------------------------------------------------------

/// Cap on requeue attempts for one drained unit of work before giving up and letting
/// the periodic reconcile sweep (`indexing.reconcile_interval_secs`) catch it instead.
///
/// This exists only to stop a persistently-bad target (Qdrant down for hours, DNS
/// broken) from retrying forever; the reconcile sweep is the actual safety net, and
/// with its interval measured in minutes rather than seconds, biasing toward more
/// retries here (see [`is_permanent_failure`]) is cheap insurance against a real
/// transient outage outlasting a smaller cap.
const MAX_RETRY_ATTEMPTS: u32 = 6;
/// First retry's delay. Doubles each attempt after that, capped at
/// [`RETRY_MAX_BACKOFF`].
const RETRY_BASE_BACKOFF: Duration = Duration::from_secs(5);
/// Ceiling on a single retry's delay, so `MAX_RETRY_ATTEMPTS` bounds total wall-clock
/// wait to a few minutes rather than growing unboundedly.
const RETRY_MAX_BACKOFF: Duration = Duration::from_secs(300);

/// Exponential backoff for the given 1-indexed attempt number, capped at
/// [`RETRY_MAX_BACKOFF`].
fn backoff_for_attempt(attempt: u32) -> Duration {
    let exponent = attempt.saturating_sub(1).min(10); // headroom before any overflow
    RETRY_BASE_BACKOFF
        .saturating_mul(1u32 << exponent)
        .min(RETRY_MAX_BACKOFF)
}

/// Whether a failed run should be dropped (permanent) rather than requeued
/// (transient — the default).
///
/// The only failure `ingest::index_paths` can currently produce that retrying can
/// never fix is a `validation.strict` rejection: `ingest::process_file` bails with a
/// message containing the literal substring `"(strict mode)"` for exactly that case,
/// and nothing else on the indexing path produces that phrase. Everything else —
/// embeddings unreachable, Qdrant unreachable, git or state-DB I/O errors — is
/// environmental and often self-heals, so it is requeued with backoff by default.
///
/// This is a substring match on the rendered error rather than a typed error
/// distinction, and that is deliberate, not a shortcut taken under time pressure: the
/// direction of a MISCLASSIFICATION matters more than its likelihood. If this string
/// ever stops matching (a wording change in `ingest.rs`, a new permanent-failure kind
/// nobody taught this function about), the failure falls through to "transient" —
/// which just means it gets retried a bounded number of times and then dropped exactly
/// as it would have been anyway, not that it is lost. The reverse mistake — treating a
/// retryable infrastructure blip as permanent — would drop real work after one attempt
/// with nothing left to fall back on but the reconcile sweep. Given the choice, this
/// function is written to fail toward "retry too much" rather than "retry too little".
fn is_permanent_failure(err: &anyhow::Error) -> bool {
    format!("{err:#}").contains("(strict mode)")
}

// ---------------------------------------------------------------------------
// The worker
// ---------------------------------------------------------------------------

/// What one drained unit of work asks the runner to do: either index exactly `paths`,
/// or run a full reconcile (scan, then index whatever the scan found).
///
/// Exists so the drain/retry/coalesce loop below (`drain_and_run_with`) can be unit
/// tested against a fake runner instead of live Qdrant/embeddings — the real worker
/// (`run_worker`) supplies [`ingest_runner`], which is the only thing that talks to
/// `ingest::index_paths` / `ingest::scan_and_index` in production.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Unit {
    Paths(Vec<PathBuf>),
    FullReconcile,
}

/// Whether `unit` might have changed the schema tree, and therefore requires the
/// shared `SchemaCache` (see `schema::SharedSchemaCache`) to be rebuilt and swapped
/// in BEFORE this unit is indexed — not after.
///
/// Ordering matters: `ingest::index_paths`/`ingest::scan_and_index` always re-read
/// `.kb-schema.yaml` fresh off disk on every run (they build their own throwaway
/// `SchemaCache` internally, unrelated to this shared one), so indexing itself is
/// never stale. What WOULD go stale is every concurrent MCP call in the meantime:
/// `get_schema` describing the old rules, or a write validating against them, while
/// this unit is busy indexing documents under the new ones — a window that can last
/// as long as this unit takes to embed (seconds to minutes for a large change). So
/// the shared cache is rebuilt first, closing that window before it opens rather
/// than after.
///
/// A `FullReconcile` always rebuilds: it already means "something the queue's own
/// path-level tracking doesn't capture may have changed" (periodic sweep, startup
/// catch-up, or `write_raw_file` after ANY schema write, via `mark_full`), and
/// `scan_for_dirty` is about to do its own full walk regardless — a schema rebuild
/// alongside it is a rounding error on that cost, not worth the precision of trying
/// to detect "no, THIS particular full reconcile didn't touch a schema". A `Paths`
/// unit rebuilds only when one of the changed paths is literally a
/// `.kb-schema.yaml` — the case a webhook delivers when someone pushes straight to
/// the KB's git host without going through `update_schema` (which already rebuilds
/// synchronously on its own, before ever reaching this queue).
fn unit_touches_schema(unit: &Unit) -> bool {
    match unit {
        Unit::FullReconcile => true,
        Unit::Paths(paths) => paths.iter().any(|p| {
            p.file_name().and_then(|n| n.to_str()) == Some(crate::schema::SCHEMA_FILE_NAME)
        }),
    }
}

/// A boxed, owned future — the runner's return type. Owned (`Arc`/by-value
/// arguments) rather than borrowing `&ResolvedConfig`/`&Unit` specifically so this
/// type has no lifetime parameter: a borrowed version here forces every caller
/// (production and test alike) into a higher-ranked `Fn(&'a A, &'a B) -> Fut<'a>`
/// bound, which a capturing closure cannot satisfy — only a bare `fn` can — making the
/// fake-runner tests below impossible to write as closures. Cloning an `Arc` and a
/// small `Unit` once per drained batch is not worth fighting that for.
type RunFuture = std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send>>;

/// Same shape as [`RunFuture`], for the schema-rebuild step. No `Result`: a failed
/// rebuild (in practice only a panic inside the blocking walk — `SchemaCache::build`
/// has no fallible return) is logged and swallowed by the real implementation rather
/// than aborting the unit's indexing, for the same reason [`is_permanent_failure`]
/// biases toward retrying too much rather than too little — a stale cache for one
/// more cycle is a much smaller failure than skipping indexing entirely.
type RebuildFuture = std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>;

/// The real runner: calls into `ingest`, exactly as the module doc's diagram promises.
/// This is the ONLY place in the worker that talks to `ingest::index_paths` /
/// `ingest::scan_and_index`.
fn ingest_runner(config: Arc<ResolvedConfig>, unit: Unit) -> RunFuture {
    Box::pin(async move {
        match unit {
            Unit::FullReconcile => {
                crate::ingest::scan_and_index(&config, false, Trigger::Worker).await
            }
            Unit::Paths(paths) => {
                crate::ingest::index_paths(&config, &paths, false, Trigger::Worker).await
            }
        }
    })
}

/// The real schema-rebuild step: walk the tree (off the executor — this is blocking
/// filesystem work, same as every other `SchemaCache::build` call site) and swap the
/// result into `schema_cache`. This is the ONLY place the worker itself rebuilds the
/// shared cache; `update_schema` has its own synchronous rebuild, independent of this
/// one, for the reasons documented on that call site.
fn schema_rebuild_runner(
    config: Arc<ResolvedConfig>,
    schema_cache: SharedSchemaCache,
) -> RebuildFuture {
    Box::pin(async move {
        let data_path = config.canonical_data_path();
        let frontmatter = config.frontmatter.clone();
        match tokio::task::spawn_blocking(move || {
            crate::schema::SchemaCache::build(&data_path, &frontmatter)
        })
        .await
        {
            Ok(schemas) => crate::schema::store_shared(&schema_cache, schemas),
            Err(e) => warn!("Schema rebuild panicked in the reindex worker: {e}"),
        }
    })
}

/// Run the reindex worker forever: sleep until the queue has something, then drain and
/// run until it is empty again. Spawned once at server startup. The CLI never runs
/// this — `ingest::scan_and_index` is its synchronous, no-worker equivalent.
///
/// `schema_cache` is the same `SharedSchemaCache` the MCP handler reads from
/// (`KbSearchServer::schema_cache`) — this task is the other of the two places that
/// ever write to it, the first being `update_schema`'s own synchronous rebuild.
///
/// Takes the LIVE `SharedConfig` handle, not a one-off `Arc<ResolvedConfig>`: a
/// fresh snapshot is loaded before every drain (see the `load_shared_config` call
/// below), so `indexing.include`/`exclude`/`exclude_files`, `frontmatter.*`,
/// `validation.*`, and `chunking.*` all pick up a `POST /admin/reload` swap on the
/// worker's very next wake — no restart needed. This is what makes those settings
/// `reload::ReloadEffect::Applied` (or `ReindexRequired` for `chunking.*`, since the
/// effect only reaches documents indexed after the change) rather than
/// restart-required — see `reload.rs`'s classification table.
pub async fn run_worker(shared_config: SharedConfig, schema_cache: SharedSchemaCache) {
    // A closure rather than passing `schema_cache` straight into `drain_and_run_with`
    // so the rebuild step has the same test-injectable shape as `ingest_runner` (see
    // `RebuildFuture`'s doc comment) — production and tests both go through one
    // `Fn(Arc<ResolvedConfig>) -> RebuildFuture` parameter.
    let rebuild = move |config: Arc<ResolvedConfig>| -> RebuildFuture {
        schema_rebuild_runner(config, Arc::clone(&schema_cache))
    };
    loop {
        REINDEX_QUEUE.notify.notified().await;
        // Fresh snapshot per wake, not per process — see this function's doc comment.
        let config = crate::config::load_shared_config(&shared_config);
        drain_and_run_with(&REINDEX_QUEUE, &config, &ingest_runner, &rebuild).await;
    }
}

/// One "wake up, drain, run until empty" cycle, parameterized over the runner so tests
/// can exercise the coalesce/retry logic without live infrastructure.
///
/// Coalesce-don't-drop: each iteration re-drains the queue; if `mark_paths`/
/// `mark_full` landed while the previous unit was running, the next drain is non-empty
/// and the loop runs again immediately instead of returning to the caller (which, in
/// `run_worker`, means going back to sleep on `Notify` — exactly the bug being fixed,
/// where a webhook landing mid-reindex used to be dropped instead of picked back up).
async fn drain_and_run_with(
    queue: &ReindexQueue,
    config: &Arc<ResolvedConfig>,
    run: &(dyn Fn(Arc<ResolvedConfig>, Unit) -> RunFuture + Sync),
    rebuild_schema: &(dyn Fn(Arc<ResolvedConfig>) -> RebuildFuture + Sync),
) {
    loop {
        let (paths, full) = queue.drain();
        if paths.is_empty() && !full {
            return;
        }
        // `full` wins: a pending full reconcile already covers whatever the specific
        // paths would have done, so there is no reason to index them twice.
        let unit = if full {
            Unit::FullReconcile
        } else {
            Unit::Paths(paths.into_iter().collect())
        };
        // See `unit_touches_schema`'s doc comment for why this must happen BEFORE
        // `run_with_retry` below, not after or concurrently with it.
        if unit_touches_schema(&unit) {
            rebuild_schema(Arc::clone(config)).await;
        }
        run_with_retry(config, unit, run).await;
    }
}

async fn run_with_retry(
    config: &Arc<ResolvedConfig>,
    unit: Unit,
    run: &(dyn Fn(Arc<ResolvedConfig>, Unit) -> RunFuture + Sync),
) {
    let mut attempt = 0u32;
    loop {
        match run(Arc::clone(config), unit.clone()).await {
            Ok(()) => return,
            Err(e) if is_permanent_failure(&e) => {
                warn!(
                    ?unit,
                    "Indexing run failed with a non-retryable error; dropping it — the \
                     writer that caused this already saw the rejection: {:#}",
                    e
                );
                return;
            }
            Err(e) => {
                attempt += 1;
                if attempt > MAX_RETRY_ATTEMPTS {
                    warn!(
                        ?unit,
                        attempts = attempt - 1,
                        "Indexing run kept failing after {} attempt(s); giving up for \
                         now. The periodic reconcile sweep will retry: {:#}",
                        MAX_RETRY_ATTEMPTS,
                        e
                    );
                    return;
                }
                let backoff = backoff_for_attempt(attempt);
                warn!(
                    ?unit,
                    attempt,
                    ?backoff,
                    "Indexing run failed; treating as transient and retrying: {:#}",
                    e
                );
                tokio::time::sleep(backoff).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn path(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    // -- queue mechanics ------------------------------------------------------

    #[test]
    fn mark_paths_accumulates_and_drain_resets() {
        let q = ReindexQueue::new();
        q.mark_paths([path("a.md")]);
        q.mark_paths([path("b.md"), path("c.md")]);

        let snap = q.snapshot();
        assert_eq!(snap.pending_paths, 3);
        assert!(!snap.full_pending);

        let (drained, full) = q.drain();
        assert_eq!(drained.len(), 3);
        assert!(!full);
        assert!(drained.contains(&path("a.md")));

        // Draining resets state — nothing left to take.
        let (again, full_again) = q.drain();
        assert!(again.is_empty());
        assert!(!full_again);
    }

    #[test]
    fn mark_full_is_independent_of_paths() {
        let q = ReindexQueue::new();
        q.mark_paths([path("a.md")]);
        q.mark_full();

        let snap = q.snapshot();
        assert_eq!(snap.pending_paths, 1);
        assert!(snap.full_pending);

        let (paths, full) = q.drain();
        assert_eq!(paths.len(), 1);
        assert!(full);
    }

    #[test]
    fn marking_the_same_path_twice_does_not_duplicate() {
        let q = ReindexQueue::new();
        q.mark_paths([path("a.md")]);
        q.mark_paths([path("a.md")]);
        assert_eq!(q.snapshot().pending_paths, 1);
    }

    // -- retry policy -----------------------------------------------------------

    #[test]
    fn backoff_doubles_and_caps() {
        assert_eq!(backoff_for_attempt(1), Duration::from_secs(5));
        assert_eq!(backoff_for_attempt(2), Duration::from_secs(10));
        assert_eq!(backoff_for_attempt(3), Duration::from_secs(20));
        // Cap reached well before MAX_RETRY_ATTEMPTS, and never exceeded.
        assert_eq!(backoff_for_attempt(20), RETRY_MAX_BACKOFF);
    }

    #[test]
    fn strict_validation_failure_is_permanent() {
        let err =
            anyhow::anyhow!("Validation failed for 'x.md' (strict mode): [\"missing title\"]");
        assert!(is_permanent_failure(&err));
    }

    #[test]
    fn infrastructure_failures_are_transient() {
        for msg in [
            "Failed to embed chunk texts: connection refused",
            "Failed to connect to Qdrant: transport error",
            "git fetch timed out after 120s",
            "no such table: documents",
        ] {
            let err = anyhow::anyhow!("{msg}");
            assert!(
                !is_permanent_failure(&err),
                "expected transient, classified permanent: {msg}"
            );
        }
    }

    #[test]
    fn unrecognized_errors_default_to_transient() {
        // The documented bias: when the string match cannot confirm "permanent", the
        // classification must fall through to "transient", never the other way.
        let err = anyhow::anyhow!("some completely novel failure mode nobody anticipated");
        assert!(!is_permanent_failure(&err));
    }

    // -- coalesce-don't-drop, against a fake runner ------------------------------

    /// A fake runner that records every unit it was called with, in order, and pops
    /// its behavior for that call off a prepared script. Lets tests assert both "how
    /// many times did indexing actually run" and "did it see the coalesced work".
    struct FakeRunner {
        calls: StdMutex<Vec<Unit>>,
        /// One outcome consumed per call; `Ok` results wrapped for `anyhow::Result`,
        /// with a side-effect closure for simulating "a write landed mid-run".
        script: StdMutex<Vec<ScriptedOutcome>>,
    }

    /// One scripted call's behavior for [`FakeRunner`]: a closure so a test can run a
    /// side effect (e.g. `queue.mark_paths(...)`, simulating a write landing mid-run)
    /// before producing the call's result.
    type ScriptedOutcome = Box<dyn FnOnce() -> anyhow::Result<()> + Send>;

    impl FakeRunner {
        fn new(script: Vec<ScriptedOutcome>) -> Self {
            Self {
                calls: StdMutex::new(Vec::new()),
                script: StdMutex::new(script),
            }
        }

        /// Synchronous on purpose: everything this does (mutex locks, popping the
        /// script) is non-blocking std-mutex work, so callers can invoke it from
        /// inside a plain closure and box the (already-resolved) result into a future
        /// themselves — see `boxed_runner`.
        fn run_sync(&self, unit: &Unit) -> anyhow::Result<()> {
            self.calls.lock().unwrap().push(unit.clone());
            let next = self.script.lock().unwrap().remove(0);
            next()
        }

        fn call_count(&self) -> usize {
            self.calls.lock().unwrap().len()
        }
    }

    fn test_config() -> Arc<ResolvedConfig> {
        // A minimal config; `drain_and_run_with`'s fake runner never actually reads it.
        crate::mcp::make_test_resolved_config(std::path::Path::new("/tmp"))
    }

    /// Wrap a `Fn(&Unit) -> anyhow::Result<()>`-shaped closure into the boxed-future
    /// runner signature `drain_and_run_with` expects, so each test can write its fake
    /// behavior as plain synchronous logic.
    fn boxed_runner(
        f: impl Fn(&Unit) -> anyhow::Result<()> + Send + Sync + 'static,
    ) -> impl Fn(Arc<ResolvedConfig>, Unit) -> RunFuture {
        move |_cfg, unit| {
            let result = f(&unit);
            Box::pin(async move { result })
        }
    }

    /// A rebuild step that does nothing — for tests exercising coalesce/retry
    /// mechanics that have no opinion on schema handling.
    fn noop_rebuild() -> impl Fn(Arc<ResolvedConfig>) -> RebuildFuture {
        |_cfg| Box::pin(async {})
    }

    #[tokio::test]
    async fn a_write_that_lands_mid_run_causes_exactly_one_follow_up_run() {
        let queue = ReindexQueue::new();
        queue.mark_paths([path("a.md")]);

        // `Arc` so the closure below can reach back into the queue while a "run" for
        // the first unit is still notionally in progress.
        let queue = Arc::new(queue);
        let queue_for_script = Arc::clone(&queue);

        let runner = Arc::new(FakeRunner::new(vec![
            // First call (processing "a.md"): simulate a write landing WHILE this run
            // is in flight, by marking a new path before returning success. This is
            // exactly what `mark_paths` being called concurrently from an MCP tool
            // would do.
            Box::new(move || {
                queue_for_script.mark_paths([path("b.md")]);
                Ok(())
            }),
            // Second call (processing the coalesced "b.md"): succeeds, nothing new
            // marked, so the loop must stop here.
            Box::new(|| Ok(())),
        ]));

        let config = test_config();
        let runner_for_closure = Arc::clone(&runner);
        let run_fn = boxed_runner(move |unit| runner_for_closure.run_sync(unit));
        drain_and_run_with(&queue, &config, &run_fn, &noop_rebuild()).await;

        assert_eq!(
            runner.call_count(),
            2,
            "exactly one follow-up run for the work marked mid-run — not zero (dropped) \
             and not more (no infinite coalescing loop once the queue is quiet)"
        );
        let calls = runner.calls.lock().unwrap();
        assert_eq!(calls[0], Unit::Paths(vec![path("a.md")]));
        assert_eq!(calls[1], Unit::Paths(vec![path("b.md")]));
        drop(calls);

        // The queue must be empty afterward — nothing left unindexed.
        let snap = queue.snapshot();
        assert_eq!(snap.pending_paths, 0);
        assert!(!snap.full_pending);
    }

    #[tokio::test]
    async fn a_full_reconcile_pending_alongside_paths_only_runs_the_reconcile() {
        let queue = ReindexQueue::new();
        queue.mark_paths([path("a.md")]);
        queue.mark_full();

        let runner = Arc::new(FakeRunner::new(vec![Box::new(|| Ok(()))]));
        let config = test_config();
        let runner_for_closure = Arc::clone(&runner);
        let run_fn = boxed_runner(move |unit| runner_for_closure.run_sync(unit));
        drain_and_run_with(&queue, &config, &run_fn, &noop_rebuild()).await;

        assert_eq!(runner.call_count(), 1);
        assert_eq!(runner.calls.lock().unwrap()[0], Unit::FullReconcile);
    }

    #[tokio::test]
    async fn permanent_failure_is_dropped_without_retry() {
        let queue = ReindexQueue::new();
        queue.mark_paths([path("bad.md")]);

        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_for_closure = Arc::clone(&attempts);
        let config = test_config();

        let run_fn = boxed_runner(move |_unit| {
            attempts_for_closure.fetch_add(1, Ordering::SeqCst);
            Err(anyhow::anyhow!(
                "Validation failed for 'bad.md' (strict mode): [\"bad\"]"
            ))
        });
        drain_and_run_with(&queue, &config, &run_fn, &noop_rebuild()).await;

        assert_eq!(
            attempts.load(Ordering::SeqCst),
            1,
            "a permanent failure must be dropped on the first attempt, not retried"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn transient_failure_is_retried_up_to_the_cap_then_dropped() {
        let queue = ReindexQueue::new();
        queue.mark_paths([path("flaky.md")]);

        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_for_closure = Arc::clone(&attempts);
        let config = test_config();

        // Every attempt fails transiently — proves the loop terminates rather than
        // retrying forever, and that it stops at exactly the documented cap.
        let run_fn = boxed_runner(move |_unit| {
            attempts_for_closure.fetch_add(1, Ordering::SeqCst);
            Err(anyhow::anyhow!("embeddings service unreachable"))
        });
        drain_and_run_with(&queue, &config, &run_fn, &noop_rebuild()).await;

        assert_eq!(
            attempts.load(Ordering::SeqCst),
            MAX_RETRY_ATTEMPTS + 1,
            "one initial attempt plus MAX_RETRY_ATTEMPTS retries, then give up"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn transient_failure_that_later_succeeds_stops_retrying() {
        let queue = ReindexQueue::new();
        queue.mark_paths([path("recovers.md")]);

        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_for_closure = Arc::clone(&attempts);
        let config = test_config();

        let run_fn = boxed_runner(move |_unit| {
            let n = attempts_for_closure.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                Err(anyhow::anyhow!("qdrant unreachable"))
            } else {
                Ok(())
            }
        });
        drain_and_run_with(&queue, &config, &run_fn, &noop_rebuild()).await;

        assert_eq!(
            attempts.load(Ordering::SeqCst),
            2,
            "retried once, then succeeded"
        );
    }

    #[tokio::test]
    async fn an_empty_drain_never_invokes_the_runner() {
        let queue = ReindexQueue::new();
        let config = test_config();
        let calls = Arc::new(AtomicU32::new(0));
        let calls_for_closure = Arc::clone(&calls);

        let run_fn = boxed_runner(move |_unit| {
            calls_for_closure.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });
        drain_and_run_with(&queue, &config, &run_fn, &noop_rebuild()).await;

        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    // -- schema-cache rebuild gating and ordering --------------------------------

    #[test]
    fn full_reconcile_always_touches_schema() {
        assert!(unit_touches_schema(&Unit::FullReconcile));
    }

    #[test]
    fn paths_without_a_schema_file_do_not_touch_schema() {
        let unit = Unit::Paths(vec![path("notes/a.md"), path("food/recipes/b.md")]);
        assert!(!unit_touches_schema(&unit));
    }

    #[test]
    fn a_schema_file_among_the_paths_touches_schema() {
        let unit = Unit::Paths(vec![path("notes/a.md"), path("food/.kb-schema.yaml")]);
        assert!(unit_touches_schema(&unit));
    }

    /// The ordering this whole feature exists for: a dirtied `.kb-schema.yaml` must
    /// cause the shared cache to be rebuilt BEFORE that unit is handed to the
    /// indexer, never after and never concurrently with it. A shared timeline
    /// (rather than two separate counters) is what actually proves the ORDER, not
    /// just that both steps happened.
    #[tokio::test]
    async fn a_dirtied_schema_file_rebuilds_the_cache_before_indexing_it() {
        let queue = ReindexQueue::new();
        queue.mark_paths([path("food/.kb-schema.yaml")]);

        let timeline: Arc<StdMutex<Vec<&'static str>>> = Arc::new(StdMutex::new(Vec::new()));

        let timeline_for_rebuild = Arc::clone(&timeline);
        let rebuild = move |_cfg: Arc<ResolvedConfig>| -> RebuildFuture {
            let timeline = Arc::clone(&timeline_for_rebuild);
            Box::pin(async move {
                timeline.lock().unwrap().push("rebuild");
            })
        };

        let timeline_for_run = Arc::clone(&timeline);
        let run_fn = move |_cfg: Arc<ResolvedConfig>, _unit: Unit| -> RunFuture {
            let timeline = Arc::clone(&timeline_for_run);
            Box::pin(async move {
                timeline.lock().unwrap().push("run");
                Ok(())
            })
        };

        let config = test_config();
        drain_and_run_with(&queue, &config, &run_fn, &rebuild).await;

        assert_eq!(
            *timeline.lock().unwrap(),
            vec!["rebuild", "run"],
            "the schema cache must be rebuilt BEFORE the dirtied schema file is \
             indexed, never after"
        );
    }

    #[tokio::test]
    async fn a_dirtied_path_with_no_schema_file_never_triggers_a_rebuild() {
        let queue = ReindexQueue::new();
        queue.mark_paths([path("notes/a.md")]);

        let rebuild_calls = Arc::new(AtomicU32::new(0));
        let rebuild_calls_for_closure = Arc::clone(&rebuild_calls);
        let rebuild = move |_cfg: Arc<ResolvedConfig>| -> RebuildFuture {
            rebuild_calls_for_closure.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {})
        };

        let run_fn = boxed_runner(|_unit| Ok(()));
        let config = test_config();
        drain_and_run_with(&queue, &config, &run_fn, &rebuild).await;

        assert_eq!(
            rebuild_calls.load(Ordering::SeqCst),
            0,
            "an ordinary document change must not pay for a schema-tree rebuild"
        );
    }

    #[tokio::test]
    async fn a_full_reconcile_always_rebuilds_the_schema_cache() {
        let queue = ReindexQueue::new();
        queue.mark_full();

        let rebuild_calls = Arc::new(AtomicU32::new(0));
        let rebuild_calls_for_closure = Arc::clone(&rebuild_calls);
        let rebuild = move |_cfg: Arc<ResolvedConfig>| -> RebuildFuture {
            rebuild_calls_for_closure.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {})
        };

        let run_fn = boxed_runner(|_unit| Ok(()));
        let config = test_config();
        drain_and_run_with(&queue, &config, &run_fn, &rebuild).await;

        assert_eq!(
            rebuild_calls.load(Ordering::SeqCst),
            1,
            "a full reconcile cannot cheaply prove it didn't touch a schema, so it \
             always rebuilds"
        );
    }
}
