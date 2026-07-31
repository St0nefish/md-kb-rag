//! Runtime observability for the indexing pipeline.
//!
//! `run_index` computes everything worth knowing about a run — how many files it saw,
//! how many it embedded, how many it refused — and used to log it and throw it away.
//! Answering "is an index running, and how did the last one go?" then meant `docker exec`
//! into the container and reading logs, which is the wrong shape for a service whose
//! whole job is keeping an index in sync.
//!
//! This module keeps that state in memory so `/status` and `/metrics` can answer in one
//! call. It is deliberately process-global, mirroring [`crate::webhook::REINDEX_LOCK`]:
//! indexing is already serialized per process by that lock, and threading a handle
//! through five call sites and every test mock would buy nothing.
//!
//! Nothing here is persisted. A restart resets the run history, which is correct — the
//! interesting question is almost always "what has this process done", and the durable
//! counts (documents, points) live in SQLite and Qdrant where they belong.

use std::collections::BTreeMap;
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
    /// The initial index after a fresh clone at server startup.
    Startup,
    /// A push notification from the knowledge base's git host.
    Webhook,
    /// A `create_document` / `edit_document` / `delete_document` MCP call.
    WriteTool,
}

impl Trigger {
    /// Every variant, so state-set metrics can emit a 0 for the inactive ones.
    pub const ALL: [Self; 4] = [Self::Cli, Self::Startup, Self::Webhook, Self::WriteTool];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cli => "cli",
            Self::Startup => "startup",
            Self::Webhook => "webhook",
            Self::WriteTool => "write_tool",
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
}

impl RunCounters {
    /// Name/value pairs for the Prometheus encoder, so adding a counter above cannot
    /// silently fail to appear in `/metrics`.
    pub fn as_pairs(&self) -> [(&'static str, u64); 10] {
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
        })
    }
}

/// Query-string keys whose values are scrubbed by [`redact_error`].
const SECRET_QUERY_KEYS: [&str; 6] = ["api-key", "api_key", "apikey", "key", "token", "password"];

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
    let lower = out.to_lowercase();
    let mut replacements: Vec<(usize, usize)> = Vec::new();
    for key in SECRET_QUERY_KEYS {
        let needle = format!("{key}=");
        let mut from = 0;
        while let Some(pos) = lower[from..].find(&needle) {
            let start = from + pos;
            // Require a delimiter before the key so `monkey=` does not match `key=`.
            let preceded_ok = start == 0
                || !lower.as_bytes()[start - 1].is_ascii_alphanumeric()
                    && lower.as_bytes()[start - 1] != b'_'
                    && lower.as_bytes()[start - 1] != b'-';
            let value_start = start + needle.len();
            if preceded_ok && value_start < out.len() {
                let value_end = out[value_start..]
                    .find(|c: char| c == '&' || c == '"' || c == '\'' || c.is_whitespace())
                    .map(|i| value_start + i)
                    .unwrap_or(out.len());
                if value_end > value_start {
                    replacements.push((value_start, value_end));
                }
            }
            from = value_start.min(lower.len());
        }
    }

    // Apply right-to-left so earlier offsets stay valid.
    replacements.sort_unstable_by_key(|(start, _)| std::cmp::Reverse(*start));
    for (start, end) in replacements {
        if out.is_char_boundary(start) && out.is_char_boundary(end) {
            out.replace_range(start..end, "***");
        }
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
        };
        let pairs = c.as_pairs();
        // Every field is distinct and non-zero above, so a missing or duplicated entry
        // in as_pairs() shows up as a sum mismatch.
        assert_eq!(pairs.iter().map(|(_, v)| v).sum::<u64>(), 55);
        let names: std::collections::BTreeSet<_> = pairs.iter().map(|(n, _)| *n).collect();
        assert_eq!(names.len(), 10);
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
}
