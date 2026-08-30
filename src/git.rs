use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::process::Command;
use tokio::sync::{Mutex, MutexGuard};
use tokio::time::timeout;
use tracing::{error, info, warn};

/// Maximum time to wait for a git subprocess (clone) before treating it as
/// hung and returning an error.
pub(crate) const GIT_TIMEOUT: Duration = Duration::from_secs(120);

/// Serializes every git invocation against the knowledge-base clone.
///
/// A git working copy is not safe for concurrent mutation: `add`, `commit`,
/// `merge` and `rebase` all take `.git/index.lock`, and whichever process loses
/// the race fails outright with `Unable to create '.git/index.lock': File
/// exists`. Worse than the failure is what it leaves behind — a half-staged
/// index whose own rollback can lose the same race, wedging the clone so that
/// every later write commits locally but can never rebase (`cannot rebase: You
/// have unstaged changes`) and so never syncs again.
///
/// Two independent producers reach this clone while the server is running: the
/// write tools (`write.rs` → [`commit_and_sync`]) and the webhook handler
/// (`webhook.rs`, fetch + ff-only merge). They overlap routinely rather than
/// exceptionally, because every write pushes to the KB's git host, which fires
/// a webhook straight back at us seconds later.
///
/// Until #92 this was covered incidentally by `webhook::REINDEX_LOCK`; that lock
/// is gone, and `reindex::ReindexQueue` which replaced it serializes *indexing*,
/// not *git*. Hence an explicit lock whose only job is git.
static GIT_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

/// Proof that the holder has exclusive access to the knowledge-base clone.
///
/// Every function in this module that shells out to git in `data_path` demands
/// one by reference, so "did I take the lock?" is answered by the type checker
/// rather than by review. Because the guard is passed as a `&` argument and
/// never re-acquired internally, a call chain cannot deadlock itself on this
/// non-reentrant mutex: there is exactly one acquisition per sequence, at the
/// top.
///
/// Hold it across a whole logical operation, not per command. In particular a
/// failed write and its rollback must run under a *single* acquisition —
/// releasing in between is what lets another writer observe, and then commit,
/// a half-staged index.
#[must_use = "the git lock is released as soon as this guard is dropped"]
pub struct GitLock(#[allow(dead_code)] MutexGuard<'static, ()>);

/// Acquire exclusive access to the knowledge-base clone.
///
/// Deliberately unbounded: the operations it guards are individually capped at
/// [`GIT_TIMEOUT`], so the wait is bounded in practice by the holder's own
/// timeouts, and a caller that gave up early would just resume racing.
pub async fn lock_git() -> GitLock {
    GitLock(GIT_LOCK.lock().await)
}

/// Inject a token into an HTTPS URL for authenticated git operations.
/// SSH URLs are returned unchanged.
pub fn inject_token_into_url(url: &str, token: &str) -> String {
    if let Some(rest) = url.strip_prefix("https://") {
        format!("https://{}@{}", token, rest)
    } else if let Some(rest) = url.strip_prefix("http://") {
        format!("http://{}@{}", token, rest)
    } else {
        // SSH or other scheme — pass through unchanged
        url.to_string()
    }
}

/// Redact tokens embedded in URLs (e.g. `https://token@host/path` → `https://***@host/path`).
/// Handles URLs embedded in larger strings (like git stderr output).
pub fn redact_url(s: &str) -> String {
    let mut result = s.to_string();
    for prefix in &["https://", "http://"] {
        let mut search_from = 0;
        while let Some(start) = result[search_from..].find(prefix) {
            let abs_start = search_from + start;
            let after_scheme = abs_start + prefix.len();
            let rest = &result[after_scheme..];
            if let Some(at_pos) = rest.find('@') {
                // Check there's no '/' before the '@' — the token is between scheme and @
                let before_at = &rest[..at_pos];
                if !before_at.contains('/') && !before_at.is_empty() {
                    result = format!(
                        "{}***{}",
                        &result[..after_scheme],
                        &result[after_scheme + at_pos..]
                    );
                    // Advance past the redacted portion
                    search_from = after_scheme + 3; // len("***")
                    continue;
                }
            }
            search_from = after_scheme;
        }
    }
    result
}

/// Ensure a git repository exists at `data_path`. If the path is not already a
/// git repo and `git_url` is provided, performs a full single-branch clone.
///
/// Returns `Ok(true)` if a fresh clone was performed, `Ok(false)` if the repo
/// already existed.
pub async fn ensure_repo(
    _lock: &GitLock,
    git_url: &str,
    branch: &str,
    data_path: &str,
    token: Option<&str>,
) -> Result<bool> {
    // Already a git repo — nothing to do
    if std::path::Path::new(data_path).join(".git").exists() {
        return Ok(false);
    }

    tokio::fs::create_dir_all(data_path)
        .await
        .with_context(|| format!("Failed to create data directory: {}", data_path))?;

    let clone_url = match token {
        Some(t) if !t.is_empty() => inject_token_into_url(git_url, t),
        _ => git_url.to_string(),
    };

    info!(
        "Cloning {} (branch: {}) into {}",
        redact_url(&clone_url),
        branch,
        data_path
    );

    // Full clone (no --depth): commit_and_sync needs history to fetch/rebase/push.
    let output = timeout(
        GIT_TIMEOUT,
        Command::new("git")
            .args([
                "clone",
                "--branch",
                branch,
                "--single-branch",
                &clone_url,
                ".",
            ])
            .current_dir(data_path)
            .output(),
    )
    .await
    .map_err(|_elapsed| {
        error!("git clone timed out after {:?}", GIT_TIMEOUT);
        anyhow::anyhow!("git clone timed out after {:?}", GIT_TIMEOUT)
    })?
    .context("Failed to run git clone")?;

    if !output.status.success() {
        let stderr = redact_url(&String::from_utf8_lossy(&output.stderr));
        anyhow::bail!("git clone failed: {}", stderr);
    }

    info!("Clone complete");
    Ok(true)
}

/// Outcome of [`commit_and_sync`]: the resulting commit SHA, plus any paths pulled in
/// by the rebase from commits other than the one this call just made.
///
/// `rebased_paths` comes from diffing `OLD..NEW`, where `OLD` is HEAD right after our
/// own local commit (captured before the fetch) and `NEW` is HEAD after the rebase
/// completes. That range necessarily excludes `rel_path` itself — the caller already
/// knows about that one — and captures exactly the files touched by whatever the
/// rebase pulled in from the remote, which also need reindexing. Empty when there was
/// no remote to rebase onto (`git_url` is `None`) or the rebase was a no-op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitOutcome {
    pub sha: String,
    pub rebased_paths: Vec<std::path::PathBuf>,
}

/// Run `git rev-parse HEAD` in `data_path` and return the trimmed SHA.
///
/// `pub(crate)` so `webhook.rs` can compute its own before/after range around the
/// fetch + ff-only merge it performs, the same way `commit_and_sync` does around its
/// own fetch + rebase.
pub(crate) async fn rev_parse_head(_lock: &GitLock, data_path: &str) -> anyhow::Result<String> {
    let out = timeout(
        GIT_TIMEOUT,
        Command::new("git")
            .args([
                "-c",
                &format!("safe.directory={}", data_path),
                "rev-parse",
                "HEAD",
            ])
            .current_dir(data_path)
            .output(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("git rev-parse timed out after {:?}", GIT_TIMEOUT))?
    .context("Failed to spawn git rev-parse")?;

    if !out.status.success() {
        let stderr = redact_url(&String::from_utf8_lossy(&out.stderr));
        anyhow::bail!("git rev-parse HEAD failed: {}", stderr);
    }

    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Parse `git diff --name-status -z` output into the set of paths it touched.
///
/// With `-z`, git NUL-delimits every field of the output — the status token AND
/// each path — instead of separating status from path with a tab and one record
/// from the next with a newline. That is what this function relies on: `-z` is
/// also the one invocation shape where git does NOT C-quote/octal-escape a path
/// containing non-ASCII bytes, tabs, backslashes, or quotes (`core.quotepath`'s
/// effect and the newline-terminated `\t`-field format go together). Without
/// `-z`, a file named `café.md` comes back as the literal 15-character string
/// `"caf\303\251.md"` — quotes, backslashes, and all — which matches no real
/// path on disk; `ReindexQueue::mark_paths` would then dirty a path that does
/// not exist while the real file silently never gets reindexed (#143). Splitting
/// the flat `-z`-delimited token stream below is what receives the raw bytes
/// git actually wrote to the path, unmangled.
///
/// Handles A(dded)/M(odified)/D(eleted) as a single path, and R(enamed)/C(opied) — which
/// carry a similarity score suffix like `R100` and two NUL-separated paths — as BOTH the
/// old and new path, since both need reindexing (old: purge; new: index).
fn parse_diff_name_status(output: &str) -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();
    // Trailing (and any stray consecutive) NULs produce empty tokens; skip them
    // rather than trying to parse a status out of "".
    let mut fields = output.split('\0').filter(|s| !s.is_empty());
    while let Some(status) = fields.next() {
        match status.chars().next() {
            Some('A') | Some('M') | Some('D') => {
                if let Some(path) = fields.next() {
                    paths.push(std::path::PathBuf::from(path));
                }
            }
            Some('R') | Some('C') => {
                if let (Some(old), Some(new)) = (fields.next(), fields.next()) {
                    paths.push(std::path::PathBuf::from(old));
                    paths.push(std::path::PathBuf::from(new));
                }
            }
            _ => {
                warn!(
                    "Unrecognized 'git diff --name-status -z' status token, ignoring: {}",
                    status
                );
            }
        }
    }
    paths
}

/// `git diff --name-status -M -z old..new` in `data_path`, parsed into touched paths.
/// Local git only — no network. `-M` forces rename detection so a pure rename is
/// reported as `R`, not as a delete+add pair (which would still work — both paths end
/// up in the result — but would cost the new path an unnecessary re-embed instead of
/// letting `index_paths` see it as unchanged content under a new name... which it
/// cannot, since content hashing does not know about the old path. Either way both
/// paths are enqueued; `-M` is for a cleaner log line, not correctness here).
///
/// `-z` NUL-delimits the output instead of git's default newline/tab-delimited,
/// C-quoted format — see [`parse_diff_name_status`]'s doc comment for why that
/// matters for any path containing non-ASCII bytes or other special characters.
pub(crate) async fn git_diff_name_status(
    _lock: &GitLock,
    data_path: &str,
    old: &str,
    new: &str,
) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let range = format!("{old}..{new}");
    let out = timeout(
        GIT_TIMEOUT,
        Command::new("git")
            .args([
                "-c",
                &format!("safe.directory={}", data_path),
                "diff",
                "--name-status",
                "-M",
                "-z",
                &range,
            ])
            .current_dir(data_path)
            .output(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("git diff timed out after {:?}", GIT_TIMEOUT))?
    .context("Failed to spawn git diff")?;

    if !out.status.success() {
        let stderr = redact_url(&String::from_utf8_lossy(&out.stderr));
        anyhow::bail!("git diff --name-status failed: {}", stderr);
    }

    Ok(parse_diff_name_status(&String::from_utf8_lossy(
        &out.stdout,
    )))
}

/// Parse `git log --format=%x02%ct%x03 --name-only --diff-filter=ACMR -z` output into
/// a map from repo-relative path to the Unix timestamp (`%ct`, committer time) of the
/// most recent commit that touched it — see [`git_log_mtimes`] for why this exists
/// and why it is one batched call rather than one `git log -1` per file.
///
/// Byte-level shape (verified empirically, not assumed from docs): `-z` NUL-delimits
/// path records exactly as it does for [`parse_diff_name_status`] above, but a plain
/// `--format` string is NOT itself NUL-terminated by `-z` — only the diff/name-only
/// machinery is. So each commit's formatted line still ends with git's own `\n`, and
/// with `--name-only` that newline becomes the record separator immediately before
/// the commit's first touched path, landing as a literal leading `\n` on the NUL-split
/// token right after the format token. The `\x02…\x03` wrapper around `%ct` is what
/// makes format tokens unambiguously distinguishable from path tokens regardless of
/// that stray newline or of what characters appear in a path (an all-digit filename
/// would otherwise be indistinguishable from a bare timestamp).
///
/// `git log`'s default order is newest-first, so the FIRST time a path is seen here
/// is its most recent touch — `.or_insert` below is what makes that stick.
fn parse_git_log_mtimes(raw: &[u8]) -> HashMap<String, i64> {
    let mut map = HashMap::new();
    let mut current_ts: Option<i64> = None;

    for token in raw.split(|&b| b == 0) {
        // Strip the record-separator artifact described above: a literal leading
        // newline on the token immediately following a timestamp token (whether that
        // token turns out to be a path, or — when a commit's diff touched nothing
        // matching `--diff-filter`, e.g. a merge or a delete-only commit — the NEXT
        // commit's own timestamp token).
        let token = token.strip_prefix(b"\n").unwrap_or(token);
        if token.is_empty() {
            continue;
        }

        if token.len() >= 2 && token[0] == 0x02 && token[token.len() - 1] == 0x03 {
            let inner = &token[1..token.len() - 1];
            current_ts = std::str::from_utf8(inner)
                .ok()
                .and_then(|s| s.parse::<i64>().ok());
            continue;
        }

        if let Some(ts) = current_ts {
            let path = String::from_utf8_lossy(token).into_owned();
            map.entry(path).or_insert(ts);
        }
    }

    map
}

/// Conservative per-invocation byte budget for the pathspec arguments
/// [`git_log_mtimes`] passes to a single `git log` call (#237).
///
/// The real OS `ARG_MAX` is typically far larger (multiple MB on Linux —
/// `getconf ARG_MAX` commonly reports 2MB+), but this deliberately stays well under
/// any real-world floor rather than trying to probe or compute the actual limit at
/// call time: some container/exec environments cap lower than the host default, argv
/// and the process's environment variables share the same kernel-enforced budget (so
/// "how much room argv actually has" depends on how big `envp` happens to be, not
/// just on `ARG_MAX` itself), and probing would cost a syscall to save an amount of
/// slack that is an order of magnitude away from ever mattering at any corpus size
/// this project targets. Sized in path BYTES, not path count, because path length
/// varies wildly (a nested `domain/dev/subsystem/...` tree vs. a root-level file) and
/// a byte budget is what `ARG_MAX` itself actually measures.
const GIT_LOG_MTIMES_ARG_BUDGET_BYTES: usize = 100_000;

/// Splits `paths` into chunks whose total byte length (plus a small per-entry
/// separator allowance) stays under [`GIT_LOG_MTIMES_ARG_BUDGET_BYTES`], so
/// [`git_log_mtimes`] never builds a single `git log` argv that risks exceeding
/// `ARG_MAX` (#237) no matter how large the corpus.
///
/// A single path longer than the whole budget still gets its own one-entry chunk
/// rather than being silently dropped: the `i > start` guard below only refuses to
/// split a chunk BEFORE it holds anything, so the very first entry considered for a
/// fresh chunk is always accepted into it regardless of size. An oversized chunk like
/// that can still fail at the OS level if the one path itself is longer than real
/// `ARG_MAX` — see [`git_log_mtimes`]'s per-chunk error handling for what happens
/// then (that one chunk degrades; the others are unaffected).
fn chunk_paths_by_byte_budget(paths: &[String]) -> Vec<&[String]> {
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut used = 0usize;
    for (i, p) in paths.iter().enumerate() {
        // +1: conservative per-entry allowance for the argv pointer/separator
        // overhead the kernel also counts against the same budget as the string
        // bytes themselves.
        let cost = p.len() + 1;
        if i > start && used + cost > GIT_LOG_MTIMES_ARG_BUDGET_BYTES {
            chunks.push(&paths[start..i]);
            start = i;
            used = 0;
        }
        used += cost;
    }
    if start < paths.len() {
        chunks.push(&paths[start..]);
    }
    chunks
}

/// #164: batched, git-log-derived "true" last-modified time for every path in
/// `paths` — one (or, since #237, a handful of chunked) `git log` invocation(s)
/// rather than one `git log -1` per file, the whole point, since a per-file
/// invocation is what made this prohibitively slow across a large corpus. Local git
/// only, no network.
///
/// `paths` is passed straight through as each chunk's `git log` pathspec (`--
/// <paths>`), so this only ever walks history relevant to the exact files the caller
/// is about to index — a single scoped write costs one narrowly-filtered log walk
/// (or a couple, if its path list happens to cross the chunk budget), not a
/// full-repo one, and a full reconcile costs walks scoped to the whole corpus
/// instead of one walk per file. `paths.is_empty()` returns an empty map without
/// spawning git at all: an unfiltered `git log` with no pathspec would walk (and
/// return touched-path data for) the ENTIRE repository, which is not what an empty
/// scope means here.
///
/// #237: unlike the pre-#237 version, this never builds one argv for the WHOLE
/// `paths` list — [`chunk_paths_by_byte_budget`] splits it first, and each chunk
/// runs as its own `git log` call via [`git_log_mtimes_chunk`], with results merged
/// as they come back. This is what keeps a large corpus from exceeding `ARG_MAX` and
/// silently losing git-derived mtimes for the ENTIRE run — see that constant's doc
/// comment for the byte-budget rationale. A failing chunk (timeout, non-zero exit,
/// or — even chunked — a single pathological path too long for `ARG_MAX` on its
/// own) degrades only the paths in THAT chunk: they end up absent from the returned
/// map, same as any path git has no history for, and every other chunk's results are
/// still merged in. This function itself therefore never fails outright anymore —
/// unlike its pre-#237 signature, there is no longer an `Err` case for a caller to
/// handle, because a partial failure is now always something this function can
/// recover from internally rather than something it needs to propagate.
///
/// `--diff-filter=ACMR` excludes pure deletions (`D`): a path currently on disk that
/// this function is being asked about was necessarily added or last modified more
/// recently than any subsequent deletion of the same path could have been, so
/// including `D` entries would only ever risk shadowing the real answer with a
/// deletion timestamp for a path that was later recreated.
pub async fn git_log_mtimes(
    lock: &GitLock,
    data_path: &str,
    paths: &[String],
) -> HashMap<String, i64> {
    if paths.is_empty() {
        return HashMap::new();
    }

    let chunks = chunk_paths_by_byte_budget(paths);
    let chunk_count = chunks.len();
    let mut merged = HashMap::new();
    let mut failed_chunks = 0usize;
    let mut failed_paths = 0usize;

    for (idx, chunk) in chunks.into_iter().enumerate() {
        match git_log_mtimes_chunk(lock, data_path, chunk).await {
            Ok(map) => merged.extend(map),
            Err(e) => {
                // #237: a failing chunk degrades only the paths IN that chunk — they
                // fall back to filesystem mtime, same as any path missing from the
                // merged map already means — not the whole call, and not sibling
                // chunks already merged in or still to come.
                warn!(
                    chunk = idx,
                    of = chunk_count,
                    paths_in_chunk = chunk.len(),
                    "git log (mtime lookup) failed for one path chunk — those paths \
                     fall back to filesystem mtime: {:#}",
                    e
                );
                failed_chunks += 1;
                failed_paths += chunk.len();
            }
        }
    }

    if failed_chunks > 0 {
        warn!(
            failed_chunks,
            failed_paths,
            total_chunks = chunk_count,
            "git log (mtime lookup): {failed_chunks} of {chunk_count} chunk(s) failed \
             ({failed_paths} path(s) affected); remaining chunks still contributed \
             git-derived mtimes"
        );
    }

    merged
}

/// Runs one `git log` invocation for a single chunk of paths — the command
/// build/spawn/parse logic [`git_log_mtimes`] used to run exactly once, for the
/// WHOLE path list, before #237 batched it by byte budget. Split into its own
/// function so `git_log_mtimes` can call it per chunk and let one chunk's failure
/// degrade independently of the others, rather than the all-or-nothing fallback the
/// unbatched version had.
async fn git_log_mtimes_chunk(
    _lock: &GitLock,
    data_path: &str,
    paths: &[String],
) -> Result<HashMap<String, i64>> {
    let safe_dir = format!("safe.directory={}", data_path);
    let mut args: Vec<&str> = vec![
        "-c",
        &safe_dir,
        "log",
        "--format=%x02%ct%x03",
        "--name-only",
        "--diff-filter=ACMR",
        "-z",
        "--",
    ];
    args.extend(paths.iter().map(|p| p.as_str()));

    let out = timeout(
        GIT_TIMEOUT,
        Command::new("git")
            .args(&args)
            .current_dir(data_path)
            .output(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("git log timed out after {:?}", GIT_TIMEOUT))?
    .context("Failed to spawn git log")?;

    if !out.status.success() {
        let stderr = redact_url(&String::from_utf8_lossy(&out.stderr));
        anyhow::bail!("git log (mtime lookup) failed: {}", stderr);
    }

    Ok(parse_git_log_mtimes(&out.stdout))
}

/// The outcome of a failed [`commit_and_sync`] call, split by whether a commit landed.
///
/// `commit_and_sync` runs five git operations in sequence — add, commit, fetch,
/// rebase, push — and the two ways it can fail demand OPPOSITE recovery, so this is a
/// hard enum rather than a flattened `anyhow::Error`:
///
/// - [`PreCommit`](Self::PreCommit): `git add` or `git commit` failed. HEAD is
///   untouched — the attempted change exists only in the working tree (and possibly
///   the index, if `add` succeeded but `commit` did not). A caller can safely discard
///   it and report that nothing changed; that discarding is exactly what
///   [`restore_from_head`] / [`unstage`] are for.
/// - [`PostCommit`](Self::PostCommit): `git fetch`, `git rebase`, or `git push`
///   failed, but the commit itself landed — HEAD already includes it. Discarding the
///   working-tree change at this point would silently resurrect (on a delete) or
///   revert (on a create/edit) content that is genuinely, durably gone/changed as far
///   as the local repo is concerned. The only thing that failed is telling the remote
///   about it, so the correct move is to leave it exactly as it is and report that the
///   sync is pending.
///
/// Being a plain enum (not `anyhow::Error`) means there is no `?`-friendly blanket
/// conversion into it — a caller has to name a variant to get at the underlying
/// cause, which is what keeps the phase distinction from being silently discarded on
/// the way past.
#[derive(Debug, thiserror::Error)]
pub enum CommitSyncError {
    /// Nothing was committed. `redact_url` has already been applied to any git
    /// stderr folded into the cause.
    #[error("{0:#}")]
    PreCommit(anyhow::Error),

    /// `sha` is a real local commit — do not roll it back. `redact_url` has already
    /// been applied to any git stderr folded into the cause.
    #[error("commit {sha} landed locally but syncing to the remote failed: {source:#}")]
    PostCommit { sha: String, source: anyhow::Error },
}

/// Stage `rel_path`, commit with `message`, then (if `git_url` is Some) fetch the
/// remote branch, rebase the local branch onto it, and push. Returns the new commit
/// SHA plus any paths pulled in by the rebase — see [`CommitOutcome`].
///
/// `paths` are relative to `data_path` and are committed together as a single, atomic
/// commit — e.g. a document move can stage the old path's removal and the new path's
/// addition as one call rather than two. `message` already includes any provenance
/// trailer. If `git_url` is None, commit locally only (no fetch/rebase/push).
/// On a rebase conflict, abort the rebase (so the working tree is left clean at the local
/// commit) and return an Err whose message clearly identifies it as a rebase/merge conflict
/// on the file, distinct from other git failures.
///
/// `paths` must not be empty — see the pathspec-scoping comment below on why an
/// unscoped commit is dangerous. An empty slice returns `CommitSyncError::PreCommit`
/// before any git command runs.
///
/// The `Err` side distinguishes exactly where in that sequence things went wrong — see
/// [`CommitSyncError`]. A caller that needs to roll back a failure must match on it
/// rather than treat every failure the same way.
#[allow(clippy::too_many_arguments)]
pub async fn commit_and_sync(
    lock: &GitLock,
    git_url: Option<&str>,
    branch: &str,
    data_path: &str,
    token: Option<&str>,
    paths: &[&str],
    message: &str,
    author_name: &str,
    author_email: &str,
) -> Result<CommitOutcome, CommitSyncError> {
    if paths.is_empty() {
        return Err(CommitSyncError::PreCommit(anyhow::anyhow!(
            "commit_and_sync called with no paths"
        )));
    }

    // Helper: build a base git command with safe.directory set and cwd pointing at data_path.
    // Returns (Command,) ready to have more args appended.
    let git_cmd = |args: &[&str]| {
        let mut cmd = Command::new("git");
        cmd.args(["-c", &format!("safe.directory={}", data_path)])
            .args(args)
            .current_dir(data_path);
        cmd
    };

    // Helper: `git_cmd` plus the commit identity in the environment.
    //
    // EVERY git subcommand that creates a commit needs this, not just `commit`.
    // `rebase` replays commits, so it needs a committer identity too — and it was
    // previously invoked without one. In any environment lacking a global git config
    // that fails with "Committer identity unknown", which the deployed container hits
    // exactly: it runs as a non-root user with HOME=/tmp and no .gitconfig. It never
    // surfaced in practice only because a fetch that fast-forwards replays nothing;
    // the failure needs a genuine divergence, which is precisely the case this
    // fetch→rebase→push sequence exists to handle.
    let git_cmd_authored = |args: &[&str]| {
        let mut cmd = git_cmd(args);
        cmd.env("GIT_AUTHOR_NAME", author_name)
            .env("GIT_AUTHOR_EMAIL", author_email)
            .env("GIT_COMMITTER_NAME", author_name)
            .env("GIT_COMMITTER_EMAIL", author_email);
        cmd
    };

    // --- git add -- <paths...> ---
    // Every failure from here through the end of `git commit` is PreCommit: HEAD has
    // not moved, so a caller can discard whatever this left behind and be back to
    // exactly where it started.
    let mut add_args: Vec<&str> = vec!["add", "--"];
    add_args.extend(paths.iter().copied());
    let add_out = timeout(GIT_TIMEOUT, git_cmd(&add_args).output())
        .await
        .map_err(|_| {
            CommitSyncError::PreCommit(anyhow::anyhow!("git add timed out after {:?}", GIT_TIMEOUT))
        })?
        .map_err(|e| {
            CommitSyncError::PreCommit(anyhow::Error::new(e).context("Failed to spawn git add"))
        })?;
    if !add_out.status.success() {
        let stderr = redact_url(&String::from_utf8_lossy(&add_out.stderr));
        return Err(CommitSyncError::PreCommit(anyhow::anyhow!(
            "git add failed: {}",
            stderr
        )));
    }

    // --- git commit -m <message> -- <paths...> ---
    // Set the author identity inline so the command is self-contained even in
    // environments without a global git user configured. Both author and committer
    // derive from user.* when not otherwise specified.
    //
    // The trailing pathspec is what keeps this commit to the caller's own set of
    // paths. Without it `git commit` commits the ENTIRE index, so any unrelated
    // staged entry — left by a failed write whose rollback did not complete, by the
    // separate `index --full` CLI process, or by anything else touching the clone —
    // silently rides along in this call's commit. The `add` above was already
    // path-scoped; the commit has to be too, or the scoping accomplishes nothing.
    // Passing multiple paths here (rather than issuing one commit per path) is what
    // lets a caller land a multi-file change — e.g. a document move's delete-old +
    // add-new — as a single atomic commit instead of two.
    let mut commit_args: Vec<&str> = vec!["commit", "-m", message, "--"];
    commit_args.extend(paths.iter().copied());
    let commit_out = timeout(GIT_TIMEOUT, git_cmd_authored(&commit_args).output())
        .await
        .map_err(|_| {
            CommitSyncError::PreCommit(anyhow::anyhow!(
                "git commit timed out after {:?}",
                GIT_TIMEOUT
            ))
        })?
        .map_err(|e| {
            CommitSyncError::PreCommit(anyhow::Error::new(e).context("Failed to spawn git commit"))
        })?;
    if !commit_out.status.success() {
        let stderr = redact_url(&String::from_utf8_lossy(&commit_out.stderr));
        return Err(CommitSyncError::PreCommit(anyhow::anyhow!(
            "git commit failed: {}",
            stderr
        )));
    }

    // The commit has landed. Every failure from here on is PostCommit — HEAD already
    // includes it, so `local_sha` (captured now, before anything else can touch HEAD)
    // is a real, durable local commit no matter what the rest of this call does.
    let local_sha = rev_parse_head(lock, data_path).await.map_err(|e| {
        // rev-parse HEAD failing immediately after a successful `git commit` would
        // mean something is badly wrong with the repo itself, not with sync — but the
        // commit above DID succeed, so this is still unambiguously post-commit. There
        // is just no sha to attach to it.
        CommitSyncError::PostCommit {
            sha: "<unknown: rev-parse HEAD failed immediately after a successful commit>"
                .to_string(),
            source: e,
        }
    })?;

    let mut rebased_paths: Vec<std::path::PathBuf> = Vec::new();

    if let Some(url) = git_url {
        let auth_url = match token {
            Some(t) if !t.is_empty() => inject_token_into_url(url, t),
            _ => url.to_string(),
        };

        // `local_sha` doubles as "HEAD right after our own commit and before the
        // fetch". Diffing this against HEAD once the rebase completes isolates
        // exactly what the rebase pulled in from the remote — our own change is
        // already on both sides of that range, so it is never double-reported here.
        let old_head = local_sha.clone();

        // --- git fetch --no-tags <auth_url> <branch> ---
        info!(
            "Fetching {} branch {} for rebase",
            redact_url(&auth_url),
            branch
        );
        let fetch_out = timeout(
            GIT_TIMEOUT,
            git_cmd(&["fetch", "--no-tags", &auth_url, branch]).output(),
        )
        .await
        .map_err(|_| CommitSyncError::PostCommit {
            sha: local_sha.clone(),
            source: anyhow::anyhow!("git fetch timed out after {:?}", GIT_TIMEOUT),
        })?
        .map_err(|e| CommitSyncError::PostCommit {
            sha: local_sha.clone(),
            source: anyhow::Error::new(e).context("Failed to spawn git fetch"),
        })?;
        if !fetch_out.status.success() {
            let stderr = redact_url(&String::from_utf8_lossy(&fetch_out.stderr));
            return Err(CommitSyncError::PostCommit {
                sha: local_sha.clone(),
                source: anyhow::anyhow!("git fetch failed: {}", stderr),
            });
        }

        // --- git rebase FETCH_HEAD ---
        let rebase_out = timeout(
            GIT_TIMEOUT,
            git_cmd_authored(&["rebase", "FETCH_HEAD"]).output(),
        )
        .await
        .map_err(|_| CommitSyncError::PostCommit {
            sha: local_sha.clone(),
            source: anyhow::anyhow!("git rebase timed out after {:?}", GIT_TIMEOUT),
        })?
        .map_err(|e| CommitSyncError::PostCommit {
            sha: local_sha.clone(),
            source: anyhow::Error::new(e).context("Failed to spawn git rebase"),
        })?;
        if !rebase_out.status.success() {
            let stderr = redact_url(&String::from_utf8_lossy(&rebase_out.stderr));

            // Establish whether this was a REAL conflict before aborting — once the
            // rebase is aborted the unmerged paths are gone. `--diff-filter=U` lists
            // exactly the files left in a conflicted state, which only a genuine
            // content conflict produces.
            //
            // This previously labelled every rebase failure a "rebase conflict" on
            // `rel_path` — the file the caller happened to be writing, which in a
            // real conflict is usually not even the conflicting file. So an unrelated
            // failure (e.g. "Committer identity unknown") was reported as a phantom
            // conflict on an innocent file, sending you looking for a merge problem
            // that never existed.
            let conflicted = timeout(
                GIT_TIMEOUT,
                git_cmd(&["diff", "--name-only", "--diff-filter=U"]).output(),
            )
            .await
            .ok()
            .and_then(|r| r.ok())
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
            .unwrap_or_default();

            // Abort the rebase so the working tree is left clean at the local commit.
            let _ = git_cmd(&["rebase", "--abort"]).output().await;

            // Either way, our own commit (`local_sha`) is still there — the rebase
            // touched nothing durable, it just couldn't finish. Still PostCommit.
            if conflicted.is_empty() {
                return Err(CommitSyncError::PostCommit {
                    sha: local_sha.clone(),
                    source: anyhow::anyhow!(
                        "git rebase onto FETCH_HEAD failed with no conflicting files — \
                         this is not a merge conflict. Rebase aborted. stderr: {}",
                        stderr
                    ),
                });
            }
            return Err(CommitSyncError::PostCommit {
                sha: local_sha.clone(),
                source: anyhow::anyhow!(
                    "rebase conflict: git rebase onto FETCH_HEAD conflicted on {}. \
                     Rebase aborted. stderr: {}",
                    conflicted.replace('\n', ", "),
                    stderr
                ),
            });
        }

        // The rebase just succeeded, which means it may have REPLAYED our commit
        // onto a brand-new sha (whenever FETCH_HEAD actually had something to
        // rebase onto — the normal case, since writers and the webhook overlap
        // routinely per this module's doc comment). Capture that post-rebase HEAD
        // now and use it — not the pre-rebase `local_sha` — in every failure
        // branch from here through the end of this function (#140): `local_sha`
        // can already be a dangling, unreachable object by this point, so
        // reporting it in a push-failure error would point the caller at a commit
        // `git show`/`git log` can no longer find, with no way to locate the one
        // that is actually sitting at HEAD pending sync. `git push` never rewrites
        // local history, so this same value is also exactly the final answer on
        // the success path below — no second re-read needed there anymore.
        let post_rebase_sha =
            rev_parse_head(lock, data_path)
                .await
                .map_err(|e| CommitSyncError::PostCommit {
                    sha: "<unknown: rev-parse HEAD failed immediately after a successful rebase>"
                        .to_string(),
                    source: e,
                })?;

        // Diff the rebase range now, before pushing — this is local-only (no network)
        // and a push failure below should not prevent the caller from at least
        // learning what changed locally, though in practice a push failure aborts the
        // whole call anyway.
        rebased_paths = git_diff_name_status(lock, data_path, &old_head, &post_rebase_sha)
            .await
            .map_err(|e| CommitSyncError::PostCommit {
                sha: post_rebase_sha.clone(),
                source: e.context("Failed to diff the rebase range"),
            })?;

        // --- git push <auth_url> HEAD:<branch> ---
        info!("Pushing to {} branch {}", redact_url(&auth_url), branch);
        let push_refspec = format!("HEAD:{}", branch);
        let push_out = timeout(
            GIT_TIMEOUT,
            git_cmd(&["push", &auth_url, &push_refspec]).output(),
        )
        .await
        .map_err(|_| CommitSyncError::PostCommit {
            sha: post_rebase_sha.clone(),
            source: anyhow::anyhow!("git push timed out after {:?}", GIT_TIMEOUT),
        })?
        .map_err(|e| CommitSyncError::PostCommit {
            sha: post_rebase_sha.clone(),
            source: anyhow::Error::new(e).context("Failed to spawn git push"),
        })?;
        if !push_out.status.success() {
            let stderr = redact_url(&String::from_utf8_lossy(&push_out.stderr));
            return Err(CommitSyncError::PostCommit {
                sha: post_rebase_sha.clone(),
                source: anyhow::anyhow!("git push failed: {}", stderr),
            });
        }

        // `git push` does not move local HEAD, so `post_rebase_sha` is still
        // exactly right here — no need to re-read it a second time.
        return Ok(CommitOutcome {
            sha: post_rebase_sha,
            rebased_paths,
        });
    }

    // No remote configured: `local_sha` is already the final answer.
    Ok(CommitOutcome {
        sha: local_sha,
        rebased_paths,
    })
}

/// Restore `rel_path`'s content in BOTH the index and the working tree to match HEAD,
/// discarding any uncommitted change to it — a deletion, an overwrite, or a `git add`
/// that got staged but never committed.
///
/// Only valid to call in response to [`CommitSyncError::PreCommit`]. HEAD has not
/// moved in that case, so "match HEAD" really does mean "match how things were right
/// before `commit_and_sync` was called." Calling this after a
/// [`CommitSyncError::PostCommit`] would be wrong in the opposite direction: HEAD by
/// then already includes the very change this would erase.
///
/// Requires `rel_path` to exist at HEAD — `git restore` fails on a pathspec it has no
/// prior content for. That holds for a delete or an edit rollback (the path was
/// already tracked, or it would not have been deletable/editable). It does NOT hold
/// for a brand-new path whose first-ever commit failed before landing (`create_document`);
/// use [`unstage`] for that case instead.
pub async fn restore_from_head(
    _lock: &GitLock,
    data_path: &str,
    rel_path: &str,
) -> anyhow::Result<()> {
    let out = timeout(
        GIT_TIMEOUT,
        Command::new("git")
            .args([
                "-c",
                &format!("safe.directory={}", data_path),
                "restore",
                "--source=HEAD",
                "--staged",
                "--worktree",
                "--",
                rel_path,
            ])
            .current_dir(data_path)
            .output(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("git restore timed out after {:?}", GIT_TIMEOUT))?
    .context("Failed to spawn git restore")?;

    if !out.status.success() {
        let stderr = redact_url(&String::from_utf8_lossy(&out.stderr));
        anyhow::bail!(
            "git restore --source=HEAD -- {} failed: {}",
            rel_path,
            stderr
        );
    }
    Ok(())
}

/// Remove `rel_path` from the index without touching the working tree — i.e. undo
/// whatever `git add` staged for it. A safe no-op (exit 0, not an error) if `rel_path`
/// was never staged in the first place, which is what makes this usable
/// unconditionally rather than only when the caller knows `git add` succeeded.
///
/// Used to roll back `create_document`'s pre-commit failures: the new file has no
/// HEAD content to fall back to (`restore_from_head` would fail on it — see its
/// doc), so the caller removes the file itself and calls this to make sure `git add`
/// staging it doesn't silently ride along on some later, unrelated commit.
pub async fn unstage(_lock: &GitLock, data_path: &str, rel_path: &str) -> anyhow::Result<()> {
    let out = timeout(
        GIT_TIMEOUT,
        Command::new("git")
            .args([
                "-c",
                &format!("safe.directory={}", data_path),
                "reset",
                "--",
                rel_path,
            ])
            .current_dir(data_path)
            .output(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("git reset timed out after {:?}", GIT_TIMEOUT))?
    .context("Failed to spawn git reset")?;

    if !out.status.success() {
        let stderr = redact_url(&String::from_utf8_lossy(&out.stderr));
        anyhow::bail!("git reset -- {} failed: {}", rel_path, stderr);
    }
    Ok(())
}

/// Clean up git state left behind by a process that was killed mid-operation —
/// an interrupted rebase/merge, or a stale `.git/index.lock` — before anything else
/// in this process touches the clone.
///
/// Call once, at startup, right after the clone is confirmed to exist (i.e. right
/// after [`ensure_repo`]) and before the bootstrap scan or anything else reads from
/// it. It exists because a SIGKILL (watchtower's stop-grace timeout, or anything
/// else that doesn't give the process a chance to finish) can land mid `git rebase`
/// or mid `git commit`. Unlike the index — survivable via idempotent UUID5 upserts
/// plus the startup reconcile — the clone has no self-healing path of its own: an
/// interrupted rebase or a stale `index.lock` makes every subsequent
/// [`commit_and_sync`] fail, forever, with no recovery short of this.
///
/// ## Why removing `index.lock` unconditionally is safe here
///
/// Elsewhere, "is this lockfile stale" requires checking whether its owning process
/// is still alive, because another live process could legitimately hold it right
/// now. That check does not apply here. This process is the clone's *sole* owner
/// inside the container — no sibling process ever touches this working copy — and
/// this function runs exactly once, at the very start of this process's life,
/// before it has issued a single git command of its own. Any lockfile found at this
/// point cannot belong to "us" (we haven't done anything yet) and cannot belong to
/// a concurrent process (there isn't one); by construction it is leftover from a
/// killed predecessor.
///
/// ## Ordering
///
/// `index.lock` is removed FIRST. `git rebase --abort` / `git merge --abort` both
/// need to write to the index themselves, and would fail with "Unable to create
/// '.git/index.lock': File exists" if the stale lock were still in place — exactly
/// the failure this function exists to clear.
///
/// ## Failure handling
///
/// Never fails boot: every abort command's own failure is logged and swallowed
/// rather than propagated. A repo broken in some way this can't fix keeps failing
/// loudly on the reconcile sweep and on every write after this — a much better
/// place to surface it than blocking startup.
pub async fn recover_interrupted_state(lock: &GitLock, repo: &Path) {
    let git_dir = repo.join(".git");

    // First: the index lock, so the aborts below can actually touch the index.
    let index_lock = git_dir.join("index.lock");
    if index_lock.exists() {
        warn!(
            "Found stale .git/index.lock at startup (left behind by a killed \
             predecessor process — this process is the clone's sole owner and has \
             only just started, so no live process can hold it) — removing it"
        );
        if let Err(e) = tokio::fs::remove_file(&index_lock).await {
            error!("Failed to remove stale .git/index.lock: {e:#}");
        }
    }

    let rebase_merge = git_dir.join("rebase-merge");
    let rebase_apply = git_dir.join("rebase-apply");
    if rebase_merge.exists() || rebase_apply.exists() {
        let marker = if rebase_merge.exists() {
            "rebase-merge"
        } else {
            "rebase-apply"
        };
        warn!("Found an interrupted rebase at startup (.git/{marker} present) — aborting it");
        if let Err(e) = run_abort(lock, repo, &["rebase", "--abort"]).await {
            error!("git rebase --abort failed during startup recovery: {e:#}");
        }
    }

    let merge_head = git_dir.join("MERGE_HEAD");
    if merge_head.exists() {
        warn!("Found an interrupted merge at startup (.git/MERGE_HEAD present) — aborting it");
        if let Err(e) = run_abort(lock, repo, &["merge", "--abort"]).await {
            error!("git merge --abort failed during startup recovery: {e:#}");
        }
    }
}

/// Run a git abort subcommand (`rebase --abort` / `merge --abort`) in `repo`,
/// returning an error with redacted stderr on non-zero exit or timeout. Shared by
/// [`recover_interrupted_state`]'s two abort paths.
async fn run_abort(_lock: &GitLock, repo: &Path, args: &[&str]) -> anyhow::Result<()> {
    let data_path = repo.to_string_lossy();
    let joined = args.join(" ");
    let out = timeout(
        GIT_TIMEOUT,
        Command::new("git")
            .args(["-c", &format!("safe.directory={}", data_path)])
            .args(args)
            .current_dir(repo)
            .output(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("git {} timed out after {:?}", joined, GIT_TIMEOUT))?
    .with_context(|| format!("Failed to spawn git {}", joined))?;

    if !out.status.success() {
        let stderr = redact_url(&String::from_utf8_lossy(&out.stderr));
        anyhow::bail!("git {} failed: {}", joined, stderr);
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Build a `git` `Command` for a test fixture, pinned against the escape
    /// hatch in #218.
    ///
    /// Git resolves "the repository" by walking *up* from the working
    /// directory until it finds a `.git`. Every fixture in this module runs
    /// git inside a `tempfile::TempDir`, and several call sites (`init`,
    /// `clone` into an empty dir, the first `add`/`commit` before either has
    /// run) execute before that directory necessarily contains a `.git` of
    /// its own. Normally `TMPDIR` is `/tmp`, which sits outside any repo, so
    /// the upward walk finds nothing and the gap is invisible. Point `TMPDIR`
    /// at a path inside a real checkout instead (done twice by agents working
    /// in git worktrees, to dodge a full `/tmp` tmpfs) and that same walk
    /// finds the checkout's `.git` and silently commits to it — see #218 for
    /// two independent incidents.
    ///
    /// `GIT_CEILING_DIRECTORIES` tells git which directories it must not climb
    /// *into* while searching upward. Critically that means the ceiling has to
    /// be `dir`'s **parent**, not `dir` itself — `dir` is where the search
    /// starts, so listing it as a ceiling is a no-op and the walk still
    /// escapes (verified empirically: `GIT_CEILING_DIRECTORIES=$fixture` did
    /// nothing, `GIT_CEILING_DIRECTORIES=$(dirname $fixture)` stopped it cold,
    /// turning the escape into a clean "not a git repository" error). Every
    /// git invocation in this test module must be built through this helper
    /// rather than calling `std::process::Command::new("git")` directly —
    /// the protection is only worth anything if it is uniform across all of
    /// them (~40 call sites) — and `git_ceiling_directories_blocks_escape`
    /// below is the regression test proving it holds.
    fn git_test_cmd(dir: impl AsRef<Path>) -> std::process::Command {
        let dir = dir.as_ref();
        let mut cmd = std::process::Command::new("git");
        cmd.current_dir(dir);
        // Fall back to `dir` itself only if it has no parent (e.g. a root),
        // which no real fixture path ever is — this just avoids a panic.
        cmd.env("GIT_CEILING_DIRECTORIES", dir.parent().unwrap_or(dir));
        cmd
    }

    /// Regression test for #218: point a fixture's working directory inside a
    /// real (scratch) git repository — the exact misconfiguration that let
    /// git-backed tests silently commit to an enclosing checkout — and assert
    /// that repo's `HEAD` is untouched after running fixture git commands
    /// through [`git_test_cmd`]. Without the `GIT_CEILING_DIRECTORIES` guard
    /// this test fails: `git add`/`commit` run from `enclosing/nested` (which
    /// has no `.git` of its own) walk up, find `enclosing`'s `.git`, and stage
    /// / commit into the real repo instead of erroring out.
    #[test]
    fn git_ceiling_directories_blocks_escape() {
        let enclosing = tempfile::TempDir::new().unwrap();
        std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(enclosing.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(enclosing.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(enclosing.path())
            .output()
            .unwrap();
        std::fs::write(enclosing.path().join("seed.md"), "seed").unwrap();
        std::process::Command::new("git")
            .args(["add", "seed.md"])
            .current_dir(enclosing.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "seed commit"])
            .current_dir(enclosing.path())
            .output()
            .unwrap();
        let head_before = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(enclosing.path())
            .output()
            .unwrap();
        let head_before = String::from_utf8_lossy(&head_before.stdout).to_string();

        // A fixture directory nested inside the "real" repo, with no `.git`
        // of its own -- this is the shape `TMPDIR` pointed inside a worktree
        // produces: every `tempfile::TempDir::new()` lands somewhere under
        // an enclosing checkout.
        let fixture = enclosing.path().join("nested_fixture");
        std::fs::create_dir(&fixture).unwrap();
        std::fs::write(fixture.join("stray.md"), "should never be committed").unwrap();

        // Without the ceiling guard, both of these would walk up into
        // `enclosing`'s `.git` and stage/commit `stray.md` there.
        git_test_cmd(&fixture)
            .args(["add", "stray.md"])
            .output()
            .unwrap();
        let commit_out = git_test_cmd(&fixture)
            .args(["commit", "-m", "should fail: no repo here"])
            .output()
            .unwrap();
        assert!(
            !commit_out.status.success(),
            "commit from a non-repo fixture dir must fail once GIT_CEILING_DIRECTORIES \
             blocks upward discovery, not silently land in the enclosing repo"
        );

        let head_after = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(enclosing.path())
            .output()
            .unwrap();
        let head_after = String::from_utf8_lossy(&head_after.stdout).to_string();
        assert_eq!(
            head_before, head_after,
            "the enclosing repo's HEAD must be untouched by a fixture command \
             that never found a .git of its own"
        );

        // Check the INDEX specifically, not `git status --porcelain`: the
        // fixture directory (`nested_fixture/`, containing `stray.md`) is a
        // real subdirectory of `enclosing` and will show up there as an
        // untracked path regardless of whether the ceiling guard worked --
        // that's expected and not what this test is protecting against. What
        // must never happen is `stray.md` landing in the enclosing repo's
        // index (staged) or history (committed); the `HEAD` check above
        // already covers history, so this covers staging.
        let diff_cached = std::process::Command::new("git")
            .args(["diff", "--cached", "--name-only"])
            .current_dir(enclosing.path())
            .output()
            .unwrap();
        assert!(
            String::from_utf8_lossy(&diff_cached.stdout).is_empty(),
            "the enclosing repo's index must be untouched -- stray.md must \
             never have been staged into it"
        );
    }

    // --- inject_token_into_url tests ---

    #[test]
    fn inject_token_into_https_url() {
        let url = "https://gitea.example.com/user/repo.git";
        let result = inject_token_into_url(url, "ghp_abc123");
        assert_eq!(result, "https://ghp_abc123@gitea.example.com/user/repo.git");
    }

    #[test]
    fn inject_token_leaves_ssh_url_unchanged() {
        let url = "git@gitea.example.com:user/repo.git";
        let result = inject_token_into_url(url, "ghp_abc123");
        assert_eq!(result, url);
    }

    #[test]
    fn inject_token_empty_token() {
        let url = "https://gitea.example.com/user/repo.git";
        let result = inject_token_into_url(url, "");
        assert_eq!(result, "https://@gitea.example.com/user/repo.git");
    }

    // --- redact_url tests ---

    #[test]
    fn redact_url_hides_token() {
        let url = "https://ghp_abc123@gitea.example.com/user/repo.git";
        let result = redact_url(url);
        assert_eq!(result, "https://***@gitea.example.com/user/repo.git");
        assert!(!result.contains("ghp_abc123"));
    }

    #[test]
    fn redact_url_no_token_unchanged() {
        let url = "https://gitea.example.com/user/repo.git";
        let result = redact_url(url);
        assert_eq!(result, url);
    }

    #[test]
    fn redact_url_ssh_unchanged() {
        let url = "git@gitea.example.com:user/repo.git";
        let result = redact_url(url);
        assert_eq!(result, url);
    }

    #[test]
    fn redact_url_on_stderr_with_embedded_url() {
        let stderr = "fatal: could not read from remote repository 'https://ghp_secret@gitea.example.com/user/repo.git': not found";
        let result = redact_url(stderr);
        assert!(result.contains("https://***@gitea.example.com/user/repo.git"));
        assert!(!result.contains("ghp_secret"));
    }

    // --- ensure_repo tests ---

    #[tokio::test]
    async fn ensure_repo_short_circuits_when_git_exists() {
        let dir = tempfile::TempDir::new().unwrap();
        // Create a fake .git directory
        std::fs::create_dir(dir.path().join(".git")).unwrap();

        let lock = lock_git().await;
        let result = ensure_repo(
            &lock,
            "https://example.com/repo.git",
            "main",
            dir.path().to_str().unwrap(),
            None,
        )
        .await
        .unwrap();

        assert!(!result, "Should return false when .git already exists");
    }

    /// Helper: create a local bare git repo with one commit on the given branch.
    /// `pub(crate)` so `mcp.rs`'s write-tool tests can build a real repo fixture.
    pub(crate) fn create_bare_repo(branch: &str) -> tempfile::TempDir {
        let bare_dir = tempfile::TempDir::new().unwrap();
        let bare_path = bare_dir.path();

        // Init bare repo
        git_test_cmd(bare_path)
            .args(["init", "--bare", "--initial-branch", branch])
            .output()
            .unwrap();

        // Create a temporary working clone to make an initial commit
        let work_dir = tempfile::TempDir::new().unwrap();
        git_test_cmd(work_dir.path())
            .args(["clone", bare_path.to_str().unwrap(), "."])
            .output()
            .unwrap();
        git_test_cmd(work_dir.path())
            .args(["checkout", "-b", branch])
            .output()
            .unwrap();
        std::fs::write(work_dir.path().join("README.md"), "# Test repo").unwrap();
        git_test_cmd(work_dir.path())
            .args(["add", "README.md"])
            .output()
            .unwrap();
        git_test_cmd(work_dir.path())
            .args([
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "commit",
                "-m",
                "initial commit",
            ])
            .output()
            .unwrap();
        git_test_cmd(work_dir.path())
            .args(["push", "origin", branch])
            .output()
            .unwrap();

        bare_dir
    }

    #[tokio::test]
    async fn ensure_repo_clones_into_empty_dir() {
        let bare = create_bare_repo("main");
        let target = tempfile::TempDir::new().unwrap();
        let clone_path = target.path().join("repo");

        let lock = lock_git().await;
        let result = ensure_repo(
            &lock,
            bare.path().to_str().unwrap(),
            "main",
            clone_path.to_str().unwrap(),
            None,
        )
        .await
        .unwrap();

        assert!(result, "Should return true for fresh clone");
        assert!(
            clone_path.join(".git").exists(),
            ".git should exist after clone"
        );
        assert!(
            clone_path.join("README.md").exists(),
            "Cloned content should be present"
        );
    }

    #[tokio::test]
    async fn ensure_repo_creates_parent_dirs() {
        let bare = create_bare_repo("main");
        let target = tempfile::TempDir::new().unwrap();
        // Nested path that doesn't exist yet
        let clone_path = target.path().join("deeply/nested/repo");

        let lock = lock_git().await;
        let result = ensure_repo(
            &lock,
            bare.path().to_str().unwrap(),
            "main",
            clone_path.to_str().unwrap(),
            None,
        )
        .await
        .unwrap();

        assert!(result);
        assert!(clone_path.join(".git").exists());
    }

    #[tokio::test]
    async fn ensure_repo_idempotent_after_clone() {
        let bare = create_bare_repo("main");
        let target = tempfile::TempDir::new().unwrap();
        let clone_path = target.path().join("repo");

        let lock = lock_git().await;

        // First call — should clone
        let first = ensure_repo(
            &lock,
            bare.path().to_str().unwrap(),
            "main",
            clone_path.to_str().unwrap(),
            None,
        )
        .await
        .unwrap();
        assert!(first);

        // Second call — should short-circuit
        let second = ensure_repo(
            &lock,
            bare.path().to_str().unwrap(),
            "main",
            clone_path.to_str().unwrap(),
            None,
        )
        .await
        .unwrap();
        assert!(!second, "Second call should return false (already cloned)");
    }

    #[tokio::test]
    async fn ensure_repo_fails_on_nonempty_dir_without_git() {
        let bare = create_bare_repo("main");
        let target = tempfile::TempDir::new().unwrap();
        // Pre-populate directory with a file (but no .git)
        std::fs::write(target.path().join("stale-file.txt"), "leftover data").unwrap();

        let lock = lock_git().await;
        let result = ensure_repo(
            &lock,
            bare.path().to_str().unwrap(),
            "main",
            target.path().to_str().unwrap(),
            None,
        )
        .await;

        assert!(
            result.is_err(),
            "Should fail when dir is non-empty without .git"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("git clone failed"),
            "Error should mention clone failure, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn ensure_repo_fails_on_bad_branch() {
        let bare = create_bare_repo("main");
        let target = tempfile::TempDir::new().unwrap();
        let clone_path = target.path().join("repo");

        let lock = lock_git().await;
        let result = ensure_repo(
            &lock,
            bare.path().to_str().unwrap(),
            "nonexistent-branch",
            clone_path.to_str().unwrap(),
            None,
        )
        .await;

        assert!(result.is_err(), "Should fail when branch doesn't exist");
    }

    #[tokio::test]
    async fn ensure_repo_fails_on_bad_url() {
        let target = tempfile::TempDir::new().unwrap();
        let clone_path = target.path().join("repo");

        let lock = lock_git().await;
        let result = ensure_repo(
            &lock,
            "/nonexistent/path/to/repo.git",
            "main",
            clone_path.to_str().unwrap(),
            None,
        )
        .await;

        assert!(result.is_err(), "Should fail on invalid remote URL");
    }

    #[tokio::test]
    async fn ensure_repo_error_redacts_token_in_url() {
        let target = tempfile::TempDir::new().unwrap();
        let clone_path = target.path().join("repo");

        let lock = lock_git().await;
        let result = ensure_repo(
            &lock,
            "https://example.com/nonexistent/repo.git",
            "main",
            clone_path.to_str().unwrap(),
            Some("super_secret_token"),
        )
        .await;

        assert!(result.is_err());
        let err = format!("{:#}", result.unwrap_err());
        assert!(
            !err.contains("super_secret_token"),
            "Error message should not contain the raw token: {}",
            err
        );
    }

    #[test]
    fn inject_token_into_http_url() {
        let url = "http://gitea.local/user/repo.git";
        let result = inject_token_into_url(url, "tok123");
        assert_eq!(result, "http://tok123@gitea.local/user/repo.git");
    }

    #[test]
    fn redact_url_handles_http_scheme() {
        let url = "http://secret@gitea.local/user/repo.git";
        let result = redact_url(url);
        assert_eq!(result, "http://***@gitea.local/user/repo.git");
        assert!(!result.contains("secret"));
    }

    #[test]
    fn redact_url_handles_multiple_urls() {
        let s = "tried https://tok1@host1/r.git then https://tok2@host2/r.git";
        let result = redact_url(s);
        assert!(!result.contains("tok1"));
        assert!(!result.contains("tok2"));
        assert!(result.contains("https://***@host1/r.git"));
        assert!(result.contains("https://***@host2/r.git"));
    }

    // --- commit_and_sync tests ---

    /// Helper: create a local working clone of a bare repo.
    /// `pub(crate)` so `mcp.rs`'s write-tool tests can build a real repo fixture.
    pub(crate) fn clone_bare_repo(bare_path: &std::path::Path, branch: &str) -> tempfile::TempDir {
        let work_dir = tempfile::TempDir::new().unwrap();
        git_test_cmd(work_dir.path())
            .args(["clone", bare_path.to_str().unwrap(), "."])
            .output()
            .unwrap();
        // Ensure we're on the right branch
        git_test_cmd(work_dir.path())
            .args(["checkout", branch])
            .output()
            .unwrap();
        work_dir
    }

    /// Local-only path: commit a new file without any remote.
    #[tokio::test]
    async fn commit_and_sync_local_only() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        // Write a new file into the working repo
        std::fs::write(work.path().join("notes.md"), "# Notes\nHello world").unwrap();

        let lock = lock_git().await;
        let outcome = commit_and_sync(
            &lock,
            None,
            "main",
            work_path,
            None,
            &["notes.md"],
            "add notes.md\n\nmd-kb-rag bot commit",
            "test-bot",
            "test-bot@localhost",
        )
        .await
        .unwrap();
        let sha = &outcome.sha;

        // SHA should be a 40-char hex string
        assert_eq!(sha.len(), 40, "Expected a 40-char SHA, got: {}", sha);
        assert!(
            sha.chars().all(|c| c.is_ascii_hexdigit()),
            "SHA should be hex: {}",
            sha
        );
        assert!(
            outcome.rebased_paths.is_empty(),
            "no git_url means no fetch/rebase, so nothing to report"
        );

        // The file should be committed (git show HEAD should include it)
        let show_out = git_test_cmd(work_path)
            .args([
                "-c",
                &format!("safe.directory={}", work_path),
                "show",
                "--name-only",
                "--format=",
                "HEAD",
            ])
            .output()
            .unwrap();
        let show_str = String::from_utf8_lossy(&show_out.stdout);
        assert!(
            show_str.contains("notes.md"),
            "notes.md should appear in HEAD commit, got: {}",
            show_str
        );
    }

    /// Push path: commit a file and push to a local bare remote via file:// URL.
    #[tokio::test]
    async fn commit_and_sync_with_push_to_local_bare() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        // Use a file:// URL so git treats it like a real remote (allows push)
        let bare_url = format!("file://{}", bare.path().to_str().unwrap());

        // Write a new file into the working repo
        std::fs::write(work.path().join("article.md"), "# Article\nContent here").unwrap();

        let lock = lock_git().await;
        let outcome = commit_and_sync(
            &lock,
            Some(&bare_url),
            "main",
            work_path,
            None,
            &["article.md"],
            "add article.md",
            "test-bot",
            "test-bot@localhost",
        )
        .await
        .unwrap();

        assert_eq!(
            outcome.sha.len(),
            40,
            "Expected a 40-char SHA, got: {}",
            outcome.sha
        );
        assert!(
            outcome.rebased_paths.is_empty(),
            "nothing else landed on the remote between fetch and push"
        );

        // Verify the commit made it to the bare remote by cloning it fresh
        let verify_dir = tempfile::TempDir::new().unwrap();
        git_test_cmd(verify_dir.path())
            .args(["clone", bare.path().to_str().unwrap(), "."])
            .output()
            .unwrap();
        assert!(
            verify_dir.path().join("article.md").exists(),
            "article.md should exist in the remote after push"
        );
    }

    /// Rebase conflict: two clones diverge on the same file, second push must detect conflict.
    #[tokio::test]
    async fn commit_and_sync_rebase_conflict_returns_distinguishable_error() {
        let bare = create_bare_repo("main");
        let bare_url = format!("file://{}", bare.path().to_str().unwrap());

        let lock = lock_git().await;

        // Clone A: will push first
        let work_a = clone_bare_repo(bare.path(), "main");
        std::fs::write(work_a.path().join("conflict.md"), "version A").unwrap();
        commit_and_sync(
            &lock,
            Some(&bare_url),
            "main",
            work_a.path().to_str().unwrap(),
            None,
            &["conflict.md"],
            "add conflict.md from A",
            "test-bot",
            "test-bot@localhost",
        )
        .await
        .unwrap();

        // Clone B (made from the original bare *before* A pushed): will try to push
        // the same file with different content — rebase should conflict
        let work_b = clone_bare_repo(bare.path(), "main");
        // Manually reset work_b to the state before A's push by checking out the parent commit
        // Instead, simulate divergence: work_b was cloned before A pushed,
        // but since we clone after A pushed, we need to manually step back.
        // Simpler approach: make work_b commit to a *detached* state that doesn't include A's commit.
        // Actually the easiest way: create a second independent clone from the original state,
        // which we saved before A pushed. Since we can't travel back in time, instead:
        // - Let work_b clone from bare (which now has A's commit)
        // - Then use git reset --hard to go back to the parent and commit something diverging
        let log_out = git_test_cmd(work_b.path())
            .args(["log", "--format=%H", "-2"])
            .output()
            .unwrap();
        let commits: Vec<&str> = std::str::from_utf8(&log_out.stdout)
            .unwrap()
            .lines()
            .collect();
        // commits[0] = A's commit, commits[1] = initial commit
        let parent_sha = commits[1].trim();

        // Reset to before A's commit
        git_test_cmd(work_b.path())
            .args(["reset", "--hard", parent_sha])
            .output()
            .unwrap();

        // Now write a conflicting version of the same file
        std::fs::write(
            work_b.path().join("conflict.md"),
            "version B — conflicts with A",
        )
        .unwrap();

        let result = commit_and_sync(
            &lock,
            Some(&bare_url),
            "main",
            work_b.path().to_str().unwrap(),
            None,
            &["conflict.md"],
            "add conflict.md from B",
            "test-bot",
            "test-bot@localhost",
        )
        .await;

        // A rebase conflict happens after B's own commit has already landed locally
        // (fetch/rebase run after `git commit`), so this must be PostCommit, not
        // PreCommit — B's commit is real and must not be treated as discardable.
        let err = match result {
            Err(CommitSyncError::PostCommit { source, .. }) => source,
            other => panic!("expected CommitSyncError::PostCommit, got: {:?}", other),
        };
        let msg = err.to_string();
        assert!(
            msg.starts_with("rebase conflict:"),
            "Error should start with 'rebase conflict:', got: {}",
            msg
        );
    }

    /// Two clones each add a different file and push in turn; the second push must
    /// rebase in the first clone's commit, and `rebased_paths` must report the file
    /// THAT commit touched — not the one this call is committing itself.
    #[tokio::test]
    async fn commit_and_sync_reports_paths_pulled_in_by_the_rebase() {
        let bare = create_bare_repo("main");
        let bare_url = format!("file://{}", bare.path().to_str().unwrap());

        let lock = lock_git().await;

        // Clone A pushes first, adding other.md.
        let work_a = clone_bare_repo(bare.path(), "main");
        std::fs::write(work_a.path().join("other.md"), "from A").unwrap();
        commit_and_sync(
            &lock,
            Some(&bare_url),
            "main",
            work_a.path().to_str().unwrap(),
            None,
            &["other.md"],
            "add other.md from A",
            "test-bot",
            "test-bot@localhost",
        )
        .await
        .unwrap();

        // Clone B was cloned BEFORE A pushed (from the original bare state), so its
        // own commit_and_sync call must fetch + rebase onto A's commit to push.
        let work_b = clone_bare_repo(bare.path(), "main");
        let log_out = git_test_cmd(work_b.path())
            .args(["log", "--format=%H", "-2"])
            .output()
            .unwrap();
        let commits: Vec<&str> = std::str::from_utf8(&log_out.stdout)
            .unwrap()
            .lines()
            .collect();
        let parent_sha = commits[1].trim();
        git_test_cmd(work_b.path())
            .args(["reset", "--hard", parent_sha])
            .output()
            .unwrap();

        std::fs::write(work_b.path().join("mine.md"), "from B").unwrap();
        let outcome = commit_and_sync(
            &lock,
            Some(&bare_url),
            "main",
            work_b.path().to_str().unwrap(),
            None,
            &["mine.md"],
            "add mine.md from B",
            "test-bot",
            "test-bot@localhost",
        )
        .await
        .unwrap();

        assert_eq!(
            outcome.rebased_paths,
            vec![std::path::PathBuf::from("other.md")],
            "the rebase pulled in A's commit, which touched other.md — not mine.md, \
             which is B's own change and already known to the caller"
        );

        // A rebase REPLAYS commits, so it needs a committer identity of its own — the
        // identity used for the original `git commit` does not carry over. `rebase`
        // was previously invoked without one, which works on a developer machine with
        // a global git config and fails everywhere else with "Committer identity
        // unknown": CI, and the deployed container, which runs with HOME=/tmp and no
        // .gitconfig.
        //
        // Asserting on the replayed commit's committer catches a regression in BOTH
        // environments. Drop the identity again and a machine WITH a global config
        // silently stamps the ambient developer identity here (this assertion fails),
        // while a machine WITHOUT one fails the rebase outright (the unwrap above
        // fails). Asserting only that the call succeeded would catch it in CI alone,
        // and would pass locally while shipping the bug.
        let committer = git_test_cmd(work_b.path())
            .args(["log", "-1", "--format=%cn|%ce"])
            .output()
            .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&committer.stdout).trim(),
            "test-bot|test-bot@localhost",
            "the replayed commit must carry the committer identity commit_and_sync \
             was given, not whatever git config the host happens to have"
        );
    }

    /// #143 regression, exercised against a real `git diff` invocation rather
    /// than a hand-built string: git's default `core.quotepath=true` C-quotes
    /// and octal-escapes any path with non-ASCII bytes in `--name-status`
    /// output, so without `-z` this returned `"caf\303\251.md"` verbatim — a
    /// string matching no real file on disk — instead of `café.md`.
    #[tokio::test]
    async fn git_diff_name_status_handles_a_non_ascii_filename() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        let lock = lock_git().await;
        let old_head = rev_parse_head(&lock, work_path).await.unwrap();

        std::fs::write(work.path().join("café.md"), "content").unwrap();
        git_test_cmd(work_path)
            .args(["add", "café.md"])
            .output()
            .unwrap();
        git_test_cmd(work_path)
            .args([
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "commit",
                "-m",
                "add café.md",
            ])
            .output()
            .unwrap();
        let new_head = rev_parse_head(&lock, work_path).await.unwrap();

        let paths = git_diff_name_status(&lock, work_path, &old_head, &new_head)
            .await
            .unwrap();
        assert_eq!(
            paths,
            vec![std::path::PathBuf::from("café.md")],
            "the real, unmangled filename must come back — not git's quoted/escaped form"
        );
    }

    // --- parse_git_log_mtimes tests (#164) ------------------------------------
    //
    // Pure byte-parsing, no git subprocess — the byte layouts below were verified
    // empirically against a real `git log --format=%x02%ct%x03 --name-only
    // --diff-filter=ACMR -z` invocation (see `git_log_mtimes`'s doc comment), not
    // guessed from documentation.

    #[test]
    fn parse_git_log_mtimes_empty_input_yields_empty_map() {
        assert!(parse_git_log_mtimes(b"").is_empty());
    }

    #[test]
    fn parse_git_log_mtimes_one_commit_multiple_files() {
        // `\x02<ts>\x03` NUL `\n<path1>` NUL `<path2>` NUL — one commit's format
        // token followed by both files it touched.
        let raw = b"\x021700000000\x03\0\na/one.md\0a/two.md\0";
        let map = parse_git_log_mtimes(raw);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("a/one.md"), Some(&1_700_000_000));
        assert_eq!(map.get("a/two.md"), Some(&1_700_000_000));
    }

    #[test]
    fn parse_git_log_mtimes_first_occurrence_wins_as_the_most_recent_touch() {
        // git log's default order is newest-first, so a path appearing under two
        // different commits' timestamps must keep the FIRST (newest) one.
        let raw = b"\x021700000200\x03\0\na.md\0\x021700000100\x03\0\na.md\0";
        let map = parse_git_log_mtimes(raw);
        assert_eq!(map.get("a.md"), Some(&1_700_000_200));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn parse_git_log_mtimes_multiple_commits_each_get_their_own_paths() {
        let raw = b"\x021700000300\x03\0\nnew.md\0\x021700000100\x03\0\nold.md\0";
        let map = parse_git_log_mtimes(raw);
        assert_eq!(map.get("new.md"), Some(&1_700_000_300));
        assert_eq!(map.get("old.md"), Some(&1_700_000_100));
    }

    #[test]
    fn parse_git_log_mtimes_tolerates_back_to_back_timestamp_tokens() {
        // Defensive: a commit whose formatted header prints but whose diff (under
        // `--diff-filter`) is empty — never actually observed against real git in
        // this codebase's testing (merge and filtered-out commits are omitted
        // entirely, not printed with an empty file list — see the doc comment on
        // `git_log_mtimes`), but the parser must not panic or misattribute a path
        // to the wrong commit if it ever does happen.
        let raw = b"\x021700000300\x03\0\x021700000100\x03\0\nold.md\0";
        let map = parse_git_log_mtimes(raw);
        assert_eq!(map.get("old.md"), Some(&1_700_000_100));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn parse_git_log_mtimes_handles_non_ascii_paths() {
        let raw = "\x021700000000\x03\0\ncafé.md\0".as_bytes();
        let map = parse_git_log_mtimes(raw);
        assert_eq!(map.get("café.md"), Some(&1_700_000_000));
    }

    // --- git_log_mtimes tests (#164) -------------------------------------------

    #[tokio::test]
    async fn git_log_mtimes_returns_empty_map_for_an_empty_path_list() {
        let lock = lock_git().await;
        // No repo needed at all — an empty `paths` short-circuits before spawning
        // git, so even a nonexistent directory is fine here.
        let map = git_log_mtimes(&lock, "/nonexistent/does/not/matter", &[]).await;
        assert!(map.is_empty());
    }

    #[tokio::test]
    async fn git_log_mtimes_returns_each_paths_most_recent_commit_time() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        // Commit 1: add both files.
        std::fs::write(work.path().join("a.md"), "a v1").unwrap();
        std::fs::write(work.path().join("b.md"), "b v1").unwrap();
        git_test_cmd(work_path)
            .args(["add", "a.md", "b.md"])
            .output()
            .unwrap();
        git_test_cmd(work_path)
            .args([
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "commit",
                "-m",
                "add a.md and b.md",
            ])
            // `@<unix-seconds> <tz-offset>` is git's own unambiguous date format —
            // deliberately not an ISO string with no offset, which git parses in
            // the LOCAL timezone and would make the expected epoch value below
            // depend on whatever TZ the test happens to run under.
            .env("GIT_AUTHOR_DATE", "@1577836800 +0000")
            .env("GIT_COMMITTER_DATE", "@1577836800 +0000")
            .output()
            .unwrap();

        // Commit 2: touch only a.md, well after commit 1 — this is the case #164
        // exists for: a.md's true last-modified time must move; b.md's must not.
        std::fs::write(work.path().join("a.md"), "a v2").unwrap();
        git_test_cmd(work_path)
            .args(["add", "a.md"])
            .output()
            .unwrap();
        git_test_cmd(work_path)
            .args([
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "commit",
                "-m",
                "update a.md",
            ])
            .env("GIT_AUTHOR_DATE", "@1623758400 +0000")
            .env("GIT_COMMITTER_DATE", "@1623758400 +0000")
            .output()
            .unwrap();

        let lock = lock_git().await;
        let map = git_log_mtimes(&lock, work_path, &["a.md".to_string(), "b.md".to_string()]).await;

        // 2020-01-01T00:00:00Z and 2021-06-15T12:00:00Z as Unix seconds.
        assert_eq!(
            map.get("a.md"),
            Some(&1_623_758_400),
            "a.md's most recent touch is commit 2"
        );
        assert_eq!(
            map.get("b.md"),
            Some(&1_577_836_800),
            "b.md was never touched again after commit 1"
        );
    }

    #[tokio::test]
    async fn git_log_mtimes_omits_a_path_with_no_history() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        let lock = lock_git().await;
        // "untracked.md" was never committed — the caller must fall back to
        // filesystem mtime for it, which only works if this function leaves it
        // out of the map entirely rather than inventing a value.
        let map = git_log_mtimes(&lock, work_path, &["untracked.md".to_string()]).await;
        assert!(
            !map.contains_key("untracked.md"),
            "a path git has no history for must be absent, not defaulted to 0 or similar"
        );
    }

    // --- chunk_paths_by_byte_budget tests (#237) -------------------------------

    #[test]
    fn chunk_paths_by_byte_budget_splits_once_the_running_total_exceeds_the_budget() {
        // 5 paths of 25_000 bytes each (cost 25_001 with the +1 separator
        // allowance). Budget is 100_000: three entries fit (75_003 used), a fourth
        // would push it to 100_004 — over budget — so the split lands right before
        // it. The 4th and 5th then start a fresh chunk together (50_002, still under
        // budget).
        let paths: Vec<String> = (0..5).map(|_| "x".repeat(25_000)).collect();
        let chunks = chunk_paths_by_byte_budget(&paths);
        assert_eq!(
            chunks.iter().map(|c| c.len()).collect::<Vec<_>>(),
            vec![3, 2],
            "expected a 3-entry chunk then a 2-entry chunk, got chunk sizes {:?}",
            chunks.iter().map(|c| c.len()).collect::<Vec<_>>()
        );
        // Every input path must still be present somewhere, in order, across the
        // chunks — chunking must never drop or reorder a path.
        let flattened: Vec<&String> = chunks.into_iter().flatten().collect();
        assert_eq!(flattened, paths.iter().collect::<Vec<_>>());
    }

    #[test]
    fn chunk_paths_by_byte_budget_keeps_a_single_oversized_path_in_its_own_chunk() {
        // A path alone longer than the whole budget must not be dropped — it gets a
        // one-entry chunk of its own rather than being silently excluded, and the
        // small paths that follow it still get chunked together normally.
        let huge = "x".repeat(200_000);
        let paths = vec![huge.clone(), "a.md".to_string(), "b.md".to_string()];
        let chunks = chunk_paths_by_byte_budget(&paths);
        assert_eq!(
            chunks.iter().map(|c| c.len()).collect::<Vec<_>>(),
            vec![1, 2],
            "the oversized path must stand alone in its own chunk, not merged with \
             or dropped in favor of the smaller paths that follow it"
        );
        assert_eq!(chunks[0], [huge]);
        assert_eq!(chunks[1], ["a.md".to_string(), "b.md".to_string()]);
    }

    #[test]
    fn chunk_paths_by_byte_budget_empty_input_yields_no_chunks() {
        let paths: Vec<String> = vec![];
        assert!(chunk_paths_by_byte_budget(&paths).is_empty());
    }

    // --- git_log_mtimes ARG_MAX chunking tests (#237) ---------------------------

    /// The regression test for #237 itself: a path list whose TOTAL byte length
    /// comfortably exceeds a real OS `ARG_MAX` (this environment's `getconf ARG_MAX`
    /// is 2_097_152 bytes; the decoy paths below total roughly twice that) must not
    /// make the whole lookup come back empty.
    ///
    /// `git_log_mtimes_chunk` — the exact command-build/spawn logic the pre-#237
    /// `git_log_mtimes` used to run ONCE against the whole list — is called directly
    /// first, to prove the underlying OS limit is real and would have broken the
    /// unbatched implementation (this is the "fails before" half): a single argv
    /// this large cannot even be spawned, so it returns `Err`. `git_log_mtimes`
    /// itself is then called with the identical oversized list and must still return
    /// the correct answer for the one real tracked path folded in among the decoys —
    /// proof that #237's chunking is what makes the difference, not some other
    /// change to the fixture.
    #[tokio::test]
    async fn git_log_mtimes_survives_a_path_list_that_would_exceed_arg_max_unchunked() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        std::fs::write(work.path().join("a.md"), "content").unwrap();
        git_test_cmd(work_path)
            .args(["add", "a.md"])
            .output()
            .unwrap();
        git_test_cmd(work_path)
            .args([
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "commit",
                "-m",
                "add a.md",
            ])
            .env("GIT_AUTHOR_DATE", "@1600000000 +0000")
            .env("GIT_COMMITTER_DATE", "@1600000000 +0000")
            .output()
            .unwrap();

        // 6 decoy paths of 700_000 bytes each (4.2MB total) — none of them exist on
        // disk or in git history, which is fine: `git log -- <pathspec>` does not
        // require a pathspec entry to match anything, it just filters. Their only
        // job is to make the combined argv large enough to blow past real
        // `ARG_MAX` regardless of this environment's exact value.
        let mut paths: Vec<String> = (0..6)
            .map(|i| format!("decoy-{i}-{}", "x".repeat(700_000)))
            .collect();
        paths.push("a.md".to_string());

        let lock = lock_git().await;

        // "Before" half: the unbatched single-invocation logic really does fail
        // against this input.
        let unchunked = git_log_mtimes_chunk(&lock, work_path, &paths).await;
        assert!(
            unchunked.is_err(),
            "a ~4MB single argv should exceed this environment's real ARG_MAX and \
             fail to spawn — if this assertion itself fails, the fixture no longer \
             proves anything about #237's fix"
        );

        // "After" half: the chunked, public entry point still gets the right answer.
        let map = git_log_mtimes(&lock, work_path, &paths).await;
        assert_eq!(
            map.get("a.md"),
            Some(&1_600_000_000),
            "a.md's git-derived mtime must still come through despite the oversized \
             combined path list, because #237 chunks it below ARG_MAX per invocation"
        );
    }

    /// #237's "partial failure degrades its own batch, not the whole run": one
    /// pathological chunk (here, a single path so long it alone exceeds real
    /// `ARG_MAX` and lands in its own one-entry chunk — see
    /// `chunk_paths_by_byte_budget_keeps_a_single_oversized_path_in_its_own_chunk`)
    /// must not blank out the results of every OTHER, healthy chunk.
    #[tokio::test]
    async fn git_log_mtimes_a_failing_chunk_does_not_blank_out_other_chunks_results() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        std::fs::write(work.path().join("a.md"), "content").unwrap();
        git_test_cmd(work_path)
            .args(["add", "a.md"])
            .output()
            .unwrap();
        git_test_cmd(work_path)
            .args([
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "commit",
                "-m",
                "add a.md",
            ])
            .env("GIT_AUTHOR_DATE", "@1600000000 +0000")
            .env("GIT_COMMITTER_DATE", "@1600000000 +0000")
            .output()
            .unwrap();

        // A single ~3MB path, alone, exceeds real ARG_MAX on its own — per the byte
        // budget it forms its own one-entry chunk (never merged with `a.md`), so
        // that chunk's `git log` invocation fails to spawn while `a.md`'s separate
        // chunk succeeds normally.
        let huge_path = "x".repeat(3_000_000);
        let paths = vec![huge_path, "a.md".to_string()];

        let lock = lock_git().await;
        let map = git_log_mtimes(&lock, work_path, &paths).await;

        assert_eq!(
            map.get("a.md"),
            Some(&1_600_000_000),
            "the healthy chunk's result must survive the other chunk's failure"
        );
        assert_eq!(
            map.len(),
            1,
            "the oversized path itself must simply be absent (falls back to \
             filesystem mtime at the caller), not present with some invented value"
        );
    }

    /// Two independent new files, committed together via a two-element `paths`
    /// slice, must land as a single commit that contains both — not two commits,
    /// and not one commit missing either file.
    #[tokio::test]
    async fn commit_and_sync_commits_multiple_paths_in_one_commit() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        std::fs::write(work.path().join("one.md"), "# One").unwrap();
        std::fs::write(work.path().join("two.md"), "# Two").unwrap();

        let lock = lock_git().await;
        let head_before = rev_parse_head(&lock, work_path).await.unwrap();

        let outcome = commit_and_sync(
            &lock,
            None,
            "main",
            work_path,
            None,
            &["one.md", "two.md"],
            "add one.md and two.md",
            "test-bot",
            "test-bot@localhost",
        )
        .await
        .unwrap();

        let show_out = git_test_cmd(work_path)
            .args(["show", "--name-only", "--format=", "HEAD"])
            .output()
            .unwrap();
        let show_str = String::from_utf8_lossy(&show_out.stdout);
        assert!(
            show_str.contains("one.md") && show_str.contains("two.md"),
            "both paths should appear in the single commit, got: {show_str}"
        );

        let log = git_test_cmd(work_path)
            .args([
                "rev-list",
                "--count",
                &format!("{head_before}..{}", outcome.sha),
            ])
            .output()
            .unwrap();
        let count = String::from_utf8_lossy(&log.stdout).trim().to_string();
        assert_eq!(
            count, "1",
            "HEAD should have advanced by exactly one commit, got {count}"
        );
    }

    /// A move-shaped change — delete an existing tracked file, create a new one —
    /// committed in a single `commit_and_sync` call with both paths must produce
    /// exactly one commit in which the old path is gone and the new path exists.
    #[tokio::test]
    async fn commit_and_sync_commits_a_move_shaped_change_as_one_commit() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        // Seed and commit the file that will be "moved" away from.
        std::fs::write(work.path().join("old.md"), "# Content").unwrap();
        let lock = lock_git().await;
        commit_and_sync(
            &lock,
            None,
            "main",
            work_path,
            None,
            &["old.md"],
            "seed old.md",
            "test-bot",
            "test-bot@localhost",
        )
        .await
        .unwrap();
        let head_before_move = rev_parse_head(&lock, work_path).await.unwrap();

        // Move-shaped change: remove old.md from disk, add new.md.
        std::fs::remove_file(work.path().join("old.md")).unwrap();
        std::fs::write(work.path().join("new.md"), "# Content").unwrap();

        let outcome = commit_and_sync(
            &lock,
            None,
            "main",
            work_path,
            None,
            &["old.md", "new.md"],
            "move old.md to new.md",
            "test-bot",
            "test-bot@localhost",
        )
        .await
        .unwrap();

        let log = git_test_cmd(work_path)
            .args([
                "rev-list",
                "--count",
                &format!("{head_before_move}..{}", outcome.sha),
            ])
            .output()
            .unwrap();
        let count = String::from_utf8_lossy(&log.stdout).trim().to_string();
        assert_eq!(
            count, "1",
            "the move should land as exactly one commit, got {count}"
        );

        let name_status = git_test_cmd(work_path)
            .args(["show", "--name-status", "--format=", "HEAD"])
            .output()
            .unwrap();
        let name_status_str = String::from_utf8_lossy(&name_status.stdout);
        assert!(
            name_status_str.contains("old.md"),
            "old.md should be reflected as removed in the commit, got: {name_status_str}"
        );
        assert!(
            name_status_str.contains("new.md"),
            "new.md should be reflected as added in the commit, got: {name_status_str}"
        );
        assert!(
            !work.path().join("old.md").exists(),
            "old.md should be gone from the working tree"
        );
        assert!(
            work.path().join("new.md").exists(),
            "new.md should exist in the working tree"
        );
    }

    /// An empty `paths` slice must be rejected before any git command runs — an
    /// unscoped `git commit --` with no pathspec would commit the ENTIRE index,
    /// which is exactly what the pathspec-scoping exists to prevent.
    #[tokio::test]
    async fn commit_and_sync_empty_paths_returns_precommit_and_creates_no_commit() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        let lock = lock_git().await;
        let head_before = rev_parse_head(&lock, work_path).await.unwrap();

        let result = commit_and_sync(
            &lock,
            None,
            "main",
            work_path,
            None,
            &[],
            "empty paths",
            "test-bot",
            "test-bot@localhost",
        )
        .await;

        match result {
            Err(CommitSyncError::PreCommit(_)) => {}
            other => panic!("expected CommitSyncError::PreCommit, got: {:?}", other),
        }

        let head_after = rev_parse_head(&lock, work_path).await.unwrap();
        assert_eq!(
            head_before, head_after,
            "an empty-paths call must not create a commit or move HEAD"
        );
    }

    /// One valid path plus one path that neither exists on disk nor was deleted
    /// from HEAD (a typo'd destination in a move, say) — `git add` fails on the
    /// bad pathspec before staging anything, so this must fail cleanly as
    /// `PreCommit` with HEAD untouched, not half-commit the valid path.
    #[tokio::test]
    async fn commit_and_sync_multi_path_with_one_bad_path_returns_precommit_and_creates_no_commit()
    {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        // real.md genuinely exists and would stage cleanly on its own; typo.md was
        // never written and isn't tracked at HEAD either.
        std::fs::write(work.path().join("real.md"), "# Real").unwrap();

        let lock = lock_git().await;
        let head_before = rev_parse_head(&lock, work_path).await.unwrap();

        let result = commit_and_sync(
            &lock,
            None,
            "main",
            work_path,
            None,
            &["real.md", "typo.md"],
            "move real.md to typo.md",
            "test-bot",
            "test-bot@localhost",
        )
        .await;

        match result {
            Err(CommitSyncError::PreCommit(_)) => {}
            other => panic!("expected CommitSyncError::PreCommit, got: {:?}", other),
        }

        let head_after = rev_parse_head(&lock, work_path).await.unwrap();
        assert_eq!(
            head_before, head_after,
            "a bad path anywhere in the slice must not move HEAD, even when other \
             paths in the same slice are individually valid"
        );

        // `git add` fails on the whole pathspec atomically — nothing from this
        // call, including the otherwise-valid real.md, ends up staged.
        let staged = git_out(work_path, &["diff", "--cached", "--name-only"]);
        assert!(
            staged.trim().is_empty(),
            "no path should be left staged after a failed multi-path add, staged: {staged}"
        );
    }

    /// The same path appearing twice in the slice (e.g. a caller that doesn't
    /// dedup before calling) must be harmless: `git add`/`git commit` both accept
    /// a repeated pathspec, and this must still land as one normal commit.
    #[tokio::test]
    async fn commit_and_sync_duplicate_paths_in_slice_produce_one_normal_commit() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        std::fs::write(work.path().join("dup.md"), "# Dup").unwrap();

        let lock = lock_git().await;
        let head_before = rev_parse_head(&lock, work_path).await.unwrap();

        let outcome = commit_and_sync(
            &lock,
            None,
            "main",
            work_path,
            None,
            &["dup.md", "dup.md"],
            "add dup.md",
            "test-bot",
            "test-bot@localhost",
        )
        .await
        .unwrap();

        let log = git_test_cmd(work_path)
            .args([
                "rev-list",
                "--count",
                &format!("{head_before}..{}", outcome.sha),
            ])
            .output()
            .unwrap();
        let count = String::from_utf8_lossy(&log.stdout).trim().to_string();
        assert_eq!(
            count, "1",
            "a duplicated path must still land as exactly one commit, got {count}"
        );

        let show_str = git_out(work_path, &["show", "--name-only", "--format=", "HEAD"]);
        assert!(
            show_str.contains("dup.md"),
            "dup.md should be committed once, got: {show_str}"
        );
    }

    /// The pathspec-scoping property in
    /// `commit_is_scoped_to_its_path_and_ignores_unrelated_staged_entries` is the
    /// whole reason the commit's pathspec exists, and must still hold once that
    /// pathspec covers 2+ paths — an unrelated staged entry must not ride along
    /// just because this call is committing a move instead of a single file.
    #[tokio::test]
    async fn commit_and_sync_multi_path_commit_ignores_unrelated_staged_entries() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        std::fs::write(work.path().join("one.md"), "# One").unwrap();
        std::fs::write(work.path().join("two.md"), "# Two").unwrap();
        std::fs::write(work.path().join("stray.md"), "# Stray").unwrap();

        // Residue from something other than this write.
        git_out(work_path, &["add", "--", "stray.md"]);

        let lock = lock_git().await;
        commit_and_sync(
            &lock,
            None,
            "main",
            work_path,
            None,
            &["one.md", "two.md"],
            "add one.md and two.md",
            "test-bot",
            "test-bot@localhost",
        )
        .await
        .unwrap();
        drop(lock);

        let head = git_out(work_path, &["show", "--name-only", "--format=", "HEAD"]);
        assert!(
            head.contains("one.md") && head.contains("two.md"),
            "own paths should be committed: {head}"
        );
        assert!(
            !head.contains("stray.md"),
            "an unrelated staged entry must NOT ride along in a multi-path commit: {head}"
        );

        // The stray is left exactly as it was — scoping the commit neither commits
        // nor discards someone else's staged work.
        let staged = git_out(work_path, &["diff", "--cached", "--name-only"]);
        assert!(
            staged.contains("stray.md"),
            "unrelated staged entry should be untouched, staged: {staged}"
        );
    }

    /// A multi-path (move-shaped) commit that needs a rebase to push — the rebase
    /// itself must succeed, and `rebased_paths` must still report exactly the OTHER
    /// commit's path, not be corrupted by this call's own multi-element pathspec.
    #[tokio::test]
    async fn commit_and_sync_rebase_succeeds_around_a_multi_path_commit() {
        let bare = create_bare_repo("main");
        let bare_url = format!("file://{}", bare.path().to_str().unwrap());
        let lock = lock_git().await;

        // Clone A pushes first, adding other.md — this is the commit the rebase
        // below must pull in.
        let work_a = clone_bare_repo(bare.path(), "main");
        std::fs::write(work_a.path().join("other.md"), "from A").unwrap();
        commit_and_sync(
            &lock,
            Some(&bare_url),
            "main",
            work_a.path().to_str().unwrap(),
            None,
            &["other.md"],
            "add other.md from A",
            "test-bot",
            "test-bot@localhost",
        )
        .await
        .unwrap();

        // Clone B, rewound to before A's commit, seeds old.md directly (not
        // through commit_and_sync — no push yet) so it's a tracked file at HEAD
        // that the move below can move away from.
        let work_b = clone_bare_repo(bare.path(), "main");
        let work_b_path = work_b.path().to_str().unwrap();
        let commits = git_out(work_b_path, &["log", "--format=%H", "-2"]);
        let parent_sha = commits.lines().nth(1).unwrap().trim().to_string();
        git_out(work_b_path, &["reset", "--hard", &parent_sha]);

        std::fs::write(work_b.path().join("old.md"), "# Content").unwrap();
        git_out(work_b_path, &["add", "old.md"]);
        git_out(
            work_b_path,
            &[
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "commit",
                "-m",
                "seed old.md",
            ],
        );

        // Still unaware of A's push, perform the move-shaped multi-path commit;
        // commit_and_sync's fetch+rebase must pull A's other.md in around it.
        std::fs::remove_file(work_b.path().join("old.md")).unwrap();
        std::fs::write(work_b.path().join("new.md"), "# Content").unwrap();

        let outcome = commit_and_sync(
            &lock,
            Some(&bare_url),
            "main",
            work_b_path,
            None,
            &["old.md", "new.md"],
            "move old.md to new.md",
            "test-bot",
            "test-bot@localhost",
        )
        .await
        .unwrap();

        assert_eq!(
            outcome.rebased_paths,
            vec![std::path::PathBuf::from("other.md")],
            "the rebase pulled in A's commit, which touched other.md — a multi-path \
             move commit must not corrupt rebased_paths accounting"
        );

        let name_status = git_out(work_b_path, &["show", "--name-status", "--format=", "HEAD"]);
        assert!(
            name_status.contains("old.md") && name_status.contains("new.md"),
            "the replayed move commit should still show both paths, got: {name_status}"
        );
        assert!(
            !work_b.path().join("old.md").exists(),
            "old.md should be gone from the working tree after the rebase"
        );
        assert!(
            work_b.path().join("new.md").exists(),
            "new.md should exist in the working tree after the rebase"
        );
        assert!(
            work_b.path().join("other.md").exists(),
            "A's rebased-in commit should also be present in the working tree"
        );
    }

    // --- CommitSyncError phase-distinction tests ---

    /// `git add` fails (the path was never written to disk) — nothing is committed,
    /// and this must surface as `PreCommit`, with HEAD left exactly where it was.
    #[tokio::test]
    async fn commit_and_sync_precommit_failure_is_distinguishable_and_leaves_head_unchanged() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        let lock = lock_git().await;
        let head_before = rev_parse_head(&lock, work_path).await.unwrap();

        // "missing.md" was never written into the working tree, so `git add` has
        // nothing to stage and fails before any commit is attempted.
        let result = commit_and_sync(
            &lock,
            None,
            "main",
            work_path,
            None,
            &["missing.md"],
            "add missing.md",
            "test-bot",
            "test-bot@localhost",
        )
        .await;

        match result {
            Err(CommitSyncError::PreCommit(_)) => {}
            other => panic!("expected CommitSyncError::PreCommit, got: {:?}", other),
        }

        let head_after = rev_parse_head(&lock, work_path).await.unwrap();
        assert_eq!(
            head_before, head_after,
            "a PreCommit failure must not move HEAD"
        );
    }

    /// The commit lands locally, but the remote is unreachable (fetch fails). This
    /// must surface as `PostCommit` carrying the real local sha, and — the whole
    /// point of the distinction — the commit must still be sitting in the local repo
    /// afterward, not rolled back.
    #[tokio::test]
    async fn commit_and_sync_postcommit_failure_leaves_commit_in_place() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        std::fs::write(work.path().join("article.md"), "content").unwrap();

        let lock = lock_git().await;
        let result = commit_and_sync(
            &lock,
            // No such path — fetch fails immediately, no network required.
            Some("/nonexistent/path/to/repo.git"),
            "main",
            work_path,
            None,
            &["article.md"],
            "add article.md",
            "test-bot",
            "test-bot@localhost",
        )
        .await;

        let sha = match result {
            Err(CommitSyncError::PostCommit { sha, .. }) => sha,
            other => panic!("expected CommitSyncError::PostCommit, got: {:?}", other),
        };
        assert_eq!(sha.len(), 40, "expected a 40-char SHA, got: {}", sha);

        // The commit must actually be present in the local repo's history — a
        // PostCommit failure must never be rolled back.
        let show_out = git_test_cmd(work_path)
            .args(["show", "--name-only", "--format=", "HEAD"])
            .output()
            .unwrap();
        let show_str = String::from_utf8_lossy(&show_out.stdout);
        assert!(
            show_str.contains("article.md"),
            "the local commit must still exist after a post-commit sync failure, got: {}",
            show_str
        );
        let head = rev_parse_head(&lock, work_path).await.unwrap();
        assert_eq!(
            head, sha,
            "the sha attached to the error must match the real local HEAD"
        );
    }

    /// A push failure must redact any token embedded in the remote URL from the
    /// `PostCommit` error, exactly like the pre-existing clone/fetch paths.
    #[tokio::test]
    async fn commit_and_sync_postcommit_error_redacts_token() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        std::fs::write(work.path().join("secret.md"), "content").unwrap();

        let lock = lock_git().await;
        let result = commit_and_sync(
            &lock,
            Some("https://example.com/nonexistent/repo.git"),
            "main",
            work_path,
            Some("super_secret_token"),
            &["secret.md"],
            "add secret.md",
            "test-bot",
            "test-bot@localhost",
        )
        .await;

        let source = match result {
            Err(CommitSyncError::PostCommit { source, .. }) => source,
            other => panic!("expected CommitSyncError::PostCommit, got: {:?}", other),
        };
        let msg = format!("{:#}", source);
        assert!(
            !msg.contains("super_secret_token"),
            "token must be redacted from the PostCommit cause: {}",
            msg
        );

        // The enum's own Display must not leak it either — that's what a caller
        // actually logs/returns.
        let full = CommitSyncError::PostCommit {
            sha: "deadbeef".to_string(),
            source,
        }
        .to_string();
        assert!(!full.contains("super_secret_token"));
    }

    /// #140 regression: when the rebase replays our commit onto a NEW sha (the
    /// normal case whenever another writer already pushed — exactly the
    /// scenario `commit_and_sync_reports_paths_pulled_in_by_the_rebase` above
    /// sets up) and the subsequent push then fails, the `PostCommit` error must
    /// report that new, post-rebase sha — not the pre-rebase sha, which the
    /// replay has already made unreachable.
    ///
    /// Making the bare remote's directory tree read-only after A's push (but
    /// before B's call) is what makes the push fail deterministically while
    /// leaving `fetch` (upload-pack, read-only, writes nothing under the repo)
    /// unaffected — `git receive-pack` needs to create objects/lock refs and
    /// fails outright without write permission, so the rebase genuinely
    /// completes locally before the push failure hits.
    #[tokio::test]
    async fn commit_and_sync_postcommit_push_failure_after_rebase_reports_post_rebase_sha() {
        let bare = create_bare_repo("main");
        let bare_url = format!("file://{}", bare.path().to_str().unwrap());

        let lock = lock_git().await;

        // Clone A pushes first, so clone B's own commit below must fetch + rebase
        // onto A's commit before it can (attempt to) push.
        let work_a = clone_bare_repo(bare.path(), "main");
        std::fs::write(work_a.path().join("other.md"), "from A").unwrap();
        commit_and_sync(
            &lock,
            Some(&bare_url),
            "main",
            work_a.path().to_str().unwrap(),
            None,
            &["other.md"],
            "add other.md from A",
            "test-bot",
            "test-bot@localhost",
        )
        .await
        .unwrap();

        // Clone B, rewound to before A's push, so its own `commit_and_sync` call
        // below must fetch + rebase onto A's commit before attempting to push.
        let work_b = clone_bare_repo(bare.path(), "main");
        let log_out = git_test_cmd(work_b.path())
            .args(["log", "--format=%H", "-2"])
            .output()
            .unwrap();
        let commits: Vec<&str> = std::str::from_utf8(&log_out.stdout)
            .unwrap()
            .lines()
            .collect();
        let parent_sha = commits[1].trim();
        git_test_cmd(work_b.path())
            .args(["reset", "--hard", parent_sha])
            .output()
            .unwrap();

        std::fs::write(work_b.path().join("mine.md"), "from B").unwrap();

        // Make the bare remote reject pushes now, AFTER A's push — `git push`
        // (receive-pack) runs the `pre-receive` hook and fails on its non-zero
        // exit, while `git fetch` (upload-pack) never runs that hook and is
        // unaffected. That asymmetry is the point: `commit_and_sync` must get
        // far enough to fetch and rebase before the push fails.
        reject_pushes(bare.path());

        let result = commit_and_sync(
            &lock,
            Some(&bare_url),
            "main",
            work_b.path().to_str().unwrap(),
            None,
            &["mine.md"],
            "add mine.md from B",
            "test-bot",
            "test-bot@localhost",
        )
        .await;

        let sha = match result {
            Err(CommitSyncError::PostCommit { sha, source }) => {
                let msg = format!("{:#}", source);
                assert!(
                    msg.contains("git push failed"),
                    "expected a push failure, got: {}",
                    msg
                );
                sha
            }
            other => panic!("expected CommitSyncError::PostCommit, got: {:?}", other),
        };

        // The rebase really did replay B's commit onto a new sha — HEAD in B's
        // clone must be the reported sha, and that sha must actually exist as a
        // commit (proving it is real, not a placeholder).
        let head = rev_parse_head(&lock, work_b.path().to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(
            sha, head,
            "the reported sha must be the CURRENT local HEAD (the post-rebase \
             replay), not the pre-rebase sha the replay orphaned"
        );
        let cat_file = git_test_cmd(work_b.path())
            .args(["cat-file", "-e", &format!("{sha}^{{commit}}")])
            .output()
            .unwrap();
        assert!(
            cat_file.status.success(),
            "the reported sha must resolve to a real, findable commit"
        );
    }

    /// Install a `pre-receive` hook on a bare repo that rejects every push.
    ///
    /// Failure injection for the push-failure test above. This deliberately does
    /// NOT work by making the remote's tree read-only: `receive-pack` writing to
    /// a `chmod -w` directory is only blocked for an unprivileged user, and CI
    /// here runs on a self-hosted runner privileged enough to write anyway
    /// (root, or anything holding `CAP_DAC_OVERRIDE`, ignores the permission
    /// bits outright). That made the read-only version pass locally and fail in
    /// CI, where the push simply succeeded and no `PostCommit` error was ever
    /// produced.
    ///
    /// A `pre-receive` hook is privilege-independent: git runs it and honours a
    /// non-zero exit no matter who the pushing process is. It also targets
    /// exactly the operation under test — `fetch` (upload-pack) never runs
    /// `pre-receive`, so the fetch-then-rebase half of `commit_and_sync` still
    /// succeeds, which is precisely the sequence this test needs.
    ///
    /// Unix-only, same as the rest of this test module's assumptions
    /// (`create_bare_repo` et al. already shell out to a real `git`, which this
    /// project only targets on Linux).
    #[cfg(unix)]
    fn reject_pushes(bare: &std::path::Path) {
        use std::os::unix::fs::PermissionsExt;
        let hooks_dir = bare.join("hooks");
        let hook = hooks_dir.join("pre-receive");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        std::fs::write(
            &hook,
            "#!/bin/sh\necho 'pushes rejected by test' >&2\nexit 1\n",
        )
        .unwrap();

        // Pin the repo's hooks directory explicitly. A `core.hooksPath` in the
        // developer's or runner's GLOBAL git config silently replaces a repo's
        // own `hooks/` for every repo on the machine, which would leave this
        // hook dead and let the test pass a push it exists to reject. This
        // project's own `scripts/setup-dev.sh` sets `core.hooksPath`
        // (repo-locally), and setting it globally is a common enough habit that
        // the test must not depend on its absence. Repo-local config wins over
        // global, so writing it here makes the hook fire either way.
        git_test_cmd(bare)
            .args(["config", "core.hooksPath", hooks_dir.to_str().unwrap()])
            .output()
            .unwrap();
        std::fs::set_permissions(&hook, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    // --- restore_from_head / unstage tests ---

    /// `restore_from_head` is the rollback primitive for a PreCommit failure on an
    /// existing, already-tracked path (the shape `delete_document` and
    /// `edit_document` roll back). It must undo BOTH the working-tree change and
    /// whatever `git add` staged for it.
    #[tokio::test]
    async fn restore_from_head_undoes_a_pending_delete() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        // README.md was committed by `create_bare_repo`'s initial commit, so it
        // exists at HEAD — simulate `delete_document` staging its removal and then
        // failing before `git commit` lands.
        std::fs::remove_file(work.path().join("README.md")).unwrap();
        git_test_cmd(work_path)
            .args(["add", "--", "README.md"])
            .output()
            .unwrap();
        assert!(!work.path().join("README.md").exists());

        let lock = lock_git().await;
        restore_from_head(&lock, work_path, "README.md")
            .await
            .unwrap();

        assert!(
            work.path().join("README.md").exists(),
            "restore_from_head must bring the file back on disk"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("README.md")).unwrap(),
            "# Test repo",
            "restored content must match HEAD"
        );
        let status = git_test_cmd(work_path)
            .args(["status", "--porcelain"])
            .output()
            .unwrap();
        assert!(
            String::from_utf8_lossy(&status.stdout).trim().is_empty(),
            "the index must also be back to matching HEAD (clean working tree)"
        );
    }

    /// Same primitive, applied to `edit_document`'s rollback shape: content
    /// overwritten in place (not staged), pre-commit failure, restore.
    #[tokio::test]
    async fn restore_from_head_undoes_a_pending_edit() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        std::fs::write(work.path().join("README.md"), "clobbered content").unwrap();

        let lock = lock_git().await;
        restore_from_head(&lock, work_path, "README.md")
            .await
            .unwrap();

        assert_eq!(
            std::fs::read_to_string(work.path().join("README.md")).unwrap(),
            "# Test repo",
            "restore_from_head must revert the working-tree edit to HEAD's content"
        );
    }

    /// A path with no HEAD content at all (the shape of a failed brand-new
    /// `create_document`) cannot be "restored to HEAD" — there is nothing there.
    /// This must fail loudly rather than silently succeed, so a caller relying on it
    /// for the wrong operation finds out immediately.
    #[tokio::test]
    async fn restore_from_head_fails_on_a_path_head_has_never_seen() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        std::fs::write(work.path().join("brand-new.md"), "new content").unwrap();

        let lock = lock_git().await;
        let result = restore_from_head(&lock, work_path, "brand-new.md").await;
        assert!(
            result.is_err(),
            "restoring a path HEAD never had must fail, not silently no-op"
        );
    }

    /// `unstage` is the rollback primitive for `create_document`: a new file was
    /// `git add`ed and then `git commit` failed. It must remove the index entry
    /// without touching the working tree (the caller deletes that separately).
    #[tokio::test]
    async fn unstage_removes_a_staged_new_file_without_touching_the_worktree() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        std::fs::write(work.path().join("newfile.md"), "new content").unwrap();
        git_test_cmd(work_path)
            .args(["add", "--", "newfile.md"])
            .output()
            .unwrap();

        let status_before = git_test_cmd(work_path)
            .args(["status", "--porcelain"])
            .output()
            .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&status_before.stdout).trim(),
            "A  newfile.md"
        );

        let lock = lock_git().await;
        unstage(&lock, work_path, "newfile.md").await.unwrap();

        let status_after = git_test_cmd(work_path)
            .args(["status", "--porcelain"])
            .output()
            .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&status_after.stdout).trim(),
            "?? newfile.md",
            "the file must be unstaged (back to untracked) but still on disk"
        );
        assert!(
            work.path().join("newfile.md").exists(),
            "unstage must not touch the working tree"
        );
    }

    /// `unstage` must be safe to call even when `git add` itself was the step that
    /// failed (nothing was ever staged) — `create_document`'s rollback calls it
    /// unconditionally without knowing which of add/commit failed.
    #[tokio::test]
    async fn unstage_is_a_noop_when_nothing_was_ever_staged() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        // "never-staged.md" does not exist anywhere — not on disk, not in the index,
        // not at HEAD.
        let lock = lock_git().await;
        let result = unstage(&lock, work_path, "never-staged.md").await;
        assert!(
            result.is_ok(),
            "unstage must be a safe no-op for a path that was never staged, got: {:?}",
            result
        );
    }

    #[test]
    fn parse_diff_name_status_handles_add_modify_delete() {
        // `-z` NUL-delimits every field — status AND path — rather than
        // tab-separating fields and newline-separating records.
        let out = "A\0new.md\0M\0changed.md\0D\0gone.md\0";
        let paths = parse_diff_name_status(out);
        assert_eq!(
            paths,
            vec![
                std::path::PathBuf::from("new.md"),
                std::path::PathBuf::from("changed.md"),
                std::path::PathBuf::from("gone.md"),
            ]
        );
    }

    #[test]
    fn parse_diff_name_status_enqueues_both_sides_of_a_rename() {
        // Git reports renames with a similarity-score suffix on the status
        // ("R100"), still NUL-separated from the two paths under `-z`.
        let out = "R100\0old-name.md\0new-name.md\0";
        let paths = parse_diff_name_status(out);
        assert_eq!(
            paths,
            vec![
                std::path::PathBuf::from("old-name.md"),
                std::path::PathBuf::from("new-name.md"),
            ],
            "both the old path (to purge) and the new path (to index) must be enqueued"
        );
    }

    #[test]
    fn parse_diff_name_status_ignores_trailing_and_stray_empty_tokens() {
        // A trailing NUL (git always terminates the last record with one, same
        // as every other) must not be mistaken for an empty status token.
        let out = "A\0new.md\0\0";
        assert_eq!(
            parse_diff_name_status(out),
            vec![std::path::PathBuf::from("new.md")]
        );
    }

    /// #143 regression: without `-z`, git C-quotes/octal-escapes any path
    /// containing non-ASCII bytes (also tabs/backslashes/quotes) by default
    /// (`core.quotepath=true`), and the old tab/newline-based parser returned
    /// that quoted string verbatim — a value matching no real file on disk.
    /// `-z` output is never quoted, so the raw bytes git wrote to the path
    /// come through unmangled.
    #[test]
    fn parse_diff_name_status_handles_non_ascii_and_special_paths() {
        let out = "M\0caf\u{e9}.md\0D\0gone.md\0R100\0old.md\0new\u{201c}quoted\u{201d}.md\0";
        let paths = parse_diff_name_status(out);
        assert_eq!(
            paths,
            vec![
                std::path::PathBuf::from("caf\u{e9}.md"),
                std::path::PathBuf::from("gone.md"),
                std::path::PathBuf::from("old.md"),
                std::path::PathBuf::from("new\u{201c}quoted\u{201d}.md"),
            ]
        );
    }

    // --- #104: concurrent git access to the knowledge-base clone ---

    /// Helper: `git` in `work_path` with `safe.directory` set, returning stdout.
    fn git_out(work_path: &str, args: &[&str]) -> String {
        let out = git_test_cmd(work_path)
            .args(["-c", &format!("safe.directory={}", work_path)])
            .args(args)
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).to_string()
    }

    /// Regression for #104. Writes arriving together used to race on
    /// `.git/index.lock`: the loser failed outright with "Unable to create
    /// '.git/index.lock': File exists", and could leave a half-staged index whose
    /// own rollback lost the same race — wedging the clone so every later write
    /// committed locally but could never rebase or push again.
    ///
    /// Each task acquires `GIT_LOCK` independently, which is the contention this
    /// exercises. Multi-threaded so the tasks genuinely overlap rather than merely
    /// interleaving at await points.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_commits_serialize_instead_of_racing_the_index_lock() {
        const N: usize = 8;

        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap().to_string();

        for i in 0..N {
            std::fs::write(work.path().join(format!("doc{i}.md")), format!("# Doc {i}")).unwrap();
        }

        let mut handles = Vec::with_capacity(N);
        for i in 0..N {
            let work_path = work_path.clone();
            handles.push(tokio::spawn(async move {
                let lock = lock_git().await;
                let doc_path = format!("doc{i}.md");
                let doc_message = format!("add doc{i}.md");
                commit_and_sync(
                    &lock,
                    None,
                    "main",
                    &work_path,
                    None,
                    &[doc_path.as_str()],
                    &doc_message,
                    "test-bot",
                    "test-bot@localhost",
                )
                .await
            }));
        }

        for (i, handle) in handles.into_iter().enumerate() {
            let result = handle.await.unwrap();
            assert!(
                result.is_ok(),
                "write {i} failed under concurrency: {:?}",
                result.err()
            );
        }

        // Every document landed in history...
        let log = git_out(&work_path, &["log", "--name-only", "--format="]);
        for i in 0..N {
            assert!(
                log.contains(&format!("doc{i}.md")),
                "doc{i}.md missing from history:\n{log}"
            );
        }

        // ...and nothing was left half-staged behind them. A non-empty index here
        // is precisely the state that used to block every subsequent rebase.
        let staged = git_out(&work_path, &["diff", "--cached", "--name-only"]);
        assert!(
            staged.trim().is_empty(),
            "index should be clean after concurrent writes, still staged: {staged}"
        );
    }

    /// Regression for #104. `git commit -m MSG` commits the ENTIRE index, so a
    /// stray staged entry — left by a failed write whose rollback did not
    /// complete, or by the separate `index --full` CLI process — silently rode
    /// along inside an unrelated document's commit. The `add` was already
    /// path-scoped; the commit has to be too.
    #[tokio::test]
    async fn commit_is_scoped_to_its_path_and_ignores_unrelated_staged_entries() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        std::fs::write(work.path().join("mine.md"), "# Mine").unwrap();
        std::fs::write(work.path().join("stray.md"), "# Stray").unwrap();

        // Residue from something other than this write.
        git_out(work_path, &["add", "--", "stray.md"]);

        let lock = lock_git().await;
        commit_and_sync(
            &lock,
            None,
            "main",
            work_path,
            None,
            &["mine.md"],
            "add mine.md",
            "test-bot",
            "test-bot@localhost",
        )
        .await
        .unwrap();
        drop(lock);

        let head = git_out(work_path, &["show", "--name-only", "--format=", "HEAD"]);
        assert!(
            head.contains("mine.md"),
            "own path should be committed: {head}"
        );
        assert!(
            !head.contains("stray.md"),
            "an unrelated staged entry must NOT ride along in this commit: {head}"
        );

        // The stray is left exactly as it was — scoping the commit neither commits
        // nor discards someone else's staged work.
        let staged = git_out(work_path, &["diff", "--cached", "--name-only"]);
        assert!(
            staged.contains("stray.md"),
            "unrelated staged entry should be untouched, staged: {staged}"
        );
    }

    // --- recover_interrupted_state ---

    #[tokio::test]
    async fn recover_interrupted_state_removes_stale_index_lock() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let index_lock = work.path().join(".git").join("index.lock");
        std::fs::write(&index_lock, b"").unwrap();
        assert!(index_lock.exists(), "test setup: index.lock must exist");

        let lock = lock_git().await;
        recover_interrupted_state(&lock, work.path()).await;

        assert!(
            !index_lock.exists(),
            "stale index.lock must be removed at startup"
        );
    }

    /// A real conflicted rebase, left mid-flight (not auto-aborted, unlike what
    /// `commit_and_sync` does on a conflict) — the shape a SIGKILL landing mid
    /// `git rebase` would leave behind.
    #[tokio::test]
    async fn recover_interrupted_state_aborts_a_stuck_rebase() {
        let bare = create_bare_repo("main");
        let bare_url = format!("file://{}", bare.path().to_str().unwrap());
        let lock = lock_git().await;

        // Clone A pushes conflict.md first.
        let work_a = clone_bare_repo(bare.path(), "main");
        std::fs::write(work_a.path().join("conflict.md"), "version A").unwrap();
        commit_and_sync(
            &lock,
            Some(&bare_url),
            "main",
            work_a.path().to_str().unwrap(),
            None,
            &["conflict.md"],
            "add conflict.md from A",
            "test-bot",
            "test-bot@localhost",
        )
        .await
        .unwrap();

        // Clone B, rewound to before A's commit, commits a conflicting version of
        // the same file locally (no push).
        let work_b = clone_bare_repo(bare.path(), "main");
        let work_b_path = work_b.path().to_str().unwrap();
        let commits = git_out(work_b_path, &["log", "--format=%H", "-2"]);
        let parent_sha = commits.lines().nth(1).unwrap().trim().to_string();
        git_out(work_b_path, &["reset", "--hard", &parent_sha]);
        std::fs::write(work_b.path().join("conflict.md"), "version B").unwrap();
        git_out(work_b_path, &["add", "conflict.md"]);
        git_out(
            work_b_path,
            &[
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "commit",
                "-m",
                "add conflict.md from B",
            ],
        );

        // Fetch and rebase manually, WITHOUT aborting on failure — this is what
        // leaves `.git/rebase-merge` behind for recovery to find.
        git_out(work_b_path, &["fetch", &bare_url, "main"]);
        let rebase_out = git_test_cmd(work_b_path)
            .args([
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "rebase",
                "FETCH_HEAD",
            ])
            .output()
            .unwrap();
        assert!(
            !rebase_out.status.success(),
            "expected a genuine rebase conflict"
        );
        assert!(
            work_b.path().join(".git/rebase-merge").exists()
                || work_b.path().join(".git/rebase-apply").exists(),
            "conflicted rebase should leave a rebase-merge/rebase-apply marker"
        );

        recover_interrupted_state(&lock, work_b.path()).await;

        assert!(
            !work_b.path().join(".git/rebase-merge").exists(),
            "rebase-merge marker must be cleared"
        );
        assert!(
            !work_b.path().join(".git/rebase-apply").exists(),
            "rebase-apply marker must be cleared"
        );
        let status = git_out(work_b_path, &["status", "--porcelain"]);
        assert!(
            status.trim().is_empty(),
            "working tree must be clean after the rebase is aborted, got: {status}"
        );
    }

    /// A real conflicted merge, left mid-flight — the shape a SIGKILL landing mid
    /// `git merge` (e.g. the webhook handler's ff-only merge, if it ever fell back
    /// to a real merge) would leave behind.
    #[tokio::test]
    async fn recover_interrupted_state_aborts_a_stuck_merge() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");
        let work_path = work.path().to_str().unwrap();

        git_out(work_path, &["checkout", "-b", "other"]);
        std::fs::write(work.path().join("README.md"), "# other branch content").unwrap();
        git_out(
            work_path,
            &[
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "commit",
                "-am",
                "other branch edit",
            ],
        );

        git_out(work_path, &["checkout", "main"]);
        std::fs::write(work.path().join("README.md"), "# main branch content").unwrap();
        git_out(
            work_path,
            &[
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "commit",
                "-am",
                "main branch edit",
            ],
        );

        let merge_out = git_test_cmd(work_path)
            .args([
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "merge",
                "other",
            ])
            .output()
            .unwrap();
        assert!(!merge_out.status.success(), "expected a genuine conflict");
        assert!(work.path().join(".git/MERGE_HEAD").exists());

        let lock = lock_git().await;
        recover_interrupted_state(&lock, work.path()).await;

        assert!(
            !work.path().join(".git/MERGE_HEAD").exists(),
            "MERGE_HEAD must be cleared after the merge is aborted"
        );
        let status = git_out(work_path, &["status", "--porcelain"]);
        assert!(
            status.trim().is_empty(),
            "working tree must be clean after the merge is aborted, got: {status}"
        );
    }

    /// A clean repo — nothing to recover — must be a silent no-op.
    #[tokio::test]
    async fn recover_interrupted_state_is_a_no_op_on_a_clean_repo() {
        let bare = create_bare_repo("main");
        let work = clone_bare_repo(bare.path(), "main");

        let lock = lock_git().await;
        recover_interrupted_state(&lock, work.path()).await;

        // Still a normal, healthy repo — recovery didn't damage anything.
        let status = git_out(work.path().to_str().unwrap(), &["status", "--porcelain"]);
        assert!(status.trim().is_empty());
    }
}
