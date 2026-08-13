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

/// Parse `git diff --name-status` output into the set of paths it touched.
///
/// Handles A(dded)/M(odified)/D(eleted) as a single path, and R(enamed)/C(opied) — which
/// carry a similarity score suffix like `R100` and two tab-separated paths — as BOTH the
/// old and new path, since both need reindexing (old: purge; new: index).
fn parse_diff_name_status(output: &str) -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();
    for line in output.lines() {
        if line.is_empty() {
            continue;
        }
        let mut fields = line.split('\t');
        let Some(status) = fields.next() else {
            continue;
        };
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
                    "Unrecognized 'git diff --name-status' line, ignoring: {}",
                    line
                );
            }
        }
    }
    paths
}

/// `git diff --name-status -M old..new` in `data_path`, parsed into touched paths.
/// Local git only — no network. `-M` forces rename detection so a pure rename is
/// reported as `R`, not as a delete+add pair (which would still work — both paths end
/// up in the result — but would cost the new path an unnecessary re-embed instead of
/// letting `index_paths` see it as unchanged content under a new name... which it
/// cannot, since content hashing does not know about the old path. Either way both
/// paths are enqueued; `-M` is for a cleaner log line, not correctness here).
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

        // Diff the rebase range now, before pushing — this is local-only (no network)
        // and a push failure below should not prevent the caller from at least
        // learning what changed locally, though in practice a push failure aborts the
        // whole call anyway.
        let new_head =
            rev_parse_head(lock, data_path)
                .await
                .map_err(|e| CommitSyncError::PostCommit {
                    sha: local_sha.clone(),
                    source: e,
                })?;
        rebased_paths = git_diff_name_status(lock, data_path, &old_head, &new_head)
            .await
            .map_err(|e| CommitSyncError::PostCommit {
                sha: local_sha.clone(),
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
            sha: local_sha.clone(),
            source: anyhow::anyhow!("git push timed out after {:?}", GIT_TIMEOUT),
        })?
        .map_err(|e| CommitSyncError::PostCommit {
            sha: local_sha.clone(),
            source: anyhow::Error::new(e).context("Failed to spawn git push"),
        })?;
        if !push_out.status.success() {
            let stderr = redact_url(&String::from_utf8_lossy(&push_out.stderr));
            return Err(CommitSyncError::PostCommit {
                sha: local_sha.clone(),
                source: anyhow::anyhow!("git push failed: {}", stderr),
            });
        }

        // The rebase may have replayed our commit onto a new sha — read HEAD fresh
        // for the success return rather than reusing `local_sha`.
        let sha =
            rev_parse_head(lock, data_path)
                .await
                .map_err(|e| CommitSyncError::PostCommit {
                    sha: local_sha.clone(),
                    source: e,
                })?;
        return Ok(CommitOutcome { sha, rebased_paths });
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
        std::process::Command::new("git")
            .args(["init", "--bare", "--initial-branch", branch])
            .current_dir(bare_path)
            .output()
            .unwrap();

        // Create a temporary working clone to make an initial commit
        let work_dir = tempfile::TempDir::new().unwrap();
        std::process::Command::new("git")
            .args(["clone", bare_path.to_str().unwrap(), "."])
            .current_dir(work_dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["checkout", "-b", branch])
            .current_dir(work_dir.path())
            .output()
            .unwrap();
        std::fs::write(work_dir.path().join("README.md"), "# Test repo").unwrap();
        std::process::Command::new("git")
            .args(["add", "README.md"])
            .current_dir(work_dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args([
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "commit",
                "-m",
                "initial commit",
            ])
            .current_dir(work_dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["push", "origin", branch])
            .current_dir(work_dir.path())
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
        std::process::Command::new("git")
            .args(["clone", bare_path.to_str().unwrap(), "."])
            .current_dir(work_dir.path())
            .output()
            .unwrap();
        // Ensure we're on the right branch
        std::process::Command::new("git")
            .args(["checkout", branch])
            .current_dir(work_dir.path())
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
        let show_out = std::process::Command::new("git")
            .args([
                "-c",
                &format!("safe.directory={}", work_path),
                "show",
                "--name-only",
                "--format=",
                "HEAD",
            ])
            .current_dir(work_path)
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
        std::process::Command::new("git")
            .args(["clone", bare.path().to_str().unwrap(), "."])
            .current_dir(verify_dir.path())
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
        let log_out = std::process::Command::new("git")
            .args(["log", "--format=%H", "-2"])
            .current_dir(work_b.path())
            .output()
            .unwrap();
        let commits: Vec<&str> = std::str::from_utf8(&log_out.stdout)
            .unwrap()
            .lines()
            .collect();
        // commits[0] = A's commit, commits[1] = initial commit
        let parent_sha = commits[1].trim();

        // Reset to before A's commit
        std::process::Command::new("git")
            .args(["reset", "--hard", parent_sha])
            .current_dir(work_b.path())
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
        let log_out = std::process::Command::new("git")
            .args(["log", "--format=%H", "-2"])
            .current_dir(work_b.path())
            .output()
            .unwrap();
        let commits: Vec<&str> = std::str::from_utf8(&log_out.stdout)
            .unwrap()
            .lines()
            .collect();
        let parent_sha = commits[1].trim();
        std::process::Command::new("git")
            .args(["reset", "--hard", parent_sha])
            .current_dir(work_b.path())
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
        let committer = std::process::Command::new("git")
            .args(["log", "-1", "--format=%cn|%ce"])
            .current_dir(work_b.path())
            .output()
            .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&committer.stdout).trim(),
            "test-bot|test-bot@localhost",
            "the replayed commit must carry the committer identity commit_and_sync \
             was given, not whatever git config the host happens to have"
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

        let show_out = std::process::Command::new("git")
            .args(["show", "--name-only", "--format=", "HEAD"])
            .current_dir(work_path)
            .output()
            .unwrap();
        let show_str = String::from_utf8_lossy(&show_out.stdout);
        assert!(
            show_str.contains("one.md") && show_str.contains("two.md"),
            "both paths should appear in the single commit, got: {show_str}"
        );

        let log = std::process::Command::new("git")
            .args([
                "rev-list",
                "--count",
                &format!("{head_before}..{}", outcome.sha),
            ])
            .current_dir(work_path)
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

        let log = std::process::Command::new("git")
            .args([
                "rev-list",
                "--count",
                &format!("{head_before_move}..{}", outcome.sha),
            ])
            .current_dir(work_path)
            .output()
            .unwrap();
        let count = String::from_utf8_lossy(&log.stdout).trim().to_string();
        assert_eq!(
            count, "1",
            "the move should land as exactly one commit, got {count}"
        );

        let name_status = std::process::Command::new("git")
            .args(["show", "--name-status", "--format=", "HEAD"])
            .current_dir(work_path)
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

        let log = std::process::Command::new("git")
            .args([
                "rev-list",
                "--count",
                &format!("{head_before}..{}", outcome.sha),
            ])
            .current_dir(work_path)
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
        let show_out = std::process::Command::new("git")
            .args(["show", "--name-only", "--format=", "HEAD"])
            .current_dir(work_path)
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
        std::process::Command::new("git")
            .args(["add", "--", "README.md"])
            .current_dir(work_path)
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
        let status = std::process::Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(work_path)
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
        std::process::Command::new("git")
            .args(["add", "--", "newfile.md"])
            .current_dir(work_path)
            .output()
            .unwrap();

        let status_before = std::process::Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(work_path)
            .output()
            .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&status_before.stdout).trim(),
            "A  newfile.md"
        );

        let lock = lock_git().await;
        unstage(&lock, work_path, "newfile.md").await.unwrap();

        let status_after = std::process::Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(work_path)
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
        let out = "A\tnew.md\nM\tchanged.md\nD\tgone.md\n";
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
        // Git reports renames with a similarity-score suffix on the status ("R100").
        let out = "R100\told-name.md\tnew-name.md\n";
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
    fn parse_diff_name_status_ignores_blank_lines() {
        let out = "A\tnew.md\n\n";
        assert_eq!(
            parse_diff_name_status(out),
            vec![std::path::PathBuf::from("new.md")]
        );
    }

    // --- #104: concurrent git access to the knowledge-base clone ---

    /// Helper: `git` in `work_path` with `safe.directory` set, returning stdout.
    fn git_out(work_path: &str, args: &[&str]) -> String {
        let out = std::process::Command::new("git")
            .args(["-c", &format!("safe.directory={}", work_path)])
            .args(args)
            .current_dir(work_path)
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
        let rebase_out = std::process::Command::new("git")
            .args([
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "rebase",
                "FETCH_HEAD",
            ])
            .current_dir(work_b_path)
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

        let merge_out = std::process::Command::new("git")
            .args([
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "merge",
                "other",
            ])
            .current_dir(work_path)
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
