use std::time::Duration;

use anyhow::{Context, Result};
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{error, info, warn};

/// Maximum time to wait for a git subprocess (clone) before treating it as
/// hung and returning an error.
pub(crate) const GIT_TIMEOUT: Duration = Duration::from_secs(120);

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
pub(crate) async fn rev_parse_head(data_path: &str) -> anyhow::Result<String> {
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

/// Stage `rel_path`, commit with `message`, then (if `git_url` is Some) fetch the
/// remote branch, rebase the local branch onto it, and push. Returns the new commit
/// SHA plus any paths pulled in by the rebase — see [`CommitOutcome`].
///
/// `rel_path` is relative to `data_path`. `message` already includes any provenance trailer.
/// If `git_url` is None, commit locally only (no fetch/rebase/push).
/// On a rebase conflict, abort the rebase (so the working tree is left clean at the local
/// commit) and return an Err whose message clearly identifies it as a rebase/merge conflict
/// on the file, distinct from other git failures.
#[allow(clippy::too_many_arguments)]
pub async fn commit_and_sync(
    git_url: Option<&str>,
    branch: &str,
    data_path: &str,
    token: Option<&str>,
    rel_path: &str,
    message: &str,
    author_name: &str,
    author_email: &str,
) -> anyhow::Result<CommitOutcome> {
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

    // --- git add -- <rel_path> ---
    let add_out = timeout(GIT_TIMEOUT, git_cmd(&["add", "--", rel_path]).output())
        .await
        .map_err(|_| anyhow::anyhow!("git add timed out after {:?}", GIT_TIMEOUT))?
        .context("Failed to spawn git add")?;
    if !add_out.status.success() {
        let stderr = redact_url(&String::from_utf8_lossy(&add_out.stderr));
        anyhow::bail!("git add failed: {}", stderr);
    }

    // --- git commit -m <message> ---
    // Set the author identity inline so the command is self-contained even in
    // environments without a global git user configured. Both author and committer
    // derive from user.* when not otherwise specified.
    let commit_out = timeout(
        GIT_TIMEOUT,
        git_cmd_authored(&["commit", "-m", message]).output(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("git commit timed out after {:?}", GIT_TIMEOUT))?
    .context("Failed to spawn git commit")?;
    if !commit_out.status.success() {
        let stderr = redact_url(&String::from_utf8_lossy(&commit_out.stderr));
        anyhow::bail!("git commit failed: {}", stderr);
    }

    let mut rebased_paths: Vec<std::path::PathBuf> = Vec::new();

    if let Some(url) = git_url {
        let auth_url = match token {
            Some(t) if !t.is_empty() => inject_token_into_url(url, t),
            _ => url.to_string(),
        };

        // Capture HEAD right after our own commit and before the fetch. Diffing this
        // against HEAD once the rebase completes isolates exactly what the rebase
        // pulled in from the remote — our own change is already on both sides of that
        // range, so it is never double-reported here.
        let old_head = rev_parse_head(data_path).await?;

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
        .map_err(|_| anyhow::anyhow!("git fetch timed out after {:?}", GIT_TIMEOUT))?
        .context("Failed to spawn git fetch")?;
        if !fetch_out.status.success() {
            let stderr = redact_url(&String::from_utf8_lossy(&fetch_out.stderr));
            anyhow::bail!("git fetch failed: {}", stderr);
        }

        // --- git rebase FETCH_HEAD ---
        let rebase_out = timeout(
            GIT_TIMEOUT,
            git_cmd_authored(&["rebase", "FETCH_HEAD"]).output(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("git rebase timed out after {:?}", GIT_TIMEOUT))?
        .context("Failed to spawn git rebase")?;
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

            if conflicted.is_empty() {
                anyhow::bail!(
                    "git rebase onto FETCH_HEAD failed with no conflicting files — \
                     this is not a merge conflict. Rebase aborted. stderr: {}",
                    stderr
                );
            }
            anyhow::bail!(
                "rebase conflict: git rebase onto FETCH_HEAD conflicted on {}. \
                 Rebase aborted. stderr: {}",
                conflicted.replace('\n', ", "),
                stderr
            );
        }

        // Diff the rebase range now, before pushing — this is local-only (no network)
        // and a push failure below should not prevent the caller from at least
        // learning what changed locally, though in practice a push failure aborts the
        // whole call anyway.
        let new_head = rev_parse_head(data_path).await?;
        rebased_paths = git_diff_name_status(data_path, &old_head, &new_head)
            .await
            .context("Failed to diff the rebase range")?;

        // --- git push <auth_url> HEAD:<branch> ---
        info!("Pushing to {} branch {}", redact_url(&auth_url), branch);
        let push_refspec = format!("HEAD:{}", branch);
        let push_out = timeout(
            GIT_TIMEOUT,
            git_cmd(&["push", &auth_url, &push_refspec]).output(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("git push timed out after {:?}", GIT_TIMEOUT))?
        .context("Failed to spawn git push")?;
        if !push_out.status.success() {
            let stderr = redact_url(&String::from_utf8_lossy(&push_out.stderr));
            anyhow::bail!("git push failed: {}", stderr);
        }
    }

    let sha = rev_parse_head(data_path).await?;
    Ok(CommitOutcome { sha, rebased_paths })
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

        let result = ensure_repo(
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

        let result = ensure_repo(
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

        let result = ensure_repo(
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

        // First call — should clone
        let first = ensure_repo(
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

        let result = ensure_repo(
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

        let result = ensure_repo(
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

        let result = ensure_repo(
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

        let result = ensure_repo(
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

        let outcome = commit_and_sync(
            None,
            "main",
            work_path,
            None,
            "notes.md",
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

        let outcome = commit_and_sync(
            Some(&bare_url),
            "main",
            work_path,
            None,
            "article.md",
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

        // Clone A: will push first
        let work_a = clone_bare_repo(bare.path(), "main");
        std::fs::write(work_a.path().join("conflict.md"), "version A").unwrap();
        commit_and_sync(
            Some(&bare_url),
            "main",
            work_a.path().to_str().unwrap(),
            None,
            "conflict.md",
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
            Some(&bare_url),
            "main",
            work_b.path().to_str().unwrap(),
            None,
            "conflict.md",
            "add conflict.md from B",
            "test-bot",
            "test-bot@localhost",
        )
        .await;

        assert!(result.is_err(), "Should fail due to rebase conflict");
        let err = result.unwrap_err().to_string();
        assert!(
            err.starts_with("rebase conflict:"),
            "Error should start with 'rebase conflict:', got: {}",
            err
        );
    }

    /// Two clones each add a different file and push in turn; the second push must
    /// rebase in the first clone's commit, and `rebased_paths` must report the file
    /// THAT commit touched — not the one this call is committing itself.
    #[tokio::test]
    async fn commit_and_sync_reports_paths_pulled_in_by_the_rebase() {
        let bare = create_bare_repo("main");
        let bare_url = format!("file://{}", bare.path().to_str().unwrap());

        // Clone A pushes first, adding other.md.
        let work_a = clone_bare_repo(bare.path(), "main");
        std::fs::write(work_a.path().join("other.md"), "from A").unwrap();
        commit_and_sync(
            Some(&bare_url),
            "main",
            work_a.path().to_str().unwrap(),
            None,
            "other.md",
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
            Some(&bare_url),
            "main",
            work_b.path().to_str().unwrap(),
            None,
            "mine.md",
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
}
