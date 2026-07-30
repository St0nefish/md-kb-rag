use std::time::Duration;

use anyhow::{Context, Result};
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{error, info};

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

/// Stage `rel_path`, commit with `message`, then (if `git_url` is Some) fetch the
/// remote branch, rebase the local branch onto it, and push. Returns the new commit SHA.
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
) -> anyhow::Result<String> {
    // Helper: build a base git command with safe.directory set and cwd pointing at data_path.
    // Returns (Command,) ready to have more args appended.
    let git_cmd = |args: &[&str]| {
        let mut cmd = Command::new("git");
        cmd.args(["-c", &format!("safe.directory={}", data_path)])
            .args(args)
            .current_dir(data_path);
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
        Command::new("git")
            .args([
                "-c",
                &format!("safe.directory={}", data_path),
                "commit",
                "-m",
                message,
            ])
            .env("GIT_AUTHOR_NAME", author_name)
            .env("GIT_AUTHOR_EMAIL", author_email)
            .env("GIT_COMMITTER_NAME", author_name)
            .env("GIT_COMMITTER_EMAIL", author_email)
            .current_dir(data_path)
            .output(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("git commit timed out after {:?}", GIT_TIMEOUT))?
    .context("Failed to spawn git commit")?;
    if !commit_out.status.success() {
        let stderr = redact_url(&String::from_utf8_lossy(&commit_out.stderr));
        anyhow::bail!("git commit failed: {}", stderr);
    }

    if let Some(url) = git_url {
        let auth_url = match token {
            Some(t) if !t.is_empty() => inject_token_into_url(url, t),
            _ => url.to_string(),
        };

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
        let rebase_out = timeout(GIT_TIMEOUT, git_cmd(&["rebase", "FETCH_HEAD"]).output())
            .await
            .map_err(|_| anyhow::anyhow!("git rebase timed out after {:?}", GIT_TIMEOUT))?
            .context("Failed to spawn git rebase")?;
        if !rebase_out.status.success() {
            let stderr = redact_url(&String::from_utf8_lossy(&rebase_out.stderr));
            // Abort the rebase so the working tree is left clean at the local commit.
            let _ = git_cmd(&["rebase", "--abort"]).output().await;
            anyhow::bail!(
                "rebase conflict: git rebase onto FETCH_HEAD failed ({}). Rebase aborted. stderr: {}",
                rel_path,
                stderr
            );
        }

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

    // --- git rev-parse HEAD ---
    let rev_out = timeout(GIT_TIMEOUT, git_cmd(&["rev-parse", "HEAD"]).output())
        .await
        .map_err(|_| anyhow::anyhow!("git rev-parse timed out after {:?}", GIT_TIMEOUT))?
        .context("Failed to spawn git rev-parse")?;
    if !rev_out.status.success() {
        let stderr = redact_url(&String::from_utf8_lossy(&rev_out.stderr));
        anyhow::bail!("git rev-parse HEAD failed: {}", stderr);
    }

    let sha = String::from_utf8_lossy(&rev_out.stdout).trim().to_string();
    Ok(sha)
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

        let sha = commit_and_sync(
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

        // SHA should be a 40-char hex string
        assert_eq!(sha.len(), 40, "Expected a 40-char SHA, got: {}", sha);
        assert!(
            sha.chars().all(|c| c.is_ascii_hexdigit()),
            "SHA should be hex: {}",
            sha
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

        let sha = commit_and_sync(
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

        assert_eq!(sha.len(), 40, "Expected a 40-char SHA, got: {}", sha);

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
}
