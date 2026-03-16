use anyhow::{Context, Result};
use tokio::process::Command;
use tracing::info;

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
/// git repo and `git_url` is provided, performs a shallow clone.
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

    let output = Command::new("git")
        .args([
            "clone",
            "--branch",
            branch,
            "--single-branch",
            "--depth",
            "1",
            &clone_url,
            ".",
        ])
        .current_dir(data_path)
        .output()
        .await
        .context("Failed to run git clone")?;

    if !output.status.success() {
        let stderr = redact_url(&String::from_utf8_lossy(&output.stderr));
        anyhow::bail!("git clone failed: {}", stderr);
    }

    info!("Clone complete");
    Ok(true)
}

#[cfg(test)]
mod tests {
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
    fn create_bare_repo(branch: &str) -> tempfile::TempDir {
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
}
