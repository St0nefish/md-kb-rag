use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use globset::{Glob, GlobSet, GlobSetBuilder};
use sha2::{Digest, Sha256};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::{
    chunk,
    config::{Config, IndexingConfig},
    embed::EmbedClient,
    qdrant::{QdrantPoint, QdrantStore},
    state::{IndexedFile, StateDb},
    validate,
};

// ---------------------------------------------------------------------------
// File discovery
// ---------------------------------------------------------------------------

fn build_globset(patterns: &[String]) -> Result<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        let glob =
            Glob::new(pattern).with_context(|| format!("Invalid glob pattern: '{}'", pattern))?;
        builder.add(glob);
    }
    Ok(builder.build()?)
}

pub fn discover_files(data_path: &Path, indexing: &IndexingConfig) -> Result<Vec<PathBuf>> {
    let include_set =
        build_globset(&indexing.include).context("Failed to build include glob set")?;

    let exclude_set = if indexing.exclude.is_empty() {
        None
    } else {
        Some(build_globset(&indexing.exclude).context("Failed to build exclude glob set")?)
    };

    let exclude_filenames: HashSet<&str> =
        indexing.exclude_files.iter().map(|s| s.as_str()).collect();

    let mut matched: Vec<PathBuf> = Vec::new();

    walk_dir(
        data_path,
        data_path,
        &include_set,
        &exclude_set,
        &exclude_filenames,
        &mut matched,
    )?;

    matched.sort();
    Ok(matched)
}

fn walk_dir(
    root: &Path,
    dir: &Path,
    include_set: &GlobSet,
    exclude_set: &Option<GlobSet>,
    exclude_filenames: &HashSet<&str>,
    matched: &mut Vec<PathBuf>,
) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory: {}", dir.display()))?;

    for entry in entries {
        let entry = entry.with_context(|| format!("Failed to read entry in {}", dir.display()))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("Failed to stat: {}", path.display()))?;

        if file_type.is_symlink() {
            warn!("Skipping symlink: {}", path.display());
            continue;
        }

        if file_type.is_dir() {
            walk_dir(
                root,
                &path,
                include_set,
                exclude_set,
                exclude_filenames,
                matched,
            )?;
            continue;
        }

        if !file_type.is_file() {
            continue;
        }

        // Check exclude_files by filename
        if let Some(file_name) = path.file_name().and_then(|n| n.to_str())
            && exclude_filenames.contains(file_name)
        {
            debug!("Skipping excluded filename: {}", path.display());
            continue;
        }

        // Build relative path for glob matching
        let rel = path.strip_prefix(root).unwrap_or(&path);

        let rel_str = rel.to_string_lossy();

        // Must match at least one include pattern
        if !include_set.is_match(rel) && !include_set.is_match(rel_str.as_ref()) {
            continue;
        }

        // Must not match any exclude pattern
        if let Some(excl) = exclude_set
            && (excl.is_match(rel) || excl.is_match(rel_str.as_ref()))
        {
            debug!("Excluding file: {}", path.display());
            continue;
        }

        matched.push(path);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

pub fn compute_hash_from_bytes(content: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content);
    let digest = hasher.finalize();
    hex::encode(digest)
}

#[cfg(test)]
pub async fn compute_hash(path: &Path) -> Result<String> {
    let content = tokio::fs::read(path)
        .await
        .with_context(|| format!("Failed to read file for hashing: {}", path.display()))?;
    Ok(compute_hash_from_bytes(&content))
}

// ---------------------------------------------------------------------------
// Point ID generation
// ---------------------------------------------------------------------------

/// Project-specific UUID v5 namespace (generated once, never change after first index).
const NAMESPACE_MDKBRAG: Uuid = Uuid::from_bytes([
    0x6b, 0xa7, 0xb8, 0x14, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8,
]);

pub fn make_point_id(file_path: &str, chunk_index: usize) -> String {
    let name = format!("{}::{}", file_path, chunk_index);
    Uuid::new_v5(&NAMESPACE_MDKBRAG, name.as_bytes()).to_string()
}

// ---------------------------------------------------------------------------
// Main indexing function
// ---------------------------------------------------------------------------

/// Holds everything we need to embed and upsert for one file.
struct PendingFile {
    file_path: String,
    frontmatter: HashMap<String, serde_json::Value>,
    chunks: Vec<chunk::Chunk>,
    /// Content hash of the file on disk.
    hash: String,
    /// True when the file already existed in the state DB (so we need to delete old points first).
    was_indexed: bool,
}

/// Result of processing a single discovered file.
enum FileOutcome {
    Skipped,
    Invalid,
    Empty,
    Ready(PendingFile),
}

/// Process a single file: hash, skip-if-unchanged, validate, chunk.
async fn process_file(
    path: &Path,
    content: &str,
    full: bool,
    state_entry: Option<IndexedFile>,
    config: &Config,
) -> Result<FileOutcome> {
    let file_path = path.to_string_lossy().to_string();
    let hash = compute_hash_from_bytes(content.as_bytes());

    let was_indexed = state_entry.is_some();

    // Skip unchanged files in incremental mode
    if !full
        && let Some(ref entry) = state_entry
        && entry.content_hash == hash
    {
        debug!("Unchanged, skipping: {}", file_path);
        return Ok(FileOutcome::Skipped);
    }

    if config.validation.enabled {
        match validate::validate_content(path, content, &config.frontmatter, &config.validation)
            .await
        {
            Ok((_result, Some(validated))) => {
                let description = validated
                    .frontmatter
                    .get("description")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned);

                let chunks = chunk::chunk_markdown(
                    &validated.body,
                    description.as_deref(),
                    &config.chunking,
                );

                if chunks.is_empty() {
                    warn!("No chunks produced for: {}", file_path);
                    return Ok(FileOutcome::Empty);
                }

                debug!("  {} chunks from: {}", chunks.len(), file_path);

                Ok(FileOutcome::Ready(PendingFile {
                    file_path,
                    frontmatter: validated.frontmatter,
                    chunks,
                    hash,
                    was_indexed,
                }))
            }
            Ok((result, None)) => {
                for err in &result.errors {
                    warn!("Validation error [{}]: {}", file_path, err);
                }

                if config.validation.strict {
                    anyhow::bail!(
                        "Validation failed for '{}' (strict mode): {:?}",
                        file_path,
                        result.errors
                    );
                }

                Ok(FileOutcome::Invalid)
            }
            Err(e) => {
                error!("Failed to validate {}: {:#}", file_path, e);

                if config.validation.strict {
                    return Err(e).with_context(|| {
                        format!("Validation error in strict mode for '{}'", file_path)
                    });
                }

                Ok(FileOutcome::Invalid)
            }
        }
    } else {
        // Validation disabled — chunk the already-read content
        let chunks = chunk::chunk_markdown(content, None, &config.chunking);
        if chunks.is_empty() {
            warn!("No chunks produced for: {}", file_path);
            return Ok(FileOutcome::Empty);
        }

        Ok(FileOutcome::Ready(PendingFile {
            file_path,
            frontmatter: HashMap::new(),
            chunks,
            hash,
            was_indexed,
        }))
    }
}

/// Embed all pending files and upsert their points into Qdrant.
async fn upsert_pending(
    pending: &[PendingFile],
    embedder: &EmbedClient,
    store: &QdrantStore,
    state: &StateDb,
    collection: &str,
) -> Result<()> {
    // Flatten all chunk texts in order, recording boundaries
    let mut all_texts: Vec<String> = Vec::new();
    let mut file_boundaries: Vec<(usize, usize)> = Vec::new(); // (start_idx, count)

    for pf in pending {
        let start = all_texts.len();
        for c in &pf.chunks {
            all_texts.push(c.text.clone());
        }
        file_boundaries.push((start, pf.chunks.len()));
    }

    let all_embeddings = embedder
        .embed_texts(&all_texts)
        .await
        .context("Failed to embed chunk texts")?;

    if all_embeddings.len() != all_texts.len() {
        anyhow::bail!(
            "Embedding count mismatch: expected {}, got {}",
            all_texts.len(),
            all_embeddings.len()
        );
    }

    for (pf, (start, count)) in pending.iter().zip(file_boundaries.iter()) {
        let embeddings = &all_embeddings[*start..*start + *count];

        let mut points: Vec<QdrantPoint> = Vec::with_capacity(*count);
        for (chunk, vector) in pf.chunks.iter().zip(embeddings.iter()) {
            let mut payload: HashMap<String, serde_json::Value> = pf.frontmatter.clone();
            payload.insert(
                "file_path".to_string(),
                serde_json::Value::String(pf.file_path.clone()),
            );
            payload.insert(
                "chunk_index".to_string(),
                serde_json::Value::Number(chunk.index.into()),
            );
            payload.insert(
                "text".to_string(),
                serde_json::Value::String(chunk.text.clone()),
            );
            payload.insert(
                "line_start".to_string(),
                serde_json::Value::Number(chunk.line_start.into()),
            );
            payload.insert(
                "line_end".to_string(),
                serde_json::Value::Number(chunk.line_end.into()),
            );

            points.push(QdrantPoint {
                id: make_point_id(&pf.file_path, chunk.index),
                vector: vector.clone(),
                payload,
            });
        }

        // Delete old points first (for changed files that were already indexed)
        if pf.was_indexed {
            store
                .delete_by_file(collection, &pf.file_path)
                .await
                .with_context(|| format!("Failed to delete old points for '{}'", pf.file_path))?;
        }

        if let Err(e) = store.upsert_points(collection, points).await {
            // Upsert failed after old points were already deleted.
            // Remove the state entry so this file is re-processed on the next run
            // instead of being silently skipped due to a stale hash match.
            if pf.was_indexed
                && let Err(del_err) = state.delete(&pf.file_path).await
            {
                error!(
                    "Failed to clean up state DB entry for '{}' after upsert failure: {:#}",
                    pf.file_path, del_err
                );
            }
            return Err(e)
                .with_context(|| format!("Failed to upsert points for '{}'", pf.file_path));
        }

        state
            .upsert(&pf.file_path, &pf.hash, *count as i64)
            .await
            .with_context(|| format!("Failed to update state DB for '{}'", pf.file_path))?;

        info!("Indexed {} chunk(s) from: {}", count, pf.file_path);
    }

    Ok(())
}

/// Remove orphaned files (deleted from disk but still in the index).
async fn remove_orphans(
    orphaned: &[String],
    store: &QdrantStore,
    state: &StateDb,
    collection: &str,
) -> Result<()> {
    for file_path in orphaned {
        store
            .delete_by_file(collection, file_path)
            .await
            .with_context(|| format!("Failed to delete orphaned points for '{}'", file_path))?;

        state
            .delete(file_path)
            .await
            .with_context(|| format!("Failed to delete state DB entry for '{}'", file_path))?;

        info!("Removed orphaned file: {}", file_path);
    }
    Ok(())
}

pub async fn run_index(config: &Config, full: bool) -> Result<()> {
    info!(
        mode = if full { "full" } else { "incremental" },
        "Starting indexing run"
    );

    // ── Infrastructure ──────────────────────────────────────────────────────
    let db_path = config.state_db_path();
    let state = StateDb::new(Path::new(&db_path))
        .await
        .context("Failed to open state DB")?;

    let store = QdrantStore::new(&config.qdrant).context("Failed to connect to Qdrant")?;

    let collection = &config.qdrant.collection;
    let vector_size = config.embedding.vector_size;

    let indexed_fields = config.effective_indexed_fields();

    // ── Full-mode: wipe state and Qdrant collection so everything is clean ───
    if full {
        info!("Full reindex: clearing state DB and Qdrant collection");
        state.clear().await.context("Failed to clear state DB")?;
        store
            .drop_collection(collection)
            .await
            .context("Failed to drop Qdrant collection for full reindex")?;
    }

    store
        .ensure_collection(collection, vector_size, &indexed_fields)
        .await
        .context("Failed to ensure Qdrant collection")?;

    let embedder = EmbedClient::new(&config.embedding);

    // ── File discovery ───────────────────────────────────────────────────────
    let data_path = Path::new(config.data_path());
    let discovered =
        discover_files(data_path, &config.indexing).context("Failed to discover files")?;

    info!("Discovered {} files", discovered.len());

    // ── Determine which previously-indexed files no longer exist ─────────────
    let all_indexed = state.list_all().await.context("Failed to list state DB")?;
    let discovered_set: HashSet<String> = discovered
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();

    let orphaned: Vec<String> = all_indexed
        .iter()
        .map(|f| f.file_path.clone())
        .filter(|fp| !discovered_set.contains(fp))
        .collect();

    let indexed_map: HashMap<String, IndexedFile> = all_indexed
        .into_iter()
        .map(|f| (f.file_path.clone(), f))
        .collect();

    // ── Per-file processing ──────────────────────────────────────────────────
    let mut pending: Vec<PendingFile> = Vec::new();
    let mut skipped = 0usize;
    let mut invalid = 0usize;

    for path in &discovered {
        let file_path = path.to_string_lossy().to_string();

        // Read file once — used for hashing, validation, and chunking (fix TOCTOU #51)
        let content = match tokio::fs::read_to_string(path).await {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to read {}: {:#}", file_path, e);
                continue;
            }
        };

        let state_entry = indexed_map.get(&file_path).cloned();

        match process_file(path, &content, full, state_entry, config).await? {
            FileOutcome::Skipped => skipped += 1,
            FileOutcome::Invalid => invalid += 1,
            FileOutcome::Empty => {}
            FileOutcome::Ready(pf) => pending.push(pf),
        }
    }

    // ── Batch embedding & upsert ────────────────────────────────────────────
    let pending_count = pending.len();
    if !pending.is_empty() {
        info!("Embedding chunks for {} changed file(s)…", pending_count);
        upsert_pending(&pending, &embedder, &store, &state, collection).await?;
    }

    // ── Handle orphaned (deleted) files ──────────────────────────────────────
    if !orphaned.is_empty() {
        info!("Removing {} orphaned file(s) from index", orphaned.len());
        remove_orphans(&orphaned, &store, &state, collection).await?;
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    info!(
        discovered = discovered.len(),
        indexed = pending_count,
        skipped,
        invalid,
        orphans_removed = orphaned.len(),
        "Indexing run complete"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn make_point_id_deterministic() {
        let id1 = make_point_id("docs/test.md", 0);
        let id2 = make_point_id("docs/test.md", 0);
        let id3 = make_point_id("docs/test.md", 1);
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        uuid::Uuid::parse_str(&id1).unwrap();
    }

    #[test]
    fn compute_hash_from_bytes_consistent() {
        let h1 = compute_hash_from_bytes(b"hello world");
        let h2 = compute_hash_from_bytes(b"hello world");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // SHA256 hex
    }

    #[test]
    fn compute_hash_from_bytes_differs_on_content() {
        assert_ne!(
            compute_hash_from_bytes(b"hello"),
            compute_hash_from_bytes(b"world")
        );
    }

    #[tokio::test]
    async fn compute_hash_from_bytes_matches_file_hash() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        let content = b"hello world";
        std::fs::write(&path, content).unwrap();
        let file_hash = compute_hash(&path).await.unwrap();
        let bytes_hash = compute_hash_from_bytes(content);
        assert_eq!(file_hash, bytes_hash);
    }

    #[tokio::test]
    async fn compute_hash_consistent() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello world").unwrap();
        let h1 = compute_hash(&path).await.unwrap();
        let h2 = compute_hash(&path).await.unwrap();
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // SHA256 hex
    }

    #[tokio::test]
    async fn compute_hash_differs_on_content() {
        let dir = TempDir::new().unwrap();
        let p1 = dir.path().join("a.txt");
        let p2 = dir.path().join("b.txt");
        std::fs::write(&p1, "hello").unwrap();
        std::fs::write(&p2, "world").unwrap();
        assert_ne!(
            compute_hash(&p1).await.unwrap(),
            compute_hash(&p2).await.unwrap()
        );
    }

    /// Helper: build a Config with validation disabled for simpler test setup.
    fn config_no_validation() -> Config {
        Config {
            validation: crate::config::ValidationConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn process_file_skips_unchanged_incremental() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("doc.md");
        let content = "# Hello\nSome body text here.";
        std::fs::write(&path, content).unwrap();

        let hash = compute_hash_from_bytes(content.as_bytes());
        let state_entry = Some(IndexedFile {
            file_path: path.to_string_lossy().to_string(),
            content_hash: hash,
            chunk_count: 1,
            indexed_at: String::new(),
        });

        let config = config_no_validation();
        let outcome = process_file(&path, content, false, state_entry, &config)
            .await
            .unwrap();
        assert!(matches!(outcome, FileOutcome::Skipped));
    }

    #[tokio::test]
    async fn process_file_indexes_changed_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("doc.md");
        let content = "# Hello\nSome body text here.";
        std::fs::write(&path, content).unwrap();

        let state_entry = Some(IndexedFile {
            file_path: path.to_string_lossy().to_string(),
            content_hash: "old-hash".to_string(),
            chunk_count: 1,
            indexed_at: String::new(),
        });

        let config = config_no_validation();
        let outcome = process_file(&path, content, false, state_entry, &config)
            .await
            .unwrap();
        match outcome {
            FileOutcome::Ready(pf) => {
                assert!(!pf.chunks.is_empty());
                assert!(pf.was_indexed);
            }
            other => panic!("Expected Ready, got {:?}", outcome_name(&other)),
        }
    }

    #[tokio::test]
    async fn process_file_full_mode_ignores_matching_hash() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("doc.md");
        let content = "# Hello\nSome body text here.";
        std::fs::write(&path, content).unwrap();

        let hash = compute_hash_from_bytes(content.as_bytes());
        let state_entry = Some(IndexedFile {
            file_path: path.to_string_lossy().to_string(),
            content_hash: hash,
            chunk_count: 1,
            indexed_at: String::new(),
        });

        let config = config_no_validation();
        let outcome = process_file(&path, content, true, state_entry, &config)
            .await
            .unwrap();
        assert!(
            matches!(outcome, FileOutcome::Ready(_)),
            "Full mode should process even when hash matches"
        );
    }

    #[tokio::test]
    async fn process_file_new_file_not_was_indexed() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("doc.md");
        let content = "# Hello\nBody text.";
        std::fs::write(&path, content).unwrap();

        let config = config_no_validation();
        let outcome = process_file(&path, content, false, None, &config)
            .await
            .unwrap();
        match outcome {
            FileOutcome::Ready(pf) => assert!(!pf.was_indexed),
            other => panic!("Expected Ready, got {:?}", outcome_name(&other)),
        }
    }

    #[tokio::test]
    async fn process_file_empty_content_returns_empty() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("doc.md");
        let content = "";
        std::fs::write(&path, content).unwrap();

        let config = config_no_validation();
        let outcome = process_file(&path, content, false, None, &config)
            .await
            .unwrap();
        assert!(matches!(outcome, FileOutcome::Empty));
    }

    #[tokio::test]
    async fn process_file_with_validation_valid_frontmatter() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("doc.md");
        let content = "---\ntitle: Test\n---\n# Hello\nBody text here.";
        std::fs::write(&path, content).unwrap();

        let config = Config {
            frontmatter: crate::config::FrontmatterConfig {
                required: vec!["title".into()],
                ..Default::default()
            },
            ..Default::default()
        };

        let outcome = process_file(&path, content, false, None, &config)
            .await
            .unwrap();
        match outcome {
            FileOutcome::Ready(pf) => {
                assert!(pf.frontmatter.contains_key("title"));
            }
            other => panic!("Expected Ready, got {:?}", outcome_name(&other)),
        }
    }

    #[tokio::test]
    async fn process_file_with_validation_missing_required_field() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("doc.md");
        let content = "---\ntitle: Test\n---\n# Hello\nBody.";
        std::fs::write(&path, content).unwrap();

        let config = Config {
            frontmatter: crate::config::FrontmatterConfig {
                required: vec!["description".into()],
                ..Default::default()
            },
            ..Default::default()
        };

        let outcome = process_file(&path, content, false, None, &config)
            .await
            .unwrap();
        assert!(matches!(outcome, FileOutcome::Invalid));
    }

    #[tokio::test]
    async fn process_file_strict_validation_failure_is_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("doc.md");
        let content = "---\ntitle: Test\n---\n# Hello\nBody.";
        std::fs::write(&path, content).unwrap();

        let config = Config {
            validation: crate::config::ValidationConfig {
                enabled: true,
                strict: true,
                ..Default::default()
            },
            frontmatter: crate::config::FrontmatterConfig {
                required: vec!["description".into()],
                ..Default::default()
            },
            ..Default::default()
        };

        let result = process_file(&path, content, false, None, &config).await;
        assert!(result.is_err(), "Strict mode should propagate as Err");
    }

    /// Helper for debug output in test assertions.
    fn outcome_name(outcome: &FileOutcome) -> &'static str {
        match outcome {
            FileOutcome::Skipped => "Skipped",
            FileOutcome::Invalid => "Invalid",
            FileOutcome::Empty => "Empty",
            FileOutcome::Ready(_) => "Ready",
        }
    }

    #[test]
    fn discover_files_basic() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("doc.md"), "# Test").unwrap();
        std::fs::write(dir.path().join("other.txt"), "text").unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/nested.md"), "# Nested").unwrap();

        let indexing = IndexingConfig {
            include: vec!["**/*.md".into()],
            exclude: vec![],
            exclude_files: vec![],
        };
        let files = discover_files(dir.path(), &indexing).unwrap();
        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|p| p.ends_with("doc.md")));
        assert!(files.iter().any(|p| p.ends_with("nested.md")));
    }

    #[test]
    fn discover_files_excludes() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("keep.md"), "keep").unwrap();
        std::fs::write(dir.path().join("README.md"), "readme").unwrap();
        std::fs::create_dir_all(dir.path().join("archive")).unwrap();
        std::fs::write(dir.path().join("archive/old.md"), "old").unwrap();

        let indexing = IndexingConfig {
            include: vec!["**/*.md".into()],
            exclude: vec!["archive/**".into()],
            exclude_files: vec!["README.md".into()],
        };
        let files = discover_files(dir.path(), &indexing).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("keep.md"));
    }

    #[test]
    fn discover_files_skips_symlinks_to_files() {
        let dir = TempDir::new().unwrap();
        let real = dir.path().join("real.md");
        std::fs::write(&real, "# Real").unwrap();

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&real, dir.path().join("link.md")).unwrap();
        }

        let indexing = IndexingConfig {
            include: vec!["**/*.md".into()],
            exclude: vec![],
            exclude_files: vec![],
        };
        let files = discover_files(dir.path(), &indexing).unwrap();

        #[cfg(unix)]
        {
            assert_eq!(files.len(), 1, "Symlinked file should be skipped");
            assert!(files[0].ends_with("real.md"));
        }
    }

    #[test]
    fn discover_files_symlink_loop_does_not_hang() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("doc.md"), "# Doc").unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();

        #[cfg(unix)]
        {
            // Create a symlink loop: sub/loop -> parent dir
            std::os::unix::fs::symlink(dir.path(), dir.path().join("sub/loop")).unwrap();
        }

        let indexing = IndexingConfig {
            include: vec!["**/*.md".into()],
            exclude: vec![],
            exclude_files: vec![],
        };

        // This should complete without hanging or panicking
        let files = discover_files(dir.path(), &indexing).unwrap();
        assert!(files.iter().any(|p| p.ends_with("doc.md")));
    }
}
