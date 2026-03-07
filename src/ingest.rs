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
    state::StateDb,
    validate,
};

// ---------------------------------------------------------------------------
// File discovery
// ---------------------------------------------------------------------------

fn build_globset(patterns: &[String]) -> Result<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        let glob = Glob::new(pattern)
            .with_context(|| format!("Invalid glob pattern: '{}'", pattern))?;
        builder.add(glob);
    }
    Ok(builder.build()?)
}

pub fn discover_files(data_path: &Path, indexing: &IndexingConfig) -> Result<Vec<PathBuf>> {
    let include_set = build_globset(&indexing.include)
        .context("Failed to build include glob set")?;

    let exclude_set = if indexing.exclude.is_empty() {
        None
    } else {
        Some(
            build_globset(&indexing.exclude)
                .context("Failed to build exclude glob set")?,
        )
    };

    let exclude_filenames: HashSet<&str> = indexing
        .exclude_files
        .iter()
        .map(|s| s.as_str())
        .collect();

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
        let metadata = entry
            .metadata()
            .with_context(|| format!("Failed to stat: {}", path.display()))?;

        if metadata.is_dir() {
            walk_dir(root, &path, include_set, exclude_set, exclude_filenames, matched)?;
            continue;
        }

        if !metadata.is_file() {
            continue;
        }

        // Check exclude_files by filename
        if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
            if exclude_filenames.contains(file_name) {
                debug!("Skipping excluded filename: {}", path.display());
                continue;
            }
        }

        // Build relative path for glob matching
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path);

        let rel_str = rel.to_string_lossy();

        // Must match at least one include pattern
        if !include_set.is_match(rel) && !include_set.is_match(rel_str.as_ref()) {
            continue;
        }

        // Must not match any exclude pattern
        if let Some(excl) = exclude_set {
            if excl.is_match(rel) || excl.is_match(rel_str.as_ref()) {
                debug!("Excluding file: {}", path.display());
                continue;
            }
        }

        matched.push(path);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

pub fn compute_hash(path: &Path) -> Result<String> {
    let content = std::fs::read(path)
        .with_context(|| format!("Failed to read file for hashing: {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&content);
    let digest = hasher.finalize();
    Ok(hex::encode(digest))
}

// ---------------------------------------------------------------------------
// Point ID generation
// ---------------------------------------------------------------------------

pub fn make_point_id(file_path: &str, chunk_index: usize) -> String {
    let name = format!("{}::{}", file_path, chunk_index);
    Uuid::new_v5(&Uuid::NAMESPACE_DNS, name.as_bytes()).to_string()
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

pub async fn run_index(config: &Config, full: bool) -> Result<()> {
    info!(
        mode = if full { "full" } else { "incremental" },
        "Starting indexing run"
    );

    // ── Infrastructure ──────────────────────────────────────────────────────
    let db_path = "data/state.db";
    let state = StateDb::new(db_path)
        .await
        .context("Failed to open state DB")?;

    let store = QdrantStore::new(&config.qdrant)
        .context("Failed to connect to Qdrant")?;

    let collection = &config.qdrant.collection;
    let vector_size = config.embedding.vector_size;

    // Build the list of fields we want keyword-indexed in Qdrant: all
    // configured frontmatter indexed_fields plus the built-in "file_path".
    let mut indexed_fields = config.frontmatter.indexed_fields.clone();
    if !indexed_fields.contains(&"file_path".to_string()) {
        indexed_fields.push("file_path".to_string());
    }

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
    let discovered = discover_files(data_path, &config.indexing)
        .context("Failed to discover files")?;

    info!("Discovered {} files", discovered.len());

    // ── Determine which previously-indexed files no longer exist ─────────────
    // (do this before the per-file loop so we have the complete picture)
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

    // ── Per-file processing ──────────────────────────────────────────────────
    let mut pending: Vec<PendingFile> = Vec::new();
    let mut skipped = 0usize;
    let mut invalid = 0usize;

    for path in &discovered {
        let file_path = path.to_string_lossy().to_string();

        let hash = match compute_hash(path) {
            Ok(h) => h,
            Err(e) => {
                error!("Failed to hash {}: {:#}", file_path, e);
                continue;
            }
        };

        // Skip unchanged files in incremental mode
        let state_entry = state.get(&file_path).await.with_context(|| {
            format!("Failed to query state DB for '{}'", file_path)
        })?;

        let was_indexed = state_entry.is_some();

        if !full {
            if let Some(ref entry) = state_entry {
                if entry.content_hash == hash {
                    debug!("Unchanged, skipping: {}", file_path);
                    skipped += 1;
                    continue;
                }
            }
        }

        // Validate
        if config.validation.enabled {
            match validate::validate_file(path, &config.frontmatter, &config.validation) {
                Ok((result, Some(validated))) => {
                    if !result.warnings.is_empty() {
                        for w in &result.warnings {
                            warn!("  [{}] {}", file_path, w);
                        }
                    }

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
                        continue;
                    }

                    debug!("  {} chunks from: {}", chunks.len(), file_path);

                    pending.push(PendingFile {
                        file_path,
                        frontmatter: validated.frontmatter,
                        chunks,
                        hash,
                        was_indexed,
                    });
                }
                Ok((result, None)) => {
                    // Validation failed
                    for err in &result.errors {
                        warn!("Validation error [{}]: {}", file_path, err);
                    }
                    invalid += 1;

                    if config.validation.strict {
                        anyhow::bail!(
                            "Validation failed for '{}' (strict mode): {:?}",
                            file_path,
                            result.errors
                        );
                    }
                }
                Err(e) => {
                    error!("Failed to validate {}: {:#}", file_path, e);
                    invalid += 1;

                    if config.validation.strict {
                        return Err(e).with_context(|| {
                            format!("Validation error in strict mode for '{}'", file_path)
                        });
                    }
                }
            }
        } else {
            // Validation disabled — just chunk raw content
            let raw = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => {
                    error!("Failed to read {}: {}", file_path, e);
                    continue;
                }
            };

            let chunks = chunk::chunk_markdown(&raw, None, &config.chunking);
            if chunks.is_empty() {
                warn!("No chunks produced for: {}", file_path);
                continue;
            }

            pending.push(PendingFile {
                file_path,
                frontmatter: HashMap::new(),
                chunks,
                hash,
                was_indexed,
            });
        }
    }

    // ── Batch embedding ──────────────────────────────────────────────────────
    if !pending.is_empty() {
        info!("Embedding chunks for {} changed file(s)…", pending.len());

        // Flatten all chunk texts in order, recording boundaries
        let mut all_texts: Vec<String> = Vec::new();
        let mut file_boundaries: Vec<(usize, usize)> = Vec::new(); // (start_idx, count)

        for pf in &pending {
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

        // ── Build Qdrant points, delete stale data, upsert, update state ──────
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
                    .with_context(|| {
                        format!("Failed to delete old points for '{}'", pf.file_path)
                    })?;
            }

            store
                .upsert_points(collection, points)
                .await
                .with_context(|| format!("Failed to upsert points for '{}'", pf.file_path))?;

            state
                .upsert(&pf.file_path, &pf.hash, *count as i64)
                .await
                .with_context(|| format!("Failed to update state DB for '{}'", pf.file_path))?;

            info!("Indexed {} chunk(s) from: {}", count, pf.file_path);
        }
    }

    // ── Handle orphaned (deleted) files ──────────────────────────────────────
    if !orphaned.is_empty() {
        info!("Removing {} orphaned file(s) from index", orphaned.len());
        for file_path in &orphaned {
            store
                .delete_by_file(collection, file_path)
                .await
                .with_context(|| {
                    format!("Failed to delete orphaned points for '{}'", file_path)
                })?;

            state
                .delete(file_path)
                .await
                .with_context(|| format!("Failed to delete state DB entry for '{}'", file_path))?;

            info!("Removed orphaned file: {}", file_path);
        }
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    info!(
        discovered = discovered.len(),
        indexed = pending.len(),
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
    fn compute_hash_consistent() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello world").unwrap();
        let h1 = compute_hash(&path).unwrap();
        let h2 = compute_hash(&path).unwrap();
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // SHA256 hex
    }

    #[test]
    fn compute_hash_differs_on_content() {
        let dir = TempDir::new().unwrap();
        let p1 = dir.path().join("a.txt");
        let p2 = dir.path().join("b.txt");
        std::fs::write(&p1, "hello").unwrap();
        std::fs::write(&p2, "world").unwrap();
        assert_ne!(compute_hash(&p1).unwrap(), compute_hash(&p2).unwrap());
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
}
