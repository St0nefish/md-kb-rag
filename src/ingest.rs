use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use globset::{Glob, GlobSet, GlobSetBuilder};
use sha2::{Digest, Sha256};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::{
    chunk,
    config::{IndexingConfig, ResolvedConfig},
    embed::{EmbedClient, EmbedStore},
    qdrant::{QdrantPoint, QdrantStore, VectorStore},
    state::{IndexedFile, StateDb},
    validate,
};

// ---------------------------------------------------------------------------
// File discovery
// ---------------------------------------------------------------------------

/// Parse `patterns` into a `GlobSetBuilder`, skipping (and warning on) any
/// invalid entries. Returns the builder and the count of successfully added
/// patterns. The caller decides what to do when the count is 0.
///
/// `mcp::build_include_globset` uses a similar loop but with its own fallback
/// policy (fall back to `**/*.md`); both share this helper for the per-pattern
/// parse, so glob-library error handling stays consistent.
pub(crate) fn parse_globs(patterns: &[String]) -> (GlobSetBuilder, usize) {
    let mut builder = GlobSetBuilder::new();
    let mut valid_count = 0;
    for pattern in patterns {
        match Glob::new(pattern) {
            Ok(g) => {
                builder.add(g);
                valid_count += 1;
            }
            Err(e) => {
                tracing::warn!("Skipping invalid glob pattern '{}': {}", pattern, e);
            }
        }
    }
    (builder, valid_count)
}

/// Build a `GlobSet` from `patterns`, propagating any build errors.
/// Invalid individual patterns are skipped with a warning (via [`parse_globs`]).
fn build_globset(patterns: &[String]) -> Result<GlobSet> {
    let (builder, _count) = parse_globs(patterns);
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
        if !include_set.is_match(rel_str.as_ref()) {
            continue;
        }

        // Must not match any exclude pattern
        if let Some(excl) = exclude_set
            && excl.is_match(rel_str.as_ref())
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
    /// Number of chunks from the previous index run (0 for new files).
    /// Used to trim stale tail points after a successful upsert.
    old_chunk_count: usize,
    /// File modification time as Unix timestamp (seconds). Falls back to 0 on metadata/clock error.
    mtime: i64,
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
    rel_key: &str,
    content: &str,
    full: bool,
    state_entry: Option<IndexedFile>,
    config: &ResolvedConfig,
) -> Result<FileOutcome> {
    let file_path = rel_key.to_string();
    let hash = compute_hash_from_bytes(content.as_bytes());

    // Capture mtime now — used in PendingFile regardless of validation path.
    let mtime: i64 = tokio::fs::metadata(path)
        .await
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or_else(|| {
            warn!("Could not read mtime for '{}', defaulting to 0", file_path);
            0
        });

    let old_chunk_count = state_entry
        .as_ref()
        .map(|e| e.chunk_count as usize)
        .unwrap_or(0);

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
                    old_chunk_count,
                    mtime,
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
            old_chunk_count,
            mtime,
        }))
    }
}

/// Embed all pending files and upsert their points into Qdrant.
async fn upsert_pending<E: EmbedStore, Q: VectorStore>(
    pending: &[PendingFile],
    embedder: &E,
    store: &Q,
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

    // Build all points, then batch-upsert (no pre-delete: deterministic IDs upsert in-place)
    let mut all_points: Vec<QdrantPoint> = Vec::new();

    for (pf, (start, count)) in pending.iter().zip(file_boundaries.iter()) {
        let embeddings = &all_embeddings[*start..*start + *count];

        let base_payload = pf.frontmatter.clone();
        for (chunk, vector) in pf.chunks.iter().zip(embeddings.iter()) {
            let mut payload: HashMap<String, serde_json::Value> = base_payload.clone();
            payload.insert(
                "file_path".to_string(),
                serde_json::Value::String(pf.file_path.clone()),
            );
            payload.insert("mtime".to_string(), serde_json::json!(pf.mtime));
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

            all_points.push(QdrantPoint {
                id: make_point_id(&pf.file_path, chunk.index),
                vector: vector.clone(),
                // Sparse vector for hybrid retrieval, computed from the chunk text
                // (pure-Rust tokenizer; Qdrant applies IDF server-side). Always
                // stored so toggling search.hybrid never requires a reindex.
                sparse: Some(crate::sparse::tokenize(&chunk.text)),
                payload,
            });
        }
    }

    store
        .upsert_points(collection, all_points)
        .await
        .context("Failed to batch-upsert points")?;
    // If upsert fails, old points remain and state DB is unchanged (old hash ≠ new hash),
    // so the file will be retried on the next incremental run automatically.

    // Tail trim: for files that shrank, delete stale high-index point IDs.
    // Non-fatal: warn and continue; stale tail points will be cleaned on next --full.
    for (pf, (_start, new_count)) in pending.iter().zip(file_boundaries.iter()) {
        if pf.old_chunk_count > *new_count {
            let stale_ids: Vec<String> = (*new_count..pf.old_chunk_count)
                .map(|i| make_point_id(&pf.file_path, i))
                .collect();
            if let Err(e) = store.delete_points_by_ids(collection, stale_ids).await {
                warn!(
                    file = %pf.file_path,
                    old = pf.old_chunk_count,
                    new = new_count,
                    "Tail-trim delete failed (non-fatal, will retry on next --full): {:#}",
                    e
                );
            }
        }
    }

    // Update state DB per file
    for (pf, (_start, count)) in pending.iter().zip(file_boundaries.iter()) {
        state
            .upsert(&pf.file_path, &pf.hash, *count as i64)
            .await
            .with_context(|| format!("Failed to update state DB for '{}'", pf.file_path))?;

        info!("Indexed {} chunk(s) from: {}", count, pf.file_path);
    }

    info!(
        "Upserted {} total points across {} files",
        all_texts.len(),
        pending.len()
    );

    Ok(())
}

/// Remove orphaned files (deleted from disk but still in the index).
async fn remove_orphans<Q: VectorStore>(
    orphaned: &[String],
    store: &Q,
    state: &StateDb,
    collection: &str,
) -> Result<()> {
    let orphan_refs: Vec<&str> = orphaned.iter().map(|s| s.as_str()).collect();
    store
        .delete_by_files(collection, &orphan_refs)
        .await
        .context("Failed to batch-delete orphaned points")?;

    for file_path in orphaned {
        state
            .delete(file_path)
            .await
            .with_context(|| format!("Failed to delete state DB entry for '{}'", file_path))?;

        info!("Removed orphaned file: {}", file_path);
    }
    Ok(())
}

pub async fn run_index(config: &ResolvedConfig, full: bool) -> Result<()> {
    let run_start = std::time::Instant::now();
    info!(
        mode = if full { "full" } else { "incremental" },
        data_path = config.data_path(),
        collection = %config.qdrant.collection,
        "Starting indexing run"
    );

    // Ensure git repo exists if git_url is configured
    if let Some(ref git_url) = config.source.git_url {
        let token = std::env::var(&config.source.git_token_env)
            .ok()
            .filter(|s| !s.is_empty());
        crate::git::ensure_repo(
            git_url,
            &config.source.branch,
            config.data_path(),
            token.as_deref(),
        )
        .await
        .context("Failed to ensure git repository")?;
    }

    // ── Infrastructure ──────────────────────────────────────────────────────
    let db_path = config.state_db_path();
    let state = StateDb::new(Path::new(&db_path))
        .await
        .context("Failed to open state DB")?;

    let store = QdrantStore::new(&config.qdrant).context("Failed to connect to Qdrant")?;

    let collection = &config.qdrant.collection;
    let vector_size = config.embedding.vector_size;

    let indexed_fields = config.effective_indexed_fields();

    // ── Full-mode: drop Qdrant collection so it is recreated clean ───────────
    if full {
        info!("Full reindex: dropping Qdrant collection");
        store
            .drop_collection(collection)
            .await
            .context("Failed to drop Qdrant collection for full reindex")?;
    }

    store
        .ensure_collection(collection, vector_size, &indexed_fields)
        .await
        .context("Failed to ensure Qdrant collection")?;

    // Clear state only after the collection exists; a clear failure then leaves
    // an empty collection + populated state, which recovers on the next run.
    if full {
        state.clear().await.context("Failed to clear state DB")?;
    }

    let embedder = EmbedClient::new(&config.embedding);

    // ── File discovery ───────────────────────────────────────────────────────
    // Canonicalize data_path once so that strip_prefix is reliable even when
    // the path contains symlinks. We do this AFTER ensure_repo so the directory
    // should exist by now. If canonicalize fails (path still absent), fall back
    // to the configured path with a warning — the git clone may create it later.
    let configured_data_path = PathBuf::from(config.data_path());
    let data_path: PathBuf = match configured_data_path.canonicalize() {
        Ok(p) => p,
        Err(e) => {
            warn!(
                "Could not canonicalize data_path '{}': {} — using configured path as-is",
                configured_data_path.display(),
                e
            );
            configured_data_path.clone()
        }
    };

    // Offload the synchronous directory walk to a blocking thread so we don't
    // stall the tokio executor on large knowledge bases.
    let indexing_config = config.indexing.clone();
    let walk_path = data_path.clone();
    let discovered =
        tokio::task::spawn_blocking(move || discover_files(&walk_path, &indexing_config))
            .await
            .context("File-discovery task panicked")??;

    info!("Discovered {} files", discovered.len());

    // ── Determine which previously-indexed files no longer exist ─────────────
    let all_indexed = state.list_all().await.context("Failed to list state DB")?;
    let discovered_set: HashSet<String> = discovered
        .iter()
        .map(|p| match p.strip_prefix(&data_path) {
            Ok(rel) => rel.to_string_lossy().to_string(),
            Err(_) => {
                warn!(
                    "Discovered file '{}' does not share data_path prefix",
                    p.display()
                );
                p.to_string_lossy().to_string()
            }
        })
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
    let mut empty = 0usize;
    let mut read_errors = 0usize;

    for path in &discovered {
        let rel_key = match path.strip_prefix(&data_path) {
            Ok(rel) => rel.to_string_lossy().to_string(),
            Err(_) => {
                warn!(
                    "File path '{}' does not share data_path prefix — using absolute path as key",
                    path.display()
                );
                path.to_string_lossy().to_string()
            }
        };

        // Read file once — used for hashing, validation, and chunking (fix TOCTOU #51)
        let content = match tokio::fs::read_to_string(path).await {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to read {}: {:#}", rel_key, e);
                read_errors += 1;
                continue;
            }
        };

        let state_entry = indexed_map.get(&rel_key).cloned();

        match process_file(path, &rel_key, &content, full, state_entry, config).await? {
            FileOutcome::Skipped => skipped += 1,
            FileOutcome::Invalid => invalid += 1,
            FileOutcome::Empty => empty += 1,
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
        empty,
        read_errors,
        orphans_removed = orphaned.len(),
        elapsed_secs = run_start.elapsed().as_secs_f64(),
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

    /// Helper: build a ResolvedConfig with validation disabled for simpler test setup.
    fn config_no_validation() -> ResolvedConfig {
        ResolvedConfig {
            source: Default::default(),
            indexing: Default::default(),
            frontmatter: Default::default(),
            chunking: Default::default(),
            embedding: crate::config::ResolvedEmbeddingConfig {
                base_url: "http://test:8080/v1".into(),
                model: "test-model".into(),
                api_key: None,
                vector_size: 768,
                batch_size: 32,
            },
            qdrant: crate::config::ResolvedQdrantConfig {
                url: "http://test:6334".into(),
                collection: "knowledge-base".into(),
            },
            validation: crate::config::ValidationConfig {
                enabled: false,
                ..Default::default()
            },
            webhook: Default::default(),
            mcp: Default::default(),
            rate_limit: Default::default(),
            write: Default::default(),
            search: Default::default(),
            reranking: None,
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
        let outcome = process_file(&path, "doc.md", content, false, state_entry, &config)
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
        let outcome = process_file(&path, "doc.md", content, false, state_entry, &config)
            .await
            .unwrap();
        match outcome {
            FileOutcome::Ready(pf) => {
                assert!(!pf.chunks.is_empty());
                // old_chunk_count > 0 means the file was previously indexed
                assert!(pf.old_chunk_count > 0);
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
        let outcome = process_file(&path, "doc.md", content, true, state_entry, &config)
            .await
            .unwrap();
        assert!(
            matches!(outcome, FileOutcome::Ready(_)),
            "Full mode should process even when hash matches"
        );
    }

    #[tokio::test]
    async fn process_file_new_file_no_old_chunks() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("doc.md");
        let content = "# Hello\nBody text.";
        std::fs::write(&path, content).unwrap();

        let config = config_no_validation();
        let outcome = process_file(&path, "doc.md", content, false, None, &config)
            .await
            .unwrap();
        match outcome {
            FileOutcome::Ready(pf) => assert_eq!(pf.old_chunk_count, 0),
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
        let outcome = process_file(&path, "doc.md", content, false, None, &config)
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

        let config = {
            let mut c = config_no_validation();
            c.validation.enabled = true;
            c.frontmatter = crate::config::FrontmatterConfig {
                required: vec!["title".into()],
                ..Default::default()
            };
            c
        };

        let outcome = process_file(&path, "doc.md", content, false, None, &config)
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

        let config = {
            let mut c = config_no_validation();
            c.validation.enabled = true;
            c.frontmatter = crate::config::FrontmatterConfig {
                required: vec!["description".into()],
                ..Default::default()
            };
            c
        };

        let outcome = process_file(&path, "doc.md", content, false, None, &config)
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

        let config = {
            let mut c = config_no_validation();
            c.validation = crate::config::ValidationConfig {
                enabled: true,
                strict: true,
                ..Default::default()
            };
            c.frontmatter = crate::config::FrontmatterConfig {
                required: vec!["description".into()],
                ..Default::default()
            };
            c
        };

        let result = process_file(&path, "doc.md", content, false, None, &config).await;
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

    // -----------------------------------------------------------------------
    // Mock structs for upsert_pending / remove_orphans tests
    // -----------------------------------------------------------------------

    use crate::embed::EmbedStore;
    use crate::qdrant::VectorStore;
    use std::sync::Mutex;

    struct MockEmbedClient {
        result: Result<Vec<Vec<f32>>>,
    }

    impl MockEmbedClient {
        fn ok(vecs: Vec<Vec<f32>>) -> Self {
            Self { result: Ok(vecs) }
        }

        fn err(msg: &str) -> Self {
            Self {
                result: Err(anyhow::anyhow!("{}", msg)),
            }
        }
    }

    impl EmbedStore for MockEmbedClient {
        async fn embed_texts(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
            match &self.result {
                Ok(v) => Ok(v.clone()),
                Err(e) => anyhow::bail!("{}", e),
            }
        }
    }

    struct MockVectorStore {
        delete_result: Mutex<Result<()>>,
        upsert_result: Mutex<Result<()>>,
        upsert_called: Mutex<bool>,
        upserted_points: Mutex<Vec<crate::qdrant::QdrantPoint>>,
    }

    impl MockVectorStore {
        fn all_ok() -> Self {
            Self {
                delete_result: Mutex::new(Ok(())),
                upsert_result: Mutex::new(Ok(())),
                upsert_called: Mutex::new(false),
                upserted_points: Mutex::new(Vec::new()),
            }
        }

        fn with_delete_err(msg: &str) -> Self {
            Self {
                delete_result: Mutex::new(Err(anyhow::anyhow!("{}", msg))),
                upsert_result: Mutex::new(Ok(())),
                upsert_called: Mutex::new(false),
                upserted_points: Mutex::new(Vec::new()),
            }
        }

        fn with_upsert_err(msg: &str) -> Self {
            Self {
                delete_result: Mutex::new(Ok(())),
                upsert_result: Mutex::new(Err(anyhow::anyhow!("{}", msg))),
                upsert_called: Mutex::new(false),
                upserted_points: Mutex::new(Vec::new()),
            }
        }
    }

    impl VectorStore for MockVectorStore {
        async fn upsert_points(
            &self,
            _collection: &str,
            points: Vec<crate::qdrant::QdrantPoint>,
        ) -> Result<()> {
            *self.upsert_called.lock().unwrap() = true;
            let guard = self.upsert_result.lock().unwrap();
            match &*guard {
                Ok(()) => {
                    drop(guard);
                    self.upserted_points.lock().unwrap().extend(points);
                    Ok(())
                }
                Err(e) => anyhow::bail!("{}", e),
            }
        }

        async fn delete_by_files(&self, _collection: &str, _file_paths: &[&str]) -> Result<()> {
            let guard = self.delete_result.lock().unwrap();
            match &*guard {
                Ok(()) => Ok(()),
                Err(e) => anyhow::bail!("{}", e),
            }
        }

        async fn delete_points_by_ids(&self, _collection: &str, _ids: Vec<String>) -> Result<()> {
            Ok(())
        }
    }

    struct TrackingMockVectorStore {
        delete_by_files_calls: Mutex<Vec<Vec<String>>>,
        deleted_ids: Mutex<Vec<String>>,
        upsert_result: Mutex<Result<()>>,
        upsert_called: Mutex<bool>,
        upserted_points: Mutex<Vec<crate::qdrant::QdrantPoint>>,
    }

    impl TrackingMockVectorStore {
        fn all_ok() -> Self {
            Self {
                delete_by_files_calls: Mutex::new(Vec::new()),
                deleted_ids: Mutex::new(Vec::new()),
                upsert_result: Mutex::new(Ok(())),
                upsert_called: Mutex::new(false),
                upserted_points: Mutex::new(Vec::new()),
            }
        }
    }

    impl VectorStore for TrackingMockVectorStore {
        async fn upsert_points(
            &self,
            _collection: &str,
            points: Vec<crate::qdrant::QdrantPoint>,
        ) -> Result<()> {
            *self.upsert_called.lock().unwrap() = true;
            let guard = self.upsert_result.lock().unwrap();
            match &*guard {
                Ok(()) => {
                    drop(guard);
                    self.upserted_points.lock().unwrap().extend(points);
                    Ok(())
                }
                Err(e) => anyhow::bail!("{}", e),
            }
        }

        async fn delete_by_files(&self, _collection: &str, file_paths: &[&str]) -> Result<()> {
            self.delete_by_files_calls
                .lock()
                .unwrap()
                .push(file_paths.iter().map(|s| s.to_string()).collect());
            Ok(())
        }

        async fn delete_points_by_ids(&self, _collection: &str, ids: Vec<String>) -> Result<()> {
            self.deleted_ids.lock().unwrap().extend(ids);
            Ok(())
        }
    }

    async fn test_state_db(dir: &TempDir) -> StateDb {
        let db_path = dir.path().join("state.db");
        StateDb::new(&db_path).await.unwrap()
    }

    fn make_pending(file_path: &str, chunk_count: usize, old_chunk_count: usize) -> PendingFile {
        let chunks: Vec<chunk::Chunk> = (0..chunk_count)
            .map(|i| chunk::Chunk {
                text: format!("chunk {}", i),
                index: i,
                line_start: i * 10 + 1,
                line_end: (i + 1) * 10,
            })
            .collect();
        PendingFile {
            file_path: file_path.to_string(),
            frontmatter: HashMap::new(),
            chunks,
            hash: "abc123".to_string(),
            old_chunk_count,
            mtime: 1_700_000_000,
        }
    }

    #[tokio::test]
    async fn embedding_count_mismatch_bails() {
        let dir = TempDir::new().unwrap();
        let state = test_state_db(&dir).await;

        // 2-chunk file but embedder returns only 1 vector
        let pending = vec![make_pending("data/test.md", 2, 0)];
        let embedder = MockEmbedClient::ok(vec![vec![1.0, 2.0, 3.0]]);
        let store = MockVectorStore::all_ok();

        let result = upsert_pending(&pending, &embedder, &store, &state, "test-col").await;

        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Embedding count mismatch"),
            "Expected mismatch error, got: {}",
            msg
        );
        assert!(
            !*store.upsert_called.lock().unwrap(),
            "upsert_points should not be called after mismatch"
        );
    }

    #[tokio::test]
    async fn orphan_delete_failure_preserves_state() {
        let dir = TempDir::new().unwrap();
        let state = test_state_db(&dir).await;

        // Seed state DB with an entry
        state.upsert("data/orphan.md", "hash1", 3).await.unwrap();

        let store = MockVectorStore::with_delete_err("qdrant unavailable");

        let result =
            remove_orphans(&["data/orphan.md".to_string()], &store, &state, "test-col").await;

        assert!(result.is_err());
        // State DB entry should still exist
        let entry = state.get("data/orphan.md").await.unwrap();
        assert!(
            entry.is_some(),
            "State DB entry should be preserved on delete failure"
        );
    }

    #[tokio::test]
    async fn upsert_failure_preserves_state_for_retry() {
        let dir = TempDir::new().unwrap();
        let state = test_state_db(&dir).await;

        // Seed state with a previously-indexed file
        state.upsert("data/test.md", "old-hash", 2).await.unwrap();

        let pending = vec![make_pending("data/test.md", 2, 2)];
        let embedder = MockEmbedClient::ok(vec![vec![1.0; 3], vec![2.0; 3]]);
        let store = MockVectorStore::with_upsert_err("upsert failed");

        let result = upsert_pending(&pending, &embedder, &store, &state, "test-col").await;

        assert!(result.is_err());
        // State DB entry should be PRESERVED — old hash still differs, so file will be retried
        let entry = state.get("data/test.md").await.unwrap();
        assert!(
            entry.is_some(),
            "State DB entry should be preserved after upsert failure (enables auto-retry)"
        );
    }

    #[tokio::test]
    async fn embed_error_propagates_without_upsert() {
        let dir = TempDir::new().unwrap();
        let state = test_state_db(&dir).await;

        let pending = vec![make_pending("data/test.md", 2, 0)];
        let embedder = MockEmbedClient::err("embedding service unavailable");
        let store = MockVectorStore::all_ok();

        let result = upsert_pending(&pending, &embedder, &store, &state, "test-col").await;

        assert!(result.is_err());
        assert!(
            !*store.upsert_called.lock().unwrap(),
            "upsert_points should not be called when embedding fails"
        );
    }

    #[tokio::test]
    async fn upsert_pending_happy_path() {
        let dir = TempDir::new().unwrap();
        let state = test_state_db(&dir).await;

        let pending = vec![make_pending("data/test.md", 2, 0)];
        let embedder = MockEmbedClient::ok(vec![vec![1.0; 3], vec![2.0; 3]]);
        let store = MockVectorStore::all_ok();

        let result = upsert_pending(&pending, &embedder, &store, &state, "test-col").await;

        assert!(result.is_ok());
        assert!(
            *store.upsert_called.lock().unwrap(),
            "upsert_points should be called"
        );
        // State DB should have the entry
        let entry = state.get("data/test.md").await.unwrap();
        assert!(
            entry.is_some(),
            "State DB should have entry after successful upsert"
        );
        let entry = entry.unwrap();
        assert_eq!(entry.chunk_count, 2);
        assert_eq!(entry.content_hash, "abc123");

        // Every upserted point must carry a positive integer "mtime" payload field,
        // and file_path must be the relative key.
        let points = store.upserted_points.lock().unwrap();
        assert!(!points.is_empty(), "expected at least one upserted point");
        for point in points.iter() {
            let mtime_val = point
                .payload
                .get("mtime")
                .expect("point payload must contain 'mtime'");
            let mtime = mtime_val.as_i64().expect("'mtime' must be an integer");
            assert!(
                mtime > 0,
                "'mtime' should be a positive integer, got {mtime}"
            );
            assert_eq!(
                point.payload.get("file_path").and_then(|v| v.as_str()),
                Some("data/test.md"),
                "file_path payload must be the relative key"
            );
        }
    }

    #[tokio::test]
    async fn upsert_pending_no_pre_delete_for_changed_file() {
        let dir = TempDir::new().unwrap();
        let state = test_state_db(&dir).await;

        state.upsert("data/test.md", "old-hash", 2).await.unwrap();

        let pending = vec![make_pending("data/test.md", 2, 2)];
        let embedder = MockEmbedClient::ok(vec![vec![1.0; 3], vec![2.0; 3]]);
        let store = TrackingMockVectorStore::all_ok();

        let result = upsert_pending(&pending, &embedder, &store, &state, "test-col").await;
        assert!(result.is_ok());
        assert!(
            store.delete_by_files_calls.lock().unwrap().is_empty(),
            "delete_by_files should NOT be called for in-place update"
        );
        assert!(
            *store.upsert_called.lock().unwrap(),
            "upsert_points should be called"
        );
    }

    #[tokio::test]
    async fn upsert_pending_tail_trim_on_shrink() {
        let dir = TempDir::new().unwrap();
        let state = test_state_db(&dir).await;

        state.upsert("data/shrink.md", "old-hash", 3).await.unwrap();

        // File shrinks from 3 chunks to 1
        let mut pf = make_pending("data/shrink.md", 1, 1);
        pf.old_chunk_count = 3;

        let pending = vec![pf];
        let embedder = MockEmbedClient::ok(vec![vec![1.0; 3]]);
        let store = TrackingMockVectorStore::all_ok();

        let result = upsert_pending(&pending, &embedder, &store, &state, "test-col").await;
        assert!(result.is_ok());

        let deleted_ids = store.deleted_ids.lock().unwrap().clone();
        assert_eq!(deleted_ids.len(), 2, "should delete 2 stale tail chunks");

        let expected_id1 = make_point_id("data/shrink.md", 1);
        let expected_id2 = make_point_id("data/shrink.md", 2);
        assert!(
            deleted_ids.contains(&expected_id1),
            "should delete chunk index 1"
        );
        assert!(
            deleted_ids.contains(&expected_id2),
            "should delete chunk index 2"
        );

        let entry = state.get("data/shrink.md").await.unwrap().unwrap();
        assert_eq!(entry.chunk_count, 1);
    }

    #[tokio::test]
    async fn upsert_pending_no_tail_trim_on_grow() {
        let dir = TempDir::new().unwrap();
        let state = test_state_db(&dir).await;

        state.upsert("data/grow.md", "old-hash", 1).await.unwrap();

        let mut pf = make_pending("data/grow.md", 2, 1);
        pf.old_chunk_count = 1;

        let pending = vec![pf];
        let embedder = MockEmbedClient::ok(vec![vec![1.0; 3], vec![2.0; 3]]);
        let store = TrackingMockVectorStore::all_ok();

        let result = upsert_pending(&pending, &embedder, &store, &state, "test-col").await;
        assert!(result.is_ok());

        let deleted_ids = store.deleted_ids.lock().unwrap().clone();
        assert!(deleted_ids.is_empty(), "no tail trim when file grew");
    }

    #[test]
    fn make_point_id_portable_across_runs() {
        let id1 = make_point_id("docs/guide.md", 0);
        let id2 = make_point_id("docs/guide.md", 0);
        assert_eq!(id1, id2, "same relative path + chunk index → same point ID");

        let id_abs = make_point_id("/data/docs/guide.md", 0);
        assert_ne!(
            id1, id_abs,
            "relative and absolute paths produce different IDs"
        );
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
