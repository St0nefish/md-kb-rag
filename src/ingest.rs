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
    qdrant::{IndexedField, QdrantPoint, QdrantStore, VectorStore},
    schema::{ResolvedSchema, SchemaCache},
    state::{IndexedFile, StateDb},
    status::{INDEX_STATUS, Phase, RunMode, Trigger},
    validate,
};

/// How often a long-running phase emits a progress line.
///
/// Time-based rather than every-N-files: what matters is that the log never goes quiet
/// for long enough that a healthy run looks hung, and file counts say nothing about how
/// long each one takes.
const PROGRESS_LOG_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);

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

/// Modification time as a Unix timestamp, falling back to 0 with a warning.
///
/// `label` is the path as reported to the user, which may differ from `path` (relative
/// key vs. absolute location).
async fn file_mtime(path: &Path, label: &str) -> i64 {
    tokio::fs::metadata(path)
        .await
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or_else(|| {
            warn!("Could not read mtime for '{}', defaulting to 0", label);
            0
        })
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
    /// Byte length of `content` as read from disk. Stored alongside `mtime` so the
    /// next reconcile scan (`scan_for_dirty`) can stat-compare instead of re-reading
    /// and re-hashing unchanged files.
    size: i64,
    /// Fingerprint of the schema this file was validated against.
    schema_hash: String,
}

/// Result of processing a single discovered file.
enum FileOutcome {
    /// Unchanged since the last run. Carries the content hash so `index_paths` can tell
    /// whether the document metadata index is in sync without a per-file query.
    Skipped {
        hash: String,
    },
    Invalid,
    Empty,
    Ready(PendingFile),
}

/// Process a single file: hash, skip-if-unchanged, validate, chunk.
///
/// `force` bypasses the skip-if-unchanged check below — set only by `index_paths`'
/// destructive `md-kb-rag index --full` path, where the state DB has just been
/// cleared, so there is nothing meaningful to compare against anyway.
#[allow(clippy::too_many_arguments)]
async fn process_file(
    path: &Path,
    rel_key: &str,
    content: &str,
    force: bool,
    state_entry: Option<IndexedFile>,
    config: &ResolvedConfig,
    schema: &ResolvedSchema,
    schema_hash: &str,
) -> Result<FileOutcome> {
    let file_path = rel_key.to_string();
    let hash = compute_hash_from_bytes(content.as_bytes());
    let size = content.len() as i64;

    // Capture mtime now — used in PendingFile regardless of validation path.
    let mtime = file_mtime(path, &file_path).await;

    let old_chunk_count = state_entry
        .as_ref()
        .map(|e| e.chunk_count as usize)
        .unwrap_or(0);

    // Skip unchanged files unless forced. The schema fingerprint is part of the
    // condition: editing a .kb-schema.yaml changes no document's bytes, so without this
    // a tightened rule would never be applied to anything already indexed.
    if !force
        && let Some(ref entry) = state_entry
        && entry.content_hash == hash
        && entry.schema_hash == schema_hash
    {
        debug!("Unchanged, skipping: {}", file_path);
        return Ok(FileOutcome::Skipped { hash });
    }

    if config.validation.enabled {
        match validate::validate_content(path, content, schema, &config.validation).await {
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
                    size,
                    schema_hash: schema_hash.to_string(),
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
        // Validation disabled — still PARSE frontmatter, just don't enforce anything.
        // The metadata backfill path parses unconditionally, so returning an empty map
        // here would let Qdrant and SQLite hold different frontmatter for the same
        // document, and search filters would silently never match it.
        let (frontmatter, body) = validate::parse_frontmatter(content, schema);
        let description = frontmatter
            .get("description")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let chunks = chunk::chunk_markdown(&body, description.as_deref(), &config.chunking);
        if chunks.is_empty() {
            warn!("No chunks produced for: {}", file_path);
            return Ok(FileOutcome::Empty);
        }

        Ok(FileOutcome::Ready(PendingFile {
            file_path,
            frontmatter,
            chunks,
            hash,
            old_chunk_count,
            mtime,
            size,
            schema_hash: schema_hash.to_string(),
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

    // Publish the denominator before the call blocks. `embed_texts` reports each batch
    // as it completes, so `/status` can show real progress through what is by far the
    // longest phase of a run — on a full re-embed this single await can take minutes.
    INDEX_STATUS.set_chunks_total(all_texts.len() as u64);

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

        let base_payload = with_derived_domain(&pf.frontmatter, &pf.file_path);

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
    // The points are already in Qdrant at this stage, so a bookkeeping failure for one
    // file must not abandon the rest of the batch — that would leave later files with
    // vectors but no state row, and they would be needlessly re-embedded next run.
    // Record and continue; every failure mode here self-heals on the following run.
    let mut bookkeeping_failures = 0usize;
    for (pf, (_start, count)) in pending.iter().zip(file_boundaries.iter()) {
        if let Err(e) = state
            .upsert(
                &pf.file_path,
                &pf.hash,
                *count as i64,
                &pf.schema_hash,
                pf.mtime,
                pf.size,
            )
            .await
        {
            error!("Failed to update state DB for '{}': {:#}", pf.file_path, e);
            bookkeeping_failures += 1;
            continue;
        }

        if let Err(e) = state
            .upsert_document_metadata(
                &pf.file_path,
                &with_derived_domain(&pf.frontmatter, &pf.file_path),
                pf.mtime,
                &pf.hash,
                *count as i64,
            )
            .await
        {
            // The state row is already written, so this file's metadata is stale until
            // the next run's backfill notices the hash mismatch and repairs it.
            error!(
                "Failed to update document metadata for '{}': {:#}",
                pf.file_path, e
            );
            bookkeeping_failures += 1;
            continue;
        }

        // Per-file at debug: on a full reindex this fires once per document, which
        // drowns the progress and summary lines that actually answer "is it working?".
        // The aggregate below carries the same information for a whole batch.
        debug!(file = %pf.file_path, chunks = *count, "Indexed file");
    }

    if bookkeeping_failures > 0 {
        warn!(
            "{} file(s) had bookkeeping failures; they will be repaired on the next run",
            bookkeeping_failures
        );
    }

    info!(
        points = all_texts.len(),
        files = pending.len(),
        bookkeeping_failures,
        "Upserted points"
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

    // Vectors for the whole batch are already deleted, so stopping at the first
    // bookkeeping failure would leave later orphans visible to `list_documents` with no
    // vectors behind them. Continue; orphan detection is idempotent and retries.
    for file_path in orphaned {
        // Metadata first, bookkeeping second. Orphan detection is driven off
        // `indexed_files`, so clearing that row first and then failing would drop the
        // file out of detection permanently and strand its metadata with no sweep able
        // to find it again. This order leaves it detectable, so the next run retries.
        if let Err(e) = state.delete_document(file_path).await {
            error!(
                "Failed to delete document metadata for '{}': {:#}",
                file_path, e
            );
            continue;
        }
        if let Err(e) = state.delete(file_path).await {
            error!(
                "Failed to delete state DB entry for '{}': {:#}",
                file_path, e
            );
            continue;
        }

        info!("Removed orphaned file: {}", file_path);
    }
    Ok(())
}

/// Frontmatter with `domain` set from the document's top-level folder.
///
/// Applied identically to the Qdrant payload and the metadata index so a `domain`
/// filter behaves the same through `search` and `list_documents`. Any `domain:` key an
/// author wrote in frontmatter is overridden — location is the single source of truth.
fn with_derived_domain(
    frontmatter: &HashMap<String, serde_json::Value>,
    rel_path: &str,
) -> HashMap<String, serde_json::Value> {
    let mut out = frontmatter.clone();
    let authored = frontmatter.get("domain").and_then(|v| v.as_str());

    match derive_domain(rel_path) {
        Some(domain) => {
            // Authors migrating from the old convention may still carry a `domain:`
            // key. Overriding it silently would make `search(domain=…)` stop finding
            // their document with no indication why, so say so once per index.
            if let Some(authored) = authored
                && authored != domain
            {
                warn!(
                    file = rel_path,
                    "frontmatter says domain '{}' but the folder says '{}'; using the \
                     folder. Remove the frontmatter key — domain is derived from \
                     location now.",
                    authored,
                    domain
                );
            }
            out.insert("domain".to_string(), serde_json::Value::String(domain));
        }
        None => {
            if authored.is_some() {
                warn!(
                    file = rel_path,
                    "dropping frontmatter 'domain': documents at the knowledge-base \
                     root belong to no area"
                );
            }
            out.remove("domain");
        }
    }
    out
}

/// Top-level folder of a KB-relative path, which is what `domain` now means.
///
/// Returns `None` for a document sitting directly at the knowledge-base root, which
/// belongs to no area.
pub(crate) fn derive_domain(rel_path: &str) -> Option<String> {
    let mut components = Path::new(rel_path).components();
    let first = components.next()?;
    // Only a real directory component counts, and only when something follows it.
    components.next()?;
    match first {
        std::path::Component::Normal(name) => name.to_str().map(str::to_string),
        _ => None,
    }
}

/// Payload fields to index, unioning the schema tree with the legacy config list.
///
/// `config.effective_indexed_fields()` still contributes `file_path`, which is not a
/// frontmatter field but is needed for path lookups.
fn merge_indexed_fields(config: &ResolvedConfig, schemas: &SchemaCache) -> Vec<IndexedField> {
    let mut fields = schemas.all_indexed_fields();
    for name in config.effective_indexed_fields() {
        if !fields.iter().any(|f| f.name == name) {
            fields.push(IndexedField::keyword(name));
        }
    }
    fields.sort_by(|a, b| a.name.cmp(&b.name));
    fields
}

/// Fill in document metadata for files the incremental pass skipped.
///
/// Runs when a file is unchanged by content hash but its metadata row is missing or
/// stale — the case an existing deployment hits on its first run after this feature
/// lands. Parses frontmatter only: no chunking, no embedding, no Qdrant writes, so it
/// costs nothing beyond a file read. Per-file failures are logged and retried next run
/// rather than failing the whole index.
async fn backfill_document_metadata(
    queue: &[(String, PathBuf)],
    state: &StateDb,
    indexed: &HashMap<String, IndexedFile>,
    schemas: &SchemaCache,
) -> usize {
    let mut filled = 0usize;

    for (rel_key, path) in queue {
        let content = match tokio::fs::read_to_string(path).await {
            Ok(content) => content,
            Err(e) => {
                warn!("Metadata backfill: failed to read '{}': {:#}", rel_key, e);
                continue;
            }
        };

        let hash = compute_hash_from_bytes(content.as_bytes());
        let schema = schemas.resolve_for(Path::new(rel_key));
        let (frontmatter, _body) = validate::parse_frontmatter(&content, schema);
        let frontmatter = with_derived_domain(&frontmatter, rel_key);
        let mtime = file_mtime(path, rel_key).await;
        let chunk_count = indexed.get(rel_key).map(|e| e.chunk_count).unwrap_or(0);

        match state
            .upsert_document_metadata(rel_key, &frontmatter, mtime, &hash, chunk_count)
            .await
        {
            Ok(()) => filled += 1,
            Err(e) => warn!("Metadata backfill failed for '{}': {:#}", rel_key, e),
        }
    }

    filled
}

// ---------------------------------------------------------------------------
// Path resolution helpers shared by the scan and the scoped indexer
// ---------------------------------------------------------------------------

/// Repo-relative key for `path`, matching exactly how `indexed_files.file_path` (and
/// every Qdrant payload's `file_path`) is stored: `path` stripped of the canonical
/// `data_path` prefix.
///
/// Every producer of a path that eventually reaches [`index_paths`] — the reconcile
/// scan, a webhook's `git diff --name-status`, a write tool's own `rel_path` — has to
/// agree with this shape, or the mismatch silently orphans points: a path that never
/// matches an existing key is treated as brand new instead of as the file it actually
/// is, and the old key's rows/vectors are never revisited by anything.
fn rel_key_of(path: &Path, data_path: &Path) -> String {
    match path.strip_prefix(data_path) {
        Ok(rel) => rel.to_string_lossy().to_string(),
        Err(_) => {
            warn!(
                "Path '{}' does not share data_path prefix — using absolute path as key",
                path.display()
            );
            path.to_string_lossy().to_string()
        }
    }
}

/// Canonicalize `config.data_path()` (falling back to the configured path with a
/// warning if it does not exist yet — the git clone may create it later) and walk it
/// for every indexable file, off the executor since it is a synchronous directory
/// walk. Returns the canonical root plus every discovered file as a path relative to
/// it, in [`rel_key_of`]'s shape.
async fn discover_relative(config: &ResolvedConfig) -> Result<(PathBuf, Vec<PathBuf>)> {
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

    INDEX_STATUS.set_phase(Phase::Discovering);
    let indexing_config = config.indexing.clone();
    let walk_path = data_path.clone();
    let discovered =
        tokio::task::spawn_blocking(move || discover_files(&walk_path, &indexing_config))
            .await
            .context("File-discovery task panicked")??;

    info!("Discovered {} files", discovered.len());
    let rel: Vec<PathBuf> = discovered
        .iter()
        .map(|p| PathBuf::from(rel_key_of(p, &data_path)))
        .collect();
    Ok((data_path, rel))
}

// ---------------------------------------------------------------------------
// The reconcile scan — read-only, produces a worklist
// ---------------------------------------------------------------------------

/// How many `indexed_files` rows [`scan_for_dirty`] holds in memory at once. See
/// [`StateDb::fetch_indexed_files_page`] for why this is paged rather than loaded in
/// one query.
const SCAN_PAGE_SIZE: i64 = 1000;

/// Detect which repo-relative paths need attention from [`index_paths`], without
/// touching Qdrant or SQLite. This is the ONLY thing that decides a full-corpus
/// reconcile is needed — `index_paths` never walks the filesystem on its own — and it
/// is deliberately read-only: mutation lives in exactly one place (`index_paths`), so
/// there is exactly one place a bug in embedding/upsert/purge logic can hide.
///
/// A path is dirty for one of three reasons, and the scan does the minimum work
/// needed to catch each one without reading a single file's content:
///
/// 1. **Changed or new.** It exists on disk with no `indexed_files` row, or with an
///    `mtime`/`size` that no longer matches the row. This is a pre-filter only — the
///    content hash remains the sole authority on whether a file actually changed, and
///    that authoritative check happens when `index_paths` re-reads the file. A false
///    positive here (mtime touched, bytes unchanged) costs one wasted hash comparison
///    downstream, not a wasted re-embed; a false negative would silently drop a real
///    change, which stat cannot produce short of the clock going backwards.
/// 2. **Orphaned.** It has an `indexed_files` row but no longer exists on disk.
/// 3. **Metadata-stale.** Content is unchanged (same `indexed_files.content_hash`),
///    but `documents` has no row for it, or a different hash — the case
///    `index_paths` resolves with a cheap parse-only refresh, no re-embedding.
///
/// At a corpus size of thousands to tens of thousands of documents, content-hashing
/// (or even just fully materializing) the whole corpus on every sweep would dominate
/// the sweep's cost; this function never does either.
pub async fn scan_for_dirty(config: &ResolvedConfig) -> Result<Vec<PathBuf>> {
    let state = StateDb::new(Path::new(&config.state_db_path()))
        .await
        .context("Failed to open state DB")?;

    let (data_path, discovered) = discover_relative(config).await?;
    INDEX_STATUS.set_files_total(discovered.len() as u64);

    let schemas = SchemaCache::build(&data_path, &config.frontmatter);

    // Every path currently on disk. Needed twice: to tell an orphan (row, no file)
    // from a live one while paging `indexed_files`, and — via `visited`, below — to
    // find files that exist but have no row at all (brand new).
    let seen: HashSet<String> = discovered
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    let mut visited: HashSet<String> = HashSet::with_capacity(seen.len());
    let mut dirty: HashSet<PathBuf> = HashSet::new();

    INDEX_STATUS.set_phase(Phase::Scanning);
    let mut scanned = 0usize;
    let mut last_progress = std::time::Instant::now();
    let mut offset = 0i64;
    loop {
        let page = state
            .fetch_indexed_files_page(SCAN_PAGE_SIZE, offset)
            .await
            .context("Failed to page indexed_files during reconcile scan")?;
        if page.is_empty() {
            break;
        }
        let page_len = page.len();

        for row in &page {
            scanned += 1;
            INDEX_STATUS.set_files_done(scanned as u64);
            if last_progress.elapsed() >= PROGRESS_LOG_INTERVAL {
                info!(scanned, "Reconcile scan in progress…");
                last_progress = std::time::Instant::now();
            }

            visited.insert(row.file_path.clone());

            if !seen.contains(&row.file_path) {
                // Row survives, file does not: orphaned.
                dirty.insert(PathBuf::from(&row.file_path));
                continue;
            }

            let rel = Path::new(&row.file_path);
            if schemas.is_frozen(rel).is_some() {
                // Frozen scopes are never touched by the scan or the indexer, exactly
                // as a full walk-based run has always skipped them.
                continue;
            }

            // Reason 1a: the schema fingerprint moved. Cheap — no disk I/O, just a
            // lookup against the already-built schema tree — and it can flip a file
            // dirty even when its bytes and stat metadata are untouched.
            if schemas.resolve_for(rel).fingerprint() != row.schema_hash {
                dirty.insert(PathBuf::from(&row.file_path));
                continue;
            }

            // Reason 1b: stat pre-filter. The only per-file disk access this function
            // performs, and it is a metadata syscall, never a content read.
            let abs = data_path.join(&row.file_path);
            match tokio::fs::metadata(&abs).await {
                Ok(meta) => {
                    let mtime = meta
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs() as i64)
                        .unwrap_or(0);
                    let size = meta.len() as i64;
                    if mtime != row.mtime || size != row.size {
                        dirty.insert(PathBuf::from(&row.file_path));
                        continue;
                    }
                }
                Err(e) => {
                    // Existed moments ago in `discovered`; a stat failure now is most
                    // likely a TOCTOU race (deleted or replaced between the walk and
                    // this call). Mark it dirty rather than silently skipping —
                    // `index_paths` re-checks existence itself and will resolve this
                    // one way or the other.
                    debug!(
                        "Stat failed for '{}', marking dirty: {:#}",
                        row.file_path, e
                    );
                    dirty.insert(PathBuf::from(&row.file_path));
                    continue;
                }
            }

            // Reason 3: metadata staleness. Unchanged content, but the metadata index
            // disagrees with (or lacks) it.
            if row.doc_hash.as_deref() != Some(row.content_hash.as_str()) {
                dirty.insert(PathBuf::from(&row.file_path));
            }
        }

        if (page_len as i64) < SCAN_PAGE_SIZE {
            break;
        }
        offset += SCAN_PAGE_SIZE;
    }

    // Reason 2: files on disk with no `indexed_files` row at all — but still subject
    // to the same frozen-scope exclusion as every other path here. Without this check
    // a new file dropped into a frozen scope would be marked dirty on every sweep
    // forever (it never gets an `indexed_files` row, since `index_paths` also skips
    // frozen paths), for no benefit — it can never actually be indexed until the
    // schema is fixed.
    for rel_key in seen.difference(&visited) {
        if schemas.is_frozen(Path::new(rel_key)).is_some() {
            continue;
        }
        dirty.insert(PathBuf::from(rel_key));
    }

    info!(
        dirty = dirty.len(),
        discovered = discovered.len(),
        "Reconcile scan complete"
    );

    Ok(dirty.into_iter().collect())
}

// ---------------------------------------------------------------------------
// The scoped indexer — the only function that mutates the index
// ---------------------------------------------------------------------------

/// (Re)index exactly the given repo-relative `paths`, and nothing else. Recording
/// start/finish in [`INDEX_STATUS`] like every indexing run.
///
/// **This is the only function in the whole system that ever mutates Qdrant,
/// `indexed_files`, `documents`, or `document_fields`.** Every producer of work — the
/// MCP write tools, the webhook handler, the background reindex worker (fed by
/// [`scan_for_dirty`] or by a `git diff`), and the CLI (via [`scan_and_index`]) — first
/// turns "what changed" into a list of paths, then calls this. That is a deliberate
/// invariant, not an accident of how the code happened to get organized: there is
/// exactly one place an embedding/upsert/purge bug can hide, and exactly one place to
/// look when the index and the filesystem disagree.
///
/// For each path: if the file exists on disk it is (re)read and hashed; if its content
/// or governing schema fingerprint actually changed (or `force` is set), it is
/// chunked, embedded, and upserted. If unchanged but the metadata index is stale, only
/// that (cheap, parse-only) metadata is refreshed — **no re-embedding**. If the file
/// does not exist on disk, its points and rows are purged. This exactly mirrors
/// [`FileOutcome`] — see [`process_file`].
///
/// `force = true` bypasses the skip-if-unchanged check and, before touching any path,
/// drops and recreates the Qdrant collection and clears the state DB — this is
/// `md-kb-rag index --full`'s destructive-rebuild semantics, unchanged from before this
/// module split scanning out of indexing. **It is safe only when `paths` is the
/// complete set of files on disk**, which [`scan_and_index`] guarantees by discovering
/// fresh rather than scanning. Calling this with `force = true` on a partial path list
/// would drop the whole collection and rebuild it from just those paths, destroying
/// every other document's vectors — nothing in the worker/queue path ever sets `force`;
/// only the CLI's `--full` flag does.
pub async fn index_paths(
    config: &ResolvedConfig,
    paths: &[PathBuf],
    force: bool,
    trigger: Trigger,
) -> Result<()> {
    let mode = RunMode::from_full(force);
    let run = INDEX_STATUS.begin(mode, trigger);

    let result = index_paths_inner(config, paths, force).await;

    match &result {
        Ok(()) => run.finish(None),
        Err(e) => {
            // The failure path needs a terminal log line of its own. A run that simply
            // stops emitting is indistinguishable from one still working, which is the
            // ambiguity this whole module exists to remove.
            error!(
                mode = mode.as_str(),
                trigger = trigger.as_str(),
                "Indexing run failed: {:#}",
                e
            );
            run.finish(Some(format!("{e:#}")));
        }
    }

    result
}

async fn index_paths_inner(config: &ResolvedConfig, paths: &[PathBuf], force: bool) -> Result<()> {
    let run_start = std::time::Instant::now();
    info!(
        mode = if force { "full" } else { "scoped" },
        trigger_paths = paths.len(),
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

    // Canonicalize data_path the same way discover_relative does, so joining it with a
    // repo-relative path and later stripping it again round-trips exactly.
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

    // Discover and merge every .kb-schema.yaml once. Resolution afterwards is an
    // in-memory prefix lookup, so this stays O(schema files) rather than O(paths).
    let schemas = SchemaCache::build(&data_path, &config.frontmatter);
    for (scope, reason) in schemas.broken_scopes() {
        error!(
            "Invalid schema at {}/{}: {} — documents in this scope are frozen and will \
             not be indexed until it is fixed",
            scope.display(),
            crate::schema::SCHEMA_FILE_NAME,
            reason
        );
    }

    // Union of every `indexed` dot-path across the whole schema tree. Payload indexes
    // are collection-wide, so a field declared only in a deep scope still has to be
    // registered here or filtering on it silently fails.
    let indexed_fields = merge_indexed_fields(config, &schemas);

    // A full reindex drops the collection and rebuilds it, but frozen documents are
    // skipped during the rebuild — so their vectors would be deleted and never
    // restored, leaving them invisible to search while still listed in the metadata
    // index. Refuse rather than destroy data; scoped indexing is unaffected.
    if force && schemas.broken_scopes().count() > 0 {
        let scopes: Vec<String> = schemas
            .broken_scopes()
            .map(|(dir, _)| dir.display().to_string())
            .collect();
        anyhow::bail!(
            "Refusing a full reindex while {} schema file(s) are invalid ({}). A full \
             run rebuilds the collection from scratch and cannot reindex frozen scopes, \
             so their vectors would be lost. Fix the schema(s), or run a scoped/incremental \
             index instead.",
            scopes.len(),
            scopes.join(", ")
        );
    }

    // ── force: drop Qdrant collection so it is recreated clean ───────────────
    if force {
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
    if force {
        state.clear().await.context("Failed to clear state DB")?;
    }

    let embedder = EmbedClient::new(&config.embedding);

    INDEX_STATUS.set_files_total(paths.len() as u64);

    let rel_keys: Vec<String> = paths
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();

    // Batch-loaded for exactly the paths in scope — not the whole corpus. A dirty set
    // from a big reconcile sweep can run to thousands of paths, so this is one or a
    // few round trips (see `SQLITE_MAX_PARAMS_PER_QUERY`) rather than one per path.
    let state_map = if force {
        // The state DB was just cleared above; every lookup would be a wasted query
        // that always returns nothing.
        HashMap::new()
    } else {
        state
            .get_many(&rel_keys)
            .await
            .context("Failed to load state rows for the given paths")?
    };
    let document_hashes = if force {
        HashMap::new()
    } else {
        state
            .get_document_hashes_many(&rel_keys)
            .await
            .context("Failed to load document metadata hashes for the given paths")?
    };

    // ── Per-path processing ──────────────────────────────────────────────────
    let mut pending: Vec<PendingFile> = Vec::new();
    let mut backfill_queue: Vec<(String, PathBuf)> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    let mut skipped = 0usize;
    let mut invalid = 0usize;
    let mut empty = 0usize;
    let mut read_errors = 0usize;
    let mut frozen = 0usize;

    INDEX_STATUS.set_phase(Phase::Scanning);
    let mut scanned = 0usize;
    let mut last_progress = std::time::Instant::now();

    for rel_key in &rel_keys {
        // Counted at the top so the `continue` arms below still advance progress —
        // a scan that stalls on unreadable files should look like it is moving.
        scanned += 1;
        INDEX_STATUS.set_files_done(scanned as u64);
        if last_progress.elapsed() >= PROGRESS_LOG_INTERVAL {
            info!(scanned, total = rel_keys.len(), "Indexing given paths…");
            last_progress = std::time::Instant::now();
        }

        let abs_path = data_path.join(rel_key);

        // Missing on disk: treat as a delete, purged in a single batch below rather
        // than one Qdrant round trip per path.
        if !abs_path.exists() {
            missing.push(rel_key.clone());
            continue;
        }

        let rel = Path::new(rel_key.as_str());
        if let Some(reason) = schemas.is_frozen(rel) {
            // The schema governing this document failed to parse. Applying the
            // parent's rules instead would silently enforce rules we know are wrong
            // across a whole subtree, so the scope is frozen: nothing here is indexed
            // or re-indexed, and whatever is already in the index stays untouched.
            debug!("Frozen scope, skipping {}: {}", rel_key, reason);
            frozen += 1;
            continue;
        }

        // Read file once — used for hashing, validation, and chunking (fix TOCTOU #51)
        let content = match tokio::fs::read_to_string(&abs_path).await {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to read {}: {:#}", rel_key, e);
                read_errors += 1;
                continue;
            }
        };

        let schema = schemas.resolve_for(rel);
        let schema_hash = schema.fingerprint();
        let state_entry = state_map.get(rel_key).cloned();

        match process_file(
            &abs_path,
            rel_key,
            &content,
            force,
            state_entry,
            config,
            schema,
            &schema_hash,
        )
        .await?
        {
            FileOutcome::Skipped { hash } => {
                skipped += 1;
                // Unchanged content, but the metadata index may still be missing or
                // stale for this file — queue it for a cheap parse-only backfill.
                if document_hashes.get(rel_key) != Some(&hash) {
                    backfill_queue.push((rel_key.clone(), abs_path.clone()));
                }
            }
            FileOutcome::Invalid => invalid += 1,
            FileOutcome::Empty => empty += 1,
            FileOutcome::Ready(pf) => pending.push(pf),
        }
    }

    // ── Batch embedding & upsert ────────────────────────────────────────────
    let pending_count = pending.len();
    if !pending.is_empty() {
        INDEX_STATUS.set_phase(Phase::Embedding);
        info!("Embedding chunks for {} changed file(s)…", pending_count);
        upsert_pending(&pending, &embedder, &store, &state, collection).await?;
    }

    // ── Backfill metadata for unchanged files ────────────────────────────────
    let backfilled = if backfill_queue.is_empty() {
        0
    } else {
        INDEX_STATUS.set_phase(Phase::Backfilling);
        info!(
            "Backfilling document metadata for {} unchanged file(s)",
            backfill_queue.len()
        );
        backfill_document_metadata(&backfill_queue, &state, &state_map, &schemas).await
    };

    // ── Handle missing (deleted) files ───────────────────────────────────────
    if !missing.is_empty() {
        INDEX_STATUS.set_phase(Phase::RemovingOrphans);
        info!("Removing {} missing file(s) from index", missing.len());
        remove_orphans(&missing, &store, &state, collection).await?;
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    // `discovered` keeps its established meaning of "how many paths this run
    // considered" — for a scoped run that is the size of the given worklist, not a
    // filesystem walk count, which is now `scan_for_dirty`'s concern entirely.
    let counters = crate::status::RunCounters {
        discovered: paths.len() as u64,
        indexed: pending_count as u64,
        skipped: skipped as u64,
        invalid: invalid as u64,
        empty: empty as u64,
        read_errors: read_errors as u64,
        metadata_backfilled: backfilled as u64,
        frozen_by_broken_schema: frozen as u64,
        broken_schemas: schemas.broken_scopes().count() as u64,
        orphans_removed: missing.len() as u64,
    };
    INDEX_STATUS.set_counters(counters.clone());

    info!(
        discovered = counters.discovered,
        indexed = counters.indexed,
        skipped = counters.skipped,
        invalid = counters.invalid,
        empty = counters.empty,
        read_errors = counters.read_errors,
        metadata_backfilled = counters.metadata_backfilled,
        frozen_by_broken_schema = counters.frozen_by_broken_schema,
        broken_schemas = counters.broken_schemas,
        orphans_removed = counters.orphans_removed,
        elapsed_secs = run_start.elapsed().as_secs_f64(),
        "Indexing run complete"
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Synchronous scan-then-index, for callers with no worker
// ---------------------------------------------------------------------------

/// Scan, then index whatever the scan found — for callers that have no background
/// worker to hand a dirty-path queue to: the `md-kb-rag index` CLI subcommand, and the
/// server's own pre-worker bootstrap immediately after a fresh git clone. Both need a
/// synchronous, in-process "bring the index up to date" call, which this provides by
/// composing [`scan_for_dirty`] and [`index_paths`].
///
/// `force = true` is `--full`: rather than scanning (which would compare against state
/// this call is about to clear, and would trivially mark everything dirty once state
/// IS clear), it discovers every file directly and indexes all of them with
/// `force = true` — see [`index_paths`] for why that combination is only ever safe with
/// a complete path list, which a fresh discovery walk guarantees.
pub async fn scan_and_index(config: &ResolvedConfig, force: bool, trigger: Trigger) -> Result<()> {
    if force {
        let (_data_path, all_paths) = discover_relative(config).await?;
        index_paths(config, &all_paths, true, trigger).await
    } else {
        let dirty = scan_for_dirty(config)
            .await
            .context("Reconcile scan failed")?;
        index_paths(config, &dirty, false, trigger).await
    }
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
                request_timeout_secs: 60,
                batch_concurrency: 4,
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
            provenance: Default::default(),
        }
    }

    /// The one test that drives the process-global `INDEX_STATUS`.
    ///
    /// Every other status test uses a local `IndexStatus`, so nothing here races. If a
    /// second global-touching test is ever added, both need serializing — `cargo test`
    /// runs tests in parallel threads within one process.
    #[tokio::test]
    async fn index_paths_records_a_failed_run_in_the_global_status() {
        let dir = TempDir::new().unwrap();
        let mut config = config_no_validation();
        config.source.data_path = Some(dir.path().to_string_lossy().into_owned());
        // Port 1 refuses immediately, so `ensure_collection` fails before any per-path
        // work and the run ends in an error without needing any live service.
        config.qdrant.url = "http://127.0.0.1:1".into();

        let result = index_paths(&config, &[], false, Trigger::Cli).await;
        assert!(result.is_err(), "expected the run to fail");

        // Without this assertion, swapping the Ok/Err arms in `index_paths` — reporting a
        // failed run as a success — passes the entire suite. That would defeat the
        // point of the feature: `/status` would claim the index is healthy while every
        // run is failing, and `kb_index_last_success_timestamp_seconds` would keep
        // advancing so no age-based alert would ever fire.
        let snap = crate::status::INDEX_STATUS.snapshot();
        assert!(!snap.indexing, "the run must not still be marked in flight");

        let last = snap.last_run.expect("the failed run must be recorded");
        assert!(!last.success, "a failed run must not report success");
        assert_eq!(last.mode, RunMode::Incremental);
        assert_eq!(last.trigger, Trigger::Cli);
        assert!(
            last.error.is_some_and(|e| !e.is_empty()),
            "the failure needs a reason attached"
        );
        assert!(snap.runs_failed >= 1);
        assert!(
            snap.last_success_unix.is_none(),
            "a failing run must not stamp a success timestamp"
        );
    }

    // -- scan_for_dirty -------------------------------------------------------
    //
    // Pure state-DB + filesystem behavior — no Qdrant or embeddings involved, since
    // the scan never touches either. Every test opens a real (tempfile-backed)
    // StateDb, matching how `scan_for_dirty` itself opens one.

    /// Fingerprint of the schema `scan_for_dirty` will compute for `rel_path` given a
    /// KB rooted at `data_path` with no `.kb-schema.yaml` files — i.e. the plain
    /// config-derived root schema, computed the exact same way `scan_for_dirty` does
    /// internally, so tests never have to assume anything about the hash's shape.
    fn expected_schema_hash(
        data_path: &std::path::Path,
        frontmatter: &crate::config::FrontmatterConfig,
    ) -> String {
        let schemas = SchemaCache::build(data_path, frontmatter);
        schemas
            .resolve_for(std::path::Path::new("doc.md"))
            .fingerprint()
    }

    /// Build a config rooted at `dir`, with a real (not-yet-created) state DB path.
    fn scan_test_config(dir: &TempDir) -> ResolvedConfig {
        let mut config = config_no_validation();
        config.source.data_path = Some(dir.path().to_string_lossy().into_owned());
        config
    }

    async fn open_scan_test_db(config: &ResolvedConfig) -> StateDb {
        StateDb::new(Path::new(&config.state_db_path()))
            .await
            .unwrap()
    }

    fn stat(path: &Path) -> (i64, i64) {
        let meta = std::fs::metadata(path).unwrap();
        let mtime = meta
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        (mtime, meta.len() as i64)
    }

    #[tokio::test]
    async fn scan_for_dirty_flags_a_file_with_no_indexed_files_row() {
        let dir = TempDir::new().unwrap();
        let config = scan_test_config(&dir);
        std::fs::write(dir.path().join("new.md"), "# New").unwrap();

        let dirty = scan_for_dirty(&config).await.unwrap();
        assert_eq!(dirty, vec![PathBuf::from("new.md")]);
    }

    #[tokio::test]
    async fn scan_for_dirty_ignores_a_file_whose_stat_and_metadata_are_unchanged() {
        let dir = TempDir::new().unwrap();
        let config = scan_test_config(&dir);
        let path = dir.path().join("doc.md");
        std::fs::write(&path, "# Doc").unwrap();
        let (mtime, size) = stat(&path);
        let schema_hash = expected_schema_hash(dir.path(), &config.frontmatter);

        let db = open_scan_test_db(&config).await;
        db.upsert("doc.md", "some-hash", 1, &schema_hash, mtime, size)
            .await
            .unwrap();
        let mut fm = HashMap::new();
        fm.insert("title".into(), serde_json::json!("Doc"));
        db.upsert_document_metadata("doc.md", &fm, mtime, "some-hash", 1)
            .await
            .unwrap();

        let dirty = scan_for_dirty(&config).await.unwrap();
        assert!(
            dirty.is_empty(),
            "unchanged stat + fresh metadata must not be marked dirty: {dirty:?}"
        );
    }

    #[tokio::test]
    async fn scan_for_dirty_flags_a_file_whose_mtime_changed() {
        let dir = TempDir::new().unwrap();
        let config = scan_test_config(&dir);
        let path = dir.path().join("doc.md");
        std::fs::write(&path, "# Doc").unwrap();
        let (mtime, size) = stat(&path);
        let schema_hash = expected_schema_hash(dir.path(), &config.frontmatter);

        let db = open_scan_test_db(&config).await;
        // Record a DIFFERENT mtime than what's actually on disk, simulating a file
        // that was touched (or genuinely edited) since the last index.
        db.upsert("doc.md", "some-hash", 1, &schema_hash, mtime - 1000, size)
            .await
            .unwrap();
        let mut fm = HashMap::new();
        fm.insert("title".into(), serde_json::json!("Doc"));
        db.upsert_document_metadata("doc.md", &fm, mtime, "some-hash", 1)
            .await
            .unwrap();

        let dirty = scan_for_dirty(&config).await.unwrap();
        assert_eq!(dirty, vec![PathBuf::from("doc.md")]);
    }

    #[tokio::test]
    async fn scan_for_dirty_flags_a_row_whose_file_no_longer_exists() {
        let dir = TempDir::new().unwrap();
        let config = scan_test_config(&dir);
        let schema_hash = expected_schema_hash(dir.path(), &config.frontmatter);

        // No file written to disk at all — a row with nothing behind it.
        let db = open_scan_test_db(&config).await;
        db.upsert("gone.md", "some-hash", 1, &schema_hash, 0, 0)
            .await
            .unwrap();

        let dirty = scan_for_dirty(&config).await.unwrap();
        assert_eq!(dirty, vec![PathBuf::from("gone.md")]);
    }

    #[tokio::test]
    async fn scan_for_dirty_flags_stale_metadata_without_a_content_change() {
        let dir = TempDir::new().unwrap();
        let config = scan_test_config(&dir);
        let path = dir.path().join("doc.md");
        std::fs::write(&path, "# Doc").unwrap();
        let (mtime, size) = stat(&path);
        let schema_hash = expected_schema_hash(dir.path(), &config.frontmatter);

        // indexed_files says "some-hash", but there is no `documents` row at all —
        // the upgrade/backfill case, not a content change.
        let db = open_scan_test_db(&config).await;
        db.upsert("doc.md", "some-hash", 1, &schema_hash, mtime, size)
            .await
            .unwrap();

        let dirty = scan_for_dirty(&config).await.unwrap();
        assert_eq!(
            dirty,
            vec![PathBuf::from("doc.md")],
            "stat and schema are unchanged, but missing metadata must still surface"
        );
    }

    #[tokio::test]
    async fn scan_for_dirty_flags_a_schema_fingerprint_change_even_with_unchanged_stat() {
        let dir = TempDir::new().unwrap();
        let config = scan_test_config(&dir);
        let path = dir.path().join("doc.md");
        std::fs::write(&path, "# Doc").unwrap();
        let (mtime, size) = stat(&path);

        let db = open_scan_test_db(&config).await;
        // A fingerprint that will never match the real one built from `dir` — stands
        // in for "the schema changed since this was last indexed".
        db.upsert("doc.md", "some-hash", 1, "stale-fingerprint", mtime, size)
            .await
            .unwrap();
        let mut fm = HashMap::new();
        fm.insert("title".into(), serde_json::json!("Doc"));
        db.upsert_document_metadata("doc.md", &fm, mtime, "some-hash", 1)
            .await
            .unwrap();

        let dirty = scan_for_dirty(&config).await.unwrap();
        assert_eq!(dirty, vec![PathBuf::from("doc.md")]);
    }

    #[tokio::test]
    async fn scan_for_dirty_never_flags_a_file_under_a_frozen_scope() {
        let dir = TempDir::new().unwrap();
        let config = scan_test_config(&dir);
        std::fs::create_dir_all(dir.path().join("broken")).unwrap();
        // Deliberately invalid schema YAML — this scope is "frozen": the indexer
        // refuses to touch anything under it until the file is fixed.
        std::fs::write(
            dir.path().join("broken/.kb-schema.yaml"),
            "fields: [not, a, mapping]",
        )
        .unwrap();
        // Neither a new file nor a previously-indexed one under the broken scope
        // should ever be marked dirty.
        std::fs::write(dir.path().join("broken/new.md"), "# New").unwrap();

        let dirty = scan_for_dirty(&config).await.unwrap();
        assert!(
            dirty.is_empty(),
            "a frozen scope must never be marked dirty by the scan: {dirty:?}"
        );
    }

    // -- domain derivation ---------------------------------------------------

    #[test]
    fn domain_comes_from_the_top_level_folder() {
        assert_eq!(
            derive_domain("food/recipes/chili.md").as_deref(),
            Some("food")
        );
        assert_eq!(
            derive_domain("sysadmin/zfs.md").as_deref(),
            Some("sysadmin")
        );
    }

    #[test]
    fn documents_at_the_root_have_no_domain() {
        assert_eq!(derive_domain("README.md"), None);
        assert_eq!(derive_domain(""), None);
    }

    #[test]
    fn derived_domain_overrides_whatever_frontmatter_claimed() {
        // Location is the single source of truth; a stale `domain:` key must not win.
        let mut fm: HashMap<String, serde_json::Value> = HashMap::new();
        fm.insert(
            "domain".into(),
            serde_json::Value::String("lifestyle".into()),
        );
        let out = with_derived_domain(&fm, "food/recipes/chili.md");
        assert_eq!(out["domain"], serde_json::json!("food"));
    }

    #[test]
    fn root_documents_lose_a_stale_domain_key() {
        let mut fm: HashMap<String, serde_json::Value> = HashMap::new();
        fm.insert("domain".into(), serde_json::Value::String("old".into()));
        let out = with_derived_domain(&fm, "README.md");
        assert!(
            !out.contains_key("domain"),
            "a document in no folder belongs to no domain"
        );
    }

    // -- indexed field union -------------------------------------------------

    #[test]
    fn merge_indexed_fields_unions_schema_and_legacy_config() {
        use crate::qdrant::IndexKind;

        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("food/recipes")).unwrap();
        std::fs::write(
            dir.path().join("food/recipes/.kb-schema.yaml"),
            "fields:\n  planning.prep_minutes:\n    type: integer\n    indexed: true\n",
        )
        .unwrap();

        let mut config = config_no_validation();
        config.frontmatter = crate::config::FrontmatterConfig {
            indexed_fields: vec!["tags".into(), "planning.prep_minutes".into()],
            ..Default::default()
        };
        let schemas = SchemaCache::build(dir.path(), &config.frontmatter);

        let fields = merge_indexed_fields(&config, &schemas);
        let named = |n: &str| fields.iter().find(|f| f.name == n);

        assert!(named("tags").is_some(), "legacy config field survives");
        assert!(
            named("file_path").is_some(),
            "effective_indexed_fields still contributes file_path"
        );
        let nested = named("planning.prep_minutes").expect("deep-scope field must be indexed");
        assert_eq!(
            nested.kind,
            IndexKind::Integer,
            "the schema's declared type must win over the legacy keyword default"
        );
        assert_eq!(
            fields
                .iter()
                .filter(|f| f.name == "planning.prep_minutes")
                .count(),
            1,
            "a field named in both sources appears once"
        );
    }

    // -- document metadata backfill -----------------------------------------

    /// A cascade with no schema files and no config rules — the shape backfill sees
    /// when it only needs frontmatter parsed, not validated.
    /// The schema a test file validates against when the test does not care about
    /// schema rules — derived from the fixture config, matching production behavior
    /// for a deployment with no `.kb-schema.yaml`.
    fn test_schema() -> ResolvedSchema {
        ResolvedSchema::from_config(&Default::default())
    }

    fn empty_schemas() -> SchemaCache {
        SchemaCache::from_config_only(&Default::default())
    }

    async fn backfill_test_db() -> (StateDb, TempDir) {
        let dir = TempDir::new().unwrap();
        let db = StateDb::new(&dir.path().join("state.db")).await.unwrap();
        (db, dir)
    }

    #[tokio::test]
    async fn backfill_fills_metadata_for_unchanged_files() {
        // The upgrade case: indexed_files populated by a previous version, documents
        // empty. No embedder is constructed at all, which is the point — backfill must
        // never trigger an embedding call.
        let (db, db_dir) = backfill_test_db().await;
        let kb = TempDir::new().unwrap();
        let path = kb.path().join("recipe.md");
        std::fs::write(
            &path,
            "---\ntitle: Chili\ndescription: One pot\ntags: [recipe, dinner]\nplanning:\n  prep_minutes: 20\n---\n\nBody.",
        )
        .unwrap();

        db.upsert("recipe.md", "stale-hash", 3, "", 0, 0)
            .await
            .unwrap();
        assert_eq!(db.document_count().await.unwrap(), 0);

        let indexed: HashMap<String, IndexedFile> = db
            .list_all()
            .await
            .unwrap()
            .into_iter()
            .map(|f| (f.file_path.clone(), f))
            .collect();
        let queue = vec![("recipe.md".to_string(), path.clone())];

        let filled = backfill_document_metadata(&queue, &db, &indexed, &empty_schemas()).await;

        assert_eq!(filled, 1);
        assert_eq!(db.document_count().await.unwrap(), 1);

        let hashes = db.list_document_hashes().await.unwrap();
        assert!(hashes.contains_key("recipe.md"));

        // chunk_count is carried over from indexed_files rather than recomputed,
        // since backfill deliberately does not chunk.
        let (chunk_count,): (i64,) =
            sqlx::query_as("SELECT chunk_count FROM documents WHERE file_path = ?")
                .bind("recipe.md")
                .fetch_one(db.pool_for_test())
                .await
                .unwrap();
        assert_eq!(chunk_count, 3);

        drop(db_dir);
    }

    #[tokio::test]
    async fn backfill_projects_nested_frontmatter() {
        let (db, _db_dir) = backfill_test_db().await;
        let kb = TempDir::new().unwrap();
        let path = kb.path().join("recipe.md");
        std::fs::write(
            &path,
            "---\ntitle: Chili\ntags: [recipe]\nplanning:\n  prep_minutes: 20\n  tested: true\n---\n\nBody.",
        )
        .unwrap();

        let queue = vec![("recipe.md".to_string(), path)];
        backfill_document_metadata(&queue, &db, &HashMap::new(), &empty_schemas()).await;

        let rows: Vec<(String, String, Option<f64>)> = sqlx::query_as(
            "SELECT field, value_text, value_num FROM document_fields WHERE file_path = ?",
        )
        .bind("recipe.md")
        .fetch_all(db.pool_for_test())
        .await
        .unwrap();

        assert!(rows.contains(&("planning.prep_minutes".into(), "20".into(), Some(20.0))));
        assert!(rows.contains(&("planning.tested".into(), "true".into(), Some(1.0))));
    }

    #[tokio::test]
    async fn backfill_survives_unreadable_files() {
        // A missing file must not fail the whole index run — it is retried next time.
        let (db, _db_dir) = backfill_test_db().await;
        let queue = vec![("gone.md".to_string(), PathBuf::from("/nonexistent/gone.md"))];

        let filled =
            backfill_document_metadata(&queue, &db, &HashMap::new(), &empty_schemas()).await;

        assert_eq!(filled, 0);
        assert_eq!(db.document_count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn backfill_parses_frontmatter_that_would_fail_validation() {
        // Documents indexed under older rules must still get metadata, so backfill
        // uses parse_frontmatter rather than going through validation.
        let (db, _db_dir) = backfill_test_db().await;
        let kb = TempDir::new().unwrap();
        let path = kb.path().join("sparse.md");
        // No title, no description, no tags — would fail a strict required-fields check.
        std::fs::write(&path, "---\ntype: reference\n---\n\nBody.").unwrap();

        let queue = vec![("sparse.md".to_string(), path)];
        let filled =
            backfill_document_metadata(&queue, &db, &HashMap::new(), &empty_schemas()).await;

        assert_eq!(filled, 1, "metadata must not depend on passing validation");
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
            schema_hash: String::new(),
            mtime: 0,
            size: 0,
        });

        let config = config_no_validation();
        let outcome = process_file(
            &path,
            "doc.md",
            content,
            false,
            state_entry,
            &config,
            &test_schema(),
            "",
        )
        .await
        .unwrap();
        assert!(matches!(outcome, FileOutcome::Skipped { .. }));
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
            schema_hash: String::new(),
            mtime: 0,
            size: 0,
        });

        let config = config_no_validation();
        let outcome = process_file(
            &path,
            "doc.md",
            content,
            false,
            state_entry,
            &config,
            &test_schema(),
            "",
        )
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
            schema_hash: String::new(),
            mtime: 0,
            size: 0,
        });

        let config = config_no_validation();
        let outcome = process_file(
            &path,
            "doc.md",
            content,
            true,
            state_entry,
            &config,
            &test_schema(),
            "",
        )
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
        let outcome = process_file(
            &path,
            "doc.md",
            content,
            false,
            None,
            &config,
            &test_schema(),
            "",
        )
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
        let outcome = process_file(
            &path,
            "doc.md",
            content,
            false,
            None,
            &config,
            &test_schema(),
            "",
        )
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

        let outcome = process_file(
            &path,
            "doc.md",
            content,
            false,
            None,
            &config,
            &test_schema(),
            "",
        )
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

        let outcome = process_file(
            &path,
            "doc.md",
            content,
            false,
            None,
            &config,
            &ResolvedSchema::from_config(&config.frontmatter),
            "",
        )
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

        let result = process_file(
            &path,
            "doc.md",
            content,
            false,
            None,
            &config,
            &ResolvedSchema::from_config(&config.frontmatter),
            "",
        )
        .await;
        assert!(result.is_err(), "Strict mode should propagate as Err");
    }

    #[tokio::test]
    async fn unchanged_content_with_unchanged_schema_is_skipped() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("doc.md");
        let content = "---\ntitle: Test\n---\nBody.";
        std::fs::write(&path, content).unwrap();

        let state_entry = Some(IndexedFile {
            file_path: "doc.md".into(),
            content_hash: compute_hash_from_bytes(content.as_bytes()),
            chunk_count: 1,
            indexed_at: "now".into(),
            schema_hash: "abc".into(),
            mtime: 0,
            size: 0,
        });

        let outcome = process_file(
            &path,
            "doc.md",
            content,
            false,
            state_entry,
            &config_no_validation(),
            &test_schema(),
            "abc",
        )
        .await
        .unwrap();

        assert!(matches!(outcome, FileOutcome::Skipped { .. }));
    }

    #[tokio::test]
    async fn unchanged_content_with_changed_schema_is_reprocessed() {
        // The landmine this mechanism exists for: editing a .kb-schema.yaml changes no
        // document's bytes, so a content-hash-only skip would never revalidate anything.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("doc.md");
        let content = "---\ntitle: Test\n---\nBody.";
        std::fs::write(&path, content).unwrap();

        let state_entry = Some(IndexedFile {
            file_path: "doc.md".into(),
            content_hash: compute_hash_from_bytes(content.as_bytes()),
            chunk_count: 1,
            indexed_at: "now".into(),
            schema_hash: "old-fingerprint".into(),
            mtime: 0,
            size: 0,
        });

        let outcome = process_file(
            &path,
            "doc.md",
            content,
            false,
            state_entry,
            &config_no_validation(),
            &test_schema(),
            "new-fingerprint",
        )
        .await
        .unwrap();

        assert!(
            !matches!(outcome, FileOutcome::Skipped { .. }),
            "a tightened schema must force revalidation of unchanged content"
        );
    }

    #[tokio::test]
    async fn upgrade_reprocesses_every_file_so_qdrant_and_sqlite_agree_on_domain() {
        // Derived `domain` is written to the Qdrant payload only via the full
        // reprocess path, while metadata backfill deliberately skips Qdrant. The two
        // stores would diverge on upgrade if legacy rows were SKIPPED — they are not,
        // because their empty schema_hash never equals a real fingerprint. This test
        // pins that reasoning down rather than leaving it as an assumption.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("doc.md");
        let content = "---\ntitle: Test\ndomain: stale\n---\nBody.";
        std::fs::write(&path, content).unwrap();

        let legacy_row = Some(IndexedFile {
            file_path: "food/doc.md".into(),
            content_hash: compute_hash_from_bytes(content.as_bytes()),
            chunk_count: 1,
            indexed_at: "now".into(),
            schema_hash: String::new(),
            mtime: 0,
            size: 0,
        });

        let outcome = process_file(
            &path,
            "food/doc.md",
            content,
            false,
            legacy_row,
            &config_no_validation(),
            &test_schema(),
            &test_schema().fingerprint(),
        )
        .await
        .unwrap();

        assert!(
            matches!(outcome, FileOutcome::Ready(_)),
            "an unchanged legacy file must still be fully reprocessed on upgrade, which \
             is what rewrites its Qdrant payload with the derived domain"
        );
    }

    #[tokio::test]
    async fn upgraded_deployments_revalidate_once() {
        // Rows written before schema tracking existed carry an empty schema_hash, which
        // never equals a real fingerprint — so the first run after upgrading reprocesses
        // every file exactly once.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("doc.md");
        let content = "---\ntitle: Test\n---\nBody.";
        std::fs::write(&path, content).unwrap();

        let state_entry = Some(IndexedFile {
            file_path: "doc.md".into(),
            content_hash: compute_hash_from_bytes(content.as_bytes()),
            chunk_count: 1,
            indexed_at: "now".into(),
            schema_hash: String::new(),
            mtime: 0,
            size: 0,
        });

        let outcome = process_file(
            &path,
            "doc.md",
            content,
            false,
            state_entry,
            &config_no_validation(),
            &test_schema(),
            &test_schema().fingerprint(),
        )
        .await
        .unwrap();

        assert!(!matches!(outcome, FileOutcome::Skipped { .. }));
    }

    #[tokio::test]
    async fn ready_files_carry_the_schema_fingerprint_they_were_validated_against() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("doc.md");
        let content = "---\ntitle: Test\n---\nBody.";
        std::fs::write(&path, content).unwrap();

        let outcome = process_file(
            &path,
            "doc.md",
            content,
            false,
            None,
            &config_no_validation(),
            &test_schema(),
            "fingerprint-xyz",
        )
        .await
        .unwrap();

        match outcome {
            FileOutcome::Ready(pf) => assert_eq!(pf.schema_hash, "fingerprint-xyz"),
            other => panic!("expected Ready, got {}", outcome_name(&other)),
        }
    }

    /// Helper for debug output in test assertions.
    fn outcome_name(outcome: &FileOutcome) -> &'static str {
        match outcome {
            FileOutcome::Skipped { .. } => "Skipped",
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
            reconcile_interval_secs: 60,
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
            reconcile_interval_secs: 60,
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
            schema_hash: String::new(),
            file_path: file_path.to_string(),
            frontmatter: HashMap::new(),
            chunks,
            hash: "abc123".to_string(),
            old_chunk_count,
            mtime: 1_700_000_000,
            size: 123,
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
        state
            .upsert("data/orphan.md", "hash1", 3, "", 0, 0)
            .await
            .unwrap();

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
    async fn removing_an_orphan_clears_its_metadata_and_projection() {
        // The existing orphan test only covers the Qdrant-failure branch, so nothing
        // proved an orphaned document actually loses its metadata — it would otherwise
        // stay visible to list_documents with no vectors behind it.
        let dir = TempDir::new().unwrap();
        let state = test_state_db(&dir).await;

        let mut fm: HashMap<String, serde_json::Value> = HashMap::new();
        fm.insert("title".into(), serde_json::json!("Orphan"));
        fm.insert("tags".into(), serde_json::json!(["note", "stale"]));

        state.upsert("gone.md", "h", 1, "", 0, 0).await.unwrap();
        state
            .upsert_document_metadata("gone.md", &fm, 1, "h", 1)
            .await
            .unwrap();
        assert_eq!(state.document_count().await.unwrap(), 1);

        let store = MockVectorStore::all_ok();
        remove_orphans(&["gone.md".to_string()], &store, &state, "test-col")
            .await
            .unwrap();

        assert_eq!(state.count().await.unwrap(), 0, "bookkeeping row removed");
        assert_eq!(
            state.document_count().await.unwrap(),
            0,
            "document metadata removed"
        );

        let remaining: Vec<(String,)> =
            sqlx::query_as("SELECT field FROM document_fields WHERE file_path = ?")
                .bind("gone.md")
                .fetch_all(state.pool_for_test())
                .await
                .unwrap();
        assert!(
            remaining.is_empty(),
            "projection rows must cascade away with the document"
        );
    }

    #[tokio::test]
    async fn upsert_failure_preserves_state_for_retry() {
        let dir = TempDir::new().unwrap();
        let state = test_state_db(&dir).await;

        // Seed state with a previously-indexed file
        state
            .upsert("data/test.md", "old-hash", 2, "", 0, 0)
            .await
            .unwrap();

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

        state
            .upsert("data/test.md", "old-hash", 2, "", 0, 0)
            .await
            .unwrap();

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

        state
            .upsert("data/shrink.md", "old-hash", 3, "", 0, 0)
            .await
            .unwrap();

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

        state
            .upsert("data/grow.md", "old-hash", 1, "", 0, 0)
            .await
            .unwrap();

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
            reconcile_interval_secs: 60,
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
            reconcile_interval_secs: 60,
        };

        // This should complete without hanging or panicking
        let files = discover_files(dir.path(), &indexing).unwrap();
        assert!(files.iter().any(|p| p.ends_with("doc.md")));
    }
}
