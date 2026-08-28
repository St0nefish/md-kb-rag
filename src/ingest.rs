use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use globset::{Glob, GlobSet, GlobSetBuilder};
use qdrant_client::qdrant::{Condition, Filter};
use sha2::{Digest, Sha256};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::{
    chunk,
    config::{IndexingConfig, ResolvedConfig, SemanticEdgesConfig},
    embed::{EmbedClient, EmbedStore},
    qdrant::{CHUNK_TEXT_KEY, QdrantPoint, QdrantStore, SearchResult, VectorStore},
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

    let filter = WalkFilter {
        include_set: &include_set,
        exclude_set: &exclude_set,
        exclude_filenames: &exclude_filenames,
    };

    let mut matched: Vec<PathBuf> = Vec::new();

    walk_dir(data_path, data_path, Some(&filter), &mut matched)?;

    matched.sort();
    Ok(matched)
}

/// Recursively collect the absolute path of every regular file under `dir`
/// (which may be nested below `root`), at any depth, symlinks skipped — the
/// write path's unfiltered counterpart to [`discover_files`], sharing the same
/// recursive walk (symlink-skip, per-entry error handling, dir recursion) via
/// `walk_dir`'s `filter: None` mode rather than re-implementing it.
/// `write::move_directory` needs to see EVERY file under a prefix, indexable
/// or not, both to detect a non-empty destination and to enumerate exactly
/// what a source subtree contains — unlike `discover_files`, which exists to
/// answer "what should the indexer index" and must stay filtered.
///
/// `root` only affects how paths get stripped for `include_set`/`exclude_set`
/// matching, which does not apply in unfiltered mode — passing `dir` itself as
/// `root` is fine here.
pub(crate) fn walk_dir_unfiltered(root: &Path, dir: &Path) -> Result<Vec<PathBuf>> {
    let mut matched: Vec<PathBuf> = Vec::new();
    walk_dir(root, dir, None, &mut matched)?;
    Ok(matched)
}

/// The include/exclude filter `walk_dir` applies to each regular file it visits.
/// Grouped into one struct so `walk_dir` takes a single `Option<&WalkFilter>`
/// rather than three separate optional parameters that would all need to be
/// `Some`/`None` in lockstep.
struct WalkFilter<'a> {
    include_set: &'a GlobSet,
    exclude_set: &'a Option<GlobSet>,
    exclude_filenames: &'a HashSet<&'a str>,
}

/// Recursively walk `dir`, collecting every regular file's absolute path into
/// `matched`. Symlinks are always skipped (this walker underlies both the
/// indexer's file discovery and `write::move_directory`'s subtree scan, and a
/// symlink loop or a hostile symlink target is unwelcome in either).
///
/// `filter` is `None` for an unfiltered walk (every regular file matches — see
/// [`walk_dir_unfiltered`]) or `Some` to additionally require the entry match
/// `include_set` and not match `exclude_set`/`exclude_filenames`, relative to
/// `root` (see [`discover_files`]). A single implementation for both modes
/// means a future fix to symlink-loop or entry-error handling here reaches
/// both callers instead of only whichever one it was made in.
fn walk_dir(
    root: &Path,
    dir: &Path,
    filter: Option<&WalkFilter>,
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
            walk_dir(root, &path, filter, matched)?;
            continue;
        }

        if !file_type.is_file() {
            continue;
        }

        let Some(filter) = filter else {
            matched.push(path);
            continue;
        };

        // Check exclude_files by filename
        if let Some(file_name) = path.file_name().and_then(|n| n.to_str())
            && filter.exclude_filenames.contains(file_name)
        {
            debug!("Skipping excluded filename: {}", path.display());
            continue;
        }

        // Build relative path for glob matching
        let rel = path.strip_prefix(root).unwrap_or(&path);

        let rel_str = rel.to_string_lossy();

        // Must match at least one include pattern
        if !filter.include_set.is_match(rel_str.as_ref()) {
            continue;
        }

        // Must not match any exclude pattern
        if let Some(excl) = filter.exclude_set
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
    /// Document body with frontmatter stripped — kept alongside the already-chunked
    /// text so `upsert_pending` can extract outgoing markdown links (`extract_markdown_links`)
    /// from the whole document in one pass rather than reassembling it from chunks.
    body: String,
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
                    body: validated.body,
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
            body,
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
                CHUNK_TEXT_KEY.to_string(),
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

        // Refresh this file's outgoing markdown-link edges. Non-fatal: a failure here
        // leaves the previous run's edges in place (stale, not wrong-shaped) rather
        // than blocking the state/metadata bookkeeping above that already succeeded,
        // and it self-heals the next time this file is (re)indexed.
        let link_targets: Vec<(String, Option<f64>)> =
            extract_markdown_links(&pf.body, &pf.file_path)
                .into_iter()
                .map(|target| (target, None))
                .collect();
        if let Err(e) = state
            .replace_links(&pf.file_path, "markdown", &link_targets)
            .await
        {
            warn!(
                file = %pf.file_path,
                "Failed to update markdown links (non-fatal, will self-heal next run): {:#}",
                e
            );
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

// ---------------------------------------------------------------------------
// Semantic edge precompute (web UI graph view, issue #53)
// ---------------------------------------------------------------------------

/// Doc-level nearest-neighbor lookup behind the UI's precomputed semantic edges.
///
/// A trait — not a direct call to the inherent `QdrantStore::recommend_by_point_id`
/// — for the same reason `VectorStore`/`EmbedStore` above are traits:
/// `update_semantic_edges` runs against a fake in tests, with no live Qdrant service
/// required.
trait NeighborStore: Send + Sync {
    async fn recommend_by_point_id(
        &self,
        collection: &str,
        point_id: &str,
        limit: u64,
        filter: Option<Filter>,
    ) -> Result<Vec<SearchResult>>;
}

/// Thin delegation impl — calls the identically-named inherent method on
/// `QdrantStore`, same pattern as its `VectorStore`/`RetrievalStore` impls in
/// `qdrant.rs`.
impl NeighborStore for QdrantStore {
    async fn recommend_by_point_id(
        &self,
        collection: &str,
        point_id: &str,
        limit: u64,
        filter: Option<Filter>,
    ) -> Result<Vec<SearchResult>> {
        QdrantStore::recommend_by_point_id(self, collection, point_id, limit, filter).await
    }
}

/// Precompute each `pending` file's outgoing semantic (kNN) edges for the web UI
/// graph view. No-ops immediately when `cfg.enabled` is false (the default —
/// computing these costs a Qdrant `recommend` query per indexed document on every
/// run, see `config.rs`), so callers do not need to check the flag themselves.
///
/// Doc-level kNN reuses each file's deterministic first-chunk point id
/// (`make_point_id(path, 0)`, see above) as the query vector, and excludes the
/// source file's own chunks via a `file_path` filter. Because that query still
/// searches every chunk in the collection, multiple hits can come back from the same
/// neighbor document — those are deduped locally, keeping each target's best score,
/// before `min_score` and top-`k` are applied.
///
/// Only ever called for `pending` — the files actually (re)indexed this run — so
/// semantics match `upsert_pending`'s markdown-link refresh: incremental, and never
/// triggered when there is nothing to index (an empty `pending`, which also covers
/// the fully-offline case: a failed embed/upsert would already have returned via `?`
/// before `pending` ever reaches this function).
///
/// Non-fatal per file: a failed lookup or bookkeeping write leaves the previous run's
/// semantic edges in place (stale, not wrong-shaped) and repairs itself the next time
/// the file is reindexed — same policy as the markdown-link refresh.
async fn update_semantic_edges<N: NeighborStore>(
    pending: &[PendingFile],
    neighbors: &N,
    state: &StateDb,
    collection: &str,
    cfg: &SemanticEdgesConfig,
) {
    if !cfg.enabled {
        return;
    }

    for pf in pending {
        let point_id = make_point_id(&pf.file_path, 0);
        let exclude_self =
            Filter::must_not([Condition::matches("file_path", pf.file_path.clone())]);

        let hits = match neighbors
            .recommend_by_point_id(collection, &point_id, cfg.k, Some(exclude_self))
            .await
        {
            Ok(hits) => hits,
            Err(e) => {
                warn!(
                    file = %pf.file_path,
                    "Failed to compute semantic neighbors (non-fatal, will retry next run): {:#}",
                    e
                );
                continue;
            }
        };

        // Dedupe by target file, keeping the best score seen for each. Also drop the
        // source file itself defensively: the `must_not` filter above is the real
        // exclusion, this is a second, filter-independent guard so an incomplete
        // filter can never make a document recommend itself.
        let mut best: HashMap<String, f32> = HashMap::new();
        for hit in &hits {
            let Some(target) = hit.payload.get("file_path").and_then(|v| v.as_str()) else {
                continue;
            };
            if target == pf.file_path || hit.score < cfg.min_score {
                continue;
            }
            best.entry(target.to_string())
                .and_modify(|s| *s = s.max(hit.score))
                .or_insert(hit.score);
        }

        let mut ranked: Vec<(String, f32)> = best.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked.truncate(cfg.k as usize);

        let targets: Vec<(String, Option<f64>)> = ranked
            .into_iter()
            .map(|(target, score)| (target, Some(score as f64)))
            .collect();

        if let Err(e) = state
            .replace_links(&pf.file_path, "semantic", &targets)
            .await
        {
            warn!(
                file = %pf.file_path,
                "Failed to update semantic links (non-fatal, will self-heal next run): {:#}",
                e
            );
        }
    }
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

// ---------------------------------------------------------------------------
// Markdown link extraction (feeds the `document_links` graph)
// ---------------------------------------------------------------------------

/// Extract every local document-to-document link from `body` — inline
/// `[label](target.md)`, reference-style (`[label][ref]`/shortcut `[ref]` paired
/// with a `[ref]: target.md` definition), wiki-style `[[target]]`, and autolink
/// `<target.md>` — resolved to repo-relative paths anchored at `source_rel_path`'s
/// directory.
///
/// What counts as a link edge, deliberately narrow:
/// - Inline `[label](target)` — unchanged from before this function supported the
///   other three syntaxes below.
/// - Reference-style: `[label][ref]` (explicit) and the shortcut `[ref]`, resolved
///   through a `[ref]: target` DEFINITION anywhere in the document (definitions may
///   appear before OR after their use sites). A definition only counts as a link
///   when at least one use site in the document actually references it — an
///   unreferenced `[ref]: target` renders nothing in CommonMark, so it produces no
///   edge here either, which keeps extraction and rewriting in lockstep with what
///   the document actually contains. Label matching is case-insensitive and
///   whitespace-normalized (`[My Ref]` and `[my   ref]` are the same label), per
///   CommonMark. A label defined more than once resolves to its FIRST definition;
///   later duplicates are ignored, also per CommonMark.
/// - Wiki-style `[[target]]`: unlike every other syntax here, a target with no
///   `.md` extension is treated as `target.md` — deliberately more lenient, because
///   the double-bracket wiki convention (Obsidian and similar tools) is
///   conventionally extension-less, and requiring the literal `.md` suffix would
///   make this syntax useless for the KBs that actually write it that way. The
///   pipe-alias form `[[target|Display text]]` is NOT specially handled — no
///   attempt is made to parse the alias apart from the target, and any target
///   containing a literal `|` is rejected outright rather than fed through the
///   default-extension step above (which would otherwise turn `guide.md|Alias`
///   into a bogus resolved path like `guide.md|Alias.md`); write `[[target]]`
///   without an alias if you want it indexed.
/// - Autolinks `<target.md>`: the content between `<` and `>` must contain no
///   whitespace and (after fragment-stripping) end in `.md`, with no scheme
///   (`http://`, `https://`, `mailto:`, or any other `scheme://`) and no leading
///   `/`. This is intentionally conservative — `<` is common in HTML, and
///   CommonMark autolinks are normally absolute URIs — so an HTML tag with
///   attributes (`<a href="x.md">`, which contains whitespace) or an absolute URI
///   (`<https://example.com/x.md>`, which has a scheme) is excluded even though
///   part of its content ends in `.md`. A bare non-path token (`<div>`,
///   `<not-a-path>`) is excluded simply for not ending in `.md`.
/// - Images (`![alt](target)`), including the reference forms `![alt][ref]` and
///   shortcut `![alt]`, are skipped entirely for every syntax above — an image is
///   not a document reference.
/// - Anything inside a fenced code block (`` ``` `` or `~~~`, tracked the same
///   line-oriented way `chunk::split_sections` tracks fences for headings) or an
///   inline code span (`` `...` ``) is skipped for every syntax above, including
///   reference definitions: a definition line inside a fence is not indexed, and a
///   definition line wholly wrapped in an inline code span never matches the
///   definition syntax in the first place (it does not start with `[` at the
///   line's own indentation once the surrounding backticks are accounted for).
/// - A trailing `#fragment` is stripped before the target is judged, and an
///   anchor-only target (nothing left after stripping) is dropped.
/// - External targets (`http://`, `https://`, `mailto:`, any other `scheme://`,
///   protocol-relative `//...`), absolute paths (`/...`), and anything not ending
///   in `.md` (after the wiki-style default-extension step above) are dropped —
///   this graph only connects markdown documents to each other by relative path.
/// - `./` and `../` are resolved against `source_rel_path`'s directory; a target that
///   would climb above the knowledge-base root is rejected outright (dropped) rather
///   than clamped, since clamping could silently collide with an unrelated document.
/// - Results are deduped, preserving document order — for a reference-style link
///   that order is the DEFINITION's position, not any use site's, since the
///   definition is what this function (and the rewriter sharing its scan) treats as
///   the link's true location. Linking the same target twice produces one edge.
pub(crate) fn extract_markdown_links(body: &str, source_rel_path: &str) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();

    for (_, raw_target, kind) in scan_link_occurrences(body) {
        if let Some(resolved) = resolve_link_target(&raw_target, source_rel_path, kind)
            && seen.insert(resolved.clone())
        {
            out.push(resolved);
        }
    }

    out
}

/// One link occurrence found by [`scan_link_occurrences`], carrying enough position
/// information for a caller to replace exactly its target substring in the original
/// document.
///
/// For a reference-style link, this is the DEFINITION's occurrence — never a use
/// site's. See [`scan_link_occurrences`]'s doc comment for why that is the only
/// choice that lets a rewrite fix every use at once without also touching (and thus
/// double-rewriting) the use sites themselves.
///
/// Produced by [`find_markdown_link_occurrences`] — the span-carrying sibling of
/// [`extract_markdown_links`] — and consumed by `write::write_document_move`'s
/// incoming-link rewriter, which needs to find every link pointing at a moved
/// document's old path and know precisely which bytes to replace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LinkOccurrence {
    /// Byte range in the original document — valid for direct slicing of the `&str`
    /// passed to [`find_markdown_link_occurrences`] — covering exactly the raw target
    /// text (the same substring `raw` holds; for a reference-style link, the
    /// DEFINITION's target text, not any `[text][ref]`/`[ref]` use site). Byte
    /// offsets, not char offsets: a multi-byte UTF-8 character earlier in the
    /// document shifts a byte offset without shifting a char offset by the same
    /// amount.
    pub span: std::ops::Range<usize>,
    /// The target exactly as written (between `(`/`)` for inline, between the
    /// double brackets for wiki-style, between `<`/`>` for an autolink, or after
    /// `[ref]:` for a reference definition) — before the title-suffix and
    /// `#fragment` stripping [`resolve_link_target`] does.
    pub raw: String,
    /// The KB-root-relative path `raw` resolves to.
    pub resolved: String,
}

/// Span-carrying sibling of [`extract_markdown_links`]: every recognized link
/// occurrence in `body` that resolves to a KB-root-relative markdown path (same
/// judging rules — see that function's doc comment), paired with the exact byte
/// span of the text a rewrite should replace.
///
/// Shares [`scan_link_occurrences`] — the same fence/code-span/image-skipping walk,
/// for every syntax — with `extract_markdown_links`, so the two can never disagree
/// about which substrings in a document are "real" links versus prose/code that
/// merely looks like one: there is exactly one scanning implementation, and both
/// entry points call it.
///
/// Unlike `extract_markdown_links`, this is NOT deduped by resolved target — a
/// document that links to the same target twice via two DIFFERENT occurrences (e.g.
/// two inline links, or an inline link and a wiki link) yields two entries, one per
/// span, because a caller rewriting text needs to visit every occurrence, not just
/// learn that an edge exists. A reference-style link is the one exception in
/// practice: however many `[text][ref]`/`[ref]` use sites share one definition, that
/// definition still produces exactly ONE occurrence here (see
/// [`scan_link_occurrences`]'s doc comment) — rewriting it once is what fixes every
/// use at once, and the use sites themselves are never touched.
///
/// Used by `write::write_document_move`: both for its own self-reference rewrite
/// (scanning the moved document's own new content for a link to its old path) and,
/// per referencing document `StateDb::links_targeting` returns, to find exactly which
/// spans in that document's body to replace.
pub(crate) fn find_markdown_link_occurrences(
    body: &str,
    source_rel_path: &str,
) -> Vec<LinkOccurrence> {
    scan_link_occurrences(body)
        .into_iter()
        .filter_map(|(span, raw, kind)| {
            resolve_link_target(&raw, source_rel_path, kind).map(|resolved| LinkOccurrence {
                span,
                raw,
                resolved,
            })
        })
        .collect()
}

/// Whether a raw target may default to a `.md` extension when it lacks one. Only
/// wiki-style `[[target]]` gets this leniency — see [`extract_markdown_links`]'s doc
/// comment for why every other syntax requires the literal `.md` suffix as written.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RawLinkKind {
    /// Inline `(target)`, autolink `<target>`, and reference-style definition
    /// targets — must end in `.md` exactly as written.
    Explicit,
    /// Wiki-style `[[target]]` — an extension-less target is assumed to mean
    /// `target.md`.
    Wiki,
}

/// The one scanning implementation behind both [`extract_markdown_links`] and
/// [`find_markdown_link_occurrences`] — see [`extract_markdown_links`]'s doc comment
/// for exactly what each recognized syntax (inline, reference-style, wiki-style,
/// autolink) requires to match. Walks `body` line by line, tracking fenced code
/// blocks (`` ``` `` /`~~~` toggling, the same line-oriented way `chunk::split_sections`
/// tracks fences for headings).
///
/// Reference-style links need the WHOLE document before they can be resolved: a
/// `[ref]: target` definition may appear after every use site that names it, so a
/// single top-to-bottom pass cannot both scan and resolve them in the same step.
/// This function still walks the body's text exactly ONCE — collecting every
/// inline/wiki/autolink occurrence, every reference DEFINITION (with its target
/// span), and every reference-style label USED, all in one line-by-line pass via
/// [`parse_reference_definition`] and [`scan_line_constructs`] — and only resolves
/// reference-style links (matching used labels against definitions, keeping the
/// first definition per label per CommonMark) as a second, non-scanning step after
/// that walk completes.
///
/// For a resolved reference-style link, the occurrence emitted here spans the
/// DEFINITION's target text, never a use site's: rewriting the definition is what
/// fixes every use at once, and rewriting (or even just reporting) the use sites
/// too would either be redundant or, worse, corrupt text that was never the actual
/// edit target. A definition with NO use site referencing it is dropped entirely —
/// CommonMark renders nothing for an unreferenced definition, so treating it as a
/// link here would extract/rewrite something invisible to a reader.
///
/// Returns every raw target payload found, unfiltered and unresolved, alongside its
/// span and which [`RawLinkKind`] produced it (only wiki-style gets a default `.md`
/// extension) — judging a target (title/fragment stripping, external/absolute/
/// non-`.md` rejection, relative path resolution) is [`resolve_link_target`]'s job,
/// which each entry point applies itself after this shared walk. The result is
/// sorted by span start, which is what gives a reference-style occurrence its
/// position in document order (the DEFINITION's position) even though it was
/// resolved out of band from the inline/wiki/autolink occurrences collected during
/// the per-line walk.
fn scan_link_occurrences(body: &str) -> Vec<(std::ops::Range<usize>, String, RawLinkKind)> {
    let mut out: Vec<(std::ops::Range<usize>, String, RawLinkKind)> = Vec::new();
    let mut ref_defs: Vec<(String, std::ops::Range<usize>, String)> = Vec::new();
    let mut used_labels: HashSet<String> = HashSet::new();
    let mut in_fence = false;
    let mut offset = 0usize;

    for line in body.split_inclusive('\n') {
        // `split_inclusive` keeps the line terminator attached, so `line.len()` is
        // exactly how far `offset` must advance to reach the next line's start — but
        // the terminator itself (`\n`, or `\r\n`) must be stripped before scanning,
        // the same way `body.lines()` would strip it.
        let content = line.strip_suffix('\n').unwrap_or(line);
        let content = content.strip_suffix('\r').unwrap_or(content);

        let trimmed = content.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            offset += line.len();
            continue;
        }

        if !in_fence {
            if let Some((label, rel_span, raw_target)) = parse_reference_definition(content) {
                // A reference-definition line is not ALSO scanned for other
                // constructs — CommonMark treats it as its own line-level
                // construct, not prose that might additionally contain a link.
                ref_defs.push((
                    label,
                    offset + rel_span.start..offset + rel_span.end,
                    raw_target,
                ));
            } else {
                let scan = scan_line_constructs(content);
                for (rel_span, raw_target, kind) in scan.occurrences {
                    out.push((
                        offset + rel_span.start..offset + rel_span.end,
                        raw_target,
                        kind,
                    ));
                }
                used_labels.extend(scan.ref_uses);
            }
        }

        offset += line.len();
    }

    // Resolve reference-style: only a USED label's definition becomes an
    // occurrence, and only the FIRST definition for a given label counts
    // (CommonMark: a duplicate definition is shadowed by the first one seen).
    let mut first_def_by_label: HashMap<&str, &(String, std::ops::Range<usize>, String)> =
        HashMap::new();
    for def in &ref_defs {
        first_def_by_label.entry(def.0.as_str()).or_insert(def);
    }
    for label in &used_labels {
        if let Some(def) = first_def_by_label.get(label.as_str()) {
            out.push((def.1.clone(), def.2.clone(), RawLinkKind::Explicit));
        }
    }

    out.sort_by_key(|(span, _, _)| span.start);
    out
}

/// If `line` is (the entirety of) a CommonMark-style reference LINK DEFINITION —
/// `[label]: target` at up to 3 spaces of indentation, optionally followed by a
/// title on the same line, which is left untouched — return the label (normalized
/// per [`normalize_ref_label`]) and the byte span/raw text of just the `target`
/// portion within `line`. Returns `None` for anything else, INCLUDING a line
/// wrapped in an inline code span (`` `[ref]: target.md` `` does not start with `[`
/// at the line's own indentation, so it never matches — the surrounding backticks
/// are the first character) and any line inside a fenced code block (handled by the
/// caller, [`scan_link_occurrences`], which never calls this for a fenced line).
///
/// Deliberately narrow: the destination must be a bare, whitespace-free token on the
/// SAME line as the label (no `<angle-bracket>`-wrapped destination, and no
/// destination/title continued onto a following line) — CommonMark allows both, but
/// neither is needed for the relative `.md` paths this KB actually links with, and
/// supporting them would require a real multi-line parser rather than this
/// single-line one.
fn parse_reference_definition(line: &str) -> Option<(String, std::ops::Range<usize>, String)> {
    let trimmed = line.trim_start();
    let indent = line.len() - trimmed.len();
    if indent > 3 || !trimmed.starts_with('[') {
        return None;
    }

    let rest = &trimmed[1..];
    let colon_idx = rest.find("]:")?;
    let label = &rest[..colon_idx];
    if label.is_empty() {
        return None;
    }

    let after_colon = &rest[colon_idx + 2..];
    let target = after_colon.trim_start();
    let leading_ws = after_colon.len() - target.len();
    let target_len = target.find(char::is_whitespace).unwrap_or(target.len());
    if target_len == 0 {
        return None;
    }
    let raw_target = &target[..target_len];

    // Byte offset of `raw_target` within the original (untrimmed) `line`: indent
    // (leading whitespace) + `[` (1 byte) + colon_idx (label bytes) + `]:` (2
    // bytes) + leading_ws (whitespace after the colon).
    let start = indent + 1 + colon_idx + 2 + leading_ws;
    let end = start + raw_target.len();

    Some((
        normalize_ref_label(label),
        start..end,
        raw_target.to_string(),
    ))
}

/// CommonMark reference-label matching is case-insensitive and normalizes internal
/// whitespace runs to a single space — `[My Ref]` and `[my   ref]` name the same
/// definition. Used to normalize both a use site's label and a definition's label
/// before comparing them.
fn normalize_ref_label(label: &str) -> String {
    label
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Everything [`scan_line_constructs`] finds on one non-fenced, non-definition
/// line: real link occurrences (inline/wiki/autolink — reference-style definitions
/// are handled separately, by [`parse_reference_definition`], since they are a
/// whole-line construct) and every reference-style LABEL used on the line
/// (`[text][ref]`'s `ref`, or a bare `[ref]` shortcut candidate), normalized for
/// later matching against collected definitions in [`scan_link_occurrences`].
struct LineConstructs {
    occurrences: Vec<(std::ops::Range<usize>, String, RawLinkKind)>,
    ref_uses: Vec<String>,
}

/// Scan one non-fenced, non-definition line for every recognized link construct:
/// inline `[label](target)`, wiki `[[target]]`, autolink `<target>`, and
/// reference-style use sites (`[label][ref]` and the shortcut `[ref]`, both
/// recorded as candidate labels only — resolving them against known definitions is
/// [`scan_link_occurrences`]'s job, once the whole document has been walked).
///
/// Handles the same two things this function's inline-only predecessor handled —
/// image syntax (`![alt](target)`, including its `![alt][ref]`/`![alt]` reference
/// forms) and inline code spans (`` `...` ``) — plus the three added syntaxes.
///
/// Byte spans, not char spans: [`find_link_parens`]/[`find_double_bracket_close`]/
/// [`scan_autolink_candidate`] all work in `Vec<char>` indices, so
/// `line.char_indices()` maps each of those char indices to its actual byte offset —
/// a multi-byte UTF-8 character earlier on the line would otherwise desync a naive
/// char-count offset from the byte offset a caller needs for slicing.
fn scan_line_constructs(line: &str) -> LineConstructs {
    let chars: Vec<char> = line.chars().collect();
    // One byte offset per char index, plus a trailing entry for `line`'s total byte
    // length so a target ending at the line's last char can still compute its end.
    let mut byte_at: Vec<usize> = line.char_indices().map(|(b, _)| b).collect();
    byte_at.push(line.len());

    let mut occurrences = Vec::new();
    let mut ref_uses = Vec::new();
    let mut i = 0usize;
    let mut in_code = false;
    let mut code_run_len = 0usize;

    while i < chars.len() {
        // A backtick run opens or closes an inline code span. Per CommonMark, only a
        // run of the SAME length closes one already open — a lone backtick inside a
        // double-backtick span stays part of the code.
        if chars[i] == '`' {
            let start = i;
            while i < chars.len() && chars[i] == '`' {
                i += 1;
            }
            let run_len = i - start;
            if !in_code {
                in_code = true;
                code_run_len = run_len;
            } else if run_len == code_run_len {
                in_code = false;
            }
            continue;
        }

        if in_code {
            i += 1;
            continue;
        }

        // Image, plain or reference form: skip the whole construct, extracting
        // nothing — an image is never a document link, for any syntax.
        if chars[i] == '!' && chars.get(i + 1) == Some(&'[') {
            i = skip_image(&chars, i + 1);
            continue;
        }

        // Wiki-style [[target]].
        if chars[i] == '['
            && chars.get(i + 1) == Some(&'[')
            && let Some(close) = find_double_bracket_close(&chars, i)
        {
            let (target_start, target_end) = (i + 2, close);
            let raw: String = chars[target_start..target_end].iter().collect();
            occurrences.push((
                byte_at[target_start]..byte_at[target_end],
                raw,
                RawLinkKind::Wiki,
            ));
            i = close + 2;
            continue;
        }

        // Autolink <target>.
        if chars[i] == '<'
            && let Some((content_start, content_end, after, has_whitespace)) =
                scan_autolink_candidate(&chars, i)
        {
            if !has_whitespace {
                let raw: String = chars[content_start..content_end].iter().collect();
                occurrences.push((
                    byte_at[content_start]..byte_at[content_end],
                    raw,
                    RawLinkKind::Explicit,
                ));
            }
            i = after;
            continue;
        }

        if chars[i] == '[' {
            // Inline [label](target).
            if let Some((target_start, target_end, after)) = find_link_parens(&chars, i) {
                let raw: String = chars[target_start..target_end].iter().collect();
                occurrences.push((
                    byte_at[target_start]..byte_at[target_end],
                    raw,
                    RawLinkKind::Explicit,
                ));
                i = after;
                continue;
            }

            // Reference-style: [text][ref] (explicit) or [ref] (shortcut
            // candidate — real only if a matching definition exists, decided
            // later by the caller once the whole document has been walked).
            if let Some(label1_close) = find_bracket_close(&chars, i) {
                let label1: String = chars[i + 1..label1_close].iter().collect();
                if chars.get(label1_close + 1) == Some(&'[')
                    && let Some(label2_close) = find_bracket_close(&chars, label1_close + 1)
                {
                    let label2: String = chars[label1_close + 2..label2_close].iter().collect();
                    let effective = if label2.trim().is_empty() {
                        label1
                    } else {
                        label2
                    };
                    ref_uses.push(normalize_ref_label(&effective));
                    i = label2_close + 1;
                    continue;
                }
                ref_uses.push(normalize_ref_label(&label1));
                i = label1_close + 1;
                continue;
            }
        }

        i += 1;
    }

    LineConstructs {
        occurrences,
        ref_uses,
    }
}

/// Skip an image construct starting at `chars[bracket] == '['` (the caller has
/// already matched the preceding `!`), covering all three forms —
/// `![alt](target)`, `![alt][ref]`, and the bare `![alt]` — returning the char
/// index just past whichever form matched (or just past the alt label if neither
/// followed). An image's target/reference is never extracted as a document link,
/// for any syntax.
fn skip_image(chars: &[char], bracket: usize) -> usize {
    let Some(label_close) = find_bracket_close(chars, bracket) else {
        return bracket + 1;
    };
    if chars.get(label_close + 1) == Some(&'(') {
        match find_link_parens(chars, bracket) {
            Some((_, _, after)) => after,
            None => label_close + 1,
        }
    } else if chars.get(label_close + 1) == Some(&'[')
        && let Some(ref_close) = find_bracket_close(chars, label_close + 1)
    {
        ref_close + 1
    } else {
        label_close + 1
    }
}

/// Find the index of the `]` closing a label opened at `chars[open] == '['` (no
/// nested-bracket support — link/image/reference labels are plain text in
/// practice). Shared by inline links, images, and reference-style use sites, so all
/// of them treat "what is a label" the same way.
fn find_bracket_close(chars: &[char], open: usize) -> Option<usize> {
    let mut j = open + 1;
    while j < chars.len() && chars[j] != ']' {
        j += 1;
    }
    (j < chars.len()).then_some(j)
}

/// Find the index of the first `]` of a closing `]]` for a wiki link opened at
/// `chars[open] == chars[open + 1] == '['`. No nested `[[`/`]]` support, same
/// simplification as [`find_bracket_close`].
fn find_double_bracket_close(chars: &[char], open: usize) -> Option<usize> {
    let mut j = open + 2;
    while j + 1 < chars.len() {
        if chars[j] == ']' && chars[j + 1] == ']' {
            return Some(j);
        }
        j += 1;
    }
    None
}

/// From `chars[open] == '<'`, look for a `>` on the SAME line with nothing that
/// looks like nested markup (a second `<` before the close), returning
/// `(content_start, content_end, index_after_closing_gt, contains_whitespace)`.
/// `contains_whitespace` lets the caller reject the candidate outright: a real
/// document-link autolink never contains whitespace, while most HTML tags with
/// attributes do (`<a href="x.md">`) — see [`extract_markdown_links`]'s doc comment
/// for the full autolink acceptance policy. Returns `None` when nothing on this
/// line closes the `<` at all (leave it as an ordinary character — most commonly
/// the start of literal HTML that spans past this line).
fn scan_autolink_candidate(chars: &[char], open: usize) -> Option<(usize, usize, usize, bool)> {
    let mut j = open + 1;
    let mut has_whitespace = false;
    while j < chars.len() && chars[j] != '>' && chars[j] != '<' {
        if chars[j].is_whitespace() {
            has_whitespace = true;
        }
        j += 1;
    }
    if j >= chars.len() || chars[j] != '>' {
        return None;
    }
    Some((open + 1, j, j + 1, has_whitespace))
}

/// Given `chars[bracket] == '['`, look for the `]` closing the bracketed label and,
/// if a `(` immediately follows it, the parenthesized target after that (parens
/// balanced, so a target containing a literal `(`/`)` still resolves correctly).
///
/// Returns `(target_start, target_end, index_after_closing_paren)` as char indices,
/// with `target_end` exclusive — or `None` if `bracket` is not actually the start of
/// an inline link (a bare `[`, or reference-style `[label][ref]`, both of which look
/// identical to this point).
fn find_link_parens(chars: &[char], bracket: usize) -> Option<(usize, usize, usize)> {
    let j = find_bracket_close(chars, bracket)?;
    if chars.get(j + 1) != Some(&'(') {
        return None; // Not immediately followed by a target — not this parser's job.
    }

    let target_start = j + 2;
    let mut depth = 1i32;
    let mut k = target_start;
    while k < chars.len() {
        match chars[k] {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some((target_start, k, k + 1));
                }
            }
            _ => {}
        }
        k += 1;
    }
    None // Unterminated target.
}

/// Judge and resolve one raw link target against `source_rel_path`'s directory.
/// Returns `None` for anything [`extract_markdown_links`]'s doc comment says to
/// drop. `kind` controls the one syntax-specific rule: whether an extension-less
/// target defaults to `.md` (wiki-style only — see [`extract_markdown_links`]'s doc
/// comment for why).
fn resolve_link_target(
    raw_target: &str,
    source_rel_path: &str,
    kind: RawLinkKind,
) -> Option<String> {
    let target = raw_target.trim();

    // Strip an optional ` "title"` / ` 'title'` suffix (CommonMark link title) — take
    // everything before the first whitespace as the actual target.
    let target = match target.find(char::is_whitespace) {
        Some(idx) => &target[..idx],
        None => target,
    };

    // Strip a trailing #fragment.
    let target = match target.find('#') {
        Some(idx) => &target[..idx],
        None => target,
    };

    if target.is_empty() {
        return None;
    }

    let lower = target.to_ascii_lowercase();
    if lower.starts_with("http://")
        || lower.starts_with("https://")
        || lower.starts_with("mailto:")
        || target.starts_with('/')
        || target.contains("://")
    {
        return None;
    }

    // A wiki-style pipe-alias target (`[[target|Display text]]`) is not parsed
    // apart from its target — see `extract_markdown_links`'s doc comment — so
    // reject it outright rather than let the default-extension step below turn
    // `guide.md|Alias` into a bogus resolved path like `guide.md|Alias.md`.
    if kind == RawLinkKind::Wiki && target.contains('|') {
        return None;
    }

    let with_ext: String;
    let target = if target.ends_with(".md") {
        target
    } else if kind == RawLinkKind::Wiki {
        with_ext = format!("{target}.md");
        with_ext.as_str()
    } else {
        return None;
    };

    resolve_relative_md_path(target, source_rel_path)
}

/// Join `target` (a `.md`-relative path, possibly containing `./`/`../`) onto
/// `source_rel_path`'s directory and normalize the result component-by-component —
/// no filesystem access, since neither path need exist on disk in the same shape the
/// index sees it. Returns `None` if a `..` climbs above the knowledge-base root.
fn resolve_relative_md_path(target: &str, source_rel_path: &str) -> Option<String> {
    let mut stack: Vec<&str> = source_rel_path
        .rsplit_once('/')
        .map(|(dir, _)| dir.split('/').filter(|c| !c.is_empty()).collect())
        .unwrap_or_default();

    for comp in target.split('/') {
        match comp {
            "" | "." => {}
            ".." => {
                // `?` on `Option<&str>` propagates `None` — i.e. rejects the whole
                // target — the moment a `..` would climb above the knowledge-base root.
                stack.pop()?;
            }
            other => stack.push(other),
        }
    }

    if stack.is_empty() {
        None
    } else {
        Some(stack.join("/"))
    }
}

/// Inverse of [`resolve_relative_md_path`]: given a referencing document's
/// KB-root-relative path and a KB-root-relative target path, produce the relative link
/// text that document should contain to point at that target — e.g. from `dev/a.md` to
/// `sysadmin/b.md`, `../sysadmin/b.md`; from `dev/a.md` to `dev/b.md`, `b.md`.
///
/// Round-trip property: for any `source_rel_path`/`target_rel_path` pair, feeding this
/// function's output back through `resolve_relative_md_path(_, source_rel_path)` must
/// return `Some(target_rel_path.to_string())` — see the `relativize_md_path_round_trips`
/// test.
///
/// No filesystem access, same as `resolve_relative_md_path` — pure path algebra over
/// KB-root-relative strings. Finds the longest shared directory prefix between
/// `source_rel_path`'s directory and `target_rel_path`'s directory, climbs out of
/// `source_rel_path`'s directory with `../` past that shared prefix, then descends back
/// down to `target_rel_path`.
///
/// Used by `write::write_document_move` to compute the replacement text for every
/// link it rewrites — both a referencing document's link to the move's new
/// destination, and the moved document's own self-reference relative to its new
/// directory.
pub(crate) fn relativize_md_path(source_rel_path: &str, target_rel_path: &str) -> String {
    let source_dir: Vec<&str> = source_rel_path
        .rsplit_once('/')
        .map(|(dir, _)| dir.split('/').filter(|c| !c.is_empty()).collect())
        .unwrap_or_default();
    let target_components: Vec<&str> = target_rel_path
        .split('/')
        .filter(|c| !c.is_empty())
        .collect();
    let target_dir = &target_components[..target_components.len().saturating_sub(1)];

    let common = source_dir
        .iter()
        .zip(target_dir.iter())
        .take_while(|(a, b)| a == b)
        .count();

    let climbs = source_dir.len() - common;
    let mut parts: Vec<&str> = std::iter::repeat_n("..", climbs).collect();
    parts.extend(target_components[common..].iter().copied());

    parts.join("/")
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
        let (frontmatter, body) = validate::parse_frontmatter(&content, schema);
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

        // Refresh this file's outgoing markdown-link edges too — the incremental path
        // (`upsert_pending`) does this per changed file, but a file the backfill visits
        // is by definition unchanged, so without this an existing deployment's edge
        // graph stays incomplete until content churns or an operator runs `index
        // --full`. Same non-fatal, self-healing policy as `upsert_pending`.
        let link_targets: Vec<(String, Option<f64>)> = extract_markdown_links(&body, rel_key)
            .into_iter()
            .map(|target| (target, None))
            .collect();
        if let Err(e) = state
            .replace_links(rel_key, "markdown", &link_targets)
            .await
        {
            warn!(
                file = %rel_key,
                "Metadata backfill: failed to update markdown links (non-fatal, will self-heal next run): {:#}",
                e
            );
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
            &crate::git::lock_git().await,
            git_url,
            &config.source.branch,
            config.data_path(),
            token.as_deref(),
        )
        .await
        .context("Failed to ensure git repository")?;
    }

    // Constructed here — the only place this whole call path ever builds a *live*
    // Qdrant/embedding client — and handed to `index_paths_generic` behind the
    // `VectorStore`/`EmbedStore` traits it already accepts for
    // `upsert_pending`/`remove_orphans`. Both constructions are pure local setup (no
    // I/O), so moving `EmbedClient::new` here from its old spot later in the function
    // changes nothing observable. Everything past this point runs generically over
    // those traits, so it can be driven by fakes in tests with no live service — see
    // `index_paths_generic`'s doc comment.
    let store = QdrantStore::new(&config.qdrant).context("Failed to connect to Qdrant")?;
    let embedder = EmbedClient::new(&config.embedding);

    index_paths_generic(config, paths, force, run_start, &embedder, &store).await
}

/// The indexing pipeline body, generic over the embedding/vector-store dependencies
/// so it can be exercised with fakes — no live Qdrant or embedding service required —
/// while still being the exact code production runs. [`index_paths_inner`] is the
/// only production caller: it constructs the real `EmbedClient`/`QdrantStore` and
/// passes them straight in, so this split changes nothing about what `serve`'s
/// worker or the `index` CLI actually do — it only adds a second entry point for
/// tests to call with fakes instead of live services.
///
/// `run_start` is threaded in as a parameter rather than started internally so the
/// elapsed time in the closing summary log still covers `index_paths_inner`'s
/// git-sync step, exactly as it did before this function existed.
async fn index_paths_generic<E: EmbedStore, Q: VectorStore + NeighborStore>(
    config: &ResolvedConfig,
    paths: &[PathBuf],
    force: bool,
    run_start: std::time::Instant,
    embedder: &E,
    store: &Q,
) -> Result<()> {
    // ── Infrastructure ──────────────────────────────────────────────────────
    let db_path = config.state_db_path();
    let state = StateDb::new(Path::new(&db_path))
        .await
        .context("Failed to open state DB")?;

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
    let indexed_fields = crate::qdrant::all_indexed_fields(config, &schemas);

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
        .ensure_collection(
            collection,
            vector_size,
            &indexed_fields,
            config.search.phrase,
        )
        .await
        .context("Failed to ensure Qdrant collection")?;

    // Clear state only after the collection exists; a clear failure then leaves
    // an empty collection + populated state, which recovers on the next run.
    if force {
        state.clear().await.context("Failed to clear state DB")?;
    }

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
        upsert_pending(&pending, embedder, store, &state, collection).await?;

        // Precompute semantic (kNN) edges for the web UI graph view. No-ops when
        // `ui.semantic_edges` is disabled (the default) — see `update_semantic_edges`.
        update_semantic_edges(
            &pending,
            store,
            &state,
            collection,
            &config.ui.semantic_edges,
        )
        .await;
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
        remove_orphans(&missing, store, &state, collection).await?;
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
            ui: Default::default(),
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

    // -- index_paths_generic ---------------------------------------------------
    //
    // `index_paths_inner` (the production caller) constructs a real `QdrantStore`
    // and `EmbedClient` and hands them straight to `index_paths_generic` — this is
    // the injection point issue #84 asked for. These tests call the generic
    // function directly with fakes, exercising the real per-path routing logic with
    // no live Qdrant or embedding service.
    //
    // `expected_schema_hash` and `open_scan_test_db`, defined below for
    // `scan_for_dirty`'s tests, are reused here — same config, same on-disk state
    // DB, same "what schema hash would the pipeline itself compute" helper.

    /// A changed (new) file, an unchanged file, and a missing (orphaned) file, in one
    /// run — the exact three-way split `process_file`'s `FileOutcome` and this
    /// function's per-path loop exist to make. If a path were ever misrouted — an
    /// unchanged file re-embedded, a changed file skipped, or a missing file left
    /// untouched instead of purged — this catches it directly: the fake
    /// embedder/store only ever see the one file that should actually be
    /// (re)embedded, and the state DB ends up with exactly the expected
    /// insert/retain/delete outcome for each of the three.
    #[tokio::test]
    async fn index_paths_generic_routes_changed_unchanged_and_missing_correctly() {
        let dir = TempDir::new().unwrap();
        let mut config = config_no_validation();
        config.source.data_path = Some(dir.path().to_string_lossy().into_owned());

        std::fs::write(dir.path().join("changed.md"), "# Changed\n\nNew content.").unwrap();
        let unchanged_content = "# Unchanged\n\nSame as last run.";
        std::fs::write(dir.path().join("unchanged.md"), unchanged_content).unwrap();
        // "missing.md" deliberately has no file on disk — it exists only as a state
        // row, simulating a file deleted since the last run.

        let schema_hash = expected_schema_hash(dir.path(), &config.frontmatter);
        let unchanged_hash = compute_hash_from_bytes(unchanged_content.as_bytes());

        let state = open_scan_test_db(&config).await;
        let mut fm = HashMap::new();
        fm.insert("title".into(), serde_json::json!("Unchanged"));
        state
            .upsert("unchanged.md", &unchanged_hash, 1, &schema_hash, 0, 0)
            .await
            .unwrap();
        state
            .upsert_document_metadata("unchanged.md", &fm, 0, &unchanged_hash, 1)
            .await
            .unwrap();
        state
            .upsert("missing.md", "stale-hash", 1, &schema_hash, 0, 0)
            .await
            .unwrap();
        state
            .upsert_document_metadata("missing.md", &HashMap::new(), 0, "stale-hash", 1)
            .await
            .unwrap();
        assert_eq!(state.document_count().await.unwrap(), 2);

        let paths = vec![
            PathBuf::from("changed.md"),
            PathBuf::from("unchanged.md"),
            PathBuf::from("missing.md"),
        ];
        // Exactly 1 embedding: "changed.md" is the only file expected to produce a
        // chunk to embed. If routing ever sent a second file's text through, the
        // embedding-count mismatch check in `upsert_pending` would itself fail the
        // run — a second, independent signal on top of the assertions below.
        let embedder = MockEmbedClient::ok(vec![vec![1.0, 2.0, 3.0]]);
        let store = TrackingMockVectorStore::all_ok();

        let result = index_paths_generic(
            &config,
            &paths,
            false,
            std::time::Instant::now(),
            &embedder,
            &store,
        )
        .await;
        assert!(result.is_ok(), "run failed: {:?}", result.err());

        // Only the changed file was embedded and upserted. Scoped so the
        // `MutexGuard` is dropped before the `.await`s below.
        {
            let points = store.upserted_points.lock().unwrap();
            assert_eq!(
                points.len(),
                1,
                "only the changed file should produce points"
            );
            assert_eq!(
                points[0].payload.get("file_path").and_then(|v| v.as_str()),
                Some("changed.md"),
                "the unchanged file must not have been re-embedded"
            );
        }

        // Only the missing file was purged.
        let deletes = store.delete_by_files_calls.lock().unwrap().clone();
        assert_eq!(deletes.len(), 1, "exactly one orphan-removal batch");
        assert_eq!(deletes[0], vec!["missing.md".to_string()]);

        // State DB reflects the outcome: changed.md now tracked, unchanged.md
        // untouched, missing.md gone.
        assert!(
            state.get("changed.md").await.unwrap().is_some(),
            "changed.md must now be tracked"
        );
        let unchanged_entry = state.get("unchanged.md").await.unwrap().unwrap();
        assert_eq!(
            unchanged_entry.content_hash, unchanged_hash,
            "unchanged.md's row must be untouched"
        );
        assert!(
            state.get("missing.md").await.unwrap().is_none(),
            "missing.md's row must be purged"
        );
        assert_eq!(
            state.document_count().await.unwrap(),
            2,
            "changed.md + unchanged.md remain; missing.md's metadata is gone"
        );
    }

    /// The one link-extraction test at the `index_paths_generic` level: proves
    /// `upsert_pending` actually calls `state.replace_links` with the outgoing
    /// markdown edges extracted from a real indexed file's body — not just that the
    /// pure parser produces the right list in isolation (see
    /// `extract_markdown_links_cases` below for that).
    #[tokio::test]
    async fn index_paths_generic_wires_markdown_links_into_state() {
        let dir = TempDir::new().unwrap();
        let mut config = config_no_validation();
        config.source.data_path = Some(dir.path().to_string_lossy().into_owned());

        std::fs::create_dir_all(dir.path().join("docs")).unwrap();
        std::fs::write(
            dir.path().join("docs/a.md"),
            "# A\n\nSee [B](./b.md) and [top](../top.md).\n",
        )
        .unwrap();

        let state = open_scan_test_db(&config).await;
        let paths = vec![PathBuf::from("docs/a.md")];
        let embedder = MockEmbedClient::ok(vec![vec![1.0, 2.0, 3.0]]);
        let store = TrackingMockVectorStore::all_ok();

        let result = index_paths_generic(
            &config,
            &paths,
            false,
            std::time::Instant::now(),
            &embedder,
            &store,
        )
        .await;
        assert!(result.is_ok(), "run failed: {:?}", result.err());

        let mut links = state.all_links().await.unwrap();
        links.sort_by(|a, b| a.1.cmp(&b.1));
        assert_eq!(
            links,
            vec![
                (
                    "docs/a.md".to_string(),
                    "docs/b.md".to_string(),
                    "markdown".to_string(),
                    None
                ),
                (
                    "docs/a.md".to_string(),
                    "top.md".to_string(),
                    "markdown".to_string(),
                    None
                ),
            ]
        );
    }

    /// Re-indexing a file whose links changed must REPLACE the prior edge set, not
    /// accumulate alongside it — otherwise a removed link would leave a stale edge in
    /// the graph forever.
    #[tokio::test]
    async fn index_paths_generic_replaces_markdown_links_on_reindex() {
        let dir = TempDir::new().unwrap();
        let mut config = config_no_validation();
        config.source.data_path = Some(dir.path().to_string_lossy().into_owned());
        let path = dir.path().join("a.md");

        std::fs::write(&path, "[Old](old.md)").unwrap();
        let state = open_scan_test_db(&config).await;
        let paths = vec![PathBuf::from("a.md")];

        index_paths_generic(
            &config,
            &paths,
            false,
            std::time::Instant::now(),
            &MockEmbedClient::ok(vec![vec![1.0]]),
            &TrackingMockVectorStore::all_ok(),
        )
        .await
        .unwrap();
        assert_eq!(
            state.all_links().await.unwrap(),
            vec![(
                "a.md".to_string(),
                "old.md".to_string(),
                "markdown".to_string(),
                None
            )]
        );

        // Change the file's content (and therefore its hash) so the second run
        // actually reprocesses it instead of hitting the unchanged-skip path.
        std::fs::write(&path, "[New](new.md)").unwrap();
        index_paths_generic(
            &config,
            &paths,
            false,
            std::time::Instant::now(),
            &MockEmbedClient::ok(vec![vec![1.0]]),
            &TrackingMockVectorStore::all_ok(),
        )
        .await
        .unwrap();

        assert_eq!(
            state.all_links().await.unwrap(),
            vec![(
                "a.md".to_string(),
                "new.md".to_string(),
                "markdown".to_string(),
                None
            )],
            "the old.md edge must be gone, not accumulated alongside new.md"
        );
    }

    // -- extract_markdown_links -------------------------------------------------
    //
    // Table-driven: each case is (name, body, source_rel_path, expected resolved
    // targets in order). Adding a new case is a one-line addition to the table, not a
    // new test function.
    #[test]
    fn extract_markdown_links_cases() {
        let cases: &[(&str, &str, &str, &[&str])] = &[
            (
                "plain sibling link",
                "See [Guide](guide.md) for details.",
                "docs/page.md",
                &["docs/guide.md"],
            ),
            (
                "./ resolves relative to the source file's own directory",
                "[Guide](./guide.md)",
                "docs/page.md",
                &["docs/guide.md"],
            ),
            (
                "../ climbs to the parent directory",
                "[Top](../top.md)",
                "docs/sub/page.md",
                &["docs/top.md"],
            ),
            (
                "a trailing #fragment is stripped before judging the target",
                "[Section](guide.md#installation)",
                "docs/page.md",
                &["docs/guide.md"],
            ),
            (
                "an anchor-only target has nothing left to link and is dropped",
                "[Here](#top)",
                "docs/page.md",
                &[],
            ),
            (
                "images are skipped, not treated as document links",
                "![diagram](diagram.md)",
                "docs/page.md",
                &[],
            ),
            (
                "external links are skipped (http, mailto, protocol-relative)",
                "[a](https://x.com/x.md) [b](mailto:a@b.com) [c](//host/x.md)",
                "docs/page.md",
                &[],
            ),
            (
                "non-.md targets are skipped",
                "[Image](diagram.png) and [Site](index.html)",
                "docs/page.md",
                &[],
            ),
            (
                "an absolute path is skipped",
                "[Abs](/etc/passwd.md)",
                "docs/page.md",
                &[],
            ),
            (
                "links inside a ``` fenced code block are not real links",
                "```md\n[Guide](guide.md)\n```",
                "docs/page.md",
                &[],
            ),
            (
                "links inside a ~~~ fenced code block are not real links",
                "~~~md\n[Guide](guide.md)\n~~~",
                "docs/page.md",
                &[],
            ),
            (
                "links inside an inline code span are not real links",
                "Use `[Guide](guide.md)` literally.",
                "docs/page.md",
                &[],
            ),
            (
                "reference-style [label][ref] resolves through a [ref]: definition that \
                 appears AFTER its use site",
                "[Guide][ref]\n\n[ref]: guide.md",
                "docs/page.md",
                &["docs/guide.md"],
            ),
            (
                "reference-style resolves through a definition that appears BEFORE its \
                 use site too",
                "[ref]: guide.md\n\n[Guide][ref]",
                "docs/page.md",
                &["docs/guide.md"],
            ),
            (
                "reference-style shortcut form [ref] (no separate label text) resolves",
                "See [ref] for details.\n\n[ref]: guide.md",
                "docs/page.md",
                &["docs/guide.md"],
            ),
            (
                "reference-style label matching is case-insensitive",
                "[Guide][REF]\n\n[ref]: guide.md",
                "docs/page.md",
                &["docs/guide.md"],
            ),
            (
                "a reference definition with no use site anywhere in the document is not a \
                 link — CommonMark renders nothing for it either",
                "[ref]: guide.md",
                "docs/page.md",
                &[],
            ),
            (
                "multiple uses of one reference definition still produce one edge",
                "[A][ref] and [B][ref] and [ref]\n\n[ref]: guide.md",
                "docs/page.md",
                &["docs/guide.md"],
            ),
            (
                "a duplicate reference definition is shadowed by the FIRST one, per \
                 CommonMark",
                "[Guide][ref]\n\n[ref]: guide.md\n[ref]: other.md",
                "docs/page.md",
                &["docs/guide.md"],
            ),
            (
                "reference-style definitions inside a fenced code block are not tracked, \
                 and the use site outside the fence resolves nothing without them",
                "[Guide][ref]\n\n```md\n[ref]: guide.md\n```\n",
                "docs/page.md",
                &[],
            ),
            (
                "a reference definition wholly wrapped in an inline code span is not a \
                 real definition",
                "[Guide][ref]\n\n`[ref]: guide.md`\n",
                "docs/page.md",
                &[],
            ),
            (
                "wiki-style [[target]] resolves like inline, sibling directory",
                "[[guide]]",
                "docs/page.md",
                &["docs/guide.md"],
            ),
            (
                "wiki-style [[target.md]] with an explicit extension is not double-appended",
                "[[guide.md]]",
                "docs/page.md",
                &["docs/guide.md"],
            ),
            (
                "wiki-style resolves ./ and ../ the same way inline targets do",
                "[[../top]]",
                "docs/sub/page.md",
                &["docs/top.md"],
            ),
            (
                "wiki-style pipe-alias form is not specially handled and is dropped",
                "[[guide.md|Display Text]]",
                "docs/page.md",
                &[],
            ),
            (
                "wiki-style links inside a fenced code block are not real links",
                "```md\n[[guide]]\n```",
                "docs/page.md",
                &[],
            ),
            (
                "wiki-style links inside an inline code span are not real links",
                "Use `[[guide]]` literally.",
                "docs/page.md",
                &[],
            ),
            (
                "autolink <target.md> resolves like inline",
                "See <guide.md> for details.",
                "docs/page.md",
                &["docs/guide.md"],
            ),
            (
                "autolink with a trailing #fragment resolves like inline",
                "<guide.md#installation>",
                "docs/page.md",
                &["docs/guide.md"],
            ),
            (
                "an HTML tag with attributes (whitespace) is not an autolink",
                "<div>",
                "docs/page.md",
                &[],
            ),
            (
                "an absolute-URI autolink is not a document link, even ending in .md",
                "<https://example.com/x.md>",
                "docs/page.md",
                &[],
            ),
            (
                "a bare non-path autolink is not a document link",
                "<not-a-path>",
                "docs/page.md",
                &[],
            ),
            (
                "autolinks inside a fenced code block are not real links",
                "```md\n<guide.md>\n```",
                "docs/page.md",
                &[],
            ),
            (
                "autolinks inside an inline code span are not real links",
                "Use `<guide.md>` literally.",
                "docs/page.md",
                &[],
            ),
            (
                "duplicate links to the same resolved target are deduped, first-seen order",
                "[A](guide.md) [B](./guide.md) [C](other.md)",
                "docs/page.md",
                &["docs/guide.md", "docs/other.md"],
            ),
            (
                "an escape attempt above the knowledge-base root is rejected",
                "[Escape](../../secret.md)",
                "docs/page.md",
                &[],
            ),
            (
                "a root-level source file resolves against an empty directory",
                "[Guide](guide.md)",
                "page.md",
                &["guide.md"],
            ),
        ];

        for (name, body, source, expected) in cases {
            let got = extract_markdown_links(body, source);
            let expected: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
            assert_eq!(got, expected, "case failed: {name}");
        }
    }

    // -- find_markdown_link_occurrences ------------------------------------------
    //
    // Span-carrying sibling of `extract_markdown_links`. These tests focus on what
    // `extract_markdown_links_cases` above cannot check: exact byte spans, and that
    // fenced/code/image skipping (shared via `scan_link_occurrences`) still holds when
    // spans are in play.

    #[test]
    fn find_markdown_link_occurrences_spans_match_only_real_links() {
        let body = "See [Guide](guide.md) for details.\n\n\
                     ```md\n[Fenced](fenced.md)\n```\n\n\
                     Use `[Code](code.md)` literally.\n\n\
                     ![Diagram](diagram.md)\n\n\
                     Also [Other](other.md).\n";
        let source = "docs/page.md";

        let occurrences = find_markdown_link_occurrences(body, source);

        let raws: Vec<&str> = occurrences.iter().map(|o| o.raw.as_str()).collect();
        assert_eq!(
            raws,
            vec!["guide.md", "other.md"],
            "only the two real inline links outside the fence/code-span/image should be reported"
        );

        let resolved: Vec<&str> = occurrences.iter().map(|o| o.resolved.as_str()).collect();
        assert_eq!(resolved, vec!["docs/guide.md", "docs/other.md"]);

        for occurrence in &occurrences {
            assert_eq!(
                &body[occurrence.span.clone()],
                occurrence.raw,
                "span must slice out exactly the raw target text for raw={:?}",
                occurrence.raw
            );
        }
    }

    #[test]
    fn find_markdown_link_occurrences_multibyte_utf8_span() {
        // Multi-byte UTF-8 text (each 'é'/'π'/'—' is more than one byte) sits before the
        // link on the same line. An implementation that used char offsets as byte
        // offsets would compute a span that undershoots the real byte position of
        // "guide.md" — either panicking on a non-UTF8 char-boundary slice or slicing out
        // the wrong bytes.
        let body = "Café résumé — π ≈ 3.14: see [Guide](guide.md) for the recipe.";
        let source = "docs/page.md";

        let occurrences = find_markdown_link_occurrences(body, source);
        assert_eq!(occurrences.len(), 1);

        let occurrence = &occurrences[0];
        assert_eq!(occurrence.raw, "guide.md");
        assert_eq!(occurrence.resolved, "docs/guide.md");

        let expected_start = body
            .find("guide.md")
            .expect("target text must appear in body");
        assert_eq!(occurrence.span.start, expected_start);
        assert_eq!(occurrence.span.end, expected_start + "guide.md".len());
        assert_eq!(&body[occurrence.span.clone()], "guide.md");
    }

    #[test]
    fn find_markdown_link_occurrences_reference_style_span_is_the_definition_not_use_sites() {
        // Three use sites (explicit label, another explicit label, and the bare
        // shortcut) all sharing one definition must yield exactly ONE occurrence,
        // spanning the DEFINITION's target text — never any use site's — per
        // `scan_link_occurrences`'s documented rewrite-target decision: rewriting
        // the definition once is what fixes every use at once.
        let body = "[A][ref] and [B][ref] and [ref] too.\n\n[ref]: guide.md \"Title\"\n";
        let source = "docs/page.md";

        let occurrences = find_markdown_link_occurrences(body, source);
        assert_eq!(
            occurrences.len(),
            1,
            "three use sites sharing one definition must collapse to one occurrence, got: \
             {occurrences:?}"
        );

        let occurrence = &occurrences[0];
        assert_eq!(occurrence.raw, "guide.md");
        assert_eq!(occurrence.resolved, "docs/guide.md");

        let expected_start = body
            .find("guide.md")
            .expect("the definition's target text must appear in body");
        assert_eq!(
            occurrence.span,
            expected_start..expected_start + "guide.md".len(),
            "the span must cover the definition's target text, not a use site"
        );
        assert_eq!(&body[occurrence.span.clone()], "guide.md");
    }

    #[test]
    fn find_markdown_link_occurrences_wiki_multibyte_utf8_span() {
        let body = "Café résumé — π ≈ 3.14: see [[guide]] for the recipe.";
        let source = "docs/page.md";

        let occurrences = find_markdown_link_occurrences(body, source);
        assert_eq!(occurrences.len(), 1);

        let occurrence = &occurrences[0];
        assert_eq!(occurrence.raw, "guide");
        assert_eq!(occurrence.resolved, "docs/guide.md");

        let expected_start = body
            .find("guide]]")
            .expect("target text must appear in body");
        assert_eq!(occurrence.span.start, expected_start);
        assert_eq!(occurrence.span.end, expected_start + "guide".len());
        assert_eq!(&body[occurrence.span.clone()], "guide");
    }

    #[test]
    fn find_markdown_link_occurrences_autolink_multibyte_utf8_span() {
        let body = "Café résumé — π ≈ 3.14: see <guide.md> for the recipe.";
        let source = "docs/page.md";

        let occurrences = find_markdown_link_occurrences(body, source);
        assert_eq!(occurrences.len(), 1);

        let occurrence = &occurrences[0];
        assert_eq!(occurrence.raw, "guide.md");
        assert_eq!(occurrence.resolved, "docs/guide.md");

        let expected_start = body
            .find("guide.md>")
            .expect("target text must appear in body");
        assert_eq!(occurrence.span.start, expected_start);
        assert_eq!(occurrence.span.end, expected_start + "guide.md".len());
        assert_eq!(&body[occurrence.span.clone()], "guide.md");
    }

    #[test]
    fn find_markdown_link_occurrences_reference_definition_multibyte_utf8_span() {
        // Multi-byte UTF-8 text before the use site AND before the definition line
        // itself (in the label), proving the definition-line parser
        // (`parse_reference_definition`) computes byte-correct spans too, not just
        // the char-array-based general scanner.
        let body = "Café résumé — π ≈ 3.14: see [Guide][réf] too.\n\n[réf]: guide.md\n";
        let source = "docs/page.md";

        let occurrences = find_markdown_link_occurrences(body, source);
        assert_eq!(occurrences.len(), 1, "got: {occurrences:?}");

        let occurrence = &occurrences[0];
        assert_eq!(occurrence.raw, "guide.md");
        assert_eq!(occurrence.resolved, "docs/guide.md");

        let expected_start = body
            .rfind("guide.md")
            .expect("the definition's target text must appear in body");
        assert_eq!(occurrence.span.start, expected_start);
        assert_eq!(occurrence.span.end, expected_start + "guide.md".len());
        assert_eq!(&body[occurrence.span.clone()], "guide.md");
    }

    #[test]
    fn find_markdown_link_occurrences_wiki_and_autolink_skip_fence_and_code_span() {
        let body = "[[real]]\n\n\
                     ```md\n[[fenced]]\n<fenced.md>\n```\n\n\
                     Use `[[coded]]` and `<coded.md>` literally.\n\n\
                     Also <real2.md>.\n";
        let source = "docs/page.md";

        let occurrences = find_markdown_link_occurrences(body, source);
        let raws: Vec<&str> = occurrences.iter().map(|o| o.raw.as_str()).collect();
        assert_eq!(
            raws,
            vec!["real", "real2.md"],
            "only the two real constructs outside the fence/code-span should be reported"
        );
        for occurrence in &occurrences {
            assert_eq!(&body[occurrence.span.clone()], occurrence.raw);
        }
    }

    #[test]
    fn find_markdown_link_occurrences_html_like_autolinks_produce_no_occurrences() {
        let body = "<div> and <https://example.com/x.md> and <not-a-path> and <a href=\"x.md\">";
        let source = "docs/page.md";

        let occurrences = find_markdown_link_occurrences(body, source);
        assert_eq!(
            occurrences,
            vec![],
            "none of these should be treated as document-link autolinks"
        );
    }

    // -- relativize_md_path -------------------------------------------------------

    #[test]
    fn relativize_md_path_matches_documented_examples() {
        assert_eq!(
            relativize_md_path("dev/a.md", "sysadmin/b.md"),
            "../sysadmin/b.md"
        );
        assert_eq!(relativize_md_path("dev/a.md", "dev/b.md"), "b.md");
    }

    #[test]
    fn relativize_md_path_round_trips() {
        // Table-driven, mirroring `extract_markdown_links_cases`'s style: each case is
        // (name, source_rel_path, target_rel_path). `relativize_md_path`'s output, fed
        // back through `resolve_relative_md_path` with the same source, must return the
        // original target — that's the contract, not any particular string shape.
        let cases: &[(&str, &str, &str)] = &[
            ("same directory", "dev/a.md", "dev/b.md"),
            ("deeper subdirectory", "dev/a.md", "dev/sub/b.md"),
            ("shallower — climbs one level", "dev/sub/a.md", "dev/b.md"),
            ("sibling subtree", "dev/sub/a.md", "dev/other/b.md"),
            ("sibling top-level area", "dev/a.md", "sysadmin/b.md"),
            ("root source, nested target", "a.md", "dev/b.md"),
            ("nested source, root target", "dev/sub/a.md", "root.md"),
            ("both at root", "a.md", "b.md"),
            (
                "deep climb then deep descent",
                "dev/a/b/c.md",
                "sysadmin/x/y/z.md",
            ),
            ("target is the source itself", "dev/a.md", "dev/a.md"),
            (
                "shared multi-level prefix",
                "dev/sub/deep/a.md",
                "dev/sub/other/b.md",
            ),
        ];

        for (name, source, target) in cases {
            let relative = relativize_md_path(source, target);
            let round_tripped = resolve_relative_md_path(&relative, source);
            assert_eq!(
                round_tripped.as_deref(),
                Some(*target),
                "case failed: {name} (source={source}, target={target}, relative={relative})"
            );
        }
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

        let fields = crate::qdrant::all_indexed_fields(&config, &schemas);
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
    async fn backfill_populates_markdown_links_for_unchanged_files() {
        // Reproduces a KB indexed before markdown-link extraction existed: the file's
        // indexed_files row is already there, but document_links has never seen it.
        // Backfill (the incremental path's counterpart for unchanged files) must fill
        // in the same edges `upsert_pending` would have written for a changed file.
        let (db, db_dir) = backfill_test_db().await;
        let kb = TempDir::new().unwrap();
        let recipes_dir = kb.path().join("recipes");
        std::fs::create_dir_all(&recipes_dir).unwrap();
        let path = recipes_dir.join("chili.md");
        std::fs::write(
            &path,
            "---\ntitle: Chili\n---\n\nSee [prep](./prep.md) and [sides](../sides/beans.md).",
        )
        .unwrap();

        db.upsert("recipes/chili.md", "stale-hash", 1, "", 0, 0)
            .await
            .unwrap();
        assert!(
            db.all_links().await.unwrap().is_empty(),
            "precondition: no links rows exist before backfill runs"
        );

        let queue = vec![("recipes/chili.md".to_string(), path)];
        let filled =
            backfill_document_metadata(&queue, &db, &HashMap::new(), &empty_schemas()).await;
        assert_eq!(filled, 1);

        let links = db.all_links().await.unwrap();
        let targets: Vec<&str> = links
            .iter()
            .filter(|(source, _, kind, _)| source == "recipes/chili.md" && kind == "markdown")
            .map(|(_, target, _, _)| target.as_str())
            .collect();
        assert!(
            targets.contains(&"recipes/prep.md"),
            "expected recipes/prep.md among {targets:?}"
        );
        assert!(
            targets.contains(&"sides/beans.md"),
            "expected sides/beans.md among {targets:?}"
        );

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

        // Neither collection-lifecycle method is exercised by the tests that use
        // this mock (they drive `upsert_pending`/`remove_orphans` directly, not the
        // full `index_paths_generic` pipeline) — no-op stubs to satisfy the trait.
        async fn drop_collection(&self, _collection: &str) -> Result<()> {
            Ok(())
        }

        async fn ensure_collection(
            &self,
            _collection: &str,
            _vector_size: u64,
            _indexed_fields: &[crate::qdrant::IndexedField],
            _enable_phrase: bool,
        ) -> Result<()> {
            Ok(())
        }
    }

    struct TrackingMockVectorStore {
        delete_by_files_calls: Mutex<Vec<Vec<String>>>,
        deleted_ids: Mutex<Vec<String>>,
        upsert_result: Mutex<Result<()>>,
        upsert_called: Mutex<bool>,
        upserted_points: Mutex<Vec<crate::qdrant::QdrantPoint>>,
        /// Number of `ensure_collection` calls seen — used by
        /// `index_paths_generic` tests to confirm the collection is (re)ensured
        /// exactly once per run, without needing a real Qdrant to ensure.
        ensure_collection_calls: Mutex<usize>,
        drop_collection_calls: Mutex<usize>,
    }

    impl TrackingMockVectorStore {
        fn all_ok() -> Self {
            Self {
                delete_by_files_calls: Mutex::new(Vec::new()),
                deleted_ids: Mutex::new(Vec::new()),
                upsert_result: Mutex::new(Ok(())),
                upsert_called: Mutex::new(false),
                upserted_points: Mutex::new(Vec::new()),
                ensure_collection_calls: Mutex::new(0),
                drop_collection_calls: Mutex::new(0),
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

        async fn drop_collection(&self, _collection: &str) -> Result<()> {
            *self.drop_collection_calls.lock().unwrap() += 1;
            Ok(())
        }

        async fn ensure_collection(
            &self,
            _collection: &str,
            _vector_size: u64,
            _indexed_fields: &[crate::qdrant::IndexedField],
            _enable_phrase: bool,
        ) -> Result<()> {
            *self.ensure_collection_calls.lock().unwrap() += 1;
            Ok(())
        }
    }

    // `index_paths_generic` requires `Q: VectorStore + NeighborStore` since
    // `update_semantic_edges` runs in the same call path. None of the
    // `index_paths_generic` tests above enable `ui.semantic_edges`, so this is
    // never actually called — it exists only to satisfy the trait bound.
    impl NeighborStore for TrackingMockVectorStore {
        async fn recommend_by_point_id(
            &self,
            _collection: &str,
            _point_id: &str,
            _limit: u64,
            _filter: Option<Filter>,
        ) -> Result<Vec<SearchResult>> {
            Ok(vec![])
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
            body: String::new(),
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

    /// Every upserted point's ID must be keyed off its OWN file and chunk index —
    /// not some other file's, and not a position in a flattened, cross-file
    /// sequence. `upsert_pending` flattens every pending file's chunk texts into one
    /// `all_texts` vector before embedding, then walks `pending` again zipped with
    /// `file_boundaries` to reconstruct which embeddings belong to which file. If
    /// that reconstruction ever mispaired a file with the wrong slice — an
    /// off-by-one in the boundary zip, or generating IDs from the chunk's position
    /// in the flattened batch instead of `chunk.index` — the resulting points would
    /// upsert under IDs belonging to a *different* document: silently shadowing
    /// whatever was there and leaving the real document's true chunks unreachable.
    /// Two files, each with its own 2-chunk `make_pending`, makes that failure mode
    /// directly observable per point rather than only in aggregate.
    #[tokio::test]
    async fn upsert_pending_assigns_point_ids_keyed_by_own_file_and_chunk_index() {
        let dir = TempDir::new().unwrap();
        let state = test_state_db(&dir).await;

        let pending = vec![
            make_pending("data/first.md", 2, 0),
            make_pending("data/second.md", 2, 0),
        ];
        // 4 chunks total across the two files, in flattened order.
        let embedder = MockEmbedClient::ok(vec![vec![1.0; 3]; 4]);
        let store = TrackingMockVectorStore::all_ok();

        let result = upsert_pending(&pending, &embedder, &store, &state, "test-col").await;
        assert!(result.is_ok());

        let points = store.upserted_points.lock().unwrap();
        assert_eq!(points.len(), 4);

        for pf in &pending {
            for chunk in &pf.chunks {
                let expected_id = make_point_id(&pf.file_path, chunk.index);
                let matching = points.iter().find(|p| p.id == expected_id);
                let point = matching.unwrap_or_else(|| {
                    panic!(
                        "no upserted point with id {expected_id} for {}#{}",
                        pf.file_path, chunk.index
                    )
                });
                assert_eq!(
                    point.payload.get("file_path").and_then(|v| v.as_str()),
                    Some(pf.file_path.as_str()),
                    "point {expected_id} carries the wrong file's payload"
                );
                assert_eq!(
                    point
                        .payload
                        .get("chunk_index")
                        .and_then(serde_json::Value::as_u64),
                    Some(chunk.index as u64),
                    "point {expected_id} carries the wrong chunk_index"
                );
            }
        }
    }

    // -- update_semantic_edges -------------------------------------------------

    /// Canned-response `NeighborStore` for `update_semantic_edges` tests — returns
    /// the same fixed hit list for every call regardless of collection/point
    /// id/limit/filter, so tests only need to shape the hits, not the request.
    struct FakeNeighborStore {
        hits: Vec<SearchResult>,
    }

    impl NeighborStore for FakeNeighborStore {
        async fn recommend_by_point_id(
            &self,
            _collection: &str,
            _point_id: &str,
            _limit: u64,
            _filter: Option<Filter>,
        ) -> Result<Vec<SearchResult>> {
            Ok(self.hits.clone())
        }
    }

    /// A canned neighbor hit carrying just the `file_path` payload field
    /// `update_semantic_edges` reads.
    fn neighbor_hit(file_path: &str, score: f32) -> SearchResult {
        let mut payload = HashMap::new();
        payload.insert(
            "file_path".to_string(),
            serde_json::Value::String(file_path.to_string()),
        );
        SearchResult {
            score,
            pre_rerank_score: None,
            dense_score: Some(score),
            sparse_score: None,
            phrase_score: None,
            payload,
        }
    }

    fn semantic_edges_cfg(enabled: bool, k: u64, min_score: f32) -> SemanticEdgesConfig {
        SemanticEdgesConfig {
            enabled,
            k,
            min_score,
        }
    }

    #[tokio::test]
    async fn update_semantic_edges_writes_replace_links_with_scores() {
        let dir = TempDir::new().unwrap();
        let state = test_state_db(&dir).await;
        let pending = vec![make_pending("a.md", 1, 0)];
        let neighbors = FakeNeighborStore {
            hits: vec![neighbor_hit("b.md", 0.9), neighbor_hit("c.md", 0.7)],
        };
        let cfg = semantic_edges_cfg(true, 5, 0.6);

        update_semantic_edges(&pending, &neighbors, &state, "col", &cfg).await;

        let mut links = state.all_links().await.unwrap();
        links.sort_by(|a, b| a.1.cmp(&b.1));
        assert_eq!(
            links,
            vec![
                (
                    "a.md".to_string(),
                    "b.md".to_string(),
                    "semantic".to_string(),
                    Some(0.9_f32 as f64)
                ),
                (
                    "a.md".to_string(),
                    "c.md".to_string(),
                    "semantic".to_string(),
                    Some(0.7_f32 as f64)
                ),
            ]
        );
    }

    /// Even though the `must_not` filter on `file_path` is the real exclusion
    /// mechanism (see `qdrant.rs`), a fake store — like this test's — has no reason
    /// to honor it, so this proves `update_semantic_edges` also drops a self-hit
    /// defensively on its own.
    #[tokio::test]
    async fn update_semantic_edges_excludes_self_hit() {
        let dir = TempDir::new().unwrap();
        let state = test_state_db(&dir).await;
        let pending = vec![make_pending("a.md", 1, 0)];
        let neighbors = FakeNeighborStore {
            hits: vec![neighbor_hit("a.md", 0.99), neighbor_hit("b.md", 0.8)],
        };
        let cfg = semantic_edges_cfg(true, 5, 0.6);

        update_semantic_edges(&pending, &neighbors, &state, "col", &cfg).await;

        let links = state.all_links().await.unwrap();
        assert_eq!(
            links,
            vec![(
                "a.md".to_string(),
                "b.md".to_string(),
                "semantic".to_string(),
                Some(0.8_f32 as f64)
            )],
            "a.md must never be recommended as its own neighbor"
        );
    }

    #[tokio::test]
    async fn update_semantic_edges_filters_below_min_score() {
        let dir = TempDir::new().unwrap();
        let state = test_state_db(&dir).await;
        let pending = vec![make_pending("a.md", 1, 0)];
        let neighbors = FakeNeighborStore {
            hits: vec![neighbor_hit("b.md", 0.9), neighbor_hit("c.md", 0.4)],
        };
        let cfg = semantic_edges_cfg(true, 5, 0.6);

        update_semantic_edges(&pending, &neighbors, &state, "col", &cfg).await;

        let links = state.all_links().await.unwrap();
        assert_eq!(
            links,
            vec![(
                "a.md".to_string(),
                "b.md".to_string(),
                "semantic".to_string(),
                Some(0.9_f32 as f64)
            )],
            "c.md scored 0.4, below the 0.6 min_score threshold"
        );
    }

    #[tokio::test]
    async fn update_semantic_edges_truncates_to_top_k() {
        let dir = TempDir::new().unwrap();
        let state = test_state_db(&dir).await;
        let pending = vec![make_pending("a.md", 1, 0)];
        let neighbors = FakeNeighborStore {
            hits: vec![
                neighbor_hit("b.md", 0.9),
                neighbor_hit("c.md", 0.8),
                neighbor_hit("d.md", 0.7),
            ],
        };
        let cfg = semantic_edges_cfg(true, 2, 0.0);

        update_semantic_edges(&pending, &neighbors, &state, "col", &cfg).await;

        let links = state.all_links().await.unwrap();
        let targets: Vec<&str> = links.iter().map(|l| l.1.as_str()).collect();
        assert_eq!(
            targets,
            vec!["b.md", "c.md"],
            "only the top 2 by score should survive k=2"
        );
    }

    #[tokio::test]
    async fn update_semantic_edges_disabled_short_circuits() {
        let dir = TempDir::new().unwrap();
        let state = test_state_db(&dir).await;
        let pending = vec![make_pending("a.md", 1, 0)];
        let neighbors = FakeNeighborStore {
            hits: vec![neighbor_hit("b.md", 0.9)],
        };
        let cfg = semantic_edges_cfg(false, 5, 0.6);

        update_semantic_edges(&pending, &neighbors, &state, "col", &cfg).await;

        assert!(
            state.all_links().await.unwrap().is_empty(),
            "disabled config must not write any semantic links"
        );
    }

    /// A failed neighbor lookup for one file must not abort the batch or propagate
    /// an error — the caller (`index_paths_generic`) does not (and must not have to)
    /// handle a `Result` from this function.
    #[tokio::test]
    async fn update_semantic_edges_lookup_failure_is_non_fatal() {
        struct FailingNeighborStore;
        impl NeighborStore for FailingNeighborStore {
            async fn recommend_by_point_id(
                &self,
                _collection: &str,
                _point_id: &str,
                _limit: u64,
                _filter: Option<Filter>,
            ) -> Result<Vec<SearchResult>> {
                anyhow::bail!("qdrant unreachable")
            }
        }

        let dir = TempDir::new().unwrap();
        let state = test_state_db(&dir).await;
        let pending = vec![make_pending("a.md", 1, 0)];
        let cfg = semantic_edges_cfg(true, 5, 0.6);

        // Must not panic; must leave no links behind for a lookup that never succeeded.
        update_semantic_edges(&pending, &FailingNeighborStore, &state, "col", &cfg).await;

        assert!(state.all_links().await.unwrap().is_empty());
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
