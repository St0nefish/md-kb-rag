use std::collections::HashMap;
use std::path::{Path, PathBuf};

use globset::GlobSet;
use tracing::{debug, warn};

use crate::{
    embed::QueryEmbedder,
    qdrant::{RetrievalStore, SearchResult},
    rerank::Reranker,
    state::{DocumentIndex, DocumentQuery, DocumentQueryResult},
};

/// How many "did you mean?" suggestions to include when no basename matches.
pub const FUZZY_SUGGESTION_COUNT: usize = 3;

/// Dependencies needed by the shared retrieval functions.
/// Dependencies for document listing.
///
/// Deliberately separate from [`RetrievalDeps`] rather than a third generic parameter
/// on it: `search` never touches the metadata index, and widening its deps struct
/// would ripple through every construction site and mock that only ever searches.
/// `get_document` also needs the metadata index (for its fuzzy fallback — see its own
/// doc comment) but takes a bare `&D: DocumentIndex` parameter rather than this
/// wrapper, for the same reason: it has no listing-shaped fields to bundle it with.
pub struct DocumentIndexDeps<'a, D: DocumentIndex> {
    pub index: &'a D,
}

/// List documents by metadata, with no embedding call and no vector search.
///
/// Unlike [`search`], this is exhaustive and deterministic: the result carries the
/// total number of matches so a caller can always detect truncation.
pub async fn list_documents<D: DocumentIndex>(
    deps: &DocumentIndexDeps<'_, D>,
    query: &DocumentQuery,
) -> anyhow::Result<DocumentQueryResult> {
    deps.index.query_documents(query).await
}

pub struct RetrievalDeps<'a, E: QueryEmbedder, Q: RetrievalStore> {
    pub embed_client: &'a E,
    pub qdrant: &'a Q,
    pub collection: &'a str,
    pub data_path: &'a Path,
    pub include_patterns: &'a GlobSet,
    pub reranker: Option<&'a (dyn Reranker + Send + Sync)>,
}

/// Filters to apply when searching.
pub struct SearchFilters {
    pub domain: Option<String>,
    pub r#type: Option<String>,
    pub tags: Option<Vec<String>>,
}

/// Options controlling search behaviour.
pub struct SearchOptions {
    pub limit: u64,
    pub min_score: Option<f32>,
    /// When true, use hybrid sparse+dense retrieval with RRF fusion; otherwise
    /// dense-only. Sourced from `search.hybrid`.
    pub hybrid: bool,
    /// Candidates fetched from each arm before RRF fusion. Sourced from
    /// `search.rrf_candidates`. Only consulted when `hybrid` is true.
    pub rrf_candidates: u64,
    /// When true, surface per-result score breakdown metadata (pre-rerank score when applicable).
    pub explain: bool,
    /// Exclude documents with `mtime` payload below this Unix timestamp.
    pub modified_after: Option<i64>,
    /// Exclude documents with `mtime` payload above this Unix timestamp.
    pub modified_before: Option<i64>,
    /// When reranking is enabled, the number of candidates to fetch before reranking.
    /// Ignored when `reranker` is None on RetrievalDeps.
    pub rerank_candidate_limit: Option<u64>,
    /// Per-document cap on the final result set — see [`apply_diversity_cap`] and
    /// the diversity-cap block in [`search`] for the full funnel-placement
    /// reasoning. `None` disables diversity (a single document may fill every
    /// slot, matching pre-#86 behaviour). Sourced from
    /// `search.diversity_max_per_document`.
    pub diversity_max_per_document: Option<usize>,
}

/// How much more generous the *pre-rerank* candidate cap is than the *final*
/// result cap (`diversity_max_per_document`), when both a reranker and diversity
/// are active.
///
/// The pre-rerank pass operates on RRF/dense-fused order — a materially weaker
/// relevance signal than the cross-encoder about to run — so capping it down to
/// the same tight number as the final cap risks discarding a document's true
/// best chunk before reranking ever gets a chance to identify it (fusion order
/// and true relevance order are not the same thing; that's the entire reason a
/// reranker exists). Multiplying the final cap by a generous factor keeps that
/// risk low while still doing real work: it trims a document that has,
/// implausibly, tens of chunks in the top of the fused pool down to a bounded
/// number before they're all sent to a paid, latency-bound reranker call — most
/// of which could never survive the final cap regardless of their rerank score.
const PRERANK_DIVERSITY_MULTIPLIER: usize = 4;

/// Apply a per-document cap to an already-ranked (best-first) result list,
/// dropping entries once a document (identified by its `file_path` payload
/// field) has contributed `max_per_document` results. Preserves relative order
/// of everything kept, so a lower-ranked chunk from an under-represented
/// document naturally backfills the slot a capped chunk would have occupied.
///
/// A result with no `file_path` in its payload is never capped — this should
/// not happen for real indexed chunks (`file_path` is always written by
/// `ingest.rs`), and treating an absent grouping key as "always distinct" is
/// safer than either dropping such a result or silently grouping unrelated
/// results together under an empty-string key.
fn apply_diversity_cap(
    mut results: Vec<SearchResult>,
    max_per_document: usize,
) -> Vec<SearchResult> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    results.retain(
        |r| match r.payload.get("file_path").and_then(|v| v.as_str()) {
            Some(file_path) => {
                let count = counts.entry(file_path.to_string()).or_insert(0);
                if *count < max_per_document {
                    *count += 1;
                    true
                } else {
                    false
                }
            }
            None => true,
        },
    );
    results
}

/// Same capping rule as [`apply_diversity_cap`], applied instead to the
/// post-rerank `(original_index_into_results, relevance_score)` pairs, which
/// are already sorted best-first by cross-encoder score. `results` is the
/// pre-rerank candidate pool the indices were computed against — used here only
/// to look up each candidate's `file_path` for grouping.
fn cap_reranked_by_document(
    indexed: Vec<(usize, f32)>,
    results: &[SearchResult],
    max_per_document: usize,
) -> Vec<(usize, f32)> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    indexed
        .into_iter()
        .filter(|(orig_i, _)| {
            match results
                .get(*orig_i)
                .and_then(|r| r.payload.get("file_path"))
                .and_then(|v| v.as_str())
            {
                Some(file_path) => {
                    let count = counts.entry(file_path.to_string()).or_insert(0);
                    if *count < max_per_document {
                        *count += 1;
                        true
                    } else {
                        false
                    }
                }
                None => true,
            }
        })
        .collect()
}

/// A successfully retrieved document.
pub struct Document {
    pub path: PathBuf,
    pub content: String,
}

// ---------------------------------------------------------------------------
// Line ranges
// ---------------------------------------------------------------------------

/// A caller-requested slice of a document, expressed in **1-based, inclusive**
/// line numbers — the numbering an editor shows, so a caller can hand back the
/// same numbers it read in a diff or a lint message without converting.
///
/// `end == None` means "to the end of the document". An `end` past the last
/// line is clamped rather than rejected: asking for more document than exists
/// is not a caller error, and clamping lets `end_line` in the response report
/// what was actually served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineRange {
    pub start: usize,
    pub end: Option<usize>,
}

/// Why a requested [`LineRange`] could not be served.
///
/// These are all caller mistakes (a malformed range, or one anchored past the
/// end of the document), so every variant carries enough context for the
/// message to say what the document actually offers.
#[derive(Debug, PartialEq, Eq)]
pub enum LineRangeError {
    /// A line number of 0 was given. Lines are numbered from 1.
    ZeroLine,
    /// `end` is before `start`.
    Inverted { start: usize, end: usize },
    /// `start` is past the last line — including every range against an empty
    /// document, which has no line 1 to anchor to.
    StartPastEnd { start: usize, total_lines: usize },
}

impl std::fmt::Display for LineRangeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroLine => write!(
                f,
                "line numbers are 1-based; start_line and end_line must be 1 or greater"
            ),
            Self::Inverted { start, end } => write!(
                f,
                "end_line ({end}) is before start_line ({start}); the range is empty"
            ),
            Self::StartPastEnd { start, total_lines } => write!(
                f,
                "start_line ({start}) is past the end of the document, which has {total_lines} line(s)"
            ),
        }
    }
}

impl LineRange {
    /// Build a range from the two optional parameters a tool surface exposes.
    ///
    /// Returns `Ok(None)` when neither bound was given — the "whole document"
    /// request, kept distinct from a range so callers need not special-case a
    /// synthetic `1..=total` on the common path. A lone `end_line` implies a
    /// start of 1, and a lone `start_line` runs to the end of the document.
    ///
    /// Only the bounds' relationship to each other is checked here; whether
    /// they land inside a particular document is [`slice_lines`]'s business,
    /// since that needs the content. That split lets a caller reject a
    /// malformed range before doing the work of resolving and reading a file.
    pub fn new(
        start_line: Option<usize>,
        end_line: Option<usize>,
    ) -> Result<Option<Self>, LineRangeError> {
        if start_line.is_none() && end_line.is_none() {
            return Ok(None);
        }
        if start_line == Some(0) || end_line == Some(0) {
            return Err(LineRangeError::ZeroLine);
        }
        let start = start_line.unwrap_or(1);
        if let Some(end) = end_line
            && end < start
        {
            return Err(LineRangeError::Inverted { start, end });
        }
        Ok(Some(Self {
            start,
            end: end_line,
        }))
    }
}

/// The lines of a document that were actually served.
///
/// `start_line`/`end_line` describe the slice as delivered, not as requested —
/// a clamped `end` reports the last line of the document — so a caller can tell
/// from the response alone whether it reached the end without re-deriving it.
#[derive(Debug)]
pub struct LineSlice {
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
    pub total_lines: usize,
}

impl LineSlice {
    /// Whether any of the document was withheld. Derived from what was served
    /// rather than from whether a range was requested, so a range that happens
    /// to cover the whole document reports the same thing an unranged read does.
    pub fn partial(&self) -> bool {
        self.start_line > 1 || self.end_line < self.total_lines
    }
}

/// Count the lines in a document, using the same rule [`slice_lines`] numbers
/// by: a line is a run of text ending in `\n`, plus any unterminated remainder.
/// A document with a trailing newline therefore does *not* count a phantom
/// empty line after it, and an empty document has zero lines.
pub fn count_lines(content: &str) -> usize {
    content.split_inclusive('\n').count()
}

/// Extract a 1-based, inclusive line range from a document.
///
/// Slices on `split_inclusive('\n')`, so the returned content is a byte-exact
/// substring of the original: line terminators (including CRLF) are preserved
/// as they were, and a slice ending on an unterminated last line stays
/// unterminated. Nothing is re-joined or normalized, which is what lets a
/// caller feed the result straight back as an `edit_document` `old_string`.
/// [`slice_lines`] for the callers that treat "no range given" as "the whole
/// document" — every read surface does.
///
/// Returning the same [`LineSlice`] either way is the point: a response can then
/// report `start_line`/`end_line`/`total_lines`/`partial` unconditionally, so a
/// client never has to branch on their presence to learn how much document it is
/// holding. Takes the content by value so the unranged path — the common one —
/// hands the string straight through instead of copying it.
pub fn slice_or_whole(
    content: String,
    range: Option<&LineRange>,
) -> Result<LineSlice, LineRangeError> {
    match range {
        Some(range) => slice_lines(&content, range),
        None => {
            let total_lines = count_lines(&content);
            Ok(LineSlice {
                content,
                start_line: 1,
                end_line: total_lines,
                total_lines,
            })
        }
    }
}

pub fn slice_lines(content: &str, range: &LineRange) -> Result<LineSlice, LineRangeError> {
    let lines: Vec<&str> = content.split_inclusive('\n').collect();
    let total_lines = lines.len();

    if range.start > total_lines {
        return Err(LineRangeError::StartPastEnd {
            start: range.start,
            total_lines,
        });
    }
    // Clamp rather than reject: see `LineRange`.
    let end_line = range.end.unwrap_or(total_lines).min(total_lines);

    Ok(LineSlice {
        content: lines[range.start - 1..end_line].concat(),
        start_line: range.start,
        end_line,
        total_lines,
    })
}

/// Structured errors from `search`, distinguishing the failing stage so callers
/// can surface stage-specific messages.
#[derive(Debug)]
pub enum SearchError {
    /// The query-embedding call failed.
    Embed(anyhow::Error),
    /// The Qdrant search call failed.
    Search(anyhow::Error),
}

/// Structured errors from `get_document`.
#[derive(Debug)]
pub enum GetDocumentError {
    /// Resolved outside the data directory (path traversal). Hard fail.
    Outside,
    /// Resolved to a file type excluded by `indexing.include`. Hard fail.
    NotPermitted,
    /// File not found — eligible for fuzzy suggestions.
    /// Suggestions are already relative to the data root.
    NotFound { suggestions: Vec<String> },
    /// Multiple indexed files share the same basename.
    /// Matches are already relative to the data root.
    Ambiguous { matches: Vec<String> },
    /// Other I/O / internal error.
    Io(String),
}

// ---------------------------------------------------------------------------
// Internal path helpers
// ---------------------------------------------------------------------------

/// Strip the data-root prefix from an absolute file_path for display to
/// clients. Falls back to returning the path unchanged if it doesn't share the
/// prefix.
pub fn relative_to_data(absolute: &str, data_root: &Path) -> String {
    Path::new(absolute)
        .strip_prefix(data_root)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| absolute.to_string())
}

/// Levenshtein edit distance between two strings (chars, not bytes).
/// O(m*n) time, O(n) space. Used for "did you mean?" suggestions over a few
/// hundred basenames — fine for our scale, no extra dependency needed.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (curr[j - 1] + 1).min(prev[j] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

// ---------------------------------------------------------------------------
// Internal resolve helpers
// ---------------------------------------------------------------------------

/// Outcome of trying to resolve a user-supplied path against the data directory.
pub(crate) enum ResolveErr {
    /// File didn't exist — eligible for fuzzy basename fallback.
    NotFound,
    /// Resolved outside the data directory (path traversal). Hard fail.
    Outside,
    /// Resolved to a file type excluded by `indexing.include`. Hard fail.
    NotPermitted,
    /// Other I/O error (permissions, etc). Hard fail with the original message.
    Other(String),
}

/// Strip a leading separator so a caller can address the knowledge-base root as `/`.
pub(crate) fn kb_root_relative(raw: &str) -> &str {
    raw.trim_start_matches('/')
}

/// Resolve a user-supplied path against the data directory, applying the
/// path-traversal and file-type security checks.
pub(crate) fn resolve_within_data(
    raw: &str,
    data_path: &Path,
    include_patterns: &GlobSet,
) -> Result<PathBuf, ResolveErr> {
    let requested = PathBuf::from(raw);
    let resolved = if requested.is_absolute() {
        // A leading `/` is ambiguous: it could be a real filesystem path (accepted
        // historically) or the caller treating the knowledge-base root as `/`, which is
        // the natural reading since callers cannot know where the KB lives inside the
        // container. Try the literal path first, then fall back to KB-root-relative.
        if requested.exists() {
            requested
        } else {
            data_path.join(kb_root_relative(raw))
        }
    } else {
        data_path.join(&requested)
    };

    let canonical = resolved.canonicalize().map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => ResolveErr::NotFound,
        std::io::ErrorKind::PermissionDenied => {
            ResolveErr::Other(format!("Permission denied: {}", resolved.display()))
        }
        _ => ResolveErr::Other(format!(
            "Cannot access file '{}': {}",
            resolved.display(),
            e
        )),
    })?;

    if !canonical.starts_with(data_path) {
        return Err(ResolveErr::Outside);
    }

    let relative = canonical.strip_prefix(data_path).unwrap_or(&canonical);
    if !include_patterns.is_match(relative) {
        return Err(ResolveErr::NotPermitted);
    }

    Ok(canonical)
}

// ---------------------------------------------------------------------------
// Public retrieval functions
// ---------------------------------------------------------------------------

/// Embed the query, apply filters, search Qdrant, apply min_score floor, and
/// return raw results. Does NOT format any output — callers do that.
pub async fn search<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &RetrievalDeps<'_, E, Q>,
    query: &str,
    filters: &SearchFilters,
    opts: &SearchOptions,
) -> Result<Vec<SearchResult>, SearchError> {
    let embed_start = std::time::Instant::now();
    let vector = deps
        .embed_client
        .embed_query(query)
        .await
        .map_err(SearchError::Embed)?;
    let embed_ms = embed_start.elapsed().as_millis();

    let mut filter_map: HashMap<String, serde_json::Value> = HashMap::new();

    // Collapse three near-identical scalar-filter insertions with a local helper.
    // We borrow each field directly from `filters` and move the String value into
    // the map rather than cloning it separately — the struct fields are owned and
    // not used after this point.
    let mut insert_str = |key: &str, opt: &Option<String>| {
        if let Some(v) = opt {
            filter_map.insert(key.to_string(), serde_json::Value::String(v.clone()));
        }
    };
    insert_str("domain", &filters.domain);
    insert_str("type", &filters.r#type);

    // Empty tags list = no filter; avoid sending an empty match_any to Qdrant.
    if let Some(tags) = &filters.tags
        && !tags.is_empty()
    {
        let tag_values: Vec<serde_json::Value> = tags
            .iter()
            .map(|t| serde_json::Value::String(t.clone()))
            .collect();
        filter_map.insert("tags".to_string(), serde_json::Value::Array(tag_values));
    }

    // mtime range filter — documents indexed before mtime was stored may have
    // mtime=0 or no mtime field and will be excluded silently by this filter.
    if opts.modified_after.is_some() || opts.modified_before.is_some() {
        debug!("mtime filter active — documents without mtime in payload will be excluded");
    }
    if let Some(after) = opts.modified_after {
        filter_map.insert("mtime__gte".to_string(), serde_json::json!(after));
    }
    if let Some(before) = opts.modified_before {
        filter_map.insert("mtime__lte".to_string(), serde_json::json!(before));
    }

    let fetch_limit = if deps.reranker.is_some() {
        opts.rerank_candidate_limit.unwrap_or(opts.limit)
    } else {
        opts.limit
    };

    let search_start = std::time::Instant::now();
    let mut results = if opts.hybrid {
        // Tokenize the raw query into a sparse vector and fuse with the dense arm.
        // Fall back to dense-only if the query produces an empty sparse vector
        // (e.g. all-punctuation queries) to avoid sending an empty vector to Qdrant.
        let sparse = crate::sparse::tokenize(query);
        if sparse.0.is_empty() {
            if opts.explain {
                debug!(
                    "hybrid sparse-fallback: empty sparse vector, explain scores reflect dense-only"
                );
            }
            deps.qdrant
                .search(deps.collection, vector, filter_map, fetch_limit)
                .await
                .map_err(SearchError::Search)?
        } else {
            deps.qdrant
                .hybrid_search(
                    deps.collection,
                    vector,
                    sparse,
                    filter_map,
                    fetch_limit,
                    opts.rrf_candidates,
                    opts.explain,
                )
                .await
                .map_err(SearchError::Search)?
        }
    } else {
        deps.qdrant
            .search(deps.collection, vector, filter_map, fetch_limit)
            .await
            .map_err(SearchError::Search)?
    };
    let search_ms = search_start.elapsed().as_millis();

    // Apply min_score floor only when Some — None is a no-op, preserving
    // current behaviour.
    if let Some(s) = opts.min_score {
        results.retain(|r| r.score >= s);
    }

    if let Some(reranker) = deps.reranker {
        // Diversity, pass 1 of 2 — generous pre-rerank cap on the candidate pool.
        // See `PRERANK_DIVERSITY_MULTIPLIER`'s doc comment for why this uses a
        // looser number than the final cap rather than the same one: fusion order
        // is not relevance order, so capping tightly here (before the
        // cross-encoder runs) risks throwing away a document's true best chunk.
        // What this DOES do is stop an implausibly over-represented document from
        // burning the entire (paid, latency-bound) reranker call on chunks that
        // could never survive the final cap regardless of their rerank score.
        //
        // This only trims *within* whatever Qdrant already returned (bounded by
        // `rerank_candidate_limit` / `rrf_candidates`) — it cannot recover a
        // different document's chunks that Qdrant dropped before this function
        // ever saw them. That earlier stage is a recall problem, tuned separately
        // via `search.rrf_candidates` / `reranking.candidate_limit`, not something
        // a post-hoc diversity pass can fix.
        if let Some(max_per_document) = opts.diversity_max_per_document {
            let prerank_cap = max_per_document.saturating_mul(PRERANK_DIVERSITY_MULTIPLIER);
            results = apply_diversity_cap(results, prerank_cap);
        }

        // When explain is requested, snapshot pre-rerank scores keyed by index
        // so we can attach them to each result after reranking updates the score.
        let pre_rerank_scores: Option<Vec<f32>> = if opts.explain {
            Some(results.iter().map(|r| r.score).collect())
        } else {
            None
        };

        let docs_with_indices: Vec<(usize, &str)> = results
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                r.payload
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| (i, s))
            })
            .collect();
        let docs: Vec<&str> = docs_with_indices.iter().map(|(_, s)| *s).collect();
        let top_k = opts.limit as usize;
        if docs.is_empty() {
            if let Some(max_per_document) = opts.diversity_max_per_document {
                results = apply_diversity_cap(results, max_per_document);
            }
            results.truncate(top_k);
            return Ok(results);
        }
        match reranker.rerank(query, &docs).await {
            Ok(ranked) => {
                let mut indexed: Vec<(usize, f32)> = ranked
                    .iter()
                    .map(|r| (docs_with_indices[r.index].0, r.relevance_score))
                    .collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Diversity, pass 2 of 2 — the real, user-tunable cap, applied to
                // the cross-encoder's own ranking (the best relevance signal
                // available) and BEFORE truncating to `top_k`. Order matters here:
                // capping before truncation lets the next-best chunk from an
                // under-represented document backfill the slot a capped chunk
                // would have taken, rather than just shrinking the result count.
                if let Some(max_per_document) = opts.diversity_max_per_document {
                    indexed = cap_reranked_by_document(indexed, &results, max_per_document);
                }

                results = indexed
                    .into_iter()
                    .take(top_k)
                    .filter_map(|(orig_i, score)| {
                        results.get(orig_i).map(|r| {
                            let mut hit = r.clone();
                            hit.pre_rerank_score = pre_rerank_scores
                                .as_ref()
                                .and_then(|v| v.get(orig_i))
                                .copied();
                            hit.score = score;
                            hit
                        })
                    })
                    .collect();
            }
            Err(e) => {
                warn!("Reranker unavailable, falling back to fused order: {e:#}");
                // Fused order stands in for a relevance ranking here, same as the
                // no-reranker branch below — cap it the same way before truncating.
                if let Some(max_per_document) = opts.diversity_max_per_document {
                    results = apply_diversity_cap(results, max_per_document);
                }
                results.truncate(top_k);
            }
        }
    } else {
        // No reranker: the fused (hybrid) or dense-cosine order IS the relevance
        // ranking, so the diversity cap applies directly to it, before truncating
        // to the caller's requested `limit` — same backfill reasoning as pass 2
        // above (capping pre-truncation lets a lower-ranked, under-represented
        // document's chunk fill the slot a capped chunk would have taken).
        if let Some(max_per_document) = opts.diversity_max_per_document {
            results = apply_diversity_cap(results, max_per_document);
        }
        results.truncate(opts.limit as usize);
    }

    debug!(
        embed_ms,
        search_ms,
        results = results.len(),
        "search timing"
    );

    Ok(results)
}

/// Resolve a document path and return its full content.
///
/// On success returns `Document { path, content }`. On failure returns a
/// structured `GetDocumentError` — callers are responsible for turning these
/// into user-facing error strings.
///
/// `document_index` backs the fuzzy fallback (see step 2 below) — the SQLite
/// `documents` table, not Qdrant. See that step's comment for why, and for the
/// consistency tradeoff that follows from the choice.
pub async fn get_document<E: QueryEmbedder, Q: RetrievalStore, D: DocumentIndex>(
    deps: &RetrievalDeps<'_, E, Q>,
    document_index: &D,
    raw: &str,
) -> Result<Document, GetDocumentError> {
    // 1. Try the literal path as given.
    match resolve_within_data(raw, deps.data_path, deps.include_patterns) {
        Ok(canonical) => {
            let content = tokio::fs::read_to_string(&canonical)
                .await
                .map_err(|e| GetDocumentError::Io(e.to_string()))?;
            return Ok(Document {
                path: canonical,
                content,
            });
        }
        Err(ResolveErr::NotFound) => {
            // Fall through to fuzzy basename matching.
        }
        Err(ResolveErr::Outside) => return Err(GetDocumentError::Outside),
        Err(ResolveErr::NotPermitted) => return Err(GetDocumentError::NotPermitted),
        Err(ResolveErr::Other(msg)) => return Err(GetDocumentError::Io(msg)),
    }

    // 2. Fuzzy fallback: load every indexed path and look for a basename match.
    //    Auto-resolve a unique match; otherwise produce a helpful error.
    //
    //    Sourced from the SQLite `documents` table rather than Qdrant's file_path
    //    facet (the previous approach): `documents` already holds exactly this list
    //    — one row per successfully indexed file, maintained by the same pipeline
    //    that writes Qdrant (`ingest::index_paths_inner` / `remove_orphans`) — so a
    //    local, indexed SELECT beats a network round trip to the vector store, needs
    //    no arbitrary cap (the previous fetch capped at 10,000 and silently dropped
    //    paths past it), and stays cheap as the corpus grows into the thousands.
    //
    //    Consistency: `documents` can transiently disagree with Qdrant across a
    //    single indexing run's bookkeeping step. On create/update, Qdrant is written
    //    first and `documents` second (`ingest::index_paths_inner`); a bookkeeping
    //    failure between the two — logged, retried next run — leaves a freshly
    //    indexed file searchable via Qdrant but briefly absent from `documents`, so a
    //    typo'd path to a document indexed moments ago could momentarily miss a
    //    fuzzy match. On delete, the order reverses (Qdrant points removed first,
    //    then `documents`), so the previous Qdrant-facet approach had the mirror-image
    //    race: a path could briefly still fuzzy-match in `documents` after its
    //    content was already gone from Qdrant (the existing secondary-resolve-failure
    //    handling below already covers that case). Either way the window is one
    //    indexing run wide and self-heals on the next; `list_documents` already reads
    //    from `documents` under the identical tradeoff, so this is consistent with
    //    the rest of the codebase rather than a new risk.
    let all_paths = document_index.all_paths().await.unwrap_or_else(|e| {
        warn!("Failed to fetch indexed paths for fuzzy lookup: {e:#}");
        Vec::new()
    });

    let basename = Path::new(raw)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(raw);

    let exact: Vec<&String> = all_paths
        .iter()
        .filter(|p| Path::new(p.as_str()).file_name().and_then(|n| n.to_str()) == Some(basename))
        .collect();

    match exact.len() {
        1 => match resolve_within_data(exact[0], deps.data_path, deps.include_patterns) {
            Ok(canonical) => {
                let content = tokio::fs::read_to_string(&canonical)
                    .await
                    .map_err(|e| GetDocumentError::Io(e.to_string()))?;
                Ok(Document {
                    path: canonical,
                    content,
                })
            }
            Err(_) => {
                warn!(path = %exact[0], "Fuzzy-matched path failed secondary resolve");
                Err(GetDocumentError::NotFound {
                    suggestions: vec![],
                })
            }
        },
        0 => {
            let mut scored: Vec<(usize, &String)> = all_paths
                .iter()
                .map(|p| {
                    let bn = Path::new(p.as_str())
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("");
                    (levenshtein(basename, bn), p)
                })
                .collect();
            scored.sort_by_key(|(d, _)| *d);
            scored.truncate(FUZZY_SUGGESTION_COUNT);

            let suggestions: Vec<String> = scored
                .into_iter()
                .map(|(_, p)| relative_to_data(p, deps.data_path))
                .collect();

            Err(GetDocumentError::NotFound { suggestions })
        }
        _ => {
            let matches: Vec<String> = exact
                .into_iter()
                .map(|p| relative_to_data(p, deps.data_path))
                .collect();
            Err(GetDocumentError::Ambiguous { matches })
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::QueryEmbedder;
    use crate::qdrant::{RetrievalStore, SearchResult};

    // ------------------------------------------------------------------
    // relative_to_data
    // ------------------------------------------------------------------

    #[test]
    fn relative_to_data_strips_prefix() {
        let root = std::path::Path::new("/data");
        assert_eq!(
            relative_to_data("/data/lifestyle/foo.md", root),
            "lifestyle/foo.md"
        );
    }

    #[test]
    fn relative_to_data_handles_trailing_slash_root() {
        let root = std::path::Path::new("/data/");
        assert_eq!(
            relative_to_data("/data/lifestyle/foo.md", root),
            "lifestyle/foo.md"
        );
    }

    #[test]
    fn relative_to_data_returns_unchanged_when_no_prefix_match() {
        let root = std::path::Path::new("/data");
        assert_eq!(relative_to_data("/other/foo.md", root), "/other/foo.md");
    }

    #[test]
    fn relative_to_data_path_equal_to_root_returns_empty() {
        // This input (/data itself, not a child) is not produced in practice.
        let root = std::path::Path::new("/data");
        assert_eq!(relative_to_data("/data", root), "");
    }

    #[test]
    fn relative_to_data_partial_segment_does_not_match() {
        let root = std::path::Path::new("/data");
        let out = relative_to_data("/data-other/foo.md", root);
        assert_eq!(out, "/data-other/foo.md");
    }

    // ------------------------------------------------------------------
    // levenshtein
    // ------------------------------------------------------------------

    #[test]
    fn levenshtein_identical_is_zero() {
        assert_eq!(levenshtein("hello", "hello"), 0);
    }

    #[test]
    fn levenshtein_empty_string() {
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", ""), 0);
    }

    #[test]
    fn levenshtein_substitutions_and_inserts() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn levenshtein_ranks_close_matches_lower() {
        let target = "tacoma-2025.md";
        let close = levenshtein(target, "tacoma-2024.md");
        let far = levenshtein(target, "voron-trident.md");
        assert!(
            close < far,
            "expected close ({close}) < far ({far}) for target '{target}'"
        );
    }

    #[test]
    fn levenshtein_unicode_uses_chars_not_bytes() {
        assert_eq!(levenshtein("café", "cafe"), 1);
    }

    // ------------------------------------------------------------------
    // Path resolution tests (pure logic, no filesystem required beyond /tmp)
    // ------------------------------------------------------------------

    #[test]
    fn path_traversal_detection() {
        let data_path = std::path::PathBuf::from("/tmp/test-kb-data");
        let traversal = data_path.join("../../../etc/passwd");
        assert!(
            traversal.starts_with(&data_path),
            "raw join with .. still starts_with data_path — canonicalize() is needed"
        );
    }

    #[test]
    fn absolute_path_outside_data_rejected() {
        let data_path = std::path::PathBuf::from("/tmp/test-kb-data");
        let outside = std::path::PathBuf::from("/etc/passwd");
        assert!(
            !outside.starts_with(&data_path),
            "/etc/passwd should not start_with /tmp/test-kb-data"
        );
    }

    #[test]
    fn relative_path_inside_data_accepted() {
        let data_path = std::path::PathBuf::from("/tmp/test-kb-data");
        let inside = data_path.join("docs/guide.md");
        assert!(
            inside.starts_with(&data_path),
            "data_path/docs/guide.md should start_with data_path"
        );
    }

    #[test]
    fn canonicalize_nonexistent_file_produces_not_found_message() {
        let bad_path = std::path::PathBuf::from("/tmp/nonexistent-kb-test-dir/missing.md");
        let err = bad_path
            .canonicalize()
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => format!("File not found: {}", bad_path.display()),
                std::io::ErrorKind::PermissionDenied => {
                    format!("Permission denied: {}", bad_path.display())
                }
                _ => format!("Cannot access file '{}': {}", bad_path.display(), e),
            })
            .unwrap_err();
        assert!(
            err.contains("File not found"),
            "expected 'File not found', got: {err}"
        );
    }

    #[test]
    fn get_document_uses_canonical_path() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("test.md"), "# Hello").unwrap();

        let canonical_data = tmp.path().canonicalize().unwrap();

        let requested = PathBuf::from("docs/test.md");
        let resolved = canonical_data.join(&requested);
        let canonical_resolved = resolved.canonicalize().unwrap();

        assert!(
            canonical_resolved.starts_with(&canonical_data),
            "resolved path should be under the canonical data path"
        );
    }

    #[test]
    fn canonicalize_error_message_includes_path() {
        let bad_path = std::path::PathBuf::from("/tmp/nonexistent-kb-test-dir/missing.md");
        let err = bad_path
            .canonicalize()
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => format!("File not found: {}", bad_path.display()),
                std::io::ErrorKind::PermissionDenied => {
                    format!("Permission denied: {}", bad_path.display())
                }
                _ => format!("Cannot access file '{}': {}", bad_path.display(), e),
            })
            .unwrap_err();
        assert!(
            err.contains(&bad_path.display().to_string()),
            "error message should include the file path, got: {err}"
        );
    }

    // ------------------------------------------------------------------
    // min_score post-filter tests (pure, no network)
    // ------------------------------------------------------------------

    fn make_search_result(score: f32) -> SearchResult {
        SearchResult {
            score,
            pre_rerank_score: None,
            dense_score: None,
            sparse_score: None,
            payload: HashMap::new(),
        }
    }

    #[test]
    fn min_score_drops_below_floor_results() {
        let mut results = vec![
            make_search_result(0.9),
            make_search_result(0.5),
            make_search_result(0.3),
            make_search_result(0.7),
        ];
        let floor = 0.6f32;
        results.retain(|r| r.score >= floor);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.score >= floor));
    }

    #[test]
    fn min_score_none_is_noop() {
        let results = vec![
            make_search_result(0.9),
            make_search_result(0.1),
            make_search_result(0.5),
        ];
        let min_score: Option<f32> = None;
        let mut filtered = results.clone();
        if let Some(s) = min_score {
            filtered.retain(|r| r.score >= s);
        }
        assert_eq!(filtered.len(), 3);
    }

    // ------------------------------------------------------------------
    // diversity cap tests (pure, no network) — issue #86
    // ------------------------------------------------------------------

    fn make_result_for(file_path: &str, score: f32) -> SearchResult {
        let mut r = make_search_result(score);
        r.payload
            .insert("file_path".into(), serde_json::json!(file_path));
        r
    }

    fn file_paths_of(results: &[SearchResult]) -> Vec<&str> {
        results
            .iter()
            .map(|r| r.payload.get("file_path").and_then(|v| v.as_str()).unwrap())
            .collect()
    }

    #[test]
    fn apply_diversity_cap_limits_one_document_monopolizing() {
        // 5 chunks all from doc A, ranked best-first, plus 2 from doc B behind them.
        let results = vec![
            make_result_for("a.md", 0.9),
            make_result_for("a.md", 0.85),
            make_result_for("a.md", 0.8),
            make_result_for("a.md", 0.75),
            make_result_for("a.md", 0.7),
            make_result_for("b.md", 0.65),
            make_result_for("b.md", 0.6),
        ];
        let capped = apply_diversity_cap(results, 2);
        let a_count = file_paths_of(&capped)
            .into_iter()
            .filter(|p| *p == "a.md")
            .count();
        assert_eq!(a_count, 2, "doc A must not exceed the cap");
        assert_eq!(
            file_paths_of(&capped),
            vec!["a.md", "a.md", "b.md", "b.md"],
            "doc B's chunks must backfill the slots doc A lost, preserving relative order"
        );
    }

    #[test]
    fn apply_diversity_cap_preserves_legitimate_multi_chunk_relevance() {
        // 3 chunks from the same document, all genuinely top-ranked (e.g. three
        // sections of one guide that all answer the query) — a cap >= 3 must keep
        // all of them, not collapse to a single result.
        let results = vec![
            make_result_for("guide.md", 0.95),
            make_result_for("guide.md", 0.9),
            make_result_for("guide.md", 0.85),
            make_result_for("other.md", 0.5),
        ];
        let capped = apply_diversity_cap(results, 3);
        assert_eq!(
            file_paths_of(&capped),
            vec!["guide.md", "guide.md", "guide.md", "other.md"],
            "a cap >= actual count must not drop any legitimately top-ranked chunk"
        );
    }

    #[test]
    fn apply_diversity_cap_missing_file_path_is_never_capped() {
        let mut no_path = make_search_result(0.5);
        no_path.payload.remove("file_path");
        let results = vec![
            make_result_for("a.md", 0.9),
            make_result_for("a.md", 0.8),
            no_path,
        ];
        let capped = apply_diversity_cap(results, 1);
        assert_eq!(
            capped.len(),
            2,
            "a.md capped to 1, plus the un-groupable result always kept"
        );
    }

    #[test]
    fn cap_reranked_by_document_backfills_like_apply_diversity_cap() {
        let results = vec![
            make_result_for("a.md", 0.5), // index 0
            make_result_for("a.md", 0.5), // index 1
            make_result_for("a.md", 0.5), // index 2
            make_result_for("b.md", 0.5), // index 3
        ];
        // Post-rerank order: a.md x3 ahead of b.md, matching a cross-encoder that
        // (correctly, in this synthetic case) still finds doc A most relevant.
        let indexed = vec![(0, 9.0), (1, 8.0), (2, 7.0), (3, 6.0)];
        let capped = cap_reranked_by_document(indexed, &results, 2);
        assert_eq!(
            capped,
            vec![(0, 9.0), (1, 8.0), (3, 6.0)],
            "third a.md candidate dropped, b.md candidate survives in rank order"
        );
    }

    #[test]
    fn resolve_within_data_enforces_boundaries() {
        let tmp = tempfile::tempdir().unwrap();
        let data_path = tmp.path().canonicalize().unwrap();

        let docs = data_path.join("docs");
        std::fs::create_dir_all(&docs).unwrap();
        std::fs::write(docs.join("note.md"), "# Note").unwrap();
        std::fs::write(docs.join("note.txt"), "plain text").unwrap();

        let mut builder = globset::GlobSetBuilder::new();
        builder.add(globset::Glob::new("**/*.md").unwrap());
        let include = builder.build().unwrap();

        // Valid relative path resolves Ok.
        assert!(
            resolve_within_data("docs/note.md", &data_path, &include).is_ok(),
            "valid relative path should resolve Ok"
        );

        // Path traversal escaping the data dir returns Err.
        let result = resolve_within_data("../../../etc/passwd", &data_path, &include);
        assert!(result.is_err(), "traversal should return Err");

        // Absolute path to a real file outside the tempdir returns Outside.
        // /etc/hosts exists on both macOS and Linux.
        let outside = "/etc/hosts";
        if std::path::Path::new(outside).exists() {
            let result = resolve_within_data(outside, &data_path, &include);
            assert!(
                matches!(result, Err(ResolveErr::Outside)),
                "absolute path outside data dir should return Outside"
            );
        }

        // Non-.md file inside returns NotPermitted.
        let txt_path = docs.join("note.txt").to_string_lossy().to_string();
        assert!(
            matches!(
                resolve_within_data(&txt_path, &data_path, &include),
                Err(ResolveErr::NotPermitted)
            ),
            "non-.md file should return NotPermitted"
        );
    }

    // ------------------------------------------------------------------
    // Mock types for retrieval unit tests
    // ------------------------------------------------------------------

    struct MockEmbedder {
        err: Option<String>,
        ok: Vec<f32>,
    }

    impl MockEmbedder {
        fn ok(v: Vec<f32>) -> Self {
            Self { err: None, ok: v }
        }
        fn err(msg: &str) -> Self {
            Self {
                err: Some(msg.to_string()),
                ok: vec![],
            }
        }
    }

    impl QueryEmbedder for MockEmbedder {
        async fn embed_query(&self, _q: &str) -> anyhow::Result<Vec<f32>> {
            if let Some(ref msg) = self.err {
                anyhow::bail!("{}", msg);
            }
            Ok(self.ok.clone())
        }
    }

    struct MockRetrievalStore {
        search_err: Option<String>,
        search_ok: Vec<SearchResult>,
        received_filters: std::sync::Mutex<Option<HashMap<String, serde_json::Value>>>,
        /// Sparse vector captured by `hybrid_search` (None until called).
        received_sparse: std::sync::Mutex<Option<(Vec<u32>, Vec<f32>)>>,
        /// Which method was last invoked: "search" or "hybrid_search".
        last_call: std::sync::Mutex<Option<&'static str>>,
    }

    impl MockRetrievalStore {
        fn with_results(results: Vec<SearchResult>) -> Self {
            Self {
                search_err: None,
                search_ok: results,
                received_filters: std::sync::Mutex::new(None),
                received_sparse: std::sync::Mutex::new(None),
                last_call: std::sync::Mutex::new(None),
            }
        }
        fn with_search_err(msg: &str) -> Self {
            Self {
                search_err: Some(msg.to_string()),
                search_ok: Vec::new(),
                received_filters: std::sync::Mutex::new(None),
                received_sparse: std::sync::Mutex::new(None),
                last_call: std::sync::Mutex::new(None),
            }
        }
    }

    impl RetrievalStore for MockRetrievalStore {
        async fn search(
            &self,
            _collection: &str,
            _vector: Vec<f32>,
            filters: HashMap<String, serde_json::Value>,
            _limit: u64,
        ) -> anyhow::Result<Vec<SearchResult>> {
            *self.last_call.lock().unwrap() = Some("search");
            *self.received_filters.lock().unwrap() = Some(filters);
            if let Some(ref msg) = self.search_err {
                anyhow::bail!("{}", msg);
            }
            Ok(self.search_ok.clone())
        }

        async fn hybrid_search(
            &self,
            _collection: &str,
            _dense: Vec<f32>,
            sparse: (Vec<u32>, Vec<f32>),
            filters: HashMap<String, serde_json::Value>,
            _limit: u64,
            _rrf_candidates: u64,
            _explain: bool,
        ) -> anyhow::Result<Vec<SearchResult>> {
            *self.last_call.lock().unwrap() = Some("hybrid_search");
            *self.received_filters.lock().unwrap() = Some(filters);
            *self.received_sparse.lock().unwrap() = Some(sparse);
            if let Some(ref msg) = self.search_err {
                anyhow::bail!("{}", msg);
            }
            Ok(self.search_ok.clone())
        }
    }

    /// `DocumentIndex` mock for `get_document`'s fuzzy-fallback tests. Only
    /// `all_paths` is on that code path — `query_documents` backs `list_documents`
    /// instead and is unused here, so it stubs out rather than duplicating
    /// `state.rs`'s real query logic.
    struct MockDocumentIndex {
        paths: Vec<String>,
        err: Option<String>,
    }

    impl MockDocumentIndex {
        fn with_paths(paths: Vec<String>) -> Self {
            Self { paths, err: None }
        }
        fn with_err(msg: &str) -> Self {
            Self {
                paths: Vec::new(),
                err: Some(msg.to_string()),
            }
        }
    }

    impl DocumentIndex for MockDocumentIndex {
        async fn query_documents(
            &self,
            _query: &DocumentQuery,
        ) -> anyhow::Result<DocumentQueryResult> {
            unimplemented!("get_document's fuzzy fallback does not call query_documents")
        }

        async fn all_paths(&self) -> anyhow::Result<Vec<String>> {
            if let Some(ref msg) = self.err {
                anyhow::bail!("{}", msg);
            }
            Ok(self.paths.clone())
        }
    }

    fn make_md_globset() -> GlobSet {
        let mut builder = globset::GlobSetBuilder::new();
        builder.add(globset::Glob::new("**/*.md").unwrap());
        builder.build().unwrap()
    }

    fn make_deps<'a, E: QueryEmbedder, Q: RetrievalStore>(
        embed: &'a E,
        qdrant: &'a Q,
        data_path: &'a Path,
        include_patterns: &'a GlobSet,
    ) -> RetrievalDeps<'a, E, Q> {
        RetrievalDeps {
            embed_client: embed,
            qdrant,
            collection: "test-col",
            data_path,
            include_patterns,
            reranker: None,
        }
    }

    fn default_opts() -> SearchOptions {
        SearchOptions {
            limit: 10,
            min_score: None,
            hybrid: false,
            rrf_candidates: 50,
            explain: false,
            modified_after: None,
            modified_before: None,
            rerank_candidate_limit: None,
            // Off by default in this shared test fixture, even though the shipped
            // config default is `Some(3)` — existing tests built on `default_opts()`
            // predate diversity and assert on undiversified result sets. The
            // diversity-specific tests below opt in explicitly instead.
            diversity_max_per_document: None,
        }
    }

    // ------------------------------------------------------------------
    // search() unit tests with mocks
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn search_filters_passed_through() {
        let embed = MockEmbedder::ok(vec![0.1, 0.2, 0.3]);
        let store = MockRetrievalStore::with_results(vec![]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let filters = SearchFilters {
            domain: Some("engineering".to_string()),
            r#type: Some("guide".to_string()),
            tags: Some(vec!["rust".to_string(), "rag".to_string()]),
        };

        let _ = search(&deps, "query", &filters, &default_opts())
            .await
            .unwrap();

        let received = store.received_filters.lock().unwrap().clone().unwrap();
        assert_eq!(
            received.get("domain").and_then(|v| v.as_str()),
            Some("engineering"),
            "domain filter should be passed through"
        );
        assert_eq!(
            received.get("type").and_then(|v| v.as_str()),
            Some("guide"),
            "type filter should be passed through"
        );
        let tags = received
            .get("tags")
            .and_then(|v| v.as_array())
            .expect("tags should be an array");
        assert_eq!(tags.len(), 2);
        assert!(tags.iter().any(|t| t.as_str() == Some("rust")));
        assert!(tags.iter().any(|t| t.as_str() == Some("rag")));
    }

    #[tokio::test]
    async fn search_empty_tags_not_sent_as_filter() {
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(vec![]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let filters = SearchFilters {
            domain: None,
            r#type: None,
            tags: Some(vec![]), // empty — should not be sent
        };

        let _ = search(&deps, "q", &filters, &default_opts()).await.unwrap();

        let received = store.received_filters.lock().unwrap().clone().unwrap();
        assert!(
            !received.contains_key("tags"),
            "empty tags list should not produce a filter key"
        );
    }

    #[tokio::test]
    async fn search_embed_failure_returns_embed_error() {
        let embed = MockEmbedder::err("embedding service down");
        let store = MockRetrievalStore::with_results(vec![]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let result = search(
            &deps,
            "query",
            &SearchFilters {
                domain: None,
                r#type: None,
                tags: None,
            },
            &default_opts(),
        )
        .await;
        assert!(
            matches!(result, Err(SearchError::Embed(_))),
            "embed failure should map to SearchError::Embed"
        );
    }

    #[tokio::test]
    async fn search_store_failure_returns_search_error() {
        let embed = MockEmbedder::ok(vec![0.1, 0.2]);
        let store = MockRetrievalStore::with_search_err("qdrant unavailable");
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let result = search(
            &deps,
            "query",
            &SearchFilters {
                domain: None,
                r#type: None,
                tags: None,
            },
            &default_opts(),
        )
        .await;
        assert!(
            matches!(result, Err(SearchError::Search(_))),
            "store failure should map to SearchError::Search"
        );
    }

    #[tokio::test]
    async fn search_min_score_drops_below_floor() {
        let results = vec![
            make_search_result(0.9),
            make_search_result(0.5),
            make_search_result(0.3),
        ];
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(results);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let opts = SearchOptions {
            min_score: Some(0.6),
            ..default_opts()
        };
        let returned = search(
            &deps,
            "q",
            &SearchFilters {
                domain: None,
                r#type: None,
                tags: None,
            },
            &opts,
        )
        .await
        .unwrap();
        assert_eq!(returned.len(), 1);
        assert_eq!(returned[0].score, 0.9);
    }

    #[tokio::test]
    async fn search_min_score_none_keeps_all() {
        let results = vec![
            make_search_result(0.9),
            make_search_result(0.1),
            make_search_result(0.5),
        ];
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(results);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let returned = search(
            &deps,
            "q",
            &SearchFilters {
                domain: None,
                r#type: None,
                tags: None,
            },
            &default_opts(),
        )
        .await
        .unwrap();
        assert_eq!(returned.len(), 3, "None min_score should keep all results");
    }

    #[tokio::test]
    async fn search_empty_results_returns_empty_vec() {
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(vec![]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let returned = search(
            &deps,
            "q",
            &SearchFilters {
                domain: None,
                r#type: None,
                tags: None,
            },
            &default_opts(),
        )
        .await
        .unwrap();
        assert!(returned.is_empty());
    }

    // ------------------------------------------------------------------
    // diversity cap, end-to-end through search() — issue #86
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn search_diversity_caps_one_document_without_reranker() {
        // One document supplies every candidate but two; without a cap it would
        // occupy the entire result page and crowd the other document out.
        let results = vec![
            make_result_for("prolific.md", 0.99),
            make_result_for("prolific.md", 0.98),
            make_result_for("prolific.md", 0.97),
            make_result_for("prolific.md", 0.96),
            make_result_for("prolific.md", 0.95),
            make_result_for("quiet.md", 0.5),
        ];
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(results);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let opts = SearchOptions {
            limit: 3,
            diversity_max_per_document: Some(2),
            ..default_opts()
        };
        let returned = search(
            &deps,
            "q",
            &SearchFilters {
                domain: None,
                r#type: None,
                tags: None,
            },
            &opts,
        )
        .await
        .unwrap();

        assert_eq!(
            file_paths_of(&returned),
            vec!["prolific.md", "prolific.md", "quiet.md"],
            "the cap must free a slot for the other document instead of \
             letting prolific.md fill the whole 3-result page"
        );
    }

    #[tokio::test]
    async fn search_diversity_disabled_lets_one_document_monopolize() {
        // Same corpus as above, but the knob is off (None) — this is the
        // regression guard for the disable path: results must revert to the
        // pre-#86 behaviour where one document can fill every slot.
        let results = vec![
            make_result_for("prolific.md", 0.99),
            make_result_for("prolific.md", 0.98),
            make_result_for("prolific.md", 0.97),
            make_result_for("quiet.md", 0.5),
        ];
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(results);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let opts = SearchOptions {
            limit: 3,
            diversity_max_per_document: None,
            ..default_opts()
        };
        let returned = search(
            &deps,
            "q",
            &SearchFilters {
                domain: None,
                r#type: None,
                tags: None,
            },
            &opts,
        )
        .await
        .unwrap();

        assert_eq!(
            file_paths_of(&returned),
            vec!["prolific.md", "prolific.md", "prolific.md"],
            "diversity_max_per_document: None must disable capping entirely"
        );
    }

    #[tokio::test]
    async fn search_diversity_preserves_line_ranges() {
        // The cap must not lose or reorder the line_start/line_end payload
        // fields callers use to locate the relevant region within a document.
        let mut r0 = make_result_for("a.md", 0.9);
        r0.payload
            .insert("line_start".into(), serde_json::json!(10));
        r0.payload.insert("line_end".into(), serde_json::json!(20));
        let mut r1 = make_result_for("a.md", 0.8);
        r1.payload
            .insert("line_start".into(), serde_json::json!(30));
        r1.payload.insert("line_end".into(), serde_json::json!(40));
        let mut r2 = make_result_for("a.md", 0.7);
        r2.payload
            .insert("line_start".into(), serde_json::json!(50));
        r2.payload.insert("line_end".into(), serde_json::json!(60));

        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(vec![r0, r1, r2]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let opts = SearchOptions {
            limit: 10,
            diversity_max_per_document: Some(2),
            ..default_opts()
        };
        let returned = search(
            &deps,
            "q",
            &SearchFilters {
                domain: None,
                r#type: None,
                tags: None,
            },
            &opts,
        )
        .await
        .unwrap();

        assert_eq!(returned.len(), 2, "capped to 2 of the 3 a.md chunks");
        let get_lines = |r: &SearchResult| {
            (
                r.payload.get("line_start").and_then(|v| v.as_i64()),
                r.payload.get("line_end").and_then(|v| v.as_i64()),
            )
        };
        assert_eq!(
            get_lines(&returned[0]),
            (Some(10), Some(20)),
            "surviving chunk's own line range must be untouched by capping"
        );
        assert_eq!(
            get_lines(&returned[1]),
            (Some(30), Some(40)),
            "surviving chunk's own line range must be untouched by capping"
        );
    }

    // ------------------------------------------------------------------
    // hybrid search routing tests
    // ------------------------------------------------------------------

    fn hybrid_opts() -> SearchOptions {
        SearchOptions {
            hybrid: true,
            ..default_opts()
        }
    }

    #[tokio::test]
    async fn search_hybrid_routes_to_hybrid_search() {
        let embed = MockEmbedder::ok(vec![0.1, 0.2, 0.3]);
        let store = MockRetrievalStore::with_results(vec![make_search_result(0.9)]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let filters = SearchFilters {
            domain: Some("sysadmin".to_string()),
            r#type: None,
            tags: None,
        };

        let query = "node:ares state.db";
        let _ = search(&deps, query, &filters, &hybrid_opts())
            .await
            .unwrap();

        // Routed to the hybrid arm, not the dense-only one.
        assert_eq!(
            *store.last_call.lock().unwrap(),
            Some("hybrid_search"),
            "hybrid=true must route to hybrid_search"
        );

        // The exact query text was tokenized into a non-empty sparse vector and
        // matches what the sparse module would produce for that query.
        let captured = store.received_sparse.lock().unwrap().clone().unwrap();
        assert!(
            !captured.0.is_empty(),
            "sparse query vector must be non-empty"
        );
        let expected = crate::sparse::tokenize(query);
        let captured_set: std::collections::HashSet<u32> = captured.0.iter().copied().collect();
        let expected_set: std::collections::HashSet<u32> = expected.0.iter().copied().collect();
        assert_eq!(
            captured_set, expected_set,
            "sparse vector must come from the raw query"
        );

        // Filters carried into the hybrid path.
        let received = store.received_filters.lock().unwrap().clone().unwrap();
        assert_eq!(
            received.get("domain").and_then(|v| v.as_str()),
            Some("sysadmin"),
            "filters must be carried into hybrid_search"
        );
    }

    #[tokio::test]
    async fn search_dense_routes_to_plain_search() {
        let embed = MockEmbedder::ok(vec![0.1, 0.2, 0.3]);
        let store = MockRetrievalStore::with_results(vec![]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let _ = search(
            &deps,
            "anything",
            &SearchFilters {
                domain: None,
                r#type: None,
                tags: None,
            },
            &default_opts(), // hybrid: false
        )
        .await
        .unwrap();

        assert_eq!(
            *store.last_call.lock().unwrap(),
            Some("search"),
            "hybrid=false must route to the dense-only search"
        );
        assert!(
            store.received_sparse.lock().unwrap().is_none(),
            "dense-only path must not compute a sparse vector"
        );
    }

    #[tokio::test]
    async fn search_hybrid_surfaces_store_results() {
        // Acceptance (mock form): a keyword-only query that dense search alone would
        // rank poorly. The hybrid arm (RRF over dense+sparse, server-side in Qdrant)
        // is modeled here by the store returning the correct chunk first; we assert
        // the hybrid path returns it in position 1.
        let mut right_chunk = make_search_result(0.95);
        right_chunk
            .payload
            .insert("file_path".into(), serde_json::json!("sysadmin/ares.md"));
        let store = MockRetrievalStore::with_results(vec![right_chunk, make_search_result(0.40)]);
        let embed = MockEmbedder::ok(vec![0.1, 0.2, 0.3]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let results = search(
            &deps,
            "ares",
            &SearchFilters {
                domain: None,
                r#type: None,
                tags: None,
            },
            &hybrid_opts(),
        )
        .await
        .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].payload.get("file_path").and_then(|v| v.as_str()),
            Some("sysadmin/ares.md"),
            "hybrid result should surface the keyword chunk first"
        );
    }

    // ------------------------------------------------------------------
    // get_document() unit tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn get_document_literal_path_valid() {
        let tmp = tempfile::tempdir().unwrap();
        let data_path = tmp.path().canonicalize().unwrap();
        let docs = data_path.join("docs");
        std::fs::create_dir_all(&docs).unwrap();
        std::fs::write(docs.join("guide.md"), "# Guide\nContent here.").unwrap();

        let gs = make_md_globset();
        let embed = MockEmbedder::ok(vec![]);
        let store = MockRetrievalStore::with_results(vec![]);
        let deps = make_deps(&embed, &store, &data_path, &gs);
        let index = MockDocumentIndex::with_paths(vec![]);

        let doc = get_document(&deps, &index, "docs/guide.md").await.unwrap();
        assert!(doc.content.contains("Content here."));
    }

    #[tokio::test]
    async fn get_document_literal_path_outside_returns_outside() {
        let tmp = tempfile::tempdir().unwrap();
        let data_path = tmp.path().canonicalize().unwrap();
        let gs = make_md_globset();
        let embed = MockEmbedder::ok(vec![]);
        let store = MockRetrievalStore::with_results(vec![]);
        let deps = make_deps(&embed, &store, &data_path, &gs);
        let index = MockDocumentIndex::with_paths(vec![]);

        // /etc/hosts is a real file outside any tempdir
        if std::path::Path::new("/etc/hosts").exists() {
            let result = get_document(&deps, &index, "/etc/hosts").await;
            assert!(
                matches!(result, Err(GetDocumentError::Outside)),
                "absolute path outside data dir should return Outside"
            );
        }
    }

    #[tokio::test]
    async fn get_document_literal_path_not_permitted() {
        let tmp = tempfile::tempdir().unwrap();
        let data_path = tmp.path().canonicalize().unwrap();
        std::fs::write(data_path.join("notes.txt"), "plain text").unwrap();

        let gs = make_md_globset(); // only *.md
        let embed = MockEmbedder::ok(vec![]);
        let store = MockRetrievalStore::with_results(vec![]);
        let deps = make_deps(&embed, &store, &data_path, &gs);
        let index = MockDocumentIndex::with_paths(vec![]);

        let txt_path = data_path.join("notes.txt").to_string_lossy().to_string();
        let result = get_document(&deps, &index, &txt_path).await;
        assert!(
            matches!(result, Err(GetDocumentError::NotPermitted)),
            "non-.md file should return NotPermitted"
        );
    }

    #[tokio::test]
    async fn get_document_fuzzy_unique_basename_match() {
        let tmp = tempfile::tempdir().unwrap();
        let data_path = tmp.path().canonicalize().unwrap();
        let notes = data_path.join("notes");
        std::fs::create_dir_all(&notes).unwrap();
        std::fs::write(notes.join("foo.md"), "# Foo\nFuzzy content.").unwrap();

        let gs = make_md_globset();
        let embed = MockEmbedder::ok(vec![]);
        let store = MockRetrievalStore::with_results(vec![]);
        let deps = make_deps(&embed, &store, &data_path, &gs);
        // The index returns the relative key — literal "foo.md" won't resolve, falls to fuzzy
        let index = MockDocumentIndex::with_paths(vec!["notes/foo.md".to_string()]);

        let doc = get_document(&deps, &index, "foo.md").await.unwrap();
        assert!(doc.content.contains("Fuzzy content."));
    }

    #[tokio::test]
    async fn get_document_fuzzy_zero_matches_not_found_with_suggestions() {
        // No real filesystem needed — fuzzy 0-match path only does Levenshtein + relative_to_data.
        // Use a fake data_path; literal resolve will fail with NotFound (dir doesn't exist).
        let tmp = tempfile::tempdir().unwrap();
        let data_path = tmp.path().canonicalize().unwrap();

        // Relative keys as stored in the documents table
        let indexed_paths = vec!["tacoma-2024.md".to_string(), "voron-trident.md".to_string()];

        let gs = make_md_globset();
        let embed = MockEmbedder::ok(vec![]);
        let store = MockRetrievalStore::with_results(vec![]);
        let deps = make_deps(&embed, &store, &data_path, &gs);
        let index = MockDocumentIndex::with_paths(indexed_paths);

        // "tacoma-2025.md" doesn't exist anywhere — literal resolve: NotFound, fuzzy: 0 exact basename matches
        let result = get_document(&deps, &index, "tacoma-2025.md").await;
        match result {
            Err(GetDocumentError::NotFound { suggestions }) => {
                assert!(!suggestions.is_empty(), "should have suggestions");
                // tacoma-2024.md should rank closer than voron-trident.md
                let first = &suggestions[0];
                assert!(
                    first.contains("tacoma-2024"),
                    "closest match should be tacoma-2024, got: {first}"
                );
            }
            other => panic!(
                "Expected NotFound, got: {}",
                match other {
                    Ok(_) => "Ok",
                    Err(GetDocumentError::Outside) => "Outside",
                    Err(GetDocumentError::NotPermitted) => "NotPermitted",
                    Err(GetDocumentError::Ambiguous { .. }) => "Ambiguous",
                    Err(GetDocumentError::Io(_)) => "Io",
                    _ => "other",
                }
            ),
        }
    }

    #[tokio::test]
    async fn get_document_fuzzy_ambiguous_multiple_basename_matches() {
        let tmp = tempfile::tempdir().unwrap();
        let data_path = tmp.path().canonicalize().unwrap();

        // Two relative keys with the same basename
        let indexed_paths = vec!["a/notes.md".to_string(), "b/notes.md".to_string()];

        let gs = make_md_globset();
        let embed = MockEmbedder::ok(vec![]);
        let store = MockRetrievalStore::with_results(vec![]);
        let deps = make_deps(&embed, &store, &data_path, &gs);
        let index = MockDocumentIndex::with_paths(indexed_paths);

        // "notes.md" as basename — not on disk, so literal fails; fuzzy finds 2 exact matches
        let result = get_document(&deps, &index, "notes.md").await;
        match result {
            Err(GetDocumentError::Ambiguous { matches }) => {
                assert_eq!(matches.len(), 2, "should have 2 ambiguous matches");
            }
            other => panic!(
                "Expected Ambiguous, got: {}",
                match other {
                    Ok(_) => "Ok",
                    Err(GetDocumentError::Outside) => "Outside",
                    Err(GetDocumentError::NotPermitted) => "NotPermitted",
                    Err(GetDocumentError::NotFound { .. }) => "NotFound",
                    Err(GetDocumentError::Io(_)) => "Io",
                    _ => "other",
                }
            ),
        }
    }

    #[tokio::test]
    async fn get_document_fuzzy_finds_a_match_past_the_old_ten_thousand_cap() {
        // Regression guard for #87: the Qdrant-facet fetch this replaced capped at
        // MAX_INDEXED_PATHS_FOR_FUZZY (10,000) and silently dropped anything past it,
        // so a document indexed "late" in facet order could never be fuzzy-matched.
        // `documents` is queried directly with no such cap — assert a basename well
        // past the old boundary still resolves.
        let tmp = tempfile::tempdir().unwrap();
        let data_path = tmp.path().canonicalize().unwrap();
        let target_dir = data_path.join("late");
        std::fs::create_dir_all(&target_dir).unwrap();
        std::fs::write(target_dir.join("needle.md"), "# Needle\nFound it.").unwrap();

        let mut paths: Vec<String> = (0..10_500).map(|i| format!("filler/{i}.md")).collect();
        paths.push("late/needle.md".to_string());

        let gs = make_md_globset();
        let embed = MockEmbedder::ok(vec![]);
        let store = MockRetrievalStore::with_results(vec![]);
        let deps = make_deps(&embed, &store, &data_path, &gs);
        let index = MockDocumentIndex::with_paths(paths);

        let doc = get_document(&deps, &index, "needle.md")
            .await
            .expect("a basename past the old 10k cap must still resolve");
        assert!(doc.content.contains("Found it."));
    }

    #[tokio::test]
    async fn get_document_fuzzy_falls_back_gracefully_when_the_document_index_errors() {
        // Regression guard for the store swap (#87): a query failure against
        // `documents` must degrade to "not found, no suggestions" — same as an empty
        // index — rather than propagating the error or panicking. Mirrors the old
        // Qdrant-facet-fetch-failure behavior it replaces.
        let tmp = tempfile::tempdir().unwrap();
        let data_path = tmp.path().canonicalize().unwrap();

        let gs = make_md_globset();
        let embed = MockEmbedder::ok(vec![]);
        let store = MockRetrievalStore::with_results(vec![]);
        let deps = make_deps(&embed, &store, &data_path, &gs);
        let index = MockDocumentIndex::with_err("database is locked");

        let result = get_document(&deps, &index, "missing.md").await;
        match result {
            Err(GetDocumentError::NotFound { suggestions }) => {
                assert!(
                    suggestions.is_empty(),
                    "an unusable index has nothing to suggest from"
                );
            }
            other => panic!(
                "Expected NotFound, got: {}",
                match other {
                    Ok(_) => "Ok",
                    Err(GetDocumentError::Outside) => "Outside",
                    Err(GetDocumentError::NotPermitted) => "NotPermitted",
                    Err(GetDocumentError::Ambiguous { .. }) => "Ambiguous",
                    Err(GetDocumentError::Io(_)) => "Io",
                    _ => "other",
                }
            ),
        }
    }

    // ------------------------------------------------------------------
    // reranker tests
    // ------------------------------------------------------------------

    struct MockReranker {
        fail: bool,
    }

    impl crate::rerank::Reranker for MockReranker {
        fn rerank<'a>(
            &'a self,
            _query: &'a str,
            documents: &'a [&'a str],
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = anyhow::Result<Vec<crate::rerank::RerankResult>>>
                    + Send
                    + 'a,
            >,
        > {
            let fail = self.fail;
            let n = documents.len();
            Box::pin(async move {
                if fail {
                    anyhow::bail!("down");
                }
                // Reverse the order: highest score to lowest-indexed document
                Ok((0..n)
                    .rev()
                    .enumerate()
                    .map(|(rank, i)| crate::rerank::RerankResult {
                        index: i,
                        relevance_score: (n - rank) as f32,
                    })
                    .collect())
            })
        }
    }

    #[tokio::test]
    async fn reranker_reorders_results() {
        let mut r0 = make_search_result(0.9);
        r0.payload
            .insert("content".into(), serde_json::json!("doc A"));
        let mut r1 = make_search_result(0.8);
        r1.payload
            .insert("content".into(), serde_json::json!("doc B"));
        let mut r2 = make_search_result(0.7);
        r2.payload
            .insert("content".into(), serde_json::json!("doc C"));

        let store = MockRetrievalStore::with_results(vec![r0, r1, r2]);
        let embed = MockEmbedder::ok(vec![0.1]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let reranker = MockReranker { fail: false };
        let deps = RetrievalDeps {
            embed_client: &embed,
            qdrant: &store,
            collection: "test-col",
            data_path,
            include_patterns: &gs,
            reranker: Some(&reranker as &(dyn crate::rerank::Reranker + Send + Sync)),
        };

        let opts = SearchOptions {
            limit: 3,
            rerank_candidate_limit: None,
            ..default_opts()
        };

        let results = search(
            &deps,
            "q",
            &SearchFilters {
                domain: None,
                r#type: None,
                tags: None,
            },
            &opts,
        )
        .await
        .unwrap();

        // MockReranker reverses order: last doc gets highest score
        assert_eq!(results.len(), 3);
        // The last original result (index 2, doc C) should now be first
        assert_eq!(
            results[0].payload.get("content").and_then(|v| v.as_str()),
            Some("doc C"),
            "reranked order should put last doc first"
        );
    }

    #[tokio::test]
    async fn reranker_failure_returns_fused_order() {
        let mut r0 = make_search_result(0.9);
        r0.payload
            .insert("content".into(), serde_json::json!("doc A"));
        let mut r1 = make_search_result(0.8);
        r1.payload
            .insert("content".into(), serde_json::json!("doc B"));

        let store = MockRetrievalStore::with_results(vec![r0, r1]);
        let embed = MockEmbedder::ok(vec![0.1]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let reranker = MockReranker { fail: true };
        let deps = RetrievalDeps {
            embed_client: &embed,
            qdrant: &store,
            collection: "test-col",
            data_path,
            include_patterns: &gs,
            reranker: Some(&reranker as &(dyn crate::rerank::Reranker + Send + Sync)),
        };

        let opts = SearchOptions {
            limit: 2,
            rerank_candidate_limit: None,
            ..default_opts()
        };

        let results = search(
            &deps,
            "q",
            &SearchFilters {
                domain: None,
                r#type: None,
                tags: None,
            },
            &opts,
        )
        .await
        .unwrap();

        // Should still return results (fail-soft), in original fused order
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].payload.get("content").and_then(|v| v.as_str()),
            Some("doc A"),
            "on reranker failure, should return fused order"
        );
    }

    #[tokio::test]
    async fn search_diversity_backfills_after_reranking() {
        // Pre-rerank (fused) order: b.md ranks best, the 4 a.md chunks trail it.
        // MockReranker fully reverses whatever order it receives, so the
        // cross-encoder ends up judging all 4 a.md chunks more relevant than
        // b.md's, pushing them to the top of the REranked order. The final cap
        // must still hold against that reranked order: only 2 of a.md's chunks
        // may survive, and b.md backfills the freed slot rather than the page
        // just shrinking to 2 results. This is the two-stage design's core
        // claim — the cap acts on the cross-encoder's ranking, not the weaker
        // pre-rerank signal.
        let mut b1 = make_search_result(0.5);
        b1.payload
            .insert("file_path".into(), serde_json::json!("b.md"));
        b1.payload.insert("content".into(), serde_json::json!("b1"));
        let make_a = |content: &str| {
            let mut r = make_search_result(0.4);
            r.payload
                .insert("file_path".into(), serde_json::json!("a.md"));
            r.payload
                .insert("content".into(), serde_json::json!(content));
            r
        };
        let store = MockRetrievalStore::with_results(vec![
            b1,
            make_a("a1"),
            make_a("a2"),
            make_a("a3"),
            make_a("a4"),
        ]);
        let embed = MockEmbedder::ok(vec![0.1]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let reranker = MockReranker { fail: false };
        let deps = RetrievalDeps {
            embed_client: &embed,
            qdrant: &store,
            collection: "test-col",
            data_path,
            include_patterns: &gs,
            reranker: Some(&reranker as &(dyn crate::rerank::Reranker + Send + Sync)),
        };

        let opts = SearchOptions {
            limit: 3,
            diversity_max_per_document: Some(2),
            rerank_candidate_limit: None,
            ..default_opts()
        };

        let results = search(
            &deps,
            "q",
            &SearchFilters {
                domain: None,
                r#type: None,
                tags: None,
            },
            &opts,
        )
        .await
        .unwrap();

        assert_eq!(
            file_paths_of(&results),
            vec!["a.md", "a.md", "b.md"],
            "post-rerank cap must trim a.md to 2 and backfill with b.md instead \
             of the page shrinking to 2 results"
        );
    }

    // ------------------------------------------------------------------
    // Phase 3 feature tests: mtime filter + explain
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn search_modified_after_adds_mtime_gte_filter() {
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(vec![]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let opts = SearchOptions {
            modified_after: Some(1_000_000),
            ..default_opts()
        };
        let _ = search(
            &deps,
            "q",
            &SearchFilters {
                domain: None,
                r#type: None,
                tags: None,
            },
            &opts,
        )
        .await
        .unwrap();

        let received = store.received_filters.lock().unwrap().clone().unwrap();
        assert!(
            received.contains_key("mtime__gte"),
            "modified_after should produce a mtime__gte key in the filter map"
        );
        assert_eq!(
            received.get("mtime__gte").and_then(|v| v.as_i64()),
            Some(1_000_000),
            "mtime__gte value should match modified_after"
        );
        assert!(
            !received.contains_key("mtime__lte"),
            "mtime__lte should not appear when only modified_after is set"
        );
    }

    #[tokio::test]
    async fn search_modified_before_adds_mtime_lte_filter() {
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(vec![]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let opts = SearchOptions {
            modified_before: Some(2_000_000),
            ..default_opts()
        };
        let _ = search(
            &deps,
            "q",
            &SearchFilters {
                domain: None,
                r#type: None,
                tags: None,
            },
            &opts,
        )
        .await
        .unwrap();

        let received = store.received_filters.lock().unwrap().clone().unwrap();
        assert!(
            received.contains_key("mtime__lte"),
            "modified_before should produce a mtime__lte key in the filter map"
        );
        assert_eq!(
            received.get("mtime__lte").and_then(|v| v.as_i64()),
            Some(2_000_000),
            "mtime__lte value should match modified_before"
        );
    }

    #[tokio::test]
    async fn search_explain_populates_pre_rerank_score() {
        let mut r0 = make_search_result(0.9);
        r0.payload
            .insert("content".into(), serde_json::json!("doc A"));
        let mut r1 = make_search_result(0.8);
        r1.payload
            .insert("content".into(), serde_json::json!("doc B"));

        let store = MockRetrievalStore::with_results(vec![r0, r1]);
        let embed = MockEmbedder::ok(vec![0.1]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let reranker = MockReranker { fail: false };
        let deps = RetrievalDeps {
            embed_client: &embed,
            qdrant: &store,
            collection: "test-col",
            data_path,
            include_patterns: &gs,
            reranker: Some(&reranker as &(dyn crate::rerank::Reranker + Send + Sync)),
        };

        let opts = SearchOptions {
            explain: true,
            rerank_candidate_limit: Some(10),
            ..default_opts()
        };
        let results = search(
            &deps,
            "q",
            &SearchFilters {
                domain: None,
                r#type: None,
                tags: None,
            },
            &opts,
        )
        .await
        .unwrap();

        assert!(
            !results.is_empty(),
            "should return results when explain is active"
        );
        for r in &results {
            assert!(
                r.pre_rerank_score.is_some(),
                "explain=true + reranker active should set pre_rerank_score on every result"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Line-range tests
    // -----------------------------------------------------------------------

    /// Three terminated lines. Kept short so the assertions read as the slice
    /// they describe rather than as index arithmetic.
    const DOC: &str = "one\ntwo\nthree\n";

    #[test]
    fn line_range_new_returns_none_when_neither_bound_is_given() {
        assert_eq!(LineRange::new(None, None).unwrap(), None);
    }

    #[test]
    fn line_range_new_defaults_a_lone_end_to_start_at_one() {
        let r = LineRange::new(None, Some(5)).unwrap().unwrap();
        assert_eq!(r.start, 1);
        assert_eq!(r.end, Some(5));
    }

    #[test]
    fn line_range_new_leaves_a_lone_start_open_ended() {
        let r = LineRange::new(Some(4), None).unwrap().unwrap();
        assert_eq!(r.start, 4);
        assert_eq!(r.end, None);
    }

    #[test]
    fn line_range_new_rejects_line_zero_on_either_bound() {
        assert_eq!(
            LineRange::new(Some(0), Some(3)),
            Err(LineRangeError::ZeroLine)
        );
        assert_eq!(
            LineRange::new(None, Some(0)),
            Err(LineRangeError::ZeroLine),
            "a lone end_line=0 must not be read as the implicit start of 1"
        );
    }

    #[test]
    fn line_range_new_rejects_an_inverted_range() {
        assert_eq!(
            LineRange::new(Some(9), Some(3)),
            Err(LineRangeError::Inverted { start: 9, end: 3 })
        );
    }

    #[test]
    fn count_lines_ignores_the_trailing_newline() {
        assert_eq!(count_lines(""), 0);
        assert_eq!(count_lines("a"), 1);
        assert_eq!(
            count_lines("a\n"),
            1,
            "no phantom line after a final newline"
        );
        assert_eq!(count_lines("a\nb"), 2);
        assert_eq!(count_lines("a\n\nb\n"), 3, "a blank line is still a line");
    }

    #[test]
    fn slice_lines_returns_an_inclusive_range() {
        let s = slice_lines(
            DOC,
            &LineRange {
                start: 1,
                end: Some(2),
            },
        )
        .unwrap();
        assert_eq!(s.content, "one\ntwo\n");
        assert_eq!((s.start_line, s.end_line, s.total_lines), (1, 2, 3));
    }

    #[test]
    fn slice_lines_serves_a_single_line() {
        let s = slice_lines(
            DOC,
            &LineRange {
                start: 2,
                end: Some(2),
            },
        )
        .unwrap();
        assert_eq!(s.content, "two\n");
    }

    #[test]
    fn slice_lines_runs_to_eof_when_end_is_open() {
        let s = slice_lines(
            DOC,
            &LineRange {
                start: 2,
                end: None,
            },
        )
        .unwrap();
        assert_eq!(s.content, "two\nthree\n");
        assert_eq!(s.end_line, 3);
    }

    #[test]
    fn slice_lines_clamps_an_end_past_the_document_and_reports_the_clamp() {
        let s = slice_lines(
            DOC,
            &LineRange {
                start: 3,
                end: Some(9999),
            },
        )
        .unwrap();
        assert_eq!(s.content, "three\n");
        assert_eq!(
            s.end_line, 3,
            "end_line must report what was served, not what was asked for"
        );
        assert_eq!(s.total_lines, 3);
    }

    #[test]
    fn slice_lines_rejects_a_start_past_the_document() {
        assert_eq!(
            slice_lines(
                DOC,
                &LineRange {
                    start: 4,
                    end: None
                }
            )
            .unwrap_err(),
            LineRangeError::StartPastEnd {
                start: 4,
                total_lines: 3
            }
        );
    }

    #[test]
    fn slice_lines_rejects_any_range_against_an_empty_document() {
        assert_eq!(
            slice_lines(
                "",
                &LineRange {
                    start: 1,
                    end: Some(1)
                }
            )
            .unwrap_err(),
            LineRangeError::StartPastEnd {
                start: 1,
                total_lines: 0
            }
        );
    }

    #[test]
    fn slice_lines_preserves_the_exact_bytes_including_crlf() {
        let crlf = "one\r\ntwo\r\nthree";
        let s = slice_lines(
            crlf,
            &LineRange {
                start: 1,
                end: Some(2),
            },
        )
        .unwrap();
        assert_eq!(
            s.content, "one\r\ntwo\r\n",
            "line endings must survive the round trip so the slice can be an edit old_string"
        );
    }

    #[test]
    fn slice_lines_keeps_an_unterminated_last_line_unterminated() {
        let s = slice_lines(
            "one\ntwo",
            &LineRange {
                start: 2,
                end: None,
            },
        )
        .unwrap();
        assert_eq!(s.content, "two");
        assert_eq!(s.total_lines, 2);
    }

    #[test]
    fn slice_or_whole_without_a_range_serves_the_whole_document() {
        let s = slice_or_whole(DOC.to_string(), None).unwrap();
        assert_eq!(s.content, DOC);
        assert_eq!((s.start_line, s.end_line, s.total_lines), (1, 3, 3));
        assert!(!s.partial());
    }

    #[test]
    fn slice_or_whole_without_a_range_handles_an_empty_document() {
        // The one input `slice_lines` refuses; the unranged path must still serve it.
        let s = slice_or_whole(String::new(), None).unwrap();
        assert_eq!(s.content, "");
        assert_eq!((s.start_line, s.end_line, s.total_lines), (1, 0, 0));
        assert!(!s.partial());
    }

    #[test]
    fn slice_or_whole_with_a_range_slices() {
        let range = LineRange {
            start: 2,
            end: Some(2),
        };
        let s = slice_or_whole(DOC.to_string(), Some(&range)).unwrap();
        assert_eq!(s.content, "two\n");
        assert!(s.partial());
    }

    #[test]
    fn partial_is_false_for_a_range_covering_every_line() {
        let range = LineRange {
            start: 1,
            end: Some(3),
        };
        let s = slice_or_whole(DOC.to_string(), Some(&range)).unwrap();
        assert!(
            !s.partial(),
            "an explicit full-document range must report the same as no range at all"
        );
    }

    #[test]
    fn slice_lines_over_the_whole_document_reproduces_it_byte_for_byte() {
        let s = slice_lines(
            DOC,
            &LineRange {
                start: 1,
                end: None,
            },
        )
        .unwrap();
        assert_eq!(s.content, DOC);
    }
}
