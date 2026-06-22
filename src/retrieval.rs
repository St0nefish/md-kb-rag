use std::collections::HashMap;
use std::path::{Path, PathBuf};

use globset::GlobSet;
use tracing::{debug, warn};

use crate::{
    embed::QueryEmbedder,
    qdrant::{RetrievalStore, SearchResult},
    rerank::Reranker,
};

/// Upper bound on the number of indexed file paths to fetch for fuzzy
/// `get_document` resolution. Larger KBs will silently cap at this value.
pub const MAX_INDEXED_PATHS_FOR_FUZZY: u64 = 10_000;
/// How many "did you mean?" suggestions to include when no basename matches.
pub const FUZZY_SUGGESTION_COUNT: usize = 3;

/// Dependencies needed by the shared retrieval functions.
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
}

/// A successfully retrieved document.
pub struct Document {
    pub path: PathBuf,
    pub content: String,
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

/// Resolve a user-supplied path against the data directory, applying the
/// path-traversal and file-type security checks.
pub(crate) fn resolve_within_data(
    raw: &str,
    data_path: &Path,
    include_patterns: &GlobSet,
) -> Result<PathBuf, ResolveErr> {
    let requested = PathBuf::from(raw);
    let resolved = if requested.is_absolute() {
        requested
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
        match reranker.rerank(query, &docs).await {
            Ok(ranked) => {
                let mut indexed: Vec<(usize, f32)> = ranked
                    .iter()
                    .map(|r| (docs_with_indices[r.index].0, r.relevance_score))
                    .collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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
                results.truncate(top_k);
            }
        }
    } else {
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
pub async fn get_document<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &RetrievalDeps<'_, E, Q>,
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

    // 2. Fuzzy fallback: load every indexed path from Qdrant and look for a
    //    basename match. Auto-resolve a unique match; otherwise produce a
    //    helpful error.
    let all_paths = deps
        .qdrant
        .fetch_facet_values(deps.collection, "file_path", MAX_INDEXED_PATHS_FOR_FUZZY)
        .await
        .unwrap_or_else(|e| {
            warn!("Failed to fetch file_path facet for fuzzy lookup: {e:#}");
            Vec::new()
        });
    if all_paths.len() as u64 == MAX_INDEXED_PATHS_FOR_FUZZY {
        warn!(
            cap = MAX_INDEXED_PATHS_FOR_FUZZY,
            "fuzzy path resolver hit the cap; some paths may be missing from suggestions"
        );
    }

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
        facet_paths: Vec<String>,
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
                facet_paths: Vec::new(),
                received_filters: std::sync::Mutex::new(None),
                received_sparse: std::sync::Mutex::new(None),
                last_call: std::sync::Mutex::new(None),
            }
        }
        fn with_search_err(msg: &str) -> Self {
            Self {
                search_err: Some(msg.to_string()),
                search_ok: Vec::new(),
                facet_paths: Vec::new(),
                received_filters: std::sync::Mutex::new(None),
                received_sparse: std::sync::Mutex::new(None),
                last_call: std::sync::Mutex::new(None),
            }
        }
        fn with_facet_paths(paths: Vec<String>) -> Self {
            Self {
                search_err: None,
                search_ok: Vec::new(),
                facet_paths: paths,
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

        async fn fetch_facet_values(
            &self,
            _collection: &str,
            _field: &str,
            _limit: u64,
        ) -> anyhow::Result<Vec<String>> {
            Ok(self.facet_paths.clone())
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
        let store = MockRetrievalStore::with_facet_paths(vec![]);
        let deps = make_deps(&embed, &store, &data_path, &gs);

        let doc = get_document(&deps, "docs/guide.md").await.unwrap();
        assert!(doc.content.contains("Content here."));
    }

    #[tokio::test]
    async fn get_document_literal_path_outside_returns_outside() {
        let tmp = tempfile::tempdir().unwrap();
        let data_path = tmp.path().canonicalize().unwrap();
        let gs = make_md_globset();
        let embed = MockEmbedder::ok(vec![]);
        let store = MockRetrievalStore::with_facet_paths(vec![]);
        let deps = make_deps(&embed, &store, &data_path, &gs);

        // /etc/hosts is a real file outside any tempdir
        if std::path::Path::new("/etc/hosts").exists() {
            let result = get_document(&deps, "/etc/hosts").await;
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
        let store = MockRetrievalStore::with_facet_paths(vec![]);
        let deps = make_deps(&embed, &store, &data_path, &gs);

        let txt_path = data_path.join("notes.txt").to_string_lossy().to_string();
        let result = get_document(&deps, &txt_path).await;
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
        // Facet returns the relative key — literal "foo.md" won't resolve, falls to fuzzy
        let store = MockRetrievalStore::with_facet_paths(vec!["notes/foo.md".to_string()]);
        let deps = make_deps(&embed, &store, &data_path, &gs);

        let doc = get_document(&deps, "foo.md").await.unwrap();
        assert!(doc.content.contains("Fuzzy content."));
    }

    #[tokio::test]
    async fn get_document_fuzzy_zero_matches_not_found_with_suggestions() {
        // No real filesystem needed — fuzzy 0-match path only does Levenshtein + relative_to_data.
        // Use a fake data_path; literal resolve will fail with NotFound (dir doesn't exist).
        let tmp = tempfile::tempdir().unwrap();
        let data_path = tmp.path().canonicalize().unwrap();

        // Relative keys as stored in the new index
        let facet_paths = vec!["tacoma-2024.md".to_string(), "voron-trident.md".to_string()];

        let gs = make_md_globset();
        let embed = MockEmbedder::ok(vec![]);
        let store = MockRetrievalStore::with_facet_paths(facet_paths);
        let deps = make_deps(&embed, &store, &data_path, &gs);

        // "tacoma-2025.md" doesn't exist anywhere — literal resolve: NotFound, fuzzy: 0 exact basename matches
        let result = get_document(&deps, "tacoma-2025.md").await;
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
        let facet_paths = vec!["a/notes.md".to_string(), "b/notes.md".to_string()];

        let gs = make_md_globset();
        let embed = MockEmbedder::ok(vec![]);
        let store = MockRetrievalStore::with_facet_paths(facet_paths);
        let deps = make_deps(&embed, &store, &data_path, &gs);

        // "notes.md" as basename — not on disk, so literal fails; fuzzy finds 2 exact matches
        let result = get_document(&deps, "notes.md").await;
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
}
