use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use globset::GlobSet;
use qdrant_client::qdrant::{Condition, Filter};
use tracing::{debug, warn};

use crate::{
    embed::QueryEmbedder,
    qdrant::{CHUNK_TEXT_KEY, PATH_ANCESTORS_KEY, RetrievalStore, SearchResult},
    rerank::Reranker,
    state::{
        DocumentIndex, DocumentQuery, DocumentQueryResult, DocumentSummary, InboundLink, LinkPage,
        OutboundLink, PathMatches,
    },
    status::QUERY_METRICS,
};

/// How many "did you mean?" suggestions to include when no basename matches.
pub const FUZZY_SUGGESTION_COUNT: usize = 3;

/// Hard cap on how many edges `get_document` reports per direction (`links_out`,
/// `links_in`). A hub document's inbound edge count is unbounded in principle (any
/// number of other documents can link to it), so this needs a ceiling the same way
/// `search` caps its result count — the true total and whether the page was
/// truncated ride alongside it (`LinkPage::total`/`has_more`) rather than silently
/// dropping the tail, matching this codebase's `total`/`has_more` convention
/// everywhere else it caps a list (`search`, `list_documents`, the schema-update
/// casualty list).
pub const MAX_LINKS_PER_DIRECTION: u64 = 100;

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

/// Filters to apply when searching, already lowered to Qdrant conditions.
///
/// The MCP `search` tool's rich `filters` (see `state::FieldFilter`) get here via
/// `qdrant::lower_field_filters`, after the tool handler has checked every named
/// field carries a payload index; the web UI's plain `domain`/`type`/`tags` query
/// params build their conditions directly (`qdrant::Condition::matches`) with no such
/// check, exactly as they always have. Either way, `search`/`search_grouped` below
/// treat this as an opaque, already-valid set of extra conditions to AND onto the
/// query — this module has no opinion on where a `Condition` came from.
#[derive(Default)]
pub struct SearchFilters {
    pub conditions: Vec<Condition>,
}

/// Build `SearchFilters` from the plain `domain`/`type`/`tags` parameters the CLI
/// `search` subcommand and the web UI's `/api/search` both accept — the shape
/// neither has reason to change now that the MCP `search` tool has moved on to the
/// richer `filters` model. Each becomes a single-field Qdrant match condition
/// exactly as `retrieval::search` built them inline before this function existed;
/// an absent value, or an empty `tags` list, contributes nothing.
pub fn plain_search_filters(
    domain: Option<&str>,
    r#type: Option<&str>,
    tags: Option<&[String]>,
) -> SearchFilters {
    let mut conditions = Vec::new();
    if let Some(d) = domain {
        conditions.push(Condition::matches("domain", d.to_string()));
    }
    if let Some(t) = r#type {
        conditions.push(Condition::matches("type", t.to_string()));
    }
    if let Some(tags) = tags
        && !tags.is_empty()
    {
        conditions.push(Condition::matches("tags", tags.to_vec()));
    }
    SearchFilters { conditions }
}

/// A `path_prefix` needle already resolved to concrete document paths (#182).
///
/// The metadata index is the single authority for what a needle matches — it holds
/// every indexed path, including those of documents whose Qdrant payload predates
/// the `path_ancestors` field. Resolving there and pushing the *answer* down means
/// query mode and enumeration mode can never disagree about the same needle, which
/// is the split issue #259 was filed for.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PathFilter {
    /// The matching KB-relative paths, in the order the index returned them.
    pub paths: Vec<String>,
    /// True when the resolution cap hid matches that really exist, so a caller must
    /// not read an under-full page as "there is nothing more".
    pub truncated: bool,
}

impl From<PathMatches> for PathFilter {
    fn from(m: PathMatches) -> Self {
        let truncated = m.truncated();
        Self {
            paths: m.paths,
            truncated,
        }
    }
}

/// Cap on how many resolved paths a `path_prefix` needle may push into one Qdrant
/// filter (#182).
///
/// A very broad needle (`a`) matches most of the corpus, and the resolved list is
/// sent to Qdrant on every query, so it has to be bounded somewhere. Hitting the cap
/// is not silent: [`PathFilter::truncated`] rides through to
/// `path_prefix_truncated` on the response. The value is a starting point chosen to
/// sit comfortably above any realistic *deliberate* filter, not a measured optimum.
pub const PATH_FILTER_MAX_PATHS: u64 = 2000;

/// Options controlling search behaviour.
pub struct SearchOptions {
    pub limit: u64,
    pub min_score: Option<f32>,
    /// When true, use hybrid sparse+dense retrieval with RRF fusion; otherwise
    /// dense-only. Sourced from `search.hybrid`.
    pub hybrid: bool,
    /// Candidates fetched from each arm before RRF fusion. Sourced from
    /// `search.rrf_candidates`. Consulted whenever any fused arm is in play —
    /// sparse (when `hybrid` is true) or phrase (when `phrase` is true and the
    /// query has a quoted span) — not only when `hybrid` is true.
    pub rrf_candidates: u64,
    /// When true, double-quoted spans in `query` become exact-phrase conditions
    /// (see [`extract_phrases`]), added as a third fused prefetch arm alongside
    /// dense/sparse — independent of `hybrid`. The caller (not this module) is
    /// responsible for ANDing in whether the phrase-matching payload index is
    /// actually available on the server right now (`search.phrase` config AND
    /// `status::IndexStatus::phrase_matching_available`), so this flag alone is
    /// the single source of truth for "attempt phrase parsing" — `search`/
    /// `search_grouped` never touch global state to decide. When `false`, quoted
    /// text is treated as literal characters and the query is used unmodified,
    /// which keeps an unquoted query's behavior identical either way.
    pub phrase: bool,
    /// When true, surface per-result score breakdown metadata (pre-rerank score when applicable).
    pub explain: bool,
    /// Exclude documents with `mtime` payload below this Unix timestamp.
    pub modified_after: Option<i64>,
    /// Exclude documents with `mtime` payload above this Unix timestamp.
    pub modified_before: Option<i64>,
    /// Restrict to this already-resolved set of documents — see [`PathFilter`].
    ///
    /// #182: retrieval no longer interprets path syntax. The caller resolves a
    /// `path_prefix` needle against the metadata index
    /// (`state::DocumentIndex::paths_matching`) and hands the answer down, so the
    /// substring semantics are decided in exactly one place for both search modes
    /// rather than by a Qdrant keyword match here and a SQL `LIKE` in enumeration.
    pub path_filter: Option<PathFilter>,
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

/// Build the `mtime__gte`/`mtime__lte` sentinel entries [`crate::qdrant::build_conditions`]
/// special-cases, from [`SearchOptions`]'s recency filters. Shared by [`search`] and
/// [`search_grouped`] so the two don't drift on how a recency filter lowers.
fn mtime_filter_map(opts: &SearchOptions) -> HashMap<String, serde_json::Value> {
    let mut filter_map = HashMap::new();
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
    filter_map
}

/// Pull double-quoted spans out of `query` as exact-phrase conditions, returning
/// `(flattened_query, phrases)`: `flattened_query` is `query` with only the paired
/// quote *delimiters* removed (everything else — including the phrase words
/// themselves and all original whitespace — is preserved byte-for-byte in place),
/// and `phrases` is each non-blank phrase's trimmed text, in order. Multiple
/// quoted spans each become a separate entry — callers AND them together.
///
/// `flattened_query`, not the raw `query`, is what should still feed the dense
/// embedding and the sparse tokenizer: `deploy notes for "node:ares" rocm` flattens
/// to `deploy notes for node:ares rocm`, so the words inside a phrase keep
/// contributing to ordinary term/semantic retrieval, not just the phrase filter.
///
/// An unterminated quote (no matching close before the end of the string) is not
/// an error — the dangling `"` is left as a literal character and contributes no
/// phrase, exactly as an unquoted query would. This is also why a query with no
/// `"` at all returns `query` completely unchanged: no delimiter is ever found, so
/// nothing is removed and nothing is trimmed — an unquoted query's behavior is
/// identical whether or not phrase parsing runs at all.
pub fn extract_phrases(query: &str) -> (String, Vec<String>) {
    let chars: Vec<char> = query.chars().collect();
    let n = chars.len();
    let mut is_delimiter = vec![false; n];
    let mut phrases = Vec::new();

    let mut i = 0;
    while i < n {
        if chars[i] == '"'
            && let Some(rel_end) = chars[i + 1..].iter().position(|&c| c == '"')
        {
            let end = i + 1 + rel_end;
            is_delimiter[i] = true;
            is_delimiter[end] = true;
            let phrase: String = chars[i + 1..end].iter().collect();
            let trimmed = phrase.trim();
            if !trimmed.is_empty() {
                phrases.push(trimmed.to_string());
            }
            i = end + 1;
        } else {
            i += 1;
        }
    }

    if phrases.is_empty() {
        // No pair was matched (either no quotes, or only dangling ones) — return
        // the original string untouched rather than an equal-but-rebuilt copy.
        return (query.to_string(), phrases);
    }

    let flattened: String = chars
        .iter()
        .enumerate()
        .filter(|(idx, _)| !is_delimiter[*idx])
        .map(|(_, c)| *c)
        .collect();
    (flattened, phrases)
}

/// How much larger than the caller's requested fetch size to ask Qdrant for when
/// `path_prefix` is set in query mode, before [`apply_path_filter`] retains only
/// the matching subset.
///
/// #130: since [`path_filter_condition`] started giving Qdrant an exact filter for
/// any document carrying `path_ancestors`, this over-fetch is no longer paying for
/// imprecision across the *whole* corpus — only for the legacy slice that predates
/// that field (see that function's doc comment for the full backward-compatibility
/// reasoning). Those legacy documents pass the Qdrant filter unconditionally (via
/// its `is_empty(path_ancestors)` arm) regardless of whether their real path
/// matches the needle, so they can still crowd a `limit`-sized page the same way an
/// unfiltered corpus always could pre-#130 — over-fetching remains necessary until
/// every document has been reindexed at least once under this field.
const PATH_PREFIX_OVERFETCH_MULTIPLIER: u64 = 5;

/// Absolute ceiling on the over-fetched size computed via
/// `PATH_PREFIX_OVERFETCH_MULTIPLIER`, so a large `limit` (or a reranker's
/// `rerank_candidate_limit`) cannot turn into an unbounded Qdrant fetch.
const PATH_PREFIX_OVERFETCH_CEILING: u64 = 500;

/// Absolute ceiling on how deep into the ranked candidate pool `offset` may
/// reach in query mode (#224), mirroring [`PATH_PREFIX_OVERFETCH_CEILING`]'s
/// role for `path_prefix`: both cap the same dimension — how many ranked
/// candidates the funnel is ever asked to produce — so an unbounded `offset`
/// alone (no reranker in play, see [`search_paged`]'s doc comment for the
/// other, reranker-specific bound) cannot turn into an unbounded Qdrant fetch
/// any more than an unbounded `path_prefix` over-fetch already can't. Reusing
/// the same value isn't required for correctness, only chosen because the two
/// knobs cap the same dimension and there's no reason for them to drift apart.
const MAX_OFFSET_DEPTH: u64 = PATH_PREFIX_OVERFETCH_CEILING;

/// Scale `fetch_limit` up for the `path_filter` over-fetch when one is set,
/// otherwise return it unchanged. Shared by [`search`] and [`search_grouped`] so
/// the two apply the same policy.
fn path_prefix_fetch_limit(fetch_limit: u64, path_filter: Option<&PathFilter>) -> u64 {
    if path_filter.is_some() {
        fetch_limit
            .saturating_mul(PATH_PREFIX_OVERFETCH_MULTIPLIER)
            .min(PATH_PREFIX_OVERFETCH_CEILING)
    } else {
        fetch_limit
    }
}

/// Canonicalize a caller-supplied `path_prefix` needle into the one form every
/// consumer must see.
///
/// #182: there is now exactly one consumer of the needle itself — the metadata
/// index resolution (`state::DocumentIndex::paths_matching`, and the identical
/// `LIKE` that `push_where` applies in enumeration mode). Callers normalize here,
/// once, before either, so the two modes cannot disagree about the same input.
/// That is why this is public: the normalization is part of the needle's contract,
/// not an internal detail of this module.
///
/// Trailing slashes are stripped so `sysadmin/` and `sysadmin` are one needle. This
/// mattered more when the value had to match a `path_ancestors` entry exactly (a
/// trailing slash matched nothing at all); under substring matching it is milder —
/// `efi.md/` would simply find no path containing that literal — but the two forms
/// still ought to mean the same thing.
///
/// `trim_end_matches`, not `strip_suffix`: one strip leaves `"sysadmin//"` as
/// `"sysadmin/"`. Returns `None` for a needle that is empty once trimmed (`""`,
/// `"/"`, `"//"`) — "match everything" is the absence of a filter, not a filter on
/// the empty string, which as a substring would match every path anyway.
///
/// Idempotent.
pub fn normalize_path_needle(path_prefix: Option<&str>) -> Option<&str> {
    let trimmed = path_prefix?.trim_end_matches('/');
    (!trimmed.is_empty()).then_some(trimmed)
}

/// Build the Qdrant condition that enforces `path_prefix` (#130), and the
/// backward-compatibility story behind it.
///
/// ## Mechanics
///
/// `ingest::index_paths` writes every document's ancestor-directory segments —
/// plus its own full relative path — into the `path_ancestors` keyword-array
/// payload field (see [`crate::qdrant::PATH_ANCESTORS_KEY`] and
/// `ingest::derive_path_ancestors`'s doc comment for the exact shape). Because
/// each document's own full path is one of those entries, a *resolved path list*
/// can be matched through that same keyword index as a match-any:
/// `Condition::matches(PATH_ANCESTORS_KEY, paths)`. Qdrant does the filtering
/// against an index it already has — no new payload index, no reindex, and no
/// pulling candidates back to re-check a string client-side.
///
/// #182 is what makes this work for a *substring* needle, which a keyword index
/// could not match directly: the needle is resolved to concrete paths against the
/// metadata index before it ever gets here, so what Qdrant sees is always an exact
/// set. No filter, or a needle that matched nothing, returns `None` — "match
/// everything" and "match nothing" both need no condition here (the latter because
/// [`apply_path_filter`]'s empty-set retain drops everything anyway).
///
/// ## The backward-compatibility problem
///
/// `path_ancestors` did not exist before #130. A document that has not been
/// reindexed since carries no such field at all — and a Qdrant keyword-match
/// condition against a *missing* field matches nothing, full stop. Compiling
/// `path_prefix` to `Condition::matches(PATH_ANCESTORS_KEY, prefix)` alone would
/// therefore silently return **zero results** for every such document, corpus-wide,
/// the moment this code shipped — this project's live deployment carries roughly
/// 4,800 chunks indexed before this change, and nothing about a plain `serve`
/// restart re-embeds them (only `index --full`, or the slow drip of incremental
/// edits touching each file, ever does). That is a materially worse failure than
/// today's imprecise-but-nonempty over-fetch/retain, so it is not acceptable to
/// ship as-is — seeing this coming is the entire reason issue #130 asked for the
/// tradeoff to be argued explicitly rather than assumed away.
///
/// ## The fix: an OR-escape for documents Qdrant knows nothing about
///
/// The condition this function builds is not a bare equality — it is
/// `path_ancestors MATCHES prefix OR path_ancestors IS EMPTY`
/// (`Condition::is_empty` matches a key that is missing, `null`, or `[]` — exactly
/// the "never reindexed under this field" case). That the *missing*-key case
/// counts is the load-bearing part and is not something the qdrant-client crate
/// documents — its doc comment on the constructor says only "check if a field is
/// empty". It is Qdrant server behavior (`IsEmpty` treats an absent key as empty),
/// verified empirically against `qdrant/qdrant:v1.17.0` and pinned by
/// `qdrant::tests::path_prefix_filter_semantics_hold_on_a_live_server`, which
/// fails loudly if a future server narrows it to "empty array only".
/// Concretely, per document:
///
///   - **Reindexed** (carries `path_ancestors`): the `MATCHES` arm decides —
///     exact, no over-fetch needed for this document, no risk of losing it to a
///     retain.
///   - **Not yet reindexed** (no `path_ancestors` at all): the `IS EMPTY` arm lets
///     it through Qdrant's filter unconditionally, exactly as if `path_prefix`
///     were not applied at Qdrant level for it. [`apply_path_filter`] then runs
///     its exact set-membership retain, but *only meaningfully
///     affects this bucket* — a reindexed document's `file_path` is guaranteed
///     (by construction of `derive_path_ancestors`) to already satisfy that same
///     retain, so it is never the reason a reindexed document gets dropped.
///
/// This degrades gracefully at every point along the migration curve: a corpus
/// with zero reindexed documents behaves identically to pre-#130 (every document
/// takes the legacy retain path, `path_prefix_truncated` behaves exactly as
/// before); a fully reindexed corpus needs no `IS EMPTY` escape in practice (no
/// document will ever satisfy it) and `path_prefix_truncated` naturally tends
/// toward `false` because Qdrant's own filter, not a client-side retain, is now
/// deciding who's in the page. No reindex is required for correctness at any
/// point in between — only for precision to improve. See
/// `SearchOptions::path_prefix`'s doc comment for the field-level summary, and
/// `path_prefix_fetch_limit`'s for why the over-fetch multiplier itself is
/// unchanged rather than removed.
///
/// ## Why not detect corpus-wide coverage instead?
///
/// An alternative considered: check once (e.g. at startup, or via a Qdrant `count`
/// query for `IS EMPTY` points) whether *any* legacy document remains, and only
/// then decide whether to add the `IS EMPTY` escape at all. That was rejected —
/// not because it's unsound, but because it buys nothing this per-document OR
/// doesn't already give for free: the escape costs one extra filter clause and, for
/// an already-fully-migrated corpus, zero extra documents ever match it (the
/// escape is *present in the query* but *matches nothing*, which is
/// indistinguishable in effect from not being there). A coverage check would add a
/// second code path to keep correct, a place for the two to drift, and (per this
/// file's touched-files constraint) nowhere to persist or observe the result
/// anyway — `status.rs` is out of scope for this change. Self-adapting per document
/// is strictly simpler and cannot be stale.
fn path_filter_condition(path_filter: Option<&PathFilter>) -> Option<Condition> {
    let filter = path_filter?;
    // A needle that matched nothing (a typo'd filename, say) must not fall through
    // to an unfiltered search whose every result the retain then discards — ask
    // Qdrant for nothing instead. The same `unsatisfiable` shape an empty `any_of`
    // filter set compiles to, rather than an empty keyword list, whose behavior is
    // not a thing worth relying on.
    if filter.paths.is_empty() {
        return Some(Condition::from(Filter::should([
            crate::qdrant::unsatisfiable(PATH_ANCESTORS_KEY),
            Condition::is_empty(PATH_ANCESTORS_KEY),
        ])));
    }
    Some(Condition::from(Filter::should([
        // `derive_path_ancestors` puts each document's own full relative path in
        // `path_ancestors` alongside its ancestor directories, so a resolved path
        // list matches exactly through the keyword index that already exists — no
        // new payload index and no reindex.
        Condition::matches(PATH_ANCESTORS_KEY, filter.paths.clone()),
        Condition::is_empty(PATH_ANCESTORS_KEY),
    ])))
}

/// Restrict `results` to documents in the resolved [`PathFilter`] set, when one is
/// given.
///
/// #182: this is now an exact set-membership test, not a `str::starts_with` guess.
/// It is what enforces the filter for a legacy (not-yet-reindexed) document, which
/// carries no `path_ancestors` and therefore passes [`path_filter_condition`]'s
/// `IS EMPTY` escape unconditionally — see that function's doc comment. For a
/// reindexed document the retain is provably redundant (Qdrant already matched it
/// against the same path list), so it costs nothing and needs no per-document
/// branch to skip.
///
/// Because the membership test is exact rather than approximate, the legacy slice
/// can no longer be *wrongly* dropped or kept; the only thing left for
/// [`path_prefix_truncated`] to report is a page the over-fetch could not fill.
fn apply_path_filter(
    results: &mut Vec<SearchResult>,
    path_filter: Option<&PathFilter>,
    data_root: &Path,
) {
    let Some(filter) = path_filter else { return };
    let allowed: HashSet<&str> = filter.paths.iter().map(String::as_str).collect();
    results.retain(|r| {
        r.payload
            .get("file_path")
            .and_then(|v| v.as_str())
            .is_some_and(|fp| allowed.contains(relative_to_data(fp, data_root).as_str()))
    });
}

/// True when the path filter may have hidden matches that really exist, rather
/// than the caller genuinely running out of them.
///
/// #182: there are now two independent ways that can happen, and either one alone
/// is enough to warn the caller:
///
///   1. **Resolution was capped** ([`PathFilter::truncated`]) — the needle matched
///      more documents than `PATH_FILTER_MAX_PATHS`, so the filter handed to Qdrant
///      is itself an incomplete picture of what matches. This is the new one, and it
///      is knowable up front, independent of what any single page returned.
///   2. **The over-fetched page could not be filled** — the historical signal. Only
///      the legacy (no `path_ancestors`) slice can now be dropped by the retain,
///      since [`apply_path_filter`] is an exact membership test, but that slice can
///      still crowd out a `limit`-sized page.
///
/// For (2): `pre_retain_count` is how many hits Qdrant returned for the (possibly
/// over-fetched) `fetch_limit` requested, *before* [`apply_path_filter`] ran;
/// `post_retain_count` is the count immediately after. When Qdrant returned fewer
/// than `fetch_limit`, the fetch was exhaustive — there was nothing more to find —
/// so under-return relative to `limit` is proven benign. Only when Qdrant filled
/// the full (over-fetched) page is a post-retain shortfall unprovable, and so must
/// be surfaced rather than silently returned.
fn path_prefix_truncated(
    path_filter: Option<&PathFilter>,
    limit: u64,
    fetch_limit: u64,
    pre_retain_count: u64,
    post_retain_count: u64,
) -> bool {
    let Some(filter) = path_filter else {
        return false;
    };
    filter.truncated || (post_retain_count < limit && pre_retain_count >= fetch_limit)
}

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

/// Apply `offset` then `limit` to an already-fully-ranked (best-first) list —
/// the LAST step of the funnel (#224), run only once fusion, reranking, and
/// the diversity cap have all already finished settling the true order. See
/// [`search_paged`]'s doc comment for why offset can't be satisfied any
/// earlier than this (a pre-fusion or pre-rerank offset would page over a
/// different ranking than the one the caller ends up seeing).
///
/// `items` is expected to already hold at most as many entries as the funnel
/// was bounded to produce (`page_depth` in the callers below) — this only
/// windows into what's there, it cannot recover entries the funnel was never
/// asked to fetch in the first place.
fn paginate<T>(mut items: Vec<T>, offset: u64, limit: u64) -> Vec<T> {
    if offset > 0 {
        let skip = (offset as usize).min(items.len());
        items = items.split_off(skip);
    }
    items.truncate(limit as usize);
    items
}

/// A successfully retrieved document.
pub struct Document {
    pub path: PathBuf,
    pub content: String,
    /// Outbound `document_links` edges (both `markdown` and `semantic` kinds),
    /// capped at [`MAX_LINKS_PER_DIRECTION`]. See [`document_links`]'s doc
    /// comment for how a link-lookup failure is handled.
    pub links_out: LinkPage<OutboundLink>,
    /// Inbound `document_links` edges targeting this document, same cap and
    /// failure handling as `links_out`.
    pub links_in: LinkPage<InboundLink>,
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
    /// [`search_grouped`]'s document-metadata hydration (the `DocumentIndex` lookup)
    /// failed.
    Document(anyhow::Error),
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
        std::io::ErrorKind::NotFound => {
            // #225: a `..` component that fails to canonicalize is a traversal
            // attempt whose escaped target just doesn't happen to exist — NOT
            // a "no such document" the way a `..`-free path failing the same
            // way genuinely is. Left unchecked, `../../etc/passwd` (relative)
            // and `/../x` (leading-slash — falls into the `kb_root_relative`
            // branch above, which is itself just `data_path.join("../x")`,
            // equally escaping) report DIFFERENT error classes purely because
            // one of them happens to resolve to a real file and the other
            // doesn't: the real file's canonicalize succeeds and hits the
            // `starts_with` check below (`Outside`), while the nonexistent
            // one fails canonicalize first and would otherwise return
            // `NotFound` here — before the traversal check ever runs. Neither
            // shape leaks anything (both are hard-rejected either way), but
            // `NotFound` reads as "no such document," hiding the real reason
            // the path was refused from a caller or a developer debugging path
            // handling. Route both shapes to the same `Outside` class instead,
            // matching what the EXISTING-target case already reports a few
            // lines below — a `..`-free path that fails to canonicalize is
            // still a genuine `NotFound`, eligible for the fuzzy-basename
            // fallback the way it always was.
            if resolved
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
            {
                ResolveErr::Outside
            } else {
                ResolveErr::NotFound
            }
        }
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

/// The results of [`search`] plus whether a `path_prefix` filter may have
/// under-returned relative to `limit` (`path_prefix_truncated` — see
/// `retrieval::path_prefix_truncated` and `apply_path_filter`'s doc comment).
///
/// Implements [`IntoIterator`] over `results` (yielding owned [`SearchResult`]s,
/// same as the `Vec<SearchResult>` this replaced) so a caller that only wants the
/// result list — e.g. `write.rs`'s dedup gate, which never sets `path_prefix` —
/// needs no change at the call site.
#[derive(Debug, Clone)]
pub struct SearchOutcome {
    pub results: Vec<SearchResult>,
    /// True when the path filter may have hidden matches that really exist. Two
    /// independent causes, either sufficient — see [`path_prefix_truncated`]:
    /// the needle resolved to more documents than [`PATH_FILTER_MAX_PATHS`]
    /// (#182), or the *legacy* (not-yet-reindexed) slice of the corpus crowded out
    /// genuine matches within the over-fetch window (#130, and that cause settles
    /// to always-`false` on its own once every document has been reindexed at
    /// least once under `path_ancestors`).
    ///
    /// The field name predates both and is kept as-is: renaming it would be a
    /// breaking response-shape change for every caller — the MCP `search` tool,
    /// the CLI, `/api/search`.
    pub path_prefix_truncated: bool,
    /// True when the caller's requested `offset + limit` exceeded the ranked-
    /// candidate depth the funnel was bounded to produce (#224) — see
    /// [`search_paged`]'s doc comment for the exact bound (`rerank_candidate_limit`
    /// when reranking is active, [`MAX_OFFSET_DEPTH`] otherwise). When true,
    /// `results` may look short (or empty) not because the corpus is
    /// exhausted but because paging that deep was never attempted — the same
    /// "can't prove there isn't more" signal `path_prefix_truncated` already
    /// gives for its own best-effort retain, just for a different stage of
    /// the funnel.
    pub offset_truncated: bool,
}

impl IntoIterator for SearchOutcome {
    type Item = SearchResult;
    type IntoIter = std::vec::IntoIter<SearchResult>;

    fn into_iter(self) -> Self::IntoIter {
        self.results.into_iter()
    }
}

/// Build the cross-encoder reranker's input list: each result's original index
/// paired with its chunk text, for results that have one. Reads chunk text via
/// [`CHUNK_TEXT_KEY`] — see that constant's doc comment for why the read key
/// must never drift from `ingest`'s write key (the #61 regression this guards
/// against). Pulled out of `search` so the writer/reader agreement can be
/// pinned by a direct unit test rather than only indirectly through a full
/// search call.
fn docs_for_rerank(results: &[SearchResult]) -> Vec<(usize, &str)> {
    results
        .iter()
        .enumerate()
        .filter_map(|(i, r)| {
            r.payload
                .get(CHUNK_TEXT_KEY)
                .and_then(|v| v.as_str())
                .map(|s| (i, s))
        })
        .collect()
}

/// Embed the query, apply filters, search Qdrant, apply min_score floor, and
/// return raw results. Does NOT format any output — callers do that.
///
/// `offset` (#224) pages over the SAME funnel `limit` truncates — fusion,
/// reranking, and the diversity cap all have to finish running before either
/// applies, so paging cannot be satisfied by re-querying Qdrant "starting at
/// offset" (that would page over Qdrant's pre-fusion order, not the
/// caller-visible ranked one). Instead the funnel below is asked to produce
/// `offset + limit` ranked candidates instead of just `limit` (`page_depth`),
/// and the leading `offset` of them is sliced off at the very end — see
/// [`paginate`] — only once everything above has already settled the true
/// order.
///
/// That depth is bounded, not unlimited, by whichever cap already governs how
/// many candidates the funnel is willing to produce:
///   - `rerank_candidate_limit` (`reranking.candidate_limit`), when a
///     reranker is configured: the funnel never asks the cross-encoder to
///     score more than this many candidates in the first place, so nothing
///     past it can ever be reordered into the caller-visible ranking no
///     matter how deep `offset` reaches. This bound already existed before
///     offset paging did — it's sized independently of `opts.limit` to give
///     the reranker a real pool to choose from — and simply doubles as
///     offset's ceiling too now.
///   - [`MAX_OFFSET_DEPTH`], always: an absolute ceiling for the case above
///     (when no reranker is configured, or one is but `rerank_candidate_limit`
///     is unset), so a very large `offset` alone can't turn into an unbounded
///     Qdrant fetch — the same role [`PATH_PREFIX_OVERFETCH_CEILING`] plays
///     for `path_prefix`.
///
/// A page whose depth exceeds this bound gets [`SearchOutcome::offset_truncated`]
/// set rather than a silently short/empty result: past that depth this
/// function cannot distinguish "the corpus has no more matches" from "the
/// funnel was never asked to fetch that deep," so it says so explicitly,
/// mirroring `path_prefix_truncated`'s reasoning for its own best-effort
/// retain.
pub async fn search_paged<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &RetrievalDeps<'_, E, Q>,
    query: &str,
    filters: &SearchFilters,
    opts: &SearchOptions,
    offset: u64,
) -> Result<SearchOutcome, SearchError> {
    // Double-quoted spans become phrase conditions; the flattened (dequoted) text
    // still feeds the embedding and the sparse tokenizer below. When `opts.phrase`
    // is false this is a no-op: `flat_query` is `query` unchanged and `phrases` is
    // empty, so an unquoted query's behavior — and a quoted one, when the caller
    // has decided phrase matching isn't available — is untouched.
    let (flat_query, phrases) = if opts.phrase {
        extract_phrases(query)
    } else {
        (query.to_string(), Vec::new())
    };

    let embed_start = std::time::Instant::now();
    let vector = deps
        .embed_client
        .embed_query(&flat_query)
        .await
        .map_err(SearchError::Embed)?;
    let embed_ms = embed_start.elapsed().as_millis();

    let filter_map = mtime_filter_map(opts);
    // #182: the caller already resolved the needle against the metadata index, so
    // there is no path syntax to interpret here — the same resolved set drives the
    // Qdrant condition, the over-fetch, the post-fetch retain, and the truncation
    // signal.
    let path_filter = opts.path_filter.as_ref();
    // AND the exact-or-legacy-escape `path_ancestors` condition onto whatever
    // conditions the caller already built — see `path_filter_condition`'s doc
    // comment for the full backward-compatibility reasoning. A no-op (no condition
    // appended) when there is no filter, or when the needle matched nothing.
    let mut extra_conditions = filters.conditions.clone();
    extra_conditions.extend(path_filter_condition(path_filter));

    // See this function's doc comment for the full reasoning — `page_depth` is
    // how many ranked candidates the funnel needs to produce to serve
    // `offset + limit`, bounded by whichever depth cap already applies.
    let requested_depth = offset.saturating_add(opts.limit);
    let max_depth = match deps.reranker {
        Some(_) => opts
            .rerank_candidate_limit
            .unwrap_or(opts.limit)
            .min(MAX_OFFSET_DEPTH),
        None => MAX_OFFSET_DEPTH,
    };
    let offset_truncated = requested_depth > max_depth;
    let page_depth = requested_depth.min(max_depth);

    let fetch_limit = if deps.reranker.is_some() {
        // Unchanged by offset paging: already sized to the reranker's own
        // candidate pool, independent of `opts.limit` — see this function's
        // doc comment for why that pool already has enough headroom for
        // `offset` (up to `max_depth`) without needing to grow further here.
        opts.rerank_candidate_limit.unwrap_or(opts.limit)
    } else {
        // No reranker: the funnel's only candidate-depth knob is `opts.limit`
        // itself, so it has to grow to `page_depth` for `offset` to have
        // anything to page into.
        page_depth
    };
    let fetch_limit = path_prefix_fetch_limit(fetch_limit, path_filter);

    let search_start = std::time::Instant::now();
    // Tokenize the flattened query into a sparse vector when hybrid is on; an
    // empty result (e.g. an all-punctuation query) is treated the same as hybrid
    // being off, to avoid sending an empty vector to Qdrant.
    let sparse = if opts.hybrid {
        let sparse = crate::sparse::tokenize(&flat_query);
        if sparse.0.is_empty() {
            if opts.explain {
                debug!(
                    "hybrid sparse-fallback: empty sparse vector, explain scores reflect dense-only"
                );
            }
            None
        } else {
            Some(sparse)
        }
    } else {
        None
    };

    let mut results = if sparse.is_none() && phrases.is_empty() {
        // Neither arm needed — the plain dense-only query, unchanged from before
        // hybrid/phrase existed.
        deps.qdrant
            .search(
                deps.collection,
                vector,
                filter_map,
                extra_conditions.clone(),
                fetch_limit,
            )
            .await
            .map_err(SearchError::Search)?
    } else {
        // Fused retrieval: sparse when hybrid produced one, phrase when requested
        // — independent of each other, so this also covers "phrase only, hybrid
        // off" (dense + phrase, two arms).
        deps.qdrant
            .hybrid_search(
                deps.collection,
                vector,
                sparse,
                &phrases,
                filter_map,
                extra_conditions,
                fetch_limit,
                opts.rrf_candidates,
                opts.explain,
            )
            .await
            .map_err(SearchError::Search)?
    };
    let search_ms = search_start.elapsed().as_millis();

    let pre_prefix_count = results.len() as u64;
    apply_path_filter(&mut results, path_filter, deps.data_path);
    // Measured against `page_depth`, not `opts.limit`: with `offset` set, the
    // retain needs to have kept `page_depth` survivors to serve the requested
    // page, not merely `opts.limit` of them — see this function's doc comment.
    let path_prefix_truncated = path_prefix_truncated(
        path_filter,
        page_depth,
        fetch_limit,
        pre_prefix_count,
        results.len() as u64,
    );

    // Apply min_score floor only when Some — None is a no-op, preserving
    // current behaviour.
    if let Some(s) = opts.min_score {
        results.retain(|r| r.score >= s);
    }

    // #245: which rerank outcome (if any) to report to `record_query` at
    // whichever return point this call actually takes. `None` until a reranker
    // is genuinely attempted below (`deps.reranker` absent, or `docs.is_empty()`
    // short-circuits before ever calling it); set to `Some((rerank_ms, success))`
    // by every branch that DOES attempt one, including the "response was
    // entirely unusable" fallback below, which is the same silent-degradation
    // shape as the `Err(e)` arm and is recorded identically (`success: false`).
    let mut rerank_outcome: Option<(u64, bool)> = None;

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

        let docs_with_indices: Vec<(usize, &str)> = docs_for_rerank(&results);
        let docs: Vec<&str> = docs_with_indices.iter().map(|(_, s)| *s).collect();
        // Retain up to `page_depth` candidates — not just `opts.limit` — so
        // there's something for `offset` to page into; the final `paginate`
        // call at the bottom of this function is what narrows that back down
        // to the caller's actual page.
        let retain_depth = page_depth as usize;
        if docs.is_empty() {
            if let Some(max_per_document) = opts.diversity_max_per_document {
                results = apply_diversity_cap(results, max_per_document);
            }
            results.truncate(retain_depth);
            let results = paginate(results, offset, opts.limit);
            // #245: this early return produces zero results and, unmodified,
            // never reaches the timing `debug!`/`record_query` call at the
            // bottom of this function — without this call zero-result queries
            // are systematically undercounted whenever reranking is enabled,
            // exactly the gap the issue was filed over. `rerank: None` since no
            // rerank was attempted (there were no candidates to send it).
            QUERY_METRICS.record_query(embed_ms as u64, search_ms as u64, None, results.len());
            return Ok(SearchOutcome {
                results,
                path_prefix_truncated,
                offset_truncated,
            });
        }
        let rerank_start = std::time::Instant::now();
        match reranker.rerank(query, &docs).await {
            Ok(ranked) => {
                // The reranker's response is untrusted external input, deserialized
                // straight off the wire (`rerank.rs`'s `RerankResult`, no validation
                // there either) — a version mismatch between the configured
                // `reranking.model`/API and this client's index-mapping assumption,
                // an off-by-one in the backend's own truncation logic, or a
                // misconfigured `reranking.base_url` pointed at a different service
                // can all produce an `index` that is out of range or repeated.
                // Indexing `docs_with_indices` with an out-of-range value used to
                // panic this request outright (#136); a repeated index used to let
                // one chunk occupy two result slots, double-charging its document's
                // diversity allowance and handing the caller a literal duplicate
                // (#138). Bounds-check and dedupe here, keeping the first
                // (highest-ranked, since a reranker returns its results best-first)
                // occurrence of each valid index, so a malformed response degrades
                // the same way a reranker error already does (the `Err` arm below)
                // instead of corrupting or crashing the request.
                let mut seen_indices: std::collections::HashSet<usize> =
                    std::collections::HashSet::with_capacity(ranked.len());
                let mut out_of_range = 0usize;
                let mut duplicate = 0usize;
                let mut indexed: Vec<(usize, f32)> = ranked
                    .iter()
                    .filter(|r| {
                        if r.index >= docs_with_indices.len() {
                            out_of_range += 1;
                            false
                        } else if !seen_indices.insert(r.index) {
                            duplicate += 1;
                            false
                        } else {
                            true
                        }
                    })
                    .map(|r| (docs_with_indices[r.index].0, r.relevance_score))
                    .collect();

                if out_of_range > 0 || duplicate > 0 {
                    warn!(
                        "Reranker response had {out_of_range} out-of-range and \
                         {duplicate} duplicate indices (of {} entries against {} \
                         documents sent); dropping them and continuing with the \
                         remaining {} entries",
                        ranked.len(),
                        docs_with_indices.len(),
                        indexed.len(),
                    );
                }

                // #210: a `200` response whose entries are ALL invalid (every
                // index out of range, or repeats of the same handful of valid
                // ones) leaves `indexed` empty despite the reranker technically
                // succeeding — a deliberately non-empty `ranked` guards against
                // conflating this with "the reranker legitimately had nothing to
                // rank" (which can't happen here: `docs` was already checked
                // non-empty above, so a `ranked.is_empty()` response is itself
                // just as "unusable" as an all-invalid one and falls into this
                // branch too). Before this fix, that condition filtered the
                // caller down to a near-empty page even though the fused
                // ranking computed above is a perfectly serviceable fallback —
                // the SAME "reranker is unusable" condition the `Err` arm below
                // already degrades gracefully from. Converge the two: fall back
                // to fused order here exactly as the `Err` arm does, rather than
                // let a `200`-with-garbage-body and a network failure produce
                // two very different user-visible outcomes.
                if indexed.is_empty() {
                    warn!(
                        "Reranker returned {} entries but none were usable ({out_of_range} \
                         out-of-range, {duplicate} duplicate, against {} documents sent); \
                         falling back to fused order",
                        ranked.len(),
                        docs_with_indices.len(),
                    );
                    if let Some(max_per_document) = opts.diversity_max_per_document {
                        results = apply_diversity_cap(results, max_per_document);
                    }
                    results.truncate(retain_depth);
                    let results = paginate(results, offset, opts.limit);
                    // #245: a third `Ok(SearchOutcome)` return that bypasses the
                    // timing `debug!`/`record_query` call at the bottom of this
                    // function — the rerank WAS attempted (docs was non-empty),
                    // it just came back entirely unusable, which is the same
                    // "reranker is unusable" outcome the `Err(e)` arm below
                    // already records as a failure.
                    let rerank_ms = rerank_start.elapsed().as_millis() as u64;
                    QUERY_METRICS.record_query(
                        embed_ms as u64,
                        search_ms as u64,
                        Some((rerank_ms, false)),
                        results.len(),
                    );
                    return Ok(SearchOutcome {
                        results,
                        path_prefix_truncated,
                        offset_truncated,
                    });
                }

                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Diversity, pass 2 of 2 — the real, user-tunable cap, applied to
                // the cross-encoder's own ranking (the best relevance signal
                // available) and BEFORE truncating to `retain_depth`. Order matters
                // here: capping before truncation lets the next-best chunk from an
                // under-represented document backfill the slot a capped chunk
                // would have taken, rather than just shrinking the result count.
                if let Some(max_per_document) = opts.diversity_max_per_document {
                    indexed = cap_reranked_by_document(indexed, &results, max_per_document);
                }

                results = indexed
                    .into_iter()
                    .take(retain_depth)
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
                rerank_outcome = Some((rerank_start.elapsed().as_millis() as u64, true));
            }
            Err(e) => {
                warn!("Reranker unavailable, falling back to fused order: {e:#}");
                rerank_outcome = Some((rerank_start.elapsed().as_millis() as u64, false));
                // Fused order stands in for a relevance ranking here, same as the
                // no-reranker branch below — cap it the same way before truncating.
                if let Some(max_per_document) = opts.diversity_max_per_document {
                    results = apply_diversity_cap(results, max_per_document);
                }
                results.truncate(retain_depth);
            }
        }
    } else {
        // No reranker: the fused (hybrid) or dense-cosine order IS the relevance
        // ranking, so the diversity cap applies directly to it, before truncating
        // to `page_depth` — same backfill reasoning as pass 2 above (capping
        // pre-truncation lets a lower-ranked, under-represented document's
        // chunk fill the slot a capped chunk would have taken). `page_depth`
        // rather than `opts.limit` for the same reason as the reranker branch:
        // there needs to be something left for `offset` to page into below.
        if let Some(max_per_document) = opts.diversity_max_per_document {
            results = apply_diversity_cap(results, max_per_document);
        }
        results.truncate(page_depth as usize);
    }

    // Offset applies as the very LAST step of the funnel — see this function's
    // doc comment for why it can't run any earlier than this. Everything
    // above (fusion, min_score, reranking, diversity) has already settled the
    // true best-first order by this point; `paginate` only windows into it.
    let results = paginate(results, offset, opts.limit);

    debug!(
        embed_ms,
        search_ms,
        results = results.len(),
        "search timing"
    );
    QUERY_METRICS.record_query(
        embed_ms as u64,
        search_ms as u64,
        rerank_outcome,
        results.len(),
    );

    Ok(SearchOutcome {
        results,
        path_prefix_truncated,
        offset_truncated,
    })
}

/// Query-mode search with no result paging (`offset` fixed at 0) — see
/// [`search_paged`] for the funnel/offset-bound reasoning and the
/// paging-capable version `search_chunks`/`search_grouped` in `mcp.rs` call
/// to honor the `search` tool's `offset` parameter (#224). Kept as a
/// separate, thin entry point so this crate's non-MCP callers — the CLI
/// `search` subcommand (`main.rs`), the write-path dedup gate (`write.rs`),
/// the web UI's `/api/search` (`web.rs`), and the eval harness (`eval.rs`) —
/// don't need to carry an `offset` argument they will never set.
pub async fn search<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &RetrievalDeps<'_, E, Q>,
    query: &str,
    filters: &SearchFilters,
    opts: &SearchOptions,
) -> Result<SearchOutcome, SearchError> {
    search_paged(deps, query, filters, opts, 0).await
}

/// One document from a query+document (grouped) search: Qdrant's best-scoring chunk
/// for that document, hydrated to document-shaped metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupedDocument {
    pub score: f32,
    pub summary: DocumentSummary,
}

/// The results of [`search_grouped`] plus whether a `path_prefix` filter may have
/// under-returned relative to `limit` — same meaning as [`SearchOutcome`]'s field
/// of the same name, just for grouped-document results instead of chunks.
#[derive(Debug, Clone)]
pub struct GroupedSearchOutcome {
    pub documents: Vec<GroupedDocument>,
    pub path_prefix_truncated: bool,
    /// True when the caller's requested `offset + limit` exceeded
    /// [`MAX_OFFSET_DEPTH`] — same meaning as [`SearchOutcome::offset_truncated`],
    /// just for grouped-document results. This path has no reranker and no
    /// `rerank_candidate_limit` bound (see [`search_grouped`]'s doc comment),
    /// so `MAX_OFFSET_DEPTH` is the only depth cap that ever applies here.
    pub offset_truncated: bool,
}

/// Semantic search collapsed to one result per document — the `search` tool's
/// query+document granularity. Groups a Qdrant query by `file_path` with
/// `group_size` 1, so each group's single hit already IS that document's
/// best-scoring chunk, then hydrates every hit's title/description/mtime/
/// frontmatter from `document_index`: Qdrant's own payload only carries the
/// winning chunk's fields, not the whole-document metadata `list_documents` uses.
///
/// Runs the SAME dense/sparse/phrase retrieval arms as [`search`] (honoring
/// `opts.hybrid` and `opts.phrase` identically), fused via server-side RRF and
/// then grouped — see [`crate::qdrant::QdrantStore::search_grouped`]'s doc comment
/// for why: grouped and chunk results must differ only in shape, one row per
/// document vs one per chunk, never in which documents are retrievable. Does not
/// run a reranker, matching `search`'s own reranker-free path when hybrid/phrase
/// fusion order is the only relevance signal available. Also carries no
/// exhaustive total: grouped vector search has no notion of "how many documents
/// match", so the `search` tool's response for this combination reports `returned`
/// but omits `total`/`has_more` rather than claim a count this path cannot back up.
///
/// `offset` (#224): same end-of-funnel placement as [`search_paged`], but
/// simpler — this path runs no reranker and no diversity pass (one row per
/// document already), so the only depth bound is the absolute
/// [`MAX_OFFSET_DEPTH`] ceiling; there is no `rerank_candidate_limit` to also
/// consider. Unlike [`search`]/[`search_paged`], there is no separate
/// `offset`-free wrapper here — this crate's only caller
/// (`mcp.rs`'s `search_grouped` tool handler) always has an `offset` in hand
/// (`0` when the caller didn't ask for paging), so a second entry point would
/// have no real caller of its own.
pub async fn search_grouped<E: QueryEmbedder, Q: RetrievalStore, D: DocumentIndex>(
    deps: &RetrievalDeps<'_, E, Q>,
    document_index: &D,
    query: &str,
    filters: &SearchFilters,
    opts: &SearchOptions,
    fields: Option<&[String]>,
    offset: u64,
) -> Result<GroupedSearchOutcome, SearchError> {
    // See `search_paged`'s identical block above — same phrase-flattening contract.
    let (flat_query, phrases) = if opts.phrase {
        extract_phrases(query)
    } else {
        (query.to_string(), Vec::new())
    };

    let embed_start = std::time::Instant::now();
    let vector = deps
        .embed_client
        .embed_query(&flat_query)
        .await
        .map_err(SearchError::Embed)?;
    let embed_ms = embed_start.elapsed().as_millis();

    let sparse = if opts.hybrid {
        let sparse = crate::sparse::tokenize(&flat_query);
        if sparse.0.is_empty() {
            None
        } else {
            Some(sparse)
        }
    } else {
        None
    };

    let filter_map = mtime_filter_map(opts);
    // The caller-resolved path set, reused for every decision below, exactly as in
    // `search_paged` — see `PathFilter`'s doc comment.
    let path_filter = opts.path_filter.as_ref();
    // Same exact-or-legacy-escape condition `search_paged` ANDs in — see
    // `path_filter_condition`'s doc comment.
    let mut extra_conditions = filters.conditions.clone();
    extra_conditions.extend(path_filter_condition(path_filter));

    // See `search_paged`'s doc comment for the full reasoning — `page_depth` is
    // how many ranked candidates the funnel needs to produce to serve
    // `offset + limit`. No reranker exists on this path, so `MAX_OFFSET_DEPTH`
    // is the only bound (`search_paged`'s other, reranker-specific bound
    // doesn't apply here).
    let requested_depth = offset.saturating_add(opts.limit);
    let offset_truncated = requested_depth > MAX_OFFSET_DEPTH;
    let page_depth = requested_depth.min(MAX_OFFSET_DEPTH);

    let fetch_limit = path_prefix_fetch_limit(page_depth, path_filter);

    let search_start = std::time::Instant::now();
    let mut results = deps
        .qdrant
        .search_grouped(
            deps.collection,
            vector,
            sparse,
            &phrases,
            filter_map,
            extra_conditions,
            "file_path",
            1,
            fetch_limit,
            opts.rrf_candidates,
        )
        .await
        .map_err(SearchError::Search)?;
    let search_ms = search_start.elapsed().as_millis();

    let pre_prefix_count = results.len() as u64;
    apply_path_filter(&mut results, path_filter, deps.data_path);
    // Measured against `page_depth`, not `opts.limit` — see `search_paged`'s
    // identical adjustment for why.
    let path_prefix_truncated = path_prefix_truncated(
        path_filter,
        page_depth,
        fetch_limit,
        pre_prefix_count,
        results.len() as u64,
    );

    if let Some(s) = opts.min_score {
        results.retain(|r| r.score >= s);
    }

    // Preserve Qdrant's score-descending order through the SQLite round trip below:
    // record each hit's (relative path, score) here, look summaries up in whatever
    // order `get_summaries_by_paths` returns them, then re-walk this ranked list to
    // reassemble the final order — rather than trust the SQL result's row order.
    let ranked_paths: Vec<(String, f32)> = results
        .iter()
        .filter_map(|r| {
            let raw = r.payload.get("file_path").and_then(|v| v.as_str())?;
            Some((relative_to_data(raw, deps.data_path), r.score))
        })
        .collect();

    let paths: Vec<String> = ranked_paths.iter().map(|(p, _)| p.clone()).collect();
    let summaries = document_index
        .get_summaries_by_paths(&paths, fields)
        .await
        .map_err(SearchError::Document)?;
    let mut by_path: HashMap<String, DocumentSummary> = summaries
        .into_iter()
        .map(|s| (s.file_path.clone(), s))
        .collect();

    // A path absent from `by_path` (the metadata index transiently behind Qdrant —
    // same caveat as `get_document`'s fuzzy fallback) is skipped rather than
    // fabricated: a document search that can't back its own claim with real
    // metadata about it.
    let mut documents: Vec<GroupedDocument> = ranked_paths
        .into_iter()
        .filter_map(|(path, score)| {
            by_path
                .remove(&path)
                .map(|summary| GroupedDocument { score, summary })
        })
        .collect();

    // `fetch_limit` above may be an over-fetch (path_prefix multiplies it up to
    // 500, and `page_depth` itself grows for `offset`), and prefix/min_score
    // filtering can leave more survivors than the caller actually asked for.
    // Truncate to `page_depth` — not `opts.limit` — first, mirroring
    // `search_paged`'s own retain-then-paginate split, then apply `offset` as
    // the very last step now that this list is in its final settled order.
    documents.truncate(page_depth as usize);
    let documents = paginate(documents, offset, opts.limit);

    debug!(
        embed_ms,
        search_ms,
        results = documents.len(),
        "search timing (grouped)"
    );
    // #245: this path has no reranker at all, so `rerank` is always `None` —
    // unlike `search_paged`, `search_grouped` has exactly one return point (no
    // early "zero candidates" branch, since there is no reranker candidate
    // pool to be empty), so one call site here covers every query.
    QUERY_METRICS.record_query(embed_ms as u64, search_ms as u64, None, documents.len());

    Ok(GroupedSearchOutcome {
        documents,
        path_prefix_truncated,
        offset_truncated,
    })
}

/// Resolve a document path and return its full content, plus its link-graph
/// neighborhood (`links_out`/`links_in` — #157).
///
/// On success returns `Document { path, content, links_out, links_in }`. On
/// failure returns a structured `GetDocumentError` — callers are responsible for
/// turning these into user-facing error strings.
///
/// `document_index` backs the fuzzy fallback (see step 2 below) — the SQLite
/// `documents` table, not Qdrant. See that step's comment for why, and for the
/// consistency tradeoff that follows from the choice. It also now backs the
/// link-graph lookup (step 3), unconditionally on every successful resolution —
/// see [`document_links`]'s doc comment for why a failure there degrades instead
/// of propagating.
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
            let rel_path = relative_to_data(&canonical.to_string_lossy(), deps.data_path);
            let (links_out, links_in) = document_links(document_index, &rel_path).await;
            return Ok(Document {
                path: canonical,
                content,
                links_out,
                links_in,
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
                let rel_path = relative_to_data(&canonical.to_string_lossy(), deps.data_path);
                let (links_out, links_in) = document_links(document_index, &rel_path).await;
                Ok(Document {
                    path: canonical,
                    content,
                    links_out,
                    links_in,
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

/// Fetch `links_out`/`links_in` for `rel_path` (a `documents`-table-relative path,
/// matching what `document_links.source_path`/`target_path` store), each capped at
/// [`MAX_LINKS_PER_DIRECTION`].
///
/// Failures degrade to an empty, zero-total page rather than propagating —
/// `get_document` is a hot read path, and the link graph is supplementary to the
/// document body it serves: a transient failure on either of these two queries
/// must not turn an otherwise-successful read into an error. Same non-fatal
/// posture as the fuzzy fallback's `all_paths()` call above, and the same
/// rationale as `capped_casualties` in `mcp.rs` for why the cap is reported
/// (`LinkPage::total`/`has_more`) rather than silently applied.
async fn document_links<D: DocumentIndex>(
    document_index: &D,
    rel_path: &str,
) -> (LinkPage<OutboundLink>, LinkPage<InboundLink>) {
    let links_out = document_index
        .links_out(rel_path, MAX_LINKS_PER_DIRECTION)
        .await
        .unwrap_or_else(|e| {
            warn!(path = %rel_path, "Failed to fetch outbound links: {e:#}");
            LinkPage {
                links: Vec::new(),
                total: 0,
            }
        });
    let links_in = document_index
        .links_in(rel_path, MAX_LINKS_PER_DIRECTION)
        .await
        .unwrap_or_else(|e| {
            warn!(path = %rel_path, "Failed to fetch inbound links: {e:#}");
            LinkPage {
                links: Vec::new(),
                total: 0,
            }
        });
    (links_out, links_in)
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
    // extract_phrases
    // ------------------------------------------------------------------

    #[test]
    fn extract_phrases_unquoted_query_is_returned_completely_unchanged() {
        // The main regression risk: an unquoted query must behave EXACTLY as it
        // did before phrase parsing existed. Includes leading/trailing/internal
        // whitespace to prove nothing gets trimmed or rebuilt along the way.
        let query = "  deploy notes for node:ares rocm  ";
        let (flat, phrases) = extract_phrases(query);
        assert_eq!(flat, query);
        assert!(phrases.is_empty());
    }

    #[test]
    fn extract_phrases_pulls_a_single_quoted_span() {
        let (flat, phrases) = extract_phrases(r#"deploy notes for "node:ares" rocm"#);
        assert_eq!(flat, "deploy notes for node:ares rocm");
        assert_eq!(phrases, vec!["node:ares".to_string()]);
    }

    #[test]
    fn extract_phrases_collects_multiple_quoted_spans_in_order() {
        let (flat, phrases) = extract_phrases(r#""node:ares" and "rocm driver" notes"#);
        assert_eq!(flat, "node:ares and rocm driver notes");
        assert_eq!(
            phrases,
            vec!["node:ares".to_string(), "rocm driver".to_string()]
        );
    }

    #[test]
    fn extract_phrases_trims_whitespace_inside_a_phrase() {
        let (_, phrases) = extract_phrases(r#""  node:ares  ""#);
        assert_eq!(phrases, vec!["node:ares".to_string()]);
    }

    #[test]
    fn extract_phrases_unterminated_quote_is_a_literal_character_not_an_error() {
        let query = r#"deploy notes for "node:ares rocm"#;
        let (flat, phrases) = extract_phrases(query);
        assert_eq!(
            flat, query,
            "a dangling quote with no close must leave the string untouched"
        );
        assert!(phrases.is_empty());
    }

    #[test]
    fn extract_phrases_blank_quoted_span_contributes_no_phrase() {
        let (_, phrases) = extract_phrases(r#"hello "" world"#);
        assert!(phrases.is_empty());
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
            phrase_score: None,
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

    #[test]
    fn resolve_within_data_leading_slash_traversal_reports_the_same_error_class_as_relative_traversal()
     {
        // #225: `../x` (relative) and `/../x` (leading-slash) are the same
        // traversal attempt read two different ways, and neither `x` is
        // expected to exist anywhere reachable — a sibling of the tempdir
        // named `x`, or a top-level `/x` — so both exercise the
        // failing-`canonicalize()` path this test guards, rather than the
        // "target happens to exist" path `resolve_within_data_enforces_boundaries`
        // above already covers via `/etc/hosts`.
        //
        // Before the fix, `/../x` fell into the absolute-path branch, failed
        // the literal `canonicalize()` with `NotFound`, and returned that
        // `NotFound` directly — the traversal check a few lines below never
        // ran — while the relative shape already correctly returned `Outside`.
        // Same rejection either way (no security impact), but a caller or a
        // developer debugging path handling was told two different things for
        // the same mistake.
        let tmp = tempfile::tempdir().unwrap();
        let data_path = tmp.path().canonicalize().unwrap();
        let mut builder = globset::GlobSetBuilder::new();
        builder.add(globset::Glob::new("**/*.md").unwrap());
        let include = builder.build().unwrap();

        let relative = resolve_within_data("../x", &data_path, &include);
        let leading_slash = resolve_within_data("/../x", &data_path, &include);

        assert!(
            matches!(relative, Err(ResolveErr::Outside)),
            "relative traversal to a nonexistent target should report Outside, \
             got a different variant"
        );
        assert!(
            matches!(leading_slash, Err(ResolveErr::Outside)),
            "leading-slash traversal to a nonexistent target should report \
             Outside — the SAME class as the relative shape above, not NotFound"
        );
    }

    // ------------------------------------------------------------------
    // Mock types for retrieval unit tests
    // ------------------------------------------------------------------

    struct MockEmbedder {
        err: Option<String>,
        ok: Vec<f32>,
        /// The exact string `embed_query` was last called with — lets phrase tests
        /// assert the flattened (dequoted) text is what gets embedded, not the raw
        /// query.
        last_query: std::sync::Mutex<Option<String>>,
    }

    impl MockEmbedder {
        fn ok(v: Vec<f32>) -> Self {
            Self {
                err: None,
                ok: v,
                last_query: std::sync::Mutex::new(None),
            }
        }
        fn err(msg: &str) -> Self {
            Self {
                err: Some(msg.to_string()),
                ok: vec![],
                last_query: std::sync::Mutex::new(None),
            }
        }
    }

    impl QueryEmbedder for MockEmbedder {
        async fn embed_query(&self, q: &str) -> anyhow::Result<Vec<f32>> {
            *self.last_query.lock().unwrap() = Some(q.to_string());
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
        /// Sparse vector captured by `hybrid_search`/`search_grouped` (None until
        /// called, or if called with no sparse arm).
        received_sparse: std::sync::Mutex<Option<(Vec<u32>, Vec<f32>)>>,
        /// Phrases captured by `hybrid_search`/`search_grouped` (empty until called
        /// with a non-empty phrase list).
        received_phrases: std::sync::Mutex<Vec<String>>,
        /// Which method was last invoked: "search", "hybrid_search", or "search_grouped".
        last_call: std::sync::Mutex<Option<&'static str>>,
        /// `extra_conditions` captured by whichever method was last invoked.
        received_conditions: std::sync::Mutex<Option<Vec<Condition>>>,
    }

    impl MockRetrievalStore {
        fn with_results(results: Vec<SearchResult>) -> Self {
            Self {
                search_err: None,
                search_ok: results,
                received_filters: std::sync::Mutex::new(None),
                received_sparse: std::sync::Mutex::new(None),
                received_phrases: std::sync::Mutex::new(Vec::new()),
                last_call: std::sync::Mutex::new(None),
                received_conditions: std::sync::Mutex::new(None),
            }
        }
        fn with_search_err(msg: &str) -> Self {
            Self {
                search_err: Some(msg.to_string()),
                search_ok: Vec::new(),
                received_filters: std::sync::Mutex::new(None),
                received_sparse: std::sync::Mutex::new(None),
                received_phrases: std::sync::Mutex::new(Vec::new()),
                last_call: std::sync::Mutex::new(None),
                received_conditions: std::sync::Mutex::new(None),
            }
        }
    }

    impl RetrievalStore for MockRetrievalStore {
        async fn search(
            &self,
            _collection: &str,
            _vector: Vec<f32>,
            filters: HashMap<String, serde_json::Value>,
            extra_conditions: Vec<Condition>,
            _limit: u64,
        ) -> anyhow::Result<Vec<SearchResult>> {
            *self.last_call.lock().unwrap() = Some("search");
            *self.received_filters.lock().unwrap() = Some(filters);
            *self.received_conditions.lock().unwrap() = Some(extra_conditions);
            if let Some(ref msg) = self.search_err {
                anyhow::bail!("{}", msg);
            }
            Ok(self.search_ok.clone())
        }

        async fn hybrid_search(
            &self,
            _collection: &str,
            _dense: Vec<f32>,
            sparse: Option<(Vec<u32>, Vec<f32>)>,
            phrases: &[String],
            filters: HashMap<String, serde_json::Value>,
            extra_conditions: Vec<Condition>,
            _limit: u64,
            _rrf_candidates: u64,
            _explain: bool,
        ) -> anyhow::Result<Vec<SearchResult>> {
            *self.last_call.lock().unwrap() = Some("hybrid_search");
            *self.received_filters.lock().unwrap() = Some(filters);
            *self.received_sparse.lock().unwrap() = sparse;
            *self.received_phrases.lock().unwrap() = phrases.to_vec();
            *self.received_conditions.lock().unwrap() = Some(extra_conditions);
            if let Some(ref msg) = self.search_err {
                anyhow::bail!("{}", msg);
            }
            Ok(self.search_ok.clone())
        }

        async fn search_grouped(
            &self,
            _collection: &str,
            _vector: Vec<f32>,
            sparse: Option<(Vec<u32>, Vec<f32>)>,
            phrases: &[String],
            filters: HashMap<String, serde_json::Value>,
            extra_conditions: Vec<Condition>,
            _group_by: &str,
            _group_size: u64,
            limit: u64,
            _rrf_candidates: u64,
        ) -> anyhow::Result<Vec<SearchResult>> {
            *self.last_call.lock().unwrap() = Some("search_grouped");
            *self.received_filters.lock().unwrap() = Some(filters);
            *self.received_sparse.lock().unwrap() = sparse;
            *self.received_phrases.lock().unwrap() = phrases.to_vec();
            *self.received_conditions.lock().unwrap() = Some(extra_conditions);
            if let Some(ref msg) = self.search_err {
                anyhow::bail!("{}", msg);
            }
            // Real Qdrant never returns more groups than the requested `limit` —
            // match that here so a test that over-configures `search_ok` still
            // reflects the (possibly over-fetched) count `search_grouped` actually
            // asked for, rather than silently handing back everything preconfigured.
            let mut results = self.search_ok.clone();
            results.truncate(limit as usize);
            Ok(results)
        }
    }

    /// `DocumentIndex` mock for `get_document`'s fuzzy-fallback tests and
    /// `search_grouped`'s hydration tests.
    struct MockDocumentIndex {
        paths: Vec<String>,
        err: Option<String>,
        /// Summaries `get_summaries_by_paths` filters down to the requested paths —
        /// `query_documents` and `all_paths` don't need this, so it stays empty for
        /// the fuzzy-fallback tests that only ever construct `with_paths`/`with_err`.
        summaries: Vec<DocumentSummary>,
        /// Canned `links_out`/`links_in` pages, deliberately keyed on nothing but the
        /// call itself (every path gets the same canned answer) — the link-lookup
        /// tests below only ever query a single document, so this stays simple
        /// rather than modeling a per-path graph.
        outbound: Vec<OutboundLink>,
        inbound: Vec<InboundLink>,
        /// Separate from `err`: `err` fails `all_paths`/`get_summaries_by_paths`
        /// (simulating the metadata index being unavailable at all), which would
        /// also fail `get_document`'s fuzzy fallback and isn't what the
        /// state-DB-unavailable link test wants to exercise — that test wants the
        /// document resolve to succeed while only the link queries fail, to prove
        /// `get_document` degrades rather than propagating.
        links_err: Option<String>,
    }

    impl MockDocumentIndex {
        fn with_paths(paths: Vec<String>) -> Self {
            Self {
                paths,
                err: None,
                summaries: Vec::new(),
                outbound: Vec::new(),
                inbound: Vec::new(),
                links_err: None,
            }
        }
        fn with_err(msg: &str) -> Self {
            Self {
                paths: Vec::new(),
                err: Some(msg.to_string()),
                summaries: Vec::new(),
                outbound: Vec::new(),
                inbound: Vec::new(),
                links_err: None,
            }
        }
        fn with_summaries(summaries: Vec<DocumentSummary>) -> Self {
            Self {
                paths: Vec::new(),
                err: None,
                summaries,
                outbound: Vec::new(),
                inbound: Vec::new(),
                links_err: None,
            }
        }
        /// Paths resolve normally, but `links_out`/`links_in` both return `msg` as
        /// an error — the state-DB-unavailable-for-links case.
        fn with_paths_and_links_err(paths: Vec<String>, msg: &str) -> Self {
            Self {
                paths,
                err: None,
                summaries: Vec::new(),
                outbound: Vec::new(),
                inbound: Vec::new(),
                links_err: Some(msg.to_string()),
            }
        }
        /// Paths resolve normally and `links_out`/`links_in` return the given
        /// canned pages — the happy-path wiring test.
        fn with_paths_and_links(
            paths: Vec<String>,
            outbound: Vec<OutboundLink>,
            inbound: Vec<InboundLink>,
        ) -> Self {
            Self {
                paths,
                err: None,
                summaries: Vec::new(),
                outbound,
                inbound,
                links_err: None,
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

        async fn paths_matching(&self, needle: &str, cap: u64) -> anyhow::Result<PathMatches> {
            if let Some(ref msg) = self.err {
                anyhow::bail!("{}", msg);
            }
            // Case-insensitive substring, mirroring what SQLite's `LIKE` gives the
            // real implementation.
            let needle = needle.to_lowercase();
            let all: Vec<String> = self
                .paths
                .iter()
                .filter(|p| p.to_lowercase().contains(&needle))
                .cloned()
                .collect();
            let total = all.len() as u64;
            Ok(PathMatches {
                paths: all.into_iter().take(cap as usize).collect(),
                total,
            })
        }

        async fn get_summaries_by_paths(
            &self,
            paths: &[String],
            _fields: Option<&[String]>,
        ) -> anyhow::Result<Vec<DocumentSummary>> {
            if let Some(ref msg) = self.err {
                anyhow::bail!("{}", msg);
            }
            Ok(self
                .summaries
                .iter()
                .filter(|s| paths.contains(&s.file_path))
                .cloned()
                .collect())
        }

        async fn links_out(
            &self,
            _source_path: &str,
            limit: u64,
        ) -> anyhow::Result<LinkPage<OutboundLink>> {
            if let Some(ref msg) = self.links_err {
                anyhow::bail!("{}", msg);
            }
            let total = self.outbound.len() as u64;
            let links = self.outbound.iter().take(limit as usize).cloned().collect();
            Ok(LinkPage { links, total })
        }

        async fn links_in(
            &self,
            _target_path: &str,
            limit: u64,
        ) -> anyhow::Result<LinkPage<InboundLink>> {
            if let Some(ref msg) = self.links_err {
                anyhow::bail!("{}", msg);
            }
            let total = self.inbound.len() as u64;
            let links = self.inbound.iter().take(limit as usize).cloned().collect();
            Ok(LinkPage { links, total })
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
            phrase: false,
            explain: false,
            modified_after: None,
            modified_before: None,
            path_filter: None,
            rerank_candidate_limit: None,
            // Off by default in this shared test fixture, even though the shipped
            // config default is `Some(3)` — existing tests built on `default_opts()`
            // predate diversity and assert on undiversified result sets. The
            // diversity-specific tests below opt in explicitly instead.
            diversity_max_per_document: None,
        }
    }

    // ------------------------------------------------------------------
    // path_filter over-fetch / path_prefix_truncated (pure, no network)
    // ------------------------------------------------------------------

    /// A resolved filter over `paths`, nothing hidden by the cap.
    fn pf(paths: &[&str]) -> PathFilter {
        PathFilter {
            paths: paths.iter().map(|p| p.to_string()).collect(),
            truncated: false,
        }
    }

    /// What resolving the needle `keep/` against the `keep/*` + `skip/*` corpus the
    /// truncation tests below build would return: every `keep/` document they use.
    /// Those tests are about what the retain and the truncation signal do to a page,
    /// not about resolution, so they all share one set.
    fn keep_filter() -> PathFilter {
        pf(&["keep/only.md", "keep/a.md", "keep/b.md", "keep/c.md"])
    }

    #[test]
    fn path_prefix_fetch_limit_multiplies_and_caps() {
        let filter = pf(&["x/a.md"]);
        assert_eq!(path_prefix_fetch_limit(10, Some(&filter)), 50);
        assert_eq!(
            path_prefix_fetch_limit(200, Some(&filter)),
            PATH_PREFIX_OVERFETCH_CEILING,
            "a large base fetch_limit must still be capped at the ceiling"
        );
        assert_eq!(
            path_prefix_fetch_limit(10, None),
            10,
            "no path filter means no over-fetch at all"
        );
    }

    #[test]
    fn path_prefix_truncated_true_when_overfetch_saturated_and_short_of_limit() {
        // 10 hits came back for a 10-slot over-fetched page (saturated — no proof
        // the corpus doesn't hold more), and only 3 of those survived the retain,
        // short of the caller's limit of 5.
        assert!(path_prefix_truncated(Some(&pf(&["x/a.md"])), 5, 10, 10, 3));
    }

    #[test]
    fn path_prefix_truncated_false_when_the_fetch_proved_exhaustion() {
        // Qdrant returned fewer hits (3) than the over-fetched limit (10) — there
        // was genuinely nothing more to find, so a post-retain shortfall against
        // `limit` is proven benign rather than a possible blind spot.
        assert!(!path_prefix_truncated(Some(&pf(&["x/a.md"])), 5, 10, 3, 3));
    }

    #[test]
    fn path_prefix_truncated_false_when_enough_results_survived_the_retain() {
        assert!(!path_prefix_truncated(Some(&pf(&["x/a.md"])), 5, 10, 10, 5));
    }

    #[test]
    fn path_prefix_truncated_false_with_no_path_filter() {
        assert!(!path_prefix_truncated(None, 5, 10, 10, 3));
    }

    #[test]
    fn path_prefix_truncated_true_when_resolution_hit_the_cap() {
        // #182: the needle matched more documents than the resolution cap, so the
        // filter itself is an incomplete picture. That is knowable regardless of
        // how this particular page came back — here the page was *not* saturated
        // (3 of 10) and still filled the caller's limit, which under the old
        // page-shape-only signal alone would have reported `false`.
        let capped = PathFilter {
            paths: vec!["x/a.md".to_string()],
            truncated: true,
        };
        assert!(path_prefix_truncated(Some(&capped), 1, 10, 3, 1));
    }

    // ------------------------------------------------------------------
    // path_filter_condition() — exact match-any filter + backward-compat escape
    // ------------------------------------------------------------------

    /// The exact shape `path_filter_condition` must build: an OR of "one of these
    /// exact documents" and "this document was never reindexed under
    /// `path_ancestors`" — see that function's doc comment for why both arms are
    /// required for safety.
    fn expected_path_filter_condition(paths: &[&str]) -> Condition {
        Condition::from(Filter::should([
            Condition::matches(
                PATH_ANCESTORS_KEY,
                paths.iter().map(|p| p.to_string()).collect::<Vec<_>>(),
            ),
            Condition::is_empty(PATH_ANCESTORS_KEY),
        ]))
    }

    #[test]
    fn path_filter_condition_builds_the_exact_or_legacy_escape() {
        // #182: the resolved paths go in as a match-any over the *existing*
        // `path_ancestors` keyword index — each document's own full path is an
        // entry there, so no new index and no reindex is needed to filter exactly.
        assert_eq!(
            path_filter_condition(Some(&pf(&["sysadmin/zfs.md", "sysadmin/dns.md"]))),
            Some(expected_path_filter_condition(&[
                "sysadmin/zfs.md",
                "sysadmin/dns.md"
            ]))
        );
    }

    #[test]
    fn path_filter_condition_none_when_no_filter_given() {
        assert_eq!(path_filter_condition(None), None);
    }

    #[test]
    fn path_filter_condition_asks_for_nothing_when_the_needle_matched_nothing() {
        // Not `None`: `None` means "no filter", which would run an unfiltered
        // search and then throw every result away in the retain. See the
        // function's doc comment.
        assert_eq!(
            path_filter_condition(Some(&pf(&[]))),
            Some(Condition::from(Filter::should([
                crate::qdrant::unsatisfiable(PATH_ANCESTORS_KEY),
                Condition::is_empty(PATH_ANCESTORS_KEY),
            ])))
        );
        assert_ne!(path_filter_condition(Some(&pf(&[]))), None);
    }

    #[test]
    fn apply_path_filter_keeps_only_set_members() {
        // #182: exact membership, not a prefix guess. `keep/only.md` is in the
        // resolved set; `keeper/other.md` shares its opening characters and must
        // still be dropped, which a `starts_with` check would have got wrong.
        let mut results = vec![
            make_result_for("keep/only.md", 0.9),
            make_result_for("keeper/other.md", 0.8),
            make_result_for("skip/x.md", 0.7),
        ];
        apply_path_filter(
            &mut results,
            Some(&pf(&["keep/only.md"])),
            Path::new("/data"),
        );

        assert_eq!(file_paths_of(&results), vec!["keep/only.md"]);
    }

    #[test]
    fn apply_path_filter_drops_everything_when_the_needle_matched_nothing() {
        // Pairs with `path_filter_condition_none_when_the_needle_matched_nothing`:
        // no Qdrant condition is added in that case, so this retain is the only
        // thing enforcing "nothing matched".
        let mut results = vec![make_result_for("keep/only.md", 0.9)];
        apply_path_filter(&mut results, Some(&pf(&[])), Path::new("/data"));
        assert!(results.is_empty());
    }

    #[test]
    fn apply_path_filter_is_a_no_op_without_a_filter() {
        let mut results = vec![
            make_result_for("keep/only.md", 0.9),
            make_result_for("skip/x.md", 0.8),
        ];
        apply_path_filter(&mut results, None, Path::new("/data"));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn normalize_path_needle_trims_every_trailing_slash() {
        // `strip_suffix('/')` would leave "sysadmin/" here.
        assert_eq!(normalize_path_needle(Some("sysadmin//")), Some("sysadmin"));
        assert_eq!(normalize_path_needle(Some("sysadmin/")), Some("sysadmin"));
        assert_eq!(normalize_path_needle(Some("sysadmin")), Some("sysadmin"));
        assert_eq!(normalize_path_needle(Some("///")), None);
        assert_eq!(normalize_path_needle(Some("")), None);
        assert_eq!(normalize_path_needle(None), None);
        // Idempotent — both search modes apply it, so it must be a fixed point.
        assert_eq!(
            normalize_path_needle(normalize_path_needle(Some("a/b//"))),
            Some("a/b")
        );
    }

    // ------------------------------------------------------------------
    // search() unit tests with mocks
    // ------------------------------------------------------------------

    /// Regression: a trailing slash must not change what comes back.
    ///
    /// The original bug was a disagreement between two enforcement points —
    /// the Qdrant condition normalized its prefix and the retain did not, so a
    /// prefix naming a whole document (`.../efi.md/`) matched Qdrant-side and was
    /// then dropped by `"…/efi.md".starts_with("…/efi.md/")`. #182 removes the
    /// disagreement structurally: there is one needle, normalized once, resolved
    /// once, and retrieval only ever sees the resolved answer.
    ///
    /// Still exercised end-to-end through the whole pipeline — normalize, resolve
    /// against the index, search — rather than by comparing two normalized strings,
    /// because the string-level test above passes either way, which is exactly why
    /// it never caught the original.
    #[tokio::test]
    async fn search_path_prefix_results_are_identical_with_and_without_a_trailing_slash() {
        async fn results_for(prefix: &str) -> Vec<String> {
            let index = MockDocumentIndex::with_paths(vec![
                "sysadmin/nodes/ares/boot/efi.md".to_string(),
                "food/recipes/chili.md".to_string(),
            ]);
            let needle = normalize_path_needle(Some(prefix));
            let path_filter = match needle {
                Some(n) => Some(PathFilter::from(
                    index
                        .paths_matching(n, PATH_FILTER_MAX_PATHS)
                        .await
                        .unwrap(),
                )),
                None => None,
            };

            let store = MockRetrievalStore::with_results(vec![
                make_result_for("sysadmin/nodes/ares/boot/efi.md", 0.9),
                make_result_for("food/recipes/chili.md", 0.8),
            ]);
            let embed = MockEmbedder::ok(vec![0.1]);
            let gs = make_md_globset();
            let data_path = Path::new("/data");
            let deps = make_deps(&embed, &store, data_path, &gs);
            let opts = SearchOptions {
                path_filter,
                ..default_opts()
            };
            search(&deps, "q", &SearchFilters::default(), &opts)
                .await
                .unwrap()
                .results
                .iter()
                .map(|r| {
                    r.payload
                        .get("file_path")
                        .and_then(|v| v.as_str())
                        .unwrap()
                        .to_string()
                })
                .collect()
        }

        // A prefix naming one specific document — the shape that broke.
        let bare = results_for("sysadmin/nodes/ares/boot/efi.md").await;
        let slashed = results_for("sysadmin/nodes/ares/boot/efi.md/").await;
        assert_eq!(
            bare,
            vec!["sysadmin/nodes/ares/boot/efi.md".to_string()],
            "the document's own full path is a legitimate path_prefix target"
        );
        assert_eq!(
            slashed, bare,
            "a trailing slash must not change the result set"
        );

        // And the ordinary folder shape, including a doubled slash.
        let folder = results_for("sysadmin").await;
        assert_eq!(results_for("sysadmin/").await, folder);
        assert_eq!(results_for("sysadmin//").await, folder);
        assert_eq!(folder.len(), 1);
    }

    #[tokio::test]
    async fn search_path_prefix_truncated_set_when_overfetch_is_saturated() {
        // limit=2, no reranker, so fetch_limit = path_prefix_fetch_limit(2) = 10.
        // Qdrant hands back exactly 10 hits (the full over-fetched page — saturated),
        // of which only 1 is in the resolved set: fewer than `limit` survive the retain,
        // and the saturated fetch means that shortfall is not provably exhaustive.
        let mut results: Vec<SearchResult> = (0..9)
            .map(|i| make_result_for(&format!("skip/{i}.md"), 0.9 - i as f32 * 0.01))
            .collect();
        results.push(make_result_for("keep/only.md", 0.1));
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(results);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let opts = SearchOptions {
            limit: 2,
            path_filter: Some(keep_filter()),
            ..default_opts()
        };
        let outcome = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap();

        assert_eq!(outcome.results.len(), 1);
        assert!(
            outcome.path_prefix_truncated,
            "10 hits filling a 10-slot over-fetched page proves nothing about \
             whether more prefix matches exist beyond it"
        );
    }

    #[tokio::test]
    async fn search_path_prefix_truncated_clear_when_fetch_was_exhaustive() {
        // limit=5, no reranker, so fetch_limit = path_prefix_fetch_limit(5) = 25.
        // Qdrant returns only 2 hits total — nowhere near the 25-slot page — so the
        // fetch is proven exhaustive even though only 1 result survives the retain.
        let results = vec![
            make_result_for("keep/a.md", 0.9),
            make_result_for("skip/b.md", 0.8),
        ];
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(results);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let opts = SearchOptions {
            limit: 5,
            path_filter: Some(keep_filter()),
            ..default_opts()
        };
        let outcome = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap();

        assert_eq!(outcome.results.len(), 1);
        assert!(
            !outcome.path_prefix_truncated,
            "Qdrant returned far fewer hits than the over-fetch limit — exhaustion \
             is proven, so the shortfall must not be reported as truncation"
        );
    }

    // ------------------------------------------------------------------
    // path_prefix_truncated interaction with later result-shrinking stages
    // ------------------------------------------------------------------
    //
    // `path_prefix_truncated` is computed right after the path_prefix retain —
    // strictly before min_score filtering, reranking, and the diversity cap
    // ever run. Each pair below fixes the SAME pre-shrink saturation/exhaustion
    // state as the two tests just above, then adds one later stage that shrinks
    // (even to zero) the results that stage sees. The flag must track only the
    // pre-shrink state, in both directions.

    fn make_result_with_content(file_path: &str, score: f32, content: &str) -> SearchResult {
        let mut r = make_result_for(file_path, score);
        r.payload
            .insert(CHUNK_TEXT_KEY.into(), serde_json::json!(content));
        r
    }

    #[tokio::test]
    async fn search_path_prefix_truncated_stays_true_despite_min_score_emptying_the_page() {
        // Same saturated setup as `..._true_when_overfetch_is_saturated` above,
        // but min_score now removes the one surviving "keep" result entirely.
        let mut results: Vec<SearchResult> = (0..9)
            .map(|i| make_result_for(&format!("skip/{i}.md"), 0.9 - i as f32 * 0.01))
            .collect();
        results.push(make_result_for("keep/only.md", 0.1));
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(results);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let opts = SearchOptions {
            limit: 2,
            min_score: Some(0.5),
            path_filter: Some(keep_filter()),
            ..default_opts()
        };
        let outcome = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap();

        assert!(
            outcome.results.is_empty(),
            "min_score should have dropped the one keep/ result"
        );
        assert!(
            outcome.path_prefix_truncated,
            "the saturated over-fetch is still unresolved; min_score emptying the \
             page afterward must not clear the flag"
        );
    }

    #[tokio::test]
    async fn search_path_prefix_truncated_stays_false_despite_min_score_dropping_below_limit() {
        // limit=2, fetch_limit=10 (saturated: exactly 10 raw hits fill the
        // over-fetched page). Three "keep/" results survive the prefix retain —
        // already >= limit, so the flag is correctly false BEFORE min_score
        // ever runs, regardless of the saturation. min_score then knocks two of
        // the three below the floor, leaving only one — below `limit`. A flag
        // computed from the post-min_score count instead of the post-retain
        // count would wrongly flip this to true.
        let mut results: Vec<SearchResult> = (0..7)
            .map(|i| make_result_for(&format!("skip/{i}.md"), 0.5 - i as f32 * 0.01))
            .collect();
        results.push(make_result_for("keep/a.md", 0.9));
        results.push(make_result_for("keep/b.md", 0.5));
        results.push(make_result_for("keep/c.md", 0.1));
        assert_eq!(
            results.len(),
            10,
            "must exactly saturate the over-fetched page"
        );
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(results);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let opts = SearchOptions {
            limit: 2,
            min_score: Some(0.6),
            path_filter: Some(keep_filter()),
            ..default_opts()
        };
        let outcome = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap();

        assert_eq!(
            outcome.results.len(),
            1,
            "min_score should have knocked out keep/b.md and keep/c.md"
        );
        assert!(
            !outcome.path_prefix_truncated,
            "three keep/ results already survived the retain — enough to satisfy \
             limit — before min_score thinned them further; that later thinning \
             must not spuriously set the flag"
        );
    }

    #[tokio::test]
    async fn search_path_prefix_truncated_stays_false_despite_diversity_thinning_survivors_below_limit()
     {
        // limit=2, fetch_limit=10 (saturated: exactly 10 raw hits). Both
        // "keep/only.md" hits survive the prefix retain, already satisfying
        // `limit` on their own — a correctly-false case BEFORE diversity ever
        // runs, regardless of the saturation. They share one file_path, so a
        // diversity cap of 1 thins them to 1 — below `limit`. A flag computed
        // from the post-diversity count instead of the post-retain count would
        // wrongly flip this to true.
        let mut results: Vec<SearchResult> = (0..8)
            .map(|i| make_result_for(&format!("skip/{i}.md"), 0.5 - i as f32 * 0.01))
            .collect();
        results.push(make_result_for("keep/only.md", 0.9));
        results.push(make_result_for("keep/only.md", 0.8));
        assert_eq!(
            results.len(),
            10,
            "must exactly saturate the over-fetched page"
        );
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(results);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let opts = SearchOptions {
            limit: 2,
            path_filter: Some(keep_filter()),
            diversity_max_per_document: Some(1),
            ..default_opts()
        };
        let outcome = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap();

        assert_eq!(
            outcome.results.len(),
            1,
            "the diversity cap should have thinned the two same-document survivors to one"
        );
        assert!(
            !outcome.path_prefix_truncated,
            "the path_prefix retain already produced enough matches to satisfy limit, \
             and the fetch was exhaustive — the diversity cap thinning that further \
             must not turn the flag on"
        );
    }

    #[tokio::test]
    async fn search_path_prefix_truncated_stays_true_despite_diversity_thinning_further() {
        // Saturated fetch (15 hits filling a 15-slot over-fetched page for
        // limit=3): only 2 of them match the prefix, short of `limit`, so the
        // flag is already true before diversity runs. Both survivors share one
        // file_path, so a diversity cap of 1 thins them further, to 1.
        let mut results: Vec<SearchResult> = (0..13)
            .map(|i| make_result_for(&format!("skip/{i}.md"), 0.9 - i as f32 * 0.01))
            .collect();
        results.push(make_result_for("keep/only.md", 0.5));
        results.push(make_result_for("keep/only.md", 0.4));
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(results);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let opts = SearchOptions {
            limit: 3,
            path_filter: Some(keep_filter()),
            diversity_max_per_document: Some(1),
            ..default_opts()
        };
        let outcome = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap();

        assert_eq!(outcome.results.len(), 1);
        assert!(
            outcome.path_prefix_truncated,
            "the saturated over-fetch is still unresolved; the diversity cap thinning \
             the survivors further must not clear the flag"
        );
    }

    #[tokio::test]
    async fn search_path_prefix_truncated_stays_true_through_the_reranker_branch() {
        // Same saturated shape as the plain-search "true" test above, but with a
        // reranker configured — a different code path (rerank_candidate_limit
        // feeds fetch_limit instead of limit, and results flow through
        // `reranker.rerank` before the final truncate) that must not recompute
        // or corrupt a flag already settled before it ever runs.
        let mut results: Vec<SearchResult> = (0..9)
            .map(|i| make_result_with_content(&format!("skip/{i}.md"), 0.9 - i as f32 * 0.01, "s"))
            .collect();
        results.push(make_result_with_content("keep/only.md", 0.1, "k"));
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(results);
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
            limit: 2,
            rerank_candidate_limit: None,
            path_filter: Some(keep_filter()),
            ..default_opts()
        };
        let outcome = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap();

        assert_eq!(outcome.results.len(), 1);
        assert!(
            outcome.path_prefix_truncated,
            "the saturated over-fetch is still unresolved; taking the reranker \
             branch must not clear the flag"
        );
    }

    #[tokio::test]
    async fn search_path_prefix_truncated_stays_false_through_the_reranker_branch() {
        // Same exhaustive shape as the plain-search "false" test above, but with
        // a reranker configured.
        let results = vec![
            make_result_with_content("keep/only.md", 0.9, "k1"),
            make_result_with_content("keep/only.md", 0.8, "k2"),
            make_result_with_content("skip/other.md", 0.1, "s"),
        ];
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(results);
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
            limit: 5,
            rerank_candidate_limit: None,
            path_filter: Some(keep_filter()),
            ..default_opts()
        };
        let outcome = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap();

        assert_eq!(outcome.results.len(), 2);
        assert!(
            !outcome.path_prefix_truncated,
            "the fetch was already proven exhaustive; taking the reranker branch \
             must not spuriously set the flag"
        );
    }

    // ------------------------------------------------------------------
    // offset paging (#224)
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn search_paged_pages_do_not_overlap_no_reranker() {
        // The actual #224 bug: `offset: 0`, `offset: 2`, `offset: 4` against the
        // same query used to return byte-identical results every time, because
        // `search` never read `offset` at all. Prove the fix by paging through a
        // 6-result pool two at a time and asserting every page is disjoint from
        // every other AND that, reassembled, the pages reproduce the full
        // fused order exactly.
        let results: Vec<SearchResult> = (0..6)
            .map(|i| make_result_with_content(&format!("doc{i}.md"), 0.9 - i as f32 * 0.1, "x"))
            .collect();
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(results);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let mut opts = default_opts();
        opts.limit = 2;

        let page0 = search_paged(&deps, "q", &SearchFilters::default(), &opts, 0)
            .await
            .unwrap()
            .results;
        let page1 = search_paged(&deps, "q", &SearchFilters::default(), &opts, 2)
            .await
            .unwrap()
            .results;
        let page2 = search_paged(&deps, "q", &SearchFilters::default(), &opts, 4)
            .await
            .unwrap()
            .results;

        let paths0 = file_paths_of(&page0);
        let paths1 = file_paths_of(&page1);
        let paths2 = file_paths_of(&page2);

        assert_eq!(paths0, vec!["doc0.md", "doc1.md"]);
        assert_eq!(paths1, vec!["doc2.md", "doc3.md"]);
        assert_eq!(paths2, vec!["doc4.md", "doc5.md"]);

        // No pairwise overlap between any two pages.
        for (a, b) in [(&paths0, &paths1), (&paths0, &paths2), (&paths1, &paths2)] {
            assert!(
                a.iter().all(|p| !b.contains(p)),
                "pages must not overlap: {a:?} vs {b:?}"
            );
        }
    }

    #[tokio::test]
    async fn search_paged_offset_zero_matches_search_wrapper() {
        // `search()` is a thin `offset: 0` wrapper around `search_paged` — prove
        // the two stay equivalent rather than silently diverging.
        let results: Vec<SearchResult> = (0..3)
            .map(|i| make_result_with_content(&format!("doc{i}.md"), 0.9 - i as f32 * 0.1, "x"))
            .collect();
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(results);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let opts = default_opts();

        let via_wrapper = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap();
        let via_paged = search_paged(&deps, "q", &SearchFilters::default(), &opts, 0)
            .await
            .unwrap();

        assert_eq!(
            file_paths_of(&via_wrapper.results),
            file_paths_of(&via_paged.results)
        );
        assert_eq!(via_wrapper.offset_truncated, via_paged.offset_truncated);
    }

    #[tokio::test]
    async fn search_paged_offset_truncated_past_absolute_depth_bound() {
        // No reranker configured, so the only depth cap is `MAX_OFFSET_DEPTH`
        // (500). `offset + limit` past that must be flagged rather than
        // silently returning whatever happened to be available.
        let results: Vec<SearchResult> = (0..3)
            .map(|i| make_result_with_content(&format!("doc{i}.md"), 0.9 - i as f32 * 0.1, "x"))
            .collect();
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(results);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let mut opts = default_opts();
        opts.limit = 5;

        let outcome = search_paged(&deps, "q", &SearchFilters::default(), &opts, 498)
            .await
            .unwrap();

        assert!(
            outcome.offset_truncated,
            "offset 498 + limit 5 = 503 exceeds the 500 depth ceiling"
        );
    }

    #[tokio::test]
    async fn search_paged_offset_not_truncated_within_absolute_depth_bound() {
        let results: Vec<SearchResult> = (0..3)
            .map(|i| make_result_with_content(&format!("doc{i}.md"), 0.9 - i as f32 * 0.1, "x"))
            .collect();
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(results);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let mut opts = default_opts();
        opts.limit = 5;

        let outcome = search_paged(&deps, "q", &SearchFilters::default(), &opts, 10)
            .await
            .unwrap();

        assert!(
            !outcome.offset_truncated,
            "offset 10 + limit 5 = 15 is well within the 500 depth ceiling"
        );
    }

    #[tokio::test]
    async fn search_paged_offset_bounded_by_rerank_candidate_limit() {
        // With a reranker configured, `rerank_candidate_limit` — not
        // `MAX_OFFSET_DEPTH` — is the binding depth cap: the funnel never asks
        // the cross-encoder to score more candidates than this, so `offset`
        // cannot reach past it even though 500 is nowhere close.
        let results = vec![
            make_result_with_content("a.md", 0.9, "a"),
            make_result_with_content("b.md", 0.8, "b"),
            make_result_with_content("c.md", 0.7, "c"),
        ];
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(results);
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
            limit: 2,
            rerank_candidate_limit: Some(3),
            ..default_opts()
        };

        // offset 2 + limit 2 = depth 4, past the configured candidate_limit of 3.
        let outcome = search_paged(&deps, "q", &SearchFilters::default(), &opts, 2)
            .await
            .unwrap();
        assert!(
            outcome.offset_truncated,
            "depth 4 exceeds rerank_candidate_limit 3, even though it's far under \
             MAX_OFFSET_DEPTH"
        );
        // Only 1 candidate is left to page into (3 retained, skip 2).
        assert_eq!(outcome.results.len(), 1);

        // offset 0 + limit 2 = depth 2, within candidate_limit 3 — not truncated.
        let outcome_ok = search_paged(&deps, "q", &SearchFilters::default(), &opts, 0)
            .await
            .unwrap();
        assert!(!outcome_ok.offset_truncated);
        assert_eq!(outcome_ok.results.len(), 2);
    }

    #[tokio::test]
    async fn search_conditions_passed_through_unchanged() {
        // `SearchFilters` carries already-lowered Qdrant conditions now — building
        // and validating them is the caller's job (mcp.rs's rich `filters`, or
        // web.rs's plain domain/type/tags); `search` itself just ANDs whatever it's
        // handed onto the query, unexamined.
        let embed = MockEmbedder::ok(vec![0.1, 0.2, 0.3]);
        let store = MockRetrievalStore::with_results(vec![]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let conditions = vec![
            Condition::matches("domain", "engineering".to_string()),
            Condition::matches("type", "guide".to_string()),
            Condition::matches("tags", vec!["rust".to_string(), "rag".to_string()]),
        ];
        let filters = SearchFilters {
            conditions: conditions.clone(),
        };

        let _ = search(&deps, "query", &filters, &default_opts())
            .await
            .unwrap();

        let received = store.received_conditions.lock().unwrap().clone().unwrap();
        assert_eq!(received, conditions);
    }

    /// #130: `search` must AND the `path_filter_condition` onto whatever the
    /// caller's own `SearchFilters` already carried — not replace it, not drop it.
    /// This is the request-shape half of the #130 fix; the server-side "does the
    /// filter actually exclude non-matching legacy documents" half is not
    /// mock-testable (`MockRetrievalStore` never evaluates the `Filter` it's
    /// handed) and is covered instead by `apply_path_filter`'s still-active retain
    /// tests below, which exercise the client-side backstop this condition's
    /// `is_empty` escape depends on.
    #[tokio::test]
    async fn search_appends_path_ancestors_condition_alongside_caller_conditions() {
        let embed = MockEmbedder::ok(vec![0.1, 0.2, 0.3]);
        let store = MockRetrievalStore::with_results(vec![]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let caller_conditions = vec![Condition::matches("domain", "sysadmin".to_string())];
        let filters = SearchFilters {
            conditions: caller_conditions.clone(),
        };
        let opts = SearchOptions {
            path_filter: Some(pf(&["sysadmin/nodes/ares.md"])),
            ..default_opts()
        };

        let _ = search(&deps, "query", &filters, &opts).await.unwrap();

        let received = store.received_conditions.lock().unwrap().clone().unwrap();
        let mut expected = caller_conditions;
        expected.push(expected_path_filter_condition(&["sysadmin/nodes/ares.md"]));
        assert_eq!(
            received, expected,
            "the path_ancestors condition must be appended, not substituted"
        );
    }

    /// Same request-shape guarantee as the test above, for the grouped-document
    /// path — `search_grouped` builds its own `extra_conditions` independently of
    /// `search`/`search_paged`, so it needs its own proof it wires the condition in.
    #[tokio::test]
    async fn search_grouped_appends_path_ancestors_condition() {
        let embed = MockEmbedder::ok(vec![0.1, 0.2, 0.3]);
        let store = MockRetrievalStore::with_results(vec![]);
        let doc_index = MockDocumentIndex::with_paths(vec![]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let opts = SearchOptions {
            path_filter: Some(pf(&["kitchen/chili.md"])),
            ..default_opts()
        };

        let _ = search_grouped(
            &deps,
            &doc_index,
            "query",
            &SearchFilters::default(),
            &opts,
            None,
            0,
        )
        .await
        .unwrap();

        let received = store.received_conditions.lock().unwrap().clone().unwrap();
        assert_eq!(
            received,
            vec![expected_path_filter_condition(&["kitchen/chili.md"])]
        );
    }

    #[tokio::test]
    async fn search_no_path_ancestors_condition_when_path_prefix_unset() {
        // No condition at all is appended when `path_prefix` is `None` — confirms
        // the #130 change is additive-only and a no-op for every existing caller
        // that never sets `path_prefix`.
        let embed = MockEmbedder::ok(vec![0.1, 0.2, 0.3]);
        let store = MockRetrievalStore::with_results(vec![]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let _ = search(&deps, "query", &SearchFilters::default(), &default_opts())
            .await
            .unwrap();

        let received = store.received_conditions.lock().unwrap().clone().unwrap();
        assert!(
            received.is_empty(),
            "no path_prefix means no path_ancestors condition"
        );
    }

    #[tokio::test]
    async fn search_embed_failure_returns_embed_error() {
        let embed = MockEmbedder::err("embedding service down");
        let store = MockRetrievalStore::with_results(vec![]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let result = search(&deps, "query", &SearchFilters::default(), &default_opts()).await;
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

        let result = search(&deps, "query", &SearchFilters::default(), &default_opts()).await;
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
        let returned = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap()
            .results;
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

        let returned = search(&deps, "q", &SearchFilters::default(), &default_opts())
            .await
            .unwrap()
            .results;
        assert_eq!(returned.len(), 3, "None min_score should keep all results");
    }

    #[tokio::test]
    async fn search_empty_results_returns_empty_vec() {
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(vec![]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let returned = search(&deps, "q", &SearchFilters::default(), &default_opts())
            .await
            .unwrap()
            .results;
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
        let returned = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap()
            .results;

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
        let returned = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap()
            .results;

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
        let returned = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap()
            .results;

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
            conditions: vec![Condition::matches("domain", "sysadmin".to_string())],
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

        // Filters carried into the hybrid path as extra conditions, not the legacy
        // filter map (that map now carries only the mtime__gte/mtime__lte sentinels).
        let received = store.received_conditions.lock().unwrap().clone().unwrap();
        assert_eq!(
            received,
            vec![Condition::matches("domain", "sysadmin".to_string())],
            "filters must be carried into hybrid_search unchanged"
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
            &SearchFilters::default(),
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

        let results = search(&deps, "ares", &SearchFilters::default(), &hybrid_opts())
            .await
            .unwrap()
            .results;

        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].payload.get("file_path").and_then(|v| v.as_str()),
            Some("sysadmin/ares.md"),
            "hybrid result should surface the keyword chunk first"
        );
    }

    // ------------------------------------------------------------------
    // phrase search routing tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn search_quoted_span_routes_to_hybrid_search_with_the_phrase_even_when_hybrid_is_off() {
        let embed = MockEmbedder::ok(vec![0.1, 0.2, 0.3]);
        let store = MockRetrievalStore::with_results(vec![make_search_result(0.9)]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let opts = SearchOptions {
            phrase: true,
            ..default_opts() // hybrid: false
        };
        let _ = search(
            &deps,
            r#"deploy notes for "node:ares" rocm"#,
            &SearchFilters::default(),
            &opts,
        )
        .await
        .unwrap();

        assert_eq!(
            *store.last_call.lock().unwrap(),
            Some("hybrid_search"),
            "a quoted span must route to the fused path independent of hybrid"
        );
        assert_eq!(
            *store.received_phrases.lock().unwrap(),
            vec!["node:ares".to_string()]
        );
        assert!(
            store.received_sparse.lock().unwrap().is_none(),
            "hybrid=false must not add a sparse arm even with a phrase present"
        );
    }

    #[tokio::test]
    async fn search_embeds_the_flattened_dequoted_text_not_the_raw_query() {
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(vec![]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let opts = SearchOptions {
            phrase: true,
            ..default_opts()
        };
        let _ = search(
            &deps,
            r#"deploy notes for "node:ares" rocm"#,
            &SearchFilters::default(),
            &opts,
        )
        .await
        .unwrap();

        assert_eq!(
            *embed.last_query.lock().unwrap(),
            Some("deploy notes for node:ares rocm".to_string())
        );
    }

    #[tokio::test]
    async fn search_with_phrase_disabled_treats_quotes_as_literal_characters() {
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(vec![]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let query = r#"deploy notes for "node:ares" rocm"#;
        let opts = SearchOptions {
            phrase: false,
            ..default_opts()
        };
        let _ = search(&deps, query, &SearchFilters::default(), &opts)
            .await
            .unwrap();

        assert_eq!(
            *embed.last_query.lock().unwrap(),
            Some(query.to_string()),
            "phrase: false must leave the raw query — quotes included — untouched"
        );
        assert_eq!(
            *store.last_call.lock().unwrap(),
            Some("search"),
            "phrase: false must not add a phrase arm, so this stays the plain dense path"
        );
    }

    #[tokio::test]
    async fn search_unquoted_query_is_unaffected_by_phrase_being_enabled() {
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(vec![]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let query = "deploy notes for node:ares rocm";
        let opts = SearchOptions {
            phrase: true,
            ..default_opts()
        };
        let _ = search(&deps, query, &SearchFilters::default(), &opts)
            .await
            .unwrap();

        assert_eq!(*embed.last_query.lock().unwrap(), Some(query.to_string()));
        assert_eq!(
            *store.last_call.lock().unwrap(),
            Some("search"),
            "no quotes means no phrase, so this must stay the plain dense path"
        );
    }

    #[tokio::test]
    async fn search_grouped_honors_hybrid_and_phrase_like_search_does() {
        // Part 1's regression: grouped search used to be hard-wired dense-only
        // regardless of `opts.hybrid`/`opts.phrase`. It must now build the same
        // arms `search` does.
        let embed = MockEmbedder::ok(vec![0.1, 0.2, 0.3]);
        let store = MockRetrievalStore::with_results(vec![make_grouped_hit("a.md", 0.9)]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);
        let index = MockDocumentIndex::with_summaries(vec![]);

        let opts = SearchOptions {
            hybrid: true,
            phrase: true,
            ..default_opts()
        };
        let _ = search_grouped(
            &deps,
            &index,
            r#"deploy notes for "node:ares" rocm"#,
            &SearchFilters::default(),
            &opts,
            None,
            0,
        )
        .await
        .unwrap();

        assert_eq!(*store.last_call.lock().unwrap(), Some("search_grouped"));
        assert!(
            store.received_sparse.lock().unwrap().is_some(),
            "hybrid=true must still add a sparse arm at document granularity"
        );
        assert_eq!(
            *store.received_phrases.lock().unwrap(),
            vec!["node:ares".to_string()]
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
    async fn get_document_populates_links_out_and_links_in() {
        let tmp = tempfile::tempdir().unwrap();
        let data_path = tmp.path().canonicalize().unwrap();
        let docs = data_path.join("docs");
        std::fs::create_dir_all(&docs).unwrap();
        std::fs::write(docs.join("guide.md"), "# Guide").unwrap();

        let gs = make_md_globset();
        let embed = MockEmbedder::ok(vec![]);
        let store = MockRetrievalStore::with_results(vec![]);
        let deps = make_deps(&embed, &store, &data_path, &gs);
        let index = MockDocumentIndex::with_paths_and_links(
            vec![],
            vec![OutboundLink {
                target_path: "docs/other.md".to_string(),
                kind: "markdown".to_string(),
                score: None,
                exists: true,
            }],
            vec![InboundLink {
                source_path: "docs/referrer.md".to_string(),
                kind: "semantic".to_string(),
                score: Some(0.8),
            }],
        );

        let doc = get_document(&deps, &index, "docs/guide.md").await.unwrap();

        assert_eq!(doc.links_out.total, 1);
        assert_eq!(doc.links_out.links[0].target_path, "docs/other.md");
        assert_eq!(doc.links_in.total, 1);
        assert_eq!(doc.links_in.links[0].source_path, "docs/referrer.md");
    }

    #[tokio::test]
    async fn get_document_link_lookup_failure_degrades_instead_of_failing_the_read() {
        // The state-DB-unavailable path: path resolution succeeds (no `all_paths`
        // call needed for a literal path), but the link queries themselves error.
        // `get_document` must still return the document, with empty/zero-total
        // link pages rather than propagating the error.
        let tmp = tempfile::tempdir().unwrap();
        let data_path = tmp.path().canonicalize().unwrap();
        let docs = data_path.join("docs");
        std::fs::create_dir_all(&docs).unwrap();
        std::fs::write(docs.join("guide.md"), "# Guide\nStill readable.").unwrap();

        let gs = make_md_globset();
        let embed = MockEmbedder::ok(vec![]);
        let store = MockRetrievalStore::with_results(vec![]);
        let deps = make_deps(&embed, &store, &data_path, &gs);
        let index = MockDocumentIndex::with_paths_and_links_err(vec![], "state db unavailable");

        let doc = get_document(&deps, &index, "docs/guide.md").await.unwrap();

        assert!(doc.content.contains("Still readable."));
        assert_eq!(doc.links_out.total, 0);
        assert!(doc.links_out.links.is_empty());
        assert_eq!(doc.links_in.total, 0);
        assert!(doc.links_in.links.is_empty());
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
            .insert(CHUNK_TEXT_KEY.into(), serde_json::json!("doc A"));
        let mut r1 = make_search_result(0.8);
        r1.payload
            .insert(CHUNK_TEXT_KEY.into(), serde_json::json!("doc B"));
        let mut r2 = make_search_result(0.7);
        r2.payload
            .insert(CHUNK_TEXT_KEY.into(), serde_json::json!("doc C"));

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

        let results = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap()
            .results;

        // MockReranker reverses order: last doc gets highest score
        assert_eq!(results.len(), 3);
        // The last original result (index 2, doc C) should now be first
        assert_eq!(
            results[0]
                .payload
                .get(CHUNK_TEXT_KEY)
                .and_then(|v| v.as_str()),
            Some("doc C"),
            "reranked order should put last doc first"
        );
    }

    #[tokio::test]
    async fn reranker_failure_returns_fused_order() {
        let mut r0 = make_search_result(0.9);
        r0.payload
            .insert(CHUNK_TEXT_KEY.into(), serde_json::json!("doc A"));
        let mut r1 = make_search_result(0.8);
        r1.payload
            .insert(CHUNK_TEXT_KEY.into(), serde_json::json!("doc B"));

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

        let results = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap()
            .results;

        // Should still return results (fail-soft), in original fused order
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0]
                .payload
                .get(CHUNK_TEXT_KEY)
                .and_then(|v| v.as_str()),
            Some("doc A"),
            "on reranker failure, should return fused order"
        );
    }

    // A reranker stub that hands back an exact, caller-specified response
    // rather than computing one from the input — needed for the #136/#138
    // tests below, which must control the returned `index` values precisely
    // (out of range, or repeated) rather than derive them from `documents`.
    struct MockRerankerFixed {
        ranked: Vec<crate::rerank::RerankResult>,
    }

    impl crate::rerank::Reranker for MockRerankerFixed {
        fn rerank<'a>(
            &'a self,
            _query: &'a str,
            _documents: &'a [&'a str],
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = anyhow::Result<Vec<crate::rerank::RerankResult>>>
                    + Send
                    + 'a,
            >,
        > {
            let ranked = self.ranked.clone();
            Box::pin(async move { Ok(ranked) })
        }
    }

    #[tokio::test]
    async fn reranker_out_of_range_index_is_dropped_not_panicking() {
        // #136 regression: `docs_with_indices[r.index]` used to be indexed
        // straight off the reranker's untrusted response with no bounds check,
        // so an out-of-range `index` panicked the whole search request. Only
        // 2 documents are sent to the reranker here (valid indices 0 and 1);
        // index 5 is malformed and must be dropped, not indexed.
        let mut r0 = make_search_result(0.9);
        r0.payload
            .insert(CHUNK_TEXT_KEY.into(), serde_json::json!("doc A"));
        let mut r1 = make_search_result(0.8);
        r1.payload
            .insert(CHUNK_TEXT_KEY.into(), serde_json::json!("doc B"));

        let store = MockRetrievalStore::with_results(vec![r0, r1]);
        let embed = MockEmbedder::ok(vec![0.1]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let reranker = MockRerankerFixed {
            ranked: vec![
                crate::rerank::RerankResult {
                    index: 0,
                    relevance_score: 1.0,
                },
                crate::rerank::RerankResult {
                    index: 5,
                    relevance_score: 2.0,
                },
            ],
        };
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

        // The old code panicked inside this call; the assertion below is only
        // reachable at all once the bounds check is in place.
        let results = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap()
            .results;

        assert_eq!(
            results.len(),
            1,
            "the out-of-range entry must be dropped, leaving only the valid one"
        );
        assert_eq!(
            results[0]
                .payload
                .get(CHUNK_TEXT_KEY)
                .and_then(|v| v.as_str()),
            Some("doc A"),
            "the surviving result must be the one with the valid index"
        );
    }

    #[tokio::test]
    async fn reranker_duplicate_index_is_deduplicated() {
        // #138 regression: a reranker response repeating the same `index`
        // used to make that one chunk occupy two slots in `indexed`, so it
        // came out twice in the final result set. The duplicate (lower-scored,
        // second) occurrence must be dropped, keeping the result set free of
        // literal duplicates.
        let mut r0 = make_search_result(0.9);
        r0.payload
            .insert(CHUNK_TEXT_KEY.into(), serde_json::json!("doc A"));
        let mut r1 = make_search_result(0.8);
        r1.payload
            .insert(CHUNK_TEXT_KEY.into(), serde_json::json!("doc B"));

        let store = MockRetrievalStore::with_results(vec![r0, r1]);
        let embed = MockEmbedder::ok(vec![0.1]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let reranker = MockRerankerFixed {
            ranked: vec![
                crate::rerank::RerankResult {
                    index: 0,
                    relevance_score: 0.9,
                },
                crate::rerank::RerankResult {
                    index: 0,
                    relevance_score: 0.5,
                },
                crate::rerank::RerankResult {
                    index: 1,
                    relevance_score: 0.7,
                },
            ],
        };
        let deps = RetrievalDeps {
            embed_client: &embed,
            qdrant: &store,
            collection: "test-col",
            data_path,
            include_patterns: &gs,
            reranker: Some(&reranker as &(dyn crate::rerank::Reranker + Send + Sync)),
        };

        // `limit` must be large enough that the spurious duplicate would
        // survive `take(top_k)` if it weren't deduplicated — at limit 2 the
        // duplicate's lower score happens to fall outside the page regardless,
        // which would make this test pass even without the fix.
        let opts = SearchOptions {
            limit: 3,
            rerank_candidate_limit: None,
            ..default_opts()
        };

        let results = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap()
            .results;

        assert_eq!(
            results.len(),
            2,
            "the duplicate index-0 entry must be dropped, not counted as a \
             second result, even though the page has room for 3"
        );
        let doc_a_count = results
            .iter()
            .filter(|r| r.payload.get(CHUNK_TEXT_KEY).and_then(|v| v.as_str()) == Some("doc A"))
            .count();
        assert_eq!(doc_a_count, 1, "doc A must appear exactly once");
    }

    #[tokio::test]
    async fn reranker_entirely_invalid_indices_falls_back_to_fused_order() {
        // #210: before this fix, a `200` response whose every entry is invalid
        // (all out of range here) left `indexed` empty, so the caller got a
        // near-empty result set even though the fused ranking computed just
        // above is a perfectly good page — a strictly worse outcome than the
        // `Err` arm's graceful fallback for the SAME underlying condition (the
        // reranker being unusable). Prove the two converge: this response is
        // "successful" at the transport level (`Ok`, not a network error) yet
        // must still produce the full fused-order page, not an almost-empty one.
        let mut r0 = make_search_result(0.9);
        r0.payload
            .insert(CHUNK_TEXT_KEY.into(), serde_json::json!("doc A"));
        let mut r1 = make_search_result(0.8);
        r1.payload
            .insert(CHUNK_TEXT_KEY.into(), serde_json::json!("doc B"));

        let store = MockRetrievalStore::with_results(vec![r0, r1]);
        let embed = MockEmbedder::ok(vec![0.1]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let reranker = MockRerankerFixed {
            ranked: vec![
                crate::rerank::RerankResult {
                    index: 5,
                    relevance_score: 0.9,
                },
                crate::rerank::RerankResult {
                    index: 9,
                    relevance_score: 0.8,
                },
            ],
        };
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

        let results = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap()
            .results;

        // Before the fix: `indexed` is empty (both indices out of range), so
        // `results` would end up empty too — this assertion is what fails
        // without the fallback.
        assert_eq!(
            results.len(),
            2,
            "an all-invalid rerank response must fall back to the full fused \
             page, not an empty one: got {} results",
            results.len()
        );
        // Fused order is score-descending: doc A (0.9) before doc B (0.8) —
        // proves this really is the pre-rerank order, not a coincidence of length.
        assert_eq!(
            results[0]
                .payload
                .get(CHUNK_TEXT_KEY)
                .and_then(|v| v.as_str()),
            Some("doc A"),
            "fallback must preserve fused (pre-rerank) order"
        );
    }

    #[tokio::test]
    async fn reranker_empty_response_falls_back_to_fused_order() {
        // A reranker that returns `Ok(vec![])` — zero entries, not an error —
        // is just as unusable as one returning garbage indices; #210's fix
        // covers this too since `indexed` ends up empty either way.
        let mut r0 = make_search_result(0.9);
        r0.payload
            .insert(CHUNK_TEXT_KEY.into(), serde_json::json!("doc A"));

        let store = MockRetrievalStore::with_results(vec![r0]);
        let embed = MockEmbedder::ok(vec![0.1]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let reranker = MockRerankerFixed { ranked: vec![] };
        let deps = RetrievalDeps {
            embed_client: &embed,
            qdrant: &store,
            collection: "test-col",
            data_path,
            include_patterns: &gs,
            reranker: Some(&reranker as &(dyn crate::rerank::Reranker + Send + Sync)),
        };

        let opts = SearchOptions {
            limit: 1,
            rerank_candidate_limit: None,
            ..default_opts()
        };

        let results = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap()
            .results;

        assert_eq!(
            results.len(),
            1,
            "an empty rerank response must still fall back to fused order"
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
        b1.payload
            .insert(CHUNK_TEXT_KEY.into(), serde_json::json!("b1"));
        let make_a = |content: &str| {
            let mut r = make_search_result(0.4);
            r.payload
                .insert("file_path".into(), serde_json::json!("a.md"));
            r.payload
                .insert(CHUNK_TEXT_KEY.into(), serde_json::json!(content));
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

        let results = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap()
            .results;

        assert_eq!(
            file_paths_of(&results),
            vec!["a.md", "a.md", "b.md"],
            "post-rerank cap must trim a.md to 2 and backfill with b.md instead \
             of the page shrinking to 2 results"
        );
    }

    // ------------------------------------------------------------------
    // #61 regression: writer (ingest) / reader (reranker input builder) key
    // agreement. The bug was that `ingest.rs` wrote chunk text under `"text"`
    // while `retrieval.rs` read it back under `"content"`, so the reranker's
    // input list was always empty and reranking silently no-op'd — every
    // result kept its pre-rerank fused order and `pre_rerank_score` stayed
    // `None`, with no error anywhere. These tests pin the fix directly rather
    // than through fixtures that could drift back to mirroring a future bug.
    // ------------------------------------------------------------------

    #[test]
    fn docs_for_rerank_finds_text_written_under_the_shared_key() {
        // Insert the payload exactly the way `ingest::index_paths` does — chunk
        // text as a `serde_json::Value::String` under `CHUNK_TEXT_KEY` — and
        // assert the reranker input builder (the reader) finds it under that
        // same key. This is the assertion that should have existed since #61:
        // it fails the instant writer and reader disagree on the key, without
        // needing a full `search()` call (and its silent fail-soft path) to
        // notice.
        let mut r = make_search_result(0.42);
        r.payload.insert(
            CHUNK_TEXT_KEY.to_string(),
            serde_json::Value::String("chunk body text".to_string()),
        );

        let docs = docs_for_rerank(std::slice::from_ref(&r));

        assert_eq!(
            docs,
            vec![(0, "chunk body text")],
            "reranker input builder must find chunk text under CHUNK_TEXT_KEY, \
             the exact key ingest::index_paths writes it under"
        );
    }

    #[test]
    fn docs_for_rerank_skips_results_missing_the_shared_key() {
        // The flip side: a result with no CHUNK_TEXT_KEY entry (or an
        // unrelated key, like the old buggy "content") contributes nothing to
        // the reranker's input list. This is the exact mechanism that made
        // #61 silent — production payloads never had the reader's old key, so
        // every result was filtered out and reranking quietly no-op'd via the
        // `docs.is_empty()` fail-soft path instead of erroring.
        let mut wrong_key = make_search_result(0.5);
        wrong_key.payload.insert(
            "content".to_string(),
            serde_json::json!("should be ignored"),
        );
        let no_key = make_search_result(0.5);

        let results = [wrong_key, no_key];
        let docs = docs_for_rerank(&results);

        assert!(
            docs.is_empty(),
            "results without CHUNK_TEXT_KEY must not reach the reranker"
        );
    }

    #[tokio::test]
    async fn reranker_is_invoked_when_results_carry_chunk_text() {
        // Proves the reranker is actually called end-to-end when results carry
        // real chunk text under the production key — not just that the input
        // builder finds text in isolation (see `docs_for_rerank_finds_text_...`
        // above). Payloads are built with `CHUNK_TEXT_KEY`, mirroring exactly
        // what `ingest::index_paths` writes, and the mock reranker's known
        // reversal is used as a witness: if the reranker were skipped (the
        // #61 bug), results would keep their original fused order instead.
        let mut r0 = make_search_result(0.9);
        r0.payload.insert(
            CHUNK_TEXT_KEY.to_string(),
            serde_json::Value::String("doc A".to_string()),
        );
        let mut r1 = make_search_result(0.8);
        r1.payload.insert(
            CHUNK_TEXT_KEY.to_string(),
            serde_json::Value::String("doc B".to_string()),
        );
        let mut r2 = make_search_result(0.7);
        r2.payload.insert(
            CHUNK_TEXT_KEY.to_string(),
            serde_json::Value::String("doc C".to_string()),
        );

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

        let results = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap()
            .results;

        // MockReranker fully reverses whatever order it's handed. If the
        // reranker had been skipped (docs.is_empty()), the fused order (A, B,
        // C by descending score) would survive untouched instead.
        assert_eq!(
            results
                .iter()
                .map(|r| r
                    .payload
                    .get(CHUNK_TEXT_KEY)
                    .and_then(|v| v.as_str())
                    .unwrap())
                .collect::<Vec<_>>(),
            vec!["doc C", "doc B", "doc A"],
            "reranker must have actually run and reordered results, proving \
             `docs` was non-empty"
        );
    }

    #[tokio::test]
    async fn search_explain_populates_nonnull_pre_rerank_score_when_reranker_runs() {
        // This is exactly how the #61 bug surfaced in production: every result
        // came back with `pre_rerank_score: null` even with `explain: true`
        // and reranking enabled, because the reranker's input list was always
        // empty and the early-return path never touches `pre_rerank_score`.
        // Here the payload carries chunk text under the real key, so the
        // reranker runs and must stamp each result's original (pre-rerank)
        // score onto `pre_rerank_score`.
        let mut r0 = make_search_result(0.9);
        r0.payload.insert(
            CHUNK_TEXT_KEY.to_string(),
            serde_json::Value::String("doc A".to_string()),
        );
        let mut r1 = make_search_result(0.8);
        r1.payload.insert(
            CHUNK_TEXT_KEY.to_string(),
            serde_json::Value::String("doc B".to_string()),
        );

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

        let results = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap()
            .results;

        assert_eq!(results.len(), 2, "reranker must have run on both results");
        for r in &results {
            assert!(
                r.pre_rerank_score.is_some(),
                "explain=true + an active reranker that actually receives chunk \
                 text must set pre_rerank_score on every result — a null here \
                 is exactly the production symptom of #61"
            );
        }
        // The two original fused scores (0.9 for "doc A", 0.8 for "doc B")
        // must both still be present as pre_rerank_score, just possibly
        // reassigned to a different result after reordering.
        let mut pre_scores: Vec<f32> = results
            .iter()
            .map(|r| r.pre_rerank_score.unwrap())
            .collect();
        pre_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(pre_scores, vec![0.8, 0.9]);
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
        let _ = search(&deps, "q", &SearchFilters::default(), &opts)
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
        let _ = search(&deps, "q", &SearchFilters::default(), &opts)
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
            .insert(CHUNK_TEXT_KEY.into(), serde_json::json!("doc A"));
        let mut r1 = make_search_result(0.8);
        r1.payload
            .insert(CHUNK_TEXT_KEY.into(), serde_json::json!("doc B"));

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
        let results = search(&deps, "q", &SearchFilters::default(), &opts)
            .await
            .unwrap()
            .results;

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
    // search_grouped: query+document granularity
    // -----------------------------------------------------------------------

    fn make_grouped_hit(file_path: &str, score: f32) -> SearchResult {
        let mut payload = HashMap::new();
        payload.insert(
            "file_path".to_string(),
            serde_json::json!(format!("/data/{file_path}")),
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

    fn make_summary(file_path: &str, title: &str) -> DocumentSummary {
        DocumentSummary {
            file_path: file_path.to_string(),
            title: Some(title.to_string()),
            description: None,
            mtime: 100,
            indexed_at: "2024-01-01T00:00:00Z".to_string(),
            frontmatter: serde_json::json!({}),
        }
    }

    #[tokio::test]
    async fn search_grouped_returns_one_row_per_document_hydrated_and_ranked() {
        // Qdrant's grouped query already collapsed each document to its best chunk;
        // `search_grouped` must hydrate each with document metadata and preserve
        // Qdrant's score-descending order through the SQLite round trip.
        let embed = MockEmbedder::ok(vec![0.1, 0.2]);
        let store = MockRetrievalStore::with_results(vec![
            make_grouped_hit("a.md", 0.9),
            make_grouped_hit("b.md", 0.7),
        ]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);
        let index = MockDocumentIndex::with_summaries(vec![
            // Deliberately out of score order, to prove `search_grouped` re-sorts by
            // Qdrant's ranking rather than trusting the index's own row order.
            make_summary("b.md", "B"),
            make_summary("a.md", "A"),
        ]);

        let documents = search_grouped(
            &deps,
            &index,
            "query",
            &SearchFilters::default(),
            &default_opts(),
            None,
            0,
        )
        .await
        .unwrap()
        .documents;

        assert_eq!(documents.len(), 2, "one row per document, not per chunk");
        assert_eq!(documents[0].summary.file_path, "a.md");
        assert_eq!(documents[0].score, 0.9);
        assert_eq!(documents[1].summary.file_path, "b.md");
        assert_eq!(documents[1].score, 0.7);
    }

    #[tokio::test]
    async fn search_grouped_skips_a_hit_missing_from_the_metadata_index() {
        // The metadata index can transiently lag Qdrant (same caveat as
        // `get_document`'s fuzzy fallback) — a hit with no matching summary is
        // dropped rather than fabricated or panicking.
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(vec![make_grouped_hit("gone.md", 0.9)]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);
        let index = MockDocumentIndex::with_summaries(vec![]);

        let documents = search_grouped(
            &deps,
            &index,
            "query",
            &SearchFilters::default(),
            &default_opts(),
            None,
            0,
        )
        .await
        .unwrap()
        .documents;

        assert!(documents.is_empty());
    }

    #[tokio::test]
    async fn search_grouped_truncates_to_limit() {
        // `path_prefix` over-fetches (up to `limit * 5`, capped) before filtering
        // by prefix, so it is easy for more than `opts.limit` documents to survive
        // the prefix/min_score filters. `search_grouped` must still cap its final
        // output at `opts.limit`, the same way chunk-granularity `search` does.
        let embed = MockEmbedder::ok(vec![0.1, 0.2]);
        let store = MockRetrievalStore::with_results(vec![
            make_grouped_hit("docs/a.md", 0.9),
            make_grouped_hit("docs/b.md", 0.8),
            make_grouped_hit("docs/c.md", 0.7),
            make_grouped_hit("docs/d.md", 0.6),
            make_grouped_hit("docs/e.md", 0.5),
        ]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);
        let index = MockDocumentIndex::with_summaries(vec![
            make_summary("docs/a.md", "A"),
            make_summary("docs/b.md", "B"),
            make_summary("docs/c.md", "C"),
            make_summary("docs/d.md", "D"),
            make_summary("docs/e.md", "E"),
        ]);

        let mut opts = default_opts();
        opts.limit = 2;
        opts.path_filter = Some(pf(&[
            "docs/a.md",
            "docs/b.md",
            "docs/c.md",
            "docs/d.md",
            "docs/e.md",
        ]));

        let documents = search_grouped(
            &deps,
            &index,
            "query",
            &SearchFilters::default(),
            &opts,
            None,
            0,
        )
        .await
        .unwrap()
        .documents;

        assert_eq!(
            documents.len(),
            2,
            "search_grouped must truncate to opts.limit like search() does"
        );
    }

    #[tokio::test]
    async fn search_grouped_pages_do_not_overlap() {
        // Document-granularity mirror of `search_paged_pages_do_not_overlap_no_reranker`
        // — #224 applies to `granularity: document` too.
        let embed = MockEmbedder::ok(vec![0.1, 0.2]);
        let store = MockRetrievalStore::with_results(vec![
            make_grouped_hit("docs/a.md", 0.9),
            make_grouped_hit("docs/b.md", 0.8),
            make_grouped_hit("docs/c.md", 0.7),
            make_grouped_hit("docs/d.md", 0.6),
        ]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);
        let index = MockDocumentIndex::with_summaries(vec![
            make_summary("docs/a.md", "A"),
            make_summary("docs/b.md", "B"),
            make_summary("docs/c.md", "C"),
            make_summary("docs/d.md", "D"),
        ]);

        let mut opts = default_opts();
        opts.limit = 2;

        let page0 = search_grouped(
            &deps,
            &index,
            "query",
            &SearchFilters::default(),
            &opts,
            None,
            0,
        )
        .await
        .unwrap()
        .documents;
        let page1 = search_grouped(
            &deps,
            &index,
            "query",
            &SearchFilters::default(),
            &opts,
            None,
            2,
        )
        .await
        .unwrap()
        .documents;

        let paths0: Vec<&str> = page0.iter().map(|d| d.summary.file_path.as_str()).collect();
        let paths1: Vec<&str> = page1.iter().map(|d| d.summary.file_path.as_str()).collect();

        assert_eq!(paths0, vec!["docs/a.md", "docs/b.md"]);
        assert_eq!(paths1, vec!["docs/c.md", "docs/d.md"]);
        assert!(
            paths0.iter().all(|p| !paths1.contains(p)),
            "pages must not overlap: {paths0:?} vs {paths1:?}"
        );
    }

    #[tokio::test]
    async fn search_grouped_offset_truncated_past_absolute_depth_bound() {
        // No reranker/diversity pass exists on this path, so `MAX_OFFSET_DEPTH`
        // is the only depth cap — see `search_grouped`'s doc comment.
        let embed = MockEmbedder::ok(vec![0.1, 0.2]);
        let store = MockRetrievalStore::with_results(vec![make_grouped_hit("docs/a.md", 0.9)]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);
        let index = MockDocumentIndex::with_summaries(vec![make_summary("docs/a.md", "A")]);

        let mut opts = default_opts();
        opts.limit = 5;

        let outcome = search_grouped(
            &deps,
            &index,
            "query",
            &SearchFilters::default(),
            &opts,
            None,
            498,
        )
        .await
        .unwrap();

        assert!(
            outcome.offset_truncated,
            "offset 498 + limit 5 = 503 exceeds the 500 depth ceiling"
        );
    }

    // -----------------------------------------------------------------------
    // #245: retrieval metrics are actually recorded by the real search path
    // -----------------------------------------------------------------------
    //
    // The bug these tests target is specifically that `record_query` was correct
    // and covered by its own unit tests in `status.rs`, but nothing in this
    // module ever called it — "after ~15 real searches every counter read 0" per
    // the live-deployment verification in the issue. A test that calls
    // `QUERY_METRICS.record_query(...)` directly would have passed before this
    // fix just as easily as after it, since `record_query` itself was never
    // broken. These tests instead go through `retrieval::search`/`search_grouped`
    // — the real, only caller in production — and assert the counters moved.
    //
    // `QUERY_METRICS` is a process-global `LazyLock`, and `cargo test` runs tests
    // in this crate concurrently in one process, so a test cannot assert an
    // exact absolute count (another test's search could race in between). Taking
    // a before/after snapshot and asserting the delta is at least what this
    // test's own call should have produced is safe under that concurrency: only
    // `fetch_add` ever touches these atomics, so a snapshot taken before this
    // test's own `search` call can only ever be a lower bound, never invalidated
    // by a concurrent increment.

    #[tokio::test]
    async fn search_records_query_metrics_via_the_real_search_path() {
        let before = crate::status::QUERY_METRICS.snapshot();

        let results = vec![make_result_with_content("doc.md", 0.9, "hello world")];
        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(results);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);

        let outcome = search(&deps, "q", &SearchFilters::default(), &default_opts())
            .await
            .unwrap();
        assert_eq!(outcome.results.len(), 1);

        let after = crate::status::QUERY_METRICS.snapshot();
        assert!(
            after.queries_total > before.queries_total,
            "queries_total must increase after a real retrieval::search call \
             (before={}, after={}) — this is the exact gap #245 was filed over: \
             record_query existed and worked, but nothing called it",
            before.queries_total,
            after.queries_total
        );
        assert!(
            after.embed_latency_ms.count > before.embed_latency_ms.count,
            "the embed-latency histogram must record a sample from the real call"
        );
        assert!(
            after.qdrant_latency_ms.count > before.qdrant_latency_ms.count,
            "the qdrant-latency histogram must record a sample from the real call"
        );
    }

    #[tokio::test]
    async fn search_zero_results_with_reranker_configured_still_records_a_query() {
        // The specific gap called out in #245: `search_paged`'s `docs.is_empty()`
        // early return (Qdrant found nothing to send the reranker) bypasses the
        // function's normal timing/record_query call entirely. Missing this call
        // site would systematically undercount the single most useful metric —
        // the zero-result rate — for every zero-hit query while reranking is on.
        let before = crate::status::QUERY_METRICS.snapshot();

        let embed = MockEmbedder::ok(vec![0.1]);
        let store = MockRetrievalStore::with_results(Vec::new());
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

        let outcome = search(&deps, "q", &SearchFilters::default(), &default_opts())
            .await
            .unwrap();
        assert!(outcome.results.is_empty());

        let after = crate::status::QUERY_METRICS.snapshot();
        assert!(
            after.queries_total > before.queries_total,
            "the docs.is_empty() early return must still record a query"
        );
        assert!(
            after.zero_result_total > before.zero_result_total,
            "a zero-result query must increment zero_result_total even when a \
             reranker is configured but never gets any candidates to rerank"
        );
    }

    #[tokio::test]
    async fn search_grouped_records_query_metrics_via_the_real_path() {
        // `search_grouped` has no reranker and, unlike `search_paged`, no early
        // "zero candidates" return — one call site covers it, added alongside
        // the `search_paged` fix since #241's task explicitly called out
        // checking this function's return paths rather than assuming.
        let before = crate::status::QUERY_METRICS.snapshot();

        let embed = MockEmbedder::ok(vec![0.1, 0.2]);
        let store = MockRetrievalStore::with_results(vec![make_grouped_hit("docs/a.md", 0.9)]);
        let gs = make_md_globset();
        let data_path = Path::new("/data");
        let deps = make_deps(&embed, &store, data_path, &gs);
        let index = MockDocumentIndex::with_summaries(vec![make_summary("docs/a.md", "A")]);

        let outcome = search_grouped(
            &deps,
            &index,
            "query",
            &SearchFilters::default(),
            &default_opts(),
            None,
            0,
        )
        .await
        .unwrap();
        assert_eq!(outcome.documents.len(), 1);

        let after = crate::status::QUERY_METRICS.snapshot();
        assert!(
            after.queries_total > before.queries_total,
            "search_grouped must also record a query on its real return path"
        );
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
