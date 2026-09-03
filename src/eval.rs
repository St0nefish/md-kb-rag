//! Retrieval-quality evaluation harness (issue #167).
//!
//! This is a RAG system whose entire product is retrieval quality, and until this
//! module existed there was no way to measure it. `search.rrf_candidates`,
//! `search.diversity_max_per_document`, `search.min_score`,
//! `reranking.candidate_limit`, `chunking.max_chunk_size`, `search.hybrid`, and the
//! embedding model itself all trade recall against precision against latency in ways
//! that are only knowable empirically — every one of them was, until now, tuned
//! blind. `mcp-md-wiki eval --queries eval.yaml` closes that loop: it runs a fixed set
//! of `{query, expected documents}` cases through the real `retrieval::search` core
//! (the same function the server and the `search` CLI subcommand use — see #84's
//! dependency injection, which is precisely what makes this cheap to build and run
//! offline) and reports whether the *deployed config* finds the documents an
//! operator already knows are right.
//!
//! Deliberately not reimplemented here: ranking, fusion, filtering. This module's
//! only job is to drive `retrieval::search` with cases from a file and score what
//! comes back — see `run_eval` below. All the actual retrieval behavior — hybrid
//! RRF, phrase matching, reranking, diversity capping — lives in `retrieval.rs`
//! exactly once.
//!
//! # Query file schema
//!
//! ```yaml
//! cases:
//!   - query: "how do I reset the qdrant collection"
//!     expect_paths:            # AND semantics — every path here must be retrieved
//!       - dev/troubleshooting/qdrant-reset.md
//!   - query: "printer bed leveling"
//!     expect_any:               # OR semantics — at least one of these must be retrieved
//!       - 3d-printing/bed-leveling.md
//!       - 3d-printing/first-layer-calibration.md
//!     filters:                  # optional, same shape as the `search` CLI subcommand's
//!       domain: 3d-printing
//!       type: guide
//!       tags: [claude-code]
//! ```
//!
//! `expect_paths` and `expect_any` answer two different questions a case author
//! actually has, and conflating them would make the schema lie about intent:
//!
//! - `expect_paths` is for a query with a known, specific, *multi-document* answer —
//!   e.g. "what are our GPU deployment options" ought to surface the ROCm guide AND
//!   the Vulkan guide AND the Nvidia guide. Every entry is independently required;
//!   partial credit is possible (recall@k is fractional) but the case only *passes*
//!   when all of them show up.
//! - `expect_any` is for a query where several documents are equally correct answers
//!   — near-duplicate reference pages, or a topic covered from two angles — and
//!   finding any one of them means retrieval did its job. Treating those as an
//!   `expect_paths` AND-set would make a case permanently unpassable (or would
//!   silently under-report recall) for a query that was never supposed to require
//!   every alternative simultaneously.
//!
//! A case may set both: `expect_paths` are still each individually required, and
//! `expect_any` contributes one additional required "slot" satisfiable by any one of
//! its members. A case with neither is a case that can never fail no matter what
//! retrieval returns, which is not a useful test — `load_cases` rejects it at load
//! time rather than silently reporting a hollow 100%.
//!
//! `filters` is optional per-case and reuses `retrieval::plain_search_filters` — the
//! same domain/type/tags shape the `search` CLI subcommand and the web UI's
//! `/api/search` already expose — so a case can pin down a query that would
//! otherwise be ambiguous across the whole corpus.
//!
//! # Metrics
//!
//! Both are computed per case against the top-`k` results `retrieval::search`
//! returns (`k` is also what is passed to `SearchOptions::limit`, so "top-k" here
//! means literally the results retrieval chose to return, not a post-hoc slice of a
//! larger set):
//!
//! - **recall@k** — for a case with `n` required "slots" (each `expect_paths` entry
//!   is one slot; a non-empty `expect_any` list is one more slot, satisfiable by any
//!   member), recall@k is `slots_found / n`. The aggregate recall@k is the mean of
//!   the per-case recall@k across all cases — NOT a pooled found/required ratio, so
//!   a case with many `expect_paths` entries cannot dominate the aggregate over a
//!   case with one.
//! - **MRR** (Mean Reciprocal Rank) — for each case, find the rank (1-based) of the
//!   first returned result whose path is in that case's relevant set
//!   (`expect_paths` ∪ `expect_any`); the case's reciprocal rank is `1/rank`, or `0`
//!   if nothing relevant was returned at all. The aggregate MRR is the mean of the
//!   per-case reciprocal rank.
//!
//! `--threshold` (see `main.rs`) gates on the aggregate **recall@k** specifically,
//! not MRR: recall@k answers "did we find the right documents at all", which is the
//! yes/no retrieval-quality question a CI gate needs; MRR answers "how high did we
//! rank them", a real but secondary concern once the guarantee is that they're found
//! at all. Reported side by side regardless of which one gates.

use std::collections::HashSet;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::embed::QueryEmbedder;
use crate::qdrant::RetrievalStore;
use crate::retrieval::{self, RetrievalDeps, SearchError, SearchOptions, plain_search_filters};

/// Per-case retrieval filters, mirroring the plain domain/type/tags shape the
/// `search` CLI subcommand and `/api/search` already use — see
/// `retrieval::plain_search_filters`, which this lowers to directly rather than
/// reimplementing the same three `Condition`s a second time.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalFilters {
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default, rename = "type")]
    pub doc_type: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
}

/// One evaluation case: a query and the documents retrieval is expected to find.
///
/// See the module doc comment for the full `expect_paths` (AND) vs `expect_any`
/// (OR) rationale.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalCase {
    pub query: String,
    /// Every one of these must appear in the top-k results for a perfect score on
    /// this case's AND-slots.
    #[serde(default)]
    pub expect_paths: Vec<String>,
    /// At least one of these must appear in the top-k results — together these
    /// count as a single required slot, not one per entry.
    #[serde(default)]
    pub expect_any: Vec<String>,
    #[serde(default)]
    pub filters: EvalFilters,
}

/// The parsed contents of an eval query file — `deny_unknown_fields` at every level
/// (see `EvalCase`/`EvalFilters`) so a typo'd key (`expect_path` for
/// `expect_paths`) is a load-time error instead of a silently-vacuous case.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvalFile {
    #[serde(default)]
    cases: Vec<EvalCase>,
}

/// Load and validate eval cases from a YAML file.
///
/// Validation beyond parsing: the file must contain at least one case, every case
/// needs a non-empty `query`, and every case needs at least one expectation
/// (`expect_paths` and/or `expect_any` non-empty) — a case with neither can never
/// fail, which would let a broken eval file silently report perfect scores forever.
pub fn load_cases(path: &Path) -> anyhow::Result<Vec<EvalCase>> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read '{}': {e}", path.display()))?;
    let file: EvalFile = serde_yaml_ng::from_str(&raw)
        .map_err(|e| anyhow::anyhow!("failed to parse '{}': {e}", path.display()))?;

    if file.cases.is_empty() {
        anyhow::bail!(
            "'{}' contains no cases — add at least one under `cases:`",
            path.display()
        );
    }

    for (i, case) in file.cases.iter().enumerate() {
        if case.query.trim().is_empty() {
            anyhow::bail!("case {} (0-based) has an empty `query`", i);
        }
        if case.expect_paths.is_empty() && case.expect_any.is_empty() {
            anyhow::bail!(
                "case {} ('{}') has neither `expect_paths` nor `expect_any` — it can \
                 never fail, which makes it useless as a test",
                i,
                case.query
            );
        }
    }

    Ok(file.cases)
}

/// The retrieval knobs `run_eval` sweeps into every case's `SearchOptions`.
///
/// A deliberately narrow subset of `SearchConfig`/`ResolvedRerankingConfig`: eval
/// cares about the knobs that change *what* comes back (hybrid/phrase/candidate
/// depth/diversity/min_score) and not about response-shaping options (`explain`,
/// `modified_after`/`modified_before`, `path_prefix`) that no eval case has any use
/// for. Copy rather than borrowed from `ResolvedConfig` directly so callers (the CLI
/// today) build it once from whatever config source they have, and this module
/// never has to know `ResolvedConfig`'s shape.
#[derive(Debug, Clone, Copy)]
pub struct EvalSearchConfig {
    /// Both the recall@k cutoff and `SearchOptions::limit` — see the module doc
    /// comment for why these are deliberately the same number.
    pub k: u64,
    pub min_score: Option<f32>,
    pub hybrid: bool,
    pub rrf_candidates: u64,
    pub phrase: bool,
    pub rerank_candidate_limit: Option<u64>,
    pub diversity_max_per_document: Option<usize>,
}

/// One case's outcome: what was retrieved, what was found/missing against its
/// expectations, and its two per-case metrics.
#[derive(Debug, Clone, Serialize)]
pub struct CaseResult {
    pub query: String,
    /// Fraction of this case's required slots that were found in the top-k.
    pub recall_at_k: f64,
    /// `1 / rank` of the first relevant result, or `0.0` if none was found.
    pub reciprocal_rank: f64,
    /// True iff every required slot was found (`recall_at_k == 1.0`).
    pub passed: bool,
    /// Expected paths (from `expect_paths`, plus the winning `expect_any` member if
    /// one hit) that were actually found in the results.
    pub found: Vec<String>,
    /// Expected paths that were NOT found — every missed `expect_paths` entry, plus
    /// the whole `expect_any` list if none of its members hit.
    pub missing: Vec<String>,
    /// The top-k file paths retrieval actually returned, in rank order — kept on
    /// the result so a human (or `--json` consumer) can see what search returned
    /// instead of what was expected, without re-running the query.
    pub retrieved: Vec<String>,
}

/// Score one case against its ranked, already-top-k-truncated list of retrieved
/// file paths.
///
/// Pure and synchronous on purpose: this is the part of the harness where a wrong
/// formula would silently poison every number the tool ever reports, so it is
/// isolated from the network/async retrieval call specifically so tests can drive
/// it directly with hand-worked inputs (see `mod tests` below) rather than only
/// indirectly through a live search.
pub fn score_case(case: &EvalCase, retrieved: &[String]) -> CaseResult {
    let mut required = 0usize;
    let mut found_count = 0usize;
    let mut found = Vec::new();
    let mut missing = Vec::new();

    for expected in &case.expect_paths {
        required += 1;
        if retrieved.iter().any(|r| r == expected) {
            found_count += 1;
            found.push(expected.clone());
        } else {
            missing.push(expected.clone());
        }
    }

    if !case.expect_any.is_empty() {
        required += 1;
        match case
            .expect_any
            .iter()
            .find(|expected| retrieved.iter().any(|r| &r == expected))
        {
            Some(hit) => {
                found_count += 1;
                found.push(hit.clone());
            }
            None => missing.extend(case.expect_any.iter().cloned()),
        }
    }

    // `required` is guaranteed >= 1 by `load_cases`' validation, but a case built
    // directly (as the unit tests below do) could construct an empty one — dividing
    // by zero would be a NaN recall silently poisoning the aggregate mean, so treat
    // "nothing was ever required" as a trivial zero rather than propagating NaN.
    let recall_at_k = if required == 0 {
        0.0
    } else {
        found_count as f64 / required as f64
    };

    let relevant: HashSet<&str> = case
        .expect_paths
        .iter()
        .chain(case.expect_any.iter())
        .map(String::as_str)
        .collect();
    let reciprocal_rank = retrieved
        .iter()
        .position(|r| relevant.contains(r.as_str()))
        .map(|idx| 1.0 / (idx as f64 + 1.0))
        .unwrap_or(0.0);

    CaseResult {
        query: case.query.clone(),
        recall_at_k,
        reciprocal_rank,
        passed: required > 0 && found_count == required,
        found,
        missing,
        retrieved: retrieved.to_vec(),
    }
}

/// The full report: every case's outcome plus the two aggregate metrics.
#[derive(Debug, Clone, Serialize)]
pub struct EvalReport {
    pub k: u64,
    pub cases: Vec<CaseResult>,
    /// Mean of each case's `recall_at_k` — see the module doc comment for why this
    /// is a mean of per-case fractions, not a pooled ratio.
    pub mean_recall_at_k: f64,
    /// Mean of each case's `reciprocal_rank`.
    pub mrr: f64,
    pub passed: usize,
    pub failed: usize,
}

impl EvalReport {
    fn from_cases(k: u64, cases: Vec<CaseResult>) -> Self {
        let n = cases.len();
        let (mean_recall_at_k, mrr) = if n == 0 {
            (0.0, 0.0)
        } else {
            (
                cases.iter().map(|c| c.recall_at_k).sum::<f64>() / n as f64,
                cases.iter().map(|c| c.reciprocal_rank).sum::<f64>() / n as f64,
            )
        };
        let passed = cases.iter().filter(|c| c.passed).count();
        let failed = n - passed;

        Self {
            k,
            cases,
            mean_recall_at_k,
            mrr,
            passed,
            failed,
        }
    }
}

/// Render a `SearchError` the same way the `search` CLI subcommand does, so a
/// failure surfaced through `eval` reads consistently with one surfaced through
/// `mcp-md-wiki search`.
fn describe_search_error(query: &str, e: SearchError) -> anyhow::Error {
    match e {
        SearchError::Embed(err) => anyhow::anyhow!("query '{query}': embedding failed: {err:#}"),
        SearchError::Search(err) => anyhow::anyhow!("query '{query}': search failed: {err:#}"),
        SearchError::Document(err) => {
            anyhow::anyhow!("query '{query}': document metadata lookup failed: {err:#}")
        }
    }
}

/// Run every case through `retrieval::search` with `search_cfg` applied, and score
/// each one.
///
/// Fails fast on the first case whose `retrieval::search` call itself errors
/// (embedding/Qdrant unreachable, etc.) — that is an infrastructure failure, not a
/// quality signal, and averaging it in as "0 recall" would make a config change and
/// a Qdrant outage indistinguishable in the report.
pub async fn run_eval<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &RetrievalDeps<'_, E, Q>,
    cases: &[EvalCase],
    search_cfg: &EvalSearchConfig,
) -> anyhow::Result<EvalReport> {
    let mut results = Vec::with_capacity(cases.len());

    for case in cases {
        let filters = plain_search_filters(
            case.filters.domain.as_deref(),
            case.filters.doc_type.as_deref(),
            case.filters.tags.as_deref(),
        );
        let opts = SearchOptions {
            limit: search_cfg.k,
            min_score: search_cfg.min_score,
            hybrid: search_cfg.hybrid,
            rrf_candidates: search_cfg.rrf_candidates,
            phrase: search_cfg.phrase,
            explain: false,
            modified_after: None,
            modified_before: None,
            path_filter: None,
            rerank_candidate_limit: search_cfg.rerank_candidate_limit,
            diversity_max_per_document: search_cfg.diversity_max_per_document,
        };

        let outcome = retrieval::search(deps, &case.query, &filters, &opts)
            .await
            .map_err(|e| describe_search_error(&case.query, e))?;

        let retrieved: Vec<String> = outcome
            .results
            .iter()
            .filter_map(|r| {
                r.payload
                    .get("file_path")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
            })
            .collect();

        results.push(score_case(case, &retrieved));
    }

    Ok(EvalReport::from_cases(search_cfg.k, results))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(query: &str, expect_paths: &[&str], expect_any: &[&str]) -> EvalCase {
        EvalCase {
            query: query.to_string(),
            expect_paths: expect_paths.iter().map(|s| s.to_string()).collect(),
            expect_any: expect_any.iter().map(|s| s.to_string()).collect(),
            filters: EvalFilters::default(),
        }
    }

    fn paths(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    // --- score_case: hand-worked recall@k / MRR -----------------------------

    #[test]
    fn expect_paths_scores_partial_recall_and_first_relevant_rank() {
        // required = 2 (a.md, b.md), retrieved has a.md at rank 2 (index 1), b.md missing.
        let c = case("q", &["a.md", "b.md"], &[]);
        let retrieved = paths(&["x.md", "a.md", "c.md"]);
        let r = score_case(&c, &retrieved);

        assert_eq!(r.recall_at_k, 0.5, "1 of 2 required slots found");
        assert_eq!(
            r.reciprocal_rank, 0.5,
            "a.md is the first relevant hit, at rank 2 -> 1/2"
        );
        assert!(!r.passed, "b.md missing means this case did not pass");
        assert_eq!(r.found, vec!["a.md".to_string()]);
        assert_eq!(r.missing, vec!["b.md".to_string()]);
    }

    #[test]
    fn expect_any_counts_as_one_slot_satisfied_by_any_member() {
        // required = 1 (the whole expect_any list), q.md hits at rank 1.
        let c = case("q", &[], &["p.md", "q.md"]);
        let retrieved = paths(&["q.md", "z.md"]);
        let r = score_case(&c, &retrieved);

        assert_eq!(r.recall_at_k, 1.0);
        assert_eq!(r.reciprocal_rank, 1.0);
        assert!(r.passed);
        assert_eq!(r.found, vec!["q.md".to_string()]);
        assert!(r.missing.is_empty());
    }

    #[test]
    fn expect_any_with_no_hit_reports_every_alternative_as_missing() {
        let c = case("q", &[], &["p.md", "q.md"]);
        let retrieved = paths(&["z.md"]);
        let r = score_case(&c, &retrieved);

        assert_eq!(r.recall_at_k, 0.0);
        assert_eq!(r.reciprocal_rank, 0.0);
        assert!(!r.passed);
        assert_eq!(r.missing, vec!["p.md".to_string(), "q.md".to_string()]);
    }

    #[test]
    fn combined_expect_paths_and_expect_any_both_count_toward_required() {
        // required = 2: a.md (expect_paths) + the expect_any slot.
        let c = case("q", &["a.md"], &["p.md", "q.md"]);
        let retrieved = paths(&["a.md", "p.md"]);
        let r = score_case(&c, &retrieved);

        assert_eq!(r.recall_at_k, 1.0);
        assert!(r.passed);
    }

    #[test]
    fn nothing_relevant_retrieved_gives_zero_reciprocal_rank() {
        let c = case("q", &["a.md"], &[]);
        let retrieved = paths(&["x.md", "y.md"]);
        let r = score_case(&c, &retrieved);

        assert_eq!(r.recall_at_k, 0.0);
        assert_eq!(r.reciprocal_rank, 0.0);
        assert!(!r.passed);
    }

    #[test]
    fn empty_case_scores_zero_recall_not_nan() {
        // Not reachable via `load_cases` (which rejects this), but `score_case` is
        // public and must not produce NaN if ever called directly on one.
        let c = case("q", &[], &[]);
        let r = score_case(&c, &paths(&["x.md"]));
        assert_eq!(r.recall_at_k, 0.0);
        assert!(!r.recall_at_k.is_nan());
        assert!(!r.passed);
    }

    // --- EvalReport aggregation: the module doc comment's worked example ----

    #[test]
    fn aggregate_metrics_are_the_mean_of_per_case_metrics() {
        // Case 1: recall 0.5, RR 0.5 (from the first test above).
        // Case 2: recall 1.0, RR 1.0 (from the second test above).
        // Mean recall = 0.75, mean RR = 0.75.
        let c1 = score_case(
            &case("q1", &["a.md", "b.md"], &[]),
            &paths(&["x.md", "a.md", "c.md"]),
        );
        let c2 = score_case(
            &case("q2", &[], &["p.md", "q.md"]),
            &paths(&["q.md", "z.md"]),
        );

        let report = EvalReport::from_cases(5, vec![c1, c2]);

        assert_eq!(report.mean_recall_at_k, 0.75);
        assert_eq!(report.mrr, 0.75);
        assert_eq!(report.passed, 1);
        assert_eq!(report.failed, 1);
    }

    #[test]
    fn empty_report_has_zeroed_aggregates_not_nan() {
        let report = EvalReport::from_cases(5, vec![]);
        assert_eq!(report.mean_recall_at_k, 0.0);
        assert_eq!(report.mrr, 0.0);
        assert_eq!(report.passed, 0);
        assert_eq!(report.failed, 0);
    }

    // --- load_cases: YAML parsing, including malformed input ----------------

    fn write_temp(contents: &str) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        f
    }

    #[test]
    fn loads_a_well_formed_file_with_paths_any_and_filters() {
        let f = write_temp(
            r#"
cases:
  - query: "how do I reset qdrant"
    expect_paths:
      - dev/troubleshooting/qdrant-reset.md
  - query: "printer bed leveling"
    expect_any:
      - 3d-printing/bed-leveling.md
      - 3d-printing/first-layer-calibration.md
    filters:
      domain: 3d-printing
      type: guide
      tags: [claude-code]
"#,
        );

        let cases = load_cases(f.path()).unwrap();
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].query, "how do I reset qdrant");
        assert_eq!(
            cases[0].expect_paths,
            vec!["dev/troubleshooting/qdrant-reset.md".to_string()]
        );
        assert_eq!(cases[1].filters.domain.as_deref(), Some("3d-printing"));
        assert_eq!(cases[1].filters.doc_type.as_deref(), Some("guide"));
        assert_eq!(
            cases[1].filters.tags.as_deref(),
            Some(["claude-code".to_string()].as_slice())
        );
    }

    #[test]
    fn rejects_malformed_yaml() {
        let f = write_temp("cases: [this is not: valid: yaml structure");
        let err = load_cases(f.path()).unwrap_err();
        assert!(err.to_string().contains("failed to parse"), "{err}");
    }

    #[test]
    fn rejects_an_unknown_field_as_a_likely_typo() {
        let f = write_temp(
            r#"
cases:
  - query: "q"
    expect_path: ["a.md"]
"#,
        );
        let err = load_cases(f.path()).unwrap_err();
        assert!(err.to_string().contains("failed to parse"), "{err}");
    }

    #[test]
    fn rejects_an_empty_case_list() {
        let f = write_temp("cases: []");
        let err = load_cases(f.path()).unwrap_err();
        assert!(err.to_string().contains("no cases"), "{err}");
    }

    #[test]
    fn rejects_a_case_with_no_expectations() {
        let f = write_temp(
            r#"
cases:
  - query: "q"
"#,
        );
        let err = load_cases(f.path()).unwrap_err();
        assert!(err.to_string().contains("can never fail"), "{err}");
    }

    #[test]
    fn rejects_a_case_with_an_empty_query() {
        let f = write_temp(
            r#"
cases:
  - query: "  "
    expect_paths: ["a.md"]
"#,
        );
        let err = load_cases(f.path()).unwrap_err();
        assert!(err.to_string().contains("empty `query`"), "{err}");
    }

    #[test]
    fn rejects_a_missing_file() {
        let err = load_cases(Path::new("/nonexistent/eval.yaml")).unwrap_err();
        assert!(err.to_string().contains("failed to read"), "{err}");
    }
}
