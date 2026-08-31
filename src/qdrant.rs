use std::collections::HashMap;

use anyhow::{Context, Result};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, CountPointsBuilder, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder,
    DeletePointsBuilder, Distance, FacetCountsBuilder, FacetHit, FieldCondition, FieldType, Filter,
    Fusion, Match, Modifier, NamedVectors, PointStruct, PrefetchQuery, PrefetchQueryBuilder, Query,
    QueryPointGroupsBuilder, QueryPointsBuilder, Range, SearchPointsBuilder,
    SparseVectorParamsBuilder, SparseVectorsConfigBuilder, TextIndexParamsBuilder, TokenizerType,
    UpsertPointsBuilder, Value as QdrantValue, Vector, VectorInput, VectorParamsBuilder,
    VectorsConfigBuilder, facet_value, value::Kind, vectors_config::Config as VectorsConfigOneof,
};
use tracing::{debug, error, info, warn};

use crate::config::ResolvedQdrantConfig;
use crate::state::FieldFilter;

pub trait VectorStore: Send + Sync {
    async fn upsert_points(&self, collection: &str, points: Vec<QdrantPoint>) -> Result<()>;
    async fn delete_by_files(&self, collection: &str, file_paths: &[&str]) -> Result<()>;
    async fn delete_points_by_ids(&self, collection: &str, ids: Vec<String>) -> Result<()>;
    /// Drop `collection` if it exists — `md-kb-rag index --full`'s destructive
    /// rebuild step. In the trait (rather than only an inherent `QdrantStore`
    /// method) so `ingest::index_paths_generic` can run this same call path against
    /// a fake in tests.
    async fn drop_collection(&self, collection: &str) -> Result<()>;
    /// Create `collection` if it does not exist and ensure every payload index in
    /// `indexed_fields` (plus the standing `mtime` range index). In the trait for
    /// the same reason as `drop_collection` above.
    ///
    /// `enable_phrase` gates creating the phrase-matching text index on the `text`
    /// field (`search.phrase`, config-controlled). When `true`, index creation
    /// failure — an older Qdrant server rejecting `phrase_matching` — is logged and
    /// tolerated exactly like every other payload index above: it never fails
    /// startup, and the caller-visible effect is `status::INDEX_STATUS`'s "text"
    /// entry going to `Failed`, which is what gates phrase matching off for the
    /// process (see `status::IndexStatus::phrase_matching_available`).
    async fn ensure_collection(
        &self,
        collection: &str,
        vector_size: u64,
        indexed_fields: &[IndexedField],
        enable_phrase: bool,
    ) -> Result<()>;
    /// Exact point count for `collection` (`0` if the collection does not exist), for
    /// #155's active self-heal: `ingest::scan_and_index` compares this against
    /// state.db's `total_chunk_count()` before a scoped reconcile sweep and escalates
    /// to a full rebuild on a large deficit — see that call site for the full
    /// rationale. In the trait (rather than only an inherent `QdrantStore` method) so
    /// that escalation decision can be driven by test fakes instead of a live Qdrant,
    /// same reasoning as `drop_collection`/`ensure_collection` above.
    ///
    /// Deliberately exact (`CountPointsBuilder::exact(true)`), unlike the approximate
    /// `points_count` `collection_info` reports for `/status` (`QdrantStore::collection_info`,
    /// server.rs's passive half of #155): that check only ever reports a number,
    /// while this one decides whether to drop and rebuild the whole collection, so it
    /// cannot afford eventual-consistency noise turning into a false-positive wipe.
    async fn collection_point_count(&self, collection: &str) -> Result<u64>;
}

pub trait RetrievalStore: Send + Sync {
    /// `extra_conditions` are ANDed onto `filters` unchanged — the caller's
    /// already-lowered rich `filters` (see [`lower_field_filters`]), separate from
    /// `filters` because that map also carries the `mtime__gte`/`mtime__lte`
    /// sentinel keys [`build_conditions`] special-cases.
    async fn search(
        &self,
        collection: &str,
        vector: Vec<f32>,
        filters: std::collections::HashMap<String, serde_json::Value>,
        extra_conditions: Vec<Condition>,
        limit: u64,
    ) -> Result<Vec<SearchResult>>;
    /// Fused retrieval via Qdrant's server-side Reciprocal Rank Fusion over up to
    /// three prefetch arms: dense (always), sparse (when `sparse` is `Some` — i.e.
    /// hybrid is enabled and the query produced a non-empty sparse vector), and
    /// phrase (when `phrases` is non-empty — one exact-phrase condition per entry,
    /// ANDed onto a dense-ranked arm). At least one of `sparse`/`phrases` must be
    /// given; a caller with neither should call [`RetrievalStore::search`] instead
    /// of paying for a needless fusion query.
    ///
    /// When `explain=false` (default): fuses server-side; `dense_score`/
    /// `sparse_score`/`phrase_score` on results are `None`.
    /// When `explain=true`: runs each present arm as a separate query, fuses
    /// client-side (k=60), and populates the per-arm score field for every arm a
    /// result appeared in — `phrase_score` is `Some` exactly when the result
    /// matched every requested phrase.
    ///
    /// `extra_conditions`: see [`RetrievalStore::search`].
    #[allow(clippy::too_many_arguments)]
    async fn hybrid_search(
        &self,
        collection: &str,
        dense: Vec<f32>,
        sparse: Option<(Vec<u32>, Vec<f32>)>,
        phrases: &[String],
        filters: std::collections::HashMap<String, serde_json::Value>,
        extra_conditions: Vec<Condition>,
        limit: u64,
        rrf_candidates: u64,
        explain: bool,
    ) -> Result<Vec<SearchResult>>;

    /// Nearest-neighbor search grouped by a payload field, collapsed to
    /// `group_size` best-scoring hits per group — the `search` tool's
    /// query+document (grouped) granularity.
    ///
    /// Shares retrieval semantics with [`RetrievalStore::hybrid_search`] rather
    /// than running its own dense-only-always path: dense-only when `sparse` is
    /// `None` and `phrases` is empty (a plain per-group nearest-neighbor query,
    /// preserving the historical dense-only behaviour exactly), otherwise the same
    /// dense/sparse/phrase prefetch arms fused via server-side RRF before grouping.
    /// This is deliberate — grouped and chunk results must differ only in shape,
    /// one row per document vs one per chunk, never in which documents are
    /// retrievable; an exact-identifier query the sparse arm alone can find must
    /// not silently return nothing at document granularity. `extra_conditions`:
    /// see [`RetrievalStore::search`].
    #[allow(clippy::too_many_arguments)]
    async fn search_grouped(
        &self,
        collection: &str,
        vector: Vec<f32>,
        sparse: Option<(Vec<u32>, Vec<f32>)>,
        phrases: &[String],
        filters: std::collections::HashMap<String, serde_json::Value>,
        extra_conditions: Vec<Condition>,
        group_by: &str,
        group_size: u64,
        limit: u64,
        rrf_candidates: u64,
    ) -> Result<Vec<SearchResult>>;
}

pub struct QdrantStore {
    client: Qdrant,
}

#[derive(Debug, Clone)]
pub struct QdrantPoint {
    pub id: String,
    pub vector: Vec<f32>,
    /// Sparse vector as `(indices, values)`. Attached as the named `sparse` vector
    /// when present; `None` stores only the dense vector.
    pub sparse: Option<(Vec<u32>, Vec<f32>)>,
    pub payload: HashMap<String, serde_json::Value>,
}

/// Kind of payload index to create for a field.
///
/// A keyword index serves equality and any-of matching; numeric and boolean fields need
/// their own index kinds for range and comparison filters to work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    Keyword,
    Integer,
    Float,
    Bool,
}

impl IndexKind {
    fn to_qdrant(self) -> FieldType {
        match self {
            IndexKind::Keyword => FieldType::Keyword,
            IndexKind::Integer => FieldType::Integer,
            IndexKind::Float => FieldType::Float,
            IndexKind::Bool => FieldType::Bool,
        }
    }
}

/// A payload field to index, and how.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedField {
    pub name: String,
    pub kind: IndexKind,
}

impl IndexedField {
    pub fn keyword(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: IndexKind::Keyword,
        }
    }
}

/// The full set of payload fields Qdrant actually has an index for: the union of
/// [`crate::schema::SchemaCache::all_indexed_fields`] (per-scope `.kb-schema.yaml`
/// declarations) and `config.effective_indexed_fields()` (the legacy
/// `frontmatter.indexed_fields` list, which also always contributes `file_path`).
///
/// This is the single canonical source for "what is indexed in Qdrant" — every
/// caller that creates payload indexes, or that validates a filter field against
/// what is indexed, must go through this rather than reimplementing the union or
/// consulting only the schema half of it. Using only `all_indexed_fields()` on its
/// own understates the real index set and rejects filters that Qdrant can in fact
/// serve.
pub fn all_indexed_fields(
    config: &crate::config::ResolvedConfig,
    schemas: &crate::schema::SchemaCache,
) -> Vec<IndexedField> {
    let mut fields = schemas.all_indexed_fields();
    for name in config.effective_indexed_fields() {
        if !fields.iter().any(|f| f.name == name) {
            fields.push(IndexedField::keyword(name));
        }
    }
    fields.sort_by(|a, b| a.name.cmp(&b.name));
    fields
}

/// The Qdrant payload key under which a chunk's raw text is stored.
///
/// There is exactly one writer — `ingest::index_paths` — and it must be the only
/// place that ever inserts this key. Every reader of chunk text off a
/// `SearchResult`/`ScoredPoint` payload must go through this constant instead of
/// a string literal:
///
///   - `retrieval::rerank_and_truncate` (builds the cross-encoder's input list)
///   - `mcp::format_search_results` (the MCP search snippet formatter)
///   - `web::to_search_hit` (the web UI search result formatter)
///
/// This exists because of the #61 regression: the writer and the reranker reader
/// drifted to different literals (`"text"` vs `"content"`), so the reranker's
/// input list was always empty and reranking silently no-op'd in production for
/// every deployment since. Nothing caught it because the tests built fixture
/// payloads with the same wrong key the reader used, instead of the key the
/// writer actually writes. Route every writer and reader through this constant
/// so that class of drift can't happen again.
pub const CHUNK_TEXT_KEY: &str = "text";

/// The Qdrant payload key under which a document's ancestor-directory keyword
/// array is stored (#130) — for `sysadmin/nodes/ares/boot/efi.md`:
/// `["sysadmin", "sysadmin/nodes", "sysadmin/nodes/ares", "sysadmin/nodes/ares/boot",
/// "sysadmin/nodes/ares/boot/efi.md"]` (every ancestor directory, deepest last,
/// plus the document's own full relative path as the terminal entry — see
/// `ingest::derive_path_ancestors`'s doc comment for why the file's own path is
/// included even though the issue's example stops at the parent directory).
///
/// There is exactly one writer — `ingest::index_paths` — and it must be the only
/// place that ever inserts this key, same discipline as [`CHUNK_TEXT_KEY`] and for
/// the same reason (the #61 regression). Every place that turns a `path_prefix`
/// into a Qdrant condition (currently `retrieval::path_filter_condition`) must go
/// through this constant instead of a string literal.
pub const PATH_ANCESTORS_KEY: &str = "path_ancestors";

#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchResult {
    pub score: f32,
    /// Retrieval score before cross-encoder reranking, when reranking was active.
    pub pre_rerank_score: Option<f32>,
    /// Dense cosine score for this result. Always `Some` for dense-only queries;
    /// `Some` for hybrid queries when `explain=true` (client-side RRF); `None` otherwise.
    pub dense_score: Option<f32>,
    /// Sparse BM25 score for this result. `Some` only for hybrid queries when `explain=true`.
    pub sparse_score: Option<f32>,
    /// Dense-ranked score from the phrase arm. `Some` only when `explain=true` and
    /// this result matched every requested phrase (see [`RetrievalStore::hybrid_search`]);
    /// its presence, not its value, is what tells a caller a hit was a phrase match.
    pub phrase_score: Option<f32>,
    pub payload: HashMap<String, serde_json::Value>,
}

// Convert serde_json::Value -> QdrantValue
fn json_to_qdrant_value(v: &serde_json::Value) -> QdrantValue {
    let kind = match v {
        serde_json::Value::Null => None,
        serde_json::Value::Bool(b) => Some(Kind::BoolValue(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(Kind::IntegerValue(i))
            } else {
                n.as_f64().map(Kind::DoubleValue)
            }
        }
        serde_json::Value::String(s) => Some(Kind::StringValue(s.clone())),
        serde_json::Value::Array(arr) => {
            let values = arr.iter().map(json_to_qdrant_value).collect();
            Some(Kind::ListValue(qdrant_client::qdrant::ListValue { values }))
        }
        serde_json::Value::Object(map) => {
            let fields = map
                .iter()
                .map(|(k, v)| (k.clone(), json_to_qdrant_value(v)))
                .collect();
            Some(Kind::StructValue(qdrant_client::qdrant::Struct { fields }))
        }
    };
    QdrantValue { kind }
}

// Convert QdrantValue -> serde_json::Value
fn qdrant_value_to_json(v: &QdrantValue) -> serde_json::Value {
    match &v.kind {
        None => serde_json::Value::Null,
        Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
        Some(Kind::IntegerValue(i)) => serde_json::Value::Number((*i).into()),
        Some(Kind::DoubleValue(f)) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Some(Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
        Some(Kind::ListValue(list)) => {
            serde_json::Value::Array(list.values.iter().map(qdrant_value_to_json).collect())
        }
        Some(Kind::StructValue(s)) => {
            let map = s
                .fields
                .iter()
                .map(|(k, v)| (k.clone(), qdrant_value_to_json(v)))
                .collect();
            serde_json::Value::Object(map)
        }
    }
}

// Convert HashMap<String, serde_json::Value> -> HashMap<String, QdrantValue> (for PointStruct payload)
fn json_payload_to_qdrant(
    payload: &HashMap<String, serde_json::Value>,
) -> HashMap<String, QdrantValue> {
    payload
        .iter()
        .map(|(k, v)| (k.clone(), json_to_qdrant_value(v)))
        .collect()
}

// Convert HashMap<String, QdrantValue> -> HashMap<String, serde_json::Value>
fn qdrant_payload_to_json(
    payload: &HashMap<String, QdrantValue>,
) -> HashMap<String, serde_json::Value> {
    payload
        .iter()
        .map(|(k, v)| (k.clone(), qdrant_value_to_json(v)))
        .collect()
}

/// Build Qdrant filter conditions from a JSON filter map.
///
/// Supports: String (keyword match), Integer (exact match),
/// Bool (boolean match), Array of strings (match_any).
/// Special keys `mtime__gte` and `mtime__lte` (integer values) are combined
/// into a single range condition on the `mtime` payload field.
/// Returns an error for float values, null, object, or other unsupported types.
fn build_conditions(filters: &HashMap<String, serde_json::Value>) -> Result<Vec<Condition>> {
    let mut conditions = Vec::new();
    let mut mtime_gte: Option<f64> = None;
    let mut mtime_lte: Option<f64> = None;

    for (key, value) in filters {
        // Special-case: mtime range sentinels inserted by retrieval.rs
        match key.as_str() {
            "mtime__gte" => {
                if let Some(i) = value.as_i64() {
                    mtime_gte = Some(i as f64);
                }
                continue;
            }
            "mtime__lte" => {
                if let Some(i) = value.as_i64() {
                    mtime_lte = Some(i as f64);
                }
                continue;
            }
            _ => {}
        }

        let condition = match value {
            serde_json::Value::Array(arr) => {
                let mut string_values: Vec<String> = Vec::with_capacity(arr.len());
                for v in arr {
                    match v.as_str() {
                        Some(s) => string_values.push(s.to_string()),
                        None => anyhow::bail!(
                            "Array filter for key '{}' contains a non-string element: {}",
                            key,
                            v
                        ),
                    }
                }
                Condition::matches(key, string_values)
            }
            serde_json::Value::String(s) => Condition::matches(key, s.clone()),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Condition::matches(key, i)
                } else {
                    anyhow::bail!(
                        "Float filter values are not supported for key '{}': \
                         exact float equality is unreliable due to floating-point precision. \
                         Use an integer filter instead.",
                        key
                    );
                }
            }
            serde_json::Value::Bool(b) => Condition::from(FieldCondition {
                key: key.clone(),
                r#match: Some(Match {
                    match_value: Some(qdrant_client::qdrant::r#match::MatchValue::Boolean(*b)),
                }),
                ..Default::default()
            }),
            serde_json::Value::Null => {
                anyhow::bail!("Unsupported filter value type: null for key '{}'", key);
            }
            serde_json::Value::Object(_) => {
                anyhow::bail!("Unsupported filter value type: object for key '{}'", key);
            }
        };
        conditions.push(condition);
    }

    if mtime_gte.is_some() || mtime_lte.is_some() {
        conditions.push(Condition::from(FieldCondition {
            key: "mtime".to_string(),
            range: Some(Range {
                gte: mtime_gte,
                lte: mtime_lte,
                ..Default::default()
            }),
            ..Default::default()
        }));
    }

    Ok(conditions)
}

/// A condition that can never match, for an empty `any_of` filter set — or, per
/// #182, a `path_prefix` needle that resolved to no documents at all.
///
/// Mirrors `state::StateDb::push_where`'s `AND 0 = 1` convention: Qdrant's filter
/// grammar has no literal boolean constant, so this ANDs a structural check
/// (`is_null`) against its own negation instead. Built from `is_null` rather than a
/// value-typed condition so it never has to guess the field's Qdrant match kind.
pub(crate) fn unsatisfiable(field: &str) -> Condition {
    Condition::from(Filter {
        must: vec![Condition::is_null(field)],
        must_not: vec![Condition::is_null(field)],
        ..Default::default()
    })
}

/// Build the single- or multi-value match condition for one field, given its
/// declared [`IndexKind`] — a field absent from the caller's indexed-fields map
/// defaults to [`IndexKind::Keyword`], matching the untyped string behaviour
/// `build_conditions` above has always had for `domain`/`type`/`tags`.
///
/// `values` are the canonicalized text spellings [`document_fields::canonical_text`]
/// produces (the same ones `state::StateDb::push_where` matches against
/// `document_fields.value_text`) — this re-parses them back to the Qdrant-native
/// type so a keyword-shaped string never gets compared against an integer or
/// boolean payload value, which would just never match.
fn match_condition(field: &str, values: &[String], kind: IndexKind) -> Result<Condition, String> {
    if values.is_empty() {
        return Ok(unsatisfiable(field));
    }
    match kind {
        IndexKind::Keyword => Ok(Condition::matches(field.to_string(), values.to_vec())),
        IndexKind::Integer => {
            let ints: Vec<i64> = values
                .iter()
                .map(|v| {
                    v.parse::<i64>().map_err(|_| {
                        format!(
                            "filter '{field}': '{v}' is not a valid integer for this field's \
                             declared type"
                        )
                    })
                })
                .collect::<Result<_, _>>()?;
            Ok(Condition::matches(field.to_string(), ints))
        }
        IndexKind::Bool => {
            let bools: Vec<bool> = values
                .iter()
                .map(|v| {
                    v.parse::<bool>().map_err(|_| {
                        format!(
                            "filter '{field}': '{v}' is not a valid boolean for this field's \
                             declared type"
                        )
                    })
                })
                .collect::<Result<_, _>>()?;
            if bools.len() == 1 {
                Ok(Condition::matches(field.to_string(), bools[0]))
            } else {
                // No `MatchValue` variant matches "any of these booleans" directly —
                // OR the individual equality conditions together instead.
                Ok(Condition::from(Filter::should(
                    bools
                        .into_iter()
                        .map(|b| Condition::matches(field.to_string(), b)),
                )))
            }
        }
        IndexKind::Float => Err(format!(
            "filter '{field}': equality/any-of/all-of matching is not supported on a float \
             field — exact float comparison is unreliable; use a numeric range \
             (gte/lte/gt/lt) instead"
        )),
    }
}

/// Lower a document listing's parsed field filters (see [`state::FieldFilter`]) to
/// Qdrant conditions, for the `search` tool's query+filters combinations (both chunk
/// and grouped-document granularity): the same parsed representation
/// `state::StateDb::push_where` lowers to SQL, lowered here to Qdrant conditions
/// instead so a call carrying both a semantic `query` and `filters` can run entirely
/// against Qdrant rather than needing a second SQLite round trip.
///
/// `indexed` supplies each field's declared payload-index kind (from
/// [`crate::schema::SchemaCache::all_indexed_fields`]) so equality/any-of/all-of
/// matching is built with the right Qdrant match type; a field absent from the map
/// is treated as a keyword. This function does not itself reject an unindexed
/// field — Qdrant filters one correctly, just by a full scan rather than an index —
/// so a caller that must refuse silently-slow filtering (the MCP `search` tool, per
/// its rich `filters` contract) checks membership in `indexed` itself before calling.
pub fn lower_field_filters(
    filters: &[(String, FieldFilter)],
    indexed: &HashMap<String, IndexKind>,
) -> Result<Vec<Condition>, String> {
    let mut conditions = Vec::with_capacity(filters.len());
    for (field, filter) in filters {
        let kind = indexed.get(field).copied().unwrap_or(IndexKind::Keyword);
        match filter {
            FieldFilter::AnyOf(values) => conditions.push(match_condition(field, values, kind)?),
            FieldFilter::AllOf(values) => {
                // ANDed together implicitly: every entry in the returned Vec becomes
                // a separate `must` condition in the caller's `Filter::must(..)`.
                for value in values {
                    conditions.push(match_condition(field, std::slice::from_ref(value), kind)?);
                }
            }
            FieldFilter::Range { gte, lte, gt, lt } => {
                if kind == IndexKind::Keyword || kind == IndexKind::Bool {
                    return Err(format!(
                        "filter '{field}': a numeric range (gte/lte/gt/lt) was requested but \
                         the field is indexed as {kind:?}; declare it `type: integer` or \
                         `type: number` in the governing .kb-schema.yaml to use a range filter"
                    ));
                }
                conditions.push(Condition::from(FieldCondition {
                    key: field.clone(),
                    range: Some(Range {
                        gte: *gte,
                        lte: *lte,
                        gt: *gt,
                        lt: *lt,
                    }),
                    ..Default::default()
                }));
            }
        }
    }
    Ok(conditions)
}

/// Build the `QueryPoints` request for [`QdrantStore::recommend_by_point_id`].
///
/// Pulled out as a pure function (mirroring `build_conditions` above) so the
/// request shape — named `dense` vector, nearest-to-an-existing-point query,
/// limit, and optional filter — is unit-testable without a live Qdrant server.
fn build_recommend_query(
    collection: &str,
    point_id: &str,
    limit: u64,
    filter: Option<Filter>,
) -> QueryPointsBuilder {
    let mut builder = QueryPointsBuilder::new(collection)
        .query(Query::new_nearest(VectorInput::new_id(point_id)))
        .using("dense")
        .limit(limit)
        .with_payload(true);
    if let Some(filter) = filter {
        builder = builder.filter(filter);
    }
    builder
}

/// Build the shared dense/sparse/phrase prefetch arm set for a fused RRF query —
/// used identically by [`QdrantStore::hybrid_search`] and
/// [`QdrantStore::search_grouped`] (via [`build_query_groups_request`]) so grouped
/// and chunk retrieval can never diverge on which documents are retrievable, only
/// on result shape. Dense is always included; sparse is added when given (hybrid
/// enabled and the query produced a non-empty sparse vector); one phrase arm is
/// added when `phrases` is non-empty, ANDing one exact-phrase condition per entry
/// onto a dense-ranked arm. Every arm carries the same `conditions`.
///
/// The phrase arm's own "score" is just the dense cosine similarity — the exact
/// phrase requirement is a *filter*, not a distinct scoring function, so it
/// contributes no scoring signal of its own. The tiering callers actually want —
/// exact phrase ranks above all-terms-present ranks above merely-similar — emerges
/// from RRF alone: a chunk containing the phrase tends to also score well on the
/// dense and sparse arms too, so it accumulates reciprocal rank from all three
/// arms instead of just one or two. This is deliberate; do not replace it with a
/// hand-tuned per-arm weight.
fn build_fusion_arms(
    dense: Vec<f32>,
    sparse: Option<(Vec<u32>, Vec<f32>)>,
    phrases: &[String],
    conditions: &[Condition],
    rrf_candidates: u64,
) -> Vec<PrefetchQuery> {
    let mut arms = Vec::with_capacity(3);

    let mut dense_arm = PrefetchQueryBuilder::default()
        .using("dense")
        .query(Query::new_nearest(VectorInput::new_dense(dense.clone())))
        .limit(rrf_candidates);
    if !conditions.is_empty() {
        dense_arm = dense_arm.filter(Filter::must(conditions.to_vec()));
    }
    arms.push(dense_arm.into());

    if let Some((indices, values)) = sparse {
        let mut sparse_arm = PrefetchQueryBuilder::default()
            .using("sparse")
            .query(Query::new_nearest(VectorInput::new_sparse(indices, values)))
            .limit(rrf_candidates);
        if !conditions.is_empty() {
            sparse_arm = sparse_arm.filter(Filter::must(conditions.to_vec()));
        }
        arms.push(sparse_arm.into());
    }

    if !phrases.is_empty() {
        let mut phrase_conditions = conditions.to_vec();
        phrase_conditions.extend(
            phrases
                .iter()
                .map(|p| Condition::matches_phrase("text", p.clone())),
        );
        let phrase_arm = PrefetchQueryBuilder::default()
            .using("dense")
            .query(Query::new_nearest(VectorInput::new_dense(dense)))
            .filter(Filter::must(phrase_conditions))
            .limit(rrf_candidates);
        arms.push(phrase_arm.into());
    }

    arms
}

/// Build the `QueryPointGroups` request for [`QdrantStore::search_grouped`], pulled
/// out as a pure function (mirroring [`build_recommend_query`] above) so the request
/// shape is unit-testable without a live Qdrant server.
///
/// Dense-only (a plain per-group nearest-neighbor query, preserving historical
/// behaviour exactly) when `sparse` is `None` and `phrases` is empty; otherwise
/// builds the same [`build_fusion_arms`] prefetch set `hybrid_search` uses and
/// fuses via server-side RRF before Qdrant groups the fused results — see that
/// function's doc comment for why grouped search shares this rather than staying
/// dense-only-always.
#[allow(clippy::too_many_arguments)]
fn build_query_groups_request(
    collection: &str,
    vector: Vec<f32>,
    sparse: Option<(Vec<u32>, Vec<f32>)>,
    phrases: &[String],
    conditions: Vec<Condition>,
    group_by: &str,
    group_size: u64,
    limit: u64,
    rrf_candidates: u64,
) -> QueryPointGroupsBuilder {
    let mut builder = QueryPointGroupsBuilder::new(collection, group_by)
        .group_size(group_size)
        .limit(limit)
        .with_payload(true);

    if sparse.is_some() || !phrases.is_empty() {
        let arms = build_fusion_arms(vector, sparse, phrases, &conditions, rrf_candidates);
        for arm in arms {
            builder = builder.add_prefetch(arm);
        }
        builder = builder.query(Query::new_fusion(Fusion::Rrf));
    } else {
        builder = builder
            .query(Query::new_nearest(VectorInput::new_dense(vector)))
            .using("dense");
        if !conditions.is_empty() {
            builder = builder.filter(Filter::must(conditions));
        }
    }
    builder
}

impl QdrantStore {
    pub fn new(config: &ResolvedQdrantConfig) -> Result<Self> {
        let client = Qdrant::from_url(&config.url)
            // The client's compatibility probe prints to *stdout*, not the tracing
            // subscriber — which corrupts `status --json` and any other machine-readable
            // output. The server/client versions are pinned together in compose, so the
            // check buys nothing here.
            .skip_compatibility_check()
            .build()
            .context("Failed to connect to Qdrant")?;
        info!("Connected to Qdrant at {}", config.url);
        Ok(Self { client })
    }

    pub async fn drop_collection(&self, collection: &str) -> Result<()> {
        let exists = self
            .client
            .collection_exists(collection)
            .await
            .context("Failed to check if collection exists")?;

        if exists {
            info!("Dropping Qdrant collection '{}'", collection);
            self.client
                .delete_collection(collection)
                .await
                .context("Failed to delete collection")?;
        }
        Ok(())
    }

    pub async fn ensure_collection(
        &self,
        collection: &str,
        vector_size: u64,
        indexed_fields: &[IndexedField],
        enable_phrase: bool,
    ) -> Result<()> {
        let exists = self
            .client
            .collection_exists(collection)
            .await
            .context("Failed to check if collection exists")?;

        if !exists {
            info!("Creating Qdrant collection '{}'", collection);

            // Named dense vector ("dense") + named sparse vector ("sparse") with the
            // server-side IDF modifier. Both are always created so toggling
            // `search.hybrid` never requires a reindex.
            let mut vectors_config = VectorsConfigBuilder::default();
            vectors_config.add_named_vector_params(
                "dense",
                VectorParamsBuilder::new(vector_size, Distance::Cosine),
            );

            let mut sparse_config = SparseVectorsConfigBuilder::default();
            sparse_config.add_named_vector_params(
                "sparse",
                SparseVectorParamsBuilder::default().modifier(Modifier::Idf as i32),
            );

            self.client
                .create_collection(
                    CreateCollectionBuilder::new(collection)
                        .vectors_config(vectors_config)
                        .sparse_vectors_config(sparse_config),
                )
                .await
                .context("Failed to create collection")?;
            info!("Created collection '{}'", collection);
        } else {
            debug!("Collection '{}' already exists", collection);

            // #159: an existing collection's dense-vector dimension is never revisited
            // here otherwise — the block above only sets `vectors_config` when the
            // collection is being created fresh. If an operator changes
            // `embedding.vector_size` (or swaps to a differently-dimensioned embedding
            // model) in config.yaml and restarts `serve` with a plain restart — not
            // `md-kb-rag index --full`, which always `drop_collection`s before calling
            // this (see `ingest::index_paths`) — the mismatch used to go uncaught here
            // and surface only much later, as a dimension-mismatch error out of
            // `upsert_pending`'s embed/upsert call. That error contains no
            // `"(strict mode)"` substring, so `reindex::is_permanent_failure`
            // misclassified it as transient: retried `MAX_RETRY_ATTEMPTS` times with
            // backoff, then repeated by every subsequent periodic reconcile sweep,
            // forever, with nothing in the log pointing at the actual cause. Catch it
            // here instead, loudly and immediately, before a single point is ever
            // embedded against the wrong dimension.
            let info = self
                .client
                .collection_info(collection)
                .await
                .context("Failed to fetch existing collection info for a dimension check")?;

            let existing_size = info
                .result
                .and_then(|r| r.config)
                .and_then(|c| c.params)
                .and_then(|p| {
                    p.vectors_config
                        .and_then(|vc| vc.config)
                        .and_then(|cfg| match cfg {
                            // The dense vector is always named "dense" (see the `!exists`
                            // branch above), so a `ParamsMap` collection looks it up by name.
                            // The unnamed `Params` variant is handled too, defensively — it
                            // would only appear if a collection were ever created outside this
                            // function's own vectors_config shape.
                            VectorsConfigOneof::Params(p) => Some(p.size),
                            VectorsConfigOneof::ParamsMap(m) => m.map.get("dense").map(|p| p.size),
                        })
                });

            if let Some(existing_size) = existing_size
                && existing_size != vector_size
            {
                anyhow::bail!(
                    "Qdrant collection '{collection}' already exists with dense-vector \
                     dimension {existing_size}, but the configured embedding model \
                     produces dimension {vector_size}. This usually means \
                     `embedding.vector_size` (or the embedding model itself) changed \
                     without a full reindex — a plain restart cannot fix this, since it \
                     never rebuilds the collection (only `index --full` does, by \
                     dropping it first). Run `md-kb-rag index --full` to drop and \
                     rebuild the collection at the new dimension, then restart `serve`."
                );
            }
        }

        for indexed in indexed_fields {
            let kind = indexed.kind.to_qdrant();
            debug!(
                "Ensuring {:?} index on field '{}' in collection '{}'",
                kind, indexed.name, collection
            );
            let result = self
                .client
                .create_field_index(CreateFieldIndexCollectionBuilder::new(
                    collection,
                    &indexed.name,
                    kind,
                ))
                .await;

            match result {
                // Recorded on success too, so a field that recovers on a later run
                // clears its previous failure instead of looking broken forever.
                Ok(_) => crate::status::INDEX_STATUS.record_payload_index(&indexed.name, None),
                Err(e) => {
                    // Creating an index that already exists with the same type is a no-op,
                    // so a failure here usually means the declared type changed and Qdrant
                    // is holding an index of the old kind. Dropping and recreating it would
                    // be destructive on a live collection, and failing the whole run because
                    // of one field is worse than proceeding — so warn precisely and carry on.
                    // Deliberate: failing the whole run — and therefore server startup — over
                    // one field is worse than proceeding without its index, and dropping a
                    // live index to recreate it is destructive. But a filter on this field
                    // may now be slow or incomplete, so this is an error, not a warning.
                    error!(
                        "Could not ensure {:?} index on '{}' in collection '{}': {:#}. \
                         Filters on this field may be slow or return incomplete results. \
                         If its declared type changed, delete the payload index in Qdrant \
                         and reindex.",
                        kind, indexed.name, collection, e
                    );
                    // An error! that scrolls out of the log buffer leaves a silently
                    // degraded filter behind; /status keeps it visible until it is fixed.
                    crate::status::INDEX_STATUS.record_payload_index(
                        &indexed.name,
                        Some(crate::status::redact_error(&format!("{e:#}"))),
                    );
                }
            }
        }

        // Ensure integer index on mtime for range-filter queries (idempotent).
        //
        // Non-fatal for the same reason as the schema-declared indexes above: Qdrant
        // filters correctly without a payload index, just more slowly, so failing
        // startup over one index is worse than proceeding loudly without it.
        match self
            .client
            .create_field_index(CreateFieldIndexCollectionBuilder::new(
                collection,
                "mtime",
                FieldType::Integer,
            ))
            .await
        {
            Ok(_) => crate::status::INDEX_STATUS.record_payload_index("mtime", None),
            Err(e) => {
                error!(
                    "Could not ensure the integer index on 'mtime' in collection '{}': {:#}. \
                     Recency filters may be slow until this is resolved.",
                    collection, e
                );
                crate::status::INDEX_STATUS.record_payload_index(
                    "mtime",
                    Some(crate::status::redact_error(&format!("{e:#}"))),
                );
            }
        }

        // Keyword index on `path_ancestors` (#130), unconditional and non-fatal for
        // the same reasons as the `mtime` index just above: it is a derived field
        // every point carries going forward (not a schema-declared one, so it isn't
        // in `indexed_fields`), and Qdrant filters correctly without the index —
        // just by an unindexed scan — so a creation failure degrades performance,
        // not correctness, and must not fail startup.
        //
        // A keyword index on an array-valued field indexes every element, which is
        // exactly what `path_filter_condition`'s `Condition::matches` (single-value
        // "does the array contain this element") needs to run as an index lookup
        // rather than a full collection scan.
        match self
            .client
            .create_field_index(CreateFieldIndexCollectionBuilder::new(
                collection,
                PATH_ANCESTORS_KEY,
                FieldType::Keyword,
            ))
            .await
        {
            Ok(_) => crate::status::INDEX_STATUS.record_payload_index(PATH_ANCESTORS_KEY, None),
            Err(e) => {
                error!(
                    "Could not ensure the keyword index on '{}' in collection '{}': {:#}. \
                     path_prefix filtering may be slow until this is resolved.",
                    PATH_ANCESTORS_KEY, collection, e
                );
                crate::status::INDEX_STATUS.record_payload_index(
                    PATH_ANCESTORS_KEY,
                    Some(crate::status::redact_error(&format!("{e:#}"))),
                );
            }
        }

        // Phrase-matching text index on `text` (search.phrase). Config-gated: when
        // disabled, skip creation entirely and leave no "text" entry behind, so
        // `IndexStatus::phrase_matching_available` reads `false` and quotes stay
        // literal characters (see that method's doc comment).
        //
        // CRITICAL — graceful degradation: an older Qdrant server does not support
        // `phrase_matching` and rejects this call. That must never fail startup, so
        // this follows the exact same tolerate-and-record pattern as the indexes
        // above: log a clear warning, record the failure, and carry on. Retrieval
        // reads the recorded outcome (not this call's return value) before ever
        // adding a phrase prefetch arm, so a query against a server that never
        // confirmed support degrades to ordinary term+semantic retrieval instead of
        // erroring.
        if enable_phrase {
            let text_index_params =
                TextIndexParamsBuilder::new(TokenizerType::Word).phrase_matching(true);
            match self
                .client
                .create_field_index(
                    CreateFieldIndexCollectionBuilder::new(collection, "text", FieldType::Text)
                        .field_index_params(text_index_params),
                )
                .await
            {
                Ok(_) => crate::status::INDEX_STATUS.record_payload_index("text", None),
                Err(e) => {
                    warn!(
                        "Could not create the phrase-matching text index on 'text' in \
                         collection '{}': {:#}. This usually means the Qdrant server is too \
                         old to support phrase_matching — phrase search (double-quoted terms \
                         in `search`) is disabled for this process; ordinary term/semantic \
                         search is unaffected.",
                        collection, e
                    );
                    crate::status::INDEX_STATUS.record_payload_index(
                        "text",
                        Some(crate::status::redact_error(&format!("{e:#}"))),
                    );
                }
            }
        }

        info!(
            collection,
            fields = indexed_fields.len(),
            "Collection ready"
        );

        Ok(())
    }

    pub async fn upsert_points(&self, collection: &str, points: Vec<QdrantPoint>) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }

        let point_count = points.len();
        let structs: Vec<PointStruct> = points
            .into_iter()
            .map(|p| {
                let payload = json_payload_to_qdrant(&p.payload);
                let mut vectors =
                    NamedVectors::default().add_vector("dense", Vector::new_dense(p.vector));
                if let Some((indices, values)) = p.sparse {
                    vectors = vectors.add_vector("sparse", Vector::new_sparse(indices, values));
                }
                PointStruct::new(p.id, vectors, payload)
            })
            .collect();

        self.client
            .upsert_points(UpsertPointsBuilder::new(collection, structs))
            .await
            .context("Failed to upsert points")?;

        debug!("Upserted {} points into '{}'", point_count, collection);
        Ok(())
    }

    pub async fn delete_by_files(&self, collection: &str, file_paths: &[&str]) -> Result<()> {
        if file_paths.is_empty() {
            return Ok(());
        }

        let values: Vec<String> = file_paths.iter().map(|s| s.to_string()).collect();
        let filter = Filter::must([Condition::matches("file_path", values)]);

        self.client
            .delete_points(DeletePointsBuilder::new(collection).points(filter))
            .await
            .context("Failed to batch-delete points by file paths")?;

        debug!(
            "Batch-deleted points for {} file(s) from collection '{}'",
            file_paths.len(),
            collection
        );
        Ok(())
    }

    pub async fn delete_points_by_ids(&self, collection: &str, ids: Vec<String>) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }

        let point_count = ids.len();

        self.client
            .delete_points(DeletePointsBuilder::new(collection).points(ids))
            .await
            .context("Failed to delete points by IDs")?;

        debug!(
            "Deleted {} points by ID from collection '{}'",
            point_count, collection
        );
        Ok(())
    }

    /// See [`VectorStore::collection_point_count`] for the full rationale (exact vs.
    /// `collection_info`'s approximate count, and why this exists on the trait at
    /// all). Mirrors `drop_collection`/`ensure_collection`'s existence-check pattern:
    /// a collection that does not exist has 0 points rather than being an error —
    /// #155's caller needs a plain deficit number even when the collection was
    /// dropped outright, not merely emptied.
    pub async fn collection_point_count(&self, collection: &str) -> Result<u64> {
        let exists = self
            .client
            .collection_exists(collection)
            .await
            .context("Failed to check if collection exists")?;

        if !exists {
            return Ok(0);
        }

        let response = self
            .client
            .count(CountPointsBuilder::new(collection).exact(true))
            .await
            .context("Failed to count points in collection")?;

        Ok(response.result.map(|r| r.count).unwrap_or(0))
    }
}

/// Thin delegation impls — each method calls the identically-named inherent method
/// on `QdrantStore`. The inherent methods are the real implementations; these impls
/// exist only to satisfy the `VectorStore` trait used by `ingest.rs`.
impl VectorStore for QdrantStore {
    async fn upsert_points(&self, collection: &str, points: Vec<QdrantPoint>) -> Result<()> {
        QdrantStore::upsert_points(self, collection, points).await
    }

    async fn delete_by_files(&self, collection: &str, file_paths: &[&str]) -> Result<()> {
        QdrantStore::delete_by_files(self, collection, file_paths).await
    }

    async fn delete_points_by_ids(&self, collection: &str, ids: Vec<String>) -> Result<()> {
        QdrantStore::delete_points_by_ids(self, collection, ids).await
    }

    async fn drop_collection(&self, collection: &str) -> Result<()> {
        QdrantStore::drop_collection(self, collection).await
    }

    async fn ensure_collection(
        &self,
        collection: &str,
        vector_size: u64,
        indexed_fields: &[IndexedField],
        enable_phrase: bool,
    ) -> Result<()> {
        QdrantStore::ensure_collection(self, collection, vector_size, indexed_fields, enable_phrase)
            .await
    }

    async fn collection_point_count(&self, collection: &str) -> Result<u64> {
        QdrantStore::collection_point_count(self, collection).await
    }
}

/// Thin delegation impls — each method calls the identically-named inherent method
/// on `QdrantStore`. The inherent methods are the real implementations; these impls
/// exist only to satisfy the `RetrievalStore` trait used by `retrieval.rs`.
impl RetrievalStore for QdrantStore {
    async fn search(
        &self,
        collection: &str,
        vector: Vec<f32>,
        filters: std::collections::HashMap<String, serde_json::Value>,
        extra_conditions: Vec<Condition>,
        limit: u64,
    ) -> Result<Vec<SearchResult>> {
        QdrantStore::search(self, collection, vector, filters, extra_conditions, limit).await
    }

    async fn hybrid_search(
        &self,
        collection: &str,
        dense: Vec<f32>,
        sparse: Option<(Vec<u32>, Vec<f32>)>,
        phrases: &[String],
        filters: std::collections::HashMap<String, serde_json::Value>,
        extra_conditions: Vec<Condition>,
        limit: u64,
        rrf_candidates: u64,
        explain: bool,
    ) -> Result<Vec<SearchResult>> {
        QdrantStore::hybrid_search(
            self,
            collection,
            dense,
            sparse,
            phrases,
            filters,
            extra_conditions,
            limit,
            rrf_candidates,
            explain,
        )
        .await
    }

    async fn search_grouped(
        &self,
        collection: &str,
        vector: Vec<f32>,
        sparse: Option<(Vec<u32>, Vec<f32>)>,
        phrases: &[String],
        filters: std::collections::HashMap<String, serde_json::Value>,
        extra_conditions: Vec<Condition>,
        group_by: &str,
        group_size: u64,
        limit: u64,
        rrf_candidates: u64,
    ) -> Result<Vec<SearchResult>> {
        QdrantStore::search_grouped(
            self,
            collection,
            vector,
            sparse,
            phrases,
            filters,
            extra_conditions,
            group_by,
            group_size,
            limit,
            rrf_candidates,
        )
        .await
    }
}

/// Extract string values from facet hits, skipping non-string variants.
fn extract_facet_strings(hits: Vec<FacetHit>) -> Vec<String> {
    hits.into_iter()
        .filter_map(|hit| {
            hit.value.and_then(|v| match v.variant {
                Some(facet_value::Variant::StringValue(s)) => Some(s),
                _ => None,
            })
        })
        .collect()
}

impl QdrantStore {
    pub async fn search(
        &self,
        collection: &str,
        vector: Vec<f32>,
        filters: HashMap<String, serde_json::Value>,
        extra_conditions: Vec<Condition>,
        limit: u64,
    ) -> Result<Vec<SearchResult>> {
        let mut conditions = build_conditions(&filters)?;
        conditions.extend(extra_conditions);

        // Target the named "dense" vector (the collection no longer has an unnamed
        // default vector after the named-vector migration).
        let mut builder = SearchPointsBuilder::new(collection, vector, limit)
            .vector_name("dense")
            .with_payload(true);
        if !conditions.is_empty() {
            builder = builder.filter(Filter::must(conditions));
        }

        let response = self
            .client
            .search_points(builder)
            .await
            .context("Failed to search points")?;

        let results = response
            .result
            .into_iter()
            .map(|scored| SearchResult {
                score: scored.score,
                pre_rerank_score: None,
                dense_score: Some(scored.score),
                sparse_score: None,
                phrase_score: None,
                payload: qdrant_payload_to_json(&scored.payload),
            })
            .collect();

        Ok(results)
    }

    /// Fused retrieval via the Qdrant Query API. Builds the shared dense/sparse/
    /// phrase prefetch arm set ([`build_fusion_arms`]) and fuses server-side with
    /// Reciprocal Rank Fusion, returning the top `limit`. At least one of
    /// `sparse`/`phrases` must be given — see the trait doc comment.
    #[allow(clippy::too_many_arguments)]
    pub async fn hybrid_search(
        &self,
        collection: &str,
        dense: Vec<f32>,
        sparse: Option<(Vec<u32>, Vec<f32>)>,
        phrases: &[String],
        filters: HashMap<String, serde_json::Value>,
        extra_conditions: Vec<Condition>,
        limit: u64,
        rrf_candidates: u64,
        explain: bool,
    ) -> Result<Vec<SearchResult>> {
        let mut conditions = build_conditions(&filters)?;
        conditions.extend(extra_conditions);

        if explain {
            return self
                .hybrid_search_explain(
                    collection,
                    dense,
                    sparse,
                    phrases,
                    conditions,
                    limit,
                    rrf_candidates,
                )
                .await;
        }

        let arms = build_fusion_arms(dense, sparse, phrases, &conditions, rrf_candidates);
        let mut builder = QueryPointsBuilder::new(collection)
            .query(Query::new_fusion(Fusion::Rrf))
            .limit(limit)
            .with_payload(true);
        for arm in arms {
            builder = builder.add_prefetch(arm);
        }

        let response = self
            .client
            .query(builder)
            .await
            .context("Failed to run hybrid search")?;

        let results = response
            .result
            .into_iter()
            .map(|scored| SearchResult {
                score: scored.score,
                pre_rerank_score: None,
                dense_score: None,
                sparse_score: None,
                phrase_score: None,
                payload: qdrant_payload_to_json(&scored.payload),
            })
            .collect();

        Ok(results)
    }

    /// Hybrid search with client-side RRF (k=60). Used when `explain=true` to
    /// surface per-arm dense/sparse/phrase scores alongside the fused score.
    #[allow(clippy::too_many_arguments)]
    async fn hybrid_search_explain(
        &self,
        collection: &str,
        dense: Vec<f32>,
        sparse: Option<(Vec<u32>, Vec<f32>)>,
        phrases: &[String],
        conditions: Vec<Condition>,
        limit: u64,
        rrf_candidates: u64,
    ) -> Result<Vec<SearchResult>> {
        // Dense arm
        let mut dense_builder = SearchPointsBuilder::new(collection, dense.clone(), rrf_candidates)
            .vector_name("dense")
            .with_payload(true);
        if !conditions.is_empty() {
            dense_builder = dense_builder.filter(Filter::must(conditions.clone()));
        }
        let dense_resp = self
            .client
            .search_points(dense_builder)
            .await
            .context("Failed to run dense arm for explain")?;

        // Sparse arm via QueryPoints (supports sparse named vectors) — only when hybrid.
        let sparse_resp = if let Some((sparse_indices, sparse_values)) = sparse {
            let mut sparse_builder = QueryPointsBuilder::new(collection)
                .query(Query::new_nearest(VectorInput::new_sparse(
                    sparse_indices,
                    sparse_values,
                )))
                .using("sparse")
                .limit(rrf_candidates)
                .with_payload(true);
            if !conditions.is_empty() {
                sparse_builder = sparse_builder.filter(Filter::must(conditions.clone()));
            }
            Some(
                self.client
                    .query(sparse_builder)
                    .await
                    .context("Failed to run sparse arm for explain")?,
            )
        } else {
            None
        };

        // Phrase arm, dense-ranked and phrase-filtered — only when phrases given.
        let phrase_resp = if !phrases.is_empty() {
            let mut phrase_conditions = conditions.clone();
            phrase_conditions.extend(
                phrases
                    .iter()
                    .map(|p| Condition::matches_phrase("text", p.clone())),
            );
            let phrase_builder = SearchPointsBuilder::new(collection, dense, rrf_candidates)
                .vector_name("dense")
                .filter(Filter::must(phrase_conditions))
                .with_payload(true);
            Some(
                self.client
                    .search_points(phrase_builder)
                    .await
                    .context("Failed to run phrase arm for explain")?,
            )
        } else {
            None
        };

        // Client-side RRF — key by file_path::chunk_index from payload
        struct RrfAccum {
            rrf_score: f32,
            dense_score: Option<f32>,
            sparse_score: Option<f32>,
            phrase_score: Option<f32>,
            payload: HashMap<String, serde_json::Value>,
        }

        // Key by file_path::chunk_index — both fields are always present in the
        // indexed schema; missing fields collapse to ""::0 (logged at debug level).
        fn payload_key(payload: &HashMap<String, serde_json::Value>) -> String {
            let fp = payload
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let ci = payload
                .get("chunk_index")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            format!("{fp}::{ci}")
        }

        let k = 60.0_f32;
        let mut accum: HashMap<String, RrfAccum> = HashMap::new();

        for (rank, scored) in dense_resp.result.iter().enumerate() {
            let payload = qdrant_payload_to_json(&scored.payload);
            let key = payload_key(&payload);
            let rrf = 1.0 / (k + rank as f32 + 1.0);
            let entry = accum.entry(key).or_insert(RrfAccum {
                rrf_score: 0.0,
                dense_score: None,
                sparse_score: None,
                phrase_score: None,
                payload,
            });
            entry.rrf_score += rrf;
            entry.dense_score = Some(scored.score);
        }

        if let Some(sparse_resp) = &sparse_resp {
            for (rank, scored) in sparse_resp.result.iter().enumerate() {
                let payload = qdrant_payload_to_json(&scored.payload);
                let key = payload_key(&payload);
                let rrf = 1.0 / (k + rank as f32 + 1.0);
                let entry = accum.entry(key).or_insert(RrfAccum {
                    rrf_score: 0.0,
                    dense_score: None,
                    sparse_score: None,
                    phrase_score: None,
                    payload,
                });
                entry.rrf_score += rrf;
                entry.sparse_score = Some(scored.score);
            }
        }

        if let Some(phrase_resp) = &phrase_resp {
            for (rank, scored) in phrase_resp.result.iter().enumerate() {
                let payload = qdrant_payload_to_json(&scored.payload);
                let key = payload_key(&payload);
                let rrf = 1.0 / (k + rank as f32 + 1.0);
                let entry = accum.entry(key).or_insert(RrfAccum {
                    rrf_score: 0.0,
                    dense_score: None,
                    sparse_score: None,
                    phrase_score: None,
                    payload,
                });
                entry.rrf_score += rrf;
                entry.phrase_score = Some(scored.score);
            }
        }

        let mut ranked: Vec<RrfAccum> = accum.into_values().collect();
        ranked.sort_by(|a, b| {
            b.rrf_score
                .partial_cmp(&a.rrf_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked.truncate(limit as usize);

        Ok(ranked
            .into_iter()
            .map(|a| SearchResult {
                score: a.rrf_score,
                pre_rerank_score: None,
                dense_score: a.dense_score,
                sparse_score: a.sparse_score,
                phrase_score: a.phrase_score,
                payload: a.payload,
            })
            .collect())
    }

    /// Nearest-neighbor search grouped by a payload field, collapsing each group to
    /// its `group_size` best-scoring hits — backs the `search` tool's
    /// query+document (grouped) granularity. See
    /// [`RetrievalStore::search_grouped`]'s doc comment for why this shares
    /// [`build_fusion_arms`] with `hybrid_search` rather than staying dense-only.
    #[allow(clippy::too_many_arguments)]
    pub async fn search_grouped(
        &self,
        collection: &str,
        vector: Vec<f32>,
        sparse: Option<(Vec<u32>, Vec<f32>)>,
        phrases: &[String],
        filters: HashMap<String, serde_json::Value>,
        extra_conditions: Vec<Condition>,
        group_by: &str,
        group_size: u64,
        limit: u64,
        rrf_candidates: u64,
    ) -> Result<Vec<SearchResult>> {
        let mut conditions = build_conditions(&filters)?;
        conditions.extend(extra_conditions);
        let fused = sparse.is_some() || !phrases.is_empty();

        let builder = build_query_groups_request(
            collection,
            vector,
            sparse,
            phrases,
            conditions,
            group_by,
            group_size,
            limit,
            rrf_candidates,
        );

        let response = self
            .client
            .query_groups(builder)
            .await
            .context("Failed to run grouped query")?;

        let results = response
            .result
            .map(|r| r.groups)
            .unwrap_or_default()
            .into_iter()
            // `group_size` bounds each group to its best-scoring hit(s), sorted
            // best-first by Qdrant; the first is exactly the one document-mode wants.
            .filter_map(|group| group.hits.into_iter().next())
            .map(|scored| SearchResult {
                score: scored.score,
                pre_rerank_score: None,
                // Only the plain dense-only path's score IS the dense cosine value;
                // a fused (RRF) score is not, so leave the per-arm fields `None` —
                // same convention as `hybrid_search`'s non-explain results.
                dense_score: if fused { None } else { Some(scored.score) },
                sparse_score: None,
                phrase_score: None,
                payload: qdrant_payload_to_json(&scored.payload),
            })
            .collect();

        Ok(results)
    }

    /// Doc-level k-NN via the Query API, using an existing point's stored vector as
    /// the query instead of a caller-supplied embedding — Qdrant resolves `point_id`
    /// server-side and searches the named `dense` vector for its nearest neighbors.
    ///
    /// Used to precompute the UI's semantic edges: callers pass the deterministic
    /// first-chunk point id for a document (`make_point_id(file_path, 0)`, see
    /// `ingest.rs`), so the neighbors returned are the nearest *documents* by their
    /// opening chunk's dense embedding. `filter`, when given, is applied the same
    /// way `search`'s conditions are — the typical use is excluding the source
    /// document's own chunks via a `file_path` `must_not` condition so a document
    /// never recommends itself.
    pub async fn recommend_by_point_id(
        &self,
        collection: &str,
        point_id: &str,
        limit: u64,
        filter: Option<Filter>,
    ) -> Result<Vec<SearchResult>> {
        let builder = build_recommend_query(collection, point_id, limit, filter);

        let response = self
            .client
            .query(builder)
            .await
            .context("Failed to run recommend-by-point-id query")?;

        let results = response
            .result
            .into_iter()
            .map(|scored| SearchResult {
                score: scored.score,
                pre_rerank_score: None,
                dense_score: Some(scored.score),
                sparse_score: None,
                phrase_score: None,
                payload: qdrant_payload_to_json(&scored.payload),
            })
            .collect();

        Ok(results)
    }

    pub async fn health_check(&self) -> Result<()> {
        self.client
            .health_check()
            .await
            .context("Qdrant health check failed")?;
        Ok(())
    }

    /// Fetch distinct values for a keyword-indexed payload field via Qdrant facets.
    ///
    /// Returns up to `limit` unique string values. Gracefully returns an empty
    /// vec on errors (e.g. empty collection, unindexed field).
    pub async fn fetch_facet_values(
        &self,
        collection: &str,
        field: &str,
        limit: u64,
    ) -> Result<Vec<String>> {
        let builder = FacetCountsBuilder::new(collection, field).limit(limit);
        let response = match self.client.facet(builder).await {
            Ok(resp) => resp,
            Err(e) => {
                debug!(
                    "Facet query for field '{}' failed (may be empty collection): {e}",
                    field
                );
                return Ok(vec![]);
            }
        };
        Ok(extract_facet_strings(response.hits))
    }

    pub async fn collection_info(&self, collection: &str) -> Result<Option<u64>> {
        let exists = self
            .client
            .collection_exists(collection)
            .await
            .context("Failed to check collection existence")?;

        if !exists {
            return Ok(None);
        }

        let info = self
            .client
            .collection_info(collection)
            .await
            .context("Failed to get collection info")?;

        let count = info.result.and_then(|r| r.points_count);

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::FutureExt;

    // --- live-Qdrant (`#[ignore]`d) test helpers -- see #231 ---
    //
    // The `#[ignore]`d tests below all talk to a real Qdrant server
    // (`docker compose up -d qdrant`; run with `cargo test -- --ignored`).
    // CI's `qdrant-integration` job runs that suite under cargo's default
    // multi-threaded runner, so several of these tests hit the same shared
    // server concurrently. Two helpers below exist specifically to make that
    // safe:

    /// Deterministic, per-test collection name. Must be called with the
    /// test's own function name, not a hand-picked string -- a name tied 1:1
    /// to the test that owns it can never collide with another test's
    /// collection (now or as tests are added later) without anyone having to
    /// remember to pick a fresh name by hand. Deterministic rather than
    /// random so a failure is reproducible: `cargo test <fn_name> --
    /// --ignored` always talks to the exact same collection a CI failure did.
    fn live_test_collection(test_name: &str) -> String {
        format!("md-kb-rag-test-{test_name}")
    }

    /// Run a live-Qdrant test body, dropping its collection afterward
    /// whether the body panics or not. Without this, a failed assertion
    /// mid-test skips the trailing `delete_collection` call and orphans the
    /// collection on the server — harmless for a single run, but the orphans
    /// accumulate across every subsequent run and the server ends up
    /// carrying stale collections from every test that ever failed once.
    async fn with_collection_cleanup<F, Fut>(store: &QdrantStore, collection: &str, body: F)
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        let result = std::panic::AssertUnwindSafe(body()).catch_unwind().await;
        let _ = store.client.delete_collection(collection).await;
        if let Err(payload) = result {
            std::panic::resume_unwind(payload);
        }
    }

    /// Retry `probe` (typically a live-Qdrant read taken shortly after a
    /// write) until `ready` accepts its result or `attempts` are exhausted,
    /// sleeping `interval` between tries.
    ///
    /// A prior fixed `tokio::time::sleep(500ms)` before reading back an
    /// upsert was the actual source of #231's reported flakiness: these
    /// tests already had unique collection names (nothing here was ever
    /// dropping a collection out from under another test), but a single
    /// fixed delay assumes indexing latency is constant, and it isn't --
    /// under the default multi-threaded `--ignored` runner, several tests
    /// hit the same shared server at once, so how long a given write takes
    /// to become visible to a subsequent read varies with how loaded the
    /// server happens to be right then. That surfaced as exactly the
    /// "unrelated assertion mismatch" #231 describes (an empty result set,
    /// or a "no point with id" error, where a fixed-length sleep had usually
    /// been enough). Confirmed serially (`--test-threads=1`) these tests
    /// passed every time; only concurrent runs were ever flaky, which is
    /// what pointed at write-visibility timing rather than a genuine
    /// collection collision. Polling with a generous ceiling removes the
    /// dependency on server load while still failing fast if something is
    /// genuinely wrong.
    async fn retry_until<T, F, Fut>(
        attempts: u32,
        interval: std::time::Duration,
        mut probe: F,
        mut ready: impl FnMut(&T) -> bool,
    ) -> T
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = T>,
    {
        let mut last = probe().await;
        for _ in 1..attempts {
            if ready(&last) {
                return last;
            }
            tokio::time::sleep(interval).await;
            last = probe().await;
        }
        last
    }

    #[test]
    fn qdrant_value_roundtrip() {
        let mut payload: HashMap<String, serde_json::Value> = HashMap::new();
        payload.insert("title".into(), serde_json::Value::String("Test Doc".into()));
        payload.insert(
            "file_path".into(),
            serde_json::Value::String("/data/test.md".into()),
        );
        payload.insert(
            "text".into(),
            serde_json::Value::String("Some chunk content".into()),
        );
        payload.insert("chunk_index".into(), serde_json::json!(0));
        payload.insert(
            "tags".into(),
            serde_json::Value::Array(vec![
                serde_json::Value::String("rust".into()),
                serde_json::Value::String("rag".into()),
            ]),
        );

        let qdrant_payload = json_payload_to_qdrant(&payload);
        let roundtripped = qdrant_payload_to_json(&qdrant_payload);

        assert_eq!(
            roundtripped.get("title").and_then(|v| v.as_str()),
            Some("Test Doc")
        );
        assert_eq!(
            roundtripped.get("file_path").and_then(|v| v.as_str()),
            Some("/data/test.md")
        );
        assert_eq!(
            roundtripped.get("text").and_then(|v| v.as_str()),
            Some("Some chunk content")
        );
        assert_eq!(
            roundtripped.get("chunk_index").and_then(|v| v.as_i64()),
            Some(0)
        );
        let tags = roundtripped.get("tags").and_then(|v| v.as_array()).unwrap();
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0].as_str(), Some("rust"));
        assert_eq!(tags[1].as_str(), Some("rag"));
    }

    /// Integration test: upsert a point, search, and verify payload is returned.
    ///
    /// Stays live-only — this exercises the actual wire round trip through a real
    /// Qdrant server: named dense-vector upsert, `search_points` against the "dense"
    /// vector, and struct/list/scalar payload values surviving protobuf encode and
    /// decode. `qdrant_value_roundtrip` above already covers the JSON<->QdrantValue
    /// conversion in isolation; what only a live server can additionally prove is
    /// that the server itself stores and returns those values unchanged, which no
    /// fake can stand in for without just re-asserting the conversion functions.
    ///
    /// Requires a running Qdrant instance at localhost:6334.
    /// Run with: cargo test qdrant_search_returns_payload -- --ignored
    #[tokio::test]
    #[ignore]
    async fn qdrant_search_returns_payload() {
        let config = ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: live_test_collection("qdrant_search_returns_payload"),
        };
        let store = QdrantStore::new(&config).unwrap();

        // Clean up from any prior run
        let _ = store.client.delete_collection(&config.collection).await;

        let vector_size = 4;
        store
            .ensure_collection(&config.collection, vector_size, &[], false)
            .await
            .unwrap();

        with_collection_cleanup(&store, &config.collection, || async {
            let mut payload: HashMap<String, serde_json::Value> = HashMap::new();
            payload.insert("title".into(), serde_json::json!("Test Document"));
            payload.insert("file_path".into(), serde_json::json!("/data/test.md"));
            payload.insert("text".into(), serde_json::json!("Hello world chunk"));

            let point = QdrantPoint {
                id: "00000000-0000-0000-0000-000000000001".into(),
                vector: vec![1.0, 0.0, 0.0, 0.0],
                sparse: None,
                payload,
            };
            store
                .upsert_points(&config.collection, vec![point])
                .await
                .unwrap();

            // Poll instead of a single fixed sleep -- see retry_until's doc
            // comment for why (#231).
            let results = retry_until(
                20,
                std::time::Duration::from_millis(250),
                || async {
                    store
                        .search(
                            &config.collection,
                            vec![1.0, 0.0, 0.0, 0.0],
                            HashMap::new(),
                            Vec::new(),
                            1,
                        )
                        .await
                        .unwrap()
                },
                |results| !results.is_empty(),
            )
            .await;

            assert_eq!(results.len(), 1);
            let result = &results[0];
            assert_eq!(
                result.payload.get("title").and_then(|v| v.as_str()),
                Some("Test Document"),
                "search results must include payload fields"
            );
            assert_eq!(
                result.payload.get("file_path").and_then(|v| v.as_str()),
                Some("/data/test.md"),
            );
            assert_eq!(
                result.payload.get("text").and_then(|v| v.as_str()),
                Some("Hello world chunk"),
            );
        })
        .await;
    }

    /// #159: `ensure_collection` against an EXISTING collection must reject a
    /// dense-vector dimension mismatch instead of silently proceeding.
    ///
    /// Before the fix, the `else` (collection-already-exists) branch never looked at
    /// the collection's actual configured dimension at all, so `ensure_collection`
    /// with a different `vector_size` than what the collection was created with
    /// returned `Ok(())` — reproducing the operator mistake #159 describes (an
    /// `embedding.vector_size` config change, or a model swap, applied with a plain
    /// `serve` restart instead of `index --full`). The mismatch would then surface
    /// only much later and far less clearly, out of `upsert_pending`'s embed/upsert
    /// call — and be misclassified as a transient failure and retried forever (see
    /// `reindex::is_permanent_failure`'s doc comment).
    ///
    /// This assertion fails before the fix (`ensure_collection` returns `Ok(())` even
    /// though the collection was created at a different dimension) and passes after
    /// it (`Err`, naming both dimensions and pointing at `index --full`).
    ///
    /// Stays live-only — the thing under test is reading back Qdrant's own
    /// server-stored `VectorParams.size` for an existing collection via
    /// `collection_info`, which no fake `VectorStore` can stand in for without just
    /// re-asserting the comparison logic.
    ///
    /// Requires a running Qdrant instance at localhost:6334.
    /// Run with: cargo test ensure_collection_rejects_a_dimension_mismatch -- --ignored
    #[tokio::test]
    #[ignore]
    async fn ensure_collection_rejects_a_dimension_mismatch() {
        let config = ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: live_test_collection("ensure_collection_rejects_a_dimension_mismatch"),
        };
        let store = QdrantStore::new(&config).unwrap();

        let _ = store.client.delete_collection(&config.collection).await;

        with_collection_cleanup(&store, &config.collection, || async {
            // Create the collection at dimension 4 — as if indexed by an older embedding
            // model.
            store
                .ensure_collection(&config.collection, 4, &[], false)
                .await
                .unwrap();

            // A plain restart with a reconfigured (or swapped) embedding model now produces
            // a different dimension, with no `drop_collection` in between — exactly what
            // `index --full` would do, and a plain `serve` restart never does.
            let result = store
                .ensure_collection(&config.collection, 8, &[], false)
                .await;

            assert!(
                result.is_err(),
                "a dimension mismatch against an existing collection must be rejected"
            );
            let msg = format!("{:#}", result.unwrap_err());
            assert!(
                msg.contains('4'),
                "error should name the existing dimension: {msg}"
            );
            assert!(
                msg.contains('8'),
                "error should name the configured dimension: {msg}"
            );
            assert!(
                msg.contains("index --full"),
                "error should point at the fix: {msg}"
            );

            // A matching call (same dimension as what the collection already holds) must
            // still be a no-op success — this is the overwhelmingly common case (every
            // ordinary scoped indexing run calls `ensure_collection` again) and must not
            // regress.
            store
                .ensure_collection(&config.collection, 4, &[], false)
                .await
                .expect("a matching dimension must not be rejected");
        })
        .await;
    }

    /// Integration test: `ensure_collection(enable_phrase: true)` creates a working
    /// phrase-matching text index, a phrase-filtered *fused* (RRF) query ranks the
    /// chunk containing the exact phrase above one that merely contains the same
    /// words in a different order, and — #133 — the identical phrase condition
    /// applied as a hard `Filter::must` genuinely EXCLUDES the non-matching chunk,
    /// not just outranks it.
    ///
    /// #212: this test originally asserted the phrase-matching chunk was the ONLY
    /// result of the FUSED query (`results.len() == 1`) — i.e. that the phrase
    /// condition acts as a hard filter there. Against a real server (this test could
    /// not previously run at all — see below) that assertion fails: both chunks come
    /// back, `/data/exact.md` first with `/data/reordered.md` second. That is not a
    /// bug; it is exactly what `build_fusion_arms`'s doc comment documents as
    /// deliberate — the phrase condition only ever applies within ONE of the fused
    /// RRF arms (`dense` always runs unfiltered too), so a document absent from the
    /// phrase arm can still surface via the dense arm, just ranked lower because it
    /// only accumulates reciprocal rank from one arm instead of two. Phrase matching
    /// is a *ranking* signal there, not an exclusion filter — the fused-query
    /// assertions below were rewritten to match that actual, intended contract
    /// instead of loosening it to "don't crash".
    ///
    /// #133: that rewrite, however, left NOTHING in the default (non-`--ignored`,
    /// and — pre-#133 — not even CI-run) suite proving Qdrant's phrase filter
    /// actually excludes anything server-side. Every other phrase test (offline,
    /// mocked) only proves the *request* is shaped correctly — that the arm exists,
    /// that `extract_phrases` splits quoted spans, that the filter condition is
    /// present in what gets sent. None of that would catch the payload index
    /// silently failing to build (`ensure_collection` tolerates that failure by
    /// design and only logs), a tokenizer mismatch between indexing and querying, a
    /// wrong field name in the phrase condition, or a Qdrant version whose phrase
    /// semantics differ from what's assumed. The block at the end of this test closes
    /// that gap directly: it applies the same phrase condition through `search`'s
    /// `extra_conditions` (the plain dense-only path, which — unlike `hybrid_search`'s
    /// fused arms — always compiles every condition into one real `Filter::must`) and
    /// asserts the reordered chunk is gone from the result set entirely.
    ///
    /// Stays live-only — this exercises Qdrant's own phrase-matching text index
    /// end to end (index creation, then a real `Condition::matches_phrase` filter
    /// evaluated server-side), which no fake can stand in for. The OTHER half of
    /// this feature — an older server rejecting `phrase_matching` degrading
    /// gracefully rather than failing startup — is covered offline by
    /// `status::tests::phrase_matching_available_false_after_a_failed_index_creation`
    /// and `retrieval::tests::search_with_phrase_disabled_treats_quotes_as_literal_characters`,
    /// since reproducing an incompatible server would need one to test against.
    ///
    /// Requires a running Qdrant instance at localhost:6334.
    /// Run with: cargo test phrase_index_enables_exact_phrase_matching -- --ignored
    #[tokio::test]
    #[ignore]
    async fn phrase_index_enables_exact_phrase_matching() {
        let config = ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: live_test_collection("phrase_index_enables_exact_phrase_matching"),
        };
        let store = QdrantStore::new(&config).unwrap();

        let _ = store.client.delete_collection(&config.collection).await;

        let vector_size = 4;
        store
            .ensure_collection(&config.collection, vector_size, &[], true)
            .await
            .unwrap();

        with_collection_cleanup(&store, &config.collection, || async {
            let make_point = |id: &str, file: &str, text: &str, vec: Vec<f32>| {
                let mut payload = HashMap::new();
                payload.insert("file_path".into(), serde_json::json!(file));
                payload.insert("text".into(), serde_json::json!(text));
                QdrantPoint {
                    id: id.into(),
                    vector: vec,
                    sparse: None,
                    payload,
                }
            };

            let points = vec![
                make_point(
                    // #212: was "...000p01" — 'p' is not a hex digit, so this was never a
                    // valid UUID and the point was rejected; the test had never actually run.
                    "00000000-0000-0000-0000-000000000a01",
                    "/data/exact.md",
                    "deploy notes for node:ares rocm",
                    vec![1.0, 0.0, 0.0, 0.0],
                ),
                make_point(
                    "00000000-0000-0000-0000-000000000a02",
                    "/data/reordered.md",
                    "rocm notes for node:ares deploy",
                    vec![1.0, 0.0, 0.0, 0.0],
                ),
            ];
            store
                .upsert_points(&config.collection, points)
                .await
                .unwrap();

            let phrases = vec!["node:ares rocm".to_string()];
            // `explain: true` so `phrase_score` is populated — its presence (not value) is
            // what proves the phrase arm actually matched, which is a much more precise
            // check than inferring it from ranking order alone.
            //
            // Poll instead of a single fixed sleep -- see retry_until's doc
            // comment for why (#231).
            let results = retry_until(
                20,
                std::time::Duration::from_millis(250),
                || async {
                    store
                        .hybrid_search(
                            &config.collection,
                            vec![1.0, 0.0, 0.0, 0.0],
                            None,
                            &phrases,
                            HashMap::new(),
                            Vec::new(),
                            10,
                            50,
                            true,
                        )
                        .await
                        .unwrap()
                },
                |results| results.len() >= 2,
            )
            .await;

            assert_eq!(
                results.len(),
                2,
                "both chunks are retrievable — the dense arm runs unfiltered alongside the \
                 phrase arm, so phrase matching is a ranking signal, not an exclusion filter"
            );
            assert_eq!(
                results[0].payload.get("file_path").and_then(|v| v.as_str()),
                Some("/data/exact.md"),
                "the exact-phrase chunk accumulates reciprocal rank from both the dense and \
                 phrase arms, so it must rank first"
            );
            assert!(
                results[0].phrase_score.is_some(),
                "the top result's phrase_score must be Some — it matched the phrase arm"
            );
            assert_eq!(
                results[1].payload.get("file_path").and_then(|v| v.as_str()),
                Some("/data/reordered.md"),
                "the reordered chunk still surfaces via the unfiltered dense arm, just ranked \
                 second since it only accumulates rank from one arm instead of two"
            );
            assert!(
                results[1].phrase_score.is_none(),
                "the reordered chunk never matched the phrase arm, so its phrase_score must \
                 be None — this is the actual signal the phrase-matching text index provides"
            );

            // #133: everything above proves phrase matching is a *ranking* signal
            // inside the fused RRF query — and, per `build_fusion_arms`'s doc
            // comment, that arm can never actually EXCLUDE a document, because the
            // dense arm always runs alongside it unfiltered. That leaves the one
            // claim the `phrase_matching` payload index actually exists to back —
            // "Qdrant can filter results down to exactly the literal phrase" —
            // completely unproven: a silently-failed index build (`ensure_collection`
            // tolerates that by design and only logs — see its own doc comment), a
            // tokenizer mismatch between how chunks were indexed and how phrases are
            // queried, or a wrong field name in the phrase condition would all still
            // let every assertion above pass, because nothing above ever asks Qdrant
            // to exclude anything.
            //
            // Prove exclusion directly instead: apply the identical phrase condition
            // as a hard `Filter::must` via `search`'s `extra_conditions` (the plain
            // dense-only path, not the fused `hybrid_search` prefetch arm) and
            // confirm the reordered chunk is excluded outright rather than merely
            // out-ranked. This is what actually distinguishes "the server-side
            // enforcement works" from "the request was shaped correctly" — the class
            // of failure #133 was filed over.
            let filtered = retry_until(
                20,
                std::time::Duration::from_millis(250),
                || async {
                    store
                        .search(
                            &config.collection,
                            vec![1.0, 0.0, 0.0, 0.0],
                            HashMap::new(),
                            vec![Condition::matches_phrase(
                                "text",
                                "node:ares rocm".to_string(),
                            )],
                            10,
                        )
                        .await
                        .unwrap()
                },
                |results| !results.is_empty(),
            )
            .await;

            assert_eq!(
                filtered.len(),
                1,
                "a hard phrase filter must exclude the reordered chunk outright, not merely \
                 rank it lower — this is the server-side enforcement #133 asked to be proven"
            );
            assert_eq!(
                filtered[0]
                    .payload
                    .get("file_path")
                    .and_then(|v| v.as_str()),
                Some("/data/exact.md"),
                "the only survivor of a hard phrase filter must be the chunk that actually \
                 contains the literal phrase"
            );
        })
        .await;
    }

    /// Integration test: upsert points for multiple files, batch-delete by file paths,
    /// and verify the targeted points are removed while others remain.
    ///
    /// Stays live-only — the thing this proves is that Qdrant's own server-side
    /// filter evaluation (`Condition::matches("file_path", values)` inside a real
    /// `delete_points` call) actually matches and deletes the right points and
    /// leaves everything else untouched. That is Qdrant's filter engine doing the
    /// work, not code in this crate; a fake `VectorStore` would only prove the fake
    /// implements the same filter logic correctly, not that Qdrant does.
    ///
    /// Requires a running Qdrant instance at localhost:6334.
    /// Run with: cargo test delete_by_files_removes_matching -- --ignored
    #[tokio::test]
    #[ignore]
    async fn delete_by_files_removes_matching() {
        let config = ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: live_test_collection("delete_by_files_removes_matching"),
        };
        let store = QdrantStore::new(&config).unwrap();

        let _ = store.client.delete_collection(&config.collection).await;

        let vector_size = 4;
        store
            .ensure_collection(
                &config.collection,
                vector_size,
                &[IndexedField::keyword("file_path")],
                false,
            )
            .await
            .unwrap();

        with_collection_cleanup(&store, &config.collection, || async {
            // Insert points for 3 different files
            let make_point = |id: &str, file: &str, vec: Vec<f32>| {
                let mut payload = HashMap::new();
                payload.insert("file_path".into(), serde_json::json!(file));
                QdrantPoint {
                    id: id.into(),
                    vector: vec,
                    sparse: None,
                    payload,
                }
            };

            let points = vec![
                make_point(
                    "00000000-0000-0000-0000-000000000001",
                    "/data/a.md",
                    vec![1.0, 0.0, 0.0, 0.0],
                ),
                make_point(
                    "00000000-0000-0000-0000-000000000002",
                    "/data/b.md",
                    vec![0.0, 1.0, 0.0, 0.0],
                ),
                make_point(
                    "00000000-0000-0000-0000-000000000003",
                    "/data/c.md",
                    vec![0.0, 0.0, 1.0, 0.0],
                ),
            ];
            store
                .upsert_points(&config.collection, points)
                .await
                .unwrap();

            // Poll instead of a single fixed sleep -- see retry_until's doc
            // comment for why (#231). Wait for all 3 points to actually be
            // visible before deleting 2 of them, or the delete's own filter
            // match could race the upsert becoming visible.
            let count_before_delete = retry_until(
                20,
                std::time::Duration::from_millis(250),
                || async {
                    store
                        .collection_point_count(&config.collection)
                        .await
                        .unwrap()
                },
                |count| *count == 3,
            )
            .await;
            assert_eq!(
                count_before_delete, 3,
                "all 3 upserted points must be visible before delete_by_files runs"
            );

            // Delete points for files a.md and b.md in one call
            store
                .delete_by_files(&config.collection, &["/data/a.md", "/data/b.md"])
                .await
                .unwrap();

            let count_after_delete = retry_until(
                20,
                std::time::Duration::from_millis(250),
                || async {
                    store
                        .collection_point_count(&config.collection)
                        .await
                        .unwrap()
                },
                |count| *count == 1,
            )
            .await;
            assert_eq!(
                count_after_delete, 1,
                "exactly 2 of the 3 points should have been deleted"
            );

            // c.md point should still be searchable
            let results = store
                .search(
                    &config.collection,
                    vec![0.0, 0.0, 1.0, 0.0],
                    HashMap::new(),
                    Vec::new(),
                    10,
                )
                .await
                .unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(
                results[0].payload.get("file_path").and_then(|v| v.as_str()),
                Some("/data/c.md"),
            );

            // a.md and b.md should return no results
            let results_a = store
                .search(
                    &config.collection,
                    vec![1.0, 0.0, 0.0, 0.0],
                    {
                        let mut f = HashMap::new();
                        f.insert("file_path".into(), serde_json::json!("/data/a.md"));
                        f
                    },
                    Vec::new(),
                    10,
                )
                .await
                .unwrap();
            assert!(results_a.is_empty(), "a.md points should be deleted");
        })
        .await;
    }

    /// `upsert_points`, `delete_by_files`, and `delete_points_by_ids` each guard
    /// their empty-input case with an early `return Ok(())` before touching the
    /// client — the point of the old (`#[ignore]`d, live-Qdrant) version of this
    /// test was to show that an empty `delete_by_files` call left an existing point
    /// untouched. But that was only ever a roundabout way of checking the same
    /// early return: nothing about "does the server leave a point alone" is in
    /// question once the call never reaches the server at all.
    ///
    /// That early-return guarantee is directly, offline-checkable: point at a URL
    /// nothing is listening on (`QdrantStore::new` never dials out — see
    /// `index_paths_records_a_failed_run_in_the_global_status` in `ingest.rs` for
    /// the same fact established the other way, via a real RPC failing) and confirm
    /// each empty-input call still returns `Ok(())`. If any of the three guards were
    /// ever removed, the call would instead try to reach the unreachable address and
    /// this test would see a connection error instead.
    #[tokio::test]
    async fn empty_input_calls_are_no_ops_without_touching_qdrant() {
        let config = ResolvedQdrantConfig {
            url: "http://127.0.0.1:1".into(),
            collection: "unused".into(),
        };
        let store = QdrantStore::new(&config).unwrap();

        assert!(
            store
                .upsert_points(&config.collection, vec![])
                .await
                .is_ok(),
            "empty upsert must short-circuit before any RPC"
        );
        assert!(
            store.delete_by_files(&config.collection, &[]).await.is_ok(),
            "empty delete_by_files must short-circuit before any RPC"
        );
        assert!(
            store
                .delete_points_by_ids(&config.collection, vec![])
                .await
                .is_ok(),
            "empty delete_points_by_ids must short-circuit before any RPC"
        );
    }

    fn make_string_facet_hit(value: &str, count: u64) -> FacetHit {
        use qdrant_client::qdrant::FacetValue;
        FacetHit {
            value: Some(FacetValue {
                variant: Some(facet_value::Variant::StringValue(value.to_string())),
            }),
            count,
        }
    }

    #[test]
    fn extract_facet_strings_returns_string_values() {
        let hits = vec![
            make_string_facet_hit("networking", 5),
            make_string_facet_hit("docker", 3),
            make_string_facet_hit("storage", 1),
        ];
        let values = extract_facet_strings(hits);
        assert_eq!(values, vec!["networking", "docker", "storage"]);
    }

    #[test]
    fn extract_facet_strings_skips_non_string_variants() {
        use qdrant_client::qdrant::FacetValue;
        let hits = vec![
            make_string_facet_hit("valid", 2),
            FacetHit {
                value: Some(FacetValue {
                    variant: Some(facet_value::Variant::IntegerValue(42)),
                }),
                count: 1,
            },
            FacetHit {
                value: Some(FacetValue {
                    variant: Some(facet_value::Variant::BoolValue(true)),
                }),
                count: 1,
            },
            make_string_facet_hit("also-valid", 1),
        ];
        let values = extract_facet_strings(hits);
        assert_eq!(values, vec!["valid", "also-valid"]);
    }

    #[test]
    fn extract_facet_strings_handles_empty_hits() {
        let values = extract_facet_strings(vec![]);
        assert!(values.is_empty());
    }

    #[test]
    fn extract_facet_strings_skips_none_value() {
        let hits = vec![
            make_string_facet_hit("present", 3),
            FacetHit {
                value: None,
                count: 1,
            },
        ];
        let values = extract_facet_strings(hits);
        assert_eq!(values, vec!["present"]);
    }

    #[test]
    fn extract_facet_strings_skips_none_variant() {
        use qdrant_client::qdrant::FacetValue;
        let hits = vec![
            make_string_facet_hit("present", 3),
            FacetHit {
                value: Some(FacetValue { variant: None }),
                count: 1,
            },
        ];
        let values = extract_facet_strings(hits);
        assert_eq!(values, vec!["present"]);
    }

    /// Integration test: upsert points with keyword fields, then fetch facet values.
    ///
    /// Stays live-only for the part that matters here: that Qdrant's facet
    /// aggregation actually groups by distinct field value and that
    /// `extract_facet_strings` (tested standalone above) is fed real `FacetHit`s
    /// shaped the way the server actually returns them. The *other* half of what
    /// this test used to check — that a failed facet query degrades to an empty
    /// list instead of an error — needs no live server at all and is now covered
    /// offline by `fetch_facet_values_degrades_to_empty_on_query_failure` below, so
    /// it is no longer duplicated here.
    ///
    /// Requires a running Qdrant instance at localhost:6334.
    /// Run with: cargo test facet_values_returns_distinct_strings -- --ignored
    #[tokio::test]
    #[ignore]
    async fn facet_values_returns_distinct_strings() {
        let config = ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: live_test_collection("facet_values_returns_distinct_strings"),
        };
        let store = QdrantStore::new(&config).unwrap();

        let _ = store.client.delete_collection(&config.collection).await;

        store
            .ensure_collection(
                &config.collection,
                4,
                &[IndexedField::keyword("domain")],
                false,
            )
            .await
            .unwrap();

        with_collection_cleanup(&store, &config.collection, || async {
            let make_point = |id: &str, domain: &str, vec: Vec<f32>| {
                let mut payload = HashMap::new();
                payload.insert("domain".into(), serde_json::json!(domain));
                QdrantPoint {
                    id: id.into(),
                    vector: vec,
                    sparse: None,
                    payload,
                }
            };

            let points = vec![
                make_point(
                    "00000000-0000-0000-0000-000000000001",
                    "networking",
                    vec![1.0, 0.0, 0.0, 0.0],
                ),
                make_point(
                    "00000000-0000-0000-0000-000000000002",
                    "docker",
                    vec![0.0, 1.0, 0.0, 0.0],
                ),
                make_point(
                    "00000000-0000-0000-0000-000000000003",
                    "networking",
                    vec![0.0, 0.0, 1.0, 0.0],
                ),
            ];
            store
                .upsert_points(&config.collection, points)
                .await
                .unwrap();

            // Poll instead of a single fixed sleep -- see retry_until's doc
            // comment for why (#231).
            let values = retry_until(
                20,
                std::time::Duration::from_millis(250),
                || async {
                    store
                        .fetch_facet_values(&config.collection, "domain", 10)
                        .await
                        .unwrap()
                },
                |values| values.len() >= 2,
            )
            .await;

            assert_eq!(values.len(), 2, "should have 2 distinct domains");
            assert!(values.contains(&"networking".to_string()));
            assert!(values.contains(&"docker".to_string()));
        })
        .await;
    }

    /// `fetch_facet_values` treats ANY facet-query failure as "no values" rather
    /// than propagating the error (see the `Err(e) => { ...; return Ok(vec![]) }`
    /// arm above) — not just an unindexed/nonexistent field on an otherwise healthy
    /// collection, which is all a live server can exercise. Pointing at an
    /// unreachable Qdrant instead proves the degradation covers the failure mode
    /// that matters most in production: a facet lookup (e.g. for `/status` or a
    /// `list_documents` filter hint) landing during a brief Qdrant outage must not
    /// itself become a hard error.
    #[tokio::test]
    async fn fetch_facet_values_degrades_to_empty_on_query_failure() {
        let config = ResolvedQdrantConfig {
            url: "http://127.0.0.1:1".into(),
            collection: "unused".into(),
        };
        let store = QdrantStore::new(&config).unwrap();

        let result = store
            .fetch_facet_values(&config.collection, "domain", 10)
            .await;

        assert_eq!(
            result.unwrap(),
            Vec::<String>::new(),
            "a facet query that can't reach Qdrant must degrade to empty, not error"
        );
    }

    #[test]
    fn filter_string_creates_match() {
        let mut filters = HashMap::new();
        filters.insert("domain".to_string(), serde_json::json!("engineering"));
        let conditions = build_conditions(&filters).unwrap();
        assert_eq!(conditions.len(), 1);
    }

    #[test]
    fn filter_integer_creates_match() {
        let mut filters = HashMap::new();
        filters.insert("priority".to_string(), serde_json::json!(42i64));
        let conditions = build_conditions(&filters).unwrap();
        assert_eq!(conditions.len(), 1);
    }

    #[test]
    fn filter_float_returns_error() {
        let mut filters = HashMap::new();
        filters.insert("score".to_string(), serde_json::json!(3.15f64));
        let err = build_conditions(&filters).unwrap_err();
        assert!(
            err.to_string()
                .contains("Float filter values are not supported")
        );
    }

    #[test]
    fn filter_bool_creates_match() {
        let mut filters = HashMap::new();
        filters.insert("active".to_string(), serde_json::json!(true));
        let conditions = build_conditions(&filters).unwrap();
        assert_eq!(conditions.len(), 1);
    }

    #[test]
    fn filter_array_creates_any_match() {
        let mut filters = HashMap::new();
        filters.insert("tags".to_string(), serde_json::json!(["rust", "rag"]));
        let conditions = build_conditions(&filters).unwrap();
        assert_eq!(conditions.len(), 1);
    }

    #[test]
    fn filter_null_returns_error() {
        let mut filters = HashMap::new();
        filters.insert("bad".to_string(), serde_json::Value::Null);
        assert!(build_conditions(&filters).is_err());
    }

    #[test]
    fn filter_nested_object_returns_error() {
        let mut filters = HashMap::new();
        filters.insert("nested".to_string(), serde_json::json!({"a": 1}));
        assert!(build_conditions(&filters).is_err());
    }

    #[test]
    fn empty_filters_returns_empty() {
        let filters = HashMap::new();
        let conditions = build_conditions(&filters).unwrap();
        assert!(conditions.is_empty());
    }

    #[test]
    fn filter_array_with_non_string_element_returns_error() {
        let mut filters = HashMap::new();
        filters.insert(
            "tags".to_string(),
            serde_json::json!(["valid", 42, "also-valid"]),
        );
        let err = build_conditions(&filters).unwrap_err();
        assert!(
            err.to_string().contains("non-string element"),
            "expected non-string error, got: {}",
            err
        );
    }

    /// `build_recommend_query` (the pure request-builder behind
    /// `recommend_by_point_id`) must target the named `dense` vector with a
    /// nearest-to-existing-point query keyed by the given point id, and carry the
    /// requested limit and payload flag through unmodified.
    #[test]
    fn recommend_query_targets_dense_vector_by_point_id() {
        use qdrant_client::qdrant::{point_id::PointIdOptions, query::Variant, vector_input};

        let request =
            build_recommend_query("kb", "00000000-0000-0000-0000-000000000042", 7, None).build();

        assert_eq!(request.collection_name, "kb");
        assert_eq!(request.using.as_deref(), Some("dense"));
        assert_eq!(request.limit, Some(7));
        assert!(request.filter.is_none());
        assert_eq!(
            request.with_payload,
            Some(qdrant_client::qdrant::WithPayloadSelector {
                selector_options: Some(
                    qdrant_client::qdrant::with_payload_selector::SelectorOptions::Enable(true)
                ),
            })
        );

        let query = request.query.expect("query must be set");
        let Some(Variant::Nearest(vector_input)) = query.variant else {
            panic!("expected a Nearest query variant, got {:?}", query.variant);
        };
        let Some(vector_input::Variant::Id(point_id)) = vector_input.variant else {
            panic!("expected the Nearest query to target a point id");
        };
        assert_eq!(
            point_id.point_id_options,
            Some(PointIdOptions::Uuid(
                "00000000-0000-0000-0000-000000000042".to_string()
            ))
        );
    }

    /// When a filter is supplied (the caller's typical use: excluding the source
    /// document's own chunks by `file_path`), it must be carried into the request
    /// unmodified rather than dropped or merged into some other condition.
    #[test]
    fn recommend_query_carries_supplied_filter() {
        let filter =
            Filter::must_not([Condition::matches("file_path", "docs/self.md".to_string())]);
        let request = build_recommend_query("kb", "some-point-id", 5, Some(filter.clone())).build();

        assert_eq!(request.filter, Some(filter));
    }

    /// A missing filter must leave the request filter-free — a `None` here means
    /// Qdrant returns neighbors from the whole collection, so this must not
    /// silently default to some other implicit condition.
    #[test]
    fn recommend_query_without_filter_is_unfiltered() {
        let request = build_recommend_query("kb", "some-point-id", 5, None).build();
        assert!(request.filter.is_none());
    }

    /// Integration test: upsert several documents' first-chunk points, then confirm
    /// `recommend_by_point_id` returns the nearest neighbor by the named `dense`
    /// vector and that an excluding filter removes a specific document.
    ///
    /// #212: this test originally asserted that an UNFILTERED query for point A's
    /// nearest neighbor returned A itself (i.e. that querying by a point's own ID
    /// includes that point in its own results). Against a real server that assertion
    /// fails — B comes back instead. This is not a bug: Qdrant's `VectorInput::new_id`
    /// query mechanism resolves the point's stored vector and searches with it, but
    /// deliberately EXCLUDES the query point itself from the results (the same way
    /// "recommend similar items" never recommends the item back to itself) — that
    /// exclusion happens server-side regardless of whether the caller also supplies a
    /// filter. So `recommend_by_point_id(a, limit=1, None)` was never going to return
    /// A; it returns A's actual nearest DIFFERENT neighbor, B — exactly what the
    /// second half of this test (with an explicit `must_not` filter) already expected.
    /// The assertion below was corrected to match that real, documented behavior.
    ///
    /// Stays live-only — what this proves is that Qdrant's server-side point-id
    /// resolution (`VectorInput::new_id`, i.e. "look up this point's own vector and
    /// use it as the query, excluding the point itself") actually works end to end,
    /// plus that a `must_not` filter is honored by the Query API for this query
    /// shape. Neither is something a fake can stand in for: it's Qdrant's own
    /// vector-lookup-by-id behavior under test, not code in this crate.
    ///
    /// Requires a running Qdrant instance at localhost:6334.
    /// Run with: cargo test recommend_by_point_id_finds_nearest_neighbor -- --ignored
    #[tokio::test]
    #[ignore]
    async fn recommend_by_point_id_finds_nearest_neighbor() {
        let config = ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: live_test_collection("recommend_by_point_id_finds_nearest_neighbor"),
        };
        let store = QdrantStore::new(&config).unwrap();

        let _ = store.client.delete_collection(&config.collection).await;

        let vector_size = 4;
        store
            .ensure_collection(
                &config.collection,
                vector_size,
                &[IndexedField::keyword("file_path")],
                false,
            )
            .await
            .unwrap();

        with_collection_cleanup(&store, &config.collection, || async {
            let make_point = |id: &str, file: &str, vec: Vec<f32>| {
                let mut payload = HashMap::new();
                payload.insert("file_path".into(), serde_json::json!(file));
                QdrantPoint {
                    id: id.into(),
                    vector: vec,
                    sparse: None,
                    payload,
                }
            };

            // "a" and "b" are near-identical vectors (should recommend each other);
            // "c" is orthogonal and should not show up as a's nearest neighbor.
            let point_a = "00000000-0000-0000-0000-0000000000a1";
            let point_b = "00000000-0000-0000-0000-0000000000b1";
            let point_c = "00000000-0000-0000-0000-0000000000c1";
            let points = vec![
                make_point(point_a, "/data/a.md", vec![1.0, 0.0, 0.0, 0.0]),
                make_point(point_b, "/data/b.md", vec![0.99, 0.01, 0.0, 0.0]),
                make_point(point_c, "/data/c.md", vec![0.0, 0.0, 1.0, 0.0]),
            ];
            store
                .upsert_points(&config.collection, points)
                .await
                .unwrap();

            // Poll instead of a single fixed sleep -- see retry_until's doc
            // comment for why (#231). `recommend_by_point_id` looks up point
            // a's own stored vector server-side; before it's visible this
            // returns a hard "no point with id" error rather than an empty
            // result, so the retry predicate checks `is_ok()`.
            //
            // Unfiltered: Qdrant's ID-based query excludes the query point itself from
            // its own results (server-side, unconditionally — see this test's doc
            // comment), so a's nearest neighbor here is b, not a itself.
            let results = retry_until(
                20,
                std::time::Duration::from_millis(250),
                || async {
                    store
                        .recommend_by_point_id(&config.collection, point_a, 1, None)
                        .await
                },
                |result| result.is_ok(),
            )
            .await
            .unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(
                results[0].payload.get("file_path").and_then(|v| v.as_str()),
                Some("/data/b.md"),
                "querying by point a's own ID excludes a itself, so its nearest neighbor is b"
            );

            // Excluding a's own file, the nearest neighbor is b, not c.
            let exclude_self =
                Filter::must_not([Condition::matches("file_path", "/data/a.md".to_string())]);
            let results = store
                .recommend_by_point_id(&config.collection, point_a, 1, Some(exclude_self))
                .await
                .unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(
                results[0].payload.get("file_path").and_then(|v| v.as_str()),
                Some("/data/b.md"),
                "nearest neighbor excluding self should be b, not the orthogonal c"
            );
        })
        .await;
    }

    /// #130/#182: pin the Qdrant server behaviors `path_prefix` is built on.
    ///
    /// `retrieval::path_filter_condition` compiles a resolved path set to
    /// `should[ matches(path_ancestors, [paths...]), is_empty(path_ancestors) ]`,
    /// ANDed into each RRF prefetch arm's `must`. That is only correct if several
    /// non-obvious things are true of the server, none of which this crate
    /// controls and none of which the qdrant-client crate documents:
    ///
    ///   1. `is_empty` matches a point whose key is **missing entirely**, not
    ///      only one whose value is `[]`. This is the whole
    ///      backward-compatibility escape: documents indexed before
    ///      `path_ancestors` existed carry no such key, and if `is_empty`
    ///      stopped matching them, `path_prefix` would return **zero results
    ///      for the entire un-reindexed corpus** — silently, with a green
    ///      mock-only suite.
    ///   2. A keyword `match` against an **array-valued** field tests element
    ///      membership, not equality against the whole array. Every ancestor
    ///      entry past the first depends on this.
    ///   3. A `should` nested inside a prefetch arm's `must` behaves as an OR
    ///      *within* that arm, rather than being flattened or dropped.
    ///
    /// All three were verified by hand against `qdrant/qdrant:v1.17.0` when this
    /// was written; this test is what makes a future server changing any of them
    /// fail loudly here instead of quietly at the top of someone's search
    /// results. `retrieval.rs`'s own coverage cannot substitute — its
    /// `MockRetrievalStore` never evaluates the `Filter` it is handed.
    ///
    /// Requires a running Qdrant instance at localhost:6334.
    /// Run with: cargo test path_prefix_filter_semantics_hold_on_a_live_server -- --ignored
    #[tokio::test]
    #[ignore]
    async fn path_prefix_filter_semantics_hold_on_a_live_server() {
        let config = ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: live_test_collection("path_prefix_filter_semantics_hold_on_a_live_server"),
        };
        let store = QdrantStore::new(&config).unwrap();

        let _ = store.client.delete_collection(&config.collection).await;

        let vector_size = 4;
        store
            .ensure_collection(
                &config.collection,
                vector_size,
                &[IndexedField::keyword("file_path")],
                false,
            )
            .await
            .unwrap();

        with_collection_cleanup(&store, &config.collection, || async {
            // `ancestors: None` is the legacy shape under test — the key is not
            // written at all, exactly as a pre-#130 point carries it.
            let make_point = |id: &str, file: &str, ancestors: Option<Vec<&str>>| {
                let mut payload = HashMap::new();
                payload.insert("file_path".into(), serde_json::json!(file));
                if let Some(ancestors) = ancestors {
                    payload.insert(PATH_ANCESTORS_KEY.into(), serde_json::json!(ancestors));
                }
                QdrantPoint {
                    id: id.into(),
                    // Identical vectors: relevance is irrelevant here, the
                    // filter is the entire subject.
                    vector: vec![1.0, 0.0, 0.0, 0.0],
                    sparse: None,
                    payload,
                }
            };

            let points = vec![
                make_point(
                    "00000000-0000-0000-0000-0000000001a1",
                    "/data/sysadmin/nodes/ares.md",
                    Some(vec!["sysadmin", "sysadmin/nodes", "sysadmin/nodes/ares.md"]),
                ),
                make_point(
                    "00000000-0000-0000-0000-0000000001b1",
                    "/data/food/chili.md",
                    Some(vec!["food", "food/chili.md"]),
                ),
                make_point(
                    "00000000-0000-0000-0000-0000000001c1",
                    "/data/legacy.md",
                    None,
                ),
            ];
            store
                .upsert_points(&config.collection, points)
                .await
                .unwrap();

            let search = async |conditions: Vec<Condition>| -> Vec<String> {
                let mut files: Vec<String> = store
                    .hybrid_search(
                        &config.collection,
                        vec![1.0, 0.0, 0.0, 0.0],
                        None,
                        &[],
                        HashMap::new(),
                        conditions,
                        10,
                        50,
                        false,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .filter_map(|r| {
                        r.payload
                            .get("file_path")
                            .and_then(|v| v.as_str())
                            .map(str::to_string)
                    })
                    .collect();
                // Sort so assertions pin membership, not RRF tie-break order
                // between three identical vectors.
                files.sort();
                files
            };

            // Establish write visibility once, unfiltered, before asserting on
            // any filter -- see retry_until's doc comment (#231).
            let all = retry_until(
                20,
                std::time::Duration::from_millis(250),
                || search(Vec::new()),
                |files| files.len() >= 3,
            )
            .await;
            assert_eq!(
                all,
                vec![
                    "/data/food/chili.md",
                    "/data/legacy.md",
                    "/data/sysadmin/nodes/ares.md"
                ],
                "all three fixture points must be visible before filters are judged"
            );

            // (1) is_empty matches the point with NO `path_ancestors` key at all
            //     -- and only that one.
            assert_eq!(
                search(vec![Condition::is_empty(PATH_ANCESTORS_KEY)]).await,
                vec!["/data/legacy.md"],
                "is_empty must match a MISSING key, not just an empty array — the \
                 entire legacy-document escape hangs on this"
            );

            // (2) a keyword match against an array-valued field is element
            //     membership: "sysadmin" is one of three entries, and matching
            //     it must not also drag in the point that has no such entry.
            assert_eq!(
                search(vec![Condition::matches(
                    PATH_ANCESTORS_KEY,
                    "sysadmin".to_string()
                )])
                .await,
                vec!["/data/sysadmin/nodes/ares.md"],
                "a keyword match on an array field tests element membership"
            );
            // The deep entry matches on the same terms — this is what lets a
            // caller scope to one specific document by its full path.
            assert_eq!(
                search(vec![Condition::matches(
                    PATH_ANCESTORS_KEY,
                    "sysadmin/nodes/ares.md".to_string()
                )])
                .await,
                vec!["/data/sysadmin/nodes/ares.md"],
            );

            // (3) keyword matching is not "starts with" — it is exact, per entry.
            //     #182 is what makes that a non-issue rather than a limitation:
            //     a substring needle is resolved to whole paths against SQLite
            //     before it ever reaches Qdrant, so what arrives here is always
            //     an exact set. This assertion pins *why* that resolution step
            //     has to exist.
            assert!(
                search(vec![Condition::matches(
                    PATH_ANCESTORS_KEY,
                    "sys".to_string()
                )])
                .await
                .is_empty(),
                "a partial path component must NOT match — which is precisely why a \
                 substring needle cannot be pushed down as-is and is resolved to \
                 concrete paths first"
            );
            assert!(
                search(vec![Condition::matches(
                    PATH_ANCESTORS_KEY,
                    "sysadmin/nodes/ares".to_string()
                )])
                .await
                .is_empty(),
                "a partial final segment must NOT match either"
            );

            // (4) a keyword match against a LIST of values is match-any, and each
            //     value still tests element membership in the array field. This is
            //     the #182 push-down: one condition carrying every path the needle
            //     resolved to.
            assert_eq!(
                search(vec![Condition::matches(
                    PATH_ANCESTORS_KEY,
                    vec![
                        "sysadmin/nodes/ares.md".to_string(),
                        "food/chili.md".to_string(),
                    ]
                )])
                .await,
                vec!["/data/food/chili.md", "/data/sysadmin/nodes/ares.md"],
                "a multi-value keyword match must be an OR across the resolved paths"
            );

            // (5) the actual shipped shape: that match-any `should`-nested inside
            //     the prefetch arm's `must`, ORed with the legacy escape — the
            //     resolved documents AND the legacy one, and nothing else.
            assert_eq!(
                search(vec![Condition::from(Filter::should([
                    Condition::matches(
                        PATH_ANCESTORS_KEY,
                        vec!["sysadmin/nodes/ares.md".to_string()]
                    ),
                    Condition::is_empty(PATH_ANCESTORS_KEY),
                ]))])
                .await,
                vec!["/data/legacy.md", "/data/sysadmin/nodes/ares.md"],
                "should[match_any, is_empty] nested in a prefetch arm's must must \
                 behave as an OR: the resolved paths plus the legacy escape, \
                 excluding the non-matching reindexed document"
            );
        })
        .await;
    }

    // ------------------------------------------------------------------
    // lower_field_filters
    // ------------------------------------------------------------------

    #[test]
    fn lower_any_of_keyword_produces_match_any() {
        let filters = vec![(
            "type".to_string(),
            FieldFilter::AnyOf(vec!["guide".into(), "recipe".into()]),
        )];
        let indexed = HashMap::from([("type".to_string(), IndexKind::Keyword)]);
        let conditions = lower_field_filters(&filters, &indexed).unwrap();
        assert_eq!(conditions.len(), 1);
    }

    #[test]
    fn lower_defaults_to_keyword_when_field_is_not_in_the_indexed_map() {
        // Mirrors the web UI's domain/type/tags filters, which never consult a
        // schema-derived indexed set at all.
        let filters = vec![(
            "domain".to_string(),
            FieldFilter::AnyOf(vec!["food".into()]),
        )];
        let conditions = lower_field_filters(&filters, &HashMap::new()).unwrap();
        assert_eq!(conditions.len(), 1);
    }

    #[test]
    fn lower_all_of_produces_one_condition_per_value() {
        let filters = vec![(
            "tags".to_string(),
            FieldFilter::AllOf(vec!["recipe".into(), "dinner".into()]),
        )];
        let indexed = HashMap::from([("tags".to_string(), IndexKind::Keyword)]);
        let conditions = lower_field_filters(&filters, &indexed).unwrap();
        assert_eq!(
            conditions.len(),
            2,
            "all_of must AND one condition per value, not fold them into one"
        );
    }

    #[test]
    fn lower_range_on_integer_field_builds_a_range_condition() {
        let filters = vec![(
            "prep_minutes".to_string(),
            FieldFilter::Range {
                gte: None,
                lte: Some(30.0),
                gt: None,
                lt: None,
            },
        )];
        let indexed = HashMap::from([("prep_minutes".to_string(), IndexKind::Integer)]);
        let conditions = lower_field_filters(&filters, &indexed).unwrap();
        assert_eq!(conditions.len(), 1);
    }

    #[test]
    fn lower_range_on_keyword_field_is_rejected() {
        let filters = vec![(
            "type".to_string(),
            FieldFilter::Range {
                gte: Some(1.0),
                lte: None,
                gt: None,
                lt: None,
            },
        )];
        let indexed = HashMap::from([("type".to_string(), IndexKind::Keyword)]);
        let err = lower_field_filters(&filters, &indexed).unwrap_err();
        assert!(err.contains("numeric range"), "got: {err}");
    }

    #[test]
    fn lower_equality_on_float_field_is_rejected() {
        let filters = vec![("rating".to_string(), FieldFilter::AnyOf(vec!["4.5".into()]))];
        let indexed = HashMap::from([("rating".to_string(), IndexKind::Float)]);
        let err = lower_field_filters(&filters, &indexed).unwrap_err();
        assert!(err.contains("float"), "got: {err}");
    }

    #[test]
    fn lower_integer_any_of_parses_values_as_integers() {
        let filters = vec![(
            "prep_minutes".to_string(),
            FieldFilter::AnyOf(vec!["20".into(), "30".into()]),
        )];
        let indexed = HashMap::from([("prep_minutes".to_string(), IndexKind::Integer)]);
        let conditions = lower_field_filters(&filters, &indexed).unwrap();
        assert_eq!(conditions.len(), 1);
    }

    #[test]
    fn lower_integer_any_of_rejects_a_non_numeric_value() {
        let filters = vec![(
            "prep_minutes".to_string(),
            FieldFilter::AnyOf(vec!["not-a-number".into()]),
        )];
        let indexed = HashMap::from([("prep_minutes".to_string(), IndexKind::Integer)]);
        let err = lower_field_filters(&filters, &indexed).unwrap_err();
        assert!(err.contains("not-a-number"), "got: {err}");
    }

    #[test]
    fn lower_bool_any_of_single_value() {
        let filters = vec![(
            "active".to_string(),
            FieldFilter::AnyOf(vec!["true".into()]),
        )];
        let indexed = HashMap::from([("active".to_string(), IndexKind::Bool)]);
        let conditions = lower_field_filters(&filters, &indexed).unwrap();
        assert_eq!(conditions.len(), 1);
    }

    #[test]
    fn lower_empty_any_of_is_unsatisfiable_not_dropped() {
        let filters = vec![("tags".to_string(), FieldFilter::AnyOf(vec![]))];
        let indexed = HashMap::from([("tags".to_string(), IndexKind::Keyword)]);
        let conditions = lower_field_filters(&filters, &indexed).unwrap();
        // A filter naming a field always contributes exactly one condition — an
        // empty value set must not silently disappear and widen the query.
        assert_eq!(conditions.len(), 1);
    }

    /// Live counterpart to `state::tests`' offline query-mode/enumeration-mode
    /// filter-equivalence suite: seeds real points and confirms a live Qdrant
    /// server honors `lower_field_filters`'s output exactly the way this
    /// crate's own condition evaluator (used offline, since no live Qdrant is
    /// normally available in this environment) predicts for the same `any_of`
    /// filter against the same fixture shape.
    ///
    /// Requires a running Qdrant instance at localhost:6334.
    /// Run with: cargo test qdrant_filters_agree_with_the_offline_prediction -- --ignored
    #[tokio::test]
    #[ignore]
    async fn qdrant_filters_agree_with_the_offline_prediction() {
        let config = ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: live_test_collection("qdrant_filters_agree_with_the_offline_prediction"),
        };
        let store = QdrantStore::new(&config).unwrap();
        let _ = store.client.delete_collection(&config.collection).await;

        let indexed_fields = [IndexedField::keyword("tags")];
        store
            .ensure_collection(&config.collection, 4, &indexed_fields, false)
            .await
            .unwrap();

        with_collection_cleanup(&store, &config.collection, || async {
            let make_point = |id: &str, file_path: &str, tags: &[&str]| {
                let mut payload: HashMap<String, serde_json::Value> = HashMap::new();
                payload.insert("file_path".into(), serde_json::json!(file_path));
                payload.insert("tags".into(), serde_json::json!(tags));
                QdrantPoint {
                    id: id.into(),
                    vector: vec![1.0, 0.0, 0.0, 0.0],
                    sparse: None,
                    payload,
                }
            };
            let points = vec![
                make_point(
                    "00000000-0000-0000-0000-000000000001",
                    "a.md",
                    &["recipe", "dinner"],
                ),
                make_point(
                    "00000000-0000-0000-0000-000000000002",
                    "b.md",
                    &["recipe", "breakfast"],
                ),
                make_point("00000000-0000-0000-0000-000000000003", "d.md", &["zfs"]),
            ];
            store
                .upsert_points(&config.collection, points)
                .await
                .unwrap();

            let filters = vec![(
                "tags".to_string(),
                FieldFilter::AnyOf(vec!["breakfast".into(), "zfs".into()]),
            )];
            let indexed = HashMap::from([("tags".to_string(), IndexKind::Keyword)]);

            // Poll instead of a single fixed sleep -- see retry_until's doc
            // comment for why (#231). `lower_field_filters` is cheap and pure,
            // so it's simplest to just rebuild the conditions fresh on each
            // attempt rather than cloning them out of the closure.
            let results = retry_until(
                20,
                std::time::Duration::from_millis(250),
                || async {
                    let conditions = lower_field_filters(&filters, &indexed).unwrap();
                    store
                        .search(
                            &config.collection,
                            vec![1.0, 0.0, 0.0, 0.0],
                            HashMap::new(),
                            conditions,
                            10,
                        )
                        .await
                        .unwrap()
                },
                |results| results.len() >= 2,
            )
            .await;
            let mut paths: Vec<&str> = results
                .iter()
                .filter_map(|r| r.payload.get("file_path").and_then(|v| v.as_str()))
                .collect();
            paths.sort();

            assert_eq!(
                paths,
                vec!["b.md", "d.md"],
                "a live Qdrant server must honor lower_field_filters's any_of the same way \
                 the offline equivalence tests in state.rs predict"
            );
        })
        .await;
    }

    // ------------------------------------------------------------------
    // all_indexed_fields
    // ------------------------------------------------------------------

    /// When a schema declares a field `indexed: true` with an explicit type
    /// AND the legacy `frontmatter.indexed_fields` config list names the same
    /// field, the schema's declared kind must win — not the config union's
    /// implicit keyword default — since that is what decides whether a
    /// `Range` filter on the field is accepted or rejected downstream.
    #[test]
    fn schema_kind_wins_over_legacy_config_default_on_name_collision() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  prep_minutes:\n    type: integer\n    indexed: true\n",
        )
        .unwrap();
        let mut config = crate::mcp::make_test_resolved_config(tmp.path());
        std::sync::Arc::make_mut(&mut config).frontmatter = crate::config::FrontmatterConfig {
            indexed_fields: vec!["prep_minutes".to_string()],
            ..Default::default()
        };
        let schemas = crate::schema::SchemaCache::build(tmp.path(), &config.frontmatter);

        let fields = all_indexed_fields(&config, &schemas);
        let prep = fields
            .iter()
            .find(|f| f.name == "prep_minutes")
            .expect("prep_minutes must be indexed");
        assert_eq!(
            prep.kind,
            IndexKind::Integer,
            "the schema's explicit `type: integer` must win over the legacy config's \
             implicit keyword default"
        );
    }

    // ------------------------------------------------------------------
    // build_query_groups_request
    // ------------------------------------------------------------------

    #[test]
    fn query_groups_request_targets_dense_vector_grouped_by_field() {
        let request = build_query_groups_request(
            "kb",
            vec![0.1, 0.2, 0.3],
            None,
            &[],
            Vec::new(),
            "file_path",
            1,
            10,
            50,
        )
        .build();

        assert_eq!(request.collection_name, "kb");
        assert_eq!(request.group_by, "file_path");
        assert_eq!(request.group_size, Some(1));
        assert_eq!(request.limit, Some(10));
        assert_eq!(request.using.as_deref(), Some("dense"));
        assert!(request.filter.is_none());
        assert!(
            request.prefetch.is_empty(),
            "dense-only (no sparse, no phrases) must stay a plain query, not a fusion \
             with prefetch arms"
        );
    }

    #[test]
    fn query_groups_request_carries_supplied_conditions() {
        let conditions = vec![Condition::matches("type", "guide".to_string())];
        let request = build_query_groups_request(
            "kb",
            vec![0.1, 0.2, 0.3],
            None,
            &[],
            conditions.clone(),
            "file_path",
            1,
            10,
            50,
        )
        .build();

        assert_eq!(request.filter, Some(Filter::must(conditions)));
    }

    #[test]
    fn query_groups_request_fuses_when_sparse_is_given() {
        let request = build_query_groups_request(
            "kb",
            vec![0.1, 0.2, 0.3],
            Some((vec![1, 2], vec![0.5, 0.5])),
            &[],
            Vec::new(),
            "file_path",
            1,
            10,
            50,
        )
        .build();

        assert_eq!(
            request.prefetch.len(),
            2,
            "sparse given, no phrases: dense + sparse arms"
        );
        assert!(
            request.using.is_none(),
            "a fused query must not also set a top-level `using` vector name"
        );
    }

    #[test]
    fn query_groups_request_fuses_when_phrases_are_given_even_without_sparse() {
        let phrases = vec!["node:ares".to_string()];
        let request = build_query_groups_request(
            "kb",
            vec![0.1, 0.2, 0.3],
            None,
            &phrases,
            Vec::new(),
            "file_path",
            1,
            10,
            50,
        )
        .build();

        assert_eq!(
            request.prefetch.len(),
            2,
            "phrases given, no sparse: dense + phrase arms — phrase search must work \
             independent of hybrid"
        );
    }

    #[test]
    fn query_groups_request_fuses_all_three_arms_when_both_given() {
        let phrases = vec!["node:ares".to_string()];
        let request = build_query_groups_request(
            "kb",
            vec![0.1, 0.2, 0.3],
            Some((vec![1], vec![1.0])),
            &phrases,
            Vec::new(),
            "file_path",
            1,
            10,
            50,
        )
        .build();

        assert_eq!(request.prefetch.len(), 3);
    }

    // ------------------------------------------------------------------
    // build_fusion_arms
    // ------------------------------------------------------------------

    #[test]
    fn fusion_arms_dense_only_when_neither_sparse_nor_phrases_given() {
        let arms = build_fusion_arms(vec![0.1, 0.2], None, &[], &[], 50);
        assert_eq!(arms.len(), 1);
    }

    #[test]
    fn fusion_arms_phrase_arm_ands_every_phrase_onto_the_dense_ranked_query() {
        let phrases = vec!["node:ares".to_string(), "rocm".to_string()];
        let conditions = vec![Condition::matches("type", "guide".to_string())];
        let arms = build_fusion_arms(vec![0.1, 0.2], None, &phrases, &conditions, 50);

        assert_eq!(arms.len(), 2, "dense + phrase, no sparse arm");
        let phrase_arm = &arms[1];
        let filter = phrase_arm
            .filter
            .clone()
            .expect("phrase arm must carry a filter");
        // The supplied condition plus one matches_phrase condition per phrase.
        assert_eq!(filter.must.len(), 1 + phrases.len());
    }

    #[test]
    fn fusion_arms_carries_conditions_into_every_arm() {
        let conditions = vec![Condition::matches("type", "guide".to_string())];
        let arms = build_fusion_arms(
            vec![0.1, 0.2],
            Some((vec![1], vec![1.0])),
            &["x".to_string()],
            &conditions,
            50,
        );
        assert_eq!(arms.len(), 3);
        for arm in &arms {
            assert!(
                arm.filter.is_some(),
                "every arm must carry the caller's conditions"
            );
        }
    }
}
