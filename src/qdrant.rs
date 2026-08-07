use std::collections::HashMap;

use anyhow::{Context, Result};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, DeletePointsBuilder,
    Distance, FacetCountsBuilder, FacetHit, FieldCondition, FieldType, Filter, Fusion, Match,
    Modifier, NamedVectors, PointStruct, PrefetchQueryBuilder, Query, QueryPointsBuilder, Range,
    SearchPointsBuilder, SparseVectorParamsBuilder, SparseVectorsConfigBuilder,
    UpsertPointsBuilder, Value as QdrantValue, Vector, VectorInput, VectorParamsBuilder,
    VectorsConfigBuilder, facet_value, value::Kind,
};
use tracing::{debug, error, info};

use crate::config::ResolvedQdrantConfig;

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
    async fn ensure_collection(
        &self,
        collection: &str,
        vector_size: u64,
        indexed_fields: &[IndexedField],
    ) -> Result<()>;
}

pub trait RetrievalStore: Send + Sync {
    async fn search(
        &self,
        collection: &str,
        vector: Vec<f32>,
        filters: std::collections::HashMap<String, serde_json::Value>,
        limit: u64,
    ) -> Result<Vec<SearchResult>>;
    /// Hybrid sparse+dense retrieval with Reciprocal Rank Fusion.
    ///
    /// When `explain=false` (default): fuses server-side via Qdrant's built-in RRF;
    /// `dense_score`/`sparse_score` on results are `None`.
    /// When `explain=true`: runs separate dense and sparse queries, fuses client-side
    /// (k=60), and populates `dense_score`/`sparse_score` on each result.
    #[allow(clippy::too_many_arguments)]
    async fn hybrid_search(
        &self,
        collection: &str,
        dense: Vec<f32>,
        sparse: (Vec<u32>, Vec<f32>),
        filters: std::collections::HashMap<String, serde_json::Value>,
        limit: u64,
        rrf_candidates: u64,
        explain: bool,
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
    ) -> Result<()> {
        QdrantStore::ensure_collection(self, collection, vector_size, indexed_fields).await
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
        limit: u64,
    ) -> Result<Vec<SearchResult>> {
        QdrantStore::search(self, collection, vector, filters, limit).await
    }

    async fn hybrid_search(
        &self,
        collection: &str,
        dense: Vec<f32>,
        sparse: (Vec<u32>, Vec<f32>),
        filters: std::collections::HashMap<String, serde_json::Value>,
        limit: u64,
        rrf_candidates: u64,
        explain: bool,
    ) -> Result<Vec<SearchResult>> {
        QdrantStore::hybrid_search(
            self,
            collection,
            dense,
            sparse,
            filters,
            limit,
            rrf_candidates,
            explain,
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
        limit: u64,
    ) -> Result<Vec<SearchResult>> {
        let conditions = build_conditions(&filters)?;

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
                payload: qdrant_payload_to_json(&scored.payload),
            })
            .collect();

        Ok(results)
    }

    /// Hybrid sparse+dense retrieval via the Qdrant Query API.
    ///
    /// Runs two prefetch arms — dense (named `dense`) and sparse (named `sparse`) —
    /// each fetching `rrf_candidates` candidates with `filters` applied, then fuses
    /// them server-side with Reciprocal Rank Fusion and returns the top `limit`.
    #[allow(clippy::too_many_arguments)]
    pub async fn hybrid_search(
        &self,
        collection: &str,
        dense: Vec<f32>,
        sparse: (Vec<u32>, Vec<f32>),
        filters: HashMap<String, serde_json::Value>,
        limit: u64,
        rrf_candidates: u64,
        explain: bool,
    ) -> Result<Vec<SearchResult>> {
        let conditions = build_conditions(&filters)?;
        let (sparse_indices, sparse_values) = sparse;

        if explain {
            return self
                .hybrid_search_explain(
                    collection,
                    dense,
                    (sparse_indices, sparse_values),
                    conditions,
                    limit,
                    rrf_candidates,
                )
                .await;
        }

        let mut dense_arm = PrefetchQueryBuilder::default()
            .using("dense")
            .query(Query::new_nearest(VectorInput::new_dense(dense)))
            .limit(rrf_candidates);
        let mut sparse_arm = PrefetchQueryBuilder::default()
            .using("sparse")
            .query(Query::new_nearest(VectorInput::new_sparse(
                sparse_indices,
                sparse_values,
            )))
            .limit(rrf_candidates);

        // Carry the same payload filters into both arms.
        if !conditions.is_empty() {
            dense_arm = dense_arm.filter(Filter::must(conditions.clone()));
            sparse_arm = sparse_arm.filter(Filter::must(conditions));
        }

        let builder = QueryPointsBuilder::new(collection)
            .add_prefetch(dense_arm)
            .add_prefetch(sparse_arm)
            .query(Query::new_fusion(Fusion::Rrf))
            .limit(limit)
            .with_payload(true);

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
                payload: qdrant_payload_to_json(&scored.payload),
            })
            .collect();

        Ok(results)
    }

    /// Hybrid search with client-side RRF (k=60). Used when `explain=true` to
    /// surface per-arm dense/sparse scores alongside the fused score.
    async fn hybrid_search_explain(
        &self,
        collection: &str,
        dense: Vec<f32>,
        sparse: (Vec<u32>, Vec<f32>),
        conditions: Vec<Condition>,
        limit: u64,
        rrf_candidates: u64,
    ) -> Result<Vec<SearchResult>> {
        let (sparse_indices, sparse_values) = sparse;

        // Dense arm
        let mut dense_builder = SearchPointsBuilder::new(collection, dense, rrf_candidates)
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

        // Sparse arm via QueryPoints (supports sparse named vectors)
        let mut sparse_builder = QueryPointsBuilder::new(collection)
            .query(Query::new_nearest(VectorInput::new_sparse(
                sparse_indices,
                sparse_values,
            )))
            .using("sparse")
            .limit(rrf_candidates)
            .with_payload(true);
        if !conditions.is_empty() {
            sparse_builder = sparse_builder.filter(Filter::must(conditions));
        }
        let sparse_resp = self
            .client
            .query(sparse_builder)
            .await
            .context("Failed to run sparse arm for explain")?;

        // Client-side RRF — key by file_path::chunk_index from payload
        struct RrfAccum {
            rrf_score: f32,
            dense_score: Option<f32>,
            sparse_score: Option<f32>,
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
                payload,
            });
            entry.rrf_score += rrf;
            entry.dense_score = Some(scored.score);
        }

        for (rank, scored) in sparse_resp.result.iter().enumerate() {
            let payload = qdrant_payload_to_json(&scored.payload);
            let key = payload_key(&payload);
            let rrf = 1.0 / (k + rank as f32 + 1.0);
            let entry = accum.entry(key).or_insert(RrfAccum {
                rrf_score: 0.0,
                dense_score: None,
                sparse_score: None,
                payload,
            });
            entry.rrf_score += rrf;
            entry.sparse_score = Some(scored.score);
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
                payload: a.payload,
            })
            .collect())
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
            collection: "test-search-payload".into(),
        };
        let store = QdrantStore::new(&config).unwrap();

        // Clean up from any prior run
        let _ = store.client.delete_collection(&config.collection).await;

        let vector_size = 4;
        store
            .ensure_collection(&config.collection, vector_size, &[])
            .await
            .unwrap();

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

        // Small delay for indexing
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let results = store
            .search(
                &config.collection,
                vec![1.0, 0.0, 0.0, 0.0],
                HashMap::new(),
                1,
            )
            .await
            .unwrap();

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

        // Clean up
        store
            .client
            .delete_collection(&config.collection)
            .await
            .unwrap();
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
            collection: "test-delete-by-files".into(),
        };
        let store = QdrantStore::new(&config).unwrap();

        let _ = store.client.delete_collection(&config.collection).await;

        let vector_size = 4;
        store
            .ensure_collection(
                &config.collection,
                vector_size,
                &[IndexedField::keyword("file_path")],
            )
            .await
            .unwrap();

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

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Delete points for files a.md and b.md in one call
        store
            .delete_by_files(&config.collection, &["/data/a.md", "/data/b.md"])
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // c.md point should still be searchable
        let results = store
            .search(
                &config.collection,
                vec![0.0, 0.0, 1.0, 0.0],
                HashMap::new(),
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
                10,
            )
            .await
            .unwrap();
        assert!(results_a.is_empty(), "a.md points should be deleted");

        store
            .client
            .delete_collection(&config.collection)
            .await
            .unwrap();
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
            collection: "test-facet-values".into(),
        };
        let store = QdrantStore::new(&config).unwrap();

        let _ = store.client.delete_collection(&config.collection).await;

        store
            .ensure_collection(&config.collection, 4, &[IndexedField::keyword("domain")])
            .await
            .unwrap();

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

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let values = store
            .fetch_facet_values(&config.collection, "domain", 10)
            .await
            .unwrap();

        assert_eq!(values.len(), 2, "should have 2 distinct domains");
        assert!(values.contains(&"networking".to_string()));
        assert!(values.contains(&"docker".to_string()));

        store
            .client
            .delete_collection(&config.collection)
            .await
            .unwrap();
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
    /// vector and that an excluding filter removes the source document itself.
    ///
    /// Stays live-only — what this proves is that Qdrant's server-side point-id
    /// resolution (`VectorInput::new_id`, i.e. "look up this point's own vector and
    /// use it as the query") actually works end to end, plus that a `must_not`
    /// filter is honored by the Query API for this query shape. Neither is
    /// something a fake can stand in for: it's Qdrant's own vector-lookup-by-id
    /// behavior under test, not code in this crate.
    ///
    /// Requires a running Qdrant instance at localhost:6334.
    /// Run with: cargo test recommend_by_point_id_finds_nearest_neighbor -- --ignored
    #[tokio::test]
    #[ignore]
    async fn recommend_by_point_id_finds_nearest_neighbor() {
        let config = ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: "test-recommend-by-point-id".into(),
        };
        let store = QdrantStore::new(&config).unwrap();

        let _ = store.client.delete_collection(&config.collection).await;

        let vector_size = 4;
        store
            .ensure_collection(
                &config.collection,
                vector_size,
                &[IndexedField::keyword("file_path")],
            )
            .await
            .unwrap();

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

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Unfiltered: a's own point is the nearest match to itself.
        let results = store
            .recommend_by_point_id(&config.collection, point_a, 1, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].payload.get("file_path").and_then(|v| v.as_str()),
            Some("/data/a.md"),
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

        store
            .client
            .delete_collection(&config.collection)
            .await
            .unwrap();
    }
}
