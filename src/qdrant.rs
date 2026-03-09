use std::collections::HashMap;

use anyhow::{Context, Result};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, DeletePointsBuilder,
    Distance, FacetCountsBuilder, FacetHit, FieldCondition, FieldType, Filter, Match, PointStruct,
    SearchPointsBuilder, UpsertPointsBuilder, Value as QdrantValue, VectorParamsBuilder,
    facet_value, value::Kind,
};
use tracing::{debug, info};

use crate::config::ResolvedQdrantConfig;

pub trait VectorStore: Send + Sync {
    async fn upsert_points(&self, collection: &str, points: Vec<QdrantPoint>) -> Result<()>;
    async fn delete_by_files(&self, collection: &str, file_paths: &[&str]) -> Result<()>;
}

pub struct QdrantStore {
    client: Qdrant,
}

#[derive(Debug, Clone)]
pub struct QdrantPoint {
    pub id: String,
    pub vector: Vec<f32>,
    pub payload: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub score: f32,
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
/// Returns an error for float values, null, object, or other unsupported types.
fn build_conditions(filters: &HashMap<String, serde_json::Value>) -> Result<Vec<Condition>> {
    let mut conditions = Vec::new();
    for (key, value) in filters {
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
    Ok(conditions)
}

impl QdrantStore {
    pub fn new(config: &ResolvedQdrantConfig) -> Result<Self> {
        let client = Qdrant::from_url(&config.url)
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
        indexed_fields: &[String],
    ) -> Result<()> {
        let exists = self
            .client
            .collection_exists(collection)
            .await
            .context("Failed to check if collection exists")?;

        if !exists {
            info!("Creating Qdrant collection '{}'", collection);
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(collection)
                        .vectors_config(VectorParamsBuilder::new(vector_size, Distance::Cosine)),
                )
                .await
                .context("Failed to create collection")?;
            info!("Created collection '{}'", collection);
        } else {
            debug!("Collection '{}' already exists", collection);
        }

        for field in indexed_fields {
            debug!(
                "Ensuring keyword index on field '{}' in collection '{}'",
                field, collection
            );
            self.client
                .create_field_index(CreateFieldIndexCollectionBuilder::new(
                    collection,
                    field,
                    FieldType::Keyword,
                ))
                .await
                .with_context(|| {
                    format!(
                        "Failed to create keyword index on field '{}' in collection '{}'",
                        field, collection
                    )
                })?;
        }

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
                PointStruct::new(p.id, p.vector, payload)
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
}

impl VectorStore for QdrantStore {
    async fn upsert_points(&self, collection: &str, points: Vec<QdrantPoint>) -> Result<()> {
        QdrantStore::upsert_points(self, collection, points).await
    }

    async fn delete_by_files(&self, collection: &str, file_paths: &[&str]) -> Result<()> {
        QdrantStore::delete_by_files(self, collection, file_paths).await
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

        let mut builder = SearchPointsBuilder::new(collection, vector, limit).with_payload(true);
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
            .ensure_collection(&config.collection, vector_size, &["file_path".to_string()])
            .await
            .unwrap();

        // Insert points for 3 different files
        let make_point = |id: &str, file: &str, vec: Vec<f32>| {
            let mut payload = HashMap::new();
            payload.insert("file_path".into(), serde_json::json!(file));
            QdrantPoint {
                id: id.into(),
                vector: vec,
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

    /// Integration test: delete_by_files with empty slice is a no-op.
    /// Requires a running Qdrant instance at localhost:6334.
    /// Run with: cargo test delete_by_files_empty_is_noop -- --ignored
    #[tokio::test]
    #[ignore]
    async fn delete_by_files_empty_is_noop() {
        let config = ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: "test-delete-by-files-empty".into(),
        };
        let store = QdrantStore::new(&config).unwrap();

        let _ = store.client.delete_collection(&config.collection).await;

        store
            .ensure_collection(&config.collection, 4, &[])
            .await
            .unwrap();

        let mut payload = HashMap::new();
        payload.insert("file_path".into(), serde_json::json!("/data/a.md"));
        let point = QdrantPoint {
            id: "00000000-0000-0000-0000-000000000001".into(),
            vector: vec![1.0, 0.0, 0.0, 0.0],
            payload,
        };
        store
            .upsert_points(&config.collection, vec![point])
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Empty delete should be fine
        store
            .delete_by_files(&config.collection, &[])
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Point should still exist
        let results = store
            .search(
                &config.collection,
                vec![1.0, 0.0, 0.0, 0.0],
                HashMap::new(),
                10,
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);

        store
            .client
            .delete_collection(&config.collection)
            .await
            .unwrap();
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
            .ensure_collection(&config.collection, 4, &["domain".to_string()])
            .await
            .unwrap();

        let make_point = |id: &str, domain: &str, vec: Vec<f32>| {
            let mut payload = HashMap::new();
            payload.insert("domain".into(), serde_json::json!(domain));
            QdrantPoint {
                id: id.into(),
                vector: vec,
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

        // Non-existent field returns empty (graceful degradation)
        let empty = store
            .fetch_facet_values(&config.collection, "nonexistent", 10)
            .await
            .unwrap();
        assert!(empty.is_empty());

        store
            .client
            .delete_collection(&config.collection)
            .await
            .unwrap();
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
        filters.insert("score".to_string(), serde_json::json!(3.14f64));
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
}
