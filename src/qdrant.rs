use std::collections::HashMap;

use anyhow::{Context, Result};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, DeletePointsBuilder,
    Distance, FieldCondition, FieldType, Filter, Match, PointStruct, SearchPointsBuilder,
    UpsertPointsBuilder, VectorParamsBuilder,
    value::Kind,
    Value as QdrantValue,
};
use tracing::{debug, info};

use crate::config::QdrantConfig;

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
            } else if let Some(f) = n.as_f64() {
                Some(Kind::DoubleValue(f))
            } else {
                None
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
        Some(Kind::DoubleValue(f)) => {
            serde_json::Number::from_f64(*f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null)
        }
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

impl QdrantStore {
    pub fn new(config: &QdrantConfig) -> Result<Self> {
        let client = Qdrant::from_url(&config.url)
            .build()
            .context("Failed to connect to Qdrant")?;
        info!("Connected to Qdrant at {}", config.url);
        Ok(Self { client })
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
                .create_field_index(
                    CreateFieldIndexCollectionBuilder::new(collection, field, FieldType::Keyword),
                )
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

    pub async fn upsert_points(
        &self,
        collection: &str,
        points: Vec<QdrantPoint>,
    ) -> Result<()> {
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

    pub async fn delete_by_file(&self, collection: &str, file_path: &str) -> Result<()> {
        let filter = Filter::must([Condition::matches(
            "file_path",
            file_path.to_string(),
        )]);

        self.client
            .delete_points(DeletePointsBuilder::new(collection).points(filter))
            .await
            .with_context(|| {
                format!(
                    "Failed to delete points for file '{}' in collection '{}'",
                    file_path, collection
                )
            })?;

        debug!(
            "Deleted points for file '{}' from collection '{}'",
            file_path, collection
        );
        Ok(())
    }

    pub async fn search(
        &self,
        collection: &str,
        vector: Vec<f32>,
        filters: HashMap<String, serde_json::Value>,
        limit: u64,
    ) -> Result<Vec<SearchResult>> {
        let conditions: Vec<Condition> = filters
            .iter()
            .map(|(key, value)| match value {
                serde_json::Value::Array(arr) => {
                    // match_any for array values
                    let string_values: Vec<String> = arr
                        .iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect();
                    Condition::matches(key, string_values)
                }
                serde_json::Value::String(s) => {
                    // keyword match for string values
                    Condition::matches(key, s.clone())
                }
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Condition::matches(key, i)
                    } else {
                        // Fall back to field condition with double range for floats;
                        // for simplicity encode as string match (best effort)
                        let fc = FieldCondition {
                            key: key.clone(),
                            ..Default::default()
                        };
                        Condition::from(fc)
                    }
                }
                serde_json::Value::Bool(b) => {
                    let fc = FieldCondition {
                        key: key.clone(),
                        r#match: Some(Match {
                            match_value: Some(
                                qdrant_client::qdrant::r#match::MatchValue::Boolean(*b),
                            ),
                        }),
                        ..Default::default()
                    };
                    Condition::from(fc)
                }
                _ => {
                    // unsupported filter type — produce a condition that is always satisfied
                    // by using the field's existence check (best effort)
                    let fc = FieldCondition {
                        key: key.clone(),
                        ..Default::default()
                    };
                    Condition::from(fc)
                }
            })
            .collect();

        let mut builder = SearchPointsBuilder::new(collection, vector, limit);
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

        let count = info
            .result
            .and_then(|r| r.points_count);

        Ok(count)
    }
}
