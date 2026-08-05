use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::RwLock;

use chrono::{DateTime, NaiveDate};

use anyhow::Context as _;
use globset::{Glob, GlobSet, GlobSetBuilder};
use rmcp::{
    ErrorData as McpError, ServerHandler, handler::server::wrapper::Parameters, model::*, schemars,
    tool, tool_handler, tool_router,
};
use tracing::{debug, error, warn};

use crate::{
    config::ResolvedConfig,
    document_fields,
    embed::EmbedClient,
    git,
    qdrant::QdrantStore,
    rerank::RerankClient,
    retrieval::{
        self, DocumentIndexDeps, GetDocumentError, RetrievalDeps, SearchFilters, SearchOptions,
    },
    schema::SchemaCache,
    state::{DocumentIndex, DocumentQuery, FieldFilter, OrderBy, StateDb},
    validate,
};

const MAX_SEARCH_LIMIT: u64 = 50;
const MAX_QUERY_LEN: usize = 4096;
const MAX_PATH_LEN: usize = 4096;
const MAX_FILTER_STR_LEN: usize = 256;
const MAX_TAG_COUNT: usize = 20;
const MAX_TAG_LEN: usize = 256;
const MAX_CONTENT_LEN: usize = 512 * 1024; // 512 KB

/// Maximum number of characters from the new document's content used to build
/// the dedup query text. Keeps the embedding request within typical token limits
/// (most embedding models cap at ~512–8192 tokens; 2000 chars ≈ 400–500 tokens).
const DEDUP_QUERY_CHAR_LIMIT: usize = 2000;

/// A near-duplicate found during dedup gate evaluation.
#[derive(Debug, Clone)]
pub struct DuplicateHit {
    pub file_path: String,
    pub score: f32,
}

/// Pure decision function: given the closest match from Qdrant (if any) and a
/// threshold, decide whether to refuse the write. Returns `Some(DuplicateHit)`
/// when the write should be blocked, `None` when it is safe to proceed.
///
/// This is factored out so it can be unit-tested without a live Qdrant/embedder.
pub fn dedup_verdict(top: Option<(String, f32)>, threshold: f32) -> Option<DuplicateHit> {
    match top {
        Some((path, score)) if score >= threshold => Some(DuplicateHit {
            file_path: path,
            score,
        }),
        _ => None,
    }
}

/// Build the dedup query text on the same textual basis the indexer uses, then
/// truncate it to `DEDUP_QUERY_CHAR_LIMIT`.
///
/// This must stay aligned with `chunk::chunk_markdown`: every indexed chunk is
/// prefixed with its document's own `description` when
/// `chunking.prepend_description` is set, so a dedup query built without that
/// prefix would be compared against a different textual basis than the
/// candidates it is scored against.
pub(crate) fn build_dedup_query(
    body: &str,
    description: Option<&str>,
    prepend_description: bool,
) -> String {
    let assembled = match (prepend_description, description) {
        (true, Some(desc)) => format!("{}\n\n{}", desc, body),
        _ => body.to_string(),
    };
    assembled.chars().take(DEDUP_QUERY_CHAR_LIMIT).collect()
}

/// Search options for the dedup gate.
///
/// Deliberately pinned to dense-only rather than inheriting `search.hybrid`, so
/// the returned score is a cosine similarity comparable to
/// `write.dedup_threshold`. Hybrid RRF scores top out around 0.03 — against a
/// cosine threshold like the 0.80 default the gate could never fire — and a
/// cross-encoder relevance score is not a similarity at all, so reranking is
/// also kept out of this path (see the `reranker: None` at the call site).
pub(crate) fn dedup_search_opts() -> SearchOptions {
    SearchOptions {
        limit: 1,
        min_score: None,
        hybrid: false,
        // Unused in the dense-only path, which performs no RRF fusion.
        rrf_candidates: 0,
        explain: false,
        modified_after: None,
        modified_before: None,
        rerank_candidate_limit: None,
    }
}

fn resolve_limit(requested: Option<u64>) -> u64 {
    requested.unwrap_or(10).min(MAX_SEARCH_LIMIT)
}

/// Parse an ISO 8601 date/datetime string to a Unix timestamp (seconds).
///
/// Accepts RFC 3339 datetimes (e.g. `2024-01-15T12:00:00Z`) and date-only
/// strings (e.g. `2024-01-15`, interpreted as midnight UTC).
pub(crate) fn parse_date_to_timestamp(s: &str) -> Result<i64, String> {
    // Try RFC 3339 / ISO 8601 datetime first
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Ok(dt.timestamp());
    }
    // Fall back to date-only YYYY-MM-DD (treated as start of day UTC)
    if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date
            .and_hms_opt(0, 0, 0)
            .expect("midnight is always valid")
            .and_utc();
        return Ok(dt.timestamp());
    }
    Err(format!(
        "invalid date '{}': expected RFC 3339 (e.g. 2024-01-15T00:00:00Z) \
         or date-only (e.g. 2024-01-15)",
        s
    ))
}

/// How many invalidated documents to name before summarizing the rest.
const MAX_REPORTED_CASUALTIES: usize = 20;
/// Cap on permitted values a single `update_schema` call may add to a field.
const MAX_SCHEMA_VALUES: usize = 500;
/// Cap on the serialized size of a `set_field` definition.
const MAX_SCHEMA_DEFINITION_LEN: usize = 8 * 1024;
/// Cap on permitted values echoed back per field by `get_schema`.
const MAX_REPORTED_VALUES: usize = 200;
/// Cap on fields echoed back per scope by `get_schema`.
const MAX_REPORTED_FIELDS: usize = 500;
/// Cap on a caller-supplied commit message.
const MAX_COMMIT_MESSAGE_LEN: usize = 1000;

/// Normalize a caller-supplied scope path into a safe KB-relative directory.
///
/// Rejects absolute paths and any `..` component — a schema written outside the KB
/// would govern nothing and could clobber unrelated files.
fn normalize_scope_path(raw: &str) -> Result<std::path::PathBuf, McpError> {
    use std::path::{Component, PathBuf};

    let trimmed = raw.trim().trim_start_matches("./").trim_matches('/');
    if trimmed.is_empty() {
        return Ok(PathBuf::new());
    }
    if trimmed.len() > MAX_PATH_LEN {
        return Err(McpError::invalid_params(
            format!(
                "path too long: {} chars (max {})",
                trimmed.len(),
                MAX_PATH_LEN
            ),
            None,
        ));
    }

    let candidate = PathBuf::from(trimmed);
    if candidate.is_absolute() {
        return Err(McpError::invalid_params(
            format!("path must be relative to the knowledge-base root, got '{raw}'"),
            None,
        ));
    }
    for component in candidate.components() {
        if !matches!(component, Component::Normal(_)) {
            return Err(McpError::invalid_params(
                format!("path must not contain '..' or absolute segments, got '{raw}'"),
                None,
            ));
        }
    }

    Ok(candidate)
}

/// Turn tool parameters into a typed schema edit.
fn build_schema_edit(params: &UpdateSchemaParams) -> Result<crate::schema::SchemaEdit, McpError> {
    use crate::schema::SchemaEdit;
    let invalid = |msg: String| McpError::invalid_params(msg, None);

    // A schema file is committed, pushed, and re-parsed on every cache build, so an
    // oversized one is a durable cost rather than a transient one. Bound the inputs
    // here, mirroring the content cap the document write tools enforce.
    if params.field.len() > MAX_FILTER_STR_LEN {
        return Err(invalid(format!(
            "field name too long: {} chars (max {})",
            params.field.len(),
            MAX_FILTER_STR_LEN
        )));
    }
    if let Some(values) = &params.values {
        if values.len() > MAX_SCHEMA_VALUES {
            return Err(invalid(format!(
                "too many values: {} (max {})",
                values.len(),
                MAX_SCHEMA_VALUES
            )));
        }
        if let Some(long) = values.iter().find(|v| v.len() > MAX_FILTER_STR_LEN) {
            return Err(invalid(format!(
                "value too long: {} chars (max {})",
                long.len(),
                MAX_FILTER_STR_LEN
            )));
        }
    }
    if let Some(definition) = &params.definition
        && definition.to_string().len() > MAX_SCHEMA_DEFINITION_LEN
    {
        return Err(invalid(format!(
            "field definition too large (max {} bytes)",
            MAX_SCHEMA_DEFINITION_LEN
        )));
    }

    let values = || -> Result<Vec<String>, McpError> {
        params
            .values
            .clone()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| {
                invalid(format!(
                    "'{}' requires a non-empty values list",
                    params.operation
                ))
            })
    };

    match params.operation.trim().to_ascii_lowercase().as_str() {
        "add_values" => Ok(SchemaEdit::AddValues {
            field: params.field.clone(),
            values: values()?,
        }),
        "remove_values" => Ok(SchemaEdit::RemoveValues {
            field: params.field.clone(),
            values: values()?,
        }),
        "set_field" => {
            let definition = params
                .definition
                .clone()
                .ok_or_else(|| invalid("'set_field' requires a definition".into()))?;
            let parsed = serde_json::from_value(definition)
                .map_err(|e| invalid(format!("invalid field definition: {e}")))?;
            Ok(SchemaEdit::SetField {
                field: params.field.clone(),
                definition: Box::new(parsed),
            })
        }
        "remove_field" => Ok(SchemaEdit::RemoveField {
            field: params.field.clone(),
        }),
        other => Err(invalid(format!(
            "unknown operation '{other}': expected add_values, remove_values, set_field, \
             or remove_field"
        ))),
    }
}

/// Resolve a possibly-partial scope reference to exactly one directory.
///
/// Mirrors `get_document`'s contract: an exact match wins, several matches are an
/// explicit ambiguity error rather than a guess, and none is a not-found error naming
/// what does exist.
fn resolve_scope_reference(
    schemas: &SchemaCache,
    requested: &std::path::Path,
) -> Result<std::path::PathBuf, McpError> {
    let matches = schemas.match_scope_dirs(requested);
    match matches.len() {
        // No scope declares its own schema here; the path still resolves through the
        // cascade to whatever ancestor governs it.
        0 => Ok(requested.to_path_buf()),
        1 => Ok(matches.into_iter().next().expect("length checked")),
        _ => Err(McpError::invalid_params(
            format!(
                "'{}' matches {} scopes: {}. Use a more specific path.",
                requested.display(),
                matches.len(),
                matches
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            None,
        )),
    }
}

/// Reject a caller-supplied commit message that git or the log would mangle.
fn validate_commit_message(message: Option<&str>) -> Result<(), McpError> {
    let Some(msg) = message else {
        return Ok(());
    };
    if msg.contains('\n') {
        return Err(McpError::invalid_params(
            "commit message must not contain newlines".to_string(),
            None,
        ));
    }
    if msg.len() > MAX_COMMIT_MESSAGE_LEN {
        return Err(McpError::invalid_params(
            format!(
                "commit message too long ({} chars); maximum is {}",
                msg.len(),
                MAX_COMMIT_MESSAGE_LEN
            ),
            None,
        ));
    }
    Ok(())
}

/// Strip control characters and cap length anywhere inside a value echoed back to an
/// agent. Defaults come from schema files in a synced repo, so an array or object
/// default is just as attacker-controlled as a string one.
fn sanitize_reflected_value(value: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match value {
        Value::String(s) => Value::String(crate::server::sanitize_facet_value(s)),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .take(MAX_REPORTED_VALUES)
                .map(sanitize_reflected_value)
                .collect(),
        ),
        Value::Object(map) => Value::Object(
            map.iter()
                .take(MAX_REPORTED_VALUES)
                .map(|(k, v)| {
                    (
                        crate::server::sanitize_facet_value(k),
                        sanitize_reflected_value(v),
                    )
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Render a casualty list for a human-readable message.
fn render_casualties(casualties: &[serde_json::Value]) -> String {
    let mut out = String::new();
    for entry in casualties.iter().take(MAX_REPORTED_CASUALTIES) {
        out.push_str(&format!(
            "  - {}: {}\n",
            entry["path"].as_str().unwrap_or("?"),
            entry["reason"].as_str().unwrap_or("?")
        ));
    }
    if casualties.len() > MAX_REPORTED_CASUALTIES {
        out.push_str(&format!(
            "  … and {} more\n",
            casualties.len() - MAX_REPORTED_CASUALTIES
        ));
    }
    out
}

/// Default page size for `list_documents` — well above `search`'s cap, since
/// enumeration is the point.
const DEFAULT_LIST_LIMIT: u64 = 100;
/// Hard cap on a single `list_documents` page.
const MAX_LIST_LIMIT: u64 = 1000;
/// Cap on how many filter fields one call may specify.
const MAX_LIST_FILTERS: usize = 20;
/// Cap on values within a single field's filter, so one call cannot generate an
/// unbounded number of bound SQL parameters.
const MAX_FILTER_VALUES: usize = 500;

/// Translate one JSON filter value into a typed [`FieldFilter`].
///
/// Accepts a scalar for equality, an array for any-of, or an object carrying
/// `any_of` / `all_of` / `gte` / `lte` / `gt` / `lt`.
fn parse_field_filter(field: &str, raw: &serde_json::Value) -> Result<FieldFilter, String> {
    use serde_json::Value;

    // Scalar values go through the same canonicalization as the write path, so a JSON
    // `false` matches a stored boolean and `45` matches a stored integer.
    let canonical = |value: &Value| -> Result<String, String> {
        document_fields::canonical_text(value).ok_or_else(|| {
            format!(
                "filter '{}': expected a string, number, or boolean, got {}",
                field, value
            )
        })
    };

    // One place enforces the value cap, so no filter form can slip past it. `all_of`
    // in particular compiles to one correlated subquery per value, so an uncapped list
    // is a query-complexity attack, not merely a large response.
    let values_of = |items: &[Value]| -> Result<Vec<String>, String> {
        if items.len() > MAX_FILTER_VALUES {
            return Err(format!(
                "filter '{}': too many values ({}, max {})",
                field,
                items.len(),
                MAX_FILTER_VALUES
            ));
        }
        items.iter().map(&canonical).collect()
    };

    match raw {
        Value::String(_) | Value::Number(_) | Value::Bool(_) => {
            Ok(FieldFilter::AnyOf(vec![canonical(raw)?]))
        }
        Value::Array(items) => Ok(FieldFilter::AnyOf(values_of(items)?)),
        Value::Object(map) => {
            let number = |key: &str| -> Result<Option<f64>, String> {
                match map.get(key) {
                    None | Some(Value::Null) => Ok(None),
                    Some(Value::Number(n)) => Ok(n.as_f64()),
                    Some(other) => Err(format!(
                        "filter '{}': '{}' must be a number, got {}",
                        field, key, other
                    )),
                }
            };

            let known = ["any_of", "all_of", "gte", "lte", "gt", "lt"];
            if let Some(unknown) = map.keys().find(|k| !known.contains(&k.as_str())) {
                return Err(format!(
                    "filter '{}': unknown operator '{}'; expected one of {}",
                    field,
                    unknown,
                    known.join(", ")
                ));
            }

            // Set matching and range matching are separate modes. Accepting a mix and
            // honoring only one silently returns a broader result set than the caller
            // asked for, which is exactly the class of silent-wrong-answer this tool
            // exists to eliminate — so reject it rather than pick a winner.
            let has_set = map.contains_key("any_of") || map.contains_key("all_of");
            let has_range = ["gte", "lte", "gt", "lt"]
                .iter()
                .any(|k| map.contains_key(*k));
            if has_set && has_range {
                return Err(format!(
                    "filter '{}': cannot combine set matching (any_of/all_of) with a \
                     numeric range (gte/lte/gt/lt); use one or the other",
                    field
                ));
            }
            if map.contains_key("any_of") && map.contains_key("all_of") {
                return Err(format!(
                    "filter '{}': specify either any_of or all_of, not both",
                    field
                ));
            }

            if let Some(values) = map.get("all_of") {
                let items = values
                    .as_array()
                    .ok_or_else(|| format!("filter '{}': 'all_of' must be an array", field))?;
                if items.is_empty() {
                    return Err(format!("filter '{}': 'all_of' must not be empty", field));
                }
                return Ok(FieldFilter::AllOf(values_of(items)?));
            }

            if let Some(values) = map.get("any_of") {
                let items = values
                    .as_array()
                    .ok_or_else(|| format!("filter '{}': 'any_of' must be an array", field))?;
                return Ok(FieldFilter::AnyOf(values_of(items)?));
            }

            let (gte, lte, gt, lt) = (number("gte")?, number("lte")?, number("gt")?, number("lt")?);
            if gte.is_none() && lte.is_none() && gt.is_none() && lt.is_none() {
                return Err(format!(
                    "filter '{}': object filters need at least one of {}",
                    field,
                    known.join(", ")
                ));
            }
            Ok(FieldFilter::Range { gte, lte, gt, lt })
        }
        Value::Null => Err(format!(
            "filter '{}': null is not a filter; omit the field instead",
            field
        )),
    }
}

/// Build a validated [`DocumentQuery`] from tool parameters.
fn build_document_query(params: &ListDocumentsParams) -> Result<DocumentQuery, McpError> {
    let invalid = |msg: String| McpError::invalid_params(msg, None);

    let mut filters = Vec::new();
    if let Some(raw_filters) = &params.filters {
        if raw_filters.len() > MAX_LIST_FILTERS {
            return Err(invalid(format!(
                "too many filters: {} (max {})",
                raw_filters.len(),
                MAX_LIST_FILTERS
            )));
        }
        for (field, raw) in raw_filters {
            if field.len() > MAX_FILTER_STR_LEN {
                return Err(invalid(format!(
                    "filter field name too long: {} chars (max {})",
                    field.len(),
                    MAX_FILTER_STR_LEN
                )));
            }
            filters.push((
                field.clone(),
                parse_field_filter(field, raw).map_err(invalid)?,
            ));
        }
        // Deterministic order keeps generated SQL stable across calls.
        filters.sort_by(|a, b| a.0.cmp(&b.0));
    }

    let order_by = match &params.order_by {
        Some(raw) => OrderBy::parse(raw).map_err(invalid)?,
        None => OrderBy::default(),
    };

    if let Some(fields) = &params.fields
        && fields.len() > MAX_LIST_FILTERS
    {
        return Err(invalid(format!(
            "too many fields requested: {} (max {})",
            fields.len(),
            MAX_LIST_FILTERS
        )));
    }

    if let Some(prefix) = &params.path_prefix
        && prefix.len() > MAX_FILTER_STR_LEN
    {
        return Err(invalid(format!(
            "path_prefix too long: {} chars (max {})",
            prefix.len(),
            MAX_FILTER_STR_LEN
        )));
    }

    Ok(DocumentQuery {
        filters,
        path_prefix: params.path_prefix.clone(),
        order_by,
        order_desc: params.descending.unwrap_or(false),
        limit: params
            .limit
            .unwrap_or(DEFAULT_LIST_LIMIT)
            .clamp(1, MAX_LIST_LIMIT),
        offset: params.offset.unwrap_or(0),
        fields: params.fields.clone(),
    })
}

fn validate_search_params(params: &SearchParams) -> Result<(), McpError> {
    if params.query.len() > MAX_QUERY_LEN {
        return Err(McpError::invalid_params(
            format!("query exceeds maximum length of {MAX_QUERY_LEN} characters"),
            None,
        ));
    }
    if let Some(domain) = &params.domain
        && domain.len() > MAX_FILTER_STR_LEN
    {
        return Err(McpError::invalid_params(
            format!("domain exceeds maximum length of {MAX_FILTER_STR_LEN} characters"),
            None,
        ));
    }
    if let Some(doc_type) = &params.r#type
        && doc_type.len() > MAX_FILTER_STR_LEN
    {
        return Err(McpError::invalid_params(
            format!("type exceeds maximum length of {MAX_FILTER_STR_LEN} characters"),
            None,
        ));
    }
    if let Some(tags) = &params.tags {
        if tags.len() > MAX_TAG_COUNT {
            return Err(McpError::invalid_params(
                format!("tags list exceeds maximum of {MAX_TAG_COUNT} entries"),
                None,
            ));
        }
        for tag in tags {
            if tag.len() > MAX_TAG_LEN {
                return Err(McpError::invalid_params(
                    format!("tag exceeds maximum length of {MAX_TAG_LEN} characters"),
                    None,
                ));
            }
        }
    }
    Ok(())
}

/// Validate that `rel_path` is safe to write inside `data_root`, returning the
/// absolute target path on success or an error string on failure.
///
/// Checks performed (in order):
/// 1. Reject absolute paths.
/// 2. Reject any `..` component.
/// 3. Lexical `starts_with` check on the joined abs path.
/// 4. Canonicalize the deepest *existing* ancestor of the target; verify it
///    still `starts_with` the canonical data_root. This catches a symlinked
///    ancestor directory that resolves to a location outside data_root.
pub fn resolve_safe_write_path(data_root: &Path, rel_path: &str) -> Result<PathBuf, String> {
    // A leading `/` means "the knowledge-base root", not a filesystem path — callers
    // have no way to know where the KB lives inside the container, so treating `/x.md`
    // and `x.md` as the same location is the only reading that makes sense here.
    let rel_path = crate::retrieval::kb_root_relative(rel_path);
    let requested = Path::new(rel_path);
    if requested.is_absolute() {
        return Err("path must be relative to the knowledge base root".to_string());
    }

    // 2. Reject any `..` component.
    for component in requested.components() {
        if component == std::path::Component::ParentDir {
            return Err("path must not contain '..' components".to_string());
        }
    }

    let abs_path = data_root.join(rel_path);

    // 3. Lexical starts_with check.
    if !abs_path.starts_with(data_root) {
        return Err("path escapes the knowledge base root".to_string());
    }

    // 4. Canonical-ancestor check: canonicalize data_root, then walk up from
    //    abs_path to find the deepest ancestor that actually exists on disk,
    //    canonicalize it, and confirm it still sits under canonical data_root.
    let canonical_root = data_root.canonicalize().map_err(|e| {
        format!(
            "cannot canonicalize data root '{}': {}",
            data_root.display(),
            e
        )
    })?;

    // Walk from abs_path upward until we find an existing ancestor.
    let mut candidate = abs_path.as_path();
    let existing_ancestor = loop {
        if candidate.exists() {
            break candidate;
        }
        match candidate.parent() {
            Some(p) => candidate = p,
            None => {
                // No ancestor exists at all (shouldn't happen since data_root must exist).
                break data_root;
            }
        }
    };

    let canonical_ancestor = existing_ancestor.canonicalize().map_err(|e| {
        format!(
            "cannot canonicalize ancestor '{}': {}",
            existing_ancestor.display(),
            e
        )
    })?;

    if !canonical_ancestor.starts_with(&canonical_root) {
        return Err("path escapes the knowledge base root (symlink detected)".to_string());
    }

    Ok(abs_path)
}

/// Parameters for the `get_document` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GetDocumentParams {
    /// Path of the document to retrieve. Accepts paths relative to the
    /// knowledge-base root (e.g. `lifestyle/vehicles/foo.md`, as returned by
    /// the `search` tool), or just a basename when it's unique across the
    /// index. Absolute paths are also accepted for backwards compatibility.
    pub path: String,
}

/// Parameters for the `list_documents` tool.
///
/// Every field is optional: with no arguments at all the tool pages through the whole
/// knowledge base.
#[derive(Debug, Default, serde::Deserialize, schemars::JsonSchema)]
pub struct ListDocumentsParams {
    /// Frontmatter criteria, keyed by field name. Nested fields use dot-paths
    /// (`planning.prep_minutes`). Values may be:
    /// a scalar for equality (`{"type": "guide"}`), an array for any-of
    /// (`{"tags": ["recipe", "dinner"]}`), or an object for all-of and numeric
    /// comparison (`{"tags": {"all_of": ["recipe", "dinner"]}}`,
    /// `{"planning.prep_minutes": {"lt": 30}}`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filters: Option<serde_json::Map<String, serde_json::Value>>,

    /// Restrict to documents whose path starts with this prefix, e.g.
    /// `lifestyle/kitchen/recipes/`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_prefix: Option<String>,

    /// Sort key: `path` (default), `title`, `mtime`, or `indexed_at`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order_by: Option<String>,

    /// Sort descending instead of ascending.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub descending: Option<bool>,

    /// Maximum documents to return (default 100, max 1000). The response always
    /// reports the full match count, so truncation is never silent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,

    /// Number of documents to skip, for paging.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<u64>,

    /// Frontmatter fields to include per document (dot-paths). Omit for all of them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fields: Option<Vec<String>>,
}

/// Parameters for the `get_schema` tool.
#[derive(Debug, Default, serde::Deserialize, schemars::JsonSchema)]
pub struct GetSchemaParams {
    /// Directory or document path whose governing rules to resolve. Omit for the root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,

    /// Only report these fields (dot-paths). Omit for all of them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fields: Option<Vec<String>>,

    /// Only report fields that declare a closed value set — the vocabulary view.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values_only: Option<bool>,
}

/// Parameters for the `update_schema` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct UpdateSchemaParams {
    /// Directory whose schema to edit, e.g. `lifestyle/kitchen/recipes`. Empty or
    /// omitted edits the knowledge-base root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,

    /// What to change: `add_values`, `remove_values`, `set_field`, or `remove_field`.
    pub operation: String,

    /// Field this operation targets. Nested fields use dot-paths.
    pub field: String,

    /// Values, for `add_values` and `remove_values`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<String>>,

    /// Field definition, for `set_field`. Accepts the same keys as a `.kb-schema.yaml`
    /// entry: `type`, `required`, `indexed`, `values`, `extend`, `default`, `open`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub definition: Option<serde_json::Value>,

    /// Report what the change would do without writing anything, including which
    /// existing documents it would invalidate. Never refuses — it always succeeds and
    /// reports. When false (the default), a change that would invalidate existing
    /// documents is refused unless `force` is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_run: Option<bool>,

    /// Apply even when existing documents would fail the new rules.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force: Option<bool>,
}

/// Parameters for the `search` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchParams {
    /// The natural-language search query.
    pub query: String,

    /// Optional: filter results to a specific domain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,

    /// Optional: filter results by document type.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,

    /// Optional: filter results to documents that have any of these tags.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,

    /// Maximum number of results to return (default: 10, max: 50).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,

    /// Minimum relevance score floor (0.0–1.0 for dense; ~0.01–0.03 for hybrid
    /// RRF scores). Results below this threshold are dropped. Overrides the
    /// global `search.min_score` config when provided.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_score: Option<f32>,

    /// When true, include a score-breakdown line per result showing retrieval
    /// mode and, when reranking was active, the pre-rerank score.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explain: Option<bool>,

    /// Exclude documents whose `mtime` is before this date. Accepts RFC 3339
    /// datetimes (e.g. `2024-01-15T00:00:00Z`) or date-only (e.g. `2024-01-15`).
    /// Documents indexed before mtime tracking was introduced may be excluded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified_after: Option<String>,

    /// Exclude documents whose `mtime` is after this date. Accepts the same
    /// formats as `modified_after`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified_before: Option<String>,
}

/// Parameters for the `create_document` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct CreateDocumentParams {
    /// Path of the new document, relative to the knowledge base root, e.g. "sysadmin/docker/foo.md"
    pub path: String,
    /// Full markdown content of the document, INCLUDING YAML frontmatter
    pub content: String,
    /// Optional commit message; if omitted, a message is generated from the path
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Skip the duplicate-detection check and create the document even if a
    /// similar one exists. Use this when you have verified the existing document
    /// is sufficiently different and want to create a distinct entry anyway.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub force_new: Option<bool>,
}

/// Parameters for the `edit_document` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct EditDocumentParams {
    /// Path of the document to edit. Resolved like get_document (relative to the KB
    /// root, a unique basename, or absolute).
    pub path: String,
    /// Surgical edit: exact text to find. Must occur EXACTLY ONCE in the document.
    /// Provide together with new_string. Mutually exclusive with `content`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_string: Option<String>,
    /// Surgical edit: text that replaces old_string. Provide together with old_string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_string: Option<String>,
    /// Full replace: the entire new content of the document, including YAML
    /// frontmatter. Mutually exclusive with old_string/new_string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Optional commit message; defaults to "docs: update {path}".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Parameters for the `delete_document` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct DeleteDocumentParams {
    /// Path of the document to delete. Resolved like get_document (relative to the
    /// KB root, a unique basename, or absolute).
    pub path: String,
    /// Optional commit message; defaults to "docs: delete {path}".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Validated edit mode, produced by `parse_edit_mode`.
#[derive(Debug, PartialEq)]
pub enum EditMode {
    /// Replace `old_string` with `new_string` (must appear exactly once).
    Surgical { old: String, new: String },
    /// Replace the entire document content.
    Full { content: String },
}

/// Parse and validate the mode fields of `EditDocumentParams`, returning a
/// typed `EditMode` or a human-readable error string.
///
/// Rules:
/// - SURGICAL = `old_string` AND `new_string` both `Some`, `content` is `None`.
/// - FULL = `content` is `Some`, both `old_string` and `new_string` are `None`.
/// - Any other combination is rejected.
/// - Surgical with `old_string == new_string` is rejected (no-op).
pub fn parse_edit_mode(params: &EditDocumentParams) -> Result<EditMode, String> {
    let has_content = params.content.is_some();
    let has_old = params.old_string.is_some();
    let has_new = params.new_string.is_some();

    match (has_content, has_old, has_new) {
        // Full mode
        (true, false, false) => Ok(EditMode::Full {
            content: params.content.clone().unwrap(),
        }),
        // Surgical mode
        (false, true, true) => {
            let old = params.old_string.clone().unwrap();
            let new = params.new_string.clone().unwrap();
            if old == new {
                return Err(
                    "old_string and new_string are identical — no change would be made".to_string(),
                );
            }
            Ok(EditMode::Surgical { old, new })
        }
        // Both modes set
        (true, _, _) if has_old || has_new => {
            Err("content is mutually exclusive with old_string/new_string; \
             provide either content (full replace) or old_string+new_string (surgical edit)"
                .to_string())
        }
        // Only one of old_string/new_string
        (false, true, false) => {
            Err("old_string requires new_string; provide both for a surgical edit".to_string())
        }
        (false, false, true) => {
            Err("new_string requires old_string; provide both for a surgical edit".to_string())
        }
        // Neither mode
        (false, false, false) => Err(
            "must provide either content (full replace) or old_string+new_string (surgical edit)"
                .to_string(),
        ),
        // Unreachable combinations (content=true, old=true, new=true or content=true, old/new only)
        _ => Err("content is mutually exclusive with old_string/new_string; \
             provide either content (full replace) or old_string+new_string (surgical edit)"
            .to_string()),
    }
}

/// Apply a surgical edit: replace the single occurrence of `old_string` with
/// `new_string` in `old_content`.
///
/// Returns the new content string on success, or a descriptive error string.
pub fn apply_surgical(
    old_content: &str,
    old_string: &str,
    new_string: &str,
) -> Result<String, String> {
    let count = old_content.matches(old_string).count();
    match count {
        0 => Err("old_string not found in document".to_string()),
        1 => Ok(old_content.replacen(old_string, new_string, 1)),
        n => Err(format!(
            "old_string is not unique in document (found {n} occurrences); \
             include more surrounding context to disambiguate"
        )),
    }
}

/// Render a unified diff between `old` and `new` content, labelled with
/// `a/<relpath>` and `b/<relpath>`. Returns an empty string if there is no
/// diff (shouldn't happen for a real change).
pub fn render_unified_diff(old: &str, new: &str, relpath: &str) -> String {
    use similar::TextDiff;
    let diff = TextDiff::from_lines(old, new);
    diff.unified_diff()
        .context_radius(3)
        .header(&format!("a/{relpath}"), &format!("b/{relpath}"))
        .to_string()
}

/// Build a commit message with git trailers identifying the tool and operation.
///
/// The resulting message has the form:
/// ```text
/// <subject line>
///
/// Tool: md-kb-rag
/// Operation: <operation>
/// ```
///
/// `user_subject` is the caller-supplied commit message (if any). When absent,
/// `default_subject` is used. The trailer block is always appended after a blank line.
pub fn build_commit_message(
    user_subject: Option<&str>,
    default_subject: &str,
    operation: &str,
) -> String {
    let subject = user_subject.unwrap_or(default_subject);
    format!("{}\n\nTool: md-kb-rag\nOperation: {}", subject, operation)
}

#[derive(Clone)]
pub struct KbSearchServer {
    embed_client: Arc<EmbedClient>,
    qdrant: Arc<QdrantStore>,
    collection: String,
    canonical_data_path: PathBuf,
    /// Glob patterns (from `indexing.include`) used to restrict `get_document` to permitted file types.
    include_patterns: Arc<GlobSet>,
    /// Dynamic MCP server instructions, refreshed periodically with discovered metadata.
    instructions: Arc<RwLock<String>>,
    /// Live config handle. Every tool call fetches its own fresh snapshot via
    /// [`Self::config`] rather than caching one at construction, so `POST
    /// /admin/reload` is observed by the very next call — see that method's doc
    /// comment for exactly which settings this makes dynamic.
    config: crate::config::SharedConfig,
    /// The shared, cached schema tree — built once at server startup and kept
    /// current by the reindex worker (which rebuilds it before indexing any
    /// dirty `.kb-schema.yaml`) and by `update_schema`'s own synchronous rebuild.
    /// `get_schema`, `update_schema`, and the write path all read this instead of
    /// re-walking the knowledge base on every call; see `schema::SharedSchemaCache`.
    schema_cache: crate::schema::SharedSchemaCache,
    rerank_client: Option<Arc<RerankClient>>,
    /// Handle to the document metadata index, opened on first use and held for the
    /// process lifetime. Under WAL this reader coexists with the short-lived writer
    /// pools that the reindex worker and CLI open.
    ///
    /// Lazy rather than constructor-injected so building a server stays synchronous
    /// and infallible with respect to SQLite availability.
    state_db: Arc<tokio::sync::OnceCell<StateDb>>,
}

/// Build an include `GlobSet` for MCP path filtering, with a `**/*.md` fallback
/// when no patterns are valid. Per-pattern parsing uses [`crate::ingest::parse_globs`]
/// so both sites share the same skip-and-warn policy; this function keeps its own
/// failure policy (fall back to `**/*.md` on empty or builder error) distinct from
/// `ingest::build_globset`, which propagates errors to the caller.
fn build_include_globset(patterns: &[String]) -> GlobSet {
    let (mut builder, valid_count) = crate::ingest::parse_globs(patterns);
    if valid_count == 0 {
        warn!("No valid include patterns configured — falling back to **/*.md");
        builder.add(Glob::new("**/*.md").unwrap());
    }
    builder.build().unwrap_or_else(|e| {
        error!(
            "Failed to build include globset: {} — falling back to **/*.md",
            e
        );
        let mut fallback = GlobSetBuilder::new();
        fallback.add(Glob::new("**/*.md").unwrap());
        fallback
            .build()
            .expect("hardcoded fallback glob '**/*.md' must compile")
    })
}

#[tool_router]
impl KbSearchServer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        embed_client: Arc<EmbedClient>,
        qdrant: Arc<QdrantStore>,
        collection: String,
        data_path: PathBuf,
        include_patterns: &[String],
        instructions: Arc<RwLock<String>>,
        config: crate::config::SharedConfig,
        schema_cache: crate::schema::SharedSchemaCache,
        rerank_client: Option<Arc<RerankClient>>,
    ) -> anyhow::Result<Self> {
        let canonical_data_path = data_path.canonicalize().with_context(|| {
            format!("Failed to canonicalize data path: {}", data_path.display())
        })?;
        Ok(Self {
            embed_client,
            qdrant,
            collection,
            canonical_data_path,
            // Compiled once from the config snapshot at construction time — a
            // `POST /admin/reload` that changes `indexing.include` does NOT change
            // what get_document accepts until a restart. See
            // `reload.rs`'s "indexing.include (MCP get_document path filter)" entry.
            include_patterns: Arc::new(build_include_globset(include_patterns)),
            instructions,
            config,
            schema_cache,
            rerank_client,
            state_db: Arc::new(tokio::sync::OnceCell::new()),
        })
    }

    /// A fresh snapshot of the live config — a lock acquisition plus an `Arc`
    /// clone, mirroring `schema::load_shared`. Every tool call fetches its own
    /// snapshot here rather than reading a value captured at construction, so a
    /// `POST /admin/reload` swap is observed starting with the very next call.
    fn config(&self) -> Arc<ResolvedConfig> {
        crate::config::load_shared_config(&self.config)
    }

    /// Write a non-document file into the KB, commit it, and queue a full reconcile.
    ///
    /// Used for `.kb-schema.yaml`, which is versioned and synced like a document but is
    /// not itself indexed. The write goes to a temp file and is renamed into place, so a
    /// failure part-way cannot leave a half-written schema that would freeze the scope.
    async fn write_raw_file(
        &self,
        rel_path: &str,
        content: &str,
        commit_message: &str,
    ) -> Result<(), McpError> {
        let config = self.config();

        // Same resolver the document write tools use. Joining the data root with a
        // caller-supplied path is NOT sufficient on its own: the knowledge base is a
        // synced git repo, and git materializes tracked symlinks on checkout, so a
        // hostile upstream commit could otherwise redirect this write outside the KB.
        let abs_path = resolve_safe_write_path(&self.canonical_data_path, rel_path)
            .map_err(|e| McpError::invalid_params(format!("Invalid schema path: {}", e), None))?;

        if let Some(parent) = abs_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                error!("Failed to create directory '{}': {}", parent.display(), e);
                McpError::internal_error(format!("Failed to create directory: {}", e), None)
            })?;
        }

        // Re-check after creating the directory: `resolve_safe_write_path` can only
        // canonicalize ancestors that existed at the time, so a newly created path
        // component is verified here.
        resolve_safe_write_path(&self.canonical_data_path, rel_path)
            .map_err(|e| McpError::invalid_params(format!("Invalid schema path: {}", e), None))?;

        // Unique per call, not merely per process: two concurrent requests inside one
        // server would otherwise share a temp path and silently clobber each other,
        // with the loser still reporting success.
        static WRITE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = WRITE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let temp_path = abs_path.with_extension(format!("tmp-{}-{}", std::process::id(), seq));
        tokio::fs::write(&temp_path, content.as_bytes())
            .await
            .map_err(|e| {
                error!("Failed to write '{}': {}", temp_path.display(), e);
                McpError::internal_error(format!("Failed to write file: {}", e), None)
            })?;
        tokio::fs::rename(&temp_path, &abs_path)
            .await
            .map_err(|e| {
                error!("Failed to install '{}': {}", abs_path.display(), e);
                McpError::internal_error(format!("Failed to write file: {}", e), None)
            })?;

        let token = std::env::var(&config.source.git_token_env)
            .ok()
            .filter(|s| !s.is_empty());

        git::commit_and_sync(
            config.source.git_url.as_deref(),
            &config.source.branch,
            self.canonical_data_path.to_str().unwrap_or_default(),
            token.as_deref(),
            rel_path,
            commit_message,
            &config.write.commit_author_name,
            &config.write.commit_author_email,
        )
        .await
        .map_err(|e| {
            error!("commit_and_sync failed for '{}': {:#}", rel_path, e);
            McpError::internal_error(format!("Git commit/sync failed: {}", e), None)
        })?;

        // A schema change revalidates its whole subtree via the schema fingerprint —
        // any document under this scope can flip from valid to invalid or vice versa —
        // and there is no cheap way to enumerate exactly which paths that touches
        // without a walk. Rather than approximate it, mark a full reconcile: the
        // worker will scan, and `index_paths`' existing schema-fingerprint check
        // (unrelated to this reconcile's OWN full-walk vs scoped distinction) is what
        // actually catches the affected documents once it re-reads them.
        crate::reindex::mark_full();

        Ok(())
    }

    /// Documents already under `rel_dir` that a candidate schema would reject.
    ///
    /// Answered from the metadata index rather than by re-reading markdown: every
    /// document's frontmatter is stored as JSON, so this is a query.
    async fn documents_broken_by(
        &self,
        rel_dir: &std::path::Path,
        schemas: &SchemaCache,
        candidate_file: &crate::schema::SchemaFile,
    ) -> Result<Vec<serde_json::Value>, McpError> {
        let index = self.state_db().await.map_err(|e| {
            error!("Schema dry-run could not open the metadata index: {:#}", e);
            McpError::internal_error(
                format!("Cannot check existing documents: {e}. Index unavailable."),
                None,
            )
        })?;

        let prefix = if rel_dir.as_os_str().is_empty() {
            None
        } else {
            Some(format!("{}/", rel_dir.to_string_lossy()))
        };

        let query = DocumentQuery {
            path_prefix: prefix,
            // The whole point is completeness; a truncated check would report a clean
            // dry-run for a change that breaks documents beyond the page.
            limit: u32::MAX as u64,
            ..Default::default()
        };

        let listing = index.query_documents(&query).await.map_err(|e| {
            error!("Schema dry-run query failed: {:#}", e);
            McpError::internal_error(format!("Cannot check existing documents: {e}"), None)
        })?;

        let mut casualties = Vec::new();
        for doc in &listing.documents {
            let Some(map) = doc.frontmatter.as_object() else {
                continue;
            };

            // Resolve each document against ITS OWN effective schema under the proposed
            // edit, not against the edited directory's. A descendant scope that
            // redefines the field being changed is unaffected by this edit, and
            // validating it against the parent's new rule would report a casualty that
            // does not exist — blocking a legitimate change.
            let doc_path = std::path::Path::new(&doc.file_path);
            let effective = schemas.resolve_with_candidate(doc_path, rel_dir, candidate_file);
            if effective.is_none() {
                continue;
            }
            let effective = effective.expect("checked above");

            let mut frontmatter: std::collections::HashMap<String, serde_json::Value> =
                map.clone().into_iter().collect();
            // The real indexing path fills in schema defaults before validating, so
            // skipping that here reports a required field WITH a default as breaking
            // every document that omits it — blocking a genuinely safe change and
            // pushing the operator toward `force`, which also bypasses the real checks.
            validate::apply_defaults(&mut frontmatter, &effective);
            let errors = validate::validate_frontmatter(&frontmatter, &effective);
            if let Some(first) = errors.first() {
                casualties.push(serde_json::json!({
                    "path": doc.file_path,
                    "reason": first.message,
                    "error_count": errors.len(),
                }));
            }
        }

        Ok(casualties)
    }

    /// The document metadata index, opened on first use.
    async fn state_db(&self) -> anyhow::Result<&StateDb> {
        self.state_db
            .get_or_try_init(|| async {
                let path = self.config().state_db_path();
                StateDb::new(std::path::Path::new(&path))
                    .await
                    .with_context(|| format!("Failed to open state DB at {}", path))
            })
            .await
    }

    /// Build a `RetrievalDeps` bundle from this server's fields.
    fn deps(&self) -> RetrievalDeps<'_, EmbedClient, QdrantStore> {
        RetrievalDeps {
            embed_client: &self.embed_client,
            qdrant: &self.qdrant,
            collection: &self.collection,
            data_path: &self.canonical_data_path,
            include_patterns: &self.include_patterns,
            reranker: self
                .rerank_client
                .as_ref()
                .map(|c| c.as_ref() as &(dyn crate::rerank::Reranker + Send + Sync)),
        }
    }

    #[tool(
        description = "Search the knowledge base using a natural-language query. \
        Returns ranked document chunks with title, relevance score, text snippet, and metadata.\n\
        \n\
        Filters: domain, type, tags — narrow results to matching documents.\n\
        \n\
        Quality controls:\n\
        - min_score: drop results below a relevance floor (float). Hybrid RRF scores are \
          ~0.01–0.03; dense cosine scores are 0.0–1.0. Set accordingly.\n\
        - explain: add a per-result score-breakdown line showing retrieval mode and, when \
          reranking was active, the pre-rerank score.\n\
        \n\
        Recency filters (ISO 8601 date or datetime, e.g. \"2024-01-15\" or \
        \"2024-01-15T00:00:00Z\"):\n\
        - modified_after: exclude documents with mtime before this date.\n\
        - modified_before: exclude documents with mtime after this date.\n\
        Note: documents indexed before mtime tracking was introduced may be excluded."
    )]
    async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        validate_search_params(&params)?;

        let limit = resolve_limit(params.limit);

        debug!(
            query = %&params.query.chars().take(100).collect::<String>(),
            limit,
            has_domain = params.domain.is_some(),
            has_type = params.r#type.is_some(),
            has_tags = params.tags.is_some(),
            "search called"
        );

        let filters = SearchFilters {
            domain: params.domain,
            r#type: params.r#type,
            tags: params.tags,
        };

        let modified_after = params
            .modified_after
            .as_deref()
            .map(parse_date_to_timestamp)
            .transpose()
            .map_err(|e| McpError::invalid_params(e, None))?;
        let modified_before = params
            .modified_before
            .as_deref()
            .map(parse_date_to_timestamp)
            .transpose()
            .map_err(|e| McpError::invalid_params(e, None))?;

        // Fetched once per call so every field below — including
        // reranking.candidate_limit — reflects the same live snapshot, rather than
        // racing a concurrent `POST /admin/reload` mid-request.
        let config = self.config();
        let explain = params.explain.unwrap_or(false);
        let opts = SearchOptions {
            limit,
            min_score: params.min_score.or(config.search.min_score),
            hybrid: config.search.hybrid,
            rrf_candidates: config.search.rrf_candidates as u64,
            explain,
            modified_after,
            modified_before,
            rerank_candidate_limit: config.reranking.as_ref().map(|r| r.candidate_limit as u64),
        };

        let results = retrieval::search(&self.deps(), &params.query, &filters, &opts)
            .await
            .map_err(|e| match e {
                retrieval::SearchError::Embed(err) => {
                    error!("Embedding query failed: {:#}", err);
                    McpError::internal_error("Failed to generate query embedding".to_string(), None)
                }
                retrieval::SearchError::Search(err) => {
                    error!("Qdrant search failed: {:#}", err);
                    McpError::internal_error("Search query failed".to_string(), None)
                }
            })?;

        debug!(result_count = results.len(), "search returned results");

        if results.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No results found.",
            )]));
        }

        // Format results as text content
        let mut output = String::new();
        for (i, result) in results.iter().enumerate() {
            let title = result
                .payload
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or("(untitled)");

            let (text_snippet, needs_ellipsis) = {
                let full_text = result
                    .payload
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let chars: Vec<char> = full_text.chars().take(801).collect();
                if chars.len() > 800 {
                    (chars[..800].iter().collect::<String>(), true)
                } else {
                    (chars.into_iter().collect::<String>(), false)
                }
            };

            let file_path_raw = result
                .payload
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let file_path = retrieval::relative_to_data(file_path_raw, &self.canonical_data_path);

            let domain = result
                .payload
                .get("domain")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let doc_type = result
                .payload
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let tags = result
                .payload
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|t| t.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();

            let lines = match (
                result.payload.get("line_start").and_then(|v| v.as_i64()),
                result.payload.get("line_end").and_then(|v| v.as_i64()),
            ) {
                (Some(s), Some(e)) => format!(" (lines {s}–{e})"),
                _ => String::new(),
            };

            output.push_str(&format!(
                "## Result {rank}\n\
                **Title**: {title}\n\
                **Score**: {score:.4}\n\
                **File**: {file_path}{lines}\n",
                rank = i + 1,
                title = title,
                score = result.score,
                file_path = file_path,
                lines = lines,
            ));

            if explain {
                let mode = if config.search.hybrid {
                    "hybrid RRF"
                } else {
                    "dense cosine"
                };
                let mut breakdown = if let Some(pre) = result.pre_rerank_score {
                    format!(
                        "**Score breakdown**: mode={mode}, rerank={:.4}, pre-rerank={:.4}",
                        result.score, pre,
                    )
                } else {
                    format!(
                        "**Score breakdown**: mode={mode}, score={:.4}",
                        result.score,
                    )
                };
                if let Some(d) = result.dense_score {
                    breakdown.push_str(&format!(", dense={d:.4}"));
                }
                if let Some(s) = result.sparse_score {
                    breakdown.push_str(&format!(", sparse={s:.4}"));
                }
                breakdown.push('\n');
                output.push_str(&breakdown);
            }

            if !domain.is_empty() {
                output.push_str(&format!("**Domain**: {domain}\n"));
            }
            if !doc_type.is_empty() {
                output.push_str(&format!("**Type**: {doc_type}\n"));
            }
            if !tags.is_empty() {
                output.push_str(&format!("**Tags**: {tags}\n"));
            }

            if !text_snippet.is_empty() {
                let ellipsis = if needs_ellipsis { "..." } else { "" };
                output.push_str(&format!("\n{text_snippet}{ellipsis}\n"));
            }

            output.push('\n');
        }

        Ok(CallToolResult::success(vec![Content::text(output.trim())]))
    }

    #[tool(
        description = "Show the frontmatter rules governing a path. Schemas cascade by \
        directory: a .kb-schema.yaml applies to its folder and everything beneath it, \
        with deeper files refining shallower ones. This returns the fully MERGED result \
        for the path you ask about, plus which schema file contributed each field.\n\
        \n\
        Call this before creating or editing a document — the rules in the server \
        instructions are root-level only and may not be complete for a given folder.\n\
        \n\
        - path: directory or document path; omit for the knowledge-base root\n\
        - fields: only report these dot-paths\n\
        - values_only: only report fields with a closed set of permitted values"
    )]
    async fn get_schema(
        &self,
        Parameters(params): Parameters<GetSchemaParams>,
    ) -> Result<CallToolResult, McpError> {
        let raw = params.path.clone().unwrap_or_default();
        let rel = normalize_scope_path(&raw)?;
        // Reads the shared, server-owned cache rather than walking the tree on every
        // call — this tool is the one the server's own instructions tell agents to
        // call before every write, so it needs to be cheap. See
        // `KbSearchServer::schema_cache`'s doc comment for how it stays current.
        let schemas = crate::schema::load_shared(&self.schema_cache);

        // A document path resolves via its parent; a directory resolves to itself. A
        // partial directory reference resolves like a partial document path does —
        // exact match wins, otherwise report the candidates rather than guessing.
        let rel = if raw.ends_with(".md") || rel.as_os_str().is_empty() {
            rel
        } else {
            resolve_scope_reference(&schemas, &rel)?
        };
        let lookup = if raw.ends_with(".md") {
            rel.clone()
        } else {
            rel.join("_")
        };
        let schema = schemas.resolve_for(&lookup);

        let values_only = params.values_only.unwrap_or(false);
        let mut reported: Vec<serde_json::Value> = Vec::new();
        let mut omitted = 0usize;
        for (field, def) in &schema.fields {
            if let Some(wanted) = &params.fields
                && !wanted.contains(field)
            {
                continue;
            }
            if values_only && def.values.is_none() {
                continue;
            }
            // Schema files arrive via git sync, so field count is attacker-controlled
            // and the per-field value cap alone does not bound the response. Counted
            // after the filters so `omitted` reflects fields the caller actually asked
            // for, not every remaining field in the schema.
            if reported.len() >= MAX_REPORTED_FIELDS {
                omitted += 1;
                continue;
            }
            // Field names, permitted values, and provenance paths all originate in
            // .kb-schema.yaml files from a synced repo. The instructions actively steer
            // agents to call this tool before every write, so it is a reliably-triggered
            // reflection point — strip control characters and cap length on everything
            // that came from the knowledge base.
            let clean_values = def.values.as_ref().map(|vs| {
                vs.iter()
                    .take(MAX_REPORTED_VALUES)
                    .map(|v| crate::server::sanitize_facet_value(v))
                    .collect::<Vec<_>>()
            });
            reported.push(serde_json::json!({
                "field": crate::server::sanitize_facet_value(field),
                "type": def.ty.map(|t| format!("{t:?}").to_lowercase()),
                "required": def.required,
                "indexed": def.indexed,
                "values": clean_values,
                "default": def.default.as_ref().map(sanitize_reflected_value),
                "open": def.open,
                "declared_in": schema
                    .origin
                    .get(field)
                    .map(|o| crate::server::sanitize_facet_value(o)),
            }));
        }

        let frozen = schemas.is_frozen(&lookup);
        let structured = serde_json::json!({
            "path": rel.to_string_lossy(),
            "frozen": frozen.is_some(),
            "frozen_reason": frozen,
            "fields": reported,
            "omitted_fields": omitted,
        });

        let mut text = format!(
            "Schema governing '{}' ({} field(s)):\n\n",
            rel.display(),
            reported.len()
        );
        if let Some(reason) = frozen {
            text.push_str(&format!(
                "WARNING: this scope is frozen — its schema file is invalid ({reason}). \
                 Documents here are not being indexed.\n\n"
            ));
        }
        if omitted > 0 {
            text.push_str(&format!(
                "({omitted} further field(s) omitted; narrow with the fields parameter.)\n\n"
            ));
        }
        for entry in &reported {
            text.push_str(&format!("- {}", entry["field"].as_str().unwrap_or("?")));
            if let Some(ty) = entry["type"].as_str() {
                text.push_str(&format!(" ({ty})"));
            }
            if entry["required"] == serde_json::json!(true) {
                text.push_str(" [required]");
            }
            if let Some(values) = entry["values"].as_array() {
                let rendered: Vec<String> = values
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect();
                text.push_str(&format!(" — one of: {}", rendered.join(", ")));
            }
            if let Some(origin) = entry["declared_in"].as_str() {
                text.push_str(&format!("  (from {origin})"));
            }
            text.push('\n');
        }

        let mut result = CallToolResult::success(vec![Content::text(text.trim_end())]);
        result.structured_content = Some(structured);
        Ok(result)
    }

    #[tool(
        description = "Change the frontmatter rules for a directory, editing its \
        .kb-schema.yaml. Use this when a new document warrants a new tag or field \
        rather than working around the rules.\n\
        \n\
        Operations: add_values / remove_values (adjust a field's permitted set), \
        set_field (declare or replace a field definition), remove_field.\n\
        \n\
        Every change is checked against the documents that already exist under this \
        scope BEFORE anything is written. If the change would invalidate any of them, \
        it is refused and they are listed — pass force to apply anyway. Pass dry_run to \
        see the effect without writing. The file is committed and pushed like any \
        document edit."
    )]
    async fn update_schema(
        &self,
        Parameters(params): Parameters<UpdateSchemaParams>,
    ) -> Result<CallToolResult, McpError> {
        let invalid = |msg: String| McpError::invalid_params(msg, None);

        let requested = normalize_scope_path(params.path.as_deref().unwrap_or(""))?;
        let edit = build_schema_edit(&params)?;

        // Read the shared cache for the pre-edit view (casualty check, raw file
        // lookup below). This does NOT need to be maximally fresh — worst case a
        // concurrent change this call doesn't see yet is caught by the casualty
        // check failing safe or by a subsequent call — but see the synchronous
        // rebuild after the write below, which is NOT optional.
        let schemas = crate::schema::load_shared(&self.schema_cache);

        // Resolve a partial reference against existing scopes, but fall back to the
        // literal path: creating a schema for a directory that has none yet is the
        // normal way to introduce one, so "no match" is not an error here.
        let matches = schemas.match_scope_dirs(&requested);
        let rel_dir = match matches.len() {
            0 => requested,
            1 => matches.into_iter().next().expect("length checked"),
            _ => {
                return Err(McpError::invalid_params(
                    format!(
                        "'{}' matches {} scopes: {}. Use a more specific path.",
                        requested.display(),
                        matches.len(),
                        matches
                            .iter()
                            .map(|p| p.display().to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    None,
                ));
            }
        };

        let mut file = schemas
            .raw_file_at(&rel_dir)
            .map_err(|e| invalid(format!("Existing schema at '{}' is unreadable: {e}. Fix it by hand before editing through this tool.", rel_dir.display())))?;
        let summary = file.apply(&edit).map_err(invalid)?;

        // A self-contradictory definition parses fine but freezes the whole subtree at
        // the next index run — after this call has already reported success. Catch it
        // here, where the caller can still act on it.
        file.validate_self().map_err(invalid)?;

        let yaml = file.to_yaml().map_err(invalid)?;

        // Re-parse what we are about to write. A schema that does not round-trip would
        // freeze this whole subtree at the next index run.
        serde_yaml_ng::from_str::<crate::schema::SchemaFile>(&yaml).map_err(|e| {
            McpError::internal_error(
                format!("Refusing to write a schema that does not parse: {e}"),
                None,
            )
        })?;

        // Dry-run the change against documents that already exist under this scope.
        let casualties = self.documents_broken_by(&rel_dir, &schemas, &file).await?;

        let dry_run = params.dry_run.unwrap_or(false);
        let force = params.force.unwrap_or(false);

        if !casualties.is_empty() && !force && !dry_run {
            return Err(McpError::invalid_params(
                format!(
                    "Refusing to apply: {} existing document(s) would fail the new rules. \
                     Fix them first, or pass force to apply anyway.\n{}",
                    casualties.len(),
                    render_casualties(&casualties)
                ),
                Some(serde_json::json!({ "would_invalidate": casualties })),
            ));
        }

        if dry_run {
            let text = format!(
                "Dry run — nothing written.\n{}\nWould affect {} existing document(s).{}\n\nResulting {}:\n{}",
                summary,
                casualties.len(),
                if casualties.is_empty() {
                    String::new()
                } else {
                    format!("\n{}", render_casualties(&casualties))
                },
                crate::schema::SCHEMA_FILE_NAME,
                yaml
            );
            let mut result = CallToolResult::success(vec![Content::text(text)]);
            result.structured_content = Some(serde_json::json!({
                "dry_run": true,
                "summary": summary,
                "would_invalidate": casualties,
                "yaml": yaml,
            }));
            return Ok(result);
        }

        let rel_file = rel_dir.join(crate::schema::SCHEMA_FILE_NAME);
        let rel_file_str = rel_file.to_string_lossy().to_string();
        let commit_message = format!("schema: {summary} in {}", rel_dir.display());

        self.write_raw_file(&rel_file_str, &yaml, &commit_message)
            .await?;

        // Rebuild the shared schema cache and swap it in SYNCHRONOUSLY — before this
        // call returns, not merely "soon". `write_raw_file` already called
        // `reindex::mark_full`, so the reindex worker will ALSO rebuild it, but that
        // happens out of band on the worker's own schedule and cannot be relied on to
        // win the race against whatever the calling agent does next. The scenario this
        // guards against: an agent calls `update_schema` to permit a new value, then
        // immediately calls `create_document`/`edit_document` relying on that new rule
        // — if the write path read a stale cache, it would validate against the schema
        // this very call just replaced and wrongly reject a now-valid document. Wrapped
        // in `spawn_blocking` because the walk itself is blocking filesystem work, not
        // because anything here needs to run off-thread for its own sake — `.await`ing
        // it still makes this call return only once the rebuild has completed.
        let rebuild_data_path = self.canonical_data_path.clone();
        let rebuild_frontmatter = self.config().frontmatter.clone();
        match tokio::task::spawn_blocking(move || {
            SchemaCache::build(&rebuild_data_path, &rebuild_frontmatter)
        })
        .await
        {
            Ok(rebuilt) => crate::schema::store_shared(&self.schema_cache, rebuilt),
            Err(e) => {
                // A panic in the walk itself (not a normal error — `SchemaCache::build`
                // has no fallible return). Leave the previous cache in place rather than
                // fail the whole call: the write already succeeded and is committed: an
                // agent's NEXT read/write may briefly see the pre-edit schema, which is
                // the same staleness window this call exists to close, not a new one —
                // failing here would not close it either, just add a spurious error on
                // top of a successful write.
                error!("Schema rebuild panicked after update_schema write: {e}");
            }
        }

        let mut text = format!("{summary}\nWrote {rel_file_str}.");
        if !casualties.is_empty() {
            text.push_str(&format!(
                "\n\nWARNING: {} existing document(s) now fail validation and will stop \
                 being re-indexed until fixed:\n{}",
                casualties.len(),
                render_casualties(&casualties)
            ));
        }

        let mut result = CallToolResult::success(vec![Content::text(text)]);
        result.structured_content = Some(serde_json::json!({
            "dry_run": false,
            "summary": summary,
            "path": rel_file_str,
            "invalidated": casualties,
        }));
        Ok(result)
    }

    #[tool(
        description = "List documents by their frontmatter, without relevance ranking. \
        Use this instead of `search` whenever you need a COMPLETE set — every recipe, \
        every config document, all docs of a given type. `search` returns ranked chunks \
        and several may come from one document, so it cannot enumerate reliably.\n\
        \n\
        All parameters are optional; with none, it pages through the whole knowledge base.\n\
        - filters: frontmatter criteria keyed by field. Nested fields use dot-paths. \
        A scalar means equality ({\"type\": \"guide\"}), an array means any-of \
        ({\"tags\": [\"recipe\", \"dinner\"]}), and an object means all-of or a numeric \
        range ({\"tags\": {\"all_of\": [\"recipe\", \"dinner\"]}}, \
        {\"planning.prep_minutes\": {\"lt\": 30}}).\n\
        - path_prefix: restrict to a folder, e.g. 'lifestyle/kitchen/recipes/'.\n\
        - order_by: path (default), title, mtime, or indexed_at; descending flips it.\n\
        - limit / offset: page size (default 100, max 1000) and starting position.\n\
        - fields: which frontmatter fields to return per document; omit for all.\n\
        \n\
        The response always reports the total number of matching documents, so \
        truncation is never silent — if has_more is true, page with offset."
    )]
    async fn list_documents(
        &self,
        Parameters(params): Parameters<ListDocumentsParams>,
    ) -> Result<CallToolResult, McpError> {
        let query = build_document_query(&params)?;

        let index = self.state_db().await.map_err(|e| {
            error!("list_documents could not open the metadata index: {:#}", e);
            McpError::internal_error(format!("Document index unavailable: {}", e), None)
        })?;

        let result = retrieval::list_documents(&DocumentIndexDeps { index }, &query)
            .await
            .map_err(|e| {
                error!("list_documents failed: {:#}", e);
                McpError::internal_error(format!("Failed to list documents: {}", e), None)
            })?;

        let has_more = result.has_more(query.offset);
        let returned = result.documents.len();

        let structured = serde_json::json!({
            "total": result.total,
            "returned": returned,
            "offset": query.offset,
            "has_more": has_more,
            "documents": result
                .documents
                .iter()
                .map(|d| serde_json::json!({
                    "file_path": d.file_path,
                    "title": d.title,
                    "description": d.description,
                    "mtime": d.mtime,
                    "frontmatter": d.frontmatter,
                }))
                .collect::<Vec<_>>(),
        });

        let mut text = if result.total == 0 {
            "No documents match those criteria.".to_string()
        } else if returned == 0 {
            format!(
                "{} document(s) match, but offset {} is past the end.",
                result.total, query.offset
            )
        } else {
            format!(
                "{} document(s) match; showing {}–{}.\n\n",
                result.total,
                query.offset + 1,
                query.offset + returned as u64
            )
        };

        for doc in &result.documents {
            text.push_str(&format!("- {}", doc.file_path));
            if let Some(title) = &doc.title {
                text.push_str(&format!(" — {}", title));
            }
            text.push('\n');
            if let Some(description) = &doc.description {
                text.push_str(&format!("  {}\n", description.trim()));
            }
        }

        if has_more {
            text.push_str(&format!(
                "\n{} more document(s) match. Page with offset={} to continue.",
                result.total - query.offset - returned as u64,
                query.offset + returned as u64
            ));
        }

        // Plain text keeps parity with the other tools; the structured half is what a
        // consuming skill checks to detect truncation without parsing prose.
        let mut call_result = CallToolResult::success(vec![Content::text(text.trim_end())]);
        call_result.structured_content = Some(structured);
        Ok(call_result)
    }

    #[tool(
        description = "Retrieve the full raw content of a document by file path. \
        Accepts paths relative to the knowledge base root (e.g. \
        'lifestyle/vehicles/foo.md', as returned by the `search` tool) or just \
        the basename if it's unique across the index. Returns the complete \
        markdown including frontmatter."
    )]
    async fn get_document(
        &self,
        Parameters(params): Parameters<GetDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        let raw = params.path.trim();
        if raw.is_empty() {
            return Err(McpError::invalid_params(
                "Path parameter is empty".to_string(),
                None,
            ));
        }
        if raw.len() > MAX_PATH_LEN {
            return Err(McpError::invalid_params(
                format!("path exceeds maximum length of {MAX_PATH_LEN} characters"),
                None,
            ));
        }

        debug!(path = %raw, "get_document called");

        match retrieval::get_document(&self.deps(), raw).await {
            Ok(doc) => {
                debug!(path = %raw, "get_document served");
                Ok(CallToolResult::success(vec![Content::text(doc.content)]))
            }
            Err(GetDocumentError::Outside) => {
                warn!(path = %raw, "get_document: path outside data directory");
                Err(McpError::invalid_params(
                    "File path is outside the data directory".to_string(),
                    None,
                ))
            }
            Err(GetDocumentError::NotPermitted) => {
                warn!(path = %raw, "get_document: file type not permitted");
                Err(McpError::invalid_params(
                    "File type not permitted".to_string(),
                    None,
                ))
            }
            Err(GetDocumentError::NotFound { suggestions }) => {
                let mut msg = format!("File not found: '{}'.", raw);
                if suggestions.is_empty() {
                    msg.push_str(" Use the `search` tool to find paths.");
                } else {
                    msg.push_str(&format!(
                        " Closest indexed files: {}. Or use the `search` tool.",
                        suggestions.join(", ")
                    ));
                }
                Err(McpError::invalid_params(msg, None))
            }
            Err(GetDocumentError::Ambiguous { matches }) => {
                let basename = std::path::Path::new(raw)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(raw);
                Err(McpError::invalid_params(
                    format!(
                        "Multiple files match basename '{}': {}. Use a more specific path.",
                        basename,
                        matches.join(", ")
                    ),
                    None,
                ))
            }
            Err(GetDocumentError::Io(msg)) => Err(McpError::invalid_params(msg, None)),
        }
    }

    /// Shared pipeline for create_document and edit_document.
    ///
    /// Callers are responsible for resolving paths and computing old/new content
    /// before calling this. This function handles validation, optional dedup
    /// gating (create only), filesystem write, git commit, reindex, and diff output.
    ///
    /// * `old_content` – empty string for create; existing file bytes for edit.
    /// * `new_content` – the content to write (already computed by caller).
    /// * `rel_path`    – repo-relative path (used for git add/commit and messages).
    /// * `is_create`   – `true` for create (dedup gate active), `false` for edit.
    /// * `message`     – optional custom commit message.
    /// * `default_verb`– verb for the default commit message, e.g. `"add"` or `"update"`.
    /// * `force_new`   – when `Some(true)`, bypasses the dedup gate on create paths.
    /// * `operation`   – label for the `Operation:` git trailer, e.g. `"create_document"`.
    ///
    /// The absolute path is deliberately NOT a parameter: it is re-resolved from
    /// `rel_path` immediately before each filesystem action, so a path validated by the
    /// caller cannot go stale across the validation and dedup awaits in between.
    /// Callers still resolve it themselves for their own existence checks.
    #[allow(clippy::too_many_arguments)]
    async fn write_document(
        &self,
        old_content: &str,
        new_content: &str,
        rel_path: &str,
        is_create: bool,
        message: Option<&str>,
        default_verb: &str,
        force_new: Option<bool>,
        operation: &str,
    ) -> Result<CallToolResult, McpError> {
        // One snapshot for the whole call, so a concurrent `POST /admin/reload`
        // cannot mix old and new values across this method's several config reads.
        let config = self.config();

        if new_content.len() > MAX_CONTENT_LEN {
            return Err(McpError::invalid_params(
                format!(
                    "content is too large ({} bytes); maximum is {} bytes",
                    new_content.len(),
                    MAX_CONTENT_LEN
                ),
                None,
            ));
        }

        // 1. Validate new_content before writing (catches frontmatter errors in
        //    both full-replace and surgical edits before touching the filesystem).
        //
        // The schema is resolved from the TARGET path's directory, so writing into
        // `lifestyle/kitchen/recipes/` is governed by that folder's rules regardless of
        // where the caller has been reading. This reads the shared, server-owned cache
        // (`KbSearchServer::schema_cache`) rather than rebuilding: at knowledge-base
        // scale a full tree walk on every write is no longer "a few milliseconds", and
        // going through `update_schema` keeps this from going stale — that tool
        // rebuilds and swaps the cache SYNCHRONOUSLY before it returns, specifically so
        // a write immediately following a schema change is validated against the new
        // rules rather than the ones it just replaced. A schema edited directly on the
        // KB's git host (bypassing `update_schema`) is instead picked up by the reindex
        // worker the next time it sees that `.kb-schema.yaml` dirty.
        let schemas = crate::schema::load_shared(&self.schema_cache);
        if let Some(reason) = schemas.is_frozen(std::path::Path::new(rel_path)) {
            return Err(McpError::invalid_params(
                format!(
                    "Cannot write '{}': the schema governing this directory is invalid ({}). \
                     Fix {} before writing here.",
                    rel_path,
                    reason,
                    crate::schema::SCHEMA_FILE_NAME
                ),
                None,
            ));
        }
        let schema = schemas.resolve_for(std::path::Path::new(rel_path));

        let (validation_result, validated) = validate::validate_content(
            std::path::Path::new(rel_path),
            new_content,
            schema,
            &config.validation,
        )
        .await
        .map_err(|e| {
            error!("Validation error for '{}': {:#}", rel_path, e);
            McpError::internal_error(format!("Failed to validate content: {}", e), None)
        })?;

        if !validation_result.valid {
            let data = Some(serde_json::json!({
                "field_errors": validation_result.field_errors
            }));
            return Err(McpError::invalid_params(
                format!(
                    "frontmatter validation failed for '{}': {}",
                    rel_path,
                    validation_result.errors.join("; ")
                ),
                data,
            ));
        }

        // 2. Dedup gate: on create paths, check for near-duplicate existing documents.
        //    Gate runs only when: this is a create (not edit), dedup is enabled in
        //    config, and the caller has not set force_new = true.
        if is_create && config.write.dedup_enabled && !matches!(force_new, Some(true)) {
            // Reuse the body already parsed during validation above rather than
            // re-deriving it here: that keeps the dedup query on exactly the
            // frontmatter-stripped basis the indexer embeds.
            let query_text = validated
                .as_ref()
                .map(|v| {
                    let description = v.frontmatter.get("description").and_then(|d| d.as_str());
                    build_dedup_query(&v.body, description, config.chunking.prepend_description)
                })
                .unwrap_or_default();

            if query_text.trim().is_empty() {
                warn!(
                    "Dedup gate skipped for '{}': no body text to compare",
                    rel_path
                );
            } else {
                let empty_filters = SearchFilters {
                    domain: None,
                    r#type: None,
                    tags: None,
                };
                // Detach the reranker: `dedup_threshold` is a cosine similarity,
                // and a cross-encoder relevance score is not comparable to it.
                let dedup_deps = RetrievalDeps {
                    reranker: None,
                    ..self.deps()
                };
                match retrieval::search(
                    &dedup_deps,
                    &query_text,
                    &empty_filters,
                    &dedup_search_opts(),
                )
                .await
                {
                    Ok(results) => {
                        let top = results.into_iter().next().map(|r| {
                            let path = r
                                .payload
                                .get("file_path")
                                .and_then(|v| v.as_str())
                                .map(|p| retrieval::relative_to_data(p, &self.canonical_data_path))
                                .unwrap_or_default();
                            (path, r.score)
                        });
                        if let Some((path, score)) = top.as_ref() {
                            debug!(
                                "Dedup gate for '{}': nearest '{}' at dense cosine {:.4} \
                                 (threshold {:.2})",
                                rel_path, path, score, config.write.dedup_threshold
                            );
                        }
                        if let Some(hit) = dedup_verdict(top, config.write.dedup_threshold) {
                            let threshold = config.write.dedup_threshold;
                            return Err(McpError::invalid_params(
                                format!(
                                    "A similar document already exists: '{}' \
                                     (similarity {:.2} ≥ threshold {:.2}). \
                                     Edit it with edit_document, or pass \
                                     force_new=true to create a new document anyway.",
                                    hit.file_path, hit.score, threshold
                                ),
                                Some(serde_json::json!({
                                    "duplicate_of": hit.file_path,
                                    "similarity": hit.score,
                                    "threshold": threshold,
                                })),
                            ));
                        }
                    }
                    Err(e) => {
                        warn!(
                            "Dedup search failed for '{}' (proceeding with write): {:#?}",
                            rel_path, e
                        );
                    }
                }
            }
        }

        // 3. Create parent directories and write the file
        // Validate the commit message BEFORE touching the filesystem. Rejecting it
        // afterwards would leave the file written but never committed, and the index
        // purge that follows a successful commit would never run.
        validate_commit_message(message)?;

        // Resolve fresh before creating directories too. The caller's resolution
        // happened before schema validation and a Qdrant dedup query — a wide window in
        // which a concurrent git sync could swap a component for a symlink, which would
        // otherwise let create_dir_all materialize real directories outside the KB.
        let abs_path =
            &resolve_safe_write_path(&self.canonical_data_path, rel_path).map_err(|e| {
                error!(
                    "Path check failed before creating directories for '{}': {}",
                    rel_path, e
                );
                McpError::invalid_params(format!("Invalid path: {}", e), None)
            })?;

        if let Some(parent) = abs_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                error!(
                    "Failed to create parent directories for '{}': {}",
                    abs_path.display(),
                    e
                );
                McpError::internal_error(
                    format!("Failed to create parent directories: {}", e),
                    None,
                )
            })?;
        }

        // Re-verify immediately before writing. The initial resolution could only
        // canonicalize ancestors that existed at the time, and the work between then
        // and here — schema validation, an embedding call, a Qdrant dedup query — is a
        // wide window in which a concurrent git sync could swap a path component for a
        // symlink. Checking afterwards would only report an escape that already
        // happened; the verified path is what we write to.
        let abs_path =
            &resolve_safe_write_path(&self.canonical_data_path, rel_path).map_err(|e| {
                error!("Path check failed before writing '{}': {}", rel_path, e);
                McpError::invalid_params(format!("Invalid path: {}", e), None)
            })?;

        if is_create {
            use tokio::io::AsyncWriteExt as _;
            let mut file = tokio::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(abs_path)
                .await
                .map_err(|e| {
                    if e.kind() == std::io::ErrorKind::AlreadyExists {
                        McpError::invalid_params(
                            format!(
                                "File '{}' already exists; use edit_document to modify it",
                                abs_path.display()
                            ),
                            None,
                        )
                    } else {
                        error!("Failed to create file '{}': {}", abs_path.display(), e);
                        McpError::internal_error(format!("Failed to create file: {}", e), None)
                    }
                })?;
            file.write_all(new_content.as_bytes()).await.map_err(|e| {
                error!("Failed to write file '{}': {}", abs_path.display(), e);
                McpError::internal_error(format!("Failed to write file: {}", e), None)
            })?;
        } else {
            tokio::fs::write(abs_path, new_content.as_bytes())
                .await
                .map_err(|e| {
                    error!("Failed to write file '{}': {}", abs_path.display(), e);
                    McpError::internal_error(format!("Failed to write file: {}", e), None)
                })?;
        }

        let commit_message = build_commit_message(
            message,
            &format!("docs: {} {}", default_verb, rel_path),
            operation,
        );

        // 5. Resolve git token
        let token = std::env::var(&config.source.git_token_env)
            .ok()
            .filter(|s| !s.is_empty());

        // 6. Commit and sync to remote
        let commit_outcome = git::commit_and_sync(
            config.source.git_url.as_deref(),
            &config.source.branch,
            self.canonical_data_path.to_str().unwrap_or_default(),
            token.as_deref(),
            rel_path,
            &commit_message,
            &config.write.commit_author_name,
            &config.write.commit_author_email,
        )
        .await
        .map_err(|e| {
            error!("commit_and_sync failed for '{}': {:#}", rel_path, e);
            McpError::internal_error(format!("Git commit/sync failed: {}", e), None)
        })?;

        // 7. Mark this path — and anything the rebase pulled in from other commits —
        //    dirty and return immediately. The reindex worker (src/reindex.rs) does
        //    the actual chunk/embed/upsert work out of band; this call never blocks on
        //    it, which is the whole point — embedding is far slower than an MCP
        //    client's request timeout on a large document.
        crate::reindex::mark_paths(
            std::iter::once(std::path::PathBuf::from(rel_path))
                .chain(commit_outcome.rebased_paths.iter().cloned()),
        );

        // 8. Build unified diff and return success
        let action = if is_create { "Created" } else { "Edited" };
        let summary = format!(
            "{} '{}' (commit {}). Indexing has been queued and will complete shortly.",
            action, rel_path, commit_outcome.sha
        );
        let diff = render_unified_diff(old_content, new_content, rel_path);
        let mut result_text = summary;
        if !diff.is_empty() {
            result_text = format!("{}\n\n{}", result_text, diff);
        }
        Ok(CallToolResult::success(vec![Content::text(result_text)]))
    }

    #[tool(description = "Create a new document in the knowledge base. \
        Writes the file, commits it to the git repository, and queues it for indexing \
        (indexing happens in the background; the document becomes searchable shortly \
        after this call returns, not necessarily immediately). \
        The document must not already exist — use edit_document for existing files. \
        Content must include valid YAML frontmatter. \
        Required frontmatter fields and any fixed allowed values (e.g. for type/status) are \
        listed in this server's instructions. \
        If a very similar document already exists, the create is refused and the close match is \
        reported — edit that document instead, or set force_new=true to create a new one anyway. \
        SCOPE: only for durable, long-lived reference knowledge. NEVER create a document to hold \
        session notes, intermediate analysis, task/TODO state, or scratch output — every write is \
        committed, pushed, and permanently indexed. Write transient content to a local file instead.")]
    async fn create_document(
        &self,
        Parameters(params): Parameters<CreateDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        // Resolve path: must be relative, no traversal, not already existing.
        let data_root = self.canonical_data_path.clone();
        let abs_path = resolve_safe_write_path(&data_root, &params.path)
            .map_err(|e| McpError::invalid_params(e, None))?;

        // Include-pattern guard: reject paths the indexer would not pick up.
        if !self.include_patterns.is_match(&params.path) {
            return Err(McpError::invalid_params(
                format!(
                    "path '{}' does not match any indexable include pattern \
                     (e.g. must be a markdown file under an included path)",
                    params.path
                ),
                None,
            ));
        }

        // File must not already exist for create.
        if abs_path.exists() {
            return Err(McpError::invalid_params(
                format!(
                    "File '{}' already exists. Use edit_document to modify existing files.",
                    params.path
                ),
                None,
            ));
        }

        self.write_document(
            "", // old_content: empty for new files
            &params.content,
            &params.path,
            true, // is_create
            params.message.as_deref(),
            "add",
            params.force_new,
            "create_document",
        )
        .await
    }

    #[tool(description = "Edit an existing document in the knowledge base. \
        Supports two modes:\n\
        \n\
        SURGICAL MODE — provide old_string and new_string (mutually exclusive with content):\n\
        Finds old_string in the document (must appear exactly once) and replaces it with \
        new_string. Ideal for small, targeted edits without sending the whole file. \
        old_string must be unique in the document; include more surrounding context if the \
        tool reports multiple occurrences.\n\
        \n\
        FULL-REPLACE MODE — provide content (mutually exclusive with old_string/new_string):\n\
        Replaces the entire file content with the provided content, which must include valid \
        YAML frontmatter. Required frontmatter fields and any fixed allowed values \
        (e.g. for type/status) are listed in this server's instructions.\n\
        \n\
        In both modes the result is validated, committed, and queued for indexing in the \
        background (the change becomes searchable shortly after this call returns, not \
        necessarily immediately). The path is resolved like get_document: relative to the KB \
        root, a unique basename, or absolute. The document must already exist — use \
        create_document for new files.\n\
        \n\
        SCOPE: this knowledge base holds durable reference knowledge only. NEVER append session \
        notes, task state, or other transient content to a document.")]
    async fn edit_document(
        &self,
        Parameters(params): Parameters<EditDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        // Parse and validate the edit mode (surgical vs full-replace).
        let mode = parse_edit_mode(&params).map_err(|e| McpError::invalid_params(e, None))?;

        // Resolve the path using the get_document resolver (forgiving: relative/
        // absolute/basename, canonicalized, include-pattern + containment checked).
        let raw = params.path.trim();
        if raw.is_empty() {
            return Err(McpError::invalid_params(
                "path parameter is empty".to_string(),
                None,
            ));
        }

        let canonical = match retrieval::resolve_within_data(
            raw,
            &self.canonical_data_path,
            &self.include_patterns,
        ) {
            Ok(c) => c,
            Err(retrieval::ResolveErr::NotFound) => {
                return Err(McpError::invalid_params(
                    format!(
                        "Document '{}' does not exist. Use create_document to create new files.",
                        raw
                    ),
                    None,
                ));
            }
            Err(retrieval::ResolveErr::Outside) => {
                return Err(McpError::invalid_params(
                    "File path is outside the data directory".to_string(),
                    None,
                ));
            }
            Err(retrieval::ResolveErr::NotPermitted) => {
                return Err(McpError::invalid_params(
                    "File type not permitted".to_string(),
                    None,
                ));
            }
            Err(retrieval::ResolveErr::Other(msg)) => {
                return Err(McpError::invalid_params(msg, None));
            }
        };

        // Derive the repo-relative path from the canonical absolute path.
        let rel_path = canonical
            .strip_prefix(&self.canonical_data_path)
            .unwrap_or(&canonical)
            .to_string_lossy()
            .into_owned();

        // Read the existing file content.
        let old_content = tokio::fs::read_to_string(&canonical).await.map_err(|e| {
            error!("Failed to read '{}': {}", canonical.display(), e);
            McpError::internal_error(format!("Failed to read existing file: {}", e), None)
        })?;

        // Compute new_content and operation label based on mode.
        let (new_content, operation) = match mode {
            EditMode::Full { content } => (content, "edit_document (full replace)"),
            EditMode::Surgical { old, new } => {
                let result = apply_surgical(&old_content, &old, &new).map_err(|e| {
                    // Enrich the error with the resolved relative path for context.
                    let msg = e.replace("document", &format!("'{}'", rel_path));
                    McpError::invalid_params(msg, None)
                })?;
                (result, "edit_document (surgical replace)")
            }
        };

        self.write_document(
            &old_content,
            &new_content,
            &rel_path,
            false, // is_create
            params.message.as_deref(),
            "update",
            None, // no dedup gate for edit
            operation,
        )
        .await
    }

    #[tool(description = "Delete a document from the knowledge base. \
        Removes the file from disk, commits the deletion to git with provenance trailers \
        (just like create_document/edit_document), pushes the commit, and queues removal of \
        the document's vectors from the Qdrant search index and its state-DB row (this \
        happens in the background; the document stops being searchable shortly after this \
        call returns, not necessarily immediately). \
        The path resolves like get_document: relative to the KB root \
        (e.g. 'sysadmin/guide.md'), a unique basename, or absolute. \
        The document must already exist — use search to find the correct path. \
        Returns a summary line with the commit SHA and a unified diff of the removed content.")]
    async fn delete_document(
        &self,
        Parameters(params): Parameters<DeleteDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        let config = self.config();

        let raw = params.path.trim();
        if raw.is_empty() {
            return Err(McpError::invalid_params(
                "path parameter is empty".to_string(),
                None,
            ));
        }

        // 1. Resolve the path (must already exist on disk).
        let canonical = match retrieval::resolve_within_data(
            raw,
            &self.canonical_data_path,
            &self.include_patterns,
        ) {
            Ok(c) => c,
            Err(retrieval::ResolveErr::NotFound) => {
                return Err(McpError::invalid_params(
                    format!("document does not exist: '{}'", raw),
                    None,
                ));
            }
            Err(retrieval::ResolveErr::Outside) => {
                return Err(McpError::invalid_params(
                    "File path is outside the data directory".to_string(),
                    None,
                ));
            }
            Err(retrieval::ResolveErr::NotPermitted) => {
                return Err(McpError::invalid_params(
                    "File type not permitted".to_string(),
                    None,
                ));
            }
            Err(retrieval::ResolveErr::Other(msg)) => {
                return Err(McpError::invalid_params(msg, None));
            }
        };

        // Derive repo-relative path (used for git staging, commit messages, index purge).
        let rel_path = canonical
            .strip_prefix(&self.canonical_data_path)
            .unwrap_or(&canonical)
            .to_string_lossy()
            .into_owned();

        // Validate the commit message BEFORE deleting anything. Rejecting it after the
        // removal would leave the file gone from disk but never committed, with the
        // Qdrant and state-DB purge — which only runs after a successful commit —
        // skipped too, so search would keep returning a document that no longer exists.
        validate_commit_message(params.message.as_deref())?;

        // 2. Read file content before removal (used for diff output).
        let old_content = tokio::fs::read_to_string(&canonical).await.map_err(|e| {
            error!("Failed to read '{}': {}", canonical.display(), e);
            McpError::internal_error(format!("Failed to read file before deletion: {}", e), None)
        })?;

        // 3. Remove the file from disk.
        tokio::fs::remove_file(&canonical).await.map_err(|e| {
            error!("Failed to remove '{}': {}", canonical.display(), e);
            McpError::internal_error(format!("Failed to remove file: {}", e), None)
        })?;

        // 4. Commit + push the deletion.
        let commit_message = build_commit_message(
            params.message.as_deref(),
            &format!("docs: delete {}", rel_path),
            "delete_document",
        );

        let token = std::env::var(&config.source.git_token_env)
            .ok()
            .filter(|s| !s.is_empty());

        let commit_outcome = git::commit_and_sync(
            config.source.git_url.as_deref(),
            &config.source.branch,
            self.canonical_data_path.to_str().unwrap_or_default(),
            token.as_deref(),
            &rel_path,
            &commit_message,
            &config.write.commit_author_name,
            &config.write.commit_author_email,
        )
        .await
        .map_err(|e| {
            error!(
                "commit_and_sync failed for '{}' after file was removed from disk: {:#}",
                rel_path, e
            );
            McpError::internal_error(
                format!(
                    "File was removed from disk but git commit/sync failed: {}. \
                     The file is gone locally but the git repo may be out of sync.",
                    e
                ),
                None,
            )
        })?;

        // 5. Mark this path — and anything the rebase pulled in — dirty and return
        //    immediately. The worker's scoped indexer purges a path's Qdrant points
        //    and state rows itself once it re-checks and finds the file gone (the
        //    missing-file branch of `ingest::index_paths`), so there is no separate
        //    purge to do here anymore — this is "one reindex path" applied to deletes
        //    too, not a special case.
        crate::reindex::mark_paths(
            std::iter::once(std::path::PathBuf::from(&rel_path))
                .chain(commit_outcome.rebased_paths.iter().cloned()),
        );

        // 6. Return success with summary + diff of removed content.
        let summary = format!(
            "Deleted '{}' (commit {}). Index cleanup has been queued and will complete shortly.",
            rel_path, commit_outcome.sha
        );
        let diff = render_unified_diff(&old_content, "", &rel_path);
        let mut result_text = summary;
        if !diff.is_empty() {
            result_text = format!("{}\n\n{}", result_text, diff);
        }
        Ok(CallToolResult::success(vec![Content::text(result_text)]))
    }
}

/// Default instructions used when no custom instructions are configured.
/// The server appends discovered filter values ("Available ...") and a full
/// write-authoring section after this base, so keep this short.
pub const DEFAULT_INSTRUCTIONS: &str = "Knowledge base server. \
Read with `search` (natural-language query; optional domain/type/tags filters) \
and `get_document`. \
Write with `create_document`, `edit_document`, and `delete_document`.";

#[tool_handler]
impl ServerHandler for KbSearchServer {
    fn get_info(&self) -> ServerInfo {
        let instructions = self
            .instructions
            .read()
            .unwrap_or_else(|poisoned| {
                warn!("Instructions RwLock poisoned on read; using last value");
                poisoned.into_inner()
            })
            .clone();
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::from_build_env())
            .with_instructions(instructions)
    }
}

/// Builds a minimal `ResolvedConfig` for tests. Shared with `server.rs`'s test
/// module, which needs the same handler-construction pattern for the MCP
/// service without pulling in a real config file.
#[cfg(test)]
pub(crate) fn make_test_resolved_config(data_path: &std::path::Path) -> Arc<ResolvedConfig> {
    Arc::new(ResolvedConfig {
        source: crate::config::ResolvedSourceConfig {
            git_url: None,
            branch: "master".into(),
            data_path: Some(data_path.to_string_lossy().into_owned()),
            git_token_env: "GIT_PULL_TOKEN".into(),
        },
        indexing: crate::config::IndexingConfig::default(),
        frontmatter: crate::config::FrontmatterConfig::default(),
        chunking: crate::config::ChunkingConfig::default(),
        embedding: crate::config::ResolvedEmbeddingConfig {
            base_url: "http://localhost:8080/v1".into(),
            model: "test".into(),
            api_key: None,
            vector_size: 768,
            batch_size: 32,
            request_timeout_secs: 60,
            batch_concurrency: 4,
        },
        qdrant: crate::config::ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: "test".into(),
        },
        validation: crate::config::ValidationConfig::default(),
        webhook: crate::config::WebhookConfig::default(),
        mcp: crate::config::ResolvedMcpConfig::default(),
        rate_limit: crate::config::RateLimitConfig::default(),
        write: crate::config::WriteConfig::default(),
        search: crate::config::SearchConfig::default(),
        reranking: None,
        provenance: Default::default(),
    })
}

/// An empty `SharedSchemaCache`, for tests that exercise a `KbSearchServer` but do
/// not care about schema content (e.g. instructions plumbing, path validation).
/// Tests that DO care build a real one from a temp dir's `.kb-schema.yaml` files —
/// see `make_write_test_server` below.
#[cfg(test)]
pub(crate) fn empty_test_schema_cache() -> crate::schema::SharedSchemaCache {
    Arc::new(RwLock::new(Arc::new(crate::schema::SchemaCache::default())))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- list_documents filter parsing ---

    fn filters_from(json: serde_json::Value) -> ListDocumentsParams {
        ListDocumentsParams {
            filters: Some(json.as_object().unwrap().clone()),
            ..Default::default()
        }
    }

    fn parsed_filters(json: serde_json::Value) -> Vec<(String, FieldFilter)> {
        build_document_query(&filters_from(json)).unwrap().filters
    }

    #[test]
    fn scalar_filter_becomes_equality() {
        let filters = parsed_filters(serde_json::json!({ "type": "guide" }));
        assert_eq!(
            filters,
            vec![("type".to_string(), FieldFilter::AnyOf(vec!["guide".into()]))]
        );
    }

    #[test]
    fn array_filter_becomes_any_of() {
        let filters = parsed_filters(serde_json::json!({ "tags": ["recipe", "dinner"] }));
        assert_eq!(
            filters,
            vec![(
                "tags".to_string(),
                FieldFilter::AnyOf(vec!["recipe".into(), "dinner".into()])
            )]
        );
    }

    #[test]
    fn all_of_object_becomes_all_of() {
        let filters =
            parsed_filters(serde_json::json!({ "tags": { "all_of": ["recipe", "dinner"] } }));
        assert_eq!(
            filters,
            vec![(
                "tags".to_string(),
                FieldFilter::AllOf(vec!["recipe".into(), "dinner".into()])
            )]
        );
    }

    #[test]
    fn numeric_operators_become_a_range() {
        let filters =
            parsed_filters(serde_json::json!({ "planning.prep_minutes": { "gte": 10, "lt": 30 } }));
        assert_eq!(
            filters,
            vec![(
                "planning.prep_minutes".to_string(),
                FieldFilter::Range {
                    gte: Some(10.0),
                    lte: None,
                    gt: None,
                    lt: Some(30.0),
                }
            )]
        );
    }

    #[test]
    fn booleans_and_numbers_canonicalize_like_the_write_path() {
        // The property that makes {"planning.needs_recipe": false} match stored rows.
        let filters = parsed_filters(
            serde_json::json!({ "planning.needs_recipe": false, "planning.rating": 5 }),
        );
        assert!(filters.contains(&(
            "planning.needs_recipe".to_string(),
            FieldFilter::AnyOf(vec!["false".into()])
        )));
        assert!(filters.contains(&(
            "planning.rating".to_string(),
            FieldFilter::AnyOf(vec!["5".into()])
        )));
    }

    #[test]
    fn unknown_filter_operator_is_rejected_with_guidance() {
        let err = build_document_query(&filters_from(
            serde_json::json!({ "planning.prep_minutes": { "lte_": 30 } }),
        ))
        .unwrap_err();
        let msg = format!("{:?}", err);
        assert!(msg.contains("unknown operator"), "got: {msg}");
        assert!(msg.contains("all_of"), "error should list valid operators");
    }

    #[test]
    fn empty_object_filter_is_rejected() {
        assert!(
            build_document_query(&filters_from(serde_json::json!({ "tags": {} }))).is_err(),
            "an operator-less object would otherwise match everything"
        );
    }

    #[test]
    fn null_filter_is_rejected() {
        assert!(build_document_query(&filters_from(serde_json::json!({ "tags": null }))).is_err());
    }

    #[test]
    fn non_numeric_range_bound_is_rejected() {
        let err = build_document_query(&filters_from(
            serde_json::json!({ "planning.prep_minutes": { "lt": "thirty" } }),
        ))
        .unwrap_err();
        assert!(format!("{:?}", err).contains("must be a number"));
    }

    #[test]
    fn nested_object_filter_value_is_rejected() {
        assert!(
            build_document_query(&filters_from(
                serde_json::json!({ "tags": [{ "nested": true }] })
            ))
            .is_err()
        );
    }

    #[test]
    fn list_defaults_are_applied_when_nothing_is_supplied() {
        let query = build_document_query(&ListDocumentsParams::default()).unwrap();
        assert_eq!(query.limit, DEFAULT_LIST_LIMIT);
        assert_eq!(query.offset, 0);
        assert_eq!(query.order_by, OrderBy::Path);
        assert!(!query.order_desc);
        assert!(query.filters.is_empty());
        assert!(query.path_prefix.is_none());
        assert!(query.fields.is_none());
    }

    #[test]
    fn list_limit_is_clamped_to_the_cap() {
        let query = build_document_query(&ListDocumentsParams {
            limit: Some(999_999),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(query.limit, MAX_LIST_LIMIT);

        let query = build_document_query(&ListDocumentsParams {
            limit: Some(0),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(query.limit, 1, "a zero-size page would return nothing");
    }

    #[test]
    fn invalid_order_by_is_rejected() {
        let err = build_document_query(&ListDocumentsParams {
            order_by: Some("file_path; DROP TABLE documents".into()),
            ..Default::default()
        })
        .unwrap_err();
        assert!(format!("{:?}", err).contains("unknown order_by"));
    }

    #[test]
    fn filters_are_sorted_for_stable_sql() {
        let query = build_document_query(&filters_from(
            serde_json::json!({ "zeta": "a", "alpha": "b", "mid": "c" }),
        ))
        .unwrap();
        let names: Vec<&str> = query.filters.iter().map(|(f, _)| f.as_str()).collect();
        assert_eq!(names, vec!["alpha", "mid", "zeta"]);
    }

    #[test]
    fn too_many_filters_is_rejected() {
        let mut map = serde_json::Map::new();
        for i in 0..(MAX_LIST_FILTERS + 1) {
            map.insert(format!("field{i}"), serde_json::json!("x"));
        }
        let err = build_document_query(&ListDocumentsParams {
            filters: Some(map),
            ..Default::default()
        })
        .unwrap_err();
        assert!(format!("{:?}", err).contains("too many filters"));
    }

    #[test]
    fn overlong_path_prefix_is_rejected() {
        let err = build_document_query(&ListDocumentsParams {
            path_prefix: Some("a".repeat(MAX_FILTER_STR_LEN + 1)),
            ..Default::default()
        })
        .unwrap_err();
        assert!(format!("{:?}", err).contains("path_prefix too long"));
    }

    // --- schema tool helpers ---

    #[test]
    fn scope_paths_normalize_to_relative_dirs() {
        assert_eq!(normalize_scope_path("").unwrap(), std::path::PathBuf::new());
        assert_eq!(
            normalize_scope_path("/food/recipes/").unwrap(),
            std::path::PathBuf::from("food/recipes")
        );
        assert_eq!(
            normalize_scope_path("./food/recipes").unwrap(),
            std::path::PathBuf::from("food/recipes")
        );
    }

    #[test]
    fn scope_paths_reject_traversal() {
        // A schema written outside the KB governs nothing and could clobber other files.
        assert!(normalize_scope_path("../../etc").is_err());
        assert!(normalize_scope_path("food/../../etc").is_err());
    }

    #[test]
    fn scope_paths_reject_overlong_input() {
        assert!(normalize_scope_path(&"a".repeat(MAX_PATH_LEN + 1)).is_err());
    }

    fn update_params(operation: &str, field: &str) -> UpdateSchemaParams {
        UpdateSchemaParams {
            path: None,
            operation: operation.into(),
            field: field.into(),
            values: None,
            definition: None,
            dry_run: None,
            force: None,
        }
    }

    #[test]
    fn add_values_requires_a_non_empty_list() {
        let mut params = update_params("add_values", "tags");
        assert!(build_schema_edit(&params).is_err());

        params.values = Some(vec![]);
        assert!(build_schema_edit(&params).is_err());

        params.values = Some(vec!["recipe".into()]);
        assert!(build_schema_edit(&params).is_ok());
    }

    #[test]
    fn unknown_operation_lists_the_valid_ones() {
        let err = build_schema_edit(&update_params("delete_everything", "tags")).unwrap_err();
        let msg = format!("{:?}", err);
        assert!(msg.contains("unknown operation"));
        assert!(msg.contains("add_values"));
    }

    #[test]
    fn set_field_parses_a_definition() {
        let mut params = update_params("set_field", "planning.prep_minutes");
        params.definition = Some(serde_json::json!({ "type": "integer", "indexed": true }));

        match build_schema_edit(&params).unwrap() {
            crate::schema::SchemaEdit::SetField { field, definition } => {
                assert_eq!(field, "planning.prep_minutes");
                assert_eq!(definition.ty, Some(crate::schema::FieldType::Integer));
                assert!(definition.indexed);
            }
            other => panic!("expected SetField, got {other:?}"),
        }
    }

    #[test]
    fn set_field_rejects_an_unknown_key() {
        let mut params = update_params("set_field", "tags");
        params.definition = Some(serde_json::json!({ "typ": "integer" }));
        assert!(
            build_schema_edit(&params).is_err(),
            "a typo'd key must not be silently dropped"
        );
    }

    #[test]
    fn schema_edits_round_trip_through_yaml() {
        // The property that matters: whatever update_schema writes must parse back,
        // because an unparseable schema freezes its whole subtree.
        let mut file = crate::schema::SchemaFile::default();
        file.apply(&crate::schema::SchemaEdit::AddValues {
            field: "tags".into(),
            values: vec!["recipe".into(), "dinner".into()],
        })
        .unwrap();
        file.apply(&crate::schema::SchemaEdit::SetField {
            field: "planning.prep_minutes".into(),
            definition: Box::new(
                serde_json::from_value(serde_json::json!({ "type": "integer", "indexed": true }))
                    .unwrap(),
            ),
        })
        .unwrap();

        let yaml = file.to_yaml().unwrap();
        let reparsed: crate::schema::SchemaFile = serde_yaml_ng::from_str(&yaml).unwrap();

        assert_eq!(
            reparsed.fields["tags"].values,
            Some(vec!["dinner".to_string(), "recipe".to_string()]),
            "values are sorted for a stable diff"
        );
        assert_eq!(
            reparsed.fields["planning.prep_minutes"].ty,
            Some(crate::schema::FieldType::Integer)
        );
    }

    #[test]
    fn adding_an_existing_value_is_reported_as_a_no_op() {
        let mut file = crate::schema::SchemaFile::default();
        let edit = crate::schema::SchemaEdit::AddValues {
            field: "tags".into(),
            values: vec!["recipe".into()],
        };
        file.apply(&edit).unwrap();
        let summary = file.apply(&edit).unwrap();

        assert!(summary.contains("already permitted"), "got: {summary}");
        assert_eq!(file.fields["tags"].values, Some(vec!["recipe".to_string()]));
    }

    #[test]
    fn removing_an_undeclared_field_is_an_error() {
        let mut file = crate::schema::SchemaFile::default();
        assert!(
            file.apply(&crate::schema::SchemaEdit::RemoveField {
                field: "nope".into()
            })
            .is_err()
        );
    }

    #[test]
    fn casualty_rendering_summarizes_beyond_the_cap() {
        let many: Vec<serde_json::Value> = (0..MAX_REPORTED_CASUALTIES + 5)
            .map(|i| serde_json::json!({ "path": format!("d{i}.md"), "reason": "missing" }))
            .collect();
        let rendered = render_casualties(&many);

        assert!(rendered.contains("d0.md"));
        assert!(rendered.contains("and 5 more"));
    }

    // --- schema tool handlers (no Qdrant, no embeddings, no git required) ---

    /// A server whose state DB lives under the temp KB, so metadata-backed tools work.
    fn schema_tool_server(tmp: &tempfile::TempDir) -> KbSearchServer {
        let config = make_test_resolved_config(tmp.path());
        make_write_test_server(tmp, &["**/*.md".to_string()], config)
    }

    async fn seed_document(
        server: &KbSearchServer,
        rel_path: &str,
        frontmatter: serde_json::Value,
    ) {
        let map = match frontmatter {
            serde_json::Value::Object(m) => {
                m.into_iter().collect::<std::collections::HashMap<_, _>>()
            }
            _ => panic!("frontmatter fixture must be an object"),
        };
        server
            .state_db()
            .await
            .unwrap()
            .upsert_document_metadata(rel_path, &map, 100, "hash", 1)
            .await
            .unwrap();
    }

    fn write_schema_file(tmp: &tempfile::TempDir, dir: &str, yaml: &str) {
        let target = tmp.path().join(dir);
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join(crate::schema::SCHEMA_FILE_NAME), yaml).unwrap();
    }

    #[tokio::test]
    async fn get_schema_reports_merged_fields_with_provenance() {
        let tmp = tempfile::tempdir().unwrap();
        write_schema_file(&tmp, "", "fields:\n  title:\n    required: true\n");
        write_schema_file(
            &tmp,
            "food/recipes",
            "fields:\n  prep:\n    type: integer\n    indexed: true\n",
        );
        let server = schema_tool_server(&tmp);

        let result = server
            .get_schema(Parameters(GetSchemaParams {
                path: Some("food/recipes".into()),
                ..Default::default()
            }))
            .await
            .unwrap();

        let structured = result.structured_content.unwrap();
        let fields = structured["fields"].as_array().unwrap();
        let names: Vec<&str> = fields
            .iter()
            .map(|f| f["field"].as_str().unwrap())
            .collect();

        assert!(names.contains(&"title"), "inherited from the root scope");
        assert!(names.contains(&"prep"), "declared in this scope");
        assert_eq!(structured["frozen"], serde_json::json!(false));

        let prep = fields.iter().find(|f| f["field"] == "prep").unwrap();
        assert_eq!(prep["type"], serde_json::json!("integer"));
        assert!(
            prep["declared_in"]
                .as_str()
                .unwrap()
                .contains("food/recipes"),
            "provenance points at the declaring file"
        );
    }

    #[tokio::test]
    async fn schema_tools_accept_a_leading_slash_as_the_kb_root() {
        let tmp = tempfile::tempdir().unwrap();
        write_schema_file(
            &tmp,
            "food/recipes",
            "fields:\n  prep:\n    type: integer\n",
        );
        let server = schema_tool_server(&tmp);

        let with_slash = server
            .get_schema(Parameters(GetSchemaParams {
                path: Some("/food/recipes".into()),
                ..Default::default()
            }))
            .await
            .unwrap();
        let without = server
            .get_schema(Parameters(GetSchemaParams {
                path: Some("food/recipes".into()),
                ..Default::default()
            }))
            .await
            .unwrap();

        assert_eq!(
            with_slash.structured_content, without.structured_content,
            "callers cannot know where the KB lives, so `/x` and `x` must agree"
        );
    }

    #[tokio::test]
    async fn get_schema_resolves_a_partial_directory() {
        let tmp = tempfile::tempdir().unwrap();
        write_schema_file(
            &tmp,
            "food/recipes",
            "fields:\n  prep:\n    type: integer\n",
        );
        let server = schema_tool_server(&tmp);

        let result = server
            .get_schema(Parameters(GetSchemaParams {
                path: Some("recipes".into()),
                ..Default::default()
            }))
            .await
            .expect("a unique trailing match resolves");

        let fields = result.structured_content.unwrap()["fields"]
            .as_array()
            .unwrap()
            .clone();
        assert!(
            fields.iter().any(|f| f["field"] == "prep"),
            "should have resolved to food/recipes, got: {fields:?}"
        );
    }

    #[tokio::test]
    async fn an_ambiguous_partial_scope_reports_the_candidates() {
        let tmp = tempfile::tempdir().unwrap();
        write_schema_file(&tmp, "food/recipes", "fields:\n  a:\n    type: text\n");
        write_schema_file(&tmp, "archive/recipes", "fields:\n  b:\n    type: text\n");
        let server = schema_tool_server(&tmp);

        let err = server
            .get_schema(Parameters(GetSchemaParams {
                path: Some("recipes".into()),
                ..Default::default()
            }))
            .await
            .expect_err("two scopes end in recipes; guessing would be wrong");

        let msg = format!("{:?}", err);
        assert!(msg.contains("matches 2 scopes"), "got: {msg}");
        assert!(msg.contains("food/recipes"));
        assert!(msg.contains("archive/recipes"));
    }

    #[tokio::test]
    async fn update_schema_can_still_create_a_scope_that_does_not_exist_yet() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);

        let _ = server
            .update_schema(Parameters(UpdateSchemaParams {
                path: Some("/brand/new".into()),
                operation: "add_values".into(),
                field: "tags".into(),
                values: Some(vec!["x".into()]),
                definition: None,
                dry_run: None,
                force: None,
            }))
            .await;

        assert!(
            tmp.path()
                .join("brand/new")
                .join(crate::schema::SCHEMA_FILE_NAME)
                .exists(),
            "an unmatched path must be taken literally so new scopes can be created"
        );
    }

    #[tokio::test]
    async fn get_schema_surfaces_a_frozen_scope() {
        let tmp = tempfile::tempdir().unwrap();
        write_schema_file(&tmp, "broken", "fields: [not a mapping\n");
        let server = schema_tool_server(&tmp);

        let result = server
            .get_schema(Parameters(GetSchemaParams {
                path: Some("broken".into()),
                ..Default::default()
            }))
            .await
            .unwrap();

        let structured = result.structured_content.unwrap();
        assert_eq!(structured["frozen"], serde_json::json!(true));
        assert!(structured["frozen_reason"].is_string());
    }

    #[tokio::test]
    async fn get_schema_values_only_filters_to_vocabularies() {
        let tmp = tempfile::tempdir().unwrap();
        write_schema_file(
            &tmp,
            "",
            "fields:\n  title:\n    required: true\n  status:\n    type: enum\n    values: [active]\n",
        );
        let server = schema_tool_server(&tmp);

        let result = server
            .get_schema(Parameters(GetSchemaParams {
                values_only: Some(true),
                ..Default::default()
            }))
            .await
            .unwrap();

        let fields = result.structured_content.unwrap()["fields"]
            .as_array()
            .unwrap()
            .clone();
        let names: Vec<&str> = fields
            .iter()
            .map(|f| f["field"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["status"], "only closed-set fields are reported");
    }

    #[tokio::test]
    async fn update_schema_dry_run_writes_nothing_and_reports_casualties() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);
        seed_document(&server, "notes/a.md", serde_json::json!({ "title": "A" })).await;

        let result = server
            .update_schema(Parameters(UpdateSchemaParams {
                path: Some("notes".into()),
                operation: "set_field".into(),
                field: "status".into(),
                values: None,
                definition: Some(serde_json::json!({ "required": true })),
                dry_run: Some(true),
                force: None,
            }))
            .await
            .unwrap();

        let structured = result.structured_content.unwrap();
        assert_eq!(structured["dry_run"], serde_json::json!(true));
        assert_eq!(
            structured["would_invalidate"].as_array().unwrap().len(),
            1,
            "the seeded document has no status and would fail the new rule"
        );
        assert!(
            !tmp.path()
                .join("notes")
                .join(crate::schema::SCHEMA_FILE_NAME)
                .exists(),
            "a dry run must not touch the filesystem"
        );
    }

    #[tokio::test]
    async fn update_schema_refuses_a_breaking_change_without_force() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);
        seed_document(&server, "notes/a.md", serde_json::json!({ "title": "A" })).await;

        let err = server
            .update_schema(Parameters(UpdateSchemaParams {
                path: Some("notes".into()),
                operation: "set_field".into(),
                field: "status".into(),
                values: None,
                definition: Some(serde_json::json!({ "required": true })),
                dry_run: None,
                force: None,
            }))
            .await
            .expect_err("must refuse rather than silently invalidate documents");

        let msg = format!("{:?}", err);
        assert!(msg.contains("Refusing to apply"), "got: {msg}");
        assert!(
            !tmp.path()
                .join("notes")
                .join(crate::schema::SCHEMA_FILE_NAME)
                .exists(),
            "a refused change must leave the filesystem untouched"
        );
    }

    #[tokio::test]
    async fn update_schema_accepts_a_change_that_breaks_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);
        seed_document(
            &server,
            "notes/a.md",
            serde_json::json!({ "title": "A", "status": "active" }),
        )
        .await;

        // Reaches the git step, which fails in this harness — but only AFTER the file
        // has been written, which is what this test pins down.
        let _ = server
            .update_schema(Parameters(UpdateSchemaParams {
                path: Some("notes".into()),
                operation: "add_values".into(),
                field: "status".into(),
                values: Some(vec!["active".into(), "draft".into()]),
                definition: None,
                dry_run: None,
                force: None,
            }))
            .await;

        let written = tmp
            .path()
            .join("notes")
            .join(crate::schema::SCHEMA_FILE_NAME);
        assert!(
            written.exists(),
            "a non-breaking change must be written before the git step"
        );
        let yaml = std::fs::read_to_string(&written).unwrap();
        let reparsed: crate::schema::SchemaFile = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(
            reparsed.fields["status"].values,
            Some(vec!["active".to_string(), "draft".to_string()])
        );
    }

    #[tokio::test]
    async fn update_schema_dry_run_ignores_documents_a_deeper_scope_governs() {
        // A descendant scope that redefines the edited field shadows the parent, so its
        // documents are unaffected and must not be reported as casualties.
        let tmp = tempfile::tempdir().unwrap();
        write_schema_file(
            &tmp,
            "notes/archive",
            "fields:\n  status:\n    type: enum\n    values: [archived]\n",
        );
        let server = schema_tool_server(&tmp);
        seed_document(
            &server,
            "notes/archive/old.md",
            serde_json::json!({ "title": "Old", "status": "archived" }),
        )
        .await;

        let result = server
            .update_schema(Parameters(UpdateSchemaParams {
                path: Some("notes".into()),
                operation: "set_field".into(),
                field: "status".into(),
                values: Some(vec!["active".into()]),
                definition: Some(
                    serde_json::json!({ "type": "enum", "values": ["active"], "required": true }),
                ),
                dry_run: Some(true),
                force: None,
            }))
            .await
            .unwrap();

        let casualties = result.structured_content.unwrap()["would_invalidate"]
            .as_array()
            .unwrap()
            .clone();
        assert!(
            casualties.is_empty(),
            "notes/archive/ has its own status rule and is unaffected, got: {casualties:?}"
        );
    }

    #[tokio::test]
    async fn update_schema_force_applies_despite_casualties() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);
        seed_document(&server, "notes/a.md", serde_json::json!({ "title": "A" })).await;

        // Reaches the git step and fails there, but only after writing — which is the
        // point: force must not be blocked by the casualty check.
        let _ = server
            .update_schema(Parameters(UpdateSchemaParams {
                path: Some("notes".into()),
                operation: "set_field".into(),
                field: "status".into(),
                values: None,
                definition: Some(serde_json::json!({ "required": true })),
                dry_run: None,
                force: Some(true),
            }))
            .await;

        assert!(
            tmp.path()
                .join("notes")
                .join(crate::schema::SCHEMA_FILE_NAME)
                .exists(),
            "force must write the schema even though a document would fail it"
        );
    }

    #[tokio::test]
    async fn update_schema_rejects_a_self_contradictory_definition() {
        // Parses fine, but declaring a scalar type alongside nested children freezes
        // the whole subtree at the next index run — long after this call reported
        // success. It must be caught here instead.
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);

        let err = server
            .update_schema(Parameters(UpdateSchemaParams {
                path: Some("notes".into()),
                operation: "set_field".into(),
                field: "planning".into(),
                values: None,
                definition: Some(serde_json::json!({
                    "type": "integer",
                    "fields": { "prep": { "type": "integer" } }
                })),
                dry_run: None,
                force: None,
            }))
            .await
            .expect_err("a field cannot be both a value and a container");

        assert!(format!("{:?}", err).contains("not both"));
        assert!(
            !tmp.path()
                .join("notes")
                .join(crate::schema::SCHEMA_FILE_NAME)
                .exists()
        );
    }

    #[tokio::test]
    async fn update_schema_rejects_a_path_escaping_the_knowledge_base() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);

        let err = server
            .update_schema(Parameters(UpdateSchemaParams {
                path: Some("../escape".into()),
                operation: "add_values".into(),
                field: "tags".into(),
                values: Some(vec!["x".into()]),
                definition: None,
                dry_run: None,
                force: None,
            }))
            .await
            .expect_err("traversal must be rejected");

        assert!(format!("{:?}", err).contains(".."));
    }

    #[tokio::test]
    async fn list_documents_reports_total_and_truncation() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);
        for i in 0..5 {
            seed_document(
                &server,
                &format!("notes/{i}.md"),
                serde_json::json!({ "title": format!("Doc {i}"), "tags": ["note"] }),
            )
            .await;
        }

        let result = server
            .list_documents(Parameters(ListDocumentsParams {
                limit: Some(2),
                ..Default::default()
            }))
            .await
            .unwrap();

        let structured = result.structured_content.unwrap();
        assert_eq!(structured["total"], serde_json::json!(5));
        assert_eq!(structured["returned"], serde_json::json!(2));
        assert_eq!(
            structured["has_more"],
            serde_json::json!(true),
            "truncation must never be silent"
        );
    }

    #[tokio::test]
    async fn list_documents_filters_through_the_tool_surface() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);
        seed_document(
            &server,
            "food/chili.md",
            serde_json::json!({ "title": "Chili", "tags": ["recipe"], "prep": 20 }),
        )
        .await;
        seed_document(
            &server,
            "food/stew.md",
            serde_json::json!({ "title": "Stew", "tags": ["recipe"], "prep": 90 }),
        )
        .await;

        let mut filters = serde_json::Map::new();
        filters.insert("prep".into(), serde_json::json!({ "lt": 30 }));
        let result = server
            .list_documents(Parameters(ListDocumentsParams {
                filters: Some(filters),
                ..Default::default()
            }))
            .await
            .unwrap();

        let structured = result.structured_content.unwrap();
        assert_eq!(structured["total"], serde_json::json!(1));
        assert_eq!(
            structured["documents"][0]["file_path"],
            serde_json::json!("food/chili.md")
        );
    }

    #[tokio::test]
    async fn list_documents_reports_an_empty_result_clearly() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);

        let result = server
            .list_documents(Parameters(ListDocumentsParams::default()))
            .await
            .unwrap();

        assert_eq!(
            result.structured_content.unwrap()["total"],
            serde_json::json!(0)
        );
    }

    #[test]
    fn compound_filters_are_rejected_rather_than_silently_narrowed() {
        // Honoring only one half of a mixed filter returns a broader set than asked for.
        let err = build_document_query(&filters_from(
            serde_json::json!({ "tags": { "all_of": ["a"], "gte": 5 } }),
        ))
        .unwrap_err();
        assert!(format!("{:?}", err).contains("cannot combine"));

        let err = build_document_query(&filters_from(
            serde_json::json!({ "tags": { "any_of": ["a"], "all_of": ["b"] } }),
        ))
        .unwrap_err();
        assert!(format!("{:?}", err).contains("not both"));
    }

    #[test]
    fn oversized_filter_value_lists_are_rejected() {
        let many: Vec<serde_json::Value> = (0..MAX_FILTER_VALUES + 1)
            .map(|i| serde_json::json!(format!("v{i}")))
            .collect();
        let err = build_document_query(&filters_from(
            serde_json::json!({ "tags": serde_json::Value::Array(many) }),
        ))
        .unwrap_err();
        assert!(format!("{:?}", err).contains("too many values"));
    }

    #[test]
    fn oversized_schema_edits_are_rejected() {
        let mut params = update_params("add_values", "tags");
        params.values = Some(
            (0..MAX_SCHEMA_VALUES + 1)
                .map(|i| format!("v{i}"))
                .collect(),
        );
        assert!(build_schema_edit(&params).is_err());

        let mut params = update_params("add_values", "tags");
        params.values = Some(vec!["x".repeat(MAX_FILTER_STR_LEN + 1)]);
        assert!(build_schema_edit(&params).is_err());
    }

    // --- build_commit_message tests ---

    #[test]
    fn commit_message_create_document_trailer() {
        let msg = build_commit_message(None, "docs: add notes/guide.md", "create_document");
        assert!(
            msg.contains("Tool: md-kb-rag"),
            "should contain Tool trailer: {msg}"
        );
        assert!(
            msg.contains("Operation: create_document"),
            "should contain Operation trailer: {msg}"
        );
        assert!(
            msg.starts_with("docs: add notes/guide.md"),
            "should start with default subject: {msg}"
        );
    }

    #[test]
    fn commit_message_edit_surgical_trailer() {
        let msg = build_commit_message(
            None,
            "docs: update notes/guide.md",
            "edit_document (surgical replace)",
        );
        assert!(
            msg.contains("Operation: edit_document (surgical replace)"),
            "should contain surgical operation label: {msg}"
        );
    }

    #[test]
    fn commit_message_edit_full_replace_trailer() {
        let msg = build_commit_message(
            None,
            "docs: update notes/guide.md",
            "edit_document (full replace)",
        );
        assert!(
            msg.contains("Operation: edit_document (full replace)"),
            "should contain full replace operation label: {msg}"
        );
    }

    #[test]
    fn commit_message_user_subject_overrides_default() {
        let msg = build_commit_message(
            Some("fix: correct typo in introduction"),
            "docs: update notes/guide.md",
            "edit_document (surgical replace)",
        );
        assert!(
            msg.starts_with("fix: correct typo in introduction"),
            "user subject should take precedence: {msg}"
        );
        assert!(
            msg.contains("Operation: edit_document (surgical replace)"),
            "trailer should still be appended: {msg}"
        );
    }

    #[test]
    fn commit_message_trailer_separated_by_blank_line() {
        let msg = build_commit_message(None, "docs: add test.md", "create_document");
        // Git requires a blank line between subject and trailer block
        assert!(
            msg.contains("\n\nTool: md-kb-rag"),
            "blank line must precede trailer block: {msg}"
        );
    }

    #[test]
    fn default_limit_is_ten() {
        assert_eq!(resolve_limit(None), 10);
    }

    #[test]
    fn requested_limit_within_max_is_preserved() {
        assert_eq!(resolve_limit(Some(25)), 25);
    }

    #[test]
    fn requested_limit_above_max_is_clamped() {
        assert_eq!(resolve_limit(Some(1_000_000)), MAX_SEARCH_LIMIT);
    }

    #[test]
    fn zero_limit_is_passed_through() {
        assert_eq!(resolve_limit(Some(0)), 0);
    }

    #[test]
    fn ellipsis_uses_char_count_not_byte_len() {
        // 800 chars of a 2-byte character = 1600 bytes
        let text: String = "é".repeat(801);
        assert!(text.len() > 800, "byte len should exceed 800");
        assert!(text.chars().count() > 800, "char count should exceed 800");
        // If we used .len() on a 800-char string it would wrongly trigger ellipsis
        let short: String = "é".repeat(800);
        assert!(
            short.len() > 800,
            "byte len of 800 2-byte chars exceeds 800"
        );
        assert_eq!(short.chars().count(), 800, "char count is exactly 800");
    }

    #[test]
    fn include_globset_matches_markdown() {
        let patterns = vec!["**/*.md".to_string()];
        let gs = build_include_globset(&patterns);
        assert!(
            gs.is_match("docs/guide.md"),
            "**/*.md should match docs/guide.md"
        );
        assert!(
            gs.is_match("README.md"),
            "**/*.md should match top-level README.md"
        );
    }

    #[test]
    fn include_globset_rejects_non_markdown() {
        let patterns = vec!["**/*.md".to_string()];
        let gs = build_include_globset(&patterns);
        assert!(
            !gs.is_match("state.db"),
            "**/*.md should not match state.db"
        );
        assert!(
            !gs.is_match("scripts/run.sh"),
            "**/*.md should not match shell scripts"
        );
        assert!(
            !gs.is_match(".env"),
            "**/*.md should not match credential files"
        );
    }

    #[test]
    fn include_globset_respects_custom_patterns() {
        let patterns = vec!["**/*.md".to_string(), "**/*.txt".to_string()];
        let gs = build_include_globset(&patterns);
        assert!(gs.is_match("notes/todo.txt"), "should match *.txt");
        assert!(!gs.is_match("data.json"), "should not match *.json");
    }

    fn make_params(query: &str) -> SearchParams {
        SearchParams {
            query: query.to_string(),
            domain: None,
            r#type: None,
            tags: None,
            limit: None,
            min_score: None,
            explain: None,
            modified_after: None,
            modified_before: None,
        }
    }

    #[test]
    fn valid_params_accepted() {
        let params = make_params("find documents about authentication");
        assert!(validate_search_params(&params).is_ok());
    }

    #[test]
    fn query_at_limit_is_accepted() {
        let params = make_params(&"a".repeat(MAX_QUERY_LEN));
        assert!(validate_search_params(&params).is_ok());
    }

    #[test]
    fn query_too_long_is_rejected() {
        let params = make_params(&"a".repeat(MAX_QUERY_LEN + 1));
        assert!(validate_search_params(&params).is_err());
    }

    #[test]
    fn domain_too_long_is_rejected() {
        let params = SearchParams {
            domain: Some("x".repeat(MAX_FILTER_STR_LEN + 1)),
            ..make_params("query")
        };
        assert!(validate_search_params(&params).is_err());
    }

    #[test]
    fn type_too_long_is_rejected() {
        let params = SearchParams {
            r#type: Some("x".repeat(MAX_FILTER_STR_LEN + 1)),
            ..make_params("query")
        };
        assert!(validate_search_params(&params).is_err());
    }

    #[test]
    fn too_many_tags_rejected() {
        let params = SearchParams {
            tags: Some(vec!["tag".to_string(); MAX_TAG_COUNT + 1]),
            ..make_params("query")
        };
        assert!(validate_search_params(&params).is_err());
    }

    #[test]
    fn tag_too_long_is_rejected() {
        let params = SearchParams {
            tags: Some(vec!["x".repeat(MAX_TAG_LEN + 1)]),
            ..make_params("query")
        };
        assert!(validate_search_params(&params).is_err());
    }

    #[test]
    fn max_tags_at_limit_accepted() {
        let params = SearchParams {
            tags: Some(vec!["tag".to_string(); MAX_TAG_COUNT]),
            ..make_params("query")
        };
        assert!(validate_search_params(&params).is_ok());
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
    fn get_info_returns_dynamic_instructions() {
        use rmcp::ServerHandler;

        let tmp = tempfile::tempdir().unwrap();
        let custom_text = "Custom KB instructions.\nAvailable domain: infra, networking";
        let instructions = Arc::new(RwLock::new(custom_text.to_string()));

        let qdrant_config = crate::config::ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: "test".into(),
        };
        let qdrant = Arc::new(QdrantStore::new(&qdrant_config).unwrap());
        let embed_config = crate::config::ResolvedEmbeddingConfig {
            base_url: "http://localhost:8080/v1".into(),
            model: "test".into(),
            api_key: None,
            vector_size: 768,
            batch_size: 32,
            request_timeout_secs: 60,
            batch_concurrency: 4,
        };
        let embed = Arc::new(EmbedClient::new(&embed_config));

        let server = KbSearchServer::new(
            embed,
            qdrant,
            "test".into(),
            tmp.path().to_path_buf(),
            &["**/*.md".to_string()],
            instructions,
            crate::config::shared_config(make_test_resolved_config(tmp.path())),
            empty_test_schema_cache(),
            None,
        )
        .unwrap();

        let info = server.get_info();
        let returned = info.instructions.unwrap();
        assert_eq!(returned, custom_text);
    }

    #[test]
    fn get_info_reflects_updated_instructions() {
        use rmcp::ServerHandler;

        let tmp = tempfile::tempdir().unwrap();
        let instructions = Arc::new(RwLock::new("Initial instructions".to_string()));

        let qdrant_config = crate::config::ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: "test".into(),
        };
        let qdrant = Arc::new(QdrantStore::new(&qdrant_config).unwrap());
        let embed_config = crate::config::ResolvedEmbeddingConfig {
            base_url: "http://localhost:8080/v1".into(),
            model: "test".into(),
            api_key: None,
            vector_size: 768,
            batch_size: 32,
            request_timeout_secs: 60,
            batch_concurrency: 4,
        };
        let embed = Arc::new(EmbedClient::new(&embed_config));

        let server = KbSearchServer::new(
            embed,
            qdrant,
            "test".into(),
            tmp.path().to_path_buf(),
            &["**/*.md".to_string()],
            Arc::clone(&instructions),
            crate::config::shared_config(make_test_resolved_config(tmp.path())),
            empty_test_schema_cache(),
            None,
        )
        .unwrap();

        *instructions.write().unwrap() = "Updated with metadata".to_string();

        let info = server.get_info();
        assert_eq!(info.instructions.unwrap(), "Updated with metadata");
    }

    #[test]
    fn test_get_info_recovers_from_poisoned_lock() {
        use std::panic;

        let lock = Arc::new(RwLock::new("valid instructions".to_string()));
        let lock_clone = Arc::clone(&lock);

        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _guard = lock_clone.write().unwrap();
            panic!("intentional panic to poison the lock");
        }));

        assert!(lock.read().is_err(), "lock should be poisoned");

        let recovered = lock
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();

        assert_eq!(recovered, "valid instructions");
    }

    #[test]
    fn default_instructions_constant_is_reasonable() {
        assert!(DEFAULT_INSTRUCTIONS.contains("search"));
        assert!(
            DEFAULT_INSTRUCTIONS
                .to_lowercase()
                .contains("knowledge base")
        );
    }

    #[test]
    fn write_tool_descriptions_assert_scope() {
        let tools = KbSearchServer::tool_router().list_all();

        // The write tools are the ones an agent can use to park transient content
        // in the KB, so each must carry the scope boundary. Their descriptions are
        // compiled in, unlike the server instructions, which a deployment overrides
        // via `mcp.instructions`.
        for name in ["create_document", "edit_document"] {
            let tool = tools
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("tool '{name}' not registered"));
            let description = tool
                .description
                .as_deref()
                .unwrap_or_else(|| panic!("tool '{name}' has no description"));

            assert!(
                description.contains("SCOPE"),
                "'{name}' description should carry a scope assertion: {description}"
            );
            assert!(
                description.contains("NEVER"),
                "'{name}' scope assertion should be emphatic: {description}"
            );
            assert!(
                description.contains("session notes"),
                "'{name}' should name the transient-content anti-pattern: {description}"
            );
        }
    }

    #[test]
    fn include_globset_empty_patterns_falls_back_to_markdown() {
        let gs = build_include_globset(&[]);
        assert!(
            gs.is_match("docs/guide.md"),
            "empty patterns should fall back to **/*.md"
        );
        assert!(
            gs.is_match("README.md"),
            "empty patterns should match top-level .md"
        );
        assert!(
            !gs.is_match("state.db"),
            "empty patterns fallback should not match non-markdown"
        );
    }

    #[test]
    fn include_globset_all_invalid_falls_back_to_markdown() {
        let gs = build_include_globset(&["[invalid".into()]);
        assert!(
            gs.is_match("docs/guide.md"),
            "all-invalid patterns should fall back to **/*.md"
        );
        assert!(
            !gs.is_match("data.json"),
            "all-invalid patterns fallback should not match non-markdown"
        );
    }

    #[tokio::test]
    async fn get_document_rejects_overlong_path() {
        let tmp = tempfile::tempdir().unwrap();
        let instructions = Arc::new(RwLock::new("test".to_string()));
        let config = crate::config::ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: "test".into(),
        };
        let qdrant = Arc::new(QdrantStore::new(&config).unwrap());
        let embed_config = crate::config::ResolvedEmbeddingConfig {
            base_url: "http://localhost:8080/v1".into(),
            model: "test".into(),
            api_key: None,
            vector_size: 768,
            batch_size: 32,
            request_timeout_secs: 60,
            batch_concurrency: 4,
        };
        let embed = Arc::new(EmbedClient::new(&embed_config));
        let server = KbSearchServer::new(
            embed,
            qdrant,
            "test".into(),
            tmp.path().to_path_buf(),
            &["**/*.md".to_string()],
            instructions,
            crate::config::shared_config(make_test_resolved_config(tmp.path())),
            empty_test_schema_cache(),
            None,
        )
        .unwrap();

        let overlong_path = "a".repeat(MAX_PATH_LEN + 1);
        let params = GetDocumentParams {
            path: overlong_path,
        };
        let result = server.get_document(Parameters(params)).await;
        assert!(result.is_err(), "overlong path should return an error");
    }

    // -----------------------------------------------------------------------
    // write_document helper tests
    // -----------------------------------------------------------------------

    /// Build a KbSearchServer suitable for write_document unit tests.
    /// Uses a temp directory as the data path. No real Qdrant/git required
    /// for tests that fail before those steps.
    fn make_write_test_server(
        tmp: &tempfile::TempDir,
        include_patterns: &[String],
        config: Arc<ResolvedConfig>,
    ) -> KbSearchServer {
        let qdrant_config = crate::config::ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: "test".into(),
        };
        let qdrant = Arc::new(QdrantStore::new(&qdrant_config).unwrap());
        let embed_config = crate::config::ResolvedEmbeddingConfig {
            base_url: "http://localhost:8080/v1".into(),
            model: "test".into(),
            api_key: None,
            vector_size: 768,
            batch_size: 32,
            request_timeout_secs: 60,
            batch_concurrency: 4,
        };
        let embed = Arc::new(EmbedClient::new(&embed_config));
        let instructions = Arc::new(RwLock::new(String::new()));
        // Built from whatever `.kb-schema.yaml` files already exist under `tmp` at
        // this point — callers that need a test to see a schema written must write it
        // before calling this, exactly as they already do for `write_schema_file`.
        let canonical = tmp.path().canonicalize().unwrap();
        let schema_cache: crate::schema::SharedSchemaCache = Arc::new(RwLock::new(Arc::new(
            crate::schema::SchemaCache::build(&canonical, &config.frontmatter),
        )));
        KbSearchServer::new(
            embed,
            qdrant,
            "test".into(),
            tmp.path().to_path_buf(),
            include_patterns,
            instructions,
            crate::config::shared_config(config),
            schema_cache,
            None,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn edit_document_on_nonexistent_file_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        // edit_document now goes through resolve_within_data; NotFound → clear error.
        let params = EditDocumentParams {
            path: "docs/nonexistent.md".to_string(),
            old_string: None,
            new_string: None,
            content: Some("---\ntitle: Test\n---\n# Body".to_string()),
            message: None,
        };
        let result = server.edit_document(Parameters(params)).await;

        assert!(
            result.is_err(),
            "edit of non-existent file should return Err"
        );
        let err = result.unwrap_err();
        assert!(
            err.message.contains("does not exist"),
            "error message should mention 'does not exist', got: {}",
            err.message
        );
        assert!(
            err.message.contains("create_document"),
            "error should mention create_document, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn create_document_on_existing_file_returns_use_edit_error() {
        let tmp = tempfile::tempdir().unwrap();
        // Pre-create the file
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("existing.md"), "# Already here").unwrap();

        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        let params = CreateDocumentParams {
            path: "docs/existing.md".to_string(),
            content: "---\ntitle: Test\n---\n# New content".to_string(),
            message: None,
            force_new: None,
        };
        let result = server.create_document(Parameters(params)).await;

        assert!(result.is_err(), "create on existing file should return Err");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("already exists"),
            "error message should mention 'already exists', got: {}",
            err.message
        );
        assert!(
            err.message.contains("edit_document"),
            "error should mention edit_document, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn include_pattern_guard_rejects_non_matching_path() {
        let tmp = tempfile::tempdir().unwrap();
        let config = make_test_resolved_config(tmp.path());
        // Only markdown files are indexed
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        // Try to write a .txt file (not matched by **/*.md)
        let params = CreateDocumentParams {
            path: "notes.txt".to_string(),
            content: "Some plain text".to_string(),
            message: None,
            force_new: None,
        };
        let result = server.create_document(Parameters(params)).await;

        assert!(
            result.is_err(),
            "non-matching path should be rejected by include-pattern guard"
        );
        let err = result.unwrap_err();
        assert!(
            err.message.contains("indexable include pattern"),
            "error should mention include pattern, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn include_pattern_guard_rejects_absolute_path() {
        let tmp = tempfile::tempdir().unwrap();
        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        // Absolute path should be caught by the absolute-path guard before include check
        let params = CreateDocumentParams {
            path: "/etc/passwd".to_string(),
            content: "# Evil".to_string(),
            message: None,
            force_new: None,
        };
        let result = server.create_document(Parameters(params)).await;

        // A leading `/` now means the knowledge-base root, so this resolves to
        // `<kb>/etc/passwd` — safely inside the KB — and is then rejected by the
        // include-pattern guard because it is not an indexable markdown path.
        assert!(
            result.is_err(),
            "a non-indexable path must still be rejected"
        );
        let err = result.unwrap_err();
        assert!(
            err.message.contains("include pattern"),
            "error should cite the include-pattern guard, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn validation_failure_carries_field_errors_in_data() {
        let tmp = tempfile::tempdir().unwrap();

        // Config with validation enabled requiring "title" field
        let config = Arc::new(ResolvedConfig {
            source: crate::config::ResolvedSourceConfig {
                git_url: None,
                branch: "master".into(),
                data_path: Some(tmp.path().to_string_lossy().into_owned()),
                git_token_env: "GIT_PULL_TOKEN".into(),
            },
            indexing: crate::config::IndexingConfig::default(),
            frontmatter: crate::config::FrontmatterConfig {
                required: vec!["title".into()],
                ..Default::default()
            },
            chunking: crate::config::ChunkingConfig::default(),
            embedding: crate::config::ResolvedEmbeddingConfig {
                base_url: "http://localhost:8080/v1".into(),
                model: "test".into(),
                api_key: None,
                vector_size: 768,
                batch_size: 32,
                request_timeout_secs: 60,
                batch_concurrency: 4,
            },
            qdrant: crate::config::ResolvedQdrantConfig {
                url: "http://localhost:6334".into(),
                collection: "test".into(),
            },
            validation: crate::config::ValidationConfig {
                enabled: true,
                strict: false,
                lint_command: None,
            },
            webhook: crate::config::WebhookConfig::default(),
            mcp: crate::config::ResolvedMcpConfig::default(),
            rate_limit: crate::config::RateLimitConfig::default(),
            write: crate::config::WriteConfig::default(),
            search: crate::config::SearchConfig::default(),
            reranking: None,
            provenance: Default::default(),
        });

        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        // Content intentionally missing the "title" frontmatter field
        let params = CreateDocumentParams {
            path: "guide/missing-title.md".to_string(),
            content: "---\ntype: guide\n---\n# No title in frontmatter".to_string(),
            message: None,
            force_new: None,
        };
        let result = server.create_document(Parameters(params)).await;

        assert!(result.is_err(), "validation failure should return Err");
        let err = result.unwrap_err();

        // Message should be human-readable
        assert!(
            err.message.contains("frontmatter validation failed"),
            "error message should describe validation failure, got: {}",
            err.message
        );

        // Data field must contain structured field_errors
        let data = err.data.expect("error should carry structured data");
        let field_errors = data
            .get("field_errors")
            .expect("data must have field_errors key");
        assert!(
            field_errors.is_array(),
            "field_errors must be a JSON array, got: {}",
            field_errors
        );
        let arr = field_errors.as_array().unwrap();
        assert!(
            !arr.is_empty(),
            "field_errors array must be non-empty for a validation failure"
        );

        // At least one entry should mention "title" as the failed field
        let mentions_title = arr.iter().any(|fe| {
            fe.get("field")
                .and_then(|f| f.as_str())
                .map(|f| f == "title")
                .unwrap_or(false)
        });
        assert!(
            mentions_title,
            "field_errors should contain an entry for 'title', got: {}",
            serde_json::to_string_pretty(&data).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // dedup_verdict unit tests — no live Qdrant/embedder required
    // -----------------------------------------------------------------------

    #[test]
    fn dedup_verdict_score_above_threshold_returns_hit() {
        let result = dedup_verdict(Some(("docs/existing.md".into(), 0.92)), 0.85);
        assert!(
            result.is_some(),
            "score 0.92 >= threshold 0.85 should refuse"
        );
        let hit = result.unwrap();
        assert_eq!(hit.file_path, "docs/existing.md");
        assert!((hit.score - 0.92).abs() < 1e-6);
    }

    #[test]
    fn dedup_verdict_score_at_threshold_returns_hit() {
        // Boundary: score exactly equal to threshold is also a duplicate.
        let result = dedup_verdict(Some(("docs/boundary.md".into(), 0.85)), 0.85);
        assert!(
            result.is_some(),
            "score == threshold should be treated as duplicate"
        );
    }

    #[test]
    fn dedup_verdict_score_below_threshold_allows() {
        let result = dedup_verdict(Some(("docs/different.md".into(), 0.70)), 0.85);
        assert!(
            result.is_none(),
            "score 0.70 < threshold 0.85 should allow the write"
        );
    }

    #[test]
    fn dedup_verdict_no_results_allows() {
        // Empty collection or no results → no duplicate → allow.
        let result = dedup_verdict(None, 0.85);
        assert!(
            result.is_none(),
            "no results should allow the write (empty collection case)"
        );
    }

    #[test]
    fn dedup_verdict_hit_carries_correct_fields() {
        let hit = dedup_verdict(Some(("sysadmin/networking/dns.md".into(), 0.95)), 0.85)
            .expect("should be a hit");
        assert_eq!(hit.file_path, "sysadmin/networking/dns.md");
        assert!((hit.score - 0.95).abs() < 1e-6, "score should be preserved");
    }

    // -----------------------------------------------------------------------
    // dedup query construction + search options — pure, no live services
    // -----------------------------------------------------------------------

    #[test]
    fn build_dedup_query_prepends_description() {
        let q = build_dedup_query("Body text here.", Some("A short summary."), true);
        assert_eq!(q, "A short summary.\n\nBody text here.");
    }

    #[test]
    fn build_dedup_query_omits_description_when_disabled() {
        let q = build_dedup_query("Body text here.", Some("A short summary."), false);
        assert_eq!(
            q, "Body text here.",
            "prepend_description=false must match the indexer, which also omits it"
        );
    }

    #[test]
    fn build_dedup_query_omits_description_when_absent() {
        let q = build_dedup_query("Body text here.", None, true);
        assert_eq!(q, "Body text here.");
    }

    #[test]
    fn build_dedup_query_truncates_to_limit() {
        let long_body = "x".repeat(DEDUP_QUERY_CHAR_LIMIT * 2);
        let q = build_dedup_query(&long_body, None, false);
        assert_eq!(q.chars().count(), DEDUP_QUERY_CHAR_LIMIT);
    }

    #[test]
    fn build_dedup_query_truncation_counts_chars_not_bytes() {
        // Multi-byte input must not panic or split a character.
        let long_body = "é".repeat(DEDUP_QUERY_CHAR_LIMIT * 2);
        let q = build_dedup_query(&long_body, None, false);
        assert_eq!(q.chars().count(), DEDUP_QUERY_CHAR_LIMIT);
    }

    /// The dedup query must be built on the same textual basis the indexer
    /// embeds, otherwise the gate scores a query against candidates that were
    /// assembled differently. Pin the two together so they cannot drift.
    #[test]
    fn build_dedup_query_matches_chunk_prepend_format() {
        let body = "## Heading\n\nSome body content.";
        let description = "A short summary.";
        let chunking = crate::config::ChunkingConfig::default();
        assert!(
            chunking.prepend_description,
            "this test assumes the indexer default prepends description"
        );

        let chunks = crate::chunk::chunk_markdown(body, Some(description), &chunking);
        let first_chunk = &chunks.first().expect("body should produce a chunk").text;
        let query = build_dedup_query(body, Some(description), chunking.prepend_description);

        assert!(
            first_chunk.starts_with(&format!("{}\n\n", description)),
            "indexed chunk should carry the description prefix, got: {:?}",
            first_chunk
        );
        assert_eq!(
            query, *first_chunk,
            "dedup query and indexed chunk text must share one textual basis"
        );
    }

    /// Regression guard for issue #67: `write.dedup_threshold` is a cosine
    /// similarity, so the dedup search must never inherit `search.hybrid`
    /// (RRF scores top out near 0.03 and would make the gate unable to fire).
    #[test]
    fn dedup_search_opts_is_dense_only() {
        let opts = dedup_search_opts();
        assert!(
            !opts.hybrid,
            "dedup must be dense-only so its score is a cosine similarity"
        );
        assert!(
            crate::config::SearchConfig::default().hybrid,
            "search.hybrid defaults to true — this is exactly what dedup must not inherit"
        );
        assert_eq!(opts.limit, 1, "dedup only needs the nearest neighbour");
        assert!(
            opts.min_score.is_none(),
            "thresholding is dedup_verdict's job, not the search floor's"
        );
    }

    /// A cross-encoder relevance score is not a cosine similarity, so the gate
    /// must not request rerank candidate expansion either.
    #[test]
    fn dedup_search_opts_requests_no_rerank_expansion() {
        assert!(dedup_search_opts().rerank_candidate_limit.is_none());
    }

    /// Test that the gating booleans (`dedup_enabled`, `must_already_exist`, `force_new`)
    /// correctly bypass the dedup gate.  We use a server with dedup_enabled=false / true
    /// and call write_document up to the point where the gate would fire — since the
    /// embed client isn't reachable the gate's embed call fails-open (logs a warning and
    /// continues), but with dedup_enabled=false the gate is never entered at all, so we
    /// reach a different error (validation or file existence) and NOT a dedup refusal.
    #[tokio::test]
    async fn dedup_gate_disabled_via_config_does_not_embed() {
        let tmp = tempfile::tempdir().unwrap();

        // Config with dedup disabled.
        let config = Arc::new(ResolvedConfig {
            source: crate::config::ResolvedSourceConfig {
                git_url: None,
                branch: "master".into(),
                data_path: Some(tmp.path().to_string_lossy().into_owned()),
                git_token_env: "GIT_PULL_TOKEN".into(),
            },
            indexing: crate::config::IndexingConfig::default(),
            frontmatter: crate::config::FrontmatterConfig::default(),
            chunking: crate::config::ChunkingConfig::default(),
            embedding: crate::config::ResolvedEmbeddingConfig {
                base_url: "http://localhost:8080/v1".into(),
                model: "test".into(),
                api_key: None,
                vector_size: 768,
                batch_size: 32,
                request_timeout_secs: 60,
                batch_concurrency: 4,
            },
            qdrant: crate::config::ResolvedQdrantConfig {
                url: "http://localhost:6334".into(),
                collection: "test".into(),
            },
            validation: crate::config::ValidationConfig::default(),
            webhook: crate::config::WebhookConfig::default(),
            mcp: crate::config::ResolvedMcpConfig::default(),
            rate_limit: crate::config::RateLimitConfig::default(),
            write: crate::config::WriteConfig {
                dedup_enabled: false,
                dedup_threshold: 0.85,
                commit_author_name: "md-kb-rag".to_string(),
                commit_author_email: "md-kb-rag@localhost".to_string(),
            },
            search: crate::config::SearchConfig::default(),
            reranking: None,
            provenance: Default::default(),
        });

        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        // When dedup is disabled we should NOT get a dedup refusal.
        // The write will fail at git/reindex (no live services) — that's fine.
        let params = CreateDocumentParams {
            path: "docs/new.md".to_string(),
            content: "---\ntitle: Test Doc\n---\n# Content".to_string(),
            message: None,
            force_new: None,
        };
        let result = server.create_document(Parameters(params)).await;

        // We expect an error (git/reindex will fail in unit test), but it must
        // NOT be a dedup refusal — i.e. it should not mention "similar document".
        if let Err(e) = result {
            assert!(
                !e.message.contains("similar document"),
                "dedup disabled: error must not be a dedup refusal, got: {}",
                e.message
            );
        }
        // (Ok is also fine — means we somehow reached the write step, which is
        // unexpected in a unit test but not a test failure for this assertion.)
    }

    #[tokio::test]
    async fn dedup_gate_bypassed_by_force_new() {
        let tmp = tempfile::tempdir().unwrap();

        // Config with dedup ENABLED (default).
        let config = make_test_resolved_config(tmp.path());

        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        // With force_new=Some(true), the gate must be skipped even when dedup is
        // enabled.  The embed/qdrant will fail-open (no live services), so we will
        // reach git/reindex and fail there — but NOT with a dedup message.
        let params = CreateDocumentParams {
            path: "docs/forced.md".to_string(),
            content: "---\ntitle: Forced Doc\n---\n# Content".to_string(),
            message: None,
            force_new: Some(true),
        };
        let result = server.create_document(Parameters(params)).await;

        if let Err(e) = result {
            assert!(
                !e.message.contains("similar document"),
                "force_new=true must bypass dedup gate, got: {}",
                e.message
            );
        }
    }

    #[tokio::test]
    async fn dedup_gate_skipped_for_edit_path() {
        let tmp = tempfile::tempdir().unwrap();
        // Create the file so the edit path can proceed past existence check.
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(
            sub.join("edit-me.md"),
            "---\ntitle: Old Doc\n---\n# old content",
        )
        .unwrap();

        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        // Edit path should never trigger the dedup gate.
        // It will fail at git/reindex — but NOT with a dedup message.
        let params = EditDocumentParams {
            path: "docs/edit-me.md".to_string(),
            old_string: None,
            new_string: None,
            content: Some("---\ntitle: Edited Doc\n---\n# New content".to_string()),
            message: None,
        };
        let result = server.edit_document(Parameters(params)).await;

        if let Err(e) = result {
            assert!(
                !e.message.contains("similar document"),
                "edit path must never trigger dedup gate, got: {}",
                e.message
            );
        }
    }

    // -----------------------------------------------------------------------
    // delete_document unit tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn delete_document_nonexistent_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        let params = DeleteDocumentParams {
            path: "docs/nonexistent.md".to_string(),
            message: None,
        };
        let result = server.delete_document(Parameters(params)).await;

        assert!(
            result.is_err(),
            "delete of non-existent file should return Err"
        );
        let err = result.unwrap_err();
        assert!(
            err.message.contains("does not exist"),
            "error should mention 'does not exist', got: {}",
            err.message
        );
    }

    #[test]
    fn delete_document_relpath_derivation() {
        // Verify that the relpath strip logic produces the expected relative path.
        let canonical_data = std::path::PathBuf::from("/data/kb");
        let canonical_file = std::path::PathBuf::from("/data/kb/sysadmin/networking/dns.md");
        let rel = canonical_file
            .strip_prefix(&canonical_data)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        assert_eq!(rel, "sysadmin/networking/dns.md");
    }

    #[test]
    fn delete_document_diff_shows_all_as_removals() {
        // When deleting a file, render_unified_diff(old, "", path) should show all
        // lines as removals (every non-header line starts with '-').
        let old = "---\ntitle: My Doc\n---\n# Content\nSome text.\n";
        let diff = render_unified_diff(old, "", "docs/my-doc.md");
        assert!(!diff.is_empty(), "delete diff should be non-empty");
        for line in diff.lines() {
            if !line.starts_with("---")
                && !line.starts_with("+++")
                && !line.starts_with("@@")
                && !line.is_empty()
            {
                assert!(
                    line.starts_with('-'),
                    "all content lines in a delete diff should be removals, got: {line}"
                );
            }
        }
    }

    #[test]
    fn delete_document_commit_message_has_correct_trailers() {
        let msg = build_commit_message(None, "docs: delete notes/guide.md", "delete_document");
        assert!(
            msg.contains("Tool: md-kb-rag"),
            "should contain Tool trailer: {msg}"
        );
        assert!(
            msg.contains("Operation: delete_document"),
            "should contain Operation: delete_document trailer: {msg}"
        );
        assert!(
            msg.starts_with("docs: delete notes/guide.md"),
            "should start with delete subject: {msg}"
        );
    }

    #[test]
    fn delete_document_user_message_overrides_default() {
        let msg = build_commit_message(
            Some("chore: remove obsolete guide"),
            "docs: delete notes/guide.md",
            "delete_document",
        );
        assert!(
            msg.starts_with("chore: remove obsolete guide"),
            "user subject should override default: {msg}"
        );
        assert!(
            msg.contains("Operation: delete_document"),
            "trailer still present: {msg}"
        );
    }

    #[tokio::test]
    async fn delete_document_empty_path_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        let params = DeleteDocumentParams {
            path: "   ".to_string(), // whitespace-only
            message: None,
        };
        let result = server.delete_document(Parameters(params)).await;

        assert!(result.is_err(), "empty path should return Err");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("empty"),
            "error should mention empty path, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn delete_document_existing_file_proceeds_to_git_step() {
        // Create a real file in tmp; the tool should resolve it and proceed past
        // path-resolution and file-removal to the git step (which will fail since
        // there's no git repo). The error must NOT be a path-resolution error.
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(
            sub.join("delete-me.md"),
            "---\ntitle: Delete Me\n---\n# Body",
        )
        .unwrap();

        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        let params = DeleteDocumentParams {
            path: "docs/delete-me.md".to_string(),
            message: None,
        };
        let result = server.delete_document(Parameters(params)).await;

        // File removal happens before git; the file should be gone regardless.
        assert!(
            !sub.join("delete-me.md").exists(),
            "file should have been removed from disk"
        );

        // The error (if any) should be git- or index-related, not path-resolution.
        if let Err(e) = result {
            assert!(
                !e.message.contains("does not exist"),
                "error should not be path-resolution, got: {}",
                e.message
            );
        }
    }

    // -----------------------------------------------------------------------
    // resolve_safe_write_path unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn safe_write_path_treats_a_leading_slash_as_the_kb_root() {
        // Callers cannot know where the KB lives inside the container, so `/x.md` and
        // `x.md` must address the same document. The escape checks still apply — this
        // resolves under the data root, it does not reach the real /etc.
        let tmp = tempfile::tempdir().unwrap();

        let rooted = resolve_safe_write_path(tmp.path(), "/notes/a.md").unwrap();
        let relative = resolve_safe_write_path(tmp.path(), "notes/a.md").unwrap();
        assert_eq!(rooted, relative);
        assert!(rooted.starts_with(tmp.path()));

        // And traversal is still rejected however it is spelled.
        assert!(resolve_safe_write_path(tmp.path(), "/../escape.md").is_err());
        assert!(resolve_safe_write_path(tmp.path(), "../escape.md").is_err());
    }

    #[test]
    fn safe_write_path_rejects_parent_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let result = resolve_safe_write_path(tmp.path(), "../../etc/shadow");
        assert!(result.is_err(), "parent-dir component must be rejected");
        let msg = result.unwrap_err();
        assert!(msg.contains(".."), "error should mention '..', got: {msg}");
    }

    #[test]
    fn safe_write_path_accepts_normal_nested_path() {
        let tmp = tempfile::tempdir().unwrap();
        // The file doesn't need to exist; the ancestor (tmp itself) does.
        let result = resolve_safe_write_path(tmp.path(), "subdir/docs/guide.md");
        assert!(
            result.is_ok(),
            "normal nested path should be accepted, got: {:?}",
            result
        );
        let abs = result.unwrap();
        assert!(
            abs.starts_with(tmp.path()),
            "returned path should be under data_root"
        );
    }

    #[test]
    fn safe_write_path_rejects_symlinked_ancestor_outside_root() {
        // Create two separate temp directories.
        let inside_tmp = tempfile::tempdir().unwrap();
        let outside_tmp = tempfile::tempdir().unwrap();

        // Create a subdirectory inside inside_tmp that is actually a symlink
        // pointing to outside_tmp.
        let escaped_dir = inside_tmp.path().join("escaped");
        std::os::unix::fs::symlink(outside_tmp.path(), &escaped_dir)
            .expect("failed to create symlink");

        // A path through the symlink: "escaped/secret.md"
        // Lexically this is under inside_tmp, but canonically it resolves outside.
        let result = resolve_safe_write_path(inside_tmp.path(), "escaped/secret.md");

        assert!(
            result.is_err(),
            "path through symlinked ancestor pointing outside root must be rejected"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("symlink") || msg.contains("escapes"),
            "error should mention symlink or escape, got: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // parse_edit_mode unit tests
    // -----------------------------------------------------------------------

    fn make_edit_params(
        content: Option<&str>,
        old_string: Option<&str>,
        new_string: Option<&str>,
    ) -> EditDocumentParams {
        EditDocumentParams {
            path: "docs/test.md".to_string(),
            content: content.map(|s| s.to_string()),
            old_string: old_string.map(|s| s.to_string()),
            new_string: new_string.map(|s| s.to_string()),
            message: None,
        }
    }

    #[test]
    fn parse_edit_mode_full_replace_is_recognized() {
        let params = make_edit_params(Some("new content"), None, None);
        let mode = parse_edit_mode(&params).unwrap();
        assert_eq!(
            mode,
            EditMode::Full {
                content: "new content".to_string()
            }
        );
    }

    #[test]
    fn parse_edit_mode_surgical_is_recognized() {
        let params = make_edit_params(None, Some("old text"), Some("new text"));
        let mode = parse_edit_mode(&params).unwrap();
        assert_eq!(
            mode,
            EditMode::Surgical {
                old: "old text".to_string(),
                new: "new text".to_string()
            }
        );
    }

    #[test]
    fn parse_edit_mode_both_modes_rejected() {
        let params = make_edit_params(Some("full content"), Some("old"), Some("new"));
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(
            err.contains("mutually exclusive"),
            "expected 'mutually exclusive' in error, got: {err}"
        );
    }

    #[test]
    fn parse_edit_mode_neither_mode_rejected() {
        let params = make_edit_params(None, None, None);
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(
            err.contains("must provide"),
            "expected 'must provide' in error, got: {err}"
        );
    }

    #[test]
    fn parse_edit_mode_only_old_string_rejected() {
        let params = make_edit_params(None, Some("old"), None);
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(
            err.contains("new_string"),
            "expected mention of new_string in error, got: {err}"
        );
    }

    #[test]
    fn parse_edit_mode_only_new_string_rejected() {
        let params = make_edit_params(None, None, Some("new"));
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(
            err.contains("old_string"),
            "expected mention of old_string in error, got: {err}"
        );
    }

    #[test]
    fn parse_edit_mode_identical_old_new_rejected() {
        let params = make_edit_params(None, Some("same text"), Some("same text"));
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(
            err.contains("identical"),
            "expected 'identical' in error, got: {err}"
        );
    }

    // -----------------------------------------------------------------------
    // apply_surgical unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn apply_surgical_single_occurrence_replaced() {
        let old = "Hello world!\nGoodbye earth!";
        let result = apply_surgical(old, "world", "Rust").unwrap();
        assert_eq!(result, "Hello Rust!\nGoodbye earth!");
    }

    #[test]
    fn apply_surgical_not_found_returns_error() {
        let old = "Hello world!";
        let err = apply_surgical(old, "missing text", "replacement").unwrap_err();
        assert!(
            err.contains("not found"),
            "expected 'not found' in error, got: {err}"
        );
    }

    #[test]
    fn apply_surgical_multiple_occurrences_returns_error_with_count() {
        let old = "foo bar foo baz foo";
        let err = apply_surgical(old, "foo", "qux").unwrap_err();
        assert!(
            err.contains("3"),
            "error should mention occurrence count (3), got: {err}"
        );
        assert!(
            err.contains("not unique"),
            "error should mention 'not unique', got: {err}"
        );
    }

    #[test]
    fn apply_surgical_exact_single_unique_string() {
        let old = "---\ntitle: My Doc\n---\n# Content\nSome text here.";
        let result = apply_surgical(old, "Some text here.", "Updated text.").unwrap();
        assert_eq!(result, "---\ntitle: My Doc\n---\n# Content\nUpdated text.");
    }

    // -----------------------------------------------------------------------
    // render_unified_diff unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn render_unified_diff_shows_added_lines() {
        let old = "line1\nline2\n";
        let new = "line1\nline2\nline3\n";
        let diff = render_unified_diff(old, new, "docs/test.md");
        assert!(
            !diff.is_empty(),
            "diff should be non-empty for a changed doc"
        );
        assert!(
            diff.contains("+line3"),
            "diff should show added line, got:\n{diff}"
        );
        assert!(
            diff.contains("a/docs/test.md"),
            "diff header should name the file, got:\n{diff}"
        );
    }

    #[test]
    fn render_unified_diff_shows_removed_lines() {
        let old = "line1\nline2\nline3\n";
        let new = "line1\nline3\n";
        let diff = render_unified_diff(old, new, "docs/test.md");
        assert!(
            diff.contains("-line2"),
            "diff should show removed line, got:\n{diff}"
        );
    }

    #[test]
    fn render_unified_diff_identical_content_is_empty() {
        let content = "line1\nline2\n";
        let diff = render_unified_diff(content, content, "docs/test.md");
        assert!(
            diff.is_empty(),
            "identical content should produce empty diff, got:\n{diff}"
        );
    }

    #[test]
    fn render_unified_diff_create_shows_all_as_additions() {
        let old = "";
        let new = "---\ntitle: New Doc\n---\n# Hello\n";
        let diff = render_unified_diff(old, new, "docs/new.md");
        assert!(!diff.is_empty(), "new file diff should be non-empty");
        // Every non-header line should be an addition.
        for line in diff.lines() {
            if !line.starts_with("---")
                && !line.starts_with("+++")
                && !line.starts_with("@@")
                && !line.is_empty()
            {
                assert!(
                    line.starts_with('+'),
                    "all content lines in a create diff should be additions, got: {line}"
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // parse_date_to_timestamp tests
    // ------------------------------------------------------------------

    #[test]
    fn parse_date_rfc3339_returns_unix_timestamp() {
        // 2024-01-15T00:00:00Z is a known timestamp
        let ts = parse_date_to_timestamp("2024-01-15T00:00:00Z").unwrap();
        assert_eq!(
            ts, 1_705_276_800,
            "RFC 3339 midnight UTC should parse correctly"
        );
    }

    #[test]
    fn parse_date_date_only_treated_as_midnight_utc() {
        let ts = parse_date_to_timestamp("2024-01-15").unwrap();
        assert_eq!(
            ts, 1_705_276_800,
            "date-only should be treated as midnight UTC"
        );
    }

    #[test]
    fn parse_date_invalid_string_returns_err() {
        let result = parse_date_to_timestamp("not-a-date");
        assert!(
            result.is_err(),
            "invalid date string should return an error"
        );
    }

    #[test]
    fn parse_date_rfc3339_with_offset_returns_utc_equivalent() {
        // 2024-01-15T01:00:00+01:00 == 2024-01-15T00:00:00Z
        let ts = parse_date_to_timestamp("2024-01-15T01:00:00+01:00").unwrap();
        assert_eq!(ts, 1_705_276_800, "offset datetime should convert to UTC");
    }

    // --- write tools queue instead of reindexing inline ---
    //
    // Before the async reindex worker, these tools awaited `ingest::run_index` inline
    // and used `REINDEX_LOCK` to keep that from racing the webhook. Now they just mark
    // paths dirty on `reindex::REINDEX_QUEUE` and return — which also means these tests
    // no longer need a live Qdrant/embeddings service to reach that point, since
    // nothing here calls into the indexer at all.

    /// Build a `KbSearchServer` backed by a real git working clone, so write tools get
    /// past `commit_and_sync` and reach the point where they mark paths dirty.
    fn make_git_backed_server(
        work: &tempfile::TempDir,
    ) -> (KbSearchServer, Arc<crate::config::ResolvedConfig>) {
        let mut config = make_test_resolved_config(work.path());
        // Bypass the dedup gate: it would otherwise call out to a (nonexistent)
        // embedding service before we ever reach the commit.
        Arc::get_mut(&mut config).unwrap().write.dedup_enabled = false;
        let server = make_write_test_server(work, &["**/*.md".to_string()], Arc::clone(&config));
        (server, config)
    }

    #[tokio::test]
    async fn create_document_reports_queued_indexing_without_touching_the_indexer() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let (server, _config) = make_git_backed_server(&work);

        let pending_before = crate::reindex::REINDEX_QUEUE.snapshot().pending_paths;

        let result = server
            .create_document(Parameters(CreateDocumentParams {
                path: "docs/queued.md".to_string(),
                content:
                    "---\ntitle: Queued\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n"
                        .to_string(),
                message: None,
                force_new: Some(true),
            }))
            .await;

        let result = result.expect("write must succeed even though nothing indexes it inline");
        let text = format!("{:?}", result.content);
        assert!(
            text.contains("Created 'docs/queued.md'"),
            "must report the successful create: {text}"
        );
        assert!(
            text.contains("queued"),
            "must tell the caller indexing is queued, not synchronous: {text}"
        );
        assert!(
            !text.contains("SKIPPED"),
            "the old skipped-index warning language must be gone: {text}"
        );

        let pending_after = crate::reindex::REINDEX_QUEUE.snapshot().pending_paths;
        assert!(
            pending_after > pending_before,
            "create_document must mark its path dirty on the queue"
        );
    }

    /// The correctness case the synchronous rebuild in `update_schema` exists for:
    /// an agent that widens a rule and then immediately writes against it must be
    /// validated against the NEW schema, not whatever was cached when the server
    /// started. If `update_schema` only marked a full reconcile (`mark_full`) and
    /// left the refresh to the reindex worker, this would fail here, because
    /// nothing in this test spawns that worker — exactly the regression this test
    /// is meant to catch.
    #[tokio::test]
    async fn create_document_immediately_after_update_schema_validates_against_the_new_rules() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        // The schema baked into the server's cache at construction time below: only
        // "active" is permitted yet.
        write_schema_file(
            &work,
            "notes",
            "fields:\n  status:\n    type: enum\n    values: [active]\n",
        );
        let (server, _config) = make_git_backed_server(&work);

        // Sanity check that the OLD schema really does reject "beta" — otherwise the
        // second half of this test would not be proving anything.
        let rejected = server
            .create_document(Parameters(CreateDocumentParams {
                path: "notes/before.md".to_string(),
                content: "---\ntitle: Before\nstatus: beta\n---\n\n# Body\n".to_string(),
                message: None,
                force_new: Some(true),
            }))
            .await;
        assert!(
            rejected.is_err(),
            "sanity check failed: 'beta' must not be permitted before the schema \
             change — got {rejected:?}"
        );

        // Widen the rule through the tool an agent would actually use.
        server
            .update_schema(Parameters(UpdateSchemaParams {
                path: Some("notes".into()),
                operation: "add_values".into(),
                field: "status".into(),
                values: Some(vec!["beta".into()]),
                definition: None,
                dry_run: None,
                force: None,
            }))
            .await
            .expect("update_schema must succeed against this git-backed harness");

        // Same server, same cached schema handle, next call: must see the new rule.
        let accepted = server
            .create_document(Parameters(CreateDocumentParams {
                path: "notes/after.md".to_string(),
                content: "---\ntitle: After\nstatus: beta\n---\n\n# Body\n".to_string(),
                message: None,
                force_new: Some(true),
            }))
            .await;
        assert!(
            accepted.is_ok(),
            "the write immediately after update_schema must validate against the \
             NEW schema, not a stale cached copy: {:?}",
            accepted.err()
        );
    }

    #[tokio::test]
    async fn delete_document_reports_queued_cleanup_without_touching_the_indexer() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::write(
            work.path().join("doomed.md"),
            "---\ntitle: D\n---\n\n# Body\n",
        )
        .unwrap();
        // delete_document git-adds the removed path, so the file must already be tracked.
        std::process::Command::new("git")
            .args(["add", "doomed.md"])
            .current_dir(work.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args([
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "commit",
                "-m",
                "add doomed.md",
            ])
            .current_dir(work.path())
            .output()
            .unwrap();
        let (server, _config) = make_git_backed_server(&work);

        let pending_before = crate::reindex::REINDEX_QUEUE.snapshot().pending_paths;

        let result = server
            .delete_document(Parameters(DeleteDocumentParams {
                path: "doomed.md".to_string(),
                message: None,
            }))
            .await;

        let result = result.expect("delete must succeed even though nothing purges it inline");
        let text = format!("{:?}", result.content);
        assert!(
            text.contains("Deleted 'doomed.md'"),
            "must report the successful delete: {text}"
        );
        assert!(
            text.contains("queued"),
            "must tell the caller cleanup is queued, not synchronous: {text}"
        );
        assert!(
            !text.contains("REMAINS in the search index"),
            "the old skipped-purge warning language must be gone: {text}"
        );

        let pending_after = crate::reindex::REINDEX_QUEUE.snapshot().pending_paths;
        assert!(
            pending_after > pending_before,
            "delete_document must mark its path dirty on the queue rather than purging inline"
        );
    }
}
