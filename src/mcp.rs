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
    write::{
        DirectoryMoveError, DirectoryMoveSuccess, WriteDeps, WriteError,
        WriteOutcome as CoreWriteOutcome, WriteRequest, WriteSuccess,
    },
};

const MAX_QUERY_LEN: usize = 4096;
const MAX_PATH_LEN: usize = 4096;
const MAX_FILTER_STR_LEN: usize = 256;
const MAX_TAG_COUNT: usize = 20;
const MAX_TAG_LEN: usize = 256;
const MAX_CONTENT_LEN: usize = 512 * 1024; // 512 KB

/// Resolve a caller's requested `limit` against the configured default and ceiling.
///
/// An over-large request is clamped rather than rejected — a caller asking for more
/// than the maximum gets the maximum, which is friendlier than an error and matches
/// the historical behaviour when both values were hardcoded.
fn resolve_limit(requested: Option<u64>, default_limit: u64, max_limit: u64) -> u64 {
    requested.unwrap_or(default_limit).min(max_limit)
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

/// Keys a `set_field` definition accepts, kept in one place so every error that names
/// them — and the parameter's own doc comment — say exactly the same thing.
const FIELD_DEFINITION_KEYS: &str =
    "`type`, `required`, `indexed`, `values`, `extend`, `default`, `open`";

/// A `set_field` definition as delivered by an MCP client.
///
/// The `update_schema` tool schema advertises this parameter as the plain JSON object
/// described by [`crate::schema::RawFieldDef`] — the same shape a `.kb-schema.yaml`
/// entry uses. That's a deliberate fix: `serde_json::Value` (the old type here) produces
/// no `type` constraint at all in the advertised schema, and at least one real MCP
/// client responded to that ambiguity by sending the definition as a JSON-encoded
/// *string* instead of an object, which the old handler rejected with an error naming a
/// Rust struct the caller has no way to act on.
///
/// This type's [`Deserialize`](serde::Deserialize) impl still tolerates that string
/// form as a runtime fallback — some clients stringify nested-object arguments
/// regardless of what the schema says — but the *advertised* schema is not widened to
/// document it: a `oneOf: [object, string]` schema would just reopen the same
/// ambiguity for clients that DO read it. A conforming client only ever needs to send
/// the object.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldDefinitionInput(pub crate::schema::RawFieldDef);

impl<'de> serde::Deserialize<'de> for FieldDefinitionInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        parse_field_definition(value)
            .map(FieldDefinitionInput)
            .map_err(serde::de::Error::custom)
    }
}

// Delegate schema generation to `RawFieldDef`'s own derived `JsonSchema` impl rather
// than hand-duplicating its fields here, so the advertised shape and the accepted shape
// can never drift apart. This is what actually fixes the bug: the tool schema now
// advertises a real object with named, typed properties instead of `{}`.
impl schemars::JsonSchema for FieldDefinitionInput {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        crate::schema::RawFieldDef::schema_name()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        crate::schema::RawFieldDef::schema_id()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        crate::schema::RawFieldDef::json_schema(generator)
    }
}

/// Parse a `set_field` definition from JSON: a JSON object directly, or (see
/// [`FieldDefinitionInput`]) a string containing one. Every error names the expected
/// shape in caller-facing terms — never a bare Rust type name, which means nothing to
/// an MCP client on the other end of the wire.
fn parse_field_definition(value: serde_json::Value) -> Result<crate::schema::RawFieldDef, String> {
    use serde_json::Value;

    let object = match value {
        Value::Object(_) => value,
        Value::String(s) => match serde_json::from_str::<Value>(&s) {
            Ok(parsed @ Value::Object(_)) => parsed,
            Ok(other) => return Err(definition_shape_error(&other)),
            Err(e) => {
                return Err(format!(
                    "field definition must be a JSON object with keys \
                     {FIELD_DEFINITION_KEYS} (mirroring a .kb-schema.yaml entry). A JSON \
                     string containing that object is also accepted, but this string is \
                     not valid JSON: {e}"
                ));
            }
        },
        other => return Err(definition_shape_error(&other)),
    };

    serde_json::from_value(object).map_err(|e| format!("invalid field definition: {e}"))
}

/// Build the "wrong shape entirely" error for [`parse_field_definition`], naming what
/// was actually received without echoing its (possibly large) content.
fn definition_shape_error(value: &serde_json::Value) -> String {
    format!(
        "field definition must be a JSON object with keys {FIELD_DEFINITION_KEYS} \
         (mirroring a .kb-schema.yaml entry), got {}",
        json_value_kind(value)
    )
}

/// Describe a JSON value's kind in a few words, for error messages.
fn json_value_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
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
    if let Some(definition) = &params.definition {
        // Measured on the parsed-and-reserialized form rather than whatever bytes the
        // client happened to send: that's what actually gets committed to the schema
        // file (via `SchemaFile::to_yaml`), and it makes the cap apply identically
        // whether the definition arrived as an object or as the string fallback.
        let size = serde_json::to_string(&definition.0)
            .map(|s| s.len())
            .unwrap_or(usize::MAX);
        if size > MAX_SCHEMA_DEFINITION_LEN {
            return Err(invalid(format!(
                "field definition too large (max {} bytes)",
                MAX_SCHEMA_DEFINITION_LEN
            )));
        }
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
            // Parsing already happened when the tool call's arguments were
            // deserialized into `UpdateSchemaParams` (see `FieldDefinitionInput`), so
            // there's nothing left to do here but unwrap it.
            let definition = params
                .definition
                .clone()
                .ok_or_else(|| invalid("'set_field' requires a definition".into()))?;
            Ok(SchemaEdit::SetField {
                field: params.field.clone(),
                definition: Box::new(definition.0),
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

/// Parameters for the `get_document` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GetDocumentParams {
    /// Path of the document to retrieve. Accepts paths relative to the
    /// knowledge-base root (e.g. `lifestyle/vehicles/foo.md`, as returned by
    /// the `search` tool), or just a basename when it's unique across the
    /// index. Absolute paths are also accepted for backwards compatibility.
    pub path: String,
    /// First line to return, 1-based and inclusive. Omit to start at the top of
    /// the document. Given without `end_line`, the rest of the document is
    /// returned.
    #[serde(default)]
    pub start_line: Option<usize>,
    /// Last line to return, 1-based and inclusive. Omit to read to the end of
    /// the document. Given without `start_line`, reading starts at line 1. A
    /// value past the last line is clamped, not an error.
    #[serde(default)]
    pub end_line: Option<usize>,
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

    /// Field definition, for `set_field`: a JSON object with the same keys as a
    /// `.kb-schema.yaml` entry — `type`, `required`, `indexed`, `values`, `extend`,
    /// `default`, `open` — all optional. As a fallback for clients that stringify
    /// nested-object arguments, a JSON string containing that same object is also
    /// accepted, though the object form is what this schema describes and what a
    /// conforming client should send.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub definition: Option<FieldDefinitionInput>,

    /// Report what the change would do without writing anything, including which
    /// existing documents it would invalidate. Never refuses — it always succeeds and
    /// reports. When false (the default), a change that would invalidate existing
    /// documents is refused unless `force` is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_run: Option<bool>,

    /// Apply even when existing documents would fail the new rules.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force: Option<bool>,

    /// Required to let `add_values`, `set_field`, or `remove_field` apply against the
    /// knowledge-base root scope (path omitted/empty). Root-scope mutations are guarded
    /// per the KB's `meta/schema-tag-policy.md` — pass `true` only when the change is a
    /// deliberate design decision consistent with that policy, not as a routine default.
    /// Not needed for `remove_values`, for `dry_run` calls, or for any non-root path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acknowledge_root_change: Option<bool>,
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
    /// Optional stale-read guard: the SHA-256 hex digest of the document's full
    /// content (frontmatter included) at the time you read it — the same
    /// `content_hash` `get_document` returns in `structured_content`, and the same
    /// hashing this project already uses to detect changed files. When set, the edit
    /// is refused with an explicit "changed since you read it" error if the file's
    /// current content hash does not match, instead of a confusing old_string/content
    /// mismatch. Omit to skip this check.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_hash: Option<String>,
    /// Relocate the document to this new repo-relative path in the same commit as
    /// the edit. The destination must not already exist. The server reads the
    /// document's CURRENT content itself — you never need to re-send the document
    /// body just to move it. Combines with either edit mode (surgical or
    /// full-replace) to relocate AND change content in one commit, or may be
    /// provided with neither old_string/new_string nor content for a pure move
    /// that leaves the content unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_path: Option<String>,
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

/// Parameters for the `move_directory` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct MoveDirectoryParams {
    /// Source directory prefix, relative to the knowledge base root (e.g.
    /// "sysadmin/old-project"). Every document under this prefix, at any depth,
    /// is moved. Must exist and contain at least one indexable document.
    pub source_path: String,
    /// Destination directory prefix, relative to the knowledge base root. Must
    /// not already have any file living under it — this tool never merges into
    /// or overwrites an existing prefix.
    pub dest_path: String,
    /// Optional commit message; if omitted, a message is generated from the two
    /// prefixes.
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
/// typed `Option<EditMode>` or a human-readable error string.
///
/// Rules:
/// - SURGICAL = `old_string` AND `new_string` both `Some`, `content` is `None`.
/// - FULL = `content` is `Some`, both `old_string` and `new_string` are `None`.
/// - Neither SURGICAL nor FULL, but `new_path` is `Some`: `Ok(None)` — a pure
///   move with content left unchanged. The caller (`edit_document`) reads the
///   document's current content itself and passes it through untouched.
/// - Neither SURGICAL nor FULL, and `new_path` is also `None`: rejected — at
///   least one of the three (surgical, full-replace, move) must be requested.
/// - SURGICAL and FULL together are always rejected, regardless of `new_path` —
///   the two edit modes remain mutually exclusive WITH EACH OTHER; `new_path` is
///   an orthogonal, independent axis that may combine with either one (or
///   neither).
/// - Surgical with `old_string == new_string` is rejected (no-op).
pub fn parse_edit_mode(params: &EditDocumentParams) -> Result<Option<EditMode>, String> {
    let has_content = params.content.is_some();
    let has_old = params.old_string.is_some();
    let has_new = params.new_string.is_some();
    let has_move = params.new_path.is_some();

    match (has_content, has_old, has_new) {
        // Full mode (optionally combined with a move — new_path is orthogonal
        // and applied by the caller, not read here)
        (true, false, false) => Ok(Some(EditMode::Full {
            content: params.content.clone().unwrap(),
        })),
        // Surgical mode (optionally combined with a move, same as above)
        (false, true, true) => {
            let old = params.old_string.clone().unwrap();
            let new = params.new_string.clone().unwrap();
            if old == new {
                return Err(
                    "old_string and new_string are identical — no change would be made".to_string(),
                );
            }
            Ok(Some(EditMode::Surgical { old, new }))
        }
        // Both modes set: still mutually exclusive with each other even when
        // new_path is also present — a move never resolves which of the two
        // conflicting edits to apply before relocating.
        (true, _, _) if has_old || has_new => {
            Err("content is mutually exclusive with old_string/new_string; \
             provide either content (full replace) or old_string+new_string (surgical edit) \
             — not both. new_path may be combined with either one (or with neither, for a \
             pure move), but it does not resolve a conflict between the two edit modes \
             themselves."
                .to_string())
        }
        // Only one of old_string/new_string. new_path does not change this —
        // a surgical edit always needs both halves, move or no move.
        (false, true, false) => Err(
            "old_string requires new_string; provide both for a surgical edit. (If you only \
             meant to move the document, omit old_string entirely and pass new_path alone.)"
                .to_string(),
        ),
        (false, false, true) => Err(
            "new_string requires old_string; provide both for a surgical edit. (If you only \
             meant to move the document, omit new_string entirely and pass new_path alone.)"
                .to_string(),
        ),
        // Neither edit mode: a pure move if new_path was given, otherwise an error.
        (false, false, false) => {
            if has_move {
                Ok(None)
            } else {
                Err(
                    "must provide content (full replace), old_string+new_string (surgical \
                     edit), or new_path (move) — at least one is required"
                        .to_string(),
                )
            }
        }
        // Unreachable combinations (content=true, old=true, new=true or content=true, old/new only)
        _ => Err("content is mutually exclusive with old_string/new_string; \
             provide either content (full replace) or old_string+new_string (surgical edit)"
            .to_string()),
    }
}

/// Number of leading characters of `old_string` used as a search anchor when no exact
/// or whitespace-normalized match exists at all. Long enough to be a specific location
/// in most real documents, short enough that a single `str::find` over the document
/// stays cheap.
const NOT_FOUND_ANCHOR_CHARS: usize = 40;

/// How many characters of surrounding document text to show on each side of an anchor
/// match, when reporting a near-match diagnostic. Kept small deliberately: this text
/// goes into an error a caller (and possibly its logs) will see, so it is an
/// orientation snippet, not a document excerpt.
const NOT_FOUND_CONTEXT_CHARS: usize = 80;

/// Above this size, skip the near-match/anchor diagnostics below and fall back to the
/// plain not-found message. `old_content` is the on-disk document and, unlike
/// `new_content`, is not itself bounded by `MAX_CONTENT_LEN` — it could predate that
/// cap or have been written outside these tools entirely — so this is a second,
/// independent bound rather than an assumption that the write-path cap already covers
/// it.
const NOT_FOUND_DIAGNOSTIC_MAX_BYTES: usize = MAX_CONTENT_LEN;

/// Collapse every run of whitespace (including newlines) to a single space and trim
/// the ends, for a whitespace-insensitive comparison.
///
/// This treats differing indentation, trailing spaces, and CRLF-vs-LF line endings as
/// equal — by far the most common reason a caller's `old_string` fails to match
/// verbatim (e.g. it was retyped or reformatted rather than copied from
/// `get_document`'s output). One linear pass, no allocation beyond the output string,
/// so it stays cheap up to `NOT_FOUND_DIAGNOSTIC_MAX_BYTES`.
fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A small window of `content` centered on the char-boundary byte offset `pos`,
/// clamped to `radius` characters on each side. Used to show just enough surrounding
/// text to orient a caller, never a large slice of the document.
fn context_window(content: &str, pos: usize, radius: usize) -> &str {
    let start = content[..pos]
        .char_indices()
        .rev()
        .nth(radius.saturating_sub(1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let end = content[pos..]
        .char_indices()
        .nth(radius)
        .map(|(i, _)| pos + i)
        .unwrap_or(content.len());
    &content[start..end]
}

/// Apply a surgical edit: replace the single occurrence of `old_string` with
/// `new_string` in `old_content`.
///
/// Returns the new content string on success, or a descriptive error string.
/// `display_path` is the document's KB-relative path, used only to make the error
/// message name the file directly rather than the generic word "document".
///
/// When `old_string` occurs zero times, the error is built to help the caller decide
/// what to do next rather than just restating the failure (issue #88):
/// 1. If a whitespace-insensitive match exists, say so — that is almost always the
///    actual cause, and the fix is "copy it verbatim" rather than "re-read the file".
/// 2. Otherwise, anchor on the start of `old_string` and, if that much appears in the
///    document, show the surrounding text so the caller can see exactly how reality
///    diverges (a stale read, a typo partway through, wrong section, etc).
/// 3. Otherwise, nothing in the document resembles `old_string` at all — most likely
///    the wrong document or content that changed substantially.
///
/// Cost is bounded: every diagnostic pass here is a single linear scan (`split_whitespace`,
/// `contains`, or `str::find`) over `old_content`, and the whole diagnostic block is
/// skipped above `NOT_FOUND_DIAGNOSTIC_MAX_BYTES`, so a failed edit is never
/// accidentally quadratic in document size.
pub fn apply_surgical(
    old_content: &str,
    old_string: &str,
    new_string: &str,
    display_path: &str,
) -> Result<String, String> {
    let count = old_content.matches(old_string).count();
    match count {
        0 => {
            if old_content.len() > NOT_FOUND_DIAGNOSTIC_MAX_BYTES {
                return Err(format!("old_string not found in '{display_path}'"));
            }

            if !old_string.trim().is_empty()
                && normalize_whitespace(old_content).contains(&normalize_whitespace(old_string))
            {
                return Err(format!(
                    "old_string not found in '{display_path}', but a near-match exists \
                     differing only in whitespace — indentation, trailing spaces, or \
                     line endings. Copy old_string exactly as returned by get_document \
                     rather than retyping or reformatting it."
                ));
            }

            let anchor_end = old_string
                .char_indices()
                .nth(NOT_FOUND_ANCHOR_CHARS)
                .map(|(i, _)| i)
                .unwrap_or(old_string.len());
            let anchor = old_string[..anchor_end].trim();

            if !anchor.is_empty()
                && let Some(pos) = old_content.find(anchor)
            {
                let window = context_window(old_content, pos, NOT_FOUND_CONTEXT_CHARS);
                return Err(format!(
                    "old_string not found in '{display_path}'. Its start ('{anchor}') \
                     does appear, but the text after that point differs from \
                     old_string — nearby document content: \"…{window}…\". The file may \
                     have changed since you read it, or old_string may have a typo past \
                     that point."
                ));
            }

            Err(format!(
                "old_string not found in '{display_path}', and nothing resembling it \
                 was located either. This may be the wrong document, or its content has \
                 changed substantially since it was last read — re-read it with \
                 get_document before editing."
            ))
        }
        1 => Ok(old_content.replacen(old_string, new_string, 1)),
        n => Err(format!(
            "old_string is not unique in '{display_path}' (found {n} occurrences); \
             include more surrounding context to disambiguate"
        )),
    }
}

/// The definitive, machine-readable outcome of a write tool call
/// (`create_document`/`edit_document`/`delete_document`), exposed via
/// `CallToolResult::structured_content` under the `"outcome"` key so a caller can
/// branch on `result.structured_content["outcome"]` instead of pattern-matching the
/// human-readable summary text.
///
/// The last three variants map directly onto `git::CommitSyncError`'s pre-commit /
/// post-commit split (see that type's docs for why the two need opposite recovery):
/// `FailedNoChange` and `FailedInconsistentState` both come from a `PreCommit`
/// failure (the former rolled back cleanly, the latter's rollback itself failed);
/// `CommittedPendingSync` comes from a `PostCommit` failure, which is deliberately
/// left uncorrected. `NotFound` and validation failures never reach `write_document`/
/// `delete_document` at all — they are rejected earlier via `McpError::invalid_params`
/// — so they are not modeled here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriteOutcome {
    /// Committed and pushed. The tool's happy path.
    Synced,
    /// Committed locally, but the remote push failed. NOT rolled back — the commit is
    /// real. Will sync on the next successful write, or on manual intervention.
    CommittedPendingSync,
    /// Nothing was committed and the pre-commit rollback succeeded: filesystem and
    /// git state are back to exactly how they were before this call. Safe to retry.
    FailedNoChange,
    /// Nothing was committed AND the pre-commit rollback itself failed. Filesystem
    /// and git are now inconsistent with each other and with HEAD. This needs
    /// operator attention, not a blind retry — logged at `error!` for that reason.
    FailedInconsistentState,
}

impl WriteOutcome {
    fn as_str(self) -> &'static str {
        match self {
            WriteOutcome::Synced => "synced",
            WriteOutcome::CommittedPendingSync => "committed_pending_sync",
            WriteOutcome::FailedNoChange => "failed_no_change",
            WriteOutcome::FailedInconsistentState => "failed_inconsistent_state",
        }
    }
}

/// Attach a machine-readable `{"outcome": ...}` discriminant to a successful
/// `CallToolResult`, alongside its human-readable text content.
fn with_outcome(mut result: CallToolResult, outcome: WriteOutcome) -> CallToolResult {
    result.structured_content = Some(serde_json::json!({ "outcome": outcome.as_str() }));
    result
}

/// Like [`with_outcome`], but for `create_document`/`edit_document` specifically:
/// also attaches `rewritten_paths`, the repo-relative paths of OTHER documents a
/// MOVE rewrote incoming links in (always `[]` for a non-move write, or a move
/// with nothing to rewrite). A move that silently edits other documents without
/// surfacing which ones is not acceptable, so this rides in `structured_content`
/// on every create/edit result, not just moves.
fn with_outcome_and_rewrites(
    mut result: CallToolResult,
    outcome: WriteOutcome,
    rewritten_paths: &[String],
) -> CallToolResult {
    result.structured_content = Some(serde_json::json!({
        "outcome": outcome.as_str(),
        "rewritten_paths": rewritten_paths,
    }));
    result
}

/// Build the `data` payload for an `McpError` reporting a failed write-tool outcome,
/// so the same `{"outcome": ...}` discriminant is available on the error path too
/// (`ErrorData::data`), not just on success.
fn outcome_data(outcome: WriteOutcome) -> Option<serde_json::Value> {
    Some(serde_json::json!({ "outcome": outcome.as_str() }))
}

/// Map a successful `write::write_document` result (create or edit) onto this
/// tool surface's `CallToolResult`, preserving the exact text and
/// `structured_content` shape the existing create/edit tests pin down.
fn create_edit_success_to_result(
    success: WriteSuccess,
    rel_path: &str,
    is_create: bool,
) -> CallToolResult {
    let action = if is_create { "Created" } else { "Edited" };
    // Shared by both outcomes below: a one-line addendum naming exactly which
    // OTHER documents a move rewrote incoming links in, so an agent moving a
    // document is told about the side effect rather than discovering it later.
    // Empty for every non-move write and for a move with nothing to rewrite.
    let rewrite_note = if success.rewritten_paths.is_empty() {
        String::new()
    } else {
        format!(
            "\n\nUpdated links in {} document(s): {}.",
            success.rewritten_paths.len(),
            success.rewritten_paths.join(", ")
        )
    };
    match success.outcome {
        CoreWriteOutcome::Synced => {
            let summary = format!(
                "{} '{}' (commit {}). Indexing has been queued and will complete shortly.{}",
                action, rel_path, success.sha, rewrite_note
            );
            let mut result_text = summary;
            if !success.diff.is_empty() {
                result_text = format!("{}\n\n{}", result_text, success.diff);
            }
            with_outcome_and_rewrites(
                CallToolResult::success(vec![Content::text(result_text)]),
                WriteOutcome::Synced,
                &success.rewritten_paths,
            )
        }
        CoreWriteOutcome::CommittedPendingSync => {
            let cause = success
                .sync_failure_cause
                .as_deref()
                .unwrap_or("unknown error");
            let summary = format!(
                "{} '{}' (commit {}) — committed locally, but the push to the remote \
                 failed: {}. It will sync on the next successful write or manual \
                 intervention. Indexing has been queued from the local copy.{}",
                action, rel_path, success.sha, cause, rewrite_note
            );
            let mut result_text = summary;
            if !success.diff.is_empty() {
                result_text = format!("{}\n\n{}", result_text, success.diff);
            }
            with_outcome_and_rewrites(
                CallToolResult::success(vec![Content::text(result_text)]),
                WriteOutcome::CommittedPendingSync,
                &success.rewritten_paths,
            )
        }
    }
}

/// Map a `write::write_document` failure (create or edit) onto this tool
/// surface's `McpError`, preserving the exact text/data shapes the existing
/// create/edit tests pin down.
///
/// `canonical_data_path` is used only to reconstruct the absolute path for the
/// `AlreadyExists` race (see that arm's comment) — every other arm reports
/// against `rel_path`, matching what the tool surface's other errors already do.
///
/// `dest_path` is `Some` only when this call is (or was attempting to be) a
/// document MOVE — i.e. `edit_document` was called with `new_path` set. It
/// disambiguates the `AlreadyExists` arm, which for a move reports a collision
/// at the DESTINATION, not at `rel_path` (the source, which — for a move — is
/// expected to already exist). `create_document`'s own TOCTOU race, and every
/// non-move `edit_document` call, pass `None` here, preserving the original
/// `AlreadyExists` wording keyed on `rel_path`.
fn create_edit_error_to_mcp_error(
    err: WriteError,
    rel_path: &str,
    is_create: bool,
    canonical_data_path: &Path,
    dest_path: Option<&str>,
) -> McpError {
    match err {
        WriteError::Frozen { reason } => McpError::invalid_params(
            format!(
                "Cannot write '{}': the schema governing this directory is invalid ({}). \
                 Fix {} before writing here.",
                rel_path,
                reason,
                crate::schema::SCHEMA_FILE_NAME
            ),
            None,
        ),
        WriteError::Validation { result } => McpError::invalid_params(
            format!(
                "frontmatter validation failed for '{}': {}",
                rel_path,
                result.errors.join("; ")
            ),
            Some(serde_json::json!({ "field_errors": result.field_errors })),
        ),
        WriteError::DedupHit {
            duplicate_of,
            similarity,
            threshold,
        } => McpError::invalid_params(
            format!(
                "A similar document already exists: '{}' \
                 (similarity {:.2} ≥ threshold {:.2}). \
                 Edit it with edit_document, or pass \
                 force_new=true to create a new document anyway.",
                duplicate_of, similarity, threshold
            ),
            Some(serde_json::json!({
                "duplicate_of": duplicate_of,
                "similarity": similarity,
                "threshold": threshold,
            })),
        ),
        WriteError::InvalidCommitMessage { reason } => McpError::invalid_params(reason, None),
        WriteError::UnsafePath { msg } => McpError::invalid_params(msg, None),
        // Same text MCP callers already saw for this failure before
        // `WriteError::Internal` existed to split it out of `UnsafePath` — see
        // that variant's doc comment. MCP is a trusted surface, so unlike
        // `web.rs` (which maps this to a generic message) this stays verbatim.
        WriteError::Internal { msg } => McpError::invalid_params(msg, None),
        // Two distinct races share this variant:
        //
        // 1. `create_document`'s own pre-check (`abs_path.exists()`) already
        //    passed, so the file was created between that check and
        //    `write::write_document`'s `create_new` open. Restored to the
        //    pre-`write.rs`-extraction wording, which reported the absolute
        //    filesystem path rather than the repo-relative one — reconstructed
        //    here via `resolve_safe_write_path` since `WriteError::AlreadyExists`
        //    itself carries no path (kept a unit variant so `web.rs`'s exhaustive
        //    match needs no change to accommodate this). `dest_path` is `None`
        //    here, so this is the arm that runs.
        //
        // 2. `edit_document` was called with `new_path` set (a MOVE, possibly
        //    combined with a content edit) and the DESTINATION already exists —
        //    previously impossible for `edit_document`, which had no way to
        //    reach `AlreadyExists` before moves existed. `rel_path` here is the
        //    move's SOURCE (which legitimately exists — that's what made this an
        //    edit rather than a create), so reporting the collision against
        //    `rel_path` would misdirect the caller at the wrong file. `dest_path`
        //    disambiguates: when it's `Some`, this arm names the DESTINATION as
        //    what collided, not the source.
        WriteError::AlreadyExists => {
            if let Some(dest) = dest_path {
                let abs_dest = crate::write::resolve_safe_write_path(canonical_data_path, dest)
                    .unwrap_or_else(|_| {
                        canonical_data_path.join(crate::retrieval::kb_root_relative(dest))
                    });
                McpError::invalid_params(
                    format!(
                        "Cannot move '{}' to '{}': the destination '{}' already exists. \
                         A move never overwrites — choose a different new_path, or delete \
                         or move the document already at the destination first.",
                        rel_path,
                        dest,
                        abs_dest.display()
                    ),
                    None,
                )
            } else {
                let abs_path = crate::write::resolve_safe_write_path(canonical_data_path, rel_path)
                    .unwrap_or_else(|_| {
                        canonical_data_path.join(crate::retrieval::kb_root_relative(rel_path))
                    });
                McpError::invalid_params(
                    format!(
                        "File '{}' already exists; use edit_document to modify it",
                        abs_path.display()
                    ),
                    None,
                )
            }
        }
        WriteError::NotFound => {
            McpError::invalid_params(format!("File '{}' does not exist", rel_path), None)
        }
        WriteError::StaleHash { expected, actual } => McpError::invalid_params(
            format!(
                "'{}' has changed since you read it: expected content_hash '{}' \
                 but the current document hash is '{}'. Re-read it with \
                 get_document and reapply your edit against the current content.",
                rel_path, expected, actual
            ),
            None,
        ),
        WriteError::PreCommitFailed {
            rolled_back: true,
            msg,
        } => McpError::internal_error(
            format!(
                "'{}' was not {}: git commit failed and the attempted change has \
                 been rolled back — nothing changed, safe to retry. Cause: {}",
                rel_path,
                if is_create { "created" } else { "edited" },
                msg
            ),
            outcome_data(WriteOutcome::FailedNoChange),
        ),
        WriteError::PreCommitFailed {
            rolled_back: false,
            msg,
        } => McpError::internal_error(
            format!(
                "'{}' is in an INCONSISTENT state: git commit failed AND the \
                 rollback attempt itself failed. The working tree may not \
                 match git history for this path — do not assume this \
                 operation did or did not take effect. Manual inspection is \
                 required. {}",
                rel_path, msg
            ),
            outcome_data(WriteOutcome::FailedInconsistentState),
        ),
        WriteError::Io { msg } => McpError::internal_error(msg, None),
    }
}

/// Map a successful `write::delete_document` result onto this tool surface's
/// `CallToolResult`, preserving the exact text/`structured_content` shape the
/// existing delete tests pin down.
fn delete_success_to_result(success: WriteSuccess, rel_path: &str) -> CallToolResult {
    match success.outcome {
        CoreWriteOutcome::Synced => {
            let summary = format!(
                "Deleted '{}' (commit {}). Index cleanup has been queued and will complete shortly.",
                rel_path, success.sha
            );
            let mut result_text = summary;
            if !success.diff.is_empty() {
                result_text = format!("{}\n\n{}", result_text, success.diff);
            }
            with_outcome(
                CallToolResult::success(vec![Content::text(result_text)]),
                WriteOutcome::Synced,
            )
        }
        CoreWriteOutcome::CommittedPendingSync => {
            let cause = success
                .sync_failure_cause
                .as_deref()
                .unwrap_or("unknown error");
            let summary = format!(
                "Deleted '{}' (commit {}) — committed locally, but the push to the remote \
                 failed: {}. It will sync on the next successful write or manual \
                 intervention. Index cleanup has been queued from the local copy.",
                rel_path, success.sha, cause
            );
            let mut result_text = summary;
            if !success.diff.is_empty() {
                result_text = format!("{}\n\n{}", result_text, success.diff);
            }
            with_outcome(
                CallToolResult::success(vec![Content::text(result_text)]),
                WriteOutcome::CommittedPendingSync,
            )
        }
    }
}

/// Map a `write::delete_document` failure onto this tool surface's `McpError`,
/// preserving the exact text/data shapes the existing delete tests pin down.
fn delete_error_to_mcp_error(err: WriteError, rel_path: &str) -> McpError {
    match err {
        WriteError::InvalidCommitMessage { reason } => McpError::invalid_params(reason, None),
        WriteError::UnsafePath { msg } => McpError::invalid_params(msg, None),
        // See `create_edit_error_to_mcp_error`'s identical arm: same text MCP
        // callers already saw before `WriteError::Internal` existed.
        WriteError::Internal { msg } => McpError::invalid_params(msg, None),
        WriteError::NotFound => {
            McpError::invalid_params(format!("document does not exist: '{}'", rel_path), None)
        }
        WriteError::PreCommitFailed {
            rolled_back: true,
            msg,
        } => McpError::internal_error(
            format!(
                "'{}' was NOT deleted: git commit failed and the file has been \
                 restored from HEAD — nothing changed, safe to retry. \
                 Cause: {}",
                rel_path, msg
            ),
            outcome_data(WriteOutcome::FailedNoChange),
        ),
        WriteError::PreCommitFailed {
            rolled_back: false,
            msg,
        } => McpError::internal_error(
            format!(
                "'{}' is in an INCONSISTENT state: git commit failed AND the \
                 attempt to restore the file from HEAD also failed. The file is \
                 gone from disk but was never committed as deleted — do not \
                 assume it exists or that the deletion is durable. Manual \
                 inspection is required. {}",
                rel_path, msg
            ),
            outcome_data(WriteOutcome::FailedInconsistentState),
        ),
        WriteError::Io { msg } => McpError::internal_error(msg, None),
        // `write::delete_document` never produces these — they are create/edit-only
        // failure modes (schema-frozen check, frontmatter validation, the dedup
        // gate, and create-vs-exists) that the delete pipeline doesn't run.
        other => McpError::internal_error(format!("unexpected write error: {:?}", other), None),
    }
}

/// Map a successful `write::move_directory` result onto this tool surface's
/// `CallToolResult`. Mirrors `create_edit_success_to_result`'s two-outcome shape
/// (`Synced` / `CommittedPendingSync`), scaled to a whole batch of documents: the
/// summary line names how many documents moved and lists every `old -> new` pair,
/// plus the same rewrite-note addendum `create_edit_success_to_result` uses for a
/// single-document move's incoming-link rewrites.
fn move_directory_success_to_result(
    success: DirectoryMoveSuccess,
    source_dir: &str,
    dest_dir: &str,
) -> CallToolResult {
    let rewrite_note = if success.rewritten_paths.is_empty() {
        String::new()
    } else {
        format!(
            "\n\nUpdated links in {} document(s) outside the moved subtree: {}.",
            success.rewritten_paths.len(),
            success.rewritten_paths.join(", ")
        )
    };
    let moved_lines = success
        .moved
        .iter()
        .map(|(old, new)| format!("  {} -> {}", old, new))
        .collect::<Vec<_>>()
        .join("\n");
    let moved_json = success
        .moved
        .iter()
        .map(|(old, new)| serde_json::json!({ "from": old, "to": new }))
        .collect::<Vec<_>>();

    let (outcome, summary) = match success.outcome {
        CoreWriteOutcome::Synced => (
            WriteOutcome::Synced,
            format!(
                "Moved {} document(s) from '{}' to '{}' (commit {}). Indexing has been \
                 queued and will complete shortly.\n\n{}{}",
                success.moved.len(),
                source_dir,
                dest_dir,
                success.sha,
                moved_lines,
                rewrite_note
            ),
        ),
        CoreWriteOutcome::CommittedPendingSync => {
            let cause = success
                .sync_failure_cause
                .as_deref()
                .unwrap_or("unknown error");
            (
                WriteOutcome::CommittedPendingSync,
                format!(
                    "Moved {} document(s) from '{}' to '{}' (commit {}) — committed \
                     locally, but the push to the remote failed: {}. It will sync on the \
                     next successful write or manual intervention. Indexing has been \
                     queued from the local copy.\n\n{}{}",
                    success.moved.len(),
                    source_dir,
                    dest_dir,
                    success.sha,
                    cause,
                    moved_lines,
                    rewrite_note
                ),
            )
        }
    };

    let mut result = CallToolResult::success(vec![Content::text(summary)]);
    result.structured_content = Some(serde_json::json!({
        "outcome": outcome.as_str(),
        "moved": moved_json,
        "rewritten_paths": success.rewritten_paths,
    }));
    result
}

/// Map a `write::move_directory` failure onto this tool surface's `McpError`.
/// Mirrors `create_edit_error_to_mcp_error`'s shape, scaled to
/// `DirectoryMoveError`'s directory-move-specific variants.
fn move_directory_error_to_mcp_error(
    err: DirectoryMoveError,
    source_dir: &str,
    dest_dir: &str,
) -> McpError {
    match err {
        DirectoryMoveError::SourceEmpty { msg } => {
            McpError::invalid_params(format!("Cannot move '{}': {}", source_dir, msg), None)
        }
        DirectoryMoveError::AlreadyExists => McpError::invalid_params(
            format!(
                "Cannot move '{}' to '{}': the destination already has at least one file \
                 living under it. A directory move never merges into or overwrites an \
                 existing prefix — choose a different destination, or clear it first.",
                source_dir, dest_dir
            ),
            None,
        ),
        DirectoryMoveError::BrokenSchemaInSource { path, reason } => McpError::invalid_params(
            format!(
                "Cannot move '{}': the schema file '{}' under the source subtree is invalid \
                 ({}). A {} that cannot be read cannot be verified safe to relocate — moving \
                 documents governed by rules this process cannot parse is exactly the case \
                 this refuses. Fix it (or remove it) before retrying the move.",
                source_dir,
                path,
                reason,
                crate::schema::SCHEMA_FILE_NAME
            ),
            None,
        ),
        DirectoryMoveError::Frozen { reason } => McpError::invalid_params(
            format!(
                "Cannot move '{}' to '{}': the schema governing one of the directories \
                 involved is invalid ({}). Fix {} before moving.",
                source_dir,
                dest_dir,
                reason,
                crate::schema::SCHEMA_FILE_NAME
            ),
            None,
        ),
        DirectoryMoveError::Validation {
            failures,
            moved_schema_files,
        } => {
            let summary = failures
                .iter()
                .map(|(path, result)| format!("{}: {}", path, result.errors.join("; ")))
                .collect::<Vec<_>>()
                .join(" | ");
            let schema_note = if moved_schema_files.is_empty() {
                String::new()
            } else {
                let relocated = moved_schema_files
                    .iter()
                    .map(|(old, new)| format!("{} -> {}", old, new))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "\n\nThis subtree carries its own {}, which is relocating along with it \
                     ({}). That means these documents are being checked against a \
                     GENUINELY DIFFERENT schema cascade than the one that governed them at \
                     the source — a relocated schema file re-parents onto the destination's \
                     ancestors, not the source's, so a document that was valid moments ago \
                     can legitimately stop being valid. Either adjust the destination's \
                     cascade to still admit these documents, fix the documents themselves, \
                     or move the schema file separately first.",
                    crate::schema::SCHEMA_FILE_NAME,
                    relocated
                )
            };
            McpError::invalid_params(
                format!(
                    "Cannot move '{}' to '{}': frontmatter validation against the \
                     DESTINATION's schema cascade failed for {} document(s): {}{}",
                    source_dir,
                    dest_dir,
                    failures.len(),
                    summary,
                    schema_note
                ),
                Some(serde_json::json!({
                    "failures": failures.iter().map(|(path, result)| serde_json::json!({
                        "path": path,
                        "field_errors": result.field_errors,
                    })).collect::<Vec<_>>(),
                    "moved_schema_files": moved_schema_files.iter().map(|(old, new)| serde_json::json!({
                        "from": old,
                        "to": new,
                    })).collect::<Vec<_>>(),
                })),
            )
        }
        DirectoryMoveError::UnsafePath { msg } => McpError::invalid_params(msg, None),
        DirectoryMoveError::Internal { msg } => McpError::invalid_params(msg, None),
        DirectoryMoveError::InvalidCommitMessage { reason } => {
            McpError::invalid_params(reason, None)
        }
        DirectoryMoveError::PreCommitFailed {
            rolled_back: true,
            msg,
        } => McpError::internal_error(
            format!(
                "Directory move from '{}' to '{}' was NOT applied: git commit failed and \
                 every document has been rolled back — nothing changed, safe to retry. \
                 Cause: {}",
                source_dir, dest_dir, msg
            ),
            outcome_data(WriteOutcome::FailedNoChange),
        ),
        DirectoryMoveError::PreCommitFailed {
            rolled_back: false,
            msg,
        } => McpError::internal_error(
            format!(
                "Directory move from '{}' to '{}' is in an INCONSISTENT state: git commit \
                 failed AND the rollback attempt itself failed. Filesystem and git state \
                 may not match each other — do not assume this move did or did not take \
                 effect. Manual inspection is required. {}",
                source_dir, dest_dir, msg
            ),
            outcome_data(WriteOutcome::FailedInconsistentState),
        ),
        DirectoryMoveError::Io { msg } => McpError::internal_error(msg, None),
    }
}

/// Outcome of a `write_raw_file` call whose commit actually landed in local
/// history. The two `WriteOutcome` *failure* variants (`FailedNoChange`,
/// `FailedInconsistentState`) are deliberately NOT modeled here — `write_raw_file`
/// reports those directly as `Err(McpError)` (via `outcome_data`), exactly like
/// `write_document`/`delete_document` do, rather than folding every case into one
/// return type.
///
/// `update_schema` (the only caller) matches on this to decide the `WriteOutcome`
/// discriminant and message text to hand back to its own caller. The schema-cache
/// rebuild and `reindex::mark_full` queuing happen identically for both variants
/// inside `write_raw_file` itself — the commit is durable in local history either
/// way, only the remote push status differs — so `update_schema` does not need to
/// (and must not) branch on this to decide whether to do those.
enum RawFileOutcome {
    /// Committed and pushed.
    Synced { sha: String },
    /// Committed locally; the remote push failed. `cause` is the redacted,
    /// already-`{:#}`-formatted `CommitSyncError` source, ready to interpolate
    /// into a user-facing message.
    CommittedPendingSync { sha: String, cause: String },
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
    /// The dirty-path queue every write tool marks paths on (see
    /// `write::WriteDeps::queue`) and `write_raw_file`/`update_schema` call
    /// `mark_full` on directly. `server::run_server` constructs exactly one
    /// `ReindexQueue`, clones this `Arc` into `KbSearchServer`, `UiState`,
    /// `WebhookState`, and `reindex::run_worker` — all four MUST share the same
    /// instance, since the worker only ever drains the one it was handed. There
    /// is no ambient/global fallback (see `reindex::ReindexQueue`'s doc
    /// comment); a server built without one has no way to get indexing to
    /// happen at all, which is deliberate — it makes "which queue does this
    /// producer feed" a constructor argument instead of a runtime assumption.
    reindex_queue: Arc<crate::reindex::ReindexQueue>,
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
        reindex_queue: Arc<crate::reindex::ReindexQueue>,
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
            reindex_queue,
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
    /// failure part-way through the *filesystem* write cannot leave a half-written
    /// schema that would freeze the scope.
    ///
    /// `commit_and_sync` below has its own two-phase failure mode — see
    /// `git::CommitSyncError` — and is rolled back exactly like `write_document`'s: a
    /// `PreCommit` failure undoes the filesystem write (remove + `unstage` for a
    /// brand-new schema file that has no HEAD content to fall back to;
    /// `restore_from_head` for an overwrite of an existing, already-tracked one) and
    /// reports `FailedNoChange`, or `FailedInconsistentState` if that rollback itself
    /// fails. A `PostCommit` failure leaves the local commit in place — it is real —
    /// and reports `CommittedPendingSync`.
    ///
    /// `reindex::mark_full` fires only once the commit has actually landed locally
    /// (`Synced` or `CommittedPendingSync`), never on a rolled-back write: queuing a
    /// full reconcile against a schema change that was never actually committed (or,
    /// worse, against a filesystem/git state a failed rollback left inconsistent)
    /// would be pointless at best and actively misleading at worst — the reconcile
    /// would revalidate every document under the scope against content that is not,
    /// in fact, what's in git history.
    async fn write_raw_file(
        &self,
        rel_path: &str,
        content: &str,
        commit_message: &str,
    ) -> Result<RawFileOutcome, McpError> {
        let config = self.config();

        // Same resolver the document write tools use. Joining the data root with a
        // caller-supplied path is NOT sufficient on its own: the knowledge base is a
        // synced git repo, and git materializes tracked symlinks on checkout, so a
        // hostile upstream commit could otherwise redirect this write outside the KB.
        let abs_path = crate::write::resolve_safe_write_path(&self.canonical_data_path, rel_path)
            .map_err(|e| {
            McpError::invalid_params(format!("Invalid schema path: {}", e), None)
        })?;

        if let Some(parent) = abs_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                error!("Failed to create directory '{}': {}", parent.display(), e);
                McpError::internal_error(format!("Failed to create directory: {}", e), None)
            })?;
        }

        // Re-check after creating the directory: `resolve_safe_write_path` can only
        // canonicalize ancestors that existed at the time, so a newly created path
        // component is verified here.
        crate::write::resolve_safe_write_path(&self.canonical_data_path, rel_path)
            .map_err(|e| McpError::invalid_params(format!("Invalid schema path: {}", e), None))?;

        // Whether this call is creating `rel_path` for the first time or overwriting
        // an existing, already-tracked one — determines which rollback primitive
        // applies if `commit_and_sync` fails before landing (see the match below).
        // Checked as late as possible, immediately before the write, to keep the
        // TOCTOU window against a concurrent writer as small as the temp+rename
        // strategy below allows.
        let is_new = !abs_path.exists();

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

        // `commit_and_sync` distinguishes WHERE it failed — see `git::CommitSyncError`
        // — and the two phases demand opposite handling, exactly as in
        // `write_document`. A `PreCommit` failure means HEAD never moved, so the
        // filesystem write above is rolled back and reported as "nothing changed". A
        // `PostCommit` failure means the commit is a real, durable part of local
        // history — rolling it back here would silently undo a schema change that
        // genuinely happened, so it is left alone and reported as "committed, sync
        // pending" instead.
        let data_path_str = self.canonical_data_path.to_str().unwrap_or_default();

        // Held across the commit AND any rollback below — see `write::write_document`.
        // A schema write races document writes and the webhook for the same index.
        let git_lock = git::lock_git().await;

        let commit_outcome = match git::commit_and_sync(
            &git_lock,
            config.source.git_url.as_deref(),
            &config.source.branch,
            data_path_str,
            token.as_deref(),
            &[rel_path],
            commit_message,
            &config.write.commit_author_name,
            &config.write.commit_author_email,
        )
        .await
        {
            Ok(outcome) => outcome,

            Err(git::CommitSyncError::PreCommit(source)) => {
                error!(
                    "commit_and_sync pre-commit failure writing schema '{}', rolling back: {:#}",
                    rel_path, source
                );

                // For a brand-new schema file, there is no HEAD content to restore
                // to — remove it from disk directly and unstage whatever `git add`
                // staged. For an overwrite of an existing schema, HEAD already has
                // the previous content, so restore it (this also un-stages any
                // partial `git add`, in one step).
                let rollback = if is_new {
                    match tokio::fs::remove_file(&abs_path).await {
                        Ok(()) => git::unstage(&git_lock, data_path_str, rel_path).await,
                        Err(e) => Err(anyhow::Error::new(e)
                            .context("Failed to remove newly-written schema file during rollback")),
                    }
                } else {
                    git::restore_from_head(&git_lock, data_path_str, rel_path).await
                };

                return match rollback {
                    Ok(()) => Err(McpError::internal_error(
                        format!(
                            "Schema at '{}' was NOT changed: git commit failed and the write \
                             has been rolled back — nothing changed, safe to retry. \
                             Cause: {:#}",
                            rel_path, source
                        ),
                        outcome_data(WriteOutcome::FailedNoChange),
                    )),
                    // The rollback ITSELF failed — a third, worse state than either of
                    // the above. The schema file may now be gone/changed on disk with
                    // no corresponding commit, or the index may not match HEAD.
                    // Report it distinctly and loudly rather than letting it
                    // masquerade as a clean no-op.
                    Err(rollback_err) => {
                        error!(
                            "Rollback FAILED after a pre-commit git failure writing schema \
                             '{}': {:#}. Original cause: {:#}. Filesystem and git state may \
                             now be inconsistent.",
                            rel_path, rollback_err, source
                        );
                        Err(McpError::internal_error(
                            format!(
                                "Schema at '{}' is in an INCONSISTENT state: git commit \
                                 failed AND the rollback attempt itself failed. The working \
                                 tree may not match git history for this path — do not \
                                 assume the schema change did or did not take effect. Manual \
                                 inspection is required. Commit cause: {:#}. \
                                 Rollback cause: {:#}",
                                rel_path, source, rollback_err
                            ),
                            outcome_data(WriteOutcome::FailedInconsistentState),
                        ))
                    }
                };
            }

            Err(git::CommitSyncError::PostCommit { sha, source }) => {
                warn!(
                    "commit_and_sync post-commit (sync) failure writing schema '{}', commit {} \
                     stands uncorrected: {:#}",
                    rel_path, sha, source
                );

                // The commit landed locally regardless of push status, so the schema
                // change is real and durable as far as this clone's git history is
                // concerned — queue the same full reconcile a clean success would.
                // See this method's doc comment for why that reconcile must NOT run
                // on the rolled-back (PreCommit) branch above but must here.
                self.reindex_queue.mark_full();

                return Ok(RawFileOutcome::CommittedPendingSync {
                    sha,
                    cause: format!("{:#}", source),
                });
            }
        };

        // A schema change revalidates its whole subtree via the schema fingerprint —
        // any document under this scope can flip from valid to invalid or vice versa —
        // and there is no cheap way to enumerate exactly which paths that touches
        // without a walk. Rather than approximate it, mark a full reconcile: the
        // worker will scan, and `index_paths`' existing schema-fingerprint check
        // (unrelated to this reconcile's OWN full-walk vs scoped distinction) is what
        // actually catches the affected documents once it re-reads them.
        self.reindex_queue.mark_full();

        Ok(RawFileOutcome::Synced {
            sha: commit_outcome.sha,
        })
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

        let config = self.config();
        let limit = resolve_limit(
            params.limit,
            config.search.default_limit,
            config.search.max_limit,
        );

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
            diversity_max_per_document: config.search.diversity_max_per_document,
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

        // Root-scope mutations need an explicit opt-in (see `acknowledge_root_change`'s
        // doc comment). `remove_values` is exempt — shrinking the root vocabulary is the
        // policy-aligned direction, and the casualty check below already guards it — and
        // `dry_run` is exempt everywhere, since it writes nothing. This must run before
        // `file.apply()` below: the point is to refuse before any edit is computed or
        // written, not merely before the commit.
        let is_root = rel_dir.as_os_str().is_empty();
        let is_gated_op = !matches!(edit, crate::schema::SchemaEdit::RemoveValues { .. });
        let dry_run_requested = params.dry_run.unwrap_or(false);
        let acknowledged = params.acknowledge_root_change.unwrap_or(false);
        if is_root && is_gated_op && !dry_run_requested && !acknowledged {
            return Err(McpError::invalid_params(
                "root schema changes are guarded: the root tag vocabulary is \
                 identity-only by policy (see meta/schema-tag-policy.md in this \
                 knowledge base). Pass acknowledge_root_change=true only if this change \
                 is a deliberate design decision consistent with that policy."
                    .to_string(),
                None,
            ));
        }

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

        // `write_raw_file` rolls itself back on a pre-commit `commit_and_sync`
        // failure and returns `Err` in that case (see its doc comment) — this `?`
        // propagates that `Err` (with its `FailedNoChange`/`FailedInconsistentState`
        // outcome data already attached) WITHOUT reaching the cache rebuild below.
        // That is exactly what must happen: a rolled-back write means the schema on
        // disk is unchanged (or, in the inconsistent-state case, of unknown
        // trustworthiness), so rebuilding the shared cache from it here would either
        // be a no-op at best or propagate bad state at worst. Only a call that
        // actually landed a local commit — `Synced` or `CommittedPendingSync` —
        // reaches the code below.
        let write_outcome = self
            .write_raw_file(&rel_file_str, &yaml, &commit_message)
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
        //
        // This runs for BOTH `RawFileOutcome` variants, not just `Synced`: a
        // `CommittedPendingSync` write is still a real local commit — the new schema
        // is genuinely in effect for this clone regardless of whether the push to the
        // remote landed — so the cache must reflect it just the same.
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

        // Same `WriteOutcome` discriminant `write_document`/`delete_document` attach
        // via `with_outcome` — not called directly here because this response also
        // carries schema-specific fields (`summary`, `path`, `invalidated`) that
        // `with_outcome` would clobber, but the discriminant string itself comes from
        // the same enum, not a parallel literal.
        let (outcome, mut text) = match write_outcome {
            RawFileOutcome::Synced { sha } => (
                WriteOutcome::Synced,
                format!("{summary}\nWrote {rel_file_str} (commit {sha})."),
            ),
            RawFileOutcome::CommittedPendingSync { sha, cause } => (
                WriteOutcome::CommittedPendingSync,
                format!(
                    "{summary}\nWrote {rel_file_str} (commit {sha}) — committed locally, but \
                     the push to the remote failed: {cause}. It will sync on the next \
                     successful write or manual intervention.",
                ),
            ),
        };
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
            "outcome": outcome.as_str(),
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
        markdown including frontmatter, in both the text content and \
        structured_content.content. \
        Pass start_line and/or end_line (1-based, inclusive) to read only part of \
        a long document; structured_content always reports start_line, end_line, \
        total_lines, and partial, so you can tell what you got and page through \
        the rest. \
        structured_content.content_hash is a SHA-256 hex digest of the WHOLE \
        document, not of the returned slice — pass it as edit_document's \
        expected_hash to guard against editing stale content, whether you read the \
        document in full or in part."
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

        // Checked before the path is resolved: a malformed range is wrong no
        // matter which document it was aimed at, so it should not cost a
        // metadata-index open and a file read to say so.
        let range = retrieval::LineRange::new(params.start_line, params.end_line)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;

        debug!(path = %raw, ?range, "get_document called");

        // The fuzzy-basename fallback resolves against the SQLite metadata index
        // rather than a Qdrant facet fetch, so the index has to be opened here.
        // Only the fallback needs it — an exact path hit is served from disk and
        // never touches this — but the resolve happens inside `get_document`, so
        // it is passed unconditionally.
        let index = self.state_db().await.map_err(|e| {
            error!("get_document could not open the metadata index: {:#}", e);
            McpError::internal_error(format!("Document index unavailable: {}", e), None)
        })?;

        match retrieval::get_document(&self.deps(), index, raw).await {
            Ok(doc) => {
                debug!(path = %raw, "get_document served");
                // Same hash indexed_files.content_hash already stores for this exact
                // content, so a caller can round-trip it straight into edit_document's
                // expected_hash without this project introducing a second hash scheme.
                // Hence hashing here, before any slicing, and always over the whole
                // file: that expected_hash guards the document on disk, so hashing a
                // slice would hand back a token that can never match — turning every
                // partial read into a dead end for the edit that motivated it.
                let content_hash = crate::ingest::compute_hash_from_bytes(doc.content.as_bytes());
                let slice = retrieval::slice_or_whole(doc.content, range.as_ref())
                    .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
                // structured_content must mirror the text block: MCP clients that
                // prefer structuredContent render ONLY it, so a hash-only payload
                // makes the document invisible to them (observed in practice).
                //
                // The line fields are reported unconditionally, including on a full
                // read, so a client never has to branch on their presence to learn
                // how much document it is holding.
                let structured = serde_json::json!({
                    "path": retrieval::relative_to_data(
                        &doc.path.to_string_lossy(),
                        &self.canonical_data_path,
                    ),
                    "content": &slice.content,
                    "content_hash": content_hash,
                    "start_line": slice.start_line,
                    "end_line": slice.end_line,
                    "total_lines": slice.total_lines,
                    "partial": slice.partial(),
                });
                let mut result = CallToolResult::success(vec![Content::text(slice.content)]);
                result.structured_content = Some(structured);
                Ok(result)
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
    /// Thin adapter over `write::write_document`: builds `WriteDeps`/`WriteRequest`
    /// from this server's fields and the current config snapshot, then maps the
    /// structured `WriteSuccess`/`WriteError` back onto this tool surface's exact
    /// `CallToolResult`/`McpError` shapes (see `create_edit_success_to_result` /
    /// `create_edit_error_to_mcp_error`).
    ///
    /// Callers are responsible for resolving paths and computing old/new content
    /// before calling this — see `write::WriteRequest`'s doc comment.
    ///
    /// * `old_content` – empty string for create; existing file bytes for edit.
    /// * `new_content` – the content to write (already computed by caller).
    /// * `rel_path`    – repo-relative path (used for git add/commit and messages).
    /// * `is_create`   – `true` for create (dedup gate active), `false` for edit.
    /// * `message`     – optional custom commit message.
    /// * `default_verb`– verb for the default commit message, e.g. `"add"` or `"update"`.
    /// * `force_new`   – when `Some(true)`, bypasses the dedup gate on create paths.
    /// * `operation`   – label for the `Operation:` git trailer, e.g. `"create_document"`.
    /// * `dest_path`   – when `Some`, turns this into a document MOVE: `rel_path` is
    ///   the source, this is the destination. `None` (the default for every
    ///   `create_document` call, and for a plain `edit_document` call with no
    ///   `new_path`) is the existing create/edit behavior. See
    ///   `write::WriteRequest::dest_path`.
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
        dest_path: Option<&str>,
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

        let token = std::env::var(&config.source.git_token_env)
            .ok()
            .filter(|s| !s.is_empty());

        // Only opened for a MOVE (`dest_path.is_some()`) — `write::write_document`
        // itself never reads `deps.state` outside `write_document_move`, so a
        // plain create/edit has no use for it, and opening the state DB lazily
        // here (rather than unconditionally on every write) avoids materializing
        // `state.db` on disk for calls that will never touch it. Best-effort: a
        // state DB that fails to open degrades a move to "without link
        // rewriting" (see `WriteDeps::state`'s doc comment) rather than failing
        // the write.
        let state_db = if dest_path.is_some() {
            self.state_db().await.ok()
        } else {
            None
        };

        let deps = WriteDeps {
            retrieval: self.deps(),
            canonical_data_path: &self.canonical_data_path,
            schema_cache: &self.schema_cache,
            validation: &config.validation,
            prepend_description: config.chunking.prepend_description,
            dedup_enabled: config.write.dedup_enabled,
            dedup_threshold: config.write.dedup_threshold,
            git_url: config.source.git_url.as_deref(),
            branch: &config.source.branch,
            token: token.as_deref(),
            commit_author_name: &config.write.commit_author_name,
            commit_author_email: &config.write.commit_author_email,
            queue: &self.reindex_queue,
            state: state_db,
        };

        let req = WriteRequest {
            rel_path,
            old_content,
            new_content,
            is_create,
            message,
            default_verb,
            force_new,
            operation,
            // `edit_document` already enforces the stale-read guard itself, ahead
            // of applying a surgical old_string/new_string replacement — so a
            // stale read surfaces as an explicit "changed since you read it"
            // error rather than a confusing old_string-not-found one. Passing
            // `None` here avoids redundantly re-hashing the same in-memory
            // `old_content` a second time; it can never disagree with the first
            // check since both compare against the identical string.
            expected_hash: None,
            // Threaded straight from this method's own `dest_path` parameter —
            // `Some` turns this call into a move. See
            // `write::WriteRequest::dest_path`.
            dest_path,
        };

        match crate::write::write_document(&deps, req).await {
            Ok(success) => Ok(create_edit_success_to_result(success, rel_path, is_create)),
            Err(err) => Err(create_edit_error_to_mcp_error(
                err,
                rel_path,
                is_create,
                &self.canonical_data_path,
                dest_path,
            )),
        }
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
        let abs_path = crate::write::resolve_safe_write_path(&data_root, &params.path)
            .map_err(|e| McpError::invalid_params(e, None))?;

        // The include-pattern eligibility guard is enforced inside
        // `write::write_document` itself too — see that module's
        // `check_include_pattern` — so every caller of the shared write pipeline
        // (this tool, `edit_document`/`delete_document` via `resolve_within_data`,
        // and the HTTP UI in `web.rs`) gets it for free instead of each transport
        // maintaining its own copy that a future caller could forget. It is ALSO
        // run here, explicitly, ahead of the `exists()` pre-check just below:
        // pre-refactor, a path that both exists on disk and fails this check was
        // reported with the include-pattern message, not "already exists; use
        // edit_document" — which, for such a path, is misleading circular
        // guidance, since edit_document would then reject the same path as not
        // permitted. Running `write::write_document`'s check later is not
        // sufficient to restore that priority on its own, since the `exists()`
        // check below returns before ever reaching it. Same message text as
        // `write::write_document`'s own check (see `create_edit_error_to_mcp_error`'s
        // `UnsafePath` arm), so existing callers see the exact same wording.
        crate::write::check_include_pattern_against(&self.include_patterns, &params.path).map_err(
            |e| {
                create_edit_error_to_mcp_error(
                    e,
                    &params.path,
                    true,
                    &self.canonical_data_path,
                    None,
                )
            },
        )?;

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
            None, // create_document never moves a document
        )
        .await
    }

    #[tool(
        description = "Edit an existing document in the knowledge base, optionally also \
        relocating it. Content changes and relocation are independent axes: pick zero or one \
        content mode, and independently choose whether to also move the document.\n\
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
        MOVE MODE — provide new_path to relocate the document to a different repo-relative \
        path, in the same commit as any content change:\n\
        Unlike SURGICAL and FULL-REPLACE, which are mutually exclusive with EACH OTHER, MOVE \
        may be combined with either one — or with neither, for a pure relocation that leaves \
        content unchanged. Pass new_path alone to move the document as-is; pass it together \
        with old_string+new_string or content to fix up the document and relocate it in one \
        atomic commit. You never need to re-send the document body just to move it — the \
        server reads its current content itself. The destination (new_path) must not already \
        exist; this tool never overwrites. Frontmatter is validated against the DESTINATION \
        directory's schema, which may differ from the source directory's — call get_schema on \
        the destination path first if you are not sure what it requires. Moving a document \
        also automatically updates every recognized link in OTHER documents that point at it — \
        inline [text](path.md), reference-style [text][ref] and the shortcut [ref] (the \
        [ref]: path.md definition is rewritten once; the use sites are left untouched), \
        wiki-style [[path]] (an extension-less target is treated as path.md), and autolinks \
        <path.md> — committing those documents in the SAME commit as the move; the updated \
        paths are reported back in rewritten_paths. LIMITATION: the wiki pipe-alias form \
        [[path|Display text]] is not recognized as a link at all — write [[path]] without an \
        alias if you need it tracked.\n\
        \n\
        At least one of {content, old_string+new_string, new_path} must be provided — a call \
        with none of them changes nothing and is rejected.\n\
        \n\
        In every mode the result is validated, committed, and queued for indexing in the \
        background (the change becomes searchable shortly after this call returns, not \
        necessarily immediately). The path parameter (the SOURCE) is resolved like \
        get_document: relative to the KB root, a unique basename, or absolute. The document at \
        path must already exist — use create_document for new files. new_path, by contrast, is \
        taken literally as a repo-relative destination and must NOT already exist.\n\
        \n\
        OPTIONAL STALE-READ GUARD — expected_hash:\n\
        Pass the content_hash get_document returned in structured_content when you read this \
        document, and the edit is refused with an explicit error if the file has changed since \
        then, instead of a confusing old_string/content mismatch. Applies to moves too, guarding \
        against a stale read of the source.\n\
        \n\
        SCOPE: this knowledge base holds durable reference knowledge only. NEVER append session \
        notes, task state, or other transient content to a document."
    )]
    async fn edit_document(
        &self,
        Parameters(params): Parameters<EditDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        // Parse and validate the content-edit mode (surgical vs full-replace vs
        // neither). `new_path` is an orthogonal axis handled below, independent of
        // this — see `parse_edit_mode`'s doc comment.
        let mode = parse_edit_mode(&params).map_err(|e| McpError::invalid_params(e, None))?;
        let dest_path = params.new_path.as_deref();

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

        // Optional stale-read guard: if the caller tells us what content it based this
        // edit on, refuse up front when the file has since changed, rather than let a
        // shifted `old_string` fail with a confusing (and, for a full replace, silent)
        // mismatch. Same hash `get_document` reports back as `content_hash` and
        // `indexed_files.content_hash` already uses — see `EditDocumentParams::expected_hash`.
        if let Some(expected) = params.expected_hash.as_deref() {
            let actual = crate::ingest::compute_hash_from_bytes(old_content.as_bytes());
            if !expected.trim().eq_ignore_ascii_case(&actual) {
                return Err(McpError::invalid_params(
                    format!(
                        "'{}' has changed since you read it: expected content_hash '{}' \
                         but the current document hash is '{}'. Re-read it with \
                         get_document and reapply your edit against the current content.",
                        rel_path,
                        expected.trim(),
                        actual
                    ),
                    None,
                ));
            }
        }

        // Compute new_content and operation label based on mode. `None` is a pure
        // move (guaranteed by `parse_edit_mode` to only occur when `dest_path` is
        // `Some` — see its doc comment): the destination gets the source's current
        // content, byte-for-byte, since `write::write_document` never reads
        // `rel_path`'s content itself, move or not.
        let (new_content, operation) = match mode {
            None => (old_content.clone(), "edit_document (move)"),
            Some(EditMode::Full { content }) => (
                content,
                if dest_path.is_some() {
                    "edit_document (full replace + move)"
                } else {
                    "edit_document (full replace)"
                },
            ),
            Some(EditMode::Surgical { old, new }) => {
                let result = apply_surgical(&old_content, &old, &new, &rel_path)
                    .map_err(|e| McpError::invalid_params(e, None))?;
                (
                    result,
                    if dest_path.is_some() {
                        "edit_document (surgical replace + move)"
                    } else {
                        "edit_document (surgical replace)"
                    },
                )
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
            dest_path,
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

        // Resolve the path (must already exist on disk). This is the same fuzzy
        // resolver `get_document`/`edit_document` use — relative to the KB root, a
        // unique basename, or absolute — and produces this tool's richer NotFound
        // text. It stays here rather than in `write::delete_document`, which does
        // its own plain existence check as a defense-in-depth fallback for callers
        // (like the HTTP UI) that address a document by exact path instead.
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

        let token = std::env::var(&config.source.git_token_env)
            .ok()
            .filter(|s| !s.is_empty());

        let deps = WriteDeps {
            retrieval: self.deps(),
            canonical_data_path: &self.canonical_data_path,
            schema_cache: &self.schema_cache,
            validation: &config.validation,
            prepend_description: config.chunking.prepend_description,
            dedup_enabled: config.write.dedup_enabled,
            dedup_threshold: config.write.dedup_threshold,
            git_url: config.source.git_url.as_deref(),
            branch: &config.source.branch,
            token: token.as_deref(),
            commit_author_name: &config.write.commit_author_name,
            commit_author_email: &config.write.commit_author_email,
            queue: &self.reindex_queue,
            // `delete_document` never moves a document — `write::write_document_move`
            // is the only reader of `WriteDeps::state` — so there is nothing here
            // for a state DB to do. `None` per `WriteDeps::state`'s doc comment.
            state: None,
        };

        match crate::write::delete_document(&deps, &rel_path, params.message.as_deref()).await {
            Ok(success) => Ok(delete_success_to_result(success, &rel_path)),
            Err(err) => Err(delete_error_to_mcp_error(err, &rel_path)),
        }
    }

    #[tool(
        description = "Relocate every document under a source directory prefix to a \
        destination prefix, as ONE atomic commit. Use this to reorganize a whole subtree \
        at once — for a single document, use edit_document's MOVE MODE instead; this tool \
        has no content-editing ability (no body, no expected_hash, no surgical/full-replace \
        modes).\n\
        \n\
        ALL-OR-NOTHING: every moved document's frontmatter is validated against the \
        DESTINATION path's schema (which may differ per document, since each keeps its \
        position under the destination prefix) BEFORE anything is written. If even one \
        document fails that validation, the whole move is refused and nothing is \
        mutated — the response names every document that failed, not just the first.\n\
        \n\
        PRECONDITIONS: source_path must exist and contain at least one indexable document. \
        dest_path must not already have any file living under it — this tool never merges \
        into or overwrites an existing prefix.\n\
        \n\
        SCHEMA FILES: a source subtree containing its own .kb-schema.yaml is supported — the \
        schema file moves along with the documents it governs. But relocating it re-parents \
        its cascade: its own declarations are unchanged, but they now merge onto whatever \
        governs the DESTINATION instead of the source, per field and per attribute. If the \
        destination's ancestors declare different required fields, values sets, defaults, or \
        types than the source's did, that difference reaches every document under the moved \
        subtree — so a document that was valid at the source can legitimately fail validation \
        at the destination, even though its own content and its own schema file's declarations \
        never changed. This is checked BEFORE anything is written (see ALL-OR-NOTHING above), \
        and the error names which schema file relocated and why. An unparseable \
        .kb-schema.yaml anywhere in the source subtree blocks the move outright — rules that \
        cannot be read cannot be verified safe to relocate.\n\
        \n\
        LINK REWRITING: a link between two documents that are BOTH moving keeps pointing \
        at each other post-move; a link to a document that stays in place keeps that exact \
        target, with only its relative spelling updated for the mover's new location. \
        Documents OUTSIDE the moved subtree that link INTO it also have those links \
        rewritten to the new location, committed in the SAME commit as the move — the \
        rewritten paths are reported back in rewritten_paths. Every recognized link syntax \
        is covered: inline [text](path.md), reference-style [text][ref] and the shortcut \
        [ref] (the [ref]: path.md definition is rewritten once; the use sites are left \
        untouched), wiki-style [[path]] (an extension-less target is treated as path.md), \
        and autolinks <path.md>. LIMITATION: the wiki pipe-alias form [[path|Display text]] \
        is not recognized as a link at all — write [[path]] without an alias if you need it \
        tracked.\n\
        \n\
        Returns the number of documents moved (with their old -> new paths) and every \
        rewritten path. Indexing of every moved and rewritten document is queued in the \
        background — the changes become searchable shortly after this call returns, not \
        necessarily immediately."
    )]
    async fn move_directory(
        &self,
        Parameters(params): Parameters<MoveDirectoryParams>,
    ) -> Result<CallToolResult, McpError> {
        let config = self.config();

        let source_dir = params.source_path.trim();
        let dest_dir = params.dest_path.trim();
        if source_dir.is_empty() {
            return Err(McpError::invalid_params(
                "source_path parameter is empty".to_string(),
                None,
            ));
        }
        if dest_dir.is_empty() {
            return Err(McpError::invalid_params(
                "dest_path parameter is empty".to_string(),
                None,
            ));
        }

        let token = std::env::var(&config.source.git_token_env)
            .ok()
            .filter(|s| !s.is_empty());

        // Best-effort, same as `write_document`'s own lazy state-DB open for a
        // single-document MOVE: a state DB that fails to open degrades the
        // incoming-link rewrite to "skip it", not a failed move — see
        // `WriteDeps::state`'s doc comment.
        let state_db = self.state_db().await.ok();

        let deps = WriteDeps {
            retrieval: self.deps(),
            canonical_data_path: &self.canonical_data_path,
            schema_cache: &self.schema_cache,
            validation: &config.validation,
            prepend_description: config.chunking.prepend_description,
            dedup_enabled: config.write.dedup_enabled,
            dedup_threshold: config.write.dedup_threshold,
            git_url: config.source.git_url.as_deref(),
            branch: &config.source.branch,
            token: token.as_deref(),
            commit_author_name: &config.write.commit_author_name,
            commit_author_email: &config.write.commit_author_email,
            queue: &self.reindex_queue,
            state: state_db,
        };

        match crate::write::move_directory(&deps, source_dir, dest_dir, params.message.as_deref())
            .await
        {
            Ok(success) => Ok(move_directory_success_to_result(
                success, source_dir, dest_dir,
            )),
            Err(err) => Err(move_directory_error_to_mcp_error(err, source_dir, dest_dir)),
        }
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
        ui: crate::config::UiConfig::default(),
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
    // These now live in `write.rs` (the dedup gate and commit-message/diff
    // helpers moved there with the rest of the write pipeline); imported here
    // so the tests below — ported verbatim — keep compiling unchanged.
    use crate::write::{
        build_commit_message, build_dedup_query, dedup_search_opts, dedup_verdict,
        render_unified_diff,
    };

    /// Mirrors `write::DEDUP_QUERY_CHAR_LIMIT` (private there), so the ported
    /// `build_dedup_query_*` tests below keep compiling unchanged.
    const DEDUP_QUERY_CHAR_LIMIT: usize = 2000;

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
            acknowledge_root_change: None,
        }
    }

    /// Build a `set_field` definition the way a real MCP client's JSON arrives: through
    /// `FieldDefinitionInput`'s own `Deserialize` impl, not by constructing
    /// `RawFieldDef` directly. Fixtures here are expected to be valid; use
    /// `serde_json::from_value::<FieldDefinitionInput>` directly in tests that assert on
    /// a parse failure.
    fn definition(json: serde_json::Value) -> FieldDefinitionInput {
        serde_json::from_value(json).expect("test fixture must be a valid definition")
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
    fn update_schema_definition_advertises_as_a_typed_object() {
        // This is the regression test for the actual bug: `definition` used to be typed
        // `serde_json::Value`, which schemars turns into an unconstrained `{}` schema —
        // no `type` keyword, no listed properties, nothing telling a client this must be
        // an object. A client with no other signal is then free to encode the value
        // however it likes, including as a JSON-encoded string, which is exactly what
        // happened in practice (see `FieldDefinitionInput`'s doc comment).
        let schema = schemars::schema_for!(UpdateSchemaParams);
        let root = schema.as_value();

        let definition_schema = &root["properties"]["definition"];
        // `Option<FieldDefinitionInput>` becomes `anyOf: [<real schema>, {"type": "null"}]`;
        // find the non-null branch.
        let object_schema = definition_schema["anyOf"]
            .as_array()
            .expect("definition must offer a typed alternative, not a bare {}")
            .iter()
            .find(|branch| branch["type"] != serde_json::json!("null"))
            .expect("definition must have a non-null branch");

        // schemars refs the RawFieldDef schema into `$defs` rather than inlining it;
        // resolve it so the assertions below see the real shape.
        let resolved = match object_schema["$ref"].as_str() {
            Some(reference) => &root["$defs"][reference.rsplit('/').next().unwrap()],
            None => object_schema,
        };

        assert_eq!(
            resolved["type"],
            serde_json::json!("object"),
            "definition must advertise as an object, got: {resolved}"
        );
        assert_eq!(
            resolved["additionalProperties"],
            serde_json::json!(false),
            "an unknown key must be rejected by a conforming client's own schema \
             validation too, not just our runtime check, got: {resolved}"
        );
        for key in [
            "type", "required", "indexed", "values", "extend", "default", "open",
        ] {
            assert!(
                !resolved["properties"][key].is_null(),
                "definition schema is missing documented key '{key}': {resolved}"
            );
        }
    }

    #[test]
    fn set_field_parses_a_definition() {
        let mut params = update_params("set_field", "planning.prep_minutes");
        params.definition = Some(definition(
            serde_json::json!({ "type": "integer", "indexed": true }),
        ));

        match build_schema_edit(&params).unwrap() {
            crate::schema::SchemaEdit::SetField { field, definition } => {
                assert_eq!(field, "planning.prep_minutes");
                assert_eq!(definition.ty, Some(crate::schema::FieldType::Integer));
                assert_eq!(definition.indexed, Some(true));
            }
            other => panic!("expected SetField, got {other:?}"),
        }
    }

    #[test]
    fn set_field_accepts_a_json_encoded_string_as_a_fallback() {
        // At least one real MCP client sends nested-object tool arguments as a
        // JSON-encoded string rather than an object, regardless of what the tool
        // schema advertises. `FieldDefinitionInput` tolerates that as a fallback.
        let mut params = update_params("set_field", "planning.prep_minutes");
        params.definition = Some(definition(serde_json::Value::String(
            r#"{"type":"integer","indexed":true}"#.to_string(),
        )));

        match build_schema_edit(&params).unwrap() {
            crate::schema::SchemaEdit::SetField { field, definition } => {
                assert_eq!(field, "planning.prep_minutes");
                assert_eq!(definition.ty, Some(crate::schema::FieldType::Integer));
                assert_eq!(definition.indexed, Some(true));
            }
            other => panic!("expected SetField, got {other:?}"),
        }
    }

    #[test]
    fn set_field_rejects_a_string_that_is_not_valid_json() {
        let err = serde_json::from_value::<FieldDefinitionInput>(serde_json::Value::String(
            "not json at all".to_string(),
        ))
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not valid JSON"),
            "expected a message explaining the string wasn't parseable JSON, got: {msg}"
        );
    }

    #[test]
    fn set_field_rejects_a_json_array_naming_the_expected_shape() {
        let err =
            serde_json::from_value::<FieldDefinitionInput>(serde_json::json!(["type", "integer"]))
                .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("JSON object"),
            "expected the error to name the expected shape, got: {msg}"
        );
        assert!(
            !msg.contains("RawFieldDef"),
            "a Rust type name is meaningless to an MCP client, got: {msg}"
        );
    }

    #[test]
    fn set_field_rejects_an_unknown_key() {
        let err =
            serde_json::from_value::<FieldDefinitionInput>(serde_json::json!({ "typ": "integer" }))
                .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("typ"),
            "a typo'd key must be named in the error, not just silently dropped: {msg}"
        );
        assert!(
            !msg.contains("RawFieldDef"),
            "a Rust type name is meaningless to an MCP client, got: {msg}"
        );
    }

    #[test]
    fn update_schema_params_reject_a_misspelled_definition_key() {
        // The same check, but through the exact path a real tool call takes: the whole
        // `UpdateSchemaParams` deserialized from one JSON blob, the way rmcp's
        // `Parameters<T>` extractor does it.
        let err = serde_json::from_value::<UpdateSchemaParams>(serde_json::json!({
            "operation": "set_field",
            "field": "tags",
            "definition": { "typ": "integer" },
        }))
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("typ"), "got: {msg}");
        assert!(!msg.contains("RawFieldDef"), "got: {msg}");
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
    async fn get_schema_root_ignores_config_frontmatter_once_a_root_file_exists() {
        // A root .kb-schema.yaml declares `title`. config.yaml ALSO declares a
        // required field (`legacy_field`) that the root file never mentions. Per issue
        // #91's policy, a root schema file is authoritative for the KB root: config's
        // `frontmatter` block must not still be contributing fields through
        // get_schema, and every field it does report must be attributed to the root
        // schema file, not "config.yaml".
        let tmp = tempfile::tempdir().unwrap();
        write_schema_file(&tmp, "", "fields:\n  title:\n    required: true\n");

        let mut config = (*make_test_resolved_config(tmp.path())).clone();
        config.frontmatter.required = vec!["legacy_field".into()];
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], Arc::new(config));

        let result = server
            .get_schema(Parameters(GetSchemaParams::default()))
            .await
            .unwrap();

        let structured = result.structured_content.unwrap();
        let fields = structured["fields"].as_array().unwrap();
        let names: Vec<&str> = fields
            .iter()
            .map(|f| f["field"].as_str().unwrap())
            .collect();

        assert!(
            names.contains(&"title"),
            "the root file's own field is reported"
        );
        assert!(
            !names.contains(&"legacy_field"),
            "a config-only field must not leak into the root once a root schema file exists"
        );
        for field in fields {
            assert_eq!(
                field["declared_in"],
                serde_json::json!(".kb-schema.yaml"),
                "every reported root field must be attributed to the root schema file, \
                 not to config.yaml, once one exists"
            );
        }
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
        // Needs a real git-backed harness (not `schema_tool_server`'s bare tempdir):
        // `write_raw_file` now rolls a failed `commit_and_sync` back (see
        // `write_raw_file`'s doc comment), so a harness where the git step can never
        // succeed would have this call fail and its rollback remove the very file
        // this test is checking for.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let (server, _config) = make_git_backed_server(&work);

        server
            .update_schema(Parameters(UpdateSchemaParams {
                path: Some("/brand/new".into()),
                operation: "add_values".into(),
                field: "tags".into(),
                values: Some(vec!["x".into()]),
                definition: None,
                dry_run: None,
                force: None,
                acknowledge_root_change: None,
            }))
            .await
            .expect("update_schema must succeed against this git-backed harness");

        assert!(
            work.path()
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
                definition: Some(definition(serde_json::json!({ "required": true }))),
                dry_run: Some(true),
                force: None,
                acknowledge_root_change: None,
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
                definition: Some(definition(serde_json::json!({ "required": true }))),
                dry_run: None,
                force: None,
                acknowledge_root_change: None,
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
        // Git-backed harness — see the comment on
        // `update_schema_can_still_create_a_scope_that_does_not_exist_yet` for why
        // `schema_tool_server`'s bare tempdir no longer works for a write that must
        // actually land: `write_raw_file` now rolls back a failed `commit_and_sync`
        // instead of leaving the file behind.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let (server, _config) = make_git_backed_server(&work);
        seed_document(
            &server,
            "notes/a.md",
            serde_json::json!({ "title": "A", "status": "active" }),
        )
        .await;

        server
            .update_schema(Parameters(UpdateSchemaParams {
                path: Some("notes".into()),
                operation: "add_values".into(),
                field: "status".into(),
                values: Some(vec!["active".into(), "draft".into()]),
                definition: None,
                dry_run: None,
                force: None,
                acknowledge_root_change: None,
            }))
            .await
            .expect("a non-breaking change must succeed against this git-backed harness");

        let written = work
            .path()
            .join("notes")
            .join(crate::schema::SCHEMA_FILE_NAME);
        assert!(
            written.exists(),
            "a non-breaking change must be written and committed"
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
                definition: Some(definition(
                    serde_json::json!({ "type": "enum", "values": ["active"], "required": true }),
                )),
                dry_run: Some(true),
                force: None,
                acknowledge_root_change: None,
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
        // Git-backed harness — see the comment on
        // `update_schema_can_still_create_a_scope_that_does_not_exist_yet`: this test
        // asserts the write survives, so the git step must actually succeed rather
        // than trigger `write_raw_file`'s rollback.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let (server, _config) = make_git_backed_server(&work);
        seed_document(&server, "notes/a.md", serde_json::json!({ "title": "A" })).await;

        // The point of this test: force must not be blocked by the casualty check.
        server
            .update_schema(Parameters(UpdateSchemaParams {
                path: Some("notes".into()),
                operation: "set_field".into(),
                field: "status".into(),
                values: None,
                definition: Some(definition(serde_json::json!({ "required": true }))),
                dry_run: None,
                force: Some(true),
                acknowledge_root_change: None,
            }))
            .await
            .expect("force must succeed against this git-backed harness");

        assert!(
            work.path()
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
                definition: Some(definition(serde_json::json!({
                    "type": "integer",
                    "fields": { "prep": { "type": "integer" } }
                }))),
                dry_run: None,
                force: None,
                acknowledge_root_change: None,
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
                acknowledge_root_change: None,
            }))
            .await
            .expect_err("traversal must be rejected");

        assert!(format!("{:?}", err).contains(".."));
    }

    #[tokio::test]
    async fn update_schema_root_add_values_refused_without_acknowledgment() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);

        let err = server
            .update_schema(Parameters(UpdateSchemaParams {
                path: None,
                operation: "add_values".into(),
                field: "tags".into(),
                values: Some(vec!["x".into()]),
                definition: None,
                dry_run: None,
                force: None,
                acknowledge_root_change: None,
            }))
            .await
            .expect_err("root add_values must be refused without acknowledge_root_change");

        let msg = format!("{:?}", err);
        assert!(
            msg.contains("root schema changes are guarded"),
            "got: {msg}"
        );
        assert!(msg.contains("meta/schema-tag-policy.md"), "got: {msg}");
        assert!(msg.contains("acknowledge_root_change"), "got: {msg}");
        assert!(
            !tmp.path().join(crate::schema::SCHEMA_FILE_NAME).exists(),
            "a refused change must leave the filesystem untouched"
        );
    }

    #[tokio::test]
    async fn update_schema_root_set_field_and_remove_field_are_refused_without_acknowledgment() {
        let tmp = tempfile::tempdir().unwrap();
        write_schema_file(
            &tmp,
            "",
            "fields:\n  tags:\n    type: enum\n    values: [x]\n",
        );
        let server = schema_tool_server(&tmp);

        let err = server
            .update_schema(Parameters(UpdateSchemaParams {
                path: None,
                operation: "set_field".into(),
                field: "status".into(),
                values: None,
                definition: Some(definition(serde_json::json!({ "type": "text" }))),
                dry_run: None,
                force: None,
                acknowledge_root_change: None,
            }))
            .await
            .expect_err("root set_field must be refused without acknowledge_root_change");
        assert!(format!("{:?}", err).contains("root schema changes are guarded"));

        let err = server
            .update_schema(Parameters(UpdateSchemaParams {
                path: None,
                operation: "remove_field".into(),
                field: "tags".into(),
                values: None,
                definition: None,
                dry_run: None,
                force: None,
                acknowledge_root_change: None,
            }))
            .await
            .expect_err("root remove_field must be refused without acknowledge_root_change");
        assert!(format!("{:?}", err).contains("root schema changes are guarded"));
    }

    #[tokio::test]
    async fn update_schema_root_add_values_allowed_with_acknowledgment() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let (server, _config) = make_git_backed_server(&work);

        server
            .update_schema(Parameters(UpdateSchemaParams {
                path: None,
                operation: "add_values".into(),
                field: "tags".into(),
                values: Some(vec!["identity".into()]),
                definition: None,
                dry_run: None,
                force: None,
                acknowledge_root_change: Some(true),
            }))
            .await
            .expect("root add_values must succeed once acknowledged");

        assert!(
            work.path().join(crate::schema::SCHEMA_FILE_NAME).exists(),
            "the acknowledged change must actually be written"
        );
    }

    #[tokio::test]
    async fn update_schema_root_add_values_allowed_with_dry_run() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);

        let result = server
            .update_schema(Parameters(UpdateSchemaParams {
                path: None,
                operation: "add_values".into(),
                field: "tags".into(),
                values: Some(vec!["identity".into()]),
                definition: None,
                dry_run: Some(true),
                force: None,
                acknowledge_root_change: None,
            }))
            .await
            .expect("a root dry run must not be gated");

        let structured = result.structured_content.unwrap();
        assert_eq!(structured["dry_run"], serde_json::json!(true));
        assert_eq!(
            structured["summary"],
            serde_json::json!("added to 'tags': identity"),
            "the dry-run result must be exactly what the same edit against a non-root \
             scope would report — the root guard must not alter dry-run behavior"
        );
        assert!(
            !tmp.path().join(crate::schema::SCHEMA_FILE_NAME).exists(),
            "a dry run must not touch the filesystem"
        );
    }

    #[tokio::test]
    async fn update_schema_root_remove_values_allowed_without_acknowledgment() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        write_schema_file(
            &work,
            "",
            "fields:\n  tags:\n    type: enum\n    values: [x, y]\n",
        );
        git_commit_all(&work, crate::schema::SCHEMA_FILE_NAME, "add root schema");
        let (server, _config) = make_git_backed_server(&work);

        server
            .update_schema(Parameters(UpdateSchemaParams {
                path: None,
                operation: "remove_values".into(),
                field: "tags".into(),
                values: Some(vec!["y".into()]),
                definition: None,
                dry_run: None,
                force: None,
                acknowledge_root_change: None,
            }))
            .await
            .expect("root remove_values must not require acknowledge_root_change");

        let written = work.path().join(crate::schema::SCHEMA_FILE_NAME);
        let yaml = std::fs::read_to_string(&written).unwrap();
        let reparsed: crate::schema::SchemaFile = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(reparsed.fields["tags"].values, Some(vec!["x".to_string()]));
    }

    #[tokio::test]
    async fn update_schema_non_root_add_values_allowed_without_acknowledgment() {
        // Already exercised incidentally by other update_schema tests (e.g.
        // `update_schema_accepts_a_change_that_breaks_nothing`), but this test names
        // the property the root guard must not regress: the gate is root-only.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let (server, _config) = make_git_backed_server(&work);

        server
            .update_schema(Parameters(UpdateSchemaParams {
                path: Some("notes".into()),
                operation: "add_values".into(),
                field: "tags".into(),
                values: Some(vec!["x".into()]),
                definition: None,
                dry_run: None,
                force: None,
                acknowledge_root_change: None,
            }))
            .await
            .expect("non-root scopes must not require acknowledge_root_change");

        assert!(
            work.path()
                .join("notes")
                .join(crate::schema::SCHEMA_FILE_NAME)
                .exists()
        );
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
    fn omitted_limit_uses_the_configured_default() {
        assert_eq!(resolve_limit(None, 10, 50), 10);
        // The default is configurable, not baked in — a deployment that raised it
        // must see its own value, not the historical 10.
        assert_eq!(resolve_limit(None, 25, 50), 25);
    }

    #[test]
    fn requested_limit_within_max_is_preserved() {
        assert_eq!(resolve_limit(Some(25), 10, 50), 25);
    }

    #[test]
    fn requested_limit_above_max_is_clamped_to_the_configured_max() {
        assert_eq!(resolve_limit(Some(1_000_000), 10, 50), 50);
        // Clamped to the CONFIGURED ceiling, not a hardcoded one: raising
        // max_limit must actually raise what a caller can ask for.
        assert_eq!(resolve_limit(Some(1_000_000), 10, 200), 200);
    }

    #[test]
    fn zero_limit_is_passed_through() {
        assert_eq!(resolve_limit(Some(0), 10, 50), 0);
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
            Arc::new(crate::reindex::ReindexQueue::new()),
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
            Arc::new(crate::reindex::ReindexQueue::new()),
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
            Arc::new(crate::reindex::ReindexQueue::new()),
        )
        .unwrap();

        let overlong_path = "a".repeat(MAX_PATH_LEN + 1);
        let params = GetDocumentParams {
            path: overlong_path,
            start_line: None,
            end_line: None,
        };
        let result = server.get_document(Parameters(params)).await;
        assert!(result.is_err(), "overlong path should return an error");
    }

    // --- get_document line ranges -------------------------------------------

    /// Numbered lines so a failed assertion names the line it actually got.
    const RANGE_DOC: &str = "l1\nl2\nl3\nl4\nl5\n";

    /// Read `range_doc.md` through the real handler and return
    /// (text content, structured_content).
    async fn get_range(
        server: &KbSearchServer,
        start_line: Option<usize>,
        end_line: Option<usize>,
    ) -> Result<(String, serde_json::Value), McpError> {
        let result = server
            .get_document(Parameters(GetDocumentParams {
                path: "range_doc.md".into(),
                start_line,
                end_line,
            }))
            .await?;
        let text = match &result.content[0].raw {
            rmcp::model::RawContent::Text(t) => t.text.clone(),
            other => panic!("expected a text content block, got {other:?}"),
        };
        Ok((text, result.structured_content.unwrap()))
    }

    fn range_test_server(tmp: &tempfile::TempDir) -> KbSearchServer {
        std::fs::write(tmp.path().join("range_doc.md"), RANGE_DOC).unwrap();
        schema_tool_server(tmp)
    }

    #[tokio::test]
    async fn get_document_without_a_range_serves_the_whole_document() {
        let tmp = tempfile::tempdir().unwrap();
        let server = range_test_server(&tmp);

        let (text, structured) = get_range(&server, None, None).await.unwrap();

        assert_eq!(text, RANGE_DOC);
        assert_eq!(structured["start_line"], 1);
        assert_eq!(structured["end_line"], 5);
        assert_eq!(structured["total_lines"], 5);
        assert_eq!(
            structured["partial"], false,
            "a full read must not advertise itself as partial"
        );
    }

    #[tokio::test]
    async fn get_document_serves_an_inclusive_line_range() {
        let tmp = tempfile::tempdir().unwrap();
        let server = range_test_server(&tmp);

        let (text, structured) = get_range(&server, Some(2), Some(4)).await.unwrap();

        assert_eq!(text, "l2\nl3\nl4\n");
        assert_eq!(
            structured["content"], "l2\nl3\nl4\n",
            "structured_content must mirror the text block, sliced the same way"
        );
        assert_eq!(structured["start_line"], 2);
        assert_eq!(structured["end_line"], 4);
        assert_eq!(structured["total_lines"], 5);
        assert_eq!(structured["partial"], true);
    }

    #[tokio::test]
    async fn get_document_reads_to_eof_with_only_a_start_line() {
        let tmp = tempfile::tempdir().unwrap();
        let server = range_test_server(&tmp);

        let (text, structured) = get_range(&server, Some(4), None).await.unwrap();

        assert_eq!(text, "l4\nl5\n");
        assert_eq!(structured["end_line"], 5);
    }

    #[tokio::test]
    async fn get_document_reads_from_the_top_with_only_an_end_line() {
        let tmp = tempfile::tempdir().unwrap();
        let server = range_test_server(&tmp);

        let (text, structured) = get_range(&server, None, Some(2)).await.unwrap();

        assert_eq!(text, "l1\nl2\n");
        assert_eq!(structured["start_line"], 1);
    }

    #[tokio::test]
    async fn get_document_clamps_an_end_line_past_the_document() {
        let tmp = tempfile::tempdir().unwrap();
        let server = range_test_server(&tmp);

        let (text, structured) = get_range(&server, Some(4), Some(900)).await.unwrap();

        assert_eq!(text, "l4\nl5\n");
        assert_eq!(structured["end_line"], 5);
        assert_eq!(
            structured["partial"], true,
            "a clamped tail still omits the head of the document"
        );
    }

    #[tokio::test]
    async fn get_document_covering_every_line_is_not_reported_as_partial() {
        let tmp = tempfile::tempdir().unwrap();
        let server = range_test_server(&tmp);

        let (_, structured) = get_range(&server, Some(1), Some(5)).await.unwrap();

        assert_eq!(structured["partial"], false);
    }

    #[tokio::test]
    async fn get_document_hashes_the_whole_document_even_for_a_partial_read() {
        let tmp = tempfile::tempdir().unwrap();
        let server = range_test_server(&tmp);

        let (_, full) = get_range(&server, None, None).await.unwrap();
        let (_, partial) = get_range(&server, Some(2), Some(3)).await.unwrap();

        assert_eq!(
            partial["content_hash"], full["content_hash"],
            "content_hash is edit_document's expected_hash: it must describe the file \
             on disk, not the slice served"
        );
        assert_eq!(
            partial["content_hash"],
            crate::ingest::compute_hash_from_bytes(RANGE_DOC.as_bytes())
        );
    }

    #[tokio::test]
    async fn get_document_rejects_a_malformed_range() {
        let tmp = tempfile::tempdir().unwrap();
        let server = range_test_server(&tmp);

        assert!(
            get_range(&server, Some(0), None).await.is_err(),
            "line 0 does not exist; lines are 1-based"
        );
        assert!(
            get_range(&server, Some(4), Some(2)).await.is_err(),
            "an inverted range should be refused, not silently swapped"
        );
    }

    #[tokio::test]
    async fn get_document_rejects_a_start_line_past_the_document() {
        let tmp = tempfile::tempdir().unwrap();
        let server = range_test_server(&tmp);

        let err = get_range(&server, Some(99), None).await.unwrap_err();
        assert!(
            err.to_string().contains('5'),
            "the error should say how many lines the document actually has, got: {err}"
        );
    }

    #[tokio::test]
    async fn get_document_validates_the_range_before_resolving_the_path() {
        let tmp = tempfile::tempdir().unwrap();
        let server = range_test_server(&tmp);

        let err = server
            .get_document(Parameters(GetDocumentParams {
                path: "no/such/document.md".into(),
                start_line: Some(9),
                end_line: Some(2),
            }))
            .await
            .unwrap_err()
            .to_string();

        assert!(
            !err.contains("not found"),
            "a bad range should be reported as such, not masked by the path lookup: {err}"
        );
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
            Arc::new(crate::reindex::ReindexQueue::new()),
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
            expected_hash: None,
            new_path: None,
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
    async fn create_document_on_existing_but_not_permitted_file_returns_include_pattern_error() {
        // G3 regression: pre-refactor, a path that BOTH exists on disk AND fails
        // the include-pattern check was reported with the include-pattern
        // message, not "already exists" — reporting "already exists; use
        // edit_document" here would be misleading circular guidance, since
        // edit_document rejects the same path as not permitted right back.
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        // Exists on disk as a `.md` file, but the server below only permits
        // `.txt` files — so it both exists AND fails the include-pattern check.
        std::fs::write(sub.join("existing.md"), "# Already here").unwrap();

        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.txt".to_string()], config);

        let params = CreateDocumentParams {
            path: "docs/existing.md".to_string(),
            content: "---\ntitle: Test\n---\n# New content".to_string(),
            message: None,
            force_new: None,
        };
        let result = server.create_document(Parameters(params)).await;

        assert!(result.is_err(), "create should be rejected");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("indexable include pattern"),
            "error should report the include-pattern rejection, got: {}",
            err.message
        );
        assert!(
            !err.message.contains("already exists"),
            "error should not fall back to the misleading 'already exists' message, got: {}",
            err.message
        );
    }

    #[test]
    fn already_exists_race_error_reports_the_absolute_path() {
        // `create_document`'s own pre-check (`abs_path.exists()`) catches the
        // ordinary "already exists" case before ever reaching `write.rs` — see
        // `create_document_on_existing_file_returns_use_edit_error` above. The
        // `WriteError::AlreadyExists` arm this test drives is only reachable via
        // a genuine TOCTOU race (the file appears between that pre-check and
        // `write::write_document`'s `create_new` open), which pre-`write.rs`-
        // extraction code reported with the absolute filesystem path rather than
        // the repo-relative one — restore that wording (N2).
        let tmp = tempfile::tempdir().unwrap();
        let canonical_data_path = tmp.path().canonicalize().unwrap();
        let rel_path = "docs/existing.md";

        let err = create_edit_error_to_mcp_error(
            WriteError::AlreadyExists,
            rel_path,
            true,
            &canonical_data_path,
            None,
        );

        let expected_abs = canonical_data_path.join(rel_path);
        assert!(
            err.message.contains(&expected_abs.display().to_string()),
            "expected the absolute path '{}' in the message, got: {}",
            expected_abs.display(),
            err.message
        );
        assert!(
            err.message.contains("edit_document"),
            "error should still mention edit_document, got: {}",
            err.message
        );
    }

    #[test]
    fn internal_write_error_reaches_mcp_callers_with_the_same_text_as_before() {
        // G2: `WriteError::Internal` exists so `web.rs` can hide a canonicalize
        // failure's embedded absolute path from an untrusted caller. MCP is a
        // trusted surface that was already relaying this exact text via
        // `WriteError::UnsafePath` before that split — both adapters must keep
        // doing so, verbatim, for `Internal` too.
        let msg = "Invalid path: cannot canonicalize data root '/data/kb': \
                    No such file or directory (os error 2)"
            .to_string();
        let tmp = tempfile::tempdir().unwrap();
        let canonical_data_path = tmp.path().canonicalize().unwrap();

        let create_err = create_edit_error_to_mcp_error(
            WriteError::Internal { msg: msg.clone() },
            "docs/x.md",
            true,
            &canonical_data_path,
            None,
        );
        assert_eq!(create_err.message, msg);

        let delete_err =
            delete_error_to_mcp_error(WriteError::Internal { msg: msg.clone() }, "docs/x.md");
        assert_eq!(delete_err.message, msg);
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
            ui: crate::config::UiConfig::default(),
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
            ui: crate::config::UiConfig::default(),
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
            expected_hash: None,
            new_path: None,
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

    // `delete_document_existing_file_proceeds_to_git_step` used to live here: create
    // a file in a plain tempdir with no git repo, delete it, and check the failure
    // wasn't a path-resolution error. `delete_document_with_no_git_repo_reports_
    // inconsistent_state` (below, in the pre-commit/post-commit test group) drives
    // the exact same scenario with assertions that actually pin down the new
    // behavior — the FailedInconsistentState outcome and message — so it replaces
    // this test rather than sitting alongside a strictly weaker duplicate.

    // `resolve_safe_write_path` unit tests moved to `write.rs` alongside the
    // function itself — see that module's test suite.

    // -----------------------------------------------------------------------
    // parse_edit_mode unit tests
    // -----------------------------------------------------------------------

    fn make_edit_params(
        content: Option<&str>,
        old_string: Option<&str>,
        new_string: Option<&str>,
        new_path: Option<&str>,
    ) -> EditDocumentParams {
        EditDocumentParams {
            path: "docs/test.md".to_string(),
            content: content.map(|s| s.to_string()),
            old_string: old_string.map(|s| s.to_string()),
            new_string: new_string.map(|s| s.to_string()),
            message: None,
            expected_hash: None,
            new_path: new_path.map(|s| s.to_string()),
        }
    }

    #[test]
    fn parse_edit_mode_full_replace_is_recognized() {
        let params = make_edit_params(Some("new content"), None, None, None);
        let mode = parse_edit_mode(&params).unwrap();
        assert_eq!(
            mode,
            Some(EditMode::Full {
                content: "new content".to_string()
            })
        );
    }

    #[test]
    fn parse_edit_mode_surgical_is_recognized() {
        let params = make_edit_params(None, Some("old text"), Some("new text"), None);
        let mode = parse_edit_mode(&params).unwrap();
        assert_eq!(
            mode,
            Some(EditMode::Surgical {
                old: "old text".to_string(),
                new: "new text".to_string()
            })
        );
    }

    #[test]
    fn parse_edit_mode_both_modes_rejected() {
        let params = make_edit_params(Some("full content"), Some("old"), Some("new"), None);
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(
            err.contains("mutually exclusive"),
            "expected 'mutually exclusive' in error, got: {err}"
        );
    }

    #[test]
    fn parse_edit_mode_neither_mode_rejected() {
        let params = make_edit_params(None, None, None, None);
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(
            err.contains("must provide"),
            "expected 'must provide' in error, got: {err}"
        );
        assert!(
            err.contains("new_path"),
            "the error must name new_path as a third option now that moves exist, got: {err}"
        );
    }

    #[test]
    fn parse_edit_mode_only_old_string_rejected() {
        let params = make_edit_params(None, Some("old"), None, None);
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(
            err.contains("new_string"),
            "expected mention of new_string in error, got: {err}"
        );
    }

    #[test]
    fn parse_edit_mode_only_new_string_rejected() {
        let params = make_edit_params(None, None, Some("new"), None);
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(
            err.contains("old_string"),
            "expected mention of old_string in error, got: {err}"
        );
    }

    #[test]
    fn parse_edit_mode_identical_old_new_rejected() {
        let params = make_edit_params(None, Some("same text"), Some("same text"), None);
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(
            err.contains("identical"),
            "expected 'identical' in error, got: {err}"
        );
    }

    // -----------------------------------------------------------------------
    // parse_edit_mode: new_path (move) arms
    // -----------------------------------------------------------------------

    #[test]
    fn parse_edit_mode_move_alone_is_a_pure_move() {
        // Neither content mode, but new_path set: Ok(None) — a pure move, content
        // unchanged.
        let params = make_edit_params(None, None, None, Some("docs/new-home.md"));
        let mode = parse_edit_mode(&params).unwrap();
        assert_eq!(
            mode, None,
            "move-only should parse to Ok(None), got: {mode:?}"
        );
    }

    #[test]
    fn parse_edit_mode_surgical_combined_with_move_is_recognized() {
        let params = make_edit_params(
            None,
            Some("old text"),
            Some("new text"),
            Some("docs/new-home.md"),
        );
        let mode = parse_edit_mode(&params).unwrap();
        assert_eq!(
            mode,
            Some(EditMode::Surgical {
                old: "old text".to_string(),
                new: "new text".to_string()
            }),
            "surgical + new_path must still parse as Ok(Some(Surgical))"
        );
    }

    #[test]
    fn parse_edit_mode_full_replace_combined_with_move_is_recognized() {
        let params = make_edit_params(Some("new content"), None, None, Some("docs/new-home.md"));
        let mode = parse_edit_mode(&params).unwrap();
        assert_eq!(
            mode,
            Some(EditMode::Full {
                content: "new content".to_string()
            }),
            "full-replace + new_path must still parse as Ok(Some(Full))"
        );
    }

    #[test]
    fn parse_edit_mode_both_modes_still_rejected_even_with_move() {
        // surgical and full-replace remain mutually exclusive WITH EACH OTHER
        // regardless of whether new_path is also present.
        let params = make_edit_params(
            Some("full content"),
            Some("old"),
            Some("new"),
            Some("docs/new-home.md"),
        );
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(
            err.contains("mutually exclusive"),
            "expected 'mutually exclusive' in error even with new_path set, got: {err}"
        );
    }

    // -----------------------------------------------------------------------
    // apply_surgical unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn apply_surgical_single_occurrence_replaced() {
        let old = "Hello world!\nGoodbye earth!";
        let result = apply_surgical(old, "world", "Rust", "d.md").unwrap();
        assert_eq!(result, "Hello Rust!\nGoodbye earth!");
    }

    #[test]
    fn apply_surgical_not_found_names_the_document_path() {
        // Regression coverage for issue #88: the error must name the actual document,
        // not the word "document" — and, now that the message is built directly
        // rather than via a blind `.replace("document", ...)`, an occurrence of the
        // word "document" elsewhere in the message (e.g. in "get_document") must
        // survive untouched.
        let old = "Hello world! Nothing here resembles the needle at all, at all.";
        let err = apply_surgical(
            old,
            "missing text entirely unrelated to this content, over forty chars long",
            "replacement",
            "food/plans/2026-07-30.md",
        )
        .unwrap_err();
        assert!(
            err.contains("not found in 'food/plans/2026-07-30.md'"),
            "expected the document path in the error, got: {err}"
        );
        assert!(
            err.contains("get_document"),
            "the word 'document' inside 'get_document' must survive intact, got: {err}"
        );
    }

    #[test]
    fn apply_surgical_not_found_but_whitespace_normalized_match_exists() {
        // The most common real cause per issue #88: same text, different indentation /
        // line endings / trailing whitespace. Must be called out explicitly rather than
        // left for the caller to guess.
        let old = "line one\n    line two   \nline three";
        let old_string = "line one\nline two\nline three"; // no indentation, no trailing spaces
        let err = apply_surgical(old, old_string, "replacement", "notes.md").unwrap_err();
        assert!(
            err.contains("whitespace"),
            "expected a whitespace near-match callout, got: {err}"
        );
        assert!(
            err.contains("notes.md"),
            "expected the document path in the error, got: {err}"
        );
    }

    #[test]
    fn apply_surgical_not_found_but_anchor_matches_shows_context() {
        // old_string's first 40 chars appear verbatim in the document, but the text
        // diverges after that point — the caller gets to see exactly where and how.
        let anchor = "0123456789012345678901234567890123456789"; // exactly 40 chars
        let old_string = format!("{anchor}XYZ_EXPECTED_TAIL");
        let old = format!("prefix text before it {anchor}ABC_ACTUAL_TAIL and trailing text after");
        let err = apply_surgical(&old, &old_string, "replacement", "d.md").unwrap_err();
        assert!(
            err.contains(anchor),
            "expected the matched anchor text in the error, got: {err}"
        );
        assert!(
            err.contains("ABC_ACTUAL_TAIL"),
            "expected surrounding document context in the error, got: {err}"
        );
    }

    #[test]
    fn apply_surgical_not_found_and_nothing_resembles_it() {
        let old = "A short paragraph about something else entirely.";
        let old_string = "Completely unrelated text that will never appear anywhere in the source, over forty chars.";
        let err = apply_surgical(old, old_string, "replacement", "d.md").unwrap_err();
        assert!(
            err.contains("wrong document") || err.contains("changed substantially"),
            "expected guidance that nothing resembles old_string, got: {err}"
        );
    }

    #[test]
    fn apply_surgical_not_found_diagnostics_are_skipped_above_the_size_cap() {
        // Bounds the diagnostic work: past NOT_FOUND_DIAGNOSTIC_MAX_BYTES, fall back to
        // the plain message rather than running whitespace-normalization or an anchor
        // search over an arbitrarily large document.
        let old = "x".repeat(NOT_FOUND_DIAGNOSTIC_MAX_BYTES + 1);
        let err = apply_surgical(
            &old,
            "missing text over forty characters long, easily",
            "r",
            "big.md",
        )
        .unwrap_err();
        assert_eq!(
            err, "old_string not found in 'big.md'",
            "past the size cap the message must be the plain fallback, got: {err}"
        );
    }

    #[test]
    fn apply_surgical_multiple_occurrences_returns_error_with_count() {
        let old = "foo bar foo baz foo";
        let err = apply_surgical(old, "foo", "qux", "d.md").unwrap_err();
        assert!(
            err.contains("3"),
            "error should mention occurrence count (3), got: {err}"
        );
        assert!(
            err.contains("not unique in 'd.md'"),
            "error should name the document and say 'not unique', got: {err}"
        );
    }

    #[test]
    fn apply_surgical_exact_single_unique_string() {
        let old = "---\ntitle: My Doc\n---\n# Content\nSome text here.";
        let result = apply_surgical(old, "Some text here.", "Updated text.", "d.md").unwrap();
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
    // paths dirty on their server's `ReindexQueue` and return — which also means these
    // tests no longer need a live Qdrant/embeddings service to reach that point, since
    // nothing here calls into the indexer at all.

    /// Build a `KbSearchServer` backed by a real git working clone, so write tools get
    /// past `commit_and_sync` and reach the point where they mark paths dirty.
    ///
    /// `make_write_test_server` gives this server its own private `ReindexQueue`
    /// (see that function), so the tests below read it back via
    /// `server.reindex_queue` — no other test in the binary can have marked
    /// anything on it.
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

        crate::reindex::test_support::assert_marked_dirty(
            &server.reindex_queue,
            &["docs/queued.md"],
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
                acknowledge_root_change: None,
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
            work.path().join("doomed-queued-cleanup-test.md"),
            "---\ntitle: D\n---\n\n# Body\n",
        )
        .unwrap();
        // delete_document git-adds the removed path, so the file must already be tracked.
        std::process::Command::new("git")
            .args(["add", "doomed-queued-cleanup-test.md"])
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
                "add doomed-queued-cleanup-test.md",
            ])
            .current_dir(work.path())
            .output()
            .unwrap();
        let (server, _config) = make_git_backed_server(&work);

        let result = server
            .delete_document(Parameters(DeleteDocumentParams {
                path: "doomed-queued-cleanup-test.md".to_string(),
                message: None,
            }))
            .await;

        let result = result.expect("delete must succeed even though nothing purges it inline");
        let text = format!("{:?}", result.content);
        assert!(
            text.contains("Deleted 'doomed-queued-cleanup-test.md'"),
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

        crate::reindex::test_support::assert_marked_dirty(
            &server.reindex_queue,
            &["doomed-queued-cleanup-test.md"],
        );
    }

    // -----------------------------------------------------------------------
    // Pre-commit vs. post-commit failure handling (git::CommitSyncError) —
    // create_document / edit_document / delete_document
    // -----------------------------------------------------------------------

    /// `HEAD` of `work`, as a trimmed hex string.
    fn head_sha(work: &tempfile::TempDir) -> String {
        let out = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(work.path())
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// `git status --porcelain` of `work`, as a trimmed string ("" means clean).
    fn git_status(work: &tempfile::TempDir) -> String {
        let out = std::process::Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(work.path())
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn git_commit_all(work: &tempfile::TempDir, rel_path: &str, message: &str) {
        std::process::Command::new("git")
            .args(["add", "--", rel_path])
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
                message,
            ])
            .current_dir(work.path())
            .output()
            .unwrap();
    }

    /// Force `git commit` to fail in `work` while leaving the repo otherwise
    /// completely healthy: `git add` still succeeds, HEAD is still valid, and a
    /// restore from HEAD should still succeed. This is `CommitSyncError::PreCommit`
    /// with the failure specifically at the `commit` step rather than `add`.
    ///
    /// Deliberately NOT a `.git/hooks/pre-commit` script: a machine (or CI image)
    /// with a global `core.hooksPath` override — common for commit-signing or
    /// linting setups — would silently ignore a repo-local hook, making that
    /// approach environment-dependent. Repo-local git CONFIG, by contrast, always
    /// applies: enabling commit signing and pointing at a signing key that does not
    /// exist makes `git commit` fail deterministically on any machine, with or
    /// without gpg installed (no gpg binary fails it too, just with a different
    /// message), and without touching global state.
    fn force_git_commit_to_fail(work: &tempfile::TempDir) {
        for args in [
            ["config", "commit.gpgsign", "true"],
            ["config", "user.signingkey", "nonexistent-bogus-key-id"],
        ] {
            std::process::Command::new("git")
                .args(args)
                .current_dir(work.path())
                .output()
                .unwrap();
        }
    }

    /// Extract `structured_content["outcome"]` (success path) or `data["outcome"]`
    /// (error path) as a `&str`, so tests can assert on the machine-readable
    /// discriminant instead of parsing prose.
    fn outcome_of(value: &Option<serde_json::Value>) -> Option<&str> {
        value.as_ref()?.get("outcome")?.as_str()
    }

    #[tokio::test]
    async fn delete_document_precommit_failure_restores_the_file_and_reports_no_change() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original = "---\ntitle: D\n---\n\n# Body\n";
        std::fs::write(work.path().join("doomed.md"), original).unwrap();
        git_commit_all(&work, "doomed.md", "add doomed.md");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let (server, _config) = make_git_backed_server(&work);

        let result = server
            .delete_document(Parameters(DeleteDocumentParams {
                path: "doomed.md".to_string(),
                message: None,
            }))
            .await;

        let err = result.expect_err("a rejected pre-commit hook must fail the delete");
        assert_eq!(
            outcome_of(&err.data),
            Some("failed_no_change"),
            "error data must carry the outcome discriminant, got: {:?}",
            err
        );

        assert!(
            work.path().join("doomed.md").exists(),
            "the file must be restored to disk after a rolled-back pre-commit failure"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("doomed.md")).unwrap(),
            original,
            "restored content must match what was at HEAD"
        );
        assert_eq!(
            head_before,
            head_sha(&work),
            "HEAD must not move on a rolled-back pre-commit failure"
        );
        assert_eq!(
            git_status(&work),
            "",
            "working tree must be clean after rollback"
        );
    }

    #[tokio::test]
    async fn delete_document_postcommit_failure_leaves_commit_and_reports_pending_sync() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::write(
            work.path().join("doomed.md"),
            "---\ntitle: D\n---\n\n# Body\n",
        )
        .unwrap();
        git_commit_all(&work, "doomed.md", "add doomed.md");

        let mut config = make_test_resolved_config(work.path());
        {
            let c = Arc::get_mut(&mut config).unwrap();
            c.write.dedup_enabled = false;
            // No such path — `git fetch` fails immediately, no network required, but
            // only AFTER the deletion's `git add`/`git commit` have already
            // succeeded locally.
            c.source.git_url = Some("/nonexistent/path/to/repo.git".to_string());
        }
        let server = make_write_test_server(&work, &["**/*.md".to_string()], config);

        let result = server
            .delete_document(Parameters(DeleteDocumentParams {
                path: "doomed.md".to_string(),
                message: None,
            }))
            .await;

        let result = result.expect("a post-commit sync failure must still report as success");
        assert_eq!(
            outcome_of(&result.structured_content),
            Some("committed_pending_sync"),
            "got: {:?}",
            result
        );
        let text = format!("{:?}", result.content);
        assert!(
            text.contains("push") && text.contains("sync on the next successful write"),
            "must explain the push failure and that sync is pending: {text}"
        );

        // The deletion IS a real local commit — the file must remain gone, and HEAD
        // must record the deletion. None of this is rolled back.
        assert!(
            !work.path().join("doomed.md").exists(),
            "a post-commit failure must NOT resurrect the file"
        );
        let show = std::process::Command::new("git")
            .args(["show", "--name-only", "--format=", "HEAD"])
            .current_dir(work.path())
            .output()
            .unwrap();
        assert!(
            String::from_utf8_lossy(&show.stdout).contains("doomed.md"),
            "the deletion commit must be present in local HEAD"
        );
    }

    /// When there is no git repository at all, `git add` fails (`PreCommit`) and the
    /// rollback attempt (`git restore`) ALSO fails, since there is nothing to restore
    /// from. This is the third, worse outcome: the file is gone from disk with no
    /// corresponding commit anywhere. It must be reported distinctly rather than
    /// masquerading as either a clean delete or a clean no-op.
    #[tokio::test]
    async fn delete_document_with_no_git_repo_reports_inconsistent_state() {
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

        let err = result.expect_err("deleting with no git repo must fail");
        assert_eq!(
            outcome_of(&err.data),
            Some("failed_inconsistent_state"),
            "got: {:?}",
            err
        );
        assert!(
            err.message.contains("INCONSISTENT"),
            "the message must call out the inconsistent state loudly, got: {}",
            err.message
        );

        // The restore could not put it back (there is no repo to restore from), so
        // the file really is gone — that IS the inconsistent state being reported.
        assert!(!sub.join("delete-me.md").exists());
    }

    #[tokio::test]
    async fn create_document_precommit_failure_removes_the_new_file_and_reports_no_change() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let (server, _config) = make_git_backed_server(&work);

        let result = server
            .create_document(Parameters(CreateDocumentParams {
                path: "docs/new.md".to_string(),
                content: "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n"
                    .to_string(),
                message: None,
                force_new: Some(true),
            }))
            .await;

        let err = result.expect_err("a rejected pre-commit hook must fail the create");
        assert_eq!(
            outcome_of(&err.data),
            Some("failed_no_change"),
            "got: {:?}",
            err
        );

        assert!(
            !work.path().join("docs/new.md").exists(),
            "the newly-written file must be removed on rollback — there is no HEAD \
             content for a brand-new create to fall back to"
        );
        assert_eq!(
            head_before,
            head_sha(&work),
            "HEAD must not move on a rolled-back pre-commit failure"
        );
        assert_eq!(
            git_status(&work),
            "",
            "the aborted `git add` must be unstaged too — no leftover addition that \
             could ride along on a later, unrelated commit"
        );
    }

    #[tokio::test]
    async fn edit_document_precommit_failure_restores_previous_content_and_reports_no_change() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original =
            "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
        std::fs::write(work.path().join("edit-me.md"), original).unwrap();
        git_commit_all(&work, "edit-me.md", "add edit-me.md");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let (server, _config) = make_git_backed_server(&work);

        let result = server
            .edit_document(Parameters(EditDocumentParams {
                path: "edit-me.md".to_string(),
                old_string: None,
                new_string: None,
                content: Some(
                    "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New body\n"
                        .to_string(),
                ),
                message: None,
                expected_hash: None,
                new_path: None,
            }))
            .await;

        let err = result.expect_err("a rejected pre-commit hook must fail the edit");
        assert_eq!(
            outcome_of(&err.data),
            Some("failed_no_change"),
            "got: {:?}",
            err
        );

        assert_eq!(
            std::fs::read_to_string(work.path().join("edit-me.md")).unwrap(),
            original,
            "the edit must be rolled back to the previous HEAD content"
        );
        assert_eq!(
            head_before,
            head_sha(&work),
            "HEAD must not move on a rolled-back pre-commit failure"
        );
        assert_eq!(
            git_status(&work),
            "",
            "working tree must be clean after rollback"
        );
    }

    #[tokio::test]
    async fn edit_document_postcommit_failure_leaves_commit_and_reports_pending_sync() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original =
            "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
        std::fs::write(work.path().join("edit-me.md"), original).unwrap();
        git_commit_all(&work, "edit-me.md", "add edit-me.md");

        let mut config = make_test_resolved_config(work.path());
        {
            let c = Arc::get_mut(&mut config).unwrap();
            c.write.dedup_enabled = false;
            c.source.git_url = Some("/nonexistent/path/to/repo.git".to_string());
        }
        let server = make_write_test_server(&work, &["**/*.md".to_string()], config);

        let new_content =
            "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New body\n";
        let result = server
            .edit_document(Parameters(EditDocumentParams {
                path: "edit-me.md".to_string(),
                old_string: None,
                new_string: None,
                content: Some(new_content.to_string()),
                message: None,
                expected_hash: None,
                new_path: None,
            }))
            .await;

        let result = result.expect("a post-commit sync failure must still report as success");
        assert_eq!(
            outcome_of(&result.structured_content),
            Some("committed_pending_sync"),
            "got: {:?}",
            result
        );

        // The edit IS a real local commit — the new content stays on disk, and HEAD
        // records it. None of this is rolled back just because the push failed.
        assert_eq!(
            std::fs::read_to_string(work.path().join("edit-me.md")).unwrap(),
            new_content,
            "a post-commit failure must NOT revert the edit"
        );
        let show = std::process::Command::new("git")
            .args(["show", "HEAD:edit-me.md"])
            .current_dir(work.path())
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&show.stdout), new_content);
    }

    // -----------------------------------------------------------------------
    // expected_hash — optional stale-read guard (issue #88)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn get_document_reports_a_content_hash_matching_indexed_files_hashing() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        let content = "---\ntitle: T\ntype: guide\n---\n# Body\n";
        std::fs::write(sub.join("guide.md"), content).unwrap();

        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        let result = server
            .get_document(Parameters(GetDocumentParams {
                path: "docs/guide.md".to_string(),
                start_line: None,
                end_line: None,
            }))
            .await
            .unwrap();

        let structured = result
            .structured_content
            .expect("get_document must report a content_hash in structured_content");
        let hash = structured["content_hash"]
            .as_str()
            .expect("content_hash must be a string");
        assert_eq!(
            hash,
            crate::ingest::compute_hash_from_bytes(content.as_bytes()),
            "must be the exact same hashing indexed_files.content_hash uses, so a \
             caller can round-trip it into edit_document's expected_hash"
        );
        assert_eq!(
            structured["content"].as_str(),
            Some(content),
            "structured_content must carry the full document: clients that prefer \
             structuredContent render only it, so a hash-only payload hides the \
             document entirely"
        );
        assert_eq!(structured["path"].as_str(), Some("docs/guide.md"));
    }

    #[tokio::test]
    async fn edit_document_with_a_stale_expected_hash_is_rejected_before_touching_the_file() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        let original = "---\ntitle: Old\ntype: guide\n---\n# Old body\n";
        std::fs::write(sub.join("edit-me.md"), original).unwrap();

        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        // A hash of some OTHER content — as if the caller read the document at an
        // earlier revision.
        let stale_hash = crate::ingest::compute_hash_from_bytes(b"not the current content");

        let result = server
            .edit_document(Parameters(EditDocumentParams {
                path: "docs/edit-me.md".to_string(),
                old_string: None,
                new_string: None,
                content: Some("---\ntitle: New\ntype: guide\n---\n# New body\n".to_string()),
                message: None,
                expected_hash: Some(stale_hash),
                new_path: None,
            }))
            .await;

        let err = result.expect_err("a stale expected_hash must be rejected");
        assert!(
            err.message.contains("changed since you read it"),
            "expected an explicit stale-read message, got: {}",
            err.message
        );
        assert!(
            err.message.contains("get_document"),
            "expected guidance to re-read via get_document, got: {}",
            err.message
        );

        // Rejected before any write: the file on disk must be untouched.
        assert_eq!(
            std::fs::read_to_string(sub.join("edit-me.md")).unwrap(),
            original,
            "a stale expected_hash must fail before the file is touched"
        );
    }

    #[tokio::test]
    async fn edit_document_with_a_matching_expected_hash_proceeds_to_a_synced_write() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original =
            "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
        std::fs::write(work.path().join("edit-me.md"), original).unwrap();
        git_commit_all(&work, "edit-me.md", "add edit-me.md");
        let (server, _config) = make_git_backed_server(&work);

        let correct_hash = crate::ingest::compute_hash_from_bytes(original.as_bytes());
        let new_content =
            "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New body\n";

        let result = server
            .edit_document(Parameters(EditDocumentParams {
                path: "edit-me.md".to_string(),
                old_string: None,
                new_string: None,
                content: Some(new_content.to_string()),
                message: None,
                expected_hash: Some(correct_hash),
                new_path: None,
            }))
            .await;

        let result = result.expect("a correct expected_hash must not block the edit");
        assert_eq!(
            outcome_of(&result.structured_content),
            Some("synced"),
            "got: {:?}",
            result
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("edit-me.md")).unwrap(),
            new_content
        );
    }

    // -----------------------------------------------------------------------
    // edit_document: new_path (move), end-to-end through the tool
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn edit_document_move_alone_relocates_content_unchanged() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original =
            "---\ntitle: Old Home\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::create_dir_all(work.path().join("docs")).unwrap();
        std::fs::write(work.path().join("docs/old-home.md"), original).unwrap();
        git_commit_all(&work, "docs/old-home.md", "add docs/old-home.md");
        let (server, _config) = make_git_backed_server(&work);

        let result = server
            .edit_document(Parameters(EditDocumentParams {
                path: "docs/old-home.md".to_string(),
                old_string: None,
                new_string: None,
                content: None,
                message: None,
                expected_hash: None,
                new_path: Some("docs/new-home.md".to_string()),
            }))
            .await;

        let result = result.expect("a pure move (new_path alone) must succeed");
        assert_eq!(
            outcome_of(&result.structured_content),
            Some("synced"),
            "got: {:?}",
            result
        );

        assert_eq!(
            std::fs::read_to_string(work.path().join("docs/new-home.md")).unwrap(),
            original,
            "the destination must have the source's exact original content, unchanged"
        );
        assert!(
            !work.path().join("docs/old-home.md").exists(),
            "the source must be gone after a move"
        );
    }

    #[tokio::test]
    async fn edit_document_move_combined_with_edit_relocates_transformed_content() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original =
            "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
        std::fs::write(work.path().join("edit-me.md"), original).unwrap();
        git_commit_all(&work, "edit-me.md", "add edit-me.md");
        let (server, _config) = make_git_backed_server(&work);

        let new_content =
            "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New body\n";

        let result = server
            .edit_document(Parameters(EditDocumentParams {
                path: "edit-me.md".to_string(),
                old_string: None,
                new_string: None,
                content: Some(new_content.to_string()),
                message: None,
                expected_hash: None,
                new_path: Some("archive/edit-me.md".to_string()),
            }))
            .await;

        let result = result.expect("a combined move+edit must succeed");
        assert_eq!(
            outcome_of(&result.structured_content),
            Some("synced"),
            "got: {:?}",
            result
        );

        assert_eq!(
            std::fs::read_to_string(work.path().join("archive/edit-me.md")).unwrap(),
            new_content,
            "the destination must hold the TRANSFORMED content, not the pre-move original"
        );
        assert!(
            !work.path().join("edit-me.md").exists(),
            "the source must be gone after a move"
        );
    }

    #[tokio::test]
    async fn edit_document_move_onto_existing_destination_reports_the_destination_as_the_collision()
    {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let source_content =
            "---\ntitle: Source\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Source body\n";
        let dest_content =
            "---\ntitle: Dest\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Dest body\n";
        std::fs::write(work.path().join("source.md"), source_content).unwrap();
        std::fs::write(work.path().join("dest.md"), dest_content).unwrap();
        git_commit_all(&work, "source.md", "add source.md");
        git_commit_all(&work, "dest.md", "add dest.md");
        let (server, _config) = make_git_backed_server(&work);

        let result = server
            .edit_document(Parameters(EditDocumentParams {
                path: "source.md".to_string(),
                old_string: None,
                new_string: None,
                content: None,
                message: None,
                expected_hash: None,
                new_path: Some("dest.md".to_string()),
            }))
            .await;

        let err = result.expect_err("moving onto an existing destination must be rejected");
        assert!(
            err.message.contains("dest.md"),
            "error should name the destination path, got: {}",
            err.message
        );
        assert!(
            err.message.contains("destination"),
            "error should make clear it is the DESTINATION that collided, got: {}",
            err.message
        );
        assert!(
            !err.message.contains("'source.md' already exists"),
            "error must not misattribute the collision to the source, got: {}",
            err.message
        );

        // Rejected before any filesystem mutation (write_document_move checks the
        // destination's existence before writing anything) — both files, source and
        // pre-existing destination, must be untouched.
        assert_eq!(
            std::fs::read_to_string(work.path().join("source.md")).unwrap(),
            source_content
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("dest.md")).unwrap(),
            dest_content
        );
    }

    // -----------------------------------------------------------------------
    // update_schema / write_raw_file — the same PreCommit/PostCommit rollback
    // treatment as create_document/edit_document/delete_document above, applied to
    // the one write path (`write_raw_file`, used only by `update_schema`) PR #93
    // deferred. See `write_raw_file`'s doc comment for the rollback rules and
    // `update_schema`'s comment above its call to it for why a rolled-back write
    // must skip both the schema-cache rebuild and `reindex::mark_full`.
    // -----------------------------------------------------------------------

    /// `git_status`, filtered to drop the SQLite state-DB files (`state.db` plus its
    /// `-shm`/`-wal` siblings). `update_schema`'s casualty check
    /// (`documents_broken_by`) opens the metadata index on first use, which lazily
    /// creates those files under `work` as an ordinary, expected side effect of
    /// running the tool at all — unrelated to whether a `commit_and_sync` rollback
    /// left the WRITE ITSELF clean. `create_document`/`edit_document`/
    /// `delete_document`'s equivalent tests never touch the metadata index this way,
    /// which is why their plain `git_status` assertions can stay exact.
    fn git_status_ignoring_state_db(work: &tempfile::TempDir) -> String {
        git_status(work)
            .lines()
            .filter(|line| {
                let path = line.split_whitespace().last().unwrap_or("");
                !path.starts_with("state.db")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[tokio::test]
    async fn update_schema_precommit_failure_on_new_schema_rolls_it_back() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let head_before = head_sha(&work);

        // No `.kb-schema.yaml` exists anywhere under `notes/` yet, so this write
        // creates one from scratch — the `is_new` branch of `write_raw_file`'s
        // rollback.
        force_git_commit_to_fail(&work);
        let (server, _config) = make_git_backed_server(&work);

        let result = server
            .update_schema(Parameters(UpdateSchemaParams {
                path: Some("notes".into()),
                operation: "add_values".into(),
                field: "tags".into(),
                values: Some(vec!["x".into()]),
                definition: None,
                dry_run: None,
                force: None,
                acknowledge_root_change: None,
            }))
            .await;

        let err = result.expect_err("a rejected pre-commit hook must fail the schema write");
        assert_eq!(
            outcome_of(&err.data),
            Some("failed_no_change"),
            "got: {:?}",
            err
        );

        assert!(
            !work
                .path()
                .join("notes")
                .join(crate::schema::SCHEMA_FILE_NAME)
                .exists(),
            "the newly-written schema file must be removed on rollback — there is no \
             HEAD content for a brand-new scope to fall back to"
        );
        assert_eq!(
            head_before,
            head_sha(&work),
            "HEAD must not move on a rolled-back pre-commit failure"
        );
        assert_eq!(
            git_status_ignoring_state_db(&work),
            "",
            "the aborted `git add` must be unstaged too — no leftover addition that \
             could ride along on a later, unrelated commit"
        );
    }

    #[tokio::test]
    async fn update_schema_precommit_failure_on_existing_schema_restores_previous_content() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");

        // Commit an existing schema for `notes/` BEFORE forcing commits to fail, so
        // there is real HEAD content for the rollback to restore.
        let original = "fields:\n  status:\n    type: enum\n    values: [active]\n";
        write_schema_file(&work, "notes", original);
        git_commit_all(
            &work,
            &format!("notes/{}", crate::schema::SCHEMA_FILE_NAME),
            "add notes schema",
        );
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        // Built AFTER the real commit above, so the cache actually knows about the
        // existing `notes/` scope (needed for `update_schema` to resolve it and for
        // `write_raw_file` to see the write as an overwrite, not a create).
        let (server, _config) = make_git_backed_server(&work);

        let result = server
            .update_schema(Parameters(UpdateSchemaParams {
                path: Some("notes".into()),
                operation: "add_values".into(),
                field: "status".into(),
                values: Some(vec!["beta".into()]),
                definition: None,
                dry_run: None,
                force: None,
                acknowledge_root_change: None,
            }))
            .await;

        let err = result.expect_err("a rejected pre-commit hook must fail the schema write");
        assert_eq!(
            outcome_of(&err.data),
            Some("failed_no_change"),
            "got: {:?}",
            err
        );

        let written = work
            .path()
            .join("notes")
            .join(crate::schema::SCHEMA_FILE_NAME);
        assert_eq!(
            std::fs::read_to_string(&written).unwrap(),
            original,
            "the overwrite must be rolled back to the previous HEAD content, not left \
             holding the new (uncommitted) schema"
        );
        assert_eq!(
            head_before,
            head_sha(&work),
            "HEAD must not move on a rolled-back pre-commit failure"
        );
        assert_eq!(
            git_status_ignoring_state_db(&work),
            "",
            "working tree must be clean after rollback"
        );
    }

    #[tokio::test]
    async fn update_schema_postcommit_failure_leaves_commit_and_reports_pending_sync() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");

        let mut config = make_test_resolved_config(work.path());
        {
            let c = Arc::get_mut(&mut config).unwrap();
            c.write.dedup_enabled = false;
            // No such path — `git fetch` fails immediately, no network required, but
            // only AFTER the schema write's `git add`/`git commit` have already
            // succeeded locally.
            c.source.git_url = Some("/nonexistent/path/to/repo.git".to_string());
        }
        let server = make_write_test_server(&work, &["**/*.md".to_string()], config);

        let result = server
            .update_schema(Parameters(UpdateSchemaParams {
                path: Some("notes".into()),
                operation: "add_values".into(),
                field: "status".into(),
                values: Some(vec!["active".into()]),
                definition: None,
                dry_run: None,
                force: None,
                acknowledge_root_change: None,
            }))
            .await;

        let result = result.expect("a post-commit sync failure must still report as success");
        assert_eq!(
            outcome_of(&result.structured_content),
            Some("committed_pending_sync"),
            "got: {:?}",
            result
        );
        let text = format!("{:?}", result.content);
        assert!(
            text.contains("push") && text.contains("sync on the next successful write"),
            "must explain the push failure and that sync is pending: {text}"
        );

        // The schema change IS a real local commit — the file stays written, and HEAD
        // records it. None of this is rolled back just because the push failed.
        let written = work
            .path()
            .join("notes")
            .join(crate::schema::SCHEMA_FILE_NAME);
        assert!(written.exists());
        let show = std::process::Command::new("git")
            .args(["show", "--name-only", "--format=", "HEAD"])
            .current_dir(work.path())
            .output()
            .unwrap();
        assert!(
            String::from_utf8_lossy(&show.stdout).contains(crate::schema::SCHEMA_FILE_NAME),
            "the schema commit must be present in local HEAD"
        );

        // The reasoning behind NOT rolling back a PostCommit failure only pays off if
        // the shared schema cache was actually rebuilt from the new (committed)
        // content despite the push failure — mirrors
        // `create_document_immediately_after_update_schema_validates_against_the_new_rules`,
        // but for the pending-sync outcome instead of a clean sync. If `update_schema`
        // skipped the cache rebuild whenever the write wasn't a clean `Synced`, this
        // next call would wrongly reject "beta" against the stale pre-change schema.
        let accepted = server
            .create_document(Parameters(CreateDocumentParams {
                path: "notes/after.md".to_string(),
                content: "---\ntitle: After\nstatus: beta\n---\n\n# Body\n".to_string(),
                message: None,
                force_new: Some(true),
            }))
            .await;
        assert!(
            accepted.is_err(),
            "sanity check: this call's OWN schema edit only added 'active', not \
             'beta' — this must still be rejected, otherwise this test would not be \
             distinguishing 'cache rebuilt' from 'no schema at all'"
        );
        let err_msg = format!("{:?}", accepted.err());
        assert!(
            err_msg.contains("beta"),
            "must be rejected specifically for the 'beta' value, not some unrelated \
             failure: {err_msg}"
        );
    }

    /// When there is no git repository at all, `git add` fails (`PreCommit`) and the
    /// rollback attempt (`git reset` via `unstage`) ALSO fails, since there is no
    /// repo to reset anything in. This is the third, worse outcome: the schema file
    /// is gone from disk with no corresponding commit anywhere. It must be reported
    /// distinctly rather than masquerading as either a clean write or a clean no-op —
    /// mirrors `delete_document_with_no_git_repo_reports_inconsistent_state`.
    #[tokio::test]
    async fn update_schema_rollback_failure_reports_inconsistent_state() {
        let tmp = tempfile::tempdir().unwrap();
        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        let result = server
            .update_schema(Parameters(UpdateSchemaParams {
                path: Some("notes".into()),
                operation: "add_values".into(),
                field: "tags".into(),
                values: Some(vec!["x".into()]),
                definition: None,
                dry_run: None,
                force: None,
                acknowledge_root_change: None,
            }))
            .await;

        let err = result.expect_err("writing a schema with no git repo must fail");
        assert_eq!(
            outcome_of(&err.data),
            Some("failed_inconsistent_state"),
            "got: {:?}",
            err
        );
        assert!(
            err.message.contains("INCONSISTENT"),
            "the message must call out the inconsistent state loudly, got: {}",
            err.message
        );

        // The remove succeeded (there is no repo to fail that part), but the
        // subsequent `unstage` could not run against a nonexistent repo — that
        // mismatch (file gone, no git awareness of it ever having existed) IS the
        // inconsistent state being reported.
        assert!(
            !tmp.path()
                .join("notes")
                .join(crate::schema::SCHEMA_FILE_NAME)
                .exists()
        );
    }

    // -- move_directory_error_to_mcp_error: destination-cascade wording -----

    #[test]
    fn validation_error_names_the_destination_when_a_schema_file_relocated() {
        // The crux requirement: when the source subtree's own schema file is
        // moving too, the MCP-facing error must make clear the DESTINATION's
        // (re-parented) cascade is why documents that were valid at the source
        // now fail, not leave the caller staring at a bare validation failure.
        let err = move_directory_error_to_mcp_error(
            DirectoryMoveError::Validation {
                failures: vec![(
                    "dest/target/sub/a.md".to_string(),
                    crate::validate::ValidationResult {
                        file_path: "dest/target/sub/a.md".to_string(),
                        valid: false,
                        errors: vec!["missing required field 'extra_required'".to_string()],
                        field_errors: vec![],
                    },
                )],
                moved_schema_files: vec![(
                    format!("src/sub/{}", crate::schema::SCHEMA_FILE_NAME),
                    format!("dest/target/sub/{}", crate::schema::SCHEMA_FILE_NAME),
                )],
            },
            "src",
            "dest/target",
        );

        assert!(
            err.message.contains("DESTINATION"),
            "must name the destination cascade as the reason: {}",
            err.message
        );
        assert!(
            err.message.contains("dest/target"),
            "must name the destination path: {}",
            err.message
        );
        assert!(
            err.message.contains("re-parent"),
            "must explain the schema file re-parented onto the destination: {}",
            err.message
        );
        assert!(
            err.message
                .contains(&format!("src/sub/{}", crate::schema::SCHEMA_FILE_NAME)),
            "must name which schema file relocated: {}",
            err.message
        );
    }

    #[test]
    fn validation_error_without_a_relocated_schema_file_omits_the_reparenting_note() {
        let err = move_directory_error_to_mcp_error(
            DirectoryMoveError::Validation {
                failures: vec![(
                    "dest/a.md".to_string(),
                    crate::validate::ValidationResult {
                        file_path: "dest/a.md".to_string(),
                        valid: false,
                        errors: vec!["missing required field 'x'".to_string()],
                        field_errors: vec![],
                    },
                )],
                moved_schema_files: vec![],
            },
            "src",
            "dest",
        );

        assert!(
            !err.message.contains("re-parent"),
            "no schema file moved, so there is nothing to explain re-parenting for: {}",
            err.message
        );
    }
}
