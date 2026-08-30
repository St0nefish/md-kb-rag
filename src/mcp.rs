use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::RwLock;

use chrono::{DateTime, NaiveDate};

use anyhow::Context as _;
use globset::{Glob, GlobSet, GlobSetBuilder};
use rmcp::{
    ErrorData as McpError, RoleServer, ServerHandler, handler::server::wrapper::Parameters,
    model::*, schemars, service::RequestContext, tool, tool_handler, tool_router,
};
use tracing::{debug, error, warn};

use crate::{
    config::ResolvedConfig,
    document_fields,
    embed::EmbedClient,
    git,
    qdrant::{CHUNK_TEXT_KEY, IndexKind, QdrantStore},
    rerank::RerankClient,
    retrieval::{
        self, DocumentIndexDeps, GetDocumentError, RetrievalDeps, SearchFilters, SearchOptions,
    },
    schema::SchemaCache,
    state::{DocumentIndex, DocumentQuery, FieldFilter, OrderBy, StateDb},
    validate,
    write::{
        self, DirectoryMoveError, DirectoryMoveSuccess, FrontmatterEdit, WriteDeps, WriteError,
        WriteOutcome as CoreWriteOutcome, WriteRequest, WriteSuccess,
    },
};

const MAX_QUERY_LEN: usize = 4096;
const MAX_PATH_LEN: usize = 4096;
const MAX_FILTER_STR_LEN: usize = 256;
const MAX_CONTENT_LEN: usize = 512 * 1024; // 512 KB
/// Cap on the number of operations a single `write_document` `frontmatter_patch`
/// call may carry — a document write should rarely need more than a handful of
/// field edits at once; this bounds the work `write::apply_frontmatter_patch`
/// does per call, mirroring `MAX_SCHEMA_VALUES`'s reasoning for `update_schema`.
const MAX_FRONTMATTER_PATCH_OPS: usize = 20;
/// Cap on the `values` list of a single `add_values`/`remove_values`
/// `frontmatter_patch` operation.
const MAX_FRONTMATTER_PATCH_VALUES: usize = 200;
/// Cap on the serialized size of a single `frontmatter_patch` value —
/// mirrors `MAX_SCHEMA_DEFINITION_LEN`'s identical reasoning for `update_schema`'s
/// `set_field` definitions: a document's frontmatter is committed and re-parsed
/// on every read, so an oversized value is a durable cost, not a transient one.
const MAX_FRONTMATTER_VALUE_LEN: usize = 4 * 1024;

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
/// Cap on the unified diff embedded in a write/delete tool's `structured_content`
/// (fix #129). Mirrors `MAX_SCHEMA_DEFINITION_LEN`'s convention for bounding a
/// single text blob: `WriteSuccess::diff` is unbounded by design (a full replace
/// of a large document produces a large diff, and the text channel already
/// carries it whole), but embedding it verbatim into `structured_content` too
/// would let one write emit a multi-megabyte tool result. See
/// [`capped_diff`] for the same unbounded-source/bounded-payload split
/// [`capped_casualties`] already applies to `update_schema`'s casualty list.
const MAX_STRUCTURED_DIFF_BYTES: usize = 8 * 1024;

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

/// `search`'s `filters` parameter as delivered by an MCP client.
///
/// Same fix as [`FieldDefinitionInput`], applied to the same shape of bug: the old
/// type here, `Option<serde_json::Map<String, serde_json::Value>>`, advertises no
/// schema constraint at all (schemars emits `{}` for a bare `serde_json::Map`), so a
/// calling model has no way to learn from the tool schema that a scalar means
/// equality, an array means any-of, or that an object accepts
/// `any_of`/`all_of`/`gte`/`lte`/`gt`/`lt`. It has to learn that from prose alone —
/// and, per the same failure mode `FieldDefinitionInput` exists to cover, at least
/// one client class responds to an under-specified object parameter by sending it
/// JSON-encoded as a string instead, which the old `Option<Map<...>>` field rejected
/// with a raw deserialize error rather than a caller-actionable one.
///
/// Unlike `FieldDefinitionInput`, there is no existing typed Rust struct to delegate
/// to for the advertised schema — `filters` is a map keyed by arbitrary caller-chosen
/// field names, each valued by one of several shapes — so [`json_schema`] below
/// builds that schema by hand instead of deriving it. Actual per-condition parsing
/// still happens in `parse_field_filter`, unchanged: this type only fixes what the
/// tool schema advertises and adds the same string-tolerance fallback, it does not
/// duplicate that function's shape checking or its field-named error messages.
///
/// [`json_schema`]: schemars::JsonSchema::json_schema
#[derive(Debug, Clone, PartialEq)]
pub struct SearchFiltersInput(pub serde_json::Map<String, serde_json::Value>);

impl<'de> serde::Deserialize<'de> for SearchFiltersInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        parse_search_filters_input(value)
            .map(SearchFiltersInput)
            .map_err(serde::de::Error::custom)
    }
}

/// Parse `search`'s `filters` argument from JSON: an object directly, or (see
/// [`SearchFiltersInput`]) a string containing one. Mirrors
/// [`parse_field_definition`]'s two-shape acceptance and error style.
fn parse_search_filters_input(
    value: serde_json::Value,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    use serde_json::Value;

    match value {
        Value::Object(map) => Ok(map),
        Value::String(s) => match serde_json::from_str::<Value>(&s) {
            Ok(Value::Object(map)) => Ok(map),
            Ok(other) => Err(filters_shape_error(&other)),
            Err(e) => Err(format!(
                "filters must be a JSON object mapping frontmatter field names to \
                 conditions. A JSON string containing that object is also accepted, \
                 but this string is not valid JSON: {e}"
            )),
        },
        other => Err(filters_shape_error(&other)),
    }
}

/// Build the "wrong shape entirely" error for [`parse_search_filters_input`], naming
/// what was actually received without echoing its (possibly large) content.
fn filters_shape_error(value: &serde_json::Value) -> String {
    format!(
        "filters must be a JSON object mapping frontmatter field names to conditions, \
         got {}",
        json_value_kind(value)
    )
}

impl schemars::JsonSchema for SearchFiltersInput {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "SearchFilters".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::SearchFiltersInput"))
    }

    // Hand-built rather than derived (see this type's doc comment): describes an
    // object whose values ("filter conditions") are, per field, a scalar
    // (equality), an array of scalars (any-of), or an object carrying
    // `any_of`/`all_of`/`gte`/`lte`/`gt`/`lt` — exactly the shapes
    // `parse_field_filter` accepts. Kept tight and caller-facing (see #126): no
    // implementation rationale leaks into the emitted schema, only the shape a
    // caller needs to construct a valid `filters` value.
    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "object",
            "description": "Frontmatter criteria by field (dot-path keys). Each \
                value is either a scalar (equality), an array of scalars (any-of), \
                or an object with any_of/all_of (explicit set match) or \
                gte/lte/gt/lt (numeric range).",
            "additionalProperties": {
                "description": "One field's filter condition.",
                "anyOf": [
                    { "type": ["string", "number", "boolean"] },
                    {
                        "type": "array",
                        "items": { "type": ["string", "number", "boolean"] }
                    },
                    {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "any_of": {
                                "type": "array",
                                "items": { "type": ["string", "number", "boolean"] }
                            },
                            "all_of": {
                                "type": "array",
                                "items": { "type": ["string", "number", "boolean"] }
                            },
                            "gte": { "type": "number" },
                            "lte": { "type": "number" },
                            "gt": { "type": "number" },
                            "lt": { "type": "number" }
                        }
                    }
                ]
            }
        })
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

/// Cap a casualty list for `structured_content`, mirroring the cap
/// [`render_casualties`] already applies to the text half.
///
/// `documents_broken_by` deliberately returns the *complete* casualty list — the
/// force/refuse decision needs completeness, so that query stays unbounded — but
/// embedding the full `Vec` verbatim into `structured_content` let a schema
/// tightening that broke thousands of documents emit a multi-megabyte tool result
/// while the text channel silently stayed capped at [`MAX_REPORTED_CASUALTIES`].
/// Returns the capped list alongside the true total and whether it was truncated,
/// the same `total`/`has_more` shape `search` uses elsewhere in this file, so a
/// client that only reads `structured_content` can still tell "empty" from
/// "truncated" rather than only ever seeing the first page.
fn capped_casualties(casualties: &[serde_json::Value]) -> (Vec<serde_json::Value>, usize, bool) {
    let total = casualties.len();
    let capped = casualties
        .iter()
        .take(MAX_REPORTED_CASUALTIES)
        .cloned()
        .collect();
    (capped, total, total > MAX_REPORTED_CASUALTIES)
}

/// Cap `diff` (a `WriteSuccess`/`DirectoryMoveSuccess` unified diff) for
/// `structured_content` — see [`MAX_STRUCTURED_DIFF_BYTES`]'s doc comment.
/// Returns `(capped_diff, diff_truncated, diff_total_bytes)`, the same
/// capped-value/truncated-flag/true-total shape [`capped_casualties`] returns,
/// so a client reading only `structured_content` can tell "the whole diff" from
/// "cut off, and by how much" rather than only ever seeing the first slice.
///
/// Cuts at a `char_boundary` at or before the byte cap — `diff` is arbitrary
/// document text, so a naive byte-index cut could otherwise land inside a
/// multi-byte UTF-8 character and produce an invalid `&str` slice.
fn capped_diff(diff: &str) -> (String, bool, usize) {
    let total = diff.len();
    if total <= MAX_STRUCTURED_DIFF_BYTES {
        return (diff.to_string(), false, total);
    }
    let mut end = MAX_STRUCTURED_DIFF_BYTES;
    while end > 0 && !diff.is_char_boundary(end) {
        end -= 1;
    }
    (diff[..end].to_string(), true, total)
}

/// Render `WriteSuccess::rebased_paths`/`DirectoryMoveSuccess::rebased_paths`
/// (a `Vec<PathBuf>`) as `Vec<String>` for `structured_content` — mirrors
/// `web.rs`'s `write_success_response` doing the identical `to_string_lossy`
/// conversion for its own fixed HTTP contract, rather than letting `json!`
/// serialize `PathBuf` directly (which errors on a non-UTF-8 path instead of
/// degrading gracefully).
fn rebased_paths_json(rebased_paths: &[PathBuf]) -> Vec<String> {
    rebased_paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect()
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
    // `false` matches a stored boolean and `45` matches a stored integer. Also caps
    // each value's length — master enforced this per-value (domain_too_long_is_rejected
    // et al.); folding those scalar params into this generic `filters` map must not
    // lose it, since an unbounded value is still a durable cost against Qdrant/SQLite
    // regardless of which filter form carried it in.
    let canonical = |value: &Value| -> Result<String, String> {
        let text = document_fields::canonical_text(value).ok_or_else(|| {
            format!(
                "filter '{}': expected a string, number, or boolean, got {}",
                field, value
            )
        })?;
        if text.len() > MAX_FILTER_STR_LEN {
            return Err(format!(
                "filter '{}': value too long ({} chars, max {})",
                field,
                text.len(),
                MAX_FILTER_STR_LEN
            ));
        }
        Ok(text)
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

/// Parse a `search` call's raw `filters` map into the typed representation shared by
/// both backends: SQLite (`state::StateDb::push_where`, enumeration mode) and Qdrant
/// (`qdrant::lower_field_filters`, query mode).
fn parse_filters(
    raw_filters: &Option<SearchFiltersInput>,
) -> Result<Vec<(String, FieldFilter)>, McpError> {
    let invalid = |msg: String| McpError::invalid_params(msg, None);

    let mut filters = Vec::new();
    if let Some(raw_filters) = raw_filters {
        let raw_filters = &raw_filters.0;
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
    Ok(filters)
}

/// Shared length check for `path_prefix`, used by both enumeration mode
/// ([`build_document_query`]) and query mode ([`KbSearchServer::search`]).
fn validate_path_prefix(path_prefix: &Option<String>) -> Result<(), McpError> {
    if let Some(prefix) = path_prefix
        && prefix.len() > MAX_FILTER_STR_LEN
    {
        return Err(McpError::invalid_params(
            format!(
                "path_prefix too long: {} chars (max {})",
                prefix.len(),
                MAX_FILTER_STR_LEN
            ),
            None,
        ));
    }
    Ok(())
}

/// Shared count check for `fields`, used by both enumeration mode
/// ([`build_document_query`]) and query+document mode ([`KbSearchServer::search`]).
fn validate_fields_count(fields: &Option<Vec<String>>) -> Result<(), McpError> {
    if let Some(fields) = fields
        && fields.len() > MAX_LIST_FILTERS
    {
        return Err(McpError::invalid_params(
            format!(
                "too many fields requested: {} (max {})",
                fields.len(),
                MAX_LIST_FILTERS
            ),
            None,
        ));
    }
    Ok(())
}

/// Build a validated [`DocumentQuery`] from tool parameters — the `search` tool's
/// document-granularity, no-query (enumeration) combination.
fn build_document_query(params: &SearchParams) -> Result<DocumentQuery, McpError> {
    let invalid = |msg: String| McpError::invalid_params(msg, None);

    let filters = parse_filters(&params.filters)?;

    let order_by = match &params.order_by {
        Some(raw) => OrderBy::parse(raw).map_err(invalid)?,
        None => OrderBy::default(),
    };

    validate_fields_count(&params.fields)?;
    validate_path_prefix(&params.path_prefix)?;

    let mtime_after = params
        .modified_after
        .as_deref()
        .map(parse_date_to_timestamp)
        .transpose()
        .map_err(invalid)?;
    let mtime_before = params
        .modified_before
        .as_deref()
        .map(parse_date_to_timestamp)
        .transpose()
        .map_err(invalid)?;

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
        mtime_after,
        mtime_before,
    })
}

/// Parse `search`'s `filters` for query mode (chunk or grouped-document
/// granularity, both against Qdrant) and lower them to Qdrant conditions.
///
/// Every named field must carry a payload index — checked here against
/// [`crate::qdrant::all_indexed_fields`] (the union of schema-declared and legacy
/// config-declared fields, i.e. the real Qdrant index set) and rejected by name
/// otherwise: Qdrant can filter an unindexed field, just by a full scan rather than
/// an index, and silently doing that would return more than what was asked for.
/// This is the one piece of validation [`crate::qdrant::lower_field_filters`]
/// deliberately leaves to its caller — see that function's doc comment.
fn build_query_conditions(
    params: &SearchParams,
    config: &ResolvedConfig,
    schemas: &SchemaCache,
) -> Result<Vec<qdrant_client::qdrant::Condition>, McpError> {
    let filters = parse_filters(&params.filters)?;

    let indexed: std::collections::HashMap<String, IndexKind> =
        crate::qdrant::all_indexed_fields(config, schemas)
            .into_iter()
            .map(|f| (f.name, f.kind))
            .collect();

    if let Some((field, _)) = filters.iter().find(|(f, _)| !indexed.contains_key(f)) {
        return Err(McpError::invalid_params(
            format!(
                "filter '{field}' is not indexed for Qdrant queries; mark it `indexed: true` \
                 in the governing .kb-schema.yaml to filter on it with a search query"
            ),
            None,
        ));
    }

    crate::qdrant::lower_field_filters(&filters, &indexed)
        .map_err(|e| McpError::invalid_params(e, None))
}

/// Which kind of result `search` returns: one row per chunk, or one per document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Granularity {
    Chunk,
    Document,
}

/// Parse a caller-supplied `granularity`, rejecting anything unrecognized.
fn parse_granularity(raw: &str) -> Result<Granularity, McpError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "chunk" => Ok(Granularity::Chunk),
        "document" => Ok(Granularity::Document),
        other => Err(McpError::invalid_params(
            format!("unknown granularity '{other}': expected 'chunk' or 'document'"),
            None,
        )),
    }
}

/// Resolve the effective granularity for a `search` call: the caller's explicit
/// choice if given, otherwise `chunk` when a query is present and `document` when it
/// is not — reproducing the old `search` and `list_documents` tools' behaviour
/// exactly. Pure and I/O-free so the default-in-both-directions property is
/// unit-testable without a live index.
fn resolve_granularity(
    query_present: bool,
    requested: Option<&str>,
) -> Result<Granularity, McpError> {
    match requested {
        Some(raw) => parse_granularity(raw),
        None if query_present => Ok(Granularity::Chunk),
        None => Ok(Granularity::Document),
    }
}

/// Whether `search`'s `query` should be treated as present. A blank/whitespace-only
/// string is treated the same as an absent query — there is nothing to embed —
/// rather than sent to the embedder as a no-op vector search.
fn query_is_present(query: &Option<String>) -> bool {
    query.as_deref().is_some_and(|q| !q.trim().is_empty())
}

fn validate_search_params(params: &SearchParams) -> Result<(), McpError> {
    if let Some(query) = &params.query
        && query.len() > MAX_QUERY_LEN
    {
        return Err(McpError::invalid_params(
            format!("query exceeds maximum length of {MAX_QUERY_LEN} characters"),
            None,
        ));
    }
    Ok(())
}

/// Parameters for `get_document`.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GetDocumentParams {
    /// Relative path, unique basename, or absolute path.
    pub path: String,
    /// First line to return (1-based, inclusive).
    #[serde(default)]
    pub start_line: Option<usize>,
    /// Last line to return (1-based, inclusive).
    #[serde(default)]
    pub end_line: Option<usize>,
}

/// Parameters for `get_schema`.
#[derive(Debug, Default, serde::Deserialize, schemars::JsonSchema)]
pub struct GetSchemaParams {
    /// Directory or document path; omit for the root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,

    /// Only report these fields (dot-paths).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fields: Option<Vec<String>>,

    /// Only fields with a closed value set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values_only: Option<bool>,
}

/// Parameters for `update_schema`.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct UpdateSchemaParams {
    /// Directory to edit; omit for the KB root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,

    /// Which operation to perform.
    pub operation: String,

    /// Field this operation targets (dot-path).
    pub field: String,

    /// Values, for add_values/remove_values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<String>>,

    /// Field definition for set_field; a JSON string also works.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub definition: Option<FieldDefinitionInput>,

    /// Preview the change without writing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_run: Option<bool>,

    /// Apply even if existing documents would fail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force: Option<bool>,

    /// Required for root-scope changes; see schema-tag-policy.md.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acknowledge_root_change: Option<bool>,
}

/// Parameters for `search` (also covers enumeration).
#[derive(Debug, Default, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchParams {
    /// Semantic query. Omit to enumerate everything.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,

    /// Frontmatter criteria by field (dot-paths).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filters: Option<SearchFiltersInput>,

    /// Restrict to paths starting with this prefix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_prefix: Option<String>,

    /// "chunk" or "document"; defaults per query presence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub granularity: Option<String>,

    /// Maximum results to return.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,

    /// Number to skip, for paging. Exhaustive (no depth limit) in enumeration
    /// mode. In query mode (chunk or document granularity), pages over the
    /// already-ranked results, so `offset + limit` is capped at
    /// reranking.candidate_limit when reranking is enabled, or a fixed depth
    /// otherwise — a request past that bound gets `offset_truncated: true` in
    /// the response instead of a silently short or empty page.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<u64>,

    /// Sort key, enumeration only: path/title/mtime/indexed_at.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order_by: Option<String>,

    /// Sort descending (enumeration only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub descending: Option<bool>,

    /// Relevance floor (query only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_score: Option<f32>,

    /// Add a score-breakdown line per result (query + chunk granularity only;
    /// rejected at document granularity, since a grouped result collapses to
    /// one row per document with no per-arm chunk score to report).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explain: Option<bool>,

    /// Frontmatter fields to include per result (dot-paths; document
    /// granularity only, enumeration or query — rejected at chunk
    /// granularity, since a chunk result never joins the document metadata
    /// index this draws from).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fields: Option<Vec<String>>,

    /// Exclude docs modified before this date.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified_after: Option<String>,

    /// Exclude docs modified after this date.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified_before: Option<String>,
}

/// Parameters for `write_document`.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct WriteDocumentParams {
    /// Document or directory path, relative to the KB root.
    pub path: String,
    /// Whole file content, including frontmatter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Surgical edit: exact text to replace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_string: Option<String>,
    /// Surgical edit: its replacement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_string: Option<String>,
    /// Structured frontmatter edits; combines with `append`, not with `content`
    /// or `old_string`/`new_string`. See `write::FrontmatterEdit`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frontmatter_patch: Option<Vec<FrontmatterPatchOp>>,
    /// Append this text to the end of the document body; combines with
    /// `frontmatter_patch`, not with `content` or `old_string`/`new_string`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub append: Option<String>,
    /// Relocate here; combines with an edit, or stands alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_path: Option<String>,
    /// Commit message; a default is generated if omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Stale-read guard: content_hash from a prior get_document.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_hash: Option<String>,
    /// Skip the near-duplicate check when creating.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force_new: Option<bool>,
}

/// One structured frontmatter edit, mirroring `update_schema`'s
/// `operation`/`field`/`values`/`definition` shape — this codebase's
/// established idiom for "a structured edit instead of a free-text patch"
/// (see `UpdateSchemaParams`, `build_schema_edit`) — applied here to a
/// document's own frontmatter values. See `write::FrontmatterEdit`.
#[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
pub struct FrontmatterPatchOp {
    /// "set_field" | "remove_field" | "add_values" | "remove_values".
    pub operation: String,
    /// Frontmatter field this operation targets (dot-path).
    pub field: String,
    /// New value, for set_field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<serde_json::Value>,
    /// Values to add/remove, for add_values/remove_values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<serde_json::Value>>,
}

/// Parameters for `delete_document`.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct DeleteDocumentParams {
    /// Relative path, unique basename, or absolute path.
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
    /// Structured frontmatter edits only — see `write::FrontmatterEdit`.
    Patch { edits: Vec<FrontmatterEdit> },
    /// Append only.
    Append { text: String },
    /// Structured frontmatter edits AND an append, in the same call — applied
    /// patch-then-append (see `write_document_edit`).
    PatchAppend {
        edits: Vec<FrontmatterEdit>,
        text: String,
    },
}

/// Build a single `write::FrontmatterEdit` from the wire shape a caller sent,
/// mirroring `build_schema_edit`'s identical role for `update_schema`.
fn build_frontmatter_edit(op: &FrontmatterPatchOp) -> Result<FrontmatterEdit, String> {
    if op.field.trim().is_empty() {
        return Err("frontmatter_patch: field must not be empty".to_string());
    }
    if op.field.len() > MAX_FILTER_STR_LEN {
        return Err(format!(
            "frontmatter_patch: field name too long: {} chars (max {})",
            op.field.len(),
            MAX_FILTER_STR_LEN
        ));
    }
    let check_value_size = |v: &serde_json::Value| -> Result<(), String> {
        let size = serde_json::to_string(v)
            .map(|s| s.len())
            .unwrap_or(usize::MAX);
        if size > MAX_FRONTMATTER_VALUE_LEN {
            return Err(format!(
                "frontmatter_patch: value for '{}' too large (max {} bytes)",
                op.field, MAX_FRONTMATTER_VALUE_LEN
            ));
        }
        Ok(())
    };

    match op.operation.trim().to_ascii_lowercase().as_str() {
        "set_field" => {
            let value = op
                .value
                .clone()
                .ok_or_else(|| "frontmatter_patch: 'set_field' requires a value".to_string())?;
            check_value_size(&value)?;
            Ok(FrontmatterEdit::SetField {
                field: op.field.clone(),
                value,
            })
        }
        "remove_field" => Ok(FrontmatterEdit::RemoveField {
            field: op.field.clone(),
        }),
        op_name @ ("add_values" | "remove_values") => {
            let values = op.values.clone().filter(|v| !v.is_empty()).ok_or_else(|| {
                format!("frontmatter_patch: '{op_name}' requires a non-empty values list")
            })?;
            if values.len() > MAX_FRONTMATTER_PATCH_VALUES {
                return Err(format!(
                    "frontmatter_patch: too many values for '{}': {} (max {})",
                    op.field,
                    values.len(),
                    MAX_FRONTMATTER_PATCH_VALUES
                ));
            }
            for v in &values {
                check_value_size(v)?;
            }
            if op_name == "add_values" {
                Ok(FrontmatterEdit::AddValues {
                    field: op.field.clone(),
                    values,
                })
            } else {
                Ok(FrontmatterEdit::RemoveValues {
                    field: op.field.clone(),
                    values,
                })
            }
        }
        other => Err(format!(
            "frontmatter_patch: unknown operation '{other}': expected set_field, remove_field, \
             add_values, or remove_values"
        )),
    }
}

/// Parse every op in a `frontmatter_patch` list, in order, applying the same
/// per-call size cap `MAX_FRONTMATTER_PATCH_OPS` bounds.
fn parse_frontmatter_patch_ops(ops: &[FrontmatterPatchOp]) -> Result<Vec<FrontmatterEdit>, String> {
    if ops.len() > MAX_FRONTMATTER_PATCH_OPS {
        return Err(format!(
            "frontmatter_patch: too many operations: {} (max {})",
            ops.len(),
            MAX_FRONTMATTER_PATCH_OPS
        ));
    }
    ops.iter().map(build_frontmatter_edit).collect()
}

/// Parse and validate the content-edit fields of `WriteDocumentParams`,
/// returning a typed `Option<EditMode>` or a human-readable error string.
///
/// Rules:
/// - SURGICAL = `old_string` AND `new_string` both `Some`, `content` is `None`.
/// - FULL = `content` is `Some`, both `old_string` and `new_string` are `None`.
/// - PATCH/APPEND/PATCH_APPEND = `frontmatter_patch` and/or `append` is `Some`.
///   These two combine freely with EACH OTHER (patch is applied first, then
///   append — see `write_document_edit`), but neither may combine with FULL
///   or SURGICAL: those are whole-document edits, these are structured edits
///   to part of the document, and there is no well-defined way to apply both
///   to the same call.
/// - Neither FULL, SURGICAL, nor PATCH/APPEND, but `new_path` is `Some`:
///   `Ok(None)` — a pure move with content left unchanged. The caller
///   (`write_document`'s edit path) reads the document's current content
///   itself and passes it through unchanged.
/// - None of the above, and `new_path` is also `None`: rejected — at least
///   one edit mode (or a move) must be requested.
/// - FULL and SURGICAL together are always rejected, regardless of `new_path`
///   — the two whole-document edit modes remain mutually exclusive WITH EACH
///   OTHER; `new_path` is an orthogonal, independent axis that may combine
///   with any one edit mode (or neither, for a pure move).
/// - Surgical with `old_string == new_string` is rejected (no-op), same as an
///   `append` that is empty or whitespace-only.
pub fn parse_edit_mode(params: &WriteDocumentParams) -> Result<Option<EditMode>, String> {
    let has_content = params.content.is_some();
    let has_move = params.new_path.is_some();

    // old_string/new_string must arrive as a pair, independent of every other
    // mode — checked first so a caller who supplied only one gets that
    // specific error rather than a generic "mutually exclusive" one.
    match (params.old_string.is_some(), params.new_string.is_some()) {
        (true, false) => {
            return Err(
                "old_string requires new_string; provide both for a surgical edit. (If you only \
                 meant to move the document, omit old_string entirely and pass new_path alone.)"
                    .to_string(),
            );
        }
        (false, true) => {
            return Err(
                "new_string requires old_string; provide both for a surgical edit. (If you only \
                 meant to move the document, omit new_string entirely and pass new_path alone.)"
                    .to_string(),
            );
        }
        _ => {}
    }
    let has_surgical = params.old_string.is_some();
    let has_patch = params.frontmatter_patch.is_some();
    let has_append = params.append.is_some();

    if (has_content || has_surgical) && (has_patch || has_append) {
        return Err(
            "content and old_string/new_string are whole-document edits, mutually exclusive \
             with frontmatter_patch/append (structured edits to part of the document). \
             Provide one or the other — frontmatter_patch and append may combine with each \
             other, just not with content or old_string/new_string."
                .to_string(),
        );
    }
    if has_content && has_surgical {
        return Err("content is mutually exclusive with old_string/new_string; \
             provide either content (full replace) or old_string+new_string (surgical edit) \
             — not both. new_path may be combined with either one (or with neither, for a \
             pure move), but it does not resolve a conflict between the two edit modes \
             themselves."
            .to_string());
    }

    if has_content {
        return Ok(Some(EditMode::Full {
            content: params.content.clone().unwrap(),
        }));
    }
    if has_surgical {
        let old = params.old_string.clone().unwrap();
        let new = params.new_string.clone().unwrap();
        if old == new {
            return Err(
                "old_string and new_string are identical — no change would be made".to_string(),
            );
        }
        return Ok(Some(EditMode::Surgical { old, new }));
    }
    if has_patch || has_append {
        let edits = match &params.frontmatter_patch {
            Some(ops) if !ops.is_empty() => Some(parse_frontmatter_patch_ops(ops)?),
            Some(_) => {
                return Err("frontmatter_patch must contain at least one operation".to_string());
            }
            None => None,
        };
        let text = match params.append.as_deref() {
            Some(t) if !t.trim().is_empty() => Some(t.to_string()),
            Some(_) => return Err("append must not be empty".to_string()),
            None => None,
        };
        return Ok(Some(match (edits, text) {
            (Some(edits), Some(text)) => EditMode::PatchAppend { edits, text },
            (Some(edits), None) => EditMode::Patch { edits },
            (None, Some(text)) => EditMode::Append { text },
            (None, None) => unreachable!(
                "has_patch || has_append guarantees at least one of edits/text is Some"
            ),
        }));
    }

    // No edit mode at all: a pure move if new_path was given, otherwise an error.
    if has_move {
        Ok(None)
    } else {
        Err(
            "must provide content (full replace), old_string+new_string (surgical edit), \
             frontmatter_patch (structured frontmatter edit), append (add to the end of the \
             body), or new_path (move) — at least one is required"
                .to_string(),
        )
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
/// (`write_document`/`delete_document`), exposed via
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

/// Build the fields every write/delete `structured_content` payload shares
/// (fix #129): `sha` and `diff` (capped — see [`capped_diff`] — with
/// `diff_truncated`/`diff_total_bytes` alongside it) and `rebased_paths`, plus
/// `sync_failure_cause` when the write landed as `committed_pending_sync`.
/// Before this, `structured_content` carried only `{"outcome", ...}` — a
/// client that prefers structured content over prose (Claude Code does) had
/// no programmatic way to learn the commit SHA or see the diff at all, the
/// same class of bug as #124 (`search`'s `structured_content` once regressed
/// to a bare `{"path_prefix_truncated": ...}`).
///
/// Returns a `serde_json::Map` rather than a full `Value` so
/// [`with_outcome_and_rewrites`]/[`with_outcome_and_referencing`] can merge in
/// their own move/delete-specific field on top.
fn write_success_structured_fields(
    outcome: WriteOutcome,
    success: &WriteSuccess,
) -> serde_json::Map<String, serde_json::Value> {
    let (diff, diff_truncated, diff_total_bytes) = capped_diff(&success.diff);
    let mut fields = serde_json::Map::new();
    fields.insert("outcome".to_string(), serde_json::json!(outcome.as_str()));
    fields.insert("sha".to_string(), serde_json::json!(success.sha));
    fields.insert("diff".to_string(), serde_json::json!(diff));
    fields.insert(
        "diff_truncated".to_string(),
        serde_json::json!(diff_truncated),
    );
    fields.insert(
        "diff_total_bytes".to_string(),
        serde_json::json!(diff_total_bytes),
    );
    fields.insert(
        "rebased_paths".to_string(),
        serde_json::json!(rebased_paths_json(&success.rebased_paths)),
    );
    if let Some(cause) = &success.sync_failure_cause {
        fields.insert("sync_failure_cause".to_string(), serde_json::json!(cause));
    }
    fields
}

/// Attach a machine-readable `{"outcome": ...}` discriminant to a successful
/// `CallToolResult`, alongside its human-readable text content — for
/// `write_document`'s create/edit path specifically: also attaches `sha`,
/// `diff`/`diff_truncated`/`diff_total_bytes`, `rebased_paths`, and
/// `sync_failure_cause` (see [`write_success_structured_fields`], fix #129),
/// plus `rewritten_paths`, the repo-relative paths of OTHER documents a MOVE
/// rewrote incoming links in (always `[]` for a non-move write, or a move
/// with nothing to rewrite). A move that silently edits other documents
/// without surfacing which ones is not acceptable, so this rides in
/// `structured_content` on every create/edit result, not just moves. See
/// [`with_outcome_and_referencing`] for `delete_document`'s equivalent.
fn with_outcome_and_rewrites(
    mut result: CallToolResult,
    outcome: WriteOutcome,
    success: &WriteSuccess,
) -> CallToolResult {
    let mut fields = write_success_structured_fields(outcome, success);
    fields.insert(
        "rewritten_paths".to_string(),
        serde_json::json!(success.rewritten_paths),
    );
    result.structured_content = Some(serde_json::Value::Object(fields));
    result
}

/// (#229) Like [`with_outcome_and_rewrites`], but for `delete_document`
/// specifically: attaches `referencing_paths`, the repo-relative paths of
/// OTHER documents that still link to the just-deleted document (always `[]`
/// when none exist, or when no `StateDb` was available to check — see
/// `write::WriteSuccess::referencing_paths`'s doc comment), alongside the same
/// `sha`/`diff`/`rebased_paths`/`sync_failure_cause` fields
/// [`write_success_structured_fields`] attaches for create/edit (fix #129).
/// The `referencing_paths` half is the caller-visible counterpart of the
/// reverse-link check #181 already logs server-side — an agent has no access
/// to that log, so the same information must reach it through the tool result
/// too.
fn with_outcome_and_referencing(
    mut result: CallToolResult,
    outcome: WriteOutcome,
    success: &WriteSuccess,
) -> CallToolResult {
    let mut fields = write_success_structured_fields(outcome, success);
    fields.insert(
        "referencing_paths".to_string(),
        serde_json::json!(success.referencing_paths),
    );
    result.structured_content = Some(serde_json::Value::Object(fields));
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
                &success,
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
                &success,
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
/// document MOVE — i.e. `write_document` was called with `new_path` set. It
/// disambiguates the `AlreadyExists` arm, which for a move reports a collision
/// at the DESTINATION, not at `rel_path` (the source, which — for a move — is
/// expected to already exist). The create path's own TOCTOU race, and every
/// non-move edit call, pass `None` here, preserving the original
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
                 Edit it with write_document, or pass \
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
        // 1. `write_document`'s own create-path pre-check (`abs_path.exists()`)
        //    already passed, so the file was created between that check and
        //    `write::write_document`'s `create_new` open. Restored to the
        //    pre-`write.rs`-extraction wording, which reported the absolute
        //    filesystem path rather than the repo-relative one — reconstructed
        //    here via `resolve_safe_write_path` since `WriteError::AlreadyExists`
        //    itself carries no path (kept a unit variant so `web.rs`'s exhaustive
        //    match needs no change to accommodate this). `dest_path` is `None`
        //    here, so this is the arm that runs.
        //
        // 2. `write_document` was called with `new_path` set (a MOVE, possibly
        //    combined with a content edit) and the DESTINATION already exists.
        //    `rel_path` here is the move's SOURCE (which legitimately exists —
        //    that's what made this an edit rather than a create), so reporting
        //    the collision against `rel_path` would misdirect the caller at the
        //    wrong file. `dest_path` disambiguates: when it's `Some`, this arm
        //    names the DESTINATION as what collided, not the source.
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
                        "File '{}' already exists; use write_document to modify it",
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
    // (#229) Named here so it can be included in both outcomes' text below —
    // empty for every delete with nothing (or nothing known) still linking to
    // the removed document.
    let referencing_note = if success.referencing_paths.is_empty() {
        String::new()
    } else {
        format!(
            "\n\nStill linked from {} other document(s): {}. This delete did not rewrite \
             or remove those links — they will dangle until each referencing document's own \
             next reindex drops the now-stale edge.",
            success.referencing_paths.len(),
            success.referencing_paths.join(", ")
        )
    };
    match success.outcome {
        CoreWriteOutcome::Synced => {
            let summary = format!(
                "Deleted '{}' (commit {}). Index cleanup has been queued and will complete shortly.{}",
                rel_path, success.sha, referencing_note
            );
            let mut result_text = summary;
            if !success.diff.is_empty() {
                result_text = format!("{}\n\n{}", result_text, success.diff);
            }
            with_outcome_and_referencing(
                CallToolResult::success(vec![Content::text(result_text)]),
                WriteOutcome::Synced,
                &success,
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
                 intervention. Index cleanup has been queued from the local copy.{}",
                rel_path, success.sha, cause, referencing_note
            );
            let mut result_text = summary;
            if !success.diff.is_empty() {
                result_text = format!("{}\n\n{}", result_text, success.diff);
            }
            with_outcome_and_referencing(
                CallToolResult::success(vec![Content::text(result_text)]),
                WriteOutcome::CommittedPendingSync,
                &success,
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
    // (fix #129) Same sha/rebased_paths/sync_failure_cause parity
    // `write_success_structured_fields` gives the single-document create/edit/delete
    // path — this tool surface is still `write_document`, just dispatched to a
    // directory move, so a caller reading only `structured_content` deserves the
    // same commit-identifying fields here too. No `diff`: `DirectoryMoveSuccess`
    // carries no unified diff (a directory move's text summary has none either —
    // see `moved_lines`/`moved_json` above for its equivalent).
    let mut structured = serde_json::Map::new();
    structured.insert("outcome".to_string(), serde_json::json!(outcome.as_str()));
    structured.insert("sha".to_string(), serde_json::json!(success.sha));
    structured.insert(
        "rebased_paths".to_string(),
        serde_json::json!(rebased_paths_json(&success.rebased_paths)),
    );
    structured.insert("moved".to_string(), serde_json::json!(moved_json));
    structured.insert(
        "rewritten_paths".to_string(),
        serde_json::json!(success.rewritten_paths),
    );
    if let Some(cause) = &success.sync_failure_cause {
        structured.insert("sync_failure_cause".to_string(), serde_json::json!(cause));
    }
    result.structured_content = Some(serde_json::Value::Object(structured));
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
    /// Per-tool description overlay, keyed by tool name, applied over the
    /// router's own `Tool` entries in `list_tools`/`get_tool` (see this
    /// module's hand-written `ServerHandler` impl). `#[tool_handler]`
    /// regenerates the router from `Self::tool_router()` on every call, so a
    /// description baked into a `#[tool(...)]` attribute can never be swapped
    /// at runtime — this overlay is what makes `descriptions::compose_all`
    /// (recomputed by the same periodic refresh that updates `instructions`
    /// above) actually reach `tools/list` and `tools/get` without a restart.
    /// `std::sync::RwLock`, not tokio's: `get_tool` is generated as a
    /// non-async fn by the trait, so it cannot `.await` a tokio lock.
    description_overlay: Arc<RwLock<HashMap<String, String>>>,
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
        description_overlay: Arc<RwLock<HashMap<String, String>>>,
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
            description_overlay,
        })
    }

    /// A fresh snapshot of the live config — a lock acquisition plus an `Arc`
    /// clone, mirroring `schema::load_shared`. Every tool call fetches its own
    /// snapshot here rather than reading a value captured at construction, so a
    /// `POST /admin/reload` swap is observed starting with the very next call.
    fn config(&self) -> Arc<ResolvedConfig> {
        crate::config::load_shared_config(&self.config)
    }

    /// Apply the live description overlay to one router-provided `Tool`,
    /// recovering from a poisoned lock the same way `get_info` does for
    /// `instructions`. A tool name absent from the overlay (should not
    /// happen — every tool in `tool_router()` is one of `descriptions::TOOL_NAMES`)
    /// is left with whatever the router itself produced, which is `None`
    /// since every `#[tool(...)]` attribute below carries no `description`.
    fn overlay_description(&self, mut tool: Tool) -> Tool {
        let overlay = self.description_overlay.read().unwrap_or_else(|poisoned| {
            warn!("Description overlay RwLock poisoned on read; using last value");
            poisoned.into_inner()
        });
        if let Some(desc) = overlay.get(tool.name.as_ref()) {
            tool.description = Some(std::borrow::Cow::Owned(desc.clone()));
        }
        tool
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

    #[tool]
    async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        validate_search_params(&params)?;

        let query_present = query_is_present(&params.query);
        let granularity = resolve_granularity(query_present, params.granularity.as_deref())?;

        debug!(
            query_present,
            granularity = ?granularity,
            has_filters = params.filters.is_some(),
            "search called"
        );

        match (query_present, granularity) {
            (false, Granularity::Chunk) => Err(McpError::invalid_params(
                "chunk granularity requires a query; omit granularity (or set it to \
                 'document') to enumerate without one"
                    .to_string(),
                None,
            )),
            (false, Granularity::Document) => self.search_enumerate(&params).await,
            (true, Granularity::Chunk) => self.search_chunks(&params).await,
            (true, Granularity::Document) => self.search_grouped(&params).await,
        }
    }

    /// query+chunk: the original `search` tool's behavior, unchanged output shape.
    async fn search_chunks(&self, params: &SearchParams) -> Result<CallToolResult, McpError> {
        let query = params.query.as_deref().unwrap_or_default();

        validate_path_prefix(&params.path_prefix)?;
        // #132 (audit follow-up): `fields` selects per-result frontmatter fields
        // from the document metadata index — a chunk result is hydrated straight
        // from the Qdrant chunk payload and never joins that index, so `fields`
        // has nothing to draw from here. `search_grouped`/`build_document_query`
        // (document granularity, query or enumeration) are its only real
        // consumers; reject explicitly here rather than silently drop it the way
        // it used to (this was the SAME silent-no-op bug `explain` at document
        // granularity is, just for the mirror-image parameter/granularity pair).
        if params.fields.is_some() {
            return Err(McpError::invalid_params(
                "fields is document-granularity only (fields draws from the document \
                 metadata index, which a chunk result never joins) — omit it, or set \
                 granularity to 'document'"
                    .to_string(),
                None,
            ));
        }
        let schemas = crate::schema::load_shared(&self.schema_cache);
        // Fetched once per call so every field below — including
        // reranking.candidate_limit — reflects the same live snapshot, rather than
        // racing a concurrent `POST /admin/reload` mid-request.
        let config = self.config();
        let conditions = build_query_conditions(params, &config, &schemas)?;

        let limit = resolve_limit(
            params.limit,
            config.search.default_limit,
            config.search.max_limit,
        );

        let filters = SearchFilters { conditions };

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
        let explain = params.explain.unwrap_or(false);
        let opts = SearchOptions {
            limit,
            min_score: params.min_score.or(config.search.min_score),
            hybrid: config.search.hybrid,
            rrf_candidates: config.search.rrf_candidates as u64,
            // Config-enabled AND confirmed available on this server right now —
            // see `status::IndexStatus::phrase_matching_available`'s doc comment
            // for why an unconfirmed index must not attempt a phrase arm.
            phrase: config.search.phrase && crate::status::INDEX_STATUS.phrase_matching_available(),
            explain,
            modified_after,
            modified_before,
            path_prefix: params.path_prefix.clone(),
            rerank_candidate_limit: config.reranking.as_ref().map(|r| r.candidate_limit as u64),
            diversity_max_per_document: config.search.diversity_max_per_document,
        };

        // The `explain` mode label must describe the query that actually ran, not
        // the config that permitted it. An enabled phrase arm only fires when the
        // query carries a quoted span, and with `hybrid` off a phrase query is
        // still a two-arm RRF fusion — reporting that as "dense cosine" would be a
        // lie in exactly the situation someone turned `explain` on to diagnose.
        let phrase_arm_ran = opts.phrase && !retrieval::extract_phrases(query).1.is_empty();

        // #224: `offset` pages over the already fused/reranked/diversity-capped
        // ranking — see `retrieval::search_paged`'s doc comment for the funnel
        // placement and the depth bound `offset_truncated` reports against.
        let offset = params.offset.unwrap_or(0);
        let outcome = retrieval::search_paged(&self.deps(), query, &filters, &opts, offset)
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
                retrieval::SearchError::Document(err) => {
                    error!("Document metadata lookup failed: {:#}", err);
                    McpError::internal_error("Document metadata lookup failed".to_string(), None)
                }
            })?;
        let path_prefix_truncated = outcome.path_prefix_truncated;
        let offset_truncated = outcome.offset_truncated;
        let results = outcome.results;

        debug!(
            result_count = results.len(),
            path_prefix_truncated, offset_truncated, "search returned results"
        );

        let mode = match (config.search.hybrid, phrase_arm_ran) {
            (true, true) => "hybrid RRF + phrase",
            (true, false) => "hybrid RRF",
            (false, true) => "dense + phrase RRF",
            (false, false) => "dense cosine",
        };

        let (text, structured) = build_chunk_search_payload(
            &results,
            &self.canonical_data_path,
            explain,
            mode,
            path_prefix_truncated,
            offset_truncated,
            offset,
            opts.rerank_candidate_limit,
        );

        let mut call_result = CallToolResult::success(vec![Content::text(text)]);
        call_result.structured_content = Some(structured);
        Ok(call_result)
    }

    /// query+document: Qdrant grouped by `file_path`, collapsed to each document's
    /// best-scoring chunk, hydrated with document-level metadata from `StateDb`. New
    /// combination — there is no output shape to preserve, only the two the design
    /// specifies: the document shape `search_enumerate` returns, plus a per-document
    /// `score`, and no `total`/`has_more` (grouped vector search cannot back either).
    async fn search_grouped(&self, params: &SearchParams) -> Result<CallToolResult, McpError> {
        let query = params.query.as_deref().unwrap_or_default();

        // #132: `explain` produces a per-result score breakdown (dense/sparse/
        // phrase/pre-rerank) from the per-arm scores `search_chunks` collects on
        // each chunk `SearchResult`. This path collapses every document to its
        // best-scoring chunk via Qdrant's server-side grouping (see this
        // function's doc comment) with no explain-mode fusion query behind it —
        // there is no per-arm breakdown to attach, so `GroupedDocument` carries
        // none. Implementing the real thing would mean threading an `explain`
        // flag through `RetrievalStore::search_grouped` and adding a
        // client-side-fused, per-arm-scored grouped-query path in `qdrant.rs`
        // mirroring `hybrid_search_explain` — real work, out of scope for this
        // fix. Reject explicitly instead of the silent no-op this used to be:
        // a caller that turned `explain` on to diagnose a confusing document
        // ranking deserves an error telling them so, not a response that quietly
        // drops the one thing they asked for.
        if params.explain == Some(true) {
            return Err(McpError::invalid_params(
                "explain is chunk-granularity only; document-granularity results \
                 collapse to one row per document with no per-arm score breakdown \
                 available to report — omit it, or set granularity to 'chunk'"
                    .to_string(),
                None,
            ));
        }

        validate_path_prefix(&params.path_prefix)?;
        validate_fields_count(&params.fields)?;
        let schemas = crate::schema::load_shared(&self.schema_cache);
        let config = self.config();
        let conditions = build_query_conditions(params, &config, &schemas)?;

        let limit = resolve_limit(
            params.limit,
            config.search.default_limit,
            config.search.max_limit,
        );

        let filters = SearchFilters { conditions };

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

        let opts = SearchOptions {
            limit,
            min_score: params.min_score.or(config.search.min_score),
            // Grouped queries now share `search_chunks`'s dense/sparse/phrase arms
            // (see `retrieval::search_grouped`'s doc comment) — a query only the
            // sparse or phrase arm can find must be just as retrievable at
            // document granularity as at chunk granularity. Reranking and
            // diversity still don't apply here: there is one row per document
            // already, so per-document diversity is a no-op and there is no
            // per-chunk candidate pool for a cross-encoder to rerank.
            hybrid: config.search.hybrid,
            rrf_candidates: config.search.rrf_candidates as u64,
            phrase: config.search.phrase && crate::status::INDEX_STATUS.phrase_matching_available(),
            // Always false: `explain: true` is rejected above (#132) before this
            // point is reached, and `retrieval::search_grouped` never reads this
            // field regardless — grouped results carry no per-arm score to
            // explain in the first place. `false` here (rather than
            // `params.explain.unwrap_or(false)`) says so plainly instead of
            // leaving a dead-looking reference to a value that can only be
            // `None`/`Some(false)` by now.
            explain: false,
            modified_after,
            modified_before,
            path_prefix: params.path_prefix.clone(),
            rerank_candidate_limit: None,
            diversity_max_per_document: None,
        };

        let index = self.state_db().await.map_err(|e| {
            error!(
                "search (query+document) could not open the metadata index: {:#}",
                e
            );
            McpError::internal_error(format!("Document index unavailable: {}", e), None)
        })?;

        // #224: same paging entry point and depth-bound reasoning as
        // `search_chunks` — see `retrieval::search_grouped`'s doc comment.
        let offset = params.offset.unwrap_or(0);
        let outcome = retrieval::search_grouped(
            &self.deps(),
            index,
            query,
            &filters,
            &opts,
            params.fields.as_deref(),
            offset,
        )
        .await
        .map_err(|e| match e {
            retrieval::SearchError::Embed(err) => {
                error!("Embedding query failed: {:#}", err);
                McpError::internal_error("Failed to generate query embedding".to_string(), None)
            }
            retrieval::SearchError::Search(err) => {
                error!("Qdrant grouped search failed: {:#}", err);
                McpError::internal_error("Search query failed".to_string(), None)
            }
            retrieval::SearchError::Document(err) => {
                error!("Document metadata lookup failed: {:#}", err);
                McpError::internal_error("Document metadata lookup failed".to_string(), None)
            }
        })?;
        let path_prefix_truncated = outcome.path_prefix_truncated;
        let offset_truncated = outcome.offset_truncated;
        let documents = outcome.documents;

        let (text, structured) = build_grouped_search_payload(
            &documents,
            path_prefix_truncated,
            offset_truncated,
            offset,
        );

        let mut call_result = CallToolResult::success(vec![Content::text(text)]);
        call_result.structured_content = Some(structured);
        Ok(call_result)
    }

    /// document, no query: the former `list_documents` tool's behavior, unchanged
    /// output shape (`{total, returned, offset, has_more, documents}`).
    async fn search_enumerate(&self, params: &SearchParams) -> Result<CallToolResult, McpError> {
        let query = build_document_query(params)?;

        let index = self.state_db().await.map_err(|e| {
            error!(
                "search (enumerate) could not open the metadata index: {:#}",
                e
            );
            McpError::internal_error(format!("Document index unavailable: {}", e), None)
        })?;

        let result = retrieval::list_documents(&DocumentIndexDeps { index }, &query)
            .await
            .map_err(|e| {
                error!("search (enumerate) failed: {:#}", e);
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

    #[tool]
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

    #[tool]
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
            let (would_invalidate, casualties_total, casualties_truncated) =
                capped_casualties(&casualties);
            return Err(McpError::invalid_params(
                format!(
                    "Refusing to apply: {} existing document(s) would fail the new rules. \
                     Fix them first, or pass force to apply anyway.\n{}",
                    casualties.len(),
                    render_casualties(&casualties)
                ),
                Some(serde_json::json!({
                    "would_invalidate": would_invalidate,
                    "casualties_total": casualties_total,
                    "casualties_truncated": casualties_truncated,
                })),
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
            let (would_invalidate, casualties_total, casualties_truncated) =
                capped_casualties(&casualties);
            let mut result = CallToolResult::success(vec![Content::text(text)]);
            result.structured_content = Some(serde_json::json!({
                "dry_run": true,
                "summary": summary,
                "would_invalidate": would_invalidate,
                "casualties_total": casualties_total,
                "casualties_truncated": casualties_truncated,
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
        // immediately calls `write_document` relying on that new rule
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

        let (invalidated, casualties_total, casualties_truncated) = capped_casualties(&casualties);
        let mut result = CallToolResult::success(vec![Content::text(text)]);
        result.structured_content = Some(serde_json::json!({
            "outcome": outcome.as_str(),
            "dry_run": false,
            "summary": summary,
            "path": rel_file_str,
            "invalidated": invalidated,
            "casualties_total": casualties_total,
            "casualties_truncated": casualties_truncated,
        }));
        Ok(result)
    }

    #[tool]
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
                // content, so a caller can round-trip it straight into write_document's
                // expected_hash without this project introducing a second hash scheme.
                // Hence hashing here, before any slicing, and always over the whole
                // file: that expected_hash guards the document on disk, so hashing a
                // slice would hand back a token that can never match — turning every
                // partial read into a dead end for the edit that motivated it.
                let content_hash = crate::ingest::compute_hash_from_bytes(doc.content.as_bytes());
                // Snapshot the link-graph neighborhood before `doc.content` is moved
                // into `slice_or_whole` below — `doc.links_out`/`doc.links_in` are
                // untouched by that move (distinct fields), so this ordering is not
                // load-bearing, just where it reads most naturally alongside the hash.
                //
                // `has_more` mirrors `search`'s truncation contract: a hub document
                // can have far more inbound links than `retrieval::MAX_LINKS_PER_DIRECTION`
                // allows through, and silently dropping the tail would misrepresent
                // the graph rather than just page it. `score` is `null` for every
                // `markdown` edge (only `semantic` neighbors carry one) and `exists`
                // appears only on `links_out`: an inbound edge's source cannot dangle
                // by construction (`delete_document` removes a deleted file's own
                // outgoing rows), but an outbound edge's target can point at a file
                // that was never indexed or was renamed out from under it — see
                // `OutboundLink`'s doc comment. Both kinds (`markdown`, author-written;
                // `semantic`, a machine-inferred kNN neighbor, only populated when
                // `ui.semantic_edges.enabled` is on) ride the same list, tagged by
                // `kind`, rather than being split into separate fields — mirroring how
                // `/api/graph` already exposes them, just without that endpoint's
                // dangling-edge drop.
                let links_out = serde_json::json!({
                    "total": doc.links_out.total,
                    "has_more": doc.links_out.has_more(),
                    "links": doc.links_out.links.iter().map(|l| serde_json::json!({
                        "target_path": l.target_path,
                        "kind": l.kind,
                        "score": l.score,
                        "exists": l.exists,
                    })).collect::<Vec<_>>(),
                });
                let links_in = serde_json::json!({
                    "total": doc.links_in.total,
                    "has_more": doc.links_in.has_more(),
                    "links": doc.links_in.links.iter().map(|l| serde_json::json!({
                        "source_path": l.source_path,
                        "kind": l.kind,
                        "score": l.score,
                    })).collect::<Vec<_>>(),
                });
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
                    "links_out": links_out,
                    "links_in": links_in,
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

    /// Shared pipeline for `write_document`'s create and edit paths.
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
    /// * `operation`   – label for the `Operation:` git trailer, e.g. `"write_document"`.
    /// * `dest_path`   – when `Some`, turns this into a document MOVE: `rel_path` is
    ///   the source, this is the destination. `None` (the default for a plain
    ///   create, and for an edit call with no `new_path`) is the existing
    ///   create/edit behavior. See `write::WriteRequest::dest_path`.
    /// * `expected_hash` – the caller's `content_hash` from a prior read, threaded
    ///   straight into `WriteRequest::expected_hash`. See that field's doc comment
    ///   for why this is not redundant with `write_document_edit`'s own up-front
    ///   check: this one is `write::write_document`'s live-disk re-check,
    ///   immediately before the overwrite under `GIT_LOCK`, which catches a
    ///   modification that landed during this call's own awaits (e.g. a slow
    ///   `validation.lint_command`) — a window the up-front check cannot see.
    #[allow(clippy::too_many_arguments)]
    async fn run_document_write(
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
        expected_hash: Option<&str>,
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
            // Threaded straight from this method's own `expected_hash` parameter.
            // `write_document_edit`'s up-front check (above) and this one check
            // different things: the up-front check compares against the
            // in-memory `old_content` this call already read, catching a stale
            // caller read; this one is `write::write_document`'s live-disk
            // re-check immediately before the overwrite under `GIT_LOCK`, which
            // catches a concurrent modification that lands *during* this call —
            // e.g. while awaiting `validate::validate_content`, which can exec an
            // arbitrarily slow `validation.lint_command`. Passing `None` here
            // would leave that second window unguarded, silently dropping #142's
            // protection for every MCP write (see #243).
            expected_hash,
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

    /// `write_document`'s directory-move dispatch: `source_dir` resolved to an
    /// existing directory, so this relocates the whole subtree via
    /// `write::move_directory` instead of writing a single document. Mirrors the
    /// old standalone `move_directory` tool.
    ///
    /// Every field that only makes sense for a single document (`content`,
    /// `old_string`/`new_string`, `expected_hash`, `force_new`) is rejected here —
    /// a directory move has no body to replace and no dedup gate to bypass.
    /// `new_path` is required: it is the destination prefix.
    async fn write_document_move_dir(
        &self,
        params: &WriteDocumentParams,
        source_dir: &str,
    ) -> Result<CallToolResult, McpError> {
        let dest_dir = match params.new_path.as_deref().map(str::trim) {
            Some(p) if !p.is_empty() => p,
            _ => {
                return Err(McpError::invalid_params(
                    "new_path is required to move the directory at path".to_string(),
                    None,
                ));
            }
        };

        for (field, is_set) in [
            ("content", params.content.is_some()),
            ("old_string", params.old_string.is_some()),
            ("new_string", params.new_string.is_some()),
            ("frontmatter_patch", params.frontmatter_patch.is_some()),
            ("append", params.append.is_some()),
            ("expected_hash", params.expected_hash.is_some()),
            ("force_new", params.force_new.is_some()),
        ] {
            if is_set {
                return Err(McpError::invalid_params(
                    format!(
                        "{} is not valid when path is a directory: a directory move \
                         only takes new_path (and optionally message)",
                        field
                    ),
                    None,
                ));
            }
        }

        let config = self.config();
        let token = std::env::var(&config.source.git_token_env)
            .ok()
            .filter(|s| !s.is_empty());

        // Best-effort, same as `run_document_write`'s own lazy state-DB open for a
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

    /// `write_document`'s create path: `path` did not resolve to an existing,
    /// permitted file. Mirrors the old standalone `create_document` tool.
    async fn write_document_create(
        &self,
        params: WriteDocumentParams,
        raw: &str,
    ) -> Result<CallToolResult, McpError> {
        if params.old_string.is_some() || params.new_string.is_some() {
            return Err(McpError::invalid_params(
                "cannot surgically edit a document that does not exist".to_string(),
                None,
            ));
        }
        if params.frontmatter_patch.is_some() {
            return Err(McpError::invalid_params(
                "cannot patch frontmatter on a document that does not exist — use content to \
                 create it"
                    .to_string(),
                None,
            ));
        }
        if params.append.is_some() {
            return Err(McpError::invalid_params(
                "cannot append to a document that does not exist — use content to create it"
                    .to_string(),
                None,
            ));
        }
        if params.new_path.is_some() {
            return Err(McpError::invalid_params(
                "new_path is not valid when creating a new document — there is \
                 nothing at path yet to move; create it directly at the final path"
                    .to_string(),
                None,
            ));
        }
        let Some(content) = params.content.as_deref() else {
            return Err(McpError::invalid_params(
                "content is required to create a new document".to_string(),
                None,
            ));
        };

        // Resolve path: must be relative, no traversal, not already existing.
        let data_root = self.canonical_data_path.clone();
        let abs_path = crate::write::resolve_safe_write_path(&data_root, raw)
            .map_err(|e| McpError::invalid_params(e, None))?;

        // The include-pattern eligibility guard is enforced inside
        // `write::write_document` itself too — see that module's
        // `check_include_pattern` — so every caller of the shared write pipeline
        // (this tool, the HTTP UI in `web.rs`) gets it for free instead of each
        // transport maintaining its own copy that a future caller could forget. It
        // is ALSO run here, explicitly, ahead of the `exists()` pre-check just
        // below: a path that both exists on disk and fails this check must be
        // reported with the include-pattern message, not "already exists" — which,
        // for such a path, is misleading circular guidance, since a retry would
        // reject the same path as not permitted right back. Running
        // `write::write_document`'s check later is not sufficient to restore that
        // priority on its own, since the `exists()` check below returns before
        // ever reaching it. Same message text as `write::write_document`'s own
        // check (see `create_edit_error_to_mcp_error`'s `UnsafePath` arm), so
        // existing callers see the exact same wording.
        crate::write::check_include_pattern_against(&self.include_patterns, raw).map_err(|e| {
            create_edit_error_to_mcp_error(e, raw, true, &self.canonical_data_path, None)
        })?;

        // File must not already exist for create.
        if abs_path.exists() {
            return Err(McpError::invalid_params(
                format!(
                    "File '{}' already exists. Call write_document again without \
                     old_string/new_string to edit it in place.",
                    raw
                ),
                None,
            ));
        }

        self.run_document_write(
            "", // old_content: empty for new files
            content,
            raw,
            true, // is_create
            params.message.as_deref(),
            "add",
            params.force_new,
            "write_document",
            None, // a create never moves a document
            params.expected_hash.as_deref(),
        )
        .await
    }

    /// `write_document`'s edit path: `path` resolved to an existing, permitted
    /// file. Mirrors the old standalone `edit_document` tool.
    async fn write_document_edit(
        &self,
        params: WriteDocumentParams,
        canonical: PathBuf,
    ) -> Result<CallToolResult, McpError> {
        // Parse and validate the content-edit mode (surgical vs full-replace vs
        // neither). `new_path` is an orthogonal axis handled below, independent of
        // this — see `parse_edit_mode`'s doc comment.
        let mode = parse_edit_mode(&params).map_err(|e| McpError::invalid_params(e, None))?;
        let dest_path = params.new_path.as_deref();

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
        // `indexed_files.content_hash` already uses — see `WriteDocumentParams::expected_hash`.
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
            None => (old_content.clone(), "write_document (move)"),
            Some(EditMode::Full { content }) => (
                content,
                if dest_path.is_some() {
                    "write_document (full replace + move)"
                } else {
                    "write_document (full replace)"
                },
            ),
            Some(EditMode::Surgical { old, new }) => {
                let result = apply_surgical(&old_content, &old, &new, &rel_path)
                    .map_err(|e| McpError::invalid_params(e, None))?;
                (
                    result,
                    if dest_path.is_some() {
                        "write_document (surgical replace + move)"
                    } else {
                        "write_document (surgical replace)"
                    },
                )
            }
            Some(EditMode::Patch { edits }) => {
                let result = write::apply_frontmatter_patch(&old_content, &edits)
                    .map_err(|e| McpError::invalid_params(e, None))?;
                (
                    result,
                    if dest_path.is_some() {
                        "write_document (frontmatter patch + move)"
                    } else {
                        "write_document (frontmatter patch)"
                    },
                )
            }
            Some(EditMode::Append { text }) => {
                let result = write::apply_append(&old_content, &text);
                (
                    result,
                    if dest_path.is_some() {
                        "write_document (append + move)"
                    } else {
                        "write_document (append)"
                    },
                )
            }
            Some(EditMode::PatchAppend { edits, text }) => {
                // Patch first, then append — the patch only ever touches the
                // frontmatter block, so applying it first and re-deriving the
                // frontmatter/body split for the append (see `apply_append`'s
                // own doc comment) composes cleanly with no ordering ambiguity.
                let patched = write::apply_frontmatter_patch(&old_content, &edits)
                    .map_err(|e| McpError::invalid_params(e, None))?;
                let result = write::apply_append(&patched, &text);
                (
                    result,
                    if dest_path.is_some() {
                        "write_document (frontmatter patch + append + move)"
                    } else {
                        "write_document (frontmatter patch + append)"
                    },
                )
            }
        };

        self.run_document_write(
            &old_content,
            &new_content,
            &rel_path,
            false, // is_create
            params.message.as_deref(),
            "update",
            None, // no dedup gate for edit
            operation,
            dest_path,
            params.expected_hash.as_deref(),
        )
        .await
    }

    #[tool]
    async fn write_document(
        &self,
        Parameters(params): Parameters<WriteDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        let raw = params.path.trim().to_string();
        if raw.is_empty() {
            return Err(McpError::invalid_params(
                "path parameter is empty".to_string(),
                None,
            ));
        }
        // Mirrors `get_document`'s length guard (see its comment): rejecting an
        // oversized path here avoids paying for `resolve_safe_write_path`, an
        // include-pattern check, and git staging on an input that was never going
        // to resolve to anything.
        if raw.len() > MAX_PATH_LEN {
            return Err(McpError::invalid_params(
                format!("path exceeds maximum length of {MAX_PATH_LEN} characters"),
                None,
            ));
        }
        if let Some(new_path) = params.new_path.as_deref()
            && new_path.len() > MAX_PATH_LEN
        {
            return Err(McpError::invalid_params(
                format!("new_path exceeds maximum length of {MAX_PATH_LEN} characters"),
                None,
            ));
        }

        // Branch 1: a directory move. Detected via the literal (non-fuzzy)
        // resolver — a directory can never satisfy `resolve_within_data`'s
        // include-pattern check, so that resolver cannot be used for this test.
        if let Ok(abs) = crate::write::resolve_safe_write_path(&self.canonical_data_path, &raw)
            && abs.is_dir()
        {
            return self.write_document_move_dir(&params, &raw).await;
        }

        // Branch 2: a document create or edit. `resolve_within_data` (the same
        // resolver `get_document` uses to try a literal path before its own
        // basename fallback, which lives only there, not here) tells them apart:
        // a path that resolves to an existing, permitted file is an edit;
        // `NotFound` (nothing there yet) or `NotPermitted` (something exists
        // there, but of a non-indexable type) falls through to create, which
        // re-validates the path itself via `resolve_safe_write_path` and
        // `check_include_pattern_against` — independent, literal, KB-root-relative
        // checks that do not share `resolve_within_data`'s "try the raw absolute
        // path against the real filesystem first" ambiguity. That re-validation is
        // what makes it safe to fall through for those two cases rather than treat
        // every non-success as a hard failure, and it is also what restores the
        // pre-merge priority: a path that both exists on disk and fails the
        // include-pattern check is reported with that specific message, not a
        // generic "not found" one.
        //
        // `Outside` and `Other` are different: `resolve_within_data` only returns
        // `Outside` when the literal absolute path actually EXISTS outside the
        // data root (see its doc comment) — falling through to create would strip
        // the leading `/` and join it KB-relative, silently writing a *new* file
        // inside the KB at a path the caller never asked for while reporting
        // success. `Other` (a real I/O error resolving the path) has no safe
        // fallback interpretation either. Both are hard errors here, same as
        // `delete_document`'s handling of the same resolver just above.
        match retrieval::resolve_within_data(
            &raw,
            &self.canonical_data_path,
            &self.include_patterns,
        ) {
            Ok(canonical) => self.write_document_edit(params, canonical).await,
            Err(retrieval::ResolveErr::NotFound) | Err(retrieval::ResolveErr::NotPermitted) => {
                self.write_document_create(params, &raw).await
            }
            Err(retrieval::ResolveErr::Outside) => Err(McpError::invalid_params(
                "File path is outside the data directory".to_string(),
                None,
            )),
            Err(retrieval::ResolveErr::Other(msg)) => Err(McpError::invalid_params(msg, None)),
        }
    }

    #[tool]
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
        // Mirrors `get_document`'s length guard (see its comment) and
        // `write_document`'s identical check: reject before the fuzzy resolver, the
        // metadata index, or git ever see an input that was never going to resolve.
        if raw.len() > MAX_PATH_LEN {
            return Err(McpError::invalid_params(
                format!("path exceeds maximum length of {MAX_PATH_LEN} characters"),
                None,
            ));
        }

        // Resolve the path (must already exist on disk). This is the same fuzzy
        // resolver `get_document`/`write_document` use — relative to the KB root, a
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

        // (#229) Best-effort, same as `run_document_write`'s and
        // `write_document_move_dir`'s own lazy state-DB opens: a state DB that
        // fails to open degrades `delete_document`'s inbound-link check to
        // "skip it" (see `WriteDeps::state`'s doc comment), not a failed
        // delete. Without this, `write::delete_document`'s reverse-link query
        // never runs at all — `WriteDeps::state == None` — and
        // `referencing_paths` would always come back empty regardless of what
        // actually links to the document being deleted.
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

        match crate::write::delete_document(&deps, &rel_path, params.message.as_deref()).await {
            Ok(success) => Ok(delete_success_to_result(success, &rel_path)),
            Err(err) => Err(delete_error_to_mcp_error(err, &rel_path)),
        }
    }
}

/// Builds `search_chunks`' text and structured payload from already-fetched
/// results. Pulled out of the tool handler as its own pure function so the
/// "results -> CallToolResult content" seam — the exact spot where
/// `structured_content` once regressed to a bare `{"path_prefix_truncated": ...}"`
/// while the text content still carried full results — is reachable by a plain
/// unit test: no network, no mocked `KbSearchServer`, none of `EmbedClient`'s
/// retry/backoff to defeat.
///
/// `mode` is the caller-computed `explain` label (see `search_chunks`'s
/// `phrase_arm_ran` derivation) — constant across the whole result set, never
/// per-result. The empty-results branch lives here too, so one seam covers both.
///
/// `offset` and `rerank_candidate_limit` (#240) exist purely to hand to
/// [`offset_truncated_note`] when `offset_truncated` is set — see that
/// function's doc comment for why the note needs both to give accurate advice.
#[allow(clippy::too_many_arguments)]
fn build_chunk_search_payload(
    results: &[crate::qdrant::SearchResult],
    data_root: &Path,
    explain: bool,
    mode: &str,
    path_prefix_truncated: bool,
    offset_truncated: bool,
    offset: u64,
    rerank_candidate_limit: Option<u64>,
) -> (String, serde_json::Value) {
    if results.is_empty() {
        let mut text = "No results found.".to_string();
        if path_prefix_truncated {
            text.push_str(
                "\n\nNote: path_prefix matched more candidates than could be over-fetched, \
                 so this may not be exhaustive — narrow the prefix or lower limit to be sure.",
            );
        }
        if offset_truncated {
            text.push_str(&offset_truncated_note(offset, rerank_candidate_limit));
        }
        let structured = serde_json::json!({
            "returned": 0,
            "results": [],
            "path_prefix_truncated": path_prefix_truncated,
            "offset_truncated": offset_truncated,
        });
        return (text, structured);
    }

    // Format results as text content, and mirror them into `structured_content`.
    //
    // Both, not either: `search` historically returned text only, so a client
    // that prefers structured content had nothing to render but the truncation
    // flag — which reads as "no results" even when the text body is full. The
    // other two granularities (`search_enumerate`, `search_grouped`) already
    // return a full structured payload, so chunk mode returning a bare flag was
    // the odd one out.
    let mut output = String::new();
    let mut structured_results: Vec<serde_json::Value> = Vec::with_capacity(results.len());
    for (i, result) in results.iter().enumerate() {
        let title = result
            .payload
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("(untitled)");

        let (text_snippet, needs_ellipsis) = {
            let full_text = result
                .payload
                .get(CHUNK_TEXT_KEY)
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
        let file_path = retrieval::relative_to_data(file_path_raw, data_root);

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
            // Presence, not magnitude, is the signal: the phrase arm's "score" is
            // just the dense-ranked query re-run under a phrase filter, so its
            // value carries no information beyond dense=. What a caller actually
            // wants to know is whether this result matched every requested
            // phrase at all.
            if result.phrase_score.is_some() {
                breakdown.push_str(", phrase=matched");
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

        let null = serde_json::Value::Null;
        structured_results.push(serde_json::json!({
            "file_path": file_path,
            "title": title,
            "score": result.score,
            "text": text_snippet,
            "text_truncated": needs_ellipsis,
            "domain": domain,
            "type": doc_type,
            "tags": result.payload.get("tags").cloned().unwrap_or(null.clone()),
            "line_start": result.payload.get("line_start").cloned().unwrap_or(null.clone()),
            "line_end": result.payload.get("line_end").cloned().unwrap_or(null),
            "dense_score": result.dense_score,
            "sparse_score": result.sparse_score,
            "pre_rerank_score": result.pre_rerank_score,
            "phrase_matched": result.phrase_score.is_some(),
        }));

        output.push('\n');
    }

    if path_prefix_truncated {
        output.push_str(
            "\nNote: path_prefix matched more candidates than could be over-fetched, so \
             fewer results than `limit` were returned and more may exist — narrow the \
             prefix or lower limit to be sure this is exhaustive.\n",
        );
    }
    if offset_truncated {
        output.push_str(&offset_truncated_note(offset, rerank_candidate_limit));
    }

    let structured = serde_json::json!({
        "returned": structured_results.len(),
        "results": structured_results,
        "path_prefix_truncated": path_prefix_truncated,
        "offset_truncated": offset_truncated,
    });

    (output.trim().to_string(), structured)
}

/// Shared prose for both `build_chunk_search_payload` and
/// `build_grouped_search_payload`'s `offset_truncated` note (#224 / #240).
///
/// The bound this note explains can be tripped two different ways, and they
/// need different advice:
///
/// - `offset == 0`: `limit` alone already exceeds the ranked-candidate depth
///   bound — the whole first page is inside the untouched region, so there is
///   no offset to lower (it's already zero). The actionable fix is raising the
///   bound (`reranking.candidate_limit`, when a reranker sized it — passed as
///   `rerank_candidate_limit`) or lowering `limit`. Telling a caller with
///   `offset: 0` to "lower offset" is nonsensical advice pointing at a knob
///   they never touched — that was #240.
/// - `offset > 0`: the usual paging-too-deep case the original #224 note
///   described — lowering `offset` (or narrowing the query so fewer pages are
///   needed) is the real fix here.
fn offset_truncated_note(offset: u64, rerank_candidate_limit: Option<u64>) -> String {
    if offset == 0 {
        match rerank_candidate_limit {
            Some(limit) => format!(
                "\nNote: limit alone already reached past this query's ranked-candidate depth \
                 bound — reranking.candidate_limit is currently {limit}. This page may be short \
                 or empty not because there are no more matches, but because paging that deep \
                 was never attempted. Raise reranking.candidate_limit or lower limit to be sure \
                 this is exhaustive.\n"
            ),
            None => "\nNote: limit alone already reached past this query's ranked-candidate \
                 depth bound (a fixed ceiling; no reranker is configured to size a larger one) \
                 — this page may be short or empty not because there are no more matches, but \
                 because paging that deep was never attempted. Lower limit to be sure this is \
                 exhaustive.\n"
                .to_string(),
        }
    } else {
        "\nNote: offset + limit reached past this query's ranked-candidate depth bound \
         (reranking.candidate_limit when reranking is active, otherwise a fixed ceiling) — \
         this page may be short or empty not because there are no more matches, but because \
         paging that deep was never attempted. Narrow the query or lower offset to be sure \
         this is exhaustive.\n"
            .to_string()
    }
}

/// Builds `search_grouped`'s text and structured payload from already-fetched
/// grouped documents. Same untested-adapter shape and risk as
/// `build_chunk_search_payload` (wrong JSON key, an accidentally-included
/// `total`/`has_more` — grouped must claim neither — or a dropped `score`) —
/// pulled out so a hand-built `Vec<GroupedDocument>` can drive it directly, with
/// no network involved.
fn build_grouped_search_payload(
    documents: &[retrieval::GroupedDocument],
    path_prefix_truncated: bool,
    offset_truncated: bool,
    offset: u64,
) -> (String, serde_json::Value) {
    let returned = documents.len();

    let structured = serde_json::json!({
        "returned": returned,
        "path_prefix_truncated": path_prefix_truncated,
        "offset_truncated": offset_truncated,
        "documents": documents
            .iter()
            .map(|d| serde_json::json!({
                "file_path": d.summary.file_path,
                "title": d.summary.title,
                "description": d.summary.description,
                "mtime": d.summary.mtime,
                "frontmatter": d.summary.frontmatter,
                "score": d.score,
            }))
            .collect::<Vec<_>>(),
    });

    let mut text = if returned == 0 {
        "No documents matched.".to_string()
    } else {
        format!("{returned} document(s) matched, ranked by relevance.\n\n")
    };

    for doc in documents {
        text.push_str(&format!(
            "- {} (score {:.4})",
            doc.summary.file_path, doc.score
        ));
        if let Some(title) = &doc.summary.title {
            text.push_str(&format!(" — {}", title));
        }
        text.push('\n');
        if let Some(description) = &doc.summary.description {
            text.push_str(&format!("  {}\n", description.trim()));
        }
    }

    if path_prefix_truncated {
        text.push_str(
            "\nNote: path_prefix matched more candidates than could be over-fetched, so \
             fewer results than `limit` were returned and more may exist — narrow the \
             prefix or lower limit to be sure this is exhaustive.\n",
        );
    }
    if offset_truncated {
        // Grouped search never runs a reranker (see `search_grouped`'s doc
        // comment / `retrieval::GroupedSearchOutcome::offset_truncated`), so
        // there is no `reranking.candidate_limit` to name here — the bound is
        // always the fixed absolute ceiling.
        text.push_str(&offset_truncated_note(offset, None));
    }

    (text.trim_end().to_string(), structured)
}

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

    // Hand-written rather than left for `#[tool_handler]` to generate: the
    // macro regenerates the tool list from `Self::tool_router()` (and its
    // compile-time `#[tool(...)]` attributes) on every call, so a description
    // that needs to change at runtime — per `descriptions.rs`'s whole point —
    // cannot be baked into that attribute. These two methods apply this
    // server's live `description_overlay` on top of the router's own `Tool`
    // entries; `call_tool` is left for the macro to generate unchanged, since
    // dispatch itself does not depend on the description text.
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let tools = Self::tool_router()
            .list_all()
            .into_iter()
            .map(|tool| self.overlay_description(tool))
            .collect();
        Ok(ListToolsResult::with_all_items(tools))
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        Self::tool_router()
            .get(name)
            .cloned()
            .map(|tool| self.overlay_description(tool))
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

/// An empty description overlay, for tests that exercise a `KbSearchServer` but
/// do not care about tool descriptions — `list_tools`/`get_tool` fall back to
/// whatever the router itself produced (`None`, since no `#[tool(...)]`
/// attribute carries one) when a tool has no overlay entry.
#[cfg(test)]
pub(crate) fn empty_test_description_overlay() -> Arc<RwLock<HashMap<String, String>>> {
    Arc::new(RwLock::new(HashMap::new()))
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

    fn filters_from(json: serde_json::Value) -> SearchParams {
        SearchParams {
            filters: Some(SearchFiltersInput(json.as_object().unwrap().clone())),
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
        let query = build_document_query(&SearchParams::default()).unwrap();
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
        let query = build_document_query(&SearchParams {
            limit: Some(999_999),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(query.limit, MAX_LIST_LIMIT);

        let query = build_document_query(&SearchParams {
            limit: Some(0),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(query.limit, 1, "a zero-size page would return nothing");
    }

    #[test]
    fn invalid_order_by_is_rejected() {
        let err = build_document_query(&SearchParams {
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
        let err = build_document_query(&SearchParams {
            filters: Some(SearchFiltersInput(map)),
            ..Default::default()
        })
        .unwrap_err();
        assert!(format!("{:?}", err).contains("too many filters"));
    }

    #[test]
    fn search_filters_advertises_as_a_typed_object() {
        // Regression test for #151: `filters` used to be typed
        // `Option<serde_json::Map<String, serde_json::Value>>`, which schemars turns
        // into an unconstrained `{"type": "object"}` with no `additionalProperties`
        // constraint at all — no signal to a calling client about what a field's
        // condition may look like, and (per `SearchFiltersInput`'s doc comment) the
        // same under-specification that drove at least one real client to send a
        // nested-object parameter JSON-encoded as a string instead of an object.
        let schema = schemars::schema_for!(SearchParams);
        let root = schema.as_value();

        let filters_schema = &root["properties"]["filters"];
        // `Option<SearchFiltersInput>` becomes `anyOf: [<real schema>, {"type":
        // "null"}]`; find the non-null branch.
        let object_schema = filters_schema["anyOf"]
            .as_array()
            .expect("filters must offer a typed alternative, not a bare {}")
            .iter()
            .find(|branch| branch["type"] != serde_json::json!("null"))
            .expect("filters must have a non-null branch");

        // schemars refs a named type's schema into `$defs` rather than inlining it;
        // resolve it so the assertions below see the real shape (mirrors
        // `update_schema_definition_advertises_as_a_typed_object`'s resolution).
        let resolved = match object_schema["$ref"].as_str() {
            Some(reference) => &root["$defs"][reference.rsplit('/').next().unwrap()],
            None => object_schema,
        };

        assert_eq!(
            resolved["type"],
            serde_json::json!("object"),
            "filters must advertise as an object, got: {resolved}"
        );
        let condition_schema = &resolved["additionalProperties"];
        assert_ne!(
            *condition_schema,
            serde_json::json!(true),
            "a bare `additionalProperties: true` (schemars' rendering of an \
             unconstrained serde_json::Map) tells a client nothing about a \
             condition's shape — this is the exact bug being fixed, got: {resolved}"
        );

        let branches = condition_schema["anyOf"]
            .as_array()
            .expect("a filter condition must advertise its scalar/array/object forms");

        // Scalar branch: equality against a string, number, or boolean.
        assert!(
            branches
                .iter()
                .any(|b| b["type"].as_array().is_some_and(|types| {
                    let types: Vec<&str> = types.iter().filter_map(|t| t.as_str()).collect();
                    types.contains(&"string")
                        && types.contains(&"number")
                        && types.contains(&"boolean")
                })),
            "missing the scalar-equality branch, got: {condition_schema}"
        );

        // Array branch: any-of against a list of scalars.
        assert!(
            branches
                .iter()
                .any(|b| b["type"] == serde_json::json!("array") && !b["items"].is_null()),
            "missing the any-of-array branch, got: {condition_schema}"
        );

        // Object branch: named any_of/all_of/gte/lte/gt/lt properties.
        let object_branch = branches
            .iter()
            .find(|b| b["type"] == serde_json::json!("object"))
            .expect("missing the any_of/all_of/range object branch");
        for key in ["any_of", "all_of", "gte", "lte", "gt", "lt"] {
            assert!(
                !object_branch["properties"][key].is_null(),
                "condition object schema is missing documented key '{key}': \
                 {object_branch}"
            );
        }
    }

    #[test]
    fn search_filters_accepts_a_json_encoded_string_as_a_fallback() {
        // At least one real MCP client sends nested-object tool arguments as a
        // JSON-encoded string rather than an object, regardless of what the tool
        // schema advertises (same failure mode `FieldDefinitionInput` exists to
        // cover — see #151). `SearchFiltersInput` must tolerate that as a fallback,
        // and the parsed result must behave identically to the equivalent object.
        let params: SearchParams = serde_json::from_value(serde_json::json!({
            "filters": r#"{"type":"guide"}"#,
        }))
        .expect("a JSON-encoded filters string must deserialize");
        let query = build_document_query(&params).unwrap();
        assert_eq!(
            query.filters,
            vec![("type".to_string(), FieldFilter::AnyOf(vec!["guide".into()]))]
        );
    }

    #[test]
    fn search_filters_rejects_a_string_that_is_not_valid_json() {
        let err = serde_json::from_value::<SearchFiltersInput>(serde_json::Value::String(
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
    fn search_filters_rejects_a_json_array_naming_the_expected_shape() {
        let err = serde_json::from_value::<SearchFiltersInput>(serde_json::json!(["type", "x"]))
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("JSON object"),
            "expected the error to name the expected shape, got: {msg}"
        );
        assert!(
            !msg.contains("SearchFiltersInput"),
            "a Rust type name is meaningless to an MCP client, got: {msg}"
        );
    }

    #[test]
    fn overlong_path_prefix_is_rejected() {
        let err = build_document_query(&SearchParams {
            path_prefix: Some("a".repeat(MAX_FILTER_STR_LEN + 1)),
            ..Default::default()
        })
        .unwrap_err();
        assert!(format!("{:?}", err).contains("path_prefix too long"));
    }

    // --- search: granularity resolution ---

    #[test]
    fn granularity_defaults_to_chunk_when_a_query_is_present() {
        assert_eq!(resolve_granularity(true, None).unwrap(), Granularity::Chunk);
    }

    #[test]
    fn granularity_defaults_to_document_when_no_query_is_present() {
        assert_eq!(
            resolve_granularity(false, None).unwrap(),
            Granularity::Document
        );
    }

    #[test]
    fn explicit_granularity_overrides_the_default_either_direction() {
        assert_eq!(
            resolve_granularity(true, Some("document")).unwrap(),
            Granularity::Document
        );
        assert_eq!(
            resolve_granularity(false, Some("chunk")).unwrap(),
            Granularity::Chunk
        );
    }

    #[test]
    fn unknown_granularity_is_rejected() {
        let err = resolve_granularity(true, Some("paragraph")).unwrap_err();
        assert!(format!("{:?}", err).contains("unknown granularity"));
    }

    #[test]
    fn blank_query_is_treated_as_absent_for_granularity_purposes() {
        assert!(!query_is_present(&Some("   ".to_string())));
        assert!(!query_is_present(&None));
        assert!(query_is_present(&Some("pasta".to_string())));
    }

    #[tokio::test]
    async fn chunk_granularity_without_a_query_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);

        let err = server
            .search(Parameters(SearchParams {
                granularity: Some("chunk".to_string()),
                ..Default::default()
            }))
            .await
            .unwrap_err();
        assert!(
            format!("{:?}", err).contains("requires a query"),
            "got: {err:?}"
        );
    }

    // --- search: search_grouped adapter, through the real tool entry point ---
    //
    // A live embeddings endpoint and a live Qdrant are both unavailable in this
    // environment (and `EmbedClient`'s retry/backoff classifies a refused local
    // connection as transient, retrying for up to two minutes — see
    // `embed::embed_backoff` — so even a deliberate connection failure is too
    // slow to use as a fast test signal). These tests therefore cover only what
    // is reachable BEFORE `retrieval::search_grouped` ever calls the embedder:
    // routing, and that `modified_after`/`fields` are parsed by this specific
    // branch rather than silently dropped. The response-shape assembly and
    // `min_score` forwarding (which has no offline-observable effect) need a
    // live Qdrant + embeddings stack to verify; see this module's Report for
    // that gap.

    #[tokio::test]
    async fn search_grouped_routes_a_document_query_to_the_grouped_path_not_enumeration() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);
        seed_document(
            &server,
            "notes/a.md",
            serde_json::json!({ "title": "A", "random_field": "x" }),
        )
        .await;

        let mut filters = serde_json::Map::new();
        filters.insert("random_field".into(), serde_json::json!("x"));

        // Control: enumeration mode (no query) accepts this filter fine, since
        // it never requires a Qdrant payload index — so the divergence below is
        // attributable to routing, not to the filter being universally invalid.
        let enumerate_result = server
            .search(Parameters(SearchParams {
                filters: Some(SearchFiltersInput(filters.clone())),
                ..Default::default()
            }))
            .await
            .expect("enumeration mode does not require a Qdrant payload index");
        assert_eq!(
            enumerate_result.structured_content.unwrap()["total"],
            serde_json::json!(1)
        );

        // The same filter, with a query AND an explicit `document` granularity,
        // must route to the grouped (query+document) path — which DOES require
        // every filter field to carry a Qdrant payload index — not silently
        // fall back to enumeration.
        let err = server
            .search(Parameters(SearchParams {
                query: Some("test".to_string()),
                granularity: Some("document".to_string()),
                filters: Some(SearchFiltersInput(filters)),
                ..Default::default()
            }))
            .await
            .unwrap_err();
        assert!(
            format!("{:?}", err).contains("not indexed for Qdrant queries"),
            "a query+document search must route to the grouped path, which rejects an \
             unindexed filter field before ever reaching Qdrant; got: {err:?}"
        );
    }

    #[tokio::test]
    async fn search_grouped_parses_modified_after_before_reaching_qdrant() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);

        let err = server
            .search(Parameters(SearchParams {
                query: Some("test".to_string()),
                granularity: Some("document".to_string()),
                modified_after: Some("not-a-date".to_string()),
                ..Default::default()
            }))
            .await
            .unwrap_err();
        assert!(
            format!("{:?}", err).contains("invalid date"),
            "the grouped adapter must parse modified_after itself rather than dropping \
             it before retrieval::search_grouped ever runs; got: {err:?}"
        );
    }

    #[tokio::test]
    async fn search_grouped_validates_fields_count_before_reaching_qdrant() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);

        let err = server
            .search(Parameters(SearchParams {
                query: Some("test".to_string()),
                granularity: Some("document".to_string()),
                fields: Some(
                    (0..(MAX_LIST_FILTERS + 1))
                        .map(|i| format!("f{i}"))
                        .collect(),
                ),
                ..Default::default()
            }))
            .await
            .unwrap_err();
        assert!(
            format!("{:?}", err).contains("too many fields requested"),
            "the grouped adapter must validate fields itself, proving the param actually \
             reaches this branch rather than being silently dropped; got: {err:?}"
        );
    }

    #[tokio::test]
    async fn search_grouped_rejects_explain_true() {
        // #132: `explain: true` used to be a silent no-op at document
        // granularity — accepted, never producing a score breakdown, with no
        // signal to the caller that it did nothing. Must now be rejected
        // outright, and rejected before ever reaching Qdrant (no live Qdrant is
        // configured in this test harness, so a Qdrant-side error here would
        // prove the rejection did NOT happen early).
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);

        let err = server
            .search(Parameters(SearchParams {
                query: Some("test".to_string()),
                granularity: Some("document".to_string()),
                explain: Some(true),
                ..Default::default()
            }))
            .await
            .unwrap_err();
        assert!(
            format!("{:?}", err).contains("chunk-granularity only"),
            "explain: true at document granularity must be rejected with an \
             explicit error, not silently ignored; got: {err:?}"
        );
    }

    #[tokio::test]
    async fn search_chunks_rejects_fields() {
        // #132 audit follow-up: `fields` used to be a silent no-op at chunk
        // granularity — accepted, never read by `search_chunks` or
        // `build_chunk_search_payload`, the mirror image of `explain` silently
        // no-opping at document granularity. Must now be rejected outright.
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);

        let err = server
            .search(Parameters(SearchParams {
                query: Some("test".to_string()),
                granularity: Some("chunk".to_string()),
                fields: Some(vec!["status".to_string()]),
                ..Default::default()
            }))
            .await
            .unwrap_err();
        assert!(
            format!("{:?}", err).contains("document-granularity only"),
            "fields at chunk granularity must be rejected with an explicit error, \
             not silently ignored; got: {err:?}"
        );
    }

    // --- search: query-mode filter lowering ---

    /// A `ResolvedConfig` (with `frontmatter.indexed_fields` set to `fields`) paired
    /// with the `SchemaCache` built from that same config with no `.kb-schema.yaml`
    /// on disk, whose root schema falls back to `config.indexed_fields` (see
    /// `SchemaCache::build`'s doc comment) — enough to exercise
    /// `build_query_conditions`'s indexed-field check (now the
    /// `qdrant::all_indexed_fields` union) without a real KB tree. Callers that
    /// want to exercise the *config-only* (legacy) half of that union build a
    /// `SchemaCache` from an empty field list directly instead.
    fn schema_cache_with_indexed(fields: &[&str]) -> (Arc<ResolvedConfig>, SchemaCache) {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = make_test_resolved_config(tmp.path());
        Arc::make_mut(&mut config).frontmatter = crate::config::FrontmatterConfig {
            indexed_fields: fields.iter().map(|f| f.to_string()).collect(),
            ..Default::default()
        };
        let schemas = SchemaCache::build(tmp.path(), &config.frontmatter);
        (config, schemas)
    }

    fn filters_param(json: serde_json::Value) -> SearchParams {
        SearchParams {
            filters: Some(SearchFiltersInput(json.as_object().unwrap().clone())),
            ..Default::default()
        }
    }

    #[test]
    fn query_mode_rejects_a_filter_on_a_field_with_no_payload_index() {
        let (config, schemas) = schema_cache_with_indexed(&["type"]);
        let err = build_query_conditions(
            &filters_param(serde_json::json!({ "untracked_field": "x" })),
            &config,
            &schemas,
        )
        .unwrap_err();
        let msg = format!("{:?}", err);
        assert!(msg.contains("untracked_field"), "got: {msg}");
        assert!(msg.contains("indexed: true"), "got: {msg}");
    }

    #[test]
    fn query_mode_accepts_a_filter_on_an_indexed_field() {
        let (config, schemas) = schema_cache_with_indexed(&["type"]);
        let conditions = build_query_conditions(
            &filters_param(serde_json::json!({ "type": "guide" })),
            &config,
            &schemas,
        )
        .unwrap();
        assert_eq!(conditions.len(), 1);
    }

    /// A field indexed only via the legacy `frontmatter.indexed_fields` config
    /// list — never declared `indexed: true` in any `.kb-schema.yaml` — is
    /// genuinely filterable in Qdrant (`qdrant::all_indexed_fields` unions both
    /// sources when creating payload indexes), so `build_query_conditions` must
    /// accept it too rather than rejecting it by name for only checking the
    /// schema half of that union.
    #[test]
    fn query_mode_accepts_a_legacy_config_only_indexed_field() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = make_test_resolved_config(tmp.path());
        Arc::make_mut(&mut config).frontmatter = crate::config::FrontmatterConfig {
            indexed_fields: vec!["legacy_only".to_string()],
            ..Default::default()
        };
        // Built from an empty field list: nothing is `indexed: true` in any schema,
        // so the schema half of the union contributes nothing for this field.
        let schemas = SchemaCache::build(tmp.path(), &crate::config::FrontmatterConfig::default());

        let conditions = build_query_conditions(
            &filters_param(serde_json::json!({ "legacy_only": "x" })),
            &config,
            &schemas,
        )
        .unwrap();
        assert_eq!(conditions.len(), 1);
    }

    /// When the SAME field name is declared indexed by both a `.kb-schema.yaml`
    /// (with an explicit type) and the legacy `frontmatter.indexed_fields` list,
    /// `qdrant::all_indexed_fields` must let the schema's kind win — not fall
    /// back to the legacy union's implicit keyword default — since that is
    /// exactly what determines whether a numeric range filter on the field is
    /// accepted here or rejected as "declared as Keyword".
    #[test]
    fn query_mode_accepts_a_range_filter_when_schema_kind_beats_the_legacy_default() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  prep_minutes:\n    type: integer\n    indexed: true\n",
        )
        .unwrap();
        let mut config = make_test_resolved_config(tmp.path());
        Arc::make_mut(&mut config).frontmatter = crate::config::FrontmatterConfig {
            indexed_fields: vec!["prep_minutes".to_string()],
            ..Default::default()
        };
        let schemas = SchemaCache::build(tmp.path(), &config.frontmatter);

        let conditions = build_query_conditions(
            &filters_param(serde_json::json!({ "prep_minutes": { "gte": 10 } })),
            &config,
            &schemas,
        )
        .expect(
            "a numeric range must be accepted once the schema's integer kind wins over \
             the legacy keyword default",
        );
        assert_eq!(conditions.len(), 1);
    }

    #[test]
    fn query_mode_filters_reproduce_the_old_domain_type_tags_conditions() {
        // `filters` is the one and only path to narrowing a query-mode search now;
        // this pins that {"domain": ..., "type": ..., "tags": [...]} through it
        // produces the exact same Qdrant conditions the deleted domain/type/tags
        // params used to build directly.
        let (config, schemas) = schema_cache_with_indexed(&["domain", "type", "tags"]);
        let conditions = build_query_conditions(
            &filters_param(serde_json::json!({
                "domain": "sysadmin",
                "type": "guide",
                "tags": ["rust", "rag"],
            })),
            &config,
            &schemas,
        )
        .unwrap();

        assert_eq!(conditions.len(), 3);
        // Every keyword-kind `AnyOf` — regardless of value count — lowers through the
        // same `Condition::matches(_, Vec<String>)` (match-any) shape the old
        // `tags: Vec<String>` array filter always used; `domain`/`type` (formerly
        // single-value scalar matches) now go through that same shape too, since
        // `filters` no longer distinguishes "one value" from "any of these values".
        assert!(
            conditions.contains(&qdrant_client::qdrant::Condition::matches(
                "domain",
                vec!["sysadmin".to_string()]
            ))
        );
        assert!(
            conditions.contains(&qdrant_client::qdrant::Condition::matches(
                "type",
                vec!["guide".to_string()]
            ))
        );
        assert!(
            conditions.contains(&qdrant_client::qdrant::Condition::matches(
                "tags",
                vec!["rust".to_string(), "rag".to_string()]
            ))
        );
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
    fn update_schema_definition_keeps_all_named_properties_after_slimming_descriptions() {
        // Regression test for the `#[schemars(description = "...")]` overrides added to
        // `RawFieldDef` (see that struct in schema.rs): they replace the huge doc-comment
        // text schemars would otherwise copy into every field's `description`, but must
        // not touch the shape of the schema itself. This asserts the full set of named,
        // typed properties the delegation exists to guarantee — including the recursive
        // `fields` map, which is what would silently degrade to an empty object if the
        // override attributes were misapplied (e.g. put on the wrong item, or swallowing
        // the field itself via a typo) rather than just shortening a description.
        let schema = schemars::schema_for!(UpdateSchemaParams);
        let root = schema.as_value();

        let definition_schema = &root["properties"]["definition"];
        let object_schema = definition_schema["anyOf"]
            .as_array()
            .expect("definition must offer a typed alternative, not a bare {}")
            .iter()
            .find(|branch| branch["type"] != serde_json::json!("null"))
            .expect("definition must have a non-null branch");
        let resolved = match object_schema["$ref"].as_str() {
            Some(reference) => &root["$defs"][reference.rsplit('/').next().unwrap()],
            None => object_schema,
        };

        assert_eq!(resolved["type"], serde_json::json!("object"));
        for key in [
            "type", "required", "indexed", "values", "default", "open", "fields",
        ] {
            let prop = &resolved["properties"][key];
            assert!(
                !prop.is_null(),
                "definition schema is missing documented key '{key}': {resolved}"
            );
            let desc = prop["description"]
                .as_str()
                .unwrap_or_else(|| panic!("property '{key}' lost its description: {prop}"));
            assert!(
                desc.len() <= 80,
                "property '{key}' description should be a short override, not the full \
                 doc comment ({} chars): {desc:?}",
                desc.len()
            );
        }

        // `fields` is a map keyed by field name, whose values recurse back into the same
        // definition — confirm that recursion still resolves to the real object schema
        // (via `$ref`) rather than being erased or inlined without bound.
        let fields_prop = &resolved["properties"]["fields"];
        let nested_ref = fields_prop["additionalProperties"]["$ref"]
            .as_str()
            .expect("recursive 'fields' map must $ref back into the definition schema");
        let nested = &root["$defs"][nested_ref.rsplit('/').next().unwrap()];
        assert_eq!(
            nested["type"],
            serde_json::json!("object"),
            "recursive fields entry must resolve to the real object schema: {nested}"
        );
        assert!(
            !nested["properties"]["values"].is_null(),
            "recursive fields entry must keep its own named properties: {nested}"
        );
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
    async fn update_schema_caps_structured_casualties_like_it_caps_the_text() {
        // #148: `documents_broken_by` deliberately returns every casualty — the
        // force/refuse decision needs completeness — but before this fix that full
        // list went straight into `structured_content` while the text half was
        // already capped at MAX_REPORTED_CASUALTIES via `render_casualties`. Seed
        // enough documents to blow past the cap and assert the structured half is
        // bounded too, with a total/truncated flag so a client reading only
        // `structured_content` can still tell it was cut off.
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);
        let seeded = MAX_REPORTED_CASUALTIES + 5;
        for i in 0..seeded {
            seed_document(
                &server,
                &format!("notes/doc{i}.md"),
                serde_json::json!({ "title": format!("Doc {i}") }),
            )
            .await;
        }

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
        assert_eq!(
            structured["would_invalidate"].as_array().unwrap().len(),
            MAX_REPORTED_CASUALTIES,
            "structured_content must cap the casualty list the same way the text \
             rendering does, not embed all {seeded} verbatim"
        );
        assert_eq!(
            structured["casualties_total"],
            serde_json::json!(seeded),
            "the true count must still be reported even though the list is capped"
        );
        assert_eq!(
            structured["casualties_truncated"],
            serde_json::json!(true),
            "truncation must never be silent"
        );
    }

    #[tokio::test]
    async fn update_schema_reports_untruncated_casualties_below_the_cap() {
        // Companion to the truncation test above: when the casualty count is at or
        // under the cap, `casualties_truncated` must read false and
        // `casualties_total` must match the (uncapped) list length exactly, so a
        // client cannot mistake "small" for "truncated."
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
        assert_eq!(structured["would_invalidate"].as_array().unwrap().len(), 1);
        assert_eq!(structured["casualties_total"], serde_json::json!(1));
        assert_eq!(structured["casualties_truncated"], serde_json::json!(false));
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
    async fn search_enumerate_reports_total_and_truncation() {
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
            .search(Parameters(SearchParams {
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
    async fn search_enumerate_filters_through_the_tool_surface() {
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
            .search(Parameters(SearchParams {
                filters: Some(SearchFiltersInput(filters)),
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
    async fn search_enumerate_reports_an_empty_result_clearly() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);

        let result = server
            .search(Parameters(SearchParams::default()))
            .await
            .unwrap();

        assert_eq!(
            result.structured_content.unwrap()["total"],
            serde_json::json!(0)
        );
    }

    /// Enumeration mode's `path_prefix` compiles to an exact SQL `LIKE prefix%`
    /// (see `state::query_documents`) — never the query-mode post-fetch retain
    /// the over-fetch/`path_prefix_truncated` fix exists for. A selective prefix
    /// must still report an exact `total`/`returned`/`has_more`, and the response
    /// must carry no `path_prefix_truncated` key at all: that concept only exists
    /// where a fetch can come up short of what actually matches.
    #[tokio::test]
    async fn search_enumerate_path_prefix_is_unaffected_by_query_mode_overfetch() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);
        for i in 0..5 {
            seed_document(
                &server,
                &format!("keep/{i}.md"),
                serde_json::json!({ "title": format!("Doc {i}") }),
            )
            .await;
        }
        for i in 0..20 {
            seed_document(
                &server,
                &format!("skip/{i}.md"),
                serde_json::json!({ "title": format!("Other {i}") }),
            )
            .await;
        }

        let result = server
            .search(Parameters(SearchParams {
                path_prefix: Some("keep/".to_string()),
                limit: Some(2),
                ..Default::default()
            }))
            .await
            .unwrap();

        let structured = result.structured_content.unwrap();
        assert_eq!(
            structured["total"],
            serde_json::json!(5),
            "an exact SQL LIKE prefix match, unaffected by the query-mode over-fetch \
             even though far more non-matching documents exist elsewhere"
        );
        assert_eq!(structured["returned"], serde_json::json!(2));
        assert_eq!(structured["has_more"], serde_json::json!(true));
        assert!(
            structured.get("path_prefix_truncated").is_none(),
            "path_prefix_truncated is a query-mode-only concept"
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
    fn oversized_filter_scalar_value_is_rejected() {
        // Master enforced MAX_FILTER_STR_LEN on each scalar filter value
        // (domain_too_long_is_rejected et al., since folded into the generic
        // `filters` map). `parse_filters`/`parse_field_filter` cap the field NAME
        // length and the values COUNT, but a single value's length must be capped
        // too, or one call can smuggle an arbitrarily large string into a Qdrant
        // query / SQLite bound parameter.
        let long = "x".repeat(MAX_FILTER_STR_LEN + 1);
        let err = build_document_query(&filters_from(serde_json::json!({ "tags": long.clone() })))
            .unwrap_err();
        assert!(
            format!("{:?}", err).contains("too long"),
            "scalar-equality value over MAX_FILTER_STR_LEN must be rejected: {:?}",
            err
        );
    }

    #[test]
    fn oversized_filter_any_of_value_is_rejected() {
        let long = "x".repeat(MAX_FILTER_STR_LEN + 1);
        let err = build_document_query(&filters_from(
            serde_json::json!({ "tags": { "any_of": [long] } }),
        ))
        .unwrap_err();
        assert!(
            format!("{:?}", err).contains("too long"),
            "an any_of value over MAX_FILTER_STR_LEN must be rejected: {:?}",
            err
        );
    }

    #[test]
    fn oversized_filter_all_of_value_is_rejected() {
        let long = "x".repeat(MAX_FILTER_STR_LEN + 1);
        let err = build_document_query(&filters_from(
            serde_json::json!({ "tags": { "all_of": [long] } }),
        ))
        .unwrap_err();
        assert!(
            format!("{:?}", err).contains("too long"),
            "an all_of value over MAX_FILTER_STR_LEN must be rejected: {:?}",
            err
        );
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
            query: Some(query.to_string()),
            ..Default::default()
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
    fn no_query_is_accepted() {
        // Enumeration mode: `query` is entirely optional now.
        let params = SearchParams::default();
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
            empty_test_description_overlay(),
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
            empty_test_description_overlay(),
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
    fn write_document_tool_description_covers_every_mode() {
        // The COMPILED description owns the API contract and nothing else: the
        // three content modes and how they combine. What belongs in this KB —
        // durable reference material, a shared scratchpad, anything else — is a
        // property of the deployment, not of the binary, so it lives in the
        // per-tool extension (`<kb>/meta/mcp/tools/write_document.md`) which is
        // appended at runtime. One binary serves knowledge bases whose scope
        // policies contradict each other; baking either one in here would ship a
        // description that lies to half of them.
        //
        // The compiled description no longer lives on the `#[tool(...)]`
        // attribute (see `descriptions.rs`) — every tool method carries a bare
        // `#[tool]`, and `KbSearchServer::tool_router().list_all()` therefore
        // returns `description: None` for all six. This asserts against the
        // actual runtime source of the description instead.
        let name = "write_document";
        let description = crate::descriptions::compose_tool_description(name, false, None)
            .unwrap_or_else(|| panic!("no compiled description for tool '{name}'"));

        for mode in [
            "content",
            "old_string",
            "new_path",
            "frontmatter_patch",
            "append",
        ] {
            assert!(
                description.contains(mode),
                "'{name}' description should document the '{mode}' mode: {description}"
            );
        }
    }

    #[test]
    fn tool_router_tools_carry_no_compiled_description() {
        // Every `#[tool(...)]` attribute deliberately carries no `description`
        // (and no doc comment that would leak in as one via the macro's
        // fallback) — the description overlay in `list_tools`/`get_tool` is the
        // only source of a tool's description. This guards against a future
        // edit accidentally reintroducing one, which would silently bypass the
        // overlay for that tool (`list_tools` always applies the overlay
        // unconditionally, but a stray compiled description would still betray
        // the "compiled layer can't state per-KB policy" rule the moment
        // someone writes one back in).
        for tool in KbSearchServer::tool_router().list_all() {
            assert!(
                tool.description.is_none(),
                "tool '{}' unexpectedly carries a compiled description: {:?}",
                tool.name,
                tool.description
            );
        }
    }

    // --- description overlay: list_tools / get_tool ---------------------

    /// Build a bare-bones `KbSearchServer` for overlay tests — same
    /// construction pattern as `get_document_rejects_overlong_path` below,
    /// parameterized by the description overlay under test.
    fn make_overlay_test_server(overlay: HashMap<String, String>) -> KbSearchServer {
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
        // `KbSearchServer::new` canonicalizes `tmp`'s path during construction
        // but never touches the filesystem again afterward (neither does
        // anything these overlay tests exercise — `get_tool`/`list_tools` are
        // pure lock reads), so `tmp` can safely drop once construction
        // returns.
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
            Arc::new(RwLock::new(overlay)),
        )
        .unwrap();
        drop(tmp);
        server
    }

    #[test]
    fn list_tools_returns_the_overlay_composed_descriptions() {
        // `list_tools`'s hand-written body (see the `ServerHandler` impl) is
        // exactly `tool_router().list_all().map(overlay_description)` behind a
        // thin async/RequestContext wrapper the framework provides — `Peer`'s
        // constructor needed to build that context is `pub(crate)` inside
        // `rmcp` and unreachable from here, so this exercises the same overlay
        // logic directly. The wrapper itself (real dispatch through the MCP
        // transport) is covered end-to-end by
        // `server::tests::tools_list_without_session_header_succeeds`
        // and its sibling assertions on `tools/list` descriptions.
        let overlay = crate::descriptions::compose_tool_descriptions(None, false);
        let server = make_overlay_test_server(overlay.clone());

        let tools: Vec<Tool> = KbSearchServer::tool_router()
            .list_all()
            .into_iter()
            .map(|tool| server.overlay_description(tool))
            .collect();

        assert_eq!(tools.len(), crate::descriptions::TOOL_NAMES.len());
        for tool in &tools {
            let expected = overlay
                .get(tool.name.as_ref())
                .unwrap_or_else(|| panic!("no overlay entry for tool '{}'", tool.name));
            assert_eq!(
                tool.description.as_deref(),
                Some(expected.as_str()),
                "tool '{}' description mismatch",
                tool.name
            );
        }
    }

    #[test]
    fn get_tool_returns_the_same_text_as_list_tools_would_for_that_tool() {
        let mut overlay = HashMap::new();
        overlay.insert(
            "search".to_string(),
            "Overlay search description.".to_string(),
        );
        let server = make_overlay_test_server(overlay);

        let tool = server
            .get_tool("search")
            .expect("search tool should be registered");
        assert_eq!(
            tool.description.as_deref(),
            Some("Overlay search description.")
        );

        // Same value the router itself produces, run through the same
        // `overlay_description` path `list_tools` uses.
        let router_tool = KbSearchServer::tool_router()
            .get("search")
            .cloned()
            .unwrap();
        let via_list_tools_path = server.overlay_description(router_tool);
        assert_eq!(tool.description, via_list_tools_path.description);
    }

    #[test]
    fn get_tool_returns_none_for_an_unknown_name() {
        let server = make_overlay_test_server(HashMap::new());
        assert!(server.get_tool("not_a_real_tool").is_none());
    }

    #[test]
    fn description_overlay_recovers_from_a_poisoned_lock() {
        use std::panic;

        let mut overlay = HashMap::new();
        overlay.insert("search".to_string(), "Before poison.".to_string());
        let server = make_overlay_test_server(overlay);

        let overlay_lock = Arc::clone(&server.description_overlay);
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _guard = overlay_lock.write().unwrap();
            panic!("intentional panic to poison the description overlay lock");
        }));
        assert!(
            overlay_lock.read().is_err(),
            "overlay lock should be poisoned"
        );

        // get_tool must not panic, and must recover the last-good value.
        let tool = server
            .get_tool("search")
            .expect("search tool should still be registered after recovery");
        assert_eq!(tool.description.as_deref(), Some("Before poison."));
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
            empty_test_description_overlay(),
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
    async fn get_document_reports_links_out_and_links_in() {
        // End-to-end through the real adapter: seed `document_links` directly via
        // the state DB (bypassing ingest, same shortcut `seed_document` takes for
        // `documents`), then confirm the JSON shape `get_document` hands back —
        // key names, the `exists` flag on a dangling outbound target, and both
        // edge kinds riding the same list distinguished only by `kind`.
        let tmp = tempfile::tempdir().unwrap();
        let server = range_test_server(&tmp);
        let db = server.state_db().await.unwrap();

        // "range_doc.md" is itself indexed as a document so the markdown edge
        // below resolves to an existing target; "missing.md" never is, so the
        // semantic edge to it must come back with exists: false.
        db.upsert_document_metadata(
            "range_doc.md",
            &std::collections::HashMap::new(),
            100,
            "hash",
            1,
        )
        .await
        .unwrap();
        db.replace_links(
            "range_doc.md",
            "markdown",
            &[("missing.md".to_string(), None)],
        )
        .await
        .unwrap();
        db.replace_links(
            "referrer.md",
            "semantic",
            &[("range_doc.md".to_string(), Some(0.42))],
        )
        .await
        .unwrap();

        let (_, structured) = get_range(&server, None, None).await.unwrap();

        let links_out = &structured["links_out"];
        assert_eq!(links_out["total"], 1);
        assert_eq!(links_out["has_more"], false);
        assert_eq!(links_out["links"][0]["target_path"], "missing.md");
        assert_eq!(links_out["links"][0]["kind"], "markdown");
        assert_eq!(links_out["links"][0]["score"], serde_json::Value::Null);
        assert_eq!(
            links_out["links"][0]["exists"], false,
            "missing.md was never indexed, so the outbound edge must be flagged dangling"
        );

        let links_in = &structured["links_in"];
        assert_eq!(links_in["total"], 1);
        assert_eq!(links_in["has_more"], false);
        assert_eq!(links_in["links"][0]["source_path"], "referrer.md");
        assert_eq!(links_in["links"][0]["kind"], "semantic");
        assert_eq!(links_in["links"][0]["score"], 0.42);
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
            empty_test_description_overlay(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn write_document_on_nonexistent_path_upserts_as_a_create() {
        // The merged tool UPSERTS: a `content` write against a path that does not
        // exist creates it (as `create_document` used to), rather than the old
        // standalone `edit_document`'s "does not exist" refusal.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let (server, _config) = make_git_backed_server(&work);

        let params = WriteDocumentParams {
            path: "docs/nonexistent.md".to_string(),
            old_string: None,
            new_string: None,
            content: Some("---\ntitle: Test\n---\n# Body".to_string()),
            message: None,
            expected_hash: None,
            new_path: None,
            force_new: Some(true),
            frontmatter_patch: None,
            append: None,
        };
        let result = server.write_document(Parameters(params)).await;

        let result = result.expect("a write against a nonexistent path must create it");
        let text = format!("{:?}", result.content);
        assert!(
            text.contains("Created 'docs/nonexistent.md'"),
            "a write against a nonexistent path is a CREATE: {text}"
        );
    }

    #[tokio::test]
    async fn write_document_on_existing_path_upserts_as_an_edit() {
        // The merged write_document tool UPSERTS: calling it with `content` against
        // a path that already exists no longer refuses (as `create_document` used
        // to) — it replaces the file, the same as an explicit edit would.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("docs")).unwrap();
        std::fs::write(
            work.path().join("docs/existing.md"),
            "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n",
        )
        .unwrap();
        git_commit_all(&work, "docs/existing.md", "add docs/existing.md");
        let (server, _config) = make_git_backed_server(&work);

        let params = WriteDocumentParams {
            path: "docs/existing.md".to_string(),
            content: Some(
                "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New body\n"
                    .to_string(),
            ),
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_hash: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
        };
        let result = server.write_document(Parameters(params)).await;

        let result =
            result.expect("write_document must upsert rather than refuse an existing path");
        let text = format!("{:?}", result.content);
        assert!(
            text.contains("Edited 'docs/existing.md'"),
            "an upsert onto an existing path is an EDIT, not a create: {text}"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("docs/existing.md")).unwrap(),
            "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New body\n",
        );
    }

    #[tokio::test]
    async fn write_document_frontmatter_patch_end_to_end() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("docs")).unwrap();
        std::fs::write(
            work.path().join("docs/log.md"),
            "---\ntitle: Log\nstatus: draft\n---\n\n# Body\n",
        )
        .unwrap();
        git_commit_all(&work, "docs/log.md", "add docs/log.md");
        let (server, _config) = make_git_backed_server(&work);

        let params = WriteDocumentParams {
            path: "docs/log.md".to_string(),
            content: None,
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_hash: None,
            force_new: None,
            frontmatter_patch: Some(vec![FrontmatterPatchOp {
                operation: "set_field".to_string(),
                field: "status".to_string(),
                value: Some(serde_json::json!("active")),
                values: None,
            }]),
            append: None,
        };
        let result = server.write_document(Parameters(params)).await;
        let result = result.expect("frontmatter_patch must succeed against an existing document");
        let text = format!("{:?}", result.content);
        assert!(text.contains("Edited 'docs/log.md'"), "got: {text}");

        let on_disk = std::fs::read_to_string(work.path().join("docs/log.md")).unwrap();
        assert!(on_disk.contains("status: active"), "got: {on_disk}");
        assert!(on_disk.contains("title: Log"), "title must be preserved");
        assert!(
            on_disk.ends_with("# Body\n"),
            "body must be untouched: {on_disk}"
        );
    }

    #[tokio::test]
    async fn write_document_append_end_to_end() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("docs")).unwrap();
        std::fs::write(
            work.path().join("docs/log.md"),
            "---\ntitle: Log\n---\n\n# Log\n- entry one\n",
        )
        .unwrap();
        git_commit_all(&work, "docs/log.md", "add docs/log.md");
        let (server, _config) = make_git_backed_server(&work);

        let params = WriteDocumentParams {
            path: "docs/log.md".to_string(),
            content: None,
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_hash: None,
            force_new: None,
            frontmatter_patch: None,
            append: Some("- entry two".to_string()),
        };
        let result = server.write_document(Parameters(params)).await;
        result.expect("append must succeed against an existing document");

        assert_eq!(
            std::fs::read_to_string(work.path().join("docs/log.md")).unwrap(),
            "---\ntitle: Log\n---\n\n# Log\n- entry one\n- entry two\n"
        );
    }

    #[tokio::test]
    async fn write_document_frontmatter_patch_and_append_combine_end_to_end() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("docs")).unwrap();
        std::fs::write(
            work.path().join("docs/log.md"),
            "---\ntitle: Log\nstatus: draft\n---\n\n# Log\n- entry one\n",
        )
        .unwrap();
        git_commit_all(&work, "docs/log.md", "add docs/log.md");
        let (server, _config) = make_git_backed_server(&work);

        let params = WriteDocumentParams {
            path: "docs/log.md".to_string(),
            content: None,
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_hash: None,
            force_new: None,
            frontmatter_patch: Some(vec![FrontmatterPatchOp {
                operation: "set_field".to_string(),
                field: "status".to_string(),
                value: Some(serde_json::json!("active")),
                values: None,
            }]),
            append: Some("- entry two".to_string()),
        };
        let result = server.write_document(Parameters(params)).await;
        result.expect("frontmatter_patch + append must succeed together");

        let on_disk = std::fs::read_to_string(work.path().join("docs/log.md")).unwrap();
        assert!(on_disk.contains("status: active"), "got: {on_disk}");
        assert!(
            on_disk.ends_with("- entry one\n- entry two\n"),
            "got: {on_disk}"
        );
    }

    #[tokio::test]
    async fn write_document_frontmatter_patch_validation_failure_reports_error_and_writes_nothing()
    {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::write(
            work.path().join("notes.md"),
            "---\ntitle: T\n---\n\n# Body\n",
        )
        .unwrap();
        git_commit_all(&work, "notes.md", "add notes.md");

        let mut config = make_test_resolved_config(work.path());
        Arc::get_mut(&mut config).unwrap().write.dedup_enabled = false;
        Arc::get_mut(&mut config).unwrap().frontmatter.required = vec!["title".into()];
        let server = make_write_test_server(&work, &["**/*.md".to_string()], config);

        // The patch removes the very field the schema requires.
        let params = WriteDocumentParams {
            path: "notes.md".to_string(),
            content: None,
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_hash: None,
            force_new: None,
            frontmatter_patch: Some(vec![FrontmatterPatchOp {
                operation: "remove_field".to_string(),
                field: "title".to_string(),
                value: None,
                values: None,
            }]),
            append: None,
        };
        let result = server.write_document(Parameters(params)).await;
        let err = result.expect_err("a schema-violating patch must fail validation");
        assert!(err.message.contains("validation"), "got: {}", err.message);
        assert_eq!(
            std::fs::read_to_string(work.path().join("notes.md")).unwrap(),
            "---\ntitle: T\n---\n\n# Body\n",
            "a rejected patch must never touch the file on disk"
        );
    }

    #[tokio::test]
    async fn write_document_frontmatter_patch_on_a_nonexistent_document_is_rejected() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let (server, _config) = make_git_backed_server(&work);

        let params = WriteDocumentParams {
            path: "does-not-exist.md".to_string(),
            content: None,
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_hash: None,
            force_new: None,
            frontmatter_patch: Some(vec![FrontmatterPatchOp {
                operation: "set_field".to_string(),
                field: "status".to_string(),
                value: Some(serde_json::json!("active")),
                values: None,
            }]),
            append: None,
        };
        let err = server
            .write_document(Parameters(params))
            .await
            .expect_err("cannot patch frontmatter on a document that does not exist");
        assert!(
            err.message.contains("does not exist"),
            "got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn write_document_append_on_a_nonexistent_document_is_rejected() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let (server, _config) = make_git_backed_server(&work);

        let params = WriteDocumentParams {
            path: "does-not-exist.md".to_string(),
            content: None,
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_hash: None,
            force_new: None,
            frontmatter_patch: None,
            append: Some("text".to_string()),
        };
        let err = server
            .write_document(Parameters(params))
            .await
            .expect_err("cannot append to a document that does not exist");
        assert!(
            err.message.contains("does not exist"),
            "got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn write_document_on_existing_but_not_permitted_file_returns_include_pattern_error() {
        // G3 regression, preserved across the merge: a path that BOTH exists on
        // disk AND fails the include-pattern check must be reported with the
        // include-pattern message, not "already exists" — the latter would be
        // misleading circular guidance, since a retry would reject the same path
        // as not permitted right back.
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        // Exists on disk as a `.md` file, but the server below only permits
        // `.txt` files — so it both exists AND fails the include-pattern check.
        std::fs::write(sub.join("existing.md"), "# Already here").unwrap();

        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.txt".to_string()], config);

        let params = WriteDocumentParams {
            path: "docs/existing.md".to_string(),
            content: Some("---\ntitle: Test\n---\n# New content".to_string()),
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_hash: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
        };
        let result = server.write_document(Parameters(params)).await;

        assert!(result.is_err(), "write should be rejected");
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
        // `write_document`'s own create-path pre-check (`abs_path.exists()`)
        // catches the ordinary "already exists" case before ever reaching
        // `write.rs` — see `write_document_on_existing_path_upserts_as_an_edit`
        // above (which never reaches this arm at all, since resolving to an
        // existing file routes to the edit path instead of the create path). The
        // `WriteError::AlreadyExists` arm this test drives is only reachable via a
        // genuine TOCTOU race (the file appears between that pre-check and
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
            err.message.contains("write_document"),
            "error should still mention write_document, got: {}",
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
        let params = WriteDocumentParams {
            path: "notes.txt".to_string(),
            content: Some("Some plain text".to_string()),
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_hash: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
        };
        let result = server.write_document(Parameters(params)).await;

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

        // Absolute path should be caught by a guard before any write happens.
        let params = WriteDocumentParams {
            path: "/etc/passwd".to_string(),
            content: Some("# Evil".to_string()),
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_hash: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
        };
        let result = server.write_document(Parameters(params)).await;

        // `/etc/passwd` really exists on the filesystem, so `resolve_within_data`
        // resolves the literal absolute path first (see its doc comment) and
        // returns `Outside` — a hard error from the dispatch above, caught before
        // ever reaching create's include-pattern check. This differs from the
        // "leading `/` means the KB root" convenience: that convenience only
        // applies once the literal absolute path does NOT exist (see
        // `write_document_on_absolute_path_to_real_file_outside_kb_is_hard_error`
        // for that case, and for why the misrouted version of this dispatch used
        // to make this test pass for the wrong reason — the include-pattern guard
        // it originally meant to exercise never actually ran on this input).
        assert!(
            result.is_err(),
            "an absolute path resolving to a real file outside the KB must be rejected"
        );
        let err = result.unwrap_err();
        assert!(
            err.message.contains("outside the data directory"),
            "error should cite the outside-data-directory guard, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn write_document_on_absolute_path_to_real_file_outside_kb_is_hard_error() {
        // `resolve_within_data` tries an absolute path literally first (see its
        // doc comment). When that literal path EXISTS but is outside the data
        // root, it returns `ResolveErr::Outside` — which the dispatch above must
        // treat as a hard failure, not fall through to `write_document_create`.
        // Falling through would strip the leading `/` and join it KB-relative,
        // silently creating a *new* file inside the KB at a path the caller never
        // asked for, while reporting success.
        let tmp = tempfile::tempdir().unwrap();
        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        // A real, existing .md file OUTSIDE the KB data root.
        let outside = tempfile::tempdir().unwrap();
        let outside_file = outside.path().join("real.md");
        std::fs::write(&outside_file, "# Not part of the KB").unwrap();

        let params = WriteDocumentParams {
            path: outside_file.to_str().unwrap().to_string(),
            content: Some("# Evil overwrite".to_string()),
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_hash: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
        };
        let result = server.write_document(Parameters(params)).await;

        assert!(
            result.is_err(),
            "a real absolute path outside the KB must be a hard error, not silently \
             redirected into a create inside the KB"
        );
        let err = result.unwrap_err();
        assert!(
            err.message.contains("outside the data directory"),
            "error should say the path is outside the data directory, got: {}",
            err.message
        );

        // The file outside the KB must be untouched, and nothing must have been
        // created inside the KB at the stripped-leading-slash path.
        assert_eq!(
            std::fs::read_to_string(&outside_file).unwrap(),
            "# Not part of the KB",
            "the real file outside the KB must not be overwritten"
        );
        let bogus_created_path = tmp.path().join(crate::retrieval::kb_root_relative(
            outside_file.to_str().unwrap(),
        ));
        assert!(
            !bogus_created_path.exists(),
            "a bogus file must not have been created inside the KB at {}",
            bogus_created_path.display()
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
                ..Default::default()
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
        let params = WriteDocumentParams {
            path: "guide/missing-title.md".to_string(),
            content: Some("---\ntype: guide\n---\n# No title in frontmatter".to_string()),
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_hash: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
        };
        let result = server.write_document(Parameters(params)).await;

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
        let params = WriteDocumentParams {
            path: "docs/new.md".to_string(),
            content: Some("---\ntitle: Test Doc\n---\n# Content".to_string()),
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_hash: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
        };
        let result = server.write_document(Parameters(params)).await;

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
        let params = WriteDocumentParams {
            path: "docs/forced.md".to_string(),
            content: Some("---\ntitle: Forced Doc\n---\n# Content".to_string()),
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_hash: None,
            force_new: Some(true),
            frontmatter_patch: None,
            append: None,
        };
        let result = server.write_document(Parameters(params)).await;

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
        let params = WriteDocumentParams {
            path: "docs/edit-me.md".to_string(),
            old_string: None,
            new_string: None,
            content: Some("---\ntitle: Edited Doc\n---\n# New content".to_string()),
            message: None,
            expected_hash: None,
            new_path: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
        };
        let result = server.write_document(Parameters(params)).await;

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
    async fn delete_document_overlong_path_rejected() {
        // Mirrors `get_document`'s overlong-path test (see that test's comment on
        // MAX_PATH_LEN): `delete_document` must reject the same class of input
        // before the fuzzy resolver ever runs, not fall through to a resolver-level
        // "not found" error.
        let tmp = tempfile::tempdir().unwrap();
        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        let params = DeleteDocumentParams {
            path: "a".repeat(MAX_PATH_LEN + 1),
            message: None,
        };
        let err = server
            .delete_document(Parameters(params))
            .await
            .expect_err("overlong path should return Err");
        assert!(
            err.message.contains("exceeds maximum length"),
            "error should name the length problem, got: {}",
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
    ) -> WriteDocumentParams {
        WriteDocumentParams {
            path: "docs/test.md".to_string(),
            content: content.map(|s| s.to_string()),
            old_string: old_string.map(|s| s.to_string()),
            new_string: new_string.map(|s| s.to_string()),
            message: None,
            expected_hash: None,
            new_path: new_path.map(|s| s.to_string()),
            force_new: None,
            frontmatter_patch: None,
            append: None,
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
    // (#179) parse_edit_mode: frontmatter_patch / append arms
    // -----------------------------------------------------------------------

    fn set_status_op(value: &str) -> FrontmatterPatchOp {
        FrontmatterPatchOp {
            operation: "set_field".to_string(),
            field: "status".to_string(),
            value: Some(serde_json::json!(value)),
            values: None,
        }
    }

    #[test]
    fn parse_edit_mode_patch_alone_is_recognized() {
        let mut params = make_edit_params(None, None, None, None);
        params.frontmatter_patch = Some(vec![set_status_op("active")]);
        let mode = parse_edit_mode(&params).unwrap();
        match mode {
            Some(EditMode::Patch { edits }) => {
                assert_eq!(edits.len(), 1);
                assert_eq!(
                    edits[0],
                    FrontmatterEdit::SetField {
                        field: "status".to_string(),
                        value: serde_json::json!("active"),
                    }
                );
            }
            other => panic!("expected Patch, got {other:?}"),
        }
    }

    #[test]
    fn parse_edit_mode_append_alone_is_recognized() {
        let mut params = make_edit_params(None, None, None, None);
        params.append = Some("- new entry".to_string());
        let mode = parse_edit_mode(&params).unwrap();
        assert_eq!(
            mode,
            Some(EditMode::Append {
                text: "- new entry".to_string()
            })
        );
    }

    #[test]
    fn parse_edit_mode_patch_and_append_combine() {
        let mut params = make_edit_params(None, None, None, None);
        params.frontmatter_patch = Some(vec![set_status_op("active")]);
        params.append = Some("- new entry".to_string());
        let mode = parse_edit_mode(&params).unwrap();
        match mode {
            Some(EditMode::PatchAppend { edits, text }) => {
                assert_eq!(edits.len(), 1);
                assert_eq!(text, "- new entry");
            }
            other => panic!("expected PatchAppend, got {other:?}"),
        }
    }

    #[test]
    fn parse_edit_mode_patch_combined_with_move_is_recognized() {
        let mut params = make_edit_params(None, None, None, Some("docs/new-home.md"));
        params.frontmatter_patch = Some(vec![set_status_op("active")]);
        let mode = parse_edit_mode(&params).unwrap();
        assert!(matches!(mode, Some(EditMode::Patch { .. })));
    }

    #[test]
    fn parse_edit_mode_patch_with_content_is_rejected() {
        let mut params = make_edit_params(Some("full content"), None, None, None);
        params.frontmatter_patch = Some(vec![set_status_op("active")]);
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(
            err.contains("mutually exclusive"),
            "expected 'mutually exclusive' in error, got: {err}"
        );
    }

    #[test]
    fn parse_edit_mode_append_with_surgical_is_rejected() {
        let mut params = make_edit_params(None, Some("old"), Some("new"), None);
        params.append = Some("more text".to_string());
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(
            err.contains("mutually exclusive"),
            "expected 'mutually exclusive' in error, got: {err}"
        );
    }

    #[test]
    fn parse_edit_mode_empty_frontmatter_patch_list_is_rejected() {
        let mut params = make_edit_params(None, None, None, None);
        params.frontmatter_patch = Some(vec![]);
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(err.contains("at least one operation"), "got: {err}");
    }

    #[test]
    fn parse_edit_mode_blank_append_is_rejected() {
        let mut params = make_edit_params(None, None, None, None);
        params.append = Some("   ".to_string());
        let err = parse_edit_mode(&params).unwrap_err();
        assert!(err.contains("empty"), "got: {err}");
    }

    // -----------------------------------------------------------------------
    // (#179) build_frontmatter_edit / parse_frontmatter_patch_ops
    // -----------------------------------------------------------------------

    #[test]
    fn build_frontmatter_edit_set_field_requires_a_value() {
        let op = FrontmatterPatchOp {
            operation: "set_field".to_string(),
            field: "status".to_string(),
            value: None,
            values: None,
        };
        let err = build_frontmatter_edit(&op).unwrap_err();
        assert!(err.contains("set_field"), "got: {err}");
    }

    #[test]
    fn build_frontmatter_edit_add_values_requires_non_empty_values() {
        let op = FrontmatterPatchOp {
            operation: "add_values".to_string(),
            field: "tags".to_string(),
            value: None,
            values: Some(vec![]),
        };
        let err = build_frontmatter_edit(&op).unwrap_err();
        assert!(err.contains("non-empty"), "got: {err}");
    }

    #[test]
    fn build_frontmatter_edit_rejects_unknown_operation() {
        let op = FrontmatterPatchOp {
            operation: "delete_everything".to_string(),
            field: "status".to_string(),
            value: None,
            values: None,
        };
        let err = build_frontmatter_edit(&op).unwrap_err();
        assert!(err.contains("unknown operation"), "got: {err}");
    }

    #[test]
    fn build_frontmatter_edit_rejects_an_empty_field() {
        let op = FrontmatterPatchOp {
            operation: "remove_field".to_string(),
            field: "  ".to_string(),
            value: None,
            values: None,
        };
        let err = build_frontmatter_edit(&op).unwrap_err();
        assert!(err.contains("field"), "got: {err}");
    }

    #[test]
    fn build_frontmatter_edit_remove_values_parses() {
        let op = FrontmatterPatchOp {
            operation: "remove_values".to_string(),
            field: "tags".to_string(),
            value: None,
            values: Some(vec![serde_json::json!("a")]),
        };
        let edit = build_frontmatter_edit(&op).unwrap();
        assert_eq!(
            edit,
            FrontmatterEdit::RemoveValues {
                field: "tags".to_string(),
                values: vec![serde_json::json!("a")],
            }
        );
    }

    #[test]
    fn parse_frontmatter_patch_ops_rejects_too_many_operations() {
        let ops: Vec<FrontmatterPatchOp> = (0..(MAX_FRONTMATTER_PATCH_OPS + 1))
            .map(|i| set_status_op(&i.to_string()))
            .collect();
        let err = parse_frontmatter_patch_ops(&ops).unwrap_err();
        assert!(err.contains("too many operations"), "got: {err}");
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
            .write_document(Parameters(WriteDocumentParams {
                path: "docs/queued.md".to_string(),
                content: Some(
                    "---\ntitle: Queued\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n"
                        .to_string(),
                ),
                old_string: None,
                new_string: None,
                new_path: None,
                message: None,
                expected_hash: None,
                force_new: Some(true),
                frontmatter_patch: None,
                append: None,
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
            .write_document(Parameters(WriteDocumentParams {
                path: "notes/before.md".to_string(),
                content: Some("---\ntitle: Before\nstatus: beta\n---\n\n# Body\n".to_string()),
                old_string: None,
                new_string: None,
                new_path: None,
                message: None,
                expected_hash: None,
                force_new: Some(true),
                frontmatter_patch: None,
                append: None,
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
            .write_document(Parameters(WriteDocumentParams {
                path: "notes/after.md".to_string(),
                content: Some("---\ntitle: After\nstatus: beta\n---\n\n# Body\n".to_string()),
                old_string: None,
                new_string: None,
                new_path: None,
                message: None,
                expected_hash: None,
                force_new: Some(true),
                frontmatter_patch: None,
                append: None,
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

    #[tokio::test]
    async fn delete_document_surfaces_referencing_paths_end_to_end() {
        // (#229) The reverse-link check #181 added only ever reached a server
        // log — this proves it now also reaches the caller, through both the
        // human-readable text and `structured_content`.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::write(
            work.path().join("linked.md"),
            "---\ntitle: Linked\n---\n\n# Body\n",
        )
        .unwrap();
        git_commit_all(&work, "linked.md", "add linked.md");
        let (server, _config) = make_git_backed_server(&work);

        // Seed the reverse-link index directly (same shortcut other link-graph
        // tests in this module use) rather than depending on a real reindex.
        let db = server.state_db().await.unwrap();
        db.replace_links(
            "referencer.md",
            "markdown",
            &[("linked.md".to_string(), None)],
        )
        .await
        .unwrap();

        let result = server
            .delete_document(Parameters(DeleteDocumentParams {
                path: "linked.md".to_string(),
                message: None,
            }))
            .await
            .expect("an inbound link must not block the delete");

        let text = format!("{:?}", result.content);
        assert!(
            text.contains("referencer.md"),
            "the human-readable summary must name the referencing document: {text}"
        );

        let structured = result
            .structured_content
            .expect("delete_document must attach structured_content");
        assert_eq!(
            structured["referencing_paths"],
            serde_json::json!(["referencer.md"])
        );
    }

    #[tokio::test]
    async fn delete_document_reports_empty_referencing_paths_when_nothing_links_to_it() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::write(
            work.path().join("unlinked.md"),
            "---\ntitle: Unlinked\n---\n\n# Body\n",
        )
        .unwrap();
        git_commit_all(&work, "unlinked.md", "add unlinked.md");
        let (server, _config) = make_git_backed_server(&work);

        let result = server
            .delete_document(Parameters(DeleteDocumentParams {
                path: "unlinked.md".to_string(),
                message: None,
            }))
            .await
            .unwrap();

        let structured = result.structured_content.unwrap();
        assert_eq!(structured["referencing_paths"], serde_json::json!([]));
    }

    // -----------------------------------------------------------------------
    // structured_content parity with the text summary (sha/diff/rebased_paths) —
    // fix #129
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn write_document_structured_content_carries_sha_and_diff() {
        // Before the fix, `structured_content` for a create/edit carried only
        // `{"outcome", "rewritten_paths"}` — a client reading only structured
        // content (Claude Code does) had no programmatic way to learn the commit
        // SHA or see the diff, even though both are in the text summary.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let (server, _config) = make_git_backed_server(&work);

        let result = server
            .write_document(Parameters(WriteDocumentParams {
                path: "docs/new.md".to_string(),
                old_string: None,
                new_string: None,
                content: Some("---\ntitle: Test\n---\n# Body".to_string()),
                message: None,
                expected_hash: None,
                new_path: None,
                force_new: Some(true),
                frontmatter_patch: None,
                append: None,
            }))
            .await
            .unwrap();

        let text = match &result.content[0].raw {
            rmcp::model::RawContent::Text(t) => t.text.clone(),
            other => panic!("expected a text content block, got {other:?}"),
        };
        let structured = result
            .structured_content
            .expect("write_document must attach structured_content");

        let sha = structured["sha"]
            .as_str()
            .expect("structured_content must carry the commit sha as a string");
        assert_eq!(
            sha,
            head_sha(&work),
            "structured sha must match the commit the text summary names"
        );
        assert!(
            text.contains(sha),
            "sanity: the text summary must name the same sha, got: {text}"
        );

        let diff = structured["diff"]
            .as_str()
            .expect("structured_content must carry the diff as a string");
        assert!(
            !diff.is_empty(),
            "a real create must produce a non-empty diff"
        );
        assert!(
            text.contains(diff),
            "the structured diff must match what the text channel already carries, got \
             text: {text} structured diff: {diff}"
        );
        assert_eq!(structured["diff_truncated"], serde_json::json!(false));
        assert_eq!(
            structured["diff_total_bytes"],
            serde_json::json!(diff.len())
        );
        assert_eq!(structured["rebased_paths"], serde_json::json!([]));
        assert!(
            structured.get("sync_failure_cause").is_none(),
            "a fully synced write must not carry sync_failure_cause"
        );
    }

    #[tokio::test]
    async fn write_document_structured_diff_is_capped_for_a_large_write() {
        // The diff can be large (a full replace of a big document) — this
        // codebase never truncates a structured payload silently (`search`'s
        // `has_more`, `update_schema`'s `casualties_truncated`), so a large diff
        // must be capped WITH a flag saying so, the same convention.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let (server, _config) = make_git_backed_server(&work);

        // Comfortably over MAX_STRUCTURED_DIFF_BYTES (8 KiB) so the added-lines
        // diff itself, not just the raw content, exceeds the cap.
        let big_body: String = (0..2000)
            .map(|i| format!("line {i} of filler body text\n"))
            .collect();
        let content = format!("---\ntitle: Big\n---\n# Body\n{big_body}");

        let result = server
            .write_document(Parameters(WriteDocumentParams {
                path: "docs/big.md".to_string(),
                old_string: None,
                new_string: None,
                content: Some(content),
                message: None,
                expected_hash: None,
                new_path: None,
                force_new: Some(true),
                frontmatter_patch: None,
                append: None,
            }))
            .await
            .unwrap();

        let text = match &result.content[0].raw {
            rmcp::model::RawContent::Text(t) => t.text.clone(),
            other => panic!("expected a text content block, got {other:?}"),
        };
        let structured = result.structured_content.unwrap();

        let diff = structured["diff"].as_str().unwrap();
        assert!(
            diff.len() <= MAX_STRUCTURED_DIFF_BYTES,
            "capped diff must not exceed the byte cap, got {} bytes",
            diff.len()
        );
        assert_eq!(structured["diff_truncated"], serde_json::json!(true));
        let diff_total_bytes = structured["diff_total_bytes"].as_u64().unwrap() as usize;
        assert!(
            diff_total_bytes > MAX_STRUCTURED_DIFF_BYTES,
            "diff_total_bytes must report the TRUE, uncapped length"
        );
        assert!(
            text.contains(diff),
            "the capped structured diff must still be a prefix of the full text diff"
        );
        assert!(
            text.len() > diff.len(),
            "the TEXT channel must never be truncated by this cap — only structured_content"
        );
    }

    #[tokio::test]
    async fn delete_document_structured_content_carries_sha_and_diff() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::write(
            work.path().join("gone.md"),
            "---\ntitle: Gone\n---\n\n# Body\n",
        )
        .unwrap();
        git_commit_all(&work, "gone.md", "add gone.md");
        let (server, _config) = make_git_backed_server(&work);

        let result = server
            .delete_document(Parameters(DeleteDocumentParams {
                path: "gone.md".to_string(),
                message: None,
            }))
            .await
            .unwrap();

        let structured = result.structured_content.unwrap();
        assert_eq!(
            structured["sha"].as_str().unwrap(),
            head_sha(&work),
            "delete_document's structured_content must carry the commit sha too"
        );
        let diff = structured["diff"].as_str().unwrap();
        assert!(
            !diff.is_empty(),
            "a delete must produce a non-empty (all-removals) diff"
        );
        assert_eq!(structured["rebased_paths"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn write_document_directory_move_structured_content_carries_sha_and_rebased_paths() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old-project7")).unwrap();
        let doc_content = "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old-project7/a.md"), doc_content).unwrap();
        git_commit_all(&work, "old-project7/a.md", "add old-project7/a.md");
        let (server, _config) = make_git_backed_server(&work);

        let result = server
            .write_document(Parameters(WriteDocumentParams {
                path: "old-project7".to_string(),
                content: None,
                old_string: None,
                new_string: None,
                new_path: Some("archive/new-project7".to_string()),
                message: None,
                expected_hash: None,
                force_new: None,
                frontmatter_patch: None,
                append: None,
            }))
            .await
            .unwrap();

        let structured = result.structured_content.unwrap();
        assert_eq!(
            structured["sha"].as_str().unwrap(),
            head_sha(&work),
            "a directory move's structured_content must carry the commit sha too"
        );
        assert_eq!(structured["rebased_paths"], serde_json::json!([]));
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
        // (#229) `delete_document` now unconditionally opens the state DB for
        // the inbound-link check, which lazily materializes `state.db`/`-shm`/
        // `-wal` under `work` — see `git_status_ignoring_state_db`'s doc
        // comment for why that is an expected, unrelated side effect rather
        // than evidence the rollback itself left something dirty.
        assert_eq!(
            git_status_ignoring_state_db(&work),
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
            .write_document(Parameters(WriteDocumentParams {
                path: "docs/new.md".to_string(),
                content: Some(
                    "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n"
                        .to_string(),
                ),
                old_string: None,
                new_string: None,
                new_path: None,
                message: None,
                expected_hash: None,
                force_new: Some(true),
                frontmatter_patch: None,
                append: None,
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
            .write_document(Parameters(WriteDocumentParams {
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
                force_new: None,
                frontmatter_patch: None,
                append: None,
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
            .write_document(Parameters(WriteDocumentParams {
                path: "edit-me.md".to_string(),
                old_string: None,
                new_string: None,
                content: Some(new_content.to_string()),
                message: None,
                expected_hash: None,
                new_path: None,
                force_new: None,
                frontmatter_patch: None,
                append: None,
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
            .write_document(Parameters(WriteDocumentParams {
                path: "docs/edit-me.md".to_string(),
                old_string: None,
                new_string: None,
                content: Some("---\ntitle: New\ntype: guide\n---\n# New body\n".to_string()),
                message: None,
                expected_hash: Some(stale_hash),
                new_path: None,
                force_new: None,
                frontmatter_patch: None,
                append: None,
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
            .write_document(Parameters(WriteDocumentParams {
                path: "edit-me.md".to_string(),
                old_string: None,
                new_string: None,
                content: Some(new_content.to_string()),
                message: None,
                expected_hash: Some(correct_hash),
                new_path: None,
                force_new: None,
                frontmatter_patch: None,
                append: None,
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

    /// #243 regression, at the MCP layer specifically — `write.rs`'s own
    /// `stale_hash_re_check_catches_a_change_made_after_the_first_check`
    /// already covers the re-check itself, but that test (and every
    /// `expected_hash` test above this one) constructs the concurrent change
    /// BEFORE the call, which `write_document_edit`'s own up-front check
    /// (comparing against the `old_content` it freshly reads from disk right
    /// then) catches on its own regardless of whether `run_document_write`
    /// threads `expected_hash` into `WriteRequest` at all — so none of those
    /// would have failed before the fix. This test instead makes the change
    /// land DURING the call, in the `validate::validate_content` await, via a
    /// `lint_command` that overwrites the file mid-flight — the exact failure
    /// scenario #243 describes (a webhook merge landing while an arbitrarily
    /// slow lint command runs). Before the fix, `run_document_write` passed
    /// `expected_hash: None` into `WriteRequest` no matter what the caller
    /// sent, so `write::write_document`'s live-disk re-check was skipped
    /// entirely and the write proceeded, silently clobbering the concurrent
    /// change. After the fix, the re-check catches it and the write is
    /// rejected before ever reaching the filesystem overwrite.
    #[tokio::test]
    async fn write_document_tool_re_checks_expected_hash_against_a_change_made_during_the_call() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original =
            "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
        std::fs::write(work.path().join("edit-me.md"), original).unwrap();
        git_commit_all(&work, "edit-me.md", "add edit-me.md");
        let head_before = head_sha(&work);

        // Simulates a concurrent change (a webhook merge, in production)
        // landing between `write_document_edit`'s own read of the file and the
        // actual overwrite inside `write::write_document` — modeled here as a
        // `lint_command` that overwrites the file mid-flight, during the
        // `validate::validate_content` await that runs before the re-check.
        let concurrent = "---\ntitle: Concurrent\ndescription: d\ntype: guide\ntags: \
                           [t]\n---\n\n# Concurrent body\n";
        let abs_path = work.path().join("edit-me.md");

        let mut config = make_test_resolved_config(work.path());
        Arc::get_mut(&mut config).unwrap().validation.lint_command = Some(vec![
            "sh".to_string(),
            "-c".to_string(),
            format!("printf '%s' '{}' > '{}'", concurrent, abs_path.display()),
        ]);
        let server = make_write_test_server(&work, &["**/*.md".to_string()], config);

        let expected_hash = crate::ingest::compute_hash_from_bytes(original.as_bytes());
        let concurrent_hash = crate::ingest::compute_hash_from_bytes(concurrent.as_bytes());
        let new_content =
            "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New body\n";

        let result = server
            .write_document(Parameters(WriteDocumentParams {
                path: "edit-me.md".to_string(),
                old_string: None,
                new_string: None,
                content: Some(new_content.to_string()),
                message: None,
                expected_hash: Some(expected_hash),
                new_path: None,
                force_new: None,
                frontmatter_patch: None,
                append: None,
            }))
            .await;

        let err = result.expect_err(
            "a concurrent on-disk change made during the call must be caught by the \
             live-disk re-check, not silently clobbered",
        );
        assert!(
            err.message.contains("changed since you read it"),
            "expected the stale-hash message, got: {}",
            err.message
        );
        assert!(
            err.message.contains(&concurrent_hash),
            "the rejection must report the LIVE on-disk hash (the lint_command's \
             concurrent write), not the caller's original — got: {}",
            err.message
        );

        // The concurrent write must survive untouched — that is the whole
        // point of the re-check: it must never be silently clobbered.
        assert_eq!(
            std::fs::read_to_string(&abs_path).unwrap(),
            concurrent,
            "the concurrent change must survive the rejected write"
        );
        assert_eq!(head_before, head_sha(&work), "no commit must be made");
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
            .write_document(Parameters(WriteDocumentParams {
                path: "docs/old-home.md".to_string(),
                old_string: None,
                new_string: None,
                content: None,
                message: None,
                expected_hash: None,
                new_path: Some("docs/new-home.md".to_string()),
                force_new: None,
                frontmatter_patch: None,
                append: None,
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
            .write_document(Parameters(WriteDocumentParams {
                path: "edit-me.md".to_string(),
                old_string: None,
                new_string: None,
                content: Some(new_content.to_string()),
                message: None,
                expected_hash: None,
                new_path: Some("archive/edit-me.md".to_string()),
                force_new: None,
                frontmatter_patch: None,
                append: None,
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
            .write_document(Parameters(WriteDocumentParams {
                path: "source.md".to_string(),
                old_string: None,
                new_string: None,
                content: None,
                message: None,
                expected_hash: None,
                new_path: Some("dest.md".to_string()),
                force_new: None,
                frontmatter_patch: None,
                append: None,
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
    // write_document: directory-move dispatch (path resolves to a directory)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn write_document_dispatches_to_directory_move_when_path_is_a_directory() {
        // `path` resolving to an existing directory dispatches write_document to a
        // whole-subtree move via `write::move_directory`, mirroring the old
        // standalone `move_directory` tool.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old-project")).unwrap();
        let doc_content = "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old-project/a.md"), doc_content).unwrap();
        git_commit_all(&work, "old-project/a.md", "add old-project/a.md");
        let (server, _config) = make_git_backed_server(&work);

        let result = server
            .write_document(Parameters(WriteDocumentParams {
                path: "old-project".to_string(),
                content: None,
                old_string: None,
                new_string: None,
                new_path: Some("archive/new-project".to_string()),
                message: None,
                expected_hash: None,
                force_new: None,
                frontmatter_patch: None,
                append: None,
            }))
            .await;

        let result = result.expect("a directory path must dispatch to a directory move");
        let text = format!("{:?}", result.content);
        assert!(
            text.contains("Moved 1 document(s)"),
            "must report the directory move, not a single-document write: {text}"
        );
        assert!(
            !work.path().join("old-project/a.md").exists(),
            "the source directory's document must be gone after the move"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("archive/new-project/a.md")).unwrap(),
            doc_content,
        );
    }

    #[tokio::test]
    async fn write_document_directory_move_requires_new_path() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("some-dir")).unwrap();
        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        let result = server
            .write_document(Parameters(WriteDocumentParams {
                path: "some-dir".to_string(),
                content: None,
                old_string: None,
                new_string: None,
                new_path: None,
                message: None,
                expected_hash: None,
                force_new: None,
                frontmatter_patch: None,
                append: None,
            }))
            .await;

        let err = result.expect_err("a directory move without new_path must be rejected");
        assert!(
            err.message.contains("new_path"),
            "error should name new_path as required, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn write_document_directory_move_rejects_single_document_fields() {
        // Every field that only makes sense for a single document must be
        // rejected, by name, when `path` resolves to a directory.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("some-dir")).unwrap();
        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        let base = || WriteDocumentParams {
            path: "some-dir".to_string(),
            content: None,
            old_string: None,
            new_string: None,
            new_path: Some("other-dir".to_string()),
            message: None,
            expected_hash: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
        };

        let cases: Vec<(&str, WriteDocumentParams)> = vec![
            (
                "content",
                WriteDocumentParams {
                    content: Some("x".to_string()),
                    ..base()
                },
            ),
            (
                "old_string",
                WriteDocumentParams {
                    old_string: Some("x".to_string()),
                    ..base()
                },
            ),
            (
                "new_string",
                WriteDocumentParams {
                    new_string: Some("x".to_string()),
                    ..base()
                },
            ),
            (
                "expected_hash",
                WriteDocumentParams {
                    expected_hash: Some("deadbeef".to_string()),
                    ..base()
                },
            ),
            (
                "force_new",
                WriteDocumentParams {
                    force_new: Some(true),
                    ..base()
                },
            ),
            (
                "frontmatter_patch",
                WriteDocumentParams {
                    frontmatter_patch: Some(vec![FrontmatterPatchOp {
                        operation: "set_field".to_string(),
                        field: "status".to_string(),
                        value: Some(serde_json::json!("active")),
                        values: None,
                    }]),
                    ..base()
                },
            ),
            (
                "append",
                WriteDocumentParams {
                    append: Some("more text".to_string()),
                    ..base()
                },
            ),
        ];

        for (field, params) in cases {
            let result = server.write_document(Parameters(params)).await;
            let err = result.expect_err(&format!(
                "a directory move with {field} set must be rejected"
            ));
            assert!(
                err.message.contains(field),
                "error should name '{field}' as the rejected field, got: {}",
                err.message
            );
        }
    }

    // -----------------------------------------------------------------------
    // write_document: create-path guards (surgical/content on a nonexistent path)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn write_document_surgical_edit_on_nonexistent_path_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        let result = server
            .write_document(Parameters(WriteDocumentParams {
                path: "docs/nonexistent.md".to_string(),
                content: None,
                old_string: Some("old".to_string()),
                new_string: Some("new".to_string()),
                new_path: None,
                message: None,
                expected_hash: None,
                force_new: None,
                frontmatter_patch: None,
                append: None,
            }))
            .await;

        let err = result.expect_err("a surgical edit against a nonexistent path must be rejected");
        assert!(
            err.message
                .contains("cannot surgically edit a document that does not exist"),
            "got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn write_document_overlong_path_rejected() {
        // Mirrors `get_document`'s overlong-path test: `write_document` must reject
        // the same class of input before `resolve_safe_write_path`, the
        // include-pattern check, or git staging ever see it — see #153.
        let tmp = tempfile::tempdir().unwrap();
        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        let err = server
            .write_document(Parameters(WriteDocumentParams {
                path: "a".repeat(MAX_PATH_LEN + 1),
                content: Some("---\ntitle: Test\n---\n# Body".to_string()),
                old_string: None,
                new_string: None,
                new_path: None,
                message: None,
                expected_hash: None,
                force_new: Some(true),
                frontmatter_patch: None,
                append: None,
            }))
            .await
            .expect_err("overlong path should return Err");
        assert!(
            err.message.contains("exceeds maximum length"),
            "error should name the length problem, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn write_document_overlong_new_path_rejected() {
        // Same guard, applied to `new_path` (used for moves) rather than `path` —
        // see #153.
        let tmp = tempfile::tempdir().unwrap();
        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);
        std::fs::write(
            tmp.path().join("existing.md"),
            "---\ntitle: Old\n---\n# Old\n",
        )
        .unwrap();

        let err = server
            .write_document(Parameters(WriteDocumentParams {
                path: "existing.md".to_string(),
                content: None,
                old_string: None,
                new_string: None,
                new_path: Some("a".repeat(MAX_PATH_LEN + 1)),
                message: None,
                expected_hash: None,
                force_new: None,
                frontmatter_patch: None,
                append: None,
            }))
            .await
            .expect_err("overlong new_path should return Err");
        assert!(
            err.message.contains("exceeds maximum length"),
            "error should name the length problem, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn write_document_create_without_content_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        let result = server
            .write_document(Parameters(WriteDocumentParams {
                path: "docs/nonexistent.md".to_string(),
                content: None,
                old_string: None,
                new_string: None,
                new_path: None,
                message: None,
                expected_hash: None,
                force_new: None,
                frontmatter_patch: None,
                append: None,
            }))
            .await;

        let err = result.expect_err("a create with no content must be rejected");
        assert!(
            err.message
                .contains("content is required to create a new document"),
            "got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn write_document_new_path_on_a_create_is_rejected() {
        // Not explicitly specced, but forced by `write::WriteRequest::dest_path`'s
        // own contract: `is_create` must be `false` whenever `dest_path` is
        // `Some`, since a create has no source to move from. write_document must
        // reject this combination outright rather than silently drop new_path or
        // pass an invalid request down to the write pipeline.
        let tmp = tempfile::tempdir().unwrap();
        let config = make_test_resolved_config(tmp.path());
        let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

        let result = server
            .write_document(Parameters(WriteDocumentParams {
                path: "docs/nonexistent.md".to_string(),
                content: Some("---\ntitle: Test\n---\n# Body".to_string()),
                old_string: None,
                new_string: None,
                new_path: Some("docs/elsewhere.md".to_string()),
                message: None,
                expected_hash: None,
                force_new: None,
                frontmatter_patch: None,
                append: None,
            }))
            .await;

        let err = result.expect_err("new_path on a create must be rejected");
        assert!(err.message.contains("new_path"), "got: {}", err.message);
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
    /// left the WRITE ITSELF clean. `delete_document` does the same (#229: it always
    /// opens the state DB for the inbound-link check, unlike a MOVE's conditional
    /// open), so its tests need this filtered helper too. `create_document`/plain
    /// `edit_document` (no `new_path`) never touch the metadata index at all, so
    /// their equivalent tests can keep the plain, exact `git_status` assertion.
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
            .write_document(Parameters(WriteDocumentParams {
                path: "notes/after.md".to_string(),
                content: Some("---\ntitle: After\nstatus: beta\n---\n\n# Body\n".to_string()),
                old_string: None,
                new_string: None,
                new_path: None,
                message: None,
                expected_hash: None,
                force_new: Some(true),
                frontmatter_patch: None,
                append: None,
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

    // --- build_chunk_search_payload / build_grouped_search_payload ---
    //
    // These drive the response-assembly seam directly with hand-built
    // `SearchResult`/`GroupedDocument` values — no network, no mocked
    // `KbSearchServer`, none of `EmbedClient`'s retry/backoff to defeat. This is
    // exactly the gap that let `search_chunks` ship with `structured_content`
    // carrying only `{"path_prefix_truncated": ...}` while the text content had
    // full results: nothing between "retrieval returned results" and
    // "CallToolResult handed to the client" was reachable by any test.

    fn payload_search_result(
        file_path: &str,
        title: &str,
        score: f32,
    ) -> crate::qdrant::SearchResult {
        let mut payload = HashMap::new();
        payload.insert("file_path".to_string(), serde_json::json!(file_path));
        payload.insert("title".to_string(), serde_json::json!(title));
        payload.insert("text".to_string(), serde_json::json!("some body text"));
        crate::qdrant::SearchResult {
            score,
            pre_rerank_score: None,
            dense_score: None,
            sparse_score: None,
            phrase_score: None,
            payload,
        }
    }

    #[test]
    fn build_chunk_search_payload_structured_results_present_and_populated() {
        // The regression itself: a non-empty result set must produce a
        // `structured_content.results` array of the same length as the input,
        // with each entry carrying `file_path`/`title`/`score`/`text`. Against
        // the old `{"path_prefix_truncated": ...}`-only payload, `structured
        // ["results"]` would be `Value::Null` and every index below would fail.
        let results = vec![
            payload_search_result("/data/notes/a.md", "A", 0.9),
            payload_search_result("/data/notes/b.md", "B", 0.5),
        ];

        let (_text, structured) = build_chunk_search_payload(
            &results,
            Path::new("/data"),
            false,
            "dense cosine",
            false,
            false,
            0,
            None,
        );

        let arr = structured["results"]
            .as_array()
            .expect("results must be an array, not missing/null");
        assert_eq!(arr.len(), results.len());
        for entry in arr {
            assert!(entry["file_path"].is_string());
            assert!(entry["title"].is_string());
            assert!(entry["score"].is_number());
            assert!(entry["text"].is_string());
        }
        assert_eq!(structured["returned"], serde_json::json!(2));
    }

    #[test]
    fn build_chunk_search_payload_empty_results_report_zero_not_missing_key() {
        let (text, structured) = build_chunk_search_payload(
            &[],
            Path::new("/data"),
            false,
            "dense cosine",
            false,
            false,
            0,
            None,
        );

        assert_eq!(text, "No results found.");
        assert_eq!(structured["returned"], serde_json::json!(0));
        assert_eq!(
            structured["results"]
                .as_array()
                .expect("must be an array, not missing"),
            &Vec::<serde_json::Value>::new()
        );
    }

    #[test]
    fn build_chunk_search_payload_text_and_structured_agree() {
        let results = vec![
            payload_search_result("/data/notes/a.md", "A", 0.9),
            payload_search_result("/data/notes/b.md", "B", 0.5),
            payload_search_result("/data/notes/c.md", "C", 0.1),
        ];

        let (text, structured) = build_chunk_search_payload(
            &results,
            Path::new("/data"),
            false,
            "dense cosine",
            false,
            false,
            0,
            None,
        );

        let arr = structured["results"].as_array().unwrap();
        assert_eq!(arr.len(), 3);
        let structured_paths: Vec<&str> = arr
            .iter()
            .map(|e| e["file_path"].as_str().unwrap())
            .collect();
        assert_eq!(
            structured_paths,
            vec!["notes/a.md", "notes/b.md", "notes/c.md"]
        );

        // Text content lists the same files, in the same order.
        let text_result_1_pos = text.find("notes/a.md").unwrap();
        let text_result_2_pos = text.find("notes/b.md").unwrap();
        let text_result_3_pos = text.find("notes/c.md").unwrap();
        assert!(text_result_1_pos < text_result_2_pos);
        assert!(text_result_2_pos < text_result_3_pos);
    }

    #[test]
    fn build_chunk_search_payload_path_prefix_truncated_note_and_flag() {
        let results = vec![payload_search_result("/data/notes/a.md", "A", 0.9)];

        let (text, structured) = build_chunk_search_payload(
            &results,
            Path::new("/data"),
            false,
            "dense cosine",
            true,
            false,
            0,
            None,
        );

        assert_eq!(structured["path_prefix_truncated"], serde_json::json!(true));
        assert!(
            text.contains("path_prefix matched more candidates"),
            "text must render the truncation note; got: {text}"
        );

        // Same for the empty-results branch.
        let (empty_text, empty_structured) = build_chunk_search_payload(
            &[],
            Path::new("/data"),
            false,
            "dense cosine",
            true,
            false,
            0,
            None,
        );
        assert_eq!(
            empty_structured["path_prefix_truncated"],
            serde_json::json!(true)
        );
        assert!(
            empty_text.contains("path_prefix matched more candidates"),
            "empty-results text must also render the truncation note; got: {empty_text}"
        );
    }

    #[test]
    fn build_chunk_search_payload_offset_truncated_note_and_flag() {
        // #224: mirrors the path_prefix_truncated test above, but for the
        // offset-depth-bound signal — proves the flag reaches structured_content
        // AND that the text body explains why the page may be short, on both
        // the non-empty and empty-results branches. `offset` is non-zero here
        // (the "usual" paging-too-deep case), so the note still tells the
        // caller to narrow the query or lower offset — see the offset == 0
        // variant below (#240) for the case where that advice is wrong.
        let results = vec![payload_search_result("/data/notes/a.md", "A", 0.9)];

        let (text, structured) = build_chunk_search_payload(
            &results,
            Path::new("/data"),
            false,
            "dense cosine",
            false,
            true,
            5,
            None,
        );

        assert_eq!(structured["offset_truncated"], serde_json::json!(true));
        assert!(
            text.contains("offset + limit reached past"),
            "text must render the offset-truncation note; got: {text}"
        );
        assert!(
            text.contains("Narrow the query or lower offset"),
            "a non-zero offset must still get the lower-offset advice; got: {text}"
        );

        let (empty_text, empty_structured) = build_chunk_search_payload(
            &[],
            Path::new("/data"),
            false,
            "dense cosine",
            false,
            true,
            5,
            None,
        );
        assert_eq!(
            empty_structured["offset_truncated"],
            serde_json::json!(true)
        );
        assert!(
            empty_text.contains("offset + limit reached past"),
            "empty-results text must also render the offset-truncation note; got: {empty_text}"
        );
    }

    /// #240: at `offset == 0` the depth bound was tripped by `limit` alone —
    /// there is no offset to lower (it's already zero), so telling the caller
    /// to "narrow the query or lower offset" is nonsensical advice pointing at
    /// a knob they never touched. The note must instead point at `limit` and
    /// (when a reranker sized the bound) name `reranking.candidate_limit`
    /// directly, rather than reuse the offset > 0 wording.
    #[test]
    fn build_chunk_search_payload_offset_truncated_at_offset_zero_names_the_right_knob() {
        let results = vec![payload_search_result("/data/notes/a.md", "A", 0.9)];

        // No reranker configured: the bound is the fixed absolute ceiling, not
        // a tunable `reranking.candidate_limit`.
        let (text_no_reranker, _) = build_chunk_search_payload(
            &results,
            Path::new("/data"),
            false,
            "dense cosine",
            false,
            true,
            0,
            None,
        );
        assert!(
            !text_no_reranker.contains("lower offset"),
            "offset is already 0 — must not tell the caller to lower it: {text_no_reranker}"
        );
        assert!(
            text_no_reranker.contains("Lower limit"),
            "must point at limit instead: {text_no_reranker}"
        );
        assert!(
            !text_no_reranker.contains("reranking.candidate_limit"),
            "no reranker is configured, so the note must not name a knob that \
             does not apply: {text_no_reranker}"
        );

        // Reranker configured with a candidate_limit: the note should name it
        // directly, as the actual knob to raise.
        let (text_reranker, _) = build_chunk_search_payload(
            &results,
            Path::new("/data"),
            false,
            "dense cosine",
            false,
            true,
            0,
            Some(50),
        );
        assert!(
            !text_reranker.contains("lower offset"),
            "offset is already 0 — must not tell the caller to lower it: {text_reranker}"
        );
        assert!(
            text_reranker.contains("reranking.candidate_limit is currently 50"),
            "must name the actual configured bound: {text_reranker}"
        );
        assert!(
            text_reranker.contains("Raise reranking.candidate_limit or lower limit"),
            "must give actionable advice naming the real knobs: {text_reranker}"
        );
    }

    #[test]
    fn build_chunk_search_payload_explain_toggles_score_breakdown_line() {
        let results = vec![payload_search_result("/data/notes/a.md", "A", 0.9)];

        let (text_off, _) = build_chunk_search_payload(
            &results,
            Path::new("/data"),
            false,
            "dense cosine",
            false,
            false,
            0,
            None,
        );
        assert!(!text_off.contains("Score breakdown"));

        let (text_on, _) = build_chunk_search_payload(
            &results,
            Path::new("/data"),
            true,
            "dense cosine",
            false,
            false,
            0,
            None,
        );
        assert!(text_on.contains("Score breakdown"));
    }

    #[test]
    fn build_chunk_search_payload_mode_label_appears_verbatim() {
        let results = vec![payload_search_result("/data/notes/a.md", "A", 0.9)];

        for mode in [
            "hybrid RRF + phrase",
            "hybrid RRF",
            "dense + phrase RRF",
            "dense cosine",
        ] {
            let (text, _) = build_chunk_search_payload(
                &results,
                Path::new("/data"),
                true,
                mode,
                false,
                false,
                0,
                None,
            );
            assert!(
                text.contains(&format!("mode={mode}")),
                "expected mode label {mode:?} verbatim in: {text}"
            );
        }
    }

    #[test]
    fn build_chunk_search_payload_phrase_matched_tracks_phrase_score_presence() {
        let mut matched = payload_search_result("/data/notes/a.md", "A", 0.9);
        matched.phrase_score = Some(0.7);
        let mut unmatched = payload_search_result("/data/notes/b.md", "B", 0.5);
        unmatched.phrase_score = None;

        let (_text, structured) = build_chunk_search_payload(
            &[matched, unmatched],
            Path::new("/data"),
            false,
            "dense + phrase RRF",
            false,
            false,
            0,
            None,
        );

        let arr = structured["results"].as_array().unwrap();
        assert_eq!(arr[0]["phrase_matched"], serde_json::json!(true));
        assert_eq!(arr[1]["phrase_matched"], serde_json::json!(false));
    }

    fn payload_grouped_document(
        file_path: &str,
        title: &str,
        score: f32,
    ) -> retrieval::GroupedDocument {
        retrieval::GroupedDocument {
            score,
            summary: crate::state::DocumentSummary {
                file_path: file_path.to_string(),
                title: Some(title.to_string()),
                description: None,
                mtime: 0,
                indexed_at: "2026-01-01T00:00:00Z".to_string(),
                frontmatter: serde_json::json!({}),
            },
        }
    }

    #[test]
    fn build_grouped_search_payload_omits_total_and_has_more_but_keeps_score() {
        let documents = vec![
            payload_grouped_document("notes/a.md", "A", 0.9),
            payload_grouped_document("notes/b.md", "B", 0.5),
        ];

        let (_text, structured) = build_grouped_search_payload(&documents, false, false, 0);

        let obj = structured.as_object().unwrap();
        assert!(
            !obj.contains_key("total"),
            "grouped search cannot back `total` and must not claim one: {structured}"
        );
        assert!(
            !obj.contains_key("has_more"),
            "grouped search cannot back `has_more` and must not claim one: {structured}"
        );

        let docs = structured["documents"].as_array().unwrap();
        assert_eq!(docs.len(), 2);
        for doc in docs {
            assert!(
                doc["score"].is_number(),
                "each document must carry a score: {doc}"
            );
        }
        assert_eq!(structured["returned"], serde_json::json!(2));
    }

    #[test]
    fn build_grouped_search_payload_empty_reports_zero() {
        let (text, structured) = build_grouped_search_payload(&[], false, false, 0);
        assert_eq!(text, "No documents matched.");
        assert_eq!(structured["returned"], serde_json::json!(0));
        assert_eq!(structured["documents"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn build_grouped_search_payload_path_prefix_truncated_note_and_flag() {
        let documents = vec![payload_grouped_document("notes/a.md", "A", 0.9)];
        let (text, structured) = build_grouped_search_payload(&documents, true, false, 0);

        assert_eq!(structured["path_prefix_truncated"], serde_json::json!(true));
        assert!(
            text.contains("path_prefix matched more candidates"),
            "text must render the truncation note; got: {text}"
        );
    }

    #[test]
    fn build_grouped_search_payload_offset_truncated_note_and_flag() {
        // #224: grouped granularity's mirror of the chunk-payload test above.
        // `offset` is non-zero, so the note still tells the caller to narrow
        // the query or lower offset.
        let documents = vec![payload_grouped_document("notes/a.md", "A", 0.9)];
        let (text, structured) = build_grouped_search_payload(&documents, false, true, 5);

        assert_eq!(structured["offset_truncated"], serde_json::json!(true));
        assert!(
            text.contains("offset + limit reached past"),
            "text must render the offset-truncation note; got: {text}"
        );
        assert!(text.contains("Narrow the query or lower offset"));
    }

    /// #240: grouped granularity's mirror of the chunk-payload zero-offset
    /// test above — grouped search never runs a reranker (there is no
    /// `reranking.candidate_limit` for it to name), so the note must point at
    /// `limit` and the fixed ceiling, not tell the caller to lower an offset
    /// that is already 0.
    #[test]
    fn build_grouped_search_payload_offset_truncated_at_offset_zero_names_the_right_knob() {
        let documents = vec![payload_grouped_document("notes/a.md", "A", 0.9)];
        let (text, _) = build_grouped_search_payload(&documents, false, true, 0);

        assert!(
            !text.contains("lower offset"),
            "offset is already 0 — must not tell the caller to lower it: {text}"
        );
        assert!(
            text.contains("Lower limit"),
            "must point at limit instead: {text}"
        );
    }
}
