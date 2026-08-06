use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::process::Command;
use tokio::task::JoinSet;

use gray_matter::engine::YAML;
use gray_matter::{Matter, Pod};
use serde_json::Value;

use crate::config::ValidationConfig;
use crate::schema::{self, ResolvedSchema, SchemaCache};

/// A structured, machine-readable description of a single validation failure.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FieldError {
    /// The frontmatter field name (or `"<lint>"` for external lint failures).
    pub field: String,
    /// Rule that was violated: `"required"` | `"allowed_value"` | `"lint"`.
    pub rule: String,
    /// Human-readable message (identical text to what was previously in `errors`).
    pub message: String,
    /// The offending value, if any.
    pub got: Option<String>,
    /// The set of allowed values, populated for `rule == "allowed_value"`.
    pub expected: Option<Vec<String>>,
    /// Which schema file declared the violated rule — a `.kb-schema.yaml` path
    /// relative to the KB root, or `"config.yaml"` for the legacy global config.
    /// `None` for non-schema rules (`"lint"`, `"io"`). The cascade means the rule
    /// may come from an ancestor directory rather than the document's own, so a
    /// caller cannot fix what it cannot locate — this is also embedded in
    /// `message`, matching the same field `get_schema` reports as `declared_in`.
    pub schema_origin: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub file_path: String,
    pub valid: bool,
    /// Flat error strings — kept for backward compatibility.
    /// Derived from `field_errors`; always mirrors `field_errors[*].message`.
    pub errors: Vec<String>,
    /// Structured per-field errors — the machine-readable form.
    pub field_errors: Vec<FieldError>,
}

#[derive(Debug, Clone)]
pub struct ValidatedFile {
    pub frontmatter: HashMap<String, Value>,
    pub body: String,
}

fn pod_to_value(pod: Pod) -> Value {
    match pod {
        Pod::String(s) => Value::String(s),
        Pod::Integer(i) => Value::Number(i.into()),
        Pod::Float(f) => {
            if let Some(n) = serde_json::Number::from_f64(f) {
                Value::Number(n)
            } else {
                Value::Null
            }
        }
        Pod::Boolean(b) => Value::Bool(b),
        Pod::Array(arr) => Value::Array(arr.into_iter().map(pod_to_value).collect()),
        Pod::Hash(map) => {
            let obj = map.into_iter().map(|(k, v)| (k, pod_to_value(v))).collect();
            Value::Object(obj)
        }
        Pod::Null => Value::Null,
    }
}

pub async fn validate_file(
    path: &Path,
    schema: &ResolvedSchema,
    validation: &ValidationConfig,
) -> anyhow::Result<(ValidationResult, Option<ValidatedFile>)> {
    let content = tokio::fs::read_to_string(path).await?;
    validate_content(path, &content, schema, validation).await
}

/// Which schema file declared `field`'s rule, formatted both as a bracketed message
/// suffix and as the bare (sanitized) origin string for `FieldError::schema_origin`.
///
/// The cascade means the rule enforced against a document in `a/b/c.md` may come from
/// `a/.kb-schema.yaml`, `a/b/.kb-schema.yaml`, or the implicit root schema — naming it
/// is the difference between a caller that can go fix the right file and one that has
/// to guess. Sanitized the same way `get_schema`'s `declared_in` is: schema paths
/// originate in directory names from a synced git repo and are reflected straight into
/// an error a caller reads, so they get the same control-character/length treatment as
/// any other knowledge-base-controlled string reaching an agent.
fn origin_suffix(schema: &ResolvedSchema, field: &str) -> (Option<String>, String) {
    match schema.origin.get(field) {
        Some(origin) => {
            let clean = crate::server::sanitize_facet_value(origin);
            let suffix = format!(" [declared in {clean}]");
            (Some(clean), suffix)
        }
        None => (None, String::new()),
    }
}

/// Render a value compactly for an error message, without dumping a whole document.
fn compact_value(value: &Value) -> String {
    const MAX: usize = 80;
    let rendered = match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    if rendered.chars().count() > MAX {
        let truncated: String = rendered.chars().take(MAX).collect();
        format!("{truncated}…")
    } else {
        rendered
    }
}

/// Parse frontmatter and body, applying schema defaults, without validating.
///
/// Split out from [`validate_content`] because that function deliberately returns no
/// frontmatter when validation fails — metadata backfill needs the parsed fields for
/// documents that were indexed under whatever rules applied at the time, regardless of
/// whether they satisfy today's rules.
pub(crate) fn parse_frontmatter(
    content: &str,
    schema: &ResolvedSchema,
) -> (HashMap<String, Value>, String) {
    let (mut frontmatter, body) = parse_frontmatter_raw(content);
    apply_defaults(&mut frontmatter, schema);
    (frontmatter, body)
}

/// Parse frontmatter and body with no schema involvement at all.
pub(crate) fn parse_frontmatter_raw(content: &str) -> (HashMap<String, Value>, String) {
    let matter = Matter::<YAML>::new();
    let parsed = matter.parse(content);

    let mut frontmatter: HashMap<String, Value> = HashMap::new();
    if let Some(Pod::Hash(map)) = parsed.data {
        for (k, v) in map {
            frontmatter.insert(k, pod_to_value(v));
        }
    }

    (frontmatter, parsed.content)
}

/// Fill in schema-declared defaults for fields the document omitted.
pub(crate) fn apply_defaults(frontmatter: &mut HashMap<String, Value>, schema: &ResolvedSchema) {
    for (path, def) in &schema.fields {
        let Some(default) = &def.default else {
            continue;
        };
        if schema::get_by_dotpath(frontmatter, path).is_none() {
            schema::set_by_dotpath(frontmatter, path, default.clone());
        }
    }
}

/// Check parsed frontmatter against a resolved schema.
///
/// Separate from [`validate_content`] so a schema change can be dry-run against the
/// frontmatter already stored in the metadata index, without re-reading any documents.
pub fn validate_frontmatter(
    frontmatter: &HashMap<String, Value>,
    schema: &ResolvedSchema,
) -> Vec<FieldError> {
    let mut field_errors: Vec<FieldError> = Vec::new();

    // One pass over the resolved schema, dispatching per field definition. Field paths
    // may be nested (`planning.prep_minutes`), so lookups go through dot-path access.
    for (field, def) in &schema.fields {
        let value = schema::get_by_dotpath(frontmatter, field);

        // An empty string satisfies "present" for lookup purposes but not for the
        // required check — matching the pre-cascade behavior exactly.
        let empty_string = value.and_then(Value::as_str) == Some("");
        let absent = matches!(value, None | Some(Value::Null));

        if absent || empty_string {
            if def.required {
                let (schema_origin, suffix) = origin_suffix(schema, field);
                field_errors.push(FieldError {
                    field: field.clone(),
                    rule: "required".into(),
                    message: format!("Missing required frontmatter field: '{}'{}", field, suffix),
                    got: None,
                    expected: None,
                    schema_origin,
                });
            }
            if absent {
                // A genuinely absent field has nothing to type- or value-check.
                continue;
            }
            // An empty string that IS present still gets checked below: the old
            // validator ran it against the allowed set and reported a value error,
            // and skipping that here would silently accept documents it rejected.
        }

        let Some(value) = value else {
            continue;
        };

        if let Some(ty) = def.ty {
            if let Err(reason) = schema::check_type(ty, value) {
                let (schema_origin, suffix) = origin_suffix(schema, field);
                field_errors.push(FieldError {
                    field: field.clone(),
                    rule: "type_mismatch".into(),
                    message: format!("field '{}': {}{}", field, reason, suffix),
                    got: Some(compact_value(value)),
                    expected: None,
                    schema_origin,
                });
                // A wrong-typed value cannot be meaningfully checked against a value
                // set or an object's key list.
                continue;
            }

            if ty == schema::FieldType::Object
                && !def.open
                && let Some(map) = value.as_object()
            {
                // The rule lives on the parent object field (e.g. `planning`), not the
                // undeclared child key (`planning.typo_key`) — look origin up by that.
                let (schema_origin, suffix) = origin_suffix(schema, field);
                for key in map.keys() {
                    let child = format!("{}.{}", field, key);
                    if !schema.fields.contains_key(&child) {
                        field_errors.push(FieldError {
                            field: child.clone(),
                            rule: "closed_object".into(),
                            message: format!(
                                "field '{}' is not declared, and '{}' does not allow undeclared keys{}",
                                child, field, suffix
                            ),
                            got: None,
                            expected: None,
                            schema_origin: schema_origin.clone(),
                        });
                    }
                }
            }
        }

        // A field whose type was never declared keeps the pre-cascade value-checking
        // semantics: only strings and string array elements are checked, everything
        // else is exempt. Explicitly declared enum/list fields get the strict check.
        let value_check = match def.ty {
            None => schema::check_values_lenient(value, def.values.as_deref()),
            Some(_) => def
                .values
                .as_ref()
                .map_or(Ok(()), |permitted| schema::check_values(value, permitted)),
        };

        if let Some(permitted) = &def.values
            && let Err(reason) = value_check
        {
            let (schema_origin, suffix) = origin_suffix(schema, field);
            field_errors.push(FieldError {
                field: field.clone(),
                rule: "allowed_value".into(),
                message: format!("field '{}': {}{}", field, reason, suffix),
                got: Some(compact_value(value)),
                expected: Some(permitted.clone()),
                schema_origin,
            });
        }
    }

    field_errors
}

pub async fn validate_content(
    path: &Path,
    content: &str,
    schema: &ResolvedSchema,
    validation: &ValidationConfig,
) -> anyhow::Result<(ValidationResult, Option<ValidatedFile>)> {
    let file_path = path.to_string_lossy().to_string();
    let mut field_errors: Vec<FieldError> = Vec::new();

    let (frontmatter, body) = parse_frontmatter(content, schema);

    if !validation.enabled {
        let result = ValidationResult {
            file_path: file_path.clone(),
            valid: true,
            errors: vec![],
            field_errors: vec![],
        };
        return Ok((result, Some(ValidatedFile { frontmatter, body })));
    }

    field_errors.extend(validate_frontmatter(&frontmatter, schema));

    // Run lint command if configured
    if let Some(lint_cmd) = &validation.lint_command
        && let Some((program, args)) = lint_cmd.split_first()
    {
        let output = Command::new(program).args(args).arg(path).output().await;
        match output {
            Ok(out) if !out.status.success() => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                let stdout = String::from_utf8_lossy(&out.stdout);
                let raw = if !stderr.is_empty() {
                    stderr.trim().to_string()
                } else {
                    stdout.trim().to_string()
                };
                let msg = format!("Lint command failed: {}", raw);
                field_errors.push(FieldError {
                    field: "<lint>".into(),
                    rule: "lint".into(),
                    message: msg,
                    got: None,
                    expected: None,
                    // Not a schema rule — nothing in .kb-schema.yaml to point at.
                    schema_origin: None,
                });
            }
            Err(e) => {
                field_errors.push(FieldError {
                    field: "<lint>".into(),
                    rule: "lint".into(),
                    message: format!("Failed to run lint command: {}", e),
                    got: None,
                    expected: None,
                    schema_origin: None,
                });
            }
            _ => {}
        }
    }

    // Derive the flat `errors` vec from field_errors for backward-compat
    let errors: Vec<String> = field_errors.iter().map(|e| e.message.clone()).collect();
    let valid = field_errors.is_empty();

    let result = ValidationResult {
        file_path: file_path.clone(),
        valid,
        errors,
        field_errors,
    };

    let validated_file = if valid {
        Some(ValidatedFile { frontmatter, body })
    } else {
        None
    };

    Ok((result, validated_file))
}

pub async fn validate_all(
    files: &[PathBuf],
    data_path: &Path,
    schemas: &SchemaCache,
    validation: &ValidationConfig,
) -> Vec<(ValidationResult, Option<ValidatedFile>)> {
    let mut set = JoinSet::new();

    for (i, file) in files.iter().enumerate() {
        let file = file.clone();
        // Each file validates against the schema governing its own directory.
        let rel = file.strip_prefix(data_path).unwrap_or(&file).to_path_buf();
        let schema = schemas.resolve_for(&rel).clone();
        let validation = validation.clone();
        set.spawn(async move {
            let pair = match validate_file(&file, &schema, &validation).await {
                Ok(pair) => pair,
                Err(e) => {
                    let msg = format!("Failed to read or parse file: {}", e);
                    let fe = FieldError {
                        field: "<io>".into(),
                        rule: "io".into(),
                        message: msg.clone(),
                        got: None,
                        expected: None,
                        schema_origin: None,
                    };
                    let result = ValidationResult {
                        file_path: file.to_string_lossy().to_string(),
                        valid: false,
                        errors: vec![msg],
                        field_errors: vec![fe],
                    };
                    (result, None)
                }
            };
            (i, pair)
        });
    }

    let mut indexed: Vec<(usize, (ValidationResult, Option<ValidatedFile>))> =
        Vec::with_capacity(files.len());
    while let Some(res) = set.join_next().await {
        match res {
            Ok(item) => indexed.push(item),
            Err(e) => tracing::warn!("validation task panicked: {e}"),
        }
    }
    indexed.sort_by_key(|(i, _)| *i);
    indexed.into_iter().map(|(_, pair)| pair).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FrontmatterConfig;
    use crate::schema::SchemaFile;

    /// Route a legacy `FrontmatterConfig` fixture through the backward-compat adapter.
    ///
    /// Every pre-cascade test below runs against the adapted schema rather than the
    /// config directly, which makes the whole existing suite a golden master proving
    /// a deployment with no `.kb-schema.yaml` behaves exactly as it did.
    fn as_schema(config: &FrontmatterConfig) -> ResolvedSchema {
        ResolvedSchema::from_config(config)
    }
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn default_fm_config() -> FrontmatterConfig {
        FrontmatterConfig {
            required: vec!["title".into(), "type".into()],
            indexed_fields: vec![],
            defaults: HashMap::from([("status".into(), "active".into())]),
            allowed: HashMap::new(),
        }
    }

    fn default_val_config() -> ValidationConfig {
        ValidationConfig {
            enabled: true,
            strict: false,
            lint_command: None,
        }
    }

    fn write_temp(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[tokio::test]
    async fn valid_frontmatter() {
        let content = "---\ntitle: Test\ntype: guide\n---\n# Hello\nBody text";
        let f = write_temp(content);
        let (result, validated) = validate_file(
            f.path(),
            &as_schema(&default_fm_config()),
            &default_val_config(),
        )
        .await
        .unwrap();
        assert!(result.valid);
        assert!(result.errors.is_empty());
        let vf = validated.unwrap();
        assert_eq!(
            vf.frontmatter.get("title").unwrap().as_str().unwrap(),
            "Test"
        );
        assert_eq!(
            vf.frontmatter.get("status").unwrap().as_str().unwrap(),
            "active"
        );
        assert!(vf.body.contains("Hello"));
    }

    #[tokio::test]
    async fn missing_required_field() {
        let content = "---\ntitle: Test\n---\nBody";
        let f = write_temp(content);
        let (result, validated) = validate_file(
            f.path(),
            &as_schema(&default_fm_config()),
            &default_val_config(),
        )
        .await
        .unwrap();
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("type")));
        assert!(validated.is_none());
    }

    #[tokio::test]
    async fn no_frontmatter() {
        let content = "# Just markdown\nNo frontmatter here";
        let f = write_temp(content);
        let (result, _) = validate_file(
            f.path(),
            &as_schema(&default_fm_config()),
            &default_val_config(),
        )
        .await
        .unwrap();
        assert!(!result.valid);
        assert_eq!(result.errors.len(), 2); // missing title and type
    }

    #[tokio::test]
    async fn defaults_applied() {
        let content = "---\ntitle: Test\ntype: guide\n---\nBody";
        let f = write_temp(content);
        let (_, validated) = validate_file(
            f.path(),
            &as_schema(&default_fm_config()),
            &default_val_config(),
        )
        .await
        .unwrap();
        let vf = validated.unwrap();
        assert_eq!(
            vf.frontmatter.get("status").unwrap().as_str().unwrap(),
            "active"
        );
    }

    #[tokio::test]
    async fn validate_content_matches_validate_file() {
        let content = "---\ntitle: Test\ntype: guide\n---\n# Hello\nBody text";
        let f = write_temp(content);
        let (file_result, file_validated) = validate_file(
            f.path(),
            &as_schema(&default_fm_config()),
            &default_val_config(),
        )
        .await
        .unwrap();
        let (content_result, content_validated) = validate_content(
            f.path(),
            content,
            &as_schema(&default_fm_config()),
            &default_val_config(),
        )
        .await
        .unwrap();
        assert_eq!(file_result.valid, content_result.valid);
        assert_eq!(file_result.errors, content_result.errors);
        let fv = file_validated.unwrap();
        let cv = content_validated.unwrap();
        assert_eq!(fv.frontmatter, cv.frontmatter);
        assert_eq!(fv.body, cv.body);
    }

    #[tokio::test]
    async fn validate_content_invalid_frontmatter() {
        let content = "---\ntitle: Test\n---\nBody";
        let f = write_temp(content);
        let (result, validated) = validate_content(
            f.path(),
            content,
            &as_schema(&default_fm_config()),
            &default_val_config(),
        )
        .await
        .unwrap();
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("type")));
        assert!(validated.is_none());
    }

    #[tokio::test]
    async fn validate_content_applies_defaults() {
        let content = "---\ntitle: Test\ntype: guide\n---\nBody";
        let f = write_temp(content);
        let (_, validated) = validate_content(
            f.path(),
            content,
            &as_schema(&default_fm_config()),
            &default_val_config(),
        )
        .await
        .unwrap();
        let vf = validated.unwrap();
        assert_eq!(
            vf.frontmatter.get("status").unwrap().as_str().unwrap(),
            "active"
        );
    }

    #[tokio::test]
    async fn nested_defaults_create_their_parent_object() {
        let schema: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  planning:\n    type: object\n    fields:\n      effort:\n        default: medium\n",
        )
        .unwrap();
        let resolved = ResolvedSchema::default().merged_with_for_test(&schema, "s");

        let (fm, _body) = parse_frontmatter("---\ntitle: X\n---\nBody", &resolved);

        assert_eq!(
            fm["planning"]["effort"],
            serde_json::json!("medium"),
            "a dot-path default must create the intermediate object"
        );
    }

    #[tokio::test]
    async fn closed_objects_reject_undeclared_keys() {
        let schema: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  planning:\n    type: object\n    open: false\n    fields:\n      effort:\n        type: text\n",
        )
        .unwrap();
        let resolved = ResolvedSchema::default().merged_with_for_test(&schema, "s");

        let (result, _) = validate_content(
            Path::new("d.md"),
            "---\nplanning:\n  effort: low\n  typo_key: x\n---\nBody",
            &resolved,
            &default_val_config(),
        )
        .await
        .unwrap();

        assert!(!result.valid);
        let err = result
            .field_errors
            .iter()
            .find(|e| e.rule == "closed_object")
            .expect("undeclared key inside a closed object must be reported");
        assert_eq!(err.field, "planning.typo_key");
    }

    #[tokio::test]
    async fn open_objects_permit_undeclared_keys() {
        let schema: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  planning:\n    type: object\n    fields:\n      effort:\n        type: text\n",
        )
        .unwrap();
        let resolved = ResolvedSchema::default().merged_with_for_test(&schema, "s");

        let (result, _) = validate_content(
            Path::new("d.md"),
            "---\nplanning:\n  effort: low\n  extra: x\n---\nBody",
            &resolved,
            &default_val_config(),
        )
        .await
        .unwrap();

        assert!(result.valid, "containers are open by default");
    }

    #[tokio::test]
    async fn an_empty_string_counts_as_missing_for_required() {
        let (result, _) = validate_content(
            Path::new("d.md"),
            "---\ntitle: \"\"\ntype: guide\n---\nBody",
            &as_schema(&default_fm_config()),
            &default_val_config(),
        )
        .await
        .unwrap();

        assert!(!result.valid);
        assert!(
            result
                .field_errors
                .iter()
                .any(|e| e.field == "title" && e.rule == "required")
        );
    }

    #[tokio::test]
    async fn an_empty_string_is_still_checked_against_a_closed_set() {
        // The pre-cascade validator reported an allowed_value error here; treating an
        // empty string purely as "missing" would silently accept it on a non-required
        // field, which is a behavior regression for config-only deployments.
        let (result, _) = validate_content(
            Path::new("d.md"),
            "---\ntitle: T\ntype: guide\nstatus: \"\"\n---\nBody",
            &as_schema(&fm_config_with_allowed()),
            &default_val_config(),
        )
        .await
        .unwrap();

        assert!(
            result
                .field_errors
                .iter()
                .any(|e| e.field == "status" && e.rule == "allowed_value"),
            "expected an allowed_value error, got: {:?}",
            result.field_errors
        );
    }

    #[tokio::test]
    async fn legacy_allowed_fields_still_exempt_non_strings() {
        // The old validator only checked strings and string array elements; numbers and
        // booleans passed untouched. A config-only deployment must not start failing.
        let (result, _) = validate_content(
            Path::new("d.md"),
            "---\ntitle: T\ntype: guide\nstatus: 42\n---\nBody",
            &as_schema(&fm_config_with_allowed()),
            &default_val_config(),
        )
        .await
        .unwrap();

        assert!(
            !result.field_errors.iter().any(|e| e.field == "status"),
            "a numeric value in an `allowed` field was exempt before and must stay exempt"
        );
    }

    #[tokio::test]
    async fn validate_all_reports_unreadable_files_rather_than_dropping_them() {
        // A file deleted between discovery and validation must surface as an explicit
        // failure, not silently vanish from the report.
        let good = write_temp("---\ntitle: Good\ntype: guide\n---\nBody");
        let files = vec![
            good.path().to_path_buf(),
            std::path::PathBuf::from("/nonexistent/gone.md"),
        ];
        let schemas = SchemaCache::from_config_only(&default_fm_config());

        let results = validate_all(&files, Path::new("/"), &schemas, &default_val_config()).await;

        assert_eq!(results.len(), 2, "every input yields a result");
        let failed = results
            .iter()
            .find(|(r, _)| !r.valid)
            .expect("the unreadable file must be reported");
        assert!(
            failed.0.field_errors.iter().any(|e| e.rule == "io"),
            "expected an io rule, got: {:?}",
            failed.0.field_errors
        );
    }

    #[tokio::test]
    async fn validate_all_mixed() {
        let good = write_temp("---\ntitle: Good\ntype: guide\n---\nBody");
        let bad = write_temp("---\ntitle: Bad\n---\nMissing type");
        let files = vec![good.path().to_path_buf(), bad.path().to_path_buf()];
        let schemas = SchemaCache::from_config_only(&default_fm_config());
        let results = validate_all(&files, Path::new("/"), &schemas, &default_val_config()).await;
        assert_eq!(results.len(), 2);
        assert!(results[0].0.valid);
        assert!(!results[1].0.valid);
    }

    #[tokio::test]
    async fn lint_command_passing_exits_zero() {
        let content = "---\ntitle: Test\ntype: guide\n---\nBody";
        let f = write_temp(content);
        let val_config = ValidationConfig {
            enabled: true,
            strict: false,
            lint_command: Some(vec!["true".into()]),
        };
        let (result, _) = validate_file(f.path(), &as_schema(&default_fm_config()), &val_config)
            .await
            .unwrap();
        assert!(result.valid);
        assert!(result.errors.is_empty());
    }

    #[tokio::test]
    async fn lint_command_failing_adds_error() {
        let content = "---\ntitle: Test\ntype: guide\n---\nBody";
        let f = write_temp(content);
        let val_config = ValidationConfig {
            enabled: true,
            strict: false,
            lint_command: Some(vec!["false".into()]),
        };
        let (result, validated) =
            validate_file(f.path(), &as_schema(&default_fm_config()), &val_config)
                .await
                .unwrap();
        assert!(!result.valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("Lint command failed"))
        );
        assert!(validated.is_none());
    }

    #[tokio::test]
    async fn lint_command_receives_path_as_argument() {
        // Use `sh -c 'test -f "$1"' -- ` to verify the path was passed as a
        // distinct argument and actually points to an existing file.
        let content = "---\ntitle: Test\ntype: guide\n---\nBody";
        let f = write_temp(content);
        let val_config = ValidationConfig {
            enabled: true,
            strict: false,
            lint_command: Some(vec![
                "sh".into(),
                "-c".into(),
                "test -f \"$1\"".into(),
                "--".into(),
            ]),
        };
        let (result, _) = validate_file(f.path(), &as_schema(&default_fm_config()), &val_config)
            .await
            .unwrap();
        assert!(result.valid, "errors: {:?}", result.errors);
    }

    #[tokio::test]
    async fn lint_command_empty_vec_is_noop() {
        let content = "---\ntitle: Test\ntype: guide\n---\nBody";
        let f = write_temp(content);
        let val_config = ValidationConfig {
            enabled: true,
            strict: false,
            lint_command: Some(vec![]),
        };
        let (result, _) = validate_file(f.path(), &as_schema(&default_fm_config()), &val_config)
            .await
            .unwrap();
        assert!(result.valid);
    }

    // -----------------------------------------------------------------------
    // Allowed-value (enum) enforcement tests
    // -----------------------------------------------------------------------

    fn fm_config_with_allowed() -> FrontmatterConfig {
        let mut allowed = HashMap::new();
        allowed.insert(
            "type".into(),
            vec![
                "guide".into(),
                "reference".into(),
                "research".into(),
                "config".into(),
                "troubleshooting".into(),
                "architecture".into(),
                "project".into(),
                "decision-record".into(),
                "migration".into(),
            ],
        );
        allowed.insert(
            "status".into(),
            vec!["active".into(), "draft".into(), "archived".into()],
        );
        FrontmatterConfig {
            required: vec!["title".into(), "type".into()],
            indexed_fields: vec![],
            defaults: HashMap::new(),
            allowed,
        }
    }

    #[tokio::test]
    async fn allowed_value_violation_produces_field_error() {
        // "type" present but value not in the allowed set
        let content = "---\ntitle: Test\ntype: invalid-type\n---\nBody";
        let f = write_temp(content);
        let (result, validated) = validate_file(
            f.path(),
            &as_schema(&fm_config_with_allowed()),
            &default_val_config(),
        )
        .await
        .unwrap();
        assert!(!result.valid);
        assert!(validated.is_none());

        // There must be exactly one field_error for the "type" field
        let fe = result
            .field_errors
            .iter()
            .find(|e| e.field == "type" && e.rule == "allowed_value")
            .expect("expected an allowed_value FieldError for 'type'");
        assert_eq!(fe.got.as_deref(), Some("invalid-type"));
        assert!(fe.expected.as_ref().unwrap().contains(&"guide".to_string()));

        // backward-compat: errors must mirror field_errors messages
        assert!(result.errors.iter().any(|e| e.contains("invalid-type")));
    }

    #[tokio::test]
    async fn allowed_value_valid_passes() {
        let content = "---\ntitle: Test\ntype: guide\n---\nBody";
        let f = write_temp(content);
        let (result, validated) = validate_file(
            f.path(),
            &as_schema(&fm_config_with_allowed()),
            &default_val_config(),
        )
        .await
        .unwrap();
        assert!(result.valid, "errors: {:?}", result.errors);
        assert!(validated.is_some());
        assert!(result.field_errors.is_empty());
    }

    #[tokio::test]
    async fn allowed_absent_field_does_not_error() {
        // "status" is in the allowed map but not required and not present in frontmatter.
        // No defaults configured in this fixture either, so it is simply absent.
        let content = "---\ntitle: Test\ntype: guide\n---\nBody";
        let f = write_temp(content);
        let (result, _) = validate_file(
            f.path(),
            &as_schema(&fm_config_with_allowed()),
            &default_val_config(),
        )
        .await
        .unwrap();
        assert!(
            result.field_errors.iter().all(|e| e.field != "status"),
            "absent allowed-field 'status' should not produce an error"
        );
        assert!(result.valid, "errors: {:?}", result.errors);
    }

    #[tokio::test]
    async fn errors_mirrors_field_errors_messages_backward_compat() {
        // A required-field violation should appear in both errors and field_errors
        let content = "---\ntitle: Test\n---\nBody"; // missing 'type'
        let f = write_temp(content);
        let (result, _) = validate_file(
            f.path(),
            &as_schema(&fm_config_with_allowed()),
            &default_val_config(),
        )
        .await
        .unwrap();
        assert!(!result.valid);

        // Every field_error message must appear verbatim in errors
        for fe in &result.field_errors {
            assert!(
                result.errors.contains(&fe.message),
                "errors should contain field_error message: {}",
                fe.message
            );
        }
        // errors and field_errors must have the same length
        assert_eq!(result.errors.len(), result.field_errors.len());
    }

    // -----------------------------------------------------------------------
    // Schema-origin provenance (issue #76): a caller must be able to locate the
    // schema file that imposed a violated rule, not just what the rule was.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn required_field_error_names_its_declaring_schema() {
        // The legacy config-only cascade root is a synthetic schema whose origin is
        // always "config.yaml" (see `ResolvedSchema::from_config`).
        let content = "---\ntitle: Test\n---\nBody"; // missing 'type'
        let f = write_temp(content);
        let (result, _) = validate_file(
            f.path(),
            &as_schema(&default_fm_config()),
            &default_val_config(),
        )
        .await
        .unwrap();

        let fe = result
            .field_errors
            .iter()
            .find(|e| e.field == "type" && e.rule == "required")
            .expect("expected a required-field error for 'type'");
        assert_eq!(fe.schema_origin.as_deref(), Some("config.yaml"));
        assert!(
            fe.message.contains("[declared in config.yaml]"),
            "expected the message to name the declaring schema, got: {}",
            fe.message
        );
    }

    #[tokio::test]
    async fn allowed_value_error_names_the_cascaded_schema_file_not_the_root() {
        // A field redeclared by a directory-level .kb-schema.yaml must report THAT
        // file, not the root the cascade started from — the whole point of provenance
        // is that a deeper scope can override a shallower one's rule.
        let schema: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  status:\n    type: enum\n    values: [archived]\n")
                .unwrap();
        let resolved =
            ResolvedSchema::default().merged_with_for_test(&schema, "food/.kb-schema.yaml");

        let (result, _) = validate_content(
            Path::new("food/d.md"),
            "---\nstatus: draft\n---\nBody",
            &resolved,
            &default_val_config(),
        )
        .await
        .unwrap();

        let fe = result
            .field_errors
            .iter()
            .find(|e| e.field == "status" && e.rule == "allowed_value")
            .expect("expected an allowed_value error for 'status'");
        assert_eq!(fe.schema_origin.as_deref(), Some("food/.kb-schema.yaml"));
        assert!(
            fe.message.contains("[declared in food/.kb-schema.yaml]"),
            "got: {}",
            fe.message
        );
        // The permitted set must still be listed too — origin is additive, not a
        // replacement for the existing "allowed: ..." detail.
        assert!(
            fe.expected
                .as_ref()
                .unwrap()
                .contains(&"archived".to_string())
        );
    }

    #[tokio::test]
    async fn closed_object_error_is_attributed_to_the_parent_fields_origin() {
        // The undeclared key is `planning.typo_key`, which never appears in
        // `schema.origin` itself — the rule ("no undeclared keys") belongs to the
        // parent object field `planning`, so provenance must be looked up under that
        // key, not the synthesized child path.
        let schema: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  planning:\n    type: object\n    open: false\n    fields:\n      effort:\n        type: text\n",
        )
        .unwrap();
        let resolved = ResolvedSchema::default().merged_with_for_test(&schema, "s");

        let (result, _) = validate_content(
            Path::new("d.md"),
            "---\nplanning:\n  effort: low\n  typo_key: x\n---\nBody",
            &resolved,
            &default_val_config(),
        )
        .await
        .unwrap();

        let fe = result
            .field_errors
            .iter()
            .find(|e| e.rule == "closed_object")
            .expect("expected a closed_object error");
        assert_eq!(fe.field, "planning.typo_key");
        assert_eq!(
            fe.schema_origin.as_deref(),
            Some("s"),
            "origin must come from the parent field 'planning', not the child path"
        );
    }

    #[tokio::test]
    async fn lint_and_io_errors_have_no_schema_origin() {
        // Neither failure mode comes from a frontmatter rule, so there is no schema
        // file to point at — `schema_origin` must stay `None` rather than fabricate one.
        let content = "---\ntitle: Test\ntype: guide\n---\nBody";
        let f = write_temp(content);
        let val_config = ValidationConfig {
            enabled: true,
            strict: false,
            lint_command: Some(vec!["false".into()]),
        };
        let (result, _) = validate_file(f.path(), &as_schema(&default_fm_config()), &val_config)
            .await
            .unwrap();
        let fe = result
            .field_errors
            .iter()
            .find(|e| e.rule == "lint")
            .expect("expected a lint error");
        assert!(fe.schema_origin.is_none());
    }
}
