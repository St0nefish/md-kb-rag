use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::process::Command;
use tokio::task::JoinSet;

use gray_matter::engine::YAML;
use gray_matter::{Matter, Pod};
use serde_json::Value;

use crate::config::{FrontmatterConfig, ValidationConfig};

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
    config: &FrontmatterConfig,
    validation: &ValidationConfig,
) -> anyhow::Result<(ValidationResult, Option<ValidatedFile>)> {
    let content = tokio::fs::read_to_string(path).await?;
    validate_content(path, &content, config, validation).await
}

pub async fn validate_content(
    path: &Path,
    content: &str,
    config: &FrontmatterConfig,
    validation: &ValidationConfig,
) -> anyhow::Result<(ValidationResult, Option<ValidatedFile>)> {
    let file_path = path.to_string_lossy().to_string();
    let mut field_errors: Vec<FieldError> = Vec::new();

    let matter = Matter::<YAML>::new();
    let parsed = matter.parse(content);

    // Parse frontmatter fields
    let mut frontmatter: HashMap<String, Value> = HashMap::new();

    if let Some(Pod::Hash(map)) = parsed.data {
        for (k, v) in map {
            frontmatter.insert(k, pod_to_value(v));
        }
    }

    // Apply defaults for missing fields
    for (key, default_val) in &config.defaults {
        frontmatter
            .entry(key.clone())
            .or_insert_with(|| Value::String(default_val.clone()));
    }

    // Check required fields
    for field in &config.required {
        if !frontmatter.contains_key(field) {
            field_errors.push(FieldError {
                field: field.clone(),
                rule: "required".into(),
                message: format!("Missing required frontmatter field: '{}'", field),
                got: None,
                expected: None,
            });
        }
    }

    // Check allowed values for fields that are present
    for (field, allowed_values) in &config.allowed {
        if let Some(value) = frontmatter.get(field) {
            match value {
                Value::String(s) => {
                    if !allowed_values.contains(s) {
                        let expected_list = allowed_values.join(", ");
                        field_errors.push(FieldError {
                            field: field.clone(),
                            rule: "allowed_value".into(),
                            message: format!(
                                "field '{}' has value '{}', expected one of: {}",
                                field, s, expected_list
                            ),
                            got: Some(s.clone()),
                            expected: Some(allowed_values.clone()),
                        });
                    }
                }
                Value::Array(arr) => {
                    // For list-valued fields, check each element against the allowed set
                    for elem in arr {
                        if let Value::String(s) = elem
                            && !allowed_values.contains(s)
                        {
                            let expected_list = allowed_values.join(", ");
                            field_errors.push(FieldError {
                                field: field.clone(),
                                rule: "allowed_value".into(),
                                message: format!(
                                    "field '{}' has value '{}', expected one of: {}",
                                    field, s, expected_list
                                ),
                                got: Some(s.clone()),
                                expected: Some(allowed_values.clone()),
                            });
                        }
                    }
                }
                // Non-string, non-array values: skip enforcement (not a closed-set scenario)
                _ => {}
            }
        }
        // Field absent: no error — presence is governed by `required`, not `allowed`
    }

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
                });
            }
            Err(e) => {
                field_errors.push(FieldError {
                    field: "<lint>".into(),
                    rule: "lint".into(),
                    message: format!("Failed to run lint command: {}", e),
                    got: None,
                    expected: None,
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
        Some(ValidatedFile {
            frontmatter,
            body: parsed.content,
        })
    } else {
        None
    };

    Ok((result, validated_file))
}

pub async fn validate_all(
    files: &[PathBuf],
    config: &FrontmatterConfig,
    validation: &ValidationConfig,
) -> Vec<(ValidationResult, Option<ValidatedFile>)> {
    let mut set = JoinSet::new();

    for (i, file) in files.iter().enumerate() {
        let file = file.clone();
        let config = config.clone();
        let validation = validation.clone();
        set.spawn(async move {
            let pair = match validate_file(&file, &config, &validation).await {
                Ok(pair) => pair,
                Err(e) => {
                    let msg = format!("Failed to read or parse file: {}", e);
                    let fe = FieldError {
                        field: "<io>".into(),
                        rule: "lint".into(),
                        message: msg.clone(),
                        got: None,
                        expected: None,
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
        let (result, validated) =
            validate_file(f.path(), &default_fm_config(), &default_val_config())
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
        let (result, validated) =
            validate_file(f.path(), &default_fm_config(), &default_val_config())
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
        let (result, _) = validate_file(f.path(), &default_fm_config(), &default_val_config())
            .await
            .unwrap();
        assert!(!result.valid);
        assert_eq!(result.errors.len(), 2); // missing title and type
    }

    #[tokio::test]
    async fn defaults_applied() {
        let content = "---\ntitle: Test\ntype: guide\n---\nBody";
        let f = write_temp(content);
        let (_, validated) = validate_file(f.path(), &default_fm_config(), &default_val_config())
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
        let (file_result, file_validated) =
            validate_file(f.path(), &default_fm_config(), &default_val_config())
                .await
                .unwrap();
        let (content_result, content_validated) = validate_content(
            f.path(),
            content,
            &default_fm_config(),
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
            &default_fm_config(),
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
            &default_fm_config(),
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
    async fn validate_all_mixed() {
        let good = write_temp("---\ntitle: Good\ntype: guide\n---\nBody");
        let bad = write_temp("---\ntitle: Bad\n---\nMissing type");
        let files = vec![good.path().to_path_buf(), bad.path().to_path_buf()];
        let results = validate_all(&files, &default_fm_config(), &default_val_config()).await;
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
        let (result, _) = validate_file(f.path(), &default_fm_config(), &val_config)
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
        let (result, validated) = validate_file(f.path(), &default_fm_config(), &val_config)
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
        let (result, _) = validate_file(f.path(), &default_fm_config(), &val_config)
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
        let (result, _) = validate_file(f.path(), &default_fm_config(), &val_config)
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
        let (result, validated) =
            validate_file(f.path(), &fm_config_with_allowed(), &default_val_config())
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
        let (result, validated) =
            validate_file(f.path(), &fm_config_with_allowed(), &default_val_config())
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
        let (result, _) = validate_file(f.path(), &fm_config_with_allowed(), &default_val_config())
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
        let (result, _) = validate_file(f.path(), &fm_config_with_allowed(), &default_val_config())
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
}
