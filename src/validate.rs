use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use gray_matter::engine::YAML;
use gray_matter::{Matter, Pod};
use serde_json::Value;

use crate::config::{FrontmatterConfig, ValidationConfig};

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub file_path: String,
    pub valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
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

pub fn validate_file(
    path: &Path,
    config: &FrontmatterConfig,
    validation: &ValidationConfig,
) -> anyhow::Result<(ValidationResult, Option<ValidatedFile>)> {
    let file_path = path.to_string_lossy().to_string();
    let mut errors: Vec<String> = Vec::new();
    let warnings: Vec<String> = Vec::new();

    let content = std::fs::read_to_string(path)?;

    let matter = Matter::<YAML>::new();
    let parsed = matter.parse(&content);

    // Parse frontmatter fields
    let mut frontmatter: HashMap<String, Value> = HashMap::new();

    if let Some(data) = parsed.data {
        if let Pod::Hash(map) = data {
            for (k, v) in map {
                frontmatter.insert(k, pod_to_value(v));
            }
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
            errors.push(format!("Missing required frontmatter field: '{}'", field));
        }
    }

    // Run lint command if configured
    if let Some(lint_cmd) = &validation.lint_command {
        let cmd_str = lint_cmd.replace("{file}", &file_path);
        let parts: Vec<&str> = cmd_str.split_whitespace().collect();
        if let Some((program, args)) = parts.split_first() {
            let output = Command::new(program).args(args).output();
            match output {
                Ok(out) if !out.status.success() => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let msg = if !stderr.is_empty() {
                        stderr.trim().to_string()
                    } else {
                        stdout.trim().to_string()
                    };
                    errors.push(format!("Lint command failed: {}", msg));
                }
                Err(e) => {
                    errors.push(format!("Failed to run lint command: {}", e));
                }
                _ => {}
            }
        }
    }

    let valid = errors.is_empty();

    let result = ValidationResult {
        file_path: file_path.clone(),
        valid,
        errors,
        warnings,
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

pub fn validate_all(
    _data_path: &Path,
    files: &[PathBuf],
    config: &FrontmatterConfig,
    validation: &ValidationConfig,
) -> Vec<(ValidationResult, Option<ValidatedFile>)> {
    files
        .iter()
        .filter_map(|file| match validate_file(file, config, validation) {
            Ok(pair) => Some(pair),
            Err(e) => {
                let result = ValidationResult {
                    file_path: file.to_string_lossy().to_string(),
                    valid: false,
                    errors: vec![format!("Failed to read or parse file: {}", e)],
                    warnings: Vec::new(),
                };
                Some((result, None))
            }
        })
        .collect()
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

    #[test]
    fn valid_frontmatter() {
        let content = "---\ntitle: Test\ntype: guide\n---\n# Hello\nBody text";
        let f = write_temp(content);
        let (result, validated) =
            validate_file(f.path(), &default_fm_config(), &default_val_config()).unwrap();
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

    #[test]
    fn missing_required_field() {
        let content = "---\ntitle: Test\n---\nBody";
        let f = write_temp(content);
        let (result, validated) =
            validate_file(f.path(), &default_fm_config(), &default_val_config()).unwrap();
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("type")));
        assert!(validated.is_none());
    }

    #[test]
    fn no_frontmatter() {
        let content = "# Just markdown\nNo frontmatter here";
        let f = write_temp(content);
        let (result, _) =
            validate_file(f.path(), &default_fm_config(), &default_val_config()).unwrap();
        assert!(!result.valid);
        assert_eq!(result.errors.len(), 2); // missing title and type
    }

    #[test]
    fn defaults_applied() {
        let content = "---\ntitle: Test\ntype: guide\n---\nBody";
        let f = write_temp(content);
        let (_, validated) =
            validate_file(f.path(), &default_fm_config(), &default_val_config()).unwrap();
        let vf = validated.unwrap();
        assert_eq!(
            vf.frontmatter.get("status").unwrap().as_str().unwrap(),
            "active"
        );
    }

    #[test]
    fn validate_all_mixed() {
        let good = write_temp("---\ntitle: Good\ntype: guide\n---\nBody");
        let bad = write_temp("---\ntitle: Bad\n---\nMissing type");
        let files = vec![good.path().to_path_buf(), bad.path().to_path_buf()];
        let results = validate_all(
            Path::new("/tmp"),
            &files,
            &default_fm_config(),
            &default_val_config(),
        );
        assert_eq!(results.len(), 2);
        assert!(results[0].0.valid);
        assert!(!results[1].0.valid);
    }
}
