//! Projection of document frontmatter into flat, filterable rows.
//!
//! `documents.frontmatter` stores the faithful JSON record; this module derives the
//! `document_fields` inverted index from it. Nested objects flatten to dot-paths
//! (`planning.prep_minutes`), so a filter never needs to know a document's shape.
//!
//! The projection is rebuildable — see `StateDb::reproject_all_fields`. Changing the
//! rules here requires no re-read of markdown and no re-embedding.

use serde_json::Value;
use std::collections::HashMap;
use tracing::{debug, warn};

/// How deep nested objects are flattened before giving up. Frontmatter this deep is
/// almost certainly structured data that belongs in the document body, not a filter key.
const MAX_FLATTEN_DEPTH: usize = 4;

/// Upper bound on projection rows from a single document.
const MAX_PROJECTED_ROWS: usize = 2_000;

/// Fields with a dedicated column on `documents`. They are prose, not filter-shaped,
/// and projecting them would bloat the index with never-matched rows.
const PROMOTED_FIELDS: [&str; 2] = ["title", "description"];

/// One row of the `document_fields` projection.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldRow {
    /// Dot-path field name, e.g. `planning.prep_minutes`.
    pub field: String,
    /// Canonical text form, used for equality and any-of/all-of matching.
    pub value_text: String,
    /// Numeric form when the value is coercible, enabling range queries.
    pub value_num: Option<f64>,
}

/// Canonical text form of a scalar, used identically when writing rows and when
/// comparing a filter value. Keeping both sides on this one function is what makes
/// `{"planning.needs_recipe": false}` match a stored `needs_recipe: false`.
///
/// Returns `None` for values that are not scalars.
pub fn canonical_text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Bool(b) => Some(b.to_string()),
        // `Number::to_string` preserves representation rather than value: a YAML
        // `45.0` and a JSON `45` are both the number forty-five but stringify as
        // "45.0" and "45". Since the write side and the filter side both come through
        // here, a mismatch would silently drop documents from equality matches — so
        // integral values get one canonical spelling regardless of how they arrived.
        Value::Number(n) => Some(canonical_number(n)),
        _ => None,
    }
}

/// One spelling per numeric value, so equality matching cannot depend on whether the
/// author wrote `45` or `45.0`.
fn canonical_number(n: &serde_json::Number) -> String {
    if let Some(i) = n.as_i64() {
        return i.to_string();
    }
    if let Some(u) = n.as_u64() {
        return u.to_string();
    }
    match n.as_f64() {
        Some(f) if f.fract() != 0.0 || !f.is_finite() => f.to_string(),
        // Integral floats render as integers so `45.0` and `45` agree. The u64 arm
        // matters because a value above i64::MAX arriving as an integer takes the
        // as_u64 path above; without this it would arrive as a float and render in
        // exponential form, giving one value two spellings.
        Some(f) if f >= 0.0 && f <= u64::MAX as f64 => (f as u64).to_string(),
        // Both integer casts must be bounded on BOTH sides: Rust saturates float→int
        // casts, so an unbounded arm would collapse every value beyond the range onto
        // the same integer — giving distinct values one spelling, which is the exact
        // inverse of the bug this function exists to prevent.
        Some(f) if f < 0.0 && f >= i64::MIN as f64 => (f as i64).to_string(),
        Some(f) => f.to_string(),
        None => n.to_string(),
    }
}

/// Numeric form of a scalar, or `None` when it cannot participate in range queries.
///
/// Booleans map to 1.0/0.0 so they are usable by both equality and range operators
/// without needing a third column.
pub fn numeric_value(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn leaf_row(field: &str, value: &Value) -> Option<FieldRow> {
    canonical_text(value).map(|value_text| FieldRow {
        field: field.to_string(),
        value_text,
        value_num: numeric_value(value),
    })
}

/// Flatten a document's frontmatter into `document_fields` rows.
///
/// - Nested objects become dot-paths, recursively, capped at [`MAX_FLATTEN_DEPTH`].
///   The container key itself never gets a row — whole objects are not filterable.
/// - Arrays of scalars become one row per element sharing a field name, which is what
///   makes any-of and all-of matching on `tags` possible.
/// - Arrays holding objects or arrays are skipped; they have no useful flat projection.
/// - `null` produces no row, so "absent" and "explicitly null" filter identically.
pub fn flatten_frontmatter(frontmatter: &HashMap<String, Value>) -> Vec<FieldRow> {
    let mut rows = Vec::new();
    // Sorted for deterministic output — makes tests stable and diffs readable.
    let mut keys: Vec<&String> = frontmatter.keys().collect();
    keys.sort();
    for key in keys {
        if PROMOTED_FIELDS.contains(&key.as_str()) {
            continue;
        }
        flatten_value(key, &frontmatter[key], 1, &mut rows);
        if rows.len() >= MAX_PROJECTED_ROWS {
            warn!(
                rows = rows.len(),
                "frontmatter projection hit the row cap; later fields are not filterable"
            );
            rows.truncate(MAX_PROJECTED_ROWS);
            break;
        }
    }
    rows
}

fn flatten_value(path: &str, value: &Value, depth: usize, out: &mut Vec<FieldRow>) {
    match value {
        Value::Null => {}
        Value::String(_) | Value::Number(_) | Value::Bool(_) => {
            if out.len() >= MAX_PROJECTED_ROWS {
                return;
            }
            if let Some(row) = leaf_row(path, value) {
                out.push(row);
            }
        }
        Value::Array(items) => {
            if items.iter().any(|v| v.is_array() || v.is_object()) {
                debug!(field = path, "skipping array with non-scalar elements");
                return;
            }
            for item in items {
                // The cap is checked here, inside the expansion, so the WORK is bounded
                // and not just the result. Documents arriving via git bypass the MCP
                // write tools' size cap, so a single huge array would otherwise be
                // fully materialized before any outer check saw it.
                if out.len() >= MAX_PROJECTED_ROWS {
                    return;
                }
                if let Some(row) = leaf_row(path, item) {
                    out.push(row);
                }
            }
        }
        Value::Object(map) => {
            if out.len() >= MAX_PROJECTED_ROWS {
                return;
            }
            if depth >= MAX_FLATTEN_DEPTH {
                warn!(
                    field = path,
                    depth, "max flatten depth reached; deeper fields are not filterable"
                );
                return;
            }
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for key in keys {
                // Per key, not merely once before the loop: MAX_FLATTEN_DEPTH bounds
                // nesting depth but not key count, so a single very wide object would
                // otherwise be fully materialized before any outer check saw it.
                if out.len() >= MAX_PROJECTED_ROWS {
                    return;
                }
                flatten_value(&format!("{path}.{key}"), &map[key], depth + 1, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fm(value: Value) -> HashMap<String, Value> {
        match value {
            Value::Object(map) => map.into_iter().collect(),
            _ => panic!("frontmatter fixture must be an object"),
        }
    }

    fn find<'a>(rows: &'a [FieldRow], field: &str) -> Vec<&'a FieldRow> {
        rows.iter().filter(|r| r.field == field).collect()
    }

    /// The real shape from lifestyle/kitchen/recipes/stir-fry.md — the document that
    /// motivated nested-field support in the first place.
    fn stir_fry() -> HashMap<String, Value> {
        fm(json!({
            "title": "Stir Fry",
            "description": "Shopping-list-only entry for the weekly improvised stir fry.",
            "type": "reference",
            "domain": "lifestyle",
            "tags": ["recipe", "stir-fry", "stovetop", "wok"],
            "planning": {
                "servings": 4,
                "nights_covered": 3,
                "nights_confidence": "estimated",
                "tested": true,
                "rating": 5,
                "prep_minutes": 45,
                "cook_minutes": 15,
                "effort": "medium",
                "cuisine": "asian",
                "scalable": true,
                "needs_recipe": false
            }
        }))
    }

    #[test]
    fn promoted_fields_are_not_projected() {
        let rows = flatten_frontmatter(&stir_fry());
        assert!(find(&rows, "title").is_empty(), "title has its own column");
        assert!(
            find(&rows, "description").is_empty(),
            "description has its own column"
        );
    }

    #[test]
    fn scalars_project_with_canonical_text() {
        let rows = flatten_frontmatter(&stir_fry());
        let ty = find(&rows, "type");
        assert_eq!(ty.len(), 1);
        assert_eq!(ty[0].value_text, "reference");
        assert_eq!(ty[0].value_num, None);
    }

    #[test]
    fn arrays_produce_one_row_per_element() {
        let rows = flatten_frontmatter(&stir_fry());
        let tags = find(&rows, "tags");
        assert_eq!(tags.len(), 4, "one row per tag enables any-of/all-of");
        let values: Vec<&str> = tags.iter().map(|r| r.value_text.as_str()).collect();
        assert!(values.contains(&"recipe"));
        assert!(values.contains(&"wok"));
    }

    #[test]
    fn nested_objects_flatten_to_dot_paths() {
        let rows = flatten_frontmatter(&stir_fry());
        let prep = find(&rows, "planning.prep_minutes");
        assert_eq!(prep.len(), 1);
        assert_eq!(prep[0].value_text, "45");
        assert_eq!(prep[0].value_num, Some(45.0));
    }

    #[test]
    fn container_key_itself_gets_no_row() {
        let rows = flatten_frontmatter(&stir_fry());
        assert!(
            find(&rows, "planning").is_empty(),
            "whole objects are not filterable"
        );
    }

    #[test]
    fn booleans_project_to_text_and_number() {
        let rows = flatten_frontmatter(&stir_fry());
        let needs = find(&rows, "planning.needs_recipe");
        assert_eq!(needs.len(), 1);
        assert_eq!(needs[0].value_text, "false");
        assert_eq!(needs[0].value_num, Some(0.0));

        let tested = find(&rows, "planning.tested");
        assert_eq!(tested[0].value_text, "true");
        assert_eq!(tested[0].value_num, Some(1.0));
    }

    #[test]
    fn filter_value_canonicalizes_to_stored_text() {
        // The property that makes {"planning.needs_recipe": false} match the stored row.
        let rows = flatten_frontmatter(&stir_fry());
        let stored = find(&rows, "planning.needs_recipe")[0].value_text.clone();
        assert_eq!(canonical_text(&json!(false)), Some(stored));

        let prep = find(&rows, "planning.prep_minutes")[0].value_text.clone();
        assert_eq!(canonical_text(&json!(45)), Some(prep));
    }

    #[test]
    fn nulls_produce_no_row() {
        let rows = flatten_frontmatter(&fm(json!({ "status": null, "type": "guide" })));
        assert!(find(&rows, "status").is_empty());
        assert_eq!(find(&rows, "type").len(), 1);
    }

    #[test]
    fn arrays_of_objects_are_skipped() {
        let rows = flatten_frontmatter(&fm(json!({
            "steps": [{ "n": 1 }, { "n": 2 }],
            "type": "guide"
        })));
        assert!(find(&rows, "steps").is_empty());
        assert_eq!(find(&rows, "type").len(), 1, "siblings still project");
    }

    #[test]
    fn a_huge_array_is_bounded_during_expansion_not_after() {
        let many: Vec<Value> = (0..50_000).map(|i| json!(format!("t{i}"))).collect();
        let rows = flatten_frontmatter(&fm(json!({ "tags": many })));

        assert!(
            rows.len() <= MAX_PROJECTED_ROWS,
            "one field must not exceed the cap, got {}",
            rows.len()
        );
    }

    #[test]
    fn a_very_wide_object_is_bounded_during_expansion() {
        let mut wide = serde_json::Map::new();
        for i in 0..50_000 {
            wide.insert(format!("k{i}"), json!(i));
        }
        let rows = flatten_frontmatter(&fm(json!({ "custom": Value::Object(wide) })));

        assert!(
            rows.len() <= MAX_PROJECTED_ROWS,
            "breadth must be bounded too, got {}",
            rows.len()
        );
    }

    #[test]
    fn values_beyond_u64_keep_distinct_spellings() {
        // Rust saturates float->int casts, so an unbounded integer arm would give
        // 2e19 and 5e25 the same canonical text and make them match each other.
        let a = canonical_text(&json!(2.0e19f64)).unwrap();
        let b = canonical_text(&json!(5.0e25f64)).unwrap();
        assert_ne!(a, b, "distinct values must not collapse onto one spelling");

        let very_negative = canonical_text(&json!(-5.0e25f64)).unwrap();
        assert_ne!(very_negative, canonical_text(&json!(-2.0e19f64)).unwrap());
    }

    #[test]
    fn flattening_stops_at_max_depth() {
        let rows = flatten_frontmatter(&fm(json!({
            "a": { "b": { "c": { "d": { "e": "too deep" } } } }
        })));
        assert!(
            rows.iter().all(|r| r.field != "a.b.c.d.e"),
            "depth beyond the cap is not projected"
        );
    }

    #[test]
    fn floats_keep_numeric_form() {
        let rows = flatten_frontmatter(&fm(json!({ "score": 1.5 })));
        let score = find(&rows, "score");
        assert_eq!(score[0].value_num, Some(1.5));
        assert_eq!(score[0].value_text, "1.5");
    }

    #[test]
    fn integral_floats_and_integers_share_one_spelling() {
        // A document written `prep_minutes: 45.0` must be findable by a filter for 45.
        // Without normalization the stored text is "45.0", the filter canonicalizes to
        // "45", and the document silently vanishes from equality matches.
        assert_eq!(canonical_text(&json!(45.0)), canonical_text(&json!(45)));
        assert_eq!(canonical_text(&json!(45.0)).as_deref(), Some("45"));

        let stored = flatten_frontmatter(&fm(json!({ "planning": { "prep_minutes": 45.0 } })));
        assert_eq!(
            find(&stored, "planning.prep_minutes")[0].value_text,
            canonical_text(&json!(45)).unwrap()
        );
    }

    #[test]
    fn values_above_i64_agree_whether_written_as_int_or_float() {
        // u64::MAX-scale values take the as_u64 path when written as an integer; the
        // float path must produce the same spelling or the two never match.
        let big = 9_300_000_000_000_000_000u64;
        let as_int = serde_json::Number::from(big);
        let as_float = serde_json::Number::from_f64(big as f64).unwrap();

        assert_eq!(
            canonical_text(&Value::Number(as_int)),
            canonical_text(&Value::Number(as_float)),
            "one value must have exactly one canonical spelling"
        );
    }

    #[test]
    fn non_finite_floats_do_not_panic() {
        // serde_json cannot construct NaN/Infinity, but the guard must hold regardless.
        assert!(canonical_text(&json!(1e308)).is_some());
    }

    #[test]
    fn negative_and_large_numbers_canonicalize() {
        assert_eq!(canonical_text(&json!(-7)).as_deref(), Some("-7"));
        assert_eq!(canonical_text(&json!(-7.0)).as_deref(), Some("-7"));
        assert_eq!(
            canonical_text(&json!(i64::MAX)).as_deref(),
            Some("9223372036854775807")
        );
        // Non-integral stays as written.
        assert_eq!(canonical_text(&json!(-7.25)).as_deref(), Some("-7.25"));
    }

    #[test]
    fn output_is_deterministic() {
        // HashMap iteration order must not leak into the projection, or every
        // reindex would look like a change.
        let first = flatten_frontmatter(&stir_fry());
        for _ in 0..8 {
            assert_eq!(flatten_frontmatter(&stir_fry()), first);
        }
    }
}
