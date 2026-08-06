//! Directory-scoped frontmatter schemas.
//!
//! A `.kb-schema.yaml` file governs every document at or below its directory, the way
//! `CLAUDE.md` cascades. Deeper files refine shallower ones, so a `recipes/` folder can
//! require `planning.cook_minutes` without that field meaning anything elsewhere.
//!
//! Internally a schema is a **flat map keyed by dot-path** (`planning.prep_minutes`).
//! Nested YAML is accepted as authoring sugar and flattened at parse time: merging
//! nested structures raises questions a flat set simply does not have — whether
//! redefining a container replaces its children, what happens when one level calls a
//! path a leaf and another calls it a container — and none of those ambiguities buy
//! anything.
//!
//! Deployments with no schema files keep working: the global `frontmatter` block in
//! `config.yaml` becomes the implicit root schema via [`ResolvedSchema::from_config`].
//! This is a **deprecated fallback**, though — a schema describes the knowledge base's
//! own content rules, and `config.yaml` is deployment config that lives on the
//! container host, not in the KB's git repo. A root `.kb-schema.yaml` is the
//! non-deprecated way to declare root rules, and once one exists it is authoritative:
//! it REPLACES the config-derived root outright rather than layering onto it, so a KB
//! carries its root rules with it wherever it is cloned or served, independent of
//! whatever `config.yaml` the deploying host happens to have. `config.yaml`'s
//! `frontmatter` block is consulted only when no root `.kb-schema.yaml` exists at all
//! — see [`SchemaCache::build`].

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use serde::Deserialize;
use serde_json::Value;
use tracing::{debug, warn};

use crate::config::FrontmatterConfig;
use crate::qdrant::{IndexKind, IndexedField};

/// Filename that declares a schema for its directory and everything beneath it.
pub const SCHEMA_FILE_NAME: &str = ".kb-schema.yaml";

/// Largest schema file we will attempt to parse.
///
/// Schema files arrive through git sync and are therefore untrusted. Deeply nested
/// YAML costs superlinear time to parse — hundreds of kilobytes can burn seconds of
/// CPU before the parser's own recursion guard even rejects it — and this parse runs
/// on every write, every instructions refresh, and every index run. A real schema is
/// a few kilobytes; anything approaching this cap is not a schema.
const MAX_SCHEMA_FILE_BYTES: u64 = 256 * 1024;

/// Declared type of a frontmatter field. Undeclared fields are not type-checked.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Deserialize, serde::Serialize, schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum FieldType {
    Text,
    Integer,
    Number,
    Boolean,
    /// A scalar drawn from a closed set. Also accepts an array, checking each element —
    /// this preserves how the pre-cascade `allowed` map behaved.
    Enum,
    /// An array; elements are checked against `values` when present.
    List,
    /// `YYYY-MM-DD`.
    Date,
    /// RFC 3339 datetime.
    Timestamp,
    /// A container for dot-path children rather than a value of its own.
    Object,
}

impl FieldType {
    fn describe(self) -> &'static str {
        match self {
            FieldType::Text => "text",
            FieldType::Integer => "an integer",
            FieldType::Number => "a number",
            FieldType::Boolean => "a boolean",
            FieldType::Enum => "a scalar value",
            FieldType::List => "a list",
            FieldType::Date => "a date (YYYY-MM-DD)",
            FieldType::Timestamp => "an RFC 3339 timestamp",
            FieldType::Object => "an object",
        }
    }
}

/// Placeholder inside a `values:` list that splices in the inherited value set at that
/// position — see [`RawFieldDef::values`] and [`ResolvedSchema::merged_with`] for the
/// full splicing/dedup rules. The `$` prefix is reserved: any other `$`-prefixed token
/// in a `values:` list is a hard error rather than a literal value (see `validate_raw`),
/// so a typo here can never silently degrade into "just another permitted tag."
pub const VALUES_SENTINEL: &str = "$values";

/// A field definition exactly as written in a `.kb-schema.yaml`.
///
/// Also doubles as the shape the `update_schema` MCP tool advertises for `set_field`'s
/// `definition` parameter (see `mcp::FieldDefinitionInput`), via a derived
/// [`schemars::JsonSchema`] impl. `deny_unknown_fields` here becomes
/// `additionalProperties: false` in that advertised schema, so a client's own
/// validation — not just our runtime error — can catch a typo'd key.
#[derive(Debug, Clone, PartialEq, Deserialize, serde::Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RawFieldDef {
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub ty: Option<FieldType>,
    /// `None` means "not declared here" and inherits the parent scope's `required`
    /// (`false` if there is no parent definition either) — see
    /// [`ResolvedSchema::merged_with`]. This is why the field is `Option<bool>` rather
    /// than a plain `bool` defaulting to `false`: a plain bool cannot distinguish "the
    /// author wrote `required: false`" from "the author said nothing," and per-attribute
    /// inheritance needs that distinction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required: Option<bool>,
    /// Same absent-means-inherit rule as `required`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indexed: Option<bool>,
    /// Closed set of permitted values, for `enum` and `list`.
    ///
    /// Enforcement strictness is keyed off `ty`, not off whether `values` is present,
    /// and the two regimes are not equivalent: a field with `type: enum` is checked by
    /// [`check_values`] (every scalar, of any JSON type, must canonicalize to a
    /// permitted string); a field that sets `values` but leaves `ty` unset — which is
    /// how every legacy `config.yaml` `allowed` entry arrives, via
    /// [`ResolvedSchema::from_config`], but also any hand-written `.kb-schema.yaml`
    /// field that forgets `type: enum` — is checked by [`check_values_lenient`]
    /// instead, which exempts non-string, non-array values entirely. This is
    /// deliberate (see both functions' docs), not an oversight: it preserves
    /// pre-cascade validation outcomes for configs that never declared types. An
    /// author who wants strict enforcement must write `type: enum` explicitly.
    ///
    /// `None` here inherits the parent's `values` wholesale, same as every other
    /// attribute. `Some(list)` **replaces** the parent's set outright unless `list`
    /// contains the [`VALUES_SENTINEL`] placeholder (`$values`), which splices the
    /// inherited set in at that position — see [`ResolvedSchema::merged_with`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<String>>,
    /// **Deprecated** alias for a leading [`VALUES_SENTINEL`]: `extend: true` behaves
    /// exactly like writing `values: [$values, ...]` (see [`ResolvedSchema::merged_with`]
    /// for the exact expansion), and using it logs a warning naming the offending schema
    /// file. Kept only so schema files written before the sentinel existed keep parsing
    /// and cascading correctly; new schemas should write `$values` directly.
    /// `validate_raw` rejects declaring both on the same field — the two ways of saying
    /// "inherit" must not be able to disagree about where the inherited values land.
    #[serde(default, skip_serializing_if = "is_false")]
    pub extend: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    /// For `object`: whether undeclared child keys are permitted. `None` inherits the
    /// parent's `open` (`true`, the same default a fresh top-level declaration gets,
    /// when there is no parent definition either).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open: Option<bool>,
    /// Nested authoring sugar, flattened into dot-paths at parse time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fields: Option<HashMap<String, RawFieldDef>>,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// A merged field definition, keyed elsewhere by its dot-path.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldDef {
    pub ty: Option<FieldType>,
    pub required: bool,
    pub indexed: bool,
    /// See [`RawFieldDef::values`]: whether this is enforced strictly or leniently
    /// depends on `ty`, and that split is deliberate.
    pub values: Option<Vec<String>>,
    pub default: Option<Value>,
    pub open: bool,
}

impl FieldDef {
    /// Merge one child scope's explicit declarations (`raw`) onto the definition it
    /// inherits from its nearest ancestor (`inherited`, `None` when no ancestor scope
    /// declares this field at all).
    ///
    /// Per-attribute inheritance: every attribute `raw` leaves unset (`None`) falls
    /// through to `inherited`'s value for that attribute, and only defaults outright
    /// (`false`/`true`/absent) when there is no inherited definition either. This is
    /// the whole of [`ResolvedSchema::merged_with`]'s per-field logic — see that
    /// function's doc for why, and [`Self::merge_values`] for the one attribute
    /// (`values`) that has an in-band way to request a merge instead of a plain
    /// override.
    ///
    /// `origin`/`path` are used only to attribute a `warn!` if `raw` uses the
    /// deprecated `extend: true` or an unsatisfiable `$values` sentinel — they do not
    /// affect the result.
    fn merged(raw: &RawFieldDef, inherited: Option<&FieldDef>, origin: &str, path: &str) -> Self {
        Self {
            ty: raw.ty.or(inherited.and_then(|f| f.ty)),
            required: raw
                .required
                .unwrap_or(inherited.is_some_and(|f| f.required)),
            indexed: raw.indexed.unwrap_or(inherited.is_some_and(|f| f.indexed)),
            values: Self::merge_values(raw, inherited, origin, path),
            default: raw
                .default
                .clone()
                .or_else(|| inherited.and_then(|f| f.default.clone())),
            open: raw.open.unwrap_or(inherited.is_none_or(|f| f.open)),
        }
    }

    /// Resolve `raw.values` against `inherited`'s value set.
    ///
    /// - `raw.values` is `None` (the field's `values` is never mentioned at all): plain
    ///   per-attribute inheritance, same as every other attribute — take whatever the
    ///   parent had, verbatim.
    /// - `raw.values` is `Some(list)`: `list` **replaces** the inherited set outright
    ///   — this is the default, deliberately, the same way a shell assignment
    ///   `PATH=/only/this` replaces rather than extends — *unless* `list` contains
    ///   [`VALUES_SENTINEL`] (`$values`), which splices the inherited set in at that
    ///   exact position (`$PATH:/usr/local/bin` names where the inherited part goes).
    ///   The result is deduplicated, keeping the first occurrence of each value, so a
    ///   value listed both explicitly and inherited appears once.
    /// - `raw.extend` (deprecated) is a shorthand for a leading sentinel: `extend: true`
    ///   behaves exactly like `values: [$values, ...raw.values]`. `validate_raw` already
    ///   rejects combining `extend: true` with an explicit `$values` in the same list,
    ///   so at most one of these two paths ever contributes the sentinel.
    fn merge_values(
        raw: &RawFieldDef,
        inherited: Option<&FieldDef>,
        origin: &str,
        path: &str,
    ) -> Option<Vec<String>> {
        let inherited_values = inherited.and_then(|f| f.values.as_deref());

        let tokens: Vec<String> = if raw.extend {
            warn!(
                scope = %origin,
                field = %path,
                "'extend: true' is deprecated; write 'values: [{VALUES_SENTINEL}, ...]' \
                 instead (see deploy/USAGE.md)"
            );
            std::iter::once(VALUES_SENTINEL.to_string())
                .chain(raw.values.iter().flatten().cloned())
                .collect()
        } else {
            match &raw.values {
                Some(list) => list.clone(),
                // Not mentioned at all: inherit the parent's set verbatim, no splicing
                // involved.
                None => return inherited_values.map(<[String]>::to_vec),
            }
        };

        let mut out: Vec<String> = Vec::with_capacity(tokens.len());
        let push_dedup = |out: &mut Vec<String>, v: &str| {
            if !out.iter().any(|existing| existing == v) {
                out.push(v.to_string());
            }
        };
        for token in &tokens {
            if token == VALUES_SENTINEL {
                match inherited_values {
                    Some(values) if !values.is_empty() => {
                        for v in values {
                            push_dedup(&mut out, v);
                        }
                    }
                    // Loud, not silent (see module docs on the project's general
                    // stance): a sentinel with nothing to splice most often means the
                    // author expected an ancestor to declare values for this field and
                    // it doesn't (a typo'd path, a missing intermediate schema, etc).
                    // Resolving to "no values contributed" here — rather than treating
                    // the whole `values:` as absent — keeps the field CLOSED (nothing
                    // permitted) instead of silently making it unconstrained, so a
                    // document that sets this field fails validation immediately and
                    // visibly instead of the check quietly stopping enforcement.
                    _ => {
                        warn!(
                            scope = %origin,
                            field = %path,
                            "'{VALUES_SENTINEL}' has nothing to inherit here (no ancestor \
                             scope declares values for this field); it contributes no \
                             values, so any other literals in this list are the complete \
                             permitted set — declare values on an ancestor, or drop the \
                             sentinel if this list is meant to stand alone"
                        );
                    }
                }
            } else {
                push_dedup(&mut out, token);
            }
        }
        Some(out)
    }
}

/// One parsed `.kb-schema.yaml`, before merging with ancestors.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchemaFile {
    #[serde(default)]
    pub fields: HashMap<String, RawFieldDef>,
}

impl SchemaFile {
    /// Check every definition for internal contradictions.
    pub fn validate_self(&self) -> Result<(), String> {
        for (name, raw) in &self.fields {
            validate_raw(name, raw)?;
        }
        Ok(())
    }

    /// Flatten nested authoring sugar into dot-path entries.
    fn flattened(&self) -> BTreeMap<String, RawFieldDef> {
        let mut out = BTreeMap::new();
        for (name, raw) in &self.fields {
            flatten_raw(name, raw, &mut out);
        }
        out
    }
}

/// Reject definitions that contradict themselves before they can freeze a scope.
fn validate_raw(path: &str, raw: &RawFieldDef) -> Result<(), String> {
    if raw.fields.is_some()
        && let Some(ty) = raw.ty
        && ty != FieldType::Object
    {
        // Otherwise the field flattens to BOTH a scalar leaf and a set of dot-path
        // children, and any document satisfying the leaf can never satisfy the
        // children — producing an error that blames the document for a broken schema.
        return Err(format!(
            "field '{path}' declares type '{}' but also nested fields; a field is \
             either a value or a container, not both",
            format!("{ty:?}").to_lowercase()
        ));
    }

    if let Some(values) = &raw.values {
        // The `$` prefix is reserved for placeholders. An unrecognized `$`-prefixed
        // token is always a mistake — usually a typo of `$values` — and must be a hard
        // error here, not a literal value: silently accepting it as "just another
        // permitted tag" is exactly the class of quiet failure this cascade otherwise
        // goes out of its way to avoid (see the reserved-token note on
        // [`VALUES_SENTINEL`]).
        if let Some(bad) = values
            .iter()
            .find(|v| v.starts_with('$') && v.as_str() != VALUES_SENTINEL)
        {
            return Err(format!(
                "field '{path}' has an unrecognized placeholder '{bad}' in its values \
                 list; the only recognized '$'-prefixed token is '{VALUES_SENTINEL}'"
            ));
        }

        let sentinel_count = values
            .iter()
            .filter(|v| v.as_str() == VALUES_SENTINEL)
            .count();
        if sentinel_count > 1 {
            return Err(format!(
                "field '{path}' lists '{VALUES_SENTINEL}' {sentinel_count} times; at \
                 most one placeholder is allowed per values list"
            ));
        }

        // `extend: true` is a deprecated alias for a leading `$values` (see
        // `RawFieldDef::extend`); declaring both on the same field is ambiguous about
        // where the inherited values land; refuse it and make the author pick one.
        if raw.extend && sentinel_count > 0 {
            return Err(format!(
                "field '{path}' sets both 'extend: true' and a '{VALUES_SENTINEL}' \
                 placeholder; 'extend' is a deprecated alias for a leading \
                 '{VALUES_SENTINEL}', so combining them is ambiguous — use one or the \
                 other, not both"
            ));
        }
    }

    for (name, child) in raw.fields.iter().flatten() {
        validate_raw(&format!("{path}.{name}"), child)?;
    }
    Ok(())
}

fn flatten_raw(path: &str, raw: &RawFieldDef, out: &mut BTreeMap<String, RawFieldDef>) {
    if let Some(children) = &raw.fields {
        // The container itself is still a definition (it may declare `open: false`),
        // but without its nested children, which become their own entries.
        let mut container = raw.clone();
        container.fields = None;
        if container.ty.is_none() {
            container.ty = Some(FieldType::Object);
        }
        out.insert(path.to_string(), container);
        for (name, child) in children {
            flatten_raw(&format!("{path}.{name}"), child, out);
        }
    } else {
        out.insert(path.to_string(), raw.clone());
    }
}

/// A constrained edit to a schema file.
///
/// Deliberately not free-form text: a bad schema silently freezes a whole subtree, so
/// callers describe intent and the server renders the YAML.
#[derive(Debug, Clone)]
pub enum SchemaEdit {
    /// Add values to a field's permitted set, creating the field if absent.
    AddValues { field: String, values: Vec<String> },
    /// Remove values from a field's permitted set.
    RemoveValues { field: String, values: Vec<String> },
    /// Declare or replace a field definition outright.
    SetField {
        field: String,
        definition: Box<RawFieldDef>,
    },
    /// Remove a field declaration from this scope.
    RemoveField { field: String },
}

impl SchemaFile {
    /// Apply an edit, returning a description of what changed.
    pub fn apply(&mut self, edit: &SchemaEdit) -> Result<String, String> {
        match edit {
            SchemaEdit::AddValues { field, values } => {
                let def = self
                    .fields
                    .entry(field.clone())
                    .or_insert_with(|| RawFieldDef {
                        ty: Some(FieldType::Enum),
                        // Left unset rather than `Some(false)`/`Some(true)`: this
                        // scope may not be the field's first declaration, and a brand
                        // new definition created just to add a value must not clobber
                        // whatever an ancestor scope already said about `required`,
                        // `indexed`, or `open` for this field — see per-attribute
                        // inheritance in `ResolvedSchema::merged_with`.
                        required: None,
                        indexed: None,
                        values: Some(Vec::new()),
                        extend: false,
                        default: None,
                        open: None,
                        fields: None,
                    });
                let existing = def.values.get_or_insert_with(Vec::new);
                let mut added = Vec::new();
                for value in values {
                    if !existing.contains(value) {
                        existing.push(value.clone());
                        added.push(value.clone());
                    }
                }
                existing.sort();
                if added.is_empty() {
                    Ok(format!(
                        "'{}' already permitted every requested value",
                        field
                    ))
                } else {
                    Ok(format!("added to '{}': {}", field, added.join(", ")))
                }
            }
            SchemaEdit::RemoveValues { field, values } => {
                let def = self
                    .fields
                    .get_mut(field)
                    .ok_or_else(|| format!("field '{}' is not declared in this scope", field))?;
                let existing = def
                    .values
                    .as_mut()
                    .ok_or_else(|| format!("field '{}' has no value list", field))?;
                let before = existing.len();
                existing.retain(|v| !values.contains(v));
                Ok(format!(
                    "removed {} value(s) from '{}'",
                    before - existing.len(),
                    field
                ))
            }
            SchemaEdit::SetField { field, definition } => {
                self.fields
                    .insert(field.clone(), definition.as_ref().clone());
                Ok(format!("declared '{}'", field))
            }
            SchemaEdit::RemoveField { field } => {
                if self.fields.remove(field).is_none() {
                    return Err(format!("field '{}' is not declared in this scope", field));
                }
                Ok(format!("removed declaration of '{}'", field))
            }
        }
    }

    /// Render back to YAML for writing to disk.
    pub fn to_yaml(&self) -> Result<String, String> {
        // BTreeMap for deterministic key order, so a rewrite produces a minimal diff.
        let ordered: BTreeMap<&String, &RawFieldDef> = self.fields.iter().collect();
        let doc = serde_yaml_ng::to_string(&SchemaFileOut { fields: ordered })
            .map_err(|e| format!("could not serialize schema: {e}"))?;
        Ok(format!(
            "# Frontmatter schema for this directory and everything beneath it.\n\
             # Managed by the update_schema MCP tool; hand edits are fine but must stay\n\
             # valid — a malformed file freezes indexing for this whole subtree.\n{doc}"
        ))
    }
}

/// Serialization view of [`SchemaFile`] with deterministic field ordering.
#[derive(serde::Serialize)]
struct SchemaFileOut<'a> {
    fields: BTreeMap<&'a String, &'a RawFieldDef>,
}

/// A fully merged schema for one directory.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolvedSchema {
    pub fields: BTreeMap<String, FieldDef>,
    /// Which schema file contributed each field's current definition. Drives error
    /// messages and `get_schema` provenance.
    pub origin: BTreeMap<String, String>,
}

impl ResolvedSchema {
    /// Adapt the global `frontmatter` config block into the implicit root schema.
    ///
    /// **Deprecated fallback**, used only when the knowledge base has no root
    /// `.kb-schema.yaml` of its own — see [`SchemaCache::build`] and the module docs.
    /// Lossless with respect to the pre-cascade behavior: `allowed` becomes `enum`
    /// fields, which still accept either a scalar or an array of scalars.
    pub fn from_config(config: &FrontmatterConfig) -> Self {
        let blank = || FieldDef {
            ty: None,
            required: false,
            indexed: false,
            values: None,
            default: None,
            open: true,
        };
        let mut fields: BTreeMap<String, FieldDef> = BTreeMap::new();

        for name in &config.required {
            fields.entry(name.clone()).or_insert_with(blank).required = true;
        }
        for name in &config.indexed_fields {
            fields.entry(name.clone()).or_insert_with(blank).indexed = true;
        }
        for (name, default) in &config.defaults {
            fields.entry(name.clone()).or_insert_with(blank).default =
                Some(Value::String(default.clone()));
        }
        for (name, values) in &config.allowed {
            // Deliberately leaves `ty` unset. Declaring these `Enum` would subject
            // numbers, booleans, and non-string array elements to value checking that
            // the pre-cascade validator exempted, newly rejecting documents that used
            // to pass. Undeclared type keeps the lenient path.
            let field = fields.entry(name.clone()).or_insert_with(blank);
            field.values = Some(values.clone());
        }

        let origin = fields
            .keys()
            .map(|k| (k.clone(), "config.yaml".to_string()))
            .collect();

        Self { fields, origin }
    }

    /// Test-only accessor for the private merge, so validation tests can build a
    /// resolved schema without going through a filesystem cascade.
    #[cfg(test)]
    pub(crate) fn merged_with_for_test(&self, child: &SchemaFile, origin: &str) -> Self {
        self.merged_with(child, origin)
    }

    /// Merge a child schema file onto this one.
    ///
    /// The set of fields unions. Merging is **per attribute**, not per field: a child
    /// that redefines a field overrides only the attributes it explicitly writes
    /// (`type`, `required`, `indexed`, `default`, `open`, `values`) — every attribute it
    /// leaves unset still inherits from the parent's definition of that same field. A
    /// child that writes `required: true` and nothing else, say, does not reset the
    /// parent's `values` or `default` to nothing; it only tightens `required`.
    ///
    /// This is deliberately NOT the old rule (a redefinition replacing the whole
    /// definition wholesale, `extend: true` as the sole opt-in to union `values`): that
    /// rule silently discarded a parent's `required`/`indexed`/`default` the moment any
    /// child so much as narrowed `values`, which is exactly the shape of footgun that
    /// let a root-level `required: true` on `tags` go unenforced everywhere, since
    /// every domain redeclared `tags` for its own `values` list. Per-attribute
    /// inheritance means only fields whose redefinition genuinely intends to override a
    /// given attribute do — see [`FieldDef::merged`] for the exact per-attribute rule.
    ///
    /// `values` is the one attribute with an in-band way to request a merge instead of
    /// a plain override — see [`FieldDef::merge_values`] for the `$values` placeholder
    /// and the deprecated `extend: true` alias for it.
    fn merged_with(&self, child: &SchemaFile, origin: &str) -> Self {
        let mut fields = self.fields.clone();
        let mut origins = self.origin.clone();

        for (path, raw) in child.flattened() {
            let inherited = self.fields.get(&path);
            let def = FieldDef::merged(&raw, inherited, origin, &path);

            origins.insert(path.clone(), origin.to_string());
            fields.insert(path, def);
        }

        Self {
            fields,
            origin: origins,
        }
    }

    /// Dot-paths declared `indexed`.
    pub fn indexed_fields(&self) -> Vec<String> {
        self.fields
            .iter()
            .filter(|(_, def)| def.indexed)
            .map(|(path, _)| path.clone())
            .collect()
    }

    /// The payload index kind a declared type needs.
    ///
    /// Numeric and boolean fields need their own index kinds for range and comparison
    /// filters to work; everything else, including undeclared fields, is a keyword.
    pub fn index_kind(ty: Option<FieldType>) -> IndexKind {
        match ty {
            Some(FieldType::Integer) => IndexKind::Integer,
            Some(FieldType::Number) => IndexKind::Float,
            Some(FieldType::Boolean) => IndexKind::Bool,
            _ => IndexKind::Keyword,
        }
    }

    /// A stable fingerprint of this schema.
    ///
    /// Used to detect that a document needs revalidating because the rules changed,
    /// even though its content did not. Must not depend on map iteration order, or
    /// every incremental run would look like a schema change.
    pub fn fingerprint(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        for (path, def) in &self.fields {
            hasher.update(path.as_bytes());
            hasher.update([0u8]);
            hasher.update(format!("{:?}", def.ty).as_bytes());
            hasher.update([def.required as u8, def.indexed as u8, def.open as u8]);
            if let Some(values) = &def.values {
                let mut sorted = values.clone();
                sorted.sort();
                for value in sorted {
                    hasher.update(value.as_bytes());
                    hasher.update([1u8]);
                }
            }
            if let Some(default) = &def.default {
                hasher.update(default.to_string().as_bytes());
            }
            hasher.update([0xffu8]);
        }
        hex::encode(hasher.finalize())
    }
}

/// The discovered schema tree, with per-directory merge results precomputed.
#[derive(Debug, Clone, Default)]
pub struct SchemaCache {
    /// Resolved schema per governing directory (KB-relative), longest path last.
    scopes: Vec<(PathBuf, ResolvedSchema)>,
    /// Raw, unmerged schema files by governing directory, so a proposed edit can be
    /// re-cascaded exactly rather than approximated.
    raw: BTreeMap<PathBuf, SchemaFile>,
    /// Directories whose schema file failed to parse, and why.
    broken: BTreeMap<PathBuf, String>,
    root: ResolvedSchema,
    /// KB root, so a scope's raw schema file can be read back for editing.
    root_path: PathBuf,
}

/// A `SchemaCache` shared across the server, kept current by a single owner rather
/// than rebuilt (a full recursive tree walk) by every caller that needs it.
///
/// `RwLock<Arc<SchemaCache>>` rather than `arc-swap`: a reader takes the lock,
/// clones the `Arc`, and drops the guard immediately (see [`load_shared`]) — a
/// handful of atomic operations — which is cheap enough for a read-mostly value
/// that pulling in a new dependency for lock-free swaps is not justified. The
/// outer `Arc` is what makes this cloneable across the MCP handler, the reindex
/// worker, and the instructions-refresh timer, all of which hold a handle to the
/// SAME lock rather than independent copies.
pub type SharedSchemaCache = Arc<RwLock<Arc<SchemaCache>>>;

/// Clone the current cache out of `shared`. Cheap: a lock acquisition plus an
/// `Arc` clone, with the guard dropped before returning — never held across an
/// `.await`.
///
/// A poisoned lock (a reader or writer panicked while holding it) is recovered
/// rather than propagated, the same policy `server.rs` already applies to the
/// instructions lock: a panic in one caller must not brick every subsequent
/// `get_schema`/write for the rest of the process's life.
pub fn load_shared(shared: &SharedSchemaCache) -> Arc<SchemaCache> {
    shared
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// Swap a freshly built `SchemaCache` into `shared`, replacing whatever was there.
///
/// Callers of [`load_shared`] that are already mid-read hold their own `Arc` clone
/// and are unaffected by a swap landing underneath them — they simply keep using
/// the snapshot they took, and the next `load_shared` call sees the new one.
pub fn store_shared(shared: &SharedSchemaCache, new: SchemaCache) {
    let new = Arc::new(new);
    match shared.write() {
        Ok(mut guard) => *guard = new,
        Err(poisoned) => *poisoned.into_inner() = new,
    }
}

impl SchemaCache {
    /// Walk `data_path` for schema files and precompute every directory's merged schema.
    ///
    /// One pass over the tree, not one per document: resolution afterwards is an
    /// in-memory prefix lookup that touches no filesystem.
    ///
    /// A root `.kb-schema.yaml` (governing directory `""`) is handled differently from
    /// every other scope: instead of merging onto its nearest ancestor — which, at the
    /// root, would mean merging onto the config-derived schema — it REPLACES the
    /// config-derived root outright. Config-derived root rules apply only when no root
    /// `.kb-schema.yaml` exists at all. See the module docs for why: the config block
    /// is deployment config on the container host, and a KB that brings its own root
    /// schema file must not have that schema silently blended with whatever
    /// `frontmatter` block the current host's `config.yaml` happens to declare.
    pub fn build(data_path: &Path, fallback: &FrontmatterConfig) -> Self {
        let config_root = ResolvedSchema::from_config(fallback);
        let mut discovered: Vec<(PathBuf, PathBuf)> = Vec::new();
        collect_schema_files(data_path, data_path, &mut discovered);

        // Shallowest first so each merge sees its parent already resolved, then by path
        // so two scopes at the same depth resolve in a stable order. Depth alone would
        // leave siblings in `read_dir` order, which is not guaranteed — a field declared
        // with conflicting types in two sibling scopes could silently pick a different
        // index kind on different hosts or runs.
        discovered.sort_by(|(a, _), (b, _)| {
            a.components()
                .count()
                .cmp(&b.components().count())
                .then_with(|| a.cmp(b))
        });

        let mut scopes: Vec<(PathBuf, ResolvedSchema)> = Vec::new();
        let mut raw: BTreeMap<PathBuf, SchemaFile> = BTreeMap::new();
        let mut broken: BTreeMap<PathBuf, String> = BTreeMap::new();
        let mut root_file_found = false;

        for (rel_dir, abs_file) in discovered {
            let origin = rel_dir.join(SCHEMA_FILE_NAME).to_string_lossy().to_string();
            let parsed = std::fs::metadata(&abs_file)
                .map_err(|e| format!("could not stat: {e}"))
                .and_then(|meta| {
                    if meta.len() > MAX_SCHEMA_FILE_BYTES {
                        Err(format!(
                            "file is {} bytes, over the {} byte limit; a schema this \
                             large is not parsed",
                            meta.len(),
                            MAX_SCHEMA_FILE_BYTES
                        ))
                    } else {
                        Ok(())
                    }
                })
                .and_then(|()| {
                    std::fs::read_to_string(&abs_file).map_err(|e| format!("could not read: {e}"))
                })
                .and_then(|text| {
                    serde_yaml_ng::from_str::<SchemaFile>(&text).map_err(|e| e.to_string())
                })
                .and_then(|file| file.validate_self().map(|()| file));

            match parsed {
                Ok(file) => {
                    let is_root = rel_dir.as_os_str().is_empty();
                    let merged = if is_root {
                        root_file_found = true;
                        if !config_root.fields.is_empty() {
                            // Loud, not silent: config.yaml declares root rules that are
                            // about to stop applying. This fires on every build (like
                            // the "conflicting types" and "malformed schema" warnings
                            // below), not once — a build runs on every write and every
                            // reconcile sweep, so an operator tailing logs sees it
                            // consistently rather than only at the one moment it first
                            // became true.
                            warn!(
                                "a root {} exists at the knowledge-base root; config.yaml's \
                                 `frontmatter` block no longer applies there — its \
                                 required/indexed_fields/defaults/allowed entries are \
                                 ignored unless the same fields are also declared in the \
                                 root {}. Move anything still needed into it.",
                                SCHEMA_FILE_NAME, SCHEMA_FILE_NAME
                            );
                        }
                        // Replaces, not merges: an empty base, not `config_root`.
                        ResolvedSchema::default().merged_with(&file, &origin)
                    } else {
                        let parent = nearest_schema(&scopes, &rel_dir).unwrap_or(&config_root);
                        parent.merged_with(&file, &origin)
                    };
                    debug!(scope = %rel_dir.display(), "loaded schema");
                    raw.insert(rel_dir.clone(), file);
                    scopes.push((rel_dir, merged));
                }
                Err(e) => {
                    warn!(
                        scope = %rel_dir.display(),
                        "invalid {}: {} — documents in this scope will not be indexed",
                        SCHEMA_FILE_NAME, e
                    );
                    broken.insert(rel_dir, e);
                }
            }
        }

        if !root_file_found && !config_root.fields.is_empty() {
            // The deprecated fallback: no root `.kb-schema.yaml` anywhere, so
            // config.yaml's `frontmatter` block is standing in as the root schema.
            // Still fully supported (see module docs), but this is the direction we
            // want deployments to move away from — flag it every build, same as the
            // "config overridden" warning above, so it stays visible for as long as it
            // is true rather than only at startup.
            warn!(
                "no root {} found; falling back to the deprecated `frontmatter` block \
                 in config.yaml for root-level rules. This still works, but a root {} \
                 is the non-deprecated way to declare them — see deploy/USAGE.md.",
                SCHEMA_FILE_NAME, SCHEMA_FILE_NAME
            );
        }

        // Longest paths last so prefix lookup can scan backwards for the deepest match,
        // with path as a stable tiebreaker among equal depths.
        scopes.sort_by(|(a, _), (b, _)| {
            a.components()
                .count()
                .cmp(&b.components().count())
                .then_with(|| a.cmp(b))
        });

        Self {
            scopes,
            raw,
            broken,
            root: config_root,
            root_path: data_path.to_path_buf(),
        }
    }

    /// The raw, unmerged schema file governing `rel_dir`, or an empty one when that
    /// directory has no schema of its own.
    pub fn raw_file_at(&self, rel_dir: &Path) -> Result<SchemaFile, String> {
        let file = self.root_path.join(rel_dir).join(SCHEMA_FILE_NAME);
        match std::fs::read_to_string(&file) {
            Ok(text) => serde_yaml_ng::from_str(&text).map_err(|e| e.to_string()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(SchemaFile::default()),
            Err(e) => Err(e.to_string()),
        }
    }

    /// The schema `doc_path` would resolve to if `edited_dir`'s schema file held
    /// `candidate`.
    ///
    /// Rebuilds the document's full ancestor chain with the candidate substituted in,
    /// rather than guessing whether a deeper scope shadows the edit. Merge semantics are
    /// per FIELD: a descendant that redeclares one field still inherits every other one,
    /// so "a deeper scope exists" tells you nothing about whether this edit reaches the
    /// document. Returns `None` only when the document lies outside the edited subtree.
    pub fn resolve_with_candidate(
        &self,
        doc_path: &Path,
        edited_dir: &Path,
        candidate: &SchemaFile,
    ) -> Option<ResolvedSchema> {
        let dir = doc_path.parent().unwrap_or(Path::new(""));
        if !path_covers(edited_dir, dir) {
            return None;
        }

        // Every scope governing this document, shallowest first, with the candidate
        // standing in for the edited directory's own file.
        let mut chain: Vec<(&Path, &SchemaFile)> = Vec::new();
        for (scope, file) in &self.raw {
            if path_covers(scope, dir) {
                let source = if scope == edited_dir { candidate } else { file };
                chain.push((scope.as_path(), source));
            }
        }
        if !self.raw.contains_key(edited_dir) {
            // The edited directory has no schema file yet; insert the candidate at its
            // correct depth so deeper scopes still layer on top of it.
            chain.push((edited_dir, candidate));
        }
        chain.sort_by_key(|(scope, _)| scope.components().count());

        // Mirrors `build`'s root policy: when a root `.kb-schema.yaml` governs this
        // document — one already exists on disk, or this very edit is creating one —
        // `chain` already contains an entry for `""` (real or candidate) that fully
        // determines the root's fields, so starting from the config-derived root here
        // would let config re-contaminate fields the root file doesn't mention. Only
        // fall back to the config-derived root when no root schema file is in play at
        // all — the same "root file present, if any, wins outright" rule `build` uses.
        let root_governed =
            self.raw.contains_key(Path::new("")) || edited_dir.as_os_str().is_empty();
        let mut resolved = if root_governed {
            ResolvedSchema::default()
        } else {
            self.root.clone()
        };
        for (scope, file) in chain {
            let origin = scope.join(SCHEMA_FILE_NAME).to_string_lossy().to_string();
            resolved = resolved.merged_with(file, &origin);
        }
        Some(resolved)
    }

    /// Build a cache with no schema files, backed only by the global config.
    ///
    /// Used where a cascade is required by signature but the caller has no tree to walk.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn from_config_only(fallback: &FrontmatterConfig) -> Self {
        Self {
            scopes: Vec::new(),
            raw: BTreeMap::new(),
            broken: BTreeMap::new(),
            root: ResolvedSchema::from_config(fallback),
            root_path: PathBuf::new(),
        }
    }

    /// The effective schema for a KB-relative document path.
    pub fn resolve_for(&self, rel_path: &Path) -> &ResolvedSchema {
        let dir = rel_path.parent().unwrap_or(Path::new(""));
        nearest_schema(&self.scopes, dir).unwrap_or(&self.root)
    }

    /// The root schema: from a root `.kb-schema.yaml` when one exists, otherwise the
    /// config-derived fallback. See [`SchemaCache::build`] for why these don't merge.
    pub fn root(&self) -> &ResolvedSchema {
        self.scopes
            .iter()
            .find(|(dir, _)| dir.as_os_str().is_empty())
            .map(|(_, schema)| schema)
            .unwrap_or(&self.root)
    }

    /// Whether a document lies under a scope whose schema failed to parse.
    ///
    /// Such documents are frozen: not indexed, not re-indexed, and left exactly as they
    /// are in the index. Falling back to the parent schema would silently apply rules we
    /// know to be wrong across a whole subtree.
    pub fn is_frozen(&self, rel_path: &Path) -> Option<&str> {
        let dir = rel_path.parent().unwrap_or(Path::new(""));
        self.broken
            .iter()
            .find(|(broken_dir, _)| path_covers(broken_dir, dir))
            .map(|(_, reason)| reason.as_str())
    }

    /// Directories whose schema file failed to parse, with the reason.
    pub fn broken_scopes(&self) -> impl Iterator<Item = (&PathBuf, &str)> {
        self.broken.iter().map(|(dir, why)| (dir, why.as_str()))
    }

    /// Every dot-path declared `indexed` anywhere in the tree, with its index kind.
    ///
    /// Payload indexes are created once for the whole collection, so a field declared
    /// only in a deep scope still has to be registered up front — otherwise filtering on
    /// it silently fails until the collection is rebuilt.
    ///
    /// When two scopes declare the same path with different types, the first wins and a
    /// warning is emitted: one collection cannot hold two index kinds for one path.
    pub fn all_indexed_fields(&self) -> Vec<IndexedField> {
        let mut fields: Vec<IndexedField> = Vec::new();

        let mut add = |schema: &ResolvedSchema| {
            for (path, def) in &schema.fields {
                if !def.indexed {
                    continue;
                }
                let kind = ResolvedSchema::index_kind(def.ty);
                match fields.iter_mut().find(|f| &f.name == path) {
                    Some(existing) if existing.kind == kind => {}
                    // Keyword is the fallback for an undeclared type, so an explicit
                    // declaration anywhere in the tree beats it — otherwise a field
                    // also named in the legacy config's `indexed_fields` would get a
                    // keyword index and every range filter on it would quietly fail.
                    Some(existing) if existing.kind == IndexKind::Keyword => {
                        existing.kind = kind;
                    }
                    Some(existing) if kind == IndexKind::Keyword => {
                        // Keep the more specific declaration already recorded.
                        let _ = existing;
                    }
                    Some(existing) => {
                        warn!(
                            field = %path,
                            "declared with conflicting types across scopes; indexing as {:?}",
                            existing.kind
                        );
                    }
                    None => fields.push(IndexedField {
                        name: path.clone(),
                        kind,
                    }),
                }
            }
        };

        add(&self.root);
        for (_, schema) in &self.scopes {
            add(schema);
        }

        fields.sort_by(|a, b| a.name.cmp(&b.name));
        fields
    }

    /// Resolve a possibly-partial directory reference to concrete scope directories.
    ///
    /// An exact match wins outright. Otherwise every directory whose trailing segments
    /// match is returned, so the caller can report an ambiguity rather than guessing —
    /// the same contract `get_document` offers for files.
    pub fn match_scope_dirs(&self, needle: &Path) -> Vec<PathBuf> {
        if needle.as_os_str().is_empty() {
            return vec![PathBuf::new()];
        }

        let mut candidates: Vec<&PathBuf> = self.raw.keys().collect();
        candidates.sort();

        if let Some(exact) = candidates.iter().find(|dir| *dir == &needle) {
            return vec![(*exact).clone()];
        }

        candidates
            .into_iter()
            .filter(|dir| dir.ends_with(needle))
            .cloned()
            .collect()
    }

    /// Scopes that declare their own schema, shallowest first.
    pub fn scope_paths(&self) -> impl Iterator<Item = &PathBuf> {
        self.scopes.iter().map(|(dir, _)| dir)
    }
}

/// Whether `ancestor` is `dir` or one of its parents.
fn path_covers(ancestor: &Path, dir: &Path) -> bool {
    ancestor.as_os_str().is_empty() || dir == ancestor || dir.starts_with(ancestor)
}

/// Deepest scope covering `dir`.
fn nearest_schema<'a>(
    scopes: &'a [(PathBuf, ResolvedSchema)],
    dir: &Path,
) -> Option<&'a ResolvedSchema> {
    scopes
        .iter()
        .rev()
        .find(|(scope, _)| path_covers(scope, dir))
        .map(|(_, schema)| schema)
}

/// Recursively collect `(relative dir, absolute schema file)` pairs.
///
/// Deliberately independent of `indexing.include`/`exclude`: a schema governs its
/// subtree even where the markdown there is not indexed.
fn collect_schema_files(root: &Path, dir: &Path, out: &mut Vec<(PathBuf, PathBuf)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };

        if file_type.is_symlink() {
            continue;
        }

        if file_type.is_dir() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == ".git" || name.starts_with('.') {
                continue;
            }
            collect_schema_files(root, &path, out);
        } else if file_type.is_file() && entry.file_name() == SCHEMA_FILE_NAME {
            let rel_dir = dir
                .strip_prefix(root)
                .unwrap_or(Path::new(""))
                .to_path_buf();
            out.push((rel_dir, path));
        }
    }
}

// ---------------------------------------------------------------------------
// Value lookup and type checking
// ---------------------------------------------------------------------------

/// Look up a dot-path inside parsed frontmatter.
pub fn get_by_dotpath<'a>(
    frontmatter: &'a HashMap<String, Value>,
    path: &str,
) -> Option<&'a Value> {
    let mut segments = path.split('.');
    let first = segments.next()?;
    let mut current = frontmatter.get(first)?;
    for segment in segments {
        current = current.as_object()?.get(segment)?;
    }
    Some(current)
}

/// Insert a value at a dot-path, creating intermediate objects.
pub fn set_by_dotpath(frontmatter: &mut HashMap<String, Value>, path: &str, value: Value) {
    let segments: Vec<&str> = path.split('.').collect();
    if segments.len() == 1 {
        frontmatter.insert(segments[0].to_string(), value);
        return;
    }

    let entry = frontmatter
        .entry(segments[0].to_string())
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    let Some(mut cursor) = entry.as_object_mut() else {
        warn!(
            path,
            "cannot apply schema default: '{}' is not an object in this document", segments[0]
        );
        return;
    };
    for segment in &segments[1..segments.len() - 1] {
        let next = cursor
            .entry(*segment)
            .or_insert_with(|| Value::Object(serde_json::Map::new()));
        match next.as_object_mut() {
            Some(map) => cursor = map,
            None => {
                warn!(
                    path,
                    "cannot apply schema default: '{}' is not an object in this document", segment
                );
                return;
            }
        }
    }
    cursor.insert(segments[segments.len() - 1].to_string(), value);
}

fn is_date(s: &str) -> bool {
    chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").is_ok()
}

fn is_timestamp(s: &str) -> bool {
    chrono::DateTime::parse_from_rfc3339(s).is_ok()
}

/// Describe a JSON value's kind for error messages.
fn kind_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(n) => {
            if n.is_f64() {
                "a decimal number"
            } else {
                "an integer"
            }
        }
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "an object",
    }
}

/// Check a value against a declared type. Returns a human-readable reason on failure.
pub fn check_type(ty: FieldType, value: &Value) -> Result<(), String> {
    let ok = match ty {
        FieldType::Text => value.is_string(),
        FieldType::Integer => value.as_i64().is_some() || value.as_u64().is_some(),
        FieldType::Number => value.is_number(),
        FieldType::Boolean => value.is_boolean(),
        // Enum accepts a scalar, or an array of scalars — the latter preserves how the
        // pre-cascade `allowed` map treated tag lists.
        FieldType::Enum => match value {
            Value::Array(items) => items.iter().all(|v| !v.is_array() && !v.is_object()),
            other => !other.is_null(),
        },
        FieldType::List => value.is_array(),
        FieldType::Date => value.as_str().map(is_date).unwrap_or(false),
        FieldType::Timestamp => value.as_str().map(is_timestamp).unwrap_or(false),
        FieldType::Object => value.is_object(),
    };

    if ok {
        Ok(())
    } else {
        Err(format!(
            "expected {}, got {}",
            ty.describe(),
            kind_of(value)
        ))
    }
}

/// Check a value against a closed set of permitted values, with pre-cascade semantics,
/// for fields whose type was never declared.
///
/// Arrays are checked element-wise, so a tag list satisfies the set when every tag does.
///
/// The legacy `allowed` map only ever enforced against strings and the string elements
/// of an array; numbers, booleans, and non-string elements were exempt. Preserving that
/// exactly is what keeps a config-only deployment's validation outcomes unchanged.
///
/// This is the *lenient* counterpart to [`check_values`] — same `values` concept, two
/// enforcement regimes. `validate::field_errors` is the dispatch point: it calls this
/// function when `def.ty` is `None` and `check_values` when `def.ty` is `Some(_)`,
/// regardless of which config surface (`config.yaml` `allowed` vs `.kb-schema.yaml`
/// `values`) produced the field. The split is deliberate (see [`RawFieldDef::values`]
/// for the full rationale) — do not converge these two functions.
pub fn check_values_lenient(value: &Value, permitted: Option<&[String]>) -> Result<(), String> {
    let Some(permitted) = permitted else {
        return Ok(());
    };

    // Non-scalars are exempt, matching the pre-cascade validator, which logged when it
    // declined to enforce. Keep that signal.
    if !matches!(value, Value::String(_) | Value::Array(_)) {
        debug!(
            "skipping value enforcement on a non-string, non-array value (legacy \
             `allowed` semantics)"
        );
    }

    let check_string = |s: &String| -> Result<(), String> {
        if permitted.contains(s) {
            Ok(())
        } else {
            Err(format!(
                "'{}' is not permitted here (allowed: {})",
                s,
                permitted.join(", ")
            ))
        }
    };

    match value {
        Value::String(s) => check_string(s),
        Value::Array(items) => items.iter().try_for_each(|item| match item {
            Value::String(s) => check_string(s),
            _ => Ok(()),
        }),
        _ => Ok(()),
    }
}

/// Check a value against a closed set of permitted values, for a field with an
/// explicitly declared type (`type: enum` or `type: list`).
///
/// Every scalar is canonicalized to text and compared, regardless of JSON type — unlike
/// [`check_values_lenient`], nothing is exempt. See that function's doc for why the two
/// differ and where the split is made.
pub fn check_values(value: &Value, permitted: &[String]) -> Result<(), String> {
    let matches = |v: &Value| -> Result<(), String> {
        let as_text = crate::document_fields::canonical_text(v)
            .ok_or_else(|| format!("{} cannot be checked against a value list", kind_of(v)))?;
        if permitted.iter().any(|p| p == &as_text) {
            Ok(())
        } else {
            Err(format!(
                "'{}' is not permitted here (allowed: {})",
                as_text,
                permitted.join(", ")
            ))
        }
    };

    match value {
        Value::Array(items) => items.iter().try_for_each(matches),
        other => matches(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use tempfile::TempDir;

    fn write_schema(root: &Path, dir: &str, yaml: &str) {
        let target = if dir.is_empty() {
            root.to_path_buf()
        } else {
            root.join(dir)
        };
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join(SCHEMA_FILE_NAME), yaml).unwrap();
    }

    fn empty_config() -> FrontmatterConfig {
        FrontmatterConfig::default()
    }

    // -- flattening ---------------------------------------------------------

    #[test]
    fn nested_and_flat_authoring_produce_identical_schemas() {
        let nested: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  planning:\n    type: object\n    fields:\n      prep_minutes:\n        type: integer\n        indexed: true\n",
        )
        .unwrap();
        let flat: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  planning:\n    type: object\n  planning.prep_minutes:\n    type: integer\n    indexed: true\n",
        )
        .unwrap();

        let base = ResolvedSchema::default();
        assert_eq!(
            base.merged_with(&nested, "a").fields,
            base.merged_with(&flat, "a").fields,
            "nested authoring is sugar for dot-paths"
        );
    }

    #[test]
    fn nested_container_defaults_to_object_type() {
        let file: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  planning:\n    fields:\n      rating:\n        type: integer\n",
        )
        .unwrap();
        let flattened = file.flattened();
        assert_eq!(flattened["planning"].ty, Some(FieldType::Object));
        assert_eq!(flattened["planning.rating"].ty, Some(FieldType::Integer));
    }

    // -- merge semantics ----------------------------------------------------

    #[test]
    fn deeper_scopes_inherit_shallower_fields() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  title:\n    required: true\n");
        write_schema(
            dir.path(),
            "kitchen/recipes",
            "fields:\n  cook_minutes:\n    type: integer\n",
        );

        let cache = SchemaCache::build(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("kitchen/recipes/chili.md"));

        assert!(schema.fields["title"].required, "root field is inherited");
        assert_eq!(schema.fields["cook_minutes"].ty, Some(FieldType::Integer));
    }

    #[test]
    fn redefinition_overrides_only_the_attributes_it_declares() {
        // This is the exact footgun the per-attribute rewrite exists to close: the
        // child only mentions `values` (and replaces it outright — no sentinel), so
        // `type` and `required` must still come from the root. Under the old
        // wholesale-replace rule this redefinition would silently drop `required`,
        // which is why the live KB's root `tags: { required: true }` never actually
        // applied anywhere — every domain redeclared `tags` for its own `values`.
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "",
            "fields:\n  status:\n    type: enum\n    required: true\n    values: [active, draft]\n",
        );
        write_schema(
            dir.path(),
            "scratch",
            "fields:\n  status:\n    values: [wip]\n",
        );

        let cache = SchemaCache::build(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("scratch/note.md"));

        assert_eq!(
            schema.fields["status"].values,
            Some(vec!["wip".into()]),
            "values has no sentinel, so it replaces outright"
        );
        assert_eq!(
            schema.fields["status"].ty,
            Some(FieldType::Enum),
            "type was never redeclared, so it still comes from the root"
        );
        assert!(
            schema.fields["status"].required,
            "required was never redeclared, so it still comes from the root — the \
             whole point of per-attribute inheritance"
        );
    }

    #[test]
    fn a_child_can_still_explicitly_override_an_inherited_attribute() {
        // The other half of the same rule: an attribute the child DOES mention still
        // wins, same as before. Per-attribute inheritance only changes what happens to
        // attributes the child stays silent on.
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  status:\n    required: true\n");
        write_schema(
            dir.path(),
            "scratch",
            "fields:\n  status:\n    required: false\n",
        );

        let cache = SchemaCache::build(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("scratch/note.md"));

        assert!(!schema.fields["status"].required);
    }

    #[test]
    fn omitting_values_entirely_inherits_the_parents_set_verbatim() {
        // A child that redeclares a DIFFERENT attribute and never mentions `values` at
        // all inherits the parent's values set unchanged — ordinary per-attribute
        // inheritance, no splicing involved (that's only for when `values` itself is
        // redeclared).
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "",
            "fields:\n  status:\n    type: enum\n    values: [active, draft]\n",
        );
        write_schema(
            dir.path(),
            "scratch",
            "fields:\n  status:\n    required: true\n",
        );

        let cache = SchemaCache::build(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("scratch/note.md"));

        assert_eq!(
            schema.fields["status"].values,
            Some(vec!["active".into(), "draft".into()])
        );
        assert!(schema.fields["status"].required);
    }

    #[test]
    fn per_attribute_inheritance_covers_every_attribute() {
        // One field, every attribute set at the root, a child that overrides exactly
        // one of them (`indexed`). Everything else — type, required, default, open —
        // must survive untouched.
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "",
            "fields:\n  note:\n    type: object\n    open: false\n    required: true\n    \
             indexed: false\n    default: {}\n",
        );
        write_schema(dir.path(), "child", "fields:\n  note:\n    indexed: true\n");

        let cache = SchemaCache::build(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("child/doc.md"));
        let note = &schema.fields["note"];

        assert_eq!(note.ty, Some(FieldType::Object), "type inherited");
        assert!(note.required, "required inherited");
        assert!(note.indexed, "indexed is the one attribute the child set");
        assert!(!note.open, "open inherited");
        assert!(note.default.is_some(), "default inherited");
    }

    #[test]
    fn three_level_cascade_with_each_level_setting_a_different_attribute() {
        // root -> domain -> subdirectory, each level touching only ONE attribute of
        // the same field. The document under the subdirectory must see all three.
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  x:\n    required: true\n");
        write_schema(dir.path(), "a", "fields:\n  x:\n    indexed: true\n");
        write_schema(
            dir.path(),
            "a/b",
            "fields:\n  x:\n    type: enum\n    values: [one, two]\n",
        );

        let cache = SchemaCache::build(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("a/b/doc.md"));
        let x = &schema.fields["x"];

        assert!(x.required, "from root");
        assert!(x.indexed, "from the domain level");
        assert_eq!(x.ty, Some(FieldType::Enum), "from the subdirectory");
        assert_eq!(x.values, Some(vec!["one".into(), "two".into()]));
    }

    // -- $values sentinel -----------------------------------------------------

    #[test]
    fn values_replace_by_default_with_no_sentinel() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  tags:\n    values: [a, b]\n");
        write_schema(dir.path(), "child", "fields:\n  tags:\n    values: [c]\n");

        let cache = SchemaCache::build(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("child/doc.md"));

        assert_eq!(
            schema.fields["tags"].values,
            Some(vec!["c".into()]),
            "no sentinel present, so the child's list replaces outright — the shell \
             `PATH=/only/this` case"
        );
    }

    #[test]
    fn leading_sentinel_splices_inherited_values_first() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  tags:\n    values: [a, b]\n");
        write_schema(
            dir.path(),
            "child",
            "fields:\n  tags:\n    values: [$values, c]\n",
        );

        let cache = SchemaCache::build(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("child/doc.md"));

        assert_eq!(
            schema.fields["tags"].values,
            Some(vec!["a".into(), "b".into(), "c".into()])
        );
    }

    #[test]
    fn trailing_sentinel_splices_inherited_values_last() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  tags:\n    values: [a, b]\n");
        write_schema(
            dir.path(),
            "child",
            "fields:\n  tags:\n    values: [c, $values]\n",
        );

        let cache = SchemaCache::build(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("child/doc.md"));

        assert_eq!(
            schema.fields["tags"].values,
            Some(vec!["c".into(), "a".into(), "b".into()]),
            "position is meaningful: the sentinel sits after 'c', so inherited values \
             land after it too"
        );
    }

    #[test]
    fn splicing_deduplicates_keeping_first_occurrence_order() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  tags:\n    values: [a, b]\n");
        write_schema(
            dir.path(),
            "child",
            "fields:\n  tags:\n    values: [b, $values, c]\n",
        );

        let cache = SchemaCache::build(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("child/doc.md"));

        assert_eq!(
            schema.fields["tags"].values,
            Some(vec!["b".into(), "a".into(), "c".into()]),
            "'b' keeps its first (literal, pre-sentinel) position and is not repeated \
             when the sentinel splices in the inherited set that also contains it"
        );
    }

    #[test]
    fn sentinel_with_nothing_inherited_resolves_to_only_the_literals() {
        // No ancestor declares any values for this field at all. The sentinel
        // contributes nothing (a loud warning is logged, but not asserted on here —
        // see `merge_values`'s doc for why this degrades rather than hard-erroring),
        // and any literal tokens still in the list are the complete permitted set.
        let dir = TempDir::new().unwrap();
        let cache = SchemaCache::build(dir.path(), &empty_config());
        let candidate: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  tags:\n    values: [$values, only]\n").unwrap();

        let effective = cache
            .resolve_with_candidate(Path::new("doc.md"), Path::new(""), &candidate)
            .unwrap();

        assert_eq!(effective.fields["tags"].values, Some(vec!["only".into()]));
    }

    #[test]
    fn sentinel_alone_with_nothing_inherited_closes_the_field_rather_than_leaving_it_unconstrained()
    {
        // The sharper edge of the same case: NOTHING resolves (no inherited values, no
        // other literals), so the field ends up `Some(vec![])` — permitting nothing —
        // rather than `None` — permitting anything. An empty closed set fails loudly
        // the moment a document sets the field; `None` would fail silently by not
        // checking at all. See `merge_values`'s doc for the full reasoning.
        let dir = TempDir::new().unwrap();
        let cache = SchemaCache::build(dir.path(), &empty_config());
        let candidate: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  tags:\n    values: [$values]\n").unwrap();

        let effective = cache
            .resolve_with_candidate(Path::new("doc.md"), Path::new(""), &candidate)
            .unwrap();

        assert_eq!(effective.fields["tags"].values, Some(Vec::new()));
    }

    #[test]
    fn multi_level_cascade_splices_against_the_immediately_inherited_set_not_the_root() {
        // root -> domain -> subdirectory, each splicing in turn. The subdirectory's
        // sentinel must resolve against the DOMAIN's already-merged set ([a, b]), not
        // the root's raw set ([a]) — otherwise 'b' would silently disappear for any
        // document two levels down.
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  tags:\n    values: [a]\n");
        write_schema(
            dir.path(),
            "domain",
            "fields:\n  tags:\n    values: [$values, b]\n",
        );
        write_schema(
            dir.path(),
            "domain/sub",
            "fields:\n  tags:\n    values: [$values, c]\n",
        );

        let cache = SchemaCache::build(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("domain/sub/doc.md"));

        assert_eq!(
            schema.fields["tags"].values,
            Some(vec!["a".into(), "b".into(), "c".into()])
        );
    }

    #[test]
    fn deprecated_extend_true_behaves_like_a_leading_sentinel() {
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "",
            "fields:\n  tags:\n    type: list\n    values: [reference, guide]\n",
        );
        write_schema(
            dir.path(),
            "kitchen",
            "fields:\n  tags:\n    type: list\n    required: true\n    extend: true\n    values: [recipe]\n",
        );

        let cache = SchemaCache::build(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("kitchen/chili.md"));

        assert_eq!(
            schema.fields["tags"].values,
            Some(vec!["reference".into(), "guide".into(), "recipe".into()]),
            "extend: true == a leading $values placeholder"
        );
        assert!(
            schema.fields["tags"].required,
            "extend only ever affected values; every other attribute follows the same \
             per-attribute rule as always, and here the child explicitly sets it"
        );
    }

    #[test]
    fn an_unrecognized_dollar_token_is_a_loud_parse_error() {
        let file: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  tags:\n    values: [$typo, a]\n").unwrap();
        let err = file.validate_self().unwrap_err();
        assert!(err.contains("$typo"), "names the offending token: {err}");
        assert!(err.contains("$values"), "names what IS recognized: {err}");
    }

    #[test]
    fn more_than_one_sentinel_in_a_values_list_is_a_parse_error() {
        let file: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  tags:\n    values: [$values, a, $values]\n")
                .unwrap();
        let err = file.validate_self().unwrap_err();
        assert!(err.contains("tags"), "got: {err}");
    }

    #[test]
    fn combining_extend_true_with_an_explicit_sentinel_is_a_parse_error() {
        let file: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  tags:\n    extend: true\n    values: [$values, a]\n",
        )
        .unwrap();
        let err = file.validate_self().unwrap_err();
        assert!(err.contains("extend"), "got: {err}");
        assert!(err.contains("$values"), "got: {err}");
    }

    #[test]
    fn a_schema_using_the_unrecognized_token_freezes_its_scope() {
        // The parse-time rejection above must actually reach the cascade build, not
        // just the standalone validator — a broken schema freezes its subtree like any
        // other invalid one.
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "bad", "fields:\n  tags:\n    values: [$oops]\n");

        let cache = SchemaCache::build(dir.path(), &empty_config());

        assert!(cache.is_frozen(Path::new("bad/doc.md")).is_some());
    }

    #[test]
    fn three_level_cascade_takes_the_nearest_definition() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  scope:\n    values: [root]\n");
        write_schema(dir.path(), "a", "fields:\n  scope:\n    values: [mid]\n");
        fs::create_dir_all(dir.path().join("a/b")).unwrap();

        let cache = SchemaCache::build(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("a/b/doc.md"));

        assert_eq!(
            schema.fields["scope"].values,
            Some(vec!["mid".into()]),
            "the nearest ancestor wins, not the root"
        );
    }

    #[test]
    fn sibling_scopes_do_not_leak() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "a", "fields:\n  only_a:\n    required: true\n");
        write_schema(dir.path(), "b", "fields:\n  only_b:\n    required: true\n");

        let cache = SchemaCache::build(dir.path(), &empty_config());

        assert!(
            !cache
                .resolve_for(Path::new("b/doc.md"))
                .fields
                .contains_key("only_a")
        );
        assert!(
            !cache
                .resolve_for(Path::new("a/doc.md"))
                .fields
                .contains_key("only_b")
        );
    }

    #[test]
    fn intermediate_directories_resolve_to_nearest_ancestor() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  level:\n    values: [root]\n");
        write_schema(dir.path(), "a", "fields:\n  level:\n    values: [a]\n");
        write_schema(
            dir.path(),
            "a/b/c",
            "fields:\n  level:\n    values: [abc]\n",
        );

        let cache = SchemaCache::build(dir.path(), &empty_config());

        // a/b/ has no schema of its own; it must fall to a/, not root and not a/b/c.
        assert_eq!(
            cache.resolve_for(Path::new("a/b/doc.md")).fields["level"].values,
            Some(vec!["a".into()])
        );
        assert_eq!(
            cache.resolve_for(Path::new("a/b/c/doc.md")).fields["level"].values,
            Some(vec!["abc".into()])
        );
    }

    // -- backward compatibility ---------------------------------------------

    #[test]
    fn config_block_becomes_the_implicit_root_schema() {
        let mut config = FrontmatterConfig {
            required: vec!["title".into(), "description".into()],
            indexed_fields: vec!["type".into(), "tags".into()],
            ..Default::default()
        };
        config
            .allowed
            .insert("status".into(), vec!["active".into(), "draft".into()]);
        config
            .defaults
            .insert("status".into(), "active".to_string());

        let schema = ResolvedSchema::from_config(&config);

        assert!(schema.fields["title"].required);
        assert!(schema.fields["tags"].indexed);
        assert_eq!(
            schema.fields["status"].values,
            Some(vec!["active".into(), "draft".into()])
        );
        assert_eq!(
            schema.fields["status"].ty, None,
            "the adapter must NOT declare a type: doing so would subject numbers and \
             booleans to value checks the pre-cascade validator exempted"
        );
        assert_eq!(
            schema.fields["status"].default,
            Some(json!("active")),
            "defaults stay strings, exactly as before"
        );
    }

    #[test]
    fn a_tree_with_no_schema_files_uses_the_config_root() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("doc.md"), "# hi").unwrap();
        let config = FrontmatterConfig {
            required: vec!["title".into()],
            ..Default::default()
        };

        let cache = SchemaCache::build(dir.path(), &config);

        assert!(
            cache.resolve_for(Path::new("doc.md")).fields["title"].required,
            "existing deployments keep working untouched"
        );
        assert_eq!(cache.broken_scopes().count(), 0);
    }

    #[test]
    fn a_root_schema_file_replaces_the_config_root_instead_of_merging_with_it() {
        // config.yaml declares two rules: `title` required, `legacy_only` indexed. A
        // root `.kb-schema.yaml` exists and redeclares `title` but says nothing about
        // `legacy_only`. Under the old (pre-issue-#91) behavior these merged, so
        // `legacy_only` would still show up, config-sourced, in the resolved root. The
        // whole point of this change is that it must not: once a root schema file
        // exists, it is authoritative and config.yaml's block stops applying.
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "",
            "fields:\n  title:\n    type: text\n    required: true\n",
        );
        let config = FrontmatterConfig {
            required: vec!["title".into()],
            indexed_fields: vec!["legacy_only".into()],
            ..Default::default()
        };

        let cache = SchemaCache::build(dir.path(), &config);
        let root = cache.root();

        assert!(
            root.fields["title"].required,
            "the root file's own rule applies"
        );
        assert!(
            !root.fields.contains_key("legacy_only"),
            "a config-only field must NOT leak into the root once a root schema file \
             exists — that would mean the KB's validation rules depend on whichever \
             config.yaml the deploying host happens to have"
        );
    }

    #[test]
    fn a_root_schema_files_override_reaches_subdirectories_not_the_config_leftover() {
        // Subdirectories inherit from the root scope. That inherited root must be the
        // REPLACED (root-file-only) schema, not a config-merged one — otherwise a
        // config-only field would reach every document in the tree via inheritance
        // even though the root schema itself no longer reports it.
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  title:\n    required: true\n");
        fs::create_dir_all(dir.path().join("food")).unwrap();
        let config = FrontmatterConfig {
            indexed_fields: vec!["legacy_only".into()],
            ..Default::default()
        };

        let cache = SchemaCache::build(dir.path(), &config);
        let resolved = cache.resolve_for(Path::new("food/chili.md"));

        assert!(
            resolved.fields["title"].required,
            "root rule still cascades"
        );
        assert!(
            !resolved.fields.contains_key("legacy_only"),
            "the config leftover must not reach subdirectories through inheritance either"
        );
    }

    #[test]
    fn resolve_with_candidate_replaces_config_when_creating_a_root_schema_file() {
        // update_schema's dry-run path (`resolve_with_candidate`) must apply the same
        // override policy as `build`: proposing a brand-new root `.kb-schema.yaml`
        // must not silently keep enforcing whatever config.yaml declared but the
        // candidate omits.
        let dir = TempDir::new().unwrap();
        let config = FrontmatterConfig {
            required: vec!["legacy_required".into()],
            ..Default::default()
        };
        let cache = SchemaCache::build(dir.path(), &config);

        let candidate: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  title:\n    required: true\n").unwrap();
        let effective = cache
            .resolve_with_candidate(Path::new("doc.md"), Path::new(""), &candidate)
            .expect("root always governs a root-level document");

        assert!(effective.fields["title"].required);
        assert!(
            !effective.fields.contains_key("legacy_required"),
            "the candidate root file replaces config, so a field only config declares \
             must not appear as still-enforced in the dry run"
        );
    }

    #[test]
    fn resolve_with_candidate_still_falls_back_to_config_with_no_root_file_in_play() {
        // The other half of the same policy: editing a NON-root scope while no root
        // schema file exists anywhere must still fall back to the config-derived root
        // for fields the edit itself doesn't touch — the deprecated fallback keeps
        // working until a root file actually shows up.
        let dir = TempDir::new().unwrap();
        let config = FrontmatterConfig {
            required: vec!["title".into()],
            ..Default::default()
        };
        let cache = SchemaCache::build(dir.path(), &config);

        let candidate: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  cook_minutes:\n    type: integer\n").unwrap();
        let effective = cache
            .resolve_with_candidate(Path::new("food/chili.md"), Path::new("food"), &candidate)
            .expect("food/ covers this document");

        assert!(
            effective.fields["title"].required,
            "no root schema file exists, so the config fallback must still reach this document"
        );
        assert_eq!(
            effective.fields["cook_minutes"].ty,
            Some(FieldType::Integer)
        );
    }

    // -- broken schemas -----------------------------------------------------

    #[test]
    fn malformed_schema_freezes_its_scope_without_failing_the_tree() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  title:\n    required: true\n");
        write_schema(dir.path(), "bad", "fields: [this is not a map\n");
        write_schema(dir.path(), "good", "fields:\n  ok:\n    required: true\n");

        let cache = SchemaCache::build(dir.path(), &empty_config());

        assert!(
            cache.is_frozen(Path::new("bad/doc.md")).is_some(),
            "documents under a broken schema must not be indexed under guessed rules"
        );
        assert!(cache.is_frozen(Path::new("good/doc.md")).is_none());
        assert!(cache.is_frozen(Path::new("doc.md")).is_none());
        assert_eq!(cache.broken_scopes().count(), 1);
    }

    #[test]
    fn freezing_covers_the_whole_subtree() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "bad", "not: a: valid: mapping:\n");
        fs::create_dir_all(dir.path().join("bad/deeper")).unwrap();

        let cache = SchemaCache::build(dir.path(), &empty_config());

        assert!(cache.is_frozen(Path::new("bad/deeper/doc.md")).is_some());
    }

    #[test]
    fn an_oversized_schema_file_is_rejected_without_parsing() {
        // Schema files arrive via git sync and are untrusted; deeply nested YAML costs
        // superlinear parse time, and this parse runs on every write and refresh tick.
        let dir = TempDir::new().unwrap();
        let huge = format!("fields:\n{}", "  a: {}\n".repeat(60_000));
        assert!(huge.len() as u64 > super::MAX_SCHEMA_FILE_BYTES);
        write_schema(dir.path(), "big", &huge);

        let started = std::time::Instant::now();
        let cache = SchemaCache::build(dir.path(), &empty_config());
        let elapsed = started.elapsed();

        assert!(
            cache.is_frozen(Path::new("big/doc.md")).is_some(),
            "an unparseable-by-policy schema freezes its scope like any other"
        );
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "the cap must short-circuit before parsing, took {elapsed:?}"
        );
    }

    #[test]
    fn a_normal_sized_schema_is_still_accepted() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "ok", "fields:\n  title:\n    required: true\n");
        let cache = SchemaCache::build(dir.path(), &empty_config());

        assert!(cache.is_frozen(Path::new("ok/doc.md")).is_none());
        assert!(cache.resolve_for(Path::new("ok/doc.md")).fields["title"].required);
    }

    #[test]
    fn unknown_schema_keys_are_rejected() {
        let parsed: Result<SchemaFile, _> =
            serde_yaml_ng::from_str("fields:\n  title:\n    requried: true\n");
        assert!(parsed.is_err(), "a typo'd key must not be silently ignored");
    }

    // -- indexed field union ------------------------------------------------

    #[test]
    fn indexed_fields_union_across_every_scope() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  tags:\n    indexed: true\n");
        write_schema(
            dir.path(),
            "kitchen/recipes",
            "fields:\n  planning.prep_minutes:\n    type: integer\n    indexed: true\n",
        );

        let cache = SchemaCache::build(dir.path(), &empty_config());
        let fields = cache.all_indexed_fields();
        let named = |name: &str| fields.iter().find(|f| f.name == name);

        assert!(named("tags").is_some());
        let nested = named("planning.prep_minutes")
            .expect("a field declared only in a deep scope still needs a payload index");
        assert_eq!(
            nested.kind,
            IndexKind::Integer,
            "an integer field needs an integer index for range filters to work"
        );
    }

    // -- candidate resolution for the schema dry-run -------------------------

    #[test]
    fn candidate_reaches_documents_a_deeper_scope_does_not_override() {
        // Merge is per FIELD: recipes/ declares only cook_minutes, so a new root field
        // still reaches documents under it. Treating "a deeper scope exists" as a
        // shadow would silently report this change as safe.
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "recipes",
            "fields:\n  cook_minutes:\n    type: integer\n",
        );
        let cache = SchemaCache::build(dir.path(), &empty_config());

        let candidate: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  format:\n    required: true\n").unwrap();
        let effective = cache
            .resolve_with_candidate(Path::new("recipes/chili.md"), Path::new(""), &candidate)
            .expect("the edit reaches this document");

        assert!(
            effective.fields["format"].required,
            "a field the deeper scope never touches must still apply"
        );
        assert_eq!(
            effective.fields["cook_minutes"].ty,
            Some(FieldType::Integer),
            "the deeper scope's own declarations survive"
        );
    }

    #[test]
    fn a_deeper_scope_still_wins_for_the_attribute_it_redeclares_but_inherits_the_rest() {
        // `archive/` only ever redeclares `values` (no sentinel — a plain replace) and
        // never mentions `required`. Under per-attribute inheritance that means
        // `values` still wins locally, but `required` — set by this candidate edit at
        // the root — reaches `archive/` anyway, since nothing there overrides it. This
        // is the deliberate reversal from wholesale replacement: a descendant no
        // longer needs to repeat every attribute it doesn't want to lose.
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "archive",
            "fields:\n  status:\n    type: enum\n    values: [archived]\n",
        );
        let cache = SchemaCache::build(dir.path(), &empty_config());

        let candidate: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  status:\n    type: enum\n    values: [active]\n    required: true\n",
        )
        .unwrap();
        let effective = cache
            .resolve_with_candidate(Path::new("archive/old.md"), Path::new(""), &candidate)
            .expect("still inside the edited subtree");

        assert_eq!(
            effective.fields["status"].values,
            Some(vec!["archived".into()]),
            "archive/'s own values (no sentinel) still replace the root's outright"
        );
        assert!(
            effective.fields["status"].required,
            "required was never redeclared by archive/, so the root edit's `required: \
             true` reaches it — the opposite of the old wholesale-replace behavior"
        );
    }

    #[test]
    fn documents_outside_the_edited_subtree_are_unaffected() {
        let dir = TempDir::new().unwrap();
        let cache = SchemaCache::build(dir.path(), &empty_config());
        let candidate: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  x:\n    required: true\n").unwrap();

        assert!(
            cache
                .resolve_with_candidate(
                    Path::new("sysadmin/note.md"),
                    Path::new("food"),
                    &candidate
                )
                .is_none()
        );
    }

    #[test]
    fn a_candidate_for_a_directory_with_no_schema_yet_still_applies() {
        let dir = TempDir::new().unwrap();
        let cache = SchemaCache::build(dir.path(), &empty_config());
        let candidate: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  x:\n    required: true\n").unwrap();

        let effective = cache
            .resolve_with_candidate(Path::new("food/a.md"), Path::new("food"), &candidate)
            .expect("creating a scope must still evaluate against its documents");
        assert!(effective.fields["x"].required);
    }

    #[test]
    fn index_kind_follows_the_declared_type() {
        assert_eq!(
            ResolvedSchema::index_kind(Some(FieldType::Integer)),
            IndexKind::Integer
        );
        assert_eq!(
            ResolvedSchema::index_kind(Some(FieldType::Number)),
            IndexKind::Float
        );
        assert_eq!(
            ResolvedSchema::index_kind(Some(FieldType::Boolean)),
            IndexKind::Bool
        );
        assert_eq!(
            ResolvedSchema::index_kind(Some(FieldType::Enum)),
            IndexKind::Keyword
        );
        assert_eq!(
            ResolvedSchema::index_kind(None),
            IndexKind::Keyword,
            "undeclared fields fall back to keyword"
        );
    }

    #[test]
    fn conflicting_declared_types_take_the_first_and_warn() {
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "a",
            "fields:\n  size:\n    type: integer\n    indexed: true\n",
        );
        write_schema(
            dir.path(),
            "b",
            "fields:\n  size:\n    type: text\n    indexed: true\n",
        );

        let cache = SchemaCache::build(dir.path(), &empty_config());
        let fields = cache.all_indexed_fields();

        // One collection cannot hold two index kinds for one payload path.
        assert_eq!(fields.iter().filter(|f| f.name == "size").count(), 1);
    }

    // -- fingerprint --------------------------------------------------------

    #[test]
    fn fingerprint_is_stable_and_order_independent() {
        let a: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  b:\n    values: [x, y]\n  a:\n    required: true\n",
        )
        .unwrap();
        let b: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  a:\n    required: true\n  b:\n    values: [y, x]\n",
        )
        .unwrap();

        let base = ResolvedSchema::default();
        let first = base.merged_with(&a, "s").fingerprint();
        for _ in 0..8 {
            assert_eq!(base.merged_with(&a, "s").fingerprint(), first);
        }
        assert_eq!(
            base.merged_with(&b, "s").fingerprint(),
            first,
            "declaration order must not change the fingerprint"
        );
    }

    #[test]
    fn fingerprint_changes_when_a_rule_tightens() {
        let loose: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  title:\n    required: false\n").unwrap();
        let tight: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  title:\n    required: true\n").unwrap();

        let base = ResolvedSchema::default();
        assert_ne!(
            base.merged_with(&loose, "s").fingerprint(),
            base.merged_with(&tight, "s").fingerprint()
        );
    }

    // -- dot-path access ----------------------------------------------------

    #[test]
    fn dotpath_reads_nested_values() {
        let fm: HashMap<String, Value> = match json!({
            "planning": { "prep_minutes": 45, "nested": { "deep": true } },
            "title": "x"
        }) {
            Value::Object(map) => map.into_iter().collect(),
            _ => unreachable!(),
        };

        assert_eq!(
            get_by_dotpath(&fm, "planning.prep_minutes"),
            Some(&json!(45))
        );
        assert_eq!(
            get_by_dotpath(&fm, "planning.nested.deep"),
            Some(&json!(true))
        );
        assert_eq!(get_by_dotpath(&fm, "title"), Some(&json!("x")));
        assert_eq!(get_by_dotpath(&fm, "planning.missing"), None);
        assert_eq!(get_by_dotpath(&fm, "title.nope"), None);
    }

    #[test]
    fn dotpath_writes_create_intermediate_objects() {
        let mut fm: HashMap<String, Value> = HashMap::new();
        set_by_dotpath(&mut fm, "planning.effort", json!("medium"));
        set_by_dotpath(&mut fm, "status", json!("active"));

        assert_eq!(fm["planning"]["effort"], json!("medium"));
        assert_eq!(fm["status"], json!("active"));
    }

    // -- type checking ------------------------------------------------------

    #[test]
    fn declared_types_are_enforced_strictly() {
        assert!(check_type(FieldType::Integer, &json!(45)).is_ok());
        assert!(
            check_type(FieldType::Integer, &json!("45")).is_err(),
            "a quoted number is the exact mistake strict typing exists to catch"
        );
        assert!(check_type(FieldType::Boolean, &json!(true)).is_ok());
        assert!(check_type(FieldType::Boolean, &json!("true")).is_err());
        assert!(check_type(FieldType::Number, &json!(1.5)).is_ok());
        assert!(check_type(FieldType::List, &json!(["a"])).is_ok());
        assert!(check_type(FieldType::List, &json!("a")).is_err());
        assert!(check_type(FieldType::Object, &json!({"a": 1})).is_ok());
    }

    #[test]
    fn type_errors_name_both_expected_and_actual() {
        let err = check_type(FieldType::Integer, &json!("five")).unwrap_err();
        assert!(err.contains("an integer"), "got: {err}");
        assert!(err.contains("a string"), "got: {err}");
    }

    #[test]
    fn dates_and_timestamps_validate_their_formats() {
        assert!(check_type(FieldType::Date, &json!("2026-07-31")).is_ok());
        assert!(check_type(FieldType::Date, &json!("31/07/2026")).is_err());
        assert!(check_type(FieldType::Timestamp, &json!("2026-07-31T12:00:00Z")).is_ok());
        assert!(
            check_type(FieldType::Timestamp, &json!("2026-07-31")).is_err(),
            "a bare date is not a timestamp"
        );
    }

    #[test]
    fn enum_accepts_scalars_and_arrays_alike() {
        // Backward compatibility: the old `allowed` map checked both shapes.
        assert!(check_type(FieldType::Enum, &json!("guide")).is_ok());
        assert!(check_type(FieldType::Enum, &json!(["a", "b"])).is_ok());
        assert!(check_type(FieldType::Enum, &json!([{"nested": 1}])).is_err());
    }

    #[test]
    fn value_sets_check_arrays_element_wise() {
        let permitted = vec!["a".to_string(), "b".to_string()];
        assert!(check_values(&json!("a"), &permitted).is_ok());
        assert!(check_values(&json!(["a", "b"]), &permitted).is_ok());
        assert!(check_values(&json!(["a", "z"]), &permitted).is_err());
        assert!(check_values(&json!("z"), &permitted).is_err());
    }

    #[test]
    fn value_set_errors_list_what_is_permitted() {
        let err = check_values(&json!("z"), &["a".to_string(), "b".to_string()]).unwrap_err();
        assert!(err.contains("'z'"));
        assert!(err.contains("a, b"), "error should name the allowed set");
    }

    #[test]
    fn value_sets_compare_booleans_and_numbers_by_canonical_text() {
        assert!(check_values(&json!(true), &["true".to_string()]).is_ok());
        assert!(check_values(&json!(5), &["5".to_string()]).is_ok());
    }

    /// Pins the deliberate divergence documented on `check_values_lenient` and
    /// `RawFieldDef::values` (issue #77): a declared `type: enum` is strict about what
    /// counts as a value at all, while an undeclared-type field with the same
    /// `permitted` set waves non-string, non-array values through untouched. If either
    /// function is ever changed to converge with the other, this fails.
    #[test]
    fn strict_and_lenient_checks_diverge_on_non_string_values_by_design() {
        let permitted = vec!["true".to_string()];

        assert!(
            check_values(&json!(5), &permitted).is_err(),
            "strict check rejects a number not in the permitted (text-compared) set"
        );
        assert!(
            check_values_lenient(&json!(5), Some(&permitted)).is_ok(),
            "lenient check exempts numbers entirely, matching pre-cascade `allowed` semantics"
        );
    }
}
