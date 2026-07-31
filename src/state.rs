use crate::document_fields;
use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
use sqlx::{QueryBuilder, Sqlite, SqlitePool};
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use tracing::{debug, warn};

/// How a single field is matched when listing documents.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldFilter {
    /// Document has the field with any of these values. Multi-valued fields (a tag
    /// list) match if any element matches.
    AnyOf(Vec<String>),
    /// Document has the field with every one of these values — only meaningful for
    /// multi-valued fields.
    AllOf(Vec<String>),
    /// Numeric bounds. Only values that projected to a number participate.
    Range {
        gte: Option<f64>,
        lte: Option<f64>,
        gt: Option<f64>,
        lt: Option<f64>,
    },
}

/// Ordering for a document listing. An allowlist rather than a free string, because
/// `ORDER BY` cannot be a bound parameter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OrderBy {
    #[default]
    Path,
    Title,
    Mtime,
    IndexedAt,
}

impl OrderBy {
    fn column(self) -> &'static str {
        match self {
            OrderBy::Path => "d.file_path",
            OrderBy::Title => "d.title",
            OrderBy::Mtime => "d.mtime",
            OrderBy::IndexedAt => "d.indexed_at",
        }
    }

    /// Parse a caller-supplied ordering, rejecting anything unrecognized.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "path" | "file_path" => Ok(OrderBy::Path),
            "title" => Ok(OrderBy::Title),
            "mtime" | "modified" => Ok(OrderBy::Mtime),
            "indexed_at" | "indexed" => Ok(OrderBy::IndexedAt),
            other => Err(format!(
                "unknown order_by '{}': expected one of path, title, mtime, indexed_at",
                other
            )),
        }
    }
}

/// A document listing request. Every field is optional at the tool boundary; defaults
/// are resolved before reaching here.
#[derive(Debug, Clone)]
pub struct DocumentQuery {
    /// Field path to filter, ANDed across entries.
    pub filters: Vec<(String, FieldFilter)>,
    pub path_prefix: Option<String>,
    pub order_by: OrderBy,
    pub order_desc: bool,
    pub limit: u64,
    pub offset: u64,
    /// Dot-paths to include in each result's frontmatter. `None` returns all of it.
    pub fields: Option<Vec<String>>,
}

impl Default for DocumentQuery {
    fn default() -> Self {
        Self {
            filters: Vec::new(),
            path_prefix: None,
            order_by: OrderBy::default(),
            order_desc: false,
            limit: 100,
            offset: 0,
            fields: None,
        }
    }
}

/// One document in a listing.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentSummary {
    pub file_path: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub mtime: i64,
    pub indexed_at: String,
    pub frontmatter: Value,
}

/// A page of documents plus the total matching the same filters, so a caller can
/// always tell whether the listing was truncated.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentQueryResult {
    pub documents: Vec<DocumentSummary>,
    pub total: u64,
}

impl DocumentQueryResult {
    pub fn has_more(&self, offset: u64) -> bool {
        offset + (self.documents.len() as u64) < self.total
    }
}

/// Read side of the document metadata index. Separate from [`StateDb`]'s inherent API
/// so retrieval can be tested against a mock without SQLite.
pub trait DocumentIndex: Send + Sync {
    async fn query_documents(&self, query: &DocumentQuery) -> Result<DocumentQueryResult>;
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct IndexedFile {
    pub file_path: String,
    pub content_hash: String,
    pub chunk_count: i64,
    pub indexed_at: String,
    /// Fingerprint of the schema this file was last validated against. Empty for rows
    /// written before schema tracking existed, which forces one revalidation pass.
    pub schema_hash: String,
}

/// What happened to one document during reprojection.
enum ReprojectOutcome {
    Done,
    /// The row disappeared mid-run; nothing to do.
    Gone,
    /// Left as-is, with a logged reason.
    Skipped,
    /// Placeholder before the first attempt; counted as skipped if it survives.
    Retry,
}

/// Whether an error is SQLite reporting the database as busy or locked.
///
/// A deferred transaction that opens with a read establishes a snapshot; if another
/// connection commits before it writes, SQLite rejects the write outright rather than
/// serializing it, and `busy_timeout` does not help because retrying the same
/// transaction can never succeed. Only a fresh transaction can.
fn is_busy_or_locked(e: &sqlx::Error) -> bool {
    match e {
        sqlx::Error::Database(db) => {
            let msg = db.message().to_ascii_lowercase();
            msg.contains("database is locked") || msg.contains("database table is locked")
        }
        _ => false,
    }
}

/// Whether an error is SQLite's "duplicate column name", i.e. the column already exists.
fn is_duplicate_column(e: &sqlx::Error) -> bool {
    // Match on the database error specifically rather than any stringified error, so an
    // unrelated failure that happens to mention the phrase cannot be swallowed.
    match e {
        sqlx::Error::Database(db) => db
            .message()
            .to_ascii_lowercase()
            .contains("duplicate column"),
        _ => false,
    }
}

/// Add a column to an existing table when it is not already present.
///
/// SQLite has no `ADD COLUMN IF NOT EXISTS`, and this project has no migration runner,
/// so the presence check goes through `PRAGMA table_info`.
async fn add_column_if_missing(
    pool: &SqlitePool,
    table: &str,
    column: &str,
    definition: &str,
) -> Result<()> {
    let existing: Vec<(i64, String)> = sqlx::query_as(&format!("PRAGMA table_info({table})"))
        .fetch_all(pool)
        .await
        .with_context(|| format!("Failed to inspect columns of '{table}'"))?;

    if existing.iter().any(|(_, name)| name == column) {
        return Ok(());
    }

    match sqlx::query(&format!(
        "ALTER TABLE {table} ADD COLUMN {column} {definition}"
    ))
    .execute(pool)
    .await
    {
        Ok(_) => Ok(()),
        // The check and the ALTER are not atomic, so two processes upgrading at the
        // same instant can both see the column as missing. Losing that race means the
        // column exists, which is precisely the desired end state.
        Err(e) if is_duplicate_column(&e) => {
            debug!("Column '{column}' on '{table}' was added concurrently");
            Ok(())
        }
        Err(e) => {
            Err(anyhow::Error::new(e)
                .context(format!("Failed to add column '{column}' to '{table}'")))
        }
    }
}

pub struct StateDb {
    pool: SqlitePool,
}

impl StateDb {
    pub async fn new(db_path: &Path) -> Result<Self> {
        let db_str = db_path.to_str().ok_or_else(|| {
            anyhow::anyhow!("State DB path is not valid UTF-8: {}", db_path.display())
        })?;
        let options = SqliteConnectOptions::from_str(&format!("sqlite:{}?mode=rwc", db_str))?
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(std::time::Duration::from_secs(5))
            // SQLite disables foreign keys per connection by default. Without this the
            // ON DELETE CASCADE from documents to document_fields silently never fires,
            // leaving orphaned projection rows behind on every delete.
            .foreign_keys(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS indexed_files (
                file_path    TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                chunk_count  INTEGER NOT NULL,
                indexed_at   TEXT NOT NULL DEFAULT (datetime('now'))
            )",
        )
        .execute(&pool)
        .await?;

        // `CREATE TABLE IF NOT EXISTS` is a no-op against a table that already exists,
        // so a column added after a deployment shipped needs an explicit ALTER. There
        // is no migration runner in this project; this guarded add is the mechanism.
        //
        // Existing rows get '', which never equals a real fingerprint, so the first
        // index run after upgrading revalidates every file exactly once. That is the
        // correct behavior, but it is worth expecting rather than being surprised by.
        add_column_if_missing(
            &pool,
            "indexed_files",
            "schema_hash",
            "TEXT NOT NULL DEFAULT ''",
        )
        .await?;

        // Document-level metadata index. `frontmatter` is the faithful JSON record and
        // the source of truth; document_fields below is a rebuildable projection of it.
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS documents (
                file_path    TEXT PRIMARY KEY,
                title        TEXT,
                description  TEXT,
                frontmatter  TEXT NOT NULL,
                mtime        INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                chunk_count  INTEGER NOT NULL,
                indexed_at   TEXT NOT NULL DEFAULT (datetime('now'))
            )",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS document_fields (
                file_path  TEXT NOT NULL REFERENCES documents(file_path) ON DELETE CASCADE,
                field      TEXT NOT NULL,
                value_text TEXT NOT NULL,
                value_num  REAL,
                PRIMARY KEY (file_path, field, value_text)
            )",
        )
        .execute(&pool)
        .await?;

        // The composite primary key is file_path-first, which does not serve the
        // "which documents have field = value" lookup that filtering actually performs.
        // These secondary indexes are load-bearing, not an optimization.
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_document_fields_field_text
             ON document_fields(field, value_text)",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_document_fields_field_num
             ON document_fields(field, value_num)",
        )
        .execute(&pool)
        .await?;

        Ok(Self { pool })
    }

    /// Raw pool access so tests can assert on rows this API does not expose.
    #[cfg(test)]
    pub(crate) fn pool_for_test(&self) -> &SqlitePool {
        &self.pool
    }

    #[cfg(test)]
    pub async fn get(&self, file_path: &str) -> Result<Option<IndexedFile>> {
        let row = sqlx::query_as::<_, IndexedFile>(
            "SELECT file_path, content_hash, chunk_count, indexed_at, schema_hash
             FROM indexed_files WHERE file_path = ?",
        )
        .bind(file_path)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row)
    }

    pub async fn upsert(
        &self,
        file_path: &str,
        content_hash: &str,
        chunk_count: i64,
        schema_hash: &str,
    ) -> Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO indexed_files
                (file_path, content_hash, chunk_count, indexed_at, schema_hash)
             VALUES (?, ?, ?, datetime('now'), ?)",
        )
        .bind(file_path)
        .bind(content_hash)
        .bind(chunk_count)
        .bind(schema_hash)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn delete(&self, file_path: &str) -> Result<()> {
        sqlx::query("DELETE FROM indexed_files WHERE file_path = ?")
            .bind(file_path)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    pub async fn list_all(&self) -> Result<Vec<IndexedFile>> {
        let rows = sqlx::query_as::<_, IndexedFile>(
            "SELECT file_path, content_hash, chunk_count, indexed_at, schema_hash
             FROM indexed_files ORDER BY file_path",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    pub async fn clear(&self) -> Result<()> {
        sqlx::query("DELETE FROM indexed_files")
            .execute(&self.pool)
            .await?;

        // Also clear the metadata index. A full reindex clears `indexed_files` before
        // orphan detection reads it, so orphans are never computed on a full run —
        // leaving any document whose file is gone as a phantom row that no future run
        // can ever detect, because detection is driven off `indexed_files`.
        // document_fields cascades.
        sqlx::query("DELETE FROM documents")
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    pub async fn count(&self) -> Result<i64> {
        let row: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM indexed_files")
            .fetch_one(&self.pool)
            .await?;

        Ok(row.0)
    }

    /// Replace a document's metadata row and its full field projection.
    ///
    /// Runs as a single transaction: the projection is deleted and rebuilt rather than
    /// merged, so a field removed from frontmatter does not leave a stale row behind.
    pub async fn upsert_document_metadata(
        &self,
        file_path: &str,
        frontmatter: &HashMap<String, Value>,
        mtime: i64,
        content_hash: &str,
        chunk_count: i64,
    ) -> Result<()> {
        let title = frontmatter.get("title").and_then(|v| v.as_str());
        let description = frontmatter.get("description").and_then(|v| v.as_str());
        let frontmatter_json = serde_json::to_string(frontmatter)
            .with_context(|| format!("Failed to serialize frontmatter for '{}'", file_path))?;

        let mut tx = self.pool.begin().await?;

        sqlx::query(
            "INSERT INTO documents
                (file_path, title, description, frontmatter, mtime, content_hash, chunk_count, indexed_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
             ON CONFLICT(file_path) DO UPDATE SET
                title        = excluded.title,
                description  = excluded.description,
                frontmatter  = excluded.frontmatter,
                mtime        = excluded.mtime,
                content_hash = excluded.content_hash,
                chunk_count  = excluded.chunk_count,
                indexed_at   = excluded.indexed_at",
        )
        .bind(file_path)
        .bind(title)
        .bind(description)
        .bind(&frontmatter_json)
        .bind(mtime)
        .bind(content_hash)
        .bind(chunk_count)
        .execute(&mut *tx)
        .await?;

        sqlx::query("DELETE FROM document_fields WHERE file_path = ?")
            .bind(file_path)
            .execute(&mut *tx)
            .await?;

        for row in document_fields::flatten_frontmatter(frontmatter) {
            // INSERT OR IGNORE: a document may legitimately repeat a value within an
            // array (duplicate tags), which collides on the primary key.
            sqlx::query(
                "INSERT OR IGNORE INTO document_fields (file_path, field, value_text, value_num)
                 VALUES (?, ?, ?, ?)",
            )
            .bind(file_path)
            .bind(&row.field)
            .bind(&row.value_text)
            .bind(row.value_num)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    /// Remove a document's metadata. Projection rows cascade via the foreign key.
    pub async fn delete_document(&self, file_path: &str) -> Result<()> {
        sqlx::query("DELETE FROM documents WHERE file_path = ?")
            .bind(file_path)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// Content hash of every document that has metadata, keyed by path.
    ///
    /// Used to detect files the incremental indexer skipped but whose metadata is
    /// missing or stale, without a per-file query.
    pub async fn list_document_hashes(&self) -> Result<HashMap<String, String>> {
        let rows: Vec<(String, String)> =
            sqlx::query_as("SELECT file_path, content_hash FROM documents")
                .fetch_all(&self.pool)
                .await?;

        Ok(rows.into_iter().collect())
    }

    /// Rebuild `document_fields` for every document from the stored frontmatter JSON.
    ///
    /// The escape hatch for when the projection rules change: no markdown is re-read,
    /// nothing is re-embedded, and Qdrant is untouched. Returns the number of documents
    /// reprojected.
    pub async fn reproject_all_fields(&self) -> Result<u64> {
        // Only the PATHS are snapshotted. Each document's frontmatter is re-read inside
        // the same transaction that rewrites its projection, so a concurrent reindex
        // can never be reverted: either it lands first and we reproject its new value,
        // or it lands after and overwrites ours. Snapshotting the frontmatter up front
        // would silently roll back any document updated mid-run — and an in-process
        // mutex cannot help here, since this runs as a separate process from `serve`.
        let paths: Vec<(String,)> =
            sqlx::query_as("SELECT file_path FROM documents ORDER BY file_path")
                .fetch_all(&self.pool)
                .await?;

        let mut reprojected = 0u64;
        let mut skipped = 0usize;

        const MAX_ATTEMPTS: usize = 5;

        for (file_path,) in paths {
            let mut outcome = ReprojectOutcome::Retry;

            for attempt in 1..=MAX_ATTEMPTS {
                match self.reproject_one(&file_path).await {
                    Ok(result) => {
                        outcome = result;
                        break;
                    }
                    // A concurrent commit invalidated this transaction's snapshot.
                    // Retrying the SAME transaction can never succeed, but a fresh one
                    // re-reads and will — so one colliding write from the server must
                    // not abandon the rest of the run.
                    Err(e) if is_busy_or_locked(&e) && attempt < MAX_ATTEMPTS => {
                        debug!(file = %file_path, attempt, "reprojection lost a race; retrying");
                        continue;
                    }
                    Err(e) => {
                        warn!(file = %file_path, "skipping reprojection: {e:#}");
                        outcome = ReprojectOutcome::Skipped;
                        break;
                    }
                }
            }

            match outcome {
                ReprojectOutcome::Done => reprojected += 1,
                ReprojectOutcome::Gone => {}
                ReprojectOutcome::Skipped | ReprojectOutcome::Retry => skipped += 1,
            }
        }

        if skipped > 0 {
            warn!("{skipped} document(s) skipped during reprojection; see logs above");
        }

        Ok(reprojected)
    }

    /// Reproject one document, re-reading its frontmatter inside the write transaction.
    async fn reproject_one(&self, file_path: &str) -> Result<ReprojectOutcome, sqlx::Error> {
        let mut tx = self.pool.begin().await?;

        let row: Option<(String,)> =
            sqlx::query_as("SELECT frontmatter FROM documents WHERE file_path = ?")
                .bind(file_path)
                .fetch_optional(&mut *tx)
                .await?;
        let Some((frontmatter_json,)) = row else {
            // Deleted while we were working; nothing to reproject.
            return Ok(ReprojectOutcome::Gone);
        };

        // One unparseable row must not abandon every document after it — this is the
        // designated repair tool for exactly that class of problem.
        let Ok(frontmatter) = serde_json::from_str::<HashMap<String, Value>>(&frontmatter_json)
        else {
            warn!(file = %file_path, "stored frontmatter is unparseable; leaving it alone");
            return Ok(ReprojectOutcome::Skipped);
        };

        sqlx::query("DELETE FROM document_fields WHERE file_path = ?")
            .bind(file_path)
            .execute(&mut *tx)
            .await?;

        for row in document_fields::flatten_frontmatter(&frontmatter) {
            sqlx::query(
                "INSERT OR IGNORE INTO document_fields (file_path, field, value_text, value_num)
                 VALUES (?, ?, ?, ?)",
            )
            .bind(file_path)
            .bind(&row.field)
            .bind(&row.value_text)
            .bind(row.value_num)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(ReprojectOutcome::Done)
    }

    /// Append the shared `WHERE` clause for a document query.
    ///
    /// Field paths and filter values are always bound parameters — under this schema a
    /// field name is row data, never a SQL identifier, so caller-supplied field names
    /// cannot reach the statement text.
    fn push_where(builder: &mut QueryBuilder<'_, Sqlite>, query: &DocumentQuery) {
        builder.push(" WHERE 1 = 1");

        if let Some(prefix) = &query.path_prefix {
            builder.push(" AND d.file_path LIKE ");
            builder.push_bind(escape_like_prefix(prefix));
            builder.push(" ESCAPE '\\'");
        }

        for (field, filter) in &query.filters {
            // `title` and `description` live in dedicated columns and are deliberately
            // excluded from the projection, so filter them directly rather than letting
            // them silently match nothing.
            if let Some(column) = promoted_column(field) {
                match filter {
                    FieldFilter::AllOf(values) if values.len() != 1 => {
                        // These columns hold one scalar, so requiring two distinct
                        // values is unsatisfiable. Falling back to IN would silently
                        // widen the query to any-of, which is the opposite of asked.
                        builder.push(" AND 0 = 1");
                    }
                    FieldFilter::AnyOf(values) | FieldFilter::AllOf(values) => {
                        if values.is_empty() {
                            builder.push(" AND 0 = 1");
                            continue;
                        }
                        builder.push(" AND ");
                        builder.push(column);
                        builder.push(" IN (");
                        let mut separated = builder.separated(", ");
                        for value in values {
                            separated.push_bind(value.clone());
                        }
                        builder.push(")");
                    }
                    FieldFilter::Range { .. } => {
                        // Prose columns have no numeric ordering; an unsatisfiable
                        // clause is more honest than ignoring the filter.
                        builder.push(" AND 0 = 1");
                    }
                }
                continue;
            }

            match filter {
                FieldFilter::AnyOf(values) => {
                    if values.is_empty() {
                        // An empty set matches nothing; say so explicitly rather than
                        // silently dropping the filter.
                        builder.push(" AND 0 = 1");
                        continue;
                    }
                    builder.push(
                        " AND EXISTS (SELECT 1 FROM document_fields f \
                         WHERE f.file_path = d.file_path AND f.field = ",
                    );
                    builder.push_bind(field.clone());
                    builder.push(" AND f.value_text IN (");
                    let mut separated = builder.separated(", ");
                    for value in values {
                        separated.push_bind(value.clone());
                    }
                    builder.push("))");
                }
                FieldFilter::AllOf(values) => {
                    if values.is_empty() {
                        // Mirror the AnyOf convention: an empty required set is an
                        // unsatisfiable filter, not an absent one. Emitting no clause
                        // would silently widen the query to match everything.
                        builder.push(" AND 0 = 1");
                        continue;
                    }
                    for value in values {
                        builder.push(
                            " AND EXISTS (SELECT 1 FROM document_fields f \
                             WHERE f.file_path = d.file_path AND f.field = ",
                        );
                        builder.push_bind(field.clone());
                        builder.push(" AND f.value_text = ");
                        builder.push_bind(value.clone());
                        builder.push(")");
                    }
                }
                FieldFilter::Range { gte, lte, gt, lt } => {
                    builder.push(
                        " AND EXISTS (SELECT 1 FROM document_fields f \
                         WHERE f.file_path = d.file_path AND f.field = ",
                    );
                    builder.push_bind(field.clone());
                    builder.push(" AND f.value_num IS NOT NULL");
                    for (op, bound) in [
                        (" AND f.value_num >= ", gte),
                        (" AND f.value_num <= ", lte),
                        (" AND f.value_num > ", gt),
                        (" AND f.value_num < ", lt),
                    ] {
                        if let Some(value) = bound {
                            builder.push(op);
                            builder.push_bind(*value);
                        }
                    }
                    builder.push(")");
                }
            }
        }
    }

    /// Number of documents carrying metadata. Diverging from [`Self::count`] means the
    /// metadata index needs a backfill pass.
    pub async fn document_count(&self) -> Result<i64> {
        let row: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM documents")
            .fetch_one(&self.pool)
            .await?;

        Ok(row.0)
    }

    /// Every projected field with how many distinct values it takes, most varied first.
    ///
    /// Used to decide which fields are worth breaking down in a status report: a field
    /// with three values (`status`) is a useful histogram, one with several hundred
    /// (`title`) is noise.
    pub async fn field_cardinality(&self) -> Result<Vec<(String, i64)>> {
        let rows: Vec<(String, i64)> = sqlx::query_as(
            "SELECT field, COUNT(DISTINCT value_text) FROM document_fields \
             GROUP BY field ORDER BY COUNT(DISTINCT value_text) DESC, field ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// Document counts per value of `field`, most common first.
    ///
    /// Counts distinct documents rather than rows: multi-valued fields such as `tags`
    /// store one row per value, so a `COUNT(*)` here would be counting projections
    /// rather than documents the moment a field ever repeats within one document.
    pub async fn count_by_field(&self, field: &str, limit: i64) -> Result<Vec<(String, i64)>> {
        // SQLite reads a negative LIMIT as "no limit", so a caller passing one through
        // from arithmetic would silently get the whole vocabulary instead of a page.
        let limit = limit.max(0);
        let rows: Vec<(String, i64)> = sqlx::query_as(
            "SELECT value_text, COUNT(DISTINCT file_path) AS n FROM document_fields \
             WHERE field = ? GROUP BY value_text ORDER BY n DESC, value_text ASC LIMIT ?",
        )
        .bind(field)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// Document counts per top-level directory, derived from stored paths.
    ///
    /// Deliberately sourced from the metadata index rather than a directory walk: this
    /// answers "what has actually been indexed", which is the question a status report
    /// is for. A directory present on disk but absent here is precisely the discrepancy
    /// worth seeing.
    pub async fn area_counts(&self) -> Result<Vec<(String, i64)>> {
        let rows: Vec<(String, i64)> = sqlx::query_as(
            "SELECT CASE WHEN instr(file_path, '/') > 0 \
                         THEN substr(file_path, 1, instr(file_path, '/') - 1) \
                         ELSE '' END AS area, \
                    COUNT(*) AS n \
             FROM documents GROUP BY area ORDER BY n DESC, area ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }
}

/// The `documents` column backing a promoted field, if it is one.
fn promoted_column(field: &str) -> Option<&'static str> {
    match field {
        "title" => Some("d.title"),
        "description" => Some("d.description"),
        _ => None,
    }
}

/// Escape a caller-supplied path prefix for use with `LIKE ... ESCAPE '\'`.
///
/// Real KB paths routinely contain `_`, which is a single-character LIKE wildcard —
/// left unescaped, `lifestyle/my_notes/` would also match `lifestyle/myXnotes/`.
fn escape_like_prefix(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 1);
    for ch in raw.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '%' => out.push_str("\\%"),
            '_' => out.push_str("\\_"),
            c => out.push(c),
        }
    }
    // Our wildcard, appended after escaping so it is never caller-controlled.
    out.push('%');
    out
}

/// Look up a dot-path inside parsed frontmatter.
fn value_at_dotpath<'a>(root: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = root;
    for segment in path.split('.') {
        current = current.as_object()?.get(segment)?;
    }
    Some(current)
}

/// Reduce frontmatter to the requested dot-paths, preserving nesting so the shape a
/// caller sees matches the shape in the document.
fn project_fields(frontmatter: &Value, fields: &[String]) -> Value {
    let mut out = serde_json::Map::new();
    for field in fields {
        let Some(value) = value_at_dotpath(frontmatter, field) else {
            continue;
        };
        let mut cursor = &mut out;
        let segments: Vec<&str> = field.split('.').collect();
        for segment in &segments[..segments.len() - 1] {
            cursor = cursor
                .entry(*segment)
                .or_insert_with(|| Value::Object(serde_json::Map::new()))
                .as_object_mut()
                .expect("intermediate segments are always objects");
        }
        cursor.insert(segments[segments.len() - 1].to_string(), value.clone());
    }
    Value::Object(out)
}

/// A `documents` row as selected by the listing query, in SELECT order:
/// file_path, title, description, mtime, indexed_at, frontmatter.
type DocumentRow = (String, Option<String>, Option<String>, i64, String, String);

impl DocumentIndex for StateDb {
    async fn query_documents(&self, query: &DocumentQuery) -> Result<DocumentQueryResult> {
        // Count and page run on one connection inside a transaction so `total` and the
        // returned rows reflect the same committed snapshot. Without this a concurrent
        // reindex could make has_more disagree with the page.
        let mut tx = self.pool.begin().await?;

        let mut count_builder: QueryBuilder<Sqlite> =
            QueryBuilder::new("SELECT COUNT(*) FROM documents d");
        Self::push_where(&mut count_builder, query);
        let total: i64 = count_builder
            .build_query_scalar()
            .fetch_one(&mut *tx)
            .await?;

        let mut page_builder: QueryBuilder<Sqlite> = QueryBuilder::new(
            "SELECT d.file_path, d.title, d.description, d.mtime, d.indexed_at, d.frontmatter \
             FROM documents d",
        );
        Self::push_where(&mut page_builder, query);
        page_builder.push(" ORDER BY ");
        page_builder.push(query.order_by.column());
        page_builder.push(if query.order_desc { " DESC" } else { " ASC" });
        // Mandatory tiebreaker: ties on mtime are common (many docs share a reindex
        // run) and would otherwise make LIMIT/OFFSET paging non-deterministic.
        if query.order_by != OrderBy::Path {
            page_builder.push(", d.file_path ASC");
        }
        page_builder.push(" LIMIT ");
        page_builder.push_bind(query.limit as i64);
        page_builder.push(" OFFSET ");
        page_builder.push_bind(query.offset as i64);

        let rows: Vec<DocumentRow> = page_builder.build_query_as().fetch_all(&mut *tx).await?;

        tx.commit().await?;

        let documents = rows
            .into_iter()
            .map(
                |(file_path, title, description, mtime, indexed_at, frontmatter_json)| {
                    let frontmatter: Value = serde_json::from_str(&frontmatter_json)
                        .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
                    let frontmatter = match &query.fields {
                        Some(fields) => project_fields(&frontmatter, fields),
                        None => frontmatter,
                    };
                    DocumentSummary {
                        file_path,
                        title,
                        description,
                        mtime,
                        indexed_at,
                        frontmatter,
                    }
                },
            )
            .collect();

        Ok(DocumentQueryResult {
            documents,
            total: total.max(0) as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn test_db() -> (StateDb, TempDir) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        let db = StateDb::new(&path).await.unwrap();
        (db, dir)
    }

    #[tokio::test]
    async fn upsert_and_get() {
        let (db, _dir) = test_db().await;
        db.upsert("test.md", "abc123", 3, "").await.unwrap();
        let entry = db.get("test.md").await.unwrap().unwrap();
        assert_eq!(entry.file_path, "test.md");
        assert_eq!(entry.content_hash, "abc123");
        assert_eq!(entry.chunk_count, 3);
    }

    #[tokio::test]
    async fn upsert_replaces() {
        let (db, _dir) = test_db().await;
        db.upsert("test.md", "hash1", 2, "").await.unwrap();
        db.upsert("test.md", "hash2", 5, "").await.unwrap();
        let entry = db.get("test.md").await.unwrap().unwrap();
        assert_eq!(entry.content_hash, "hash2");
        assert_eq!(entry.chunk_count, 5);
    }

    #[tokio::test]
    async fn delete_removes() {
        let (db, _dir) = test_db().await;
        db.upsert("test.md", "hash", 1, "").await.unwrap();
        db.delete("test.md").await.unwrap();
        assert!(db.get("test.md").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn list_and_count() {
        let (db, _dir) = test_db().await;
        db.upsert("a.md", "h1", 1, "").await.unwrap();
        db.upsert("b.md", "h2", 2, "").await.unwrap();
        assert_eq!(db.count().await.unwrap(), 2);
        let all = db.list_all().await.unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].file_path, "a.md"); // sorted
    }

    #[tokio::test]
    async fn clear_removes_all() {
        let (db, _dir) = test_db().await;
        db.upsert("a.md", "h1", 1, "").await.unwrap();
        db.upsert("b.md", "h2", 2, "").await.unwrap();
        db.clear().await.unwrap();
        assert_eq!(db.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn get_missing_returns_none() {
        let (db, _dir) = test_db().await;
        assert!(db.get("nonexistent.md").await.unwrap().is_none());
    }

    /// Regression: WAL journal mode must be enabled (#9)
    #[tokio::test]
    async fn wal_mode_enabled() {
        let (db, _dir) = test_db().await;
        let row: (String,) = sqlx::query_as("PRAGMA journal_mode")
            .fetch_one(&db.pool)
            .await
            .unwrap();
        assert_eq!(row.0, "wal");
    }

    /// Regression: concurrent writes must not fail with SQLITE_BUSY (#9)
    #[tokio::test]
    async fn concurrent_writes_succeed() {
        let (db, _dir) = test_db().await;
        let mut handles = Vec::new();
        // Share the pool across tasks via Arc
        let pool = db.pool.clone();
        for i in 0..10 {
            let pool = pool.clone();
            handles.push(tokio::spawn(async move {
                let path = format!("file_{}.md", i);
                sqlx::query(
                    "INSERT OR REPLACE INTO indexed_files (file_path, content_hash, chunk_count, indexed_at) VALUES (?, ?, ?, datetime('now'))",
                )
                .bind(&path)
                .bind("hash")
                .bind(1i64)
                .execute(&pool)
                .await
                .unwrap();
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(db.count().await.unwrap(), 10);
    }

    /// Regression: deleting state entry after upsert failure ensures re-processing (#4)
    #[tokio::test]
    async fn delete_after_failure_allows_reprocessing() {
        let (db, _dir) = test_db().await;
        // Simulate: file was indexed with hash1
        db.upsert("doc.md", "hash1", 3, "").await.unwrap();

        // Simulate: upsert to Qdrant fails, so we delete the state entry
        // (this is what ingest.rs now does on failure)
        db.delete("doc.md").await.unwrap();

        // On next run, the file should appear as new (not in state DB)
        assert!(db.get("doc.md").await.unwrap().is_none());
    }

    // -- schema migration ----------------------------------------------------

    /// Build a state DB with the pre-`schema_hash` `indexed_files` schema and rows in
    /// it, exactly as a deployment running the previous release would have.
    async fn legacy_db(dir: &TempDir) -> std::path::PathBuf {
        let path = dir.path().join("legacy.db");
        let options =
            SqliteConnectOptions::from_str(&format!("sqlite:{}?mode=rwc", path.to_str().unwrap()))
                .unwrap()
                .journal_mode(SqliteJournalMode::Wal);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await
            .unwrap();

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS indexed_files (
                file_path    TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                chunk_count  INTEGER NOT NULL,
                indexed_at   TEXT NOT NULL DEFAULT (datetime('now'))
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        for (path_str, hash) in [("a.md", "hash-a"), ("b.md", "hash-b")] {
            sqlx::query(
                "INSERT INTO indexed_files (file_path, content_hash, chunk_count)
                 VALUES (?, ?, 2)",
            )
            .bind(path_str)
            .bind(hash)
            .execute(&pool)
            .await
            .unwrap();
        }

        pool.close().await;
        path
    }

    #[tokio::test]
    async fn upgrading_a_populated_legacy_db_adds_the_column_and_keeps_rows() {
        // The actual upgrade path a second deployment takes. Everything else in this
        // module starts from a fresh file, where the ALTER never has rows to preserve.
        let dir = TempDir::new().unwrap();
        let path = legacy_db(&dir).await;

        let db = StateDb::new(&path).await.expect("upgrade must not fail");

        assert_eq!(db.count().await.unwrap(), 2, "existing rows must survive");
        let entry = db.get("a.md").await.unwrap().unwrap();
        assert_eq!(entry.content_hash, "hash-a");
        assert_eq!(
            entry.schema_hash, "",
            "pre-migration rows carry an empty fingerprint, which forces exactly one \
             revalidation pass"
        );
    }

    #[tokio::test]
    async fn upgrading_is_idempotent() {
        let dir = TempDir::new().unwrap();
        let path = legacy_db(&dir).await;

        let first = StateDb::new(&path).await.unwrap();
        drop(first);
        let second = StateDb::new(&path).await.expect("second open must succeed");

        assert_eq!(second.count().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn upserting_a_legacy_row_after_upgrade_works() {
        let dir = TempDir::new().unwrap();
        let path = legacy_db(&dir).await;
        let db = StateDb::new(&path).await.unwrap();

        db.upsert("a.md", "hash-a2", 3, "fingerprint")
            .await
            .expect("writing to a migrated row must work");

        let entry = db.get("a.md").await.unwrap().unwrap();
        assert_eq!(entry.content_hash, "hash-a2");
        assert_eq!(entry.schema_hash, "fingerprint");
    }

    #[tokio::test]
    async fn concurrent_first_open_does_not_fail_either_caller() {
        // Two independent pools racing the check-then-ALTER on first upgrade. The loser
        // sees "duplicate column name", which is the desired end state, not an error.
        let dir = TempDir::new().unwrap();
        let path = legacy_db(&dir).await;

        let (a, b) = tokio::join!(StateDb::new(&path), StateDb::new(&path));

        assert!(a.is_ok(), "first opener failed: {:?}", a.err());
        assert!(b.is_ok(), "second opener failed: {:?}", b.err());
    }

    #[tokio::test]
    async fn duplicate_column_is_recognized_but_other_errors_are_not() {
        // The migration swallows exactly one error. Anything broader would hide a real
        // failure behind a successful-looking open.
        let dir = TempDir::new().unwrap();
        let path = legacy_db(&dir).await;
        let db = StateDb::new(&path).await.unwrap();

        let duplicate = sqlx::query("ALTER TABLE indexed_files ADD COLUMN schema_hash TEXT")
            .execute(db.pool_for_test())
            .await
            .expect_err("the column already exists");
        assert!(is_duplicate_column(&duplicate));

        let unrelated = sqlx::query("SELECT * FROM no_such_table")
            .execute(db.pool_for_test())
            .await
            .expect_err("table does not exist");
        assert!(
            !is_duplicate_column(&unrelated),
            "an unrelated database error must not be treated as a benign race"
        );
    }

    #[tokio::test]
    async fn a_fresh_db_already_has_the_column() {
        let (db, _dir) = test_db().await;
        db.upsert("new.md", "h", 1, "fp").await.unwrap();
        assert_eq!(db.get("new.md").await.unwrap().unwrap().schema_hash, "fp");
    }

    // -- document metadata index --------------------------------------------

    fn recipe_frontmatter() -> HashMap<String, Value> {
        let json = serde_json::json!({
            "title": "Stir Fry",
            "description": "Weekly improvised stir fry.",
            "type": "reference",
            "tags": ["recipe", "stir-fry", "wok"],
            "planning": { "prep_minutes": 45, "needs_recipe": false, "effort": "medium" }
        });
        match json {
            Value::Object(map) => map.into_iter().collect(),
            _ => unreachable!(),
        }
    }

    async fn field_rows(db: &StateDb, file_path: &str) -> Vec<(String, String, Option<f64>)> {
        sqlx::query_as(
            "SELECT field, value_text, value_num FROM document_fields
             WHERE file_path = ? ORDER BY field, value_text",
        )
        .bind(file_path)
        .fetch_all(&db.pool)
        .await
        .unwrap()
    }

    /// Frontmatter with a chosen type, area-agnostic, plus two tags.
    fn doc_frontmatter(doc_type: &str, tags: &[&str]) -> HashMap<String, Value> {
        let json = serde_json::json!({
            "title": "T",
            "type": doc_type,
            "tags": tags,
        });
        match json {
            Value::Object(map) => map.into_iter().collect(),
            _ => unreachable!(),
        }
    }

    async fn seed_breakdown_corpus(db: &StateDb) {
        for (path, ty, tags) in [
            ("food/recipes/a.md", "recipe", vec!["dinner", "quick"]),
            ("food/recipes/b.md", "recipe", vec!["dinner"]),
            ("food/plans/c.md", "project", vec!["quick"]),
            ("sysadmin/d.md", "guide", vec!["docker"]),
            ("top-level.md", "guide", vec![]),
        ] {
            db.upsert_document_metadata(path, &doc_frontmatter(ty, &tags), 1700, "h", 1)
                .await
                .unwrap();
        }
    }

    #[tokio::test]
    async fn count_by_field_counts_documents_per_value() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;

        let by_type = db.count_by_field("type", 50).await.unwrap();
        assert_eq!(
            by_type,
            vec![
                ("guide".to_string(), 2),
                ("recipe".to_string(), 2),
                ("project".to_string(), 1),
            ],
            "ordered by count desc, then value asc for a stable report"
        );

        // `tags` stores one row per value, so this is the case where counting rows
        // instead of distinct documents would go wrong.
        let by_tag = db.count_by_field("tags", 50).await.unwrap();
        assert_eq!(by_tag.iter().find(|(v, _)| v == "dinner").unwrap().1, 2);
        assert_eq!(by_tag.iter().find(|(v, _)| v == "quick").unwrap().1, 2);
        assert_eq!(by_tag.iter().find(|(v, _)| v == "docker").unwrap().1, 1);
    }

    #[tokio::test]
    async fn count_by_field_respects_the_limit() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;

        let limited = db.count_by_field("type", 2).await.unwrap();
        assert_eq!(limited.len(), 2);
        // The most common values survive truncation.
        assert_eq!(limited[0].1, 2);
    }

    #[tokio::test]
    async fn count_by_field_clamps_a_negative_limit() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;
        // SQLite reads a negative LIMIT as "unbounded", so an unclamped negative would
        // silently return the whole vocabulary instead of nothing.
        assert!(db.count_by_field("type", -1).await.unwrap().is_empty());
        assert!(db.count_by_field("type", 0).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn count_by_field_is_empty_for_an_unknown_field() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;
        assert!(db.count_by_field("nope", 50).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn field_cardinality_reports_distinct_value_counts() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;

        let cards: HashMap<String, i64> =
            db.field_cardinality().await.unwrap().into_iter().collect();
        assert_eq!(cards.get("type"), Some(&3));
        // dinner, quick, docker
        assert_eq!(cards.get("tags"), Some(&3));
        // `title`/`description` are promoted to columns on `documents` and never
        // projected into `document_fields`, so the highest-cardinality free-text fields
        // cannot reach a breakdown at all.
        assert_eq!(cards.get("title"), None);
        assert_eq!(cards.get("description"), None);
    }

    #[tokio::test]
    async fn area_counts_group_by_top_level_directory() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;

        let areas = db.area_counts().await.unwrap();
        assert_eq!(
            areas,
            vec![
                ("food".to_string(), 3),
                ("".to_string(), 1),
                ("sysadmin".to_string(), 1),
            ],
            "a document at the KB root has no area and must not be dropped"
        );
    }

    #[tokio::test]
    async fn aggregates_are_empty_on_a_fresh_database() {
        let (db, _dir) = test_db().await;
        assert!(db.field_cardinality().await.unwrap().is_empty());
        assert!(db.area_counts().await.unwrap().is_empty());
        assert!(db.count_by_field("type", 50).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn aggregates_follow_document_deletion() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;
        db.delete_document("food/recipes/a.md").await.unwrap();

        let by_type: HashMap<String, i64> = db
            .count_by_field("type", 50)
            .await
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(by_type.get("recipe"), Some(&1));

        let areas: HashMap<String, i64> = db.area_counts().await.unwrap().into_iter().collect();
        assert_eq!(areas.get("food"), Some(&2));
    }

    #[tokio::test]
    async fn upsert_document_metadata_stores_promoted_columns() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("r.md", &recipe_frontmatter(), 1700, "h1", 2)
            .await
            .unwrap();

        let row: (String, Option<String>, Option<String>, i64, i64) = sqlx::query_as(
            "SELECT file_path, title, description, mtime, chunk_count FROM documents WHERE file_path = ?",
        )
        .bind("r.md")
        .fetch_one(&db.pool)
        .await
        .unwrap();

        assert_eq!(row.0, "r.md");
        assert_eq!(row.1.as_deref(), Some("Stir Fry"));
        assert_eq!(row.2.as_deref(), Some("Weekly improvised stir fry."));
        assert_eq!(row.3, 1700);
        assert_eq!(row.4, 2);
    }

    #[tokio::test]
    async fn upsert_document_metadata_projects_nested_and_list_fields() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("r.md", &recipe_frontmatter(), 1700, "h1", 2)
            .await
            .unwrap();

        let rows = field_rows(&db, "r.md").await;
        assert!(rows.contains(&("planning.prep_minutes".into(), "45".into(), Some(45.0))));
        assert!(rows.contains(&("planning.needs_recipe".into(), "false".into(), Some(0.0))));
        assert_eq!(
            rows.iter().filter(|r| r.0 == "tags").count(),
            3,
            "one row per tag"
        );
    }

    #[tokio::test]
    async fn frontmatter_json_is_stored_faithfully() {
        let (db, _dir) = test_db().await;
        let fm = recipe_frontmatter();
        db.upsert_document_metadata("r.md", &fm, 1700, "h1", 2)
            .await
            .unwrap();

        let (json,): (String,) =
            sqlx::query_as("SELECT frontmatter FROM documents WHERE file_path = ?")
                .bind("r.md")
                .fetch_one(&db.pool)
                .await
                .unwrap();
        let round_tripped: HashMap<String, Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(round_tripped, fm, "stored JSON must be lossless");
    }

    #[tokio::test]
    async fn reupsert_removes_fields_dropped_from_frontmatter() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("r.md", &recipe_frontmatter(), 1700, "h1", 2)
            .await
            .unwrap();

        // Same document, planning block removed entirely.
        let mut trimmed = recipe_frontmatter();
        trimmed.remove("planning");
        db.upsert_document_metadata("r.md", &trimmed, 1800, "h2", 2)
            .await
            .unwrap();

        let rows = field_rows(&db, "r.md").await;
        assert!(
            rows.iter().all(|r| !r.0.starts_with("planning.")),
            "stale projection rows must not survive a re-upsert"
        );
        assert_eq!(rows.iter().filter(|r| r.0 == "tags").count(), 3);
    }

    #[tokio::test]
    async fn delete_document_cascades_to_fields() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("r.md", &recipe_frontmatter(), 1700, "h1", 2)
            .await
            .unwrap();
        assert!(!field_rows(&db, "r.md").await.is_empty());

        db.delete_document("r.md").await.unwrap();

        assert_eq!(db.document_count().await.unwrap(), 0);
        assert!(
            field_rows(&db, "r.md").await.is_empty(),
            "ON DELETE CASCADE requires foreign_keys(true) to be set on the connection"
        );
    }

    #[tokio::test]
    async fn list_document_hashes_maps_paths_to_hashes() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("a.md", &recipe_frontmatter(), 1, "hash-a", 1)
            .await
            .unwrap();
        db.upsert_document_metadata("b.md", &recipe_frontmatter(), 1, "hash-b", 1)
            .await
            .unwrap();

        let hashes = db.list_document_hashes().await.unwrap();
        assert_eq!(hashes.get("a.md").map(String::as_str), Some("hash-a"));
        assert_eq!(hashes.get("b.md").map(String::as_str), Some("hash-b"));
        assert_eq!(hashes.len(), 2);
    }

    #[tokio::test]
    async fn reproject_rebuilds_fields_from_stored_json() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("r.md", &recipe_frontmatter(), 1700, "h1", 2)
            .await
            .unwrap();

        // Simulate a projection corrupted or emptied by a rules change.
        sqlx::query("DELETE FROM document_fields")
            .execute(&db.pool)
            .await
            .unwrap();
        assert!(field_rows(&db, "r.md").await.is_empty());

        let count = db.reproject_all_fields().await.unwrap();

        assert_eq!(count, 1);
        let rows = field_rows(&db, "r.md").await;
        assert!(rows.contains(&("planning.prep_minutes".into(), "45".into(), Some(45.0))));
    }

    #[tokio::test]
    async fn duplicate_array_values_do_not_break_upsert() {
        let (db, _dir) = test_db().await;
        let fm: HashMap<String, Value> = match serde_json::json!({
            "title": "Dupes",
            "tags": ["recipe", "recipe", "wok"]
        }) {
            Value::Object(map) => map.into_iter().collect(),
            _ => unreachable!(),
        };

        db.upsert_document_metadata("d.md", &fm, 1, "h", 1)
            .await
            .unwrap();

        // Collides on the (file_path, field, value_text) primary key; must not error.
        assert_eq!(field_rows(&db, "d.md").await.len(), 2);
    }

    // -- listing / query builder --------------------------------------------

    fn doc(title: &str, tags: &[&str], prep: i64, tested: bool) -> HashMap<String, Value> {
        let json = serde_json::json!({
            "title": title,
            "description": format!("{} description", title),
            "type": "reference",
            "tags": tags,
            "planning": { "prep_minutes": prep, "tested": tested }
        });
        match json {
            Value::Object(map) => map.into_iter().collect(),
            _ => unreachable!(),
        }
    }

    async fn seeded_db() -> (StateDb, TempDir) {
        let (db, dir) = test_db().await;
        for (path, title, tags, prep, tested) in [
            (
                "kitchen/recipes/chili.md",
                "Chili",
                &["recipe", "dinner"][..],
                20,
                true,
            ),
            (
                "kitchen/recipes/congee.md",
                "Congee",
                &["recipe", "breakfast"][..],
                45,
                true,
            ),
            (
                "kitchen/recipes/stir_fry.md",
                "Stir Fry",
                &["recipe", "dinner"][..],
                45,
                false,
            ),
            ("sysadmin/zfs.md", "ZFS", &["zfs"][..], 0, true),
        ] {
            db.upsert_document_metadata(path, &doc(title, tags, prep, tested), 100, "h", 1)
                .await
                .unwrap();
        }
        (db, dir)
    }

    fn paths(result: &DocumentQueryResult) -> Vec<&str> {
        result
            .documents
            .iter()
            .map(|d| d.file_path.as_str())
            .collect()
    }

    #[tokio::test]
    async fn listing_with_no_filters_returns_everything_ordered_by_path() {
        let (db, _dir) = seeded_db().await;
        let result = db.query_documents(&DocumentQuery::default()).await.unwrap();

        assert_eq!(result.total, 4);
        assert_eq!(
            paths(&result),
            vec![
                "kitchen/recipes/chili.md",
                "kitchen/recipes/congee.md",
                "kitchen/recipes/stir_fry.md",
                "sysadmin/zfs.md",
            ]
        );
    }

    #[tokio::test]
    async fn any_of_matches_documents_with_either_tag() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![(
                "tags".into(),
                FieldFilter::AnyOf(vec!["breakfast".into(), "zfs".into()]),
            )],
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        assert_eq!(result.total, 2);
        assert_eq!(
            paths(&result),
            vec!["kitchen/recipes/congee.md", "sysadmin/zfs.md"]
        );
    }

    #[tokio::test]
    async fn all_of_requires_every_value() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![(
                "tags".into(),
                FieldFilter::AllOf(vec!["recipe".into(), "dinner".into()]),
            )],
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        assert_eq!(result.total, 2, "congee is a recipe but not dinner");
        assert_eq!(
            paths(&result),
            vec!["kitchen/recipes/chili.md", "kitchen/recipes/stir_fry.md"]
        );
    }

    #[tokio::test]
    async fn filters_are_anded_across_fields() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![
                ("tags".into(), FieldFilter::AnyOf(vec!["recipe".into()])),
                (
                    "planning.tested".into(),
                    FieldFilter::AnyOf(vec!["false".into()]),
                ),
            ],
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        assert_eq!(paths(&result), vec!["kitchen/recipes/stir_fry.md"]);
    }

    #[tokio::test]
    async fn numeric_range_filters_on_nested_field() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![(
                "planning.prep_minutes".into(),
                FieldFilter::Range {
                    gte: None,
                    lte: None,
                    gt: None,
                    lt: Some(30.0),
                },
            )],
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        // chili is 20; zfs is 0. String comparison would have ordered "45" before "5".
        assert_eq!(
            paths(&result),
            vec!["kitchen/recipes/chili.md", "sysadmin/zfs.md"]
        );
    }

    #[tokio::test]
    async fn range_bounds_combine_as_between() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![(
                "planning.prep_minutes".into(),
                FieldFilter::Range {
                    gte: Some(10.0),
                    lte: Some(30.0),
                    gt: None,
                    lt: None,
                },
            )],
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        assert_eq!(paths(&result), vec!["kitchen/recipes/chili.md"]);
    }

    #[tokio::test]
    async fn path_prefix_narrows_to_a_folder() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            path_prefix: Some("kitchen/recipes/".into()),
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        assert_eq!(result.total, 3);
        assert!(paths(&result).iter().all(|p| p.starts_with("kitchen/")));
    }

    #[tokio::test]
    async fn path_prefix_escapes_like_wildcards() {
        // `_` is a single-character LIKE wildcard. Unescaped, this prefix would also
        // match a hypothetical `kitchen/recipesX/`.
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            path_prefix: Some("kitchen/recipes/stir_fry".into()),
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        assert_eq!(paths(&result), vec!["kitchen/recipes/stir_fry.md"]);

        // And a prefix whose underscore does not literally match finds nothing.
        let query = DocumentQuery {
            path_prefix: Some("kitchen/recipes/stir_fr".into()),
            ..Default::default()
        };
        assert_eq!(db.query_documents(&query).await.unwrap().total, 1);

        let query = DocumentQuery {
            path_prefix: Some("kitchen/recipes/stirXfry".into()),
            ..Default::default()
        };
        assert_eq!(
            db.query_documents(&query).await.unwrap().total,
            0,
            "underscore must not behave as a wildcard"
        );
    }

    #[tokio::test]
    async fn total_reflects_all_matches_not_just_the_page() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            limit: 2,
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        assert_eq!(result.documents.len(), 2);
        assert_eq!(result.total, 4, "truncation must never be silent");
        assert!(result.has_more(0));
    }

    #[tokio::test]
    async fn offset_pages_deterministically() {
        let (db, _dir) = seeded_db().await;
        let page = |offset| DocumentQuery {
            limit: 2,
            offset,
            ..Default::default()
        };

        let first = db.query_documents(&page(0)).await.unwrap();
        let second = db.query_documents(&page(2)).await.unwrap();

        assert_eq!(
            paths(&first),
            vec!["kitchen/recipes/chili.md", "kitchen/recipes/congee.md"]
        );
        assert_eq!(
            paths(&second),
            vec!["kitchen/recipes/stir_fry.md", "sysadmin/zfs.md"]
        );
        assert!(!second.has_more(2));
    }

    #[tokio::test]
    async fn ordering_by_mtime_still_paginates_deterministically() {
        // Every seeded document shares mtime 100, so only the file_path tiebreaker
        // keeps paging stable.
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            order_by: OrderBy::Mtime,
            order_desc: true,
            limit: 2,
            ..Default::default()
        };

        let first = db.query_documents(&query).await.unwrap();
        for _ in 0..5 {
            assert_eq!(
                paths(&db.query_documents(&query).await.unwrap()),
                paths(&first)
            );
        }
    }

    #[tokio::test]
    async fn empty_any_of_matches_nothing_rather_than_everything() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![("tags".into(), FieldFilter::AnyOf(vec![]))],
            ..Default::default()
        };

        assert_eq!(db.query_documents(&query).await.unwrap().total, 0);
    }

    #[tokio::test]
    async fn fields_projection_preserves_nesting() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            path_prefix: Some("kitchen/recipes/chili.md".into()),
            fields: Some(vec!["planning.prep_minutes".into(), "type".into()]),
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        let fm = &result.documents[0].frontmatter;
        assert_eq!(fm["planning"]["prep_minutes"], serde_json::json!(20));
        assert_eq!(fm["type"], serde_json::json!("reference"));
        assert!(fm.get("tags").is_none(), "unrequested fields are omitted");
        assert!(
            fm["planning"].get("tested").is_none(),
            "sibling nested fields are omitted"
        );
    }

    #[tokio::test]
    async fn full_frontmatter_is_returned_when_no_fields_requested() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            path_prefix: Some("kitchen/recipes/chili.md".into()),
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        let fm = &result.documents[0].frontmatter;
        assert_eq!(fm["planning"]["tested"], serde_json::json!(true));
        assert_eq!(fm["tags"], serde_json::json!(["recipe", "dinner"]));
    }

    #[tokio::test]
    async fn order_by_parse_rejects_unknown_values() {
        assert_eq!(OrderBy::parse("path"), Ok(OrderBy::Path));
        assert_eq!(OrderBy::parse("MTIME"), Ok(OrderBy::Mtime));
        assert!(OrderBy::parse("d.file_path; DROP TABLE documents").is_err());
    }

    #[tokio::test]
    async fn offset_at_or_past_the_end_returns_an_empty_page_not_an_error() {
        let (db, _dir) = seeded_db().await;

        let at_end = db
            .query_documents(&DocumentQuery {
                offset: 4,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(at_end.documents.is_empty());
        assert_eq!(at_end.total, 4);
        assert!(!at_end.has_more(4));

        let past_end = db
            .query_documents(&DocumentQuery {
                offset: 999,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(past_end.documents.is_empty());
        assert_eq!(
            past_end.total, 4,
            "total still reports the full match count"
        );
        assert!(!past_end.has_more(999));
    }

    #[tokio::test]
    async fn ordering_by_a_nullable_column_paginates_without_gaps_or_repeats() {
        // `title` is nullable, and SQLite groups NULLs together — only the file_path
        // tiebreaker keeps paging stable across the NULL group.
        let (db, _dir) = test_db().await;
        for (path, title) in [
            ("a.md", Some("Zebra")),
            ("b.md", None),
            ("c.md", Some("Apple")),
            ("d.md", None),
        ] {
            let mut fm: HashMap<String, Value> = HashMap::new();
            if let Some(t) = title {
                fm.insert("title".into(), Value::String(t.into()));
            }
            fm.insert("type".into(), Value::String("reference".into()));
            db.upsert_document_metadata(path, &fm, 100, "h", 1)
                .await
                .unwrap();
        }

        let page = |offset| DocumentQuery {
            order_by: OrderBy::Title,
            limit: 2,
            offset,
            ..Default::default()
        };
        let first = db.query_documents(&page(0)).await.unwrap();
        let second = db.query_documents(&page(2)).await.unwrap();

        let mut seen: Vec<&str> = paths(&first);
        seen.extend(paths(&second));
        seen.sort();
        assert_eq!(
            seen,
            vec!["a.md", "b.md", "c.md", "d.md"],
            "every document appears exactly once across the two pages"
        );
    }

    #[tokio::test]
    async fn promoted_fields_are_filterable_through_their_columns() {
        // title/description are excluded from the projection, so without the dedicated
        // column path a filter on them would silently match nothing.
        let (db, _dir) = seeded_db().await;

        let result = db
            .query_documents(&DocumentQuery {
                filters: vec![("title".into(), FieldFilter::AnyOf(vec!["Chili".into()]))],
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(paths(&result), vec!["kitchen/recipes/chili.md"]);
    }

    #[tokio::test]
    async fn all_of_on_a_scalar_column_is_unsatisfiable_not_widened() {
        // title holds one value, so requiring two is impossible. Reusing the any-of
        // IN(...) form would silently return documents matching either.
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![(
                "title".into(),
                FieldFilter::AllOf(vec!["Chili".into(), "Congee".into()]),
            )],
            ..Default::default()
        };

        assert_eq!(db.query_documents(&query).await.unwrap().total, 0);
    }

    #[tokio::test]
    async fn all_of_with_one_value_still_matches_a_scalar_column() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![("title".into(), FieldFilter::AllOf(vec!["Chili".into()]))],
            ..Default::default()
        };

        assert_eq!(
            paths(&db.query_documents(&query).await.unwrap()),
            vec!["kitchen/recipes/chili.md"]
        );
    }

    #[tokio::test]
    async fn description_is_filterable_too() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![(
                "description".into(),
                FieldFilter::AnyOf(vec!["Chili description".into()]),
            )],
            ..Default::default()
        };

        assert_eq!(
            paths(&db.query_documents(&query).await.unwrap()),
            vec!["kitchen/recipes/chili.md"]
        );
    }

    #[tokio::test]
    async fn empty_all_of_matches_nothing_like_empty_any_of() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![("tags".into(), FieldFilter::AllOf(vec![]))],
            ..Default::default()
        };

        assert_eq!(
            db.query_documents(&query).await.unwrap().total,
            0,
            "an empty required set is unsatisfiable, not absent"
        );
    }

    #[tokio::test]
    async fn deleted_documents_leave_the_listing() {
        let (db, _dir) = seeded_db().await;
        db.delete_document("sysadmin/zfs.md").await.unwrap();

        let result = db.query_documents(&DocumentQuery::default()).await.unwrap();
        assert_eq!(result.total, 3);
        assert!(!paths(&result).contains(&"sysadmin/zfs.md"));
    }

    #[tokio::test]
    async fn clear_also_wipes_document_metadata() {
        // A full reindex clears state BEFORE orphan detection reads it, so orphans are
        // never computed on a full run. If clear() left `documents` behind, a file
        // deleted from disk would remain listed forever with no run able to detect it.
        let (db, _dir) = test_db().await;
        db.upsert("gone.md", "h", 1, "").await.unwrap();
        db.upsert_document_metadata("gone.md", &recipe_frontmatter(), 1, "h", 1)
            .await
            .unwrap();

        db.clear().await.unwrap();

        assert_eq!(db.count().await.unwrap(), 0);
        assert_eq!(
            db.document_count().await.unwrap(),
            0,
            "metadata must not survive a full-reindex clear"
        );
        assert!(field_rows(&db, "gone.md").await.is_empty());
    }

    #[tokio::test]
    async fn reprojection_reads_current_data_not_a_snapshot() {
        // The whole point of re-reading inside the write transaction: a value updated
        // after the run began must be reprojected as its NEW value, never reverted.
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("r.md", &recipe_frontmatter(), 1, "h", 1)
            .await
            .unwrap();

        let mut updated = recipe_frontmatter();
        updated.insert(
            "planning".into(),
            serde_json::json!({ "prep_minutes": 99, "needs_recipe": true }),
        );
        db.upsert_document_metadata("r.md", &updated, 2, "h2", 1)
            .await
            .unwrap();

        db.reproject_all_fields().await.unwrap();

        let rows = field_rows(&db, "r.md").await;
        assert!(
            rows.contains(&("planning.prep_minutes".into(), "99".into(), Some(99.0))),
            "must reproject the current value, got: {rows:?}"
        );
    }

    #[tokio::test]
    async fn reprojection_skips_a_corrupt_row_and_finishes_the_rest() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("good.md", &recipe_frontmatter(), 1, "h", 1)
            .await
            .unwrap();
        db.upsert_document_metadata("zz-later.md", &recipe_frontmatter(), 1, "h", 1)
            .await
            .unwrap();

        // Corrupt one row's stored JSON; it sorts before the other.
        sqlx::query("UPDATE documents SET frontmatter = ? WHERE file_path = ?")
            .bind("{not json")
            .bind("good.md")
            .execute(db.pool_for_test())
            .await
            .unwrap();

        let count = db
            .reproject_all_fields()
            .await
            .expect("one bad row must not fail the whole repair run");

        assert_eq!(count, 1, "the healthy document was still reprojected");
        assert!(!field_rows(&db, "zz-later.md").await.is_empty());
    }

    #[tokio::test]
    async fn document_metadata_is_independent_of_indexed_files() {
        // The two tables track different things; an existing deployment has one
        // populated and the other empty until a backfill runs.
        let (db, _dir) = test_db().await;
        db.upsert("r.md", "h1", 2, "").await.unwrap();

        assert_eq!(db.count().await.unwrap(), 1);
        assert_eq!(db.document_count().await.unwrap(), 0);
    }
}
