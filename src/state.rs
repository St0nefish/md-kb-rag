use crate::document_fields;
use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
use sqlx::{QueryBuilder, Sqlite, SqlitePool};
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use tracing::{debug, warn};

/// Conservative cap on values bound into a single `WHERE file_path IN (...)` query.
/// SQLite's default limit is 999 bound parameters per statement; staying well under it
/// leaves room for drivers/wrappers that reserve a few, and keeps a single query from
/// dominating the connection while a large batch is chunked through it.
const SQLITE_MAX_PARAMS_PER_QUERY: usize = 500;

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
    /// Exclude documents whose `mtime` is before this Unix timestamp.
    pub mtime_after: Option<i64>,
    /// Exclude documents whose `mtime` is after this Unix timestamp.
    pub mtime_before: Option<i64>,
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
            mtime_after: None,
            mtime_before: None,
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

    /// Every indexed document's file path, unfiltered and unordered.
    ///
    /// Backs `retrieval::get_document`'s fuzzy fallback: basename matching and
    /// "did you mean?" suggestions need the full path list, and `documents` already
    /// holds exactly that — one row per successfully indexed file — so this is a
    /// plain local `SELECT`, no cap and no round trip to the vector store required.
    async fn all_paths(&self) -> Result<Vec<String>>;

    /// Document summaries for exactly these paths, in unspecified order.
    ///
    /// Backs `retrieval::search`'s query+document (grouped) mode: Qdrant's
    /// `query_groups` collapses each document to its single best-scoring chunk, whose
    /// payload carries only that chunk's fields — not the whole-document metadata
    /// (`title`/`description`/`mtime`/full `frontmatter`) a document-shaped result
    /// needs. The caller re-associates each returned summary with its Qdrant score by
    /// `file_path` and restores the score-descending order itself; this method makes
    /// no ordering promise and may return fewer rows than `paths` if the metadata
    /// index is transiently behind Qdrant (same consistency caveat as
    /// `get_document`'s fuzzy fallback).
    async fn get_summaries_by_paths(
        &self,
        paths: &[String],
        fields: Option<&[String]>,
    ) -> Result<Vec<DocumentSummary>>;
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
    /// File modification time (Unix seconds) as of the last successful index of this
    /// file, or the last stat-only refresh of an unchanged file (`process_file`'s skip
    /// path via `StateDb::update_stat`, #139). The reconcile scan
    /// (`ingest::scan_for_dirty`) reads these two columns directly via [`ScanRow`]
    /// rather than through this struct; `index_paths`'s scoped scan reads them via
    /// `get_many` to decide whether a skipped file's stat baseline needs refreshing.
    pub mtime: i64,
    /// File size in bytes as of the last successful index or stat-only refresh, same
    /// purpose and caveat as `mtime`. `-1` for rows written before this column existed,
    /// which never matches a real size and so forces one stat-mismatch (and therefore a
    /// content-hash check) per file on the first reconcile scan after upgrading.
    pub size: i64,
}

/// One row from [`StateDb::scan_indexed_files`]: an `indexed_files` row plus the
/// `documents.content_hash` for the same path, if a metadata row exists at all.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct ScanRow {
    pub file_path: String,
    pub content_hash: String,
    pub schema_hash: String,
    pub mtime: i64,
    pub size: i64,
    /// `documents.content_hash` for this path, or `None` if no metadata row exists yet.
    pub doc_hash: Option<String>,
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

        // Stat-based pre-filter columns for the reconcile scan (`ingest::scan_for_dirty`):
        // at corpus sizes in the thousands-to-tens-of-thousands range, content-hashing
        // every file on every sweep is too expensive, so the scan compares mtime/size
        // first and only asks the scoped indexer to open (and therefore hash) a file
        // when one of them changed. `size` defaults to -1, which never matches a real
        // file size, so every row written before this column existed gets one forced
        // stat-mismatch (and therefore a real hash check) on the first post-upgrade scan.
        add_column_if_missing(
            &pool,
            "indexed_files",
            "mtime",
            "INTEGER NOT NULL DEFAULT 0",
        )
        .await?;
        add_column_if_missing(
            &pool,
            "indexed_files",
            "size",
            "INTEGER NOT NULL DEFAULT -1",
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

        // `query_documents` supports ordering by mtime/title/indexed_at, each with a
        // mandatory `d.file_path ASC` tiebreaker (ties are common: a batch reindex
        // gives many rows the same mtime/indexed_at). Without an index shaped to match,
        // SQLite must materialize and sort the whole filtered set before LIMIT/OFFSET
        // can trim it — `CREATE INDEX IF NOT EXISTS` is already idempotent against an
        // existing production database, same as the document_fields indexes below.
        //
        // A single ASC-ordered composite index does not cover DESC queries: SQLite can
        // only satisfy an ORDER BY straight from an index when the whole key (including
        // the tiebreaker) matches the index's declared direction, or is its exact
        // reverse. `(col ASC, file_path ASC)` scanned backward yields `(col DESC,
        // file_path DESC)` — not what these queries ask for, since the tiebreaker is
        // always ASC. So each orderable column needs one index per direction; verified
        // with `EXPLAIN QUERY PLAN` (see the `query_documents_order_by_uses_index`
        // tests) that omitting either half still falls back to `USE TEMP B-TREE FOR
        // ORDER BY`. `file_path` itself needs no such pair: it is the primary key, and
        // a single-column index can satisfy both directions by scanning in reverse.
        for (name, column) in [
            ("idx_documents_mtime_asc", "mtime ASC"),
            ("idx_documents_mtime_desc", "mtime DESC"),
            ("idx_documents_title_asc", "title ASC"),
            ("idx_documents_title_desc", "title DESC"),
            ("idx_documents_indexed_at_asc", "indexed_at ASC"),
            ("idx_documents_indexed_at_desc", "indexed_at DESC"),
        ] {
            sqlx::query(&format!(
                "CREATE INDEX IF NOT EXISTS {name} ON documents({column}, file_path ASC)"
            ))
            .execute(&pool)
            .await?;
        }

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

        // Edges for the KB web UI's graph view: markdown links extracted at ingest
        // (`kind = 'markdown'`) and semantic kNN neighbors precomputed at index time
        // (`kind = 'semantic'`, carrying a similarity `score`). No `REFERENCES ...
        // documents(file_path)` / `ON DELETE CASCADE` here — unlike document_fields,
        // a link's target need not exist yet (or ever) as a `documents` row: a link may
        // point at a file that hasn't been indexed yet, or one that was renamed out from
        // under it; the graph handler drops dangling edges at read time instead. Rows
        // are removed explicitly via `delete_links_for` (called from `delete_document`)
        // and replaced wholesale per (source_path, kind) via `replace_links`.
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS document_links (
                source_path TEXT NOT NULL,
                target_path TEXT NOT NULL,
                kind        TEXT NOT NULL DEFAULT 'markdown',
                score       REAL,
                PRIMARY KEY (source_path, target_path, kind)
            )",
        )
        .execute(&pool)
        .await?;

        // The reverse lookup — "what points at this target" — which the primary key
        // (source-first) does not serve. `delete_links_for` does NOT need this: it
        // purges a renamed/deleted file's OWN outgoing rows, filtered by
        // `source_path`, which the primary key already serves directly. The live user
        // of this index is `links_targeting`, which answers "which documents must be
        // updated when this path moves" for the document-move link rewriter.
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_document_links_target
             ON document_links(target_path)",
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
            "SELECT file_path, content_hash, chunk_count, indexed_at, schema_hash, mtime, size
             FROM indexed_files WHERE file_path = ?",
        )
        .bind(file_path)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn upsert(
        &self,
        file_path: &str,
        content_hash: &str,
        chunk_count: i64,
        schema_hash: &str,
        mtime: i64,
        size: i64,
    ) -> Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO indexed_files
                (file_path, content_hash, chunk_count, indexed_at, schema_hash, mtime, size)
             VALUES (?, ?, ?, datetime('now'), ?, ?, ?)",
        )
        .bind(file_path)
        .bind(content_hash)
        .bind(chunk_count)
        .bind(schema_hash)
        .bind(mtime)
        .bind(size)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Refresh only the stat pre-filter baseline (`mtime`/`size`) for an already-indexed
    /// row, leaving `content_hash`/`chunk_count`/`schema_hash`/`indexed_at` untouched.
    ///
    /// Exists for the "content unchanged, mtime moved" case (#139): `process_file`'s
    /// skip path reads a fresh mtime/size but has no changed content to justify a full
    /// [`Self::upsert`] (which would also bump `indexed_at`, misrepresenting the file as
    /// just-reindexed). A no-op `UPDATE` still costs a write, so the caller is expected
    /// to compare against the stored value first and only call this when it differs —
    /// otherwise every skip on every scan would rewrite the row it exists to avoid
    /// rewriting.
    pub async fn update_stat(&self, file_path: &str, mtime: i64, size: i64) -> Result<()> {
        sqlx::query("UPDATE indexed_files SET mtime = ?, size = ? WHERE file_path = ?")
            .bind(mtime)
            .bind(size)
            .bind(file_path)
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
            "SELECT file_path, content_hash, chunk_count, indexed_at, schema_hash, mtime, size
             FROM indexed_files ORDER BY file_path",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// One page of `indexed_files` joined with `documents`, ordered by path.
    ///
    /// Used only by the reconcile scan (`ingest::scan_for_dirty`), which needs to
    /// interleave a `tokio::fs::metadata` stat call (async) between rows — a plain
    /// callback-based "visit each row" API cannot do that without either boxing
    /// futures or blocking the executor, so the scan owns its own paging loop and
    /// calls this once per page instead. At corpus sizes in the thousands-to-tens-of-
    /// thousands range, a single `list_all()`-style `Vec<IndexedFile>` covering the
    /// whole table would be a multi-megabyte allocation held for the whole scan; paging
    /// keeps peak memory bounded by `limit` regardless of corpus size. `LIMIT ...
    /// OFFSET` is good enough here — a row inserted or deleted by a concurrent write
    /// mid-scan can shift a page and be seen twice or missed once, but the scan is a
    /// best-effort detector with a self-healing follow-up (the next scheduled
    /// reconcile), not the system of record.
    ///
    /// `documents.content_hash` is carried along as `doc_hash` so the caller can detect
    /// metadata staleness — a file whose content is unchanged but whose `documents` row
    /// is missing or stale relative to `indexed_files.content_hash` — without a second
    /// query per row.
    pub async fn fetch_indexed_files_page(&self, limit: i64, offset: i64) -> Result<Vec<ScanRow>> {
        let rows: Vec<ScanRow> = sqlx::query_as(
            "SELECT i.file_path, i.content_hash, i.schema_hash, i.mtime, i.size, \
                    d.content_hash AS doc_hash \
             FROM indexed_files i LEFT JOIN documents d ON d.file_path = i.file_path \
             ORDER BY i.file_path LIMIT ? OFFSET ?",
        )
        .bind(limit.max(1))
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// Batch equivalent of [`Self::get`], for a caller that already knows exactly which
    /// paths it cares about (the scoped indexer's dirty-path list) and wants one or a
    /// few round trips instead of one per path. Chunks the `IN (...)` list so a large
    /// batch cannot exceed SQLite's bound-parameter limit (default 999).
    pub async fn get_many(&self, file_paths: &[String]) -> Result<HashMap<String, IndexedFile>> {
        let mut out = HashMap::with_capacity(file_paths.len());
        for chunk in file_paths.chunks(SQLITE_MAX_PARAMS_PER_QUERY) {
            if chunk.is_empty() {
                continue;
            }
            let mut builder = QueryBuilder::<Sqlite>::new(
                "SELECT file_path, content_hash, chunk_count, indexed_at, schema_hash, mtime, size \
                 FROM indexed_files WHERE file_path IN (",
            );
            let mut separated = builder.separated(", ");
            for path in chunk {
                separated.push_bind(path);
            }
            builder.push(")");
            let rows: Vec<IndexedFile> = builder.build_query_as().fetch_all(&self.pool).await?;
            for row in rows {
                out.insert(row.file_path.clone(), row);
            }
        }
        Ok(out)
    }

    /// Batch equivalent of [`Self::get_document_hash`], for the same reason as
    /// [`Self::get_many`].
    pub async fn get_document_hashes_many(
        &self,
        file_paths: &[String],
    ) -> Result<HashMap<String, String>> {
        let mut out = HashMap::with_capacity(file_paths.len());
        for chunk in file_paths.chunks(SQLITE_MAX_PARAMS_PER_QUERY) {
            if chunk.is_empty() {
                continue;
            }
            let mut builder = QueryBuilder::<Sqlite>::new(
                "SELECT file_path, content_hash FROM documents WHERE file_path IN (",
            );
            let mut separated = builder.separated(", ");
            for path in chunk {
                separated.push_bind(path);
            }
            builder.push(")");
            let rows: Vec<(String, String)> =
                builder.build_query_as().fetch_all(&self.pool).await?;
            out.extend(rows);
        }
        Ok(out)
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

    /// Remove a document's metadata. Projection rows cascade via the foreign key;
    /// `document_links` does not (its targets may not be `documents` rows at all), so
    /// its outgoing edges are cleared explicitly here.
    pub async fn delete_document(&self, file_path: &str) -> Result<()> {
        sqlx::query("DELETE FROM documents WHERE file_path = ?")
            .bind(file_path)
            .execute(&self.pool)
            .await?;

        self.delete_links_for(file_path).await?;

        Ok(())
    }

    /// Replace every `document_links` row for `(source_path, kind)` with `targets`.
    ///
    /// Scoped to `kind` rather than the whole `source_path`, so refreshing one edge
    /// kind (e.g. recomputed semantic neighbors) never touches the other (markdown
    /// links extracted from the same file's body). Runs as delete-then-insert inside a
    /// transaction so a reader never observes a partially-replaced edge set.
    pub async fn replace_links(
        &self,
        source_path: &str,
        kind: &str,
        targets: &[(String, Option<f64>)],
    ) -> Result<()> {
        let mut tx = self.pool.begin().await?;

        sqlx::query("DELETE FROM document_links WHERE source_path = ? AND kind = ?")
            .bind(source_path)
            .bind(kind)
            .execute(&mut *tx)
            .await?;

        for (target_path, score) in targets {
            sqlx::query(
                "INSERT OR IGNORE INTO document_links (source_path, target_path, kind, score)
                 VALUES (?, ?, ?, ?)",
            )
            .bind(source_path)
            .bind(target_path)
            .bind(kind)
            .bind(score)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    /// Every edge in the graph, unfiltered: (source_path, target_path, kind, score).
    ///
    /// Backs `GET /api/graph`, which filters dangling edges (targets not among the
    /// current node set) itself — this is a plain unlimited read, same rationale as
    /// [`DocumentIndex::all_paths`].
    pub async fn all_links(&self) -> Result<Vec<(String, String, String, Option<f64>)>> {
        let rows: Vec<(String, String, String, Option<f64>)> =
            sqlx::query_as("SELECT source_path, target_path, kind, score FROM document_links")
                .fetch_all(&self.pool)
                .await?;

        Ok(rows)
    }

    /// Distinct source paths whose `document_links` rows target `target_path`, scoped
    /// to `kind` (callers pass `"markdown"` to find only real inline-link edges, not
    /// precomputed `"semantic"` neighbors).
    ///
    /// Backs `write::write_document_move`'s "which documents must be updated"
    /// question: when `target_path` moves, every source this returns has a body that
    /// needs `find_markdown_link_occurrences` re-run and its link text rewritten. Uses
    /// `idx_document_links_target`, unlike every other query on this table, which is
    /// source-first and served by the primary key.
    pub async fn links_targeting(&self, target_path: &str, kind: &str) -> Result<Vec<String>> {
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT DISTINCT source_path FROM document_links
             WHERE target_path = ? AND kind = ?
             ORDER BY source_path",
        )
        .bind(target_path)
        .bind(kind)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|(path,)| path).collect())
    }

    /// Batch equivalent of [`Self::links_targeting`], for a caller that already knows
    /// every target it cares about up front (`write::move_directory`'s reverse-link
    /// scan, run once for every moved document) and wants one or a few round trips
    /// instead of one `links_targeting` call per target.
    ///
    /// Returns every `(target_path, source_path)` pair among `target_paths`, scoped to
    /// `kind` exactly like `links_targeting`, grouped by target: a target with no
    /// referencing sources simply has no key in the returned map (mirroring
    /// `links_targeting`'s empty-`Vec` result for the same case), and each target's
    /// sources come back distinct and sorted, matching `links_targeting`'s own
    /// `SELECT DISTINCT ... ORDER BY source_path`.
    ///
    /// Chunks `target_paths` at [`SQLITE_MAX_PARAMS_PER_QUERY`] (500) bound
    /// parameters per statement, same as [`Self::get_many`] — SQLite's default limit is
    /// 999 total bound parameters per statement, and each chunk here also binds `kind`,
    /// so 500 leaves comfortable headroom under that ceiling for a single query.
    pub async fn links_targeting_many(
        &self,
        target_paths: &[String],
        kind: &str,
    ) -> Result<HashMap<String, Vec<String>>> {
        let mut out: HashMap<String, Vec<String>> = HashMap::new();
        for chunk in target_paths.chunks(SQLITE_MAX_PARAMS_PER_QUERY) {
            if chunk.is_empty() {
                continue;
            }
            let mut builder = QueryBuilder::<Sqlite>::new(
                "SELECT DISTINCT target_path, source_path FROM document_links WHERE target_path IN (",
            );
            let mut separated = builder.separated(", ");
            for path in chunk {
                separated.push_bind(path);
            }
            builder.push(") AND kind = ");
            builder.push_bind(kind);
            builder.push(" ORDER BY target_path, source_path");
            let rows: Vec<(String, String)> =
                builder.build_query_as().fetch_all(&self.pool).await?;
            for (target_path, source_path) in rows {
                out.entry(target_path).or_default().push(source_path);
            }
        }
        Ok(out)
    }

    /// Remove every edge originating from `path`, in either direction of kind.
    ///
    /// Called when a document is deleted/purged (from [`Self::delete_document`]) so a
    /// removed file's outgoing links do not linger. Deliberately does not also delete
    /// edges where `path` is the *target* — those become dangling edges that
    /// `GET /api/graph` drops at read time, which is the same self-healing behavior a
    /// renamed target gets, rather than a special case here.
    pub async fn delete_links_for(&self, path: &str) -> Result<()> {
        sqlx::query("DELETE FROM document_links WHERE source_path = ?")
            .bind(path)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// Every document's summary metadata, unfiltered, unpaged, ordered by path.
    ///
    /// Backs `GET /api/graph`'s node listing, which needs every document in one shot —
    /// unlike [`DocumentIndex::query_documents`], which is built around a caller-facing
    /// `limit`/`offset` page. Same row shape and frontmatter-parsing behavior (a
    /// malformed frontmatter JSON blob degrades to an empty object rather than failing
    /// the whole listing) as `query_documents`'s mapping.
    pub async fn all_document_summaries(&self) -> Result<Vec<DocumentSummary>> {
        let rows: Vec<DocumentRow> = sqlx::query_as(
            "SELECT file_path, title, description, mtime, indexed_at, frontmatter
             FROM documents ORDER BY file_path",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(
                |(file_path, title, description, mtime, indexed_at, frontmatter_json)| {
                    let frontmatter: Value = serde_json::from_str(&frontmatter_json)
                        .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
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
            .collect())
    }

    /// Content hash of every document that has metadata, keyed by path.
    ///
    /// The scoped indexer (`ingest::index_paths`) now uses [`Self::get_document_hashes_many`]
    /// instead, scoped to just the paths it is processing — loading every document's
    /// hash at once does not scale to a large corpus. This whole-table equivalent is
    /// kept as a general-purpose `StateDb` accessor (parallel to [`Self::list_all`])
    /// rather than removed with its test coverage; `#[allow(dead_code)]` because
    /// nothing in production calls it now.
    #[allow(dead_code)]
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

        if let Some(after) = query.mtime_after {
            builder.push(" AND d.mtime >= ");
            builder.push_bind(after);
        }
        if let Some(before) = query.mtime_before {
            builder.push(" AND d.mtime <= ");
            builder.push_bind(before);
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

    /// Fields worth breaking down, with how many distinct values each takes,
    /// widest document coverage first.
    ///
    /// Two decisions matter here, and both are in SQL because a caller filtering
    /// afterwards would get the wrong rows:
    ///
    /// - `max_distinct` bounds *value* cardinality. A field with three values
    ///   (`status`) is a useful histogram; one with several hundred is noise.
    /// - Ordering is by how many documents carry the field, not by how many values it
    ///   takes. A scoped schema can declare twenty fields that apply to one folder;
    ///   ordering by value count floats all of them above `type` and `status`, and the
    ///   fields that describe the whole knowledge base fall off the end of `limit`.
    pub async fn breakdown_fields(
        &self,
        max_distinct: i64,
        limit: i64,
    ) -> Result<Vec<(String, i64)>> {
        // SQLite reads a negative LIMIT as "no limit".
        let limit = limit.max(0);
        let rows: Vec<(String, i64, i64)> = sqlx::query_as(
            "SELECT field, COUNT(DISTINCT value_text) AS n, COUNT(DISTINCT file_path) AS docs \
             FROM document_fields GROUP BY field HAVING n <= ? \
             ORDER BY docs DESC, n DESC, field ASC LIMIT ?",
        )
        .bind(max_distinct)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|(f, n, _docs)| (f, n)).collect())
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

    async fn all_paths(&self) -> Result<Vec<String>> {
        let rows: Vec<(String,)> = sqlx::query_as("SELECT file_path FROM documents")
            .fetch_all(&self.pool)
            .await?;
        Ok(rows.into_iter().map(|(file_path,)| file_path).collect())
    }

    async fn get_summaries_by_paths(
        &self,
        paths: &[String],
        fields: Option<&[String]>,
    ) -> Result<Vec<DocumentSummary>> {
        let mut out = Vec::with_capacity(paths.len());
        for chunk in paths.chunks(SQLITE_MAX_PARAMS_PER_QUERY) {
            if chunk.is_empty() {
                continue;
            }
            let mut builder: QueryBuilder<Sqlite> = QueryBuilder::new(
                "SELECT file_path, title, description, mtime, indexed_at, frontmatter \
                 FROM documents WHERE file_path IN (",
            );
            let mut separated = builder.separated(", ");
            for path in chunk {
                separated.push_bind(path);
            }
            builder.push(")");

            let rows: Vec<DocumentRow> = builder.build_query_as().fetch_all(&self.pool).await?;
            out.extend(rows.into_iter().map(
                |(file_path, title, description, mtime, indexed_at, frontmatter_json)| {
                    let frontmatter: Value = serde_json::from_str(&frontmatter_json)
                        .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
                    let frontmatter = match fields {
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
            ));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qdrant;
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
        db.upsert("test.md", "abc123", 3, "", 0, 0).await.unwrap();
        let entry = db.get("test.md").await.unwrap().unwrap();
        assert_eq!(entry.file_path, "test.md");
        assert_eq!(entry.content_hash, "abc123");
        assert_eq!(entry.chunk_count, 3);
    }

    #[tokio::test]
    async fn upsert_replaces() {
        let (db, _dir) = test_db().await;
        db.upsert("test.md", "hash1", 2, "", 0, 0).await.unwrap();
        db.upsert("test.md", "hash2", 5, "", 0, 0).await.unwrap();
        let entry = db.get("test.md").await.unwrap().unwrap();
        assert_eq!(entry.content_hash, "hash2");
        assert_eq!(entry.chunk_count, 5);
    }

    #[tokio::test]
    async fn delete_removes() {
        let (db, _dir) = test_db().await;
        db.upsert("test.md", "hash", 1, "", 0, 0).await.unwrap();
        db.delete("test.md").await.unwrap();
        assert!(db.get("test.md").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn list_and_count() {
        let (db, _dir) = test_db().await;
        db.upsert("a.md", "h1", 1, "", 0, 0).await.unwrap();
        db.upsert("b.md", "h2", 2, "", 0, 0).await.unwrap();
        assert_eq!(db.count().await.unwrap(), 2);
        let all = db.list_all().await.unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].file_path, "a.md"); // sorted
    }

    #[tokio::test]
    async fn clear_removes_all() {
        let (db, _dir) = test_db().await;
        db.upsert("a.md", "h1", 1, "", 0, 0).await.unwrap();
        db.upsert("b.md", "h2", 2, "", 0, 0).await.unwrap();
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
        db.upsert("doc.md", "hash1", 3, "", 0, 0).await.unwrap();

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

        db.upsert("a.md", "hash-a2", 3, "fingerprint", 0, 0)
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
        db.upsert("new.md", "h", 1, "fp", 0, 0).await.unwrap();
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
    async fn breakdown_fields_reports_distinct_value_counts() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;

        let cards: HashMap<String, i64> = db
            .breakdown_fields(50, 50)
            .await
            .unwrap()
            .into_iter()
            .collect();
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
    async fn breakdown_fields_excludes_high_cardinality_fields_in_sql() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;

        // `type` and `tags` each take 3 distinct values. With the ceiling at 2, both
        // drop — the filter is applied in SQL, not by the caller.
        let fields: Vec<String> = db
            .breakdown_fields(2, 50)
            .await
            .unwrap()
            .into_iter()
            .map(|(f, _)| f)
            .collect();
        assert!(!fields.contains(&"type".to_string()), "{fields:?}");
        assert!(!fields.contains(&"tags".to_string()), "{fields:?}");
    }

    #[tokio::test]
    async fn breakdown_fields_limit_keeps_usable_fields_not_the_widest() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;
        // A field far too wide to be a useful histogram.
        for i in 0..40 {
            let fm: HashMap<String, Value> = match serde_json::json!({
                "title": "T", "type": "guide", "serial": format!("sn-{i}")
            }) {
                Value::Object(m) => m.into_iter().collect(),
                _ => unreachable!(),
            };
            db.upsert_document_metadata(&format!("wide/{i}.md"), &fm, 1700, "h", 1)
                .await
                .unwrap();
        }

        // Rows come back richest-first, so an unfiltered `ORDER BY n DESC LIMIT 1`
        // would return `serial` — and every reportable field would be lost to the cap.
        let fields: Vec<String> = db
            .breakdown_fields(10, 1)
            .await
            .unwrap()
            .into_iter()
            .map(|(f, _)| f)
            .collect();
        assert_eq!(fields.len(), 1);
        assert!(
            !fields.contains(&"serial".to_string()),
            "the 40-value field must not consume the only slot: {fields:?}"
        );
        // `tags` and `type` both take 3 values and tie; either is a usable histogram.
        assert!(
            fields[0] == "tags" || fields[0] == "type",
            "expected a low-cardinality field, got {fields:?}"
        );
    }

    #[tokio::test]
    async fn breakdown_fields_orders_by_document_coverage_not_value_count() {
        let (db, _dir) = test_db().await;
        // `type` covers every document with few values; a scoped field covers a handful
        // of documents with many. Ordering by value count would float the scoped field
        // above the one that describes the whole corpus.
        for i in 0..30 {
            let mut fm: HashMap<String, Value> = match serde_json::json!({"title": "T", "type": "guide"})
            {
                Value::Object(m) => m.into_iter().collect(),
                _ => unreachable!(),
            };
            if i < 5 {
                fm.insert("scoped".into(), Value::String(format!("v{i}")));
            }
            db.upsert_document_metadata(&format!("d{i}.md"), &fm, 1700, "h", 1)
                .await
                .unwrap();
        }

        let fields: Vec<String> = db
            .breakdown_fields(500, 10)
            .await
            .unwrap()
            .into_iter()
            .map(|(f, _)| f)
            .collect();
        assert_eq!(
            fields.first().map(String::as_str),
            Some("type"),
            "the field covering every document must lead: {fields:?}"
        );
        assert!(fields.contains(&"scoped".to_string()), "{fields:?}");
    }

    #[tokio::test]
    async fn breakdown_fields_clamps_a_negative_limit() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;
        assert!(db.breakdown_fields(50, -1).await.unwrap().is_empty());
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
        assert!(db.breakdown_fields(50, 50).await.unwrap().is_empty());
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

    // ------------------------------------------------------------------
    // query mode / enumeration mode: filter semantics must agree
    // ------------------------------------------------------------------
    //
    // `search`'s query-mode filters lower to Qdrant conditions via
    // `qdrant::lower_field_filters`; its enumeration-mode filters lower to SQL
    // via `push_where`, exercised here through the real `query_documents` path.
    // No live Qdrant is available in this environment to run the lowered
    // conditions for real (see `qdrant::tests::qdrant_filters_agree_with_the_offline_prediction`
    // for the `#[ignore]`d live counterpart), so this section evaluates the
    // lowered `Condition` tree directly against the same frontmatter this
    // SQLite fixture was seeded from, using a small interpreter that only needs
    // to understand the shapes `lower_field_filters` can actually produce
    // (`Match`/`Range` field conditions, and a `Filter::should` OR for
    // multi-value boolean equality) — anything else means this test's
    // evaluator is out of date with production, so it panics rather than
    // silently passing.

    fn equivalence_frontmatter(
        type_: &str,
        tags: &[&str],
        prep_minutes: i64,
        tested: bool,
        rating: f64,
    ) -> HashMap<String, Value> {
        let json = serde_json::json!({
            "title": type_,
            "type": type_,
            "tags": tags,
            "rating": rating,
            "planning": { "prep_minutes": prep_minutes, "tested": tested }
        });
        match json {
            Value::Object(map) => map.into_iter().collect(),
            _ => unreachable!(),
        }
    }

    fn equivalence_docs() -> Vec<(&'static str, HashMap<String, Value>)> {
        vec![
            (
                "a.md",
                equivalence_frontmatter("guide", &["recipe", "dinner"], 20, true, 4.5),
            ),
            (
                "b.md",
                equivalence_frontmatter("recipe", &["recipe", "breakfast"], 45, true, 3.0),
            ),
            (
                "c.md",
                equivalence_frontmatter("reference", &["recipe", "dinner", "zfs"], 45, false, 4.5),
            ),
            (
                "d.md",
                equivalence_frontmatter("guide", &["zfs"], 15, true, 2.0),
            ),
        ]
    }

    async fn equivalence_db() -> (StateDb, TempDir) {
        let (db, dir) = test_db().await;
        for (path, fm) in equivalence_docs() {
            db.upsert_document_metadata(path, &fm, 100, "h", 1)
                .await
                .unwrap();
        }
        (db, dir)
    }

    /// Navigate a dot-path into a JSON object the same way Qdrant addresses a
    /// nested payload field natively.
    fn payload_at<'a>(payload: &'a Value, key: &str) -> Option<&'a Value> {
        let mut cur = payload;
        for part in key.split('.') {
            cur = cur.as_object()?.get(part)?;
        }
        Some(cur)
    }

    fn qdrant_range_matches(val: &Value, range: &qdrant_client::qdrant::Range) -> bool {
        let n = match val {
            Value::Number(n) => n.as_f64(),
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        };
        let Some(n) = n else { return false };
        if let Some(gte) = range.gte
            && n < gte
        {
            return false;
        }
        if let Some(lte) = range.lte
            && n > lte
        {
            return false;
        }
        if let Some(gt) = range.gt
            && n <= gt
        {
            return false;
        }
        if let Some(lt) = range.lt
            && n >= lt
        {
            return false;
        }
        true
    }

    fn qdrant_match_matches(val: &Value, m: &qdrant_client::qdrant::Match) -> bool {
        use qdrant_client::qdrant::r#match::MatchValue;
        let Some(mv) = &m.match_value else {
            return false;
        };
        // Qdrant matches an array-valued payload field by checking whether any
        // element matches, not by comparing the whole array as one value.
        let candidates: Vec<&Value> = match val {
            Value::Array(items) => items.iter().collect(),
            other => vec![other],
        };
        candidates.into_iter().any(|c| match mv {
            MatchValue::Keyword(s) => c.as_str() == Some(s.as_str()),
            MatchValue::Integer(i) => c.as_i64() == Some(*i),
            MatchValue::Boolean(b) => c.as_bool() == Some(*b),
            MatchValue::Keywords(ks) => ks.strings.iter().any(|s| c.as_str() == Some(s.as_str())),
            MatchValue::Integers(is) => is.integers.iter().any(|i| c.as_i64() == Some(*i)),
            other => panic!(
                "equivalence test's evaluator does not model match variant {other:?} — \
                 lower_field_filters should not be producing this shape"
            ),
        })
    }

    fn qdrant_condition_matches(cond: &qdrant_client::qdrant::Condition, payload: &Value) -> bool {
        use qdrant_client::qdrant::condition::ConditionOneOf;
        match cond
            .condition_one_of
            .as_ref()
            .expect("every lowered condition sets one variant")
        {
            ConditionOneOf::Field(fc) => {
                let Some(val) = payload_at(payload, &fc.key) else {
                    return false;
                };
                if let Some(range) = &fc.range {
                    qdrant_range_matches(val, range)
                } else if let Some(m) = &fc.r#match {
                    qdrant_match_matches(val, m)
                } else {
                    panic!(
                        "equivalence test's evaluator only models match/range field \
                         conditions"
                    )
                }
            }
            ConditionOneOf::Filter(f) => {
                // The only `Filter` shape `lower_field_filters` produces is a bare
                // `should` OR (multi-value boolean equality) — must/must_not/
                // min_should here would mean this evaluator is out of date.
                assert!(
                    f.must.is_empty() && f.must_not.is_empty() && f.min_should.is_none(),
                    "unexpected Filter shape from lower_field_filters: {f:?}"
                );
                f.should
                    .iter()
                    .any(|c| qdrant_condition_matches(c, payload))
            }
            other => panic!(
                "equivalence test's evaluator does not model condition variant {other:?} — \
                 lower_field_filters should not be producing this shape"
            ),
        }
    }

    /// Seed [`equivalence_db`], run `filters` through both the real
    /// enumeration-mode SQL path (`StateDb::query_documents`) and
    /// `qdrant::lower_field_filters` (evaluated offline against the same
    /// fixture's frontmatter — see this section's header comment), then assert
    /// both name exactly the same document set.
    async fn assert_query_and_enumeration_modes_agree(
        filters: Vec<(String, FieldFilter)>,
        indexed: HashMap<String, qdrant::IndexKind>,
        expected: &[&str],
    ) {
        let (db, _dir) = equivalence_db().await;

        let sql_result = db
            .query_documents(&DocumentQuery {
                filters: filters.clone(),
                ..Default::default()
            })
            .await
            .unwrap();
        let mut sql_paths: Vec<&str> = sql_result
            .documents
            .iter()
            .map(|d| d.file_path.as_str())
            .collect();
        sql_paths.sort();

        let conditions = qdrant::lower_field_filters(&filters, &indexed)
            .expect("query mode should accept this filter");
        let mut qdrant_paths: Vec<&str> = equivalence_docs()
            .into_iter()
            .filter(|(_, fm)| {
                let payload = Value::Object(fm.clone().into_iter().collect());
                conditions
                    .iter()
                    .all(|c| qdrant_condition_matches(c, &payload))
            })
            .map(|(path, _)| path)
            .collect();
        qdrant_paths.sort();

        let mut expected_sorted: Vec<&str> = expected.to_vec();
        expected_sorted.sort();

        assert_eq!(
            sql_paths, expected_sorted,
            "enumeration mode (SQL) produced a different document set than expected"
        );
        assert_eq!(
            qdrant_paths, expected_sorted,
            "query mode (Qdrant conditions) produced a different document set than expected"
        );
    }

    #[tokio::test]
    async fn query_and_enumeration_agree_on_scalar_equality() {
        assert_query_and_enumeration_modes_agree(
            vec![("type".into(), FieldFilter::AnyOf(vec!["guide".into()]))],
            HashMap::from([("type".to_string(), qdrant::IndexKind::Keyword)]),
            &["a.md", "d.md"],
        )
        .await;
    }

    #[tokio::test]
    async fn query_and_enumeration_agree_on_any_of() {
        assert_query_and_enumeration_modes_agree(
            vec![(
                "tags".into(),
                FieldFilter::AnyOf(vec!["breakfast".into(), "zfs".into()]),
            )],
            HashMap::from([("tags".to_string(), qdrant::IndexKind::Keyword)]),
            &["b.md", "c.md", "d.md"],
        )
        .await;
    }

    #[tokio::test]
    async fn query_and_enumeration_agree_on_all_of() {
        assert_query_and_enumeration_modes_agree(
            vec![(
                "tags".into(),
                FieldFilter::AllOf(vec!["recipe".into(), "dinner".into()]),
            )],
            HashMap::from([("tags".to_string(), qdrant::IndexKind::Keyword)]),
            &["a.md", "c.md"],
        )
        .await;
    }

    #[tokio::test]
    async fn query_and_enumeration_agree_on_numeric_range() {
        assert_query_and_enumeration_modes_agree(
            vec![(
                "planning.prep_minutes".into(),
                FieldFilter::Range {
                    gte: Some(10.0),
                    lte: None,
                    gt: None,
                    lt: Some(45.0),
                },
            )],
            HashMap::from([(
                "planning.prep_minutes".to_string(),
                qdrant::IndexKind::Integer,
            )]),
            &["a.md", "d.md"],
        )
        .await;
    }

    #[tokio::test]
    async fn query_and_enumeration_agree_on_a_nested_dot_path_field() {
        assert_query_and_enumeration_modes_agree(
            vec![(
                "planning.tested".into(),
                FieldFilter::AnyOf(vec!["true".into()]),
            )],
            HashMap::from([("planning.tested".to_string(), qdrant::IndexKind::Bool)]),
            &["a.md", "b.md", "d.md"],
        )
        .await;
    }

    #[tokio::test]
    async fn query_mode_rejects_float_equality_that_enumeration_mode_accepts() {
        // Deliberate divergence, not a bug: exact float equality is unreliable,
        // so query mode refuses it outright (`lower_field_filters` -> `Err`)
        // while enumeration mode's SQL just does a text comparison and accepts
        // it. Pin both halves so a future change can't silently narrow or widen
        // either side without this test noticing.
        let (db, _dir) = equivalence_db().await;
        let filters = vec![("rating".to_string(), FieldFilter::AnyOf(vec!["4.5".into()]))];

        let sql_result = db
            .query_documents(&DocumentQuery {
                filters: filters.clone(),
                ..Default::default()
            })
            .await
            .unwrap();
        let mut sql_paths: Vec<&str> = sql_result
            .documents
            .iter()
            .map(|d| d.file_path.as_str())
            .collect();
        sql_paths.sort();
        assert_eq!(
            sql_paths,
            vec!["a.md", "c.md"],
            "enumeration mode must still allow float equality"
        );

        let indexed = HashMap::from([("rating".to_string(), qdrant::IndexKind::Float)]);
        let err = qdrant::lower_field_filters(&filters, &indexed).unwrap_err();
        assert!(
            err.contains("float"),
            "query mode must reject float equality; got: {err}"
        );
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

    /// Rebuild the exact page query `query_documents` runs for a given `DocumentQuery`
    /// — same `push_where`, same `ORDER BY` construction — prefixed with `EXPLAIN QUERY
    /// PLAN`, and return the concatenated `detail` column of the plan.
    ///
    /// `query_documents` doesn't expose its SQL, so this mirrors it exactly rather than
    /// re-deriving a plan from first principles: a mismatch here would mean the test
    /// stopped testing what production runs, not that the index is wrong.
    async fn explain_page_query(db: &StateDb, query: &DocumentQuery) -> String {
        let mut builder: QueryBuilder<Sqlite> = QueryBuilder::new(
            "EXPLAIN QUERY PLAN SELECT d.file_path, d.title, d.description, d.mtime, \
             d.indexed_at, d.frontmatter FROM documents d",
        );
        StateDb::push_where(&mut builder, query);
        builder.push(" ORDER BY ");
        builder.push(query.order_by.column());
        builder.push(if query.order_desc { " DESC" } else { " ASC" });
        if query.order_by != OrderBy::Path {
            builder.push(", d.file_path ASC");
        }
        builder.push(" LIMIT ");
        builder.push_bind(query.limit as i64);
        builder.push(" OFFSET ");
        builder.push_bind(query.offset as i64);

        // EXPLAIN QUERY PLAN rows are (id, parent, notused, detail).
        let rows: Vec<(i64, i64, i64, String)> = builder
            .build_query_as()
            .fetch_all(db.pool_for_test())
            .await
            .unwrap();
        rows.into_iter()
            .map(|(_, _, _, detail)| detail)
            .collect::<Vec<_>>()
            .join(" | ")
    }

    #[tokio::test]
    async fn query_documents_order_by_uses_an_index_not_a_temp_sort() {
        // Regression for #90: every supported (order_by, order_desc) combination must
        // be servable straight from an index. A plan containing "USE TEMP B-TREE" means
        // SQLite is sorting the whole filtered set in memory before LIMIT/OFFSET can
        // trim it — exactly the bug the new indexes exist to fix. Asserting this on the
        // real planner output (rather than just asserting rows come back correct) is
        // the point: a wrong-shaped or missing index still returns correct rows, just
        // slowly, and a correctness-only test would never catch that regressing.
        let (db, _dir) = seeded_db().await;

        for order_by in [
            OrderBy::Path,
            OrderBy::Title,
            OrderBy::Mtime,
            OrderBy::IndexedAt,
        ] {
            for order_desc in [false, true] {
                let query = DocumentQuery {
                    order_by,
                    order_desc,
                    ..Default::default()
                };
                let plan = explain_page_query(&db, &query).await;
                assert!(
                    !plan.to_ascii_uppercase().contains("TEMP B-TREE"),
                    "order_by={order_by:?} desc={order_desc} should be served by an \
                     index without a temp-b-tree sort; plan: {plan}"
                );
                assert!(
                    plan.to_ascii_uppercase().contains("INDEX"),
                    "order_by={order_by:?} desc={order_desc} should scan via an index; \
                     plan: {plan}"
                );
            }
        }
    }

    #[tokio::test]
    async fn query_documents_order_by_with_path_prefix_still_uses_an_index() {
        // A filter alongside the ordering (the common real-world shape: "recent docs
        // under sysadmin/") must not knock the planner off the ordering index either.
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            path_prefix: Some("kitchen/".into()),
            order_by: OrderBy::Mtime,
            order_desc: true,
            ..Default::default()
        };
        let plan = explain_page_query(&db, &query).await;
        assert!(
            !plan.to_ascii_uppercase().contains("TEMP B-TREE"),
            "filtered + ordered query should still avoid a temp-b-tree sort; plan: {plan}"
        );
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
    async fn all_paths_returns_every_indexed_document_path() {
        // Backs `retrieval::get_document`'s fuzzy fallback (#87): it needs the
        // complete, current path list, unfiltered and with no cap.
        let (db, _dir) = seeded_db().await;
        let mut result = db.all_paths().await.unwrap();
        result.sort();
        assert_eq!(
            result,
            vec![
                "kitchen/recipes/chili.md",
                "kitchen/recipes/congee.md",
                "kitchen/recipes/stir_fry.md",
                "sysadmin/zfs.md",
            ]
        );
    }

    #[tokio::test]
    async fn all_paths_reflects_deletions() {
        let (db, _dir) = seeded_db().await;
        db.delete_document("sysadmin/zfs.md").await.unwrap();

        let result = db.all_paths().await.unwrap();
        assert_eq!(result.len(), 3);
        assert!(!result.contains(&"sysadmin/zfs.md".to_string()));
    }

    #[tokio::test]
    async fn all_paths_on_an_empty_database_is_empty() {
        let (db, _dir) = test_db().await;
        assert!(db.all_paths().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn clear_also_wipes_document_metadata() {
        // A full reindex clears state BEFORE orphan detection reads it, so orphans are
        // never computed on a full run. If clear() left `documents` behind, a file
        // deleted from disk would remain listed forever with no run able to detect it.
        let (db, _dir) = test_db().await;
        db.upsert("gone.md", "h", 1, "", 0, 0).await.unwrap();
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
        db.upsert("r.md", "h1", 2, "", 0, 0).await.unwrap();

        assert_eq!(db.count().await.unwrap(), 1);
        assert_eq!(db.document_count().await.unwrap(), 0);
    }

    // -- pagination / chunking boundaries -------------------------------------
    //
    // `fetch_indexed_files_page`, `get_many`, and `get_document_hashes_many` are
    // exercised end-to-end by `ingest.rs`'s `scan_for_dirty` tests, but those
    // fixtures hold only a handful of files — never enough to reach a second page or
    // a second 500-param chunk. That leaves exactly the boundary these methods exist
    // for untested: an off-by-one here would silently skip files during a reconcile
    // sweep, and nothing would ever error — search would just quietly go stale. The
    // tests below insert through hand-rolled multi-row `INSERT`s rather than
    // `upsert`/`upsert_document_metadata` (one round trip per row) specifically to
    // keep the 500-1000+ row cases fast.

    /// Insert `entries` (file_path, content_hash) into `indexed_files` via a handful
    /// of multi-row `INSERT`s rather than one round trip per row, so the 500+ row
    /// boundary tests below stay fast. Each statement stays well under SQLite's
    /// 999-bound-parameter ceiling on its own — this helper's job is to seed data
    /// quickly, not to probe that limit itself.
    async fn bulk_insert_indexed_files(db: &StateDb, entries: &[(String, String)]) {
        const ROWS_PER_STATEMENT: usize = 100;
        for chunk in entries.chunks(ROWS_PER_STATEMENT) {
            let mut builder = QueryBuilder::<Sqlite>::new(
                "INSERT INTO indexed_files (file_path, content_hash, chunk_count, schema_hash, mtime, size) ",
            );
            builder.push_values(chunk, |mut b, (path, hash)| {
                b.push_bind(path)
                    .push_bind(hash)
                    .push_bind(1i64)
                    .push_bind("")
                    .push_bind(0i64)
                    .push_bind(0i64);
            });
            builder.build().execute(db.pool_for_test()).await.unwrap();
        }
    }

    /// Insert `entries` (file_path, content_hash) directly into `documents`,
    /// bypassing `upsert_document_metadata`'s per-row transaction and field
    /// projection — this helper only needs to produce rows `get_document_hashes_many`
    /// can read, not realistic frontmatter, and it needs to do it fast.
    async fn bulk_insert_documents(db: &StateDb, entries: &[(String, String)]) {
        const ROWS_PER_STATEMENT: usize = 100;
        for chunk in entries.chunks(ROWS_PER_STATEMENT) {
            let mut builder = QueryBuilder::<Sqlite>::new(
                "INSERT INTO documents (file_path, frontmatter, mtime, content_hash, chunk_count) ",
            );
            builder.push_values(chunk, |mut b, (path, hash)| {
                b.push_bind(path)
                    .push_bind("{}")
                    .push_bind(0i64)
                    .push_bind(hash)
                    .push_bind(1i64);
            });
            builder.build().execute(db.pool_for_test()).await.unwrap();
        }
    }

    // -- fetch_indexed_files_page ---------------------------------------------

    /// Walk `fetch_indexed_files_page` to exhaustion at `page_size`, reassembling
    /// every returned `file_path` (in page order) and counting how many pages it
    /// took. Guards against a regression that makes the loop never terminate — the
    /// exact failure an off-by-one in the `LIMIT`/`OFFSET` boundary would cause.
    async fn drain_all_pages(db: &StateDb, page_size: i64) -> (Vec<String>, usize) {
        let mut all = Vec::new();
        let mut offset = 0i64;
        let mut pages = 0usize;
        loop {
            let page = db
                .fetch_indexed_files_page(page_size, offset)
                .await
                .unwrap();
            if page.is_empty() {
                break;
            }
            pages += 1;
            all.extend(page.into_iter().map(|r| r.file_path));
            offset += page_size;
            assert!(
                pages < 10_000,
                "pagination did not terminate for page_size={page_size} — likely an \
                 off-by-one that never advances past a non-empty page"
            );
        }
        (all, pages)
    }

    #[tokio::test]
    async fn fetch_indexed_files_page_on_an_empty_table_returns_no_pages() {
        let (db, _dir) = test_db().await;
        let (all, pages) = drain_all_pages(&db, 5).await;
        assert_eq!(pages, 0);
        assert!(all.is_empty());
    }

    #[tokio::test]
    async fn pagination_covers_every_row_exactly_once_at_an_exact_page_multiple() {
        let (db, _dir) = test_db().await;
        let entries: Vec<(String, String)> = (0..10)
            .map(|i| (format!("f-{i:03}.md"), "h".to_string()))
            .collect();
        bulk_insert_indexed_files(&db, &entries).await;

        let (all, pages) = drain_all_pages(&db, 5).await;

        assert_eq!(
            pages, 2,
            "10 rows at page size 5 must take exactly 2 pages, not a trailing empty \
             third page"
        );
        let mut expected: Vec<String> = entries.into_iter().map(|(p, _)| p).collect();
        expected.sort();
        let mut got = all;
        got.sort();
        assert_eq!(
            got, expected,
            "every row must be reassembled exactly once — no gaps, no duplicates"
        );
    }

    #[tokio::test]
    async fn pagination_handles_one_less_than_a_page_multiple() {
        let (db, _dir) = test_db().await;
        let entries: Vec<(String, String)> = (0..9)
            .map(|i| (format!("f-{i:03}.md"), "h".to_string()))
            .collect();
        bulk_insert_indexed_files(&db, &entries).await;

        let (all, pages) = drain_all_pages(&db, 5).await;

        assert_eq!(
            pages, 2,
            "9 rows at page size 5: one full page of 5, one partial page of 4"
        );
        assert_eq!(all.len(), 9);
    }

    #[tokio::test]
    async fn pagination_handles_one_more_than_a_page_multiple() {
        let (db, _dir) = test_db().await;
        let entries: Vec<(String, String)> = (0..11)
            .map(|i| (format!("f-{i:03}.md"), "h".to_string()))
            .collect();
        bulk_insert_indexed_files(&db, &entries).await;

        let (all, pages) = drain_all_pages(&db, 5).await;

        assert_eq!(
            pages, 3,
            "11 rows at page size 5: two full pages plus one row spilling into a \
             third — an off-by-one here would either drop that 11th row or spin \
             forever re-fetching an empty page"
        );
        assert_eq!(all.len(), 11);
    }

    #[tokio::test]
    async fn fetch_indexed_files_page_limit_is_clamped_to_at_least_one() {
        // The implementation binds `limit.max(1)`. SQLite reads `LIMIT 0` as "zero
        // rows" — unlike the negative-limit-means-unbounded convention this file uses
        // elsewhere (`count_by_field`, `breakdown_fields`), so this method cannot
        // reuse that clamp. An unclamped 0 (or a negative value slipping through)
        // here would make `ingest::scan_for_dirty`'s paging loop spin forever, since
        // `offset` would never advance past a permanently empty page.
        let (db, _dir) = test_db().await;
        db.upsert("only.md", "h", 1, "", 0, 0).await.unwrap();

        let zero = db.fetch_indexed_files_page(0, 0).await.unwrap();
        assert_eq!(
            zero.len(),
            1,
            "a limit of 0 must be clamped to at least 1 row, not returned empty"
        );

        let negative = db.fetch_indexed_files_page(-5, 0).await.unwrap();
        assert_eq!(negative.len(), 1, "a negative limit must also clamp to 1");
    }

    // -- get_many ---------------------------------------------------------------

    /// Insert `n` distinct rows and assert `get_many` returns every single one with
    /// its correct content hash. `n` is chosen by each call site to sit at or
    /// straddle the 500-bind-parameter chunk boundary, so this checks both that
    /// nothing is dropped at a chunk seam and that no row's data gets attached to the
    /// wrong key while chunks are reassembled into one map.
    async fn assert_get_many_returns_every_requested_key(n: usize) {
        let (db, _dir) = test_db().await;
        let entries: Vec<(String, String)> = (0..n)
            .map(|i| (format!("file-{i:05}.md"), format!("hash-{i}")))
            .collect();
        bulk_insert_indexed_files(&db, &entries).await;

        let paths: Vec<String> = entries.iter().map(|(p, _)| p.clone()).collect();
        let result = db.get_many(&paths).await.unwrap();

        assert_eq!(result.len(), n, "every requested key must come back, n={n}");
        for (path, hash) in &entries {
            let row = result.get(path).unwrap_or_else(|| {
                panic!("'{path}' missing from get_many's result — a chunk-seam drop at n={n}")
            });
            assert_eq!(
                &row.content_hash, hash,
                "wrong row attached to '{path}' at n={n}"
            );
        }
    }

    #[tokio::test]
    async fn get_many_at_499_below_the_chunk_boundary() {
        assert_get_many_returns_every_requested_key(499).await;
    }

    #[tokio::test]
    async fn get_many_at_exactly_500_the_chunk_boundary() {
        assert_get_many_returns_every_requested_key(500).await;
    }

    #[tokio::test]
    async fn get_many_at_501_one_past_the_chunk_boundary() {
        assert_get_many_returns_every_requested_key(501).await;
    }

    #[tokio::test]
    async fn get_many_at_1001_spanning_three_chunks() {
        assert_get_many_returns_every_requested_key(1001).await;
    }

    #[tokio::test]
    async fn get_many_distinguishes_absent_keys_from_present_ones() {
        let (db, _dir) = test_db().await;
        let entries: Vec<(String, String)> = (0..5)
            .map(|i| (format!("present-{i}.md"), format!("hash-{i}")))
            .collect();
        bulk_insert_indexed_files(&db, &entries).await;

        let mut requested: Vec<String> = entries.iter().map(|(p, _)| p.clone()).collect();
        requested.push("missing-1.md".into());
        requested.push("missing-2.md".into());

        let result = db.get_many(&requested).await.unwrap();

        assert_eq!(
            result.len(),
            5,
            "only rows that actually exist come back; a missing key must not be \
             confused with a chunking bug"
        );
        assert!(!result.contains_key("missing-1.md"));
        assert!(!result.contains_key("missing-2.md"));
        for (path, _) in &entries {
            assert!(result.contains_key(path));
        }
    }

    #[tokio::test]
    async fn get_many_empty_input_returns_empty_map() {
        let (db, _dir) = test_db().await;
        assert!(db.get_many(&[]).await.unwrap().is_empty());
    }

    // -- get_document_hashes_many ------------------------------------------------

    /// Same shape as [`assert_get_many_returns_every_requested_key`], for the
    /// `documents`-table equivalent.
    async fn assert_get_document_hashes_many_returns_every_requested_key(n: usize) {
        let (db, _dir) = test_db().await;
        let entries: Vec<(String, String)> = (0..n)
            .map(|i| (format!("doc-{i:05}.md"), format!("hash-{i}")))
            .collect();
        bulk_insert_documents(&db, &entries).await;

        let paths: Vec<String> = entries.iter().map(|(p, _)| p.clone()).collect();
        let result = db.get_document_hashes_many(&paths).await.unwrap();

        assert_eq!(result.len(), n, "every requested key must come back, n={n}");
        for (path, hash) in &entries {
            assert_eq!(
                result.get(path).map(String::as_str),
                Some(hash.as_str()),
                "wrong hash attached to '{path}' at n={n}"
            );
        }
    }

    #[tokio::test]
    async fn get_document_hashes_many_at_499_below_the_chunk_boundary() {
        assert_get_document_hashes_many_returns_every_requested_key(499).await;
    }

    #[tokio::test]
    async fn get_document_hashes_many_at_exactly_500_the_chunk_boundary() {
        assert_get_document_hashes_many_returns_every_requested_key(500).await;
    }

    #[tokio::test]
    async fn get_document_hashes_many_at_501_one_past_the_chunk_boundary() {
        assert_get_document_hashes_many_returns_every_requested_key(501).await;
    }

    #[tokio::test]
    async fn get_document_hashes_many_at_1001_spanning_three_chunks() {
        assert_get_document_hashes_many_returns_every_requested_key(1001).await;
    }

    #[tokio::test]
    async fn get_document_hashes_many_distinguishes_absent_keys_from_present_ones() {
        let (db, _dir) = test_db().await;
        let entries: Vec<(String, String)> = (0..5)
            .map(|i| (format!("present-{i}.md"), format!("hash-{i}")))
            .collect();
        bulk_insert_documents(&db, &entries).await;

        let mut requested: Vec<String> = entries.iter().map(|(p, _)| p.clone()).collect();
        requested.push("missing-1.md".into());

        let result = db.get_document_hashes_many(&requested).await.unwrap();

        assert_eq!(result.len(), 5);
        assert!(!result.contains_key("missing-1.md"));
    }

    #[tokio::test]
    async fn get_document_hashes_many_empty_input_returns_empty_map() {
        let (db, _dir) = test_db().await;
        assert!(db.get_document_hashes_many(&[]).await.unwrap().is_empty());
    }

    // -- document_links --------------------------------------------------------

    #[tokio::test]
    async fn replace_links_round_trip() {
        let (db, _dir) = test_db().await;
        db.replace_links(
            "a.md",
            "markdown",
            &[("b.md".to_string(), None), ("c.md".to_string(), None)],
        )
        .await
        .unwrap();

        let mut links = db.all_links().await.unwrap();
        links.sort_by(|a, b| (&a.0, &a.1, &a.2).cmp(&(&b.0, &b.1, &b.2)));
        assert_eq!(
            links,
            vec![
                (
                    "a.md".to_string(),
                    "b.md".to_string(),
                    "markdown".to_string(),
                    None
                ),
                (
                    "a.md".to_string(),
                    "c.md".to_string(),
                    "markdown".to_string(),
                    None
                ),
            ]
        );
    }

    #[tokio::test]
    async fn replace_links_carries_score() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "semantic", &[("b.md".to_string(), Some(0.87))])
            .await
            .unwrap();

        let links = db.all_links().await.unwrap();
        assert_eq!(
            links,
            vec![(
                "a.md".to_string(),
                "b.md".to_string(),
                "semantic".to_string(),
                Some(0.87)
            )]
        );
    }

    #[tokio::test]
    async fn replace_links_overwrites_prior_call_for_same_source_and_kind() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("old.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("a.md", "markdown", &[("new.md".to_string(), None)])
            .await
            .unwrap();

        let links = db.all_links().await.unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].1, "new.md");
    }

    #[tokio::test]
    async fn replace_links_is_scoped_to_kind() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("md-target.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links(
            "a.md",
            "semantic",
            &[("sem-target.md".to_string(), Some(0.9))],
        )
        .await
        .unwrap();

        // Replacing the semantic edges must not disturb the markdown edges for the
        // same source.
        db.replace_links(
            "a.md",
            "semantic",
            &[("sem-target-2.md".to_string(), Some(0.5))],
        )
        .await
        .unwrap();

        let mut links = db.all_links().await.unwrap();
        links.sort_by(|a, b| (&a.0, &a.1, &a.2).cmp(&(&b.0, &b.1, &b.2)));
        assert_eq!(
            links,
            vec![
                (
                    "a.md".to_string(),
                    "md-target.md".to_string(),
                    "markdown".to_string(),
                    None
                ),
                (
                    "a.md".to_string(),
                    "sem-target-2.md".to_string(),
                    "semantic".to_string(),
                    Some(0.5)
                ),
            ]
        );
    }

    #[tokio::test]
    async fn replace_links_empty_targets_clears_existing() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("b.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("a.md", "markdown", &[]).await.unwrap();

        assert!(db.all_links().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn links_targeting_returns_distinct_sources_ordered() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("target.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("c.md", "markdown", &[("target.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("b.md", "markdown", &[("target.md".to_string(), None)])
            .await
            .unwrap();
        // A source unrelated to `target.md` must not show up.
        db.replace_links("d.md", "markdown", &[("other.md".to_string(), None)])
            .await
            .unwrap();

        let sources = db.links_targeting("target.md", "markdown").await.unwrap();
        assert_eq!(sources, vec!["a.md", "b.md", "c.md"]);
    }

    #[tokio::test]
    async fn links_targeting_filters_by_kind() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("target.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("b.md", "semantic", &[("target.md".to_string(), Some(0.9))])
            .await
            .unwrap();

        let markdown_sources = db.links_targeting("target.md", "markdown").await.unwrap();
        assert_eq!(
            markdown_sources,
            vec!["a.md"],
            "a semantic-kind row must not come back when asking for markdown"
        );

        let semantic_sources = db.links_targeting("target.md", "semantic").await.unwrap();
        assert_eq!(semantic_sources, vec!["b.md"]);
    }

    #[tokio::test]
    async fn links_targeting_no_results_returns_empty() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("other.md".to_string(), None)])
            .await
            .unwrap();

        let sources = db
            .links_targeting("nonexistent.md", "markdown")
            .await
            .unwrap();
        assert!(sources.is_empty());
    }

    // -- links_targeting_many ----------------------------------------------------

    /// Insert `(source_path, target_path, kind)` rows into `document_links` via a
    /// handful of multi-row `INSERT`s, so the chunk-boundary test below (500+ distinct
    /// targets) stays fast — same rationale as `bulk_insert_indexed_files`/
    /// `bulk_insert_documents` above. `score` is always bound `NULL`: none of these
    /// tests care about it.
    async fn bulk_insert_document_links(db: &StateDb, rows: &[(String, String, String)]) {
        const ROWS_PER_STATEMENT: usize = 100;
        for chunk in rows.chunks(ROWS_PER_STATEMENT) {
            let mut builder = QueryBuilder::<Sqlite>::new(
                "INSERT INTO document_links (source_path, target_path, kind, score) ",
            );
            builder.push_values(chunk, |mut b, (source_path, target_path, kind)| {
                b.push_bind(source_path)
                    .push_bind(target_path)
                    .push_bind(kind)
                    .push_bind(Option::<f64>::None);
            });
            builder.build().execute(db.pool_for_test()).await.unwrap();
        }
    }

    #[tokio::test]
    async fn links_targeting_many_returns_matches_grouped_by_target() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("target1.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("c.md", "markdown", &[("target1.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("b.md", "markdown", &[("target1.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("d.md", "markdown", &[("target2.md".to_string(), None)])
            .await
            .unwrap();
        // A source unrelated to either queried target must not show up.
        db.replace_links("e.md", "markdown", &[("other.md".to_string(), None)])
            .await
            .unwrap();

        let by_target = db
            .links_targeting_many(
                &["target1.md".to_string(), "target2.md".to_string()],
                "markdown",
            )
            .await
            .unwrap();

        assert_eq!(
            by_target.get("target1.md"),
            Some(&vec![
                "a.md".to_string(),
                "b.md".to_string(),
                "c.md".to_string()
            ]),
            "multiple sources targeting the same path must all come back, sorted, \
             matching links_targeting's own ordering"
        );
        assert_eq!(by_target.get("target2.md"), Some(&vec!["d.md".to_string()]));
        assert_eq!(
            by_target.len(),
            2,
            "a target with no referencing sources must have no key at all"
        );
    }

    #[tokio::test]
    async fn links_targeting_many_filters_by_kind() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("target.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("b.md", "semantic", &[("target.md".to_string(), Some(0.9))])
            .await
            .unwrap();

        let markdown = db
            .links_targeting_many(&["target.md".to_string()], "markdown")
            .await
            .unwrap();
        assert_eq!(
            markdown.get("target.md"),
            Some(&vec!["a.md".to_string()]),
            "a semantic-kind row must not come back when asking for markdown"
        );

        let semantic = db
            .links_targeting_many(&["target.md".to_string()], "semantic")
            .await
            .unwrap();
        assert_eq!(semantic.get("target.md"), Some(&vec!["b.md".to_string()]));
    }

    #[tokio::test]
    async fn links_targeting_many_empty_input_returns_empty_map() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("target.md".to_string(), None)])
            .await
            .unwrap();

        let result = db.links_targeting_many(&[], "markdown").await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn links_targeting_many_spans_a_500_param_chunk_boundary() {
        // 501 distinct target paths: one full 500-param chunk plus one row spilling
        // into a second chunk. Each has exactly one source pointing at it, so an
        // off-by-one at the chunk seam would show up as a missing target rather than
        // silently merging two targets' sources together.
        let (db, _dir) = test_db().await;
        let targets: Vec<String> = (0..501).map(|i| format!("target-{i:05}.md")).collect();
        let rows: Vec<(String, String, String)> = targets
            .iter()
            .map(|target| {
                (
                    format!("source-for-{target}"),
                    target.clone(),
                    "markdown".to_string(),
                )
            })
            .collect();
        bulk_insert_document_links(&db, &rows).await;

        let by_target = db.links_targeting_many(&targets, "markdown").await.unwrap();

        assert_eq!(
            by_target.len(),
            501,
            "every target across the chunk boundary must come back"
        );
        for target in &targets {
            assert_eq!(
                by_target.get(target),
                Some(&vec![format!("source-for-{target}")]),
                "wrong (or missing) source attached to '{target}' at the chunk boundary"
            );
        }
    }

    #[tokio::test]
    async fn delete_links_for_removes_outgoing_edges_only() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("b.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("c.md", "markdown", &[("a.md".to_string(), None)])
            .await
            .unwrap();

        db.delete_links_for("a.md").await.unwrap();

        let links = db.all_links().await.unwrap();
        assert_eq!(
            links,
            vec![(
                "c.md".to_string(),
                "a.md".to_string(),
                "markdown".to_string(),
                None
            )],
            "incoming edges (a.md as target) must survive; only a.md's outgoing edges are removed"
        );
    }

    #[tokio::test]
    async fn delete_document_also_clears_outgoing_links() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("a.md", &recipe_frontmatter(), 1700, "h1", 1)
            .await
            .unwrap();
        db.replace_links("a.md", "markdown", &[("b.md".to_string(), None)])
            .await
            .unwrap();

        db.delete_document("a.md").await.unwrap();

        assert!(db.all_links().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn all_document_summaries_returns_every_row_unpaged() {
        let (db, _dir) = test_db().await;
        let entries: Vec<(String, String)> = (0..150)
            .map(|i| (format!("doc-{i:03}.md"), format!("hash-{i}")))
            .collect();
        bulk_insert_documents(&db, &entries).await;

        let summaries = db.all_document_summaries().await.unwrap();

        assert_eq!(
            summaries.len(),
            150,
            "all_document_summaries must not apply a hidden limit"
        );
        // ORDER BY file_path, ascending.
        assert_eq!(summaries[0].file_path, "doc-000.md");
        assert_eq!(summaries[149].file_path, "doc-149.md");
    }

    #[tokio::test]
    async fn all_document_summaries_degrades_malformed_frontmatter() {
        let (db, _dir) = test_db().await;
        sqlx::query(
            "INSERT INTO documents
                (file_path, title, description, frontmatter, mtime, content_hash, chunk_count, indexed_at)
             VALUES ('bad.md', NULL, NULL, 'not valid json{{', 0, 'h', 0, datetime('now'))",
        )
        .execute(db.pool_for_test())
        .await
        .unwrap();

        let summaries = db.all_document_summaries().await.unwrap();

        assert_eq!(summaries.len(), 1);
        assert_eq!(
            summaries[0].frontmatter,
            Value::Object(serde_json::Map::new())
        );
    }

    #[tokio::test]
    async fn all_document_summaries_maps_fields_from_upsert() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("r.md", &recipe_frontmatter(), 1700, "h1", 2)
            .await
            .unwrap();

        let summaries = db.all_document_summaries().await.unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].file_path, "r.md");
        assert_eq!(summaries[0].mtime, 1700);
        assert!(summaries[0].frontmatter.is_object());
    }
}
