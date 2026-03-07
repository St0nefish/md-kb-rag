use anyhow::Result;
use sqlx::SqlitePool;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
use std::path::Path;
use std::str::FromStr;

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct IndexedFile {
    pub file_path: String,
    pub content_hash: String,
    pub chunk_count: i64,
    pub indexed_at: String,
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
            .busy_timeout(std::time::Duration::from_secs(5));

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

        Ok(Self { pool })
    }

    #[cfg(test)]
    pub async fn get(&self, file_path: &str) -> Result<Option<IndexedFile>> {
        let row = sqlx::query_as::<_, IndexedFile>(
            "SELECT file_path, content_hash, chunk_count, indexed_at FROM indexed_files WHERE file_path = ?",
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
    ) -> Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO indexed_files (file_path, content_hash, chunk_count, indexed_at)
             VALUES (?, ?, ?, datetime('now'))",
        )
        .bind(file_path)
        .bind(content_hash)
        .bind(chunk_count)
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
            "SELECT file_path, content_hash, chunk_count, indexed_at FROM indexed_files ORDER BY file_path",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    pub async fn clear(&self) -> Result<()> {
        sqlx::query("DELETE FROM indexed_files")
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
        db.upsert("test.md", "abc123", 3).await.unwrap();
        let entry = db.get("test.md").await.unwrap().unwrap();
        assert_eq!(entry.file_path, "test.md");
        assert_eq!(entry.content_hash, "abc123");
        assert_eq!(entry.chunk_count, 3);
    }

    #[tokio::test]
    async fn upsert_replaces() {
        let (db, _dir) = test_db().await;
        db.upsert("test.md", "hash1", 2).await.unwrap();
        db.upsert("test.md", "hash2", 5).await.unwrap();
        let entry = db.get("test.md").await.unwrap().unwrap();
        assert_eq!(entry.content_hash, "hash2");
        assert_eq!(entry.chunk_count, 5);
    }

    #[tokio::test]
    async fn delete_removes() {
        let (db, _dir) = test_db().await;
        db.upsert("test.md", "hash", 1).await.unwrap();
        db.delete("test.md").await.unwrap();
        assert!(db.get("test.md").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn list_and_count() {
        let (db, _dir) = test_db().await;
        db.upsert("a.md", "h1", 1).await.unwrap();
        db.upsert("b.md", "h2", 2).await.unwrap();
        assert_eq!(db.count().await.unwrap(), 2);
        let all = db.list_all().await.unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].file_path, "a.md"); // sorted
    }

    #[tokio::test]
    async fn clear_removes_all() {
        let (db, _dir) = test_db().await;
        db.upsert("a.md", "h1", 1).await.unwrap();
        db.upsert("b.md", "h2", 2).await.unwrap();
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
        db.upsert("doc.md", "hash1", 3).await.unwrap();

        // Simulate: upsert to Qdrant fails, so we delete the state entry
        // (this is what ingest.rs now does on failure)
        db.delete("doc.md").await.unwrap();

        // On next run, the file should appear as new (not in state DB)
        assert!(db.get("doc.md").await.unwrap().is_none());
    }
}
