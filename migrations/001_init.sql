CREATE TABLE IF NOT EXISTS indexed_files (
    file_path    TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL,
    indexed_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
