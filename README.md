# md-kb-rag

A Docker-first RAG server that indexes markdown knowledge bases with YAML frontmatter into Qdrant and exposes semantic search via MCP (Streamable HTTP).

Built as a single Rust binary for type safety, small Docker images, and simple deployment.

## Quick Start

```bash
# Clone and configure
git clone https://github.com/you/md-kb-rag.git
cd md-kb-rag
cp config.example.yaml config.yaml
cp .env.example .env
# Edit config.yaml and .env with your settings

# Start the stack
docker compose up -d

# Initial full index
docker compose exec md-kb-rag md-kb-rag index --full --config /app/config.yaml

# Add MCP to Claude Code
claude mcp add --transport http kb-search \
  https://your-host:8001/mcp \
  --header "Authorization: Bearer $TOKEN"
```

## Architecture

Three Docker services:

| Service | Purpose |
|---|---|
| `qdrant` | Vector database (gRPC + REST) |
| `embeddings` | Local embedding server (llama.cpp, OpenAI-compatible API) |
| `md-kb-rag` | Indexer, MCP server, and webhook handler (single Rust binary) |

## CLI Commands

```bash
md-kb-rag serve              # Start server (MCP + webhook endpoints)
md-kb-rag index              # Incremental index (only changed files)
md-kb-rag index --full       # Full re-index (clear state, re-embed everything)
md-kb-rag validate           # Validate all markdown files without indexing
md-kb-rag status             # Print collection stats + state DB info
```

## Configuration

See [config.example.yaml](config.example.yaml) for all options:

- **source** — Git URL or bind-mount path for your knowledge base
- **indexing** — Include/exclude glob patterns
- **frontmatter** — Required fields, indexed fields, defaults
- **chunking** — Markdown-aware splitting with configurable chunk size
- **embedding** — OpenAI-compatible endpoint (works with llama.cpp, vLLM, etc.)
- **qdrant** — Connection URL and collection name
- **validation** — Strict/lenient mode, optional lint command
- **webhook** — HMAC verification for Gitea/GitHub/GitLab
- **mcp** — Server port and bearer token authentication

## MCP Search Tool

The `search` tool accepts:

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | string | yes | Natural-language search query |
| `domain` | string | no | Filter by domain field |
| `type` | string | no | Filter by document type |
| `tags` | string[] | no | Filter by tags (match any) |
| `limit` | integer | no | Max results (default: 10) |

## Webhook

POST to `/hooks/reindex` triggers:

1. HMAC signature verification (Gitea/GitHub/GitLab)
2. Branch matching
3. `git pull --ff-only` (if git_url configured)
4. Incremental reindex

See `actions/` for sample CI workflows.

## Incremental Indexing

Files are tracked by SHA256 content hash in a SQLite state database. On each run:

- **New files** — validate, chunk, embed, upsert
- **Changed files** — delete old vectors, re-process
- **Deleted files** — remove vectors and state entry
- **Unchanged files** — skip

Point IDs are deterministic UUIDs (v5) derived from `file_path::chunk_index`.

## Development

```bash
cargo build
cargo run -- serve --config config.yaml
cargo run -- index --full --config config.yaml
```

## License

MIT
