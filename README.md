# md-kb-rag

A Docker-first RAG server that indexes markdown knowledge bases with YAML frontmatter into Qdrant and exposes semantic search via MCP (Streamable HTTP).

Built as a single Rust binary for type safety, small Docker images, and simple deployment.

## Quick Start

```bash
# Clone and configure
git clone https://github.com/you/md-kb-rag.git
cd md-kb-rag
cp .env.example .env
# Edit .env: set MCP_BEARER_TOKEN and MODEL_PATH/MODEL_FILE

# Start the stack
docker compose up -d

# Initial full index
docker compose exec kb-rag md-kb-rag index --full

# Add MCP to Claude Code
claude mcp add --transport http kb-search \
  https://your-host:8001/mcp \
  --header "Authorization: Bearer $TOKEN"
```

No `config.yaml` needed — connection settings are wired through environment variables in `docker-compose.yml`. To customize behavior (chunking, frontmatter rules, etc.), mount a config file:

```yaml
# docker-compose.yml override
volumes:
  - ./config.yaml:/app/config.yaml:ro
```

See [config.example.yaml](config.example.yaml) for all available options and their defaults.

## Usage Guide

See [docs/USAGE.md](docs/USAGE.md) for detailed setup instructions, including:

- Sample markdown document with frontmatter
- How the chunking algorithm works
- Frontmatter validation configuration
- Step-by-step project setup walkthrough

## Architecture

Three Docker services:

| Service | Purpose |
|---|---|
| `qdrant` | Vector database (gRPC + REST) |
| `embeddings` | Local embedding server (llama.cpp, OpenAI-compatible API) |
| `kb-rag` | Indexer, MCP server, and webhook handler (single Rust binary) |

## CLI Commands

```bash
md-kb-rag serve              # Start server (MCP + webhook endpoints)
md-kb-rag index              # Incremental index (only changed files)
md-kb-rag index --full       # Full re-index (clear state, re-embed everything)
md-kb-rag validate           # Validate all markdown files without indexing
md-kb-rag status             # Print collection stats + state DB info
md-kb-rag health             # Check if server is healthy
```

## Configuration

Config is loaded from `config.yaml` (or the path passed via `--config`). Every field has a sensible default, so the file is optional. Connection settings can be set via environment variables:

| Env Var | Config Path | Default (in compose) |
|---|---|---|
| `EMBEDDING_BASE_URL` | `embedding.base_url` | `http://embeddings:8080/v1` |
| `EMBEDDING_MODEL` | `embedding.model` | `nomic-embed-text-v2-moe` |
| `EMBEDDING_VECTOR_SIZE` | `embedding.vector_size` | `768` |
| `QDRANT_URL` | `qdrant.url` | `http://qdrant:6334` |

Env vars take priority over config file values. If neither is set for required fields (`embedding.base_url`, `embedding.model`, `qdrant.url`), the server exits with a clear error.

See [config.example.yaml](config.example.yaml) for all options:

- **source** — Git URL or bind-mount path for your knowledge base
- **indexing** — Include/exclude glob patterns
- **frontmatter** — Required fields, indexed fields, defaults
- **chunking** — Markdown-aware splitting with configurable chunk size
- **embedding** — OpenAI-compatible endpoint (works with llama.cpp, vLLM, etc.)
- **qdrant** — Connection URL and collection name
- **validation** — Strict/lenient mode, optional lint command
- **webhook** — HMAC verification for Gitea/GitHub/GitLab (disabled if `WEBHOOK_SECRET` is unset)
- **mcp** — Server port and bearer token authentication

## Embedding Models

The default config is tuned for **nomic-embed-text-v2-moe** (768 dimensions, GGUF via llama.cpp). Download the GGUF file and set `MODEL_PATH`/`MODEL_FILE` in `.env`.

To use a different model, override in `.env`:

```env
EMBEDDING_MODEL=bge-large-en-v1.5
EMBEDDING_VECTOR_SIZE=1024
MODEL_FILE=bge-large-en-v1.5-q8_0.gguf
```

Or in `config.yaml` (env vars take priority if both are set).

Common alternatives:

| Model | `vector_size` | Notes |
|---|---|---|
| nomic-embed-text-v2-moe (default) | 768 | Recommended. MoE, strong quality/speed. |
| nomic-embed-text-v1.5 | 768 | Older nomic, same dimensions. |
| all-MiniLM-L6-v2 | 384 | Lightweight, lower quality. |
| bge-large-en-v1.5 | 1024 | Strong quality, larger vectors. |
| mxbai-embed-large-v1 | 1024 | Good alternative to bge. |

**Note:** Changing `vector_size` requires a full reindex (`index --full`) which drops and recreates the Qdrant collection.

## MCP Search Tool

The `search` tool accepts:

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | string | yes | Natural-language search query |
| `domain` | string | no | Filter by domain field |
| `type` | string | no | Filter by document type |
| `tags` | string[] | no | Filter by tags (match any) |
| `limit` | integer | no | Max results (default: 10, max: 50) |

## Webhook

POST to `/hooks/reindex` triggers:

1. HMAC signature verification (Gitea/GitHub/GitLab)
2. Branch matching
3. `git pull --ff-only` (if git_url configured)
4. Incremental reindex

The webhook endpoint is only available if `WEBHOOK_SECRET` is set to a non-empty value. See `actions/` for sample CI workflows.

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
cargo run -- serve
cargo run -- index --full
```

## License

MIT
