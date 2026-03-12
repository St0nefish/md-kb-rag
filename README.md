# md-kb-rag

A Docker-first RAG server that indexes markdown knowledge bases with YAML frontmatter into Qdrant and exposes semantic search via MCP (Streamable HTTP).

Built as a single Rust binary for type safety, small Docker images, and simple deployment.

## Documentation

- [`deploy/USAGE.md`](deploy/USAGE.md) — Setup guide, configuration, frontmatter, chunking
- [`deploy/TROUBLESHOOTING.md`](deploy/TROUBLESHOOTING.md) — Common issues and fixes
- [`deploy/config.example.yaml`](deploy/config.example.yaml) — Full annotated config reference
- [`deploy/ci-examples/`](deploy/ci-examples/) — Sample CI workflows for webhook-triggered reindex

## Quick Start

```bash
# Clone and configure
git clone https://github.com/St0nefish/md-kb-rag.git
cd md-kb-rag
cp deploy/.env.example .env
# Edit .env: set MCP_BEARER_TOKEN and MODEL_PATH/MODEL_FILE

# Download the embedding model (see "Embedding Models" below)

# Start the stack (CPU mode by default)
docker compose up -d

# Initial full index
docker compose exec kb-rag md-kb-rag index --full

# Add MCP to Claude Code
claude mcp add --transport http kb-search \
  https://your-host:8001/mcp \
  --header "Authorization: Bearer $TOKEN"
```

No `config.yaml` needed — connection settings are wired through environment variables in `docker-compose.yml`. To customize behavior (chunking, frontmatter rules, etc.), copy the example and mount it:

```bash
cp deploy/config.example.yaml config.yaml
# Edit config.yaml, then uncomment the volume mount in docker-compose.yml:
#   - ./config.yaml:/app/config.yaml:ro
```

See [deploy/config.example.yaml](deploy/config.example.yaml) for all available options and their defaults.

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

See [deploy/config.example.yaml](deploy/config.example.yaml) for all options:

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

The default config is tuned for **nomic-embed-text-v2-moe** (768 dimensions, GGUF via llama.cpp).

### Download

```bash
# Download from Hugging Face (requires huggingface-cli: pip install huggingface_hub)
huggingface-cli download nomic-ai/nomic-embed-text-v2-moe-GGUF \
  nomic-embed-text-v2-moe-Q8_0.gguf --local-dir ./data/models

# Or download directly from:
# https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF
```

Then set in `.env`:

```env
MODEL_PATH=./data/models
MODEL_FILE=nomic-embed-text-v2-moe-Q8_0.gguf
```

### Alternative Models

To use a different model, override in `.env`:

```env
EMBEDDING_MODEL=bge-large-en-v1.5
EMBEDDING_VECTOR_SIZE=1024
MODEL_FILE=bge-large-en-v1.5-q8_0.gguf
```

Or in `config.yaml` (env vars take priority if both are set).

| Model | `vector_size` | Notes |
|---|---|---|
| nomic-embed-text-v2-moe (default) | 768 | Recommended. MoE, strong quality/speed. |
| nomic-embed-text-v1.5 | 768 | Older nomic, same dimensions. |
| all-MiniLM-L6-v2 | 384 | Lightweight, lower quality. |
| bge-large-en-v1.5 | 1024 | Strong quality, larger vectors. |
| mxbai-embed-large-v1 | 1024 | Good alternative to bge. |

**Note:** Changing `vector_size` requires a full reindex (`index --full`) which drops and recreates the Qdrant collection.

## Embedding Backends

The dev `docker-compose.yml` defaults to **CPU mode** which works on any hardware. For production deployment, pick a hardware-specific template from `deploy/templates/`.

### CPU (default)

Works everywhere with no special drivers. Good for small knowledge bases or initial testing. The compose file uses `ghcr.io/ggml-org/llama.cpp:server`.

### NVIDIA CUDA

Most common GPU backend. Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed on the host. Uses `server-cuda12` image with `deploy.resources.reservations.devices` for GPU access.

### AMD ROCm

Best performance on AMD GPUs. Requires ROCm userspace drivers on the host. Uses `server-rocm` image with `/dev/kfd` and `/dev/dri` device access.

### AMD Vulkan

Simpler driver setup than ROCm — works with standard Mesa Vulkan drivers. Uses `server-vulkan` image with `/dev/dri` device access. Supports multi-GPU setups.

### Apple Silicon (Metal)

Metal GPU acceleration is **not available in Docker** (Docker on macOS runs a Linux VM). Options:

1. **Run llama-server natively** — `brew install llama.cpp`, then start it with your model and point `EMBEDDING_BASE_URL` at it (`http://host.docker.internal:8080/v1` if kb-rag runs in Docker).
2. **Use the CPU Docker image** — works but is slower than native Metal.

### External API

Skip the bundled embedding service entirely. Point `EMBEDDING_BASE_URL` at any OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, TEI) and remove the `embeddings` service from compose.

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

The webhook endpoint is only available if `WEBHOOK_SECRET` is set to a non-empty value. See [`deploy/ci-examples/`](deploy/ci-examples/) for sample CI workflows.

## Incremental Indexing

Files are tracked by SHA256 content hash in a SQLite state database. On each run:

- **New files** — validate, chunk, embed, upsert
- **Changed files** — delete old vectors, re-process
- **Deleted files** — remove vectors and state entry
- **Unchanged files** — skip

Point IDs are deterministic UUIDs (v5) derived from `file_path::chunk_index`.

## Deployment

All deployment artifacts live in [`deploy/`](deploy/):

- **Compose templates** — `deploy/templates/` has self-contained compose files for each hardware backend (CPU, NVIDIA, ROCm, Vulkan, Apple Silicon)
- **Config examples** — `deploy/.env.example` and `deploy/config.example.yaml`
- **CI examples** — `deploy/ci-examples/` has sample webhook workflows for Gitea and GitHub
- **Deploy script** — `deploy/deploy.sh` pulls and restarts via Docker context (configure with `deploy/deploy.env`)

**Claude Code users:** Run `/deploy-md-rag` for an interactive guided setup that walks through hardware selection, model download, configuration, and MCP client connection.

**Manual setup:** Copy the matching template from `deploy/templates/` to your target as `docker-compose.yml`, configure `.env` from the example, and follow [`deploy/USAGE.md`](deploy/USAGE.md).

## Development

```bash
# Set up git hooks (fmt + clippy on commit)
./scripts/setup-dev.sh

# Start only the dependencies
docker compose up qdrant embeddings -d

# Run the server locally (requires env vars for connection settings)
export EMBEDDING_BASE_URL=http://localhost:8080/v1
export EMBEDDING_MODEL=nomic-embed-text-v2-moe
export QDRANT_URL=http://localhost:6334
export MCP_BEARER_TOKEN=dev-token
cargo run -- serve
```

Typical workflow: develop locally, push to a feature branch, CI builds and tests, merge via PR. See [deploy/USAGE.md](deploy/USAGE.md) for full setup walkthrough.

## License

MIT
