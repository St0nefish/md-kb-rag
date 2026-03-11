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

### Context Window Override

nomic-embed-text-v2-moe natively supports 8192-token context windows, but the GGUF file metadata incorrectly reports a 512-token limit. The `docker-compose.yml` command includes `--override-kv nomic-bert-moe.context_length=int:8192` to correct this, along with matching `--ctx-size`, `--batch-size`, and `--ubatch-size` flags. This allows embedding larger markdown chunks in a single pass. If you switch to a different model, adjust or remove these overrides accordingly.

## GPU Backends

The `embeddings` service uses [llama.cpp](https://github.com/ggml-org/llama.cpp) for local GPU-accelerated inference. The `LLAMA_IMAGE_TAG` env var selects the GPU backend (defaults to `server-vulkan`).

| Backend | `LLAMA_IMAGE_TAG` | GPU Support | Device Passthrough |
|---|---|---|---|
| Vulkan (default) | `server-vulkan` | AMD, Intel, NVIDIA | `/dev/dri` |
| ROCm | `server-rocm` | AMD (RX 7000/9000) | `/dev/kfd` + specific render nodes |
| CUDA | `server-cuda` | NVIDIA | NVIDIA Container Toolkit runtime |

### Vulkan (default)

Works out of the box on most GPUs. No extra configuration needed beyond the default `docker-compose.yml`.

### ROCm (AMD)

For AMD GPUs using ROCm (e.g. RX 7900 XTX, RX 9700 XT), replace the device passthrough and add ROCm-specific settings. Create a `docker-compose.override.yml`:

```yaml
services:
  embeddings:
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri/cardN:/dev/dri/cardN       # replace N with your GPU card number
      - /dev/dri/renderDN:/dev/dri/renderDN  # replace N with your GPU render node
    group_add:
      - "video"
      - "render"
    security_opt:
      - seccomp=unconfined
    environment:
      - HSA_OVERRIDE_GFX_VERSION=12.0.1  # adjust for your GPU architecture
      - HIP_VISIBLE_DEVICES=0
```

Set in `.env`:

```env
LLAMA_IMAGE_TAG=server-rocm
```

Find your device nodes with `ls /dev/dri/` and match card/render numbers to your target GPU. The `group_add` GIDs correspond to the `video` and `render` groups — check `getent group video render` for the correct values on your system. `HSA_OVERRIDE_GFX_VERSION` depends on your GPU — e.g. `11.0.0` for RDNA 3 (RX 7000), `12.0.1` for RDNA 4 (RX 9000).

### CUDA (NVIDIA)

For NVIDIA GPUs, use the CUDA image with the NVIDIA Container Toolkit. Create a `docker-compose.override.yml`:

```yaml
services:
  embeddings:
    devices: []  # clear the default /dev/dri
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Set in `.env`:

```env
LLAMA_IMAGE_TAG=server-cuda
```

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
