# Troubleshooting

Common issues and fixes for md-kb-rag.

## Startup Failures

### `unknown field '...'` on startup

Your `config.yaml` has fields that no longer exist (e.g. `chunking.strategy`, `chunking.chunk_overlap`, `webhook.port`). The schema uses `#[serde(deny_unknown_fields)]`, so any unrecognized key is a hard error.

**Fix:** Compare your config against [config.example.yaml](../config.example.yaml) and remove any fields not present in the example.

### `Missing required configuration` (embedding.base_url, embedding.model, qdrant.url)

Running outside Docker without the required environment variables set. The error lists all missing fields at once.

**Fix:** Either run via `docker compose` (which wires env vars automatically) or export them manually:

```bash
export EMBEDDING_BASE_URL=http://localhost:8080/v1
export EMBEDDING_MODEL=nomic-embed-text-v2-moe
export QDRANT_URL=http://localhost:6334
```

### `MCP_BEARER_TOKEN` not set

The server refuses to start without an MCP bearer token (unless `mcp.allow_unauthenticated: true` is set in config).

**Fix:** Set `MCP_BEARER_TOKEN` in your `.env` file.

## Embedding Service

### Container exits immediately

Usually means the Docker image doesn't match your hardware. The CPU image (`server`) works everywhere but is slower. GPU images need matching drivers.

**Fix:** Check the [Embedding Backends](../README.md#embedding-backends) section in the README and pick the right image for your hardware.

### `model file not found`

The `MODEL_PATH` or `MODEL_FILE` in `.env` doesn't match where the model is on disk.

**Fix:** Verify the model file exists at `$MODEL_PATH/$MODEL_FILE` and the path is correct in `.env`.

### Slow embeddings

You're likely running the CPU image when a GPU is available.

**Fix:** Switch to the appropriate GPU image in `docker-compose.yml` (see commented blocks) and add `-ngl 999` to offload all layers to GPU.

## Indexing

### Files skipped: "missing required frontmatter field"

Files are missing a field listed in `frontmatter.required`.

**Fix:** Either add the missing field to the file's frontmatter, or remove it from the `required` list in your config. Run `md-kb-rag validate` to check all files.

### `target_chunk_size must be <= max_chunk_size`

The values are swapped — `target_chunk_size` must be the smaller value.

**Fix:** Swap the values in your config. Example: `target_chunk_size: 1000`, `max_chunk_size: 1500`.

### Qdrant connection refused

Qdrant isn't running or hasn't finished starting.

**Fix:** Check `docker compose ps` — wait for the Qdrant healthcheck to pass. The default healthcheck has a 10-second start period. If running outside Docker, verify the URL in `QDRANT_URL` points to the gRPC port (6334, not 6333).

## Webhook

### 404 on `/hooks/reindex`

The webhook route is only mounted when `WEBHOOK_SECRET` is set to a non-empty value.

**Fix:** Set `WEBHOOK_SECRET` in your `.env` and restart the service.

### 401 Unauthorized

The HMAC secret in your Git forge doesn't match `WEBHOOK_SECRET`.

**Fix:** Ensure the secret value is identical in both your `.env` and your Git forge's webhook settings. Also check that `webhook.provider` matches your forge (`gitea`, `github`, or `gitlab`).

### 200 OK but no reindex happens

The push was to a branch that doesn't match `source.branch`.

**Fix:** Check that the webhook fires on pushes to the branch configured in `source.branch` (default: `master`).

## Model / Vector Issues

### Qdrant dimension mismatch after model change

Changing the embedding model (or `vector_size`) makes existing vectors incompatible.

**Fix:** Run `md-kb-rag index --full` to drop and recreate the Qdrant collection with the new dimensions.

## Apple Silicon / macOS

### No Metal support in Docker

The llama.cpp Docker images don't support Apple Metal GPU acceleration. Docker on macOS runs a Linux VM, which doesn't have access to the Metal API.

**Options:**

1. **Run llama-server natively** — `brew install llama.cpp`, then run `llama-server` with your model and point `EMBEDDING_BASE_URL` at it (e.g. `http://host.docker.internal:8080/v1` if kb-rag is still in Docker).
2. **Use the CPU Docker image** — works but is slower than native Metal.
3. **Use an external API** — point `EMBEDDING_BASE_URL` at OpenAI, Ollama, or any OpenAI-compatible endpoint and remove the `embeddings` service from compose.
