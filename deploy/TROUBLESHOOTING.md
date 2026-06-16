# Troubleshooting

Common issues and fixes for md-kb-rag.

## Startup Failures

### `unknown field '...'` on startup

Your `config.yaml` has fields that no longer exist (e.g. `chunking.strategy`, `chunking.chunk_overlap`, `webhook.port`). The schema uses `#[serde(deny_unknown_fields)]`, so any unrecognized key is a hard error.

**Fix:** Compare your config against [config.example.yaml](config.example.yaml) and remove any fields not present in the example.

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

### `destination path '.' already exists and is not an empty directory`

The auto-clone on startup found a non-empty `data_path` without a `.git` directory. This happens when files already exist in the volume (e.g. leftover data from a previous setup).

**Fix:** Remove the volume and let the auto-clone start fresh:

```bash
docker compose down kb-rag
docker volume rm <project>_kb_data
docker compose up -d kb-rag
```

### `/data/.git: Permission denied` on startup

The named volume was created as root but the container runs with a non-root `user:` directive. The container can't write to the volume.

**Fix:** Set the volume ownership to match your container user before starting:

```bash
docker run --rm -v <volume_name>:/data --user root --entrypoint chown \
  ghcr.io/st0nefish/md-kb-rag:latest <uid>:<gid> /data
```

Replace `<uid>:<gid>` with the values from your compose `user:` setting (e.g. `1000:1000`). This only needs to be done once per volume.

## Embedding Service

### Container exits immediately

Usually means the Docker image doesn't match your hardware. The CPU image (`server`) works everywhere but is slower. GPU images need matching drivers.

**Fix:** Check the compose templates in `deploy/templates/` or the [Embedding Backends](../README.md#embedding-backends) section in the README and pick the right image for your hardware.

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

### Git fetch failed (500 Internal Server Error)

The webhook verified successfully but the in-container `git fetch` or `git merge` failed. Common causes:

- **Wrong `source.git_url`** — check the URL is correct and reachable from inside the container.
- **Bad or expired `GIT_PULL_TOKEN`** — for private HTTPS repos, the token needs read repository access for webhook pulls (and **write** access if you use the MCP write tools, which push commits back). Regenerate it in your forge's settings.
- **SSH URL without keys** — SSH URLs bypass token injection, but the container needs SSH keys configured. For Docker deployments, HTTPS with a token is simpler.
- **Diverged history** — the merge uses `--ff-only` and will fail if the local branch has diverged. This usually means someone modified files directly in the bind-mounted directory. Using a named volume (recommended) avoids this by keeping the repo inaccessible from the host.

**Fix:** Check `docker logs kb-rag` for the specific error (tokens are redacted in log output). Verify the URL and token work from the host:

```bash
git ls-remote "https://<token>@your-forge.example.com/org/repo.git"
```

### 302 redirect on webhook

Your reverse proxy is redirecting HTTP to HTTPS, but the webhook is configured with an HTTP URL.

**Fix:** Update the webhook URL in your forge to use `https://`.

## Model / Vector Issues

### Qdrant dimension mismatch after model change

Changing the embedding model (or `vector_size`) makes existing vectors incompatible.

**Fix:** Run `md-kb-rag index --full` to drop and recreate the Qdrant collection with the new dimensions.

## Logging and Observability

### `RUST_LOG` syntax and useful presets

Set `RUST_LOG` in your `.env` file or compose environment. The server warns on stderr if the value is unparseable and falls back to `info`.

| Value | Effect |
|---|---|
| `info` | Default — startup events, webhook accepts, indexing summaries |
| `md_kb_rag=debug` | Verbose app logging: per-file decisions, search timing, per-batch embed progress |
| `info,md_kb_rag::webhook=debug` | Info everywhere + detailed webhook trace |
| `debug` | Very verbose — includes library internals (noisy) |

### Key log events and how to find them

`Bearer auth rejected` (WARN) — every request rejected by MCP bearer-token auth. A flood of these means a client is using the wrong token.

`Webhook signature verification failed` (WARN) — HMAC mismatch between the forge secret and `WEBHOOK_SECRET`, logged with the provider name.

`Webhook accepted, spawning incremental reindex` / `Reindex already in progress; coalescing/skipping this webhook` — the accept/coalesce audit trail. The first means the webhook passed signature + branch checks and a reindex was started. The second means a reindex was already running, so this webhook was **skipped** (coalesced) rather than queued — only one reindex runs at a time. Because reindexing is incremental over the repo's current state, the in-flight run already picks up everything pulled so far; a push that lands after that run's `git fetch` is caught by the next webhook. (If a forge sends a burst of pushes, expect one `accepted` followed by several `coalescing/skipping` lines.)

`Indexing run complete` (INFO) — a structured per-run summary logged after every indexing run:

```text
Indexing run complete discovered=42 indexed=5 skipped=36 invalid=0 empty=0 read_errors=0 orphans_removed=1 elapsed_secs=3.2
```

| Field | Meaning |
|---|---|
| `discovered` | Total files matched by include/exclude patterns |
| `indexed` | Files successfully embedded and upserted |
| `skipped` | Unchanged files (hash matched state DB) — normal in incremental mode |
| `invalid` | Files that failed frontmatter validation |
| `empty` | Files that produced no chunks (blank body after frontmatter) |
| `read_errors` | Files that could not be read (permissions, encoding errors) |
| `orphans_removed` | Qdrant points removed for files no longer on disk |
| `elapsed_secs` | Wall-clock time for the run |

If a file you expected to be indexed shows up under `skipped`, its hash matches the last indexed version — it hasn't changed since the last run. If it shows under `invalid`, run `md-kb-rag validate` for details.

`git fetch timed out after Xs` / `git merge timed out after Xs` (ERROR) — webhook-triggered git subprocess exceeded the 120-second timeout. Check that `source.git_url` is reachable from inside the container.

`Could not read mtime for '...', defaulting to 0` (WARN) — filesystem metadata was unavailable. The file is still indexed; `mtime` in the Qdrant payload will be `0`.

## Apple Silicon / macOS

### No Metal support in Docker

The llama.cpp Docker images don't support Apple Metal GPU acceleration. Docker on macOS runs a Linux VM, which doesn't have access to the Metal API.

**Options:**

1. **Run llama-server natively** — `brew install llama.cpp`, then run `llama-server` with your model and point `EMBEDDING_BASE_URL` at it (e.g. `http://host.docker.internal:8080/v1` if kb-rag is still in Docker).
2. **Use the CPU Docker image** — works but is slower than native Metal.
3. **Use an external API** — point `EMBEDDING_BASE_URL` at OpenAI, Ollama, or any OpenAI-compatible endpoint and remove the `embeddings` service from compose.
