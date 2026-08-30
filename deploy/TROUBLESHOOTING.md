# Troubleshooting

Common issues and fixes for md-kb-rag.

## Backup and Recovery

Of the three stateful things this project touches — the KB's git clone, `state.db`,
and the Qdrant collection — **only the git repo is authoritative.** `state.db` (the
SQLite state and metadata index) and Qdrant's vectors are both fully **derived** from
the corpus: every row and every point can be rebuilt from the markdown files and their
frontmatter via `md-kb-rag index --full`. So the short version of a backup policy
here is: back up the git repo the way you'd back up any other git-hosted content
(forge-level backups, a mirror remote, or trusting the forge's own durability), and
you've covered the one thing that genuinely cannot be reconstructed. Backing up
`state.db` or the `qdrant` volume is optional — see "What backing up the derived
stores buys you" below for what it saves you over a rebuild.

### Recovering after losing `state.db` and/or the Qdrant volume

1. Confirm (or restore) that the KB clone is intact and reachable at `GIT_URL`/
   `GIT_BRANCH`. If the container's data volume was wiped entirely, a fresh start
   re-clones it automatically — see [Knowledge Base Storage](USAGE.md#knowledge-base-storage)
   in USAGE.md.
2. Run a full reindex:

   ```bash
   docker compose exec kb-rag md-kb-rag index --full
   ```

   This drops and recreates the Qdrant collection, clears `state.db`'s
   `indexed_files`/`documents`/`document_fields` tables, and re-processes every file
   from scratch — re-chunking, re-embedding, and rebuilding the metadata index.
3. Confirm the recovery with `md-kb-rag status --json` (or `GET /status`): a healthy
   result has `qdrant_points_deficit` at or near zero, and `indexed_files`/`documents`
   counts matching your corpus size.

### `index --full` refuses to run: a frozen schema blocks recovery

If any `.kb-schema.yaml` in the tree is currently invalid, `index --full` **refuses to
run at all**, naming the offending directories rather than silently skipping them —
because a full run drops and rebuilds the whole collection, and a frozen scope's
documents would be skipped during that rebuild, permanently losing their vectors
instead of merely staying stale:

```text
Refusing a full reindex while 1 schema file(s) are invalid (food/recipes). A full
run rebuilds the collection from scratch and cannot reindex frozen scopes, so
their vectors would be lost. Fix the schema(s), or run a scoped/incremental
index instead.
```

This means a disaster recovery can be blocked by a pre-existing, unrelated schema
problem — see [Documents stop being reindexed after a schema edit](#documents-stop-being-reindexed-after-a-schema-edit-schema-frozen)
below for how to find and fix it. Once the schema is valid again, retry
`index --full`. An incremental `md-kb-rag index` (no `--full`) is unaffected by scopes
frozen elsewhere in the tree and can make partial progress in the meantime, but it
will not touch the frozen scope's own documents either way.

### A Qdrant wipe with `state.db` intact

If only the Qdrant volume is lost — a `docker volume rm` on the wrong volume, disk
corruption isolated to Qdrant's storage — while `state.db` survives, the failure mode
is different from a full loss: the server's startup `ensure_collection` step silently
recreates an empty collection, and because `state.db` still believes every file is
already indexed and unchanged, an ordinary incremental reindex or reconcile sweep
finds nothing to do. Historically (#155) this was a **silent, permanent blackout** —
search would run against an empty collection with no error surfaced anywhere.

As of the current server, this is detected passively: `GET /status` and
`GET /metrics` compare `state.db`'s summed chunk count against Qdrant's actual point
count on every request, and report an error (in `store.errors`, and via the
`kb_qdrant_points_deficit` Prometheus gauge) whenever Qdrant is short by more than a
small slack — a real wipe blows past that slack by orders of magnitude, since it
zeroes the whole collection at once rather than losing a handful of points. Detection
is currently passive only — the check tells you something is wrong, it does not
itself trigger a reindex. If you see this error, or a `kb_qdrant_points_deficit`
metric that stays large and positive, run `md-kb-rag index --full` to force a full
reconcile (subject to the frozen-schema caveat above).

### What backing up the derived stores buys you

Skipping a backup of `state.db`/the Qdrant volume is safe — nothing about document
*content* is ever at risk, since that only ever lived in the git repo — but a rebuild
from scratch isn't free:

- **Time, and on a paid embedding endpoint, money.** A full reindex re-embeds every
  chunk in the corpus from nothing; there's no partial-credit path. For a large KB, or
  a metered embedding API, that's a real cost worth weighing against the effort of
  also backing up the Qdrant volume and `state.db`.
- **`indexed_at` history.** Rebuilt rows get a fresh `indexed_at`, so anything that
  trends "when was this document last touched" loses that history across the rebuild.
- **In-flight `/status` counters.** The last completed run's outcome tallies and any
  run-in-progress state are process-global and reset on restart regardless of what's
  backed up — they aren't part of either persisted store to begin with.

## Startup Failures

### `unknown field '...'` on startup

Your `config.yaml` has fields that no longer exist (e.g. `chunking.strategy`, `chunking.chunk_overlap`, `webhook.port`). The schema uses `#[serde(deny_unknown_fields)]`, so any unrecognized key is a hard error.

**Fix:** Compare your config against [config.example.yaml](config.example.yaml) and remove any fields not present in the example.

### `Missing required environment variable(s)` (EMBEDDING_BASE_URL, EMBEDDING_MODEL, QDRANT_URL)

Running outside Docker without the required environment variables set. The error lists all missing vars at once, by bare env var name:

```text
Missing required environment variable(s):
  - EMBEDDING_BASE_URL
  - EMBEDDING_MODEL
  - QDRANT_URL
```

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

The auto-clone on startup found a non-empty `DATA_PATH` without a `.git` directory. This happens when files already exist in the volume (e.g. leftover data from a previous setup).

**Fix:** Remove the volume and let the auto-clone start fresh:

```bash
docker compose down kb-rag
docker volume rm <project>_kb_data
docker compose up -d kb-rag
```

### `/data/.git: Permission denied` on startup

The image runs as `nonroot` (uid 65532) unconditionally — this is baked into the Dockerfile, not something you opt into with a compose `user:` directive. Docker initializes a fresh named volume mount point by copying ownership from the image's directory at that path, if one exists. Images built before this was fixed had no `/data` directory in the image, so Docker created the mount point `root:root`, and the non-root container couldn't write to it — hitting every fresh deployment, `user:` uncommented or not.

Images from this fix forward create and chown `/data` in the image, so a fresh named volume comes up already writable by `nonroot` and this shouldn't occur. If you hit it anyway (e.g. a volume created by an older image before upgrading), fix the existing volume's ownership once:

```bash
docker run --rm -v <volume_name>:/data --user root --entrypoint chown \
  ghcr.io/st0nefish/md-kb-rag:latest 65532:65532 /data
```

If you're running the container with a custom compose `user:` override instead of the image default, replace `65532:65532` with that uid:gid. This only needs to be done once per pre-existing volume.

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

On AMD RDNA3/RDNA4, check `rocm-smi` at idle afterwards — offloading an encoder
through ROCm can pin the card at 100% forever. See [GPU Backends](#gpu-backends).

## GPU Backends

### GPU pinned at 100% while completely idle (AMD ROCm)

**Symptom.** An AMD GPU serving embeddings or reranking sits at 100% utilisation,
core clock at maximum, and 80–100W draw — continuously, while the server handles
no requests at all. Memory clock stays at idle, which is the tell: real inference
saturates memory bandwidth, a busy-wait loop does not.

Confirm it with `rocm-smi` while nothing is being indexed or searched:

```text
GPU  Temp    AvgPwr  SCLK     MCLK   VRAM%  GPU%
0    68.0c   97.0W   3423Mhz  96Mhz  76%    100%    <- spinning
1    38.0c   14.0W   41Mhz    96Mhz  78%    3%      <- healthy idle
```

Stop the container; if utilisation drops to ~3% and power to ~15W, that container
is the cause.

**Cause.** An AMD MES firmware bug affecting RDNA3/RDNA4 (gfx11xx/gfx12xx), tracked
as [ROCm/ROCm#5706](https://github.com/ROCm/ROCm/issues/5706). It is a HIP runtime
problem, not a llama.cpp one — it also reproduces under PyTorch-ROCm with no
workload at all ([ROCm/ROCm#6298](https://github.com/ROCm/ROCm/issues/6298)).

On this project's workload it appears specifically when an **encoder** model is
offloaded to GPU (`-ngl` > 0): both embedding and reranking servers trigger it, in
either `--embeddings` or `--reranking` mode. Decoder models on the same image and
host do not.

**Fix, in order of preference:**

1. **Update the GPU firmware.** MES firmware `0x8b` (amdgpu 31.20.0+) resolves it at
   the source. Check what you have:

   ```bash
   sudo grep -i "^MES feature" /sys/kernel/debug/dri/*/amdgpu_firmware_info
   ```

   `0x81` and similar are affected. Note that a distribution `linux-firmware`
   package may be far older than your GPU — check its snapshot date before
   assuming an upgrade will help.

2. **Use the Vulkan backend for the encoders** (`compose-vulkan.yml`). Vulkan does
   not go through HIP and does not exhibit the bug. It costs roughly 40–50% more
   time per rerank batch than ROCm, which is usually a good trade against ~80W
   burned continuously. GGUF models and all server flags carry over unchanged —
   only the image tag differs.

3. **Run the encoders on CPU.** Avoids the spin, but for reranking specifically
   expect roughly 25–30× the latency; measured at 76s versus 2.8s for 150
   candidates on one deployment. Viable for embeddings, which are cheap per query;
   generally not viable for reranking.

Things that do **not** fix it, all tested: `--parallel 1`, `--poll 0`,
`HSA_ENABLE_INTERRUPT=1`, removing `HSA_OVERRIDE_GFX_VERSION`, and
`GPU_MAX_HW_QUEUES=1` (which is reported to work for decoder models, but does not
help encoders).

### Reranking returns 500 on every request

**Symptom.** `Reranker unavailable, falling back to fused order` in the logs, or
searches that take tens of seconds and then return unranked results. Querying the
reranker directly returns:

```json
{"error":{"code":500,"message":"input (518 tokens) is too large to process. increase the physical batch size (current batch size: 512)"}}
```

**Cause.** A single candidate chunk exceeds the reranker's physical batch size, and
the server rejects the whole request rather than that one document. llama.cpp
defaults `--ubatch-size` to 512 tokens; chunks at the default `max_chunk_size` of
1500 characters can exceed that.

**Fix.** md-kb-rag now truncates each document to `chunking.max_chunk_size` bytes
before sending it, so an over-long chunk degrades *that document's* score instead of
costing the whole query its reranking. When this happens you get one warning per
request rather than a silent fallback:

```text
Truncated 1 of 50 rerank documents to the 1500-byte budget (longest was 2199 bytes).
```

The budget deliberately has no knob of its own — it follows `chunking.max_chunk_size`,
so the reranker accepts exactly what the chunker is configured to emit. It is applied
only to the reranker's view of the text; returned document content is unaffected.

That bound is in **bytes**, not tokens, and the two are not the same thing. Roughly
four bytes per token holds for English prose (1500 bytes ≈ 375 tokens, comfortably
inside llama.cpp's 512 default), but CJK text, base64 blobs and long unbroken
identifiers tokenize far worse per byte. So if you raise `max_chunk_size` much past
~2000, or your corpus is not mostly English prose, also raise the reranker's batch and
context sizing so a query-plus-document pair still fits:

```yaml
      --ctx-size 16384
      --batch-size 2048
      --ubatch-size 2048
```

Size `--ctx-size` for the number of parallel slots as well — llama.cpp divides it
across them, and it will cap per-slot context to the model's training context.
Going too large fails at startup with `failed to fit params to free device memory`
when the card is shared with another model.

**Note on the "tens of seconds" symptom.** A rerank failure is now given a ~5s retry
budget rather than 120s. A 500 like the one above is deterministic — it never succeeds
on retry — so the old budget turned a sub-second failure into an MCP tool timeout that
made search look broken. Reranking is an enhancement whose fallback (fused order) is
serviceable, so it now gives up quickly and returns results.

### Reranking is enabled but never runs

**Symptom.** `reranking.enabled: true`, the reranker container is healthy, but
`search` with `explain: true` returns `pre_rerank_score: null` on every result and
ranking looks like plain vector fusion.

**Cause.** The reranker is being called with an empty document list, so it returns
immediately and the fused order stands. Historically this was caused by a payload
key mismatch between what indexing wrote and what retrieval read (fixed in #125,
which routes both through `qdrant::CHUNK_TEXT_KEY`).

**Fix.** Confirm the reranker is reachable and returns scores when called directly.
If it does, and `pre_rerank_score` is still null, the candidate list is empty before
it is sent — check that chunks carry text in the payload field retrieval reads.

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

The push was to a branch that doesn't match `GIT_BRANCH`.

**Fix:** Check that the webhook fires on pushes to the branch configured in `GIT_BRANCH` (default: `master`).

### Git fetch failed (500 Internal Server Error)

The webhook verified successfully but the in-container `git fetch` or `git merge` failed. Common causes:

- **Wrong `GIT_URL`** — check the URL is correct and reachable from inside the container.
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

Changing the embedding model (or `EMBEDDING_VECTOR_SIZE`) makes existing vectors incompatible.

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

`Webhook pull applied; marking changed paths dirty` (INFO, logged with `provider`, `branch`, and a `changed` count) — the webhook passed signature + branch checks, fetched and merged the pull, diffed the range it pulled in, and marked exactly those paths dirty for the reindex worker. `Webhook accepted with no git_url configured; marking a full reconcile` (INFO) is the same acceptance path when `source.git_url` is unset: there's nothing to fetch or diff, so the whole corpus is marked for a full reconcile instead of specific paths.

Either way, marking is instant — the response (`200 "Changes queued for indexing"`) returns before any actual indexing work happens. There is no "reindex already in progress, coalescing/skipping" log line anymore, because there is no longer a single-flight lock a webhook can lose a race for: every accepted delivery's paths land on the same `ReindexQueue`, which coalesces overlapping work rather than dropping any of it. A burst of pushes shows as a run of `Webhook pull applied` lines, one per delivery, each with its own `changed` count — none of them skipped. To confirm the marked work actually got indexed (as opposed to just accepted), look for the reindex worker's own output: the `Indexing run complete` summary below, or a `warn!` naming a unit that's being retried or dropped after repeated failure.

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

`git fetch timed out after Xs` / `git merge timed out after Xs` (ERROR) — webhook-triggered git subprocess exceeded the 120-second timeout. Check that `GIT_URL` is reachable from inside the container.

`Could not read mtime for '...', defaulting to 0` (WARN) — filesystem metadata was unavailable. The file is still indexed; `mtime` in the Qdrant payload will be `0`.

## Apple Silicon / macOS

### No Metal support in Docker

The llama.cpp Docker images don't support Apple Metal GPU acceleration. Docker on macOS runs a Linux VM, which doesn't have access to the Metal API.

**Options:**

1. **Run llama-server natively** — `brew install llama.cpp`, then run `llama-server` with your model and point `EMBEDDING_BASE_URL` at it (e.g. `http://host.docker.internal:8080/v1` if kb-rag is still in Docker).
2. **Use the CPU Docker image** — works but is slower than native Metal.
3. **Use an external API** — point `EMBEDDING_BASE_URL` at OpenAI, Ollama, or any OpenAI-compatible endpoint and remove the `embeddings` service from compose.
