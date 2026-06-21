# Architecture

Current-state reference for md-kb-rag. For setup instructions see [`deploy/USAGE.md`](../deploy/USAGE.md); for config options see [`deploy/config.example.yaml`](../deploy/config.example.yaml).

## Overview

`md-kb-rag` is a single Rust binary that combines three concerns:

- **Indexing pipeline** — walks a markdown knowledge base, chunks and embeds files, and stores vectors + state
- **MCP server** — exposes `search` and `get_document` tools over Streamable HTTP (port 8001)
- **Webhook handler** — receives push events from a Git forge, pulls changes, and triggers incremental reindex

All three share the same binary and config. The `serve` subcommand runs the server (MCP + webhook); the remaining subcommands are standalone CLI operations.

### Subcommands

| Subcommand | Purpose |
|---|---|
| `serve` | Start the MCP server and webhook handler |
| `index` | Incremental reindex (changed files only) |
| `index --full` | Full reindex — drops Qdrant collection and re-embeds everything |
| `validate` | Check frontmatter on all files without indexing |
| `status` | Print collection stats and state DB info |
| `health` | Query the running server's `/health` endpoint |

## Docker Topology

Three services, all defined in `docker-compose.yml` (or a hardware-specific template from `deploy/templates/`):

```text
┌─────────────────────────────────────────┐
│                 kb-rag                  │
│  (MCP :8001, webhook /hooks/reindex)    │
│                                         │
│  ┌──────────────┐  ┌───────────────┐   │
│  │  ingest.rs   │  │   mcp.rs      │   │
│  │  (indexer)   │  │   server.rs   │   │
│  │  webhook.rs  │  │   retrieval.rs│   │
│  └──────┬───────┘  └──────┬────────┘   │
└─────────┼─────────────────┼────────────┘
          │ gRPC :6334       │ gRPC :6334
          ▼                  ▼
┌─────────────────┐   (same Qdrant instance)
│    qdrant       │
│  gRPC :6334     │
│  REST  :6333    │
└─────────────────┘

          ▲ HTTP :8080
┌─────────────────┐
│   embeddings    │
│  (llama.cpp)    │
│  OpenAI-compat  │
└─────────────────┘
```

| Service | Image | Port(s) |
|---|---|---|
| `qdrant` | `qdrant/qdrant` | 6334 (gRPC, used by kb-rag), 6333 (REST, debugging) |
| `embeddings` | `ghcr.io/ggml-org/llama.cpp:server[-cuda12/-rocm/-vulkan]` | 8080 (internal only) |
| `kb-rag` | `ghcr.io/st0nefish/md-kb-rag` | 8001 (MCP + webhook, exposed to host) |

The kb-rag service waits for both `qdrant` and `embeddings` to pass their healthchecks before starting.

The MCP port (8001) is the only externally exposed port. This service is designed for intranet/tailnet deployment — it is not hardened for direct public internet exposure.

## Module Layout

| File | Purpose |
|---|---|
| `main.rs` | CLI entrypoint (clap subcommands), startup wiring |
| `config.rs` | Config deserialization (`config.yaml` + env-var overrides) |
| `ingest.rs` | Indexing pipeline: discover → hash → validate → chunk → embed → upsert |
| `chunk.rs` | Section-aware markdown chunker |
| `embed.rs` | Embedding API client (async-openai, batched, exponential backoff) |
| `qdrant.rs` | Qdrant gRPC operations: upsert, delete, search, facet fetch |
| `state.rs` | SQLite state DB (sqlx): tracks relative path → content hash + chunk count |
| `retrieval.rs` | Shared retrieval core: `search` and `get_document` logic used by `mcp.rs` |
| `mcp.rs` | MCP tool handlers (`search`, `get_document`): input validation, result formatting; delegates to `retrieval.rs` |
| `server.rs` | Axum server: MCP route, webhook route, bearer-token middleware, rate limiter, metadata refresh |
| `webhook.rs` | Webhook handler: provider signature verification, branch filter, git subprocess, reindex dispatch |
| `validate.rs` | Frontmatter validation against `frontmatter` config |
| `git.rs` | Git subprocess helpers: token injection, URL redaction, fetch/merge with timeout |

## Indexing Pipeline

```text
discover files (relative paths)
        │
        ▼
 read file content
        │
        ▼
 compute SHA256 hash
        │
   unchanged? ──── yes ──► skip (increment skipped counter)
        │ no
        ▼
 validate frontmatter
        │
  invalid? ─── yes ──► warn + skip (increment invalid counter)
        │ no
        ▼
 chunk markdown (section-aware)
        │
  empty? ──── yes ──► warn + skip (increment empty counter)
        │ no
        ▼
 embed chunks in batches
 (async-openai, configurable batch_size, exponential backoff)
        │
        ▼
 upsert points in-place by deterministic UUID5 id
 (id = UUID5(namespace, "relative_path::chunk_index"))
        │
 file shrank? ─── yes ──► delete tail points (old_count - new_count)
        │
        ▼
 update SQLite state (relative_path → hash + chunk_count)
        │
        ▼ (after all files)
 orphan removal: delete Qdrant points + state rows
 for paths in state DB not found on disk
        │
        ▼
 log structured summary:
 discovered / indexed / skipped / invalid / empty /
 read_errors / orphans_removed / elapsed_secs
```

### Key design decisions

**Relative paths as canonical keys.** File paths are stored relative to `source.data_path` everywhere: as the Qdrant `file_path` payload field, as the SQLite state key, and as the UUID5 input for point IDs. This makes the index portable across mount points. Upgrading from an older version that stored absolute paths requires a `--full` reindex.

**Upsert-in-place, no pre-delete window.** Points are upserted by their deterministic ID. A file that grows adds new tail points; a file that shrinks has its tail trimmed after the upsert. There is no window where a file's points are absent from the index.

**`mtime` payload field.** Each point carries an integer Unix-timestamp `mtime` field (indexed as a Qdrant integer index). This is stored for future filtered retrieval (e.g. "only docs modified after date X") but is not yet used in the search path.

## State Model

### SQLite (`state.db`)

Written to `<source.data_path>/state.db` — by default `/data/state.db`, which is inside the knowledge-base volume. No separate mount is required.

| Column | Type | Notes |
|---|---|---|
| `file_path` | TEXT PK | Relative path from `data_path` |
| `content_hash` | TEXT | SHA256 hex digest of file content |
| `chunk_count` | INTEGER | Number of chunks produced on last index |

### Qdrant payload schema

Each indexed chunk is stored as a Qdrant point with this payload:

| Field | Type | Indexed | Notes |
|---|---|---|---|
| `file_path` | keyword | yes | Relative path from `data_path` |
| `chunk_index` | integer | no | 0-based chunk position within the file |
| `text` | text | no | Chunk content (used in search results) |
| `line_start` | integer | no | First line of the chunk in the source file |
| `line_end` | integer | no | Last line of the chunk in the source file |
| `mtime` | integer | yes | File modification time as Unix timestamp |
| frontmatter fields | keyword / array | yes | Fields listed in `frontmatter.indexed_fields` (e.g. `type`, `domain`, `tags`, `title`) |

Keyword and array fields listed in `frontmatter.indexed_fields` get Qdrant keyword indexes, enabling exact-match and match-any filtering in the `search` tool.

## Retrieval

`retrieval.rs` provides two shared functions consumed by `mcp.rs`:

**`search`** — embeds the query, builds a Qdrant filter map from the optional `domain`/`type`/`tags` parameters, runs the retrieval (see below), applies an optional `min_score` floor, and returns raw results. Timing (embed + search ms) is logged at `debug`.

When `search.hybrid` is enabled (the default), retrieval is **hybrid**: the query is embedded into a dense vector *and* tokenized into a BM25-style sparse vector (`sparse.rs`, pure-Rust; Qdrant applies IDF weighting server-side via the `sparse` named vector's `Modifier::Idf`). Both arms run as Query-API prefetches — each fetching `search.rrf_candidates` candidates with the same payload filters — and are fused server-side with Reciprocal Rank Fusion (RRF). This sharply improves recall for exact tokens (hostnames, error codes, CLI flags, config keys) that pure dense search ranks poorly. With `search.hybrid: false`, only the dense (`dense`) named vector is queried.

**Vector schema & migration** — collections are created with two named vectors: `dense` (cosine, `embedding.vector_size`) and `sparse` (`Modifier::Idf`). Both are always written at index time, so toggling `search.hybrid` never requires a reindex. Upgrading a knowledge base indexed by a pre-hybrid version (single unnamed vector) *does* require a one-time full reindex (`index --full`) to migrate to the named-vector schema.

**`get_document`** — resolves a user-supplied path to a file on disk:

1. **Literal resolution** — joins the path against `data_path`, canonicalizes, and checks two security conditions: the resolved path must be under `data_path` (path-traversal guard) and must match `indexing.include` patterns (file-type guard).
2. **Fuzzy basename fallback** — if the literal path is not found, fetches all indexed `file_path` values from Qdrant (capped at 10,000) and looks for an exact basename match. A unique match auto-resolves; multiple matches return an `Ambiguous` error with the candidates listed. Zero matches return a `NotFound` error with up to 3 Levenshtein-ranked suggestions.

`mcp.rs` wraps these functions with MCP-specific input validation (length limits, tag count limits) and formats the results as text for the MCP response.

## Webhook Flow

```text
POST /hooks/reindex
        │
        ▼
 verify HMAC signature
 (GitHub: x-hub-signature-256 / Gitea: x-gitea-signature / GitLab: x-gitlab-token)
        │
  fail? ──► 401 + warn log
        │ ok
        ▼
 check branch filter (source.branch)
        │
  mismatch? ──► 200 (ignored, no reindex)
        │ match
        ▼
 acquire reindex try-lock (single-flight)
        │
  busy? ──► coalesce/skip (logged) + 200
        │ acquired
        ▼
 git fetch + git merge --ff-only
 (with source.git_url; git_token injected transiently, never written to disk)
 (120-second timeout on each subprocess)
        │
        ▼
 incremental reindex (same pipeline as `index` subcommand)
        │
        ▼
 release lock
```

The single-flight lock (`REINDEX_LOCK`) is an `Arc<tokio::Mutex<()>>`. The handler attempts `try_lock_owned()`: if it succeeds, the reindex runs in a spawned task holding the owned guard; if the lock is already held, the webhook is **skipped** (coalesced) — it is not queued or replayed — and the handler logs the coalesce and returns 200. Rationale: reindexing is incremental over the repo's current state, so a single in-flight run subsumes concurrent triggers. The one caveat is that a push landing *after* the in-flight run's `git fetch` is not seen by that run; it is picked up by the next webhook-triggered reindex.

## Security Model

This service is designed for intranet/tailnet deployment — the threat model assumes network-level access control at the perimeter.

| Control | Mechanism |
|---|---|
| MCP authentication | Bearer token (`mcp.bearer_token_env`); rejections logged at WARN |
| Webhook authentication | HMAC-SHA256 (`webhook.secret_env`); failures logged at WARN with provider |
| Path traversal | `get_document` resolves and canonicalizes paths, then checks `starts_with(data_path)` |
| File-type restriction | `get_document` checks the resolved path against `indexing.include` glob patterns |
| Facet value sanitization | MCP server instructions embed live facet values (available domains/types/tags) read from **indexed document frontmatter** in Qdrant; these are sanitized (control characters stripped, length-capped) before inclusion to mitigate prompt-injection against connected AI clients |
| Rate limiting | Configurable token-bucket rate limiter on the MCP endpoint (`rate_limit.per_second`, `rate_limit.burst_size`) |

## Configuration and Environment Variables

See [`deploy/config.example.yaml`](../deploy/config.example.yaml) for all options with defaults and [`README.md`](../README.md#configuration) for the env-var table.

## Roadmap

Hybrid sparse+dense retrieval with RRF fusion (#55) is implemented (see [Retrieval](#retrieval)). Remaining retrieval enhancements — cross-encoder reranking (#56) and power-ups such as score explanation, recency filters, and a local CLI search (#57) — are tracked in GitHub issues.
