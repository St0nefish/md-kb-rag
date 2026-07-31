# Architecture

Current-state reference for md-kb-rag. For setup instructions see [`deploy/USAGE.md`](../deploy/USAGE.md); for config options see [`deploy/config.example.yaml`](../deploy/config.example.yaml).

## Overview

`md-kb-rag` is a single Rust binary that combines three concerns:

- **Indexing pipeline** — walks a markdown knowledge base, chunks and embeds files, and stores vectors + state
- **MCP server** — exposes `search`, `get_document`, `list_documents`, `get_schema`, and `update_schema` tools over Streamable HTTP (port 8001)
- **Webhook handler** — receives push events from a Git forge, pulls changes, and triggers incremental reindex

All three share the same binary and config. The `serve` subcommand runs the server (MCP + webhook); the remaining subcommands are standalone CLI operations.

### Subcommands

| Subcommand | Purpose |
|---|---|
| `serve` | Start the MCP server and webhook handler |
| `index` | Incremental reindex (changed files only) |
| `index --full` | Full reindex — drops Qdrant collection and re-embeds everything |
| `validate` | Check frontmatter on all files without indexing |
| `status` | Print collection stats and state DB info, including the document-metadata count (warns if it lags behind indexed files) |
| `health` | Query the running server's `/health` endpoint |
| `reproject-fields` | Rebuild `document_fields` from stored frontmatter JSON (state DB only, no re-embed) |

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
| `state.rs` | SQLite state DB (sqlx): tracks relative path → content hash + chunk count + schema fingerprint, plus the `documents` and `document_fields` metadata index |
| `document_fields.rs` | Projects frontmatter JSON into filterable `document_fields` rows (dot-path flattening, array expansion, numeric coercion) |
| `schema.rs` | `.kb-schema.yaml` cascade: parsing, cascade merge, `SchemaCache` tree walk + resolution, type/value checking, schema fingerprinting |
| `retrieval.rs` | Shared retrieval core: `search` and `get_document` logic used by `mcp.rs` |
| `mcp.rs` | MCP tool handlers (`search`, `get_document`, `list_documents`, `get_schema`, `update_schema`): input validation, result formatting; delegates to `retrieval.rs` / `state.rs` / `schema.rs` |
| `server.rs` | Axum server: MCP route, webhook route, bearer-token middleware, rate limiter, metadata refresh |
| `webhook.rs` | Webhook handler: provider signature verification, branch filter, git subprocess, reindex dispatch |
| `validate.rs` | Frontmatter validation against the resolved `.kb-schema.yaml` cascade for each file's path (falls back to the `frontmatter` config as the implicit root schema) |
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

**`mtime` payload field.** Each point carries an integer Unix-timestamp `mtime` field (indexed as a Qdrant integer index). `search`'s `modified_after`/`modified_before` parameters filter on it directly (see [Retrieval](#retrieval) below); documents indexed before `mtime` tracking was introduced may have `mtime = 0` and can be silently excluded by a recency filter.

## State Model

### SQLite (`state.db`)

Written to `<source.data_path>/state.db` — by default `/data/state.db`, which is inside the knowledge-base volume. No separate mount is required.

| Column | Type | Notes |
|---|---|---|
| `file_path` | TEXT PK | Relative path from `data_path` |
| `content_hash` | TEXT | SHA256 hex digest of file content |
| `chunk_count` | INTEGER | Number of chunks produced on last index |
| `schema_hash` | TEXT | Fingerprint of the `.kb-schema.yaml` cascade the file was last validated against (see [Schema Cascade](#schema-cascade)); added via a guarded `ALTER TABLE ... ADD COLUMN`, since there is no migration runner |

### Document metadata index (`documents`, `document_fields`)

Alongside `indexed_files`, `state.db` holds a document metadata index that backs `list_documents`.

**`documents`** — one row per file:

| Column | Type | Notes |
|---|---|---|
| `file_path` | TEXT PK | Relative path from `data_path` |
| `title` | TEXT | From frontmatter |
| `description` | TEXT | From frontmatter |
| `frontmatter` | TEXT (JSON) | Full frontmatter, stored faithfully |
| `mtime` | INTEGER | File modification time (Unix timestamp) |
| `content_hash` | TEXT | SHA256 hex digest of file content |
| `chunk_count` | INTEGER | Number of chunks produced on last index |
| `indexed_at` | INTEGER | When this row was last written |

**`document_fields`** — inverted index over frontmatter, one row per (file, field, value):

| Column | Type | Notes |
|---|---|---|
| `file_path` | TEXT | Joins to `documents.file_path` |
| `field` | TEXT | Dot-path (nested frontmatter flattens, e.g. `planning.prep_minutes`) |
| `value_text` | TEXT | String form of the value; booleans store as `"true"`/`"false"` |
| `value_num` | REAL | Numeric form, when applicable (enables `gte`/`lte`/`gt`/`lt` range queries); booleans store as `1.0`/`0.0` |

Arrays produce one row per element. This table is projected by `document_fields.rs` from the `documents.frontmatter` JSON and is what `list_documents`'s `filters` and `order_by` query against — with one exception: `title` and `description` are deliberately excluded from this projection and filtered against their dedicated `documents.title`/`documents.description` columns instead (`PROMOTED_FIELDS` in `document_fields.rs`), so a `filters` entry for either field still works, it just doesn't go through `document_fields`. Because those columns each hold one scalar, `all_of` with more than one value on `title` or `description` is unsatisfiable — the query resolves to no matches rather than erroring.

**Backfill:** existing deployments self-heal — on the next `index` run, files unchanged by content hash but missing metadata get their frontmatter parsed and stored into `documents`/`document_fields`, with no re-embedding and no Qdrant writes. No operator migration step is needed. After a field-projection rule change, `md-kb-rag reproject-fields` rebuilds `document_fields` from the stored frontmatter JSON alone (no markdown re-read, no re-embed).

`reproject-fields` is safe to run against a live server: `StateDb::reproject_all_fields` re-reads each document's stored frontmatter *inside* the same transaction that rewrites its projection (rather than snapshotting paths and frontmatter up front), and retries on `SQLITE_BUSY`/`SQLITE_LOCKED` — so a concurrent index run or write-tool commit can never be reverted by a reprojection that raced it. A document whose stored frontmatter JSON is unparseable is skipped (logged as a warning) rather than aborting the whole run, and the command prints the count of documents successfully reprojected.

A full reindex (`index --full`) also clears the `documents`/`document_fields` metadata index via `StateDb::clear`, alongside `indexed_files` — so a file removed from disk since the last full run cannot leave a phantom `list_documents` entry behind.

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

`domain` is not read from frontmatter — `ingest.rs`'s `with_derived_domain` (built on `derive_domain`) computes it from the document's top-level folder and inserts it into the frontmatter map before it's written to the Qdrant payload, so from the payload's perspective it's an ordinary keyword field. The same derived map is what's persisted to `documents`/`document_fields` in the state DB (see [State Model](#state-model)), so `domain` behaves identically as a `search` payload filter and as a `list_documents` filter — only its origin (derived vs. author-written) changed. A `domain:` key in a file's own frontmatter is discarded; `with_derived_domain` logs a warning when it disagreed with the folder-derived value. Documents at the knowledge-base root (no top-level folder) get no `domain` at all.

## Retrieval

`retrieval.rs` provides two shared functions consumed by `mcp.rs`:

**`search`** — embeds the query, builds a Qdrant filter map from the optional `domain`/`type`/`tags` parameters, runs the retrieval (see below), applies an optional `min_score` floor, and returns raw results. Timing (embed + search ms) is logged at `debug`.

When `search.hybrid` is enabled (the default), retrieval is **hybrid**: the query is embedded into a dense vector *and* tokenized into a BM25-style sparse vector (`sparse.rs`, pure-Rust; Qdrant applies IDF weighting server-side via the `sparse` named vector's `Modifier::Idf`). Both arms run as Query-API prefetches — each fetching `search.rrf_candidates` candidates with the same payload filters — and are fused server-side with Reciprocal Rank Fusion (RRF). This sharply improves recall for exact tokens (hostnames, error codes, CLI flags, config keys) that pure dense search ranks poorly. With `search.hybrid: false`, only the dense (`dense`) named vector is queried.

**Vector schema & migration** — collections are created with two named vectors: `dense` (cosine, `embedding.vector_size`) and `sparse` (`Modifier::Idf`). Both are always written at index time, so toggling `search.hybrid` never requires a reindex. Upgrading a knowledge base indexed by a pre-hybrid version (single unnamed vector) *does* require a one-time full reindex (`index --full`) to migrate to the named-vector schema.

**`get_document`** — resolves a user-supplied path to a file on disk:

1. **Literal resolution** — joins the path against `data_path`, canonicalizes, and checks two security conditions: the resolved path must be under `data_path` (path-traversal guard) and must match `indexing.include` patterns (file-type guard). Before this, `retrieval::kb_root_relative` strips a leading `/` so `/food/chili.md` and `food/chili.md` resolve identically — a caller has no way to know where the KB actually lives inside the container, so a leading `/` is read as "the KB root," not a filesystem path. (For backwards compatibility, an absolute path that exists literally on disk is still tried first; only when that lookup misses does the KB-root-relative reading apply.)
2. **Fuzzy basename fallback** — if the literal path is not found, fetches all indexed `file_path` values from Qdrant (capped at 10,000) and looks for an exact basename match. A unique match auto-resolves; multiple matches return an `Ambiguous` error with the candidates listed. Zero matches return a `NotFound` error with up to 3 Levenshtein-ranked suggestions.

`mcp.rs` wraps these functions with MCP-specific input validation (length limits, tag count limits) and formats the results as text for the MCP response.

This same `kb_root_relative` reading of a leading `/` is shared by every path-taking tool, not just `get_document`. The write tools (`create_document`/`edit_document`/`delete_document`, via `resolve_safe_write_path` in `mcp.rs`) and the schema tools (`get_schema`/`update_schema`, via `normalize_scope_path`) all strip a leading `/` the same way before applying their own traversal checks — `/../x` is rejected exactly like `../x` in every one of them. `edit_document` and `delete_document` resolve their `path` through the same literal-then-fuzzy-basename logic as `get_document`; `get_schema` and `update_schema` instead resolve a partial *directory* reference against the scopes `SchemaCache` knows about (`SchemaCache::match_scope_dirs`), matching on trailing path segments — a unique match resolves, several matches are refused with the candidates listed (never a silent guess), and for `update_schema` specifically, zero matches falls back to the literal path rather than erroring, since declaring a schema for a directory that has none yet is the normal way to introduce one.

**`list_documents`** — a separate path, not part of `retrieval.rs`: it queries the `documents`/`document_fields` tables in `state.db` directly (via `state.rs`/`document_fields.rs`), with no embedding call and no relevance ranking. It lists documents by frontmatter — `filters` (equality, any-of, all-of, or numeric range per dot-path field), `path_prefix`, `order_by` (`path`/`title`/`mtime`/`indexed_at`), `descending`, `limit`/`offset` for paging, and an optional `fields` projection. It complements `search`, which returns ranked *chunks* and cannot reliably enumerate a complete set of documents. The MCP response carries both plain text and `structured_content` (`total`, `returned`, `offset`, `has_more`, `documents[]`), so truncation is never silent.

`parse_field_filter` (`mcp.rs`) rejects combining set matching (`any_of`/`all_of`) with a numeric range (`gte`/`lte`/`gt`/`lt`) on the same field, and rejects `any_of` together with `all_of` on the same field — each is a validation error, not a silent pick-one.

## Schema Cascade

`schema.rs` lets a `.kb-schema.yaml` file govern its directory and everything beneath it, cascading like `CLAUDE.md`. It replaces the single global `frontmatter` config block as the source of field rules — though that block is still honored, as the implicit root schema, when no `.kb-schema.yaml` files exist.

**Resolution.** `SchemaCache::build` walks `data_path` once (independent of `indexing.include`/`exclude`, since a schema governs its subtree even where the markdown there isn't indexed), parsing every `.kb-schema.yaml` it finds and merging each against its nearest ancestor. This produces one `ResolvedSchema` per directory that declares a schema of its own. After that single walk, resolving a document's effective rules is an in-memory longest-prefix lookup (`SchemaCache::resolve_for`) — no filesystem access, and not repeated per file during indexing.

**Merge semantics.** The set of fields unions across cascade levels; a field redefined at a deeper level replaces its inherited definition wholesale. `extend: true` is the sole exception, unioning only `values` with the inherited set while everything else on the child definition still wins. `ResolvedSchema` tracks, per field, which schema file (`origin`) contributed its current definition — this backs `get_schema`'s provenance output.

**Schema fingerprint and incremental indexing.** `ResolvedSchema::fingerprint()` hashes a schema in a way that's stable regardless of map iteration order. `ingest.rs` compares both the file's content hash *and* this fingerprint against the stored `schema_hash` (see [State Model](#state-model)) before skipping a file as unchanged — matching content alone is not enough, since editing a schema doesn't touch any document's bytes. Rows written before this feature carry an empty `schema_hash`, which never matches a real fingerprint, so the first index run after upgrading revalidates every file once. Editing a root-level `.kb-schema.yaml` changes the root fingerprint and so revalidates the whole KB on the next run. That same forced revalidation is also when `with_derived_domain` (re)writes every document's `domain` from its folder — see the note on the Qdrant payload schema above — so an upgrade can change the effective `domain` of documents whose old frontmatter value disagreed with their folder.

**Scope freezing.** A `.kb-schema.yaml` that fails to parse freezes its entire subtree (`SchemaCache::is_frozen`): nothing under it is indexed or re-indexed, and existing index entries are left exactly as they are — the cache never falls back to the parent schema, since that would silently apply the wrong rules across a whole subtree. `md-kb-rag validate` lists every broken scope in a `SCHEMA ERRORS` section (see `main.rs`) and treats it as a failure under `validation.strict`. A file over `MAX_SCHEMA_FILE_BYTES` (256 KB, `schema.rs`) is rejected on its `fs::metadata` size alone, before `read_to_string`/parsing ever runs, and freezes its subtree through the same `broken` map as any other unparseable schema.

`index --full` refuses to run at all while `schemas.broken_scopes()` is non-empty, naming the offending directories in the error rather than proceeding: a full run drops and recreates the Qdrant collection, and a frozen scope is skipped during the rebuild, so its vectors would be lost outright instead of merely going stale. Incremental indexing is unaffected by scopes frozen elsewhere in the tree and is the way to make progress while a schema is broken.

**Field shape validation.** `validate_raw` (`schema.rs`) rejects a field definition that declares both a scalar `type` and nested `fields:` — a field is either a value or a container, never both (`type: object` is exempt, since `object` inherently means "has nested fields"). This applies uniformly whether the schema was hand-edited or produced by `update_schema`'s `set_field`.

**Typed Qdrant indexes.** `SchemaCache::all_indexed_fields()` collects every dot-path declared `indexed: true` anywhere in the tree, along with the payload index kind its declared type needs (`ResolvedSchema::index_kind`): integer/number/boolean fields get Integer/Float/Bool payload indexes instead of a blanket Keyword index, enabling range and comparison filters. Payload indexes are created once for the whole collection, so a field declared only in a deep scope is still registered up front. If two scopes declare the same path with different types, the first one encountered wins and the server logs a warning — one collection can't hold two index kinds for one path; the operator must delete the stale payload index in Qdrant and reindex to pick up a type change.

Separately, the Qdrant call that actually creates a payload index (any schema-declared field, plus the built-in `mtime` index) can itself fail — typically because the field is already indexed in Qdrant under a different kind. `QdrantStore::ensure_collection` logs that at `error` level and continues rather than aborting the run or server startup: a filter on the affected field still returns correct results, just without the index to speed it up. As above, the fix is to delete the stale payload index in Qdrant and reindex.

**`get_schema` / `update_schema`.** Both tools build a fresh `SchemaCache` per call rather than caching across requests, so they always see the on-disk schema state. Both resolve a possibly-partial `path` against known scopes first (see [Retrieval](#retrieval) above for the shared partial-match/ambiguity contract); `get_schema` then resolves the merged rules for that path (a document resolves via its parent directory) and returns them with per-field provenance. `update_schema` applies a constrained `SchemaEdit` (`add_values` / `remove_values` / `set_field` / `remove_field`) to the target directory's raw schema file via `SchemaFile::apply`, renders it back to YAML, and — critically — re-parses that YAML before writing, so a change that would fail to round-trip (and freeze the subtree) is refused instead of committed. Before writing, it calls `SchemaCache::resolve_with_candidate` per document to see what that document's own effective schema *would* be under the proposed edit — rebuilding its full ancestor chain with the candidate substituted in, since merge is per-field and a deeper scope that redeclares one field still inherits the rest. Each document already indexed under that scope is then validated against its own resolved candidate (via the `documents`/`document_fields` metadata index, not a markdown re-read), with schema defaults applied first so a required field that carries a default is not reported as breaking documents that omit it. Documents that would fail block the write unless `force` is set; `dry_run` reports the same check without writing. A successful write goes through the same commit-and-push path as the document write tools and triggers an incremental reindex.

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
| Path traversal | Every path-taking tool (`get_document`, `edit_document`, `delete_document`, `create_document`, `get_schema`, `update_schema`) resolves and canonicalizes paths, then checks `starts_with(data_path)`; a leading `/` is treated as the KB root, not a filesystem escape hatch, and `..` components are rejected either way (see [Retrieval](#retrieval)) |
| File-type restriction | `get_document` and the write tools check the resolved path against `indexing.include` glob patterns |
| Facet value sanitization | MCP server instructions embed live facet values (available domains/types/tags) read from **indexed document frontmatter** in Qdrant; these are sanitized (control characters stripped, length-capped) before inclusion to mitigate prompt-injection against connected AI clients |
| Rate limiting | Configurable token-bucket rate limiter on the MCP endpoint (`rate_limit.per_second`, `rate_limit.burst_size`) |

## Configuration and Environment Variables

See [`deploy/config.example.yaml`](../deploy/config.example.yaml) for all options with defaults and [`README.md`](../README.md#configuration) for the env-var table.

## Roadmap

Hybrid sparse+dense retrieval with RRF fusion (#55) is implemented (see [Retrieval](#retrieval)). Remaining retrieval enhancements — cross-encoder reranking (#56) and power-ups such as score explanation, recency filters, and a local CLI search (#57) — are tracked in GitHub issues.
