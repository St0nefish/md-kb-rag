# md-kb-rag

A Docker-first RAG server that indexes markdown knowledge bases with YAML frontmatter into Qdrant and exposes them over MCP (Streamable HTTP) — semantic search, document retrieval, and agent-driven writes (create/edit/delete) that commit straight back to the knowledge base's git repo.

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
# (GIT_PULL_TOKEN is optional — only needed to clone/fetch a private knowledge-base repo)

# Download the embedding model (see "Embedding Models" below)

# Set source.git_url in config.yaml to point at your knowledge base repo
cp deploy/config.example.yaml config.yaml
# Edit config.yaml — at minimum, set source.git_url and uncomment the mount:
#   - ./config.yaml:/app/config.yaml:ro

# Start the stack (CPU mode by default)
# With git_url set, the server auto-clones the repo and runs a full index on first start
docker compose up -d

# Add MCP to Claude Code
claude mcp add --transport http kb-search \
  https://your-host:8001/mcp \
  --header "Authorization: Bearer $TOKEN"
```

### Claude Desktop

Claude Desktop has no native remote-MCP transport for bearer-token servers, so it
connects through the [`mcp-remote`](https://www.npmjs.com/package/mcp-remote) stdio
bridge. Add this to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "kb-search": {
      "command": "npx",
      "args": [
        "-y", "mcp-remote", "https://your-host:8001/mcp",
        "--header", "Authorization:${KB_AUTH}"
      ],
      "env": { "KB_AUTH": "Bearer YOUR_TOKEN" }
    }
  }
}
```

Note the `Authorization:${KB_AUTH}` form — no space after the colon. Claude Desktop
mangles arguments containing spaces, so the token is passed via the `env` block and
substituted by `mcp-remote`.

The server runs its Streamable HTTP transport in stateless mode, which matters most
for this path: there is no session for a dropped connection to invalidate, so the
bridge recovers on its own after a laptop sleep or a server restart.

The recommended setup uses a **named Docker volume** for the knowledge base. The container clones the repo on first start and pulls updates via webhook — no host-side git operations needed. See [deploy/USAGE.md](deploy/USAGE.md#knowledge-base-storage) for details on this approach vs. bind-mounting.

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
md-kb-rag get PATH           # Print one document (path resolves like the MCP tool)
md-kb-rag get PATH --start-line 40 --end-line 60   # ...or just those lines, 1-based inclusive
md-kb-rag status             # Aggregate counts + metadata breakdown
md-kb-rag status --json      # Same data as the server's /status endpoint
md-kb-rag status --files     # List every indexed file instead
md-kb-rag health             # Check if server is healthy
md-kb-rag reproject-fields   # Rebuild document_fields from stored frontmatter (no re-embed)
```

## Configuration

Every setting has **exactly one source**. Settings needed at startup — connection wiring, model identity, and secrets — come from environment variables only. Runtime and tuning settings come from `config.yaml` (or the path passed via `--config`) only. Nothing is readable from both, so there is no precedence to reason about.

**Environment only** — startup bindings, not settable in `config.yaml`:

| Env Var | Purpose | Default |
|---|---|---|
| `EMBEDDING_BASE_URL` | embeddings endpoint | *(required)* |
| `EMBEDDING_MODEL` | embedding model | *(required)* |
| `EMBEDDING_VECTOR_SIZE` | vector dimension | `768` |
| `QDRANT_URL` | Qdrant gRPC endpoint | *(required)* |
| `QDRANT_COLLECTION` | collection name | `knowledge-base` |
| `RERANKING_BASE_URL` | reranker endpoint | *(required when reranking is on)* |
| `RERANKING_MODEL` | reranker model | *(required when reranking is on)* |
| `GIT_URL` | knowledge-base repo to clone/pull | *(unset — no git integration)* |
| `GIT_BRANCH` | branch to track | `master` |
| `DATA_PATH` | knowledge base + state DB location | `/data` |
| `MCP_PORT` | listen port | `8001` |

Missing required env vars are named together in a single startup error rather than surfacing one per restart. An env var that is set but no longer honored logs a warning instead of being silently ignored.

**Indirected secret env vars** — the config field names *which* env var holds the secret, never the value itself:

| Config Field | Default env var name |
|---|---|
| `webhook.secret_env` | `WEBHOOK_SECRET` |
| `source.git_token_env` | `GIT_PULL_TOKEN` |
| `mcp.bearer_token_env` | `MCP_BEARER_TOKEN` |
| `embedding.api_key_env` | `EMBEDDING_API_KEY` |
| `reranking.api_key_env` | `RERANKING_API_KEY` |

For example, `webhook.secret_env: "WEBHOOK_SECRET"` tells the server to read the HMAC secret from the env var named `WEBHOOK_SECRET`. Change it in config if you want a different env var name.

Everything else — `search`, `reranking`, `chunking`, `indexing`, `validation`, `write`, `rate_limit`, `frontmatter`, `mcp.instructions` — lives in `config.yaml` only, and can be changed without a restart via [Config reload](#config-reload). Startup logs and `GET /status` report where every setting came from (env, yaml, or default).

See [deploy/config.example.yaml](deploy/config.example.yaml) for all options:

- **source** — Git URL (auto-cloned on first start) or bind-mount path for your knowledge base
- **indexing** — Include/exclude glob patterns
- **frontmatter** *(deprecated)* — Required fields, indexed fields, defaults, and `allowed` closed-set enums (enforced by `validate` and the write tools); prefer a root `.kb-schema.yaml` instead — see [`.kb-schema.yaml` Directory Schemas](#kb-schemayaml-directory-schemas)
- **chunking** — Markdown-aware splitting with configurable chunk size
- **embedding** — OpenAI-compatible endpoint (works with llama.cpp, vLLM, etc.)
- **qdrant** — Connection URL and collection name
- **validation** — Strict/lenient mode, optional lint command
- **webhook** — HMAC verification for Gitea/GitHub/GitLab (disabled if `WEBHOOK_SECRET` is unset)
- **mcp** — Server port, bearer token authentication, and the instructions narrative
- **write** — Behaviour of the write tools: near-duplicate detection (`dedup_enabled`, `dedup_threshold`) and the git commit identity
- **search** — Retrieval behaviour: hybrid sparse+dense search with RRF fusion (`hybrid`, default `true`) and per-arm candidate count (`rrf_candidates`). Set `hybrid: false` for legacy dense-only search. See the migration note below.

> **Hybrid search migration:** collections now use named `dense` + `sparse` vectors. Upgrading a knowledge base that was indexed by a pre-hybrid version requires a one-time full reindex (`md-kb-rag index --full`) — the old single-unnamed-vector schema is incompatible. After that, toggling `search.hybrid` needs no reindex (both vectors are always stored).

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

These are environment-only — `embedding.model` and `embedding.vector_size` are not settable in `config.yaml`. Changing either invalidates every vector already in the collection, so it requires a full reindex, not just a restart.

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

### Context Window Override

nomic-embed-text-v2-moe natively supports 8192-token context windows, but the GGUF file metadata incorrectly reports a 512-token limit. The `docker-compose.yml` command includes `--override-kv nomic-bert-moe.context_length=int:8192` to correct this, along with matching `--ctx-size`, `--batch-size`, and `--ubatch-size` flags. This allows embedding larger markdown chunks in a single pass. If you switch to a different model, adjust or remove these overrides accordingly.

### CPU (default)

Works everywhere with no special drivers. Good for small knowledge bases or initial testing. The compose file uses `ghcr.io/ggml-org/llama.cpp:server`.

### NVIDIA CUDA

Most common GPU backend. Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed on the host. In `docker-compose.yml`, uncomment the `## --- NVIDIA CUDA ---` block (which uses `server-cuda12` with `deploy.resources.reservations.devices` for GPU access), or use the `deploy/templates/compose-nvidia.yml` template.

### AMD ROCm

Best performance on AMD GPUs. Requires ROCm userspace drivers on the host. In `docker-compose.yml`, uncomment the `## --- AMD ROCm ---` block (which uses `server-rocm` with `/dev/kfd` and `/dev/dri` device access), or use the `deploy/templates/compose-rocm.yml` template.

For fine-grained control (e.g. targeting a specific GPU render node or setting `HSA_OVERRIDE_GFX_VERSION`), use a `docker-compose.override.yml`:

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
      - HSA_OVERRIDE_GFX_VERSION=12.0.1  # 11.0.0 for RDNA 3 (RX 7000), 12.0.1 for RDNA 4 (RX 9000)
      - HIP_VISIBLE_DEVICES=0
```

Find your device nodes with `ls /dev/dri/` and match card/render numbers to your target GPU. Check `getent group video render` for the correct GIDs on your system.

### AMD Vulkan

Simpler driver setup than ROCm — works with standard Mesa Vulkan drivers. In `docker-compose.yml`, uncomment the `## --- AMD Vulkan ---` block (which uses `server-vulkan` with `/dev/dri` device access), or use the `deploy/templates/compose-vulkan.yml` template. Supports multi-GPU setups.

### Apple Silicon (Metal)

Metal GPU acceleration is **not available in Docker** (Docker on macOS runs a Linux VM). Options:

1. **Run llama-server natively** — `brew install llama.cpp`, then start it with your model and point `EMBEDDING_BASE_URL` at it (`http://host.docker.internal:8080/v1` if kb-rag runs in Docker).
2. **Use the CPU Docker image** — works but is slower than native Metal.

### External API

Skip the bundled embedding service entirely. Point `EMBEDDING_BASE_URL` at any OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, TEI) and remove the `embeddings` service from compose.

## MCP Tools

The server exposes a full read/write surface over MCP. Read tools (`search`, `get_document`, `list_documents`, `get_schema`) are always safe; write tools (`create_document`, `edit_document`, `delete_document`, `update_schema`) mutate the knowledge base and commit to git.

**Path handling is unified across every tool that takes a `path`.** A leading `/` means the knowledge-base root, not a filesystem path — callers have no way to know where the KB actually lives inside the container, so `/food/recipes/chili.md` and `food/recipes/chili.md` address the same document. Path-escape protections still apply on top of that: `/../x` is rejected the same as `../x`. Partial paths resolve to a best match — `get_document` accepts a bare basename when it's unique across the index, and `get_schema`/`update_schema` accept a partial directory when it matches on trailing segments. A single match resolves silently; multiple matches are refused with the candidates listed rather than guessed at. `update_schema` is the one exception: when a partial directory matches *nothing* existing, it falls back to the literal path instead of erroring, since creating a schema for a directory that doesn't have one yet is the normal way to introduce one.

### Read

**`search`** — semantic search; returns ranked chunks with title, score, snippet, and metadata.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | string | yes | Natural-language search query |
| `domain` | string | no | Filter by domain — see note below |
| `type` | string | no | Filter by document type |
| `tags` | string[] | no | Filter by tags (match any) |
| `limit` | integer | no | Max results (default: 10, max: 50) |

**`domain` is derived, not authored.** `domain` is computed from each document's top-level folder name (a file at `infrastructure/docker-compose.md` gets `domain: infrastructure`; a file at the KB root has no domain) and written into both the Qdrant payload and the SQLite metadata index — it is not read from a `domain:` frontmatter key. A `domain:` key an author writes anyway is overwritten on the next index run, and the server logs a warning when the two disagree. This changes where the value comes from, not how you filter on it: `domain=...` here, the CLI's `--domain` flag, and `list_documents(filters={"domain": ...})` all still work exactly as before.

**`get_document`** — fetch the raw markdown (including frontmatter) for one document, in full or by line range.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | Path relative to the KB root (as returned by `search`), or a unique basename. A leading `/` also means the KB root — see path handling above |
| `start_line` | integer | no | First line to return, 1-based and inclusive. Omit to start at line 1 |
| `end_line` | integer | no | Last line to return, 1-based and inclusive. Omit to read to the end |

Returns both plain text and `structured_content` with `path`, `content`, `content_hash`, `start_line`, `end_line`, `total_lines`, and `partial`. The line fields are always present, on a full read as well as a partial one, so paging through a long document never requires guessing where it ended.

Line ranges are inclusive on both ends, and an `end_line` past the last line is clamped rather than rejected — `end_line` in the response reports what was actually served. A `start_line` past the last line *is* an error, and says how many lines the document has. Content is sliced byte-exactly: line endings and an unterminated final line survive, so a slice can be handed straight back to `edit_document` as an `old_string`.

**`content_hash` always covers the whole document, never the slice.** Its purpose is `edit_document`'s `expected_hash`, which guards the file on disk — so reading lines 40–60 of a document and then editing it works exactly as it does after a full read. The flip side is that it is not a checksum of the bytes you were handed.

**`list_documents`** — lists documents by frontmatter, with no relevance ranking and no embedding call. Complements `search`, which returns ranked *chunks* and cannot reliably enumerate a complete set.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `filters` | object | no | Keyed by frontmatter field (dot-paths for nested fields, e.g. `planning.prep_minutes`). A scalar means equality (`{"type": "guide"}`); an array means any-of (`{"tags": ["recipe","dinner"]}`); an object means all-of or a numeric range (`{"tags": {"all_of": ["recipe","dinner"]}}`, `{"planning.prep_minutes": {"lt": 30}}`). Range operators: `gte`, `lte`, `gt`, `lt`. Also `any_of`, `all_of` |
| `path_prefix` | string | no | Restrict to a folder, e.g. `lifestyle/kitchen/recipes/` |
| `order_by` | string | no | `path` (default), `title`, `mtime`, `indexed_at` |
| `descending` | boolean | no | Reverse sort order |
| `limit` | integer | no | Max results (default: 100, max: 1000) |
| `offset` | integer | no | Paging offset |
| `fields` | string[] | no | Frontmatter dot-paths to return per document; omit for all |

Per field, the operators are **mutually exclusive**: use set matching (`any_of`/`all_of`) *or* a numeric range (`gte`/`lte`/`gt`/`lt`), never both on the same field, and never `any_of` together with `all_of`. Mixing them is a validation error, not silently-honor-one-of-them behavior.

`title` and `description` are filterable too — they're matched against dedicated columns rather than the generic dot-path projection every other frontmatter field goes through. Since each holds a single scalar, `all_of` with more than one value on `title` or `description` can never match anything (it returns zero results rather than erroring).

Returns both plain text and MCP `structured_content` with `total`, `returned`, `offset`, `has_more`, and `documents[]` — truncation is never silent.

**`get_schema`** — show the fully merged frontmatter rules governing a path, with per-field provenance (which `.kb-schema.yaml` declared each field). See [`.kb-schema.yaml` Directory Schemas](#kb-schemayaml-directory-schemas) below for how the cascade works.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | no | Directory or document path to resolve rules for; a partial directory resolves if it matches one scope uniquely (multiple matches are refused with the candidates listed); omit for the root |
| `fields` | string[] | no | Only report these dot-paths; omit for all |
| `values_only` | boolean | no | Only report fields that declare a closed value set |

Returns plain text plus `structured_content` (`path`, `frozen`, `frozen_reason`, `fields[]`, each field carrying `type`, `required`, `indexed`, `values`, `default`, `open`, and `declared_in`).

### Write

All three write tools share the same pipeline: **path-safety guard** (no `..`, no symlink escapes, must match `indexing.include`) → **frontmatter validation** → **filesystem write** → **git commit with provenance trailers** (`Tool: md-kb-rag`, `Operation: <tool>`) → **push to the remote** → **incremental reindex** (serialized against webhook reindexes via an internal lock). Each returns a summary line with the commit SHA plus a unified diff. Commits are authored under the `write.commit_author_*` identity so tool edits are easy to spot in `git log`.

**`create_document`** — create a new file. Validates that the document doesn't already exist, then runs a **near-duplicate check**: it embeds the content and, if an existing document scores above `write.dedup_threshold`, refuses the write and names the match (pass `force_new: true` to override). That score is always a dense cosine similarity — the check is pinned to dense-only retrieval with reranking detached, so it is unaffected by `search.hybrid` and `reranking.enabled`.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | Path of the new document, relative to the KB root (e.g. `sysadmin/docker/foo.md`); a leading `/` also means the KB root |
| `content` | string | yes | Full markdown including YAML frontmatter |
| `message` | string | no | Commit subject (default: `docs: add <path>`) |
| `force_new` | boolean | no | Skip the duplicate-detection gate (default: `false`) |

**`edit_document`** — modify an existing file. Two mutually-exclusive modes:

- **Surgical** — `old_string` + `new_string`: replaces a single unique occurrence (Claude Code-style). Errors if `old_string` is missing or appears more than once.
- **Full-replace** — `content`: swaps the entire file (must include valid frontmatter).

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | Resolved like `get_document` |
| `old_string` | string | conditional | Surgical mode: exact text to find (must be unique) |
| `new_string` | string | conditional | Surgical mode: replacement text |
| `content` | string | conditional | Full-replace mode: entire new file content |
| `message` | string | no | Commit subject (default: `docs: update <path>`) |

**`delete_document`** — remove a file. Commits the deletion (with provenance trailers) and pushes, then purges the document's vectors from Qdrant and its row from the state DB directly.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | Resolved like `get_document` |
| `message` | string | no | Commit subject (default: `docs: delete <path>`) |

### Schema

**`update_schema`** — edit a directory's `.kb-schema.yaml` through constrained operations instead of free-form text. Before writing, the change is validated against every document already under that scope, using the frontmatter stored in the metadata index (no markdown re-read). If any document would fail the new rules, the change is refused and they're listed; `force` applies anyway, `dry_run` reports what would happen without writing. The rendered YAML is re-parsed before writing, so an unparseable schema can never be committed. Like the write tools above, the file is written temp-then-rename, committed and pushed, and triggers a reindex.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | no | Directory whose schema to edit; a partial directory resolves if it uniquely matches an existing scope (multiple matches are refused with the candidates listed, no match falls back to the literal path so a new scope can be introduced); omit for the root |
| `operation` | string | yes | One of `add_values`, `remove_values`, `set_field`, `remove_field` |
| `field` | string | yes | Field the operation targets (dot-path for nested fields) |
| `values` | string[] | conditional | For `add_values` / `remove_values` |
| `definition` | object | conditional | For `set_field` — accepts the same keys as a `.kb-schema.yaml` entry (`type`, `required`, `indexed`, `values`, `extend`, `default`, `open`) |
| `dry_run` | boolean | no | Report the effect without writing (default: `false`) |
| `force` | boolean | no | Apply even if existing documents would fail the new rules (default: `false`) |

When validation fails, write tools return a structured error whose `data.field_errors` array names each offending `field`, the `rule` it broke (`required` / `allowed_value` / `lint` / `type_mismatch` / `closed_object`), and — for closed-set fields — the value it `got` and the values it `expected`, so an agent can self-correct.

The server also advertises **dynamic instructions** to connected clients: the configured `mcp.instructions` narrative, the distinct filter values (domains, types, tags) discovered in the live index, and a write-authoring section listing the required frontmatter fields and any fixed allowed values. See the [`write` and `mcp` sections of the config reference](deploy/config.example.yaml) and [deploy/USAGE.md](deploy/USAGE.md#agent-write-tools) for details.

## `.kb-schema.yaml` Directory Schemas

A file named `.kb-schema.yaml` governs its directory and everything beneath it, cascading like `CLAUDE.md` — including at the knowledge-base root. `frontmatter` in `config.yaml` used to be the *only* way to declare root-level rules; it's now a deprecated fallback (see [Backward compatibility](#backward-compatibility) below), and a root `.kb-schema.yaml` — authored with the exact same syntax as any other directory — is the preferred way to declare them, since it's part of the knowledge base's own git repo rather than deployment config on the container host.

Top-level folder names are the KB's areas — this is also what `domain` is derived from (see the [`search`](#read) note above). The MCP server's dynamic instructions list them from a directory read, in addition to any `Available domain: ...` facet it advertises when `domain` is indexed at the root, whether via the deprecated `frontmatter.indexed_fields` or an `indexed: true` entry for `domain` in a root `.kb-schema.yaml`.

### Syntax

```yaml
fields:
  planning:
    type: object
    open: false          # reject undeclared keys under planning.* (default: true)
    fields:
      prep_minutes: { type: integer, indexed: true }
      effort:       { type: enum, values: [low, medium, high], indexed: true }
  tags:
    type: list
    extend: true          # union values with the inherited definition instead of replacing
    values: [dinner, quick]
```

Nested authoring (as above) and flat dot-paths (`planning.prep_minutes:`) are equivalent — nesting is sugar flattened at parse time. Internally, schemas are a flat dot-path map.

**Types:** `text`, `integer`, `number`, `boolean`, `enum`, `list`, `date` (`YYYY-MM-DD`), `timestamp` (RFC 3339), `object`. Declared types are strictly enforced with no coercion — `prep_minutes: "45"` fails against `type: integer`. Undeclared fields are not type-checked and remain legal.

A field can't declare both a scalar `type` and nested `fields:` — it's either a value or a container, not both (`type: object` is the exception, since `object` means "has nested fields"). `update_schema` enforces this the same way a hand-edited `.kb-schema.yaml` does.

### Cascade and merge rules

- The **set** of fields unions across cascade levels. A field redefined at a deeper level **replaces** its inherited definition wholesale.
- `extend: true` is the opt-out: it unions only `values` with the inherited set. Everything else on the child definition still wins.
- Resolution is a single tree walk at startup/reindex time, with in-memory longest-prefix lookup per document afterward — not a walk per file.

### Freezing

A malformed `.kb-schema.yaml` **freezes its subtree**: nothing under it is indexed or re-indexed, and existing index entries are left untouched. It never silently falls back to the parent's rules. `md-kb-rag validate` reports broken schema files loudly in a `SCHEMA ERRORS` section, and they count as a failure under strict mode (`validation.strict: true`).

`md-kb-rag index --full` **refuses to run** while any scope is frozen, naming the offending directories — a full run rebuilds the Qdrant collection from scratch and cannot reindex a frozen scope, so its vectors would be lost rather than merely stale. Fix the schema(s) first, or reindex incrementally in the meantime; incremental indexing is unaffected by frozen scopes elsewhere in the tree.

A `.kb-schema.yaml` over 256 KB is refused purely on file size — it's never read or parsed — and freezes its subtree through that same mechanism.

### Backward compatibility

`config.yaml`'s `frontmatter` block is a deprecated fallback for root-level rules, used only when no root `.kb-schema.yaml` exists anywhere — in that case, existing deployments keep working unchanged, though every index run logs a warning naming the fallback.

The moment a root `.kb-schema.yaml` is added, it **replaces** `config.yaml`'s `frontmatter` block outright rather than merging with it: `required`/`indexed_fields`/`defaults`/`allowed` there stop applying unless the same field is also declared in the root `.kb-schema.yaml` (every index run also logs a warning about this, so the switch is never silent). This is deliberate — a schema describes the knowledge base's own content rules, and a KB that carries its own root `.kb-schema.yaml` must validate the same way regardless of which host's `config.yaml` happens to be serving it. `get_schema` (omit `path` for the root) always shows exactly what's in effect and where each field came from.

### Schema-change detection

A `schema_hash` fingerprint is stored per file alongside its content hash. The incremental indexer skips a file only when **both** match, so editing a schema revalidates every document under it even though their bytes didn't change. Two consequences: the first index run after upgrading to this feature revalidates every file once (backfilling the fingerprint), and editing a root-level schema revalidates the whole KB.

That same first-run revalidation is also when every document's `domain` gets (re)computed from its folder and written to Qdrant and the state DB. If a document's old, hand-authored `domain:` disagreed with its folder, the effective value changes at that point — update any saved filters that assumed the old value, and remove the now-redundant `domain:` key from frontmatter (it's ignored either way).

### Qdrant payload indexes

Declared fields get typed Qdrant payload indexes — integer/number/boolean fields get Integer/Float/Bool indexes (enabling range filters) instead of a blanket Keyword index. The same applies to the built-in `mtime` index used by `search`'s recency filters. Index-creation failures are logged as errors but never abort startup or indexing; a filter on an unindexed field still returns correct results, just more slowly. If a field's declared type changed, delete the stale payload index in Qdrant and reindex to pick up the new type.

## Observability

Four HTTP endpoints, with different audiences and different auth:

| Endpoint | Auth | Purpose |
|---|---|---|
| `/health` | open | Liveness/readiness. Reports Qdrant and embedding-service reachability only. Returns 503 when either is down. |
| `/status` | bearer | Full runtime state as JSON. |
| `/metrics` | bearer | The same data in Prometheus text exposition format. |
| `POST /admin/reload` | bearer | Re-read and re-validate `config.yaml` and swap it in, without a restart. See [Config reload](#config-reload). |

`/status`, `/metrics`, and `/admin/reload` require the same bearer token as `/mcp` (`MCP_BEARER_TOKEN`), because unlike `/health` they enumerate tag vocabularies, area names and document counts (`/status`/`/metrics`) or can change how the write tools authenticate content and which webhook provider is trusted (`/admin/reload`) — none of that gets a weaker gate than `/mcp` itself. Scrape `/status`/`/metrics` with an `authorization` stanza:

```yaml
scrape_configs:
  - job_name: md-kb-rag
    metrics_path: /metrics
    authorization:
      credentials: ${MCP_BEARER_TOKEN}
    static_configs:
      - targets: ["kb-rag:8001"]
```

Both report:

- **Indexing state** — whether a run is in flight right now, its phase (`discovering` → `scanning` → `embedding` → `backfilling` → `removing_orphans`), how far through it is, what triggered it (`cli`, `startup`, `webhook`, `write_tool`), and how long it has been going.
- **Last run** — outcome, duration, error message on failure, and the full per-outcome tallies (`discovered`, `indexed`, `skipped`, `invalid`, `empty`, `read_errors`, `metadata_backfilled`, `frozen_by_broken_schema`, `broken_schemas`, `orphans_removed`).
- **Store counts** — `indexed_files` (state DB), `documents` (metadata index), and `qdrant_points`. `documents_missing_metadata` is the divergence between the first two; non-zero means the metadata index is behind and the next run will backfill it.
- **Metadata breakdown** — document counts per value for each indexed field, widest document coverage first, plus a synthetic `area` field grouping by top-level directory. Fields are ordered by how many documents carry them rather than how many values they take, so a scoped schema's twenty recipe fields can't crowd out `type` and `status`. Broad vocabularies like `tags` report their most common values with `truncated: true`. `domain` is omitted (it is derived from the top-level folder, so it duplicates `area`), as are date and timestamp fields.
- **Payload index health** — which Qdrant payload indexes are in place and which failed. Failures are non-fatal by design, so this is the only lasting signal that a filter may be slow or incomplete.

`kb_index_last_success_timestamp_seconds` is the metric worth alerting on: its age answers "is the index actually keeping up", which neither `/health` nor a bare error count can. It is absent until a run succeeds, so an alert on timestamp age will not fire spuriously against a freshly started process.

If a backing store is unreachable, `/status` still answers — the failure is reported in `store.errors` (and counted by `kb_status_errors`) rather than failing the request, so "is it indexing?" stays answerable while Qdrant is down. Those error strings are scrubbed of credentials first: the Qdrant client renders its full connection URL on a transport failure, and since there is no separate `qdrant.api_key` setting, an authenticated Qdrant can only be reached by embedding the credential in `QDRANT_URL`.

Responses are cached for 5 seconds and collected single-flight. One request costs roughly two dozen SQLite queries plus a Qdrant round trip, and nothing about the answer changes meaningfully within that window — a scrape burst collapses into one refresh instead of a query storm.

`documents_missing_metadata` can legitimately read negative, and is reported that way rather than clamped. It means the metadata index holds more documents than the state DB tracks, which orphan removal cannot produce on its own — the likely cause is a CLI `index` run interleaving with the server's, since the reindex lock only serializes within one process.

### Config reload

`POST /admin/reload` re-reads `config.yaml` from disk, re-validates it exactly the way startup does (same checks, same errors), and swaps it into the running server — no restart, no dropped connections. Every setting has exactly one legal source (see [Configuration](#configuration) above): ENV vars are for startup/secrets and are never re-read by a reload, so a reload always means the same thing — re-run the YAML side of config resolution and swap the result in.

```bash
curl -X POST -H "Authorization: Bearer $MCP_BEARER_TOKEN" http://localhost:8001/admin/reload
```

The response reports exactly what happened, bucketed by whether the change actually took effect:

- **`applied`** — read fresh by the code that uses it (an MCP tool call, a webhook request, the reindex worker's next drain, a periodic timer's next tick), so the very next read observes the new value. `search.*`, `write.*`, `webhook.provider`, `indexing.include`/`exclude`, `frontmatter.*`, `validation.*`, `mcp.instructions`, and more fall here.
- **`restart_required`** — baked into a value or service built once at server startup (the embedding client's `reqwest::Client` timeout, the MCP path-filter `GlobSet`, the rate limiter, anything security-critical like the bearer token or `allow_unauthenticated`). The swap updates what everything else sees, but this particular consumer keeps behaving exactly as it did before the reload.
- **`reindex_required`** — `chunking.*`. The indexer reads the new value on its next run, but only for documents that run touches; existing Qdrant chunks keep the old boundaries. Run `md-kb-rag index --full` for a consistent corpus.

A malformed or invalid `config.yaml` — bad YAML, a value that fails validation, a missing required env var — is rejected with a 400 and that error message, and the running config is left **completely untouched**, the same guarantee a failed restart on that file would give you. A successful reload also queues an immediate full reconcile, so indexing-observing changes reach the corpus on the reindex worker's very next wake rather than waiting for the periodic sweep.

### Logging

`RUST_LOG` defaults to `info,rmcp=warn`. The `rmcp` demotion matters on an actively used server: the MCP transport logs three INFO lines per request, which otherwise buries the indexing pipeline's output entirely. Its warnings and errors, including tool-call failures, still come through.

Long phases report progress on a 10-second cadence rather than staying silent — a full re-embed is a single API sequence that can run for many minutes, and a run that simply stops logging is indistinguishable from one that is still working. Every run ends with an explicit terminal line on both the success and failure paths.

## Webhook

POST to `/hooks/reindex` triggers:

1. HMAC signature verification (Gitea/GitHub/GitLab)
2. Branch matching against `source.branch`
3. `git fetch` + `git merge --ff-only` (if `source.git_url` is configured)
4. Incremental reindex of changed files

The webhook endpoint is only available if `WEBHOOK_SECRET` is set to a non-empty value.

**Setup options:**

- **Native forge webhook** (recommended) — configure directly in your Git forge's webhook settings (or via `tea`/`gh` CLI). No CI runner needed.
- **CI workflow** — trigger from a pipeline step. See [`deploy/ci-examples/`](deploy/ci-examples/) for sample Gitea and GitHub workflows.

**Git pull on webhook:** Set `source.git_url` in your config and `GIT_PULL_TOKEN` in `.env` (for private HTTPS repos) to have the container pull changes automatically when the webhook fires. See [`deploy/USAGE.md`](deploy/USAGE.md#7-set-up-incremental-reindexing-optional) for detailed setup instructions.

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
