# Usage Guide

This guide explains how to set up a markdown knowledge base for md-kb-rag, how documents are processed, and how to configure the system for your project.

## Knowledge Base Structure

A knowledge base is a directory of markdown files. Each file should have YAML frontmatter at the top with metadata about the document. The indexer walks the directory recursively, validates frontmatter, chunks the content, generates embeddings, and stores everything in Qdrant for semantic search.

### Sample Document

Every document in your knowledge base should look like this:

```markdown
---
title: Deploying with Docker Compose
description: Step-by-step guide to deploying services with Docker Compose.
type: guide
tags:
  - docker
  - deployment
status: active
---

## Deploying with Docker Compose

Your markdown content goes here. Use headings, code blocks, lists —
any standard markdown.

## Section Two

The chunker splits at heading boundaries, so each major section
becomes part of a chunk.
```

A complete sample file is available at [`docs/sample-document.md`](../docs/sample-document.md). Note there's no `domain:` key — `domain` isn't author-written frontmatter; see the note below.

**Frontmatter fields used by the system:**

| Field | Purpose |
|---|---|
| `title` | Document title (stored as Qdrant payload) |
| `description` | Summary text; optionally prepended to every chunk for better embedding context |
| `type` | Document type (e.g. `guide`, `reference`, `runbook`); filterable in MCP search |
| `domain` | **Not a frontmatter field.** Derived automatically from the document's top-level folder — see below. Still filterable in MCP search |
| `tags` | List of tags; filterable in MCP search (match-any) |

**`domain` is derived, not authored.** `domain` is computed from the document's top-level folder name (e.g. a file at `infrastructure/docker-compose.md` gets `domain: infrastructure`) and written into both the Qdrant payload and the SQLite metadata index — it is *not* read from a `domain:` key in frontmatter. If you write one anyway, it's overwritten on the next index run and the server logs a warning when the two disagree. Documents sitting directly at the knowledge-base root (no top-level folder) have no domain at all. This only changes where the value comes from: `search(domain=...)`, the CLI's `--domain` flag, and `list_documents(filters={"domain": ...})` all still work exactly as before.

You can add any other fields you like. Only fields listed in `frontmatter.indexed_fields` are stored as Qdrant payload for filtering by `search`. All frontmatter fields, however, are stored (as JSON, and projected into a filterable dot-path index) in the state DB, so any field — indexed or not — can be filtered, ranged, and sorted on via the `list_documents` MCP tool.

## How Chunking Works

Documents are split into chunks before embedding. The chunker is **section-aware** — it uses markdown headings as natural boundaries rather than blindly splitting at character counts.

### The Algorithm

1. **Split at headings** — The document body is divided into sections at each line starting with `#`. Each section includes its heading plus all content until the next heading.

2. **Accumulate sections** — Sections are greedily combined into chunks. The chunker adds sections to the current chunk as long as the total stays under `target_chunk_size` (default: 1000 characters).

3. **Flush on overflow** — When adding the next section would exceed `target_chunk_size`, the current chunk is finalized and a new one starts.

4. **Force-split oversized sections** — If a single section exceeds `max_chunk_size` (default: 1500 characters), it is split further by a secondary markdown-aware text splitter. Small fragments (under 200 characters, e.g. a lone heading) are merged into adjacent chunks to avoid orphaned headings.

5. **Prepend description** — If `chunking.prepend_description` is enabled (default: `true`) and the document has a `description` frontmatter field, that description is prepended to every chunk. This gives the embedding model context about what the chunk relates to.

### Example

Given a document with three sections of ~400 characters each and `target_chunk_size: 1000`:

- Sections 1 + 2 (800 chars) fit together → **Chunk 0**
- Section 3 (400 chars) alone → **Chunk 1**

### Tuning Chunk Size

```yaml
chunking:
  target_chunk_size: 1000   # ideal chunk size (characters)
  max_chunk_size: 1500      # hard upper limit
  prepend_description: true # prepend description to every chunk
```

- **Smaller chunks** (500–800) → more precise search results, more vectors, higher storage/compute cost.
- **Larger chunks** (1500–2000) → broader context per result, fewer vectors, may dilute relevance.
- `target_chunk_size` should be ≤ `max_chunk_size`. The target controls when new chunks start; the max controls when oversized sections are force-split.

## Frontmatter Validation

Configure which frontmatter fields are required, which are indexed for filtering, and what defaults to apply.

### Sample Config

```yaml
frontmatter:
  # Files missing these fields are skipped during indexing (with a warning).
  required:
    - title
    - description
    - type

  # These fields become filterable Qdrant payload fields.
  # The MCP search tool can filter on domain, type, and tags.
  indexed_fields:
    - type
    - domain
    - tags
    - status

  # Auto-injected if missing from a file's frontmatter.
  defaults:
    status: "active"

  # Closed-set (enum) enforcement. Maps a field to its exhaustive list of
  # allowed values; a present field whose value isn't in the list fails
  # validation. Absent fields are governed by `required`, not here. Leave a
  # field out of this map to keep it open-ended (e.g. tags). `domain` isn't
  # author-set at all — it's derived from the folder — so it never belongs here.
  allowed:
    type: [guide, reference, research, config, troubleshooting, architecture, project, decision-record, migration]
    status: [active, draft, archived]

validation:
  enabled: true     # set false to skip all frontmatter checks
  strict: false     # true = abort indexing on first invalid file
  lint_command: null # e.g. "markdownlint" to run an external linter
```

**What happens when validation fails:**

- `strict: false` (default) — invalid files are skipped with a warning; indexing continues.
- `strict: true` — the first invalid file aborts the entire indexing run.

With `validation.enabled: false`, frontmatter is still parsed — just not checked against `required`/`allowed`/lint rules — so Qdrant and the state DB's metadata index still reflect each file's actual frontmatter, rather than treating unvalidated files as fieldless.

`frontmatter.allowed` is enforced by both `md-kb-rag validate` and the MCP write tools. When a write tool rejects a document, it returns a structured error (`field_errors`) naming the offending field, the rule it broke (`required` / `allowed_value` / `lint` / `type_mismatch` / `closed_object`), and — for closed-set fields — the value it `got` versus the values it `expected`, so an agent can fix and retry without guessing.

Run `md-kb-rag validate` to check all files without indexing — useful for CI or pre-commit hooks.

## Directory Schemas (`.kb-schema.yaml`)

The `frontmatter` block above is a single, global rule set. For a knowledge base where different folders need different fields — recipes need `planning.prep_minutes`, runbooks need `severity` — drop a `.kb-schema.yaml` file into a directory. It governs that directory and everything beneath it, cascading like `CLAUDE.md`.

### Authoring

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

Nested authoring (as above) and flat dot-paths (`planning.prep_minutes:`) are equivalent — nesting is sugar flattened at parse time.

**Types:** `text`, `integer`, `number`, `boolean`, `enum`, `list`, `date` (`YYYY-MM-DD`), `timestamp` (RFC 3339), `object`. Types are strictly enforced with no coercion (`prep_minutes: "45"` fails against `type: integer`). Undeclared fields are never type-checked and remain legal.

A field definition can't declare both a scalar `type` and nested `fields:` — a field is either a value or a container, not both. `type: object` is the exception, since `object` inherently means "has nested fields." `update_schema` rejects this the same way a hand-edited `.kb-schema.yaml` does.

`.kb-schema.yaml` files themselves are not indexed as documents.

### Cascade and merge rules

- The **set** of fields unions across levels; a field redefined at a deeper level **replaces** its inherited definition wholesale.
- `extend: true` is the opt-out — it unions only `values` with the inherited set, everything else on the child definition still wins.
- Top-level folder names are the KB's areas (this is also what `domain` is derived from — see [Sample Document](#sample-document) above); the MCP server's dynamic instructions list them from a directory read, in addition to any `Available domain: ...` facet it advertises when `domain` is in `frontmatter.indexed_fields`.

### Freezing

A malformed `.kb-schema.yaml` **freezes its subtree**: nothing under it is indexed or re-indexed, and existing index entries are left untouched — it never silently falls back to the parent's rules. `md-kb-rag validate` reports broken schema files in a `SCHEMA ERRORS` section, and they count as a failure under `validation.strict: true`.

A `.kb-schema.yaml` larger than 256 KB is rejected outright — it's never read or parsed, just refused on its file size — and freezes its subtree the same way any other invalid schema does.

`md-kb-rag index --full` refuses to run at all while any scope is frozen, naming the offending directories: a full run drops and recreates the Qdrant collection, and a frozen scope's documents would be skipped during the rebuild — losing their vectors outright rather than merely leaving them stale. Fix the schema first, or keep making progress with an incremental `md-kb-rag index`, which is unaffected by scopes frozen elsewhere in the tree.

### Backward compatibility and upgrade note

With no `.kb-schema.yaml` files anywhere, the global `frontmatter` block continues to act as the implicit root schema — existing deployments keep working unchanged.

Under the hood, each indexed file now also tracks a `schema_hash` fingerprint (a new `indexed_files` column, added automatically via a guarded `ALTER TABLE ... ADD COLUMN` — no manual migration step). The incremental indexer skips a file only when both its content hash *and* schema fingerprint are unchanged, since editing a schema doesn't touch a document's bytes. Two practical consequences:

- The **first index run after upgrading** to a version with schema support revalidates every file once, to backfill the fingerprint. This is also the run where every document's `domain` gets (re)computed from its folder and written to Qdrant and the state DB — see [Sample Document](#sample-document) above. If a document's old, hand-authored `domain:` disagreed with its folder, its effective `domain` value changes at that point, and any saved `search`/`list_documents` filters built around the old value will need updating. Remove now-redundant `domain:` keys from your frontmatter — they're ignored either way.
- Editing a **root-level** schema revalidates the entire knowledge base on the next run.

If you only need to rebuild the `list_documents` metadata projection (e.g. after changing a field-projection rule) without a full reindex, `md-kb-rag reproject-fields` does that from the frontmatter JSON already stored in the state DB — no markdown re-read, no re-embedding. It's safe to run against a live server: each document's frontmatter is re-read inside the same transaction that rewrites it, so it retries past contention with a running index or write tool rather than reverting a concurrent update. A document whose stored frontmatter is unparseable is skipped with a warning instead of aborting the run, and the command reports how many documents were reprojected. Note also that a full reindex (`index --full`) clears the metadata index along with the vector collection, so a file deleted from disk since the last full run can't leave a phantom `list_documents` entry behind.

Declared fields also get typed Qdrant payload indexes (Integer/Float/Bool for numeric and boolean fields, instead of a blanket Keyword index), which is what makes `list_documents` range filters (`gte`/`lte`/`gt`/`lt`) usable on them. The same applies to the built-in `mtime` index used by `search`'s `modified_after`/`modified_before` filters. If a payload index fails to create — most often because a field's declared type changed and Qdrant is still holding an index of the old kind — it's logged as an error but never aborts startup or indexing; filters on that field keep returning correct results, just more slowly until you delete the stale payload index in Qdrant and reindex.

## Agent Write Tools

Beyond read-only search, the MCP server lets a connected agent **author the knowledge base directly**: `create_document`, `edit_document`, and `delete_document`. This turns the KB into a living document store that an assistant can curate as it learns, rather than a static index you maintain by hand.

### What a write call does

All three tools run the same pipeline server-side:

1. **Resolve and guard the path** — relative to the KB root, or a unique basename. A leading `/` is also accepted, and means the KB root, not a filesystem path: a caller has no way to know where the KB actually lives inside the container, so `/food/chili.md` and `food/chili.md` resolve to the same file. `..` components and symlinked ancestors that escape the data root are still rejected — `/../x` is refused exactly like `../x` — as are paths that don't match `indexing.include` (a file the indexer would never pick up).
2. **Validate frontmatter** — required fields, `allowed` enums, and any `validation.lint_command`. Failures come back as structured `field_errors` (see [Frontmatter Validation](#frontmatter-validation)) so the agent can self-correct.
3. **Write to disk** — in the container-owned KB clone.
4. **Commit with provenance** — the commit message gets `Tool: md-kb-rag` and `Operation: <tool>` trailers, authored under the `write.commit_author_*` identity. Tool-authored commits are trivially distinguishable from your own in `git log`.
5. **Push to the remote** — `add → commit → fetch → rebase → push`, so the KB's git host stays the source of truth.
6. **Reindex** — incrementally, holding the same internal lock the webhook uses, so a write and a webhook-triggered pull can never race.

Each tool returns a one-line summary with the commit SHA plus a unified diff of the change.

### The tools

- **`create_document`** — new file only (errors if it already exists). Before writing, it runs a **near-duplicate check**: it embeds the content and searches the collection; if an existing document scores at or above `write.dedup_threshold`, the write is refused and the close match is named. The score is always a **dense cosine similarity** — this check is pinned to dense-only retrieval with reranking detached, regardless of `search.hybrid` and `reranking.enabled`, because hybrid RRF scores (~0.01–0.03) and cross-encoder relevance scores are not on the same scale as the threshold. Pass `force_new: true` to create anyway. Disable the check globally with `write.dedup_enabled: false` (useful during bulk migrations). The check fails open — if the embedder or Qdrant is unreachable, the write proceeds.
- **`edit_document`** — existing file only. **Surgical mode** (`old_string` + `new_string`) replaces a single unique occurrence; **full-replace mode** (`content`) swaps the whole file and re-validates its frontmatter. The two modes are mutually exclusive.
- **`delete_document`** — removes the file, commits and pushes the deletion, then purges the document's vectors from Qdrant and its row from the state DB directly (no full reindex needed).

### Schema tools

Two more MCP tools let a connected agent inspect and evolve the [directory schema cascade](#directory-schemas-kb-schemayaml) itself, rather than working around it:

- **`get_schema`** — shows the fully merged rules governing a path (directory or document; omit `path` for the root), with per-field provenance naming which `.kb-schema.yaml` declared each field. Optional `fields` restricts the report to specific dot-paths; `values_only` limits it to fields with a closed value set.
- **`update_schema`** — edits a directory's `.kb-schema.yaml` through constrained operations (`add_values`, `remove_values`, `set_field`, `remove_field`) rather than free-form text. Before writing anything, the change is validated against every document already indexed under that scope, using the frontmatter stored in the metadata index — no markdown re-read. If any document would fail the new rules, the change is refused and they're listed; pass `force` to apply anyway, or `dry_run` to see the effect without writing. The rendered YAML is re-parsed before writing, so an unparseable schema can never be committed. Like the document write tools, the file is written temp-then-rename, committed and pushed, and triggers an incremental reindex.

Both tools accept a partial directory for `path`, matching on trailing segments — e.g. `recipes` resolves to `lifestyle/kitchen/recipes` if that's the only scope ending in `recipes`. A unique match resolves silently; several matches are refused with the candidates listed rather than guessed at. `update_schema` alone treats zero matches as success rather than an error: it falls back to the literal path, since declaring a schema for a directory that doesn't have one yet is the normal way to introduce one.

### Configuration

The `write` section of `config.yaml` (see [config.example.yaml](config.example.yaml)) controls this behaviour:

```yaml
write:
  dedup_enabled: true            # near-duplicate check on create_document
  dedup_threshold: 0.80          # dense cosine similarity at/above which a create is refused
  commit_author_name: "md-kb-rag"
  commit_author_email: "md-kb-rag@localhost"
```

For writes to push successfully, the container needs a writable, non-shallow clone of the KB and push credentials — i.e. `GIT_URL` set and `GIT_PULL_TOKEN` carrying a token with **write** access (read-only is enough for webhook pulls, but not for the write tools).

### Telling the agent what the KB is for

The server advertises **dynamic instructions** to MCP clients on connect (and refreshes them periodically). They combine three pieces:

1. Your `mcp.instructions` narrative — a short description of what this knowledge base covers. If omitted, a generic read+write description is used.
2. The distinct filter values (domains, types, tags) discovered in the live index, so the agent knows what's already in use.
3. A write-authoring section listing the required frontmatter fields and any fixed `allowed` values.

Independently of this, the `create_document` and `edit_document` tool descriptions state that the knowledge base is for durable, long-lived reference knowledge and must not be used as a scratchpad for session notes, intermediate analysis, or task state. That wording is compiled in and always reaches the client. If you want the same boundary stated in the server instructions — worth doing, since a client reads those before it reads any tool description — add it to your `mcp.instructions` narrative yourself; the narrative is yours to write, so nothing is prepended to it.

Keep `mcp.instructions` short — the dynamic sections supply the detail. Example:

```yaml
mcp:
  instructions: "Homelab infrastructure wiki covering networking, Docker, storage, and monitoring. Read with search/get_document/list_documents; write with create_document, edit_document, delete_document."
```

## Knowledge Base Storage

There are two ways to provide your knowledge base to the container. The **named volume** approach is recommended for most deployments.

**State database location:** The SQLite state database (`state.db`) is written to `<DATA_PATH>/state.db` — by default `/data/state.db`, i.e. inside the knowledge-base volume. With the named-volume setup this means `state.db` is automatically persisted alongside the repository content in `kb_data`. No separate mount is needed.

### Named volume with `GIT_URL` (recommended)

The container manages the knowledge base itself: it clones the repo on first start, and pulls updates via webhook. No host-side git operations needed.

```env
# .env
GIT_URL=https://your-forge.example.com/org/knowledge-base.git
GIT_BRANCH=master
```

```yaml
# docker-compose.yml (kb-rag service)
volumes:
  - kb_data:/data:rw

# top-level
volumes:
  kb_data:
```

On first start with an empty volume, the server automatically shallow-clones the repo and runs a full index. Subsequent updates come through the webhook (`git fetch` + `git merge --ff-only` + incremental reindex).

**Why this is preferred:**

- No risk of accidental edits on the host breaking `git merge --ff-only`
- Simpler setup — no pre-cloning step, no `KB_PATH` to configure
- The container owns the data lifecycle end-to-end
- Rebuilding from scratch is just `docker volume rm` + restart

**Volume ownership:** If you run the container with a non-root `user:` directive, the named volume is created as root and the container won't be able to write to it. Fix this once after creating the volume:

```bash
docker run --rm -v kb_data:/data --user root --entrypoint chown \
  ghcr.io/st0nefish/md-kb-rag:latest 1000:1000 /data
```

Replace `1000:1000` with the UID:GID from your compose `user:` setting.

### Bind-mount (alternative)

Mount a pre-cloned repo from the host. Useful when you need direct host access to the files or can't use `GIT_URL` (e.g. local-only repos).

```yaml
# docker-compose.yml (kb-rag service)
volumes:
  - ${KB_PATH:-./data/repo}:/data:rw
```

With this approach, you're responsible for keeping the directory up to date. If `GIT_URL` is also set, the webhook will still run `git fetch` + `git merge` inside the container, but having the directory accessible on the host risks accidental modifications that could cause merge conflicts.

Without `GIT_URL`, you'll need an external process to update the bind-mounted directory and trigger a reindex (either via webhook or by running `docker compose exec kb-rag md-kb-rag index`).

## Configuring Your Project

### 1. Prepare your knowledge base

Organize your markdown files in a git repository. Subdirectories are fine — the indexer walks recursively. Add YAML frontmatter to each file with at least the fields you mark as required.

### 2. Create your config (optional)

Skip this step if the default chunking and frontmatter settings work for your knowledge base. Otherwise, start from the [config.example.yaml](config.example.yaml) and customize:

```yaml
# config.yaml — minimal production config
indexing:
  include: ["**/*.md"]
  exclude:
    - ".git/**"
    - "node_modules/**"
  exclude_files:
    - "README.md"
    - "CLAUDE.md"

frontmatter:
  required: [title, description, type]
  indexed_fields: [type, domain, tags]

chunking:
  target_chunk_size: 1000
  max_chunk_size: 1500
```

Point `GIT_URL` (and optionally `GIT_BRANCH`) at your knowledge base repo via environment variables — see step 3. All other sections (`embedding`, `mcp`, `webhook`) use defaults that work with the Docker Compose stack. Override only if you need different values.

### 3. Set up environment variables

Create a `.env` file (see [`.env.example`](.env.example)):

```env
# Required
MCP_BEARER_TOKEN=your-secret-token-here
MODEL_PATH=/path/to/your/models
MODEL_FILE=nomic-embed-text-v2-moe-Q8_0.gguf

# Optional — needed for private repos over HTTPS
GIT_PULL_TOKEN=your-gitea-or-github-pat
WEBHOOK_SECRET=your-webhook-secret
RUST_LOG=info
```

If you set `GIT_URL`, also set `GIT_PULL_TOKEN` to a personal access token. Read access is enough for cloning and webhook pulls; grant **write** access if you plan to use the MCP write tools (they push commits back to the repo — see [Agent Write Tools](#agent-write-tools)). The token is injected transiently into the HTTPS clone/fetch URL and never written to disk. SSH URLs don't need a token.

### 4. Start the stack

```bash
docker compose up -d
```

This starts Qdrant, the embedding server, and the md-kb-rag service. The kb-rag service waits for both dependencies to be healthy before starting.

If `GIT_URL` is set and the data volume is empty, the server **automatically clones the repo and runs a full index** — no manual step needed. Check progress with `docker logs -f kb-rag`.

### 5. Run the initial index (bind-mount only)

If you're using the bind-mount approach without `GIT_URL`, run the initial index manually:

```bash
docker compose exec kb-rag md-kb-rag index --full
```

Full index drops any existing Qdrant collection and re-processes every file. Also use this after changing `EMBEDDING_VECTOR_SIZE`.

### 6. Connect an MCP client

```bash
claude mcp add --transport http kb-search \
  http://localhost:8001/mcp \
  --header "Authorization: Bearer $MCP_BEARER_TOKEN"
```

### 7. Set up incremental reindexing (optional)

When a webhook fires, the service verifies the HMAC signature, runs `git fetch` + `git merge --ff-only` (if `GIT_URL` is set), and triggers an incremental reindex of changed files.

#### Option A: Native forge webhook (recommended)

The simplest approach — configure the webhook directly in your Git forge's settings. No CI runner required.

**Gitea** — via UI or CLI:

```bash
# Using tea CLI
tea webhooks create \
  --repo org/knowledge-base \
  --login your-login \
  --type gitea \
  --secret "$WEBHOOK_SECRET" \
  --events push \
  --active \
  --branch-filter master \
  "https://your-host/hooks/reindex"
```

Or in the Gitea UI: Repository → Settings → Webhooks → Add Webhook (Gitea), set the target URL, secret, and push event.

**GitHub** — via CLI:

```bash
# Using gh CLI
gh api repos/org/knowledge-base/hooks --method POST \
  -f name=web \
  -f active=true \
  -f 'events[]=push' \
  -f 'config[url]=https://your-host/hooks/reindex' \
  -f 'config[content_type]=json' \
  -f "config[secret]=$WEBHOOK_SECRET"
```

Or in the GitHub UI: Repository → Settings → Webhooks → Add webhook.

**GitLab** — in the UI: Repository → Settings → Webhooks. GitLab uses a shared token (not HMAC), so set the same value for both `X-Gitlab-Token` and `WEBHOOK_SECRET`.

Make sure `webhook.provider` in your config matches your forge (`gitea`, `github`, or `gitlab`). The default is `gitea`.

#### Option B: CI workflow

If you prefer to trigger the webhook from a CI pipeline (e.g. to add logging or conditional logic), see the sample workflows in [`ci-examples/`](ci-examples/). These craft and send the HMAC-signed request as a CI step.

> **Note:** The CI examples use `master` (Gitea) and `main` (GitHub) as default branch names. Adjust the branch name in both the trigger and the payload `ref` field to match your repository.

### File Include/Exclude Patterns

The `indexing` section controls which files are processed:

- `include` — glob patterns for files to index (default: `["**/*.md"]`)
- `exclude` — glob patterns for directories/files to skip (matched against paths relative to `DATA_PATH`)
- `exclude_files` — exact filenames to skip regardless of path (e.g. `README.md`)

Setting any list **replaces** the default — it does not merge. If you add a custom exclude pattern, include the defaults too or they won't apply.
