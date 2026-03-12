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
domain: infrastructure
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

A complete sample file is available at [`docs/sample-document.md`](../docs/sample-document.md).

**Frontmatter fields used by the system:**

| Field | Purpose |
|---|---|
| `title` | Document title (stored as Qdrant payload) |
| `description` | Summary text; optionally prepended to every chunk for better embedding context |
| `type` | Document type (e.g. `guide`, `reference`, `runbook`); filterable in MCP search |
| `domain` | Knowledge domain (e.g. `infrastructure`, `backend`); filterable in MCP search |
| `tags` | List of tags; filterable in MCP search (match-any) |

You can add any other fields you like. Only fields listed in `frontmatter.indexed_fields` are stored as Qdrant payload for filtering. Everything else is ignored during search but preserved in the state DB.

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

validation:
  enabled: true     # set false to skip all frontmatter checks
  strict: false     # true = abort indexing on first invalid file
  lint_command: null # e.g. "markdownlint" to run an external linter
```

**What happens when validation fails:**

- `strict: false` (default) — invalid files are skipped with a warning; indexing continues.
- `strict: true` — the first invalid file aborts the entire indexing run.

Run `md-kb-rag validate` to check all files without indexing — useful for CI or pre-commit hooks.

## Configuring Your Project

### 1. Prepare your knowledge base

Organize your markdown files in a directory. Subdirectories are fine — the indexer walks recursively. Add YAML frontmatter to each file with at least the fields you mark as required.

### 2. Create your config (optional)

Skip this step if the default chunking and frontmatter settings work for your knowledge base. Otherwise, start from the [config.example.yaml](config.example.yaml) and customize:

```yaml
# config.yaml — minimal production config
source:
  data_path: "/data"              # where your KB is mounted

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

All other sections (`embedding`, `qdrant`, `mcp`, `webhook`) use defaults that work with the Docker Compose stack. Override only if you need different values.

### 3. Set up environment variables

Create a `.env` file (see [`.env.example`](.env.example)):

```env
# Required
MCP_BEARER_TOKEN=your-secret-token-here
MODEL_PATH=/path/to/your/models
MODEL_FILE=nomic-embed-text-v2-moe-Q8_0.gguf

# Optional
KB_PATH=/path/to/your/knowledge-base
WEBHOOK_SECRET=your-webhook-secret
RUST_LOG=info
```

### 4. Start the stack

```bash
docker compose up -d
```

This starts Qdrant, the embedding server, and the md-kb-rag service. The kb-rag service waits for both dependencies to be healthy before starting.

### 5. Run the initial index

```bash
docker compose exec kb-rag md-kb-rag index --full
```

Full index drops any existing Qdrant collection and re-processes every file. Use this on first run or after changing `vector_size`.

### 6. Connect an MCP client

```bash
claude mcp add --transport http kb-search \
  http://localhost:8001/mcp \
  --header "Authorization: Bearer $MCP_BEARER_TOKEN"
```

### 7. Set up incremental reindexing (optional)

Configure a webhook in your Git forge (Gitea, GitHub, GitLab) pointing at `http://your-host:8001/hooks/reindex`. Set the `WEBHOOK_SECRET` env var to the same secret configured in the webhook. On push to the tracked branch, the service pulls changes and reindexes only modified files.

### File Include/Exclude Patterns

The `indexing` section controls which files are processed:

- `include` — glob patterns for files to index (default: `["**/*.md"]`)
- `exclude` — glob patterns for directories/files to skip (matched against paths relative to `data_path`)
- `exclude_files` — exact filenames to skip regardless of path (e.g. `README.md`)

Setting any list **replaces** the default — it does not merge. If you add a custom exclude pattern, include the defaults too or they won't apply.
