# RAG Server Implementation — `md-kb-rag`

## Context

Build a portable, Docker-first RAG server for markdown knowledge bases with YAML frontmatter. Designed to be reusable — any markdown repo with frontmatter can use it by providing a YAML config and bind-mounting their docs. The project lives at `~/Workspace/md-kb-rag` and will be pushed to Gitea.

**Prerequisite for our deployment**: Frontmatter schema migration (see `rag-server-frontmatter-refactor.md`).

## Project: `md-kb-rag`

### Repo Location

`~/Workspace/md-kb-rag` — new git repo, pushed to Gitea

### What It Is

A Docker-compose stack that:

1. Indexes a markdown knowledge base with YAML frontmatter into Qdrant
2. Exposes semantic search with metadata filtering via MCP (Streamable HTTP)
3. Re-indexes automatically on git push via webhook
4. Validates markdown and frontmatter before indexing
5. Runs entirely self-hosted — no cloud dependencies

### User Experience

```bash
# Clone the project
git clone https://github.com/you/md-kb-rag.git
cd md-kb-rag

# Edit config
cp config.example.yaml config.yaml
# Set: git repo URL, frontmatter schema, embedding model, etc.

# Start
docker compose up -d

# Initial index
docker compose exec ingester md-kb-rag index --full

# Point your Gitea/GitHub webhook at:
#   https://your-host:9000/hooks/reindex

# Add MCP to Claude Code:
claude mcp add --transport http kb-search \
    https://your-host:8001/mcp \
    --header "Authorization: Bearer $TOKEN"
```

### Project Structure

```text
md-kb-rag/
├── docker-compose.yml          # Full stack (qdrant, embeddings, ingester, mcp-server)
├── config.example.yaml         # Example configuration
├── .env.example                # Example secrets
├── ingester/
│   ├── Dockerfile
│   ├── kb_ingest.py            # Ingestion engine (~400-600 lines)
│   ├── kb_validate.py          # Frontmatter + markdown validation
│   ├── requirements.txt
│   └── entrypoint.sh           # webhook listener + reindex handler
├── mcp-server/
│   ├── Dockerfile
│   ├── kb_mcp_server.py        # FastMCP search server (~100-150 lines)
│   └── requirements.txt
├── actions/
│   ├── gitea-reindex.yml       # Sample Gitea Actions workflow
│   └── github-reindex.yml      # Sample GitHub Actions workflow
├── README.md
└── CLAUDE.md
```

### Configuration (`config.yaml`)

```yaml
# Knowledge base source
source:
    git_url: "https://git.example.com/user/knowledge-base.git"
    branch: "master"
    # Or bind-mount to /data and omit git_url

# What to index
indexing:
    include:
        - "**/*.md"
    exclude:
        - "archive/**"
        - ".git/**"
        - ".claude/**"
        - ".tools/**"
        - "node_modules/**"
    exclude_files:
        - "CLAUDE.md"
        - "README.md"
        - "index.md"

# Frontmatter schema — which fields become filterable Qdrant payload
frontmatter:
    # Required fields (docs missing these are skipped with warning)
    required:
        - title
        - description
        - type
        - tags
    # Fields to create Qdrant keyword indexes on (for filtered search)
    indexed_fields:
        - type        # keyword
        - domain      # keyword
        - tags        # keyword array
        - status      # keyword
    # Default values for optional fields
    defaults:
        status: "active"

# Chunking
chunking:
    strategy: "markdown"       # heading-aware splitting
    max_chunk_size: 1500       # characters
    chunk_overlap: 200
    prepend_description: true  # prepend frontmatter description to first chunk

# Embedding
embedding:
    base_url: "http://embeddings:8080/v1"
    model: "nomic-embed-text-v2-moe"
    vector_size: 768
    batch_size: 32

# Qdrant
qdrant:
    url: "http://qdrant:6333"
    collection: "knowledge-base"

# Validation (runs before indexing)
validation:
    enabled: true
    strict: false              # false = skip bad files, true = abort on any failure
    # frontmatter validation against the schema above
    # markdown linting is optional — provide a custom lint command
    lint_command: null          # e.g. "rumdl check {file}" if rumdl is available

# Webhook
webhook:
    port: 9000
    secret_env: "WEBHOOK_SECRET"  # env var name containing the HMAC secret
    # Supported: gitea, github, gitlab (determines header parsing)
    provider: "gitea"

# MCP server
mcp:
    port: 8001
    bearer_token_env: "MCP_BEARER_TOKEN"
```

### Docker Compose

```yaml
services:
    qdrant:
        image: qdrant/qdrant:latest
        volumes:
            - ./data/qdrant:/qdrant/storage
        healthcheck:
            test: ["CMD", "curl", "-f", "http://localhost:6333/health"]

    # User provides their own embedding server, or uses the bundled one
    embeddings:
        image: ghcr.io/ggml-org/llama.cpp:server-vulkan
        # GPU config in .env / override file
        volumes:
            - ${MODEL_PATH}:/models:ro
        command: >
            --model /models/${MODEL_FILE}
            --host 0.0.0.0 --port 8080
            --embeddings --flash-attn on -ngl 999

    ingester:
        build: ./ingester
        volumes:
            - ./config.yaml:/app/config.yaml:ro
            - ${KB_PATH:-./data/repo}:/data:rw    # bind-mount or git clone
            - ./data/state:/state:rw
        environment:
            - WEBHOOK_SECRET=${WEBHOOK_SECRET}
        depends_on:
            qdrant: { condition: service_healthy }
            embeddings: { condition: service_healthy }

    mcp-server:
        build: ./mcp-server
        volumes:
            - ./config.yaml:/app/config.yaml:ro
        environment:
            - MCP_BEARER_TOKEN=${MCP_BEARER_TOKEN}
        depends_on:
            qdrant: { condition: service_healthy }
            embeddings: { condition: service_healthy }
```

Users with existing embedding servers just remove the `embeddings` service and point `embedding.base_url` at their endpoint.

### Ingestion Engine (`kb_ingest.py`)

**CLI:**

```bash
md-kb-rag index --full       # Full re-index (clear state, re-embed everything)
md-kb-rag index              # Incremental (only changed files)
md-kb-rag validate           # Validate all files without indexing
md-kb-rag status             # Print collection stats + state DB info
```

**Pipeline per file:**

1. **Validate** — Parse frontmatter, check required fields, check against schema
2. **Parse** — `python-frontmatter` extracts metadata + body
3. **Chunk** — `MarkdownNodeParser` (heading-aware, code block protection) + `SentenceSplitter` for oversized chunks
4. **Embed** — Batch POST to `/v1/embeddings` endpoint
5. **Upsert** — Write to Qdrant with frontmatter fields as payload

**Incremental indexing:**

- SQLite state DB at `/state/state.db`
- Tracks `(file_path, content_hash, chunk_count, indexed_at)`
- On each run: scan → SHA256 compare → process changed/new/deleted
- Changed: delete-by-filter(file_path) → re-chunk → embed → upsert
- Point IDs: UUID5 from `file_path::chunk_index`

**Validation integration:**

- Every file is validated before chunking
- Failed files: logged with reason, skipped (or abort in strict mode)
- Validation covers: required frontmatter fields present, field types match schema, values in allowed enums
- Optional lint command for markdown formatting checks

### Webhook / Re-index Trigger (`entrypoint.sh`)

The ingester container runs `almir/webhook` as PID 1, listening on port 9000. On webhook:

1. Verify HMAC signature (Gitea/GitHub/GitLab format per config)
2. Verify branch matches configured branch
3. `git pull --ff-only` (if git_url configured) or assume bind-mount is already current
4. Determine changed .md files via `git diff`
5. Validate changed files
6. Index only changed files (incremental)

**For bind-mount users** (no git): The webhook just triggers a full incremental scan — the state DB hash comparison handles the "what changed" logic.

### MCP Server (`kb_mcp_server.py`)

FastMCP server exposing a `search` tool:

**Tool: `search`**

- `query` (string, required) — semantic search query
- `domain` (string, optional) — filter by domain field
- `type` (string, optional) — filter by type field
- `tags` (list[string], optional) — filter by tags (match any)
- `limit` (int, optional, default 10) — max results

Embeds the query via the same embedding endpoint used for indexing, searches Qdrant with vector similarity + payload filters, returns results with metadata.

### Sample CI Actions

**`actions/gitea-reindex.yml`:**

```yaml
name: Trigger RAG Reindex
on:
    push:
        branches: [master]
jobs:
    reindex:
        runs-on: ubuntu-latest
        steps:
            - name: Trigger webhook
              run: |
                  curl -s -X POST "${{ secrets.RAG_WEBHOOK_URL }}/hooks/reindex" \
                      -H "Content-Type: application/json" \
                      -H "X-Gitea-Signature: $(echo -n '{"ref":"refs/heads/master"}' | openssl dgst -sha256 -hmac '${{ secrets.WEBHOOK_SECRET }}' | cut -d' ' -f2)" \
                      -d '{"ref":"refs/heads/master"}'
```

**`actions/github-reindex.yml`:** — Same pattern with `X-Hub-Signature-256` header.

### Dependencies

**Ingester container:**

- python-frontmatter
- llama-index-core (MarkdownNodeParser, SentenceSplitter)
- qdrant-client
- httpx
- pyyaml
- almir/webhook (multi-stage Docker build)

**MCP server container:**

- mcp (FastMCP SDK)
- qdrant-client
- httpx
- pyyaml

## Deployment to Our Atlas

After building and testing `md-kb-rag` as a standalone project:

1. Push to Gitea
2. Clone on Atlas: `/npool/docker/data/kb-rag/`
3. Create `config.yaml` pointing to our knowledge base
4. Create `.env` with secrets + GPU device paths
5. `docker compose up -d`
6. `docker compose exec ingester md-kb-rag index --full`
7. Configure Gitea webhook on knowledge-base repo
8. `claude mcp add` on client machines
9. Update `dev/local-ai/knowledge-base-rag-server.md` to reference the new project and mark Active

## Verification

1. **Unit**: Ingester correctly parses frontmatter, chunks markdown, handles code blocks
2. **Integration**: Full index of knowledge base → Qdrant has points with correct payload fields
3. **Filtered search**: Query with `domain=sysadmin` returns only sysadmin docs
4. **Incremental**: Modify one file, re-index, verify only that file's vectors changed
5. **Validation**: File with missing frontmatter is skipped (or fails in strict mode)
6. **Webhook**: POST to webhook endpoint triggers pull + re-index
7. **MCP end-to-end**: Claude Code searches via kb-search tool, gets filtered results with metadata
