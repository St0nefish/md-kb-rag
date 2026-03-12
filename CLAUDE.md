# md-kb-rag

Rust binary with subcommands: `serve`, `index`, `validate`, `status`.

## Hosting context

This project is hosted on **GitHub** (issues, PRs, CI). The knowledge bases it indexes live on separate Git hosts (typically Gitea). Do not conflate the two — webhook provider config, signature headers, and the `deploy/ci-examples/gitea-reindex.yml` workflow all refer to the *indexed knowledge base's* Git host, not this repo's host.

## Architecture

Single binary (`md-kb-rag`) that combines MCP server, webhook handler, and CLI indexer. Docker Compose runs 3 services: qdrant, embeddings, md-kb-rag.

## Key conventions

- All async code uses tokio
- Config loaded from `config.yaml` (deserialized in `src/config.rs`)
- State tracked in SQLite via sqlx (`data/state.db`)
- Point IDs are UUID5 from `file_path::chunk_index`
- Qdrant accessed via gRPC (port 6334)
- Embeddings via OpenAI-compatible API (async-openai)
- MCP via rmcp with Streamable HTTP transport

## Module layout

| File | Purpose |
|---|---|
| `main.rs` | CLI entrypoint (clap subcommands) |
| `config.rs` | Config deserialization |
| `validate.rs` | Frontmatter validation |
| `ingest.rs` | Indexing pipeline |
| `chunk.rs` | Markdown chunking |
| `embed.rs` | Embedding API client |
| `qdrant.rs` | Qdrant operations |
| `state.rs` | SQLite state DB |
| `mcp.rs` | MCP search tool (rmcp) |
| `webhook.rs` | Webhook handler |
| `server.rs` | Axum server (MCP + webhook routes) |

## Workflow

- **Branch protection** on `master`: direct push disabled, status checks required (`test` job must pass)
- Work on feature branches, open PRs — auto-merge on CI pass (via `auto-merge.yaml` workflow)
- `fix #N` in merge commit auto-closes GitHub issues
- Branches auto-delete after merge
- Pre-commit hook enforces `cargo fmt` + `cargo clippy` (activate with `./scripts/setup-dev.sh` after cloning)

## Issue tracking

Bugs, features, and enhancements are tracked as GitHub issues (not in-repo TODO files).

## Build & run

```bash
cargo build
cargo run -- serve          # Start server (MCP + webhook)
cargo run -- index --full   # Full reindex
cargo run -- validate       # Validate frontmatter
cargo run -- status         # Collection stats
```
