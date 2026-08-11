# md-kb-rag

Rust binary with subcommands: `serve`, `index`, `validate`, `status`.

## Hosting context

This project is hosted on **GitHub** (issues, PRs, CI). The knowledge bases it indexes live on separate Git hosts (typically Gitea). Do not conflate the two — webhook provider config, signature headers, and the `deploy/ci-examples/gitea-reindex.yml` workflow all refer to the *indexed knowledge base's* Git host, not this repo's host.

## Architecture

Single binary (`md-kb-rag`) that combines MCP server, webhook handler, and CLI indexer. Docker Compose runs 3 services: qdrant, embeddings, md-kb-rag.

In `serve` mode, indexing is asynchronous: MCP write tools and the webhook handler never call the indexer directly — they mark repo-relative paths (or a full reconcile) dirty on `reindex::REINDEX_QUEUE` and return immediately. A single background worker (`reindex::run_worker`) drains that queue and is the only thing that calls `ingest::index_paths`, which is itself the only function that mutates Qdrant or the state DB. `ingest::scan_for_dirty` is the read-only detector behind a full reconcile — it walks the corpus and stat/hash-compares against `indexed_files`, producing a worklist for `index_paths` rather than indexing anything itself. The `index` CLI subcommand has no worker: it runs `ingest::scan_and_index` synchronously in-process.

Every git invocation against the KB clone is serialized by `git::GIT_LOCK`. A working copy cannot take concurrent mutation — `add`/`commit`/`merge`/`rebase` all contend for `.git/index.lock` — and the write tools and the webhook handler reach the same clone routinely, because each write pushes and the push webhooks straight back. Functions in `git.rs` therefore take a `&GitLock` as their first argument, so the type checker, not review, answers "is the lock held". Acquire once per logical sequence and hold it across the whole thing: a failed write and its rollback must share one acquisition, or the rollback races the very writers it is protecting against. The guard is never re-acquired internally, which is what keeps the non-reentrant mutex from deadlocking a call chain against itself.

## Key conventions

- All async code uses tokio
- Config loaded from `config.yaml` (deserialized in `src/config.rs`)
- State tracked in SQLite via sqlx (default `/data/state.db`, i.e. `<source.data_path>/state.db`)
- Point IDs are UUID5 from `file_path::chunk_index`
- Qdrant accessed via gRPC (port 6334)
- Embeddings via OpenAI-compatible API (async-openai)
- MCP via rmcp with Streamable HTTP transport

## Module layout

| File | Purpose |
|---|---|
| `main.rs` | CLI entrypoint (clap subcommands) |
| `config.rs` | Config deserialization |
| `validate.rs` | Frontmatter validation against the resolved `.kb-schema.yaml` cascade for each file's path (not the global config directly — `frontmatter` in `config.yaml` is only the implicit root schema) |
| `ingest.rs` | Indexing: `index_paths` (the only function that mutates Qdrant/state — chunk/embed/upsert changed paths, purge missing ones, refresh stale metadata) and `scan_for_dirty` (read-only detector: stat/hash-compares the corpus against `indexed_files` and produces a dirty-path worklist). `scan_and_index` composes the two for callers with no worker (CLI, startup bootstrap) |
| `reindex.rs` | The dirty-path queue (`REINDEX_QUEUE`, `mark_paths`/`mark_full`) and its single background worker (`run_worker`), which drains the queue into `ingest::index_paths` with coalesce-don't-drop semantics and transient-vs-permanent retry/backoff |
| `chunk.rs` | Markdown chunking |
| `embed.rs` | Embedding API client |
| `qdrant.rs` | Qdrant operations |
| `state.rs` | SQLite state DB: file bookkeeping (`indexed_files`, including `mtime`/`size` for the reconcile scan's stat pre-filter) plus the document metadata index (`documents`, `document_fields`) backing `list_documents` |
| `document_fields.rs` | Projects frontmatter JSON into filterable `document_fields` rows (dot-path flattening, array/range support) |
| `schema.rs` | Directory-cascading `.kb-schema.yaml` support: parse, cascade merge, `SchemaCache` tree resolution, type/value checking, schema fingerprinting |
| `retrieval.rs` | Shared retrieval core (`search` + `get_document`) used by MCP and (future) CLI |
| `mcp.rs` | MCP tools (rmcp): `search`, `get_document`, `list_documents`, `get_schema`, `update_schema` (thin handlers delegating to `retrieval`/`state`/`schema`), and write tools `create_document`/`edit_document`/`delete_document` — thin adapters over `write::write_document`/`write::delete_document` that map `WriteSuccess`/`WriteError` back onto the exact `CallToolResult`/`McpError` text and `data` payloads; they never index inline |
| `write.rs` | Transport-agnostic write pipeline extracted from `mcp.rs`'s write tools: `write_document`/`delete_document` taking a `WriteDeps` bundle, returning `WriteSuccess`/`WriteError`. Owns schema-frozen check, frontmatter validation, dedup gate, commit-message validation, the pre-commit rollback (remove/unstage on create, restore-from-HEAD on edit), `git::commit_and_sync`, and marking paths dirty on `reindex::REINDEX_QUEUE`. Used by both `mcp.rs` (rmcp) and `web.rs` (HTTP) so the two transports share one behavior |
| `status.rs` | Process-global indexing run state (`INDEX_STATUS`) backing `/status` and `/metrics`: in-flight phase/progress, last-run outcome and counters, payload-index health |
| `webhook.rs` | Webhook handler: fetches + ff-only merges the push, diffs the range (`git::git_diff_name_status`), and marks the changed paths dirty on `reindex::REINDEX_QUEUE` — does not index inline |
| `git.rs` | Git operations for the KB clone (clone, fetch with timeout, `commit_and_sync`: add→commit→fetch→rebase→push, returning the changed paths the rebase pulled in alongside the commit SHA). Owns `GIT_LOCK` and the `GitLock` guard — see the git serialization rule below |
| `server.rs` | Axum server (MCP + webhook + `/health`, `/status`, `/metrics` routes, plus the unauthenticated web UI routes from `web.rs` — `/`, `/assets/*`, `/api/graph`, `/api/search`, `/api/doc/{*path}`, `/api/schema/{*path}` — all merged in before the rate-limit `GovernorLayer` wrap; owns the Prometheus encoder); spawns the reindex worker and the periodic reconcile-sweep timer (`indexing.reconcile_interval_secs`) |
| `web.rs` | The knowledge-base web UI, served straight from the binary (`assets/ui/`, embedded via `include_str!`, no filesystem reads at request time). Docs-first shell: hash-routed home/browse/doc views with a sidebar document tree and a semantic-search results panel; the Cytoscape.js graph is secondary, reachable as a per-doc neighborhood view or a full-KB view (fcose layout, hover/zoom-gated labels). `UiState` mirrors `StatusState`/`KbSearchServer::deps()`, reusing the same `Arc<OnceCell<StateDb>>` as `StatusState` rather than opening a second pool. Handlers: static shell/asset routes, `/api/graph` (nodes from `StateDb::all_document_summaries`, edges from `StateDb::all_links` filtered to the current node set, server-computed type palette), `/api/search` (thin wrapper over `retrieval::search`), `/api/doc/{*path}` GET/POST/DELETE (GET via `retrieval::get_document`; POST/DELETE thin adapters over `write::write_document`/`write::delete_document`), `/api/schema/{*path}`. Deliberately unauthenticated — same open-route posture as `/health` — because the deployment sits behind Authentik via Traefik |

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
cargo run -- reproject-fields # Rebuild document_fields from stored frontmatter (no re-embed)
```
