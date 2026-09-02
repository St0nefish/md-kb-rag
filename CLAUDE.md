# md-kb-rag

Rust binary with subcommands: `serve`, `index`, `validate`, `status`.

## Hosting context

This project is hosted on **GitHub** (issues, PRs, CI). The knowledge bases it indexes live on separate Git hosts (typically Gitea). Do not conflate the two — webhook provider config, signature headers, and the `deploy/ci-examples/gitea-reindex.yml` workflow all refer to the *indexed knowledge base's* Git host, not this repo's host.

## Architecture

Single binary (`md-kb-rag`) that combines MCP server, webhook handler, and CLI indexer. Docker Compose runs 3 services: qdrant, embeddings, md-kb-rag.

In `serve` mode, indexing is asynchronous: MCP write tools and the webhook handler never call the indexer directly — they mark repo-relative paths (or a full reconcile) dirty on a `reindex::ReindexQueue` and return immediately. That queue is an injected dependency, not a global: `server::run_server` builds exactly one `Arc<ReindexQueue>` and clones it into every producer (`KbSearchServer`, `UiState`, `WebhookState`, `AdminState`) and into the worker, so every producer and the worker are provably talking to the same queue rather than relying on convention — `write::WriteDeps::queue` is how the write pipeline receives its handle. A single background worker (`reindex::run_worker`) drains that queue and is the only thing that calls `ingest::index_paths`, which is itself the only function that mutates Qdrant or the state DB. `ingest::scan_for_dirty` is the read-only detector behind a full reconcile — it walks the corpus and stat/hash-compares against `indexed_files`, producing a worklist for `index_paths` rather than indexing anything itself. The `index` CLI subcommand has no worker: it runs `ingest::scan_and_index` synchronously in-process.

Every git invocation against the KB clone is serialized by `git::GIT_LOCK`. A working copy cannot take concurrent mutation — `add`/`commit`/`merge`/`rebase` all contend for `.git/index.lock` — and the write tools and the webhook handler reach the same clone routinely, because each write pushes and the push webhooks straight back. Functions in `git.rs` therefore take a `&GitLock` as their first argument, so the type checker, not review, answers "is the lock held". Acquire once per logical sequence and hold it across the whole thing: a failed write and its rollback must share one acquisition, or the rollback races the very writers it is protecting against. The guard is never re-acquired internally, which is what keeps the non-reentrant mutex from deadlocking a call chain against itself.

That rule covers every git call that *mutates* the clone. Read-only history calls (`git::recent_commits`, `document_history`, `document_commit_diff`) deliberately take no `&GitLock`, and the deviation is argued in their doc comment: a read cannot corrupt an immutable object store, while taking the lock would queue history reads behind an in-flight write for up to `GIT_TIMEOUT` — the inverse of the contention #236 was filed for. Verified empirically against a repo held mid-rebase-conflict: `git log` exits 0 with a transiently reverted view (the in-flight commit hidden while HEAD is detached), never an error or invalid output, and a stray `.git/index.lock` does not affect reads at all. A new *mutating* git function still takes the guard.

Authentication on the protected routes (`/mcp`, `/status`, `/metrics`, `/admin/reload`) is **dual-mode**, not either/or: `server::bearer_auth` admits a request if the static bearer token matches (constant-time, tried first — it is a string compare and it is what Claude Code sends) **or** a presented JWT validates through `oauth::OAuthValidator`. Adding OAuth did not deprecate, gate or reshape the static path; a deployment can run either, both, or neither (`mcp.allow_unauthenticated`). What OAuth changes about refusals is the `WWW-Authenticate` header, which goes on *every* 401/403 once OAuth is configured — including a failed static-token request, because the server cannot tell which credential the caller meant to present, and claude.ai will not start the authorization flow at all without the `resource_metadata` parameter (Claude Code tolerates its absence, which is why a missing header is easy to ship and hard to notice). The two `/.well-known/oauth-protected-resource*` routes are registered outside the auth layer for the same reason `/health` is: discovery has to work for a caller who has no credential yet, and gating it behind the authentication it bootstraps makes the flow unstartable.

**Per-tool write-scope enforcement is not implemented.** `bearer_auth` sits in front of the whole `/mcp` endpoint and cannot see which MCP tool a request invokes — that is in the JSON-RPC body, which only rmcp parses — so any valid token currently grants full access, writes included. The validated scopes are parked in request extensions as an `oauth::AuthorizedToken` so a later change can enforce them at tool dispatch.

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
| `reindex.rs` | `ReindexQueue` (`mark_paths`/`mark_full`), an injected dependency (not a global — see the Architecture note above) cloned as an `Arc` into every producer and into its single background worker (`run_worker`), which drains the queue into `ingest::index_paths` with coalesce-don't-drop semantics and transient-vs-permanent retry/backoff |
| `chunk.rs` | Markdown chunking |
| `embed.rs` | Embedding API client |
| `qdrant.rs` | Qdrant operations |
| `state.rs` | SQLite state DB: file bookkeeping (`indexed_files`, including `mtime`/`size` for the reconcile scan's stat pre-filter) plus the document metadata index (`documents`, `document_fields`) backing `list_documents` |
| `document_fields.rs` | Projects frontmatter JSON into filterable `document_fields` rows (dot-path flattening, array/range support) |
| `schema.rs` | Directory-cascading `.kb-schema.yaml` support: parse, cascade merge, `SchemaCache` tree resolution, type/value checking, schema fingerprinting |
| `retrieval.rs` | Shared retrieval core (`search` + `get_document`) used by MCP and (future) CLI, plus the line-range slicer behind `get_document`'s `start_line`/`end_line` (`LineRange`/`slice_lines`: 1-based inclusive, `end` clamped, byte-exact substrings) |
| `sparse.rs` | Pure-Rust BM25-style sparse-vector tokenizer (FNV-1a term hashing) feeding the `sparse` named vector for hybrid retrieval — no model, no network |
| `rerank.rs` | Cross-encoder reranking client; truncates each candidate to a byte budget derived from `chunking.max_chunk_size` before sending, with exponential-backoff retry |
| `mcp.rs` | The six MCP tools (rmcp): `search` (query-mode chunk/document retrieval and, with no `query`, the exhaustive enumeration formerly served by `list_documents`), `get_document`, `get_schema`, `update_schema` — thin handlers delegating to `retrieval`/`state`/`schema` — and the write tools `write_document`/`delete_document` (the former `create_document`/`edit_document`/`move_directory` unified into one upsert-and/or-relocate tool) — thin adapters over `write::write_document`/`write::delete_document` that map `WriteSuccess`/`WriteError` back onto the exact `CallToolResult`/`McpError` text and `data` payloads; they never index inline. Tool and server descriptions come from `descriptions.rs`'s compiled+config-derived overlay, not literal strings in this file |
| `descriptions.rs` | Assembles MCP tool/server descriptions from three layers: compiled-in `assets/mcp/*.md` (mechanics true of every deployment), config-derived sentences (e.g. the hybrid/phrase retrieval-mode sentence), and per-KB policy loaded at runtime from `<mcp.extensions_path>/` in the served knowledge base (append-only, editable via `write_document`, no restart) |
| `write.rs` | Transport-agnostic write pipeline extracted from `mcp.rs`'s write tools: `write_document`/`delete_document` taking a `WriteDeps` bundle, returning `WriteSuccess`/`WriteError`. Owns schema-frozen check, frontmatter validation, dedup gate, commit-message validation, the pre-commit rollback (remove/unstage on create, restore-from-HEAD on edit), `git::commit_and_sync`, and marking paths dirty on `WriteDeps::queue` (a `&ReindexQueue`, unlike `WriteDeps::state` no `None` mode — every write must mark its path). Used by both `mcp.rs` (rmcp) and `web.rs` (HTTP) so the two transports share one behavior |
| `oauth.rs` | OAuth 2.1 **resource server** (RFC 9728 + MCP authorization spec): `OAuthValidator` verifies RS256 access tokens against a lazily-fetched, in-memory JWKS (unknown `kid` refetches at most once per 60s, since `kid` is attacker-controlled; every fetch failure fails closed), checks `iss`/`aud`/`exp` inside one `jsonwebtoken::decode` so the signature can never be reordered after the claim checks, then the required scope. Also owns the RFC 9728 metadata document and both `WWW-Authenticate` challenge strings. **`aud` is the OAuth client_id, not the resource URL** — Authentik does not implement RFC 8707 resource indicators; this is deliberate, argued in the code, and must not be "fixed". This is strictly additive to the static bearer token, never a replacement — see `server.rs` |
| `status.rs` | Process-global indexing run state (`INDEX_STATUS`) backing `/status` and `/metrics`: in-flight phase/progress, last-run outcome and counters, payload-index health |
| `webhook.rs` | Webhook handler: fetches + ff-only merges the push, diffs the range (`git::git_diff_name_status`), and marks the changed paths dirty on `WebhookState::reindex_queue` — does not index inline |
| `reload.rs` | `POST /admin/reload`: re-reads and re-validates `config.yaml`, swaps it into the live `SharedConfig`, and classifies each changed setting `applied` (read fresh on next use) / `restart_required` (baked into a startup-built value) / `reindex_required` (`chunking.*`) |
| `git.rs` | Git operations for the KB clone (clone, fetch with timeout, `commit_and_sync`: add→commit→fetch→rebase→push, returning the changed paths the rebase pulled in alongside the commit SHA). Owns `GIT_LOCK` and the `GitLock` guard — see the git serialization rule below |
| `server.rs` | Axum server (MCP + webhook + `/health`, `/status`, `/metrics` routes, plus the unauthenticated web UI routes from `web.rs` — `/`, `/assets/*`, `/api/graph`, `/api/search`, `/api/doc/{*path}`, `/api/schema/{*path}` — and the unauthenticated OAuth discovery routes `/.well-known/oauth-protected-resource[/mcp]`, all merged in before the rate-limit `GovernorLayer` wrap; owns the Prometheus encoder); spawns the reindex worker and the periodic reconcile-sweep timer (`indexing.reconcile_interval_secs`). `bearer_auth` is dual-mode — see below |
| `web.rs` | The knowledge-base web UI, served straight from the binary (`assets/ui/`, embedded via `include_str!`, no filesystem reads at request time). Docs-first shell: hash-routed home/browse/doc views with a sidebar document tree and a semantic-search results panel; the Cytoscape.js graph is secondary, reachable as a per-doc neighborhood view or a full-KB view (fcose layout, hover/zoom-gated labels). `UiState` mirrors `StatusState`/`KbSearchServer::deps()`, reusing the same `Arc<OnceCell<StateDb>>` as `StatusState` rather than opening a second pool. Handlers: static shell/asset routes, `/api/graph` (nodes from `StateDb::all_document_summaries`, edges from `StateDb::all_links` filtered to the current node set, server-computed type palette), `/api/search` (thin wrapper over `retrieval::search`), `/api/doc/{*path}` GET/POST/DELETE (GET via `retrieval::get_document`, with optional `?start_line=&end_line=` slicing through `retrieval::slice_lines` — same contract as the MCP tool, `content_hash` always over the whole file; POST/DELETE thin adapters over `write::write_document`/`write::delete_document`), `/api/schema/{*path}`. Deliberately unauthenticated — same open-route posture as `/health` — because the deployment sits behind Authentik via Traefik |

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
