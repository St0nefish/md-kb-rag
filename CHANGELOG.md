# Changelog

This project has not yet cut a tagged release. Every merge to `master` builds and
pushes a new container image tagged by commit sha (`.github/workflows/release.yml`),
with `:latest` following `master` directly — there is no version number to pin to
yet, and no historical version list to reconstruct here. Until a tagging scheme
exists, this file tracks notable, operator-relevant changes on `master` under a
running `[Unreleased]` heading, in roughly chronological order (most recent first).
`fix #N` references are GitHub issues; see the repo's closed-issues list for the
complete history — recent activity included an automated multi-agent documentation
and correctness audit that closed roughly forty issues across security hardening,
indexing correctness, and doc drift, of which the entries below are a representative
sample rather than an exhaustive list.

## [Unreleased]

### Documentation

- Added a Backup and Recovery section to `deploy/TROUBLESHOOTING.md` covering what's
  actually authoritative (the git-hosted corpus), how to rebuild `state.db` and the
  Qdrant collection from it, the frozen-schema interaction that can block that
  rebuild, and the current (passive-only) detection for a Qdrant data wipe (fix #184).
- Added a `deploy/TROUBLESHOOTING.md` entry for the schema-freezing failure mode —
  symptoms, `md-kb-rag validate`'s `SCHEMA ERRORS`/`FROZEN` output, and the fix
  (fix #188).
- Added a worked, three-level `.kb-schema.yaml` cascade example to `deploy/USAGE.md`,
  showing the resolved schema at each level and a matching `get_schema` excerpt
  (fix #190).
- Documented the `ui`/`ui.semantic_edges` config block in `deploy/config.example.yaml`
  (previously undocumented despite being a real, deployable feature), and closed a
  smaller drift gap around `search.min_score` found during the same pass (fix #197).
- Added this file, `CONTRIBUTING.md`, and `.github/ISSUE_TEMPLATE`/
  `.github/pull_request_template.md` (fix #189).

### Server hardening

- `POST /admin/reload` now warns explicitly about the class of settings it cannot
  apply live; the reindex worker is supervised and restarted rather than silently
  dying; `/mcp` request bodies are size-bounded; passive detection was added for a
  Qdrant collection wiped out from under a surviving `state.db` (`kb_qdrant_points_deficit`
  in `/status`/`/metrics`) (fix #154, fix #163, fix #205).
- Config provenance drift, a reranking `candidate_limit` bound, `min_score`
  documentation, and a schema-fingerprint collision were fixed together (fix #137,
  fix #144, fix #149, fix #152).

### Indexing and retrieval correctness

- Strict-mode indexing no longer drops an entire batch when one rejection occurs, and
  embedding-dimension mismatches are now caught rather than silently corrupting the
  collection (fix #156, fix #159).
- Git-layer correctness fixes: an orphaned commit sha, a stale content-hash race, and
  mishandling of non-ASCII paths in diff output (fix #140, fix #142, fix #143).
- MCP `search` filter, casualty-reporting, and path-length surface cleanup (fix #148,
  fix #151, fix #153).
- Wiki-style `[[link]]` rendering fixed in the web UI graph, and `/api/graph` node
  count capped (fix #170, fix #174).
- `validation.lint_command` is now bounded by a configurable timeout, so a hanging
  external linter can no longer stall the whole indexing pipeline (fix #146).

### Accessibility and UI

- Document navigation links in the web UI now carry real `href`s, restoring keyboard
  and assistive-technology navigation (fix #165).

### Earlier structural changes worth knowing about

These predate the `[Unreleased]` window above but are still the kind of thing an
operator upgrading an old deployment needs to know happened at some point:

- **Hybrid sparse+dense retrieval with RRF fusion** (#59) — added a `sparse` named
  vector alongside `dense`; an existing pre-hybrid collection (single unnamed vector)
  needs a one-time `md-kb-rag index --full` to migrate to the named-vector schema.
- **Relative-path indexing** (#58, part of a broader shared-retrieval-core refactor)
  — internal state moved from absolute to KB-relative paths.
- **Directory-cascading `.kb-schema.yaml` support** — every indexed file gained a
  `schema_hash` fingerprint column (added automatically via a guarded
  `ALTER TABLE ... ADD COLUMN`, no manual migration step); the first index run after
  upgrading to a version with schema support revalidates every file once to backfill
  it, and is also the run where every document's `domain` is (re)computed from its
  folder rather than trusted from frontmatter. See [Backward compatibility and
  upgrade note](deploy/USAGE.md#backward-compatibility-and-upgrade-note) in
  `deploy/USAGE.md` for the full detail.
