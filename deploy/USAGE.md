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

**`domain` is derived, not authored.** `domain` is computed from the document's top-level folder name (e.g. a file at `infrastructure/docker-compose.md` gets `domain: infrastructure`) and written into both the Qdrant payload and the SQLite metadata index — it is *not* read from a `domain:` key in frontmatter. If you write one anyway, it's overwritten on the next index run and the server logs a warning when the two disagree. Documents sitting directly at the knowledge-base root (no top-level folder) have no domain at all. This only changes where the value comes from: `search(filters={"domain": ...})` — whether ranking or enumerating — and the CLI's `--domain` flag all still work exactly as before.

You can add any other fields you like. Only fields listed in `frontmatter.indexed_fields` are stored as Qdrant payload for filtering by `search`. All frontmatter fields, however, are stored (as JSON, and projected into a filterable dot-path index) in the state DB, so any field — indexed or not — can be filtered, ranged, and sorted on via `search`'s enumeration mode (omit `query`).

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

**This whole section describes the deprecated way to do it.** The `frontmatter` block below lives in `config.yaml`, which is deployment config on the container host — not part of the knowledge base's own git repo. Prefer a root `.kb-schema.yaml` (see [Directory Schemas](#directory-schemas-kb-schemayaml) below): it's the same mechanism every subdirectory already uses, it travels with the KB wherever it's cloned or served, and it's the one level `update_schema` can actually edit for you. `config.yaml`'s `frontmatter` block still works — it's consulted only when the knowledge base has no root `.kb-schema.yaml` at all — but every build logs a warning while it's in use, and once you add a root `.kb-schema.yaml`, this block stops applying (see [Backward compatibility and upgrade note](#backward-compatibility-and-upgrade-note)).

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

The `frontmatter` block above is the deprecated, single, global rule set. `.kb-schema.yaml` is the non-deprecated replacement, and it isn't limited to subdirectories — a `.kb-schema.yaml` at the knowledge-base root replaces `frontmatter` entirely (see [Root schema](#root-schema-kb-schemayaml-at-the-kb-root) below). For a knowledge base where different folders need different fields — recipes need `planning.prep_minutes`, runbooks need `severity` — drop a `.kb-schema.yaml` file into a directory. It governs that directory and everything beneath it, cascading like `CLAUDE.md`.

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
    values: [$values, dinner, quick]   # inherited values, then these — see below
```

Nested authoring (as above) and flat dot-paths (`planning.prep_minutes:`) are equivalent — nesting is sugar flattened at parse time.

**Types:** `text`, `integer`, `number`, `boolean`, `enum`, `list`, `date` (`YYYY-MM-DD`), `timestamp` (RFC 3339), `object`. Types are strictly enforced with no coercion (`prep_minutes: "45"` fails against `type: integer`). Undeclared fields are never type-checked and remain legal.

A field definition can't declare both a scalar `type` and nested `fields:` — a field is either a value or a container, not both. `type: object` is the exception, since `object` inherently means "has nested fields." `update_schema` rejects this the same way a hand-edited `.kb-schema.yaml` does.

**`values:` without a `type:` is enforced leniently.** A field declaring `values:` *and* `type: enum` is checked strictly — any value outside the list fails, whatever its YAML type. A field declaring `values:` with **no** `type:` exempts non-string values from the check, so `status: 3` passes a `values: [active, draft]` list that `status: "retired"` would fail. That is deliberate: it preserves the behaviour of the pre-cascade global `frontmatter.allowed` map so existing deployments don't start failing, and it applies to any field authored that way — including in a `.kb-schema.yaml`, not just the legacy `config.yaml` block. If you want a closed set actually enforced, declare `type: enum`.

`.kb-schema.yaml` files themselves are not indexed as documents.

### Cascade and merge rules

- The **set** of fields unions across levels.
- **Merging is per attribute, not per field.** A field redefined at a deeper level overrides only the attributes it explicitly writes (`type`, `required`, `indexed`, `default`, `open`, `values`) — every attribute it leaves unwritten still inherits from the nearest ancestor that declared this same field. A `recipes/` scope that writes only `values: [recipe]` for `type` does not reset that field's `required`/`indexed`/`default` to nothing; it only changes `values`. This is deliberate: the previous rule (a redefinition replacing the whole definition wholesale) meant a deeper scope narrowing one attribute silently discarded every other attribute the root had set for that field, with nothing reported — a real footgun for a field like `tags` that nearly every domain redeclares just to set its own `values`.
- **`values` is the one attribute with an in-band way to request a merge instead of a plain override**, via a `$values` placeholder inside the list — see below. Every other attribute either inherits wholesale (unwritten) or overrides wholesale (written); there is no partial merge for them.
- Top-level folder names are the KB's areas (this is also what `domain` is derived from — see [Sample Document](#sample-document) above); the MCP server's dynamic instructions list them from a directory read, in addition to any `Available domain: ...` facet it advertises when `domain` is indexed at the root — either via `frontmatter.indexed_fields` (deprecated fallback) or an `indexed: true` entry for `domain` in a root `.kb-schema.yaml`. `domain` isn't author-written frontmatter (see the note above), so if you migrate off `config.yaml`, remember to declare it explicitly — it does not carry over automatically.

#### The `$values` placeholder

Closed value sets (`type: enum`, or `type: list` with `values:`) have a merge mode of their own, because "replace" and "extend" are both legitimate things to want from a deeper scope and only the author knows which. The model is the shell `PATH` idiom:

```sh
PATH=$PATH:/usr/local/bin     # inherit, then add
PATH=/only/this                # replace outright
```

```yaml
values: [$values, one, two]    # inherit, then add
values: [one, two]             # replace outright
```

- **Omitting `$values` replaces the inherited set outright — this is the default**, the same way `PATH=/only/this` clobbers rather than extends. There is no implicit merging; a `values:` list you write is the complete set unless you say otherwise.
- **`$values` inside the list splices the inherited set in at that exact position.** Position is meaningful, same as `$PATH` in a shell assignment: `[$values, one, two]` puts inherited values first, `[one, two, $values]` puts them last, `[one, $values, two]` puts them in the middle.
- The result is **deduplicated**, keeping each value's first occurrence — a value listed both explicitly and inherited appears once.
- `$` is a reserved prefix inside a `values:` list. Any other `$`-prefixed token (`$value`, `$parent`, a typo of `$values`) is a hard parse error naming the offending token, never a silently-accepted literal value — a typo here degrading into "just another permitted tag" is exactly the kind of quiet failure this project avoids elsewhere.
- **`$values` with nothing to inherit** (no ancestor scope declares any `values` for this field, or there is no ancestor definition at all) resolves to contributing *no* values — a loud warning is logged naming the scope and field — rather than silently falling back to "no values declared at all." The field ends up with whatever literal values remain in the list (or a totally empty, closed set if the sentinel was the only token), not an unconstrained one: an empty closed set fails the moment a document sets the field, which is a visible, immediate signal that something upstream is missing; treating the whole `values:` declaration as absent would instead silently stop enforcing the field at all. Declare `values` on an ancestor, or drop the sentinel, to clear the warning.
- **`extend: true` is deprecated** — it was the mechanism before `$values` existed, and is kept only so schema files written before this change keep parsing and cascading identically. `extend: true` behaves exactly like a leading `$values` (`values: [$values, ...]`); using it logs a warning naming the schema file. New schemas should write `$values` directly. Declaring both `extend: true` and an explicit `$values` on the same field is a parse error — the two ways of saying "inherit" must not be able to disagree about where the inherited values land.

### Freezing

A malformed `.kb-schema.yaml` **freezes its subtree**: nothing under it is indexed or re-indexed, and existing index entries are left untouched — it never silently falls back to the parent's rules. `md-kb-rag validate` reports broken schema files in a `SCHEMA ERRORS` section, and they count as a failure under `validation.strict: true`.

A `.kb-schema.yaml` larger than 256 KB is rejected outright — it's never read or parsed, just refused on its file size — and freezes its subtree the same way any other invalid schema does.

`md-kb-rag index --full` refuses to run at all while any scope is frozen, naming the offending directories: a full run drops and recreates the Qdrant collection, and a frozen scope's documents would be skipped during the rebuild — losing their vectors outright rather than merely leaving them stale. Fix the schema first, or keep making progress with an incremental `md-kb-rag index`, which is unaffected by scopes frozen elsewhere in the tree.

### Root schema (`.kb-schema.yaml` at the KB root)

Every level of the cascade — including the root — is a `.kb-schema.yaml`. A root schema file is authored exactly like any other: drop one at the top of the knowledge base and it governs every document that no deeper scope claims, the same as a `.kb-schema.yaml` in any subdirectory.

**A root `.kb-schema.yaml`, when present, REPLACES `config.yaml`'s `frontmatter` block outright — it does not merge with it.** This is deliberate, not an oversight: a schema describes the knowledge base's own content rules, and once the KB carries its own root rules, they must not be silently blended with whatever `frontmatter` block the current deploying host's `config.yaml` happens to declare — that would mean the same KB validates differently depending on where it's hosted. So:

- **No root `.kb-schema.yaml` anywhere** — `config.yaml`'s `frontmatter` block is used as the root schema, exactly as before. This is the deprecated fallback described above; it still fully works, but every index run logs a warning naming it.
- **A root `.kb-schema.yaml` exists** — it is the entire root schema. Anything `config.yaml`'s `frontmatter` block still declares (`required`, `indexed_fields`, `defaults`, `allowed`) is ignored for root purposes unless the same field is *also* declared in the root `.kb-schema.yaml`. Every index run logs a warning naming this too, so the interaction is never silent.

Either way, `get_schema` (omit `path` for the root) tells you exactly what's in effect and where each field came from — `.kb-schema.yaml` for a root schema file, `config.yaml` for the deprecated fallback.

**Migrating an existing deployment:** translate `config.yaml`'s `frontmatter` block into `.kb-schema.yaml` field syntax (see [Authoring](#authoring) above) and commit it as `.kb-schema.yaml` at the knowledge-base root. Do this *before* or *at the same time as* removing anything from `config.yaml` — since the two don't merge, an incomplete root file written first (with the config block still present) can quietly narrow root's effective rules to only what the config block still contributes, until the root file catches up. `required` fields become `required: true`; `indexed_fields` become `indexed: true`; `defaults` become `default: <value>`; `allowed` becomes `values: [...]` **without** a `type:` — adding `type: enum` changes enforcement strictness (see [Authoring](#authoring) above) and is a separate decision, not a mechanical translation. If a subdirectory's own `.kb-schema.yaml` already redeclares a field the root also declares, double-check after migrating that the subdirectory's definition is still what you want — merging is per attribute (see [Cascade and merge rules](#cascade-and-merge-rules) above), so any attribute the subdirectory's redefinition never mentioned will start inheriting whatever the new root file says about it, even though it did not before the root file existed.

### Backward compatibility and upgrade note

Under the hood, each indexed file now also tracks a `schema_hash` fingerprint (a new `indexed_files` column, added automatically via a guarded `ALTER TABLE ... ADD COLUMN` — no manual migration step). The incremental indexer skips a file only when both its content hash *and* schema fingerprint are unchanged, since editing a schema doesn't touch a document's bytes. Two practical consequences:

- The **first index run after upgrading** to a version with schema support revalidates every file once, to backfill the fingerprint. This is also the run where every document's `domain` gets (re)computed from its folder and written to Qdrant and the state DB — see [Sample Document](#sample-document) above. If a document's old, hand-authored `domain:` disagreed with its folder, its effective `domain` value changes at that point, and any saved `search` filters built around the old value will need updating. Remove now-redundant `domain:` keys from your frontmatter — they're ignored either way.
- Editing a **root-level** schema revalidates the entire knowledge base on the next run.

If you only need to rebuild the document metadata projection backing `search`'s enumeration mode (e.g. after changing a field-projection rule) without a full reindex, `md-kb-rag reproject-fields` does that from the frontmatter JSON already stored in the state DB — no markdown re-read, no re-embedding. It's safe to run against a live server: each document's frontmatter is re-read inside the same transaction that rewrites it, so it retries past contention with a running index or write tool rather than reverting a concurrent update. A document whose stored frontmatter is unparseable is skipped with a warning instead of aborting the run, and the command reports how many documents were reprojected. Note also that a full reindex (`index --full`) clears the metadata index along with the vector collection, so a file deleted from disk since the last full run can't leave a phantom entry behind in that projection.

Declared fields also get typed Qdrant payload indexes (Integer/Float/Bool for numeric and boolean fields, instead of a blanket Keyword index). Query-mode `search` filters run against Qdrant, so a range filter (`gte`/`lte`/`gt`/`lt`) on a field there needs this — an unindexed field is rejected by name rather than filtered slowly. Enumeration mode (no `query`) filters against the SQLite `document_fields` projection instead, which every frontmatter field lands in regardless of `indexed: true`, so its range filters need no Qdrant index at all. The same Qdrant indexing applies to the built-in `mtime` index used by `search`'s `modified_after`/`modified_before` filters in query mode. If a payload index fails to create — most often because a field's declared type changed and Qdrant is still holding an index of the old kind — it's logged as an error but never aborts startup or indexing; a query-mode filter on that field is still accepted (the schema still declares it indexed) and returns correct results, just more slowly via a full scan, until you delete the stale payload index in Qdrant and reindex.

## Agent Write Tools

Beyond read-only search, the MCP server lets a connected agent **author the knowledge base directly**: `write_document` and `delete_document`. This turns the KB into a living document store that an assistant can curate as it learns, rather than a static index you maintain by hand.

### What a write call does

Both tools run the same pipeline server-side:

1. **Resolve and guard the path** — relative to the KB root, or a unique basename. A leading `/` is also accepted, and means the KB root, not a filesystem path: a caller has no way to know where the KB actually lives inside the container, so `/food/chili.md` and `food/chili.md` resolve to the same file. `..` components and symlinked ancestors that escape the data root are still rejected — `/../x` is refused exactly like `../x` — as are paths that don't match `indexing.include` (a file the indexer would never pick up).
2. **Validate frontmatter** — required fields, `allowed` enums, and any `validation.lint_command`, against the *destination* directory's schema when the call also relocates the document. Failures come back as structured `field_errors` (see [Frontmatter Validation](#frontmatter-validation)) so the agent can self-correct.
3. **Write to disk** — in the container-owned KB clone.
4. **Commit with provenance** — the commit message gets `Tool: md-kb-rag` and `Operation: <tool>` trailers, authored under the `write.commit_author_*` identity. Tool-authored commits are trivially distinguishable from your own in `git log`.
5. **Push to the remote** — `add → commit → fetch → rebase → push`, so the KB's git host stays the source of truth.
6. **Reindex** — incrementally, holding the same internal lock the webhook uses, so a write and a webhook-triggered pull can never race.

Each tool returns a one-line summary with the commit SHA plus a unified diff of the change.

### The tools

- **`write_document`** — upserts, edits, and/or moves a document, replacing the old `create_document`, `edit_document`, and `move_directory` tools:
  - **Full-replace** (`content`) — the whole file, including frontmatter. Creates `path` if it's new, replaces it if it already exists.
  - **Surgical** (`old_string` + `new_string`) — replaces a single unique occurrence instead of resending the whole file. Mutually exclusive with `content`.
  - **Move** (`new_path`) — relocates a document. Combines with either edit mode above (edit-then-move, one commit), or stands alone for a pure move — the server reads the current body itself and revalidates it against the destination schema. If `path` names a *directory* instead, `write_document` detects that and moves the whole subtree there in one commit, no `content`/`old_string`/`new_string` allowed — this replaces the old `move_directory` tool. Links pointing at whatever moved are rewritten either way.

  On create, it runs a **near-duplicate check**: it embeds the content and searches the collection; if an existing document scores at or above `write.dedup_threshold`, the write is refused and the close match is named. The score is always a **dense cosine similarity** — this check is pinned to dense-only retrieval with reranking detached, regardless of `search.hybrid` and `reranking.enabled`, because hybrid RRF scores (~0.01–0.03) and cross-encoder relevance scores are not on the same scale as the threshold. Pass `force_new: true` to create anyway. Disable the check globally with `write.dedup_enabled: false` (useful during bulk migrations). The check fails open — if the embedder or Qdrant is unreachable, the write proceeds.

  Pass `expected_hash` (a `content_hash` from a prior `get_document`) to reject an edit built on a stale read.
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
  dedup_enabled: true            # near-duplicate check on write_document's create path
  dedup_threshold: 0.80          # dense cosine similarity at/above which a create is refused
  commit_author_name: "md-kb-rag"
  commit_author_email: "md-kb-rag@localhost"
```

For writes to push successfully, the container needs a writable, non-shallow clone of the KB and push credentials — i.e. `GIT_URL` set and `GIT_PULL_TOKEN` carrying a token with **write** access (read-only is enough for webhook pulls, but not for the write tools).

### Telling the agent what the KB is for

Server and tool descriptions are assembled from three layers, appended in this order:

1. **Compiled mechanics** — what's true of *every* deployment this binary could ever serve (how paths resolve, that there's no regex, what `write_document`'s parameters mean). Baked into the binary from `assets/mcp/`; you can't change this short of a fork.
2. **Config-derived mechanics** — short sentences generated from your live config and corpus: whether search is hybrid and/or phrase-matching (`search.hybrid`/`search.phrase`), the distinct filter values (domains, types, tags) discovered in the live index, and a write-authoring section listing required frontmatter fields and any fixed `allowed` values.
3. **Your knowledge base's own policy** — what this KB is for, what belongs in it, tagging conventions, writing style. This is yours to write, and it lives in the *served knowledge base itself*, not in `config.yaml`.

**Nothing about what a knowledge base is for, or what belongs in it, is compiled in or config-derived — that's layer 3, and it's entirely up to you to write.** One binary may serve several knowledge bases with contradictory policies (a durable-reference KB and a scratch-space KB, say), so the compiled and config layers can only ever state what's true of *every* KB the binary might serve. If you want the connected agent to know this KB holds durable reference knowledge only, or that it should tag things a certain way, or anything else about *this* KB specifically — write it yourself, in `<mcp.extensions_path>/` (default `meta/mcp`, relative to the knowledge base root):

- `server.md` — appended to the server instructions (layers 1+2 above).
- `tools/<tool>.md` — appended to that one tool's description, e.g. `tools/write_document.md`.

Both are ordinary indexed KB documents (frontmatter included, stripped before the body reaches a description) — so editing them is a normal `write_document` call, no ssh, no restart: the change is picked up on the next `mcp.metadata_refresh_secs` tick (default 300s), or immediately after a `POST /admin/reload`. Extensions are **append-only** — they can add policy but never suppress or contradict a compiled or config-derived sentence.

```yaml
# meta/mcp/server.md, in the served knowledge base
---
title: MCP server policy
type: reference
status: active
---

This knowledge base holds durable, long-lived reference knowledge only. Do not use it
as a scratchpad for session notes, intermediate analysis, or task state.
```

`mcp.instructions` in `config.yaml` is a **deprecated** narrative override — still honored (logged as deprecated at startup), but a `server.md` extension file, if present, wins over it entirely. New deployments should write policy into `<extensions_path>/server.md` instead; it travels with the knowledge base rather than living on the deploying host.

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

Commit a `.kb-schema.yaml` at the repository root declaring those rules — see [Root schema](#root-schema-kb-schemayaml-at-the-kb-root) above. This is the preferred place for root-level frontmatter rules: it's part of the knowledge base's own repo, so it travels with it wherever the KB is cloned or served, and `update_schema` can edit it for you later.

```yaml
# .kb-schema.yaml — at the knowledge-base root
fields:
  title:       { type: text, required: true }
  description: { type: text, required: true }
  type:        { type: enum, required: true, indexed: true, values: [guide, reference, howto] }
  tags:        { type: list, indexed: true }
```

### 2. Create your config (optional)

Skip this step if the default chunking settings work for your knowledge base. Otherwise, start from the [config.example.yaml](config.example.yaml) and customize:

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

chunking:
  target_chunk_size: 1000
  max_chunk_size: 1500
```

Point `GIT_URL` (and optionally `GIT_BRANCH`) at your knowledge base repo via environment variables — see step 3. All other sections (`embedding`, `mcp`, `webhook`) use defaults that work with the Docker Compose stack. Override only if you need different values.

`config.yaml` still has a `frontmatter` block (see [config.example.yaml](config.example.yaml)) for deployments that haven't moved to a root `.kb-schema.yaml` yet — it's the deprecated fallback described in [Frontmatter Validation](#frontmatter-validation) above, not something a new deployment should reach for.

**Changing `config.yaml` later:** most settings in this file can be applied to a
running server without a restart — edit the file and call:

```bash
curl -X POST -H "Authorization: Bearer $MCP_BEARER_TOKEN" \
  http://localhost:8001/admin/reload
```

The response reports exactly what happened: settings that took effect immediately,
settings that need a restart (rate limiting, embedding batch tuning, and anything
tied to authentication — these are baked into services built once at startup), and
settings that need `md-kb-rag index --full` to be meaningful (`chunking.*` — a new
chunk size only applies to documents indexed after the change, so the corpus is
inconsistent until a full reindex rewrites it). A malformed or invalid file is
rejected with the parse/validation error and the running server is left completely
untouched — same as a failed restart would leave it. See [README.md](../README.md#observability)
for the full endpoint reference.

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
