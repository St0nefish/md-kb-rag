Create, edit, and/or move a document. Commits and pushes; the change becomes
searchable shortly after this returns.

- `content` — the whole file, including YAML frontmatter. Creates `path` if it is
  new, replaces it if it exists.
- `old_string` / `new_string` — replace one exact, unique occurrence instead of
  resending the whole file.
- `frontmatter_patch` — structured edits to JUST the frontmatter, leaving the body
  untouched: a list of `{operation, field, value}` or `{operation, field, values}`
  objects, `field` a dot-path. `operation` is one of `set_field` (set/replace a
  value; `value`), `remove_field` (delete a field; errors if not set), `add_values`
  (append to a list field, creating it if absent, de-duplicated; `values`), or
  `remove_values` (remove from a list field; errors if the field is absent).
  Mirrors `update_schema`'s operation vocabulary, applied to the document's own
  frontmatter values instead of a schema's field declarations. The most common
  small edit — e.g. flipping `status: draft` to `active`, or adding a tag — needs
  no `content` at all.
- `append` — add text to the end of the document body, with no need to read or
  resend existing content. Exactly one newline separates it from what was already
  there; include your own blank line in `append` for one. Never lands inside the
  frontmatter block, even for a document with no body yet.
- `new_path` — relocate. Combines with any edit mode, or stands alone for a pure
  move; the server reads the current body itself. Links pointing at the document are
  rewritten for you. If `path` is a directory, its whole subtree moves.

`content` and `old_string`/`new_string` are whole-document edits, mutually
exclusive with each other AND with `frontmatter_patch`/`append` (which may
combine with each other — patch applies first, then append — but not with a
whole-document edit). Exactly one edit mode is required unless this call is a
pure move (`new_path` alone).

Frontmatter is validated against the *destination* directory's schema — call
`get_schema` first when writing somewhere unfamiliar; this applies to
`frontmatter_patch` results too. Pass `expected_hash` to reject an edit built on
a stale read — it still guards the whole file, even for a `frontmatter_patch`
that only touches a few fields, since the body is carried through from the same
read the hash was checked against.

`structured_content` mirrors what the text summary already says, so you don't have
to parse prose to act on it: `outcome`, `sha` (the commit), `rebased_paths` (paths a
concurrent push pulled in during the pre-push rebase), and `rewritten_paths` (other
documents whose links into a moved document were rewritten). It also carries `diff`
(the unified diff), capped for size — `diff_truncated`/`diff_total_bytes` tell you
whether it was cut and how large the real diff is; the text content always has the
diff in full, uncapped. `sync_failure_cause` appears only when `outcome` is
`committed_pending_sync`.
