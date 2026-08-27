Create, edit, and/or move a document. Commits and pushes; the change becomes
searchable shortly after this returns.

- `content` — the whole file, including YAML frontmatter. Creates `path` if it is
  new, replaces it if it exists.
- `old_string` / `new_string` — replace one exact, unique occurrence instead of
  resending the whole file.
- `new_path` — relocate. Combines with either edit mode, or stands alone for a pure
  move; the server reads the current body itself. Links pointing at the document are
  rewritten for you. If `path` is a directory, its whole subtree moves.

Frontmatter is validated against the *destination* directory's schema — call
`get_schema` first when writing somewhere unfamiliar. Pass `expected_hash` to reject
an edit built on a stale read.
