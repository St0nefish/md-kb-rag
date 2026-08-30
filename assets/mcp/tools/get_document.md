Read a document by path — repo-relative (`sysadmin/docker/foo.md`) or a unique
basename. Returns the complete markdown, frontmatter included.

Pass `start_line`/`end_line` to read part of a long document. The returned
`content_hash` always covers the whole file — hand it to `write_document` as
`expected_hash` to guard against editing content that moved under you.

The response also carries `links_out` (documents this one links to) and
`links_in` (documents that link to this one) — the link graph, not just the
prose. Each entry has a `kind`: `markdown` for a link written in the document's
own body, or `semantic` for a machine-inferred similarity neighbor (only present
when semantic edges are enabled for this knowledge base). An entry in
`links_out` also carries `exists: false` when its target isn't currently an
indexed document — a broken link, not a link this tool failed to resolve. Both
lists are capped per call; check `has_more`/`total` before assuming you have
seen every edge.
