Find documents. With `query`, results are ranked by semantic relevance. Without
one, every match is returned in a stable order with an exact total — use that when
you need a *complete* set rather than the best few.

`granularity` decides what a result is: `document` (one row each) or `chunk` (scored
snippets, several per document). Defaults to `chunk` with a query, `document`
without.

Narrow with `filters`, keyed by frontmatter field — a scalar means equals
(`{"type": "guide"}`), an array means any-of, an object means all-of or a numeric
range (`{"planning.prep_minutes": {"lt": 30}}`). Nested fields use dot-paths.
`path_prefix` restricts by location, and its granularity differs by mode. With a
`query` it matches whole path components — a folder or one document's full path;
`sysadmin` matches the `sysadmin` folder, `sys` matches nothing. Without a query
(enumeration) it is a plain string prefix, so a partial final segment matches too
(`kitchen/recipes/stir_fr` finds `stir_fry.md`). `sysadmin/` and `sysadmin`
behave identically either way. In query mode a document indexed before this
exactness was added still falls back to the string-prefix check until it is next
reindexed; if that ever under-returns against a very selective prefix, the
response carries `path_prefix_truncated: true` rather than silently handing back
fewer matches than actually exist — it settles to always `false` once every
document has been reindexed at least once.

`offset` pages results. Enumeration (no query) is exhaustive — page as deep as
you like. With a query, paging reaches only as far as this search's own ranked
candidate pool, not the whole corpus: past that depth the response marks
`offset_truncated: true` instead of quietly returning less than you asked for —
narrow the query or stop paging rather than trust an empty page as "no more
results".
