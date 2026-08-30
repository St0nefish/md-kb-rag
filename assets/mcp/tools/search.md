Find documents. With `query`, results are ranked by semantic relevance. Without
one, every match is returned in a stable order with an exact total — use that when
you need a *complete* set rather than the best few.

`granularity` decides what a result is: `document` (one row each) or `chunk` (scored
snippets, several per document). Defaults to `chunk` with a query, `document`
without.

Narrow with `filters`, keyed by frontmatter field — a scalar means equals
(`{"type": "guide"}`), an array means any-of, an object means all-of or a numeric
range (`{"planning.prep_minutes": {"lt": 30}}`). Nested fields use dot-paths.
`path_prefix` restricts to a folder.

`offset` pages results. Enumeration (no query) is exhaustive — page as deep as
you like. With a query, paging reaches only as far as this search's own ranked
candidate pool, not the whole corpus: past that depth the response marks
`offset_truncated: true` instead of quietly returning less than you asked for —
narrow the query or stop paging rather than trust an empty page as "no more
results".
