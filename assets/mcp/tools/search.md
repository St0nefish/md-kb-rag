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
