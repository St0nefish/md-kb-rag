Delete a document. Removes the file, commits and pushes, and drops it from the
search index. `path` resolves as in `get_document`.

The delete does not refuse or rewrite anything if other documents still link to
it — those links are left dangling and self-heal on each referencing document's
own next reindex. If any exist, their paths are returned as `referencing_paths`,
so you can decide whether to fix or repoint them yourself.

`structured_content` mirrors the text summary: `outcome`, `sha` (the commit),
`diff` (the unified, all-removals diff — capped for size, with
`diff_truncated`/`diff_total_bytes` alongside it; the text content always has it in
full), `rebased_paths`, and `referencing_paths`. `sync_failure_cause` appears only
when `outcome` is `committed_pending_sync`.
