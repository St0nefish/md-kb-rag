Read a document by path — repo-relative (`sysadmin/docker/foo.md`) or a unique
basename. Returns the complete markdown, frontmatter included.

Pass `start_line`/`end_line` to read part of a long document. The returned
`content_hash` always covers the whole file — hand it to `write_document` as
`expected_hash` to guard against editing content that moved under you.
