Change a directory's frontmatter rules by editing its `.kb-schema.yaml`. Use this
when a document warrants a new tag or field rather than working around the rules.

Operations: `add_values`, `remove_values`, `set_field`, `remove_field`.

A change that would invalidate existing documents is refused and the offenders
listed — pass `force` to apply anyway, `dry_run` to preview. Committed and pushed
like any document edit.
