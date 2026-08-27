Show the frontmatter rules governing a path. Schemas cascade by directory — a
`.kb-schema.yaml` applies to its folder and everything below it, deeper files
refining shallower ones. Returns the merged result plus which file contributed each
field.

Call this before writing into a folder you do not already know; the rules in the
server instructions are root-level only.
