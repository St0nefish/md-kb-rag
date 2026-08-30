## Summary

What changed and why. Link the issue(s) this addresses — use `Fixes #N` (or `fix #N`)
for each one you want auto-closed when this merges; squash merge means it's this PR's
title/description GitHub reads for that, not any individual commit.

## Test plan

How you verified this. For a code change: which tests you added/ran
(`cargo test`, and `cargo test -- --ignored` against a live Qdrant if you touched
`src/qdrant.rs`). For a doc-only change: what you checked the prose against (source
file/line, an actual command's output, an existing test) — "read it over" isn't
enough for anything making a factual claim about behavior.

## Checklist

- [ ] `cargo fmt -- --check` and `cargo clippy --all-targets -- -D warnings` pass
      (the pre-commit hook from `./scripts/setup-dev.sh` runs both automatically)
- [ ] `cargo test` passes
- [ ] If `deploy/config.example.yaml` changed, it still deserializes against
      `src/config.rs` (the pre-commit hook checks this for you)
- [ ] Docs updated if this changes user-facing behavior, a config default, or an MCP
      tool's contract
