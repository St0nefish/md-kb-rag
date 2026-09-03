# Contributing

mcp-md-wiki is hosted on GitHub (this repo, its issues, and its PRs). The knowledge
bases it indexes live on separate git hosts entirely — don't confuse the two when
you're reading webhook or `deploy/` config; that side of the config always refers to
the *indexed* knowledge base's git host, never this repo's.

## Workflow

`master` is branch-protected: direct pushes are disabled, and a passing status check
is required before a PR can merge. Concretely:

1. **Branch.** Work on a feature branch, not directly on `master` (you can't push to
   it anyway). There's no enforced naming convention beyond "descriptive" —
   `fix/reranking-timeout`, `docs/backup-recovery`, that kind of thing.
2. **Open a PR against `master`.** CI (`.github/workflows/ci.yml`) runs on every PR:
   `cargo fmt -- --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`,
   and `cargo audit`, followed by a Docker build. The `test` job only runs when your
   diff touches `src/**`, `assets/**`, `Cargo.toml`/`Cargo.lock`, the `Dockerfile`,
   `docker-compose.yml`, `migrations/**`, or `ci.yml` itself — a docs-only PR like a
   `deploy/*.md` or `CONTRIBUTING.md` change skips it. The actual required status
   check is a gate job, `ci-pass`, which passes whether `test` succeeded *or* was
   skipped, so a doc-only PR still goes green without spinning up the full suite.
3. **Auto-merge.** A separate `auto-merge.yml` workflow enables GitHub's native
   auto-merge (squash) on every PR as soon as it's opened. Once `ci-pass` reports
   success, GitHub merges it automatically — there's no separate manual merge step
   for a PR that's ready and green.
4. **Closing issues.** Include `fix #N` (or `Fixes #N`, `Closes #N` — GitHub's usual
   set of magic words) in the PR body or the squash commit message to auto-close the
   corresponding issue when the PR merges. Squash means the *PR's* commit message is
   what GitHub actually reads for this, not any individual commit within it.
5. **Cleanup.** Branches auto-delete on merge — no need to clean up your own feature
   branch afterward.

Bugs, features, and enhancements are all tracked as GitHub issues; this repo doesn't
keep in-repo TODO files.

## Local setup

After cloning, run the one-time setup script:

```bash
./scripts/setup-dev.sh
```

This points `git config core.hooksPath` at `.githooks/`, which activates a
pre-commit hook that:

- Runs `cargo fmt` on your staged `.rs` files and re-stages the result.
- Runs `cargo clippy --all-targets -- -D warnings` (the `--all-targets` matters —
  without it, lint failures confined to `#[cfg(test)]` code slip past both the hook
  and, historically, CI too) and **blocks the commit** if it fails.
- If your change touches `deploy/config.example.yaml`, also runs
  `cargo test -q -- config::tests::example_config_deserializes` — a single test that
  parses the example config against the live `Config` struct and spot-checks several
  values, specifically to catch the file drifting out of sync with `src/config.rs`.
  A key that no longer exists, or a value that no longer matches its default, fails
  the commit rather than merging silently.

You can bypass the hook with `git commit --no-verify` when you genuinely need to, but
anything it would have caught still has to pass in CI before the PR can merge.

## Rust toolchain / MSRV

The pinned toolchain — currently **1.89** — lives in one place, `rust-toolchain.toml`
at the repo root, and everything else derives from it (#235):

- **Local dev** needs nothing extra: rustup reads `rust-toolchain.toml` automatically
  for every `cargo`/`rustc` invocation anywhere under this directory tree.
- **CI** (`.github/workflows/ci.yml`) extracts the file's `channel` value in a "Read
  pinned Rust toolchain version" step and feeds it explicitly to
  `dtolnay/rust-toolchain@master` (that action doesn't read the file itself) — both the
  `test` and `qdrant-integration` jobs do this, so neither can end up compiling against
  a different toolchain than the other.
- **The Docker build** takes the same value as a `--build-arg RUST_VERSION=...`, passed
  by that same CI step, so the release image is built with the exact toolchain CI just
  tested against — not `rust:X.Y-alpine` pinned by hand in the `Dockerfile` and left to
  drift out of sync with whatever `stable` CI happened to resolve to that day.
- **`rust-version` in `Cargo.toml`** carries the same number, so a plain `cargo build`
  under a too-old local toolchain fails fast with a clear "package requires rustc X but
  Y is installed" message instead of a confusing type error partway through a build.

This exists because the split bit twice in one session before #235: CI resolving
`dtolnay/rust-toolchain@stable` while the `Dockerfile` pinned an explicit, unrelated
version meant a feature stabilized after the Dockerfile's pin (`std::fs::File::lock`,
1.89.0) compiled clean through every `cargo` step CI ran and failed only in the Docker
build, at the end of a long job. Bumping the MSRV means updating `channel` in
`rust-toolchain.toml`, `rust-version` in `Cargo.toml`, and the `ARG RUST_VERSION`
default in `Dockerfile` together — CI and the Docker build then just follow.

## Testing

There is no `tests/` directory — every test lives inline in the module it tests,
inside a `#[cfg(test)] mod tests { ... }` block. Add tests next to the code they
cover rather than in a separate top-level tree.

```bash
cargo test                          # full suite (unit + most integration-style tests)
cargo fmt -- --check                # what the pre-commit hook and CI both enforce
cargo clippy --all-targets -- -D warnings
```

A small number of tests in `src/qdrant.rs` are `#[ignore]`d because they need a real
Qdrant server, not the mocked `VectorStore` the rest of the suite uses — plain
`cargo test` never runs them. Exercise them against this project's own Qdrant
service:

```bash
docker compose up -d qdrant
cargo test -- --ignored
```

CI runs this as a separate `qdrant-integration` job, deliberately `continue-on-error`
for now (see the comment above that job in `ci.yml` for why) — it isn't part of the
`ci-pass` gate yet, so a failure there won't block your PR, but a genuine regression
is still worth fixing.

## Style

Skim any of `src/schema.rs`, `src/ingest.rs`, or `src/server.rs` before writing new
code here — the house style leans heavily on dense explanatory comments that state
*why* a piece of logic exists (the failure mode it prevents, the issue number it
traces to, the tradeoff it made and the alternative it rejected), not just what the
code does line by line. A one-line "what" comment on a non-obvious function is
usually a sign more context belongs there. This applies to config field doc comments
too — `src/config.rs` and `deploy/config.example.yaml` are meant to explain the
reasoning behind a default, not just name it.

Formatting and linting are enforced mechanically (`cargo fmt`, `cargo clippy -D
warnings`) rather than by convention, so there's nothing further to memorize there.
