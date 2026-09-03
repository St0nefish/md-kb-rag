# Single source of truth for the pinned Rust version is rust-toolchain.toml
# at the repo root (`[toolchain] channel = "..."`, #235). CI reads that
# file's channel and passes it here via `--build-arg RUST_VERSION=...`, so
# this pin can no longer silently drift from what CI's own cargo steps (and
# `dtolnay/rust-toolchain`) actually use. The default below only matters for
# a plain local `docker build .` run with no build args — same convention as
# the VERSION/REVISION ARGs further down.
#
# MSRV 1.89: `ingest::acquire_reindex_lock` uses `std::fs::File::lock` /
# `lock_shared`, stabilized in 1.89.0. Do not lower this pin (here or in
# rust-toolchain.toml) without replacing that call. Before #235, this pin and
# CI's `dtolnay/rust-toolchain@stable` had no relationship at all: a too-old
# pin here compiled clean through every cargo step CI ran and failed only in
# this Docker build, at the end of a long job.
ARG RUST_VERSION=1.89
FROM rust:${RUST_VERSION}-alpine AS builder

RUN apk add --no-cache musl-dev openssl-dev openssl-libs-static perl

WORKDIR /build

# Cache dependencies
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,id=md-rag-target,target=/build/target \
    cargo build --release && \
    rm -rf src

# Build real binary
COPY src/ src/
COPY migrations/ migrations/
COPY assets/ assets/
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,id=md-rag-target,target=/build/target \
    touch src/main.rs && cargo build --release && \
    cp target/release/mcp-md-wiki /usr/local/bin/mcp-md-wiki

# Runtime image
FROM alpine:3.21

# Populated by CI (`docker buildx build --build-arg VERSION=... --build-arg
# REVISION=...`); default to "unknown" so a plain local `docker build .` with no
# build args still produces a valid, if uninformative, label instead of an empty
# one. REVISION matters beyond documentation: the arm64 nightly (fix #194) reads
# org.opencontainers.image.revision back out of a previously-built image's config
# to decide whether anything has changed since its last build, so CI actually
# setting this is what makes that skip-if-unchanged check work at all.
ARG VERSION=unknown
ARG REVISION=unknown

LABEL org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.source="https://github.com/St0nefish/md-kb-rag"

RUN apk add --no-cache ca-certificates git

COPY --from=builder /usr/local/bin/mcp-md-wiki /usr/local/bin/mcp-md-wiki

RUN addgroup -g 65532 -S nonroot && adduser -u 65532 -S nonroot -G nonroot

WORKDIR /app

# The app's actual default data_path is /data (source.data_path, config.rs), which is
# where every compose file and deploy template mounts the named volume. Pre-creating
# and chowning it here means Docker propagates that ownership when it initializes a
# fresh named volume, instead of the mountpoint coming up root-owned and unwritable
# by the non-root user below.
RUN mkdir -p /data && chown nonroot:nonroot /data

USER nonroot

HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=10s \
  CMD ["mcp-md-wiki", "health"]

ENTRYPOINT ["mcp-md-wiki"]
CMD ["serve"]
