FROM rust:1.88-alpine AS builder

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
    cp target/release/md-kb-rag /usr/local/bin/md-kb-rag

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

COPY --from=builder /usr/local/bin/md-kb-rag /usr/local/bin/md-kb-rag

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
  CMD ["md-kb-rag", "health"]

ENTRYPOINT ["md-kb-rag"]
CMD ["serve"]
