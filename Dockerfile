FROM rust:1.88-alpine AS builder

RUN apk add --no-cache musl-dev openssl-dev openssl-libs-static perl

WORKDIR /build

# Cache dependencies
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release && \
    rm -rf src

# Build real binary
COPY src/ src/
COPY migrations/ migrations/
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    touch src/main.rs && cargo build --release && \
    cp target/release/md-kb-rag /usr/local/bin/md-kb-rag

# Runtime image
FROM alpine:3.21

RUN apk add --no-cache ca-certificates git

COPY --from=builder /usr/local/bin/md-kb-rag /usr/local/bin/md-kb-rag

RUN addgroup -g 65532 -S nonroot && adduser -u 65532 -S nonroot -G nonroot

WORKDIR /app

RUN mkdir -p /app/data && chown nonroot:nonroot /app/data

USER nonroot

HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
  CMD ["md-kb-rag", "health"]

ENTRYPOINT ["md-kb-rag"]
CMD ["serve"]
