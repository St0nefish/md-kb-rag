FROM rust:1.88-alpine AS builder

RUN apk add --no-cache musl-dev openssl-dev openssl-libs-static perl

WORKDIR /build

# Cache dependencies
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs && \
    cargo build --release && \
    rm -rf src

# Build real binary
COPY src/ src/
COPY migrations/ migrations/
RUN touch src/main.rs && cargo build --release

# Runtime image
FROM alpine:3.21

RUN apk add --no-cache ca-certificates git

COPY --from=builder /build/target/release/md-kb-rag /usr/local/bin/md-kb-rag

WORKDIR /app

HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
  CMD ["md-kb-rag", "health"]

ENTRYPOINT ["md-kb-rag"]
CMD ["serve"]
