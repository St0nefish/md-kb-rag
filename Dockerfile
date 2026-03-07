FROM rust:1.88-alpine AS chef
RUN apk add --no-cache musl-dev openssl-dev openssl-libs-static perl
RUN cargo install cargo-chef
WORKDIR /build

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /build/recipe.json recipe.json
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release && \
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
