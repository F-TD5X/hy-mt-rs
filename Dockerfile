# syntax=docker/dockerfile:1

# Stage 1: Build binary
FROM rust:alpine AS builder

WORKDIR /app

# Install git for cargo git dependencies (candle-core) and build tools
RUN apk add --no-cache git musl-dev

# Copy dependency manifests and source code
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/

# Build release binary with locked dependencies and cache mounts
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --release --locked && \
    cp /app/target/release/hy-mt-rs /usr/local/bin/hy-mt-rs

# Stage 2: Minimal runtime
FROM alpine:latest AS runtime

# Install CA certificates for HTTPS model downloads and curl for healthchecks
RUN apk add --no-cache ca-certificates curl

# Create non-root user and data directory
RUN addgroup -g 10001 -S appuser && \
    adduser -u 10001 -S -D -H -s /sbin/nologin -G appuser appuser && \
    mkdir -p /data && \
    chown -R appuser:appuser /data

# Copy compiled binary from builder
COPY --from=builder /usr/local/bin/hy-mt-rs /usr/local/bin/hy-mt-rs

USER appuser:appuser
WORKDIR /data

ENV HY_MT_DATA_DIR=/data

EXPOSE 8080

ENTRYPOINT ["hy-mt-rs"]
CMD ["serve", "--listen", "0.0.0.0:8080"]
