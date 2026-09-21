# syntax=docker/dockerfile:1

# Stage 1: Build binary
FROM rust:bookworm AS builder

WORKDIR /app

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
FROM debian:bookworm-slim AS runtime

# Install CA certificates for HTTPS model downloads
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user and data directory
RUN useradd -u 10001 -U -M -s /usr/sbin/nologin appuser && \
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
