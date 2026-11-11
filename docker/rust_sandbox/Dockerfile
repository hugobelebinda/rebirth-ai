# Build: docker build -t rebirth-ai/rust-sandbox:latest docker/rust_sandbox
# Rust 1.99: every edition (2015–2024).
FROM rust:1.99-bookworm
WORKDIR /workspace
RUN useradd --create-home --uid 1000 --shell /bin/bash rebirth \
    && chown -R rebirth:rebirth /workspace
USER rebirth
CMD ["rustc", "--version"]
