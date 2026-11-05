# Build: docker build -t rebirth-ai/rust-sandbox:latest docker/rust_sandbox
FROM rust:1.82-bookworm
WORKDIR /workspace
RUN useradd --create-home --uid 1000 --shell /bin/bash rebirth \
    && chown -R rebirth:rebirth /workspace
USER rebirth
CMD ["rustc", "--version"]
