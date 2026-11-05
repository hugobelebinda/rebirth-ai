# Build: docker build -t rebirth-ai/go-sandbox:latest docker/go_sandbox
FROM golang:1.22-bookworm
WORKDIR /workspace
RUN useradd --create-home --uid 1000 --shell /bin/bash rebirth \
    && chown -R rebirth:rebirth /workspace
USER rebirth
CMD ["go", "version"]
