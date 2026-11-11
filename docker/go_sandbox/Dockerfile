# Build: docker build -t rebirth-ai/go-sandbox:latest docker/go_sandbox
# Go 1.27 builds modules declaring any older `go` line; GOTOOLCHAIN=local (the image default) never downloads one.
FROM golang:1.27-bookworm
WORKDIR /workspace
RUN useradd --create-home --uid 1000 --shell /bin/bash rebirth \
    && chown -R rebirth:rebirth /workspace
USER rebirth
CMD ["go", "version"]
