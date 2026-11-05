# Build: docker build -t rebirth-ai/swift-sandbox:latest docker/swift_sandbox
# Linux Swift toolchain (macOS host path documented if needed).
FROM swift:5.10-jammy
WORKDIR /workspace
RUN useradd --create-home --uid 1000 --shell /bin/bash rebirth \
    && chown -R rebirth:rebirth /workspace
USER rebirth
CMD ["swift", "--version"]
