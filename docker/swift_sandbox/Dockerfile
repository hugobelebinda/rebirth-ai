# Build: docker build -t rebirth-ai/swift-sandbox:latest docker/swift_sandbox
# Linux Swift toolchain (macOS host path documented if needed).
# Swift 6.4: Package.swift tools versions 5.x (Swift 5 language mode) and 6.x.
FROM swift:6.4-bookworm
WORKDIR /workspace
RUN useradd --create-home --uid 1000 --shell /bin/bash rebirth \
    && chown -R rebirth:rebirth /workspace
USER rebirth
CMD ["swift", "--version"]
