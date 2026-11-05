# Build: docker build -t rebirth-ai/cpp-sandbox:latest docker/cpp_sandbox
FROM gcc:14-bookworm
WORKDIR /workspace
RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake make \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /bin/bash rebirth \
    && chown -R rebirth:rebirth /workspace
USER rebirth
CMD ["g++", "--version"]
