# Ephemeral JS/TS test sandbox for Rebirth AI.
# Build: docker build -t rebirth-ai/js-ts-sandbox:latest docker/js_ts_sandbox
FROM node:22-bookworm-slim

WORKDIR /workspace

# Handy CLIs; project deps come from the mounted workspace (npm ci/test).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Default to non-root for safer mounts (node user exists in official image).
USER node

CMD ["node", "--version"]
