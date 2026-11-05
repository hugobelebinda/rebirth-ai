# Ephemeral Python / pytest sandbox for Rebirth AI.
# Build: docker build -t rebirth-ai/python-sandbox:latest docker/python_sandbox
FROM python:3.12-slim-bookworm

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir pytest \
    && useradd --create-home --uid 1000 --shell /bin/bash rebirth \
    && chown -R rebirth:rebirth /workspace

USER rebirth

CMD ["python", "--version"]
