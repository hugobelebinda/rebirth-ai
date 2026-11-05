# Build: docker build -t rebirth-ai/java-sandbox:latest docker/java_sandbox
FROM maven:3.9-eclipse-temurin-21
WORKDIR /workspace
# Base image may already own UID 1000 — reuse it.
RUN if getent passwd 1000 >/dev/null; then \
      EXISTING="$(getent passwd 1000 | cut -d: -f1)"; \
      chown -R "$EXISTING" /workspace || true; \
    else \
      useradd --create-home --uid 1000 --shell /bin/bash rebirth \
      && chown -R rebirth:rebirth /workspace; \
    fi
USER 1000
CMD ["mvn", "-version"]
