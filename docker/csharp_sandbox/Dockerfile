# Build: docker build -t rebirth-ai/csharp-sandbox:latest docker/csharp_sandbox
FROM mcr.microsoft.com/dotnet/sdk:8.0-bookworm-slim
WORKDIR /workspace
RUN useradd --create-home --uid 1000 --shell /bin/bash rebirth \
    && chown -R rebirth:rebirth /workspace
USER rebirth
CMD ["dotnet", "--info"]
