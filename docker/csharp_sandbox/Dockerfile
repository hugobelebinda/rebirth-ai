# Build: docker build -t rebirth-ai/csharp-sandbox:latest docker/csharp_sandbox
# .NET 10 SDK (LTS) with the .NET 8 SDK, runtime and targeting packs beside it: projects targeting net8.0
# build and run offline, with no targeting-pack download (the test step has no network).
FROM mcr.microsoft.com/dotnet/sdk:8.0-noble AS sdk8

FROM mcr.microsoft.com/dotnet/sdk:10.0-noble
COPY --from=sdk8 /usr/share/dotnet/sdk /usr/share/dotnet/sdk
COPY --from=sdk8 /usr/share/dotnet/shared /usr/share/dotnet/shared
COPY --from=sdk8 /usr/share/dotnet/packs /usr/share/dotnet/packs
WORKDIR /workspace
# Ubuntu images already have uid 1000 (ubuntu): rename it to rebirth, the user the C# adapter runs as.
RUN if id ubuntu >/dev/null 2>&1; then \
      usermod -l rebirth -d /home/rebirth -m -G "" ubuntu && groupmod -n rebirth ubuntu; \
    else \
      useradd --create-home --uid 1000 --shell /bin/bash rebirth; \
    fi \
    && chown -R rebirth:rebirth /workspace
USER rebirth
CMD ["dotnet", "--info"]
