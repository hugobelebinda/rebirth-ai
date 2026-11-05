"""C# language adapter (Phase 9)."""

from __future__ import annotations

import os
from pathlib import Path

from rebirth_ai.adapters.base import SandboxSpec
from rebirth_ai.ast.chunker import CodeChunk, CodeChunker

DEFAULT_IMAGE = os.environ.get(
    "REBIRTH_CSHARP_SANDBOX_IMAGE", "rebirth-ai/csharp-sandbox:latest"
)
SOURCE_SUFFIXES = {".cs"}
SKIP_DIR_PARTS = frozenset({".git", "bin", "obj", ".vs", "packages"})


class CSharpAdapter:
    name = "csharp"
    SOURCE_SUFFIXES = SOURCE_SUFFIXES

    def __init__(self, *, image: str | None = None, timeout_sec: int | None = None) -> None:
        self._image = image or DEFAULT_IMAGE
        self._timeout_sec = int(
            timeout_sec
            if timeout_sec is not None
            else os.environ.get("REBIRTH_SANDBOX_TIMEOUT_SEC", "300")
        )
        self._chunker = CodeChunker("c_sharp")

    def sandbox_spec(self) -> SandboxSpec:
        return SandboxSpec(
            image=self._image,
            workdir="/workspace",
            test_command="dotnet test --nologo -v q",
            timeout_sec=self._timeout_sec,
            env={"DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_NOLOGO": "1"},
            user="rebirth",
            dockerfile_dir="docker/csharp_sandbox",
            network_mode="bridge",
        )

    def is_source_file(self, path: Path) -> bool:
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            return False
        if any(part in SKIP_DIR_PARTS for part in path.parts):
            return False
        name = path.name
        if name.endswith("Tests.cs") or name.endswith("Test.cs"):
            return False
        if "Tests" in path.parts or ".Tests" in "".join(path.parts):
            # Heuristic: project folders named *.Tests
            if any(part.endswith(".Tests") or part == "Tests" for part in path.parts):
                return False
        return True

    def parse(self, path: Path, source: str | None = None) -> list[CodeChunk]:
        return self._chunker.chunk_file(path, source=source)

    def test_command_for(self, relative_file: str | None = None) -> str:
        # Run against the test project when the common fixture layout is present.
        return (
            "if [ -f MathOps.Tests/MathOps.Tests.csproj ]; then "
            "dotnet test MathOps.Tests/MathOps.Tests.csproj --nologo -v q; "
            "else dotnet test --nologo -v q; fi"
        )
