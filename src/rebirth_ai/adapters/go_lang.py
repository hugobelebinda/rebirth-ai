"""Go language adapter (Phase 9)."""

from __future__ import annotations

import os
from pathlib import Path

from rebirth_ai.adapters.base import SandboxSpec
from rebirth_ai.ast.chunker import CodeChunk, CodeChunker

DEFAULT_IMAGE = os.environ.get("REBIRTH_GO_SANDBOX_IMAGE", "rebirth-ai/go-sandbox:latest")
SOURCE_SUFFIXES = {".go"}
SKIP_DIR_PARTS = frozenset({"vendor", ".git", "bin", "dist"})


class GoAdapter:
    name = "go"
    SOURCE_SUFFIXES = SOURCE_SUFFIXES

    def __init__(self, *, image: str | None = None, timeout_sec: int | None = None) -> None:
        self._image = image or DEFAULT_IMAGE
        self._timeout_sec = int(
            timeout_sec
            if timeout_sec is not None
            else os.environ.get("REBIRTH_SANDBOX_TIMEOUT_SEC", "180")
        )
        self._chunker = CodeChunker("go")

    def sandbox_spec(self) -> SandboxSpec:
        return SandboxSpec(
            image=self._image,
            workdir="/workspace",
            test_command="go test ./...",
            timeout_sec=self._timeout_sec,
            env={"CGO_ENABLED": "0", "GOFLAGS": "-mod=mod"},
            user="rebirth",
            dockerfile_dir="docker/go_sandbox",
        )

    def is_source_file(self, path: Path) -> bool:
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            return False
        if any(part in SKIP_DIR_PARTS for part in path.parts):
            return False
        return not path.name.endswith("_test.go")

    def parse(self, path: Path, source: str | None = None) -> list[CodeChunk]:
        return self._chunker.chunk_file(path, source=source)

    def setup_command_for(self, project_dir: Path, relative_file: str | None = None) -> str | None:
        """Download the modules ``go.mod`` requires (the tests then run offline)."""
        gomod = Path(project_dir) / "go.mod"
        if not gomod.is_file() or "require" not in gomod.read_text(encoding="utf-8", errors="replace"):
            return None
        return "go mod download"

    def test_command_for(self, relative_file: str | None = None) -> str:
        if relative_file:
            # Directory or package path for `go test`
            target = relative_file
            if target.endswith(".go"):
                target = str(Path(target).parent)
            if not target.startswith("./"):
                target = "./" + target.strip("/")
            return f"go test {target}"
        return "go test ./..."
