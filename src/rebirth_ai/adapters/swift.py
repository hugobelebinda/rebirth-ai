"""Swift language adapter (Phase 9/10 — Linux toolchain image)."""

from __future__ import annotations

import os
from pathlib import Path

from rebirth_ai.adapters.base import SandboxSpec
from rebirth_ai.ast.chunker import CodeChunk, CodeChunker

DEFAULT_IMAGE = os.environ.get(
    "REBIRTH_SWIFT_SANDBOX_IMAGE", "rebirth-ai/swift-sandbox:latest"
)
SOURCE_SUFFIXES = {".swift"}
SKIP_DIR_PARTS = frozenset({".git", ".build", ".swiftpm"})


class SwiftAdapter:
    name = "swift"
    SOURCE_SUFFIXES = SOURCE_SUFFIXES

    def __init__(self, *, image: str | None = None, timeout_sec: int | None = None) -> None:
        self._image = image or DEFAULT_IMAGE
        self._timeout_sec = int(
            timeout_sec
            if timeout_sec is not None
            else os.environ.get("REBIRTH_SANDBOX_TIMEOUT_SEC", "300")
        )
        self._chunker = CodeChunker("swift")

    def sandbox_spec(self) -> SandboxSpec:
        return SandboxSpec(
            image=self._image,
            workdir="/workspace",
            test_command="swift test",
            timeout_sec=self._timeout_sec,
            env={},
            user="rebirth",
            dockerfile_dir="docker/swift_sandbox",
        )

    def is_source_file(self, path: Path) -> bool:
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            return False
        if any(part in SKIP_DIR_PARTS for part in path.parts):
            return False
        # SPM: Sources/** are migrate targets; Tests/** are not
        parts = path.parts
        if "Tests" in parts:
            return False
        return True

    def parse(self, path: Path, source: str | None = None) -> list[CodeChunk]:
        return self._chunker.chunk_file(path, source=source)

    def setup_command_for(self, project_dir: Path, relative_file: str | None = None) -> str | None:
        """``swift package resolve`` when ``Package.swift`` has package dependencies."""
        manifest = Path(project_dir) / "Package.swift"
        if not manifest.is_file() or ".package(" not in manifest.read_text(encoding="utf-8", errors="replace"):
            return None
        return "swift package resolve"

    def test_command_for(self, relative_file: str | None = None) -> str:
        return "swift test"
