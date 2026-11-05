"""Java language adapter (Phase 9)."""

from __future__ import annotations

import os
from pathlib import Path

from rebirth_ai.adapters.base import SandboxSpec
from rebirth_ai.ast.chunker import CodeChunk, CodeChunker

DEFAULT_IMAGE = os.environ.get("REBIRTH_JAVA_SANDBOX_IMAGE", "rebirth-ai/java-sandbox:latest")
SOURCE_SUFFIXES = {".java"}
SKIP_DIR_PARTS = frozenset({".git", "target", "build", ".gradle", ".idea"})


class JavaAdapter:
    name = "java"
    SOURCE_SUFFIXES = SOURCE_SUFFIXES

    def __init__(self, *, image: str | None = None, timeout_sec: int | None = None) -> None:
        self._image = image or DEFAULT_IMAGE
        self._timeout_sec = int(
            timeout_sec
            if timeout_sec is not None
            else os.environ.get("REBIRTH_SANDBOX_TIMEOUT_SEC", "300")
        )
        self._chunker = CodeChunker("java")

    def sandbox_spec(self) -> SandboxSpec:
        return SandboxSpec(
            image=self._image,
            workdir="/workspace",
            test_command="mvn -q test",
            timeout_sec=self._timeout_sec,
            env={"MAVEN_OPTS": "-Xmx512m"},
            user="1000",
            dockerfile_dir="docker/java_sandbox",
            network_mode="bridge",
        )

    def is_source_file(self, path: Path) -> bool:
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            return False
        if any(part in SKIP_DIR_PARTS for part in path.parts):
            return False
        name = path.name
        if name.endswith("Test.java") or name.endswith("Tests.java"):
            return False
        if "src/test" in path.as_posix().replace("\\", "/"):
            return False
        return True

    def parse(self, path: Path, source: str | None = None) -> list[CodeChunk]:
        return self._chunker.chunk_file(path, source=source)

    def test_command_for(self, relative_file: str | None = None) -> str:
        if relative_file and relative_file.endswith(".java"):
            stem = Path(relative_file).stem
            if stem.endswith("Test"):
                return f"mvn -q -Dtest={stem} test"
            return f"mvn -q -Dtest={stem}Test test"
        return "mvn -q test"
