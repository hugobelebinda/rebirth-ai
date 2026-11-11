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

    @staticmethod
    def _mvn(relative_file: str | None, *, offline: bool) -> str:
        mvn = "mvn -q -o" if offline else "mvn -q -B"
        if relative_file and relative_file.endswith(".java"):
            stem = Path(relative_file).stem
            return f"{mvn} -Dtest={stem if stem.endswith('Test') else stem + 'Test'} test"
        return f"{mvn} test"

    def test_command_for(self, relative_file: str | None = None) -> str:
        """Offline (``-o``): the sandbox's setup step has already downloaded everything."""
        return self._mvn(relative_file, offline=True)

    def setup_command_for(self, project_dir: Path, relative_file: str | None = None) -> str | None:
        """Run the same tests online on the project as it is, before the draft is written in: Maven
        downloads the test runner's own plugin (surefire provider) only when tests actually run, so a
        plain ``dependency:go-offline`` is not enough. Its result is ignored."""
        if not (Path(project_dir) / "pom.xml").is_file():
            return None
        return self._mvn(relative_file, offline=False)
