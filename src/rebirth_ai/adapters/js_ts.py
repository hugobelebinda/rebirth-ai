"""JavaScript / TypeScript language adapter (first vertical)."""

from __future__ import annotations

import os
from pathlib import Path

from rebirth_ai.adapters.base import SandboxSpec
from rebirth_ai.ast.chunker import CodeChunk, CodeChunker

DEFAULT_IMAGE = os.environ.get("REBIRTH_SANDBOX_IMAGE", "rebirth-ai/js-ts-sandbox:latest")
SOURCE_SUFFIXES = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}


class JsTsAdapter:
    """Adapter for Node.js / JS / TS projects."""

    name = "js_ts"
    SOURCE_SUFFIXES = SOURCE_SUFFIXES

    def __init__(
        self,
        *,
        image: str | None = None,
        timeout_sec: int | None = None,
    ) -> None:
        self._image = image or DEFAULT_IMAGE
        self._timeout_sec = int(
            timeout_sec
            if timeout_sec is not None
            else os.environ.get("REBIRTH_SANDBOX_TIMEOUT_SEC", "120")
        )
        self._js_chunker = CodeChunker("javascript")
        self._ts_chunker = CodeChunker("typescript")
        self._tsx_chunker = CodeChunker("tsx")

    def sandbox_spec(self) -> SandboxSpec:
        return SandboxSpec(
            image=self._image,
            workdir="/workspace",
            test_command="npm test",
            timeout_sec=self._timeout_sec,
            env={"CI": "1", "NODE_ENV": "test"},
            user="node",
            dockerfile_dir="docker/js_ts_sandbox",
        )

    def install_command(self, packages: list[str]) -> str:
        pkgs = " ".join(packages)
        return f"npm install --no-save --no-package-lock {pkgs}"

    def is_source_file(self, path: Path) -> bool:
        return path.suffix.lower() in SOURCE_SUFFIXES

    def parse(self, path: Path, source: str | None = None) -> list[CodeChunk]:
        suffix = path.suffix.lower()
        if suffix == ".tsx":
            chunker = self._tsx_chunker
        elif suffix in {".ts"}:
            chunker = self._ts_chunker
        else:
            chunker = self._js_chunker
        return chunker.chunk_file(path, source=source)

    def test_command_for(self, relative_file: str | None = None) -> str:
        """Use Node's built-in test runner; optionally target one test file."""
        if relative_file:
            # Escape is minimal — paths come from our fixture/temp dirs.
            return f"node --test {relative_file}"
        return "npm test"


def get_adapter() -> JsTsAdapter:
    return JsTsAdapter()
