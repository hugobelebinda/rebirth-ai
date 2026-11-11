"""Rust language adapter (Phase 9)."""

from __future__ import annotations

import os
from pathlib import Path

from rebirth_ai.adapters.base import SandboxSpec
from rebirth_ai.ast.chunker import CodeChunk, CodeChunker

DEFAULT_IMAGE = os.environ.get("REBIRTH_RUST_SANDBOX_IMAGE", "rebirth-ai/rust-sandbox:latest")
SOURCE_SUFFIXES = {".rs"}
SKIP_DIR_PARTS = frozenset({".git", "target", ".cargo"})


class RustAdapter:
    name = "rust"
    SOURCE_SUFFIXES = SOURCE_SUFFIXES
    package_level_tests = True

    def __init__(self, *, image: str | None = None, timeout_sec: int | None = None) -> None:
        self._image = image or DEFAULT_IMAGE
        self._timeout_sec = int(
            timeout_sec
            if timeout_sec is not None
            else os.environ.get("REBIRTH_SANDBOX_TIMEOUT_SEC", "300")
        )
        self._chunker = CodeChunker("rust")

    def sandbox_spec(self) -> SandboxSpec:
        return SandboxSpec(
            image=self._image,
            workdir="/workspace",
            test_command="cargo test --quiet",
            timeout_sec=self._timeout_sec,
            env={"CARGO_TERM_COLOR": "never"},
            user="rebirth",
            dockerfile_dir="docker/rust_sandbox",
        )

    def is_source_file(self, path: Path) -> bool:
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            return False
        if any(part in SKIP_DIR_PARTS for part in path.parts):
            return False
        # Prefer lib/main under src; skip dedicated tests/ tree
        if "tests" in path.parts and path.parent.name == "tests":
            return False
        return True

    def parse(self, path: Path, source: str | None = None) -> list[CodeChunk]:
        return self._chunker.chunk_file(path, source=source)

    def setup_command_for(self, project_dir: Path, relative_file: str | None = None) -> str | None:
        """``cargo fetch`` when ``Cargo.toml`` has dependencies (the tests then run offline)."""
        import tomllib

        try:
            data = tomllib.loads((Path(project_dir) / "Cargo.toml").read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            return None
        sections = ("dependencies", "dev-dependencies", "build-dependencies")
        has = any(data.get(s) for s in sections) or any(
            (t or {}).get(s) for t in (data.get("target") or {}).values() for s in sections)
        return "cargo fetch" if has or (data.get("workspace") or {}).get("dependencies") else None

    def test_command_for(self, relative_file: str | None = None) -> str:
        return "cargo test --quiet"
