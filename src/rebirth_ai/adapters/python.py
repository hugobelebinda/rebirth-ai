"""Python language adapter (Phase 8 vertical)."""

from __future__ import annotations

import os
from pathlib import Path

from rebirth_ai.adapters.base import SandboxSpec
from rebirth_ai.ast.chunker import CodeChunk, CodeChunker

DEFAULT_IMAGE = os.environ.get(
    "REBIRTH_PYTHON_SANDBOX_IMAGE",
    "rebirth-ai/python-sandbox:latest",
)
SOURCE_SUFFIXES = {".py"}
SKIP_DIR_PARTS = frozenset(
    {
        "venv",
        ".venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "__pycache__",
        "site-packages",
        ".eggs",
        "dist",
        "build",
    }
)


class PythonAdapter:
    """Adapter for pytest-based Python projects."""

    name = "python"
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
        self._chunker = CodeChunker("python")

    def sandbox_spec(self) -> SandboxSpec:
        return SandboxSpec(
            image=self._image,
            workdir="/workspace",
            test_command="pytest -q",
            timeout_sec=self._timeout_sec,
            env={"PYTHONDONTWRITEBYTECODE": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
            user="rebirth",
            dockerfile_dir="docker/python_sandbox",
        )

    def is_source_file(self, path: Path) -> bool:
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            return False
        if any(part in SKIP_DIR_PARTS for part in path.parts):
            return False
        name = path.name.lower()
        # Skip tests / conftest as migration sources
        if name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py":
            return False
        return True

    def parse(self, path: Path, source: str | None = None) -> list[CodeChunk]:
        return self._chunker.chunk_file(path, source=source)

    def test_command_for(self, relative_file: str | None = None) -> str:
        if relative_file:
            return f"pytest -q {relative_file}"
        return "pytest -q"

    def install_command(self, packages: list[str]) -> str:
        pkgs = " ".join(packages)
        return f"pip install --no-cache-dir {pkgs}"


def get_adapter() -> PythonAdapter:
    return PythonAdapter()
