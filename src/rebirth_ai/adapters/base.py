"""Language adapter protocol — pluggable per target language/ecosystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from rebirth_ai.ast.chunker import CodeChunk


@dataclass(frozen=True)
class SandboxSpec:
    """How to execute verification for a language."""

    image: str
    workdir: str = "/workspace"
    # Shell form executed inside the container (after optional file overlay).
    # `{file}` is replaced with the container-relative path when provided.
    test_command: str = "npm test"
    timeout_sec: int = 120
    # Extra env vars injected into the container.
    env: dict[str, str] = field(default_factory=dict)
    # Docker user for exec (js_ts: node; python: rebirth).
    user: str = "node"
    # Hint for doctor / CLI: relative Dockerfile directory under repo root.
    dockerfile_dir: str = "docker/js_ts_sandbox"
    # Docker network for the optional setup step (dependency download, before the draft is written).
    # The test step itself never has network: the runner disconnects the container first.
    network_mode: str = "none"
    # Time limit of the setup step (downloads).
    setup_timeout_sec: int = 600


@runtime_checkable
class LanguageAdapter(Protocol):
    """Contract every language plugin must satisfy."""

    name: str

    def sandbox_spec(self) -> SandboxSpec:
        """Return Docker image + test invocation defaults."""
        ...

    def parse(self, path: Path, source: str | None = None) -> list[CodeChunk]:
        """AST-chunk a source file."""
        ...

    def test_command_for(self, relative_file: str | None = None) -> str:
        """Build the container test command for an optional single file."""
        ...

    def is_source_file(self, path: Path) -> bool:
        """Whether this adapter owns the given path."""
        ...

    # Optional: ``setup_command_for(project_dir, relative_test) -> str | None`` — the dependency step
    # the sandbox runs with network *before* the draft is written in (see ``sandbox.runner``).
