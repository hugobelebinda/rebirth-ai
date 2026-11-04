"""Select the active code generator (Ollama vs local PEFT)."""

from __future__ import annotations

import os
from typing import Protocol

from rebirth_ai.generate.ollama_client import GenerateResult, OllamaClient


class Generator(Protocol):
    def migrate(
        self,
        *,
        objective: str,
        legacy_code: str,
        file_path: str,
        language: str = "javascript",
        retrieved_context: list[str] | None = None,
        error_class: str | None = None,
        test_logs: str | None = None,
    ) -> GenerateResult: ...


def get_generator(backend: str | None = None) -> Generator:
    """Return generator from ``REBIRTH_GENERATOR`` (``ollama`` | ``peft``)."""
    name = (backend or os.environ.get("REBIRTH_GENERATOR", "ollama")).strip().lower()
    if name in {"peft", "lora", "local"}:
        from rebirth_ai.generate.peft_client import PeftClient

        return PeftClient()
    return OllamaClient()
