"""Select the active code generator (Ollama vs local PEFT)."""

from __future__ import annotations

import os
import sys
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


def get_generator(backend: str | None = None, *, language: str | None = None) -> Generator:
    """Return generator from ``REBIRTH_GENERATOR`` (``ollama`` | ``peft``).

    With ``peft`` and a known non-JS ``language`` that has no trained adapter
    (``data/models/lora_<lang>_7b/adapter``) and no ``REBIRTH_PEFT_*`` override,
    fall back to Ollama rather than running the JS/TS adapter on another language.
    """
    name = (backend or os.environ.get("REBIRTH_GENERATOR", "ollama")).strip().lower()
    if name in {"peft", "lora", "local"}:
        from rebirth_ai.generate.peft_client import PeftClient, lang_adapter, peft_env_override
        from rebirth_ai.lora import normalize_lang

        lang_id = normalize_lang(language)
        if (
            lang_id not in (None, "js_ts")
            and not peft_env_override()
            and lang_adapter(lang_id) is None
        ):
            print(
                f"[peft] no LoRA adapter for {lang_id} (data/models/lora_{lang_id}_7b/adapter); "
                "falling back to Ollama",
                file=sys.stderr,
            )
            return OllamaClient()
        return PeftClient(language=lang_id)
    return OllamaClient()
