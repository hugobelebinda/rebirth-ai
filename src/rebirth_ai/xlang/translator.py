"""Generation backends for translation (zero-shot until a cross-language adapter exists).

- ``ollama`` (default; ``auto`` = this): plain Qwen2.5-Coder 14B in Ollama (``OLLAMA_MODEL``)
- ``peft``: base 7B with a chat prompt, no adapter (``REBIRTH_PEFT_BASE`` / ``REBIRTH_PEFT_ADAPTER``
  override; the per-language same-language adapters are *not* used — they were trained to
  modernize within one language)

Instances are cached per backend so a retry loop or an eval run loads the 7B once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from rebirth_ai.xlang.prompts import TRANSLATE_SYSTEM_PROMPT


@dataclass
class Completion:
    raw: str
    model: str


def max_new_tokens_for(source_code: str) -> int:
    # Code in a different language is about as long as the source (C++/Java/Go usually longer
    # than Python): ~2× the source's token count (~3.5 chars / token), floor 512, ceiling 4096.
    return max(512, min(4096, int(len(source_code) / 3.5 * 2) + 256))


class Translator:
    def __init__(self, backend: str = "ollama") -> None:
        name = backend.strip().lower()
        self.backend = "ollama" if name in {"auto", "routed", "rebirth", "ollama"} else name
        if self.backend not in {"ollama", "peft"}:
            raise ValueError(f"unknown translation backend {backend!r} (ollama | peft)")
        self._client = None

    def _get(self):
        if self._client is None:
            if self.backend == "ollama":
                from rebirth_ai.generate.ollama_client import OllamaClient

                self._client = OllamaClient()
            else:
                from rebirth_ai.generate.peft_client import DEFAULT_BASE_7B, NO_ADAPTER, PeftClient

                self._client = PeftClient(
                    base_model=os.environ.get("REBIRTH_PEFT_BASE", DEFAULT_BASE_7B),
                    adapter_path=os.environ.get("REBIRTH_PEFT_ADAPTER", NO_ADAPTER),
                    prompt_style="chat",
                )
        return self._client

    @property
    def model_name(self) -> str:
        c = self._get()
        if self.backend == "ollama":
            return f"ollama:{c.model}"
        return f"peft:{c.base_model}@{c.adapter_path}"

    def complete(
        self, prompt: str, *, max_new_tokens: int, system: str = TRANSLATE_SYSTEM_PROMPT, temperature: float = 0.1
    ) -> Completion:
        """``temperature`` applies to Ollama; the PEFT backend decodes greedily."""
        c = self._get()
        if self.backend == "ollama":
            raw = c.generate_raw(prompt, system=system, num_predict=max_new_tokens, temperature=temperature)
        else:
            c.max_new_tokens = max_new_tokens
            raw = c.generate_raw(prompt, system=system, clean=False)
        return Completion(raw=raw, model=self.model_name)


_CACHE: dict[str, Translator] = {}


def get_translator(backend: str | None = None) -> Translator:
    key = (backend or os.environ.get("REBIRTH_TRANSLATOR", "ollama")).strip().lower()
    if key not in _CACHE:
        _CACHE[key] = Translator(key)
    return _CACHE[key]
