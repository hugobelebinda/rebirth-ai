"""Generation backends for translation (``rebirth migrate --to`` and the translation evals).

- ``auto`` (``migrate --to`` default; also ``routed`` / ``rebirth``): the served cross-language
  model in Ollama when it is installed, else the plain 14B (``OLLAMA_MODEL``). Served = the
  ``translation:`` route's alias in ``configs/routing.yaml`` (``rebirth-xlang:14b``, moved by
  ``rebirth serve publish-xlang`` / ``rebirth serve rollback --lang xlang``; ``rebirth install``
  creates it). ``REBIRTH_XLANG_MODEL`` overrides it; ``REBIRTH_XLANG_MODEL=""`` turns it off.
- ``ollama``: always the plain Qwen2.5-Coder 14B in Ollama (``OLLAMA_MODEL``); the zero-shot
  baseline the evals compare against.
- ``peft``: base 7B with a chat prompt, no adapter (``REBIRTH_PEFT_BASE`` / ``REBIRTH_PEFT_ADAPTER``
  override; the per-language same-language adapters are *not* used — they were trained to
  modernize within one language)

The cross-language model is served through the base model's chat template with the product
prompt (the same request as the plain 14B), as it was evaluated. v2 (Phase 19: LoRA merged, q6_K):
667 / 931 program translations verified vs 551 for the plain 14B, HumanEval 2,065 / 2,229 first try
vs 2,030 (2026-11-11).

Instances are cached per backend so a retry loop or an eval run loads the model once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from rebirth_ai.xlang.prompts import TRANSLATE_SYSTEM_PROMPT

DEFAULT_XLANG_MODEL = "rebirth-xlang:14b"  # the alias `rebirth install` creates


def xlang_model() -> str:
    """Ollama tag of the served cross-language model ("" = do not use one): ``REBIRTH_XLANG_MODEL``,
    else the translation route's alias (configs/routing.yaml), else ``DEFAULT_XLANG_MODEL``."""
    env = os.environ.get("REBIRTH_XLANG_MODEL")
    if env is not None:
        return env.strip()
    try:
        from rebirth_ai import routing

        route = routing.translation_route() or {}
    except Exception:  # unreadable routing.yaml: same as no route
        route = {}
    return str(route.get("model") or DEFAULT_XLANG_MODEL).strip()


def installed_xlang_model(client) -> str | None:
    """The packaged cross-language adapter's tag if Ollama has it, else None (also when Ollama
    cannot be reached: the plain model is used and the request itself reports the error)."""
    want = xlang_model()
    if not want:
        return None
    try:
        names = set(client.list_models())
    except Exception:  # any connection / HTTP error: not available
        return None
    if want in names:
        return want
    if ":" not in want and f"{want}:latest" in names:
        return f"{want}:latest"
    return None


def retry_temperature(attempt: int) -> float:
    """0.1 for the first translation, then 0.4, 0.7 (capped at 0.8) — retries explore: near-greedy
    decoding repeats the same wrong answer for a near-identical prompt."""
    return 0.1 if attempt == 0 else min(0.8, 0.1 + 0.3 * attempt)


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
        self.backend = "auto" if name in {"auto", "routed", "rebirth"} else name
        if self.backend not in {"auto", "ollama", "peft"}:
            raise ValueError(f"unknown translation backend {backend!r} (auto | ollama | peft)")
        self._client = None

    def _get(self):
        if self._client is None:
            if self.backend in {"auto", "ollama"}:
                from rebirth_ai.generate.ollama_client import OllamaClient

                self._client = OllamaClient()
                if self.backend == "auto":
                    packaged = installed_xlang_model(self._client)
                    if packaged:
                        self._client = OllamaClient(model=packaged)
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
        if self.backend in {"auto", "ollama"}:
            return f"ollama:{c.model}"
        return f"peft:{c.base_model}@{c.adapter_path}"

    def complete(
        self, prompt: str, *, max_new_tokens: int, system: str = TRANSLATE_SYSTEM_PROMPT, temperature: float = 0.1
    ) -> Completion:
        """``temperature`` applies to Ollama; the PEFT backend decodes greedily."""
        c = self._get()
        if self.backend in {"auto", "ollama"}:
            raw = c.generate_raw(prompt, system=system, num_predict=max_new_tokens, temperature=temperature)
        else:
            c.max_new_tokens = max_new_tokens
            raw = c.generate_raw(prompt, system=system, clean=False)
        return Completion(raw=raw, model=self.model_name)


_CACHE: dict[str, Translator] = {}


def get_translator(backend: str | None = None) -> Translator:
    key = (backend or os.environ.get("REBIRTH_TRANSLATOR", "auto")).strip().lower()
    if key not in _CACHE:
        _CACHE[key] = Translator(key)
    return _CACHE[key]
