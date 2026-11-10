"""Translation backend for a vLLM OpenAI-compatible server (cloud stage-3 generation, Phase 17).

Same contract as :class:`rebirth_ai.xlang.translator.Translator` (``complete`` → ``Completion``,
``model_name``) so ``training/gen_xlang_pairs.py`` can swap it in; unlike Ollama, vLLM batches
concurrent requests, so callers send many at once from threads (``complete`` is thread-safe).

Sends the same system + user messages as the Ollama backend (Ollama's ``/api/generate`` with a
``system`` field applies the model's chat template the same way). Thinking is switched off for
hybrid-thinking models (Qwen3.x: ``chat_template_kwargs.enable_thinking=false``; other templates
ignore it) and any ``<think>`` block that still appears is dropped before code extraction.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

from rebirth_ai.xlang.prompts import TRANSLATE_SYSTEM_PROMPT
from rebirth_ai.xlang.translator import Completion

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class VllmRejected(RuntimeError):
    """The server answered 4xx (bad request: retrying the same payload will not help)."""


def _post(url: str, payload: dict, *, timeout: float, retries: int = 4) -> dict:
    body = json.dumps(payload).encode("utf-8")
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            if exc.code < 500:
                raise VllmRejected(f"HTTP {exc.code}: {detail}") from exc
            last = RuntimeError(f"HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"vLLM request failed after {retries} tries: {last}")


def strip_thinking(text: str) -> str:
    text = _THINK.sub("", text)
    # an unterminated think block (hit max_tokens while thinking) holds no final answer
    return "" if "<think>" in text else text


class VllmTranslator:
    def __init__(self, server: str, model: str, *, max_model_len: int = 8192, timeout: float = 900) -> None:
        self.server = server.rstrip("/")
        self.model = model
        self.max_model_len = max_model_len
        self.timeout = timeout

    @property
    def model_name(self) -> str:
        return f"vllm:{self.model}"

    def _budget(self, system: str, prompt: str, max_new_tokens: int) -> int:
        # keep prompt + output inside the server's context (vLLM rejects the request otherwise);
        # ~3 chars per token is a conservative estimate for code
        prompt_tokens = (len(system) + len(prompt)) // 3 + 64
        return max(256, min(max_new_tokens, self.max_model_len - prompt_tokens))

    def complete(
        self, prompt: str, *, max_new_tokens: int, system: str = TRANSLATE_SYSTEM_PROMPT, temperature: float = 0.1
    ) -> Completion:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "max_tokens": self._budget(system, prompt, max_new_tokens),
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        url = f"{self.server}/chat/completions"
        try:
            out = _post(url, payload, timeout=self.timeout)
        except VllmRejected as exc:
            if "context" not in str(exc).lower() and "max_tokens" not in str(exc).lower():
                raise
            payload["max_tokens"] = max(128, payload["max_tokens"] // 2)  # estimate was too low once
            out = _post(url, payload, timeout=self.timeout)
        raw = out["choices"][0]["message"].get("content") or ""
        return Completion(raw=strip_thinking(raw), model=self.model_name)

    def complete_sft(self, prompt: str, *, max_new_tokens: int, temperature: float = 0.0) -> Completion:
        """Raw completion for the shared adapter's SFT prompt (``build_xlang_sft_prompt``): the
        adapter was trained to emit the target code then EOS; a new section header also stops it."""
        payload = {"model": self.model, "prompt": prompt, "temperature": temperature,
                   "max_tokens": self._budget("", prompt, max_new_tokens), "stop": ["\n### Source language"]}
        out = _post(f"{self.server}/completions", payload, timeout=self.timeout)
        return Completion(raw=out["choices"][0]["text"], model=self.model_name)
