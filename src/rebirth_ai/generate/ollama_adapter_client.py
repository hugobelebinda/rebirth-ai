"""Serve a per-language LoRA through Ollama (Phase 16): the shipped artifact.

The Ollama model is ``FROM <quantized base>`` + ``ADAPTER <lora>.gguf`` (see
``rebirth_ai.packaging``). This client reproduces ``PeftClient`` exactly:

* same plan (``plan_generation``): SFT completion prompt for short files, chat for long
  files / retries, same token budgets;
* same raw text the HF model saw: requests use ``raw: true`` and the client writes the
  Qwen ChatML itself for chat-style plans (what ``apply_chat_template`` produces);
* same cleanup (``_clean_sft_completion`` + ``extract_code_block``).

Greedy, seed 0. Ollama stops at the GGUF EOS (``<|im_end|>``) and at the plan's stop strings.
"""

from __future__ import annotations

import os

import httpx

from rebirth_ai.generate.ollama_client import GenerateResult
from rebirth_ai.generate.peft_client import _clean_sft_completion, plan_generation
from rebirth_ai.generate.prompts import extract_code_block

IM_END = "<|im_end|>"
_CTX_BUCKETS = (4096, 8192, 16384, 32768)  # few distinct num_ctx values → few model reloads


def chatml(system: str, user: str) -> str:
    """Qwen2.5 chat template with a generation prompt (matches the HF tokenizer)."""
    return (
        f"<|im_start|>system\n{system}{IM_END}\n"
        f"<|im_start|>user\n{user}{IM_END}\n"
        "<|im_start|>assistant\n"
    )


def num_ctx_for(prompt: str, max_new_tokens: int) -> int:
    need = len(prompt) // 3 + max_new_tokens + 64  # ~3 chars / token for code (conservative)
    return next((b for b in _CTX_BUCKETS if b >= need), _CTX_BUCKETS[-1])


class OllamaAdapterClient:
    def __init__(
        self,
        *,
        model: str,
        language: str | None = None,
        host: str | None = None,
        timeout_sec: float = 600.0,
        max_new_tokens: int = 1024,
        prompt_style: str = "sft",
    ) -> None:
        self.model = model
        self.language = language
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_new_tokens = max_new_tokens
        self.prompt_style = prompt_style

    def request_payload(self, plan) -> dict:
        if plan.style == "sft":
            prompt, stop = plan.prompt, list(plan.stop)
        else:
            prompt, stop = chatml(plan.system, plan.prompt), [IM_END]
        return {
            "model": self.model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "seed": 0,
                "num_predict": plan.max_new_tokens,
                "num_ctx": num_ctx_for(prompt, plan.max_new_tokens),
                "stop": stop,
            },
        }

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
    ) -> GenerateResult:
        plan = plan_generation(
            prompt_style=self.prompt_style,
            objective=objective,
            legacy_code=legacy_code,
            file_path=file_path,
            language=language,
            max_new_tokens=self.max_new_tokens,
            retrieved_context=retrieved_context,
            error_class=error_class,
            test_logs=test_logs,
        )
        payload = self.request_payload(plan)
        with httpx.Client(timeout=self.timeout_sec) as client:
            resp = client.post(f"{self.host}/api/generate", json=payload)
            resp.raise_for_status()
            raw = str(resp.json().get("response", ""))
        cleaned = _clean_sft_completion(raw)
        return GenerateResult(
            model=f"ollama:{self.model}",
            raw_text=raw,
            code=extract_code_block(cleaned),
            prompt_chars=len(plan.prompt),
        )
