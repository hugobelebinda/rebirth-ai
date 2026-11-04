"""Local PEFT LoRA generation (0.5B / 7B specialization path)."""

from __future__ import annotations

import os
import re
from pathlib import Path

from rebirth_ai.generate.ollama_client import GenerateResult
from rebirth_ai.generate.prompts import (
    SYSTEM_PROMPT,
    build_migration_prompt,
    build_sft_prompt,
    extract_code_block,
)

# Prefer the proven 7B ROCm adapter; fall back to 0.5B if 7B is missing.
DEFAULT_BASE_7B = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_ADAPTER_7B = Path("data/models/lora_js_ts_7b/adapter")
DEFAULT_BASE_05B = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEFAULT_ADAPTER_05B = Path("data/models/lora_js_ts/adapter")
# Back-compat aliases
DEFAULT_BASE = DEFAULT_BASE_7B
DEFAULT_ADAPTER = DEFAULT_ADAPTER_7B


def resolve_peft_paths(
    *,
    base_model: str | None = None,
    adapter_path: Path | str | None = None,
) -> tuple[str, Path]:
    """Pick PEFT base + adapter: explicit args → env → 7B if present → 0.5B."""
    env_base = os.environ.get("REBIRTH_PEFT_BASE")
    env_adapter = os.environ.get("REBIRTH_PEFT_ADAPTER")

    if base_model is not None or adapter_path is not None:
        base = base_model or env_base or DEFAULT_BASE_7B
        raw = adapter_path or env_adapter or str(DEFAULT_ADAPTER_7B)
        return base, Path(raw)

    if env_base or env_adapter:
        base = env_base or DEFAULT_BASE_7B
        raw = env_adapter or str(DEFAULT_ADAPTER_7B)
        return base, Path(raw)

    if DEFAULT_ADAPTER_7B.exists():
        return DEFAULT_BASE_7B, DEFAULT_ADAPTER_7B
    if DEFAULT_ADAPTER_05B.exists():
        return DEFAULT_BASE_05B, DEFAULT_ADAPTER_05B
    return DEFAULT_BASE_7B, DEFAULT_ADAPTER_7B

# Stop strings for SFT-style completions (needs tokenizer= in generate)
_SFT_STOP_STRINGS = [
    "\n### Objective",
    "\n### Legacy",
    "\n### Modern",
    "\n###",
    "\n```",
]

# Trailing junk patterns seen from under-trained adapters (not real file structure)
_LEAK_MARKERS = (
    "\n### Objective",
    "\n### Legacy",
    "\n### Modern",
    "\n###import",
    "\n###",
    "\ndescribe(",
    "\n## [",
    "\n```\n",
)


def _clean_sft_completion(text: str) -> str:
    """Drop leaked SFT headers, fences, and trailing prose from completions."""
    cleaned = text.strip()
    for marker in ("### Modern", "### Legacy", "### Objective", "###"):
        if cleaned.startswith(marker):
            cleaned = cleaned[len(marker) :].lstrip("\n# ")

    for marker in _LEAK_MARKERS:
        if marker in cleaned:
            head, tail = cleaned.split(marker, 1)
            if head.strip():
                cleaned = head
                break
            cleaned = tail.lstrip("\n# ")

    if cleaned.lstrip().startswith("```"):
        cleaned = extract_code_block(cleaned)

    cleaned = cleaned.rstrip()
    return cleaned + "\n" if cleaned else ""

def _sft_max_new_tokens(legacy_code: str, requested: int) -> int:
    """Bound generation length from input size so short idioms stop sooner."""
    # ~4 chars/token rough; allow 2× rewrite headroom, clamp for short prompts
    approx_in = max(16, len(legacy_code) // 3)
    bounded = max(64, min(requested, approx_in * 2, 256))
    return bounded


def _chat_max_new_tokens(legacy_code: str, requested: int) -> int:
    """Scale chat decode budget with file size (Phase 6 whole-file repairs)."""
    lines = legacy_code.count("\n") + 1
    chars = len(legacy_code)
    # ~3 chars/token + rewrite headroom; floor 256, ceiling requested
    approx = max(256, min(requested, max(lines * 8, chars // 2, 512)))
    return int(approx)


def _default_prompt_style() -> str:
    return os.environ.get("REBIRTH_PEFT_PROMPT_STYLE", "sft").strip().lower() or "sft"


class PeftClient:
    """Load base + LoRA adapter once; generate migration drafts on CPU/GPU."""

    def __init__(
        self,
        *,
        base_model: str | None = None,
        adapter_path: Path | str | None = None,
        max_new_tokens: int = 1024,
        prompt_style: str | None = None,
    ) -> None:
        self.base_model, self.adapter_path = resolve_peft_paths(
            base_model=base_model,
            adapter_path=adapter_path,
        )
        self.max_new_tokens = max_new_tokens
        self.prompt_style = prompt_style or _default_prompt_style()  # sft | chat
        self._model = None
        self._tokenizer = None
        self._device = None
        self._sft_token_cap: int | None = None
        self._chat_token_cap: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.adapter_path.exists():
            raise FileNotFoundError(
                f"LoRA adapter not found at {self.adapter_path}. "
                "Train 7B: see docs/DATASET.md (or 0.5B: python training/train_lora.py --cpu)"
            )
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # REBIRTH_PEFT_DEVICE=cpu|cuda|auto (default auto). Prefer CPU if you
        # must keep the desktop display stable on the same AMD dGPU.
        pref = os.environ.get("REBIRTH_PEFT_DEVICE", "auto").strip().lower()
        want_cuda = torch.cuda.is_available() and pref in {"auto", "cuda", "gpu"}
        if pref == "cpu":
            want_cuda = False
        self._device = torch.device("cuda" if want_cuda else "cpu")
        dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        print(f"[peft] loading {self.base_model} + {self.adapter_path} on {self._device}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_path, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self._model = PeftModel.from_pretrained(base, str(self.adapter_path))
        self._model.to(self._device)
        self._model.eval()

    def generate_raw(self, prompt: str, *, system: str = SYSTEM_PROMPT) -> str:
        import torch

        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None and self._device

        if self.prompt_style == "chat":
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            if hasattr(self._tokenizer, "apply_chat_template"):
                text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = f"{system}\n\n{prompt}"
        else:
            text = prompt

        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        max_new = self.max_new_tokens
        if self.prompt_style == "sft":
            max_new = self._sft_token_cap or min(max_new, 256)
        elif self._chat_token_cap is not None:
            max_new = self._chat_token_cap

        gen_kwargs: dict = {
            "max_new_tokens": max_new,
            "do_sample": False,
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if self.prompt_style == "sft":
            gen_kwargs["stop_strings"] = _SFT_STOP_STRINGS
            gen_kwargs["tokenizer"] = self._tokenizer

        try:
            out = self._model.generate(**inputs, **gen_kwargs)
        except TypeError:
            gen_kwargs.pop("stop_strings", None)
            gen_kwargs.pop("tokenizer", None)
            out = self._model.generate(**inputs, **gen_kwargs)

        gen = out[0][inputs["input_ids"].shape[-1] :]
        decoded = self._tokenizer.decode(gen, skip_special_tokens=True)
        return _clean_sft_completion(decoded)

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
        # Phase 6: whole-file / agent repairs use chat (SFT truncates multi-hundred-line files).
        line_count = legacy_code.count("\n") + 1
        force_chat = (
            self.prompt_style == "chat"
            or bool(error_class)
            or bool(test_logs)
            or line_count >= 80
            or os.environ.get("REBIRTH_PEFT_FORCE_CHAT", "").strip() in {"1", "true", "yes"}
        )
        use_sft = self.prompt_style == "sft" and not force_chat
        if use_sft:
            prompt = build_sft_prompt(objective=objective, legacy_code=legacy_code)
            prev_cap = self._sft_token_cap
            self._sft_token_cap = _sft_max_new_tokens(legacy_code, self.max_new_tokens)
            try:
                raw = self.generate_raw(prompt)
            finally:
                self._sft_token_cap = prev_cap
        else:
            prompt = build_migration_prompt(
                objective=objective,
                legacy_code=legacy_code,
                file_path=file_path,
                language=language,
                retrieved_context=retrieved_context,
                error_class=error_class,
                test_logs=test_logs,
            )
            prev = self.prompt_style
            prev_chat_cap = self._chat_token_cap
            self.prompt_style = "chat"
            self._chat_token_cap = _chat_max_new_tokens(legacy_code, self.max_new_tokens)
            try:
                raw = self.generate_raw(prompt)
            finally:
                self.prompt_style = prev
                self._chat_token_cap = prev_chat_cap
        return GenerateResult(
            model=f"peft:{self.base_model}@{self.adapter_path.name}",
            raw_text=raw,
            code=extract_code_block(raw),
            prompt_chars=len(prompt),
        )
