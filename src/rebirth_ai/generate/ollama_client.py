"""Thin Ollama HTTP client for local code generation."""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from rebirth_ai.generate.prompts import SYSTEM_PROMPT, build_migration_prompt, extract_code_block


@dataclass(frozen=True)
class GenerateResult:
    model: str
    raw_text: str
    code: str
    prompt_chars: int

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "raw_text": self.raw_text,
            "code": self.code,
            "prompt_chars": self.prompt_chars,
        }


class OllamaClient:
    def __init__(
        self,
        *,
        host: str | None = None,
        model: str | None = None,
        timeout_sec: float = 300.0,
    ) -> None:
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip(
            "/"
        )
        self.model = model or os.environ.get(
            "OLLAMA_MODEL", "qwen2.5-coder:14b-instruct-q6_K"
        )
        self.timeout_sec = timeout_sec

    def list_models(self) -> list[str]:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{self.host}/api/tags")
            resp.raise_for_status()
            return [m.get("name", "") for m in resp.json().get("models", [])]

    def generate_raw(
        self,
        prompt: str,
        *,
        system: str = SYSTEM_PROMPT,
        num_predict: int = 4096,
        temperature: float = 0.1,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        with httpx.Client(timeout=self.timeout_sec) as client:
            resp = client.post(f"{self.host}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return str(data.get("response", ""))

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
        prompt = build_migration_prompt(
            objective=objective,
            legacy_code=legacy_code,
            file_path=file_path,
            language=language,
            retrieved_context=retrieved_context,
            error_class=error_class,
            test_logs=test_logs,
        )
        # Phase 6: scale decode budget with file size (floor 1k, ceiling 8k)
        lines = legacy_code.count("\n") + 1
        num_predict = max(1024, min(8192, max(lines * 16, len(legacy_code) // 2, 2048)))
        raw = self.generate_raw(prompt, num_predict=num_predict)
        return GenerateResult(
            model=self.model,
            raw_text=raw,
            code=extract_code_block(raw),
            prompt_chars=len(prompt),
        )
