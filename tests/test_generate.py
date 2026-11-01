"""Unit tests for prompts + Ollama client (httpx mocked)."""

from __future__ import annotations

import httpx
import pytest

from rebirth_ai.generate.ollama_client import OllamaClient
from rebirth_ai.generate.prompts import build_migration_prompt, extract_code_block


def test_extract_code_block_strips_fences() -> None:
    raw = "```js\nconst x = 1;\n```"
    assert extract_code_block(raw) == "const x = 1;\n"


def test_build_migration_prompt_includes_context_and_error() -> None:
    prompt = build_migration_prompt(
        objective="modernize",
        legacy_code="var x = 1;",
        file_path="src/a.js",
        retrieved_context=["export function helper() {}"],
        error_class="TestLogicFailure",
        test_logs="AssertionError expected 2",
    )
    assert "modernize" in prompt
    assert "helper" in prompt
    assert "TestLogicFailure" in prompt
    assert "AssertionError" in prompt


def test_ollama_migrate_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        return httpx.Response(200, json={"response": "```js\nconst y = 2;\n```"})

    transport = httpx.MockTransport(handler)
    client = OllamaClient(host="http://test", model="dummy")

    real_client = httpx.Client

    def fake_client(*args, **kwargs):  # noqa: ANN002, ANN003
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", fake_client)
    result = client.migrate(
        objective="use const",
        legacy_code="var y = 2;",
        file_path="src/b.js",
    )
    assert result.code.strip() == "const y = 2;"
    assert result.model == "dummy"
