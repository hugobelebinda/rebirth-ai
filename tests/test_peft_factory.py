"""Generator factory + SFT prompt helpers (no model download)."""

from __future__ import annotations

from rebirth_ai.generate.factory import get_generator
from rebirth_ai.generate.ollama_client import OllamaClient
from rebirth_ai.generate.prompts import build_sft_prompt


def test_get_generator_ollama_default(monkeypatch) -> None:
    monkeypatch.delenv("REBIRTH_GENERATOR", raising=False)
    assert isinstance(get_generator(), OllamaClient)
    assert isinstance(get_generator("ollama"), OllamaClient)


def test_get_generator_peft(monkeypatch) -> None:
    monkeypatch.setenv("REBIRTH_GENERATOR", "peft")
    from rebirth_ai.generate.peft_client import PeftClient

    assert isinstance(get_generator(), PeftClient)


def test_resolve_peft_paths_prefers_7b(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("REBIRTH_PEFT_BASE", raising=False)
    monkeypatch.delenv("REBIRTH_PEFT_ADAPTER", raising=False)
    from rebirth_ai.generate import peft_client as pc

    adapter_7b = tmp_path / "lora_js_ts_7b" / "adapter"
    adapter_7b.mkdir(parents=True)
    monkeypatch.setattr(pc, "DEFAULT_ADAPTER_7B", adapter_7b)
    monkeypatch.setattr(pc, "DEFAULT_ADAPTER_05B", tmp_path / "missing_05")
    base, path = pc.resolve_peft_paths()
    assert base == pc.DEFAULT_BASE_7B
    assert path == adapter_7b


def test_resolve_peft_paths_falls_back_to_05b(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("REBIRTH_PEFT_BASE", raising=False)
    monkeypatch.delenv("REBIRTH_PEFT_ADAPTER", raising=False)
    from rebirth_ai.generate import peft_client as pc

    adapter_05 = tmp_path / "lora_js_ts" / "adapter"
    adapter_05.mkdir(parents=True)
    monkeypatch.setattr(pc, "DEFAULT_ADAPTER_7B", tmp_path / "missing_7b")
    monkeypatch.setattr(pc, "DEFAULT_ADAPTER_05B", adapter_05)
    base, path = pc.resolve_peft_paths()
    assert base == pc.DEFAULT_BASE_05B
    assert path == adapter_05


def test_build_sft_prompt_matches_train_format() -> None:
    prompt = build_sft_prompt(objective="use const", legacy_code="var x = 1;\n")
    assert prompt.startswith("### Objective\n")
    assert "### Legacy\n" in prompt
    assert prompt.endswith("### Modern\n")
