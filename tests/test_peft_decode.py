"""Unit tests for PEFT SFT stop/decode cleanup (no model load)."""

from __future__ import annotations

from rebirth_ai.generate.peft_client import (
    _chat_max_new_tokens,
    _clean_sft_completion,
    _sft_max_new_tokens,
)


def test_clean_strips_sft_restart() -> None:
    raw = "function f(x) {\n  return x;\n}\n### Legacy\nvar x = 1;\n"
    assert _clean_sft_completion(raw) == "function f(x) {\n  return x;\n}\n"


def test_clean_strips_changelog_and_describe() -> None:
    raw = (
        "function pair(a, b) {\n  return { a, b };\n}\n"
        "## [1.0.0] - 2015\n\n### Features\n"
    )
    assert "Features" not in _clean_sft_completion(raw)
    assert "return { a, b }" in _clean_sft_completion(raw)

    raw2 = "function f() {}\n\ndescribe('f', () => {});\n"
    assert "describe" not in _clean_sft_completion(raw2)


def test_clean_strips_fence_wrapper() -> None:
    raw = "```js\nconst x = 1;\n```\n"
    assert _clean_sft_completion(raw) == "const x = 1;\n"


def test_clean_keeps_legitimate_exports() -> None:
    raw = "export function on(flag) {\n  return Boolean(flag);\n}\n"
    assert _clean_sft_completion(raw) == raw


def test_sft_token_cap_bounds_short_idioms() -> None:
    # Short legacy still gets room for additive rewrites (was 64 → truncation).
    assert _sft_max_new_tokens("var x = 1;", 512) == 256
    assert _sft_max_new_tokens("var x = 1;", 100) == 100
    long = "x" * 2000
    assert _sft_max_new_tokens(long, 1024) == 512
    assert _sft_max_new_tokens(long, 300) == 300


def test_chat_token_cap_scales_with_file_size() -> None:
    short = "const x = 1;\n"
    long = "\n".join(f"export function f{i}() {{ return {i}; }}" for i in range(200))
    assert _chat_max_new_tokens(short, 1024) >= 256
    assert _chat_max_new_tokens(long, 4096) > _chat_max_new_tokens(short, 4096)
    assert _chat_max_new_tokens(long, 4096) <= 4096


def test_clean_strips_trailing_closing_fence() -> None:
    raw = '# header\n__version__ = "1.26.0.dev0"\n```\n'
    assert _clean_sft_completion(raw) == '# header\n__version__ = "1.26.0.dev0"\n'


def test_clean_cuts_leaked_pretraining_tokens() -> None:
    raw = "major = 0\npatch = 18\n<|repo_name|>python']\n```\n"
    assert _clean_sft_completion(raw) == "major = 0\npatch = 18\n"
    raw2 = "x = 1\n<|file_sep|>other.py\ny = 2\n"
    assert _clean_sft_completion(raw2) == "x = 1\n"


def test_clean_keeps_fenced_block_and_inline_backticks() -> None:
    assert _clean_sft_completion("```python\nx = 1\n```\n") == "x = 1\n"
    raw = 'doc = "use ``` for code"\n'
    assert _clean_sft_completion(raw) == raw
