"""Prompt templates for legacy → modern code migration."""

from __future__ import annotations

import re

SYSTEM_PROMPT = """You are Rebirth AI, a precise code migration engine.
Rewrite legacy source into modern idioms for the target stack.
Rules:
- Output ONLY the full updated source file contents.
- Do not wrap the answer in markdown fences unless the original used them.
- Preserve public behavior and exports unless the objective says otherwise.
- Prefer the project's retrieved helpers/utilities when relevant.
- If fixing a test failure, address the error class and log excerpt.
"""


def build_migration_prompt(
    *,
    objective: str,
    legacy_code: str,
    file_path: str,
    language: str = "javascript",
    retrieved_context: list[str] | None = None,
    error_class: str | None = None,
    test_logs: str | None = None,
) -> str:
    parts: list[str] = [
        f"Migration objective: {objective}",
        f"File: {file_path}",
        f"Language: {language}",
    ]
    if retrieved_context:
        joined = "\n\n---\n\n".join(retrieved_context[:8])
        parts.append("Retrieved project context (interfaces/utilities):\n" + joined)
    if error_class:
        parts.append(f"Previous attempt failed with error class: {error_class}")
    if test_logs:
        # Keep logs bounded for local context windows
        clipped = test_logs if len(test_logs) <= 4000 else test_logs[:4000] + "\n...[truncated]"
        parts.append("Test / compiler logs:\n" + clipped)
    parts.append("Legacy source to rewrite:\n" + legacy_code)
    parts.append("Modernized source:")
    return "\n\n".join(parts)


def build_sft_prompt(*, objective: str, legacy_code: str) -> str:
    """Match ``training/train_lora.py`` example format for PEFT adapters."""
    return (
        "### Objective\n"
        f"{objective}\n\n"
        "### Legacy\n"
        f"{legacy_code}\n"
        "### Modern\n"
    )


# A chat reply that opens with a short prose sentence, then a fenced block with a language tag:
# "Here is the modernized code:\n\n```go\n…\n```\n…". Group 1 = prose, group 2 = code.
_PREAMBLE_FENCE_RE = re.compile(
    r"\A(?P<prose>[^\n]*(?:\n[^\n]*){0,2}?)\n+```[A-Za-z0-9_+#.-]+[ \t]*\n(?P<code>.*?)\n```",
    re.DOTALL,
)
_CODE_HINTS = re.compile(r"[{};=]|^\s*(def|class|import|from|func|fn|package|using|#include|let|var|const)\b", re.MULTILINE)


def strip_prose_preamble(text: str) -> str | None:
    """Code of the first tagged fenced block if ``text`` starts with ≤3 lines of prose, else None.

    Prose = the lines before the fence carry no code hints and the last one is a sentence
    ("…:" / "….") — so SFT completions (code, then a leaked fence) are never touched.
    """
    m = _PREAMBLE_FENCE_RE.match(text.strip())
    if not m:
        return None
    prose = m.group("prose").strip()
    if not prose or _CODE_HINTS.search(prose) or not prose.rstrip().endswith((":", ".", "!")):
        return None
    return m.group("code").rstrip() + "\n"


_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Drop a reasoning model's ``<think>…</think>`` block (it can hold code that is not the answer).
    An unterminated block (the token budget ran out while thinking) holds no final answer: ``""``."""
    text = _THINK.sub("", text)
    return "" if "<think>" in text else text


def extract_code_block(text: str) -> str:
    """Strip optional markdown fences (and a short prose preamble) from model output."""
    pre = strip_prose_preamble(text)
    if pre is not None:
        return pre
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # drop first fence line and last fence if present
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip() + "\n"
        return "\n".join(lines[1:]).strip() + "\n"
    return stripped if stripped.endswith("\n") else stripped + "\n"
