"""Prompt templates for legacy → modern code migration."""

from __future__ import annotations

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


def extract_code_block(text: str) -> str:
    """Strip optional markdown fences from model output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # drop first fence line and last fence if present
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip() + "\n"
        return "\n".join(lines[1:]).strip() + "\n"
    return stripped if stripped.endswith("\n") else stripped + "\n"
