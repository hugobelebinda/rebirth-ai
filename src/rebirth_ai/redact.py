"""Secret / token redaction for reports and run logs (Phase 10)."""

from __future__ import annotations

import re
from typing import Any

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)authorization\s*[=:]\s*bearer\s+\S+"), "Authorization=Bearer ***"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*"), "Bearer ***"),
    (re.compile(
        r"(?i)(api[_-]?key|token|secret|password|passwd)\s*[=:]\s*['\"]?([^\s'\"]+)"
    ), r"\1=***"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "ghp_***"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "github_pat_***"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "sk-***"),
    (re.compile(r"sk_(?:test|live)_[A-Za-z0-9]+"), "sk_***"),
    (re.compile(r"pk_(?:test|live)_[A-Za-z0-9]+"), "pk_***"),
    (re.compile(r"shippo_(?:test|live)_[A-Za-z0-9]+"), "shippo_***"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]+"), "xox***"),
    (re.compile(r"\b[MN][A-Za-z0-9_-]{23,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{20,}\b"), "discord_token_***"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA***"),
    (re.compile(
        r"(?i)-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
    ), "-----BEGIN PRIVATE KEY-----\n***\n-----END PRIVATE KEY-----"),
]


def redact_text(text: str | None) -> str:
    if not text:
        return ""
    out = text
    for pattern, repl in _PATTERNS:
        out = pattern.sub(repl, out)
    return out


def redact_obj(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_obj(v) for v in value]
    if isinstance(value, dict):
        return {k: redact_obj(v) for k, v in value.items()}
    return value
