"""String-soft + AST metrics (no model I/O)."""

from __future__ import annotations

import re
from pathlib import Path

from rebirth_ai.ast.chunker import CodeChunker


def soft_exact_match(a: str, b: str) -> bool:
    """Whitespace-insensitive equality (treats comma spacing as equivalent)."""
    return _squash(a) == _squash(b)


def _squash(code: str) -> str:
    text = code.strip()
    text = re.sub(r"\s+", "", text)
    return text


def token_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[^\s]", a.strip()))
    tb = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[^\s]", b.strip()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def ast_parse_ok(code: str, *, language: str = "javascript") -> bool:
    """True if Tree-sitter produces a tree without ERROR nodes."""
    try:
        return CodeChunker(language).parses(code)
    except Exception:
        return False


def public_symbol_set(code: str, *, language: str = "javascript") -> set[str]:
    """Names of top-level functions/classes/exports (API surface proxy)."""
    chunker = CodeChunker(language)
    path = Path("<eval>")
    chunks = chunker.chunk_source(code, file_path=str(path))
    names: set[str] = set()
    for ch in chunks:
        if ch.kind in {"function", "class", "export", "method"} and ch.name:
            if ch.name not in {"<anonymous>", "import", "export"}:
                names.add(ch.name)
    return names


def symbol_jaccard(pred: str, gold: str, *, language: str = "javascript") -> float:
    a = public_symbol_set(pred, language=language)
    b = public_symbol_set(gold, language=language)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
