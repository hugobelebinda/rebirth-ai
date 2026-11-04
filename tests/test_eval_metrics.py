"""Unit tests for soft/AST eval metrics (no model / Docker)."""

from __future__ import annotations

from rebirth_ai.eval.metrics import (
    ast_parse_ok,
    public_symbol_set,
    soft_exact_match,
    symbol_jaccard,
)


def test_soft_exact_ignores_comma_spacing() -> None:
    a = "[1,2,3].map((n) => n + 1);\n"
    b = "[1, 2, 3].map((n) => n + 1);\n"
    assert soft_exact_match(a, b)
    assert a.strip() != b.strip()


def test_ast_parse_ok_and_fail() -> None:
    assert ast_parse_ok("export function add(a, b) { return a + b; }\n")
    assert not ast_parse_ok("export function add(a, b) { return a + ;\n")


def test_symbol_jaccard_preserves_api() -> None:
    gold = "export function bad(n) {\n  return Number.isNaN(n);\n}\n"
    pred = "export function good(n) {\n  return Number.isNaN(n);\n}\n"
    # Renamed symbol → lower Jaccard (detects API drift)
    assert symbol_jaccard(pred, gold) < 1.0
    same = "export function bad(n) {\n  return Number.isNaN(n);\n}\n"
    assert symbol_jaccard(same, gold) == 1.0


def test_public_symbol_set_mathish() -> None:
    code = (
        "export function add(a, b) { return a + b; }\n"
        "export function multiply(a, b) { return a * b; }\n"
        "export class Calculator {}\n"
    )
    names = public_symbol_set(code)
    assert "add" in names
    assert "multiply" in names
    assert "Calculator" in names
