"""The ``telemetry`` block of a run log: one flat, versioned record per migrated file.

``rebirth migrate`` / ``migrate --to`` / ``migrate-repo`` already log the graph trace; this adds
what monitoring needs in a stable shape — task, language / direction, model served, outcome,
iterations, error class per attempt, wall time and an input profile (size, imports) for drift.
Old run logs have no telemetry block: :func:`from_payload` derives the same record from the
payload + trace, so the dashboard covers them too (without wall time / input profile).
"""

from __future__ import annotations

import re
from typing import Any

from rebirth_ai.monitor.imports import extract_imports, family

TELEMETRY_VERSION = 1

_MODEL = re.compile(r"^(?:generate_code|translate_code): iteration=(\d+) model=(\S+)")
_CLASS = re.compile(r"^classify_error: (\w+)")
_TEST = re.compile(r"^(?:execute_tests|verify_target): (?:\w+ )?(PASS|FAIL)\b")


def approx_tokens(text: str) -> int:
    return round(len(text) / 3.5)  # same rule of thumb as the translator's token budget


def parse_trace(trace: list[str] | None) -> dict[str, Any]:
    """Models used, error classes in order, and test verdicts from a graph trace."""
    models: list[str] = []
    classes: list[str] = []
    tests: list[str] = []
    generations = 0
    for line in trace or []:
        if m := _MODEL.match(line):
            generations += 1
            if m.group(2) not in models:
                models.append(m.group(2))
        elif m := _CLASS.match(line):
            classes.append(m.group(1))
        elif m := _TEST.match(line):
            tests.append(m.group(1))
    return {"models": models, "error_classes": classes, "tests": tests, "generations": generations}


def input_profile(code: str | None, lang: str | None) -> dict[str, Any] | None:
    if code is None:
        return None
    return {
        "chars": len(code),
        "lines": code.count("\n") + (1 if code and not code.endswith("\n") else 0),
        "approx_tokens": approx_tokens(code),
        "imports": extract_imports(code, lang),
    }


def build(
    *,
    task: str,
    source_lang: str | None,
    target_lang: str | None,
    backend: str | None,
    model: str | None,
    status: str | None,
    iterations: int | None,
    trace: list[str] | None,
    wall_seconds: float | None = None,
    source_code: str | None = None,
    draft: str | None = None,
    verify_mode: str | None = None,
) -> dict[str, Any]:
    """``task`` = ``modernize`` (same language) or ``translate`` (``--to``)."""
    parsed = parse_trace(trace)
    src = family(source_lang) or source_lang
    tgt = (target_lang or source_lang) if task == "translate" else src
    return {
        "v": TELEMETRY_VERSION,
        "task": task,
        "lang": src,
        "source_lang": source_lang,
        "target_lang": tgt,
        "direction": f"{source_lang}->{tgt}" if task == "translate" else src,
        "backend": backend,
        "model": model or (parsed["models"][-1] if parsed["models"] else None),
        "status": status,
        "success": status == "success",
        "iterations": iterations,
        "generations": parsed["generations"],
        "error_classes": parsed["error_classes"],
        "tests": parsed["tests"],
        "verify_mode": verify_mode,
        "wall_seconds": None if wall_seconds is None else round(wall_seconds, 2),
        "input": input_profile(source_code, source_lang),
        "output_chars": None if draft is None else len(draft),
    }


def from_payload(kind: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Telemetry records of one run log: its own block if present, else derived (one record per
    file; ``migrate-repo`` logs hold several files)."""
    if isinstance(payload.get("telemetry"), dict):
        return [payload["telemetry"]]
    if kind == "migrate-repo":
        if isinstance(payload.get("telemetry_files"), list):
            return list(payload["telemetry_files"])
        return [
            build(task="modernize", source_lang=payload.get("lang"), target_lang=None,
                  backend=payload.get("backend"), model=None, status=r.get("status"),
                  iterations=r.get("iterations"), trace=None)
            | {"error_classes": [r["error_class"]] if r.get("error_class") else []}
            for r in payload.get("results") or []
        ]
    if kind == "translate":
        return [build(task="translate", source_lang=payload.get("source_lang"), target_lang=payload.get("target_lang"),
                      backend=payload.get("backend"), model=payload.get("model"), status=payload.get("status"),
                      iterations=payload.get("iterations"), trace=payload.get("trace"),
                      verify_mode=payload.get("verify_mode"))]
    if kind == "migrate":
        return [build(task="modernize", source_lang=payload.get("lang") or payload.get("language"), target_lang=None,
                      backend=payload.get("backend"), model=None, status=payload.get("status"),
                      iterations=payload.get("iterations"), trace=payload.get("trace"))]
    return []
