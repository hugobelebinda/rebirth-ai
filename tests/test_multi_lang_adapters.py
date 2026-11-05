"""Phase 9/10 multi-language adapter smoke."""

from __future__ import annotations

from pathlib import Path

import pytest

from rebirth_ai.adapters.registry import (
    detect_adapter_id,
    get_adapter,
    list_adapter_ids,
)
from rebirth_ai.graph.discovery import discover_source_files, infer_test_path

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    ("go", "go_sample", "src", "src/mathops.go"),
    ("java", "java_sample", "src", "src/main/java/rebirth/math/MathOps.java"),
    ("csharp", "csharp_sample", "MathOps", "MathOps/Calculator.cs"),
    ("rust", "rust_sample", "src", "src/lib.rs"),
    ("cpp", "cpp_sample", "src", "src/math_ops.cpp"),
    ("swift", "swift_sample", "Sources", "Sources/MathOps/MathOps.swift"),
]


def test_all_eight_adapter_ids() -> None:
    ids = set(list_adapter_ids())
    assert ids == {"js_ts", "python", "go", "java", "csharp", "rust", "cpp", "swift"}


@pytest.mark.parametrize("lang,fixture,under,rel", CASES)
def test_detect_discover_parse(lang: str, fixture: str, under: str, rel: str) -> None:
    project = ROOT / "fixtures" / fixture
    assert detect_adapter_id(project) == lang
    adapter = get_adapter(lang)
    files = discover_source_files(project, adapter=adapter, under=under)
    assert Path(rel) in files
    chunks = adapter.parse(project / rel)
    assert chunks
    mapped = infer_test_path(project, rel)
    assert mapped is not None
