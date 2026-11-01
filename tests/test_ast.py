"""AST chunker tests against the JS/TS fixture."""

from __future__ import annotations

from pathlib import Path

from rebirth_ai.adapters.js_ts import JsTsAdapter
from rebirth_ai.ast.chunker import CodeChunker

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "js_ts_sample"


def test_chunk_math_js_has_imports_functions_class() -> None:
    path = FIXTURE / "src" / "math.js"
    chunks = CodeChunker("javascript").chunk_file(path)
    kinds = {c.kind for c in chunks}
    names = {c.name for c in chunks}

    assert "import" in kinds
    assert "function" in kinds
    assert "class" in kinds
    assert "./format.js" in names or any("format" in n for n in names)
    assert "add" in names
    assert "Calculator" in names
    assert any(c.kind == "method" for c in chunks)


def test_chunk_format_js() -> None:
    path = FIXTURE / "src" / "format.js"
    chunks = CodeChunker("javascript").chunk_file(path)
    names = {c.name for c in chunks}
    assert "formatResult" in names
    assert "formatList" in names


def test_chunk_greeter_ts() -> None:
    path = FIXTURE / "src" / "greeter.ts"
    chunks = CodeChunker("typescript").chunk_file(path)
    names = {c.name for c in chunks}
    kinds = {c.kind for c in chunks}
    assert "greet" in names
    assert "Greeter" in names
    assert "function" in kinds
    assert "class" in kinds


def test_adapter_parses_three_fixture_files() -> None:
    adapter = JsTsAdapter()
    files = [
        FIXTURE / "src" / "math.js",
        FIXTURE / "src" / "format.js",
        FIXTURE / "src" / "greeter.ts",
    ]
    for path in files:
        chunks = adapter.parse(path)
        assert len(chunks) >= 1, f"expected chunks for {path}"
        assert all(c.file_path.endswith(path.name) or path.name in c.file_path for c in chunks)
