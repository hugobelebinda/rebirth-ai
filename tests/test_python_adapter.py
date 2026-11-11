"""Phase 8 Python adapter + AST + test mapping."""

from __future__ import annotations

from pathlib import Path

from rebirth_ai.adapters.python import PythonAdapter
from rebirth_ai.ast.chunker import CodeChunker
from rebirth_ai.graph.discovery import discover_source_files, infer_test_path

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "python_sample"


def test_python_chunker_extracts_functions() -> None:
    src = (FIXTURE / "src" / "math_ops.py").read_text(encoding="utf-8")
    chunks = CodeChunker("python").chunk_source(src, file_path="math_ops.py")
    kinds = {c.kind for c in chunks}
    names = {c.name for c in chunks}
    assert "function" in kinds
    assert "class" in kinds
    assert "multiply" in names
    assert "Calculator" in names


def test_python_adapter_owns_py_not_tests() -> None:
    adapter = PythonAdapter()
    assert adapter.is_source_file(FIXTURE / "src" / "math_ops.py")
    assert not adapter.is_source_file(FIXTURE / "tests" / "test_math_ops.py")
    assert not adapter.is_source_file(Path("venv/lib/foo.py"))


def test_discover_and_infer_python_fixture() -> None:
    adapter = PythonAdapter()
    files = discover_source_files(FIXTURE, adapter=adapter, under="src")
    assert [p.as_posix() for p in files] == ["src/math_ops.py"]
    assert infer_test_path(FIXTURE, "src/math_ops.py") == "tests/test_math_ops.py"


def test_python_test_command() -> None:
    adapter = PythonAdapter()
    assert adapter.test_command_for(None) == "python -m pytest -q"
    assert adapter.test_command_for("tests/test_math_ops.py") == "python -m pytest -q tests/test_math_ops.py"
