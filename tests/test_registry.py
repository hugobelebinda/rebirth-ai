"""Phase 8 adapter registry + detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from rebirth_ai.adapters.registry import (
    adapter_summaries,
    detect_adapter_id,
    get_adapter,
    get_adapter_for_project,
    language_label_for_file,
    list_adapter_ids,
)

ROOT = Path(__file__).resolve().parents[1]


def test_list_includes_js_ts_and_python() -> None:
    ids = list_adapter_ids()
    assert "js_ts" in ids
    assert "python" in ids


def test_get_adapter_aliases() -> None:
    assert get_adapter("js").name == "js_ts"
    assert get_adapter("py").name == "python"
    with pytest.raises(ValueError):
        get_adapter("cobol")


def test_detect_js_fixture() -> None:
    assert detect_adapter_id(ROOT / "fixtures" / "js_ts_sample") == "js_ts"


def test_detect_python_fixture() -> None:
    assert detect_adapter_id(ROOT / "fixtures" / "python_sample") == "python"


def test_get_adapter_for_project_override(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    assert get_adapter_for_project(tmp_path).name == "js_ts"
    assert get_adapter_for_project(tmp_path, lang="python").name == "python"


def test_language_label_for_file() -> None:
    assert language_label_for_file(Path("a.py")) == "python"
    assert language_label_for_file(Path("a.ts")) == "typescript"
    assert language_label_for_file(Path("a.js")) == "javascript"


def test_adapter_summaries_have_images() -> None:
    rows = adapter_summaries()
    by_id = {r["id"]: r for r in rows}
    assert "js-ts-sandbox" in str(by_id["js_ts"]["image"])
    assert "python-sandbox" in str(by_id["python"]["image"])
