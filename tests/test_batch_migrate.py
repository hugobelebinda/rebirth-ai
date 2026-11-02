"""Unit tests for Phase 5 discovery / PR body (no Docker / Ollama)."""

from __future__ import annotations

from pathlib import Path

from rebirth_ai.graph.batch import (
    BatchReport,
    FileMigrationResult,
    discover_source_files,
    infer_test_path,
    render_pr_body,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "js_ts_sample"


def test_discover_fixture_sources() -> None:
    files = discover_source_files(FIXTURE, under="src")
    names = {p.as_posix() for p in files}
    assert "src/math.js" in names
    assert "src/format.js" in names
    assert "src/greeter.ts" in names


def test_infer_test_path_math_and_format() -> None:
    assert infer_test_path(FIXTURE, "src/math.js") == "test/math.test.js"
    assert infer_test_path(FIXTURE, "src/format.js") == "test/format.test.js"
    assert infer_test_path(FIXTURE, "src/greeter.ts") is None


def test_pr_body_lists_failures_without_aborting_summary() -> None:
    report = BatchReport(
        project_dir=str(FIXTURE),
        objective="demo",
        results=[
            FileMigrationResult(
                relative_file="src/math.js",
                relative_test="test/math.test.js",
                status="success",
                iterations=1,
                wrote=True,
            ),
            FileMigrationResult(
                relative_file="src/format.js",
                relative_test="test/format.test.js",
                status="human_review",
                error_class="TestLogicFailure",
            ),
            FileMigrationResult(
                relative_file="src/greeter.ts",
                relative_test=None,
                status="skipped_no_test",
            ),
        ],
    )
    body = render_pr_body(report)
    assert "src/math.js" in body
    assert "src/format.js" in body
    assert "TestLogicFailure" in body
    assert "Success: **1**" in body
    assert report.to_dict()["summary"]["failed"] == 1
