"""Phase 10: redaction, report view, run logs."""

from __future__ import annotations

import json
from pathlib import Path

from rebirth_ai.redact import redact_text
from rebirth_ai.report_view import load_report
from rebirth_ai.runs import write_run_log


def test_redact_tokens() -> None:
    raw = "Authorization: Bearer abcdefghijklmnop TOKEN=supersecret ghp_abcdefghijklmnopqrstuv"
    out = redact_text(raw)
    assert "supersecret" not in out
    assert "abcdefghijklmnop" not in out
    assert "***" in out


def test_write_run_log(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REBIRTH_RUNS_DIR", str(tmp_path))
    path = write_run_log(
        kind="migrate",
        payload={"status": "success", "test_logs": "password=hunter2 failed"},
        run_id="test_run",
    )
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_id"] == "test_run"
    assert "hunter2" not in path.read_text(encoding="utf-8")


def test_load_report_redacts(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(
        json.dumps(
            {
                "project_dir": "/x",
                "objective": "fix",
                "summary": {"success": 1, "failed": 0, "skipped": 0, "planned": 0, "total": 1},
                "results": [{"relative_file": "a.js", "status": "success", "iterations": 1}],
            }
        ),
        encoding="utf-8",
    )
    data = load_report(p)
    assert data["summary"]["success"] == 1
