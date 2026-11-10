"""Phase 18: telemetry, dashboard, regression alerts, drift, flywheel — all on synthetic data."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

from rebirth_ai.monitor import alerts, drift, flywheel, telemetry
from rebirth_ai.monitor.dashboard import render_html, summarize, wilson
from rebirth_ai.monitor.imports import extract_imports
from rebirth_ai.monitor.store import Record, load_records
from rebirth_ai.runs import write_run_log

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Privacy: nothing in Phase 18 may open a connection — any attempt fails the test."""
    def deny(*_a, **_k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


@pytest.mark.parametrize("lang,code,want", [
    ("python", "import os.path, sys\nfrom collections import Counter\nfrom . import local\n", ["collections", "os", "sys"]),
    ("typescript", "import x from 'react';\nimport {a} from \"@scope/pkg/sub\";\nconst f = require('./local');\nimport('node:fs')\n",
     ["@scope/pkg", "fs", "react"]),
    ("go", 'import (\n\t"fmt"\n\tlog "github.com/sirupsen/logrus"\n)\nimport "os"\n', ["fmt", "github.com/sirupsen/logrus", "os"]),
    ("java", "import java.util.List;\nimport static org.junit.Assert.assertEquals;\nimport java.io.*;\n", ["java.io", "java.util", "org.junit"]),
    ("csharp", "using System;\nusing System.Linq;\nusing Alias = Foo.Bar;\n", ["Foo.Bar", "System", "System.Linq"]),
    ("rust", "use std::io::{self, Read};\nuse serde::Deserialize;\nuse crate::local;\nextern crate libc;\n", ["libc", "serde", "std::io"]),
    ("cpp", '#include <vector>\n#include "local.h"\n# include <bits/stdc++.h>\n', ["bits/stdc++.h", "vector"]),
    ("swift", "import Foundation\n@testable import MyApp\nimport struct Darwin.size_t\n", ["Darwin", "Foundation", "MyApp"]),
])
def test_extract_imports(lang, code, want) -> None:
    assert extract_imports(code, lang) == want


def test_parse_trace_both_graphs() -> None:
    same = ["retrieve_context: 4 chunk(s)", "execute_tests: FAIL exit=1 in 0.36s",
            "classify_error: TestLogicFailure conf=0.30 (Unknown/low-confidence telemetry)",
            "generate_code: iteration=1 model=ollama:rebirth-python:7b chars=721", "execute_tests: PASS exit=0 in 0.26s"]
    xl = ["translate_code: iteration=1 model=ollama:rebirth-xlang:14b-v1 chars=602", "verify_target: target_tests FAIL exit=1 in 2.04s",
          "classify_error: CompileError (target compiler output)", "translate_code: iteration=2 model=ollama:rebirth-xlang:14b-v1 chars=623",
          "verify_target: target_tests PASS exit=0 in 2.03s"]
    a, b = telemetry.parse_trace(same), telemetry.parse_trace(xl)
    assert a == {"models": ["ollama:rebirth-python:7b"], "error_classes": ["TestLogicFailure"], "tests": ["FAIL", "PASS"], "generations": 1}
    assert b["models"] == ["ollama:rebirth-xlang:14b-v1"] and b["error_classes"] == ["CompileError"] and b["generations"] == 2


def test_telemetry_build_and_old_payloads() -> None:
    t = telemetry.build(task="translate", source_lang="python", target_lang="go", backend="auto", model=None,
                        status="success", iterations=2, trace=["translate_code: iteration=1 model=m1 chars=3"],
                        wall_seconds=12.345, source_code="import os\nprint(1)\n", draft="package main\n")
    assert t["direction"] == "python->go" and t["model"] == "m1" and t["success"] and t["wall_seconds"] == 12.35
    assert t["input"] == {"chars": 19, "lines": 2, "approx_tokens": 5, "imports": ["os"]}
    old = telemetry.from_payload("migrate", {"lang": "js_ts", "backend": "peft", "status": "failed", "iterations": 3,
                                             "trace": ["classify_error: SyntaxError"]})
    assert old[0]["direction"] == "js_ts" and old[0]["error_classes"] == ["SyntaxError"] and old[0]["input"] is None
    repo = telemetry.from_payload("migrate-repo", {"lang": "go", "results": [{"status": "success", "iterations": 1},
                                                                              {"status": "failed", "iterations": 5, "error_class": "Timeout"}]})
    assert [r["success"] for r in repo] == [True, False] and repo[1]["error_classes"] == ["Timeout"]


def _rec(i: int, *, key: str = "python->go", model: str = "m1", ok: bool = True, iters: int = 1, task: str = "translate",
         inp: dict | None = None) -> Record:
    src, _, tgt = key.partition("->")
    t = telemetry.build(task=task, source_lang=src, target_lang=tgt or None, backend="auto", model=model,
                        status="success" if ok else "failed", iterations=iters, trace=[])
    t["input"] = inp
    return Record(run_id=f"r{i:04d}", ts=f"2026-11-10T{i // 3600:02d}:{i // 60 % 60:02d}:{i % 60:02d}", kind=task, t=t)


def test_summary_and_html_are_offline() -> None:
    recs = [_rec(i, ok=i % 4 != 0) for i in range(8)] + [_rec(100 + i, key="go", task="modernize", model="m7") for i in range(3)]
    s = summarize(recs)
    assert s["by_key"]["translate:python->go"]["success_rate"] == 0.75
    assert s["by_key"]["modernize:go"]["runs"] == 3 and s["by_task"]["modernize"]["success_rate"] == 1.0
    page = render_html(s, [{"level": "warn", "message": "x <script>"}])
    assert "http" not in page and "<script>" not in page and "&lt;script&gt;" in page  # no external assets, escaped
    lo, hi = wilson(0, 5)
    assert lo == 0.0 and 0.3 < hi < 0.5


def test_regression_alerts() -> None:
    old = [_rec(i, model="v1", ok=i % 10 != 0) for i in range(20)]           # 90 %
    worse = [_rec(100 + i, model="v2", ok=i % 2 == 0) for i in range(20)]    # 50 %
    found = alerts.check(old + worse, min_runs=10, margin=10)
    assert any(f["level"] == "warn" and f["kind"] == "success_drop" and f["scope"] == "translate:python->go" for f in found)
    slight = [_rec(100 + i, model="v2", ok=i % 5 != 0) for i in range(12)]   # 80 %: −10 points, overlapping CIs
    lv = {f["level"] for f in alerts.check(old + slight, min_runs=10, margin=10) if f["kind"] == "success_drop"}
    assert lv == {"info"}
    few = alerts.check(old + worse[:3], min_runs=10)
    assert {f["kind"] for f in few} == {"too_few_runs"}
    same = [_rec(100 + i, model="v2", ok=i % 10 != 0) for i in range(20)]
    assert alerts.check(old + same, min_runs=10) == []


def test_drift_checks() -> None:
    ref = drift.profile(["import os\n" + "x = 1\n" * 40] * 200, "python")
    normal = [{"chars": 250, "imports": ["os"]}] * 30
    assert drift.check_lang("python", normal, ref) == []
    long = [{"chars": 9000, "imports": ["os"]}] * 30
    kinds = {f["kind"]: f["level"] for f in drift.check_lang("python", long, ref)}
    assert kinds == {"length_psi": "warn", "long_inputs": "warn"}
    new_libs = [{"chars": 250, "imports": ["torch", "numpy"]}] * 30
    f = drift.check_lang("python", new_libs, ref)
    assert f[0]["kind"] == "unseen_imports" and f[0]["top_unseen"][0][0] in {"torch", "numpy"}
    assert drift.check_lang("python", normal[:5], ref)[0]["kind"] == "too_few_runs"


def test_drift_check_reads_reference_files(tmp_path) -> None:
    (tmp_path / "python.json").write_text(json.dumps(drift.profile(["import os\nx=1\n"] * 50, "python")))
    recs = [_rec(i, key="python->go", inp={"chars": 12, "imports": ["pandas"]}) for i in range(25)]
    recs += [_rec(100 + i, key="go->rust", inp={"chars": 12, "imports": []}) for i in range(2)]
    found = drift.check(recs, ref_dir=tmp_path)
    assert {(f["lang"], f["kind"]) for f in found} == {("python", "unseen_imports"), ("go", "no_reference")}


def _write_run(runs: Path, *, task: str, status: str, iters: int, source: str, draft: str) -> str:
    from rebirth_ai.runs import new_run_id

    rid = new_run_id()
    lang = "python"
    arts = flywheel.save_artifacts(runs / rid, source_name="src/a.py", source=source,
                                   draft_name="a.go" if task == "translate" else "src/a.py", draft=draft)
    t = telemetry.build(task=task, source_lang=lang, target_lang="go" if task == "translate" else None,
                        backend="auto", model="m", status=status, iterations=iters, trace=[], source_code=source, draft=draft)
    kind = "translate" if task == "translate" else "migrate"
    write_run_log(kind=kind, run_id=rid, payload={"file": "src/a.py", "status": status, "objective": "Use f-strings",
                                                  "artifacts": arts, "telemetry": t})
    return rid


def test_flywheel_review_to_training_mix(tmp_path, monkeypatch) -> None:
    runs, fly = tmp_path / "runs", tmp_path / "flywheel"
    monkeypatch.setenv("REBIRTH_RUNS_DIR", str(runs))
    secret = "AKIA" + "ABCDEFGHIJKLMNOP"
    bad = _write_run(runs, task="modernize", status="failed", iters=3, source=f"KEY = '{secret}'\nprint('%s' % x)\n",
                     draft="print(f'{x}')\n")
    hard = _write_run(runs, task="translate", status="success", iters=4, source="print(input())\n",
                      draft="package main\nfunc main() {}\n")
    _write_run(runs, task="modernize", status="success", iters=1, source="ok\n", draft="ok\n")   # easy → not queued
    assert [r.run_id for r in flywheel.queue(runs)] == [bad, hard]

    arts = flywheel.artifacts(bad, runs)
    assert secret not in arts["source"] and "AKIA***" in arts["source"]                       # redacted on disk
    with pytest.raises(PermissionError):
        flywheel.accept(bad, own=False, root=runs, out_dir=fly)
    row = flywheel.accept(bad, own=True, fixed="print(f'{x!r}')\n", root=runs, out_dir=fly)
    assert row["fixed_by_user"] and row["modern"] == "print(f'{x!r}')\n" and row["license"] == "own"
    assert row["objective"] == "Use f-strings" and row["split"] in {"train", "eval"}
    with pytest.raises(ValueError):
        flywheel.accept(bad, own=True, root=runs, out_dir=fly)
    xrow = flywheel.accept(hard, own=True, root=runs, out_dir=fly)
    assert xrow["source_lang"] == "python" and xrow["target_lang"] == "go"
    assert flywheel.queue(runs) == []

    # force one train and one eval row to check both paths of the mix + rolling eval set
    rows = flywheel.rows(fly)["python"]
    rows = [rows[0] | {"split": "train"}, rows[0] | {"split": "eval", "legacy": "eval only\n"}]
    (fly / "python.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert flywheel.export_rolling_eval(fly) == {"python": 1}
    assert (fly / "rolling_eval" / "python.jsonl").read_text().count("\n") == 1

    sys.path.insert(0, str(ROOT / "training"))
    import build_flywheel_mix as mix

    base = tmp_path / "base.jsonl"
    base.write_text(json.dumps({"objective": "o", "language": "python", "legacy": "a\n", "modern": "b\n"}) + "\n")
    evalsets = tmp_path / "evalsets"
    evalsets.mkdir()
    stats = mix.build("python", base, tmp_path / "out.jsonl", fly, evalsets, repeat=2)
    assert stats["added"] == 1 and stats["total"] == 3                    # eval-split row left out
    (evalsets / "python.jsonl").write_text(json.dumps({"legacy": rows[0]["legacy"]}) + "\n")
    assert mix.build("python", base, tmp_path / "out2.jsonl", fly, evalsets, repeat=2)["added"] == 0  # frozen-set leak guard
    (fly / "xlang.jsonl").write_text(json.dumps(xrow | {"split": "train"}) + "\n")
    xs = mix.build("xlang", base, tmp_path / "x.jsonl", fly, evalsets, repeat=1)
    out = [json.loads(line) for line in (tmp_path / "x.jsonl").read_text().splitlines()]
    assert xs["added"] == 1 and out[-1]["prompt"].startswith("### Source language\npython\n### Target language\ngo")


def test_load_records_mixed_formats(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("REBIRTH_RUNS_DIR", str(tmp_path))
    write_run_log(kind="migrate", payload={"lang": "go", "status": "success", "iterations": 1, "trace": []})
    write_run_log(kind="migrate-repo", payload={"lang": "go", "results": [{"status": "failed", "iterations": 2}]})
    write_run_log(kind="eval", payload={"x": 1})
    recs = load_records(tmp_path)
    assert [r.kind for r in recs] == ["migrate", "migrate-repo"] and recs[1].t["success"] is False


def test_cli_dashboard_and_check(tmp_path) -> None:
    from typer.testing import CliRunner

    from rebirth_ai.cli import app

    runs = tmp_path / "runs"
    old = [_rec(i, model="v1", ok=True) for i in range(12)] + [_rec(100 + i, model="v2", ok=i % 2 == 0) for i in range(12)]
    for r in old:
        d = runs / r.run_id
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({"run_id": r.run_id, "ts": r.ts, "kind": "translate", "payload": {"telemetry": r.t}}))
    out = tmp_path / "dash.html"
    res = CliRunner().invoke(app, ["dashboard", "--runs-dir", str(runs), "--out", str(out), "--ref-dir", str(tmp_path / "none")])
    assert res.exit_code == 0, res.output
    page = out.read_text()
    assert "translate:python-&gt;go" in page and "Warning:" in page and "v2" in page
    res = CliRunner().invoke(app, ["monitor", "check", "--runs-dir", str(runs), "--ref-dir", str(tmp_path / "none")])
    assert res.exit_code == 1 and "warning" in res.output


class _Graph:
    def __init__(self, final: dict) -> None:
        self.final = final

    def invoke(self, state: dict) -> dict:
        return {**state, **self.final}


def test_migrate_commands_write_telemetry_and_artifacts(tmp_path, monkeypatch) -> None:
    """`rebirth migrate` and `migrate --to` (graphs stubbed: no model, no sandbox) log the Phase 18
    telemetry block + redacted input / draft copies the review queue needs."""
    from typer.testing import CliRunner

    import rebirth_ai.graph.app as graph_app
    import rebirth_ai.xlang.graph as xgraph
    from rebirth_ai.cli import app

    runs = tmp_path / "runs"
    monkeypatch.setenv("REBIRTH_RUNS_DIR", str(runs))
    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)
    (proj / "src" / "a.py").write_text("import os\nTOKEN = 'secret=abc123'\nprint('%s' % os.name)\n")
    trace = ["classify_error: TestLogicFailure conf=0.5", "generate_code: iteration=1 model=ollama:rebirth-python:7b chars=20",
             "execute_tests: FAIL exit=1 in 0.1s", "classify_error: SyntaxError conf=0.9",
             "generate_code: iteration=2 model=ollama:rebirth-python:7b chars=20", "execute_tests: FAIL exit=1 in 0.1s"]
    monkeypatch.setattr(graph_app, "build_migration_graph", lambda: _Graph(
        {"status": "failed", "iteration_count": 2, "trace": trace, "current_draft": "print(f'{os.name}')\n",
         "adapter_id": "python", "error_class": "SyntaxError"}))
    res = CliRunner().invoke(app, ["migrate", str(proj), "--file", "src/a.py", "--lang", "python", "--no-write"])
    assert res.exit_code == 1, res.output
    xtrace = ["translate_code: iteration=1 model=ollama:rebirth-xlang:14b-v1 chars=30", "verify_target: compile PASS exit=0 in 1s"]
    monkeypatch.setattr(xgraph, "build_translation_graph", lambda: _Graph(
        {"status": "success", "iteration_count": 1, "trace": xtrace, "current_draft": "package main\n",
         "model": "ollama:rebirth-xlang:14b-v1", "verify_mode": "compile"}))
    res = CliRunner().invoke(app, ["migrate", str(proj), "--file", "src/a.py", "--to", "go", "--out-dir", str(tmp_path / "out")])
    assert res.exit_code == 0, res.output

    recs = load_records(runs)
    assert [r.kind for r in recs] == ["migrate", "translate"]
    m, x = recs[0].t, recs[1].t
    assert m["task"] == "modernize" and m["model"] == "ollama:rebirth-python:7b" and m["error_classes"] == ["TestLogicFailure", "SyntaxError"]
    assert m["input"]["imports"] == ["os"] and isinstance(m["wall_seconds"], float)
    assert x["direction"] == "python->go" and x["model"] == "ollama:rebirth-xlang:14b-v1" and x["success"]
    assert [r.run_id for r in flywheel.queue(runs)] == [recs[0].run_id]          # failed → queued; easy success → not
    src = flywheel.artifacts(recs[0].run_id, runs)["source"]
    assert "abc123" not in src and "TOKEN=***" in src
