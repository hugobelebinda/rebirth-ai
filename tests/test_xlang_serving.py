"""Cross-language serving route (Phase 19 / 16): report gate, publish, rollback and roll-forward."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rebirth_ai import packaging, routing


@pytest.fixture
def fake_ollama(tmp_path, monkeypatch):
    bindir, state, log = tmp_path / "bin", tmp_path / "tags.txt", tmp_path / "ollama.log"
    bindir.mkdir()
    state.write_text("rebirth-xlang:14b-v1\nrebirth-xlang:14b-v2\n")
    script = bindir / "ollama"
    script.write_text(f"""#!/usr/bin/env bash
echo "$@" >> "{log}"
case "$1" in
  show) grep -qxF "$2" "{state}" ;;
  cp) echo "$3" >> "{state}" ;;
  rm) grep -vxF "$2" "{state}" > "{state}.n"; mv "{state}.n" "{state}" ;;
esac
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("REBIRTH_ROUTING", str(tmp_path / "routing.yaml"))
    return log


def _report(path: Path, suite: str, dirs: dict[str, tuple[int, int, int]]) -> Path:
    path.write_text(json.dumps({"suite": suite, "directions": {
        d: {"n": n, "pass": p, "pass_first": f} for d, (n, p, f) in dirs.items()}}))
    return path


def test_scores_and_gate(tmp_path):
    prog = _report(tmp_path / "p.json", "programs", {"python->go": (20, 15, 10), "java->rust": (20, 12, 9)})
    he = _report(tmp_path / "h.json", "humaneval", {"python->go": (40, 39, 38)})
    s = packaging.xlang_scores([prog, he])
    assert s == {"programs": {"pass": 27, "n": 40}, "humaneval": {"pass": 38, "n": 40}}  # verified / first try

    base = {"programs": {"pass": 27, "n": 1000}, "humaneval": {"pass": 900, "n": 1000}}
    worse = packaging.xlang_gate({"programs": {"pass": 6, "n": 1000}, "humaneval": {"pass": 900, "n": 1000}}, base)
    assert not worse.passed  # 6 < 27 - 20
    within = packaging.xlang_gate({"programs": {"pass": 10, "n": 1000}, "humaneval": {"pass": 891, "n": 1000}}, base)
    assert within.passed and within.warnings  # drops of 17 (≤ 20) and 9 (≤ 10) are noise
    assert not packaging.xlang_gate({"programs": {"pass": 30, "n": 999}, "humaneval": base["humaneval"]}, base).passed
    assert not packaging.xlang_gate(base, None).passed


def test_publish_rollback_roll_forward(fake_ollama):
    packaging.publish_translation({"tag": "rebirth-xlang:14b-v1", "version": "1"}, log=lambda s: None)
    packaging.publish_translation({"tag": "rebirth-xlang:14b-v2", "version": "2"}, log=lambda s: None)
    r = routing.translation_route()
    assert r["model"] == "rebirth-xlang:14b" and r["tag"].endswith("v2") and list(r["history"]) == ["1"]

    back = packaging.rollback_translation(log=lambda s: None)
    assert back["tag"].endswith("v1") and list(back["history"]) == ["2"]
    assert routing.translation_route()["version"] == "1"
    assert "cp rebirth-xlang:14b-v1 rebirth-xlang:14b" in fake_ollama.read_text()

    fwd = packaging.rollback_translation(log=lambda s: None)
    assert fwd["version"] == "2" and list(fwd["history"]) == ["1"]
    # same-language routes are untouched by the translation route
    assert routing.load_routing()["languages"] == {}


def test_rollback_refusals(fake_ollama, tmp_path):
    with pytest.raises(RuntimeError, match="no previous version"):
        packaging.rollback_translation(log=lambda s: None)
    packaging.publish_translation({"tag": "rebirth-xlang:14b-v1", "version": "1"}, log=lambda s: None)
    packaging.publish_translation({"tag": "rebirth-xlang:14b-v2", "version": "2"}, log=lambda s: None)
    tags = tmp_path / "tags.txt"
    tags.write_text("rebirth-xlang:14b-v2\nrebirth-xlang:14b\n")  # v1 deleted from Ollama
    with pytest.raises(RuntimeError, match="not in Ollama"):
        packaging.rollback_translation(log=lambda s: None)
    assert routing.translation_route()["version"] == "2"  # nothing changed
    with pytest.raises(RuntimeError, match="not in Ollama"):
        packaging.publish_translation({"tag": "rebirth-xlang:14b-v9", "version": "9"}, log=lambda s: None)


def test_cli_publish_gate(fake_ollama, tmp_path):
    from rebirth_ai.cli import app

    old = _report(tmp_path / "old.json", "programs", {"a->b": (100, 60, 50)})
    bad = _report(tmp_path / "bad.json", "programs", {"a->b": (100, 50, 40)})
    good = _report(tmp_path / "good.json", "programs", {"a->b": (100, 61, 55)})
    runner = CliRunner()
    r = runner.invoke(app, ["serve", "publish-xlang", "--tag", "rebirth-xlang:14b-v2", "--version", "2",
                            "--report", str(bad), "--baseline", str(old)])
    assert r.exit_code == 1 and "gate failed" in r.output and routing.translation_route() is None
    r = runner.invoke(app, ["serve", "publish-xlang", "--tag", "rebirth-xlang:14b-v2", "--version", "2",
                            "--report", str(good), "--baseline", str(old), "--note", "merged LoRA"])
    assert r.exit_code == 0, r.output
    route = routing.translation_route()
    assert route["eval"]["programs"] == {"pass": 61, "n": 100} and route["note"] == "merged LoRA"
    r = runner.invoke(app, ["serve", "rollback", "--lang", "xlang"])
    assert r.exit_code == 1 and "no previous version" in r.output
