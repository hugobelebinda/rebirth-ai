"""Bugs from the 1.0.0 release review, and the same kinds of bugs found elsewhere (no Docker, no models)."""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rebirth_ai import cli
from rebirth_ai.cli import app

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Output without colours (GitHub Actions forces them) and with line wraps undone."""
    return " ".join(_ANSI.sub("", text).split())


# --------------------------------------------------------------------------- python -m rebirth_ai.cli


def test_python_m_runs_after_every_definition() -> None:
    """The __main__ block sat above _print_diff and _assign_panels(): a passing migrate raised NameError and
    --help lost its panels when run as `python -m rebirth_ai.cli`."""
    body = ast.parse((ROOT / "src" / "rebirth_ai" / "cli.py").read_text()).body
    last = body[-1]
    assert isinstance(last, ast.If) and "__main__" in ast.unparse(last.test)
    out = subprocess.run([sys.executable, "-m", "rebirth_ai.cli", "--help"], capture_output=True, text=True,
                         timeout=120, cwd=ROOT)
    assert out.returncode == 0 and "Migrate" in _plain(out.stdout) and "Maintainers" in _plain(out.stdout)


# --------------------------------------------------------------------------- language from the file suffix


@pytest.mark.parametrize("path", ["fixtures/go_sample/src/mathops.go", "fixtures/rust_sample/src/lib.rs",
                                  "fixtures/cpp_sample/src/math_ops.cpp", "examples/billing/src/invoice.py"])
def test_parse_picks_the_language_from_the_suffix(path) -> None:
    res = runner.invoke(app, ["parse", str(ROOT / path)])
    assert res.exit_code == 0, res.output
    assert "chunk(s)" in _plain(res.output)


def test_parse_names_the_problem_for_an_unknown_suffix(tmp_path) -> None:
    f = tmp_path / "notes.txt"
    f.write_text("hello\n")
    res = runner.invoke(app, ["parse", str(f)])
    assert res.exit_code == 2 and "--lang" in _plain(res.output)


class _Result:
    model, prompt_chars = "fake", 1

    def __init__(self, code: str) -> None:
        self.code = code


@pytest.mark.parametrize(("name", "routed", "label"), [
    ("invoice.py", "python", "python"), ("main.go", "go", "go"), ("App.java", "java", "java"),
    ("app.ts", "js_ts", "typescript"), ("app.js", "js_ts", "javascript"), ("lib.rs", "rust", "rust"),
])
def test_generate_uses_the_files_language(tmp_path, monkeypatch, name, routed, label) -> None:
    """Every file that wasn't .ts / .tsx went to the JavaScript model with a JavaScript prompt."""
    from rebirth_ai.generate import factory

    seen = {}

    class Client:
        def migrate(self, **kw):
            seen["label"] = kw["language"]
            return _Result("x = 1\n")

    monkeypatch.setattr(factory, "get_generator", lambda backend=None, *, language=None: seen.update(routed=language) or Client())
    f = tmp_path / name
    f.write_text("x = 1\n")
    res = runner.invoke(app, ["generate", str(f), "--no-rag"])
    assert res.exit_code == 0, res.output
    assert seen == {"routed": routed, "label": label}


def test_generate_does_not_write_an_empty_draft(tmp_path, monkeypatch) -> None:
    from rebirth_ai.generate import factory

    class Client:
        def migrate(self, **kw):
            return _Result("\n")

    monkeypatch.setattr(factory, "get_generator", lambda backend=None, *, language=None: Client())
    f, out = tmp_path / "a.py", tmp_path / "out.py"
    f.write_text("x = 1\n")
    res = runner.invoke(app, ["generate", str(f), "--no-rag", "--out", str(out)])
    assert not out.exists() and "no code" in _plain(res.output)


def test_code_search_parses_a_file_with_its_own_language(tmp_path, monkeypatch) -> None:
    """Without an adapter, the retriever parsed every non-.py file as JS/TS."""
    monkeypatch.setenv("REBIRTH_EMBEDDER", "hash")
    from rebirth_ai.rag.retrieve import Retriever

    r = Retriever(db_path=tmp_path / "idx.db")
    asked = []
    monkeypatch.setattr(r, "query", lambda text, **kw: asked.append(text) or [])
    r.query_for_file(ROOT / "fixtures" / "go_sample" / "src" / "mathops.go")
    # Go function names found by the Go parser, not the first 1,000 characters of a file nothing could parse
    assert asked and "Add" in asked[0].split() and "package" not in asked[0]


def test_the_import_reader_only_reads_languages_it_knows(tmp_path) -> None:
    """Dependency order covers JS/TS and Python; other files are not parsed as JavaScript."""
    from rebirth_ai.graph import discovery

    (tmp_path / "a.go").write_text('package main\n\nimport "./b.go"\n')
    (tmp_path / "b.go").write_text("package main\n")
    assert discovery.extract_local_imports(tmp_path, "a.go") == []
    (tmp_path / "x.js").write_text("import { y } from './y.js';\n")
    (tmp_path / "y.js").write_text("export const y = 1;\n")
    assert discovery.extract_local_imports(tmp_path, "x.js") == ["y.js"]


# --------------------------------------------------------------------------- outputs of a run that failed


class _Graph:
    def __init__(self, final: dict) -> None:
        self.final = final

    def invoke(self, state):
        return {**state, **self.final}


@pytest.fixture
def billing(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("REBIRTH_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cli, "_preflight", lambda **kw: None)
    shutil.copytree(ROOT / "examples" / "billing", tmp_path / "billing")
    return tmp_path / "billing"


@pytest.mark.parametrize(("status", "draft", "written"), [
    ("success", "x = 2\n", True), ("human_review", "x = 2\n", False), ("human_review", "\n", False),
])
def test_modernize_out_holds_only_a_draft_that_passed(billing, monkeypatch, status, draft, written) -> None:
    from rebirth_ai.graph import app as graph_app

    monkeypatch.setattr(graph_app, "build_migration_graph",
                        lambda *a, **k: _Graph({"status": status, "current_draft": draft, "iteration_count": 1}))
    out = billing.parent / "new.py"
    res = runner.invoke(app, ["migrate", str(billing), "--file", "src/invoice.py", "--no-fallback", "--out", str(out)])
    assert out.exists() is written, res.output
    if not written:
        assert "not written: no draft passed the tests" in _plain(res.output)


@pytest.mark.parametrize(("draft", "written"), [("package main\n", True), ("\n", False), ("", False)])
def test_translate_never_writes_an_empty_file(billing, monkeypatch, draft, written) -> None:
    from rebirth_ai.xlang import graph as xgraph

    monkeypatch.setattr(xgraph, "build_translation_graph",
                        lambda *a, **k: _Graph({"status": "human_review", "current_draft": draft, "iteration_count": 2,
                                                "verify_mode": "compile"}))
    out_dir, out = billing.parent / "out", billing.parent / "copy.go"
    res = runner.invoke(app, ["migrate", str(billing), "--file", "src/invoice.py", "--to", "go",
                              "--out-dir", str(out_dir), "--out", str(out)])
    dest = out_dir / "src" / "invoice.go"
    assert dest.exists() is written and out.exists() is written, res.output
    text = _plain(res.output)
    assert ("not verified" in text) if written else ("nothing written: the model returned no code" in text)


# --------------------------------------------------------------------------- run from a clone, from anywhere


def test_eval_paths_do_not_depend_on_the_current_folder(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(subprocess, "call", lambda cmd, **kw: calls.append((cmd, kw)) or 0)
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(app, ["eval", "--suite", "heldout"])
    assert res.exit_code == 0, res.output
    cmd, kw = calls[0]
    args = dict(zip(cmd[2::2], cmd[3::2], strict=False))
    assert Path(args["--manifest"]).is_file() and Path(args["--manifest"]).is_absolute()
    assert Path(args["--out"]).is_absolute() and Path(args["--out"]).parent == ROOT / "data" / "reports"
    assert kw.get("cwd") == ROOT
    calls.clear()
    runner.invoke(app, ["eval", "--out", "mine.json"])  # a path the user gives is theirs: relative to their folder
    assert dict(zip(calls[0][0][2::2], calls[0][0][3::2], strict=False))["--out"] == str(tmp_path / "mine.json")


def test_outside_a_clone_every_command_says_how_to_install(tmp_path, monkeypatch) -> None:
    """A plain `pip install` has no configs/, docker/ or training/: one clear message instead of odd failures."""
    from rebirth_ai import paths

    monkeypatch.setattr(paths, "ROOT", tmp_path)
    res = runner.invoke(app, ["doctor"])
    assert res.exit_code == 2 and "git clone" in _plain(res.output) and "bootstrap.sh" in _plain(res.output)
    assert runner.invoke(app, ["--version"]).exit_code == 0
