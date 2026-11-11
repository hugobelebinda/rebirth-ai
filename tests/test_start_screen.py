"""Phase 19 launch UX: the start screen (`rebirth` alone), --version, the grouped help, and defaults that
never write into the folder you run Rebirth from."""

from __future__ import annotations

import io
import re

import pytest
from rich.console import Console
from typer.testing import CliRunner

from rebirth_ai import start

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


@pytest.fixture
def offline(monkeypatch):
    """No Ollama / Docker probes: a fixed machine status."""
    monkeypatch.setattr(start, "machine_status", lambda *a, **k: [("Ollama", True), ("Docker", False)])


def _console(answers: list[str]) -> tuple[Console, io.StringIO]:
    out = io.StringIO()
    console = Console(file=out, width=100, force_terminal=False, color_system=None, highlight=False)
    feed = iter(answers)

    def fake_input(prompt="", **_):
        console.print(prompt, end="")
        try:
            return next(feed)
        except StopIteration as exc:
            raise EOFError from exc

    console.input = fake_input  # type: ignore[method-assign]
    return console, out


def test_logo_rows_line_up() -> None:
    assert len(start.LOGO) == 6 and len({len(row) for row in start.LOGO}) == 1


def test_screen_shows_status_and_points_to_doctor(offline, monkeypatch) -> None:
    monkeypatch.setattr(start, "interactive", lambda: False)
    console, out = _console([])
    assert start.start(console) == 0
    text = _plain(out.getvalue())
    assert "✓ Ollama" in text and "✗ Docker" in text and "rebirth doctor" in text
    assert "JS/TS · Python · Go · Java" in text and "rebirth --help" in text
    assert "What do you want to do" not in text  # no prompt when not in a terminal


def test_narrow_terminal_skips_the_big_logo() -> None:
    out = io.StringIO()
    start.render(Console(file=out, width=50, color_system=None), rows=[], menu=False)
    assert "REBIRTH" in out.getvalue() and "██" not in out.getvalue()


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (["", ".", "", ""], ["migrate-repo", "{p}"]),
        (["1", ".", "src/a.py", "y"], ["migrate", "{p}", "--file", "src/a.py", "--write"]),
        (["mod", "{p}", "", "yes"], ["migrate-repo", "{p}", "--write"]),
        (["translate", ".", "src/a.py", "rust"], ["migrate", "{p}", "--file", "src/a.py", "--to", "rust"]),
        (["3"], ["doctor"]),
        (["help"], ["--help"]),
    ],
)
def test_wizard_builds_the_command(tmp_path, monkeypatch, offline, answers, expected) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(start, "interactive", lambda: True)
    ran = []
    console, out = _console([a.replace("{p}", str(tmp_path)) for a in answers])
    assert start.start(console, run=lambda args: ran.append(args) or 7) == 7
    assert ran == [[e.replace("{p}", str(tmp_path)) for e in expected]]
    assert "→ rebirth " in out.getvalue()  # the command is shown before it runs


def test_wizard_re_asks_on_bad_answers(tmp_path, monkeypatch, offline) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(start, "interactive", lambda: True)
    ran = []
    answers = ["9", "translate", str(tmp_path / "nope"), str(tmp_path), "../outside.py", "missing.py", "a.py",
               "klingon", "go"]
    console, out = _console(answers)
    start.start(console, run=lambda args: ran.append(args) or 0)
    text = _plain(out.getvalue())
    assert "Type a number from 1 to 4" in text and "No such folder" in text
    assert "is not inside" in text and "No such file" in text and "Pick one of" in text
    assert ran == [["migrate", str(tmp_path), "--file", "a.py", "--to", "go"]]


def test_ctrl_c_or_eof_cancels(offline, monkeypatch) -> None:
    monkeypatch.setattr(start, "interactive", lambda: True)
    console, out = _console([])
    assert start.start(console, run=lambda args: pytest.fail("must not run")) == 130
    assert "Cancelled" in out.getvalue()


def test_run_args_runs_a_command_in_process(capsys) -> None:
    assert start.run_args(["version"]) == 0
    assert "rebirth-ai" in capsys.readouterr().out
    assert start.run_args(["no-such-command"]) == 2


def test_version_flag_and_grouped_help() -> None:
    from rebirth_ai.cli import app

    runner = CliRunner()
    assert "rebirth-ai" in runner.invoke(app, ["--version"]).output
    help_text = _plain(runner.invoke(app, ["--help"]).output)
    panels = [help_text.index(p) for p in ("Migrate", "Setup", "Tools", "Maintainers")]
    assert panels == sorted(panels)  # user commands first, maintainer commands last
    assert help_text.index("migrate-repo") < help_text.index("mlflow-ui")


def test_rebirth_alone_shows_the_start_screen(monkeypatch, offline) -> None:
    from rebirth_ai.cli import app

    monkeypatch.setattr(start, "interactive", lambda: False)
    res = CliRunner().invoke(app, [])
    assert res.exit_code == 0 and "rebirth --help" in _plain(res.output)


def test_default_data_paths_are_in_the_rebirth_folder(tmp_path, monkeypatch) -> None:
    from rebirth_ai import paths, runs
    from rebirth_ai.classify.model import DEFAULT_MODEL_DIR

    monkeypatch.delenv("REBIRTH_RUNS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    for p in (paths.DEFAULT_DB, paths.DEFAULT_REPORT, paths.runs_dir_default(), DEFAULT_MODEL_DIR):
        assert p.is_absolute() and paths.ROOT in p.parents
    assert runs.runs_dir().is_relative_to(paths.ROOT)
    assert not (tmp_path / "data").exists()  # nothing was created in the current folder


def test_classifier_trains_itself_when_missing(tmp_path, monkeypatch) -> None:
    from rebirth_ai.classify import model

    monkeypatch.setattr(model, "DEFAULT_MODEL_DIR", tmp_path / "clf")
    clf = model.ErrorClassifier.load(tmp_path / "clf")
    assert (tmp_path / "clf" / "pipeline.joblib").is_file()
    assert clf.predict("SyntaxError: invalid syntax") in model.ERROR_CLASSES


def test_migrate_repo_looks_in_the_whole_project_without_src(tmp_path, monkeypatch) -> None:
    from rebirth_ai import preflight
    from rebirth_ai.cli import app

    monkeypatch.setattr(preflight, "ollama_error", lambda *a, **k: None)
    monkeypatch.setattr(preflight, "docker_error", lambda *a, **k: None)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "util.py").write_text("def f():\n    return 1\n")
    (tmp_path / "test_util.py").write_text("from pkg.util import f\n")
    res = CliRunner().invoke(app, ["migrate-repo", str(tmp_path), "--dry-run", "--report", str(tmp_path / "r.json")])
    out = _plain(res.output)
    assert res.exit_code == 0, out
    assert "under='.'" in out and "pkg/util.py" in out
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1\n")
    out = _plain(CliRunner().invoke(app, ["migrate-repo", str(tmp_path), "--dry-run",
                                          "--report", str(tmp_path / "r.json")]).output)
    assert "under='src'" in out and "src/app.py" in out and "pkg/util.py" not in out
