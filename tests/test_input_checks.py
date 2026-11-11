"""Phase 19 bug hunt: bad inputs and a missing Ollama / Docker give one-line errors, not tracebacks."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rebirth_ai import preflight
from rebirth_ai.safety import InputError, check_source_file, inside


@pytest.fixture
def proj(tmp_path: Path) -> Path:
    p = tmp_path / "proj"
    (p / "src").mkdir(parents=True)
    (p / "src" / "m.py").write_text("def f(x):\n    return x + 1\n")
    (p / ".env").write_text("SECRET=1\n")
    (p / "src" / "bin.py").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    (p / "src" / "latin1.py").write_bytes(b'x = "caf\xe9"\n')
    (p / "src" / "empty.py").write_text("  \n")
    (tmp_path / "outside.py").write_text("print(1)\n")
    (p / "src" / "link.py").symlink_to(tmp_path / "outside.py")
    return p


@pytest.mark.parametrize("rel,msg", [
    ("src/nope.py", "no such file"), ("../outside.py", "outside the project"), ("/etc/hostname", "outside the project"),
    ("src/link.py", "outside the project"), (".env", "secrets file"), ("src/bin.py", "binary"),
    ("src/latin1.py", "not UTF-8"), ("src/empty.py", "empty"), ("src", "not a file"),
])
def test_check_source_file_refuses(proj, rel, msg) -> None:
    with pytest.raises(InputError, match=msg):
        check_source_file(proj, rel)


def test_check_source_file_ok_and_size_limit(proj, monkeypatch) -> None:
    assert check_source_file(proj, "src/m.py").startswith("def f")
    monkeypatch.setenv("REBIRTH_MAX_FILE_BYTES", "10")
    with pytest.raises(InputError, match="over REBIRTH_MAX_FILE_BYTES"):
        check_source_file(proj, "src/m.py")
    assert inside(proj, "src/../src/m.py") == (proj / "src" / "m.py").resolve()


def test_preflight_messages(monkeypatch) -> None:
    msg = preflight.ollama_error("http://127.0.0.1:1")
    assert msg and "Ollama is not reachable at http://127.0.0.1:1" in msg and "rebirth doctor" in msg
    monkeypatch.setenv("DOCKER_HOST", "unix:///nonexistent-rebirth.sock")
    msg = preflight.docker_error()
    assert msg and msg.startswith("Docker is not reachable")


def test_ollama_host_forms(monkeypatch) -> None:
    from rebirth_ai.generate.ollama_client import OllamaClient, ollama_host

    assert ollama_host("127.0.0.1:11434") == "http://127.0.0.1:11434"  # Ollama's own OLLAMA_HOST form
    assert ollama_host("0.0.0.0") == "http://127.0.0.1:11434"
    assert ollama_host("https://ollama.example.com/") == "https://ollama.example.com"  # full URLs as given
    monkeypatch.setenv("OLLAMA_HOST", "gpu-box:12345")
    assert OllamaClient().host == "http://gpu-box:12345"


def test_test_paths_with_spaces_are_quoted() -> None:
    from rebirth_ai.adapters.go_lang import GoAdapter
    from rebirth_ai.adapters.js_ts import JsTsAdapter
    from rebirth_ai.adapters.python import PythonAdapter

    assert PythonAdapter().test_command_for("dir with spaces/test_u.py") == "python -m pytest -q 'dir with spaces/test_u.py'"
    assert JsTsAdapter().test_command_for("my tests/a.test.js") == "node --test 'my tests/a.test.js'"
    assert GoAdapter().test_command_for("my pkg/a_test.go") == "go test './my pkg'"


def test_file_language_beats_project_markers(tmp_path) -> None:
    from rebirth_ai.graph.app import initial_state

    (tmp_path / "package.json").write_text('{"name": "web"}')  # a JS project …
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "gen.py").write_text("print(1)\n")  # … with a Python script
    st = initial_state(project_dir=tmp_path, relative_file="tools/gen.py", relative_test="", objective="m")
    assert st["adapter_id"] == "python"


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Output without colour codes (Rich colours it on GitHub Actions) or line wrapping."""
    return " ".join(_ANSI.sub("", text).split())


def _cli(args, monkeypatch, *, ollama=None, docker=None):
    from rebirth_ai.cli import app

    monkeypatch.setattr(preflight, "ollama_error", lambda *a, **k: ollama)
    monkeypatch.setattr(preflight, "docker_error", lambda *a, **k: docker)
    return CliRunner().invoke(app, args)


def test_cli_errors_exit_2_without_traceback(proj, tmp_path, monkeypatch) -> None:
    cases = [
        (["migrate", str(proj), "--file", "../outside.py", "--write"], {}, "outside the project"),
        (["migrate", str(proj), "--file", ".env", "--write"], {}, "secrets file"),
        (["migrate", str(proj), "--file", "src/m.py", "--to", "klingon"], {}, "Unknown language"),
        (["migrate", str(proj), "--file", "src/m.py", "--to", "go", "--target-project", str(proj),
          "--target-file", "../../x.go"], {}, "outside --target-project"),
        (["migrate", str(proj), "--file", "src/m.py"], {"ollama": "Ollama is not reachable at x"}, "Ollama is not reachable"),
        (["migrate-repo", str(proj), "--under", "src"], {"docker": "Docker is not reachable (x)"}, "Docker is not reachable"),
        (["sandbox", str(tmp_path / "proj" / "src")], {"docker": "Docker is not reachable (x)"}, "Docker is not reachable"),
    ]
    for args, env, want in cases:
        res = _cli(args, monkeypatch, **env)
        assert res.exit_code == 2 and want in _plain(res.output) and "Traceback" not in res.output, (args, res.output)
    assert (tmp_path / "outside.py").read_text() == "print(1)\n"  # nothing written outside


def test_cli_report_and_sandbox_language(tmp_path, monkeypatch) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{bad")
    res = _cli(["report", str(bad)], monkeypatch)
    assert res.exit_code == 2 and "not a Rebirth JSON report" in _plain(res.output)
    (tmp_path / "empty").mkdir()
    res = _cli(["sandbox", str(tmp_path / "empty")], monkeypatch)
    assert res.exit_code == 2 and "pass --lang" in _plain(res.output)


def test_install_from_missing_folder_fails_before_any_pull(tmp_path) -> None:
    from rebirth_ai.install import InstallError, install

    with pytest.raises(InstallError, match="no such folder"):
        install("modernize", from_dir=tmp_path / "nope")


def test_open_pr_checks_git_before_running_a_model(proj, monkeypatch) -> None:
    import rebirth_ai.graph.batch as batch
    from rebirth_ai.sandbox.runner import SandboxRunner

    monkeypatch.setattr(batch, "build_migration_graph", lambda: pytest.fail("the model must not run"))
    monkeypatch.setattr(SandboxRunner, "ensure_image", lambda self: None)  # CI has no sandbox images
    res = _cli(["migrate-repo", str(proj), "--under", "src", "--open-pr"], monkeypatch)
    out = _plain(res.output)
    assert res.exit_code == 2 and "--open-pr:" in out and "not a git repository" in out


def test_migrate_requires_file_and_finds_the_test(tmp_path, monkeypatch) -> None:
    import rebirth_ai.graph.app as graph_app

    res = _cli(["migrate", str(tmp_path)], monkeypatch)
    assert res.exit_code == 2 and "--file" in _plain(res.output)  # no silent src/math.js default
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("def f():\n    return 1\n")
    (tmp_path / "test_pkg.py").write_text("import pkg\n")
    seen = {}

    class G:
        def invoke(self, state):
            seen["test"] = state["relative_test"]
            return {**state, "status": "failed", "iteration_count": 1, "trace": []}

    monkeypatch.setattr(graph_app, "build_migration_graph", lambda: G())
    _cli(["migrate", str(tmp_path), "--file", "pkg/__init__.py", "--no-write"], monkeypatch)
    assert seen["test"] == "test_pkg.py"  # inferred, not the old test/math.test.js default


def test_a_container_that_cannot_start_is_a_one_line_error(proj, monkeypatch) -> None:
    """A hand-built image without the expected user made Docker refuse to start the container: that was
    a 40-line traceback."""
    from docker.errors import APIError

    from rebirth_ai.adapters.python import PythonAdapter
    from rebirth_ai.sandbox.runner import SandboxError, SandboxRunner

    removed: list[str] = []

    class Created:  # Docker creates the container, then fails to start it
        def __init__(self, name):
            self.name = name

        def remove(self, force=False):
            removed.append(self.name)

    class Containers:
        def run(self, **kw):
            self.name = kw["name"]
            raise APIError("500", explanation="unable to find user rebirth: no matching entries in passwd file")

        def get(self, name):
            return Created(name)

    class Images:
        def get(self, name):
            return name

    class Client:
        containers = Containers()
        images = Images()

    runner = SandboxRunner(PythonAdapter(), client=Client())
    with pytest.raises(SandboxError, match="unable to find user rebirth.*docker build -t"):
        runner.run_project(proj)
    # the created-but-never-started container is not left behind
    assert removed == [Client.containers.name] and removed[0].startswith("rebirth-sandbox-")

    monkeypatch.setattr(SandboxRunner, "ensure_image", lambda self: None)
    monkeypatch.setattr(SandboxRunner, "run_project", lambda *a, **k: (_ for _ in ()).throw(SandboxError("Docker could not start x")))
    res = _cli(["sandbox", str(proj), "--lang", "python"], monkeypatch)
    assert res.exit_code == 2 and "Docker could not start x" in _plain(res.output) and "Traceback" not in res.output
