"""Sandbox confinement against a real container (skipped without Docker / the Python sandbox image)."""

from __future__ import annotations

from pathlib import Path

import pytest
from docker.errors import DockerException, ImageNotFound

import docker
from rebirth_ai.adapters.python import PythonAdapter
from rebirth_ai.sandbox.runner import SandboxRunner

IMAGE = "rebirth-ai/python-sandbox:latest"


def _ready() -> bool:
    try:
        client = docker.from_env()
        client.ping()
        client.images.get(IMAGE)
        return True
    except (DockerException, ImageNotFound, OSError):
        return False


pytestmark = pytest.mark.skipif(not _ready(), reason=f"Docker unavailable or image {IMAGE} not built")

NET_PROBE = (
    "python -c \"import socket\ntry:\n socket.create_connection(('1.1.1.1', 53), 3); print('NET')\n"
    "except OSError: print('NONET')\""
)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    return tmp_path


def run(project: Path, cmd: str, **kw):
    return SandboxRunner(PythonAdapter(image=IMAGE)).run_project(project, command=cmd, **kw)


def test_time_limit_kills_the_run(project):
    res = run(project, "sleep 60", timeout_sec=2)
    assert res.timed_out and res.exit_code == 124 and res.duration_sec < 20


def test_memory_limit(project, monkeypatch):
    monkeypatch.setenv("REBIRTH_SANDBOX_MEMORY", "128m")
    res = run(project, "python -c \"b = bytearray(512 * 1024 * 1024); print(len(b))\"")
    assert res.exit_code == 137 and not res.timed_out


def test_process_limit(project, monkeypatch):
    monkeypatch.setenv("REBIRTH_SANDBOX_PIDS", "32")
    res = run(project, "for i in $(seq 1 100); do sleep 3 & done; wait", timeout_sec=30)
    assert "fork" in (res.stdout + res.stderr).lower() or "resource temporarily unavailable" in res.stderr.lower()


def test_no_capabilities_no_new_privileges_not_root(project):
    res = run(project, "id -u; grep -E '^(CapEff|NoNewPrivs)' /proc/self/status")
    lines = res.stdout.split()
    assert lines[0] != "0" and "0000000000000000" in res.stdout and "NoNewPrivs:\t1" in res.stdout.replace(" ", "\t")


def test_no_network_without_setup(project):
    assert "NONET" in run(project, NET_PROBE).stdout


def test_setup_online_then_tests_offline_with_the_draft(project):
    res = SandboxRunner(PythonAdapter(image=IMAGE)).run_project(
        project,
        command=f"test -f src/new.py && {NET_PROBE}",
        setup_command=f"test ! -f src/new.py && {NET_PROBE}",
        overlay_text="y = 2\n",
        overlay_dest="src/new.py",
    )
    assert res.setup_exit_code == 0  # the draft was not there during setup
    assert res.ok and "NONET" in res.stdout
