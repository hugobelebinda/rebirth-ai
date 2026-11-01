"""Docker sandbox integration tests (skipped if Docker/image unavailable)."""

from __future__ import annotations

from pathlib import Path

import docker
import pytest
from docker.errors import DockerException, ImageNotFound

from rebirth_ai.adapters.js_ts import JsTsAdapter
from rebirth_ai.sandbox.runner import SandboxRunner

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "js_ts_sample"
IMAGE = "rebirth-ai/js-ts-sandbox:latest"


def _docker_ready() -> bool:
    try:
        client = docker.from_env()
        client.ping()
        client.images.get(IMAGE)
        return True
    except (DockerException, ImageNotFound, OSError):
        return False


pytestmark = pytest.mark.skipif(
    not _docker_ready(),
    reason=f"Docker unavailable or image {IMAGE} not built",
)


def test_sandbox_passing_suite() -> None:
    runner = SandboxRunner(JsTsAdapter(image=IMAGE))
    result = runner.run_project(FIXTURE, relative_test="test/math.test.js")
    assert result.ok, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert result.exit_code == 0


def test_sandbox_failing_suite() -> None:
    runner = SandboxRunner(JsTsAdapter(image=IMAGE))
    result = runner.run_project(FIXTURE, relative_test="test/math.failing.test.js")
    assert not result.ok
    assert result.exit_code != 0
    combined = (result.stdout + result.stderr).lower()
    assert "fail" in combined or "assert" in combined or "999" in combined


def test_sandbox_overlay_can_break_passing_test(tmp_path: Path) -> None:
    """Inject a broken math.js overlay and expect failure without mutating fixture."""
    broken = tmp_path / "math.js"
    broken.write_text(
        "export function add(a, b) { return 0; }\n"
        "export function multiply(a, b) { return 0; }\n"
        "export function summarize(a, b) { return 'sum=0'; }\n"
        "export class Calculator { constructor() { this.value = 0; } "
        "add(n){return 0;} multiply(n){return 0;} reset(){} }\n",
        encoding="utf-8",
    )
    runner = SandboxRunner(JsTsAdapter(image=IMAGE))
    result = runner.run_project(
        FIXTURE,
        relative_test="test/math.test.js",
        overlay_file=broken,
        overlay_dest="src/math.js",
    )
    assert not result.ok
