"""Ephemeral Docker sandbox execution via the Docker Python SDK.

Every run gets a fresh container that is destroyed afterwards, and is confined:

- **No network for code under test.** A run has up to two steps in the same container. The
  optional *setup* step (dependency download: ``mvn``, ``dotnet restore``, ``npm ci``, ``pip``) runs
  on the project as the user has it, with network. Then the container is disconnected from every
  network (checked; the run fails closed if it is still attached), the model's draft is written in,
  and the *test* step runs offline. Without a setup step the container never has a network.
- **Limits.** Memory (``REBIRTH_SANDBOX_MEMORY``, default 4g, no swap), process count
  (``REBIRTH_SANDBOX_PIDS``, default 1024), all Linux capabilities dropped, ``no-new-privileges``,
  the image's non-root user.
- **Time.** Each step runs under ``timeout -k 5 <sec>`` inside the container (the adapter's
  ``timeout_sec`` / ``REBIRTH_SANDBOX_TIMEOUT_SEC`` for tests, ``setup_timeout_sec`` for setup); the
  host kills the container 30 s after that if the step still has not returned.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path

from docker.errors import ContainerError, ImageNotFound
from docker.models.containers import Container

import docker
from rebirth_ai.adapters.base import LanguageAdapter, SandboxSpec

TIMEOUT_EXIT = 124  # GNU timeout's exit status; 137 = SIGKILL (after -k, or out of memory)
KILL_GRACE_SEC = 5
HOST_GRACE_SEC = 30
_COPY_IGNORE = shutil.ignore_patterns(
    "node_modules", ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    "target", "build", ".build", "bin", "obj",
)


def container_limits() -> dict[str, object]:
    """``containers.run`` keyword arguments that confine every sandbox container."""
    mem = os.environ.get("REBIRTH_SANDBOX_MEMORY", "4g")
    return {
        "mem_limit": mem,
        "memswap_limit": mem,  # = mem_limit: no swap on top
        "pids_limit": int(os.environ.get("REBIRTH_SANDBOX_PIDS", "1024")),
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges"],
    }


def timed(cmd: str, timeout_sec: float | None) -> list[str]:
    """argv running ``cmd`` in ``sh`` under ``timeout`` (no limit when ``timeout_sec`` is falsy / ≤ 0)."""
    if timeout_sec and timeout_sec > 0:
        return ["timeout", "-k", str(KILL_GRACE_SEC), str(int(timeout_sec)), "/bin/sh", "-c", cmd]
    return ["/bin/sh", "-c", cmd]


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    image: str
    command: str
    setup_command: str | None = None
    setup_exit_code: int | None = None
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_sec": self.duration_sec,
            "image": self.image,
            "command": self.command,
            "setup_command": self.setup_command,
            "setup_exit_code": self.setup_exit_code,
            "timed_out": self.timed_out,
            "ok": self.ok,
        }


@dataclass
class _Step:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


class SandboxImageMissing(RuntimeError):
    """The language's sandbox image is not built (the CLI prints this without a traceback)."""


class SandboxRunner:
    """Spin up → mount copy → [setup with network → disconnect] → overlay → test offline → destroy.

    Always copies the project into a temporary directory before mounting so
    overlays never mutate the caller's tree.
    """

    def __init__(
        self,
        adapter: LanguageAdapter,
        *,
        client: docker.DockerClient | None = None,
    ) -> None:
        self.adapter = adapter
        self.client = client or docker.from_env()
        self.spec: SandboxSpec = adapter.sandbox_spec()

    def ensure_image(self) -> None:
        try:
            self.client.images.get(self.spec.image)
        except ImageNotFound as exc:
            build_hint = self.spec.dockerfile_dir or "docker/<lang>_sandbox"
            raise SandboxImageMissing(
                f"Sandbox image {self.spec.image!r} not found. "
                f"Build it with: docker build -t {self.spec.image} {build_hint} "
                "(or run bash scripts/bootstrap.sh, which builds all 8)"
            ) from exc

    def setup_command(self, project_dir: Path, relative_test: str | None) -> str | None:
        """The adapter's dependency step for this project (``setup_command_for``), if it has one."""
        fn = getattr(self.adapter, "setup_command_for", None)
        return fn(project_dir, relative_test) if callable(fn) else None

    def run_project(
        self,
        project_dir: Path,
        *,
        relative_test: str | None = None,
        overlay_file: Path | None = None,
        overlay_text: str | None = None,
        overlay_dest: str | None = None,
        command: str | None = None,
        setup_command: str | None = None,
        network_mode: str | None = None,
        timeout_sec: float | None = None,
    ) -> SandboxResult:
        """Run the project's tests (or ``command``) on a copy of ``project_dir`` with the overlay applied.

        ``setup_command`` runs first, with network (``network_mode``, default: the adapter's, else
        ``bridge``) and before the overlay is written; without one, the adapter's own setup step is
        used for test runs (``command`` unset). The test step never has network. ``timeout_sec``
        overrides the adapter's test time limit (≤ 0: none).
        """
        project_dir = project_dir.resolve()
        if not project_dir.is_dir():
            raise FileNotFoundError(f"Project directory not found: {project_dir}")
        if overlay_text is not None and not overlay_dest:
            raise ValueError("overlay_dest is required when overlay_text is set")
        if network_mode in ("host",) or (network_mode or "").startswith("container:"):
            raise ValueError(f"network_mode {network_mode!r} cannot be disconnected; use a bridge network")

        self.ensure_image()
        cmd = command or self.adapter.test_command_for(relative_test)
        if setup_command is None and command is None:
            setup_command = self.setup_command(project_dir, relative_test)
        setup_net = network_mode or (self.spec.network_mode if self.spec.network_mode != "none" else "bridge")
        limit = self.spec.timeout_sec if timeout_sec is None else timeout_sec

        overlays: list[tuple[str, bytes]] = []
        if overlay_file is not None:
            overlays.append((overlay_dest or overlay_file.name, Path(overlay_file).read_bytes()))
        if overlay_text is not None:
            overlays.append((str(overlay_dest), overlay_text.encode("utf-8")))

        with tempfile.TemporaryDirectory(prefix="rebirth-sandbox-") as tmp:
            work = Path(tmp) / "project"
            shutil.copytree(project_dir, work, ignore=_COPY_IGNORE)
            return self._run_mounted(work, cmd, setup=setup_command, setup_network=setup_net,
                                     overlays=overlays, timeout_sec=limit)

    # ------------------------------------------------------------------ container lifecycle

    def _exec(self, container: Container, cmd: str, timeout_sec: float | None, user: str | None) -> _Step:
        argv = timed(cmd, timeout_sec)
        t0 = time.perf_counter()
        pool = ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(container.exec_run, cmd=argv, workdir=self.spec.workdir, demux=True, user=user)
        try:
            res = fut.result(timeout=(timeout_sec + KILL_GRACE_SEC + HOST_GRACE_SEC) if timeout_sec and timeout_sec > 0
                             else None)
        except FuturesTimeout:
            # `timeout` inside did not end it (stuck in the kernel, or the image lost its `timeout`):
            # kill the container; the exec stream closes and the worker thread returns.
            try:
                container.kill()
            except Exception:
                pass
            pool.shutdown(wait=False, cancel_futures=True)
            return _Step(137, "", f"[sandbox] killed by the host after {int(timeout_sec or 0)} s", True)
        pool.shutdown(wait=False)
        code = int(res.exit_code if res.exit_code is not None else 1)
        out_b, err_b = res.output if res.output else (b"", b"")
        out = (out_b or b"").decode("utf-8", errors="replace")
        err = (err_b or b"").decode("utf-8", errors="replace")
        # 124: `timeout` stopped it; 137 (SIGKILL) after the limit: it ignored SIGTERM; 137 before: memory
        timed_out = code == TIMEOUT_EXIT or (
            code == 137 and bool(timeout_sec) and time.perf_counter() - t0 >= (timeout_sec or 0))
        if timed_out:
            err += f"\n[sandbox] timed out after {int(timeout_sec or 0)} s"
        elif code == 137:
            err += f"\n[sandbox] killed (SIGKILL), most likely out of memory (limit {container_limits()['mem_limit']})"
        return _Step(code, out, err, timed_out)

    def _disconnect(self, container: Container) -> None:
        container.reload()
        for net in list((container.attrs.get("NetworkSettings") or {}).get("Networks") or {}):
            self.client.api.disconnect_container_from_network(container.id, net, force=True)
        container.reload()
        left = (container.attrs.get("NetworkSettings") or {}).get("Networks") or {}
        if left:
            raise RuntimeError(f"sandbox: could not disconnect the container from {sorted(left)}; not running tests")

    def _run_mounted(
        self,
        work: Path,
        cmd: str,
        *,
        setup: str | None = None,
        setup_network: str = "bridge",
        overlays: list[tuple[str, bytes]] | None = None,
        timeout_sec: float | None = None,
    ) -> SandboxResult:
        started = time.perf_counter()
        container: Container | None = None
        setup_step: _Step | None = None
        try:
            run_user = self.spec.user or None
            container = self.client.containers.run(
                image=self.spec.image,
                command=["sleep", "infinity"],
                working_dir=self.spec.workdir,
                volumes={str(work): {"bind": self.spec.workdir, "mode": "rw"}},
                environment=dict(self.spec.env),
                network_mode=setup_network if setup else "none",
                detach=True,
                remove=False,
                user=run_user,
                **container_limits(),
            )
            if setup:
                setup_step = self._exec(container, setup, self.spec.setup_timeout_sec, run_user)
                self._disconnect(container)
            for dest, data in overlays or []:
                target = work / dest
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            step = self._exec(container, cmd, timeout_sec, run_user)
            exit_code, stdout, stderr, timed_out = step.exit_code, step.stdout, step.stderr, step.timed_out
            if setup_step is not None and setup_step.exit_code != 0 and exit_code != 0:
                tail = (setup_step.stdout + setup_step.stderr)[-2000:]
                stderr += f"\n[sandbox] setup step `{setup}` exited {setup_step.exit_code}:\n{tail}"
        except ContainerError as exc:
            exit_code, stdout, timed_out = int(exc.exit_status or 1), "", False
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace") if exc.stderr else str(exc)
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

        return SandboxResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_sec=time.perf_counter() - started,
            image=self.spec.image,
            command=cmd,
            setup_command=setup,
            setup_exit_code=None if setup_step is None else setup_step.exit_code,
            timed_out=timed_out,
        )
