"""Ephemeral Docker sandbox execution via the Docker Python SDK."""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from docker.errors import ContainerError, ImageNotFound
from docker.models.containers import Container

import docker
from rebirth_ai.adapters.base import LanguageAdapter, SandboxSpec


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    image: str
    command: str

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
            "ok": self.ok,
        }


class SandboxRunner:
    """Spin up → mount copy → run tests → capture logs → destroy.

    Always copies the project into a temporary directory before mounting so
    overlays never mutate the caller's tree (bind-mount put_archive would).
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
            raise RuntimeError(
                f"Sandbox image {self.spec.image!r} not found. "
                f"Build it with: docker build -t {self.spec.image} {build_hint}"
            ) from exc

    def run_project(
        self,
        project_dir: Path,
        *,
        relative_test: str | None = None,
        overlay_file: Path | None = None,
        overlay_text: str | None = None,
        overlay_dest: str | None = None,
        command: str | None = None,
        network_mode: str | None = None,
    ) -> SandboxResult:
        project_dir = project_dir.resolve()
        if not project_dir.is_dir():
            raise FileNotFoundError(f"Project directory not found: {project_dir}")

        self.ensure_image()
        cmd = command or self.adapter.test_command_for(relative_test)
        # Use non-login shell so image ENV PATH (e.g. /usr/local/go/bin) is kept.
        shell_cmd = ["/bin/sh", "-c", cmd]
        net = network_mode if network_mode is not None else (self.spec.network_mode or "none")

        with tempfile.TemporaryDirectory(prefix="rebirth-sandbox-") as tmp:
            work = Path(tmp) / "project"
            shutil.copytree(
                project_dir,
                work,
                ignore=shutil.ignore_patterns(
                    "node_modules",
                    ".git",
                    ".venv",
                    "venv",
                    "__pycache__",
                    ".pytest_cache",
                    "target",
                    "build",
                    ".build",
                    "bin",
                    "obj",
                ),
            )
            if overlay_file is not None:
                dest = overlay_dest or overlay_file.name
                target = work / dest
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(overlay_file, target)
            if overlay_text is not None:
                if not overlay_dest:
                    raise ValueError("overlay_dest is required when overlay_text is set")
                target = work / overlay_dest
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(overlay_text, encoding="utf-8")

            return self._run_mounted(work, shell_cmd, cmd, network_mode=net)

    def _run_mounted(
        self,
        work: Path,
        shell_cmd: list[str],
        cmd: str,
        *,
        network_mode: str = "none",
    ) -> SandboxResult:
        started = time.perf_counter()
        container: Container | None = None
        try:
            run_user = self.spec.user or None
            container = self.client.containers.run(
                image=self.spec.image,
                command=["sleep", "infinity"],
                working_dir=self.spec.workdir,
                volumes={
                    str(work): {
                        "bind": self.spec.workdir,
                        "mode": "rw",
                    }
                },
                environment=dict(self.spec.env),
                network_mode=network_mode,
                detach=True,
                remove=False,
                user=run_user,
            )
            exec_result = container.exec_run(
                cmd=shell_cmd,
                workdir=self.spec.workdir,
                demux=True,
                user=run_user,
            )
            exit_code = int(exec_result.exit_code)
            out_b, err_b = exec_result.output if exec_result.output else (b"", b"")
            stdout = (out_b or b"").decode("utf-8", errors="replace")
            stderr = (err_b or b"").decode("utf-8", errors="replace")
        except ContainerError as exc:
            exit_code = int(exc.exit_status or 1)
            stdout = ""
            stderr = (
                (exc.stderr or b"").decode("utf-8", errors="replace") if exc.stderr else str(exc)
            )
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
        )
