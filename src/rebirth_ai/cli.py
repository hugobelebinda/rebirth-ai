"""CLI entrypoint for Rebirth AI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print
from rich.table import Table

from rebirth_ai import __version__

app = typer.Typer(help="Rebirth AI — autonomous migration agent", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print package version."""
    print(f"rebirth-ai {__version__}")


@app.command()
def doctor() -> None:
    """Check local prerequisites (Docker, Ollama, sandbox image, Python)."""
    import shutil
    import subprocess
    import sys

    print(f"[bold]Python[/bold]: {sys.version.split()[0]}")
    print(f"[bold]docker[/bold]: {'ok' if shutil.which('docker') else 'MISSING'}")
    print(f"[bold]ollama[/bold]: {'ok' if shutil.which('ollama') else 'MISSING'}")
    print(f"[bold]node[/bold]: {'ok' if shutil.which('node') else 'MISSING'}")

    try:
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        print(f"[bold]Docker daemon[/bold]: {'ok' if r.returncode == 0 else 'NOT RUNNING'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[bold]Docker daemon[/bold]: error ({exc})")

    try:
        import docker
        from docker.errors import ImageNotFound

        from rebirth_ai.adapters.js_ts import DEFAULT_IMAGE

        client = docker.from_env()
        try:
            client.images.get(DEFAULT_IMAGE)
            print(f"[bold]Sandbox image[/bold]: ok ({DEFAULT_IMAGE})")
        except ImageNotFound:
            print(f"[bold]Sandbox image[/bold]: MISSING ({DEFAULT_IMAGE})")
    except Exception as exc:  # noqa: BLE001
        print(f"[bold]Sandbox image[/bold]: error ({exc})")

    try:
        import httpx

        resp = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5.0)
        if resp.status_code == 200:
            models = [m.get("name") for m in resp.json().get("models", [])]
            print(f"[bold]Ollama[/bold]: ok ({len(models)} models)")
            for name in models[:8]:
                print(f"  - {name}")
        else:
            print(f"[bold]Ollama[/bold]: HTTP {resp.status_code}")

        ps = httpx.get("http://127.0.0.1:11434/api/ps", timeout=5.0)
        if ps.status_code == 200:
            loaded = ps.json().get("models", [])
            if not loaded:
                print("[bold]Ollama loaded[/bold]: (none)")
            for m in loaded:
                size = m.get("size") or 0
                vram = m.get("size_vram") or 0
                pct = (100.0 * vram / size) if size else 0.0
                where = "GPU" if vram and pct >= 99 else ("mixed" if vram else "CPU")
                print(
                    f"[bold]Ollama loaded[/bold]: {m.get('name')} "
                    f"{vram/1e9:.1f}/{size/1e9:.1f}GB → {where} ({pct:.0f}% VRAM)"
                )
    except Exception as exc:  # noqa: BLE001
        print(f"[bold]Ollama[/bold]: not reachable ({exc})")


@app.command("parse")
def parse_file(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
) -> None:
    """AST-chunk a JS/TS source file and print structured chunks."""
    from rebirth_ai.adapters.js_ts import JsTsAdapter

    adapter = JsTsAdapter()
    if not adapter.is_source_file(path):
        raise typer.BadParameter(f"Unsupported source suffix: {path.suffix}")
    chunks = adapter.parse(path)
    table = Table(title=f"Chunks: {path.name}")
    table.add_column("kind")
    table.add_column("name")
    table.add_column("lines")
    for c in chunks:
        table.add_row(c.kind, c.name, f"{c.start_line}-{c.end_line}")
    print(table)
    print(f"[dim]{len(chunks)} chunk(s)[/dim]")


@app.command("sandbox")
def sandbox_run(
    project: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Project root to mount into the sandbox",
    ),
    test: str | None = typer.Option(
        None,
        "--test",
        "-t",
        help="Container-relative test path (e.g. test/math.test.js)",
    ),
) -> None:
    """Run fixture/project tests inside the ephemeral Docker sandbox."""
    from rebirth_ai.adapters.js_ts import JsTsAdapter
    from rebirth_ai.sandbox.runner import SandboxRunner

    runner = SandboxRunner(JsTsAdapter())
    result = runner.run_project(project, relative_test=test)
    status = "[green]PASS[/green]" if result.ok else "[red]FAIL[/red]"
    print(f"{status} exit={result.exit_code} in {result.duration_sec:.2f}s")
    print(f"[dim]cmd:[/dim] {result.command}")
    if result.stdout:
        print("[bold]stdout[/bold]")
        print(result.stdout)
    if result.stderr:
        print("[bold]stderr[/bold]")
        print(result.stderr)
    raise typer.Exit(code=0 if result.ok else 1)


if __name__ == "__main__":
    app()
