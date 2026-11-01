"""CLI entrypoint (stubs until Phase 1+)."""

from __future__ import annotations

import typer
from rich import print

from rebirth_ai import __version__

app = typer.Typer(help="Rebirth AI — autonomous migration agent", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print package version."""
    print(f"rebirth-ai {__version__}")


@app.command()
def doctor() -> None:
    """Check local prerequisites (Docker, Ollama, Python). Expanded in Phase 1."""
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
        import httpx

        resp = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5.0)
        if resp.status_code == 200:
            models = [m.get("name") for m in resp.json().get("models", [])]
            print(f"[bold]Ollama[/bold]: ok ({len(models)} models)")
            for name in models[:8]:
                print(f"  - {name}")
        else:
            print(f"[bold]Ollama[/bold]: HTTP {resp.status_code}")
    except Exception as exc:  # noqa: BLE001
        print(f"[bold]Ollama[/bold]: not reachable ({exc})")


if __name__ == "__main__":
    app()
