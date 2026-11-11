"""The start screen: ``rebirth`` with no command.

It shows the logo, what Rebirth does and whether this machine is ready (Ollama, Docker, the two model
packs). In a terminal it then asks what you want to do, prints the matching ``rebirth …`` command so you learn
it, and runs it. Piped or redirected, it only prints the screen.
"""

from __future__ import annotations

import shlex
import sys
from collections.abc import Callable
from pathlib import Path

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from rebirth_ai import __version__

REPO = "github.com/hugobelebinda/rebirth-ai"

# "ANSI Shadow" block letters: solid blocks for the face, box-drawing lines for the shadow.
_GLYPHS = {
    "R": ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔══██╗", "██║  ██║", "╚═╝  ╚═╝"],
    "E": ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"],
    "B": ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔══██╗", "██████╔╝", "╚═════╝ "],
    "I": ["██╗", "██║", "██║", "██║", "██║", "╚═╝"],
    "T": ["████████╗", "╚══██╔══╝", "   ██║   ", "   ██║   ", "   ██║   ", "   ╚═╝   "],
    "H": ["██╗  ██╗", "██║  ██║", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"],
}
LOGO = ["".join(_GLYPHS[c][row] for c in "REBIRTH") for row in range(6)]

# Phoenix gradient, top to bottom: gold → orange → ember red. The shadow is the same hue, darker.
_FACE = ["#FFD54A", "#FFB33A", "#FF922E", "#F9712C", "#EB5232", "#D23A3A"]
_SHADOW = ["#8C6A10", "#8C5A12", "#8A4614", "#863516", "#7C2619", "#6E1C1F"]
ACCENT = "#F9712C"
VALUE = "#8BD49B"

ACTIONS = [
    ("modernize", "bring a file or a whole project up to current idioms"),
    ("translate", "rewrite a file in another of the 8 languages"),
    ("doctor", "check Docker, Ollama, the models and the sandboxes"),
    ("help", "every command and option"),
]
TO_LANGS = "python, javascript, typescript, go, java, csharp, rust, cpp, swift"


class Cancelled(Exception):
    """Ctrl-C or end of input at a prompt."""


def logo() -> Text:
    out = Text()
    for row, line in enumerate(LOGO):
        if row:
            out.append("\n")
        for ch in line:
            out.append(ch, style=f"bold {_FACE[row]}" if ch == "█" else _SHADOW[row])
    return out


def info_table() -> Table:
    table = Table(box=box.SQUARE, border_style=ACCENT, show_header=False, padding=(0, 1), expand=False)
    cells = [
        ("languages", "JS/TS · Python · Go · Java", "C# · Rust · C++ · Swift"),
        ("AI backend", "Ollama (local)", "no cloud, no keys"),
        ("safety", "Docker sandbox, offline", "tests check every draft"),
    ]
    for _ in cells:
        table.add_column()
    table.add_row(*(Text.assemble((label, "bold white"), "\n", (a, VALUE), "\n", (b, VALUE)) for label, a, b in cells))
    return table


def machine_status(timeout: float = 1.5) -> list[tuple[str, bool]]:
    """(label, ok) for Ollama, Docker and the two model packs; fast, and never raises."""
    rows: list[tuple[str, bool]] = []
    names: set[str] | None = None
    try:
        import httpx

        from rebirth_ai.generate.ollama_client import ollama_host

        resp = httpx.get(f"{ollama_host()}/api/tags", timeout=timeout)
        resp.raise_for_status()
        names = {m.get("name", "") for m in resp.json().get("models", [])}
    except Exception:
        pass
    rows.append(("Ollama", names is not None))
    try:
        import docker

        docker.from_env(timeout=int(max(1, timeout))).ping()
        rows.append(("Docker", True))
    except Exception:
        rows.append(("Docker", False))
    try:
        from rebirth_ai.install import status

        packs = status(has=lambda tag: names is not None and (tag in names or f"{tag}:latest" in names))
        for pack, s in packs.items():
            rows.append((f"{pack} {s['installed']}/{s['total']}", s["installed"] == s["total"]))
    except Exception:
        pass
    return rows


def status_line(rows: list[tuple[str, bool]]) -> Text:
    out = Text("  ")
    for label, ok in rows:
        out.append("✓ " if ok else "✗ ", style="bold green" if ok else "bold red")
        out.append(label, style="white" if ok else "red")
        out.append("   ")
    return out


def render(console: Console, *, rows: list[tuple[str, bool]] | None = None, menu: bool = True) -> None:
    """The screen: logo, the three boxes, tagline, machine status and (in a terminal) the menu."""
    console.print()
    if console.width >= 60:
        console.print(Padding(logo(), (0, 0, 0, 2)))
    else:
        console.print(Text("  REBIRTH", style=f"bold {ACCENT}"))
    console.print()
    if console.width >= 80:
        console.print(Padding(info_table(), (0, 0, 0, 2)))
    else:
        console.print(f"  [{VALUE}]8 languages · local models in Ollama · checked by your tests[/]")
    console.print()
    console.print("  [bold white]legacy code, modernized or translated, and checked by your own tests[/]")
    console.print(f"  [dim]v{__version__}   ·   offline   ·   {REPO}[/dim]")
    if rows is not None:
        console.print()
        console.print(status_line(rows))
        if not all(ok for _, ok in rows):
            console.print("  [dim]`rebirth doctor` says how to fix what is missing.[/dim]")
    if menu:
        console.print()
        for i, (name, what) in enumerate(ACTIONS, 1):
            console.print(f"  [bold {ACCENT}]{i}[/]  [bold]{name:<10}[/] [dim]{what}[/dim]")
    console.print()


def ask(console: Console, question: str, default: str = "") -> str:
    """``Question [default]: `` like a shell installer; Enter keeps the default."""
    try:
        answer = console.input(f"  [bold]{escape(question)}[/bold] {escape(f'[{default}]')}: ")
    except (EOFError, KeyboardInterrupt) as exc:
        raise Cancelled from exc
    return answer.strip() or default


def ask_yes_no(console: Console, question: str, default: bool = False) -> bool:
    while True:
        answer = ask(console, f"{question} (y/N)" if not default else f"{question} (Y/n)", "y" if default else "N")
        if answer.lower() in {"y", "yes"}:
            return True
        if answer.lower() in {"n", "no"}:
            return False
        console.print("  [red]Answer y or n.[/red]")


def pick_action(answer: str) -> str | None:
    a = answer.strip().lower()
    if a.isdigit() and 1 <= int(a) <= len(ACTIONS):
        return ACTIONS[int(a) - 1][0]
    matches = [name for name, _ in ACTIONS if a and name.startswith(a)]
    return matches[0] if len(matches) == 1 else None


def _ask_project(console: Console) -> Path:
    while True:
        project = Path(ask(console, "Project folder", ".")).expanduser().resolve()
        if project.is_dir():
            return project
        console.print(f"  [red]No such folder: {escape(str(project))}[/red]")


def _ask_file(console: Console, project: Path, question: str, *, required: bool) -> str | None:
    while True:
        answer = ask(console, question, "")
        if not answer:
            if not required:
                return None
            console.print("  [red]Name a file in the project, for example src/app.py.[/red]")
            continue
        path = Path(answer).expanduser()
        path = path if path.is_absolute() else project / path
        try:
            rel = path.resolve().relative_to(project)
        except ValueError:
            console.print(f"  [red]{escape(answer)} is not inside {escape(str(project))}.[/red]")
            continue
        if (project / rel).is_file():
            return rel.as_posix()
        console.print(f"  [red]No such file: {escape(str(project / rel))}[/red]")


def _ask_target(console: Console) -> str:
    from rebirth_ai.xlang.languages import get_lang

    while True:
        answer = ask(console, "Translate to", "")
        try:
            get_lang(answer)
            return answer.strip().lower()
        except ValueError:
            console.print(f"  [red]Pick one of: {TO_LANGS}.[/red]")


def wizard(console: Console) -> list[str]:
    """Ask what to do and return the ``rebirth`` arguments for it."""
    while True:
        action = pick_action(ask(console, "What do you want to do", "modernize"))
        if action:
            break
        console.print("  [red]Type a number from 1 to 4, or a name from the list.[/red]")
    if action == "doctor":
        return ["doctor"]
    if action == "help":
        return ["--help"]
    project = _ask_project(console)
    if action == "translate":
        file = _ask_file(console, project, "File to translate", required=True)
        target = _ask_target(console)
        console.print(f"  [dim]The translation goes to {escape(str(project / 'rebirth_out' / target))}/; "
                      "the source file is not changed.[/dim]")
        return ["migrate", str(project), "--file", file or "", "--to", target]
    file = _ask_file(console, project, "File (Enter = the whole project)", required=False)
    write = ask_yes_no(console, "Write the changes into your files?", False)
    args = ["migrate", str(project), "--file", file] if file else ["migrate-repo", str(project)]
    return [*args, "--write"] if write else args


def run_args(args: list[str]) -> int:
    """Run ``rebirth <args>`` in this process and return its exit code."""
    from typer.main import get_command

    from rebirth_ai.cli import app

    try:
        result = get_command(app).main(args=args, prog_name="rebirth", standalone_mode=False)
    except Exception as exc:
        # click's Abort / UsageError; matched by shape, since Typer may ship its own copy of click
        if type(exc).__name__ == "Abort":
            return 130
        if hasattr(exc, "show") and isinstance(getattr(exc, "exit_code", None), int):
            exc.show()
            return exc.exit_code
        raise
    return result if isinstance(result, int) else 0


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def start(console: Console | None = None, *, run: Callable[[list[str]], int] = run_args) -> int:
    console = console or Console(highlight=False)
    tty = interactive()
    render(console, rows=machine_status(), menu=tty)
    if not tty:
        console.print("  Run [bold]rebirth --help[/bold] for every command.")
        return 0
    try:
        args = wizard(console)
    except Cancelled:
        console.print("\n  [dim]Cancelled.[/dim]")
        return 130
    console.print()
    console.print(f"  [bold {ACCENT}]→[/] [bold]rebirth {escape(shlex.join(args))}[/bold]")
    console.print()
    return run(args)
