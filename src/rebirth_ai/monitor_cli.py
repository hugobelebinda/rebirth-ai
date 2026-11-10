"""Phase 18 commands: ``rebirth dashboard``, ``rebirth monitor …``, ``rebirth review …`` (all local)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print

monitor_app = typer.Typer(help="Regression + drift checks over local run logs")
review_app = typer.Typer(help="Data flywheel: review failed / hard migrations into curated training rows")

_RUNS = typer.Option(None, "--runs-dir", help="default: data/runs (REBIRTH_RUNS_DIR)")


def _warnings(records, ref_dir: Path, min_runs: int, margin: float) -> list[dict]:
    from rebirth_ai.monitor import alerts, drift

    return alerts.check(records, min_runs=min_runs, margin=margin) + drift.check(records, ref_dir=ref_dir)


def dashboard_cmd(
    out: Path = typer.Option(Path("data/reports/dashboard.html"), "--out", help="static HTML report"),
    runs_dir: Path | None = _RUNS,
    ref_dir: Path = typer.Option(Path("data/monitor/reference"), "--ref-dir"),
    min_runs: int = typer.Option(10, "--min-runs", help="regression check: runs per model before comparing"),
    margin: float = typer.Option(10.0, "--margin", help="regression check: success drop in points"),
) -> None:
    """Run health per language / direction and model version → terminal table + one offline HTML file."""
    from rebirth_ai.monitor.dashboard import render_html, render_text, summarize
    from rebirth_ai.monitor.store import load_records

    records = load_records(runs_dir)
    summary = summarize(records)
    warns = _warnings(records, ref_dir, min_runs, margin)
    print(render_text(summary))
    for w in warns:
        print(("[yellow]warning[/yellow] " if w["level"] == "warn" else "[dim]note[/dim] ") + w["message"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(summary, warns), encoding="utf-8")
    print(f"[green]wrote[/green] {out}")


@monitor_app.command("check")
def monitor_check(
    runs_dir: Path | None = _RUNS,
    ref_dir: Path = typer.Option(Path("data/monitor/reference"), "--ref-dir"),
    min_runs: int = typer.Option(10, "--min-runs"),
    margin: float = typer.Option(10.0, "--margin"),
) -> None:
    """Regression + drift warnings; exit code 1 if any ``warn``-level finding."""
    from rebirth_ai.monitor.store import load_records

    warns = _warnings(load_records(runs_dir), ref_dir, min_runs, margin)
    for w in warns:
        print(("[yellow]warning[/yellow] " if w["level"] == "warn" else "[dim]note[/dim] ") + w["message"])
    if not warns:
        print("[green]no regression or drift findings[/green]")
    raise typer.Exit(1 if any(w["level"] == "warn" for w in warns) else 0)


@monitor_app.command("reference")
def monitor_reference(
    lang: list[str] = typer.Option(None, "--lang", help="default: all 8"),
    out_dir: Path = typer.Option(Path("data/monitor/reference"), "--out-dir"),
) -> None:
    """Build the training-input profiles drift is measured against (corpus + stage-3 pool)."""
    from rebirth_ai.monitor import drift

    for lg in lang or drift.LANGS:
        path = drift.build_reference(lg, out_dir=out_dir)
        import json

        prof = json.loads(path.read_text(encoding="utf-8"))
        print(f"{lg:7s} {prof['rows']:6d} inputs, median {prof['length_quantiles']['0.5']} chars, "
              f"{len(prof['imports'])} imports → {path}")


@review_app.command("list")
def review_list(runs_dir: Path | None = _RUNS, hard_iterations: int = typer.Option(3, "--hard-iterations")) -> None:
    """Runs waiting for review (failed, or ≥ N attempts)."""
    from rebirth_ai.monitor.flywheel import queue

    q = queue(runs_dir, hard_iterations=hard_iterations)
    if not q:
        print("[green]review queue is empty[/green]")
    for r in q:
        classes = ",".join(r.t.get("error_classes") or []) or "—"
        print(f"{r.run_id}  {r.t.get('direction'):22s} {r.t.get('status'):12s} iters={r.t.get('iterations')}  {classes}")


@review_app.command("show")
def review_show(run_id: str, runs_dir: Path | None = _RUNS) -> None:
    """Input and final draft of one run (redacted copies)."""
    from rebirth_ai.monitor.flywheel import artifacts

    for role, text in artifacts(run_id, runs_dir).items():
        print(f"[bold]── {role} ──[/bold]")
        print(text.replace("[", "\\["))


@review_app.command("accept")
def review_accept(
    run_id: str,
    own: bool = typer.Option(False, "--own", help="confirm the code is yours to train on (required)"),
    fixed: Path | None = typer.Option(None, "--fixed", exists=True, dir_okay=False, help="your corrected version"),
    runs_dir: Path | None = _RUNS,
) -> None:
    """Add the run (model draft, or your --fixed version) to the curated flywheel rows."""
    from rebirth_ai.monitor.flywheel import accept

    if not own:
        own = typer.confirm("Is this code yours (license-clean) to use as training data?", default=False)
    try:
        row = accept(run_id, own=own, fixed=fixed.read_text(encoding="utf-8") if fixed else None, root=runs_dir)
    except (PermissionError, ValueError, KeyError) as exc:
        print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    print(f"[green]accepted[/green] {run_id} → {row['split']} row")


@review_app.command("reject")
def review_reject(run_id: str, reason: str = typer.Option("", "--reason"), runs_dir: Path | None = _RUNS) -> None:
    """Mark a run reviewed without adding it."""
    from rebirth_ai.monitor.flywheel import reject

    try:
        reject(run_id, reason=reason, root=runs_dir)
    except (ValueError, KeyError) as exc:
        print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    print(f"rejected {run_id}")


@review_app.command("export-eval")
def review_export_eval(max_rows: int = typer.Option(100, "--max-rows")) -> None:
    """Rolling real-world eval set from reviewed eval-split rows (frozen sets are untouched)."""
    from rebirth_ai.monitor.flywheel import export_rolling_eval

    counts = export_rolling_eval(max_rows=max_rows)
    print(counts or "no eval-split rows yet")
