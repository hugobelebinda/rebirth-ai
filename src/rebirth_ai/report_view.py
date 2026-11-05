"""Pretty-print batch / eval / run JSON (``rebirth report``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from rebirth_ai.redact import redact_obj


def load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return redact_obj(data)


def render_report(data: dict[str, Any], *, console: Console | None = None) -> None:
    con = console or Console()
    # Batch migrate-repo shape
    if "results" in data and "summary" in data:
        _render_batch(data, con)
        return
    # Eval behavioral shape
    if "suite" in data or "cases" in data or "sandbox" in data:
        _render_eval(data, con)
        return
    # Run summary
    if "payload" in data and "kind" in data:
        con.print(f"[bold]run[/bold] {data.get('run_id')} kind={data.get('kind')}")
        render_report(data.get("payload") or {}, console=con)
        return
    con.print_json(data=data)


def _render_batch(data: dict[str, Any], con: Console) -> None:
    summary = data.get("summary") or {}
    con.print(f"[bold]Project[/bold]: {data.get('project_dir')}")
    con.print(f"[bold]Objective[/bold]: {data.get('objective')}")
    if data.get("dry_run"):
        con.print("[yellow]Mode: dry-run[/yellow]")
    con.print(
        f"success={summary.get('success')} failed={summary.get('failed')} "
        f"skipped={summary.get('skipped')} planned={summary.get('planned')} "
        f"total={summary.get('total')}"
    )
    table = Table(title="Results")
    table.add_column("file")
    table.add_column("status")
    table.add_column("iters")
    table.add_column("error")
    for row in data.get("results") or []:
        table.add_row(
            str(row.get("relative_file") or ""),
            str(row.get("status") or ""),
            str(row.get("iterations") or 0),
            str(row.get("error_class") or ""),
        )
    con.print(table)


def _render_eval(data: dict[str, Any], con: Console) -> None:
    con.print(f"[bold]Eval[/bold] backend={data.get('backend')} suite={data.get('suite')}")
    if "summary" in data:
        con.print(data["summary"])
    cases = data.get("cases") or data.get("results") or []
    if not cases and isinstance(data.get("sandbox"), dict):
        cases = (data["sandbox"].get("cases") or [])
    if cases:
        table = Table(title="Cases")
        table.add_column("id")
        table.add_column("ok")
        table.add_column("detail")
        for row in cases:
            if not isinstance(row, dict):
                continue
            ok = row.get("ok")
            if ok is None:
                ok = row.get("sandbox_pass")
            table.add_row(
                str(row.get("id") or row.get("name") or ""),
                str(ok),
                str(row.get("status") or row.get("error") or "")[:80],
            )
        con.print(table)
