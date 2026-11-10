"""Health per language / direction and per model version, from local run logs.

``summarize`` aggregates; ``render_html`` writes one self-contained HTML file (no external
assets, no scripts — it opens offline); ``render_text`` is the terminal view.
"""

from __future__ import annotations

import html
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from rebirth_ai.monitor.store import Record


def wilson(successes: int, n: int, z: float = 1.645) -> tuple[float, float]:
    """90 % Wilson score interval for a success rate (sane for small n and 0 / 100 %)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _group(records: list[Record]) -> dict[str, Any]:
    n = len(records)
    ok = sum(1 for r in records if r.t.get("success"))
    iters = [r.t["iterations"] for r in records if isinstance(r.t.get("iterations"), int)]
    walls = [r.t["wall_seconds"] for r in records if isinstance(r.t.get("wall_seconds"), (int, float))]
    classes = Counter(c for r in records for c in r.t.get("error_classes") or [])
    lo, hi = wilson(ok, n)
    return {
        "runs": n,
        "success": ok,
        "success_rate": round(ok / n, 3) if n else None,
        "ci90": [round(lo, 3), round(hi, 3)],
        "mean_iterations": round(sum(iters) / len(iters), 2) if iters else None,
        "median_wall_seconds": sorted(walls)[len(walls) // 2] if walls else None,
        "top_error_classes": classes.most_common(3),
        "last_run": max((r.ts for r in records), default=None),
    }


def summarize(records: list[Record]) -> dict[str, Any]:
    by_key: dict[str, list[Record]] = defaultdict(list)
    by_key_model: dict[tuple[str, str], list[Record]] = defaultdict(list)
    by_task: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_key[r.key].append(r)
        by_key_model[(r.key, r.t.get("model") or "?")].append(r)
        by_task[r.t.get("task") or "?"].append(r)
    return {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "runs": len(records),
        "overall": _group(records),
        "by_task": {k: _group(v) for k, v in sorted(by_task.items())},
        "by_key": {k: _group(v) for k, v in sorted(by_key.items())},
        "by_key_model": [
            {"key": k, "model": m, **_group(v), "first_run": min(r.ts for r in v)}
            for (k, m), v in sorted(by_key_model.items(), key=lambda kv: (kv[0][0], min(r.ts for r in kv[1])))
        ],
    }


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.0f} %"


def render_text(summary: dict[str, Any]) -> str:
    lines = [f"{summary['runs']} run(s); overall success {_pct(summary['overall']['success_rate'])}", ""]
    lines.append(f"{'task:language / direction':34s} {'runs':>5s} {'success':>8s} {'90% CI':>12s} {'iters':>6s}  top error classes")
    for key, g in summary["by_key"].items():
        ci = f"{100 * g['ci90'][0]:.0f}–{100 * g['ci90'][1]:.0f} %"
        top = ", ".join(f"{c} ×{n}" for c, n in g["top_error_classes"]) or "—"
        it = "—" if g["mean_iterations"] is None else f"{g['mean_iterations']:.1f}"
        lines.append(f"{key:34s} {g['runs']:5d} {_pct(g['success_rate']):>8s} {ci:>12s} {it:>6s}  {top}")
    return "\n".join(lines)


_CSS = """
.viz-root { color-scheme: light; --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --grid:#e1e0d9; --meter:#2a78d6; --meter-track:#cde2fb; --warning:#fab219; }
@media (prefers-color-scheme: dark) {
  .viz-root { color-scheme: dark; --surface-1:#1a1a19; --text-primary:#f0efec; --text-secondary:#c3c2b7;
    --grid:#2c2c2a; --meter:#3987e5; --meter-track:#104281; }
}
body { margin: 0; }
.viz-root { font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--text-primary);
  background: var(--surface-1); min-height: 100vh; padding: 2rem max(1rem, calc(50vw - 560px)); box-sizing: border-box; }
h1 { font-size: 1.4rem; margin: 0 0 .25rem; } h2 { font-size: 1.05rem; margin: 2rem 0 .5rem; }
p.meta { color: var(--text-secondary); margin: 0 0 1.5rem; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: .4rem .6rem; border-bottom: 1px solid var(--grid); vertical-align: middle; }
th { color: var(--text-secondary); font-weight: 600; font-size: .85rem; } td.num, th.num { text-align: right; }
td.num, td:first-child, code { white-space: nowrap; }
.meter { height: 8px; width: 140px; background: var(--meter-track); border-radius: 4px; overflow: hidden; }
.meter > span { display: block; height: 100%; background: var(--meter); border-radius: 4px; }
.tiles { display: flex; gap: 1rem; flex-wrap: wrap; }
.tile { border: 1px solid var(--grid); border-radius: 6px; padding: .6rem .9rem; min-width: 160px; }
.tile .label { color: var(--text-secondary); font-size: .85rem; } .tile .value { display: block; font-size: 1.5rem; font-weight: 600; }
ul.warnings { list-style: none; padding: 0; } ul.warnings li { margin: .3rem 0; }
.status { display: inline-block; width: .7em; height: .7em; border-radius: 50%; margin-right: .4em; vertical-align: baseline; }
.status.warn { background: var(--warning); } .status.info { background: var(--text-secondary); }
code { font-size: .9em; }
"""


def _bar(g: dict[str, Any]) -> str:
    """Thin meter: blue fill on a lighter blue track; exact rate + 90% CI on hover (title)."""
    if g["success_rate"] is None:
        return ""
    lo, hi = g["ci90"]
    tip = f"{g['success']} of {g['runs']} succeeded: {_pct(g['success_rate'])} (90% CI {_pct(lo)}–{_pct(hi)})"
    return f'<div class="meter" title="{tip}"><span style="width:{100 * g["success_rate"]:.1f}%"></span></div>'


def _row(label: str, g: dict[str, Any], extra: str = "") -> str:
    top = ", ".join(f"{html.escape(c)} ×{n}" for c, n in g["top_error_classes"]) or "—"
    it = "—" if g["mean_iterations"] is None else f"{g['mean_iterations']:.1f}"
    wall = "—" if g["median_wall_seconds"] is None else f"{g['median_wall_seconds']:.0f} s"
    ci = "—" if g["success_rate"] is None else f"{100 * g['ci90'][0]:.0f}–{100 * g['ci90'][1]:.0f} %"
    return (f"<tr><td>{label}</td>{extra}<td class=num>{g['runs']}</td><td class=num>{_pct(g['success_rate'])}</td>"
            f"<td>{_bar(g)}</td><td class=num>{ci}</td><td class=num>{it}</td><td class=num>{wall}</td><td>{top}</td></tr>")


def render_html(summary: dict[str, Any], warnings: list[dict[str, Any]] | None = None) -> str:
    head = ("<th class=num>runs</th><th class=num>success</th><th></th><th class=num>90% CI</th>"
            "<th class=num>mean iterations</th><th class=num>median time</th><th>top error classes</th>")
    tiles = "".join(
        f'<div class="tile"><span class="label">{html.escape(t)} success · {g["runs"]} run(s)</span>'
        f'<span class="value">{_pct(g["success_rate"])}</span></div>'
        for t, g in summary["by_task"].items()
    )
    warn_html = ""
    if warnings:
        items = "".join(
            f'<li><span class="status {"warn" if w.get("level") == "warn" else "info"}"></span>'
            f'<b>{"Warning" if w.get("level") == "warn" else "Note"}:</b> {html.escape(w["message"])}</li>'
            for w in warnings)
        warn_html = f'<h2>Warnings</h2><ul class="warnings">{items}</ul>'
    keys = "".join(_row(html.escape(k), g) for k, g in summary["by_key"].items())
    models = "".join(_row(html.escape(g["key"]), g, f"<td><code>{html.escape(g['model'])}</code></td>")
                     for g in summary["by_key_model"])
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Rebirth — run health</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>{_CSS}</style></head><body><div class="viz-root">
<h1>Rebirth — run health</h1>
<p class="meta">{summary['runs']} run(s) from local run logs · generated {html.escape(summary['generated'])} · nothing here left this machine</p>
<div class="tiles">{tiles}</div>
{warn_html}
<h2>Per language (modernize) and direction (translate)</h2>
<table><tr><th>task : language / direction</th>{head}</tr>{keys}</table>
<h2>Per model version</h2>
<table><tr><th>task : language / direction</th><th>model</th>{head}</tr>{models}</table>
</div></body></html>
"""
