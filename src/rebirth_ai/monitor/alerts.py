"""Regression alerts: does the newest model version do worse than the one before it?

For every task + language / direction (and for each task overall), runs are grouped by the model
that served them, in the order the models first appeared. The newest model is compared with the
previous one once both have ``min_runs`` runs:

- ``warn``  success rate dropped by ≥ ``margin`` points **and** the 90 % Wilson intervals do not
  overlap (a drop that is not just noise), or the drop is ≥ 2 × ``margin`` points;
- ``info``  dropped by ≥ ``margin`` points but the intervals overlap (watch it), or the newest model
  has fewer than ``min_runs`` runs yet (nothing to say);
- mean iterations up by ≥ 1 is reported as ``info`` (slower to converge, same success).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rebirth_ai.monitor.dashboard import wilson
from rebirth_ai.monitor.store import Record


def _stats(rs: list[Record]) -> dict[str, Any]:
    n = len(rs)
    ok = sum(1 for r in rs if r.t.get("success"))
    iters = [r.t["iterations"] for r in rs if isinstance(r.t.get("iterations"), int)]
    return {"n": n, "ok": ok, "rate": ok / n if n else 0.0, "ci": wilson(ok, n),
            "iters": sum(iters) / len(iters) if iters else None}


def _compare(scope: str, rs: list[Record], *, min_runs: int, margin: float) -> list[dict[str, Any]]:
    by_model: dict[str, list[Record]] = defaultdict(list)
    for r in rs:
        by_model[r.t.get("model") or "?"].append(r)
    order = sorted(by_model, key=lambda m: min(r.ts for r in by_model[m]))
    if len(order) < 2:
        return []
    old_m, new_m = order[-2], order[-1]
    old, new = _stats(by_model[old_m]), _stats(by_model[new_m])
    base = {"scope": scope, "old_model": old_m, "new_model": new_m, "old": old, "new": new}
    if old["n"] < min_runs:
        return []
    if new["n"] < min_runs:
        return [base | {"level": "info", "kind": "too_few_runs",
                        "message": f"{scope}: {new_m} has {new['n']} run(s) so far (< {min_runs}); compared with {old_m} once it has more"}]
    out = []
    drop = 100 * (old["rate"] - new["rate"])
    if drop >= margin:
        separated = new["ci"][1] < old["ci"][0]
        level = "warn" if separated or drop >= 2 * margin else "info"
        out.append(base | {"level": level, "kind": "success_drop", "drop_points": round(drop, 1),
                           "message": f"{scope}: success {100 * new['rate']:.0f} % with {new_m} ({new['ok']}/{new['n']}) vs "
                                      f"{100 * old['rate']:.0f} % with {old_m} ({old['ok']}/{old['n']}): −{drop:.0f} points"
                                      + ("" if level == "warn" else " (within noise so far)")})
    if old["iters"] is not None and new["iters"] is not None and new["iters"] - old["iters"] >= 1:
        out.append(base | {"level": "info", "kind": "iterations_up",
                           "message": f"{scope}: mean iterations {new['iters']:.1f} with {new_m} vs {old['iters']:.1f} with {old_m}"})
    return out


def check(records: list[Record], *, min_runs: int = 10, margin: float = 10.0) -> list[dict[str, Any]]:
    by_key: dict[str, list[Record]] = defaultdict(list)
    by_task: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_key[r.key].append(r)
        by_task[f"{r.t.get('task')} (all)"].append(r)
    out: list[dict[str, Any]] = []
    for scope, rs in sorted(by_task.items()) + sorted(by_key.items()):
        out.extend(_compare(scope, rs, min_runs=min_runs, margin=margin))
    return out
