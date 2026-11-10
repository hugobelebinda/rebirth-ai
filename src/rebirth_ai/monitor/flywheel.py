"""Data flywheel: failed or hard migrations → human review → curated training rows.

Every ``rebirth migrate`` run saves its input and final draft next to its log
(``data/runs/<id>/artifacts/``, redacted). The **review queue** holds runs that failed or needed
``hard_iterations``+ attempts and have no decision yet. Reviewing one (``rebirth review``):

- **accept** — the final code is the model's draft or the user's own fix (``--fixed FILE``); the
  user confirms the code is their own (license-clean). One row is appended to
  ``data/datasets/flywheel/<lang>.jsonl`` (same-language, corpus format) or ``xlang.jsonl``
  (translation); its split is fixed by a hash of the input: ~20 % ``eval``, the rest ``train``.
- **reject** — recorded, nothing is added.

``train`` rows reach the next training round through ``training/build_flywheel_mix.py`` (DVC stage
``flywheel_mix``); ``eval`` rows form the rolling "recent real-world" eval set
(``data/datasets/flywheel/rolling_eval/<lang>.jsonl``) — the frozen eval sets never change.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rebirth_ai.monitor.imports import family
from rebirth_ai.monitor.store import Record, load_records
from rebirth_ai.redact import redact_text
from rebirth_ai.runs import runs_dir

FLYWHEEL_DIR = Path("data/datasets/flywheel")
EVAL_PERCENT = 20


def save_artifacts(run_dir: Path, *, source_name: str, source: str | None, draft_name: str, draft: str | None) -> dict[str, str]:
    """Redacted copies of the run's input and final draft; returns names relative to the run dir."""
    out: dict[str, str] = {}
    adir = run_dir / "artifacts"
    for role, name, text in (("source", source_name, source), ("draft", draft_name, draft)):
        if text is None:
            continue
        adir.mkdir(parents=True, exist_ok=True)
        path = adir / f"{role}{Path(name).suffix}"
        path.write_text(redact_text(text), encoding="utf-8")
        out[role] = str(path.relative_to(run_dir))
    return out


def split_for(text: str) -> str:
    h = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    return "eval" if h % 100 < EVAL_PERCENT else "train"


def _run_dir(run_id: str, root: Path | None = None) -> Path:
    return (Path(root) if root is not None else runs_dir()) / run_id


def decision(run_id: str, root: Path | None = None) -> dict[str, Any] | None:
    p = _run_dir(run_id, root) / "review.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def queue(root: Path | None = None, *, hard_iterations: int = 3) -> list[Record]:
    """Reviewable runs (artifacts saved), failed or hard, not yet decided — oldest first."""
    out = []
    for r in load_records(root, kinds=("migrate", "translate")):
        arts = r.payload.get("artifacts") or {}
        hard = not r.t.get("success") or (r.t.get("iterations") or 0) >= hard_iterations
        if "source" in arts and "draft" in arts and hard and decision(r.run_id, root) is None:
            out.append(r)
    return out


def _find(run_id: str, root: Path | None) -> Record:
    for r in load_records(root, kinds=("migrate", "translate")):
        if r.run_id == run_id:
            return r
    raise KeyError(f"no run log {run_id!r}")


def artifacts(run_id: str, root: Path | None = None) -> dict[str, str]:
    r = _find(run_id, root)
    base = _run_dir(run_id, root)
    return {k: (base / v).read_text(encoding="utf-8") for k, v in (r.payload.get("artifacts") or {}).items()}


def _record_decision(run_id: str, root: Path | None, doc: dict[str, Any]) -> None:
    p = _run_dir(run_id, root) / "review.json"
    p.write_text(json.dumps({"ts": datetime.now(UTC).isoformat(timespec="seconds"), **doc}, indent=1) + "\n",
                 encoding="utf-8")


def accept(run_id: str, *, own: bool, fixed: str | None = None, root: Path | None = None,
           out_dir: Path = FLYWHEEL_DIR) -> dict[str, Any]:
    """Append the reviewed migration as a training / eval row. ``own`` = the user confirms the
    code is theirs to train on (the flywheel never takes third-party code)."""
    if not own:
        raise PermissionError("only code you own can enter the training data: pass own=True / --own")
    if decision(run_id, root) is not None:
        raise ValueError(f"{run_id} was already reviewed")
    r = _find(run_id, root)
    arts = artifacts(run_id, root)
    final = redact_text(fixed) if fixed is not None else arts["draft"]
    src = arts["source"]
    t = r.t
    common = {"source": "flywheel", "run_id": run_id, "license": "own", "fixed_by_user": fixed is not None,
              "split": split_for(src), "reviewed": datetime.now(UTC).isoformat(timespec="seconds"),
              "model": t.get("model")}
    if t.get("task") == "translate":
        row = {"kind": "flywheel", "source_lang": t.get("source_lang"), "target_lang": t.get("target_lang"),
               "source_code": src, "target_code": final, **common}
        path = out_dir / "xlang.jsonl"
    else:
        lang = family(t.get("lang")) or t.get("lang")
        row = {"objective": r.payload.get("objective") or "Modernize this file", "language": lang,
               "legacy": src, "modern": final, "path": r.payload.get("file"), **common}
        path = out_dir / f"{lang}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    _record_decision(run_id, root, {"decision": "accepted", "fixed_by_user": fixed is not None,
                                    "row_file": str(path), "split": row["split"]})
    return row


def reject(run_id: str, *, reason: str = "", root: Path | None = None) -> None:
    if decision(run_id, root) is not None:
        raise ValueError(f"{run_id} was already reviewed")
    _find(run_id, root)
    _record_decision(run_id, root, {"decision": "rejected", "reason": reason})


def rows(out_dir: Path = FLYWHEEL_DIR) -> dict[str, list[dict[str, Any]]]:
    """Curated rows per file stem (language id or ``xlang``)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for p in sorted(Path(out_dir).glob("*.jsonl")):
        out[p.stem] = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return out


def export_rolling_eval(out_dir: Path = FLYWHEEL_DIR, *, max_rows: int = 100) -> dict[str, int]:
    """Newest ``max_rows`` eval-split rows per language → ``rolling_eval/<lang>.jsonl`` (same row
    format as the frozen eval sets, so ``eval_behavioral.py --eval-file`` reads them)."""
    counts: dict[str, int] = {}
    target = Path(out_dir) / "rolling_eval"
    for stem, rs in rows(out_dir).items():
        ev = [r for r in rs if r.get("split") == "eval"][-max_rows:]
        if not ev:
            continue
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{stem}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ev),
                                              encoding="utf-8")
        counts[stem] = len(ev)
    return counts
