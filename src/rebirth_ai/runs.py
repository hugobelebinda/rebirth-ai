"""Structured migrate run logs under ``<Rebirth>/data/runs/`` (Phase 10; ``REBIRTH_RUNS_DIR`` moves them)."""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rebirth_ai.paths import runs_dir_default
from rebirth_ai.redact import redact_obj


def new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def runs_dir() -> Path:
    path = runs_dir_default()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_run_log(
    *,
    kind: str,
    payload: dict[str, Any],
    run_id: str | None = None,
) -> Path:
    """Write one redacted JSONL + summary JSON for a migrate/eval run."""
    rid = run_id or new_run_id()
    root = runs_dir() / rid
    root.mkdir(parents=True, exist_ok=True)
    safe = redact_obj(
        {
            "run_id": rid,
            "kind": kind,
            "ts": datetime.now(UTC).isoformat(),
            "payload": payload,
        }
    )
    jsonl = root / "events.jsonl"
    with jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(safe, ensure_ascii=False) + "\n")
    summary = root / "summary.json"
    summary.write_text(json.dumps(safe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def append_event(run_id: str, event: dict[str, Any]) -> None:
    root = runs_dir() / run_id
    root.mkdir(parents=True, exist_ok=True)
    path = root / "events.jsonl"
    row = redact_obj({"ts": time.time(), **event})
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
