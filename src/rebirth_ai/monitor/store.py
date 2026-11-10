"""Load ``data/runs/<id>/summary.json`` run logs as flat telemetry records (oldest first)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rebirth_ai.monitor.telemetry import from_payload
from rebirth_ai.runs import runs_dir


@dataclass
class Record:
    run_id: str
    ts: str
    kind: str
    t: dict[str, Any]                     # telemetry
    payload: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def key(self) -> str:
        """Health is tracked per task + language (modernize) or direction (translate)."""
        return f"{self.t.get('task')}:{self.t.get('direction')}"


def load_records(root: Path | None = None, *, kinds: tuple[str, ...] = ("migrate", "translate", "migrate-repo")) -> list[Record]:
    base = Path(root) if root is not None else runs_dir()
    out: list[Record] = []
    if not base.is_dir():
        return out
    for summary in sorted(base.glob("*/summary.json")):
        try:
            doc = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        kind = doc.get("kind", "")
        if kind not in kinds:
            continue
        payload = doc.get("payload") or {}
        for t in from_payload(kind, payload):
            out.append(Record(run_id=doc.get("run_id", summary.parent.name), ts=doc.get("ts", ""), kind=kind,
                              t=t, payload=payload))
    out.sort(key=lambda r: (r.ts, r.run_id))
    return out
