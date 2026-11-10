"""Phase 18 — local monitoring + data flywheel.

Everything here reads and writes files on this machine only (``data/runs/``, ``data/monitor/``,
``data/datasets/flywheel/``); nothing opens a network connection.

- :mod:`.telemetry` — the ``telemetry`` block every ``rebirth migrate`` run log carries
- :mod:`.store`     — load run logs (old ones without telemetry are derived from their trace)
- :mod:`.dashboard` — per language / direction health, per model version (``rebirth dashboard``)
- :mod:`.alerts`    — regression warnings when a new model version does worse
- :mod:`.drift`     — live inputs vs the training data (length, imports)
- :mod:`.flywheel`  — review queue → curated training rows + rolling eval set (``rebirth review``)
"""
