"""Where Rebirth keeps its own files: inside the Rebirth folder, never the folder you run it from.

Rebirth is installed from a git clone (``scripts/bootstrap.sh``), so its data lives next to the code. Running
``rebirth migrate-repo ~/code/app`` from inside ``~/code/app`` must not leave a ``data/`` folder in that project.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DEFAULT_DB = DATA / "embeddings.db"
DEFAULT_REPORT = DATA / "reports" / "migrate_repo.json"
CLASSIFIER_DIR = DATA / "models" / "error_classifier"
ERROR_LOGS = DATA / "datasets" / "error_logs.jsonl"


def runs_dir_default() -> Path:
    """``REBIRTH_RUNS_DIR``, else ``<Rebirth>/data/runs``."""
    return Path(os.environ.get("REBIRTH_RUNS_DIR") or DATA / "runs")
