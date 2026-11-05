"""LangGraph migration state."""

from __future__ import annotations

import os
from typing import TypedDict


class MigrationState(TypedDict, total=False):
    file_path: str  # absolute path to the target source file
    project_dir: str
    relative_file: str  # path inside project (overlay dest)
    relative_test: str
    language: str
    adapter_id: str  # registry id: js_ts | python | …
    objective: str
    original_code: str
    current_draft: str
    retrieved_context: list[str]
    test_logs: str
    error_class: str | None
    error_confidence: float | None
    error_unknown: bool
    iteration_count: int
    max_iterations: int
    status: str  # pending|running|success|human_review|failed
    trace: list[str]
    missing_package: str | None
    extra_packages: list[str]  # packages to install before each test run
    test_first: bool  # if True and draft present, execute before first generate
    db_path: str


def default_max_iterations() -> int:
    return int(os.environ.get("REBIRTH_MAX_ITERATIONS", "5"))
