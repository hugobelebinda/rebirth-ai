"""Conditional edge helpers for the migration graph."""

from __future__ import annotations

from rebirth_ai.graph.state import MigrationState, default_max_iterations


def after_retrieve(state: MigrationState) -> str:
    if state.get("test_first") and (state.get("current_draft") or "").strip():
        return "execute_tests"
    return "generate_code"


def after_tests(state: MigrationState) -> str:
    if state.get("status") == "success":
        return "success"
    return "classify_error"


def after_classify(state: MigrationState) -> str:
    max_iter = int(state.get("max_iterations") or default_max_iterations())
    iteration = int(state.get("iteration_count") or 0)
    if iteration >= max_iter:
        return "halt_human_review"
    if state.get("error_class") == "DependencyMissing":
        return "install_dependency"
    # Syntax / type / logic failures: regenerate with error context
    return "generate_code"


def after_install(state: MigrationState) -> str:
    """After queuing a dep, re-test; if already tried, ask the LLM instead."""
    max_iter = int(state.get("max_iterations") or default_max_iterations())
    iteration = int(state.get("iteration_count") or 0)
    if iteration >= max_iter:
        return "halt_human_review"
    # Trace ends with "already queued X" when the package was previously known
    last = (state.get("trace") or [""])[-1]
    if "already queued" in last:
        return "generate_code"
    return "execute_tests"
