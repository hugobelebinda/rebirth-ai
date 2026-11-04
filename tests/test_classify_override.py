"""Heuristic DependencyMissing override (Phase 7 express path)."""

from rebirth_ai.graph.nodes import _heuristic_error_class, classify_error


def test_heuristic_detects_esm_missing_package() -> None:
    logs = (
        "Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'ms' "
        "imported from /workspace/src/duration.js"
    )
    assert _heuristic_error_class(logs) == "DependencyMissing"


def test_classify_error_overrides_to_dependency_missing() -> None:
    state = classify_error(
        {
            "test_logs": (
                "Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'ms' "
                "imported from /workspace/src/duration.js\n"
            ),
            "trace": [],
        }
    )
    assert state["error_class"] == "DependencyMissing"
    assert "DependencyMissing" in state["trace"][-1]
