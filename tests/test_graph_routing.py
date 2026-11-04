"""Unit tests for LangGraph routing + graph compile (no Docker/Ollama)."""

from __future__ import annotations

from rebirth_ai.graph.app import build_migration_graph
from rebirth_ai.graph.routing import after_classify, after_install, after_retrieve, after_tests
from rebirth_ai.graph.state import MigrationState


def test_build_migration_graph_compiles() -> None:
    app = build_migration_graph()
    assert app is not None


def test_after_retrieve_test_first() -> None:
    state: MigrationState = {"test_first": True, "current_draft": "x"}
    assert after_retrieve(state) == "execute_tests"
    assert after_retrieve({"test_first": False, "current_draft": "x"}) == "generate_code"


def test_after_tests_routes() -> None:
    assert after_tests({"status": "success"}) == "success"
    assert after_tests({"status": "running"}) == "classify_error"


def test_after_classify_routes() -> None:
    assert (
        after_classify(
            {"iteration_count": 5, "max_iterations": 5, "error_class": "SyntaxError"}
        )
        == "halt_human_review"
    )
    assert (
        after_classify(
            {
                "iteration_count": 1,
                "max_iterations": 5,
                "error_class": "DependencyMissing",
            }
        )
        == "install_dependency"
    )
    assert (
        after_classify(
            {
                "iteration_count": 1,
                "max_iterations": 5,
                "error_class": "TestLogicFailure",
            }
        )
        == "generate_code"
    )


def test_after_install_already_queued() -> None:
    assert (
        after_install(
            {
                "iteration_count": 0,
                "max_iterations": 5,
                "trace": ["install_dependency: already queued lodash"],
            }
        )
        == "generate_code"
    )
    assert (
        after_install(
            {
                "iteration_count": 0,
                "max_iterations": 5,
                "trace": ["install_dependency: queued lodash"],
            }
        )
        == "execute_tests"
    )
