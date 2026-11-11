"""Compile the Rebirth AI LangGraph migration loop."""

from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, START, StateGraph

from rebirth_ai.graph import nodes
from rebirth_ai.graph.routing import (
    after_classify,
    after_install,
    after_retrieve,
    after_tests,
)
from rebirth_ai.graph.state import MigrationState, default_max_iterations
from rebirth_ai.paths import DEFAULT_DB


def build_migration_graph():
    graph = StateGraph(MigrationState)
    graph.add_node("retrieve_context", nodes.retrieve_context)
    graph.add_node("generate_code", nodes.generate_code)
    graph.add_node("execute_tests", nodes.execute_tests)
    graph.add_node("classify_error", nodes.classify_error)
    graph.add_node("install_dependency", nodes.install_dependency)
    graph.add_node("success", nodes.success)
    graph.add_node("halt_human_review", nodes.halt_human_review)

    graph.add_edge(START, "retrieve_context")
    graph.add_conditional_edges(
        "retrieve_context",
        after_retrieve,
        {
            "generate_code": "generate_code",
            "execute_tests": "execute_tests",
        },
    )
    graph.add_edge("generate_code", "execute_tests")
    graph.add_conditional_edges(
        "execute_tests",
        after_tests,
        {
            "success": "success",
            "classify_error": "classify_error",
        },
    )
    graph.add_conditional_edges(
        "classify_error",
        after_classify,
        {
            "halt_human_review": "halt_human_review",
            "install_dependency": "install_dependency",
            "generate_code": "generate_code",
        },
    )
    graph.add_conditional_edges(
        "install_dependency",
        after_install,
        {
            "halt_human_review": "halt_human_review",
            "execute_tests": "execute_tests",
            "generate_code": "generate_code",
        },
    )
    graph.add_edge("success", END)
    graph.add_edge("halt_human_review", END)
    return graph.compile()


def initial_state(
    *,
    project_dir: Path,
    relative_file: str,
    relative_test: str,
    objective: str,
    seed_draft: str | None = None,
    test_first: bool = False,
    db_path: Path | str = DEFAULT_DB,
    max_iterations: int | None = None,
    adapter_id: str | None = None,
    language: str | None = None,
) -> MigrationState:
    from rebirth_ai.adapters.registry import (
        adapter_for_suffix,
        get_adapter_for_project,
        language_label_for_file,
    )

    project_dir = project_dir.resolve()
    file_path = project_dir / relative_file
    original = file_path.read_text(encoding="utf-8")
    adapter = get_adapter_for_project(project_dir, lang=adapter_id or adapter_for_suffix(file_path.suffix))
    resolved_adapter_id = adapter.name
    resolved_language = language or language_label_for_file(
        file_path, adapter_id=resolved_adapter_id
    )
    draft = seed_draft if seed_draft is not None else original
    return {
        "file_path": str(file_path),
        "project_dir": str(project_dir),
        "relative_file": relative_file,
        "relative_test": relative_test,
        "language": resolved_language,
        "adapter_id": resolved_adapter_id,
        "objective": objective,
        "original_code": original,
        "current_draft": draft,
        "retrieved_context": [],
        "test_logs": "",
        "error_class": None,
        "iteration_count": 0,
        "max_iterations": max_iterations or default_max_iterations(),
        "status": "pending",
        "trace": [],
        "missing_package": None,
        "extra_packages": [],
        "test_first": test_first,
        "db_path": str(db_path),
    }
