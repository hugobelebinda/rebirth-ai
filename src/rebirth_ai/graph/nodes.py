"""LangGraph node implementations."""

from __future__ import annotations

import re
from pathlib import Path

from rebirth_ai.adapters.js_ts import JsTsAdapter
from rebirth_ai.classify.model import ErrorClassifier
from rebirth_ai.generate.factory import get_generator
from rebirth_ai.graph.state import MigrationState
from rebirth_ai.rag.retrieve import Retriever
from rebirth_ai.sandbox.runner import SandboxRunner

_MISSING_PKG = re.compile(
    r"Cannot find (?:module|package) ['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)


def _trace(state: MigrationState, message: str) -> list[str]:
    return list(state.get("trace") or []) + [message]


def retrieve_context(state: MigrationState) -> MigrationState:
    path = Path(state["file_path"])
    db = Path(state.get("db_path") or "data/embeddings.db")
    context: list[str] = []
    if db.exists():
        try:
            retriever = Retriever(db_path=db)
            hits = retriever.query_for_file(path, top_k=4)
            context = [f"// {h.file_path} :: {h.kind} {h.name}\n{h.text}" for h in hits]
        except Exception as exc:  # noqa: BLE001
            return {
                **state,
                "retrieved_context": [],
                "status": "running",
                "trace": _trace(state, f"retrieve_context: skipped ({exc})"),
            }
    return {
        **state,
        "retrieved_context": context,
        "status": "running",
        "trace": _trace(state, f"retrieve_context: {len(context)} chunk(s)"),
    }


def generate_code(state: MigrationState) -> MigrationState:
    client = get_generator()
    language = state.get("language") or "javascript"
    result = client.migrate(
        objective=state.get("objective")
        or "Modernize the file and ensure project tests pass",
        legacy_code=state.get("current_draft") or state["original_code"],
        file_path=state.get("relative_file") or state["file_path"],
        language=language,
        retrieved_context=state.get("retrieved_context") or None,
        error_class=state.get("error_class"),
        test_logs=state.get("test_logs"),
    )
    iteration = int(state.get("iteration_count") or 0) + 1
    return {
        **state,
        "current_draft": result.code,
        "iteration_count": iteration,
        "status": "running",
        "trace": _trace(
            state,
            f"generate_code: iteration={iteration} model={result.model} chars={len(result.code)}",
        ),
    }


def execute_tests(state: MigrationState) -> MigrationState:
    project = Path(state["project_dir"])
    relative_file = state.get("relative_file") or Path(state["file_path"]).name
    draft = state.get("current_draft") or state["original_code"]
    packages = list(state.get("extra_packages") or [])
    runner = SandboxRunner(JsTsAdapter())

    # Temp sandbox copies discard node_modules — when extras are known, install+test
    # in one networked container so the dep is available for the suite.
    if packages:
        pkgs = " ".join(packages)
        test_cmd = runner.adapter.test_command_for(state.get("relative_test"))
        combined = f"npm install --no-save --no-package-lock {pkgs} && {test_cmd}"
        result = runner.run_project(
            project,
            command=combined,
            network_mode="bridge",
            overlay_text=draft,
            overlay_dest=relative_file,
        )
    else:
        result = runner.run_project(
            project,
            relative_test=state.get("relative_test"),
            overlay_text=draft,
            overlay_dest=relative_file,
        )
    logs = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    status_bit = "PASS" if result.ok else "FAIL"
    return {
        **state,
        "test_logs": logs,
        "error_class": None,
        "status": "success" if result.ok else "running",
        "trace": _trace(
            state,
            f"execute_tests: {status_bit} exit={result.exit_code} in {result.duration_sec:.2f}s",
        ),
    }


def _heuristic_error_class(logs: str) -> str | None:
    """High-precision overrides (beat a confused classifier on obvious CI shapes)."""
    low = logs.lower()
    if (
        "cannot find module" in low
        or "cannot find package" in low
        or "err_module_not_found" in low
        or "module not found" in low
        or "can't resolve" in low
        or "could not resolve" in low
    ):
        return "DependencyMissing"
    if "syntaxerror" in low or "unexpected token" in low or "error ts1005" in low:
        return "SyntaxError"
    if "typeerror" in low or "error ts23" in low or "is not assignable to" in low:
        return "TypeMismatch"
    return None


def classify_error(state: MigrationState) -> MigrationState:
    logs = state.get("test_logs") or ""
    confidence: float | None = None
    unknown = False
    override = _heuristic_error_class(logs)
    try:
        clf = ErrorClassifier.load()
        probs = clf.predict_proba(logs)
        label = max(probs, key=probs.get)
        confidence = float(probs[label])
        # Clear dependency/module errors always win over a low-signal model guess.
        if override == "DependencyMissing":
            label = "DependencyMissing"
            confidence = max(confidence, 0.99)
        elif confidence < 0.45:
            # Phase 7 telemetry: low-confidence → keep prediction but mark Unknown.
            unknown = True
            if override:
                label = override
    except Exception:  # noqa: BLE001
        label = override or "TestLogicFailure"
        if override is None:
            unknown = True
            confidence = 0.0
        else:
            confidence = 0.9
    conf_bit = f" conf={confidence:.2f}" if confidence is not None else ""
    unk_bit = " (Unknown/low-confidence telemetry)" if unknown else ""
    return {
        **state,
        "error_class": label,
        "error_confidence": confidence,
        "error_unknown": unknown,
        "trace": _trace(state, f"classify_error: {label}{conf_bit}{unk_bit}"),
    }


def install_dependency(state: MigrationState) -> MigrationState:
    logs = state.get("test_logs") or ""
    match = _MISSING_PKG.search(logs)
    pkg = match.group(1) if match else state.get("missing_package")
    if not pkg:
        return {
            **state,
            "trace": _trace(state, "install_dependency: no package detected"),
        }
    # Strip relative paths like "./foo" — only record bare package names
    if pkg.startswith(".") or pkg.startswith("/"):
        return {
            **state,
            "trace": _trace(state, f"install_dependency: skip relative import {pkg}"),
        }

    extras = list(state.get("extra_packages") or [])
    already = pkg in extras
    if not already:
        extras.append(pkg)
    return {
        **state,
        "missing_package": pkg,
        "extra_packages": extras,
        "trace": _trace(
            state,
            f"install_dependency: {'already queued' if already else 'queued'} {pkg}",
        ),
    }


def success(state: MigrationState) -> MigrationState:
    return {
        **state,
        "status": "success",
        "trace": _trace(state, "success: tests passed"),
    }


def halt_human_review(state: MigrationState) -> MigrationState:
    return {
        **state,
        "status": "human_review",
        "trace": _trace(
            state,
            f"halt_human_review: iterations={state.get('iteration_count')} "
            f"error={state.get('error_class')}",
        ),
    }
