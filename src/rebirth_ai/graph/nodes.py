"""LangGraph node implementations."""

from __future__ import annotations

import re
from pathlib import Path

from rebirth_ai.adapters.base import LanguageAdapter
from rebirth_ai.adapters.registry import get_adapter
from rebirth_ai.classify.model import ErrorClassifier
from rebirth_ai.generate.factory import get_generator
from rebirth_ai.graph.state import MigrationState
from rebirth_ai.rag.retrieve import Retriever
from rebirth_ai.sandbox.deps import install_decision
from rebirth_ai.sandbox.runner import SandboxRunner

_MISSING_PKG = re.compile(
    r"(?:Cannot find (?:module|package) ['\"]([^'\"]+)['\"]|"
    r"ModuleNotFoundError:\s*No module named ['\"]([^'\"]+)['\"]|"
    r"ImportError:\s*(?:No module named|cannot import name) ['\"]([^'\"]+)['\"])",
    re.IGNORECASE,
)


def _adapter_for_state(state: MigrationState) -> LanguageAdapter:
    adapter_id = state.get("adapter_id")
    if adapter_id:
        return get_adapter(str(adapter_id))
    language = (state.get("language") or "").lower()
    if language in {"python", "py"}:
        return get_adapter("python")
    return get_adapter("js_ts")


def _missing_package_name(logs: str) -> str | None:
    match = _MISSING_PKG.search(logs)
    if not match:
        return None
    return next((g for g in match.groups() if g), None)


def _trace(state: MigrationState, message: str) -> list[str]:
    return list(state.get("trace") or []) + [message]


def retrieve_context(state: MigrationState) -> MigrationState:
    path = Path(state["file_path"])
    db = Path(state.get("db_path") or "data/embeddings.db")
    context: list[str] = []
    if db.exists():
        # One index per project (`rebirth index <project>`): chunks of another project are not context
        # for this one. Checked before the embedder loads.
        from rebirth_ai.rag.store import VectorStore

        indexed = VectorStore(db).get_meta("project_dir")
        project = Path(state["project_dir"]).resolve()
        if indexed and Path(indexed).resolve() != project:
            return {
                **state,
                "retrieved_context": [],
                "status": "running",
                "trace": _trace(state, f"retrieve_context: skipped (the index {db} is for {indexed}; "
                                       f"`rebirth index {project}` builds one for this project)"),
            }
        try:
            retriever = Retriever(db_path=db, adapter=_adapter_for_state(state))
            hits = retriever.query_for_file(path, top_k=4)
            context = [f"// {h.file_path} :: {h.kind} {h.name}\n{h.text}" for h in hits]
        except Exception as exc:
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
    language = state.get("language") or "javascript"
    client = get_generator(language=state.get("adapter_id") or language)
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
    adapter = _adapter_for_state(state)
    runner = SandboxRunner(adapter)

    # Temp sandbox copies discard node_modules / venv: the project's declared dependencies and the
    # packages queued by install_dependency are installed in the sandbox's setup step, with network,
    # before the draft is written in; the draft's tests run offline.
    setup = runner.setup_command(project, state.get("relative_test"))
    install = getattr(adapter, "install_command", None)
    if packages and callable(install):
        setup = " && ".join(c for c in (setup, install(packages)) if c)
    result = runner.run_project(
        project,
        relative_test=state.get("relative_test"),
        setup_command=setup,
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
        or "modulenotfounderror" in low
        or "no module named" in low
        or "can't resolve" in low
        or "could not resolve" in low
    ):
        return "DependencyMissing"
    if (
        "syntaxerror" in low
        or "unexpected token" in low
        or "error ts1005" in low
        or "indentationerror" in low
    ):
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
    except Exception:
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


def _refusal_note(pkg: str, logs: str) -> str:
    """Test logs for the next draft, led by why the missing package stays missing (else the model
    keeps importing it: Phase 19 bench, 5 drafts in a row)."""
    return (
        f"[rebirth] The package `{pkg}` is not available and will not be installed: the project does "
        "not depend on it. Do not import it. Use the language's standard library or a dependency the "
        f"project already declares.\n\n{logs}"
    )


def install_dependency(state: MigrationState) -> MigrationState:
    logs = state.get("test_logs") or ""
    pkg = _missing_package_name(logs) or state.get("missing_package")
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

    adapter = _adapter_for_state(state)
    if not callable(getattr(adapter, "install_command", None)):
        return {
            **state,
            "missing_package": pkg,
            "test_logs": _refusal_note(pkg, logs),
            "trace": _trace(state, f"install_dependency: refused {pkg} — {adapter.name} has no package installs"),
        }
    language = "python" if adapter.name == "python" else "js_ts"
    to_install, why = install_decision(Path(state["project_dir"]), language, pkg, state.get("original_code") or "")
    if to_install is None:
        return {
            **state,
            "missing_package": pkg,
            "test_logs": _refusal_note(pkg, logs),
            "trace": _trace(state, f"install_dependency: refused {pkg} — {why}"),
        }

    extras = list(state.get("extra_packages") or [])
    already = to_install in extras
    if not already:
        extras.append(to_install)
    return {
        **state,
        "missing_package": pkg,
        "extra_packages": extras,
        "trace": _trace(
            state,
            f"install_dependency: {'already queued' if already else 'queued'} {to_install} ({why})",
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
