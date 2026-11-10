"""Closed-loop translation (LangGraph): translate → verify in the target sandbox → classify → retry.

Same loop shape as same-language migration (``graph/app.py``), but generation and verification
both use the *target* language: the target project's tests when there are any, else a compile
check. Failures go back to the model with the target compiler / test output. No dependency
installs: a translation must use the target's standard library (the prompt says so).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from rebirth_ai.graph.nodes import classify_error
from rebirth_ai.xlang.fixups import apply_fixups
from rebirth_ai.xlang.languages import get_lang, target_relpath
from rebirth_ai.xlang.prompts import build_translation_prompt, extract_code
from rebirth_ai.xlang.translator import get_translator, max_new_tokens_for, retry_temperature
from rebirth_ai.xlang.verify import verify_translation


class TranslationState(TypedDict, total=False):
    source_lang: str
    target_lang: str
    source_rel: str
    source_code: str
    target_rel: str  # relative to the output dir, or to the target project
    target_rel_fixed: bool  # --target-file given: never rename (Java class name)
    target_project: str | None
    target_test: str | None
    target_tests_text: str | None
    backend: str
    current_draft: str
    test_logs: str
    error_class: str | None
    error_confidence: float | None
    error_unknown: bool
    verify_mode: str
    iteration_count: int
    max_iterations: int
    status: str  # running | success | human_review
    trace: list[str]
    model: str


def _trace(state: TranslationState, msg: str) -> list[str]:
    return list(state.get("trace") or []) + [msg]


def translate_code(state: TranslationState) -> TranslationState:
    source, target = get_lang(state["source_lang"]), get_lang(state["target_lang"])
    retry = int(state.get("iteration_count") or 0) > 0
    prompt = build_translation_prompt(
        source=source,
        target=target,
        source_code=state["source_code"],
        source_path=state["source_rel"],
        target_path=state["target_rel"],
        target_tests=state.get("target_tests_text"),
        previous=state.get("current_draft") if retry else None,
        error_class=state.get("error_class") if retry else None,
        logs=state.get("test_logs") if retry else None,
    )
    # Retries explore: near-greedy decoding repeats the same wrong answer for a near-identical prompt.
    temperature = retry_temperature(int(state.get("iteration_count") or 0))
    out = get_translator(state.get("backend")).complete(
        prompt, max_new_tokens=max_new_tokens_for(state["source_code"]), temperature=temperature
    )
    code = extract_code(out.raw, target)
    fixed = apply_fixups(target, code)  # e.g. Go: missing stdlib imports (goimports-style)
    fix_note = " (fixups applied)" if fixed != code else ""
    code = fixed
    target_rel = state["target_rel"]
    if not state.get("target_rel_fixed"):
        target_rel = target_relpath(state["source_rel"], target, code)
    iteration = int(state.get("iteration_count") or 0) + 1
    return {
        **state,
        "current_draft": code,
        "target_rel": target_rel,
        "iteration_count": iteration,
        "model": out.model,
        "status": "running",
        "trace": _trace(state, f"translate_code: iteration={iteration} model={out.model} chars={len(code)}{fix_note}"),
    }


def verify_target(state: TranslationState) -> TranslationState:
    target = get_lang(state["target_lang"])
    project = state.get("target_project")
    res = verify_translation(
        target=target,
        code=state.get("current_draft") or "",
        target_rel=state["target_rel"],
        target_project=Path(project) if project else None,
        target_test=state.get("target_test"),
    )
    bit = "PASS" if res.ok else "FAIL"
    return {
        **state,
        "current_draft": res.tested_code or state.get("current_draft") or "",  # line numbers match the logs
        "test_logs": res.logs,
        "error_class": None,
        "verify_mode": res.mode,
        "status": "success" if res.ok else "running",
        "trace": _trace(state, f"verify_target: {res.mode} {bit} exit={res.exit_code} in {res.duration_sec:.2f}s"),
    }


# Compiler / parser errors in the 9 target toolchains (the trained error classifier only knows
# JS / Python test logs and calls a Go syntax error "TestLogicFailure").
_COMPILE_ERROR_RE = re.compile(
    r"\[build failed\]|\bsyntax error\b|error\[E\d{4}\]|error CS\d{4}|cannot find symbol|"
    r"\bundefined: |SyntaxError|ERR_INVALID_TYPESCRIPT_SYNTAX|^\S+\.(?:java|swift|cpp|hpp|h|cs|go|rs):\d+(?::\d+)?: error|"
    r"Build FAILED|error: (?:expected|use of unresolved|cannot find|could not compile)",
    re.MULTILINE,
)


def classify_target_error(state: TranslationState) -> TranslationState:
    logs = state.get("test_logs") or ""
    if _COMPILE_ERROR_RE.search(logs):
        return {
            **state,
            "error_class": "CompileError",
            "error_confidence": 0.95,
            "error_unknown": False,
            "trace": _trace(state, "classify_error: CompileError (target compiler output)"),
        }
    return classify_error(state)


def success(state: TranslationState) -> TranslationState:
    what = "target tests passed" if state.get("verify_mode") == "target_tests" else "compiles in the target sandbox"
    return {**state, "status": "success", "trace": _trace(state, f"success: {what}")}


def halt_human_review(state: TranslationState) -> TranslationState:
    return {
        **state,
        "status": "human_review",
        "trace": _trace(state, f"halt_human_review: iterations={state.get('iteration_count')} error={state.get('error_class')}"),
    }


def after_verify(state: TranslationState) -> str:
    return "success" if state.get("status") == "success" else "classify_error"


def after_classify(state: TranslationState) -> str:
    if int(state.get("iteration_count") or 0) >= int(state.get("max_iterations") or 3):
        return "halt_human_review"
    return "translate_code"


def build_translation_graph():
    g = StateGraph(TranslationState)
    g.add_node("translate_code", translate_code)
    g.add_node("verify_target", verify_target)
    g.add_node("classify_error", classify_target_error)
    g.add_node("success", success)
    g.add_node("halt_human_review", halt_human_review)
    g.add_edge(START, "translate_code")
    g.add_edge("translate_code", "verify_target")
    g.add_conditional_edges("verify_target", after_verify, {"success": "success", "classify_error": "classify_error"})
    g.add_conditional_edges(
        "classify_error", after_classify, {"translate_code": "translate_code", "halt_human_review": "halt_human_review"}
    )
    g.add_edge("success", END)
    g.add_edge("halt_human_review", END)
    return g.compile()


def target_tests_view(target, target_project: Path, target_test: str | None) -> str | None:
    """What the model sees of the target project: the tests only (never the target implementation).

    Rust inline tests: just the ``#[cfg(test)]`` module of the file. C++: plus the project headers
    the test includes (they declare the API the translation must define).
    """
    from rebirth_ai.xlang.verify import keep_inline_tests

    if not target_test:
        return None
    tp = target_project / target_test
    if not tp.is_file():
        return None
    text = tp.read_text(encoding="utf-8")
    if target.id == "rust":
        mod = keep_inline_tests(target, text, "")
        text = mod.strip() + "\n" if mod.strip() else text
    if target.id == "cpp":
        for inc in re.findall(r'^\s*#\s*include\s+"([^"]+)"', text, re.MULTILINE):
            for base in (tp.parent, target_project, target_project / "src", target_project / "include"):
                h = base / inc
                if h.is_file():
                    text += f"\n// ---- {h.relative_to(target_project)} (project header)\n" + h.read_text(encoding="utf-8")
                    break
    return text


def initial_translation_state(
    *,
    project_dir: Path,
    source_rel: str,
    target_lang: str,
    source_lang: str | None = None,
    target_project: Path | None = None,
    target_file: str | None = None,
    target_test: str | None = None,
    backend: str = "ollama",
    max_iterations: int = 3,
) -> TranslationState:
    from rebirth_ai.xlang.languages import lang_for_path

    src_path = project_dir / source_rel
    source = lang_for_path(source_rel) or (get_lang(source_lang) if source_lang else None)
    if source is None:
        raise ValueError(f"cannot tell the language of {source_rel}; pass --lang")
    target = get_lang(target_lang)
    if source.id == target.id:
        raise ValueError(f"source and target are both {target.id}; use `rebirth migrate` without --to")
    tests_text = target_tests_view(target, target_project, target_test) if target_project is not None else None
    return {
        "source_lang": source.id,
        "target_lang": target.id,
        "source_rel": source_rel,
        "source_code": src_path.read_text(encoding="utf-8"),
        "target_rel": target_file or target_relpath(source_rel, target),
        "target_rel_fixed": bool(target_file),
        "target_project": str(target_project.resolve()) if target_project else None,
        "target_test": target_test,
        "target_tests_text": tests_text,
        "backend": backend,
        "current_draft": "",
        "test_logs": "",
        "error_class": None,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "status": "pending",
        "trace": [],
    }
