"""Second opinion when a served same-language adapter changes nothing (Phase 19).

On 14 real files (``training/bench_real_repos.py``, 2026-11-11) the served 7B adapters returned
10 unchanged. One more run with a 14B on those took the clean modernizations from 2 to 6.

The fallback applies when all of these hold:

- the run used ``--backend auto``;
- a served adapter (``rebirth-<lang>:7b``) wrote the draft;
- the tests pass, but the draft is the original file (``batch.no_change``).

Then the loop runs once more with a 14B in Ollama: ``REBIRTH_FALLBACK_MODEL``, else the plain
Qwen2.5-Coder 14B (``OLLAMA_MODEL``), else ``rebirth-xlang:14b``, whichever Ollama has.
``--no-fallback`` or ``REBIRTH_FALLBACK=0`` turns it off.

The 14B gets ``REBIRTH_FALLBACK_TRIES`` tries (default 2: on the real files, every fallback that found a
change found it on the first try). When it finds no passing change, the result stays the served adapter's:
the file is unchanged and its tests pass, so it is reported as unchanged, not as a failure.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager

AUTO_BACKENDS = {"auto", "routed", "rebirth"}
PLAIN_14B = "qwen2.5-coder:14b-instruct-q6_K"
XLANG_14B = "rebirth-xlang:14b"
_MODEL = re.compile(r"\bmodel=(\S+)")


def enabled(flag: bool | None = None) -> bool:
    """``--fallback`` / ``--no-fallback`` when given, else ``REBIRTH_FALLBACK`` (on by default)."""
    if flag is not None:
        return flag
    return os.environ.get("REBIRTH_FALLBACK", "1").strip().lower() not in {"0", "false", "no", "off"}


def served_adapter_models(trace: list[str] | None) -> list[str]:
    """Served same-language adapters that wrote a draft in this run (``ollama:rebirth-go:7b``)."""
    return [m for line in trace or [] for m in _MODEL.findall(line)
            if m.startswith("ollama:rebirth-") and not m.startswith(f"ollama:{XLANG_14B}")]


def tries() -> int:
    """``REBIRTH_FALLBACK_TRIES`` (default 2, at least 1)."""
    try:
        return max(1, int(os.environ.get("REBIRTH_FALLBACK_TRIES", "2")))
    except ValueError:
        return 2


def candidates() -> list[str]:
    env = os.environ.get("REBIRTH_FALLBACK_MODEL", "").strip()
    if env:
        return [env]
    return [os.environ.get("OLLAMA_MODEL", PLAIN_14B).strip() or PLAIN_14B, XLANG_14B]


def pick_model(list_models: Callable[[], list[str]] | None = None) -> str | None:
    """The first candidate Ollama has (None when Ollama cannot be reached)."""
    if list_models is None:
        from rebirth_ai.generate.ollama_client import OllamaClient

        list_models = OllamaClient().list_models
    try:
        names = set(list_models())
    except Exception:  # connection / HTTP error
        return None
    for c in candidates():
        if c in names or (":" not in c and f"{c}:latest" in names):
            return c
    return None


@contextmanager
def _plain_ollama(model: str) -> Iterator[None]:
    keys = ("REBIRTH_GENERATOR", "OLLAMA_MODEL")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["REBIRTH_GENERATOR"], os.environ["OLLAMA_MODEL"] = "ollama", model
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def needs_fallback(final: dict, *, original: str | None, seed: str | None = None, backend: str | None = None,
                   use: bool | None = None) -> str | None:
    """The served adapter (``ollama:rebirth-go:7b``) whose passing, unchanged result should get the
    14B run, else None."""
    from rebirth_ai.graph.batch import no_change

    backend = (backend or os.environ.get("REBIRTH_GENERATOR", "auto")).strip().lower()
    draft = final.get("current_draft") or ""
    if not (enabled(use) and backend in AUTO_BACKENDS and final.get("status") == "success" and draft
            and original is not None and no_change(draft, original, seed)):
        return None
    served = served_adapter_models(final.get("trace"))
    return served[-1] if served else None


def run_fallback(graph, state: dict, final: dict, served: str, *,
                 list_models: Callable[[], list[str]] | None = None) -> dict:
    """The 14B run for ``final`` when it passes (both traces kept; ``fallback_model`` set). Otherwise
    ``final`` itself with a note: when Ollama has no 14B, or the 14B found no passing change."""
    trace = list(final.get("trace") or [])
    model = pick_model(list_models)
    if model is None:
        note = (f"fallback: skipped — {served} returned the file unchanged and Ollama has no 14B "
                f"({' / '.join(candidates())}; `ollama pull {PLAIN_14B}`)")
        return {**final, "trace": trace + [note]}
    n = tries()
    trace.append(f"fallback: {served} returned the file unchanged; one more run with {model} (up to {n} tries)")
    with _plain_ollama(model):
        second = graph.invoke({**state, "max_iterations": n})
    trace += list(second.get("trace") or [])
    if second.get("status") == "success":
        return {**second, "trace": trace, "fallback_model": model}
    note = (f"fallback: {model} found no passing change in {second.get('iteration_count') or n} tries; "
            "the file is left as it was (its tests pass)")
    return {**final, "trace": trace + [note]}


def invoke(graph, state: dict, *, original: str | None, seed: str | None = None, backend: str | None = None,
           use: bool | None = None, list_models: Callable[[], list[str]] | None = None) -> dict:
    """``graph.invoke(state)``, then the fallback run if it applies (see the module doc)."""
    final = graph.invoke(state)
    served = needs_fallback(final, original=original, seed=seed, backend=backend, use=use)
    return run_fallback(graph, state, final, served, list_models=list_models) if served else final
