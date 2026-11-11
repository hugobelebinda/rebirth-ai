"""Phase 19 fallback: a served 7B adapter that passes the tests but changes nothing → one more run
with a 14B (no Ollama / Docker: fake graph and model list)."""

from __future__ import annotations

import os

import pytest

from rebirth_ai.graph import fallback

ORIGINAL = "var x = 1;\n"


class Graph:
    """First run: the served adapter returns the original; second run: whatever the env picked."""

    def __init__(self, first_model="ollama:rebirth-js-ts:7b", first_draft=ORIGINAL, status="success"):
        self.calls = []
        self.first = (first_model, first_draft, status)

    def invoke(self, state):
        self.calls.append((os.environ.get("REBIRTH_GENERATOR"), os.environ.get("OLLAMA_MODEL")))
        if len(self.calls) == 1:
            model, draft, status = self.first
            return {"status": status, "current_draft": draft, "iteration_count": 1,
                    "trace": [f"generate_code: iteration=1 model={model} chars=10", "success: tests passed"]}
        return {"status": "success", "current_draft": "const x = 1;\n", "iteration_count": 2,
                "trace": [f"generate_code: iteration=1 model={os.environ['OLLAMA_MODEL']} chars=12"]}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for k in ("REBIRTH_FALLBACK", "REBIRTH_FALLBACK_MODEL", "OLLAMA_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("REBIRTH_GENERATOR", "auto")


def models(*names):
    return lambda: list(names)


def test_unchanged_by_the_served_adapter_reruns_with_the_14b() -> None:
    g = Graph()
    out = fallback.invoke(g, {}, original=ORIGINAL, list_models=models(fallback.PLAIN_14B, "rebirth-js-ts:7b"))
    assert g.calls == [("auto", None), ("ollama", fallback.PLAIN_14B)]
    assert out["current_draft"] == "const x = 1;\n" and out["fallback_model"] == fallback.PLAIN_14B
    assert any(t.startswith("fallback: ollama:rebirth-js-ts:7b returned the file unchanged") for t in out["trace"])
    assert os.environ["REBIRTH_GENERATOR"] == "auto" and "OLLAMA_MODEL" not in os.environ  # restored


def test_model_choice_and_skip() -> None:
    assert fallback.pick_model(models("rebirth-xlang:14b")) == "rebirth-xlang:14b"  # no plain 14B installed
    out = fallback.invoke(Graph(), {}, original=ORIGINAL, list_models=models("rebirth-js-ts:7b"))
    assert "fallback_model" not in out and out["trace"][-1].startswith("fallback: skipped")
    assert fallback.pick_model(lambda: (_ for _ in ()).throw(ConnectionError())) is None


def test_model_override(monkeypatch) -> None:
    monkeypatch.setenv("REBIRTH_FALLBACK_MODEL", "my-coder:32b")
    g = Graph()
    fallback.invoke(g, {}, original=ORIGINAL, list_models=models("my-coder:32b", fallback.PLAIN_14B))
    assert g.calls[1] == ("ollama", "my-coder:32b")


@pytest.mark.parametrize("case", ["flag_off", "env_off", "plain_backend", "changed", "failed", "plain_model", "seed"])
def test_no_fallback(case, monkeypatch) -> None:
    kw: dict = {"original": ORIGINAL, "list_models": models(fallback.PLAIN_14B)}
    g = Graph()
    if case == "flag_off":
        kw["use"] = False
    elif case == "env_off":
        monkeypatch.setenv("REBIRTH_FALLBACK", "0")
    elif case == "plain_backend":
        kw["backend"] = "ollama"
    elif case == "changed":
        g = Graph(first_draft="let x = 1;\n")
    elif case == "failed":
        g = Graph(status="human_review")
    elif case == "plain_model":
        g = Graph(first_model=fallback.PLAIN_14B)  # not routed: auto already used the plain model
    elif case == "seed":
        kw["seed"] = "var x = ;\n"  # the run repaired a broken seed: a real change
    out = fallback.invoke(g, {}, **kw)
    assert len(g.calls) == 1 and "fallback_model" not in out


def test_batch_report_names_the_fallback(tmp_path, monkeypatch) -> None:
    from rebirth_ai.graph import batch

    (tmp_path / "src").mkdir()
    (tmp_path / "test").mkdir()
    (tmp_path / "src" / "a.js").write_text(ORIGINAL, encoding="utf-8")
    (tmp_path / "test" / "a.test.js").write_text("import '../src/a.js';\n", encoding="utf-8")
    monkeypatch.setattr(batch, "build_migration_graph", lambda: Graph())
    monkeypatch.setattr(fallback, "pick_model", lambda list_models=None: fallback.PLAIN_14B)
    report = batch.migrate_repository(tmp_path, objective="modernize", under="src", write=True)
    (r,) = report.results
    assert r.status == "success" and r.wrote and r.fallback_model == fallback.PLAIN_14B
    assert (tmp_path / "src" / "a.js").read_text() == "const x = 1;\n"
    body = batch.render_pr_body(report)
    assert f"by the fallback {fallback.PLAIN_14B}" in body and "evidence, not proof" in body
