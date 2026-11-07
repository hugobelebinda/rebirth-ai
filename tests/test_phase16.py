"""Phase 16: Ollama adapter client, serving gate, routing, packaging helpers, served-score keys."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from rebirth_ai import packaging, routing
from rebirth_ai.generate.ollama_adapter_client import (
    IM_END,
    OllamaAdapterClient,
    chatml,
    num_ctx_for,
)
from rebirth_ai.generate.peft_client import _SFT_STOP_STRINGS, plan_generation
from rebirth_ai.tracking import registry_set_key

# --------------------------------------------------------------------------- client


def test_chatml_matches_qwen_template() -> None:
    # Exact output of Qwen2.5-Coder-7B-Instruct's apply_chat_template(add_generation_prompt=True).
    assert chatml("SYS", "U") == (
        "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"
    )


def _plan(legacy: str, **kw):
    return plan_generation(prompt_style="sft", objective="Use const", legacy_code=legacy,
                           file_path="a.js", language="javascript", **kw)


def test_payload_sft_is_raw_with_sft_stops() -> None:
    p = OllamaAdapterClient(model="m").request_payload(_plan("var x = 1;\n"))
    assert p["raw"] is True and p["prompt"].startswith("### Objective\nUse const")
    assert p["options"]["stop"] == _SFT_STOP_STRINGS
    assert p["options"]["temperature"] == 0.0 and p["options"]["num_predict"] == 256
    assert p["options"]["num_ctx"] == 4096


def test_payload_retry_uses_chatml_and_im_end() -> None:
    p = OllamaAdapterClient(model="m").request_payload(_plan("var x = 1;\n", error_class="syntax"))
    assert p["prompt"].startswith("<|im_start|>system\n") and p["prompt"].endswith("<|im_start|>assistant\n")
    assert p["options"]["stop"] == [IM_END]


def test_num_ctx_buckets() -> None:
    assert num_ctx_for("x" * 300, 256) == 4096
    assert num_ctx_for("x" * 30000, 4096) == 16384
    assert num_ctx_for("x" * 10**6, 4096) == 32768


class _FakeOllama(BaseHTTPRequestHandler):
    seen: list[dict] = []
    reply = ""

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append({"path": self.path, **body})
        data = json.dumps({"response": type(self).reply}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def fake_ollama():
    _FakeOllama.seen = []
    srv = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_migrate_cleans_sft_leak(fake_ollama) -> None:
    _FakeOllama.reply = "const x = 1;\n```\nThis rewrite uses const."
    res = OllamaAdapterClient(model="rebirth-js-ts:7b", host=fake_ollama).migrate(
        objective="Use const", legacy_code="var x = 1;\n", file_path="a.js")
    assert res.code == "const x = 1;\n" and res.model == "ollama:rebirth-js-ts:7b"
    assert _FakeOllama.seen[0]["path"] == "/api/generate" and _FakeOllama.seen[0]["raw"] is True


def test_migrate_chat_reply_with_preamble(fake_ollama) -> None:
    _FakeOllama.reply = "Here is the fixed code:\n\n```javascript\nconst x = 1;\n```\n"
    res = OllamaAdapterClient(model="m", host=fake_ollama).migrate(
        objective="Use const", legacy_code="var x = 1;\n", file_path="a.js", error_class="syntax")
    assert res.code == "const x = 1;\n"


# --------------------------------------------------------------------------- serving gate


S = {"n": 100, "soft_exact": 13, "parses": 99}


def test_serve_gate_passes_within_fidelity_slack() -> None:
    g = packaging.serve_gate(S, {"soft_exact": 14, "parses": 99}, {"soft_exact": 5, "parses": 100})
    assert g.passed and not g.warnings


def test_serve_gate_fails_on_fidelity_loss() -> None:
    g = packaging.serve_gate(S, {"soft_exact": 16, "parses": 99}, {"soft_exact": 5, "parses": 100})
    assert not g.passed and g.checks["soft_exact>=peft-2"] is False


def test_serve_gate_fails_below_base_and_warns_on_parses() -> None:
    g = packaging.serve_gate({"n": 93, "soft_exact": 7, "parses": 90}, {"soft_exact": 8, "parses": 91},
                             {"soft_exact": 8, "parses": 93})
    assert not g.passed and g.checks["soft_exact>=base"] is False
    assert any("parses" in w for w in g.warnings)


def test_serve_gate_without_any_reference_never_passes() -> None:
    assert not packaging.serve_gate(S, None, None).passed


# --------------------------------------------------------------------------- packaging helpers


def test_names_and_modelfile() -> None:
    assert packaging.ollama_name("js_ts") == "rebirth-js-ts"
    assert packaging.version_tag("csharp", "7b", "3") == "rebirth-csharp:7b-v3"
    assert packaging.serving_alias("go", "14b") == "rebirth-go:14b"
    mf = packaging.modelfile("qwen2.5-coder:7b-instruct", packaging.GGUF_DIR / "go.gguf")
    assert mf.startswith("FROM qwen2.5-coder:7b-instruct\nADAPTER ")
    assert 'TEMPLATE """{{ .Prompt }}"""' in mf and mf.count("PARAMETER stop") == len(packaging.SFT_STOPS)
    assert tuple(_SFT_STOP_STRINGS) == packaging.SFT_STOPS


def test_served_scores_never_overwrite_primary_registry_key() -> None:
    sid = "evalset:go:abc:n100"
    assert registry_set_key(sid, "adapter", None) == sid
    assert registry_set_key(sid, "adapter", {"rebirth.engine": "vllm"}) == sid
    assert registry_set_key(sid, "served-7b-q4", {"rebirth.engine": "ollama"}) == f"served:served-7b-q4:{sid}"


# --------------------------------------------------------------------------- routing


@pytest.fixture()
def routes(tmp_path, monkeypatch):
    p = tmp_path / "routing.yaml"
    monkeypatch.setenv("REBIRTH_ROUTING", str(p))
    return p


def test_auto_backend_routes_packaged_language_and_falls_back(routes) -> None:
    from rebirth_ai.generate.factory import get_generator
    from rebirth_ai.generate.ollama_client import OllamaClient

    routing.set_route("go", {"model": "rebirth-go:7b", "tag": "rebirth-go:7b-v1", "version": "1"})
    g = get_generator("auto", language="go")
    assert isinstance(g, OllamaAdapterClient) and g.model == "rebirth-go:7b"
    assert isinstance(get_generator("auto", language="rust"), OllamaClient)
    assert isinstance(get_generator("auto", language=None), OllamaClient)
    assert routes.read_text().startswith("# Generated by `rebirth package`")


def test_publish_keeps_history_and_rollback_repoints(routes, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(packaging, "ollama_alias", lambda src, dst: calls.append((src, dst)))
    monkeypatch.setattr(packaging, "ollama_has", lambda tag: True)
    monkeypatch.setattr(packaging, "write_model_card", lambda *a, **k: None)
    e1 = {"model": "rebirth-go:7b", "tag": "rebirth-go:7b-v1", "version": "1", "adapter": "data/models/a/adapter"}
    e2 = {**e1, "tag": "rebirth-go:7b-v2", "version": "2"}
    packaging.publish_route("go", e1, meta={})
    packaging.publish_route("go", e2, meta={})
    r = routing.route("go")
    assert r["version"] == "2" and set(r["history"]) == {"1"}

    from rebirth_ai import registry

    monkeypatch.setattr(registry, "alias_version", lambda lang, alias: "1")
    monkeypatch.setattr(registry, "rollback", lambda lang: "1")
    packaging.serve_rollback("go", log=lambda *_: None)
    r = routing.route("go")
    assert r["version"] == "1" and r["tag"] == "rebirth-go:7b-v1" and set(r["history"]) == {"2"}
    assert calls[-1] == ("rebirth-go:7b-v1", "rebirth-go:7b")


def test_rollback_refuses_without_packaged_build(routes, monkeypatch) -> None:
    from rebirth_ai import registry

    monkeypatch.setattr(packaging, "ollama_alias", lambda *a: None)
    monkeypatch.setattr(packaging, "write_model_card", lambda *a, **k: None)
    packaging.publish_route("go", {"model": "rebirth-go:7b", "tag": "rebirth-go:7b-v2", "version": "2",
                                   "adapter": "x/adapter"}, meta={})
    monkeypatch.setattr(registry, "alias_version", lambda lang, alias: "1")
    rolled: list[str] = []
    monkeypatch.setattr(registry, "rollback", lambda lang: rolled.append(lang) or "1")
    with pytest.raises(RuntimeError, match="no passing packaged build"):
        packaging.serve_rollback("go", log=lambda *_: None)
    assert rolled == []  # registry untouched when serving cannot follow
