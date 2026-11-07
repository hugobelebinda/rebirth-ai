"""Phase 14: shared generation plan, vLLM eval (fake OpenAI server), pipeline + infra files."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import yaml

from rebirth_ai.generate.peft_client import (
    _SFT_STOP_STRINGS,
    _chat_max_new_tokens,
    _sft_max_new_tokens,
    plan_generation,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))


# --------------------------------------------------------------------------- generation plan


def _plan(style: str, legacy: str):
    return plan_generation(prompt_style=style, objective="Use const", legacy_code=legacy,
                           file_path="a.js", language="javascript")


def test_plan_short_sft_uses_completion_prompt_and_caps() -> None:
    p = _plan("sft", "var x = 1;\n")
    assert p.style == "sft" and p.prompt.startswith("### Objective\nUse const")
    assert p.max_new_tokens == _sft_max_new_tokens("var x = 1;\n", 1024) == 256
    assert list(p.stop) == _SFT_STOP_STRINGS


def test_plan_long_file_or_chat_style_uses_chat() -> None:
    long = "\n".join(f"var x{i} = {i};" for i in range(100))
    assert _plan("sft", long).style == "chat"
    p = _plan("chat", "var x = 1;\n")
    assert p.style == "chat" and p.stop == ()
    assert p.max_new_tokens == _chat_max_new_tokens("var x = 1;\n", 1024)


# --------------------------------------------------------------------------- vLLM eval


class _FakeVLLM(BaseHTTPRequestHandler):
    rows: list[dict] = []
    seen: list[tuple[str, dict]] = []

    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((self.path, body))
        text_in = body.get("prompt") or body["messages"][-1]["content"]
        gold = next(r["modern"] for r in self.rows if r["legacy"] in text_in)
        if self.path.endswith("/completions") and "prompt" in body:
            out = {"choices": [{"text": gold + "\n```\nextra prose"}]}  # leak cleaned by the shared cleaner
        else:
            # Fence-first, like real base-model chat replies (preamble prose is a known limitation of
            # the shared cleaner — see PHASE_14 "Known issues").
            out = {"choices": [{"message": {"content": f"```javascript\n{gold}```\n"}}]}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def fake_vllm(tmp_path):
    rows = [
        {"objective": "Use const", "legacy": "var a = 1;\n", "modern": "const a = 1;\n", "language": "javascript"},
        {"objective": "Arrow fn", "legacy": "function f(x) { return x; }\n",
         "modern": "const f = (x) => x;\n", "language": "javascript"},
    ]
    ef = tmp_path / "js_ts.jsonl"
    ef.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _FakeVLLM.rows, _FakeVLLM.seen = rows, []
    srv = HTTPServer(("127.0.0.1", 0), _FakeVLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1", ef
    srv.shutdown()


def _args(server, ef, tmp_path, style):
    import argparse

    return argparse.Namespace(server=server, model="js14b", style=style, lang="js_ts", eval_file=ef, n=100,
                              out=tmp_path / "r.json", concurrency=4, max_new_tokens=1024, timeout=10)


@pytest.mark.parametrize("style", ["sft", "chat"])
def test_eval_vllm_matches_shared_schema_and_requests(fake_vllm, tmp_path, style) -> None:
    import eval_vllm
    from eval_behavioral import summarize

    server, ef = fake_vllm
    rep = eval_vllm.run(_args(server, ef, tmp_path, style))
    suite = rep["suites"]["synthetic"]
    assert suite["n"] == 2 and suite["soft_exact"] == 2 and suite["parses"] == 2
    assert set(summarize([], suite="s", backend="b", lang=None, model=None)) <= set(suite)
    paths = {p for p, _ in _FakeVLLM.seen}
    body = _FakeVLLM.seen[0][1]
    assert body["temperature"] == 0.0 and body["model"] == "js14b"
    if style == "sft":
        assert paths == {"/v1/completions"} and body["stop"] == _SFT_STOP_STRINGS
        assert body["max_tokens"] == 256
    else:
        assert paths == {"/v1/chat/completions"}
        assert [m["role"] for m in body["messages"]] == ["system", "user"]


# --------------------------------------------------------------------------- pipeline + infra files


def test_dvc_pipeline_references_real_files_and_freezes_evalsets() -> None:
    dvc = yaml.safe_load((ROOT / "dvc.yaml").read_text())
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    assert dvc["stages"]["evalsets"]["frozen"] is True
    for stage in ("train_v3", "eval_v2", "eval_v3"):
        assert dvc["stages"][stage]["foreach"] == "${v3}"
    assert "--recipe v3_expanded" in dvc["stages"]["train_v3"]["do"]["cmd"]
    assert (ROOT / "configs/recipes/v3_expanded.yaml").is_file()
    for lang in params["v3"]:
        assert (ROOT / f"data/datasets/evalsets/{lang}.jsonl").is_file()
        assert (ROOT / f"data/models/lora_{lang}_7b/adapter.dvc").is_file()


def test_requirements_cuda_is_rocm_free_and_pins_torch() -> None:
    lines = (ROOT / "requirements-cuda.txt").read_text().splitlines()
    assert any(ln.startswith("--extra-index-url https://download.pytorch.org/whl/cu") for ln in lines)
    assert "torch==2.9.1" in lines
    assert not [ln for ln in lines if "rocm" in ln.lower()]
    lock = (ROOT / "requirements-lock.txt").read_text()
    assert "torch==2.9.1+rocm6.4" in lock


@pytest.mark.parametrize("script", sorted(p.name for p in (ROOT / "scripts/cloud").glob("*.sh")))
def test_cloud_scripts_parse(script: str) -> None:
    assert subprocess.run(["bash", "-n", str(ROOT / "scripts/cloud" / script)]).returncode == 0


def test_cloud_scripts_dry_run_never_connect(tmp_path) -> None:
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "DRY_RUN": "1", "REBIRTH_GPU_HOST": "203.0.113.10"}
    out = subprocess.run(["bash", str(ROOT / "scripts/cloud/run_job.sh"), "j", "echo", "hi"],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("[dry-run] ssh")


def test_terraform_never_launches_gpu_by_default() -> None:
    tf = (ROOT / "infra/aws/variables.tf").read_text()
    block = tf[tf.index('variable "launch_instance"'):]
    assert "default     = false" in block.split("}")[0]
    main = (ROOT / "infra/aws/main.tf").read_text()
    assert "count                                = var.launch_instance ? 1 : 0" in main
    assert 'instance_initiated_shutdown_behavior = "stop"' in main


def test_terraform_validates_if_available() -> None:
    tf = next(iter(sorted((ROOT / ".tools").glob("terraform_*/terraform"))), None) if (ROOT / ".tools").is_dir() else None
    if tf is None or not (ROOT / "infra/aws/.terraform").is_dir() or shutil.which("bash") is None:
        pytest.skip("terraform not installed locally")
    out = subprocess.run([str(tf), "validate", "-no-color"], cwd=ROOT / "infra/aws", capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_14b_default_out_never_overwrites_7b_champion(tmp_path) -> None:
    """--lang go --recipe cloud_14b without --out must land in lora_go_14b, not lora_go_7b."""
    out = subprocess.run(
        [sys.executable, str(ROOT / "training/train_lora.py"), "--lang", "go", "--plan", "--recipe", "cloud_14b"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert out.returncode == 0, out.stderr
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("out →"))
    assert line.endswith("data/models/lora_go_14b")

