"""rebirth install (Phase 19): manifest, plan, verified install from a folder, resume, refusals."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

import pytest
import yaml

from rebirth_ai import install as inst

ROOT = Path(__file__).resolve().parents[1]


def _fake_ollama(tmp_path: Path, monkeypatch, installed: set[str] | None = None) -> Path:
    """An `ollama` on PATH that logs its arguments; `show <tag>` succeeds only for installed tags."""
    bindir, log, state = tmp_path / "bin", tmp_path / "ollama.log", tmp_path / "installed.txt"
    bindir.mkdir()
    state.write_text("\n".join(sorted(installed or ())) + "\n")
    script = bindir / "ollama"
    script.write_text(f"""#!/usr/bin/env bash
echo "$@" >> "{log}"
case "$1" in
  show) grep -qxF "$2" "{state}" ;;
  create) echo "$2" >> "{state}" ;;
  cp) echo "$3" >> "{state}" ;;
  pull) echo "$2" >> "{state}" ;;
esac
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return log


def _manifest(tmp_path: Path, files: dict[str, bytes]) -> dict:
    tmpl = tmp_path / "x.Modelfile"
    tmpl.write_text("# test\nFROM {gguf}\nTEMPLATE \"\"\"{{ .Prompt }}\"\"\"\n")
    def entry(name: str, tag: str, alias: str, **extra) -> dict:
        data = files[name]
        return {"tag": tag, "alias": alias, "file": name, "local": name, "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(), **extra}
    return {"version": 1, "hf_namespace": "", "packs": {
        "modernize": {"hf_repo": "m", "ollama_base": "qwen2.5-coder:7b-instruct",
                      "models": [entry("go.gguf", "rebirth-go:7b-v1", "rebirth-go:7b")]},
        "migrate": {"hf_repo": "x", "models": [entry("x.gguf", "rebirth-xlang:14b-v2", "rebirth-xlang:14b",
                                                     modelfile=str(tmpl))]}}}


def _release_dir(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    files = {"go.gguf": b"lora-go" * 100, "x.gguf": b"merged-14b" * 100}
    d = tmp_path / "release"
    d.mkdir()
    for k, v in files.items():
        (d / k).write_bytes(v)
    return d, files


def test_shipped_manifest_is_consistent():
    m = inst.load_manifest()
    assert set(m["packs"]) == set(inst.PACKS)
    routes = yaml.safe_load((ROOT / "configs" / "routing.yaml").read_text())["languages"]
    mod = m["packs"]["modernize"]["models"]
    assert {x["lang"] for x in mod} == set(routes) and len(mod) == 8
    for x in mod:
        assert routes[x["lang"]]["model"] == x["alias"]  # rebirth migrate routes to what install creates
    for pack in m["packs"].values():
        for x in pack["models"]:
            assert re.fullmatch(r"[0-9a-f]{64}", x["sha256"]) and x["size"] > 0 and x["file"].endswith(".gguf")
    xl = m["packs"]["migrate"]["models"][0]
    text = (ROOT / xl["modelfile"]).read_text()
    assert "FROM {gguf}" in text and "TEMPLATE" in text and "<|im_start|>" in text


def test_packs_and_unpublished_hub():
    assert inst.packs_for("both") == inst.PACKS and inst.packs_for("migrate") == ("migrate",)
    with pytest.raises(inst.InstallError):
        inst.packs_for("everything")
    with pytest.raises(inst.InstallError, match="not published yet"):
        inst.plan("both", manifest=inst.load_manifest() | {"hf_namespace": ""}, has=lambda t: False)
    m = inst.load_manifest() | {"hf_namespace": "someone"}
    steps = inst.plan("migrate", manifest=m, has=lambda t: False)
    assert steps[0].source.startswith("https://huggingface.co/someone/rebirth-xlang-14b-gguf/resolve/main/")


def test_install_from_folder(tmp_path, monkeypatch):
    rel, files = _release_dir(tmp_path)
    log = _fake_ollama(tmp_path, monkeypatch)
    monkeypatch.setattr(inst, "INSTALL_DIR", tmp_path / "ollama")
    m = _manifest(tmp_path, files)
    steps = inst.install("both", from_dir=rel, manifest=m, progress=lambda s: None)
    assert [s.action for s in steps] == ["install", "install"]
    calls = log.read_text().splitlines()
    assert "pull qwen2.5-coder:7b-instruct" in calls
    assert any(c.startswith("create rebirth-go:7b-v1 -f") for c in calls)
    assert "cp rebirth-go:7b-v1 rebirth-go:7b" in calls and "cp rebirth-xlang:14b-v2 rebirth-xlang:14b" in calls
    go_mf = (tmp_path / "ollama" / "rebirth-go_7b-v1" / "Modelfile").read_text()
    assert go_mf.startswith("FROM qwen2.5-coder:7b-instruct\nADAPTER ") and "### Legacy" in go_mf
    x_mf = (tmp_path / "ollama" / "rebirth-xlang_14b-v2" / "Modelfile").read_text()
    assert f"FROM {(rel / 'x.gguf').resolve()}" in x_mf
    assert (rel / "x.gguf").exists()  # files from --from are never deleted
    # second run: everything already there
    again = inst.install("both", from_dir=rel, manifest=m, progress=lambda s: None)
    assert [s.action for s in again] == ["skip", "skip"]


def test_checksum_mismatch_stops_before_ollama(tmp_path, monkeypatch):
    rel, files = _release_dir(tmp_path)
    log = _fake_ollama(tmp_path, monkeypatch)
    monkeypatch.setattr(inst, "INSTALL_DIR", tmp_path / "ollama")
    m = _manifest(tmp_path, files)
    (rel / "x.gguf").write_bytes(b"tampered" * 125)  # same size, different content
    with pytest.raises(inst.InstallError, match="sha256"):
        inst.install("migrate", from_dir=rel, manifest=m, progress=lambda s: None)
    assert not log.exists() or "create" not in log.read_text()


def test_download_resumes(tmp_path, monkeypatch):
    payload = b"0123456789" * 1000
    seen = {}

    class Resp:
        def __init__(self, start):
            self.status_code = 206 if start else 200
            self.body = payload[start:]
            self.headers = {"content-length": str(len(self.body))}
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def iter_bytes(self, n):
            for i in range(0, len(self.body), 777):
                yield self.body[i:i + 777]

    def fake_stream(method, url, headers=None, **kw):
        rng = (headers or {}).get("Range", "bytes=0-")
        seen["range"] = rng
        return Resp(int(rng.split("=")[1].rstrip("-")))

    import httpx
    monkeypatch.setattr(httpx, "stream", fake_stream)
    dest = tmp_path / "f.gguf"
    (tmp_path / "f.gguf.part").write_bytes(payload[:4000])  # interrupted earlier
    inst.download("https://example.invalid/f.gguf", dest, size=len(payload), progress=lambda s: None)
    assert seen["range"] == "bytes=4000-" and dest.read_bytes() == payload
    assert not (tmp_path / "f.gguf.part").exists()


def test_status_counts(tmp_path):
    rel, files = _release_dir(tmp_path)
    m = _manifest(tmp_path, files)
    st = inst.status(m, has=lambda t: t == "rebirth-go:7b")
    assert st["modernize"] == {"installed": 1, "total": 1, "missing": []}
    assert st["migrate"]["installed"] == 0 and st["migrate"]["missing"] == ["rebirth-xlang:14b"]
    json.dumps(st)
