"""Phase 11 per-language LoRA harness: registry, split, PEFT routing (no model download)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rebirth_ai import lora
from rebirth_ai.generate import peft_client as pc
from rebirth_ai.generate.factory import get_generator
from rebirth_ai.generate.ollama_client import OllamaClient

ROOT = Path(__file__).resolve().parents[1]


def _rows(n: int) -> list[dict]:
    return [{"objective": f"o{i}", "legacy": f"l{i}", "modern": f"m{i}"} for i in range(n)]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("javascript", "js_ts"),
        ("TypeScript", "js_ts"),
        ("py", "python"),
        ("c++", "cpp"),
        ("c#", "csharp"),
        ("go", "go"),
        ("cobol", None),
        (None, None),
    ],
)
def test_normalize_lang(raw, expected) -> None:
    assert lora.normalize_lang(raw) == expected


def test_lora_target_table_covers_all_langs() -> None:
    for lang_id in lora.LORA_LANGS:
        t = lora.lora_target(lang_id)
        assert t.data.name == f"{lang_id}_migration.jsonl"
        assert t.out.name == f"lora_{lang_id}_7b"
        assert t.adapter == t.out / "adapter"
        assert t.data.is_file(), f"missing corpus for {lang_id}"
    with pytest.raises(ValueError):
        lora.lora_target("cobol")


def test_split_is_deterministic_disjoint_and_sized() -> None:
    rows = _rows(4000)
    a = lora.split_rows(rows, eval_frac=0.05, seed=7)
    b = lora.split_rows(rows, eval_frac=0.05, seed=7)
    assert a.eval_indices == b.eval_indices
    assert len(a.train) + len(a.eval) == len(rows)
    assert 0.03 < len(a.eval) / len(rows) < 0.07
    c = lora.split_rows(rows, eval_frac=0.05, seed=8)
    assert c.eval_indices != a.eval_indices


def test_split_stable_under_append_and_dupes_stay_together() -> None:
    rows = _rows(1000)
    before = {json.dumps(r) for r in lora.split_rows(rows, eval_frac=0.1, seed=1).eval}
    grown = rows + _rows(1200)[1000:] + rows[:50]  # append new + duplicates
    after = lora.split_rows(grown, eval_frac=0.1, seed=1)
    after_eval = {json.dumps(r) for r in after.eval}
    assert before <= after_eval
    train_keys = {json.dumps(r) for r in after.train}
    assert not (train_keys & after_eval)


def test_split_zero_frac_keeps_everything() -> None:
    s = lora.split_rows(_rows(10), eval_frac=0.0, seed=1)
    assert len(s.train) == 10 and s.eval == []


def test_train_lora_plan_writes_split_index(tmp_path) -> None:
    data = tmp_path / "x.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in _rows(200)) + "\n", encoding="utf-8")
    out = tmp_path / "lora_x"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "training" / "train_lora.py"),
            "--data", str(data),
            "--out", str(out),
            "--eval-frac", "0.1",
            "--plan",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "rows=200" in proc.stdout
    idx = json.loads((out / "split.json").read_text(encoding="utf-8"))
    assert idx["n_total"] == 200
    assert idx["n_eval"] == len(idx["eval_indices"])
    assert idx["corpus_sha256"] == lora.file_sha256(data)


def _fake_lang_adapter(tmp_path, monkeypatch, lang_id: str, base: str = "base/x") -> Path:
    monkeypatch.setattr(lora, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(
        pc, "lora_target", lambda lang: lora.lora_target(lang, models_dir=tmp_path)
    )
    out = tmp_path / f"lora_{lang_id}_7b"
    (out / "adapter").mkdir(parents=True)
    (out / "rebirth_meta.json").write_text(json.dumps({"base_model": base}), encoding="utf-8")
    return out / "adapter"


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("REBIRTH_PEFT_BASE", "REBIRTH_PEFT_ADAPTER", "REBIRTH_GENERATOR"):
        monkeypatch.delenv(k, raising=False)


def test_resolve_uses_per_lang_adapter_and_meta_base(tmp_path, monkeypatch, clean_env) -> None:
    adapter = _fake_lang_adapter(tmp_path, monkeypatch, "python", base="Qwen/custom")
    assert pc.resolve_peft_paths(language="python") == ("Qwen/custom", adapter)


def test_env_override_beats_per_lang(tmp_path, monkeypatch, clean_env) -> None:
    _fake_lang_adapter(tmp_path, monkeypatch, "python")
    monkeypatch.setenv("REBIRTH_PEFT_ADAPTER", "/elsewhere/adapter")
    _, path = pc.resolve_peft_paths(language="python")
    assert path == Path("/elsewhere/adapter")


def test_factory_routes_peft_by_language(tmp_path, monkeypatch, clean_env) -> None:
    adapter = _fake_lang_adapter(tmp_path, monkeypatch, "python")
    client = get_generator("peft", language="python")
    assert isinstance(client, pc.PeftClient)
    assert client.adapter_path == adapter


def test_factory_falls_back_to_ollama_without_lang_adapter(tmp_path, monkeypatch, clean_env) -> None:
    monkeypatch.setattr(
        pc, "lora_target", lambda lang: lora.lora_target(lang, models_dir=tmp_path)
    )
    assert isinstance(get_generator("peft", language="go"), OllamaClient)
    # js_ts / unknown keep legacy PEFT behaviour (JS 7B → 0.5B)
    assert isinstance(get_generator("peft", language="javascript"), pc.PeftClient)
    assert isinstance(get_generator("peft"), pc.PeftClient)
    # explicit env override still forces PEFT
    monkeypatch.setenv("REBIRTH_PEFT_ADAPTER", "/elsewhere/adapter")
    assert isinstance(get_generator("peft", language="go"), pc.PeftClient)


def test_no_adapter_env_resolves_base_only(monkeypatch, clean_env) -> None:
    monkeypatch.setenv("REBIRTH_PEFT_ADAPTER", pc.NO_ADAPTER)
    client = get_generator("peft", language="python")
    assert isinstance(client, pc.PeftClient)
    assert str(client.adapter_path) == pc.NO_ADAPTER
    assert client.base_model == pc.DEFAULT_BASE_7B


def test_step_log_appends_fsynced_jsonl(tmp_path) -> None:
    sys.path.insert(0, str(ROOT / "training"))
    from train_lora import StepLog

    path = tmp_path / "run" / "steps.jsonl"
    log = StepLog(path)
    log.write("step_begin", step=1)
    log.write("error", kind="RuntimeError", message="HIP error")
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert [x["event"] for x in lines] == ["step_begin", "error"]
    assert lines[0]["step"] == 1 and "t" in lines[0]
    StepLog(None).write("noop")  # disabled log is a no-op


@pytest.fixture(scope="module")
def qwen_tok():
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    except Exception as exc:  # not cached on this machine
        pytest.skip(f"Qwen tokenizer unavailable: {exc}")


def _features(tok, rows, **kw):
    sys.path.insert(0, str(ROOT / "training"))
    from train_lora import build_features

    return build_features(rows, tok, **kw)


def test_build_features_masks_prompt_and_appends_eos(qwen_tok) -> None:
    sys.path.insert(0, str(ROOT / "training"))
    from train_lora import format_prompt

    row = {"objective": "use f-strings", "legacy": "x = '%s' % a\n", "modern": "x = f'{a}'\n"}
    feats, stats = _features(qwen_tok, [row], max_length=512)
    f = feats[0]
    n_prompt = len(qwen_tok(format_prompt(row), add_special_tokens=False)["input_ids"])
    assert f["labels"][:n_prompt] == [-100] * n_prompt
    assert f["input_ids"][-1] == qwen_tok.eos_token_id == f["labels"][-1]
    assert qwen_tok.eos_token_id != qwen_tok.pad_token_id  # EOS must not be masked as padding
    assert qwen_tok.decode(f["labels"][n_prompt:-1]) == row["modern"]
    assert stats["kept"] == 1 and stats["label_tokens"] == len(f["labels"]) - n_prompt


def test_build_features_drops_or_truncates_overlong(qwen_tok) -> None:
    row = {"objective": "o", "legacy": "a = 1\n" * 50, "modern": "b = 2\n" * 50}
    feats, stats = _features(qwen_tok, [row], max_length=64)
    assert feats == [] and stats["dropped_overlong"] == 1
    # legacy truncation keeps the row only if some target survives; here the prompt alone > 64
    feats, stats = _features(qwen_tok, [row], max_length=64, drop_overlong=False)
    assert feats == [] and stats["dropped_overlong"] == 1
    feats, stats = _features(
        qwen_tok, [row], max_length=64, drop_overlong=False, completion_only=False, eos=False
    )
    assert len(feats[0]["input_ids"]) == 64 and stats["truncated"] == 1


def test_sft_prompt_matches_training_prompt() -> None:
    sys.path.insert(0, str(ROOT / "training"))
    from train_lora import format_prompt

    from rebirth_ai.generate.prompts import build_sft_prompt

    row = {"objective": "o", "legacy": "x\n", "modern": "y\n"}
    assert format_prompt(row) == build_sft_prompt(objective="o", legacy_code="x\n")


def test_lora_gate_verdict() -> None:
    sys.path.insert(0, str(ROOT / "training"))
    from lora_gate import verdict

    base = {"n": 24, "soft_exact": 7, "parses": 24}
    assert verdict({"n": 24, "soft_exact": 7, "parses": 23}, base)["pass"]
    assert not verdict({"n": 24, "soft_exact": 4, "parses": 24}, base)["pass"]  # v1 Python case
    assert not verdict({"n": 24, "soft_exact": 9, "parses": 20}, base)["pass"]


def test_latest_checkpoint_picks_newest_complete(tmp_path) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
    from train_lora import latest_checkpoint

    assert latest_checkpoint(tmp_path / "missing") is None
    for step, complete in ((50, True), (100, True), (150, False)):
        d = tmp_path / f"checkpoint-{step}"
        d.mkdir()
        if complete:
            (d / "trainer_state.json").write_text("{}")
    assert latest_checkpoint(tmp_path).name == "checkpoint-100"
