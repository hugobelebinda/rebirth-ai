"""Phase 13 MLOps: recipes, post-hoc tracking helpers, registry gate (temp MLflow store)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rebirth_ai import tracking
from rebirth_ai.recipes import RECIPES_DIR, load_recipe

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- recipes


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--quant", default="none")
    p.add_argument("--eos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--recipe", default=None)
    return p


def test_recipe_sets_defaults_and_cli_wins(tmp_path) -> None:
    r = tmp_path / "r.yaml"
    r.write_text("name: t\nlora-r: 16\nquant: 4bit\neos: false\ndata: x/y.jsonl\n")
    p = _parser()
    defaults, meta = load_recipe(r, p)
    assert defaults == {"lora_r": 16, "quant": "4bit", "eos": False, "data": Path("x/y.jsonl")}
    assert meta["recipe_name"] == "t"
    p.set_defaults(**defaults)
    args = p.parse_args(["--lora-r", "32"])
    assert args.lora_r == 32 and args.quant == "4bit" and args.eos is False


def test_recipe_unknown_key_is_an_error(tmp_path) -> None:
    r = tmp_path / "bad.yaml"
    r.write_text("lora_rr: 16\n")
    with pytest.raises(ValueError, match="unknown option"):
        load_recipe(r, _parser())


@pytest.mark.parametrize("name", sorted(p.stem for p in RECIPES_DIR.glob("*.yaml")))
def test_shipped_recipes_resolve_in_train_lora(name: str, tmp_path) -> None:
    """Every configs/recipes/*.yaml is accepted by train_lora.py --plan (no torch import)."""
    out = subprocess.run(
        [sys.executable, str(ROOT / "training" / "train_lora.py"), "--lang", "go", "--plan",
         "--recipe", name, "--out", str(tmp_path / "out")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert out.returncode == 0, out.stderr
    assert "recipe →" in out.stdout


# --------------------------------------------------------------------------- tracking helpers


def test_flatten_and_jsonl_tolerate_torn_lines(tmp_path) -> None:
    assert tracking.flatten({"a": {"b": 1, "c": None}, "d": "x"}) == {"a.b": "1", "d": "x"}
    f = tmp_path / "s.jsonl"
    f.write_text('{"step": 1}\n{"step": 2}\n{"ste')  # hard reset mid-write
    assert [r["step"] for r in tracking.read_jsonl(f)] == [1, 2]


def test_base_label() -> None:
    assert tracking.base_label("Qwen/Qwen2.5-Coder-7B-Instruct") == "7b"
    assert tracking.base_label("Qwen/Qwen2.5-Coder-14B-Instruct") == "14b"
    assert tracking.base_label(None) == "unknown"


def test_run_out_dir_from_argv() -> None:
    rj = {"cmd": ["python", "train_lora.py", "--lang", "go", "--out", "data/models/exp/go_X"]}
    assert tracking.run_out_dir(rj) == (ROOT / "data/models/exp/go_X").resolve()
    rj = {"cmd": ["python", "train_lora.py", "--lang", "rust"]}
    assert tracking.run_out_dir(rj).name == "lora_rust_7b"


def test_find_run_dirs_follows_resume_chain(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tracking, "LOGS_TRAIN", tmp_path)
    out = ROOT / "data/models/exp/go_fake"

    def mk(name: str, start: str, first_step: int) -> Path:
        d = tmp_path / name
        d.mkdir()
        (d / "run.json").write_text(json.dumps({
            "start": start,
            "cmd": ["python", "train_lora.py", "--lang", "go", "--max-length", "512", "--out", str(out)],
        }))
        (d / "steps.jsonl").write_text(json.dumps({"event": "step_begin", "step": first_step}) + "\n")
        return d

    old = mk("go_a", "2026-11-05T06:00:00-05:00", 1)        # unrelated earlier full run
    first = mk("go_b", "2026-11-05T20:00:00-05:00", 1)       # killed at night
    resumed = mk("go_c", "2026-11-06T06:00:00-05:00", 51)    # --resume from checkpoint-50
    meta = {"hyperparams": {"max_length": 512}, "trained_at": "2026-11-06T12:00:00+00:00"}
    assert tracking.find_run_dirs(out, meta) == [first, resumed]
    assert old not in tracking.find_run_dirs(out, meta)


def test_eval_set_id_depends_on_rows(tmp_path) -> None:
    rep = {"suites": {"synthetic": {"suite": "heldout_split:go", "n": 24, "lang": "go"}}}
    assert tracking.eval_set_id(rep) == "heldout_split:go:n24"
    ef = tmp_path / "go.jsonl"
    ef.write_text('{"a": 1}\n')
    sid = tracking.eval_set_id({"suites": {"synthetic": {"n": 100, "lang": "go"}}}, ef)
    assert sid.startswith("evalset:go:") and sid.endswith(":n100")


def test_tracking_disabled_by_env(monkeypatch) -> None:
    monkeypatch.setenv("REBIRTH_TRACKING", "0")
    assert tracking.enabled() is False
    assert tracking.try_log_eval(Path("nope.json"), model_label="x") is None


# --------------------------------------------------------------------------- registry (needs MLflow)

@pytest.fixture()
def store(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    pytest.importorskip("sqlalchemy")
    monkeypatch.setenv("REBIRTH_MLFLOW_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.delenv("REBIRTH_TRACKING", raising=False)
    monkeypatch.setattr(tracking, "TRACKING_DIR", tmp_path)
    monkeypatch.setattr(tracking, "ARTIFACT_ROOT", tmp_path / "artifacts")
    return tmp_path


def _fake_adapter(root: Path, name: str, trained_at: str, loss: float) -> Path:
    out = root / name
    (out / "adapter").mkdir(parents=True)
    (out / "adapter" / "adapter_config.json").write_text("{}")
    (out / "rebirth_meta.json").write_text(json.dumps({
        "language": "go", "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct", "trained_at": trained_at,
        "train_loss": loss, "global_steps": 10, "train_rows": 80,
        "hyperparams": {"lora_r": 8, "lora_alpha": 16, "epochs": 1, "max_length": 512},
        "recipe": {"quant": "4bit", "features": {"kept": 80, "dropped_overlong": 2}},
    }))
    return out


def _report(path: Path, soft: int, parses: int) -> Path:
    path.write_text(json.dumps({"backend": "peft", "suites": {"synthetic": {
        "suite": "evalset:go", "backend": "peft", "lang": "go", "n": 100, "soft_exact": soft,
        "parses": parses, "mean_symbol_jaccard": 0.9, "mean_token_overlap": 0.8,
    }}}))
    return path


def test_register_track_gate_promote_rollback(store) -> None:
    from rebirth_ai import registry

    ef = store / "go.jsonl"
    ef.write_text('{"row": 1}\n')
    champ_dir = _fake_adapter(store, "lora_go_7b", "2026-11-05T10:00:00+00:00", 0.22)
    run_a = tracking.log_training(champ_dir, run_dirs=[])
    assert tracking.log_training(champ_dir, run_dirs=[]) == run_a  # idempotent
    v_a = registry.find_version("go", champ_dir)
    assert v_a == "1" and registry.alias_version("go", registry.CHALLENGER) == "1"
    registry.promote("go", v_a, reason="first champion")
    assert registry.alias_version("go", registry.CHAMPION) == "1"
    assert registry.champion_adapter("go") is not None

    tracking.log_eval(_report(store / "a.json", 14, 99), model_label="adapter", adapter=champ_dir / "adapter", eval_file=ef)
    set_id = tracking.eval_set_id(json.loads((store / "a.json").read_text()), ef)

    # Challenger 1: more soft-exact but 2 fewer parses → fails (parses >= champion - 1).
    c_dir = _fake_adapter(store, "go_C", "2026-11-05T20:00:00+00:00", 0.18)
    tracking.log_training(c_dir, run_dirs=[])
    v_c = registry.find_version("go", c_dir)
    assert registry.gate("go", v_c, set_id)["decision"] == "needs_eval"
    tracking.log_eval(_report(store / "c.json", 15, 97), model_label="adapter", adapter=c_dir / "adapter", eval_file=ef)
    res = registry.gate("go", v_c, set_id)
    assert res["passed"] is False and res["decision"] == "kept_champion"
    assert registry.alias_version("go", registry.CHAMPION) == v_a

    # Challenger 1b: ties on soft-exact (and parses within slack) → not enough, incumbent stays.
    t_dir = _fake_adapter(store, "go_tie", "2026-11-05T21:00:00+00:00", 0.18)
    tracking.log_training(t_dir, run_dirs=[])
    v_t = registry.find_version("go", t_dir)
    tracking.log_eval(_report(store / "t.json", 14, 98), model_label="adapter", adapter=t_dir / "adapter", eval_file=ef)
    res = registry.gate("go", v_t, set_id)
    assert res["passed"] is False and res["checks"]["soft_exact>=champion+1"] is False
    assert registry.gate("go", v_t, set_id, min_gain=0, apply=False)["passed"] is True

    # Challenger 2: passes → promoted, old champion becomes "previous".
    d_dir = _fake_adapter(store, "go_D1", "2026-11-06T02:00:00+00:00", 0.18)
    tracking.log_training(d_dir, run_dirs=[])
    v_d = registry.find_version("go", d_dir)
    tracking.log_eval(_report(store / "d.json", 16, 99), model_label="adapter", adapter=d_dir / "adapter", eval_file=ef)
    assert registry.gate("go", v_d, set_id, apply=False)["decision"] == "pass_not_applied"
    assert registry.alias_version("go", registry.CHAMPION) == v_a
    assert registry.gate("go", v_d, set_id)["decision"] == "promoted"
    assert registry.alias_version("go", registry.CHAMPION) == v_d
    assert registry.alias_version("go", registry.PREVIOUS) == v_a

    assert registry.rollback("go") == v_a
    assert registry.alias_version("go", registry.CHAMPION) == v_a


def test_log_eval_idempotent_and_reference_models(store) -> None:
    rep = _report(store / "base.json", 5, 100)
    r1 = tracking.log_eval(rep, model_label="base-7b-chat", lang="go")
    assert tracking.log_eval(rep, model_label="base-7b-chat", lang="go") == r1
    run = tracking.client().get_run(r1)
    assert run.data.metrics["soft_exact"] == 5 and run.data.tags["rebirth.model"] == "base-7b-chat"
