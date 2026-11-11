"""training/merge_lora.py: shard-by-shard merge == PEFT's own merge_and_unload (tiny random Qwen2)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

ROOT = Path(__file__).resolve().parents[1]


def _merge_module():
    spec = importlib.util.spec_from_file_location("merge_lora", ROOT / "training" / "merge_lora.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny(tmp_path: Path):
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                   tie_word_embeddings=False)
    model = transformers.Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    base = tmp_path / "base"
    model.save_pretrained(base, max_shard_size="40KB")  # several shards + an index, like the 14B
    lcfg = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                              "gate_proj", "up_proj", "down_proj"])
    pm = peft.get_peft_model(transformers.Qwen2ForCausalLM.from_pretrained(base, torch_dtype=torch.float32), lcfg)
    for n, p in pm.named_parameters():  # B starts at zero: give the adapter a real effect
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.05)
    adapter = tmp_path / "adapter"
    pm.save_pretrained(adapter)
    return base, adapter, pm


def test_merge_matches_peft(tmp_path: Path):
    base, adapter, pm = _tiny(tmp_path)
    assert len(json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]) > 0
    out = tmp_path / "merged"
    stats = _merge_module().merge(base, adapter, out)
    assert stats["lora_weights"] == 2 * 7 and stats["scale"] == 2.0

    reference = pm.merge_and_unload().to(torch.bfloat16)
    merged = transformers.Qwen2ForCausalLM.from_pretrained(out, torch_dtype=torch.bfloat16)
    ref_sd, got_sd = reference.state_dict(), merged.state_dict()
    for k, v in ref_sd.items():
        assert torch.allclose(got_sd[k].float(), v.float(), atol=1e-2, rtol=1e-2), k
    ids = torch.tensor([[1, 5, 9, 13]])
    with torch.no_grad():
        a, b = reference(ids).logits.float(), merged(ids).logits.float()
    assert torch.allclose(a, b, atol=5e-2), (a - b).abs().max()


def test_delete_base_and_guards(tmp_path: Path):
    base, adapter, _ = _tiny(tmp_path)
    mod = _merge_module()
    shards = sorted(base.glob("*.safetensors"))
    mod.merge(base, adapter, tmp_path / "m", delete_base=True)
    assert not any(p.exists() for p in shards)
    assert sorted(p.name for p in (tmp_path / "m").glob("*.safetensors")) == [p.name for p in shards]
    assert (tmp_path / "m" / "config.json").is_file()

    cfg = json.loads((adapter / "adapter_config.json").read_text())
    cfg["use_dora"] = True
    (adapter / "adapter_config.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="DoRA"):
        mod.load_adapter(adapter)
