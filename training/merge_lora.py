#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into its base model's safetensors shards, one shard at a time.

  python training/merge_lora.py --base data/models/merge_xlang_v2/base \\
      --adapter data/models/lora_xlang_14b/adapter --out data/models/merge_xlang_v2/merged [--delete-base]

Why not ``PeftModel.merge_and_unload()``: that loads the whole model (a 14B in bf16 is ~30 GB) into
RAM, which the 30 GB target machine does not have. Here each shard is read, every targeted linear
weight gets ``W += (alpha / r) * B @ A`` (fp32 math, cast back to the shard's dtype), and the shard is
written under the same name; peak RAM ≈ one shard (~5 GB). ``--delete-base`` removes each input
shard once its merged copy is written, so the disk peak is one model, not two (the base can be
downloaded again). Non-weight files (config, tokenizer, index) are copied unchanged.

The result is a plain Hugging Face checkpoint → ``convert_hf_to_gguf.py`` → ``llama-quantize``
(see ``scripts/xlang_v2_merge.sh``). Serving merged weights removes the runtime LoRA overhead of
``ADAPTER`` in Ollama (~35 % per translation for the 14B).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "base_model.model."


def load_adapter(adapter_dir: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg.get("use_dora"):
        raise SystemExit("DoRA adapters are not supported by this merger")
    if cfg.get("modules_to_save"):
        raise SystemExit(f"modules_to_save={cfg['modules_to_save']} not supported (full-weight copies)")
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            if ".lora_A." in k:
                role, base_key = "A", k.replace(".lora_A.weight", ".weight")
            elif ".lora_B." in k:
                role, base_key = "B", k.replace(".lora_B.weight", ".weight")
            else:
                raise SystemExit(f"unexpected adapter tensor {k}")
            base_key = base_key.removeprefix(PREFIX)
            pairs.setdefault(base_key, {})[role] = f.get_tensor(k)
    bad = [k for k, v in pairs.items() if set(v) != {"A", "B"}]
    if bad:
        raise SystemExit(f"incomplete LoRA pairs: {bad[:3]}")
    return {k: (v["A"], v["B"]) for k, v in pairs.items()}, scale


def merge(base: Path, adapter: Path, out: Path, *, delete_base: bool = False) -> dict:
    lora, scale = load_adapter(adapter)
    index = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    out.mkdir(parents=True, exist_ok=True)
    for p in base.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors"):
            shutil.copy2(p, out / p.name)
    used, max_rel = 0, 0.0
    for name in shards:
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(base / name), "pt") as f:
            meta = f.metadata() or {"format": "pt"}
            for k in f.keys():
                t = f.get_tensor(k)
                if k in lora:
                    a, b = lora[k]
                    delta = (b.float() @ a.float()) * scale
                    if delta.shape != t.shape:
                        raise SystemExit(f"{k}: delta {tuple(delta.shape)} vs weight {tuple(t.shape)}")
                    merged = (t.float() + delta).to(t.dtype)
                    max_rel = max(max_rel, float(delta.norm() / t.float().norm()))
                    t = merged
                    used += 1
                tensors[k] = t.contiguous()
        save_file(tensors, str(out / name), metadata=meta)
        del tensors
        if delete_base:
            (base / name).unlink()
        print(f"merged {name} ({used}/{len(lora)} LoRA weights so far)", flush=True)
    if used != len(lora):
        missing = sorted(set(lora) - set(index["weight_map"]))[:3]
        raise SystemExit(f"only {used}/{len(lora)} LoRA weights matched the base; e.g. {missing}")
    stats = {"lora_weights": used, "scale": scale, "max_relative_delta": round(max_rel, 5),
             "base": str(base), "adapter": str(adapter), "out": str(out)}
    (out / "rebirth_merge.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="HF checkpoint dir (safetensors shards + index)")
    ap.add_argument("--adapter", type=Path, required=True, help="PEFT adapter dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--delete-base", action="store_true", help="remove each base shard after merging it")
    a = ap.parse_args()
    if a.out.resolve() == a.base.resolve():
        print("--out must differ from --base", file=sys.stderr)
        return 2
    print(json.dumps(merge(a.base, a.adapter, a.out, delete_base=a.delete_base)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
