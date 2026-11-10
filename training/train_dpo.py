#!/usr/bin/env python3
"""DPO on top of an SFT LoRA adapter (Phase 15 S4; smoke-tested locally on the 7B first).

The SFT adapter is loaded *trainable*; TRL makes a frozen copy of it (adapter "ref") as the
reference model, so DPO moves the policy away from its own failures (``build_dpo_pairs.py``)
relative to the SFT champion, not relative to the base model. The output is a normal PEFT adapter
(same base, same LoRA shape) that every eval / packaging path already loads.

Locally (ROCm, 16 GB): 4-bit base, seq 512, batch 1 × grad-accum 8, gradient checkpointing.
On AWS (L40S 48 GB): ``--quant none --max-length 1024 --attn-impl sdpa`` (bf16, like the 14B SFT runs).

  python training/train_dpo.py --lang rust --pairs data/datasets/dpo/rust_pairs.jsonl \
      --sft-adapter data/models/lora_rust_7b/adapter --out data/models/exp/rust_dpo_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--sft-adapter", type=Path, required=True)
    ap.add_argument("--base-model", default=None, help="default: base_model_name_or_path of the adapter")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quant", choices=("4bit", "none"), default="4bit")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--loss", default="sigmoid", help="TRL loss_type, e.g. sigmoid | ipo | sigmoid,sft")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--attn-impl", default="eager", help="eager (ROCm-safe default) | sdpa (CUDA / L40S)")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    pairs = [json.loads(ln) for ln in args.pairs.read_text().splitlines() if ln.strip()]
    pairs = [{k: p[k] for k in ("prompt", "chosen", "rejected")} for p in pairs][: args.limit]
    if not pairs:
        raise SystemExit(f"no pairs in {args.pairs}")
    adapter_cfg = json.loads((args.sft_adapter / "adapter_config.json").read_text())
    base = args.base_model or adapter_cfg["base_model_name_or_path"]
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load: dict = {"dtype": torch.bfloat16, "attn_implementation": args.attn_impl}
    if args.quant == "4bit":
        from transformers import BitsAndBytesConfig

        load["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    if torch.cuda.is_available():  # CPU only for the tiny-model test
        load["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(base, **load)
    if args.quant == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(model, str(args.sft_adapter), is_trainable=True)  # adapter "default"

    cfg = DPOConfig(
        output_dir=str(args.out / "runs"),
        beta=args.beta,
        loss_type=args.loss.split(","),
        max_length=args.max_length,
        truncation_mode="keep_start",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        use_cpu=not torch.cuda.is_available(),
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        optim="adamw_torch",
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(pairs), processing_class=tok)
    t0 = time.time()
    out = trainer.train()
    secs = time.time() - t0

    adapter_dir = args.out / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir), selected_adapters=["default"])
    tok.save_pretrained(str(adapter_dir))
    # save_pretrained writes the non-default "ref" adapter under a subfolder if it ever slips in.
    log = [h for h in trainer.state.log_history if "loss" in h]
    meta = {
        "language": args.lang,
        "base_model": base,
        "adapter": _rel(adapter_dir),
        "kind": "dpo",
        "sft_adapter": _rel(args.sft_adapter),
        "pairs": _rel(args.pairs),
        "pair_rows": len(pairs),
        "hyperparams": {k: getattr(args, k) for k in ("beta", "loss", "lr", "epochs", "max_steps", "batch_size",
                                                      "grad_accum", "max_length", "quant", "attn_impl")},
        "global_steps": int(out.global_step),
        "train_loss": float(out.training_loss),
        "first_log": log[0] if log else None,
        "last_log": log[-1] if log else None,
        "train_seconds": round(secs, 1),
        "torch": torch.__version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (args.out / "rebirth_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (args.out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("global_steps", "train_loss", "first_log", "last_log", "train_seconds")},
                     indent=2))
    print(f"Saved DPO adapter → {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
