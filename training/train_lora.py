#!/usr/bin/env python3
"""Train a PEFT LoRA adapter on legacy→modern code pairs.

GPU path (ROCm/CUDA): bf16 LoRA + gradient checkpointing + grad accumulation.
CPU path: ``--cpu`` (safe on this desktop when ROCm hangs the display GPU).

Examples:
  # Safe local
  python training/train_lora.py --cpu --epochs 1 --max-length 512

  # GPU smoke (may hang display GPU on AMD — use wrapper script)
  python training/train_lora.py --base-model Qwen/Qwen2.5-Coder-7B-Instruct \\
    --max-steps 5 --max-length 512 --grad-accum 8 --gradient-checkpointing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "datasets" / "js_ts_migration.jsonl"
DEFAULT_OUT = ROOT / "data" / "models" / "lora_js_ts"


def format_example(row: dict) -> str:
    return (
        "### Objective\n"
        f"{row['objective']}\n\n"
        "### Legacy\n"
        f"{row['legacy']}\n"
        "### Modern\n"
        f"{row['modern']}"
    )


def load_texts(path: Path, limit: int | None = None) -> list[str]:
    rows: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(format_example(json.loads(line)))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"No training rows in {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        help="HF model id",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=-1, help="-1 = full epoch(s)")
    parser.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="Checkpoint every N optimizer steps (0 = only final adapter)",
    )
    parser.add_argument("--smoke", action="store_true", help="2-step tiny run")
    parser.add_argument("--limit", type=int, default=None, help="Cap dataset rows")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--attn-eager",
        action="store_true",
        help="Force eager attention (avoids experimental ROCm SDPA)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if ROCm/CUDA is visible",
    )
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HIP_VISIBLE_DEVICES"] = ""

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError:
        print(
            'Missing ML extras. Install with:\n  pip install -e ".[ml]"',
            file=sys.stderr,
        )
        return 2

    use_gpu = (not args.cpu) and torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if use_gpu else "cpu"
    print(f"torch={torch.__version__} device={device_name} cuda={use_gpu}")
    if args.cpu:
        print("Forced CPU training (--cpu)")
    if use_gpu:
        print(
            f"HIP_ALLOC={os.environ.get('PYTORCH_HIP_ALLOC_CONF')} "
            f"attn_eager={args.attn_eager}"
        )

    limit = 4 if args.smoke else args.limit
    texts = load_texts(args.data, limit=limit)
    print(f"Loaded {len(texts)} examples from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if use_gpu else torch.float32
    load_kwargs: dict = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if args.attn_eager:
        load_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    if use_gpu:
        model.to("cuda")
    if args.gradient_checkpointing and use_gpu:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        # Required for PEFT + gradient checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def tokenize(batch: dict) -> dict:
        # Dynamic pad via collator — avoids always materializing max_length tensors
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    ds = Dataset.from_dict({"text": texts}).map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args.out.mkdir(parents=True, exist_ok=True)
    max_steps = 2 if args.smoke else args.max_steps
    save_steps = max(0, args.save_steps)
    training_args = TrainingArguments(
        output_dir=str(args.out / "runs"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, args.grad_accum),
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        learning_rate=args.lr,
        logging_steps=1 if (args.smoke or max_steps > 0 and max_steps <= 20) else 10,
        save_strategy="steps" if save_steps > 0 else "no",
        save_steps=save_steps if save_steps > 0 else 500,
        save_total_limit=3 if save_steps > 0 else None,
        report_to=[],
        remove_unused_columns=False,
        bf16=use_gpu,
        fp16=False,
        dataloader_pin_memory=use_gpu,
        gradient_checkpointing=bool(args.gradient_checkpointing and use_gpu),
        optim="adamw_torch",  # adamw_8bit needs working CUDA bitsandbytes
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()
    adapter_dir = args.out / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    meta = {
        "base_model": args.base_model,
        "examples": len(texts),
        "smoke": args.smoke,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_steps": max_steps,
        "gradient_checkpointing": bool(args.gradient_checkpointing and use_gpu),
        "attn_eager": args.attn_eager,
        "data": str(args.data),
        "device": device_name,
        "torch": torch.__version__,
        "dtype": str(dtype),
    }
    (args.out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved LoRA adapter → {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
