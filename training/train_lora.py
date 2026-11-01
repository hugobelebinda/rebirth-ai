#!/usr/bin/env python3
"""Train a PEFT LoRA adapter on legacy→modern code pairs.

Default base model is a small Qwen2.5-Coder for local smoke tests on this machine
(AMD GPU / CPU). For production-quality adapters, point --base-model at a larger
coder checkpoint and train on a CUDA box or Hugging Face Jobs.

Examples:
  python training/train_lora.py --smoke
  python training/train_lora.py --base-model Qwen/Qwen2.5-Coder-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
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
        help="HF model id (use a small coder for local smoke)",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--smoke", action="store_true", help="2-step tiny run")
    parser.add_argument("--limit", type=int, default=None, help="Cap dataset rows")
    args = parser.parse_args()

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
            "Missing ML extras. Install with:\n  pip install -e \".[ml]\"",
            file=sys.stderr,
        )
        return 2

    limit = 4 if args.smoke else args.limit
    texts = load_texts(args.data, limit=limit)
    print(f"Loaded {len(texts)} examples from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
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
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    ds = Dataset.from_dict({"text": texts}).map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args.out.mkdir(parents=True, exist_ok=True)
    max_steps = 2 if args.smoke else -1
    training_args = TrainingArguments(
        output_dir=str(args.out / "runs"),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
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
    }
    (args.out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved LoRA adapter → {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
