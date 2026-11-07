#!/usr/bin/env python3
"""Train a PEFT LoRA adapter on legacy→modern code pairs.

GPU path (ROCm/CUDA): bf16 LoRA + gradient checkpointing + grad accumulation.
CPU path: ``--cpu`` (safe on this desktop when ROCm hangs the display GPU).

Examples:
  # Per-language (Phase 11): resolves data/out, writes split index + rebirth_meta.json
  python training/train_lora.py --lang python --plan          # split only, no GPU
  python training/train_lora.py --lang python --base-model Qwen/Qwen2.5-Coder-7B-Instruct \\
    --batch-size 1 --grad-accum 8 --max-length 128 --gradient-checkpointing --attn-eager

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
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rebirth_ai.lora import (  # noqa: E402
    DEFAULT_BASE_7B,
    DEFAULT_EVAL_FRAC,
    DEFAULT_SEED,
    LORA_LANGS,
    file_sha256,
    load_rows,
    lora_target,
    rel_to_root,
    split_rows,
    write_split_index,
)

DEFAULT_DATA = ROOT / "data" / "datasets" / "js_ts_migration.jsonl"
DEFAULT_OUT = ROOT / "data" / "models" / "lora_js_ts"


class StepLog:
    """Append-only JSONL, flushed + fsync'd per line so it survives a GPU hang / hard reboot."""

    def __init__(self, path: Path | None) -> None:
        self._fh = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")

    def write(self, event: str, **fields) -> None:
        if self._fh is None:
            return
        import time

        self._fh.write(json.dumps({"event": event, "t": round(time.time(), 2), **fields}) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())


def latest_checkpoint(runs_dir: Path) -> Path | None:
    """Newest complete HF Trainer checkpoint (``checkpoint-<step>`` with trainer_state.json)."""
    if not runs_dir.is_dir():
        return None
    done = [
        d for d in runs_dir.glob("checkpoint-*")
        if d.name.split("-", 1)[1].isdigit() and (d / "trainer_state.json").is_file()
    ]
    return max(done, key=lambda d: int(d.name.split("-", 1)[1]), default=None)


LORA_TARGET_SETS: dict[str, tuple[str, ...]] = {
    "attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "all": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
}

def make_step_callback(step_log: StepLog, torch_mod):
    from transformers import TrainerCallback

    def _mem() -> dict:
        if not torch_mod.cuda.is_available():
            return {}
        mb = 1 << 20
        return {
            "alloc_mb": torch_mod.cuda.memory_allocated() // mb,
            "reserved_mb": torch_mod.cuda.memory_reserved() // mb,
            "peak_mb": torch_mod.cuda.max_memory_allocated() // mb,
        }

    class _StepCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            step_log.write("train_begin", max_steps=state.max_steps, **_mem())

        def on_step_begin(self, args, state, control, **kw):
            # Last line before a hang names the optimizer step that never finished.
            step_log.write("step_begin", step=state.global_step + 1)

        def on_step_end(self, args, state, control, **kw):
            step_log.write("step_end", step=state.global_step, **_mem())

        def on_log(self, args, state, control, logs=None, **kw):
            step_log.write("log", step=state.global_step, **(logs or {}))

        def on_save(self, args, state, control, **kw):
            step_log.write("checkpoint", step=state.global_step)

        def on_train_end(self, args, state, control, **kw):
            step_log.write("train_end", step=state.global_step, **_mem())

    return _StepCallback()


def format_prompt(row: dict) -> str:
    """SFT prompt; must match ``rebirth_ai.generate.prompts.build_sft_prompt``."""
    return (
        "### Objective\n"
        f"{row['objective']}\n\n"
        "### Legacy\n"
        f"{row['legacy']}\n"
        "### Modern\n"
    )


def format_example(row: dict) -> str:
    return format_prompt(row) + row["modern"]


def build_features(
    rows: list[dict],
    tokenizer,
    *,
    max_length: int,
    eos: bool = True,
    completion_only: bool = True,
    drop_overlong: bool = True,
) -> tuple[list[dict], dict]:
    """Tokenize SFT rows → ``input_ids`` / ``attention_mask`` / ``labels``.

    * ``eos``: append the tokenizer EOS (Qwen instruct: ``<|im_end|>``, distinct from
      pad ``<|endoftext|>``) so the model learns where a rewrite ends.
    * ``completion_only``: prompt tokens get label -100 → loss only on the modern code.
    * ``drop_overlong``: skip rows whose prompt + modern (+EOS) exceed ``max_length``
      instead of right-truncating the target away. Off → legacy truncation.
    """
    feats: list[dict] = []
    stats = {"rows": len(rows), "kept": 0, "dropped_overlong": 0, "truncated": 0, "label_tokens": 0}
    eos_id = tokenizer.eos_token_id
    for row in rows:
        p_ids = tokenizer(format_prompt(row), add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(row["modern"], add_special_tokens=False)["input_ids"]
        if eos and eos_id is not None:
            c_ids = c_ids + [eos_id]
        ids = p_ids + c_ids
        labels = ([-100] * len(p_ids) if completion_only else list(p_ids)) + list(c_ids)
        if len(ids) > max_length:
            if drop_overlong:
                stats["dropped_overlong"] += 1
                continue
            ids, labels = ids[:max_length], labels[:max_length]
            stats["truncated"] += 1
        n_lab = sum(1 for x in labels if x != -100)
        if n_lab == 0:  # truncation removed the whole target — nothing to learn
            stats["dropped_overlong"] += 1
            continue
        stats["label_tokens"] += n_lab
        stats["kept"] += 1
        feats.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    return feats, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--lang",
        choices=LORA_LANGS,
        default=None,
        help="Resolve --data/--out/split index from the per-language table (Phase 11)",
    )
    parser.add_argument("--data", type=Path, default=None, help=f"default: {DEFAULT_DATA}")
    parser.add_argument("--out", type=Path, default=None, help=f"default: {DEFAULT_OUT}")
    parser.add_argument(
        "--eval-frac",
        type=float,
        default=None,
        help=f"Held-out fraction (default {DEFAULT_EVAL_FRAC} with --lang, else 0 = no split)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Split seed")
    parser.add_argument(
        "--quant",
        choices=["none", "4bit", "8bit"],
        default="none",
        help="Load base quantized (bitsandbytes). 4bit = QLoRA NF4; fits seq 512 on 16 GB",
    )
    parser.add_argument("--eos", action=argparse.BooleanOptionalAction, default=True,
                        help="Append EOS to each target (default on)")
    parser.add_argument("--completion-only", action=argparse.BooleanOptionalAction, default=True,
                        help="Loss on modern tokens only (default on)")
    parser.add_argument("--drop-overlong", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip rows longer than --max-length instead of truncating (default on)")
    parser.add_argument(
        "--step-log",
        type=Path,
        default=None,
        help="JSONL step/memory log, fsync'd per line (see scripts/train_logged.sh)",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Resolve paths + write split index, print plan, exit (no torch import)",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="HF model id (default: 7B Instruct with --lang, else 0.5B Instruct)",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--lora-targets",
        choices=sorted(LORA_TARGET_SETS),
        default="attn",
        help="attn = q/k/v/o (recipe v2); all = attention + MLP (QLoRA-paper style)",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=-1, help="-1 = full epoch(s)")
    parser.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="Checkpoint every N optimizer steps (0 = only final adapter)",
    )
    parser.add_argument("--smoke", action="store_true", help="2-step tiny run")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the newest <out>/runs/checkpoint-* if one exists (else start fresh)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap dataset rows")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--attn-eager",
        action="store_true",
        help="Force eager attention (avoids experimental ROCm SDPA)",
    )
    parser.add_argument(
        "--attn-impl",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
        help="Attention implementation (CUDA cloud: sdpa; flash_attention_2 needs flash-attn). Overrides --attn-eager",
    )
    parser.add_argument(
        "--group-by-length",
        action="store_true",
        help="Batch rows of similar length (less padding when --batch-size > 1)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if ROCm/CUDA is visible",
    )
    parser.add_argument(
        "--recipe",
        default=None,
        help="YAML recipe name or path (configs/recipes/<name>.yaml); explicit flags override it",
    )
    pre, _ = parser.parse_known_args()
    recipe_meta: dict = {}
    if pre.recipe:
        from rebirth_ai.recipes import load_recipe

        try:
            recipe_defaults, recipe_meta = load_recipe(pre.recipe, parser)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        parser.set_defaults(**recipe_defaults)
        print(f"recipe {recipe_meta['recipe_name']} ← {recipe_meta['recipe_file']}")
    args = parser.parse_args()

    split_index: Path | None = None
    if args.lang:
        target = lora_target(args.lang)
        # Only the canonical corpus owns splits/<lang>.split.json; an alternate --data
        # (e.g. an expanded corpus) writes its split next to its adapter instead.
        if args.data is None or args.data.resolve() == target.data.resolve():
            split_index = target.split_index
        args.data = args.data or target.data
        args.out = args.out or target.out
    args.data = args.data or DEFAULT_DATA
    args.base_model = args.base_model or (
        DEFAULT_BASE_7B if args.lang else "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    )
    if args.lang and args.out == lora_target(args.lang).out:
        # Default out dir follows the base size: a 14B run must never land in (and later
        # sync over) the 7B champion's lora_<lang>_7b directory.
        from rebirth_ai.tracking import base_label

        size = base_label(args.base_model)
        if size not in ("7b", "unknown"):
            args.out = lora_target(args.lang).out.with_name(f"lora_{lora_target(args.lang).lang}_{size}")
    args.out = args.out or DEFAULT_OUT
    eval_frac = args.eval_frac if args.eval_frac is not None else (DEFAULT_EVAL_FRAC if args.lang else 0.0)
    if not 0.0 <= eval_frac < 1.0:
        parser.error("--eval-frac must be in [0, 1)")
    if not args.data.is_file():
        parser.error(f"dataset not found: {args.data}")

    all_rows = load_rows(args.data)
    corpus_sha = file_sha256(args.data)
    split = split_rows(all_rows, eval_frac=eval_frac, seed=args.seed)
    if eval_frac > 0:
        split_index = split_index or (args.out / "split.json")
        write_split_index(
            split_index,
            lang=args.lang,
            data=args.data,
            corpus_sha256=corpus_sha,
            split=split,
            eval_frac=eval_frac,
            seed=args.seed,
        )
    print(
        f"lang={args.lang or '-'} data={args.data} rows={len(all_rows)} "
        f"train={len(split.train)} eval={len(split.eval)} sha256={corpus_sha[:12]}"
    )
    if split_index is not None:
        print(f"split index → {split_index}")
    print(f"base → {args.base_model}\nout → {args.out}")
    print(
        f"recipe → quant={args.quant} seq={args.max_length} r={args.lora_r} alpha={args.lora_alpha} "
        f"targets={args.lora_targets} epochs={args.epochs} lr={args.lr} "
        f"batch={args.batch_size}x{args.grad_accum}"
    )
    if args.plan:
        return 0

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
            DataCollatorForSeq2Seq,
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
    train_rows = split.train[:limit] if limit is not None else split.train
    if not train_rows:
        raise SystemExit(f"No training rows in {args.data}")
    print(f"Loaded {len(train_rows)} train rows from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    feats, feat_stats = build_features(
        train_rows,
        tokenizer,
        max_length=args.max_length,
        eos=args.eos,
        completion_only=args.completion_only,
        drop_overlong=args.drop_overlong,
    )
    print(f"features: {feat_stats}")
    if not feats:
        raise SystemExit("No rows left after tokenization (all overlong?) — raise --max-length")

    dtype = torch.bfloat16 if use_gpu else torch.float32
    load_kwargs: dict = {
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if args.attn_impl:
        load_kwargs["attn_implementation"] = args.attn_impl
    elif args.attn_eager:
        load_kwargs["attn_implementation"] = "eager"
    if args.quant != "none":
        if not use_gpu:
            raise SystemExit("--quant needs a GPU")
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if args.quant == "4bit"
            else BitsAndBytesConfig(load_in_8bit=True)
        )
        load_kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    if args.quant != "none":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(args.gradient_checkpointing)
        )
    elif use_gpu:
        model.to("cuda")
    if args.gradient_checkpointing and use_gpu and args.quant == "none":
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
        target_modules=list(LORA_TARGET_SETS[args.lora_targets]),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = Dataset.from_list(feats)
    # Pads input_ids with pad and labels with -100 (keeps prompt masking + EOS labels intact)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100, padding=True)

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
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.quant != "none" else None,
        optim="adamw_torch",  # adamw_8bit needs working CUDA bitsandbytes
        # transformers 5: group_by_length moved to train_sampling_strategy
        train_sampling_strategy="group_by_length" if args.group_by_length else "random",
    )
    step_log = StepLog(args.step_log)
    step_log.write(
        "config",
        lang=args.lang,
        base_model=args.base_model,
        max_length=args.max_length,
        train_rows=len(feats),
        quant=args.quant,
        features=feat_stats,
        lora_r=args.lora_r,
        grad_accum=args.grad_accum,
        device=device_name,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
        callbacks=[make_step_callback(step_log, torch)],
    )
    try:
        resume = latest_checkpoint(args.out / "runs") if args.resume else None
        if resume is not None:
            print(f"Resuming from {resume}")
            step_log.write("resume", checkpoint=str(resume))
        train_out = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    except BaseException as exc:  # OOM / HIP error / Ctrl-C — record before dying
        step_log.write("error", kind=type(exc).__name__, message=str(exc)[:2000])
        raise
    adapter_dir = args.out / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    meta = {
        "base_model": args.base_model,
        "examples": len(feats),
        "smoke": args.smoke,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_targets": args.lora_targets,
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
    rebirth_meta = {
        "language": args.lang,
        "base_model": args.base_model,
        "adapter": rel_to_root(adapter_dir),
        "data": rel_to_root(args.data),
        "corpus_sha256": corpus_sha,
        "corpus_rows": len(all_rows),
        "train_rows": len(feats),
        "split_train_rows": len(train_rows),
        "recipe": {
            "quant": args.quant,
            "eos": args.eos,
            "completion_only": args.completion_only,
            "drop_overlong": args.drop_overlong,
            "features": feat_stats,
        },
        "eval_rows": len(split.eval),
        "eval_frac": eval_frac,
        "split_seed": args.seed,
        "split_index": rel_to_root(split_index) if split_index else None,
        "global_steps": int(train_out.global_step),
        "train_loss": float(train_out.training_loss),
        "hyperparams": {
            k: meta[k]
            for k in (
                "lora_r", "lora_alpha", "lora_targets", "epochs", "max_length",
                "batch_size", "grad_accum", "max_steps",
            )
        },
        "lr": args.lr,
        "recipe_name": recipe_meta.get("recipe_name"),
        "recipe_file": rel_to_root(Path(recipe_meta["recipe_file"])) if recipe_meta else None,
        "smoke": args.smoke,
        "device": device_name,
        "torch": torch.__version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (args.out / "rebirth_meta.json").write_text(json.dumps(rebirth_meta, indent=2), encoding="utf-8")
    print(f"Saved LoRA adapter → {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
