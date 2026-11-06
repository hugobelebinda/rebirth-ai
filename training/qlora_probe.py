#!/usr/bin/env python3
"""Probe whether 4-bit (QLoRA) 7B training fits / is stable on this GPU.

Loads the base in NF4 (bitsandbytes), sanity-checks a greedy generation, then runs a
few real LoRA optimizer steps at each sequence length (forward + backward + AdamW),
logging peak VRAM and step time to a fsync'd JSONL. Stops at the first failure.

  python -u training/qlora_probe.py --lengths 256 512 --steps 3 --log logs/train/probe/probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from train_lora import StepLog  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-targets", choices=("attn", "all"), default="attn")
    ap.add_argument("--log", type=Path, required=True)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    log = StepLog(args.log)
    mb = 1 << 20

    def mem() -> dict:
        return {
            "alloc_mb": torch.cuda.memory_allocated() // mb,
            "peak_mb": torch.cuda.max_memory_allocated() // mb,
            "reserved_mb": torch.cuda.memory_reserved() // mb,
        }

    def emit(event: str, **kw) -> None:
        log.write(event, **kw)
        print(json.dumps({"event": event, **kw}), flush=True)

    emit("start", torch=torch.__version__, device=torch.cuda.get_device_name(0))
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
    )
    emit("loaded_4bit", secs=round(time.time() - t0, 1), **mem())

    # Sanity: quantized model still produces sensible code (catches broken kernels).
    msgs = [{"role": "user", "content": "Write a Python function add(a, b) that returns a + b. Code only."}]
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(0)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=40, do_sample=False)
    text = tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
    emit("generate_check", ok=("def add" in text and "return" in text), text=text[:200], **mem())

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            + (["gate_proj", "up_proj", "down_proj"] if args.lora_targets == "all" else []),
        ),
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    emit("peft_ready", **mem())

    for length in args.lengths:
        torch.cuda.reset_peak_memory_stats()
        # Random-but-valid tokens: worst case is a full-length sequence every step.
        x = torch.randint(100, 20000, (1, length), device=0)
        times = []
        for step in range(1, args.steps + 1):
            emit("step_begin", length=length, step=step)
            t = time.time()
            loss = model(input_ids=x, labels=x).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.time() - t
            times.append(dt)
            emit("step_end", length=length, step=step, secs=round(dt, 2),
                 loss=round(float(loss), 3), finite=bool(torch.isfinite(loss)), **mem())
        emit("length_ok", length=length, mean_step_secs=round(sum(times[1:] or times) / len(times[1:] or times), 2), **mem())

    emit("done", total_secs=round(time.time() - t0, 1))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    raise SystemExit(main())
