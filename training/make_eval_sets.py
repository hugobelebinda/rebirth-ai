#!/usr/bin/env python3
"""Freeze per-language eval sets: seeded random rows from a corpus's held-out split.

The 24-row "shortest held-out rows" suite is noisy (±2 rows) and biased toward trivial
edits. This samples ``--n`` rows at random (seed 1337) from the 5% hash split of the given
corpus, keeping only rows whose SFT prompt + gold + EOS fit ``--max-length`` tokens (the
training distribution). Because the split is a per-row content hash, a row that is held
out in the expanded corpus is held out in every smaller corpus too — the same frozen set
is valid for v2 adapters (trained on the original corpus) and adapters trained on
``data/datasets/expanded/``.

  python training/make_eval_sets.py --langs go rust --corpus-dir data/datasets/expanded
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rebirth_ai.lora import DEFAULT_EVAL_FRAC, DEFAULT_SEED, LORA_LANGS, load_rows, split_rows  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--langs", nargs="+", default=list(LORA_LANGS))
    ap.add_argument("--corpus-dir", type=Path, default=ROOT / "data" / "datasets" / "expanded")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "datasets" / "evalsets")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    args = ap.parse_args()

    from train_lora import format_prompt
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for lang in args.langs:
        corpus = args.corpus_dir / f"{lang}_migration.jsonl"
        held = split_rows(load_rows(corpus), eval_frac=DEFAULT_EVAL_FRAC, seed=args.seed).eval
        fit = [
            r for r in held
            if len(tok(format_prompt(r) + r["modern"])["input_ids"]) + 1 <= args.max_length
        ]
        rng = random.Random(f"{args.seed}|{lang}")
        sample = rng.sample(fit, min(args.n, len(fit)))
        out = args.out_dir / f"{lang}.jsonl"
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sample), encoding="utf-8")
        print(f"{lang}: held_out={len(held)} fit={len(fit)} sampled={len(sample)} → {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
