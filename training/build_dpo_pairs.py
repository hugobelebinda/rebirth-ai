#!/usr/bin/env python3
"""DPO preference pairs from an adapter's own failures (Phase 15 S4; local 7B smoke test first).

Two steps around a normal eval run (the eval scripts already store every prediction):

  1. ``sample``: N *training* rows (never held-out, never from an eval-set file) that fit
     ``--max-tokens`` → ``data/datasets/dpo/<lang>_prompts.jsonl``
  2. generate with the model being tuned — locally ``eval_behavioral.py --backend peft
     --eval-file <prompts>``, on AWS ``eval_vllm.py`` — the report keeps every prediction
  3. ``pairs``: report + prompts → ``data/datasets/dpo/<lang>_pairs.jsonl`` with
     ``{prompt, chosen, rejected, reason}``: chosen = gold, rejected = the model's own output when
     it is a **no-op** (returns the legacy code), **unparsable** (gold parses, output does not) or
     **low-overlap** (token overlap < ``--min-overlap``) or a **partial** edit (parses, overlaps, still
     not the gold). Soft-exact outputs are skipped; ``reason`` lets the trainer filter.
     These are the failure modes measured in Phases 11–12.

  python training/build_dpo_pairs.py sample --lang rust --n 120
  python training/build_dpo_pairs.py pairs --lang rust --report data/reports/exp/rust_dpo_gen.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rebirth_ai.lora import DEFAULT_EVAL_FRAC, DEFAULT_SEED, is_eval_row, load_rows, lora_target  # noqa: E402

DPO_DIR = ROOT / "data" / "datasets" / "dpo"
EVAL_DIR = ROOT / "data" / "datasets" / "evalsets"


def _norm(code: str) -> str:
    return " ".join((code or "").split())


def _corpus_for(lang: str) -> Path:
    for p in (ROOT / "data" / "datasets" / "p15" / f"{lang}_migration.jsonl",
              ROOT / "data" / "datasets" / "expanded" / f"{lang}_migration.jsonl"):
        if p.is_file():
            return p
    return lora_target(lang).data


def eval_guard(lang: str) -> tuple[set, set]:
    rows = [r for p in EVAL_DIR.glob(f"{lang}*.jsonl") for r in load_rows(p)]
    files = {(r["repo"], r["path"]) for r in rows if r.get("repo") and r.get("path")}
    texts = {_norm(r[k]) for r in rows for k in ("legacy", "modern")}
    return files, texts


def sample(args) -> int:
    from train_lora import format_prompt
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    corpus = args.corpus or _corpus_for(args.lang)
    files, texts = eval_guard(args.lang)
    rows = []
    for r in load_rows(corpus):
        if is_eval_row(r, eval_frac=DEFAULT_EVAL_FRAC, seed=DEFAULT_SEED):
            continue
        if (r.get("repo"), r.get("path")) in files or {_norm(r["legacy"]), _norm(r["modern"])} & texts:
            continue
        if args.tiers and (r.get("tier") or "expanded") not in args.tiers:
            continue
        rows.append(r)
    rng = random.Random(f"{args.seed}|{args.lang}|dpo")
    rng.shuffle(rows)
    picked = []
    for r in rows:
        if len(tok(format_prompt(r) + r["modern"])["input_ids"]) + 1 <= args.max_tokens:
            picked.append(r)
        if len(picked) >= args.n:
            break
    DPO_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or DPO_DIR / f"{args.lang}_prompts.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked), encoding="utf-8")
    print(f"{args.lang}: {len(picked)} prompts from {corpus.relative_to(ROOT)} → {out.relative_to(ROOT)}")
    return 0


def classify(row: dict, res: dict, min_overlap: float) -> str | None:
    """Why this prediction should be the *rejected* side, or None when it is good enough."""
    pred = res.get("pred") or ""
    ast = res.get("ast") or {}
    if ast.get("soft_exact"):
        return None
    if _norm(pred) == _norm(row["legacy"]):
        return "noop"
    if not pred.strip():
        return "empty"
    if ast.get("parses") is False:
        return "unparsable"
    if (ast.get("token_overlap") or 0.0) < min_overlap:
        return "low_overlap"
    if _norm(pred) != _norm(row["modern"]):
        return "partial"  # parses, close to gold, still wrong: the "partial edit" failure mode
    return None


def pairs(args) -> int:
    from train_lora import format_prompt

    prompts = load_rows(args.prompts or DPO_DIR / f"{args.lang}_prompts.jsonl")
    report = json.loads(args.report.read_text())["suites"]["synthetic"]
    results = report["results"]
    if len(results) != len(prompts):
        raise SystemExit(f"report has {len(results)} results for {len(prompts)} prompts — wrong pairing")
    out_rows, reasons = [], Counter()
    for row, res in zip(prompts, results, strict=True):
        if res.get("objective") and res["objective"] != row["objective"]:
            raise SystemExit(f"row {res.get('i')}: objective mismatch — report and prompts are out of order")
        why = classify(row, res, args.min_overlap)
        reasons[why or "good"] += 1
        if why:
            out_rows.append({"prompt": format_prompt(row), "chosen": row["modern"], "rejected": res.get("pred") or "",
                             "reason": why, "language": args.lang, "objective": row["objective"]})
    out = args.out or DPO_DIR / f"{args.lang}_pairs.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
    stats = {"prompts": len(prompts), "pairs": len(out_rows), "by_reason": dict(reasons), "model": report.get("model"),
             "min_overlap": args.min_overlap, "report": str(args.report)}
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--lang", required=True)
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--corpus", type=Path, default=None)
    s.add_argument("--tiers", nargs="*", default=None, help="e.g. expanded modern tool (default: all)")
    s.add_argument("--max-tokens", type=int, default=512)
    s.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    s.add_argument("--out", type=Path, default=None)
    p = sub.add_parser("pairs")
    p.add_argument("--lang", required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--prompts", type=Path, default=None)
    p.add_argument("--min-overlap", type=float, default=0.6)
    p.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    return sample(args) if args.cmd == "sample" else pairs(args)


if __name__ == "__main__":
    raise SystemExit(main())
