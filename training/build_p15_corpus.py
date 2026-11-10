#!/usr/bin/env python3
"""Build the Phase 15 training corpora (all 8 languages) and bigger (v2) frozen eval sets for Rust / C++.

Every language gets a leak-free copy of its expanded corpus: training rows from a file (repo + path)
that is in a frozen eval set, or with the same content, are dropped (found 2026-11-07: 2–23 of 100
eval rows per language had their file in training). Only Rust / C++ also get new data.

Inputs per language:
  data/datasets/expanded/<lang>_migration.jsonl   current corpus (CommitPackFT), always kept
  data/datasets/pool/<lang>_commitpack.jsonl      full-CommitPack rows, tier modern | generic
  data/datasets/pool/<lang>_toolpairs.jsonl       tool-generated pairs (cargo fix/clippy, clang-tidy)

Steps:
  1. dedupe (expanded wins), drop new rows over ``--max-tokens`` (prompt + gold + EOS, 1024 = the
     Phase 15 training length)
  2. eval set v2 (written once, then frozen): every row of the old frozen set + new held-out rows
     (same 5 % content-hash split, same 512-token fit rule as ``make_eval_sets.py``) up to ``--n``.
     New rows come from CommitPack only (no tool pairs: the set measures real commits) and never
     from a file (repo + path) that any adapter so far has trained on.
  3. leakage guard: drop every training row from a file (repo + path) that is in an eval set
  4. composition: expanded + all modern-tier rows + tool pairs (≤ ``--tool-share`` of the total,
     no single check > ``--check-share`` of the tool rows) + generic rows (seeded) up to ``--target``.
     Phase 12 showed more *generic* edits stop helping at 7B, so they only fill.

Held-out rows stay in the corpus (training splits them out by hash), so the corpus can be used
with ``train_lora.py --data`` as-is.

  python training/build_p15_corpus.py --langs rust cpp           # new data + eval v2 + guard
  python training/build_p15_corpus.py --langs go python java csharp swift js_ts   # guard only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rebirth_ai.lora import DEFAULT_EVAL_FRAC, DEFAULT_SEED, is_eval_row, load_rows  # noqa: E402

DATA = ROOT / "data" / "datasets"
OUT_DIR = DATA / "p15"
EVAL_DIR = DATA / "evalsets"


def _key(r: dict) -> tuple[str, str]:
    return (r.get("legacy") or "").strip(), (r.get("modern") or "").strip()


def _file_key(r: dict) -> tuple[str, str] | None:
    return (r["repo"], r["path"]) if r.get("repo") and r.get("path") else None


def _held(r: dict) -> bool:
    return is_eval_row(r, eval_frac=DEFAULT_EVAL_FRAC, seed=DEFAULT_SEED)


def _read(path: Path) -> list[dict]:
    return load_rows(path) if path.is_file() else []


def _cap_tool_rows(rows: list[dict], n: int, check_share: float, rng: random.Random) -> list[dict]:
    """Pick ≤ n tool rows, round-robin over the primary check so no single fix dominates."""
    by_check: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_check[(r.get("checks") or ["?"])[0]].append(r)
    for v in by_check.values():
        rng.shuffle(v)
    cap = max(1, int(n * check_share))
    out: list[dict] = []
    taken: Counter = Counter()
    while len(out) < n:
        progressed = False
        for check, v in sorted(by_check.items()):
            if v and taken[check] < cap and len(out) < n:
                out.append(v.pop())
                taken[check] += 1
                progressed = True
        if not progressed:
            break
    return out


def build(lang: str, args, tok) -> dict:
    from train_lora import format_prompt

    def n_tokens(r: dict) -> int:
        return len(tok(format_prompt(r) + r["modern"])["input_ids"]) + 1

    base = _read(DATA / "expanded" / f"{lang}_migration.jsonl")
    pool = _read(DATA / "pool" / f"{lang}_commitpack.jsonl")
    tools = _read(DATA / "pool" / f"{lang}_toolpairs.jsonl")
    stats: dict = {"expanded": len(base), "pool": len(pool), "toolpairs": len(tools)}

    seen = {_key(r) for r in base}
    new: list[dict] = []
    for r in pool + tools:
        k = _key(r)
        if k in seen:
            continue
        seen.add(k)
        new.append(r)
    stats["new_unique"] = len(new)
    for r in new:
        r["_tok"] = n_tokens(r)
    over = sum(r["_tok"] > args.max_tokens for r in new)
    new = [r for r in new if r["_tok"] <= args.max_tokens]
    stats["new_over_max_tokens"] = over

    # ---- eval set v2 (frozen once written) -------------------------------------------------
    old_eval = _read(EVAL_DIR / f"{lang}.jsonl")
    v2_path = EVAL_DIR / f"{lang}_v2.jsonl"
    if not new and not v2_path.is_file():  # no new data (6 of 8 languages): guard only, no v2 set
        eval_v2 = []
        stats["eval_v2"] = "none (no new rows)"
    elif v2_path.is_file():
        eval_v2 = _read(v2_path)
        stats["eval_v2"] = f"kept frozen ({len(eval_v2)} rows)"
    else:
        base_train = [r for r in base if not _held(r)]
        trained_files = {_file_key(r) for r in base_train} - {None}
        trained_texts = {t for r in base_train for t in _key(r)}
        old_keys = {_key(r) for r in old_eval}
        cands = [r for r in new
                 if r.get("tier") in ("modern", "generic") and _held(r) and r["_tok"] <= args.eval_max_tokens
                 and _key(r) not in old_keys and _file_key(r) not in trained_files
                 and not set(_key(r)) & trained_texts]
        rng = random.Random(f"{DEFAULT_SEED}|{lang}|v2")
        one_per_file: dict = {}
        for r in rng.sample(cands, len(cands)):
            one_per_file.setdefault(_file_key(r) or id(r), r)
        picked = list(one_per_file.values())[: max(0, args.n - len(old_eval))]
        eval_v2 = old_eval + [{k: v for k, v in r.items() if k != "_tok"} for r in picked]
        v2_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in eval_v2), encoding="utf-8")
        stats["eval_v2"] = {"old_rows": len(old_eval), "new_rows": len(picked), "candidates": len(cands),
                            "new_by_tier": dict(Counter(r.get("tier") for r in picked))}

    # ---- leakage guard ---------------------------------------------------------------------
    eval_files = {_file_key(r) for r in old_eval + eval_v2} - {None}
    eval_texts = {t for r in old_eval + eval_v2 for t in _key(r)}

    def leaks(r: dict) -> bool:  # same file (repo + path), or the same file content under a fork
        return not _held(r) and (_file_key(r) in eval_files or bool(set(_key(r)) & eval_texts))

    base_kept = [r for r in base if not leaks(r)]
    new = [r for r in new if not leaks(r)]
    stats["dropped_for_leakage"] = {"expanded": len(base) - len(base_kept),
                                    "eval_files_touched": len({_file_key(r) for r in base if leaks(r)} & eval_files)}

    # ---- composition -----------------------------------------------------------------------
    rng = random.Random(f"{DEFAULT_SEED}|{lang}|p15")
    modern = [r for r in new if r.get("tier") == "modern"]
    tool = [r for r in new if r.get("tier") == "tool"]
    generic = [r for r in new if r.get("tier") == "generic"]
    tool_n = min(len(tool), int(args.target * args.tool_share))
    tool_pick = _cap_tool_rows(tool, tool_n, args.check_share, rng)
    room = max(0, args.target - len(base_kept) - len(modern) - len(tool_pick)) if new else 0
    generic_pick = rng.sample(generic, min(room, len(generic)))
    rows = base_kept + modern + tool_pick + generic_pick
    rows = [{k: v for k, v in r.items() if k != "_tok"} for r in rows]
    rng.shuffle(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{lang}_migration.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    held = sum(_held(r) for r in rows)
    stats.update({
        "total": len(rows), "train_rows": len(rows) - held, "held_out_rows": held,
        "by_part": {"expanded": len(base_kept), "modern": len(modern), "tool": len(tool_pick),
                    "generic": len(generic_pick)},
        "available": {"modern": len(modern), "tool": len(tool), "generic": len(generic)},
        "tool_checks": dict(Counter((r.get("checks") or ["?"])[0] for r in tool_pick).most_common(12)),
        "params": {k: getattr(args, k) for k in ("target", "tool_share", "check_share", "max_tokens",
                                                  "eval_max_tokens", "n")},
    })
    (OUT_DIR / f"{lang}_migration.stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", nargs="+", default=["rust", "cpp"])
    ap.add_argument("--target", type=int, default=8000, help="total rows per language")
    ap.add_argument("--tool-share", type=float, default=0.25)
    ap.add_argument("--check-share", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--eval-max-tokens", type=int, default=512)
    ap.add_argument("--n", type=int, default=100, help="eval v2 rows")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    for lang in args.langs:
        print(f"== {lang}", flush=True)
        print(json.dumps(build(lang, args, tok), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
