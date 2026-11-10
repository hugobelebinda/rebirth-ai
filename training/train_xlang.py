#!/usr/bin/env python3
"""Phase 17 stage 4: train the shared cross-language adapter with ``train_lora.py`` unchanged.

``train_lora.py`` reads same-language SFT rows (objective / legacy / modern) and builds the
``### Objective / ### Legacy / ### Modern`` prompt. Translation rows (``gen_xlang_pairs.py pairs``,
``build-mix`` below) carry their own ``prompt`` (``build_xlang_sft_prompt``) and ``completion``.
This wrapper lets both kinds through: a row with ``prompt`` is trained on that exact prompt, any
other row keeps the normal SFT prompt (the mix keeps some modernization rows so file-level code
is not forgotten). Everything else (recipe, masking, EOS, dropping over-long rows) is train_lora's.

  python training/train_xlang.py build-mix --pairs data/datasets/xlang/stage3/pairs.jsonl \\
      --out data/datasets/xlang/stage4_mix.jsonl --max-rows 12000 --same-lang 2000
  python training/train_xlang.py train --recipe cloud_14b --data data/datasets/xlang/stage4_mix.jsonl \\
      --out data/models/lora_xlang_14b --max-length 2048 --batch-size 2 --grad-accum 4 --epochs 2
  python training/train_xlang.py eval-pool --out data/datasets/xlang/stage3/eval_pool.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

LANGS = ["go", "python", "swift", "rust", "csharp", "cpp", "java", "js_ts"]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()]


def load_mixed_rows(path: Path) -> list[dict]:
    """Translation rows → SFT-shaped rows (legacy = prompt, modern = completion, so train_lora's
    content-hash split and stats still work); same-language rows pass through unchanged."""
    rows = []
    for r in _jsonl(path):
        if "prompt" in r and "completion" in r:
            rows.append({"objective": "xlang", "legacy": r["prompt"], "modern": r["completion"], "prompt": r["prompt"]})
        elif all(k in r for k in ("objective", "legacy", "modern")):
            rows.append(r)
    return rows


def train(argv: list[str]) -> int:
    import train_lora

    plain = train_lora.format_prompt
    train_lora.format_prompt = lambda row: row["prompt"] if "prompt" in row else plain(row)
    train_lora.load_rows = load_mixed_rows
    sys.argv = ["train_lora.py", *argv]
    return train_lora.main()


def build_mix(args) -> int:
    """Balanced across the 56 directions: each direction gets up to max_rows / directions rows,
    forward / reverse first (one side is the human-written reference), then cross pairs; plus
    ``--same-lang`` modernization rows spread over the 8 languages (p15 corpora)."""
    rng = random.Random(args.seed)
    rows = [r for r in _jsonl(Path(args.pairs)) if r.get("split") == "train"]
    by_dir: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_dir[f"{r['source_lang']}->{r['target_lang']}"].append(r)
    cap = max(1, args.max_rows // max(1, len(by_dir)))
    picked = []
    for _d, rs in sorted(by_dir.items()):
        rng.shuffle(rs)
        rs.sort(key=lambda r: r["kind"] == "cross")  # stable: forward / reverse first
        picked += rs[:cap]
    leftover = [r for _d, rs in sorted(by_dir.items()) for r in rs[cap:]]  # fill up to max_rows
    rng.shuffle(leftover)
    picked += leftover[: max(0, args.max_rows - len(picked))]
    same = []
    per_lang = args.same_lang // len(LANGS)
    for lang in LANGS:
        f = ROOT / "data" / "datasets" / "p15" / f"{lang}_migration.jsonl"
        if f.is_file() and per_lang:
            lr = [r for r in _jsonl(f) if all(k in r for k in ("objective", "legacy", "modern"))]
            rng.shuffle(lr)
            same += lr[:per_lang]
    mix = [{"prompt": r["prompt"], "completion": r["completion"]} for r in picked] + same
    rng.shuffle(mix)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mix), encoding="utf-8")
    kinds = defaultdict(int)
    for r in picked:
        kinds[r["kind"]] += 1
    stats = {"translation_rows": len(picked), "kinds": dict(kinds), "directions": len(by_dir), "cap_per_direction": cap,
             "same_language_rows": len(same), "total": len(mix),
             "min_per_direction": min((min(cap, len(rs)) for rs in by_dir.values()), default=0)}
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(stats))
    return 0


def eval_pool(args) -> int:
    """Program-suite eval rows in pool format: the same problems per direction as
    ``eval_translate.py --suite programs --limit N`` (one seeded order for every direction)."""
    problems = _jsonl(ROOT / "data" / "datasets" / "xlang" / "programs.jsonl")
    order = list(range(len(problems)))
    random.Random(args.seed).shuffle(order)
    rows = []
    for src in ("python", "cpp", "java"):
        for i in [i for i in order if src in problems[i]["sources"]][: args.limit]:
            p = problems[i]
            rows.append({"id": f"{p['id']}@{src}", "src_lang": src, "source": p["sources"][src], "tests": p["tests"]})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} eval pool rows ({len(rows) * 7} translations) → {out}")
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        return train(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("build-mix")
    m.add_argument("--pairs", default=str(ROOT / "data/datasets/xlang/stage3/pairs.jsonl"))
    m.add_argument("--out", default=str(ROOT / "data/datasets/xlang/stage4_mix.jsonl"))
    m.add_argument("--max-rows", type=int, default=12000)
    m.add_argument("--same-lang", type=int, default=2000)
    m.add_argument("--seed", type=int, default=1337)
    e = sub.add_parser("eval-pool")
    e.add_argument("--out", default=str(ROOT / "data/datasets/xlang/stage3/eval_pool.jsonl"))
    e.add_argument("--limit", type=int, default=20)
    e.add_argument("--seed", type=int, default=1337)
    sub.add_parser("train", help="train_lora.py arguments follow")
    args = ap.parse_args()
    return {"build-mix": build_mix, "eval-pool": eval_pool}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
