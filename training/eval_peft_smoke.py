#!/usr/bin/env python3
"""Small offline eval for the local PEFT LoRA adapter (CPU-safe).

Picks N synthetic-style / short pairs from the migration JSONL, generates with
PeftClient, and reports exact-match + token overlap vs gold ``modern``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _norm(code: str) -> str:
    text = code.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[^\s]", a))
    tb = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[^\s]", b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "datasets" / "js_ts_migration.jsonl",
    )
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--prefer-synthetic", action="store_true", default=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "models" / "lora_js_ts" / "eval_smoke.json",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="HF base id (default: PeftClient / REBIRTH_PEFT_BASE)",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="LoRA adapter dir (default: data/models/lora_js_ts/adapter)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (default for small 0.5B; omit for 7B GPU eval)",
    )
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HIP_VISIBLE_DEVICES"] = ""

    from rebirth_ai.generate.peft_client import PeftClient

    rows: list[dict] = []
    with args.data.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)

    # Prefer short synthetic pairs for a readable smoke; fall back to short mined
    synth = [r for r in rows if r.get("source", "synthetic") == "synthetic"]
    pool = synth if (args.prefer_synthetic and synth) else rows
    pool = sorted(pool, key=lambda r: len(r.get("legacy", "")) + len(r.get("modern", "")))
    sample = pool[: args.n]

    client = PeftClient(
        base_model=args.base_model,
        adapter_path=args.adapter,
        prompt_style="sft",
        max_new_tokens=512,
    )
    device = "cpu" if args.cpu else "auto"
    print(f"base={client.base_model} adapter={client.adapter_path} device={device}")
    results = []
    exact = 0
    overlaps: list[float] = []

    print(f"Evaluating {len(sample)} pairs with PEFT (CPU)…")
    for i, row in enumerate(sample, 1):
        objective = row["objective"]
        legacy = row["legacy"]
        gold = row["modern"]
        pred = client.migrate(
            objective=objective,
            legacy_code=legacy,
            file_path=row.get("path") or "snippet.js",
            language=row.get("language", "javascript"),
        )
        pred_code = pred.code
        is_exact = _norm(pred_code) == _norm(gold)
        ov = _overlap(_norm(pred_code), _norm(gold))
        if is_exact:
            exact += 1
        overlaps.append(ov)
        results.append(
            {
                "i": i,
                "objective": objective[:120],
                "exact": is_exact,
                "overlap": round(ov, 3),
                "legacy_chars": len(legacy),
                "pred_preview": pred_code[:240],
                "gold_preview": gold[:240],
            }
        )
        mark = "EXACT" if is_exact else f"overlap={ov:.2f}"
        print(f"[{i}/{len(sample)}] {mark} :: {objective[:70]}")

    summary = {
        "n": len(sample),
        "exact_match": exact,
        "exact_rate": exact / len(sample) if sample else 0.0,
        "mean_token_overlap": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "model": results[0]["objective"] and pred.model if results else None,
        "results": results,
    }
    # fix model field
    summary["model"] = pred.model if sample else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nexact={exact}/{len(sample)} ({summary['exact_rate']:.0%}) "
        f"mean_overlap={summary['mean_token_overlap']:.3f}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
