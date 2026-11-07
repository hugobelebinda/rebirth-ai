#!/usr/bin/env python3
"""Gate a per-language LoRA against its own base model on the held-out split.

PASS iff adapter soft-exact >= base soft-exact AND adapter parses >= base parses - 1.
Exit 0 = pass, 1 = fail, 2 = missing/invalid report. Prints one JSON verdict line.

  python training/lora_gate.py --adapter data/reports/eval_heldout_go_peft7b_v2.json \\
      --base data/reports/eval_heldout_go_base7b_chat.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def suite(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["suites"]["synthetic"]


def verdict(adapter: dict, base: dict, *, parse_slack: int = 1) -> dict:
    ok_soft = adapter["soft_exact"] >= base["soft_exact"]
    ok_parse = adapter["parses"] >= base["parses"] - parse_slack
    return {
        "pass": bool(ok_soft and ok_parse),
        "adapter": {k: adapter.get(k) for k in ("n", "soft_exact", "parses", "mean_symbol_jaccard", "mean_token_overlap")},
        "base": {k: base.get(k) for k in ("n", "soft_exact", "parses", "mean_symbol_jaccard", "mean_token_overlap")},
        "checks": {"soft_exact>=base": ok_soft, f"parses>=base-{parse_slack}": ok_parse},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--parse-slack", type=int, default=1)
    args = ap.parse_args()
    try:
        v = verdict(suite(args.adapter), suite(args.base), parse_slack=args.parse_slack)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"pass": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 2
    print(json.dumps(v))
    return 0 if v["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
