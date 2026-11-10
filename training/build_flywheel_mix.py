#!/usr/bin/env python3
"""Phase 18 flywheel → training data: a base corpus + the reviewed ``train``-split flywheel rows.

  # same-language: corpus format (objective / language / legacy / modern)
  python training/build_flywheel_mix.py --lang go --base data/datasets/p15/go_migration.jsonl \\
      --out data/datasets/flywheel_mix/go_migration.jsonl
  # cross-language: the shared adapter's prompt / completion rows
  python training/build_flywheel_mix.py --lang xlang --base data/datasets/xlang/stage4_mix.jsonl \\
      --out data/datasets/flywheel_mix/xlang_mix.jsonl

Flywheel rows come from ``rebirth review accept`` (``data/datasets/flywheel/<lang>.jsonl``, own code
only, git-ignored). Kept out: ``eval``-split rows (they are the rolling eval set), rows whose input
appears in a frozen eval set, and duplicates of base rows. ``--repeat`` up-weights the few
real-world rows (default 2). DVC stage ``flywheel_mix`` runs this for every language.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _norm(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def frozen_inputs(evalsets: Path) -> set[str]:
    return {_norm(r["legacy"]) for p in sorted(evalsets.glob("*.jsonl")) for r in _jsonl(p) if r.get("legacy")}


def xlang_row(r: dict) -> dict:
    from rebirth_ai.xlang.languages import get_lang
    from rebirth_ai.xlang.prompts import build_xlang_sft_prompt

    prompt = build_xlang_sft_prompt(source=get_lang(r["source_lang"]), target=get_lang(r["target_lang"]),
                                    source_code=r["source_code"])
    return {"prompt": prompt, "completion": r["target_code"].rstrip() + "\n", "source": "flywheel", "run_id": r.get("run_id")}


def build(lang: str, base: Path, out: Path, flywheel: Path, evalsets: Path, repeat: int) -> dict:
    base_rows = _jsonl(base)
    fly = [r for r in _jsonl(flywheel / f"{lang}.jsonl") if r.get("split") == "train" and r.get("license") == "own"]
    frozen = frozen_inputs(evalsets)
    if lang == "xlang":
        seen = {_norm(r.get("prompt", "")) for r in base_rows}
        add = [xlang_row(r) for r in fly]
        add = [r for r in add if _norm(r["prompt"]) not in seen]
        dropped_eval = 0
    else:
        seen = {_norm(r.get("legacy", "")) for r in base_rows}
        keep = [r for r in fly if _norm(r["legacy"]) not in frozen]
        dropped_eval = len(fly) - len(keep)
        add = [{k: r[k] for k in ("objective", "language", "legacy", "modern", "source", "run_id", "path") if k in r}
               for r in keep if _norm(r["legacy"]) not in seen]
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = base_rows + add * repeat
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    stats = {"lang": lang, "base": str(base), "base_rows": len(base_rows), "flywheel_train_rows": len(fly),
             "dropped_frozen_eval": dropped_eval, "added": len(add), "repeat": repeat, "total": len(rows)}
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, help="corpus language id (go, js_ts, …) or xlang")
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--flywheel", type=Path, default=ROOT / "data/datasets/flywheel")
    ap.add_argument("--evalsets", type=Path, default=ROOT / "data/datasets/evalsets")
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()
    print(json.dumps(build(a.lang, a.base, a.out, a.flywheel, a.evalsets, a.repeat)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
