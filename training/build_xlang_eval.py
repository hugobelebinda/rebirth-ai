#!/usr/bin/env python3
"""Build the cross-language eval problems (Phase 17): HumanEval in 9 languages, tests + sources.

  data/cache/multipl_e/  (downloaded once, see --download)
    humaneval-<cfg>.parquet   MultiPL-E (MIT): per-language prompt (signature) + unit tests
    humaneval-py.parquet      openai/openai_humaneval (MIT): Python prompt, canonical solution, tests
    humanevalx-<l>.jsonl      THUDM/humaneval-x (Apache-2.0): C++ / Go / Java / JS solutions
  → data/datasets/xlang/humaneval.jsonl   one row per problem:
      {id, entry, sources: {lang: code}, targets: {lang: {prompt, tests} | python: {prompt, test, entry_point}}}

Sources are comment / docstring free where the original had the problem statement in them, so a
model has to translate, not re-solve from the description. Frozen once written (like the eval sets).

  python training/build_xlang_eval.py --download
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rebirth_ai.xlang.multipl_e import (  # noqa: E402
    HUMANEVAL_X,
    MULTIPLE_CONFIG,
    problem_key,
    strip_c_comments,
    strip_python_docstrings,
)

CACHE = ROOT / "data" / "cache" / "multipl_e"
OUT = ROOT / "data" / "datasets" / "xlang" / "humaneval.jsonl"


def download() -> None:
    from huggingface_hub import hf_hub_download

    CACHE.mkdir(parents=True, exist_ok=True)
    for cfg in MULTIPLE_CONFIG.values():
        p = hf_hub_download("nuprl/MultiPL-E", f"humaneval-{cfg}/test-00000-of-00001.parquet", repo_type="dataset")
        shutil.copy(p, CACHE / f"humaneval-{cfg}.parquet")
    p = hf_hub_download("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet", repo_type="dataset")
    shutil.copy(p, CACHE / "humaneval-py.parquet")
    for cfg in HUMANEVAL_X.values():
        p = hf_hub_download("THUDM/humaneval-x", f"data/{cfg}/data/humaneval.jsonl", repo_type="dataset")
        shutil.copy(p, CACHE / f"humanevalx-{cfg}.jsonl")


def humaneval_x_source(lang: str, row: dict) -> str:
    # prompt = imports + helper functions + the signature, with the problem statement in comments
    # (``declaration`` is inconsistent: it drops helpers for Go). Comments are removed.
    code = strip_c_comments(row["prompt"] + row["canonical_solution"])
    return code.strip("\n") + "\n"


def build() -> dict:
    import pandas as pd

    problems: dict[str, dict] = {}

    def entry(key: str) -> dict:
        return problems.setdefault(key, {"id": key, "sources": {}, "targets": {}})

    py = pd.read_parquet(CACHE / "humaneval-py.parquet")
    for r in py.itertuples():
        e = entry(problem_key(r.task_id))
        e["entry"] = r.entry_point
        e["sources"]["python"] = strip_python_docstrings(r.prompt + r.canonical_solution)
        e["targets"]["python"] = {"prompt": r.prompt, "test": r.test, "entry_point": r.entry_point}
    for lang, cfg in MULTIPLE_CONFIG.items():
        df = pd.read_parquet(CACHE / f"humaneval-{cfg}.parquet")
        for r in df.itertuples():
            entry(problem_key(r.name))["targets"][lang] = {"prompt": r.prompt, "tests": r.tests, "name": r.name}
    for lang, cfg in HUMANEVAL_X.items():
        for ln in (CACHE / f"humanevalx-{cfg}.jsonl").read_text().splitlines():
            r = json.loads(ln)
            entry(problem_key(r["task_id"]))["sources"][lang] = humaneval_x_source(lang, r)

    rows = sorted(problems.values(), key=lambda e: int(e["id"].split("_")[1]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    langs = sorted({lang for r in rows for lang in r["targets"]})
    stats = {
        "problems": len(rows),
        "targets": {lang: sum(lang in r["targets"] for r in rows) for lang in langs},
        "sources": {lang: sum(lang in r["sources"] for r in rows) for lang in sorted({s for r in rows for s in r["sources"]})},
        "licenses": {"MultiPL-E": "MIT", "openai_humaneval": "MIT", "humaneval-x": "Apache-2.0"},
    }
    OUT.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="fetch the source files into data/cache/multipl_e first")
    ap.add_argument("--force", action="store_true", help="rebuild even if the frozen file exists")
    args = ap.parse_args()
    if OUT.is_file() and not args.force:
        print(f"{OUT.relative_to(ROOT)} exists (frozen); --force to rebuild")
        return 0
    if args.download:
        download()
    print(json.dumps(build(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
