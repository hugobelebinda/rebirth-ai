#!/usr/bin/env python3
"""Program-level translation eval problems (Phase 17): whole programs judged by stdin → stdout tests.

HumanEval functions are short and leave little headroom (zero-shot 14B: 91 %). Here a "problem" is
a complete Codeforces program (imports, I/O, helpers, types; 20–150 lines) from DeepMind
CodeContests (CC BY 4.0) with reference solutions in Python 3 / C++ / Java and many tests. The
tests are input/output pairs, so one test set judges a translation into **any** of the 8 targets.

A problem is kept only if
- it is not interactive, reads stdin / writes stdout (no input files), and its outputs have no
  floating-point numbers (exact token comparison is then correct);
- one reference solution per language (Python 3, C++, Java; ≤ 3 candidates tried each) passes all
  its chosen tests in our sandboxes (token-wise, case-insensitive — like the Codeforces checkers);
- a known-incorrect solution from the dataset fails them (the tests discriminate).

  python training/build_xlang_programs.py --download     # CodeContests valid + test (~115 MB) → data/cache/
  python training/build_xlang_programs.py                # validate in the sandboxes, write the frozen set

Output: ``data/datasets/xlang/programs.jsonl`` (+ ``.stats.json``), frozen once written (``--force``).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rebirth_ai.xlang.languages import get_lang  # noqa: E402
from rebirth_ai.xlang.verify import run_io_programs  # noqa: E402

CACHE = ROOT / "data" / "cache" / "code_contests"
OUT = ROOT / "data" / "datasets" / "xlang" / "programs.jsonl"
HF = "https://huggingface.co/datasets/deepmind/code_contests/resolve/main/data/"
FILES = ["valid-00000-of-00001-5e672c5751f060d3.parquet", "test-00000-of-00001-9c49eeff30aacaa8.parquet"]
LANG_CODE = {"python": 3, "cpp": 2, "java": 4}  # CodeContests: 1 = Python 2, 2 = C++, 3 = Python 3, 4 = Java

MIN_LINES, MAX_LINES = 20, 150
MAX_IN, MAX_OUT = 6000, 3000  # bytes per test
N_TESTS, MIN_TESTS = 10, 6
CANDIDATES = 3
_FLOAT = re.compile(r"\d\.\d")
_BAD_SOURCE = re.compile(r"ONLINE_JUDGE|freopen|input\.txt|output\.txt|#\s*ifdef\s+LOCAL|#\s*ifndef\s+LOCAL|\bfileinput\b")


def download() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in FILES + ["../README.md"]:
        dst = CACHE / Path(name).name
        if not dst.is_file():
            print(f"downloading {name} …", flush=True)
            urllib.request.urlretrieve(HF + name, dst)


def nonblank_lines(code: str) -> int:
    return sum(1 for ln in code.splitlines() if ln.strip())


def usable_source(code: str) -> bool:
    lines = code.splitlines()
    return (
        MIN_LINES <= nonblank_lines(code) <= MAX_LINES
        and not _BAD_SOURCE.search(code)
        and max((len(ln) for ln in lines), default=0) <= 200
    )


def pick_tests(row, rng: random.Random) -> list[tuple[str, str]] | None:
    fixed, generated = [], []
    for key, bucket in (("public_tests", fixed), ("private_tests", fixed), ("generated_tests", generated)):
        t = row[key]
        bucket += list(zip(t["input"], t["output"], strict=True))
    if any(_FLOAT.search(out) for _, out in fixed + generated):
        return None  # needs a float tolerance
    ok = [(i, o) for i, o in fixed if len(i) <= MAX_IN and len(o) <= MAX_OUT and o.strip()]
    gen = [(i, o) for i, o in generated if len(i) <= MAX_IN and len(o) <= MAX_OUT and o.strip()]
    rng.shuffle(gen)
    tests = (ok + gen)[:N_TESTS]
    return tests if len(tests) >= MIN_TESTS else None


def candidates(row, lang: str, rng: random.Random, field: str = "solutions") -> list[str]:
    sols = row[field]
    pool = [s for c, s in zip(sols["language"], sols["solution"], strict=True) if c == LANG_CODE[lang] and usable_source(s)]
    rng.shuffle(pool)
    return pool[:CANDIDATES]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="fetch the CodeContests valid + test splits first")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=10, help="seconds per test run")
    ap.add_argument("--force", action="store_true", help="overwrite the frozen output")
    args = ap.parse_args()
    if args.download:
        download()
    if args.out.is_file() and not args.force:
        print(f"{args.out} exists (frozen); use --force to rebuild")
        return 0
    import pandas as pd

    files = [CACHE / f for f in FILES]
    if not all(f.is_file() for f in files):
        print("CodeContests files missing — run with --download")
        return 1
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    drop: Counter = Counter()
    pending: dict[str, dict] = {}
    for _, row in df.iterrows():
        rng = random.Random(f"{args.seed}:{row['name']}")
        desc = str(row["description"])
        if str(row["input_file"]).strip() or str(row["output_file"]).strip():
            drop["file_io"] += 1
            continue
        if re.search(r"\binteract(ive|ion)\b", desc, re.IGNORECASE):
            drop["interactive"] += 1
            continue
        tests = pick_tests(row, rng)
        if tests is None:
            drop["float_or_few_tests"] += 1
            continue
        cands = {lang: candidates(row, lang, rng) for lang in LANG_CODE}
        if not all(cands.values()):
            drop["no_short_solution_in_each_language"] += 1
            continue
        wrong = [(lang, s) for lang in ("cpp", "java", "python") for s in candidates(row, lang, rng, "incorrect_solutions")][:3]
        pid = f"CC_{row['cf_contest_id']}{row['cf_index']}"
        pending[pid] = {"id": pid, "name": str(row["name"]), "rating": int(row["cf_rating"]),
                        "tags": list(row["cf_tags"]), "tests": tests, "cands": cands, "wrong": wrong}
    print(f"{len(df)} problems → {len(pending)} candidates; dropped {dict(drop)}", flush=True)

    # validate the references: candidate k for every (problem, language) still without a passing one
    chosen: dict[tuple[str, str], str] = {}
    t0 = time.time()
    for k in range(CANDIDATES):
        for lang in LANG_CODE:
            progs = {f"{pid}__{lang}__{k}": (p["cands"][lang][k], p["tests"]) for pid, p in pending.items()
                     if (pid, lang) not in chosen and len(p["cands"][lang]) > k}
            if not progs:
                continue
            res = run_io_programs(get_lang(lang), progs, timeout_sec=args.timeout, workers=args.workers)
            for rid, r in res.items():
                if r.status == "pass":
                    pid = rid.split("__")[0]
                    chosen[(pid, lang)] = pending[pid]["cands"][lang][k]
            print(f"  round {k} {lang}: {sum(r.status == 'pass' for r in res.values())}/{len(res)} pass "
                  f"({Counter(r.status for r in res.values())}) {time.time() - t0:.0f}s", flush=True)
    full = [pid for pid in pending if all((pid, lang) in chosen for lang in LANG_CODE)]
    drop["reference_failed"] = len(pending) - len(full)

    # the tests must reject at least one known-incorrect solution (up to 3 tried; some are only too
    # slow on big inputs, which these small tests do not have)
    rejected: dict[str, bool] = {}
    for lang in LANG_CODE:
        progs = {f"{pid}__wrong{j}": (code, pending[pid]["tests"]) for pid in full
                 for j, (wl, code) in enumerate(pending[pid]["wrong"]) if wl == lang}
        if progs:
            for rid, r in run_io_programs(get_lang(lang), progs, timeout_sec=args.timeout, workers=args.workers).items():
                pid = rid.split("__")[0]
                rejected[pid] = rejected.get(pid, False) or r.status != "pass"
    kept = [pid for pid in full if rejected.get(pid, True)]
    drop["no_incorrect_solution_rejected"] = len(full) - len(kept)
    no_wrong = sum(1 for pid in kept if pid not in rejected)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid in sorted(kept):
        p = pending[pid]
        srcs = {lang: chosen[(pid, lang)] for lang in LANG_CODE}
        rows.append({"id": pid, "name": p["name"], "rating": p["rating"], "tags": p["tags"],
                     "sources": srcs, "lines": {lang: nonblank_lines(s) for lang, s in srcs.items()},
                     "tests": [list(t) for t in p["tests"]]})
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    ratings = sorted(r["rating"] for r in rows)
    stats = {
        "source": "DeepMind CodeContests valid + test splits (CC BY 4.0), Codeforces problems",
        "problems": len(rows), "dropped": dict(drop), "without_incorrect_solution_check": no_wrong,
        "tests_per_problem": Counter(len(r["tests"]) for r in rows).most_common(),
        "rating": {"min": ratings[0], "median": ratings[len(ratings) // 2], "max": ratings[-1]} if ratings else {},
        "lines": {lang: sorted(r["lines"][lang] for r in rows)[len(rows) // 2] if rows else 0 for lang in LANG_CODE},
        "seed": args.seed,
    }
    args.out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=1))
    print(f"wrote {args.out.relative_to(ROOT)} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
