#!/usr/bin/env python3
"""Mine the full ``bigcode/commitpack`` (not the FT subset) shard by shard — Phase 15 data.

CommitPackFT is exhausted for Rust (2,996 commits) and C++ (4,992). The full CommitPack has
~296K Rust and ~2.4M C++ commits in ~0.5 GB JSONL shards (30 Rust, 365 C++). Each shard is
downloaded, filtered and (by default) deleted, so C++ (~191 GB) never has to fit on disk.

The full set is NOT instruction-filtered, so on top of the usual license / size / parse filters
every row must pass a commit-quality filter and is put in one of two tiers:

  modern   commit message looks like a modernization / refactor (language-specific keywords:
           ``try!``→``?``, ``dyn``, editions, clippy …; C++11+ features, smart pointers, nullptr …)
  generic  a clean, focused edit: imperative subject (≥3 words), one file, 1–60 changed lines,
           not whitespace- or comment-only, not a merge / revert / version bump / typo fix

Output (append-only, resumable): ``data/datasets/pool/<lang>_commitpack.jsonl`` with the usual
SFT schema plus ``tier``; per-shard stats in ``<lang>_commitpack.state.json``.

  python training/mine_commitpack_full.py --lang rust                 # all 30 shards
  python training/mine_commitpack_full.py --lang cpp --sample 40      # 40 shards spread over 365
  python training/mine_commitpack_full.py --lang rust --shards 1 --dry-run
  python training/mine_commitpack_full.py --lang cpp --sample 60 --seed 7 --standalone   # tool-pair inputs
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from mine_migration_pairs import PERMISSIVE, _parseable  # noqa: E402

REPO = "bigcode/commitpack"
CACHE = ROOT / "data" / "cache" / "commitpack"
POOL = ROOT / "data" / "datasets" / "pool"

FOLDER = {"rust": "rust", "cpp": "c++"}
N_SHARDS = {"rust": 30, "cpp": 365}
SUFFIXES = {
    "rust": {".rs"},
    # .c is C, not C++ — left out on purpose (the FT miner kept it).
    "cpp": {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h", ".hxx", ".ipp", ".inl"},
}
MIN_CHARS, MAX_CHARS = 40, 6000

_COMMON_MODERN = (
    r"\bmigrat\w*|\bupgrad\w*|\bmoderni[sz]\w*|\bidiomatic\w*|\brefactor\w*|\bdeprecat\w*|"
    r"\blegacy\b|\bold[- ]style\b|\bnew[- ]style\b|\buse \S+(?: \S+)? instead of\b|"
    r"\breplace \S+(?: \S+)? (?:with|by)\b|\bremove (?:unnecessary|unneeded|redundant|needless|useless)\b|"
    r"\bsimplif\w*|\bcleaner\b|\bmore concise\b"
)
MODERN_RE = {
    "rust": re.compile(
        _COMMON_MODERN + r"|\bclippy\b|\bedition\b|\b(?:2015|2018|2021|2024) edition\b|\btry!|\btry macro\b|"
        r"\?[- ]?operator\b|\bquestion mark\b|\bdyn\b|\bimpl trait\b|\basync[/ ]await\b|\bto async\b|"
        r"\bconst fn\b|\blet[- ]else\b|\bnll\b|\bnon[- ]lexical\b|\bstd::future\b|\bfutures? 0\.[13]\b|"
        r"\btokio 0\.[123]\b|\bstabili[sz]\w*|\bfeature gate\b|\bremove (?:the )?feature\b|"
        r"\bextern crate\b|\bmacro_use\b|\bif let\b|\bmatches!|\brustfix\b|\bnew api\b|\bnewer\b",
        re.IGNORECASE,
    ),
    "cpp": re.compile(
        _COMMON_MODERN + r"|\bc\+\+ ?(?:0x|11|14|17|1z|20|2a|23)\b|\bcxx ?(?:11|14|17|20)\b|\bnullptr\b|"
        r"\bauto\b|\bunique_ptr\b|\bshared_ptr\b|\bweak_ptr\b|\bmake_(?:unique|shared)\b|\bsmart pointers?\b|"
        r"\brange[- ]based\b|\brange[- ]for\b|\bfor[- ]each\b|\bconstexpr\b|\boverride\b|"
        r"=\s?default\b|=\s?delete\b|\busing (?:alias|declaration)\b|\btypedefs?\b|\bemplace\w*|"
        r"\bstd::move\b|\bmove semantics\b|\brvalue\b|\blambdas?\b|\bstd::(?:array|optional|variant|"
        r"string_view|filesystem|thread|mutex|atomic|chrono|function|span|format)\b|\bboost\b|"
        r"\bclang[- ]tidy\b|\bcppcoreguidelines\b|\braw pointers?\b|\bNULL\b|\binitializer list\b|"
        r"\bstatic_assert\b|\benum class\b|\bnoexcept\b",
        re.IGNORECASE,
    ),
}

DENY_RE = re.compile(
    r"^(?:merge|rollup|revert|bump|release|version|wip|initial commit|auto|automatic|update readme)\b|"
    r"\bmerge (?:branch|pull request|remote)\b|\brollup merge\b|\btypos?\b|"
    r"\bspelling\b|\bwhitespace\b|\bformat(?:ting)?\b|\brustfmt\b|\bclang-format\b|\bindent\w*|"
    r"\bcopyright\b|\blicen[cs]e header\b|\bchangelog\b|\breadme\b|\bdocs?\b|\bcomments?\b|"
    r"\bsigned-off-by\b|\bgit-svn-id\b|^update \S+$|^(?:fix|fixes|fixed|cleanup|clean up|wip|"
    r"update|updates|updated|changes|minor|tweak|tweaks|stuff|misc|test|tests)\.?$",
    re.IGNORECASE,
)
SUBJECT_START_RE = re.compile(r"^[A-Za-z][A-Za-z'-]*\b")
COMMENT_LINE_RE = re.compile(r"^\s*(?://|/\*|\*|\*/|#\[doc|$)")


def _changed_lines(a: str, b: str) -> list[str]:
    out = []
    for line in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0):
        if line.startswith(("---", "+++", "@@")):
            continue
        if line[:1] in "+-":
            out.append(line[1:])
    return out


def classify(row: dict, lang: str) -> tuple[str | None, str]:
    """(tier, reason): tier is "modern" | "generic" | None (reason = why it was skipped)."""
    license_ = (row.get("license") or "").lower()
    if license_ and license_ not in PERMISSIVE:
        return None, "license"
    old_file, new_file = row.get("old_file") or "", row.get("new_file") or ""
    if old_file != new_file:
        return None, "rename"
    if Path(new_file).suffix.lower() not in SUFFIXES[lang]:
        return None, "suffix"
    legacy, modern = row.get("old_contents") or "", row.get("new_contents") or ""
    if not (MIN_CHARS <= len(legacy) <= MAX_CHARS and MIN_CHARS <= len(modern) <= MAX_CHARS):
        return None, "size"
    if legacy.split() == modern.split():
        return None, "whitespace_only"
    subject = " ".join((row.get("subject") or "").split())
    message = (row.get("message") or "")[:600]
    if not (8 <= len(subject) <= 120) or not SUBJECT_START_RE.match(subject):
        return None, "subject_shape"
    if DENY_RE.search(subject):
        return None, "subject_deny"
    changed = _changed_lines(legacy, modern)
    if not changed:
        return None, "no_diff"
    if all(COMMENT_LINE_RE.match(c) for c in changed):
        return None, "comment_only"
    n_lines = max(len(legacy.splitlines()), len(modern.splitlines()), 1)
    modern_tier = bool(MODERN_RE[lang].search(f"{subject}\n{message}"))
    cap = 120 if modern_tier else 60
    if len(changed) > cap or (not modern_tier and len(changed) > 0.6 * n_lines):
        return None, "diff_size"
    if len(subject.split()) < (2 if modern_tier else 3):
        return None, "subject_short"
    if not (_parseable(legacy, lang) and _parseable(modern, lang)):
        return None, "parse"
    return ("modern" if modern_tier else "generic"), "kept"


def _to_pair(row: dict, lang: str, tier: str) -> dict:
    return {
        "objective": " ".join((row.get("subject") or "").split())[:200],
        "language": lang,
        "legacy": row["old_contents"],
        "modern": row["new_contents"],
        "source": "commitpack_modern" if tier == "modern" else "commitpack",
        "repo": row.get("repos"),
        "commit": row.get("commit"),
        "path": row.get("new_file"),
        "license": (row.get("license") or "").lower(),
        "tier": tier,
    }


def shard_name(lang: str, i: int) -> str:
    return f"data/{FOLDER[lang]}/{FOLDER[lang]}-{i:04d}.jsonl"


def standalone(row: dict, lang: str) -> dict | None:
    """Old file version that compiles without project headers → input for make_tool_pairs.py."""
    from make_tool_pairs import cpp_candidate, rust_candidate

    license_ = (row.get("license") or "").lower()
    path = row.get("old_file") or ""
    text = row.get("old_contents") or ""
    if (license_ and license_ not in PERMISSIVE) or Path(path).suffix.lower() not in SUFFIXES[lang]:
        return None
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return None
    if not (cpp_candidate if lang == "cpp" else rust_candidate)(text):
        return None
    return {"legacy": text, "repo": row.get("repos"), "commit": row.get("commit"), "path": path,
            "license": license_, "tier": "standalone"}


def process_shard(lang: str, i: int, keep: bool, mode: str = "pairs") -> tuple[int, list[dict], dict, float]:
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    path = Path(hf_hub_download(REPO, repo_type="dataset", filename=shard_name(lang, i), local_dir=str(CACHE)))
    kept: list[dict] = []
    stats: Counter = Counter()
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            stats["scanned"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue
            if mode == "standalone":
                item = standalone(row, lang)
                stats["kept_standalone" if item else "skip"] += 1
                if item:
                    kept.append(item)
                continue
            tier, reason = classify(row, lang)
            stats[reason if tier is None else f"kept_{tier}"] += 1
            if tier:
                kept.append(_to_pair(row, lang, tier))
    if not keep:
        path.unlink(missing_ok=True)
    return i, kept, dict(stats), time.time() - t0


def parse_shards(spec: str | None, sample: int | None, lang: str, seed: int) -> list[int]:
    total = N_SHARDS[lang]
    if sample:
        return sorted(random.Random(f"{seed}|{lang}").sample(range(1, total + 1), min(sample, total)))
    if not spec or spec == "all":
        return list(range(1, total + 1))
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=sorted(FOLDER), required=True)
    ap.add_argument("--shards", default=None, help='"all" (default), "1-5", "1,7,9"')
    ap.add_argument("--sample", type=int, default=None, help="N shards spread at random (seeded)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--keep-shards", action="store_true", help="keep downloaded shards (default: delete)")
    ap.add_argument("--dry-run", action="store_true", help="stats only, write nothing")
    ap.add_argument("--standalone", action="store_true",
                    help="collect standalone old files (any commit) for make_tool_pairs.py instead of pairs")
    args = ap.parse_args()

    POOL.mkdir(parents=True, exist_ok=True)
    name = "standalone" if args.standalone else "commitpack"
    mode = "standalone" if args.standalone else "pairs"
    out = POOL / f"{args.lang}_{name}.jsonl"
    state_path = POOL / f"{args.lang}_{name}.state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() and not args.dry_run else {"shards": {}}
    todo = [i for i in parse_shards(args.shards, args.sample, args.lang, args.seed) if str(i) not in state["shards"]]
    print(f"{args.lang}: {len(todo)} shard(s) to mine, {len(state['shards'])} done", flush=True)

    total: Counter = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_shard, args.lang, i, args.keep_shards, mode) for i in todo]
        for fut in as_completed(futs):
            i, kept, stats, secs = fut.result()
            total.update(stats)
            print(f"  shard {i:4d}: scanned={stats.get('scanned', 0)} modern={stats.get('kept_modern', 0)} "
                  f"generic={stats.get('kept_generic', 0)} standalone={stats.get('kept_standalone', 0)} "
                  f"({secs:.0f}s)", flush=True)
            if args.dry_run:
                continue
            with out.open("a", encoding="utf-8") as fh:
                fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
            state["shards"][str(i)] = stats
            state_path.write_text(json.dumps(state, indent=1) + "\n")
    print(json.dumps(dict(total.most_common()), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
