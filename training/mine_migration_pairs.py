#!/usr/bin/env python3
"""Mine legacy→modern JS/TS pairs from public web sources.

Primary source: Hugging Face ``bigcode/commitpackft`` (permissive licenses),
filtered for upgrade/migration-like commit messages.

Schema (JSONL)::

    {"objective", "language", "legacy", "modern", "source", "repo", "commit", "path"}

Examples:
  python training/mine_migration_pairs.py --max-pairs 1500
  python training/mine_migration_pairs.py --dry-run --scan-limit 5000
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "datasets" / "js_ts_migration.jsonl"
SYNTHETIC_BACKUP = ROOT / "data" / "datasets" / "js_ts_migration.synthetic.jsonl"
DEFAULT_STATS = ROOT / "data" / "datasets" / "js_ts_migration.stats.json"
CACHE_DIR = ROOT / "data" / "cache" / "commitpackft"

MIGRATION_RE = re.compile(
    r"("
    r"\bmigrat(?:e|ion|ing|ed)\b|"
    r"\bupgrad(?:e|ing|ed)\b|"
    r"\bmoderniz(?:e|ing|ed|ation)\b|"
    r"\bconvert(?:ed|ing)?\s+to\b|"
    r"\brefactor(?:ed|ing)?\s+to\b|"
    r"\breplace\s+var\b|"
    r"\bvar\s+to\s+(?:const|let)\b|"
    r"\bcommonjs\b|\besm\b|\besmodules?\b|"
    r"\btypescript\b|\badd(?:ed)?\s+types?\b|"
    r"\bjs\s+to\s+ts\b|\bjavascript\s+to\s+typescript\b|"
    r"\bwebpack\b.*\bvite\b|\bvite\b.*\bwebpack\b|"
    r"\bbabel\b|\beslint\s+--fix\b|"
    r"\bdeprecated\b|\blegacy\b|"
    r"\bupdate\s+to\s+(?:react|vue|angular|next|node|typescript)\b|"
    r"\bbump\s+(?:react|vue|angular|next|node|typescript)\b|"
    r"\breact\s*1[678]\b|\bnext\.?js\b|"
    r"\basync(?:/|\s)?await\b|"
    r"\bpromis(?:e|ify)\b|"
    r"\boptional\s+chaining\b|\bnullish\b|"
    r"\btemplate\s+literal"
    r")",
    re.IGNORECASE,
)

DENY_RE = re.compile(
    r"("
    r"\bfix\s+typo\b|"
    r"\bspelling\b|"
    r"\bREADME\b|"
    r"\bchangelog\b|"
    r"\binitial\s+commit\b|"
    r"\bWIP\b|"
    r"\bmerge\s+branch\b"
    r")",
    re.IGNORECASE,
)

PERMISSIVE = {
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "cc0-1.0",
    "unlicense",
    "mpl-2.0",
}

SOURCE_SUFFIXES = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}


@dataclass
class Pair:
    objective: str
    language: str
    legacy: str
    modern: str
    source: str
    repo: str | None = None
    commit: str | None = None
    path: str | None = None


def _suffix(path: str) -> str:
    return Path(path).suffix.lower()


def _language_for(path: str, lang_field: str | None) -> str | None:
    suf = _suffix(path)
    if suf in {".ts", ".tsx"}:
        return "typescript"
    if suf in {".js", ".jsx", ".mjs", ".cjs"}:
        return "javascript"
    if lang_field:
        low = lang_field.lower()
        if "typescript" in low:
            return "typescript"
        if "javascript" in low:
            return "javascript"
    return None


def _looks_like_migration(subject: str, message: str) -> bool:
    text = f"{subject}\n{message}"
    if DENY_RE.search(text):
        return False
    return bool(MIGRATION_RE.search(text))


def _quality_ok(legacy: str, modern: str, *, min_chars: int, max_chars: int) -> bool:
    if not legacy or not modern or legacy == modern:
        return False
    if len(legacy) < min_chars or len(modern) < min_chars:
        return False
    if len(legacy) > max_chars or len(modern) > max_chars:
        return False
    return True


def _parseable(code: str, language: str) -> bool:
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_javascript as tsjs
        import tree_sitter_typescript as tsts

        parser = Parser()
        if language == "typescript":
            parser.language = Language(tsts.language_typescript())
        else:
            parser.language = Language(tsjs.language())
        tree = parser.parse(code.encode("utf-8"))
        return not tree.root_node.has_error
    except Exception:  # noqa: BLE001
        return True


def download_commitpack_lang(lang: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id="bigcode/commitpackft",
            repo_type="dataset",
            filename=f"data/{lang}/data.jsonl",
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
    )


def iter_commitpack_pairs(
    path: Path,
    *,
    language_hint: str,
    min_chars: int,
    max_chars: int,
    require_parse: bool,
    scan_limit: int | None,
) -> tuple[list[Pair], Counter]:
    kept: list[Pair] = []
    stats: Counter = Counter()
    with path.open(encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if scan_limit is not None and i >= scan_limit:
                break
            stats["scanned"] += 1
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue

            license_ = (row.get("license") or "").lower()
            if license_ and license_ not in PERMISSIVE:
                stats["license_skip"] += 1
                continue

            old_file = row.get("old_file") or ""
            new_file = row.get("new_file") or old_file
            if (
                _suffix(old_file) not in SOURCE_SUFFIXES
                and _suffix(new_file) not in SOURCE_SUFFIXES
            ):
                stats["suffix_skip"] += 1
                continue

            subject = row.get("subject") or ""
            message = row.get("message") or subject
            if not _looks_like_migration(subject, message):
                stats["keyword_skip"] += 1
                continue

            legacy = row.get("old_contents") or ""
            modern = row.get("new_contents") or ""
            if not _quality_ok(legacy, modern, min_chars=min_chars, max_chars=max_chars):
                stats["quality_skip"] += 1
                continue

            language = _language_for(new_file or old_file, row.get("lang")) or language_hint
            if require_parse and not (
                _parseable(legacy, language) and _parseable(modern, language)
            ):
                stats["parse_skip"] += 1
                continue

            objective = (subject.strip() or "Modernize legacy code")[:200]
            kept.append(
                Pair(
                    objective=objective,
                    language=language,
                    legacy=legacy,
                    modern=modern,
                    source="commitpackft",
                    repo=row.get("repos"),
                    commit=row.get("commit"),
                    path=new_file or old_file,
                )
            )
            stats["kept"] += 1
    return kept, stats


def ensure_synthetic_backup(current: Path, backup: Path) -> list[Pair]:
    """Preserve original hand-written pairs once; always reload synthetic from backup."""
    if current.exists() and not backup.exists():
        # First run: current file is still the hand-curated set
        rows = [json.loads(l) for l in current.read_text().splitlines() if l.strip()]
        if rows and all(r.get("source", "synthetic") == "synthetic" for r in rows):
            shutil.copy2(current, backup)
            print(f"Backed up synthetic → {backup}")
        else:
            # Already merged somehow without backup — extract synthetic rows if any
            synth = [r for r in rows if r.get("source", "synthetic") == "synthetic"]
            if synth:
                with backup.open("w", encoding="utf-8") as fh:
                    for r in synth:
                        r = {**r, "source": "synthetic"}
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"Extracted synthetic → {backup}")

    if not backup.exists():
        return []

    pairs: list[Pair] = []
    for line in backup.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pairs.append(
            Pair(
                objective=row["objective"],
                language=row.get("language", "javascript"),
                legacy=row["legacy"],
                modern=row["modern"],
                source="synthetic",
            )
        )
    return pairs


def dedupe(pairs: list[Pair]) -> list[Pair]:
    seen: set[tuple[str, str]] = set()
    out: list[Pair] = []
    for p in pairs:
        key = (p.legacy.strip(), p.modern.strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def write_jsonl(path: Path, pairs: list[Pair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--max-pairs", type=int, default=1500)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--scan-limit", type=int, default=None)
    parser.add_argument("--require-parse", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--langs", nargs="+", default=["javascript", "typescript"])
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))

    synthetic = ensure_synthetic_backup(args.out, SYNTHETIC_BACKUP)
    mined: list[Pair] = []
    per_lang: dict[str, dict] = {}

    for lang in args.langs:
        cache_file = args.cache_dir / "data" / lang / "data.jsonl"
        if not args.skip_download or not cache_file.exists():
            print(f"Downloading CommitPackFT/{lang}…")
            try:
                cache_file = download_commitpack_lang(lang, args.cache_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR downloading {lang}: {exc}", file=sys.stderr)
                return 1
        else:
            print(f"Using cache {cache_file}")

        print(f"Scanning {cache_file}…")
        pairs, stats = iter_commitpack_pairs(
            cache_file,
            language_hint="typescript" if lang == "typescript" else "javascript",
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            require_parse=args.require_parse,
            scan_limit=args.scan_limit,
        )
        per_lang[lang] = dict(stats)
        print(f"  {lang}: scanned={stats['scanned']} kept={stats['kept']}")
        mined.extend(pairs)

    mined = dedupe(mined)
    mined.sort(key=lambda p: (len(p.legacy) + len(p.modern), len(p.objective)))
    if args.max_pairs and len(mined) > args.max_pairs:
        mined = mined[: args.max_pairs]

    combined = dedupe(synthetic + mined)
    report = {
        "total": len(combined),
        "by_source": dict(Counter(p.source for p in combined)),
        "by_language": dict(Counter(p.language for p in combined)),
        "mined_capped": len(mined),
        "synthetic": len(synthetic),
        "filters": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "max_pairs": args.max_pairs,
            "require_parse": args.require_parse,
        },
        "commitpack_stats": per_lang,
    }
    print(json.dumps(report, indent=2))

    if args.dry_run:
        print("dry-run: not writing")
        return 0

    write_jsonl(args.out, combined)
    args.stats.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(combined)} pairs → {args.out}")
    print(f"Stats → {args.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
