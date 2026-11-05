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

LANG_GROUP = {
    "javascript": "js_ts",
    "typescript": "js_ts",
    "python": "python",
    "go": "go",
    "java": "java",
    "c-sharp": "csharp",
    "csharp": "csharp",
    "c#": "csharp",
    "c_sharp": "csharp",
    "rust": "rust",
    "c++": "cpp",
    "cpp": "cpp",
    "swift": "swift",
}
OUT_FOR_GROUP = {
    "js_ts": ROOT / "data" / "datasets" / "js_ts_migration.jsonl",
    "python": ROOT / "data" / "datasets" / "python_migration.jsonl",
    "go": ROOT / "data" / "datasets" / "go_migration.jsonl",
    "java": ROOT / "data" / "datasets" / "java_migration.jsonl",
    "csharp": ROOT / "data" / "datasets" / "csharp_migration.jsonl",
    "rust": ROOT / "data" / "datasets" / "rust_migration.jsonl",
    "cpp": ROOT / "data" / "datasets" / "cpp_migration.jsonl",
    "swift": ROOT / "data" / "datasets" / "swift_migration.jsonl",
}
STATS_FOR_GROUP = {k: v.with_suffix(".stats.json") for k, v in OUT_FOR_GROUP.items()}
SYNTHETIC_FOR_GROUP = {
    k: v.with_name(v.stem + ".synthetic.jsonl") for k, v in OUT_FOR_GROUP.items()
}

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

JS_SOURCE_SUFFIXES = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}
PY_SOURCE_SUFFIXES = {".py"}
GO_SOURCE_SUFFIXES = {".go"}
JAVA_SOURCE_SUFFIXES = {".java"}
CSHARP_SOURCE_SUFFIXES = {".cs"}
RUST_SOURCE_SUFFIXES = {".rs"}
CPP_SOURCE_SUFFIXES = {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h", ".c"}
SWIFT_SOURCE_SUFFIXES = {".swift"}
SOURCE_SUFFIXES = (
    JS_SOURCE_SUFFIXES
    | PY_SOURCE_SUFFIXES
    | GO_SOURCE_SUFFIXES
    | JAVA_SOURCE_SUFFIXES
    | CSHARP_SOURCE_SUFFIXES
    | RUST_SOURCE_SUFFIXES
    | CPP_SOURCE_SUFFIXES
    | SWIFT_SOURCE_SUFFIXES
)

PYTHON_MIGRATION_RE = re.compile(
    r"("
    r"\bmigrat(?:e|ion|ing|ed)\b|"
    r"\bupgrad(?:e|ing|ed)\b|"
    r"\bmoderniz(?:e|ing|ed|ation)\b|"
    r"\bconvert(?:ed|ing)?\s+to\b|"
    r"\brefactor(?:ed|ing)?\s+to\b|"
    r"\bpython\s*3\b|\bpy2\b|\bpy3\b|"
    r"\btype\s+hints?\b|\btyping\b|\bannotated\b|"
    r"\basync(?:io)?\b|\bawait\b|"
    r"\bpathlib\b|\bf-?string\b|"
    r"\bdataclass(?:es)?\b|"
    r"\bpydantic\b|\bfastapi\b|\bdjango\b|"
    r"\bunittest\b.*\bpytest\b|\bpytest\b.*\bunittest\b|"
    r"\bdeprecated\b|\blegacy\b|"
    r"\bprint\s*\(.*\)\s*#?\s*py2"
    r")",
    re.IGNORECASE,
)


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
    if suf == ".py":
        return "python"
    if suf == ".go":
        return "go"
    if suf == ".java":
        return "java"
    if suf == ".cs":
        return "csharp"
    if suf == ".rs":
        return "rust"
    if suf in CPP_SOURCE_SUFFIXES:
        return "cpp"
    if suf == ".swift":
        return "swift"
    if lang_field:
        low = lang_field.lower().replace("#", "sharp")
        for needle, label in (
            ("typescript", "typescript"),
            ("javascript", "javascript"),
            ("python", "python"),
            ("golang", "go"),
            ("go", "go"),
            ("java", "java"),
            ("c_sharp", "csharp"),
            ("csharp", "csharp"),
            ("c-sharp", "csharp"),
            ("rust", "rust"),
            ("c++", "cpp"),
            ("cpp", "cpp"),
            ("swift", "swift"),
        ):
            if needle in low:
                return label
    return None


def _looks_like_migration(subject: str, message: str, *, language_hint: str) -> bool:
    text = f"{subject}\n{message}"
    if DENY_RE.search(text):
        return False
    if language_hint == "python":
        return bool(PYTHON_MIGRATION_RE.search(text) or MIGRATION_RE.search(text))
    return bool(MIGRATION_RE.search(text))


def _suffixes_for_hint(language_hint: str) -> set[str]:
    return {
        "python": PY_SOURCE_SUFFIXES,
        "go": GO_SOURCE_SUFFIXES,
        "java": JAVA_SOURCE_SUFFIXES,
        "csharp": CSHARP_SOURCE_SUFFIXES,
        "c-sharp": CSHARP_SOURCE_SUFFIXES,
        "rust": RUST_SOURCE_SUFFIXES,
        "cpp": CPP_SOURCE_SUFFIXES,
        "c++": CPP_SOURCE_SUFFIXES,
        "swift": SWIFT_SOURCE_SUFFIXES,
        "typescript": JS_SOURCE_SUFFIXES,
        "javascript": JS_SOURCE_SUFFIXES,
    }.get(language_hint, JS_SOURCE_SUFFIXES)


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

        parser = Parser()
        if language == "typescript":
            import tree_sitter_typescript as tsts

            parser.language = Language(tsts.language_typescript())
        elif language == "python":
            import tree_sitter_python as tspy

            parser.language = Language(tspy.language())
        elif language == "go":
            import tree_sitter_go as tsgo

            parser.language = Language(tsgo.language())
        elif language == "java":
            import tree_sitter_java as tsjava

            parser.language = Language(tsjava.language())
        elif language in {"csharp", "c-sharp"}:
            import tree_sitter_c_sharp as tscs

            parser.language = Language(tscs.language())
        elif language == "rust":
            import tree_sitter_rust as tsrust

            parser.language = Language(tsrust.language())
        elif language in {"cpp", "c++"}:
            import tree_sitter_cpp as tscpp

            parser.language = Language(tscpp.language())
        elif language == "swift":
            import tree_sitter_swift as tsswift

            parser.language = Language(tsswift.language())
        else:
            import tree_sitter_javascript as tsjs

            parser.language = Language(tsjs.language())
        tree = parser.parse(code.encode("utf-8"))
        return not tree.root_node.has_error
    except Exception:  # noqa: BLE001
        return True


def _commitpack_folder(lang: str) -> str:
    """Map CLI/lang aliases to CommitPackFT folder names (e.g. csharp → c#)."""
    return {
        "csharp": "c#",
        "c-sharp": "c#",
        "c_sharp": "c#",
        "cpp": "c++",
        "c++": "c++",
    }.get(lang, lang)


def download_commitpack_lang(lang: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    folder = _commitpack_folder(lang)
    return Path(
        hf_hub_download(
            repo_id="bigcode/commitpackft",
            repo_type="dataset",
            filename=f"data/{folder}/data.jsonl",
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
    require_keyword: bool = True,
    source_tag: str = "commitpackft",
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
            allowed = _suffixes_for_hint(language_hint)
            old_suf = _suffix(old_file)
            new_suf = _suffix(new_file)
            # Some CommitPackFT rows omit paths but set lang= — accept when hint matches.
            if old_suf not in allowed and new_suf not in allowed:
                lang_row = (row.get("lang") or "").lower()
                hint_aliases = {
                    "go": {"go", "golang"},
                    "java": {"java"},
                    "csharp": {"c-sharp", "csharp", "c_sharp", "c#"},
                    "c-sharp": {"c-sharp", "csharp", "c_sharp", "c#"},
                    "rust": {"rust"},
                    "cpp": {"c++", "cpp"},
                    "c++": {"c++", "cpp"},
                    "swift": {"swift"},
                    "python": {"python"},
                    "javascript": {"javascript"},
                    "typescript": {"typescript"},
                }.get(language_hint, set())
                if not (lang_row and lang_row in hint_aliases):
                    stats["suffix_skip"] += 1
                    continue

            subject = row.get("subject") or ""
            message = row.get("message") or subject
            if require_keyword and not _looks_like_migration(
                subject, message, language_hint=language_hint
            ):
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
                    source=source_tag,
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


def _group_for_langs(langs: list[str]) -> str:
    groups = {LANG_GROUP.get(l, l) for l in langs}
    if len(groups) != 1:
        raise SystemExit(
            f"--langs must stay within one dataset group (got {sorted(groups)}); "
            "run JS/TS and Python mines separately"
        )
    return next(iter(groups))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--max-pairs", type=int, default=1500)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--scan-limit", type=int, default=None)
    parser.add_argument("--require-parse", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("migration", "corpus"),
        default="migration",
        help="migration=keyword filter; corpus=license+quality(+parse) fill toward 15K",
    )
    parser.add_argument("--langs", nargs="+", default=["javascript", "typescript"])
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))

    group = _group_for_langs(args.langs)
    out_path = args.out or OUT_FOR_GROUP.get(group, DEFAULT_OUT)
    stats_path = args.stats or STATS_FOR_GROUP.get(group, DEFAULT_STATS)
    synth_backup = SYNTHETIC_FOR_GROUP.get(group, SYNTHETIC_BACKUP)

    synthetic = ensure_synthetic_backup(out_path, synth_backup)
    existing: list[Pair] = []
    if out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") == "synthetic":
                continue
            existing.append(
                Pair(
                    objective=row.get("objective") or "Modernize legacy code",
                    language=row.get("language") or "javascript",
                    legacy=row["legacy"],
                    modern=row["modern"],
                    source=str(row.get("source") or "commitpackft"),
                    repo=row.get("repo"),
                    commit=row.get("commit"),
                    path=row.get("path"),
                )
            )
    mined: list[Pair] = []
    per_lang: dict[str, dict] = {}
    require_keyword = args.mode == "migration"
    source_tag = "commitpackft" if require_keyword else "commitpackft_corpus"

    for lang in args.langs:
        folder = _commitpack_folder(lang)
        cache_file = args.cache_dir / "data" / folder / "data.jsonl"
        if not args.skip_download or not cache_file.exists():
            print(f"Downloading CommitPackFT/{folder}…")
            try:
                cache_file = download_commitpack_lang(lang, args.cache_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR downloading {folder}: {exc}", file=sys.stderr)
                return 1
        else:
            print(f"Using cache {cache_file}")

        print(f"Scanning {cache_file} (mode={args.mode})…")
        hint = {
            "python": "python",
            "typescript": "typescript",
            "javascript": "javascript",
            "go": "go",
            "java": "java",
            "c-sharp": "csharp",
            "csharp": "csharp",
            "c#": "csharp",
            "rust": "rust",
            "c++": "cpp",
            "cpp": "cpp",
            "swift": "swift",
        }.get(lang, "javascript")
        pairs, stats = iter_commitpack_pairs(
            cache_file,
            language_hint=hint,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            require_parse=args.require_parse,
            scan_limit=args.scan_limit,
            require_keyword=require_keyword,
            source_tag=source_tag,
        )
        per_lang[lang] = dict(stats)
        print(f"  {lang}: scanned={stats['scanned']} kept={stats['kept']}")
        mined.extend(pairs)

    mined = dedupe(mined)
    mined.sort(key=lambda p: (len(p.legacy) + len(p.modern), len(p.objective)))
    if args.max_pairs and len(mined) > args.max_pairs:
        mined = mined[: args.max_pairs]

    combined = dedupe(synthetic + existing + mined)
    report = {
        "total": len(combined),
        "group": group,
        "mode": args.mode,
        "by_source": dict(Counter(p.source for p in combined)),
        "by_language": dict(Counter(p.language for p in combined)),
        "mined_capped": len(mined),
        "existing_kept": len(existing),
        "synthetic": len(synthetic),
        "filters": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "max_pairs": args.max_pairs,
            "require_parse": args.require_parse,
            "require_keyword": require_keyword,
        },
        "commitpack_stats": per_lang,
    }
    print(json.dumps(report, indent=2))

    if args.dry_run:
        print("dry-run: not writing")
        return 0

    write_jsonl(out_path, combined)
    stats_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(combined)} pairs → {out_path}")
    print(f"Stats → {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
