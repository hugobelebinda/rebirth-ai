#!/usr/bin/env python3
"""Build combined multi-lang corpus manifest + optional concat JSONL (Phase 10)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "data" / "datasets"

FILES = {
    "js_ts": DATASETS / "js_ts_migration.jsonl",
    "python": DATASETS / "python_migration.jsonl",
    "go": DATASETS / "go_migration.jsonl",
    "java": DATASETS / "java_migration.jsonl",
    "csharp": DATASETS / "csharp_migration.jsonl",
    "rust": DATASETS / "rust_migration.jsonl",
    "cpp": DATASETS / "cpp_migration.jsonl",
    "swift": DATASETS / "swift_migration.jsonl",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=DATASETS / "corpus_manifest.json",
    )
    parser.add_argument(
        "--concat",
        type=Path,
        default=None,
        help="Optional combined JSONL path",
    )
    args = parser.parse_args()

    by_lang: dict[str, int] = {}
    by_source: Counter[str] = Counter()
    total = 0
    concat_fh = None
    if args.concat is not None:
        args.concat.parent.mkdir(parents=True, exist_ok=True)
        concat_fh = args.concat.open("w", encoding="utf-8")

    try:
        for lang, path in FILES.items():
            n = 0
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict) or "legacy" not in row or "modern" not in row:
                        continue
                    n += 1
                    by_source[str(row.get("source") or "unknown")] += 1
                    if concat_fh is not None:
                        concat_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            by_lang[lang] = n
            total += n
    finally:
        if concat_fh is not None:
            concat_fh.close()

    manifest = {
        "total": total,
        "target": 15000,
        "met_target": total >= 15000,
        "by_language": by_lang,
        "by_source": dict(by_source),
        "files": {k: str(v.relative_to(ROOT)) for k, v in FILES.items()},
    }
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if total >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
