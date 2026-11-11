#!/usr/bin/env python3
"""Phase 7: run migrate-repo on customer-like / external JS benches; write reports.

Modes:
  --dry-run     discover + topo-order only (no LLM)
  default       full migrate, no file writes
  --write       full migrate and write successful drafts
  --only ID     run a single bench id
  --clone-external  shallow-clone public repos into data/bench/
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_BENCHES = [
    {
        "id": "bench_util",
        "path": ROOT / "fixtures" / "bench_util",
        "under": "src",
        "seeds": ["src/numbers.js=fixtures/bench_util/seeds/numbers.broken.js"],
        "files": ["src/numbers.js"],
        "class": "tiny_util_lib",
        "embedder": "hash",
    },
    {
        "id": "bench_mixed",
        "path": ROOT / "fixtures" / "bench_mixed",
        "under": "src",
        "seeds": ["src/config.js=fixtures/bench_mixed/seeds/config.broken.js"],
        "files": ["src/config.js"],
        "class": "mixed_js_ts",
        "embedder": "hash",
    },
    {
        # The seed draft imports `pretty-ms`, which neither package.json nor the original file uses: the
        # package policy refuses to install it (sandbox/deps.py) and the model must do without.
        "id": "bench_express",
        "path": ROOT / "fixtures" / "bench_express",
        "under": "src",
        "seeds": [
            "src/duration.js=fixtures/bench_express/seeds/duration.missing_dep.js",
        ],
        "files": ["src/duration.js"],
        "class": "app_with_deps",
        "embedder": "hash",
        "expect": "refused",
    },
    {
        # Same project with REBIRTH_ALLOW_UNDECLARED_PACKAGES=1: DependencyMissing → install `pretty-ms`
        # in the next run's setup step (with network, before the draft) → tests pass offline.
        "id": "bench_express_optin",
        "path": ROOT / "fixtures" / "bench_express",
        "under": "src",
        "seeds": [
            "src/duration.js=fixtures/bench_express/seeds/duration.missing_dep.js",
        ],
        "files": ["src/duration.js"],
        "class": "app_with_deps",
        "embedder": "hash",
        "env": {"REBIRTH_ALLOW_UNDECLARED_PACKAGES": "1"},
        "objective": (
            "Keep the `pretty-ms` package import in duration.js. If tests fail with "
            "ERR_MODULE_NOT_FOUND / Cannot find package 'pretty-ms', install the missing "
            "dependency rather than rewriting away from `pretty-ms`. Tests must pass."
        ),
        "expect": "queued",
    },
    {
        "id": "bench_monorepo",
        "path": ROOT / "fixtures" / "bench_monorepo",
        "under": "packages/a,packages/b",
        "seeds": [
            "packages/a/src/hello.js="
            "fixtures/bench_monorepo/packages/a/seeds/hello.broken.js",
        ],
        "files": ["packages/a/src/hello.js", "packages/b/src/world.js"],
        "class": "monorepo_lite",
        "embedder": "hash",
        # Limit LLM migrate to seeded package; dry-run uses under roots.
        "migrate_files": ["packages/a/src/hello.js"],
    },
    {
        "id": "bench_external_plain",
        "path": ROOT / "fixtures" / "bench_external_plain",
        "under": ".",
        "seeds": [
            "index.js=fixtures/bench_external_plain/seeds/index.broken.js",
        ],
        "files": ["index.js"],
        "class": "external_style_util",
        "embedder": "hash",
    },
]

EXTERNAL_CLONES = [
    {
        # Real Hub clone — discovery only (upstream uses ava/xo, not node:test)
        "id": "is-plain-obj",
        "url": "https://github.com/sindresorhus/is-plain-obj.git",
        "under": ".",
        "class": "tiny_util_lib_external",
        "seeds": [],
        "files": None,
        "full_migrate": False,
    },
]


def _clone_external(dest_root: Path) -> list[dict]:
    dest_root.mkdir(parents=True, exist_ok=True)
    cloned: list[dict] = []
    for spec in EXTERNAL_CLONES:
        target = dest_root / spec["id"]
        if not target.exists():
            print(f"[clone] {spec['url']} → {target}")
            subprocess.run(
                ["git", "clone", "--depth", "1", spec["url"], str(target)],
                check=True,
            )
        else:
            print(f"[clone] reuse {target}")
        cloned.append({**spec, "path": target})
    return cloned


def _run_one(
    spec: dict,
    *,
    backend: str,
    write: bool,
    dry_run: bool,
    out_dir: Path,
) -> dict:
    from rebirth_ai.graph.batch import migrate_repository

    os.environ["REBIRTH_GENERATOR"] = backend
    embedder = spec.get("embedder") or ("hash" if backend in {"peft", "lora", "local"} else None)
    if embedder:
        os.environ["REBIRTH_EMBEDDER"] = embedder
    if backend in {"peft", "lora", "local"}:
        os.environ.setdefault("REBIRTH_PEFT_PROMPT_STYLE", "chat")

    project = Path(spec["path"]).resolve()
    cwd_seeds: list[str] = []
    for s in spec.get("seeds") or []:
        file_part, seed_part = s.split("=", 1)
        seed_path = Path(seed_part)
        if not seed_path.is_absolute():
            seed_path = ROOT / seed_path
        cwd_seeds.append(f"{file_part}={seed_path}")

    under = spec.get("under") or "src"
    if under == "":
        under = "."

    files = spec.get("migrate_files") or spec.get("files")
    if dry_run and spec.get("under"):
        # Discovery proof for monorepo / multi-root: don't limit to migrate_files
        files = None if "," in str(spec.get("under")) else files

    objective = (
        "Fix seeded files so project tests pass; modernize idioms "
        "without changing public API. Prefer installing missing packages "
        "when the classifier reports DependencyMissing."
    )
    objective = spec.get("objective") or objective

    print(f"\n=== bench {spec['id']} ({spec.get('class')}) dry_run={dry_run} ===")
    saved = {k: os.environ.get(k) for k in spec.get("env") or {}}
    os.environ.update(spec.get("env") or {})
    try:
        report = migrate_repository(
            project,
            objective=objective,
            files=list(files) if files else None,
            under=under,
            seeds=cwd_seeds or None,
            write=write and not dry_run,
            dry_run=dry_run,
            max_iterations=5,
            skip_without_tests=bool(spec.get("skip_without_tests", True)),
            on_file=lambda p: print(f"  → {p}"),
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    out_path = out_dir / f"{spec['id']}.json"
    report.write_json(out_path)

    summary = report.to_dict()["summary"]
    actionable = [r for r in report.results if not r.status.startswith("skipped")]
    okish = [
        r
        for r in actionable
        if r.status in {"success", "human_review", "planned"}
    ]
    rate = (len(okish) / len(actionable)) if actionable else 1.0
    trace = [t for r in report.results for t in r.trace]
    installed = any(t.startswith("install_dependency: queued") for t in trace)
    refused = any(t.startswith("install_dependency: refused") for t in trace)
    print(
        f"  summary success={summary['success']} failed={summary['failed']} "
        f"skipped={summary['skipped']} planned={summary.get('planned', 0)} "
        f"okish_rate={rate:.0%} install_queued={installed} install_refused={refused} → {out_path}"
    )
    return {
        "id": spec["id"],
        "class": spec.get("class"),
        "path": str(project),
        "report": str(out_path),
        "summary": summary,
        "okish_rate": rate,
        "install_dependency_seen": installed,
        "install_refused": refused,
        "expect": spec.get("expect"),
        "migrate_order": report.migrate_order,
        "dry_run": dry_run,
        "backend": backend,
        "statuses": {r.relative_file: r.status for r in report.results},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="ollama", choices=["ollama", "peft"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--only", action="append", default=[], help="Bench id (repeatable)")
    parser.add_argument("--clone-external", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "reports" / "bench",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    targets = list(DEFAULT_BENCHES)
    if args.clone_external:
        targets.extend(_clone_external(ROOT / "data" / "bench"))
    if args.only:
        wanted = set(args.only)
        targets = [t for t in targets if t["id"] in wanted]
        if not targets:
            print(f"No benches matched --only {sorted(wanted)}")
            return 2

    results = []
    for spec in targets:
        dry = args.dry_run
        # External without seeds stays dry-run; with prepared seeds, full migrate
        if not spec.get("seeds") and not args.dry_run:
            dry = True
        results.append(
            _run_one(
                spec,
                backend=args.backend,
                write=args.write,
                dry_run=dry,
                out_dir=args.out_dir,
            )
        )

    index = {"phase": 7, "backend": args.backend, "results": results}
    index_path = args.out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {index_path}")

    # Gates
    if args.dry_run:
        return 0
    rates = [r["okish_rate"] for r in results if r["id"].startswith("bench_")]
    if rates and min(rates) < 0.8:
        return 1
    warned = False
    for r in results:
        if r.get("expect") == "queued" and not r.get("install_dependency_seen"):
            print(f"WARN: {r['id']} did not install the missing package")
            warned = True
        if r.get("expect") == "refused" and (r.get("install_dependency_seen") or not r.get("install_refused")):
            print(f"WARN: {r['id']}: the draft-only package was not refused")
            warned = True
    return 1 if warned else 0  # the dependency paths must be exercised


if __name__ == "__main__":
    raise SystemExit(main())
