#!/usr/bin/env python3
"""Phase 19: modernize real files from small public repositories, served 7B adapters vs plain 14B.

For every file: the project's own tests must pass on the original in the sandbox (else the file is
left out), then ``migrate_repository`` runs once per backend with the same objective, writing into
the clone. The script records status (success = changed and the tests pass, unchanged, failed),
the diff size and comment lines kept, then restores the file. ``--backends fallback`` runs the
shipped default (served 7B, then a 14B where it changed nothing, ``graph/fallback.py``); the
summary's ``fallback`` row composes it from separate ``auto`` and ``ollama`` runs.

    python training/bench_real_repos.py                       # all repos, backends auto + ollama
    python training/bench_real_repos.py --only go-humanize --backends auto

Repos are shallow-cloned at pinned commits into ``data/bench/<name>`` (git-ignored); all are MIT /
BSD licensed. Diffs go to ``data/bench/_diffs/`` (not committed: third-party code); the report
(numbers only) to ``data/reports/real_repos.json``. Backends run one after the other (all 7B runs,
then all 14B runs) so Ollama swaps models once.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BENCH = ROOT / "data" / "bench"

# name, clone URL, pinned commit, license, files to modernize
REPOS = [
    ("is-number", "https://github.com/jonschlinkert/is-number.git", "98e8ff1da1a89f93d1397a24d7413ed15421c139", "MIT", ["index.js"]),
    ("kind-of", "https://github.com/jonschlinkert/kind-of.git", "3c7e8fd86094cfdb1b1bd048a39f699161da62e4", "MIT", ["index.js"]),
    ("fill-range", "https://github.com/jonschlinkert/fill-range.git", "bef65700633e1a9debb1ce67bf2922731c9ef4b5", "MIT", ["index.js"]),
    ("to-regex-range", "https://github.com/micromatch/to-regex-range.git", "b9fa118cc271dff9504946a23e54f2bd8c364bdf", "MIT", ["index.js"]),
    ("repeat-string", "https://github.com/jonschlinkert/repeat-string.git", "3914965fe08212b60cdbb32ab83ef5f8b1d5df47", "MIT", ["index.js"]),
    ("minimist", "https://github.com/minimistjs/minimist.git", "ecfdaea23e7931c0d529c52b743c711c3278a8ce", "MIT", ["index.js"]),
    ("inflection", "https://github.com/jpvanhal/inflection.git", "88eefaacf7d0caaa701af7c8ab2d0ab3f17086f1", "MIT", ["inflection/__init__.py"]),
    ("shortuuid", "https://github.com/skorokithakis/shortuuid.git", "75a7803c3633d36eb2486e2168bb3df237a6cd43", "BSD-3-Clause", ["shortuuid/main.py"]),
    ("python-string-utils", "https://github.com/daveoncode/python-string-utils.git", "78929d88d90b1f90cb4837528ed955166bf0f559", "MIT", ["string_utils/generation.py"]),
    ("python-slugify", "https://github.com/un33k/python-slugify.git", "35aca63fc2672cba2c739ff6dff685a4f26a4f9a", "MIT", ["slugify/slugify.py", "slugify/special.py"]),
    ("go-homedir", "https://github.com/mitchellh/go-homedir.git", "af06845cf3004701891bf4fdb884bfe4920b3727", "MIT", ["homedir.go"]),
    ("errors", "https://github.com/pkg/errors.git", "87f8819acf6dc28bf5d3c14b334268236d686f48", "BSD-2-Clause", ["errors.go", "stack.go"]),
    ("go-humanize", "https://github.com/dustin/go-humanize.git", "f5b54ae1392365a578bb43ab728e3fcdc291f0cd", "MIT", ["bytes.go", "comma.go", "times.go", "si.go"]),
]

LABEL = {".js": "JavaScript", ".ts": "TypeScript", ".py": "Python", ".go": "Go"}
COMMENT = {".js": ("//", "/*", "*"), ".ts": ("//", "/*", "*"), ".go": ("//", "/*", "*"), ".py": ("#",)}


def objective(path: str) -> str:
    lang = LABEL.get(Path(path).suffix, "the language's")
    return (f"Modernize this file to current {lang} idioms without changing its public API or its "
            "behavior. Keep its comments and docstrings.")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def clone(name: str, url: str, sha: str) -> Path:
    dest = BENCH / name
    if not dest.is_dir():
        BENCH.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(dest)], check=True)
        git(dest, "remote", "add", "origin", url)
        git(dest, "fetch", "-q", "--depth", "1", "origin", sha)
        git(dest, "checkout", "-q", "FETCH_HEAD")
    head = git(dest, "rev-parse", "HEAD").strip()
    if head != sha:
        raise SystemExit(f"{dest} is at {head}, expected {sha}: remove it to re-clone")
    if git(dest, "status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit(f"{dest} has local changes: `git -C {dest} checkout -- .` first")
    return dest


def comment_lines(text: str, suffix: str) -> int:
    pre = COMMENT.get(suffix, ())
    n = sum(1 for line in text.splitlines() if line.strip().startswith(pre))
    return n + (text.count('"""') if suffix == ".py" else 0)  # + docstring delimiters


def baseline(repo: Path, rel: str) -> dict:
    from rebirth_ai.adapters.registry import get_adapter_for_project
    from rebirth_ai.graph.discovery import infer_test_path
    from rebirth_ai.sandbox.runner import SandboxRunner

    test = infer_test_path(repo, rel)
    if test is None:
        return {"test": None, "ok": False, "why": "no test found"}
    adapter = get_adapter_for_project(repo)
    res = SandboxRunner(adapter).run_project(repo, relative_test=test)
    return {"test": test, "ok": res.ok, "exit": res.exit_code, "seconds": round(res.duration_sec, 1),
            "why": "" if res.ok else (res.stdout + res.stderr)[-600:]}


def run_one(repo: Path, rel: str, backend: str, diff_dir: Path) -> dict:
    from rebirth_ai.graph.batch import migrate_repository

    # auto = served 7B only; fallback = auto + the 14B when the 7B changes nothing (the default)
    os.environ["REBIRTH_GENERATOR"] = "auto" if backend == "fallback" else backend
    t0 = time.time()
    report = migrate_repository(repo, objective=objective(rel), files=[rel], under=".", write=True,
                                max_iterations=5, fallback=backend == "fallback")
    (r,) = report.results
    diff = git(repo, "diff", "--", rel)
    before = git(repo, "show", f"HEAD:{rel}")
    after = (repo / rel).read_text(encoding="utf-8")
    git(repo, "checkout", "--", rel)
    if diff:
        diff_dir.mkdir(parents=True, exist_ok=True)
        (diff_dir / f"{repo.name}__{rel.replace('/', '_')}__{backend}.diff").write_text(diff, encoding="utf-8")
    models = sorted({m for t in r.trace for m in re.findall(r"model=(\S+)", t)})
    suf = Path(rel).suffix
    return {
        "status": r.status, "iterations": r.iterations, "error_class": r.error_class, "models": models,
        "fallback_model": r.fallback_model,
        "seconds": round(time.time() - t0, 1),
        "added": sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")),
        "removed": sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")),
        "comments_before": comment_lines(before, suf), "comments_after": comment_lines(after, suf),
    }


def summarize(rows: list[dict], backends: list[str]) -> dict:
    out: dict[str, dict] = {}
    ran = [r for r in rows if r["baseline"]["ok"]]
    for b in backends:
        res = [r["results"][b] for r in ran if b in r["results"]]
        out[b] = {s: sum(1 for x in res if x["status"] == s) for s in ("success", "unchanged")}
        out[b]["failed"] = len(res) - out[b]["success"] - out[b]["unchanged"]
        out[b]["files"] = len(res)
    if {"auto", "ollama"} <= set(backends):
        fb = [(r["results"]["auto"] if r["results"]["auto"]["status"] == "success" else r["results"]["ollama"])
              for r in ran if "auto" in r["results"] and "ollama" in r["results"]]
        out["fallback"] = {s: sum(1 for x in fb if x["status"] == s) for s in ("success", "unchanged")}
        out["fallback"]["failed"] = len(fb) - out["fallback"]["success"] - out["fallback"]["unchanged"]
        out["fallback"]["files"] = len(fb)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", default=[], help="repo name (repeatable)")
    ap.add_argument("--backends", nargs="+", default=["auto", "ollama"],
                    help="auto = served 7B only; ollama = plain 14B (OLLAMA_MODEL); fallback = the shipped default "
                    "(served 7B, then a 14B when it changes nothing)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "reports" / "real_repos.json")
    ap.add_argument("--diff-dir", type=Path, default=BENCH / "_diffs")
    args = ap.parse_args()
    os.environ.setdefault("REBIRTH_EMBEDDER", "hash")

    repos = [r for r in REPOS if not args.only or r[0] in args.only]
    rows: list[dict] = []
    for name, url, sha, lic, files in repos:
        repo = clone(name, url, sha)
        for rel in files:
            b = baseline(repo, rel)
            print(f"[baseline] {name}/{rel}: test={b['test']} ok={b['ok']}", flush=True)
            rows.append({"repo": name, "url": url, "commit": sha, "license": lic, "file": rel,
                         "lines": (repo / rel).read_text(encoding="utf-8").count("\n"), "baseline": b, "results": {}})
    for backend in args.backends:
        for row in rows:
            if not row["baseline"]["ok"]:
                continue
            res = run_one(BENCH / row["repo"], row["file"], backend, args.diff_dir)
            row["results"][backend] = res
            print(f"[{backend}] {row['repo']}/{row['file']}: {res['status']} it={res['iterations']} "
                  f"+{res['added']}/-{res['removed']} comments {res['comments_before']}→{res['comments_after']} "
                  f"{res['seconds']}s {res['models']}", flush=True)
    summary = summarize(rows, args.backends)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"backends": args.backends, "summary": summary, "rows": rows}, indent=1) + "\n",
                        encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
