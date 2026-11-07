#!/usr/bin/env python3
"""Tool-generated modernization pairs (Phase 15 data): correct by construction, compiler-checked.

Takes old, permissively licensed files from the mined pools (``data/datasets/pool``) and the
corpus, runs a deterministic modernizer and keeps (original → modernized) as an SFT pair when
the file compiles on its own before AND after, and the tool changed something real.

  rust  scratch crate (edition 2015, no dependencies) → ``cargo fix --edition`` to 2018 and 2021
        → ``cargo fix --edition-idioms`` → ``r#try!(e)`` → ``e?`` → ``cargo clippy --fix``
  cpp   standard-library-only files → ``clang-tidy --fix`` with ``modernize-*`` (minus purely
        stylistic checks) at -std=c++20

The objective names what changed (edition steps / fired checks), e.g.
"Modernize for C++11+: use nullptr, use override, use range-based for loops".

Output: ``data/datasets/pool/<lang>_toolpairs.jsonl`` (resumable: inputs already tried are
listed in ``<lang>_toolpairs.done``). Needs rustup (``~/.cargo/bin``) / ``clang-tidy`` (pip wheel).

  python training/make_tool_pairs.py --lang rust --limit 200 --workers 12
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data" / "datasets" / "pool"
CARGO_BIN = Path.home() / ".cargo" / "bin"
TIMEOUT = 120

# --------------------------------------------------------------------------- shared


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _changed(a: str, b: str) -> int:
    return sum(1 for ln in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0)
               if ln[:1] in "+-" and not ln.startswith(("---", "+++")))


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=TIMEOUT)


# --------------------------------------------------------------------------- rust

EXTERN_RE = re.compile(r"^\s*(?:#\[macro_use\]\s*)?extern crate (\w+)", re.M)
MOD_DECL_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod \w+\s*;", re.M)
MAIN_RE = re.compile(r"^fn main\s*\(", re.M)
RTRY = "r#try!("


def rust_candidate(text: str) -> bool:
    if "#![feature" in text or "include!" in text or MOD_DECL_RE.search(text):
        return False
    return all(x in ("std", "core", "alloc", "test") for x in EXTERN_RE.findall(text))


def _strip_rtry(src: str) -> tuple[str, int]:
    """``r#try!(expr)`` → ``expr?`` (balanced parens, skips strings naively-safe: exprs are code)."""
    out, i, n = [], 0, 0
    while True:
        j = src.find(RTRY, i)
        if j < 0:
            out.append(src[i:])
            break
        k, depth = j + len(RTRY), 1
        while k < len(src) and depth:
            depth += {"(": 1, ")": -1}.get(src[k], 0)
            k += 1
        if depth:
            out.append(src[i:])
            break
        inner = src[j + len(RTRY):k - 1].strip()
        out.append(src[i:j])
        out.append(f"({inner})?" if re.search(r"\s(?:as|\+|-|\*|/|&&|\|\||==|!=)\s", inner) else f"{inner}?")
        i, n = k, n + 1
    return "".join(out), n


def _cargo(args: list[str], crate: Path, target: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "CARGO_TARGET_DIR": str(target), "CARGO_INCREMENTAL": "0", "RUSTFLAGS": "--cap-lints=warn",
           "PATH": f"{CARGO_BIN}:{os.environ.get('PATH', '')}", "CARGO_TERM_COLOR": "never"}
    return _run(["cargo", *args], crate, env)


def _set_edition(crate: Path, edition: str, bin_: bool) -> None:
    kind = '[[bin]]\nname = "t"\npath = "src/main.rs"' if bin_ else '[lib]\npath = "src/lib.rs"'
    (crate / "Cargo.toml").write_text(
        f'[package]\nname = "t"\nversion = "0.0.0"\nedition = "{edition}"\n\n{kind}\n\n[workspace]\n')


def _clippy_lints(res: subprocess.CompletedProcess) -> Counter:
    c: Counter = Counter()
    for line in res.stderr.splitlines():
        for m in re.finditer(r"https://rust-lang\.github\.io/rust-clippy/[^#\s]*#(\w+)", line):
            c[m.group(1)] += 1
    return c


def rust_pair(text: str, work: Path) -> tuple[dict | None, str]:
    crate, target = work / "crate", work / "target"
    shutil.rmtree(crate, ignore_errors=True)
    (crate / "src").mkdir(parents=True)
    bin_ = bool(MAIN_RE.search(text))
    src = crate / "src" / ("main.rs" if bin_ else "lib.rs")
    src.write_text(text)
    _set_edition(crate, "2015", bin_)
    if _cargo(["check", "-q"], crate, target).returncode:
        return None, "no_compile"
    steps: list[str] = []
    before = text
    for edition in ("2018", "2021"):
        if _cargo(["fix", "--edition", "--allow-no-vcs", "--allow-dirty", "-q"], crate, target).returncode:
            return None, f"fix_{edition}_failed"
        _set_edition(crate, edition, bin_)
    if _cargo(["fix", "--edition-idioms", "--allow-no-vcs", "--allow-dirty", "-q"], crate, target).returncode == 0:
        pass
    cur = src.read_text()
    stripped, n_try = _strip_rtry(cur)
    if n_try:
        src.write_text(stripped)
        if _cargo(["check", "-q"], crate, target).returncode:
            src.write_text(cur)
            n_try = 0
    lint_before = _clippy_lints(_cargo(["clippy"], crate, target))
    _cargo(["clippy", "--fix", "--allow-no-vcs", "--allow-dirty", "-q"], crate, target)
    if _cargo(["check", "-q"], crate, target).returncode:
        return None, "broken_after_fix"
    lint_after = _clippy_lints(_cargo(["clippy"], crate, target))
    after = src.read_text()
    if after.split() == before.split():
        return None, "unchanged"
    if after.count("r#try!") or _changed(before, after) > 150:
        return None, "too_messy"

    removed = lambda pat: len(re.findall(pat, before)) - len(re.findall(pat, after))  # noqa: E731
    added = lambda pat: len(re.findall(pat, after)) - len(re.findall(pat, before))  # noqa: E731
    if n_try or removed(r"\btry!\(") > 0:
        steps.append("replace `try!` with the `?` operator")
    if added(r"\bdyn\s") > 0:
        steps.append("use `dyn` for trait objects")
    if removed(r"\bextern crate\b") > 0:
        steps.append("drop `extern crate`")
    if added(r"\bcrate::") > 0:
        steps.append("use `crate::` paths")
    if added(r"'_\b") > 0:
        steps.append("use elided lifetimes")
    if added(r"\.\.=") > 0 and removed(r"\.\.\.") > 0:
        steps.append("use `..=` ranges")
    fixed = sorted(k for k in lint_before if lint_before[k] > lint_after.get(k, 0))
    lints = ", ".join(k.replace("_", " ") for k in fixed[:6])
    if steps:
        objective = "Migrate to Rust 2021: " + "; ".join(steps) + (f"; idiomatic fixes: {lints}" if lints else "")
    elif lints:
        objective = f"Make the code idiomatic Rust: {lints}"
    else:
        objective = "Migrate to the Rust 2021 edition"
    return {"objective": objective[:200], "legacy": before, "modern": after,
            "checks": steps + [f"clippy::{k}" for k in fixed]}, "kept"


# --------------------------------------------------------------------------- cpp

STD_HEADERS = set("""
algorithm any array atomic bit bitset cassert cctype cerrno cfloat charconv chrono cinttypes climits
clocale cmath codecvt complex condition_variable csetjmp csignal cstdarg cstddef cstdint cstdio cstdlib
cstring ctime cwchar cwctype deque exception execution filesystem forward_list fstream functional future
initializer_list iomanip ios iosfwd iostream istream iterator limits list locale map memory mutex new
numbers numeric optional ostream queue random ranges ratio regex scoped_allocator set shared_mutex
span sstream stack stdexcept streambuf string string_view system_error thread tuple type_traits typeindex
typeinfo unordered_map unordered_set utility valarray variant vector concepts compare source_location
assert.h ctype.h errno.h float.h limits.h locale.h math.h setjmp.h signal.h stdarg.h stddef.h stdint.h
stdio.h stdlib.h string.h time.h wchar.h inttypes.h stdbool.h strstream
""".split())
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.M)
# Purely stylistic or huge-diff checks: not what "modernize legacy C++" means.
CPP_SKIP = ("use-trailing-return-type", "use-nodiscard", "avoid-c-arrays", "use-designated-initializers",
            "use-std-print", "use-std-format", "use-constraints", "use-ranges", "use-integer-sign-comparison",
            "macro-to-enum", "pass-by-value", "return-braced-init-list", "avoid-bind", "use-std-numbers")
CPP_CHECKS = "-*,modernize-*," + ",".join(f"-modernize-{c}" for c in CPP_SKIP)
CHECK_RE = re.compile(r"\[(modernize-[\w-]+)\]")
CPP_PHRASE = {
    "use-nullptr": "use nullptr", "use-override": "use override", "loop-convert": "use range-based for loops",
    "use-auto": "use auto", "make-unique": "use std::make_unique", "make-shared": "use std::make_shared",
    "use-using": "use `using` aliases instead of typedef", "use-emplace": "use emplace_back",
    "use-equals-default": "use `= default`", "use-equals-delete": "use `= delete`",
    "redundant-void-arg": "drop redundant `(void)`", "deprecated-headers": "use C++ headers",
    "use-bool-literals": "use bool literals", "use-default-member-init": "use default member initializers",
    "raw-string-literal": "use raw string literals", "use-noexcept": "use noexcept",
    "concat-nested-namespaces": "concatenate nested namespaces", "use-starts-ends-with": "use starts_with",
    "replace-auto-ptr": "replace std::auto_ptr with std::unique_ptr", "use-emplace-back": "use emplace_back",
    "shrink-to-fit": "use shrink_to_fit", "use-transparent-functors": "use transparent functors",
    "min-max-use-initializer-list": "use std::min/max initializer lists", "type-traits": "use _t/_v type traits",
    "unary-static-assert": "use unary static_assert", "use-uncaught-exceptions": "use uncaught_exceptions",
    "replace-random-shuffle": "replace random_shuffle with shuffle", "use-scoped-lock": "use std::scoped_lock",
}


def cpp_candidate(text: str) -> bool:
    return all(q == "<" and name in STD_HEADERS for q, name in INCLUDE_RE.findall(text))


def _tidy(path: Path, checks: str, fix: bool, header: bool) -> subprocess.CompletedProcess:
    lang = "c++-header" if header else "c++"
    # --allow-no-checks: with checks "-*" (compile check only) clang-tidy otherwise refuses to run
    # and prints its usage, which contains no "error:" — a silent pass.
    cmd = ["clang-tidy", str(path), f"--checks={checks}", "--quiet", "--allow-no-checks"] + (["--fix"] if fix else [])
    return _run(cmd + ["--", f"-x{lang}", "-std=c++20", "-Wno-everything"], path.parent)


TYPEDEF_RE = re.compile(r"^\s*typedef\s+([^;{}()\[\]]+?)\s*\b(\w+)\s*;", re.M)
USING_RE = re.compile(r"^\s*using\s+(\w+)\s*=\s*([^;]+);", re.M)


def _norm_type(t: str) -> str:
    return re.sub(r"\s+", "", t)


def typedefs_preserved(before: str, after: str) -> bool:
    """Guard against modernize-use-using bugs (seen: ``typedef const T& X`` → ``using X = T&``).

    Every ``using X = T;`` that is new in ``after`` must come from ``typedef T X;`` with the same
    type; anything that cannot be matched that way (function-pointer typedefs …) is rejected.
    """
    old = {name: _norm_type(t) for t, name in TYPEDEF_RE.findall(before)}
    prior = {name for name, _ in USING_RE.findall(before)}
    for name, t in USING_RE.findall(after):
        if name in prior:
            continue
        if old.get(name) != _norm_type(t):
            return False
    return True


def compiles(path: Path, header: bool) -> bool:
    res = _tidy(path, "-*", False, header)
    if "USAGE:" in res.stdout + res.stderr:
        raise RuntimeError(f"clang-tidy did not run: {(res.stdout + res.stderr)[:200]}")
    return res.returncode == 0 and "error:" not in res.stdout + res.stderr


def cpp_pair(text: str, work: Path, header: bool) -> tuple[dict | None, str]:
    work.mkdir(parents=True, exist_ok=True)
    f = work / ("t.hpp" if header else "t.cpp")
    f.write_text(text)
    if not compiles(f, header):
        return None, "no_compile"
    res = _tidy(f, CPP_CHECKS, True, header)
    fired = sorted({m.removeprefix("modernize-") for m in CHECK_RE.findall(res.stdout)})
    after = f.read_text()
    if after.split() == text.split():
        return None, "unchanged"
    if not compiles(f, header):
        return None, "broken_after_fix"
    if _changed(text, after) > 150:
        return None, "too_messy"
    if not typedefs_preserved(text, after):
        return None, "bad_typedef_fix"
    phrases = [CPP_PHRASE.get(c, c.replace("-", " ")) for c in fired] or ["apply modernize fixes"]
    objective = "Modernize for C++11+: " + ", ".join(phrases)
    return {"objective": objective[:200], "legacy": text, "modern": after,
            "checks": [f"modernize-{c}" for c in fired]}, "kept"


# --------------------------------------------------------------------------- driver


def _job(lang: str, item: dict, slot: int) -> tuple[str, dict | None, str]:
    work = Path(tempfile.gettempdir()) / f"rebirth_toolpairs_{os.getpid()}"
    try:
        if lang == "rust":
            pair, why = rust_pair(item["text"], work)
        else:
            header = Path(item.get("path") or "x.cpp").suffix.lower() in {".h", ".hh", ".hpp", ".hxx"}
            pair, why = cpp_pair(item["text"], work, header)
    except subprocess.TimeoutExpired:
        pair, why = None, "timeout"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if pair:
        pair = {**pair, "language": lang, "source": "toolpair_rust" if lang == "rust" else "toolpair_clang_tidy",
                "repo": item.get("repo"), "commit": item.get("commit"), "path": item.get("path"),
                "license": item.get("license"), "tier": "tool"}
    return item["sha"], pair, why


def candidates(lang: str) -> list[dict]:
    """Unique old file versions (``legacy``) from the CommitPack pool + the expanded corpus."""
    check = rust_candidate if lang == "rust" else cpp_candidate
    srcs = [POOL / f"{lang}_commitpack.jsonl", ROOT / "data" / "datasets" / "expanded" / f"{lang}_migration.jsonl",
            POOL / f"{lang}_standalone.jsonl"]
    seen: set[str] = set()
    out: list[dict] = []
    for p in srcs:
        if not p.is_file():
            continue
        for line in p.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # pool still being appended to
                continue
            text = r.get("legacy") or ""
            sha = _sha(text)
            if sha in seen or not check(text):
                continue
            seen.add(sha)
            out.append({"sha": sha, "text": text, "repo": r.get("repo"), "commit": r.get("commit"),
                        "path": r.get("path"), "license": r.get("license")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=("rust", "cpp"), required=True)
    ap.add_argument("--limit", type=int, default=None, help="max candidate files to try this run")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    tool = "cargo" if args.lang == "rust" else "clang-tidy"
    if not shutil.which(tool, path=f"{CARGO_BIN}:{os.environ.get('PATH', '')}"):
        print(f"{tool} not found", file=sys.stderr)
        return 1
    out = POOL / f"{args.lang}_toolpairs.jsonl"
    done_path = POOL / f"{args.lang}_toolpairs.done"
    done = set(done_path.read_text().split()) if done_path.is_file() else set()
    todo = [c for c in candidates(args.lang) if c["sha"] not in done]
    print(f"{args.lang}: {len(todo)} candidate files ({len(done)} tried before)", flush=True)
    todo = todo[: args.limit] if args.limit else todo

    stats: Counter = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex, out.open("a") as fo, done_path.open("a") as fd:
        futs = [ex.submit(_job, args.lang, c, i % args.workers) for i, c in enumerate(todo)]
        for n, fut in enumerate(as_completed(futs), 1):
            sha, pair, why = fut.result()
            stats[why] += 1
            if pair:
                fo.write(json.dumps(pair, ensure_ascii=False) + "\n")
                fo.flush()
            fd.write(sha + "\n")
            if n % 200 == 0:
                fd.flush()
                print(f"  {n}/{len(todo)} {dict(stats)}", flush=True)
    print(json.dumps(dict(stats.most_common()), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
