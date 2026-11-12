"""Source discovery, ignore rules, test mapping, and import topo-order (Phase 6)."""

from __future__ import annotations

import fnmatch
import json
import os
import re
from pathlib import Path

from rebirth_ai.adapters.base import LanguageAdapter
from rebirth_ai.adapters.js_ts import SOURCE_SUFFIXES as JS_SOURCE_SUFFIXES
from rebirth_ai.adapters.js_ts import JsTsAdapter
from rebirth_ai.adapters.registry import adapter_id_for_suffix, get_adapter
from rebirth_ai.safety import file_too_large, is_secret_path

# Always skipped even if not in .gitignore
DENY_DIR_NAMES = frozenset(
    {
        "node_modules",
        ".git",
        "dist",
        "build",
        "coverage",
        ".venv",
        "venv",
        "data",
        "__pycache__",
        ".next",
        ".turbo",
        ".cache",
        "out",
        "target",
        "vendor",
    }
)

_IMPORT_SPEC = re.compile(
    r"""(?:from|import)\s+['"]([^'"]+)['"]|require\(\s*['"]([^'"]+)['"]\s*\)""",
    re.MULTILINE,
)


def parse_under_roots(under: str | list[str] | None) -> list[str]:
    """Parse ``src,lib,app`` or ``["src","lib"]`` into root segments (empty → project root)."""
    if under is None:
        return ["src"]
    if isinstance(under, list):
        parts = [p.strip().strip("/\\") for p in under if str(p).strip()]
        return parts or ["src"]
    parts = [p.strip().strip("/\\") for p in str(under).split(",") if p.strip()]
    return parts or ["src"]


def _load_gitignore_patterns(project_dir: Path) -> tuple[list[str], list[str]]:
    """Return (ignore_patterns, negation_patterns) from ``.gitignore``."""
    path = project_dir / ".gitignore"
    if not path.is_file():
        return [], []
    ignore: list[str] = []
    negate: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("!"):
            negate.append(line[1:])
            continue
        ignore.append(line)
    return ignore, negate


def _match_one_gitignore(rel_posix: str, pat: str) -> bool:
    name = Path(rel_posix).name
    directory_only = pat.endswith("/")
    p = pat.rstrip("/")
    if p.startswith("/"):
        p = p[1:]
    candidates = [rel_posix, name]
    if directory_only:
        parts = rel_posix.split("/")
        for i in range(len(parts)):
            prefix = "/".join(parts[: i + 1])
            if fnmatch.fnmatch(prefix, p) or fnmatch.fnmatch(parts[i], p):
                return True
    for cand in candidates:
        if fnmatch.fnmatch(cand, p):
            return True
        if fnmatch.fnmatch(cand, f"**/{p}"):
            return True
        if "**" in p and fnmatch.fnmatch(cand, p):
            return True
    return False


def _path_matches_gitignore(
    rel_posix: str,
    patterns: list[str],
    *,
    negations: list[str] | None = None,
) -> bool:
    """Best-effort gitignore match (supports *, **, dirs, and basic ``!`` negation)."""
    ignored = any(_match_one_gitignore(rel_posix, pat) for pat in patterns)
    if not ignored:
        return False
    for neg in negations or []:
        if _match_one_gitignore(rel_posix, neg):
            return False
    return True


def is_ignored_path(
    project_dir: Path,
    relative: Path | str,
    *,
    gitignore_patterns: list[str] | None = None,
    gitignore_negations: list[str] | None = None,
) -> bool:
    """True if path should be skipped (deny dirs or .gitignore)."""
    rel = Path(relative)
    if any(part in DENY_DIR_NAMES for part in rel.parts):
        return True
    if gitignore_patterns is None:
        patterns, negations = _load_gitignore_patterns(project_dir)
    else:
        patterns, negations = gitignore_patterns, gitignore_negations or []
    return _path_matches_gitignore(rel.as_posix(), patterns, negations=negations)


def discover_source_files(
    project_dir: Path,
    *,
    adapter: LanguageAdapter | None = None,
    under: str | list[str] = "src",
    honor_gitignore: bool = True,
) -> list[Path]:
    """Return project-relative source paths under one or more roots, deny/.gitignore aware."""
    adapter = adapter or JsTsAdapter()
    root = project_dir.resolve()
    roots = parse_under_roots(under)
    if honor_gitignore:
        patterns, negations = _load_gitignore_patterns(root)
    else:
        patterns, negations = [], []

    owned_suffixes = set(getattr(adapter, "SOURCE_SUFFIXES", JS_SOURCE_SUFFIXES))

    seen: set[str] = set()
    out: list[Path] = []
    for segment in roots:
        base = root / segment if segment else root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(root)
            except ValueError:
                continue
            if is_ignored_path(
                root,
                rel,
                gitignore_patterns=patterns,
                gitignore_negations=negations,
            ):
                continue
            if is_secret_path(rel):
                continue
            if path.suffix.lower() not in owned_suffixes:
                continue
            if not adapter.is_source_file(path):
                continue
            # Never treat test / spec / seed / declaration stubs as migration sources
            name_l = path.name.lower()
            if (
                ".test." in name_l
                or ".spec." in name_l
                or name_l.endswith(".d.ts")
                or name_l.endswith(".test-d.ts")
                or name_l in {"test.js", "tests.js", "benchmark.js"}
                or name_l.startswith("test_")
                or name_l.endswith("_test.py")
                or name_l == "conftest.py"
                or path.parent.name in {"test", "tests", "__tests__"}
                or any(
                    part in {"seeds", ".rebirth_seeds"} or part.endswith("_seeds")
                    for part in path.parts
                )
            ):
                continue
            if file_too_large(path):
                continue
            key = rel.as_posix()
            if key in seen:
                continue
            seen.add(key)
            out.append(rel)
    return out


def infer_test_path(project_dir: Path, relative_file: Path | str) -> str | None:
    """Map a source file to a sibling/ co-located / package.json / pytest test when present."""
    rel = Path(relative_file)
    stem = rel.stem
    # strip trailing .test / .spec from stem if someone passes a test file
    for suffix in (".test", ".spec"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    parent = rel.parent
    candidates: list[Path] = [
        Path("test") / f"{stem}.test.js",
        Path("test") / f"{stem}.test.ts",
        Path("test") / f"{stem}.spec.js",
        Path("test") / f"{stem}.spec.ts",
        Path("tests") / f"{stem}.test.js",
        Path("tests") / f"{stem}.test.ts",
        Path("tests") / f"{stem}.spec.js",
        Path("tests") / f"{stem}.spec.ts",
        Path("__tests__") / f"{stem}.test.js",
        Path("__tests__") / f"{stem}.test.ts",
        Path("__tests__") / f"{stem}.spec.js",
        parent / "__tests__" / f"{stem}.test.js",
        parent / "__tests__" / f"{stem}.test.ts",
        parent / "__tests__" / f"{stem}.spec.js",
        parent / f"{stem}.test.js",
        parent / f"{stem}.test.ts",
        parent / f"{stem}.spec.js",
        parent / f"{stem}.spec.ts",
        # Python / pytest
        Path("tests") / f"test_{stem}.py",
        Path("test") / f"test_{stem}.py",
        Path("tests") / f"{stem}_test.py",
        Path("test") / f"{stem}_test.py",
        parent / f"test_{stem}.py",
        parent / f"{stem}_test.py",
        parent / "tests" / f"test_{stem}.py",
        parent / "test" / f"test_{stem}.py",
        # Go
        parent / f"{stem}_test.go",
        # Java
        parent / f"{stem}Test.java",
        # C# / Swift / C++
        Path("MathOps.Tests") / f"{stem}Tests.cs",
        Path("MathOps.Tests") / "CalculatorTests.cs",
        Path("Tests") / f"{stem}Tests" / f"{stem}Tests.swift",
        Path("Tests") / "MathOpsTests" / "MathOpsTests.swift",
        Path("tests") / f"{stem}_test.cpp",
        Path("tests") / f"{stem}_test.cc",
        Path("test") / f"{stem}_test.cpp",
    ]
    if str(rel).replace("\\", "/").startswith("src/main/java/"):
        try:
            sub = rel.relative_to(Path("src/main/java"))
            candidates.append(
                Path("src/test/java") / sub.parent / f"{stem}Test.java"
            )
        except ValueError:
            pass
    # packages/foo/src/bar.js → packages/foo/test/bar.test.js (monorepo lite)
    if parent.name == "src" and len(parent.parts) >= 1:
        pkg_root = parent.parent
        for folder in ("test", "tests", "__tests__"):
            for ext in (".test.js", ".test.ts", ".spec.js", ".spec.ts"):
                candidates.append(pkg_root / folder / f"{stem}{ext}")
            candidates.append(pkg_root / folder / f"test_{stem}.py")
            candidates.append(pkg_root / folder / f"{stem}_test.py")
            candidates.append(pkg_root / folder / f"{stem}_test.go")
            candidates.append(pkg_root / folder / f"{stem}_test.cpp")
    for cand in candidates:
        if (project_dir / cand).is_file():
            return cand.as_posix()
    # Rust: #[cfg(test)] often lives in the same crate file
    if rel.suffix.lower() == ".rs" and (project_dir / "Cargo.toml").is_file():
        return rel.as_posix()
    # Package-level suite markers (when no dedicated test file was found)
    suf = rel.suffix.lower()
    marker_by_suffix = {
        ".go": "go.mod",
        ".java": "pom.xml",
        ".rs": "Cargo.toml",
        ".cpp": "CMakeLists.txt",
        ".cc": "CMakeLists.txt",
        ".cxx": "CMakeLists.txt",
        ".hpp": "CMakeLists.txt",
        ".h": "CMakeLists.txt",
        ".swift": "Package.swift",
        ".cs": None,
    }
    marker = marker_by_suffix.get(suf)
    if marker and (project_dir / marker).is_file():
        return marker
    if suf == ".cs" and (
        any(project_dir.glob("*.sln")) or any(project_dir.glob("*.csproj"))
    ):
        sln = next(iter(project_dir.glob("*.sln")), None)
        return sln.name if sln else "dotnet"

    # package.json "test" script fallback: only if a listed path matches this stem
    pkg = project_dir / "package.json"
    if pkg.is_file():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
            script = (data.get("scripts") or {}).get("test") or ""
        except (json.JSONDecodeError, OSError):
            script = ""
        if script:
            # Collect explicit test file paths from common runners
            hinted_files = re.findall(
                r"(?:^|\s)((?:test|tests|__tests__)[^\s]*\.(?:test|spec)\.[jt]sx?)",
                script,
            )
            if not hinted_files:
                hinted_files = re.findall(
                    r"(?:^|\s)([^\s]+(?:\.test|\.spec)\.[jt]sx?)",
                    script,
                )
            for raw in hinted_files:
                hinted = Path(raw.strip("\"'"))
                if hinted.stem in {stem, f"{stem}.test", f"{stem}.spec"} or hinted.stem.startswith(
                    f"{stem}."
                ):
                    if (project_dir / hinted).is_file():
                        return hinted.as_posix()
                # stem.test.js → stem match via Path.stem of "math.test" vs "math"
                if hinted.name.startswith(f"{stem}.") and (project_dir / hinted).is_file():
                    return hinted.as_posix()
    # Last resort: a test file that imports this very file (`test.js` → `require('./')`, root
    # `test_<package>.py` → `import <package>`). JS runs the project's own `npm test` (the test may
    # need mocha / ava / jest); Python runs that test file with pytest.
    if suf in JS_SOURCE_SUFFIXES or suf == ".py":
        hit = tests_importing(project_dir, rel)
        if hit:
            return (WHOLE_SUITE_JS if suf in JS_SOURCE_SUFFIXES and _npm_test_script(project_dir) else hit)
    return None


WHOLE_SUITE_JS = "package.json"  # test "path" meaning: the project's `npm test` (JsTsAdapter)
_TEST_DIRS = ("test", "tests", "__tests__", "spec")
_MAX_TEST_FILES = 300


def _npm_test_script(project_dir: Path) -> str:
    try:
        data = json.loads((project_dir / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    script = (data.get("scripts") or {}).get("test") or ""
    return "" if "no test specified" in script else script


def _candidate_test_files(project_dir: Path, python: bool) -> list[Path]:
    """Test files by name: ``test_*.py`` / ``*_test.py``; root ``test.js``, files under test dirs,
    ``*.test.*`` / ``*.spec.*``. Skips the deny-listed and hidden directories without entering them."""
    exts = {".py"} if python else JS_SOURCE_SUFFIXES
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(project_dir):
        dirnames[:] = sorted(d for d in dirnames if d not in DENY_DIR_NAMES and not d.startswith("."))
        here = Path(dirpath).relative_to(project_dir)
        in_test_dir = any(part in _TEST_DIRS for part in here.parts)
        for name in sorted(filenames):
            path = Path(name)
            if path.suffix not in exts:
                continue
            stem = name[: -len(path.suffix)]
            if python:
                keep = stem.startswith("test_") or stem.endswith("_test") or stem in {"test", "tests"}
            else:
                keep = stem in {"test", "tests"} or in_test_dir or stem.endswith((".test", ".spec"))
            if keep:
                out.append(here / name)
                if len(out) >= _MAX_TEST_FILES:
                    return out
    return out


def _js_resolve(project_dir: Path, test_rel: Path, spec: str) -> str | None:
    """Project-relative source a JS import specifier in ``test_rel`` points at (``./``, ``..``, files)."""
    if spec.startswith("."):
        base = (project_dir / test_rel.parent / spec).resolve()
    else:  # the package's own name: `require('is-number')` in its tests
        try:
            name = json.loads((project_dir / "package.json").read_text(encoding="utf-8")).get("name")
        except (OSError, ValueError):
            return None
        if spec != name:
            return None
        base = project_dir.resolve()
    try:
        rel = base.relative_to(project_dir.resolve())
    except ValueError:
        return None
    if (project_dir / rel).is_file():
        return rel.as_posix()
    if (project_dir / rel).is_dir():
        if rel == Path("."):
            try:
                main = json.loads((project_dir / "package.json").read_text(encoding="utf-8")).get("main")
            except (OSError, ValueError):
                main = None
            if isinstance(main, str) and (project_dir / main).is_file():
                return Path(main).as_posix().removeprefix("./")
        for idx in ("index.js", "index.ts", "index.mjs", "index.cjs"):
            if (project_dir / rel / idx).is_file():
                return (rel / idx).as_posix()
        return None
    for ext in (".js", ".ts", ".mjs", ".cjs", ".jsx", ".tsx"):
        if (project_dir / f"{rel.as_posix()}{ext}").is_file():
            return f"{rel.as_posix()}{ext}"
    return None


def _py_module(rel: Path) -> str:
    """``inflection/__init__.py`` → ``inflection``; ``src/pkg/mod.py`` → ``pkg.mod``."""
    parts = list(rel.with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _py_reexported(project_dir: Path, rel: Path, parent: str, last: str) -> bool:
    """Whether the package's ``__init__.py`` imports from this module (``from .mod import *``)."""
    init = project_dir / rel.parent / "__init__.py"
    try:
        text = init.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    p, m = re.escape(parent), re.escape(last)
    return bool(re.search(
        rf"^\s*from\s+(?:\.|{p}\.){m}\s+import\b|^\s*from\s+(?:\.|{p})\s+import\s+[^\n]*\b{m}\b|"
        rf"^\s*import\s+{p}\.{m}\b", text, re.M))


def _suite_dir(tests: list[Path]) -> str:
    """Test folder that holds all of ``tests`` (``tests``, ``test``, …), else the project root."""
    tops = {next((part for part in t.parts[:-1] if part in _TEST_DIRS), None) for t in tests}
    if len(tops) == 1 and None not in tops:
        t = tests[0]
        return Path(*t.parts[: t.parts.index(tops.pop()) + 1]).as_posix()
    return "."


def tests_importing(project_dir: Path, relative_file: Path | str) -> str | None:
    """The test that covers ``relative_file``, found by what the test files import.

    JS: the first test file that imports this file. Python: the test files that import the module
    itself, else those that reach it through its package (``from pkg import name`` with
    ``pkg/__init__.py`` re-exporting it): one file → that file, several → their test folder.
    """
    rel = Path(relative_file)
    python = rel.suffix.lower() == ".py"
    direct: list[Path] = []
    via_package: list[Path] = []
    if python:
        mod = _py_module(rel)
        if not mod:
            return None
        parent, _, last = mod.rpartition(".")
        pats = [rf"^\s*import\s+{re.escape(mod)}\b", rf"^\s*from\s+{re.escape(mod)}\s+import\b"]
        pkg_pat = None
        if parent:
            if _py_reexported(project_dir, rel, parent, last):
                pkg_pat = re.compile(rf"^\s*(?:import|from)\s+{re.escape(parent)}\b", re.M)
            else:  # `from pkg import mod`: the module itself
                pats.append(rf"^\s*from\s+{re.escape(parent)}\s+import\s+[^\n]*\b{re.escape(last)}\b")
        pat = re.compile("|".join(pats), re.M)
    for test_rel in _candidate_test_files(project_dir, python):
        if test_rel == rel:
            continue
        try:
            text = (project_dir / test_rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if python:
            if pat.search(text):
                direct.append(test_rel)
            elif pkg_pat is not None and pkg_pat.search(text):
                via_package.append(test_rel)
            continue
        for m in _IMPORT_SPEC.finditer(text):
            spec = m.group(1) or m.group(2)
            if _js_resolve(project_dir, test_rel, spec) == rel.as_posix():
                return test_rel.as_posix()
    hits = direct or via_package  # a test of the module itself beats tests of the whole package
    if len(hits) == 1:
        return hits[0].as_posix()
    return _suite_dir(hits) if hits else None


def resolve_local_import(
    project_dir: Path,
    from_file: Path | str,
    specifier: str,
) -> str | None:
    """Resolve a relative import specifier to a project-relative source path, if it exists."""
    spec = specifier.strip()
    if not (spec.startswith("./") or spec.startswith("../")):
        return None
    base = (project_dir / Path(from_file).parent / spec).resolve()
    root = project_dir.resolve()
    try:
        rel = base.relative_to(root)
    except ValueError:
        return None

    candidates: list[Path] = []
    if base.suffix:
        candidates.append(rel)
    else:
        for ext in (".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".py"):
            candidates.append(rel.with_suffix(ext))
        candidates.append(rel / "index.js")
        candidates.append(rel / "index.ts")
        candidates.append(rel / "__init__.py")

    for cand in candidates:
        if (project_dir / cand).is_file():
            return cand.as_posix()
    return None


def extract_local_imports(project_dir: Path, relative_file: Path | str) -> list[str]:
    """Return project-relative local import targets from a source file.

    JS/TS and Python only: those are the languages whose imports name files. The others (a Go or Java package,
    a Rust crate, a C# or Swift module) compile their files together, so they get no edges and keep path order.
    """
    path = project_dir / relative_file
    if not path.is_file() or adapter_id_for_suffix(path) not in {"js_ts", "python"}:
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    # Prefer Tree-sitter import chunks when available
    specs: list[str] = []
    try:
        if path.suffix.lower() == ".py":
            adapter: LanguageAdapter = get_adapter("python")
        else:
            adapter = JsTsAdapter()
        for chunk in adapter.parse(path, source=text):
            if chunk.kind == "import" and chunk.name not in {"import", "<anonymous>"}:
                specs.append(chunk.name)
    except Exception:
        pass
    if not specs and path.suffix.lower() != ".py":
        for match in _IMPORT_SPEC.finditer(text):
            specs.append(match.group(1) or match.group(2) or "")
    if not specs and path.suffix.lower() == ".py":
        # from .foo import / import foo as relative helpers
        for match in re.finditer(
            r"^(?:from\s+(\.\S+)\s+import|import\s+(\.\S+))",
            text,
            re.MULTILINE,
        ):
            specs.append(match.group(1) or match.group(2) or "")

    resolved: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        target = resolve_local_import(project_dir, relative_file, spec)
        if target and target not in seen:
            seen.add(target)
            resolved.append(target)
    return resolved


def topological_migrate_order(
    project_dir: Path,
    files: list[Path | str],
) -> list[Path]:
    """
    Order sources so dependencies (imports) migrate before dependents.

    Only edges among ``files`` count. Cycles break by stable path order.
    """
    rels = [Path(f).as_posix() for f in files]
    index = {r: Path(r) for r in rels}
    # edges: dep -> importers wait on dep; Kahn: migrate deps first
    dependents: dict[str, set[str]] = {r: set() for r in rels}
    indegree: dict[str, int] = {r: 0 for r in rels}

    for rel in rels:
        for dep in extract_local_imports(project_dir, rel):
            if dep not in index or dep == rel:
                continue
            # rel depends on dep → dep must come first
            if rel not in dependents[dep]:
                dependents[dep].add(rel)
                indegree[rel] += 1

    # Kahn: start with indegree 0, stable by path
    ready = sorted([r for r, d in indegree.items() if d == 0])
    ordered: list[str] = []
    while ready:
        node = ready.pop(0)
        ordered.append(node)
        for child in sorted(dependents[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()

    # Cycle leftovers
    remaining = [r for r in rels if r not in ordered]
    ordered.extend(sorted(remaining))
    return [index[r] for r in ordered]
