"""Source discovery, ignore rules, test mapping, and import topo-order (Phase 6)."""

from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path

from rebirth_ai.adapters.js_ts import JsTsAdapter, SOURCE_SUFFIXES
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
    adapter: JsTsAdapter | None = None,
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
            if path.suffix.lower() not in SOURCE_SUFFIXES:
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
    """Map a source file to a sibling/ co-located / package.json test when present."""
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
    ]
    # packages/foo/src/bar.js → packages/foo/test/bar.test.js (monorepo lite)
    if parent.name == "src" and len(parent.parts) >= 1:
        pkg_root = parent.parent
        for folder in ("test", "tests", "__tests__"):
            for ext in (".test.js", ".test.ts", ".spec.js", ".spec.ts"):
                candidates.append(pkg_root / folder / f"{stem}{ext}")
    for cand in candidates:
        if (project_dir / cand).is_file():
            return cand.as_posix()

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
    return None


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
        for ext in (".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs"):
            candidates.append(rel.with_suffix(ext))
        candidates.append(rel / "index.js")
        candidates.append(rel / "index.ts")

    for cand in candidates:
        if (project_dir / cand).is_file():
            return cand.as_posix()
    return None


def extract_local_imports(project_dir: Path, relative_file: Path | str) -> list[str]:
    """Return project-relative local import targets from a source file."""
    path = project_dir / relative_file
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    # Prefer Tree-sitter import chunks when available
    specs: list[str] = []
    try:
        adapter = JsTsAdapter()
        for chunk in adapter.parse(path, source=text):
            if chunk.kind == "import" and chunk.name not in {"import", "<anonymous>"}:
                specs.append(chunk.name)
    except Exception:  # noqa: BLE001
        pass
    if not specs:
        for match in _IMPORT_SPEC.finditer(text):
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
