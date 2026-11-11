"""Language adapter registry + project detection (Phases 8–10)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rebirth_ai.adapters.base import LanguageAdapter
from rebirth_ai.adapters.cpp import CppAdapter
from rebirth_ai.adapters.csharp import CSharpAdapter
from rebirth_ai.adapters.go_lang import GoAdapter
from rebirth_ai.adapters.java_lang import JavaAdapter
from rebirth_ai.adapters.js_ts import JsTsAdapter
from rebirth_ai.adapters.python import PythonAdapter
from rebirth_ai.adapters.rust import RustAdapter
from rebirth_ai.adapters.swift import SwiftAdapter

AdapterFactory = Callable[[], LanguageAdapter]

_REGISTRY: dict[str, AdapterFactory] = {
    "js_ts": JsTsAdapter,
    "python": PythonAdapter,
    "go": GoAdapter,
    "java": JavaAdapter,
    "csharp": CSharpAdapter,
    "rust": RustAdapter,
    "cpp": CppAdapter,
    "swift": SwiftAdapter,
}

# Exact root filenames (and simple patterns via endswith / glob-ish checks below).
_DETECT_MARKERS: list[tuple[str, tuple[str, ...]]] = [
    ("js_ts", ("package.json",)),
    ("python", ("pyproject.toml", "requirements.txt", "setup.py", "Pipfile")),
    ("go", ("go.mod",)),
    ("java", ("pom.xml",)),
    ("rust", ("Cargo.toml",)),
    ("cpp", ("CMakeLists.txt",)),
    ("swift", ("Package.swift",)),
]

_SUFFIX_TO_ADAPTER = {
    ".js": "js_ts",
    ".jsx": "js_ts",
    ".mjs": "js_ts",
    ".cjs": "js_ts",
    ".ts": "js_ts",
    ".tsx": "js_ts",
    ".py": "python",
    ".go": "go",
    ".java": "java",
    ".cs": "csharp",
    ".rs": "rust",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".h": "cpp",
    ".swift": "swift",
}

_ALIASES = {
    "js": "js_ts",
    "javascript": "js_ts",
    "typescript": "js_ts",
    "ts": "js_ts",
    "py": "python",
    "c#": "csharp",
    "cs": "csharp",
    "c++": "cpp",
    "cplusplus": "cpp",
}


def adapter_for_suffix(suffix: str) -> str | None:
    """Adapter id for a file extension (``.py`` → ``python``), None if unknown."""
    return _SUFFIX_TO_ADAPTER.get(suffix.lower())


def list_adapter_ids() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_adapter(lang: str | None = None) -> LanguageAdapter:
    """Return an adapter instance by id (default ``js_ts``)."""
    key = (lang or "js_ts").strip().lower().replace("-", "_")
    key = _ALIASES.get(key, key)
    factory = _REGISTRY.get(key)
    if factory is None:
        known = ", ".join(list_adapter_ids())
        raise ValueError(f"Unknown adapter {lang!r}. Known: {known}")
    return factory()


def detect_adapter_id(project_dir: Path) -> str | None:
    """Infer adapter id from project root markers, or None if ambiguous/unknown."""
    root = project_dir.resolve()
    hits: list[str] = []
    for adapter_id, markers in _DETECT_MARKERS:
        if any((root / m).exists() for m in markers):
            hits.append(adapter_id)
    # C# / Gradle without exact single-file markers
    if any(root.glob("*.sln")) or any(root.glob("*.csproj")) or any(root.rglob("*.csproj")):
        # Prefer csharp if a solution/project is at or near root
        if (any(root.glob("*.sln")) or any(root.glob("*.csproj"))) and "csharp" not in hits:
            hits.append("csharp")
    if any(root.glob("build.gradle")) or any(root.glob("build.gradle.kts")):
        if "java" not in hits:
            hits.append("java")

    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        if "js_ts" in hits and (root / "package.json").is_file():
            if "python" in hits and _suffix_weight(root, {".py"}) > _suffix_weight(
                root, {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}
            ):
                return "python"
            return "js_ts"
        # Prefer non-js marker when package.json absent from conflict set
        for preferred in ("rust", "go", "java", "csharp", "cpp", "swift", "python"):
            if preferred in hits:
                return preferred
        return hits[0]
    return _by_source_files(root)


_SKIP_DIRS = {"node_modules", ".git", ".venv", "venv", "__pycache__", "data", "target", "build", "vendor"}


def _by_source_files(root: Path, *, limit: int = 400) -> str | None:
    """No marker file (e.g. a Go repo from before modules): the language most source files use."""
    counts: dict[str, int] = {}
    seen = 0
    for path in root.rglob("*"):
        if seen >= limit:
            break
        if not path.is_file() or set(path.relative_to(root).parts[:-1]) & _SKIP_DIRS:
            continue
        aid = _SUFFIX_TO_ADAPTER.get(path.suffix.lower())
        if aid:
            counts[aid] = counts.get(aid, 0) + 1
            seen += 1
    return max(counts, key=lambda a: counts[a]) if counts else None


def _suffix_weight(root: Path, suffixes: set[str], *, limit: int = 80) -> int:
    n = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.parts)
        if parts & {"node_modules", ".git", ".venv", "venv", "__pycache__", "data", "target", "build"}:
            continue
        if path.suffix.lower() in suffixes:
            n += 1
        if n >= limit:
            break
    return n


def get_adapter_for_project(project_dir: Path, *, lang: str | None = None) -> LanguageAdapter:
    """Resolve adapter from ``--lang`` override or project markers (fallback js_ts)."""
    if lang:
        return get_adapter(lang)
    detected = detect_adapter_id(project_dir)
    return get_adapter(detected or "js_ts")


def language_label_for_file(path: Path, *, adapter_id: str | None = None) -> str:
    """Prompt/language string for generation (not the adapter registry id)."""
    if adapter_id:
        labels = {
            "js_ts": None,  # fall through to suffix
            "python": "python",
            "go": "go",
            "java": "java",
            "csharp": "csharp",
            "rust": "rust",
            "cpp": "cpp",
            "swift": "swift",
        }
        mapped = labels.get(adapter_id)
        if mapped:
            return mapped
    suf = path.suffix.lower()
    if suf == ".py":
        return "python"
    if suf in {".ts", ".tsx"}:
        return "typescript"
    if suf == ".go":
        return "go"
    if suf == ".java":
        return "java"
    if suf == ".cs":
        return "csharp"
    if suf == ".rs":
        return "rust"
    if suf in {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h"}:
        return "cpp"
    if suf == ".swift":
        return "swift"
    return "javascript"


def adapter_id_for_suffix(path: Path) -> str | None:
    return _SUFFIX_TO_ADAPTER.get(path.suffix.lower())


def adapter_summaries() -> list[dict[str, object]]:
    """Rows for ``rebirth adapters``."""
    rows: list[dict[str, object]] = []
    for adapter_id in list_adapter_ids():
        adapter = get_adapter(adapter_id)
        spec = adapter.sandbox_spec()
        extensions = sorted(getattr(adapter, "SOURCE_SUFFIXES", set()) or [])
        rows.append(
            {
                "id": adapter_id,
                "name": adapter.name,
                "extensions": extensions,
                "image": spec.image,
                "test_command": spec.test_command,
                "dockerfile_dir": spec.dockerfile_dir,
            }
        )
    return rows
