"""Which libraries a source file pulls in — a cheap, regex-based fingerprint for drift detection.

Names are normalized to the package level the training data can be compared on (``os.path`` →
``os``, ``java.util.List`` → ``java.util``, ``@scope/pkg/sub`` → ``@scope/pkg``, ``std::io::Read`` →
``std::io``). Relative imports (``./x``, ``crate::``, ``super::``, quoted ``#include "x.h"``) are
project-local and skipped. Not a parser: good enough to see "real code uses libraries the
adapters never saw".
"""

from __future__ import annotations

import re

_PY = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import|import\s+([A-Za-z_][\w.]*(?:\s*,\s*[A-Za-z_][\w.]*)*))", re.M)
_JS = re.compile(
    r"""(?:^\s*import\s+(?:[\w*{}\s,$]+\s+from\s+)?|\brequire\s*\(\s*|\bimport\s*\(\s*)['"]([^'"]+)['"]""", re.M
)
_GO_BLOCK = re.compile(r"^\s*import\s*\(([^)]*)\)", re.M | re.S)
_GO_ONE = re.compile(r'^\s*import\s+(?:[\w.]+\s+)?"([^"]+)"', re.M)
_GO_PATH = re.compile(r'"([^"]+)"')
_JAVA = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)(?:\.\*)?\s*;", re.M)
_CS = re.compile(r"^\s*(?:global\s+)?using\s+(?:static\s+)?(?:\w+\s*=\s*)?([A-Z][\w.]*)\s*;", re.M)
_RUST = re.compile(r"^\s*(?:pub\s+)?use\s+([\w:]+)", re.M)
_RUST_CRATE = re.compile(r"^\s*extern\s+crate\s+(\w+)", re.M)
_CPP = re.compile(r"^\s*#\s*include\s*<([^>]+)>", re.M)
_SWIFT = re.compile(r"^\s*(?:@testable\s+)?import\s+(?:(?:class|struct|enum|protocol|func|var|let|typealias)\s+)?(\w+)", re.M)

LANG_ALIASES = {
    "js": "js_ts", "javascript": "js_ts", "typescript": "js_ts", "ts": "js_ts", "js_ts": "js_ts",
    "python": "python", "py": "python", "go": "go", "golang": "go", "java": "java",
    "csharp": "csharp", "c#": "csharp", "cs": "csharp", "rust": "rust", "rs": "rust",
    "cpp": "cpp", "c++": "cpp", "swift": "swift",
}


def family(lang: str | None) -> str | None:
    """Corpus language id for any language name the tool uses (TypeScript and JavaScript → js_ts)."""
    return LANG_ALIASES.get((lang or "").strip().lower())


def _js_name(spec: str) -> str | None:
    if spec.startswith((".", "/")):
        return None
    spec = spec.removeprefix("node:")
    parts = spec.split("/")
    return "/".join(parts[:2]) if spec.startswith("@") else parts[0]


def extract_imports(code: str, lang: str | None) -> list[str]:
    """Sorted, de-duplicated library names imported by ``code`` (empty for unknown languages)."""
    fam = family(lang)
    out: set[str] = set()
    if not code or fam is None:
        return []
    if fam == "python":
        for frm, imp in _PY.findall(code):
            for name in ([frm] if frm else [n.strip() for n in imp.split(",")]):
                if name and not name.startswith("."):
                    out.add(name.split(".")[0])
    elif fam == "js_ts":
        out.update(n for n in (_js_name(s) for s in _JS.findall(code)) if n)
    elif fam == "go":
        for block in _GO_BLOCK.findall(code):
            out.update(_GO_PATH.findall(block))
        out.update(_GO_ONE.findall(code))
    elif fam == "java":
        for name in _JAVA.findall(code):
            parts = name.split(".")
            # java.util.List → java.util; org.junit.jupiter.api.Test → org.junit (vendor level)
            out.add(".".join(parts[:2]) if len(parts) > 2 else name)
    elif fam == "csharp":
        out.update(_CS.findall(code))
    elif fam == "rust":
        for path in _RUST.findall(code):
            parts = [p for p in path.split("::") if p]
            if parts and parts[0] not in {"crate", "self", "super"}:
                out.add("::".join(parts[:2]) if parts[0] in {"std", "core", "alloc"} else parts[0])
        out.update(_RUST_CRATE.findall(code))
    elif fam == "cpp":
        out.update(_CPP.findall(code))
    elif fam == "swift":
        out.update(_SWIFT.findall(code))
    return sorted(out)
