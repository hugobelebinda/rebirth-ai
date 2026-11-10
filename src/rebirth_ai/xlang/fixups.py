"""Deterministic clean-up of a translated file before it is verified (no model involved).

Only mechanical fixes a language's own tooling would also make — never behavior changes:

- Go: add missing standard-library imports and drop unused ones (what ``goimports`` does for the
  stdlib; Go refuses to compile either). In the stage-2 baseline 42 of 55 Go compile errors were
  ``undefined: sort / strings / math / strconv``; whole programs often import ``strings`` and then
  never use it.

Used by ``rebirth migrate --to`` (translate node) and the translation eval.
"""

from __future__ import annotations

import re

from rebirth_ai.xlang.languages import XLang

# package name as used in code → import path (standard library only)
GO_STDLIB = {
    "bufio": "bufio",
    "bytes": "bytes",
    "errors": "errors",
    "fmt": "fmt",
    "io": "io",
    "math": "math",
    "os": "os",
    "reflect": "reflect",
    "regexp": "regexp",
    "sort": "sort",
    "strconv": "strconv",
    "strings": "strings",
    "sync": "sync",
    "time": "time",
    "unicode": "unicode",
    "slices": "slices",
    "maps": "maps",
    "cmp": "cmp",
    "bits": "math/bits",
    "big": "math/big",
    "cmplx": "math/cmplx",
    "rand": "math/rand",
    "heap": "container/heap",
    "list": "container/list",
    "utf8": "unicode/utf8",
}

_GO_IMPORT_BLOCK = re.compile(r"^import\s*\(\s*\n(.*?)^\)", re.MULTILINE | re.DOTALL)
_GO_IMPORT_LINE = re.compile(r'^import\s+((?:[\w.]+\s+)?"[^"]+")', re.MULTILINE)
_GO_SPEC = re.compile(r'^\s*(?:([\w.]+)\s+)?"([^"]+)"')


def go_code_only(code: str) -> str:
    """The Go source with comments and string / rune literals blanked (same length, same lines)."""
    out, i, n = [], 0, len(code)
    while i < n:
        c = code[i]
        if code.startswith("//", i):
            j = code.find("\n", i)
            j = n if j < 0 else j
        elif code.startswith("/*", i):
            j = code.find("*/", i + 2)
            j = n if j < 0 else j + 2
        elif c in "\"'":
            j = i + 1
            while j < n and code[j] != c and code[j] != "\n":
                j += 2 if code[j] == "\\" else 1
            j = min(j + 1, n)
        elif c == "`":
            j = code.find("`", i + 1)
            j = n if j < 0 else j + 1
        else:
            out.append(c)
            i += 1
            continue
        out.append(re.sub(r"[^\n]", " ", code[i:j]))
        i = j
    return "".join(out)


def _go_imported_names(code: str) -> set[str]:
    specs = [ln for m in _GO_IMPORT_BLOCK.finditer(code) for ln in m.group(1).split("\n")]
    specs += [m.group(1) for m in _GO_IMPORT_LINE.finditer(code)]
    names = set()
    for spec in specs:
        m = _GO_SPEC.match(spec)
        if m:
            names.add(m.group(1) or m.group(2).rsplit("/", 1)[-1])
    return names


def _go_declared(name: str, code: str) -> bool:
    """``name`` is a local identifier (variable, parameter, type, function), not a package."""
    n = re.escape(name)
    return bool(
        re.search(rf"(?<![\w.]){n}\s*(?:,\s*[\w]+\s*)*:=", code)  # name := / name, x :=
        or re.search(rf",\s*{n}\s*:=", code)  # x, name :=
        or re.search(rf"\b(?:var|const|type|func)\s+{n}\b", code)
        or re.search(rf"[(,]\s*{n}\s+[\w\[\]*.]", code)  # parameter: (name T / , name T
    )


def _go_package_name(path: str) -> str:
    parts = path.split("/")
    return parts[-2] if len(parts) > 1 and re.fullmatch(r"v\d+", parts[-1]) else parts[-1]  # math/rand/v2 → rand


def drop_unused_go_imports(code: str) -> str:
    """Remove standard-library imports whose package is never referenced (``_`` / ``.`` imports kept)."""
    plain = go_code_only(code)
    out, removed = [], False
    in_block = False
    for ln in code.split("\n"):
        stripped = ln.strip()
        if re.match(r"^import\s*\($", stripped):
            in_block = True
        elif in_block and stripped.startswith(")"):
            in_block = False
        spec = stripped[len("import"):].strip() if stripped.startswith("import ") and '"' in stripped else (stripped if in_block else "")
        m = _GO_SPEC.match(spec) if spec else None
        if m and m.group(1) not in ("_", ".") and "." not in m.group(2).split("/")[0]:
            name = m.group(1) or _go_package_name(m.group(2))
            if not re.search(rf"(?<![\w.]){re.escape(name)}\.\w", plain):
                removed = True
                continue
        out.append(ln)
    return "\n".join(out) if removed else code


def fix_go_imports(code: str) -> str:
    """Add ``import "<path>"`` for standard-library packages the code uses but does not import,
    then drop standard-library imports it never uses."""
    return drop_unused_go_imports(_add_missing_go_imports(code))


def _add_missing_go_imports(code: str) -> str:
    plain = go_code_only(code)
    used = {name for name in GO_STDLIB if re.search(rf"(?<![\w.]){name}\.[A-Z]\w*", plain)}
    missing = sorted(used - _go_imported_names(code) - {n for n in used if _go_declared(n, plain)})
    if not missing:
        return code
    specs = "".join(f'\t"{GO_STDLIB[n]}"\n' for n in missing)
    block = _GO_IMPORT_BLOCK.search(plain)
    if block:  # same offsets in code and plain
        at = code.index("\n", block.start()) + 1
        return code[:at] + specs + code[at:]
    pkg = re.search(r"^package\s+\w+[^\n]*\n", plain, re.MULTILINE)
    at = pkg.end() if pkg else 0
    return code[:at] + "\nimport (\n" + specs + ")\n" + ("" if pkg else "\n") + code[at:]


def apply_fixups(target: XLang, code: str) -> str:
    """All deterministic fix-ups for a translated ``target`` file (identity for most languages)."""
    if target.id == "go":
        return fix_go_imports(code)
    return code
