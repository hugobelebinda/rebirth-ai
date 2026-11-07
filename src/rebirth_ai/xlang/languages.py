"""Languages for cross-language migration: ids, file naming, and how to check code in each sandbox.

The 8 product languages map to 9 ids here because JS and TS are both in the ``js_ts`` adapter:
``javascript`` and ``typescript`` share its sandbox (Node 22, which runs ``.ts`` by stripping
types — no type check yet). All commands run inside the target language's Docker sandbox
(``adapters/<lang>.py``), offline, from the sandbox work dir.

Two kinds of command:
- :func:`check_spec`   compile-only check of a translated file (no tests available)
- :func:`program_spec` compile + run a self-contained test program (eval harness); the compile
  step exits with :data:`COMPILE_FAILED` so a build error is told apart from a failing test
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

COMPILE_FAILED = 90  # run.sh exit code when the compile step fails


@dataclass(frozen=True)
class XLang:
    id: str
    adapter_id: str  # sandbox / adapter registry id
    label: str  # name used in prompts
    ext: str
    fence: str  # markdown fence tag models use
    file_case: str  # snake | pascal | keep — target file-name convention


LANGS: dict[str, XLang] = {
    x.id: x
    for x in (
        XLang("python", "python", "Python", ".py", "python", "snake"),
        XLang("javascript", "js_ts", "JavaScript", ".js", "javascript", "keep"),
        XLang("typescript", "js_ts", "TypeScript", ".ts", "typescript", "keep"),
        XLang("go", "go", "Go", ".go", "go", "snake"),
        XLang("java", "java", "Java", ".java", "java", "pascal"),
        XLang("csharp", "csharp", "C#", ".cs", "csharp", "pascal"),
        XLang("rust", "rust", "Rust", ".rs", "rust", "snake"),
        XLang("cpp", "cpp", "C++", ".cpp", "cpp", "keep"),
        XLang("swift", "swift", "Swift", ".swift", "swift", "pascal"),
    )
}

# The 8 product languages; js_ts is TypeScript as a target (what new JS-ecosystem code is written in)
# and JavaScript as a source (what legacy code and the HumanEval-X solutions are written in).
PRODUCT_TARGETS = ("python", "typescript", "go", "java", "csharp", "rust", "cpp", "swift")
PRODUCT_SOURCES = ("python", "javascript", "go", "java", "csharp", "rust", "cpp", "swift")

ALIASES = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "js_ts": "typescript",
    "golang": "go",
    "c#": "csharp",
    "cs": "csharp",
    "rs": "rust",
    "c++": "cpp",
    "cxx": "cpp",
}

SUFFIX_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
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

HEADER_SUFFIXES = {".hpp", ".hh", ".h"}


def get_lang(name: str) -> XLang:
    key = name.strip().lower().replace("-", "_")
    key = ALIASES.get(key, key)
    if key not in LANGS:
        raise ValueError(f"Unknown language {name!r}. Known: {', '.join(LANGS)} (aliases: {', '.join(ALIASES)})")
    return LANGS[key]


def lang_for_path(path: str | PurePosixPath) -> XLang | None:
    key = SUFFIX_LANG.get(PurePosixPath(str(path)).suffix.lower())
    return LANGS[key] if key else None


# --------------------------------------------------------------------------- file naming

_JAVA_PUBLIC_TYPE = re.compile(
    r"^\s*public\s+(?:(?:final|abstract|sealed|static)\s+)*(?:class|interface|enum|record)\s+([A-Za-z_]\w*)",
    re.MULTILINE,
)
_JAVA_ANY_TYPE = re.compile(r"^(?:(?:final|abstract)\s+)*(?:class|interface|enum|record)\s+([A-Za-z_]\w*)", re.MULTILINE)


def snake_case(stem: str) -> str:
    s = re.sub(r"[-\s.]+", "_", stem)
    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", s)
    return re.sub(r"_+", "_", s).lower().strip("_") or "module"


def pascal_case(stem: str) -> str:
    parts = [p for p in re.split(r"[_\-\s.]+", stem) if p]
    if len(parts) == 1:
        return parts[0][:1].upper() + parts[0][1:]
    return "".join(p[:1].upper() + p[1:].lower() for p in parts) or "Module"


def target_relpath(source_rel: str, target: XLang, code: str = "") -> str:
    """Where the translation of ``source_rel`` goes, relative to the output dir.

    Same folders, the target extension, the target language's file-name convention. Java uses the
    name of the generated public class (javac requires it).
    """
    src = PurePosixPath(source_rel)
    stem = src.stem
    if target.id == "java":
        m = _JAVA_PUBLIC_TYPE.search(code) or _JAVA_ANY_TYPE.search(code)
        name = m.group(1) if m else pascal_case(stem)
    elif target.file_case == "snake":
        name = snake_case(stem)
    elif target.file_case == "pascal":
        name = pascal_case(stem)
    else:
        name = stem
    return str(src.with_name(name + target.ext))


# --------------------------------------------------------------------------- sandbox commands

_CSPROJ = (
    '<Project Sdk="Microsoft.NET.Sdk">\n  <PropertyGroup>\n    <OutputType>{kind}</OutputType>\n'
    "    <TargetFramework>net8.0</TargetFramework>\n    <AssemblyName>Rebirth</AssemblyName>\n"
    "    <Nullable>disable</Nullable>\n    <ImplicitUsings>{implicit}</ImplicitUsings>\n"
    "    <TreatWarningsAsErrors>false</TreatWarningsAsErrors>\n  </PropertyGroup>\n</Project>\n"
)
_GO_MOD = "module rebirth.check\n\ngo 1.22\n"


def check_spec(lang: XLang, rel: str, code: str = "") -> tuple[dict[str, str], str]:
    """Compile-only check for one translated file: (extra scaffold files, shell command)."""
    q = _sh_quote(rel)
    if lang.id == "python":
        return {}, f"python3 -m py_compile {q}"
    if lang.id == "javascript":
        return {}, f"node --check {q}"
    if lang.id == "typescript":
        # `node --check` skips .ts files (a broken file passes), so strip the types with Node's own
        # stripper, then syntax-check the JS. Parses only — no type check (no tsc in the sandbox).
        strip = (
            "node -e 'const m=require(\"node:module\"),fs=require(\"node:fs\");"
            "fs.writeFileSync(\"/tmp/rebirth_check.mjs\",m.stripTypeScriptTypes(fs.readFileSync(process.argv[1],\"utf8\")))'"
        )
        return {}, f"{strip} {q} && node --check /tmp/rebirth_check.mjs"
    if lang.id == "go":
        files = {"go.mod": _GO_MOD}
        if re.search(r"^package\s+main\b", code, re.MULTILINE) and not re.search(r"^func\s+main\s*\(", code, re.MULTILINE):
            files[str(PurePosixPath(rel).with_name("zz_rebirth_main.go"))] = "package main\n\nfunc main() {}\n"
        return files, "go build ./..."
    if lang.id == "java":
        return {}, f"javac -d /tmp/rebirth_classes {q}"
    if lang.id == "csharp":
        return {"RebirthCheck.csproj": _CSPROJ.format(kind="Library", implicit="enable")}, "dotnet build -nologo -v q -clp:NoSummary"
    if lang.id == "rust":
        return {}, f"rustc --edition 2021 --crate-type lib --emit=metadata -o /tmp/rebirth_check.rmeta {q}"
    if lang.id == "cpp":
        header = "-x c++-header " if PurePosixPath(rel).suffix.lower() in HEADER_SUFFIXES else ""
        return {}, f"g++ -std=c++20 -fsyntax-only {header}{q}"
    if lang.id == "swift":
        return {}, f"swiftc -typecheck {q}"
    raise ValueError(lang.id)


@dataclass(frozen=True)
class ProgramSpec:
    filename: str  # the program's source file inside its own directory
    scaffold: dict[str, str]
    compile: str  # "" when interpreted
    run: str

    def script(self, timeout_sec: int, compile_timeout_sec: int = 180) -> str:
        """run.sh body: compile (exit COMPILE_FAILED on error), then run under a timeout."""
        lines = ["set -u"]
        if self.compile:
            lines.append(f"timeout -k 2 {compile_timeout_sec} {self.compile} || exit {COMPILE_FAILED}")
        lines.append(f"exec timeout -k 2 {timeout_sec} {self.run}")
        return "\n".join(lines) + "\n"


def program_spec(lang: XLang) -> ProgramSpec:
    """How to build and run a self-contained test program (exit 0 = all asserts passed)."""
    # Interpreted languages get a syntax check as the "compile" step, so a parse error is reported
    # as compile_error, not as a failing test.
    if lang.id == "python":
        return ProgramSpec("prog.py", {}, "python3 -m py_compile prog.py", "python3 prog.py")
    if lang.id == "javascript":
        return ProgramSpec("prog.js", {}, "node --check prog.js", "node prog.js")
    if lang.id == "typescript":
        _, check = check_spec(lang, "prog.ts")
        return ProgramSpec("prog.ts", {}, f"sh -c {_sh_quote(check)}", "node --experimental-strip-types --no-warnings prog.ts")
    if lang.id == "go":
        return ProgramSpec("prog_test.go", {"go.mod": _GO_MOD}, "go test -c -o prog.test .", "./prog.test")
    if lang.id == "java":
        return ProgramSpec("Problem.java", {}, "javac -d out -sourcepath . Problem.java", "java -ea -cp out Problem")
    if lang.id == "csharp":
        return ProgramSpec(
            "Program.cs",
            {"Rebirth.csproj": _CSPROJ.format(kind="Exe", implicit="disable")},
            "dotnet build -nologo -v q -clp:NoSummary -o out",
            "dotnet out/Rebirth.dll",
        )
    if lang.id == "rust":
        return ProgramSpec("prog.rs", {}, "rustc --edition 2021 -o prog prog.rs", "./prog")
    if lang.id == "cpp":
        return ProgramSpec("prog.cpp", {}, "g++ -std=c++17 -o prog prog.cpp", "./prog")
    if lang.id == "swift":
        return ProgramSpec("main.swift", {}, "swiftc -o prog main.swift", "./prog")
    raise ValueError(lang.id)


def _sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"
