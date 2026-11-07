"""Benchmark problems for the translation eval (Phase 17 stage 1–2).

Tests: MultiPL-E (MIT; HumanEval translated to 18+ languages, unit tests per language) for the 8
non-Python targets, OpenAI HumanEval (MIT) for Python. Sources: HumanEval canonical solutions
(Python) and HumanEval-X (Apache-2.0) solutions in C++, Java, JavaScript, Go. Rust / C# / Swift /
TypeScript have no reference solutions; they become sources once verified translations exist.

A translation is judged like MultiPL-E judges a completion: the target program = the target
prompt's header (imports, ``class Problem {``) + the model's function + the target tests, compiled
and run in the target sandbox; exit 0 = every assert passed.
"""

from __future__ import annotations

import ast
import re

from rebirth_ai.xlang.languages import XLang

# MultiPL-E config suffix per target id (Python uses openai_humaneval directly)
MULTIPLE_CONFIG = {
    "cpp": "cpp",
    "csharp": "cs",
    "go": "go",
    "java": "java",
    "javascript": "js",
    "rust": "rs",
    "swift": "swift",
    "typescript": "ts",
}
HUMANEVAL_X = {"cpp": "cpp", "go": "go", "java": "java", "javascript": "js"}  # source languages with solutions

_COMMENT = re.compile(r"^\s*(//|/\*|\*|#(?!include))")


def problem_key(name_or_task: str) -> str:
    """``HumanEval_12_longest`` / ``HumanEval/12`` / ``CPP/12`` → ``HumanEval_12``."""
    m = re.search(r"(\d+)", name_or_task)
    if not m:
        raise ValueError(name_or_task)
    return f"HumanEval_{int(m.group(1))}"


# --------------------------------------------------------------------------- target side


def split_prompt(prompt: str) -> tuple[str, str]:
    """MultiPL-E prompt → (header, signature line). Drops the doc comment between them.

    The prompt always ends with the open signature (``fn f(…) -> T {``); the comment block right
    above it is the problem statement, which a *translation* task must not see.
    """
    lines = prompt.rstrip().split("\n")
    sig = lines[-1]
    j = len(lines) - 1
    while j > 0 and _COMMENT.match(lines[j - 1]):
        j -= 1
    return "\n".join(lines[:j]).rstrip() + "\n", sig


def signature_stub(target: XLang, sig: str) -> str:
    """The signature shown to the model: ``fn f(…) -> T {`` → ``fn f(…) -> T {\n    ...\n}``."""
    return sig if target.id == "python" else f"{sig.strip()}\n    ...\n}}"


def python_target_parts(prompt: str, entry_point: str) -> tuple[str, str]:
    """HumanEval prompt → (header: imports + helper functions, signature line of the entry point)."""
    lines = prompt.split("\n")
    idx = next(i for i, ln in enumerate(lines) if re.match(rf"def\s+{re.escape(entry_point)}\s*\(", ln))
    sig = lines[idx]
    # Multi-line signatures: join until the line ending with ':'
    k = idx
    while not lines[k].rstrip().endswith(":") and k + 1 < len(lines):
        k += 1
        sig += "\n" + lines[k]
    return "\n".join(lines[:idx]).rstrip() + "\n", sig


# --------------------------------------------------------------------------- source side


def strip_python_docstrings(code: str) -> str:
    """Remove function / class / module docstrings (the problem statement) from Python source."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree) + "\n"


def strip_c_comments(code: str) -> str:
    """Remove ``//`` and ``/* */`` comments (C++, Java, JS, Go, …); string / char / backtick literals
    are kept. Lines that held only a comment are dropped; blank lines that were blank stay."""
    out: list[str] = []
    i, n = 0, len(code)
    while i < n:
        c = code[i]
        if c in "\"'`":
            j = i + 1
            while j < n and code[j] != c:
                if code[j] == "\\" and c != "`":
                    j += 1
                elif code[j] == "\n" and c != "`":
                    break  # unterminated char / string on this line (e.g. a lone apostrophe)
                j += 1
            out.append(code[i : j + 1])
            i = j + 1
        elif code.startswith("//", i):
            j = code.find("\n", i)
            i = n if j < 0 else j
            out.append("\x00")
        elif code.startswith("/*", i):
            j = code.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append("\x00")
        else:
            out.append(c)
            i += 1
    lines = []
    for ln in "".join(out).split("\n"):
        if "\x00" in ln:
            ln = ln.replace("\x00", "").rstrip()
            if not ln.strip():
                continue
        lines.append(ln)
    return "\n".join(lines)


# --------------------------------------------------------------------------- assembly

_MAIN_RE = {
    "rust": re.compile(r"^\s*(?:pub\s+)?fn\s+main\s*\(", re.MULTILINE),
    "cpp": re.compile(r"^\s*int\s+main\s*\(", re.MULTILINE),
    "go": re.compile(r"^\s*func\s+main\s*\(", re.MULTILINE),
    "java": re.compile(r"^\s*(?:public\s+)?static\s+void\s+main\s*\(", re.MULTILINE),
    "csharp": re.compile(r"^\s*(?:public\s+|private\s+)?static\s+(?:void|int)\s+Main\s*\(", re.MULTILINE),
}
_CLASS_RE = re.compile(r"^\s*(?:public\s+|internal\s+)?(?:final\s+|static\s+)?class\s+\w+[^{]*\{", re.MULTILINE)


def _block_end(code: str, open_brace: int) -> int:
    """Index just past the brace that closes ``code[open_brace]`` (strings / chars skipped)."""
    depth, i, n = 0, open_brace, len(code)
    while i < n:
        c = code[i]
        if c in "\"'":
            q, i = c, i + 1
            while i < n and code[i] != q:
                i += 2 if code[i] == "\\" else 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def remove_main(target: XLang, code: str) -> str:
    rx = _MAIN_RE.get(target.id)
    if rx is None:
        if target.id == "python":
            m = re.search(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:.*", code, re.MULTILINE | re.DOTALL)
            return code[: m.start()] if m else code
        return code
    m = rx.search(code)
    if not m:
        return code
    brace = code.find("{", m.end())
    if brace < 0:
        return code
    return code[: m.start()] + code[_block_end(code, brace):]


def unwrap_class(code: str) -> str:
    """Java / C#: the model sometimes wraps the method in a class — keep only the members."""
    m = _CLASS_RE.search(code)
    if not m:
        return code
    end = _block_end(code, m.end() - 1)
    inner = code[m.end() : end - 1]
    return code[: m.start()] + inner + code[end:]


def _split_lines(code: str, rx: re.Pattern) -> tuple[list[str], str]:
    keep, moved = [], []
    for ln in code.split("\n"):
        (moved if rx.match(ln) else keep).append(ln)
    return moved, "\n".join(keep)


_GO_IMPORT_BLOCK = re.compile(r"^import\s*\(\s*\n(.*?)^\)\s*$", re.MULTILINE | re.DOTALL)
_GO_IMPORT_LINE = re.compile(r'^import\s+((?:\w+\s+)?"[^"]+")\s*$', re.MULTILINE)


def _go_merge(header: str, code: str) -> tuple[str, str]:
    imports: list[str] = []
    for m in _GO_IMPORT_BLOCK.finditer(code):
        imports += [x.strip() for x in m.group(1).split("\n") if x.strip()]
    code = _GO_IMPORT_BLOCK.sub("", code)
    imports += [m.group(1) for m in _GO_IMPORT_LINE.finditer(code)]
    code = _GO_IMPORT_LINE.sub("", code)
    code = re.sub(r"^package\s+\w+\s*$", "", code, flags=re.MULTILINE)
    have = set(re.findall(r'"[^"]+"', header))
    new = [i for i in dict.fromkeys(imports) if re.search(r'"[^"]+"', i).group(0) not in have]
    if new:
        header = re.sub(r"(import\s*\(\s*\n)", lambda m: m.group(1) + "".join(f"    {i}\n" for i in new), header, count=1)
    return header, code


JAVATUPLES_SHIM = {
    "org/javatuples/Pair.java": """package org.javatuples;
import java.util.Objects;
public final class Pair<A, B> {
    private final A a; private final B b;
    public Pair(A a, B b) { this.a = a; this.b = b; }
    public static <A, B> Pair<A, B> with(A a, B b) { return new Pair<>(a, b); }
    public A getValue0() { return a; }
    public B getValue1() { return b; }
    @Override public boolean equals(Object o) { return o instanceof Pair<?, ?> p && Objects.equals(a, p.a) && Objects.equals(b, p.b); }
    @Override public int hashCode() { return Objects.hash(a, b); }
    @Override public String toString() { return "[" + a + ", " + b + "]"; }
}
""",
    "org/javatuples/Triplet.java": """package org.javatuples;
import java.util.Objects;
public final class Triplet<A, B, C> {
    private final A a; private final B b; private final C c;
    public Triplet(A a, B b, C c) { this.a = a; this.b = b; this.c = c; }
    public static <A, B, C> Triplet<A, B, C> with(A a, B b, C c) { return new Triplet<>(a, b, c); }
    public A getValue0() { return a; }
    public B getValue1() { return b; }
    public C getValue2() { return c; }
    @Override public boolean equals(Object o) { return o instanceof Triplet<?, ?, ?> t && Objects.equals(a, t.a) && Objects.equals(b, t.b) && Objects.equals(c, t.c); }
    @Override public int hashCode() { return Objects.hash(a, b, c); }
    @Override public String toString() { return "[" + a + ", " + b + ", " + c + "]"; }
}
""",
}


# Upstream test bugs (no correct solution can pass) — scored as "broken_test", left out of pass rates.
BROKEN_TESTS = {
    ("go", "HumanEval_87"): "the test passes []interface{} values where the signature takes [][]int — never compiles",
}

# MultiPL-E's C# tests compare results with .Equals — reference equality for List / Dictionary, so
# every list-returning problem (45 of 158) fails even with a correct answer. assemble() rewrites
# ``Debug.Assert(A.Equals(B));`` to this structural comparison (same result for strings / tuples).
CS_EQUALITY_HELPER = """
static class RebirthEq {
    public static bool Same(object a, object b) {
        if (a is System.Collections.IDictionary da && b is System.Collections.IDictionary db) {
            if (da.Count != db.Count) return false;
            foreach (System.Collections.DictionaryEntry e in da)
                if (!db.Contains(e.Key) || !Same(e.Value, db[e.Key])) return false;
            return true;
        }
        if (a is System.Collections.IEnumerable ea && b is System.Collections.IEnumerable eb && !(a is string) && !(b is string)) {
            var ia = ea.GetEnumerator();
            var ib = eb.GetEnumerator();
            while (true) {
                bool ma = ia.MoveNext(), mb = ib.MoveNext();
                if (ma != mb) return false;
                if (!ma) return true;
                if (!Same(ia.Current, ib.Current)) return false;
            }
        }
        return object.Equals(a, b);
    }
}
"""
_CS_ASSERT = re.compile(r"^(\s*)Debug\.Assert\((.*)\);\s*$")


def _scan_depths(expr: str):
    """Yield (index, depth) for every character outside string / char literals."""
    depth, i = 0, 0
    while i < len(expr):
        c = expr[i]
        if c in "\"'":
            j = i + 1
            while j < len(expr) and expr[j] != c:
                j += 2 if expr[j] == "\\" else 1
            i = j + 1
            continue
        yield i, depth
        depth += (c in "([{") - (c in ")]}")
        i += 1


def fix_cs_equality(tests: str) -> str:
    """``Debug.Assert(A.Equals(B));`` → ``Debug.Assert(RebirthEq.Same(A, B));`` (top-level ``.Equals`` only)."""
    out = []
    for ln in tests.split("\n"):
        m = _CS_ASSERT.match(ln)
        if m:
            inner = m.group(2)
            k = next((i for i, d in _scan_depths(inner) if d == 0 and inner.startswith(".Equals(", i)), -1)
            if k > 0 and inner.endswith(")"):
                rhs = inner[k + len(".Equals(") : -1]
                depths = [d for _, d in _scan_depths(rhs + ")")]
                if depths and min(depths) >= 0 and depths[-1] == 0:  # the last ")" closes .Equals(
                    ln = f"{m.group(1)}Debug.Assert(RebirthEq.Same({inner[:k]}, {rhs}));"
        out.append(ln)
    return "\n".join(out)


def assemble(target: XLang, problem_target: dict, code: str) -> str:
    """Full test program for one translated function (see module docstring)."""
    code = remove_main(target, code).strip("\n")
    if target.id == "python":
        header, _ = python_target_parts(problem_target["prompt"], problem_target["entry_point"])
        return (
            "from typing import *\nimport math\n" + header + "\n" + code + "\n\n"
            + problem_target["test"] + f"\n\ncheck({problem_target['entry_point']})\n"
        )
    header, _ = split_prompt(problem_target["prompt"])
    tests = problem_target["tests"]
    if tests.lstrip().startswith("}"):  # closes the function the prompt opened; the model closed it
        tests = tests.lstrip()[1:]
    if target.id in ("java", "csharp"):
        code = unwrap_class(code)
        rx = re.compile(r"^\s*import\s+[\w.*]+\s*;\s*$") if target.id == "java" else re.compile(r"^\s*using\s+[\w.]+\s*;\s*$")
        moved, code = _split_lines(code, rx)
        if moved:
            header = "\n".join(m.strip() for m in moved) + "\n" + header
    elif target.id == "go":
        header, code = _go_merge(header, code)
    if target.id == "csharp":
        tests = fix_cs_equality(tests) + CS_EQUALITY_HELPER
    return header + "\n" + code.strip("\n") + "\n" + tests


def program_files(target: XLang) -> dict[str, str]:
    """Extra files a target program needs next to it (Java: the javatuples shim)."""
    return dict(JAVATUPLES_SHIM) if target.id == "java" else {}
