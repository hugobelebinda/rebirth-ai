"""Phase 17 stage 1: cross-language migration (rebirth migrate --to) and the translation eval harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rebirth_ai.xlang import graph as G
from rebirth_ai.xlang.languages import get_lang, lang_for_path, program_spec, target_relpath
from rebirth_ai.xlang.multipl_e import (
    assemble,
    fix_cs_equality,
    problem_key,
    remove_main,
    split_prompt,
    strip_c_comments,
    strip_python_docstrings,
    unwrap_class,
)
from rebirth_ai.xlang.prompts import build_translation_prompt, extract_code
from rebirth_ai.xlang.translator import Completion
from rebirth_ai.xlang.verify import VerifyResult, keep_inline_tests, status_for
from rebirth_ai.xlang.verify import tests_ran as ran_any_tests

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- languages


def test_language_ids_aliases_and_suffixes() -> None:
    assert get_lang("c++").id == "cpp" and get_lang("C#").id == "csharp" and get_lang("js_ts").id == "typescript"
    assert lang_for_path("src/a.tsx").id == "typescript" and lang_for_path("x.h").id == "cpp"
    assert lang_for_path("README.md") is None
    with pytest.raises(ValueError):
        get_lang("cobol")


def test_target_file_naming() -> None:
    assert target_relpath("src/math_ops.py", get_lang("go")) == "src/math_ops.go"
    assert target_relpath("lib/MathOps.java", get_lang("rust")) == "lib/math_ops.rs"
    assert target_relpath("src/math_ops.py", get_lang("csharp")) == "src/MathOps.cs"
    java = "package x;\n\npublic final class Calculator {\n}\n"
    assert target_relpath("src/math_ops.py", get_lang("java"), java) == "src/Calculator.java"


def test_program_script_marks_compile_failures() -> None:
    s = program_spec(get_lang("rust")).script(15)
    assert "|| exit 90" in s and "timeout -k 2 15 ./prog" in s
    assert [status_for(c) for c in (0, 90, 124, 1)] == ["pass", "compile_error", "timeout", "fail"]


# --------------------------------------------------------------------------- prompts / extraction


def test_extract_code_prefers_target_fence() -> None:
    raw = "Here:\n```python\nprint(1)\n```\nand the Go one:\n```go\npackage x\n```\n"
    assert extract_code(raw, get_lang("go")) == "package x\n"
    assert extract_code("```rust\nfn f() {}\n", get_lang("rust")) == "fn f() {}\n"  # truncated reply
    assert extract_code("fn f() {}", get_lang("rust")) == "fn f() {}\n"


def test_retry_prompt_carries_previous_draft_and_logs() -> None:
    p = build_translation_prompt(source=get_lang("python"), target=get_lang("go"), source_code="def f(): pass",
                                 source_path="a.py", target_path="a.go", target_tests="func TestF(t *testing.T) {}",
                                 previous="package a\nfunc F( {", error_class="CompileError", logs="syntax error")
    assert "func TestF" in p and "package a\nfunc F( {" in p and "syntax error" in p and "CompileError" in p


# --------------------------------------------------------------------------- MultiPL-E assembly


def test_problem_keys_and_prompt_split() -> None:
    assert problem_key("HumanEval_12_longest") == problem_key("CPP/12") == problem_key("HumanEval/12") == "HumanEval_12"
    header, sig = split_prompt("#include<assert.h>\n// statement\n// more\nlong f(long x) {\n")
    assert header == "#include<assert.h>\n" and sig == "long f(long x) {"


def test_comment_and_docstring_stripping() -> None:
    src = 'int f() { // c1\n  /* c2 */\n  return "http://x" == s; }\n'
    assert strip_c_comments(src) == 'int f() {\n  return "http://x" == s; }\n'
    py = 'def f(x):\n    """Problem statement."""\n    return x\n'
    assert '"""' not in strip_python_docstrings(py) and "return x" in strip_python_docstrings(py)


def test_assembly_cleanup() -> None:
    rust = 'fn f() -> i32 { 1 }\n\nfn main() {\n    println!("{}", f());\n}\n'
    assert "main" not in remove_main(get_lang("rust"), rust)
    assert unwrap_class("class Solution {\n    public static int f() { return 1; }\n}\n").strip() == "public static int f() { return 1; }"
    go_target = {"prompt": 'package f_test\n\nimport (\n    "testing"\n    "fmt"\n)\n\n// doc\nfunc f(x int) int {\n',
                 "tests": "func TestF(t *testing.T) { _ = fmt.Sprint(f(1)) }\n"}
    prog = assemble(get_lang("go"), go_target, 'package main\n\nimport "math"\n\nfunc f(x int) int { return int(math.Abs(float64(x))) }\n')
    assert '"math"' in prog and "package main" not in prog and prog.count("package ") == 1 and "// doc" not in prog


def test_csharp_list_asserts_compare_by_value() -> None:
    # MultiPL-E compares C# lists with .Equals (reference equality): no correct answer could pass
    line = '    Debug.Assert(F((new List<string>(new string[]{(string)"a)b"}))).Equals((new List<string>())));'
    assert fix_cs_equality(line) == ('    Debug.Assert(RebirthEq.Same(F((new List<string>(new string[]{(string)"a)b"}))), '
                                     '(new List<string>())));')
    for keep in ("    Debug.Assert(F(1L) == (true));", "    Debug.Assert(F(G(1).Equals(2)));"):
        assert fix_cs_equality(keep) == keep
    prog = assemble(get_lang("csharp"), {"prompt": "using System;\nclass Problem {\n    public static long F(long x) {\n",
                                         "tests": "    }\n    public static void Main(string[] args) {\n    Debug.Assert(F(1L).Equals(1L));\n    }\n}\n"},
                    "public static long F(long x) { return x; }")
    assert "RebirthEq.Same(F(1L), 1L)" in prog and "static class RebirthEq" in prog


# --------------------------------------------------------------------------- target-project guards


def test_rust_project_tests_always_win() -> None:
    orig = (ROOT / "fixtures/rust_sample/src/lib.rs").read_text()
    draft = "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn t() {}\n}\n"
    out = keep_inline_tests(get_lang("rust"), orig, draft)
    assert "fn t()" not in out and "test_summarize" in out


def test_zero_tests_is_not_a_pass() -> None:
    rust = get_lang("rust")
    assert ran_any_tests(rust, "test result: ok. 0 passed; 0 failed") is False
    assert ran_any_tests(rust, "test result: ok. 3 passed; 0 failed\ntest result: ok. 0 passed") is True
    assert ran_any_tests(get_lang("go"), "?   \trebirth/x\t[no test files]") is False
    assert ran_any_tests(get_lang("java"), "") is None


def test_model_sees_tests_not_the_target_implementation() -> None:
    view = G.target_tests_view(get_lang("rust"), ROOT / "fixtures/rust_sample", "src/lib.rs")
    assert view.startswith("#[cfg(test)]") and "pub fn add" not in view
    cpp = G.target_tests_view(get_lang("cpp"), ROOT / "fixtures/cpp_sample", "tests/math_ops_test.cpp")
    assert "std::string summarize(int a, int b);" in cpp  # the header the tests include


# --------------------------------------------------------------------------- the loop (no Docker, no model)


def test_translation_loop_retries_with_target_errors(monkeypatch) -> None:
    replies = iter(["```go\npackage mathops\nfunc Add(a, b int) int { return a + }\n```",
                    "```go\npackage mathops\n\nfunc Add(a, b int) int { return a + b }\n```"])
    prompts: list[str] = []

    class Stub:
        def complete(self, prompt, *, max_new_tokens, system=None, temperature=0.1):
            prompts.append(prompt)
            return Completion(raw=next(replies), model="stub")

    def fake_verify(*, target, code, target_rel, target_project=None, target_test=None):
        ok = "a + b" in code
        return VerifyResult(ok, 0 if ok else 1, "" if ok else "src/mathops.go:2:37: syntax error", "target_tests", "go test", 0.1)

    monkeypatch.setattr(G, "get_translator", lambda backend=None: Stub())
    monkeypatch.setattr(G, "verify_translation", fake_verify)
    state = G.initial_translation_state(project_dir=ROOT / "fixtures/python_sample", source_rel="src/math_ops.py",
                                        target_lang="go", target_project=ROOT / "fixtures/go_sample",
                                        target_file="src/mathops.go", target_test="src/mathops_test.go")
    final = G.build_translation_graph().invoke(state)
    assert final["status"] == "success" and final["iteration_count"] == 2
    assert "CompileError" in final["trace"][2] and "syntax error" in prompts[1] and "func TestAdd" in prompts[0]


def test_same_language_is_rejected() -> None:
    with pytest.raises(ValueError):
        G.initial_translation_state(project_dir=ROOT / "fixtures/python_sample", source_rel="src/math_ops.py",
                                    target_lang="python")


def test_oracle_covers_every_target() -> None:
    oracle = json.loads((ROOT / "fixtures/xlang_oracle/humaneval.json").read_text())
    assert set(oracle) == {"python", "javascript", "typescript", "go", "java", "csharp", "rust", "cpp", "swift"}
    assert all(any(k.endswith("!wrong") for k in v) for v in oracle.values())
