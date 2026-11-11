"""Phase 17 stage 1: cross-language migration (rebirth migrate --to) and the translation eval harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rebirth_ai.xlang import graph as G
from rebirth_ai.xlang.fixups import fix_go_imports
from rebirth_ai.xlang.languages import (
    get_lang,
    java_classes,
    lang_for_path,
    program_spec,
    standalone_spec,
    target_relpath,
)
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


def test_complete_programs_build_and_run_like_a_judge() -> None:
    java = 'import java.util.*;\npublic class Main {\n    static class FastReader { String s = "}"; }\n    public static void main(String[] a) {}\n}\n'
    assert java_classes(java) == ("Main", "Main")
    assert java_classes("public class Sol {}\nclass B { public static void main(String[] x) {} }\n") == ("Sol", "B")
    spec = standalone_spec(get_lang("java"), java)
    assert spec.filename == "Main.java" and spec.run.endswith("-cp out Main")
    assert "-O2" in standalone_spec(get_lang("cpp")).compile and "-O" in standalone_spec(get_lang("rust")).compile
    assert standalone_spec(get_lang("swift"), "@main\nstruct App { static func main() {} }\n").filename == "prog.swift"


# --------------------------------------------------------------------------- prompts / extraction


def test_extract_code_prefers_target_fence() -> None:
    raw = "Here:\n```python\nprint(1)\n```\nand the Go one:\n```go\npackage x\n```\n"
    assert extract_code(raw, get_lang("go")) == "package x\n"
    assert extract_code("```rust\nfn f() {}\n", get_lang("rust")) == "fn f() {}\n"  # truncated reply
    assert extract_code("fn f() {}", get_lang("rust")) == "fn f() {}\n"


def test_program_prompt_asks_for_same_output() -> None:
    p = build_translation_prompt(source=get_lang("cpp"), target=get_lang("go"), source_code="int main() {}",
                                 source_path="main.cpp", target_path="main.go", is_program=True)
    assert "complete program" in p and "exactly the same output" in p
    plain = build_translation_prompt(source=get_lang("cpp"), target=get_lang("go"), source_code="int f();",
                                     source_path="a.cpp", target_path="a.go")
    assert "complete program" not in plain


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


def test_go_fixup_adds_only_missing_stdlib_imports() -> None:
    prog = 'package f_test\n\nimport (\n    "testing"\n    "fmt"\n)\n\nfunc f(a []int) string {\n    sort.Ints(a)\n    return strings.Repeat("x", 2) + fmt.Sprint(a)\n}\n'
    out = fix_go_imports(prog)
    assert '\t"sort"\n' in out and '\t"strings"\n' in out and out.count('"fmt"') == 1
    assert fix_go_imports(out) == out  # idempotent
    # not imported when it is a local name, only mentioned in a string / comment, or a field
    for body in ('func f(list []int) int { return list.Len() }',
                 'func f() string { return "sort.Ints" } // math.Abs',
                 'func f(t T) int { return t.heap.Len() }'):
        code = "package main\n\n" + body + "\n"
        assert fix_go_imports(code) == code
    single = 'package main\n\nimport "fmt"\n\nfunc main() { fmt.Println(math.Sqrt(2), bits.OnesCount(3)) }\n'
    fixed = fix_go_imports(single)
    assert '"math"' in fixed and '"math/bits"' in fixed and fixed.index('"math"') < fixed.index("func main")
    # unused stdlib imports are dropped (Go will not compile them); blank / dot / non-stdlib imports stay
    unused = 'package main\n\nimport (\n\t"fmt"\n\t"strings"\n\t_ "embed"\n\t"github.com/x/y"\n)\n\nfunc main() { fmt.Println("strings.Repeat") }\n'
    kept = fix_go_imports(unused)
    assert '"strings"' not in kept and '_ "embed"' in kept and '"github.com/x/y"' in kept and '"fmt"' in kept


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


def _training_module(name: str):
    import importlib.util
    import sys

    sys.path.insert(0, str(ROOT / "training"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "training" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage3_leak_guard_matches_div1_div2_copies() -> None:
    g = _training_module("gen_xlang_pairs")
    assert g.title_key("1566_A. Median Maximization") == g.title_key("1567_C.  Median  maximization") == "median maximization"
    assert g.statement_key("You are given\n an array.") == g.statement_key("you are given an array.")
    assert g.error_class_for("compile_error", "") == "CompileError"
    assert g.error_class_for("fail", "test 000: wrong answer\n...") == "WrongAnswer"
    assert g.error_class_for("fail", "Traceback ... IndexError") == "RuntimeError"
    assert g.error_class_for("timeout", "") == "Timeout"


# ── translation backend: `migrate --to` (auto) serves the packaged cross-language adapter ────────


@pytest.fixture
def ollama_tags(monkeypatch, tmp_path):
    """Fake Ollama tag list (None = server unreachable); fresh translator cache per test; empty routing."""
    from rebirth_ai.generate import ollama_client
    from rebirth_ai.xlang import translator as T

    state = {"tags": []}

    def list_models(self):
        if state["tags"] is None:
            raise ConnectionError("ollama down")
        return list(state["tags"])

    monkeypatch.setattr(ollama_client.OllamaClient, "list_models", list_models)
    monkeypatch.delenv("REBIRTH_XLANG_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setenv("REBIRTH_ROUTING", str(tmp_path / "routing.yaml"))
    monkeypatch.setattr(T, "_CACHE", {})
    return state


def test_auto_backend_serves_the_installed_xlang_adapter(ollama_tags) -> None:
    from rebirth_ai.xlang.translator import DEFAULT_XLANG_MODEL, Translator

    ollama_tags["tags"] = ["qwen2.5-coder:14b-instruct-q6_K", DEFAULT_XLANG_MODEL]
    assert Translator("auto").model_name == f"ollama:{DEFAULT_XLANG_MODEL}"
    assert Translator("routed").model_name == f"ollama:{DEFAULT_XLANG_MODEL}"
    # explicit `ollama` stays the plain 14B: the zero-shot baseline the evals compare against
    assert Translator("ollama").model_name == "ollama:qwen2.5-coder:14b-instruct-q6_K"


def test_auto_backend_follows_the_translation_route(ollama_tags) -> None:
    from rebirth_ai import routing
    from rebirth_ai.xlang.translator import Translator

    routing.set_translation_route({"model": "rebirth-xlang:custom", "tag": "rebirth-xlang:14b-v9", "version": "9"})
    ollama_tags["tags"] = ["qwen2.5-coder:14b-instruct-q6_K", "rebirth-xlang:14b", "rebirth-xlang:custom"]
    assert Translator("auto").model_name == "ollama:rebirth-xlang:custom"


@pytest.mark.parametrize("tags", [["qwen2.5-coder:14b-instruct-q6_K"], None])
def test_auto_backend_falls_back_to_plain_14b(ollama_tags, tags) -> None:
    from rebirth_ai.xlang.translator import Translator

    ollama_tags["tags"] = tags  # adapter not installed, or Ollama unreachable
    assert Translator("auto").model_name == "ollama:qwen2.5-coder:14b-instruct-q6_K"


def test_xlang_model_env_override_and_off(ollama_tags, monkeypatch) -> None:
    from rebirth_ai.xlang.translator import Translator, get_translator

    ollama_tags["tags"] = ["rebirth-xlang:14b-v2", "rebirth-xlang:14b-v1"]
    monkeypatch.setenv("REBIRTH_XLANG_MODEL", "rebirth-xlang:14b-v2")
    assert Translator("auto").model_name == "ollama:rebirth-xlang:14b-v2"
    monkeypatch.setenv("REBIRTH_XLANG_MODEL", "")
    assert Translator("auto").model_name == "ollama:qwen2.5-coder:14b-instruct-q6_K"
    monkeypatch.setenv("REBIRTH_XLANG_MODEL", "rebirth-xlang:14b-v1")
    assert get_translator().backend == "auto"  # the default (REBIRTH_TRANSLATOR unset)
    with pytest.raises(ValueError):
        Translator("vllm")
