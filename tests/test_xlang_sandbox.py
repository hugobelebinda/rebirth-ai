"""Translation harness in the real sandboxes (skipped without Docker / the 8 sandbox images)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest
from docker.errors import DockerException, ImageNotFound

import docker
from rebirth_ai.adapters.registry import get_adapter, list_adapter_ids
from rebirth_ai.xlang.languages import get_lang
from rebirth_ai.xlang.multipl_e import assemble, program_files
from rebirth_ai.xlang.verify import run_io_programs, run_programs, verify_translation

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "data" / "datasets" / "xlang" / "humaneval.jsonl"


def _ready() -> bool:
    try:
        client = docker.from_env()
        client.ping()
        for a in list_adapter_ids():
            client.images.get(get_adapter(a).sandbox_spec().image)
        return PROBLEMS.is_file()
    except (DockerException, ImageNotFound, OSError):
        return False


pytestmark = pytest.mark.skipif(not _ready(), reason="Docker, the sandbox images or the xlang problems are missing")


def test_oracle_solutions_pass_and_wrong_ones_fail() -> None:
    problems = {json.loads(ln)["id"]: json.loads(ln) for ln in PROBLEMS.read_text().splitlines() if ln.strip()}
    oracle = json.loads((ROOT / "fixtures/xlang_oracle/humaneval.json").read_text())
    by_lang: dict[str, dict] = defaultdict(dict)
    for lang, sols in oracle.items():
        for key, code in sols.items():
            by_lang[lang][key.replace("!", "__")] = assemble(get_lang(lang), problems[key.split("!")[0]]["targets"][lang], code)
    for lang, programs in by_lang.items():
        res = run_programs(get_lang(lang), programs, extra_files=program_files(get_lang(lang)))
        got = {k: r.status for k, r in res.items()}
        want = {k: "fail" if k.endswith("__wrong") else "pass" for k in programs}
        assert got == want, (lang, {k: res[k].log_tail[-300:] for k in got if got[k] != want[k]})


def test_compile_check_catches_broken_translation() -> None:
    go = get_lang("go")
    ok = verify_translation(target=go, code="package mathops\n\nfunc Add(a, b int) int { return a + b }\n", target_rel="src/mathops.go")
    bad = verify_translation(target=go, code="package mathops\n\nfunc Add(a, b int) int { return a + }\n", target_rel="src/mathops.go")
    assert ok.ok and ok.mode == "compile" and not bad.ok and "syntax error" in bad.logs


def test_program_oracle_in_every_language() -> None:
    """A complete stdin → stdout program per language (incl. int-overflow bugs that must fail)."""
    oracle = json.loads((ROOT / "fixtures/xlang_oracle/programs.json").read_text())
    tests = [tuple(t) for t in oracle["tests"]]
    by_lang: dict[str, dict] = defaultdict(dict)
    for key, code in oracle["programs"].items():
        by_lang[key.split("!")[0]][key.replace("!", "__")] = (code, tests)
    assert set(by_lang) == {"python", "javascript", "typescript", "go", "java", "csharp", "rust", "cpp", "swift"}
    for lang, progs in by_lang.items():
        got = {k: r.status for k, r in run_io_programs(get_lang(lang), progs).items()}
        assert got == {k: "fail" if k.endswith("__wrong") else "pass" for k in progs}, (lang, got)


def test_typescript_to_javascript_keeps_behavior() -> None:
    """JS sources for the program suite / stage 3 = passing TypeScript with the types stripped."""
    from rebirth_ai.xlang.verify import ts_to_js

    oracle = json.loads((ROOT / "fixtures/xlang_oracle/programs.json").read_text())
    tests = [tuple(t) for t in oracle["tests"]]
    ts = oracle["programs"]["typescript"] + "\nenum Unused { A, B }\nlet cast = <number>(1 as unknown);\n"
    js = ts_to_js({"ok": ts, "broken": "let x: = ;\n"})
    assert set(js) == {"ok"} and ": number" not in js["ok"] and "enum" not in js["ok"]
    assert run_io_programs(get_lang("javascript"), {"ok": (js["ok"], tests)})["ok"].status == "pass"
