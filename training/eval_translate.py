#!/usr/bin/env python3
"""Cross-language translation eval (Phase 17): pass rate per direction on HumanEval.

For each direction source → target and problem: the model translates the source solution into a
function with the target signature; the target program (target header + translation + MultiPL-E /
HumanEval tests) is compiled and run in the target sandbox. Pass = every assert holds.

  # harness self-check, no model: hand-written correct (and deliberately wrong) solutions
  python training/eval_translate.py --oracle fixtures/xlang_oracle/humaneval.json

  # fixture matrix: each sample project → each other one, full loop vs the target project's tests
  python training/eval_translate.py --fixtures fixtures/xlang_fixtures.json --out data/reports/xlang/fixtures_ollama14b.json

  # zero-shot baseline (GPU / Ollama): 20 problems per direction, every source × every target
  python training/eval_translate.py --backend ollama --limit 20 --out data/reports/xlang/zeroshot_ollama14b.json

  # program-level suite: whole Codeforces programs (training/build_xlang_programs.py), judged by
  # stdin → stdout tests; the product prompt (`migrate --to`) with the "complete program" rule
  python training/eval_translate.py --suite programs --limit 20 --out data/reports/xlang/programs_ollama14b.json

  # with the product's retry loop: a failed translation gets its compiler output / first failing
  # test back and is regenerated (up to N rounds, temperature 0.1 → 0.4 → 0.7, like `migrate --to`)
  python training/eval_translate.py --suite programs --retries 2 --out data/reports/xlang/programs_ollama14b_retry.json

  # languages without reference programs: sources = passing translations from a report (JavaScript =
  # a passing TypeScript translation with its types stripped by Node, re-checked on the tests)
  python training/eval_translate.py --suite programs --sources javascript go csharp rust swift \
      --add-sources-from data/reports/xlang/programs_ollama14b_retry.json --retries 2 --out ...

Generations are appended to ``<out>.gen.jsonl`` as they are made, so a stopped run resumes;
``--rescore`` re-runs the tests on those cached generations only (no model, fails if any is missing).
Problems with a known upstream test bug (``BROKEN_TESTS``) are reported but left out of pass rates.
Sources: python, cpp, java, javascript, go (reference solutions exist); targets: all 9 ids.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rebirth_ai.xlang.fixups import apply_fixups  # noqa: E402
from rebirth_ai.xlang.languages import LANGS, get_lang  # noqa: E402
from rebirth_ai.xlang.multipl_e import (  # noqa: E402
    BROKEN_TESTS,
    assemble,
    program_files,
    python_target_parts,
    signature_stub,
    split_prompt,
)
from rebirth_ai.xlang.prompts import build_function_prompt, build_translation_prompt, extract_code  # noqa: E402

# Imported up front (not lazily) so a long run uses one consistent snapshot of the code: editing
# a module mid-run once crashed a 2,236-translation run at the scoring step.
from rebirth_ai.xlang.translator import get_translator, max_new_tokens_for, retry_temperature  # noqa: E402
from rebirth_ai.xlang.verify import run_io_programs, run_programs, ts_to_js  # noqa: E402

PROBLEMS = ROOT / "data" / "datasets" / "xlang" / "humaneval.jsonl"
PROGRAMS = ROOT / "data" / "datasets" / "xlang" / "programs.jsonl"
DEFAULT_SOURCES = ["python", "javascript", "go", "java", "cpp"]
DEFAULT_TARGETS = ["python", "typescript", "go", "java", "csharp", "rust", "cpp", "swift"]


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def load_problems(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def add_sources_from(problems: list[dict], report: Path, langs: list[str], workers: int = 4) -> dict[str, int]:
    """Bootstrap sources for languages with no reference solutions (Rust, C#, Swift, TS): a
    translation *into* that language that passed the target tests becomes its source.
    Python-sourced translations are preferred. Only problems the model already solved get one, so
    these directions run on an easier subset — reported as such.

    Programs suite + ``javascript`` (never a target): a passing TypeScript translation with its
    types stripped (Node, deterministic), kept only if it passes the problem's tests as JavaScript."""
    rows = json.loads(report.read_text(encoding="utf-8"))["rows"]
    by_id = {p["id"]: p for p in problems}
    rank = {"python": 0}
    derive_js = "javascript" in langs and all("tests" in p for p in problems)
    wanted = set(langs) | ({"typescript"} if derive_js else set())
    best: dict[tuple[str, str], dict] = {}
    for r in rows:
        if r["status"] != "pass" or r["tgt"] not in wanted or r["problem"] not in by_id:
            continue
        key = (r["problem"], r["tgt"])
        if key not in best or rank.get(r["src"], 1) < rank.get(best[key]["src"], 1):
            best[key] = r
    counts: Counter = Counter()
    for (pid, lang), r in best.items():
        if lang in langs and lang not in by_id[pid]["sources"]:
            by_id[pid]["sources"][lang] = r["code"]
            by_id[pid].setdefault("bootstrap", {})[lang] = r["src"]
            counts[lang] += 1
    if derive_js:
        ts = {pid: r["code"] for (pid, lang), r in best.items() if lang == "typescript" and "javascript" not in by_id[pid]["sources"]}
        js = ts_to_js(ts)
        res = run_io_programs(get_lang("javascript"), {pid: (code, [tuple(t) for t in by_id[pid]["tests"]]) for pid, code in js.items()},
                              workers=workers, timeout_sec=15)
        for pid, code in js.items():
            if res[pid].status == "pass":
                by_id[pid]["sources"]["javascript"] = code
                by_id[pid].setdefault("bootstrap", {})["javascript"] = f"typescript (types stripped) <- {best[(pid, 'typescript')]['src']}"
                counts["javascript"] += 1
        print(f"  javascript from typescript: {len(ts)} passing TS, {len(js)} stripped, {counts['javascript']} pass as JS", flush=True)
    print(f"bootstrapped sources from {_rel(report)}: {dict(counts)}", flush=True)
    return dict(counts)


def error_class_for(status: str, log: str) -> str:
    """What the retry prompt calls the failure (``migrate --to`` uses the same names)."""
    if status == "compile_error":
        return "CompileError"
    if status == "timeout":
        return "Timeout"
    return "WrongAnswer" if "wrong answer" in log else "RuntimeError"


def target_signature(tgt: str, t: dict) -> tuple[str, str]:
    """(signature stub, file header) the model is shown — the header is what MultiPL-E shows too."""
    lang = get_lang(tgt)
    header, sig = python_target_parts(t["prompt"], t["entry_point"]) if tgt == "python" else split_prompt(t["prompt"])
    return signature_stub(lang, sig), header.strip("\n")


def run_and_score(
    by_target: dict[str, dict[str, tuple[dict, str]]], workers: int, timeout: int, fixups: bool = True
) -> dict[str, object]:
    """by_target[tgt][rid] = (problem target entry, translated code) → rid → ProgramResult.

    ``fixups``: the deterministic clean-up ``migrate --to`` also applies (Go: missing stdlib
    imports), run on the assembled program."""
    results: dict[str, object] = {}
    for tgt, items in by_target.items():
        lang = get_lang(tgt)
        programs = {rid: assemble(lang, t, code) for rid, (t, code) in items.items()}
        if fixups:
            programs = {rid: apply_fixups(lang, prog) for rid, prog in programs.items()}
        t0 = time.time()
        res = run_programs(lang, programs, workers=workers, timeout_sec=timeout, extra_files=program_files(lang))
        print(f"  ran {len(programs)} {tgt} programs in {time.time() - t0:.0f}s", flush=True)
        results.update(res)
    return results


def program_path(lang) -> str:
    """File name a complete program gets (shown to the model as the target file)."""
    return {"java": "Main.java", "csharp": "Program.cs"}.get(lang.id, f"main{lang.ext}")


def score_programs(
    by_target: dict[str, dict[str, tuple[str, list]]], workers: int, timeout: int, fixups: bool = True
) -> dict[str, object]:
    """by_target[tgt][rid] = (translated program, [(stdin, stdout), ...]) → rid → ProgramResult."""
    results: dict[str, object] = {}
    for tgt, items in by_target.items():
        lang = get_lang(tgt)
        progs = {rid: (apply_fixups(lang, code) if fixups else code, [tuple(t) for t in tests])
                 for rid, (code, tests) in items.items()}
        t0 = time.time()
        results.update(run_io_programs(lang, progs, workers=workers, timeout_sec=timeout))
        print(f"  ran {len(progs)} {tgt} programs in {time.time() - t0:.0f}s", flush=True)
    return results


def oracle(args) -> int:
    problems = {p["id"]: p for p in load_problems(args.problems or PROBLEMS)}
    spec = json.loads(args.oracle.read_text(encoding="utf-8"))
    by_target: dict[str, dict] = defaultdict(dict)
    expect: dict[str, str] = {}
    for tgt, sols in spec.items():
        for key, code in sols.items():
            pid = key.split("!")[0]
            t = problems[pid]["targets"].get(tgt)
            if t is None:
                print(f"skip {tgt} {pid}: no target tests")
                continue
            rid = f"{tgt}__{key.replace('!', '__')}"
            by_target[tgt][rid] = (t, code)
            expect[rid] = "fail" if key.endswith("!wrong") else "pass"
    res = run_and_score(by_target, args.workers, args.timeout)
    bad = 0
    for rid, want in sorted(expect.items()):
        got = res[rid].status
        mark = "ok " if got == want else "BAD"
        bad += got != want
        print(f"{mark} {rid:40s} want={want:5s} got={got}")
        if got != want:
            print("    " + res[rid].log_tail[-800:].replace("\n", "\n    "))
    print(f"\n{len(expect) - bad}/{len(expect)} as expected")
    return 1 if bad else 0


def evaluate(args) -> int:
    programs = args.suite == "programs"
    if args.retries and not programs:
        print("--retries needs --suite programs (HumanEval rows are scored as assembled test programs)")
        return 1
    problems = load_problems(args.problems or (PROGRAMS if programs else PROBLEMS))
    bootstrapped = add_sources_from(problems, args.add_sources_from, args.sources, args.workers) if args.add_sources_from else {}
    order = list(range(len(problems)))
    random.Random(args.seed).shuffle(order)  # one order for every direction → comparable subsets
    directions = [(s, t) for s in args.sources for t in args.targets if s != t and not (s == "javascript" and t == "typescript")]
    gen_path = args.out.with_suffix(".gen.jsonl")
    done: dict[tuple, dict] = {}  # (src, tgt, problem, round) → generation
    if gen_path.is_file():
        for ln in gen_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                g = json.loads(ln)
                done[(g["src"], g["tgt"], g["problem"], g.get("round", 0))] = g
    args.out.parent.mkdir(parents=True, exist_ok=True)
    picked: list[tuple[str, str, dict]] = []
    for s, t in directions:
        rows = [problems[i] for i in order
                if s in problems[i]["sources"] and (programs or t in problems[i]["targets"])][: args.limit]
        picked += [(s, t, p) for p in rows]
    todo = [(s, t, p, None) for s, t, p in picked if (s, t, p["id"], 0) not in done]
    if args.rescore and todo:
        print(f"--rescore: {len(todo)} of {len(picked)} translations are not in {_rel(gen_path)}; run without --rescore")
        return 1
    translator = None
    model_name = next(iter({done[(s, t, p["id"], 0)]["model"] for s, t, p in picked if (s, t, p["id"], 0) in done}), "?")
    if not args.rescore:
        translator = get_translator(args.backend)
        model_name = translator.model_name
    print(f"{len(directions)} directions, {len(picked)} translations ({len(picked) - len(todo)} cached), model {model_name}"
          + (f", up to {args.retries} retry rounds" if args.retries else ""), flush=True)

    def prompt_for(s: str, t: str, p: dict, retry: tuple[str, str, str] | None) -> str:
        src, tgt = get_lang(s), get_lang(t)
        if programs:
            prev, err, logs = retry or (None, None, None)
            return build_translation_prompt(
                source=src, target=tgt, source_code=p["sources"][s], source_path=program_path(src),
                target_path=program_path(tgt), is_program=True, previous=prev, error_class=err, logs=logs,
            )
        sig, header = target_signature(t, p["targets"][t])
        return build_function_prompt(
            source=src, target=tgt, source_code=p["sources"][s], target_signature=sig,
            class_wrapped=t in ("java", "csharp"), target_header=header,
        )

    def generate(items: list, rnd: int) -> None:
        if not items:
            return
        t0 = time.time()
        with gen_path.open("a", encoding="utf-8") as fh:
            for k, (s, t, p, retry) in enumerate(items, 1):
                out = translator.complete(prompt_for(s, t, p, retry), max_new_tokens=max_new_tokens_for(p["sources"][s]),
                                          temperature=retry_temperature(rnd))
                g = {"src": s, "tgt": t, "problem": p["id"], "round": rnd, "raw": out.raw,
                     "code": extract_code(out.raw, get_lang(t)), "model": out.model}
                fh.write(json.dumps(g, ensure_ascii=False) + "\n")
                fh.flush()
                done[(s, t, p["id"], rnd)] = g
                if k % 20 == 0 or k == len(items):
                    rate = (time.time() - t0) / k
                    print(f"  [round {rnd}: {k}/{len(items)}] {s}->{t} {p['id']}  {rate:.1f}s/translation, "
                          f"~{rate * (len(items) - k) / 60:.0f} min left", flush=True)

    def score(items: list[tuple[str, str, dict]], rnd: int) -> dict[str, object]:
        by_target: dict[str, dict] = defaultdict(dict)
        for s, t, p in items:
            rid = f"{s}__{t}__{p['id']}"
            code = done[(s, t, p["id"], rnd)]["code"]
            if programs:
                by_target[t][rid] = (code, p["tests"])
            elif (t, p["id"]) not in BROKEN_TESTS:
                by_target[t][rid] = (p["targets"][t], code)
        return (score_programs if programs else run_and_score)(by_target, args.workers, args.timeout, fixups=not args.no_fixups)

    generate(todo, 0)
    meta = {f"{s}__{t}__{p['id']}": (s, t, p) for s, t, p in picked}
    res = score(picked, 0)
    first = {rid: r.status for rid, r in res.items()}
    final_round = dict.fromkeys(meta, 0)
    for rnd in range(1, args.retries + 1):
        failed = [(s, t, p) for rid, (s, t, p) in meta.items()
                  if rid in res and res[rid].status != "pass" and (t, p["id"]) not in BROKEN_TESTS]
        if not failed:
            break
        items = []
        for s, t, p in failed:
            r = res[f"{s}__{t}__{p['id']}"]
            prev = done[(s, t, p["id"], rnd - 1)]["code"]
            prev = prev if args.no_fixups else apply_fixups(get_lang(t), prev)  # as tested: line numbers match the log
            items.append((s, t, p, (prev, error_class_for(r.status, r.log_tail), r.log_tail)))
        missing = [x for x in items if (x[0], x[1], x[2]["id"], rnd) not in done]
        if args.rescore and missing:
            print(f"--rescore: {len(missing)} round-{rnd} retries are not cached; run without --rescore")
            return 1
        print(f"round {rnd}: retrying {len(failed)} failed translations ({len(missing)} to generate)", flush=True)
        generate(missing, rnd)
        res.update(score(failed, rnd))
        for s, t, p in failed:
            final_round[f"{s}__{t}__{p['id']}"] = rnd

    per: dict[str, Counter] = defaultdict(Counter)
    excluded: Counter = Counter()
    rows = []
    for rid, (s, t, p) in meta.items():
        pid, rnd = p["id"], final_round[rid]
        code = done[(s, t, pid, rnd)]["code"]
        if programs and not args.no_fixups:
            code = apply_fixups(get_lang(t), code)  # the file as tested
        if (t, pid) in BROKEN_TESTS:
            excluded[f"{s}->{t}"] += 1
            rows.append({"src": s, "tgt": t, "problem": pid, "status": "broken_test", "exit": None,
                         "code": code, "log": BROKEN_TESTS[(t, pid)]})
            continue
        r = res[rid]
        per[f"{s}->{t}"][r.status] += 1
        per[f"{s}->{t}"]["pass_first"] += first[rid] == "pass"
        rows.append({"src": s, "tgt": t, "problem": pid, "status": r.status, "first_status": first[rid], "rounds": rnd,
                     "exit": r.exit_code, "code": code, "log": r.log_tail[-600:]})
    summary = {d: {"n": sum(c[k] for k in c if k != "pass_first"), "pass": c["pass"], "pass_first": c["pass_first"],
                   "pass_rate": round(c["pass"] / max(1, sum(c[k] for k in c if k != "pass_first")), 3),
                   **{k: c[k] for k in ("compile_error", "fail", "timeout", "missing")}, "excluded": excluded[d]}
               for d, c in sorted(per.items())}
    report = {"model": model_name, "suite": args.suite, "problems": _rel(args.problems or (PROGRAMS if programs else PROBLEMS)), "limit": args.limit,
              "bootstrapped_sources": {"from": _rel(args.add_sources_from), "counts": bootstrapped} if bootstrapped else None,
              "fixups": not args.no_fixups, "retries": args.retries, "seed": args.seed, "sources": args.sources,
              "targets": args.targets, "directions": summary, "rows": rows}
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.retries:
        print_matrix(summary, args.sources, args.targets, key="pass_first", title="first try")
        print_matrix(summary, args.sources, args.targets, title=f"after up to {args.retries} retry rounds")
    else:
        print_matrix(summary, args.sources, args.targets)
    n = sum(d["n"] for d in summary.values())
    print(f"overall: {sum(d['pass_first'] for d in summary.values())}/{n} first try"
          + (f", {sum(d['pass'] for d in summary.values())}/{n} with retries" if args.retries else ""))
    print(f"wrote {_rel(args.out)}")
    return 0


def fixtures(args) -> int:
    """Every sample project → every other one through the full loop (``rebirth migrate --to``)."""
    from rebirth_ai.xlang.graph import build_translation_graph, initial_translation_state
    from rebirth_ai.xlang.verify import _rust_test_mod

    man = {k: v for k, v in json.loads(args.fixtures.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    langs = [x for x in man if not args.fixture_langs or x in args.fixture_langs]
    pairs = [(s, t) for s in langs for t in langs if s != t]
    out = args.out
    done = {}
    if out.is_file():
        done = {(r["src"], r["tgt"]): r for r in json.loads(out.read_text(encoding="utf-8")).get("rows", [])}
    graph = build_translation_graph()
    out.parent.mkdir(parents=True, exist_ok=True)

    def save(model: str) -> None:
        rows = list(done.values())
        ok = sum(r["status"] == "success" for r in rows)
        out.write_text(json.dumps({"model": model, "max_iterations": args.max_iterations, "pairs": len(rows),
                                   "success": ok, "rows": rows}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    model = ""
    for k, (s, t) in enumerate(pairs, 1):
        if (s, t) in done:
            continue
        src, tgt = man[s], man[t]
        state = initial_translation_state(
            project_dir=ROOT / src["project"], source_rel=src["source"], target_lang=t, source_lang=s,
            target_project=ROOT / tgt["project"], target_file=tgt["target_file"], target_test=tgt["target_test"],
            backend=args.backend, max_iterations=args.max_iterations,
        )
        if s == "rust" and (span := _rust_test_mod(state["source_code"])):  # translate the code, not its tests
            state["source_code"] = state["source_code"][: span[0]].rstrip() + "\n"
        t0 = time.time()
        final = graph.invoke(state)
        model = final.get("model") or model
        done[(s, t)] = {"src": s, "tgt": t, "status": final.get("status"), "iterations": final.get("iteration_count"),
                        "error_class": final.get("error_class"), "seconds": round(time.time() - t0, 1),
                        "trace": final.get("trace"), "code": final.get("current_draft"),
                        "log": (final.get("test_logs") or "")[-1500:]}
        save(model)
        print(f"[{k}/{len(pairs)}] {s:10s} -> {t:10s} {final.get('status'):12s} iter={final.get('iteration_count')} "
              f"{time.time() - t0:.0f}s", flush=True)
    save(model)
    print("\nfixture matrix (iterations to pass; x = not passed)")
    w = 11
    print(" " * w + "".join(f"{t[:w - 1]:>{w}}" for t in langs))
    for s in langs:
        cells = []
        for t in langs:
            r = done.get((s, t))
            cells.append(f"{'-' if r is None else (str(r['iterations']) if r['status'] == 'success' else 'x'):>{w}}")
        print(f"{s[:w - 1]:<{w}}" + "".join(cells))
    ok = sum(r["status"] == "success" for r in done.values())
    print(f"{ok}/{len(done)} directions pass the target project's tests; wrote {_rel(out)}")
    return 0


def print_matrix(summary: dict, sources: list[str], targets: list[str], key: str = "pass", title: str = "") -> None:
    width = 11
    print(f"\npass rate (source ↓ / target →){' — ' + title if title else ''}")
    print(" " * width + "".join(f"{t[:width - 1]:>{width}}" for t in targets))
    for s in sources:
        cells = []
        for t in targets:
            d = summary.get(f"{s}->{t}")
            cell = "-" if d is None else f"{d[key]}/{d['n']}"
            cells.append(f"{cell:>{width}}")
        print(f"{s[:width - 1]:<{width}}" + "".join(cells))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["humaneval", "programs"], default="humaneval",
                    help="humaneval: one function vs MultiPL-E tests; programs: a whole program vs stdin/stdout tests")
    ap.add_argument("--problems", type=Path, default=None, help="default: the suite's frozen problem file")
    ap.add_argument("--oracle", type=Path, default=None, help="self-check the harness with known solutions")
    ap.add_argument("--fixtures", type=Path, default=None,
                    help="fixture matrix: fixtures/xlang_fixtures.json, full loop against target-project tests")
    ap.add_argument("--fixture-langs", nargs="*", default=None, help="restrict the fixture matrix to these")
    ap.add_argument("--max-iterations", type=int, default=3, help="--fixtures: translate/verify rounds per pair")
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES, choices=sorted(LANGS))
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS, choices=sorted(LANGS))
    ap.add_argument("--limit", type=int, default=20, help="problems per direction")
    ap.add_argument("--add-sources-from", type=Path, default=None,
                    help="report whose passing translations into --sources languages become their sources")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--backend", default="ollama", help="ollama (plain Qwen2.5-Coder 14B, OLLAMA_MODEL) | auto (packaged cross-language adapter, REBIRTH_XLANG_MODEL, if installed) | peft (base 7B chat)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "reports" / "xlang" / "zeroshot.json")
    ap.add_argument("--no-fixups", action="store_true",
                    help="score the raw model output (no Go import fix-up) — the stage-2 raw numbers")
    ap.add_argument("--rescore", action="store_true", help="re-run the tests on cached generations only (no model)")
    ap.add_argument("--retries", type=int, default=0,
                    help="programs suite: retry rounds for failed translations, fed the compiler output / first "
                         "failing test (what `migrate --to` does with the project's tests)")
    ap.add_argument("--workers", type=int, default=4, help="parallel sandbox containers per target")
    ap.add_argument("--timeout", type=int, default=15, help="seconds per test program run")
    args = ap.parse_args()
    if args.oracle:
        return oracle(args)
    if args.fixtures:
        return fixtures(args)
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
