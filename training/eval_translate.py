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

from rebirth_ai.xlang.languages import LANGS, get_lang  # noqa: E402
from rebirth_ai.xlang.multipl_e import (  # noqa: E402
    BROKEN_TESTS,
    assemble,
    program_files,
    python_target_parts,
    signature_stub,
    split_prompt,
)
from rebirth_ai.xlang.prompts import build_function_prompt, extract_code  # noqa: E402

PROBLEMS = ROOT / "data" / "datasets" / "xlang" / "humaneval.jsonl"
DEFAULT_SOURCES = ["python", "javascript", "go", "java", "cpp"]
DEFAULT_TARGETS = ["python", "typescript", "go", "java", "csharp", "rust", "cpp", "swift"]


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def load_problems(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def add_sources_from(problems: list[dict], report: Path, langs: list[str]) -> dict[str, int]:
    """Bootstrap sources for languages with no reference solutions (Rust, C#, Swift, TS): a
    translation *into* that language that passed the target tests becomes its source.
    Python-sourced translations are preferred. Only problems the model already solved get one, so
    these directions run on an easier subset — reported as such."""
    rows = json.loads(report.read_text(encoding="utf-8"))["rows"]
    by_id = {p["id"]: p for p in problems}
    rank = {"python": 0}
    best: dict[tuple[str, str], dict] = {}
    for r in rows:
        if r["status"] != "pass" or r["tgt"] not in langs or r["tgt"] in by_id[r["problem"]]["sources"]:
            continue
        key = (r["problem"], r["tgt"])
        if key not in best or rank.get(r["src"], 1) < rank.get(best[key]["src"], 1):
            best[key] = r
    counts: Counter = Counter()
    for (pid, lang), r in best.items():
        by_id[pid]["sources"][lang] = r["code"]
        by_id[pid].setdefault("bootstrap", {})[lang] = r["src"]
        counts[lang] += 1
    print(f"bootstrapped sources from {_rel(report)}: {dict(counts)}", flush=True)
    return dict(counts)


def target_signature(tgt: str, t: dict) -> tuple[str, str]:
    """(signature stub, file header) the model is shown — the header is what MultiPL-E shows too."""
    lang = get_lang(tgt)
    header, sig = python_target_parts(t["prompt"], t["entry_point"]) if tgt == "python" else split_prompt(t["prompt"])
    return signature_stub(lang, sig), header.strip("\n")


def run_and_score(by_target: dict[str, dict[str, tuple[dict, str]]], workers: int, timeout: int) -> dict[str, object]:
    """by_target[tgt][rid] = (problem target entry, translated code) → rid → ProgramResult."""
    from rebirth_ai.xlang.verify import run_programs

    results: dict[str, object] = {}
    for tgt, items in by_target.items():
        lang = get_lang(tgt)
        programs = {rid: assemble(lang, t, code) for rid, (t, code) in items.items()}
        t0 = time.time()
        res = run_programs(lang, programs, workers=workers, timeout_sec=timeout, extra_files=program_files(lang))
        print(f"  ran {len(programs)} {tgt} programs in {time.time() - t0:.0f}s", flush=True)
        results.update(res)
    return results


def oracle(args) -> int:
    problems = {p["id"]: p for p in load_problems(args.problems)}
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
    from rebirth_ai.xlang.translator import get_translator, max_new_tokens_for

    problems = load_problems(args.problems)
    bootstrapped = add_sources_from(problems, args.add_sources_from, args.sources) if args.add_sources_from else {}
    order = list(range(len(problems)))
    random.Random(args.seed).shuffle(order)  # one order for every direction → comparable subsets
    directions = [(s, t) for s in args.sources for t in args.targets if s != t and not (s == "javascript" and t == "typescript")]
    gen_path = args.out.with_suffix(".gen.jsonl")
    done: dict[tuple, dict] = {}
    if gen_path.is_file():
        for ln in gen_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                g = json.loads(ln)
                done[(g["src"], g["tgt"], g["problem"])] = g
    args.out.parent.mkdir(parents=True, exist_ok=True)
    picked: list[tuple[str, str, dict]] = []
    for s, t in directions:
        rows = [problems[i] for i in order if s in problems[i]["sources"] and t in problems[i]["targets"]][: args.limit]
        picked += [(s, t, p) for p in rows]
    todo = [x for x in picked if (x[0], x[1], x[2]["id"]) not in done]
    if args.rescore:
        if todo:
            print(f"--rescore: {len(todo)} of {len(picked)} translations are not in {_rel(gen_path)}; run without --rescore")
            return 1
        model_name = next(iter({done[(s, t, p["id"])]["model"] for s, t, p in picked}), "?")
    else:
        translator = get_translator(args.backend)
        model_name = translator.model_name
    print(f"{len(directions)} directions, {len(picked)} translations ({len(picked) - len(todo)} cached), model {model_name}", flush=True)
    t0 = time.time()
    with gen_path.open("a", encoding="utf-8") as fh:
        for k, (s, t, p) in enumerate(todo, 1):
            src, tgt = get_lang(s), get_lang(t)
            sig, header = target_signature(t, p["targets"][t])
            prompt = build_function_prompt(
                source=src, target=tgt, source_code=p["sources"][s], target_signature=sig,
                class_wrapped=t in ("java", "csharp"), target_header=header,
            )
            out = translator.complete(prompt, max_new_tokens=max_new_tokens_for(p["sources"][s]))
            g = {"src": s, "tgt": t, "problem": p["id"], "raw": out.raw, "code": extract_code(out.raw, tgt), "model": out.model}
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")
            fh.flush()
            done[(s, t, p["id"])] = g
            if k % 20 == 0 or k == len(todo):
                rate = (time.time() - t0) / k
                print(f"  [{k}/{len(todo)}] {s}->{t} {p['id']}  {rate:.1f}s/translation, ~{rate * (len(todo) - k) / 60:.0f} min left", flush=True)

    by_target: dict[str, dict] = defaultdict(dict)
    meta = {}
    for s, t, p in picked:
        rid = f"{s}__{t}__{p['id']}"
        meta[rid] = (s, t, p["id"])
        if (t, p["id"]) not in BROKEN_TESTS:
            by_target[t][rid] = (p["targets"][t], done[(s, t, p["id"])]["code"])
    res = run_and_score(by_target, args.workers, args.timeout)

    per: dict[str, Counter] = defaultdict(Counter)
    excluded: Counter = Counter()
    rows = []
    for rid, (s, t, pid) in meta.items():
        if (t, pid) in BROKEN_TESTS:
            excluded[f"{s}->{t}"] += 1
            rows.append({"src": s, "tgt": t, "problem": pid, "status": "broken_test", "exit": None,
                         "code": done[(s, t, pid)]["code"], "log": BROKEN_TESTS[(t, pid)]})
            continue
        r = res[rid]
        per[f"{s}->{t}"][r.status] += 1
        rows.append({"src": s, "tgt": t, "problem": pid, "status": r.status, "exit": r.exit_code,
                     "code": done[(s, t, pid)]["code"], "log": r.log_tail[-600:]})
    summary = {d: {"n": sum(c.values()), "pass": c["pass"], "pass_rate": round(c["pass"] / max(1, sum(c.values())), 3),
                   **{k: c[k] for k in ("compile_error", "fail", "timeout", "missing")}, "excluded": excluded[d]}
               for d, c in sorted(per.items())}
    report = {"model": model_name, "problems": _rel(args.problems), "limit": args.limit,
              "bootstrapped_sources": {"from": _rel(args.add_sources_from), "counts": bootstrapped} if bootstrapped else None,
              "seed": args.seed, "sources": args.sources, "targets": args.targets, "directions": summary, "rows": rows}
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print_matrix(summary, args.sources, args.targets)
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


def print_matrix(summary: dict, sources: list[str], targets: list[str]) -> None:
    width = 11
    print("\npass rate (source ↓ / target →)")
    print(" " * width + "".join(f"{t[:width - 1]:>{width}}" for t in targets))
    for s in sources:
        cells = []
        for t in targets:
            d = summary.get(f"{s}->{t}")
            cell = "-" if d is None else f"{d['pass']}/{d['n']}"
            cells.append(f"{cell:>{width}}")
        print(f"{s[:width - 1]:<{width}}" + "".join(cells))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", type=Path, default=PROBLEMS)
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
    ap.add_argument("--backend", default="ollama", help="ollama (Qwen2.5-Coder 14B) | peft (base 7B chat)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "reports" / "xlang" / "zeroshot.json")
    ap.add_argument("--rescore", action="store_true", help="re-run the tests on cached generations only (no model)")
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
