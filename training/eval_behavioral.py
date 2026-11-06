#!/usr/bin/env python3
"""Behavioral eval: sandbox-pass + AST metrics (not exact-string only).

Two suites:
  1. Fixture sandbox cases — generate a fix from a broken seed, overlay, run tests.
  2. Short synthetic pairs — AST parse + symbol Jaccard + soft exact / overlap.

Examples:
  python training/eval_behavioral.py --backend peft --suite sandbox
  python training/eval_behavioral.py --backend peft --suite both --n-synth 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURE = ROOT / "fixtures" / "js_ts_sample"
SEEDS = ROOT / "fixtures" / "migration_seeds"

SANDBOX_CASES = [
    {
        "id": "math_broken",
        "relative_file": "src/math.js",
        "relative_test": "test/math.test.js",
        "seed": SEEDS / "math.broken.js",
        "gold": FIXTURE / "src" / "math.js",
        "objective": (
            "Fix multiply and keep exports working so math tests pass; "
            "modernize var without changing public API"
        ),
        "language": "javascript",
        "project": FIXTURE,
    },
    {
        "id": "format_broken",
        "relative_file": "src/format.js",
        "relative_test": "test/format.test.js",
        "seed": SEEDS / "format.broken.js",
        "gold": FIXTURE / "src" / "format.js",
        "objective": (
            "Fix formatResult separator to label=value so format tests pass; "
            "preserve exports"
        ),
        "language": "javascript",
        "project": FIXTURE,
    },
]

EVAL_SUITE_DIR = ROOT / "fixtures" / "eval_suite"


def load_manifest_cases(manifest: Path) -> list[dict]:
    """Load Phase 7 held-out cases from fixtures/eval_suite/manifest.json."""
    data = json.loads(manifest.read_text(encoding="utf-8"))
    base = manifest.parent
    project = (base / data.get("project", ".")).resolve()
    cases: list[dict] = []
    for raw in data.get("cases") or []:
        cases.append(
            {
                "id": raw["id"],
                "relative_file": raw["relative_file"],
                "relative_test": raw["relative_test"],
                "seed": base / raw["seed"],
                "gold": project / raw["relative_file"],
                "objective": raw["objective"],
                "language": raw.get("language") or "javascript",
                "project": project,
            }
        )
    return cases


def _get_generator(backend: str, language: str | None = None):
    from rebirth_ai.generate.factory import get_generator

    os.environ["REBIRTH_GENERATOR"] = backend
    if backend in {"peft", "lora", "local"}:
        os.environ.setdefault("REBIRTH_EMBEDDER", "hash")
    return get_generator(backend, language=language)


def _eval_ast(pred: str, gold: str, *, language: str) -> dict:
    from rebirth_ai.eval.metrics import (
        ast_parse_ok,
        soft_exact_match,
        symbol_jaccard,
        token_overlap,
    )

    return {
        "parses": ast_parse_ok(pred, language=language),
        "symbol_jaccard": round(symbol_jaccard(pred, gold, language=language), 3),
        "soft_exact": soft_exact_match(pred, gold),
        "token_overlap": round(token_overlap(pred, gold), 3),
    }


def run_sandbox_suite(
    backend: str,
    *,
    cases: list[dict] | None = None,
    suite_name: str = "sandbox",
) -> dict:
    from rebirth_ai.adapters.js_ts import JsTsAdapter
    from rebirth_ai.sandbox.runner import SandboxRunner

    case_list = cases or SANDBOX_CASES
    client = _get_generator(backend)
    # Whole-file repairs need chat style + roomy decode (SFT caps truncate).
    if hasattr(client, "prompt_style"):
        client.prompt_style = "chat"
    if hasattr(client, "max_new_tokens"):
        client.max_new_tokens = max(int(getattr(client, "max_new_tokens", 1024)), 1024)
    runner = SandboxRunner(JsTsAdapter())
    results = []
    sandbox_pass = 0
    seed_fail_ok = 0

    print(f"\n=== {suite_name} suite ({backend}) — {len(case_list)} cases ===")
    for case in case_list:
        project = Path(case.get("project") or FIXTURE)
        seed = Path(case["seed"]).read_text(encoding="utf-8")
        gold = Path(case["gold"]).read_text(encoding="utf-8")
        language = case["language"]

        # Sanity: broken seed should fail tests
        seed_result = runner.run_project(
            project,
            relative_test=case["relative_test"],
            overlay_text=seed,
            overlay_dest=case["relative_file"],
        )
        if not seed_result.ok:
            seed_fail_ok += 1

        pred = client.migrate(
            objective=case["objective"],
            legacy_code=seed,
            file_path=case["relative_file"],
            language=language,
        )
        code = pred.code
        ast = _eval_ast(code, gold, language=language)

        test_result = runner.run_project(
            project,
            relative_test=case["relative_test"],
            overlay_text=code,
            overlay_dest=case["relative_file"],
        )
        passed = test_result.ok
        if passed:
            sandbox_pass += 1

        mark = "PASS" if passed else "FAIL"
        print(
            f"[{case['id']}] sandbox={mark} parses={ast['parses']} "
            f"sym={ast['symbol_jaccard']:.2f} soft_exact={ast['soft_exact']} "
            f"model={pred.model}"
        )
        results.append(
            {
                "id": case["id"],
                "sandbox_pass": passed,
                "seed_fails_as_expected": not seed_result.ok,
                "test_exit": test_result.exit_code,
                "ast": ast,
                "model": pred.model,
                "pred_chars": len(code),
                "pred_preview": code[:240],
            }
        )

    n = len(case_list)
    return {
        "suite": suite_name,
        "backend": backend,
        "n": n,
        "sandbox_pass": sandbox_pass,
        "sandbox_pass_rate": sandbox_pass / n if n else 0.0,
        "seed_fails_baseline": seed_fail_ok,
        "results": results,
    }


def run_synth_suite(
    backend: str,
    *,
    n: int,
    data: Path,
    lang: str | None = None,
    seed: int | None = None,
    eval_file: Path | None = None,
) -> dict:
    """Soft-exact / AST suite.

    Without ``lang``: shortest synthetic rows of ``data`` (legacy JS/TS behaviour).
    With ``lang``: shortest rows from that language's **held-out eval split**
    (Phase 11 — never seen in LoRA training), routed to that language's adapter.
    With ``eval_file``: the first ``n`` rows of a frozen eval set, in file order
    (``training/make_eval_sets.py`` — seeded random held-out rows, not just the shortest).
    """
    from rebirth_ai.eval.metrics import soft_exact_match
    from rebirth_ai.lora import DEFAULT_EVAL_FRAC, DEFAULT_SEED, load_rows, lora_target, split_rows

    client = _get_generator(backend, language=lang)
    if eval_file is not None:
        pool = load_rows(eval_file)
        suite_name = f"evalset:{eval_file.stem}"
    elif lang:
        target = lora_target(lang)
        pool = split_rows(
            load_rows(target.data), eval_frac=DEFAULT_EVAL_FRAC, seed=seed or DEFAULT_SEED
        ).eval
        suite_name = f"heldout_split:{target.lang}"
    else:
        rows = load_rows(data)
        synth = [r for r in rows if r.get("source", "synthetic") == "synthetic"]
        pool = synth or rows
        suite_name = "synthetic"
    if eval_file is None:
        pool = sorted(pool, key=lambda r: len(r.get("legacy", "")) + len(r.get("modern", "")))
    sample = pool[:n]

    print(f"\n=== {suite_name} suite ({backend}) — {len(sample)} pairs ===")
    results = []
    soft = 0
    parses = 0
    jaccards: list[float] = []
    overlaps: list[float] = []

    for i, row in enumerate(sample, 1):
        language = row.get("language", "javascript")
        gold = row["modern"]
        pred = client.migrate(
            objective=row["objective"],
            legacy_code=row["legacy"],
            file_path=row.get("path") or "snippet.js",
            language=language,
        )
        ast = _eval_ast(pred.code, gold, language=language)
        if ast["soft_exact"]:
            soft += 1
        if ast["parses"]:
            parses += 1
        jaccards.append(ast["symbol_jaccard"])
        overlaps.append(ast["token_overlap"])
        mark = "SOFT_EXACT" if ast["soft_exact"] else f"sym={ast['symbol_jaccard']:.2f}"
        print(f"[{i}/{len(sample)}] {mark} parses={ast['parses']} :: {row['objective'][:60]}")
        results.append(
            {
                "i": i,
                "objective": row["objective"][:120],
                "ast": ast,
                "strict_ws_norm_was_misleading": (
                    soft_exact_match(pred.code, gold)
                    and pred.code.strip() != gold.strip()
                ),
                "pred_preview": pred.code[:200],
                "gold_preview": gold[:200],
                # Full texts so parse failures can be diagnosed (where/why the code breaks)
                "pred": pred.code,
                "gold": gold,
            }
        )

    n_s = len(sample)
    return {
        "suite": suite_name,
        "backend": backend,
        "lang": lang,
        "n": n_s,
        "soft_exact": soft,
        "soft_exact_rate": soft / n_s if n_s else 0.0,
        "parses": parses,
        "parse_rate": parses / n_s if n_s else 0.0,
        "mean_symbol_jaccard": sum(jaccards) / len(jaccards) if jaccards else 0.0,
        "mean_token_overlap": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "model": getattr(pred, "model", None) if sample else None,
        "results": results,
    }


def _track_synthetic(args: argparse.Namespace) -> None:
    """Best-effort MLflow record of the synthetic suite (never fails the eval)."""
    from rebirth_ai.lora import lora_target
    from rebirth_ai.tracking import try_log_eval

    adapter: Path | None = None
    if args.model_label:
        label = args.model_label
    elif args.backend == "ollama":
        label = "ollama-14b"
    elif os.environ.get("REBIRTH_PEFT_ADAPTER", "").lower() == "none":
        label = f"base-7b-{os.environ.get('REBIRTH_PEFT_PROMPT_STYLE', 'sft')}"
    else:
        label = "adapter"
        env_adapter = os.environ.get("REBIRTH_PEFT_ADAPTER")
        if env_adapter:
            adapter = Path(env_adapter)
        elif args.lang:
            adapter = lora_target(args.lang).adapter
    run_id = try_log_eval(
        args.out, model_label=label, lang=args.lang, adapter=adapter, eval_file=args.eval_file
    )
    if run_id:
        print(f"[tracking] eval → MLflow run {run_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="peft",
        choices=["peft", "ollama"],
        help="Generator backend",
    )
    parser.add_argument(
        "--suite",
        default="both",
        choices=["sandbox", "synthetic", "heldout", "both", "all"],
    )
    parser.add_argument("--n-synth", type=int, default=8)
    parser.add_argument(
        "--lang",
        default=None,
        help="Synthetic suite: sample this language's held-out split + route PEFT to its adapter",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=None,
        help="Frozen eval set JSONL (overrides held-out split sampling; see make_eval_sets.py)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "datasets" / "js_ts_migration.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=EVAL_SUITE_DIR / "manifest.json",
        help="Held-out suite manifest (Phase 7)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "reports" / "eval_behavioral.json",
    )
    parser.add_argument(
        "--freeze-baseline",
        type=Path,
        default=None,
        help="Also write a baseline snapshot JSON (e.g. data/reports/baselines/ollama_js_ts.json)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force PEFT on CPU")
    parser.add_argument(
        "--track",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record the synthetic-suite result in MLflow (Phase 13; no-op if MLflow is missing)",
    )
    parser.add_argument(
        "--model-label",
        default=None,
        help="MLflow label for what was scored (default: adapter / base-7b-<style> / ollama-14b)",
    )
    args = parser.parse_args()

    if args.cpu:
        os.environ["REBIRTH_PEFT_DEVICE"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HIP_VISIBLE_DEVICES"] = ""

    suite = args.suite
    report: dict = {"backend": args.backend, "suites": {}}
    if suite in {"sandbox", "both", "all"}:
        report["suites"]["sandbox"] = run_sandbox_suite(args.backend)
    if suite in {"heldout", "all"}:
        cases = load_manifest_cases(args.manifest)
        report["suites"]["heldout"] = run_sandbox_suite(
            args.backend, cases=cases, suite_name="heldout"
        )
    if suite in {"synthetic", "both", "all"}:
        report["suites"]["synthetic"] = run_synth_suite(
            args.backend, n=args.n_synth, data=args.data, lang=args.lang, eval_file=args.eval_file
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.track and "synthetic" in report["suites"]:
        _track_synthetic(args)

    # Headline
    print("\n=== headline ===")
    for key in ("sandbox", "heldout", "synthetic"):
        if key not in report["suites"]:
            continue
        s = report["suites"][key]
        if key == "synthetic":
            print(
                f"synth soft_exact={s['soft_exact']}/{s['n']} "
                f"parse={s['parses']}/{s['n']} "
                f"mean_sym={s['mean_symbol_jaccard']:.3f}"
            )
        else:
            print(
                f"{key}_pass={s['sandbox_pass']}/{s['n']} "
                f"({s['sandbox_pass_rate']:.0%})"
            )
    print(f"wrote {args.out}")

    if args.freeze_baseline is not None:
        from datetime import datetime, timezone

        baseline = {
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "backend": args.backend,
            "model": os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b-instruct-q6_K"),
            "suites": {
                k: {
                    "n": v.get("n"),
                    "sandbox_pass": v.get("sandbox_pass"),
                    "sandbox_pass_rate": v.get("sandbox_pass_rate"),
                    "soft_exact_rate": v.get("soft_exact_rate"),
                    "mean_symbol_jaccard": v.get("mean_symbol_jaccard"),
                }
                for k, v in report["suites"].items()
            },
            "source_report": str(args.out),
            "gate": "held-out / sandbox pass rate must stay within baseline − 5%",
        }
        args.freeze_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.freeze_baseline.write_text(
            json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
        )
        print(f"froze baseline → {args.freeze_baseline}")

    # Exit non-zero on sandbox/heldout regressions (any fail)
    for key in ("sandbox", "heldout"):
        sand = report["suites"].get(key)
        if sand is not None and sand["sandbox_pass"] < sand["n"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
