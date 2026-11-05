#!/usr/bin/env python3
"""Phase 10: run good-fixture sandbox tests for every registered language.

Exit 0 when pass-rate ≥ target (default 0.90).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rebirth_ai.adapters.registry import get_adapter, list_adapter_ids  # noqa: E402
from rebirth_ai.sandbox.runner import SandboxRunner  # noqa: E402

# Good sources only (no broken seeds) — verifies sandbox + fixture wiring.
FIXTURE_CASES: list[dict[str, object]] = [
    {"lang": "js_ts", "project": "fixtures/js_ts_sample", "test": "test/math.test.js"},
    {"lang": "python", "project": "fixtures/python_sample", "test": "tests/test_math_ops.py"},
    {"lang": "go", "project": "fixtures/go_sample", "test": "./src"},
    {"lang": "java", "project": "fixtures/java_sample", "test": None},
    {"lang": "csharp", "project": "fixtures/csharp_sample", "test": None},
    {"lang": "rust", "project": "fixtures/rust_sample", "test": None},
    {"lang": "cpp", "project": "fixtures/cpp_sample", "test": None},
    {"lang": "swift", "project": "fixtures/swift_sample", "test": None},
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="all", help="Adapter id or 'all'")
    parser.add_argument("--target", type=float, default=0.90)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "reports" / "eval_fixture_sandboxes.json",
    )
    args = parser.parse_args()

    wanted = set(list_adapter_ids()) if args.lang == "all" else {args.lang}
    cases = [c for c in FIXTURE_CASES if c["lang"] in wanted]
    results: list[dict[str, object]] = []
    for case in cases:
        lang = str(case["lang"])
        project = ROOT / str(case["project"])
        test = case.get("test")
        try:
            runner = SandboxRunner(get_adapter(lang))
            result = runner.run_project(
                project,
                relative_test=str(test) if test else None,
            )
            results.append(
                {
                    "id": f"{lang}_fixture",
                    "lang": lang,
                    "ok": result.ok,
                    "exit_code": result.exit_code,
                    "duration_sec": result.duration_sec,
                    "command": result.command,
                }
            )
            print(f"{lang}: {'PASS' if result.ok else 'FAIL'} ({result.duration_sec:.2f}s)")
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "id": f"{lang}_fixture",
                    "lang": lang,
                    "ok": False,
                    "error": str(exc),
                }
            )
            print(f"{lang}: ERROR {exc}")

    passed = sum(1 for r in results if r.get("ok"))
    total = len(results)
    rate = (passed / total) if total else 0.0
    report = {
        "suite": "fixture_sandboxes",
        "lang": args.lang,
        "target": args.target,
        "summary": {"passed": passed, "total": total, "rate": rate},
        "cases": results,
        "adapters_registered": list_adapter_ids(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.out}")
    return 0 if rate + 1e-9 >= args.target else 1


if __name__ == "__main__":
    raise SystemExit(main())
