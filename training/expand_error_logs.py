#!/usr/bin/env python3
"""Expand synthetic error-log dataset with more realistic CI variants.

Keeps label set: SyntaxError | DependencyMissing | TypeMismatch | TestLogicFailure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data" / "datasets" / "error_logs.jsonl"

EXTRA: list[dict[str, str]] = [
    # SyntaxError
    {
        "label": "SyntaxError",
        "text": "file:///workspace/src/index.js:1\nimport x from 'y'\n^^^^^^\n\nSyntaxError: Cannot use import statement outside a module\n    at Object.compileSourceTextModule",
    },
    {
        "label": "SyntaxError",
        "text": "/workspace/src/util.ts:8:1 - error TS1005: '}' expected.\n8 function broken(\n  ~\nFound 1 error in src/util.ts:8",
    },
    {
        "label": "SyntaxError",
        "text": "ERROR in ./src/app.js\nModule parse failed: Unexpected token (12:2)\nYou may need an appropriate loader to handle this file type.",
    },
    {
        "label": "SyntaxError",
        "text": "[eslint] Parsing error: Unexpected token <  src/Button.jsx:14:4",
    },
    # DependencyMissing
    {
        "label": "DependencyMissing",
        "text": "Error: Cannot find module 'lodash'\nRequire stack:\n- /workspace/src/helpers.js\n- /workspace/test/helpers.test.js",
    },
    {
        "label": "DependencyMissing",
        "text": "Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'zod' imported from /workspace/src/schema.ts",
    },
    {
        "label": "DependencyMissing",
        "text": "Module not found: Error: Can't resolve 'react-dom/client' in '/workspace/src'",
    },
    {
        "label": "DependencyMissing",
        "text": "npm ERR! code ELIFECYCLE\nnpm ERR! errno 1\nError: Cannot find module '@babel/core'",
    },
    {
        "label": "DependencyMissing",
        "text": "TS2307: Cannot find module 'express' or its corresponding type declarations.",
    },
    # TypeMismatch
    {
        "label": "TypeMismatch",
        "text": "src/user.ts:22:5 - error TS2322: Type 'string' is not assignable to type 'number'.\n22     age = name;\n       ~~",
    },
    {
        "label": "TypeMismatch",
        "text": "TypeError: Cannot read properties of undefined (reading 'map')\n    at renderList (/workspace/src/list.js:10:18)",
    },
    {
        "label": "TypeMismatch",
        "text": "error TS2345: Argument of type 'null' is not assignable to parameter of type 'string'.",
    },
    {
        "label": "TypeMismatch",
        "text": "TypeError: foo is not a function\n    at Object.<anonymous> (/workspace/test/math.test.js:12:12)",
    },
    # TestLogicFailure
    {
        "label": "TestLogicFailure",
        "text": "AssertionError [ERR_ASSERTION]: Expected values to be strictly equal:\n+ actual - expected\n+ 9\n- 20\n    at TestContext.<anonymous> (test/math.test.js:11:12)",
    },
    {
        "label": "TestLogicFailure",
        "text": "FAIL test/greeter.test.js\n  ● greets by name\n    expect(received).toBe(expected)\n    Expected: \"Hello, Ada\"\n    Received: \"Hello, undefined\"",
    },
    {
        "label": "TestLogicFailure",
        "text": "1) Calculator accumulates\n     AssertionError: expected 25 to equal 30\n      at Context.<anonymous> (test/math.test.js:21:12)",
    },
    {
        "label": "TestLogicFailure",
        "text": "not ok 2 - multiplies numbers\n  ---\n  error: |-\n    Expected 20 but got 9\n  ...",
    },
    # Phase 7 — more realistic CI / toolchain variants
    {
        "label": "SyntaxError",
        "text": "ERROR: Failed to compile.\n./src/App.tsx\nError: Expression expected. ts(1109)",
    },
    {
        "label": "SyntaxError",
        "text": "acorn: Unexpected token (15:8)\n    at pp.raise (/workspace/node_modules/acorn/dist/acorn.js)",
    },
    {
        "label": "DependencyMissing",
        "text": "Error: Cannot find module 'fastify'\nRequire stack:\n- /workspace/src/server.js",
    },
    {
        "label": "DependencyMissing",
        "text": "Unable to resolve \"@tanstack/react-query\" from \"src/hooks/useQuery.ts\"",
    },
    {
        "label": "DependencyMissing",
        "text": "go: module github.com/example/lib: not found",
    },
    {
        "label": "TypeMismatch",
        "text": "error TS2339: Property 'foo' does not exist on type 'Bar'.",
    },
    {
        "label": "TypeMismatch",
        "text": "TypeError: Cannot convert undefined or null to object\n    at Object.assign (<anonymous>)",
    },
    {
        "label": "TestLogicFailure",
        "text": "AssertionError: expected 'bench-mixed' to equal 'wrong-name'\n    at Context.<anonymous> (test/config.test.js:8:12)",
    },
    {
        "label": "TestLogicFailure",
        "text": "FAIL src/numbers.test.js\n  ● sums\n    expect(received).toBe(expected)\n    Expected: 6\n    Received: 6  // wrong message path — actually Received: 0",
    },
    {
        "label": "TestLogicFailure",
        "text": "× clamp (1 ms)\n  → expected 0 to deeply equal 10",
    },
    # Open-source / CI flavored (Phase 7 completeness)
    {
        "label": "DependencyMissing",
        "text": (
            "Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'ms' imported from "
            "/workspace/src/duration.js"
        ),
    },
    {
        "label": "DependencyMissing",
        "text": (
            "npm ERR! 404 Not Found - GET https://registry.npmjs.org/left-pad-typo "
            "- Not found"
        ),
    },
    {
        "label": "SyntaxError",
        "text": (
            "vite v5.4.0 building for production...\n"
            "✓ transforming...\n"
            "✗ Build failed in 120ms\n"
            "error during build:\n"
            "[vite:esbuild] Transform failed with 1 error:\n"
            "src/main.ts:12:8: ERROR: Expected \";\" but found \"function\""
        ),
    },
    {
        "label": "TypeMismatch",
        "text": (
            "src/index.ts:41:14 - error TS2304: Cannot find name 'window'.\n"
            "41   return window.localStorage.getItem(key);\n"
            "                ~~~~~~"
        ),
    },
    {
        "label": "TestLogicFailure",
        "text": (
            "FAIL packages/a/test/hello.test.js\n"
            "  ● greets\n"
            "    expect(received).toBe(expected)\n"
            "    Expected: \"hello, Ada\"\n"
            "    Received: \"hi, Ada\""
        ),
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT)
    args = parser.parse_args()

    existing = []
    if args.path.exists():
        existing = [json.loads(l) for l in args.path.read_text().splitlines() if l.strip()]

    seen = {(e["label"], e["text"]) for e in existing}
    added = 0
    for row in EXTRA:
        key = (row["label"], row["text"])
        if key in seen:
            continue
        existing.append(row)
        seen.add(key)
        added += 1

    with args.path.open("w", encoding="utf-8") as fh:
        for row in existing:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    from collections import Counter

    counts = Counter(r["label"] for r in existing)
    print(f"error_logs rows={len(existing)} added={added} by_label={dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
