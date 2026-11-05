# JS/TS sample fixture

> Still used for smoke; Phase 6–7 benches + eval_suite. Project: Phases 0–10 done; next = LoRA 12–14 — [SESSION_HANDOFF.md](../../docs/SESSION_HANDOFF.md).

Tiny Node project used by Phases 1–4 for:

- Tree-sitter AST chunking (`src/math.js`, `src/format.js`, `src/greeter.ts`)
- Docker sandbox pass/fail verification
- LangGraph closed-loop migrate (with seed under `fixtures/migration_seeds/`)

```bash
npm test                                 # runs test/math.test.js only
node --test test/math.failing.test.js    # exits non-zero on purpose
```

Via Rebirth CLI (preferred):

```bash
rebirth sandbox fixtures/js_ts_sample -t test/math.test.js
rebirth sandbox fixtures/js_ts_sample -t test/math.failing.test.js
rebirth parse fixtures/js_ts_sample/src/math.js

# Phase 4 closed loop (Ollama 14B default)
rebirth migrate fixtures/js_ts_sample \
  --seed-draft fixtures/migration_seeds/math.broken.js --no-write
```

**Note:** The sandbox runner copies this tree into a tempdir before mounting. Do not rely on in-container writes to update this fixture.

