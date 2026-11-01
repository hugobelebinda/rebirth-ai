# JS/TS sample fixture

Tiny Node project used by Phase 1 for:

- Tree-sitter AST chunking (`src/math.js`, `src/format.js`, `src/greeter.ts`)
- Docker sandbox pass/fail verification

```bash
npm test                                 # runs test/math.test.js only
node --test test/math.failing.test.js    # exits non-zero on purpose
```

Via Rebirth CLI (preferred):

```bash
rebirth sandbox fixtures/js_ts_sample -t test/math.test.js
rebirth sandbox fixtures/js_ts_sample -t test/math.failing.test.js
rebirth parse fixtures/js_ts_sample/src/math.js
```

**Note:** The sandbox runner copies this tree into a tempdir before mounting. Do not rely on in-container writes to update this fixture.

