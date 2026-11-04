# JS/TS test mapping matrix (Phase 6+)

> Status: **shipped** with Phase 6 (2026-11-03). Project handoff: [SESSION_HANDOFF.md](../SESSION_HANDOFF.md).

`infer_test_path(project_dir, relative_source)` tries candidates in order and returns the first existing file.

| Pattern | Example source | Example test |
|---------|----------------|--------------|
| Root `test/*.test.js` | `src/math.js` | `test/math.test.js` |
| Root `test/*.spec.js` | `src/util.js` | `test/util.spec.js` |
| Root `tests/` | `src/foo.ts` | `tests/foo.test.ts` |
| Root `__tests__/` | `src/bar.js` | `__tests__/bar.test.js` |
| Co-located `__tests__/` | `src/x.js` | `src/__tests__/x.test.js` |
| Sibling `*.test.js` | `lib/x.js` | `lib/x.test.js` |
| Package `src/` → `test/` | `packages/a/src/hello.js` | `packages/a/test/hello.test.js` |
| `package.json` scripts.test | only if a listed path matches the **stem** | e.g. `node --test test/math.test.js` → maps `math` only |

If nothing matches and `skip_without_tests=True`, the file is `skipped_no_test`.
