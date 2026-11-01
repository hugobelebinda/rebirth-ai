# Phase 1 — Environment & Sandbox Baseline (AST + Docker)

> Status: **PLANNED** (not started)  
> Depends on: Phase 0 complete

---

## 1. Objectives

- Programmatically spin ephemeral Docker containers, run tests on mounted files, capture stdout/stderr/exit_code, destroy containers.  
- Parse JS/TS with Tree-sitter; emit structured chunks (imports, functions, classes).  
- Introduce a `LanguageAdapter` interface; ship the first `js_ts` adapter.  
- Provide a tiny fixture project under `fixtures/js_ts_sample/`.

---

## 2. Planned deliverables

| Path | Responsibility |
|------|----------------|
| `docker/js_ts_sandbox/Dockerfile` | Node LTS image + project test deps pattern |
| `src/rebirth_ai/sandbox/runner.py` | Docker SDK orchestration |
| `src/rebirth_ai/ast/chunker.py` | Tree-sitter extraction |
| `src/rebirth_ai/adapters/base.py` | Protocol: `parse`, `test_command`, `sandbox_image` |
| `src/rebirth_ai/adapters/js_ts.py` | JS/TS implementation |
| `fixtures/js_ts_sample/` | Minimal npm project with tests |
| `tests/test_sandbox.py`, `tests/test_ast.py` | Integration coverage |

---

## 3. Exit criteria (to check when done)

- [ ] `docker build` produces sandbox image  
- [ ] Python API returns logs + exit code for a deliberate failing test  
- [ ] Same API returns exit 0 for a passing file  
- [ ] AST returns structured chunks for ≥3 fixture files  
- [ ] Adapter interface importable and used by sandbox runner  
- [ ] This doc + `GLOBAL.md` updated with commands, sample outputs, decisions  

---

## 4. What shipped / How to run / Sample outputs / Decision log

_Filled when Phase 1 is implemented._

---

## 5. Known gaps (seed)

- Multi-language sandboxes deferred.  
- Network-install inside sandbox policy TBD (Phase 4 install node).  
- Volume bind security: only mount workspace paths under fixtures or temp dirs.
