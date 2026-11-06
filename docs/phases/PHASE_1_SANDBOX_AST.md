# Phase 1 — Environment & Sandbox Baseline (AST + Docker)

> Status: **COMPLETE**  
> Depends on: Phase 0 complete  
> Last docs refresh: 2026-11-04 (status unchanged — complete)  
> Project now: Phases **0–10 complete** on `main`; next = Phase **11** per-lang LoRA — [SESSION_HANDOFF.md](../SESSION_HANDOFF.md)

---

## 1. Objectives

- Programmatically spin ephemeral Docker containers, run tests on mounted files, capture stdout/stderr/exit_code, destroy containers.  
- Parse JS/TS with Tree-sitter; emit structured chunks (imports, functions, classes).  
- Introduce a `LanguageAdapter` interface; ship the first `js_ts` adapter.  
- Provide a tiny fixture project under `fixtures/js_ts_sample/`.

---

## 2. Planned / shipped deliverables

| Path | Responsibility | Status |
|------|----------------|--------|
| `docker/js_ts_sandbox/Dockerfile` | Node 22 slim sandbox image | **Shipped** (`rebirth-ai/js-ts-sandbox:latest` built) |
| `src/rebirth_ai/sandbox/runner.py` | Docker SDK orchestration | **Shipped** (copies project to tempdir before mount) |
| `src/rebirth_ai/ast/chunker.py` | Tree-sitter extraction | **Shipped** (method extraction improved) |
| `src/rebirth_ai/adapters/base.py` | Protocol + `SandboxSpec` | **Shipped** |
| `src/rebirth_ai/adapters/js_ts.py` | JS/TS implementation | **Shipped** |
| `fixtures/js_ts_sample/` | Minimal npm + `node --test` project | **Shipped** (restore after overlay incident) |
| `tests/test_sandbox.py`, `tests/test_ast.py` | Integration coverage | **Shipped** |
| CLI `rebirth parse` / `rebirth sandbox` | Operator demos | **Shipped** |

---

## 3. Exit criteria

- [x] `docker build` produces sandbox image  
- [x] Python API returns logs + exit code for a deliberate failing test (observed via CLI)  
- [x] Same API returns exit 0 for a passing file (verified after tempdir isolation)  
- [x] AST returns structured chunks for ≥3 fixture files (imports/functions/class/methods)  
- [x] Adapter interface importable and used by sandbox runner  
- [x] This doc + `GLOBAL.md` updated with current status  

---

## 4. What shipped

### Contracts

- `LanguageAdapter`: `sandbox_spec()`, `parse()`, `test_command_for()`, `is_source_file()`  
- `SandboxRunner.run_project(project_dir, relative_test=, overlay_file=, overlay_dest=)` → `SandboxResult`  
- **Safety:** always `copytree` into a temp dir before bind-mount; overlays never write through to the caller's tree (early bug: `put_archive` on bind mount corrupted `fixtures/.../math.js` as `nobody`).

### How to run

```bash
cd "/home/hugobele/Documents/Personal Project/Rebirth AI"
source .venv/bin/activate
docker build -t rebirth-ai/js-ts-sandbox:latest docker/js_ts_sandbox
pytest -q
rebirth parse fixtures/js_ts_sample/src/math.js
rebirth sandbox fixtures/js_ts_sample -t test/math.test.js
rebirth sandbox fixtures/js_ts_sample -t test/math.failing.test.js
rebirth doctor
```

### Sample outputs (captured during Phase 1)

```text
# doctor
Sandbox image: ok (rebirth-ai/js-ts-sandbox:latest)
Ollama loaded: qwen2.5-coder:14b-instruct-q6_K 12.6/12.6GB → GPU (100% VRAM)

# failing sandbox (expected)
FAIL exit=1  cmd: node --test test/math.failing.test.js
```

### Decision log

| Decision | Detail |
|----------|--------|
| Test runner | Node built-in `node --test` (no Jest install in image) |
| Sandbox network | `network_mode=none` for Phase 1 |
| Overlay strategy | Host-side copy + overlay in tempdir (not `put_archive` onto bind mount) |
| Default image | `rebirth-ai/js-ts-sandbox:latest` |

---

## 5. Known gaps / open work

- Multi-language sandboxes deferred.  
- Network-install inside sandbox deferred to Phase 4.  
- `npm test` glob fixed to `test/math.test.js` only (failing suite is opt-in via `-t`).

---

## 6. Handoff to Phase 2

Once exit criteria are green: index AST chunks into SQLite embeddings; train error classifier on synthetic logs.
