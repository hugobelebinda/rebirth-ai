# Phase 4 — LangGraph Integration (The Brain)

> Status: **COMPLETE**  
> Depends on: Phases 1–3 modules callable from Python  
> Last docs refresh: 2026-11-04  
> Project now: Phases **0–10 complete** on `main`; next = Phases **12–14** per-lang LoRA — [SESSION_HANDOFF.md](../SESSION_HANDOFF.md)

---

## 1. Objectives

- Define `MigrationState` TypedDict and LangGraph nodes/edges.  
- Wire RAG → generate → sandbox → classify → conditional route.  
- Prove closed loop on **one** fixture file: fail → classify → rewrite → pass (or clean human-review halt).

---

## 2. What shipped

| Path | Role |
|------|------|
| `src/rebirth_ai/graph/state.py` | `MigrationState` + `REBIRTH_MAX_ITERATIONS` |
| `src/rebirth_ai/graph/nodes.py` | retrieve / generate / execute / classify / install / success / halt |
| `src/rebirth_ai/graph/routing.py` | Conditional edges (`after_retrieve`, `after_tests`, `after_classify`, `after_install`) |
| `src/rebirth_ai/graph/app.py` | `build_migration_graph()`, `initial_state()` |
| `src/rebirth_ai/cli.py` → `rebirth migrate` | Closed-loop CLI |
| `src/rebirth_ai/sandbox/runner.py` | `overlay_text` + `network_mode` for install+test |
| `fixtures/migration_seeds/math.broken.js` | Intentionally wrong `multiply` draft for E2E |
| `tests/test_graph_routing.py` | Graph compile + routing unit tests |

### Graph topology

```
START → retrieve_context
          ├─(test_first)→ execute_tests
          └─(else)→ generate_code → execute_tests
execute_tests → success | classify_error
classify_error → halt_human_review | install_dependency | generate_code
install_dependency → execute_tests | generate_code | halt
success / halt_human_review → END
```

### Guards

- `iteration_count >= max_iterations` (default 5) → `halt_human_review`  
- `error_class == DependencyMissing` → queue package into `extra_packages`, re-test with network  
- Already-queued missing package → `generate_code` (avoid install loops)  
- Temp sandbox copies discard `node_modules` — when extras exist, `npm install && test` in one bridged container  

---

## 3. How to run

```bash
source .venv/bin/activate
# Optional: index fixture for RAG context
rebirth index fixtures/js_ts_sample

# Closed loop: broken seed → fail → classify → rewrite → pass
rebirth migrate fixtures/js_ts_sample \
  --file src/math.js \
  --test test/math.test.js \
  --seed-draft fixtures/migration_seeds/math.broken.js \
  --max-iterations 3 \
  --out /tmp/rebirth_math_fixed.js \
  --no-write

pytest tests/test_graph_routing.py -q
```

---

## 4. Sample E2E output (2026-10-31)

```
status=success iterations=1
error_class=None
  • retrieve_context: 4 chunk(s)
  • execute_tests: FAIL exit=1 in 0.18s
  • classify_error: TestLogicFailure
  • generate_code: iteration=1 model=qwen2.5-coder:14b-instruct-q6_K chars=683
  • execute_tests: PASS exit=0 in 0.18s
  • success: tests passed
wrote /tmp/rebirth_math_fixed.js
```

Model fixed `multiply` from `a + b` → `a * b` in one iteration (~12s wall including RAG load).

---

## 5. Exit criteria

- [x] Graph compiles and runs from CLI (`rebirth migrate`)  
- [x] Documented successful single-file migration trace  
- [x] Failures beyond max iterations halt cleanly (routing unit-tested)  
- [x] `PHASE_4` + `GLOBAL` updated with traces  

---

## 6. Known gaps

- Install path re-downloads extras every test run (no host `node_modules` persistence).  
- RAG during migrate loads SentenceTransformers if `data/embeddings.db` exists (cold start warning if no HF token).  
- Phase 5 still needed for multi-file walk + GitHub PR open.  
- `--backend peft` defaults to **7B** adapter when present. 0.5B closed-loop was unreliable (file duplication). **7B PEFT closed-loop proven** 2026-11-01 on broken-math (1 iter PASS); auto hash-RAG to avoid ST+7B VRAM stack.

---

## 7. Decision log

| Date | Decision |
|------|----------|
| 2026-10-31 | Promote `langgraph` / `langchain-core` to core deps |
| 2026-10-31 | `--seed-draft` implies `--test-first` for fail→fix demos |
| 2026-10-31 | Keep migration seeds outside `fixtures/js_ts_sample/` so indexing stays clean |
| 2026-10-31 | DependencyMissing queues packages into state; re-test before another LLM call |
| 2026-11-01 | Keep migrate default on Ollama 14B; PEFT optional for specialization experiments |
