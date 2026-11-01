# Phase 4 — LangGraph Integration (The Brain)

> Status: **PLANNED**  
> Depends on: Phases 1–3 modules callable from Python

---

## 1. Objectives

- Define `MigrationState` TypedDict and LangGraph nodes/edges.  
- Wire RAG → generate → sandbox → classify → conditional route.  
- Prove closed loop on **one** fixture file: fail → classify → rewrite → pass (or clean human-review halt).

---

## 2. Planned nodes

| Node | Action |
|------|--------|
| `retrieve_context` | AST + RAG |
| `generate_code` | Ollama / LoRA |
| `execute_tests` | Docker sandbox |
| `classify_error` | Sklearn |
| `install_dependency` | Deterministic (no LLM) |
| `success` / `halt_human_review` | Terminal |

### Guards

- `iteration_count > REBIRTH_MAX_ITERATIONS` (default 5) → human review  
- `error_class == DependencyMissing` → install node  

---

## 3. Exit criteria

- [ ] Graph compiles and runs from CLI  
- [ ] Documented successful single-file migration trace  
- [ ] Failures beyond max iterations halt cleanly  
- [ ] `PHASE_4` + `GLOBAL` updated with traces  

---

## 4. What shipped / How to run / Sample outputs / Decision log

_Filled when Phase 4 is implemented._
