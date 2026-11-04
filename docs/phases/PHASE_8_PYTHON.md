# Phase 8 — Adapter Factory + Python Vertical

> Status: **PLANNED** (next)  
> Depends on: Phase 7 **complete**  
> Full plan: [PRODUCTION_ROADMAP.md](../PRODUCTION_ROADMAP.md) § Phase 8  
> Last docs refresh: 2026-11-03

## Objectives

- Lock the **repeatable language playbook** (`ADDING_A_LANGUAGE.md` + adapter registry).  
- Ship `PythonAdapter` + pytest Docker sandbox + fixture.  
- Auto-detect project language; closed-loop migrate on Python.  
- Mine first non-JS pairs toward the **15K+** multi-lang corpus.  
- Prove multi-language without rewriting the LangGraph core.

## Target languages context

v1.0 locked set: **JS/TS, Python, Go, Java, C++, C#, Rust, Swift**.  
Phase 8 ships Python and the factory; Phases 9–10 instantiate the rest.

## Exit criteria

- [ ] Adapter registry + `rebirth adapters` + `docs/adapters/ADDING_A_LANGUAGE.md`  
- [ ] Python fixture migrate ≤3 iterations on Ollama  
- [ ] `migrate-repo` works on `fixtures/python_sample`  
- [ ] Doctor checks Python sandbox image  
- [ ] Python migration pairs mined; count recorded in `DATASET.md`  
- [ ] Optional: tag `v0.3.0-js-python-preview`

## Playbook (must work for Phase 9+)

1. Adapter implementing `LanguageAdapter`  
2. Docker sandbox + doctor check  
3. Fixture + broken seed → tests pass ≤3 iterations  
4. Dataset mine filter → `data/datasets/<lang>_migration.jsonl`  
5. Behavioral eval case  
6. Tree-sitter parse path  

## What shipped / how to run / samples

_Filled when Phase 8 is implemented._
