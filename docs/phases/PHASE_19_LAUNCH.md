# Phase 19 — Packaging, UX & Official v1.0 Launch

> Status: **PLANNED — last phase** (was "Phase 11" before the 2026-11-05 renumbering). Runs after Phases 12–18.  
> Previous: [PHASE_18_MLOPS_MONITORING.md](PHASE_18_MLOPS_MONITORING.md)  
> Depends on: Phase 10 complete **and** LoRA specialization waves ([PHASE_11_LORA_PER_LANGUAGE.md](PHASE_11_LORA_PER_LANGUAGE.md)) — or explicit user skip of LoRA  
> Full plan: [PRODUCTION_ROADMAP.md](../PRODUCTION_ROADMAP.md) § Phase 19  
> Last docs refresh: 2026-11-04  
> Handoff: [SESSION_HANDOFF.md](../SESSION_HANDOFF.md)

## Why deferred

Phases 0–10 delivered **8 language verticals + ≥15K corpus + quality CLI**.  
Next priority is **train a LoRA adapter per language** (Phase 11, done 2026-11-05; 14B upgrade in Phase 15). Official **`v1.0.0`** packaging waits until those adapters exist (or you choose an Ollama-only launch).

## Objectives (when resumed)

- Bootstrap script installs tooling and **all** sandbox images.  
- README / docs cover every target language (+ PEFT adapter map if shipped).  
- Tag **`v1.0.0`** official launch (not preview).

## Locked launch surface

Languages: **JS/TS, Python, Go, Java, C++, C#, Rust, Swift**  
Corpus: **≥15K** mined multi-language migration pairs (**24 640** already)  
Quality: fixture sandbox target **≥90%** per language; truthful multi-file PR reports  
Preferred brain: Ollama 14B default + **per-lang PEFT** when adapters exist

## Exit criteria

- [ ] Phases 11–18 complete **or** written skip decision  
- [ ] `scripts/bootstrap.sh` + doctor green path documented for second machine  
- [ ] Fixture E2E for all 8 languages  
- [ ] ≥1 real JS/TS repo + ≥1 real non-JS repo migrate report  
- [ ] `migrate-repo --open-pr` on throwaway branch  
- [ ] Security checklist (secrets, sandbox network default `none`)  
- [ ] Corpus ≥15K documented in `DATASET.md`  
- [ ] Tag `v1.0.0`  

## What shipped / how to run / samples

_Filled when Phase 19 is implemented._
