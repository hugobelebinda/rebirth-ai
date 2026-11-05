# Phase 11 — Packaging, UX & Official v1.0 Launch

> Status: **DEFERRED** — run **after** Phases **12–14** (per-language LoRA)  
> Depends on: Phase 10 complete **and** LoRA specialization waves ([PHASE_12_LORA_PER_LANGUAGE.md](PHASE_12_LORA_PER_LANGUAGE.md)) — or explicit user skip of LoRA  
> Full plan: [PRODUCTION_ROADMAP.md](../PRODUCTION_ROADMAP.md) § Phase 11  
> Last docs refresh: 2026-11-04  
> Handoff: [SESSION_HANDOFF.md](../SESSION_HANDOFF.md)

## Why deferred

Phases 0–10 delivered **8 language verticals + ≥15K corpus + quality CLI**.  
Next priority is **train a LoRA adapter per language** (Phases 12–14). Official **`v1.0.0`** packaging waits until those adapters exist (or you choose an Ollama-only launch).

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

- [ ] Phases 12–14 complete **or** written skip decision  
- [ ] `scripts/bootstrap.sh` + doctor green path documented for second machine  
- [ ] Fixture E2E for all 8 languages  
- [ ] ≥1 real JS/TS repo + ≥1 real non-JS repo migrate report  
- [ ] `migrate-repo --open-pr` on throwaway branch  
- [ ] Security checklist (secrets, sandbox network default `none`)  
- [ ] Corpus ≥15K documented in `DATASET.md`  
- [ ] Tag `v1.0.0`  

## What shipped / how to run / samples

_Filled when Phase 11 is implemented._
