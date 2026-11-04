# Phase 11 — Packaging, UX & Official v1.0 Launch

> Status: **PLANNED**  
> Depends on: Phase 10 (all 8 languages + ≥15K pairs + quality gates)  
> Full plan: [PRODUCTION_ROADMAP.md](../PRODUCTION_ROADMAP.md) § Phase 11  
> Last docs refresh: 2026-11-03  
> Project now: Phases 0–7 done; awaiting PR #2 merge then Phase 8+ — [SESSION_HANDOFF.md](../SESSION_HANDOFF.md)

## Objectives

- Bootstrap script installs tooling and **all** sandbox images.  
- README / docs cover every target language.  
- Tag **`v1.0.0`** official launch (not preview).

## Locked launch surface

Languages: **JS/TS, Python, Go, Java, C++, C#, Rust, Swift**  
Corpus: **≥15K** mined multi-language migration pairs  
Quality: fixture sandbox target **≥90%** per language; truthful multi-file PR reports

## Exit criteria

- [ ] `scripts/bootstrap.sh` + doctor green path documented for second machine  
- [ ] Fixture E2E for all 8 languages  
- [ ] ≥1 real JS/TS repo + ≥1 real non-JS repo migrate report  
- [ ] `migrate-repo --open-pr` on throwaway branch  
- [ ] Security checklist (secrets, sandbox network default `none`)  
- [ ] Corpus ≥15K documented in `DATASET.md`  
- [ ] Tag `v1.0.0`  

## What shipped / how to run / samples

_Filled when Phase 11 is implemented._
