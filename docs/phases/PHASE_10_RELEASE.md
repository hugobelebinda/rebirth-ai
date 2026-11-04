# Phase 10 — Language Wave B (C++, Swift) + Quality + 15K Corpus

> Status: **PLANNED**  
> Depends on: Phase 9  
> Full plan: [PRODUCTION_ROADMAP.md](../PRODUCTION_ROADMAP.md) § Phase 10  
> Last docs refresh: 2026-11-02  
> Note: Official packaging / `v1.0` tag moved to [PHASE_11_LAUNCH.md](PHASE_11_LAUNCH.md).

## Objectives

- Ship **C++** and **Swift** adapters (complete the locked 8-language set).  
- Unify quality gates: `rebirth eval --lang all`, held-out suites, ≥90% fixture sandbox target.  
- Hit **≥15 000** mined migration pairs across all languages.  
- Structured run logs with redaction (`data/runs/`).

## Languages in this phase

| Language | Adapter id | Sandbox | Tests |
|----------|------------|---------|-------|
| C++ | `cpp` | `rebirth-ai/cpp-sandbox` | CMake + ctest / gtest |
| Swift | `swift` | `rebirth-ai/swift-sandbox` | `swift test` |

Swift: prefer Linux Swift toolchain image; if host-only, document macOS path in this phase file when implemented.

## Exit criteria

- [ ] C++ + Swift fixture E2E pass on Ollama  
- [ ] All **8** adapters registered and doctor-checked  
- [ ] `rebirth eval --lang all` documented; fixture sandbox ≥90% per language (or waived with issue)  
- [ ] Combined corpus **≥15K** pairs; `DATASET.md` + stats JSON updated  
- [ ] `data/runs/` artifacts with secret redaction  
- [ ] `rebirth report` for batch/eval JSON  

## What shipped / how to run / samples

_Filled when Phase 10 is implemented._
