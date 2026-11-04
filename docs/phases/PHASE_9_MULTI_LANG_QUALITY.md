# Phase 9 — Language Wave A (Go, Java, C#, Rust)

> Status: **PLANNED**  
> Depends on: Phase 8 (adapter factory + Python playbook locked)  
> Full plan: [PRODUCTION_ROADMAP.md](../PRODUCTION_ROADMAP.md) § Phase 9  
> Last docs refresh: 2026-11-02

## Objectives

- Instantiate the Phase 8 playbook for **Go, Java, C#, and Rust**.  
- Extend CLI `--lang` / auto-detect for all wave-A ids.  
- Grow the mined corpus (target ~1–2K pairs per language) toward **15K+** total.  
- Start shared multi-lang CLI surface (`adapters`, language-aware prompts).

## Languages in this phase

| Language | Adapter id | Sandbox | Tests |
|----------|------------|---------|-------|
| Go | `go` | `rebirth-ai/go-sandbox` | `go test` |
| Java | `java` | `rebirth-ai/java-sandbox` | Maven/Gradle + JUnit |
| C# | `csharp` | `rebirth-ai/csharp-sandbox` | `dotnet test` |
| Rust | `rust` | `rebirth-ai/rust-sandbox` | `cargo test` |

## Exit criteria

- [ ] All four wave-A languages: adapter + image + fixture E2E ≤3 iterations on Ollama  
- [ ] `rebirth adapters` lists `js_ts`, `python`, `go`, `java`, `csharp`, `rust`  
- [ ] Per-language pair counts in `DATASET.md`; corpus trending to 15K  
- [ ] Doctor builds/checks all wave-A sandbox images  
- [ ] Tree-sitter parse green for each language  

## What shipped / how to run / samples

_Filled when Phase 9 is implemented._
