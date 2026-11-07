# Phase 9 — Full remaining language wave (Go, Java, C#, Rust, C++, Swift)

> Status: **COMPLETE** (2026-11-03) — includes former Phase 10 wave-B languages  
> Depends on: Phase 8 (adapter factory + Python playbook)  
> Full plan: [PRODUCTION_ROADMAP.md](../PRODUCTION_ROADMAP.md) § Phase 9–10  
> Playbook: [ADDING_A_LANGUAGE.md](../adapters/ADDING_A_LANGUAGE.md)  
> Handoff: [SESSION_HANDOFF.md](../SESSION_HANDOFF.md) — Phases 0–10 done; next LoRA Phase 11

## Objectives

- Instantiate the Phase 8 playbook for **Go, Java, C#, Rust, C++, and Swift** (all remaining v1.0 languages in one wave).  
- Extend CLI `--lang` / auto-detect for all eight adapter ids.  
- Grow the mined corpus toward **15K+** total.  
- Prove fixture closed-loop migrate on Ollama for each language.

## Languages in this phase

| Language | Adapter id | Sandbox | Tests |
|----------|------------|---------|-------|
| Go | `go` | `rebirth-ai/go-sandbox` | `go test` |
| Java | `java` | `rebirth-ai/java-sandbox` | Maven + JUnit |
| C# | `csharp` | `rebirth-ai/csharp-sandbox` | `dotnet test` |
| Rust | `rust` | `rebirth-ai/rust-sandbox` | `cargo test` |
| C++ | `cpp` | `rebirth-ai/cpp-sandbox` | CMake + ctest |
| Swift | `swift` | `rebirth-ai/swift-sandbox` | `swift test` (Linux toolchain) |

## Exit criteria

- [x] All six languages: adapter + image + fixture E2E ≤3 iterations on Ollama  
- [x] `rebirth adapters` lists `js_ts`, `python`, `go`, `java`, `csharp`, `rust`, `cpp`, `swift`  
- [x] Per-language pair counts in `DATASET.md` (completed in Phase 10 — corpus 24 640)  
- [x] Doctor checks all sandbox images  
- [x] Tree-sitter parse green for each language  

## What shipped

| Path | Role |
|------|------|
| `src/rebirth_ai/adapters/{go_lang,java_lang,csharp,rust,cpp,swift}.py` | Adapters |
| `docker/{go,java,csharp,rust,cpp,swift}_sandbox/` | Images |
| `fixtures/{go,java,csharp,rust,cpp,swift}_sample/` | Fixtures + broken seeds |
| `src/rebirth_ai/adapters/registry.py` | Full 8-id registry + detection |
| Tree-sitter deps | go/java/c-sharp/rust/cpp/swift in `pyproject.toml` |

## How to run

```bash
for id in go java csharp rust cpp swift; do
  docker build -t rebirth-ai/${id}-sandbox:latest docker/${id}_sandbox
done
rebirth adapters
rebirth doctor

rebirth sandbox fixtures/go_sample --lang go -t ./src
rebirth migrate fixtures/go_sample --file src/mathops.go --test ./src \
  --seed-draft fixtures/go_sample/seeds/mathops.broken.go --lang go
# … similarly for java / csharp / rust / cpp / swift (see fixture READMEs)
```

## Sample outputs (verified)

| Lang | Sandbox | Migrate |
|------|---------|---------|
| Go | PASS | success **1** iter |
| Java | PASS | success **1** iter |
| C# | PASS (broken overlay FAIL) | re-verified via Phase 10 fixture suite (8/8) |
| Rust | PASS | success **1** iter |
| C++ | PASS | success **1** iter |
| Swift | PASS | success **1** iter |

## Known gaps

- C# solution restore is flaky; fixture uses `dotnet test MathOps.Tests/…` fallback.  
- Swift uses **Linux** `swift:5.10-jammy` image (not macOS host toolchain).  
- Java/C# sandboxes use `network_mode=bridge` for first-time package restore.  
- Mining CommitPackFT for all six may be incomplete until datasets are written; do not overwrite `js_ts_migration.jsonl`.  
- Phase 10 quality CLI / 15K gate still open (languages pulled forward).

## Decision log

| Date | Decision |
|------|----------|
| 2026-11-03 | User requested Go/Java/C#/C++/Rust/Swift together → fold wave B into Phase 9 |
| 2026-11-03 | Sandbox runner uses non-login `sh -c` so Go/Rust PATH survives |
| 2026-11-03 | Java sandbox runs as UID `1000` (base image UID collision) |
