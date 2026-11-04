# Phase 6 — Repo Hardening (JS/TS Production Path)

> Status: **COMPLETE (100%)**  
> Depends on: Phase 5 complete (PR #1 merged 2026-11-02)  
> Full plan: [PRODUCTION_ROADMAP.md](../PRODUCTION_ROADMAP.md) § Phase 6  
> Last docs refresh: 2026-11-03

## Objectives

- Harden `migrate-repo` for non-toy trees (still JS/TS only).  
- Import-aware file ordering, multi-root `--under`, `.gitignore` + deny list.  
- VRAM-safe RAG / PEFT policy; doctor warning.  
- Whole-file chat decode; `--dry-run`; stronger PR UX (`--draft`, auto base).

## What shipped

| Area | Detail |
|------|--------|
| Discovery | `src/rebirth_ai/graph/discovery.py` — multi-root, gitignore + `!` negation, deny dirs, skip tests/seeds, import topo-order |
| Test matrix | [docs/adapters/TEST_MAPPING.md](../adapters/TEST_MAPPING.md) |
| Batch | `migrate_repository(..., dry_run=True)`; unified diffs on `--no-write` |
| CLI | `--under src,lib,app`, `--dry-run`, `--draft`, auto `--pr-base` |
| RAG | `embedder_dim` + `embedder_name`; refuse mismatch |
| PEFT | Agent path chat + hash RAG; token caps by file size |
| Doctor | Warns when Ollama resident **or** `REBIRTH_GENERATOR=peft` while loaded |
| PR | Log excerpts; `--draft`; `detect_default_branch()` |
| Tests | `tests/test_discovery.py`, RAG dim, PEFT decode |

## How to run

```bash
source .venv/bin/activate
rebirth doctor
rebirth migrate-repo fixtures/js_ts_sample --dry-run
rebirth migrate-repo fixtures/bench_monorepo --under packages/a,packages/b --dry-run
```

## Exit criteria

- [x] PR #1 merged  
- [x] Topological migrate order on fixture  
- [x] `--dry-run` for migrate-repo  
- [x] Doctor warns on VRAM stacking / PEFT+Ollama  
- [x] Behavioral sandbox 2/2  
- [x] Test mapping matrix documented  
- [x] gitignore negation supported (basic `!`)  
- [x] WIP synced via commit/PR (Phase 6–7 closeout)

## Next

**[Phase 7](PHASE_7_REAL_WORLD_JS.md)** — complete → **[Phase 8](PHASE_8_PYTHON.md)**
