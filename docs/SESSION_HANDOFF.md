# Session handoff — 2026-11-04 (restart here)

> **Read this first** in a new agent window. Then sync [GLOBAL.md](GLOBAL.md) + [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md).

## Where we are

| Item | State |
|------|--------|
| Product phases **0–10** | **Complete on `main`** (committed/pushed with this handoff) |
| Live languages | **All 8 v1.0 ids** — adapters + Docker sandboxes + fixtures |
| Corpus | **24 640** pairs (`data/datasets/corpus_manifest.json`) — **≥15K met** |
| Fixture gate | `rebirth eval --suite fixtures --lang all` → **8/8** |
| LoRA today | **JS/TS 7B only** (`data/models/lora_js_ts_7b`, trained on 1339 pairs) |
| **Next** | **Phases 12–14** — per-language LoRA ([PHASE_12_LORA_PER_LANGUAGE.md](phases/PHASE_12_LORA_PER_LANGUAGE.md)) |
| **Not yet** | Phase 11 launch / `v1.0.0` — **deferred** until after LoRA waves |

## Git / GitHub

| Item | Value |
|------|--------|
| Branch | `main` (Phases 8–10 landed) |
| Prior merge | PR #2 @ `5f1540e` (Phases 6–7) |
| Tag | `v0.2.0-mvp` on origin (JS/TS MVP) |

## Exact next step

1. **Start Phase 12** — training harness + Python LoRA (+ optional JS/TS refresh on full ~9K).  
2. **Phase 13** — Go / Java / C# / Rust LoRAs.  
3. **Phase 14** — C++ / Swift + PEFT lang routing + eval.  
4. **Then** Phase 11 — bootstrap + tag **v1.0.0**.

Do **not** start Phase 11 packaging until LoRA waves finish or the user explicitly skips them.

## Phase 10 evidence (locked)

- `rebirth eval --suite fixtures --lang all` → **8/8** (`data/reports/eval_fixture_sandboxes.json`)  
- `rebirth report` works on batch + eval JSON  
- Migrate writes `data/runs/<id>/summary.json` (redacted; `data/runs/` gitignored)  
- Corpus: **24640** total, `met_target=true`  

| Lang | Pairs |
|------|------:|
| js_ts | 9103 |
| python | 5246 |
| java | 2649 |
| csharp | 1562 |
| go | 1528 |
| swift | 1535 |
| cpp | 1512 |
| rust | 1505 |

## Do not redo

Phases 0–10 adapters / golden JS benches / 15K mine. Prefer `REBIRTH_EMBEDDER=hash` with Ollama 14B resident; unload Ollama before PEFT train/migrate.
