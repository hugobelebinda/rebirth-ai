# Phase 10 — Quality gates + ≥15K corpus

> Status: **COMPLETE** (2026-11-04)  
> Depends on: Phase 9 (all 8 language adapters already live)  
> Full plan: [PRODUCTION_ROADMAP.md](../PRODUCTION_ROADMAP.md) § Phase 10  
> Handoff: [SESSION_HANDOFF.md](../SESSION_HANDOFF.md)  

> Note: C++/Swift adapters shipped in Phase 9.  
> **Next:** per-language LoRA — [PHASE_12_LORA_PER_LANGUAGE.md](PHASE_12_LORA_PER_LANGUAGE.md).  
> Official packaging / `v1.0` tag ([PHASE_11_LAUNCH.md](PHASE_11_LAUNCH.md)) is **deferred** until after LoRA waves.

## Objectives

- Unify quality gates: `rebirth eval --lang all` / `--suite fixtures`, ≥90% fixture sandbox.  
- Hit **≥15 000** mined migration pairs across all languages.  
- Structured run logs with redaction (`data/runs/`).  
- `rebirth report` for batch/eval JSON.

## Exit criteria

- [x] C++ + Swift fixture E2E (already Phase 9; re-confirmed in fixtures suite)  
- [x] All **8** adapters registered and doctor-checked  
- [x] `rebirth eval --suite fixtures --lang all` → **8/8** (100% ≥ 90%)  
- [x] Combined corpus **≥15K** — **24 640** pairs (`data/datasets/corpus_manifest.json`)  
- [x] `data/runs/` artifacts with secret redaction  
- [x] `rebirth report` for batch/eval JSON  

## What shipped

| Path | Role |
|------|------|
| `src/rebirth_ai/redact.py` | Token / key redaction |
| `src/rebirth_ai/runs.py` | JSONL + summary under `data/runs/` |
| `src/rebirth_ai/report_view.py` | Pretty printer |
| `rebirth report <json>` | CLI |
| `rebirth eval --suite fixtures --lang all` | Multi-lang sandbox gate |
| `training/eval_fixture_sandboxes.py` | Fixture runner |
| `training/build_corpus_manifest.py` | Manifest + optional concat |
| `mine_migration_pairs.py --mode corpus` | License+quality fill toward 15K |

## How to run

```bash
# Multi-lang fixture sandbox gate (≥90%)
rebirth eval --suite fixtures --lang all --out data/reports/eval_fixture_sandboxes.json
rebirth report data/reports/eval_fixture_sandboxes.json

# Corpus manifest
python training/build_corpus_manifest.py --concat data/datasets/all_migration.jsonl

# Top-up mine (example)
python training/mine_migration_pairs.py --langs go --mode corpus --max-pairs 1500 --require-parse

# Migrate writes redacted run log
export REBIRTH_EMBEDDER=hash
rebirth migrate fixtures/python_sample --file src/math_ops.py --test tests/test_math_ops.py \
  --seed-draft fixtures/python_sample/seeds/math_ops.broken.py --lang python
# → data/runs/<id>/summary.json
```

## Sample outputs

- Fixture eval: `passed=8 total=8 rate=1.0`  
- Corpus: `total=24640 met_target=true`  
- Run log path printed after migrate  

## Corpus mix (2026-11-04)

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
| **Total** | **24640** |

Sources: synthetic 62 · commitpackft (keyword) 3182 · commitpackft_corpus 21396.

## Known gaps

- JS/TS held-out LLM eval (`--suite heldout`) remains the Phase 7 gate; fixtures suite is the multi-lang gate.  
- `all_migration.jsonl` is regenerated locally (gitignored) — use `corpus_manifest.json` as the durable count.  
- Phase 11: bootstrap script, README per-lang matrix, official `v1.0.0`.

## Decision log

| Date | Decision |
|------|----------|
| 2026-11-04 | Languages already done in Phase 9 → Phase 10 = quality + 15K |
| 2026-11-04 | `--mode corpus` fills short langs without keyword filter (still license/quality/parse) |
| 2026-11-04 | Fixture sandbox 8/8 is the ≥90% multi-lang exit gate |
