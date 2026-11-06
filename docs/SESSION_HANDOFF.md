# Session handoff — 2026-11-05 (restart here)

> **Read this first** in a new agent window. Then sync [GLOBAL.md](GLOBAL.md) + [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md).
> Phases were **renumbered in execution order** on 2026-11-05 (map: [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md#phase-renumbering-2026-11-05)).

## Where we are — **Phase 12** (local tuning v3)

| Item | State |
|------|--------|
| Phases **0–11** | **Complete.** Phase 11 (per-language LoRA) finished 2026-11-05 — **not committed yet** (user has not asked) |
| Live languages | **All 8** — adapters + Docker sandboxes + fixtures; fixture gate 8/8 |
| v2 LoRA adapters | `data/models/lora_<lang>_7b` for all 8 — 4-bit QLoRA @512, EOS + completion-only loss, 1 epoch. All pass the gate vs base 7B (24 shortest held-out rows) and beat or tie Ollama 14B on soft-exact |
| Eval token-cap bug | **Fixed** — `_sft_max_new_tokens` floor 64 → 256 (`SFT_MIN_NEW_TOKENS`); the 64 floor truncated additive rewrites and caused every parse failure. Old reports kept as `*_cap64.json` |
| Expanded corpus | `data/datasets/expanded/<lang>_migration.jsonl` (all CommitPackFT rows passing filters; originals untouched). Rows ≤512 tokens: Go 2.1×, C# 2.8×, Java 2.6×, Python 4.6×, Swift 1.5×, C++ 1.2×, Rust 1.0× (pool exhausted) |
| Frozen eval sets | `data/datasets/evalsets/<lang>.jsonl` — 100 random held-out rows (Rust 56, C++ 93), `training/eval_behavioral.py --eval-file` |
| Go recipe experiment | Frozen 100 rows: base 7B **5** · Ollama 14B **6** · v2 **14** · C (2.1× data) **15** · D (all-linear r16, 2 epochs) **12** (overfit). D1 (all-linear r16, 1 epoch) scored tonight → `logs/exp/wave_v3/status.jsonl` |
| MLOps (Phase 13) | MLflow store `mlruns/mlflow.db`; 8 `rebirth-<lang>` champions = v2 adapters; Go C/D/D1 all kept-champion by the gate (strict +1 soft-exact). Training/eval auto-tracked; UI via `.venv-mlflow` |
| GPU queue | `scripts/gpu_job.sh` → `scripts/wave_v3.sh` (restartable: skips finished stages, `train_lora.py --resume`). **Runs only when the user says go in chat.** Next stage: Swift v2 eval → Swift v3 train → … |
| GPU hours | User starts/stops by hand: ~06:00–23:00 (Fri/Sat until 01:00) |
| Cloud | AWS $120 credit, G/VT quota (8 vCPU) **case open** — plan: [CLOUD_GPU_PLAN.md](CLOUD_GPU_PLAN.md) = Phase 15 |
| End product | One offline tool: modernize in-language **and** migrate between any of the 8 languages ([CLOUD_GPU_PLAN.md §7](CLOUD_GPU_PLAN.md#7-the-final-product)) |

## Git / GitHub

| Item | Value |
|------|--------|
| Branch | `main` (Phases 8–10 landed) |
| Prior merge | PR #2 @ `5f1540e` (Phases 6–7) |
| Tag | `v0.2.0-mvp` on origin (JS/TS MVP) |

## Exact next step

1. **Phase 12 (now)** — report D1; when the user says go: `systemd-run --user --unit=rebirth-gpu-job --working-directory="$PWD" bash scripts/gpu_job.sh` (Swift → C# → Java → Python → JS/TS v3 vs v2 on frozen sets). Also the GGUF→Ollama adapter test. [phases/PHASE_12_LOCAL_TUNING_V3.md](phases/PHASE_12_LOCAL_TUNING_V3.md)
2. ✅ **Phase 13** done (2026-11-05) — `rebirth track backfill`, `rebirth registry list`, `rebirth mlflow-ui`; store `mlruns/` (SQLite). Promotions go through `rebirth registry gate`.
3. **Phase 14** — MLOps 2: DVC + S3, Docker CUDA image, Terraform, `scripts/cloud/`, vLLM eval.
4. **Phase 15** — cloud 14B adapters + DPO, once the AWS quota is approved.
5. **Phase 16** — model CI/CD + Ollama packaging · **17** cross-language · **18** monitoring · **19** v1.0 launch.

Never start the GPU queue or commit/push without the user's explicit go-ahead.

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

Phases 0–10 adapters / golden JS benches / 15K mine / Phase 11.1 harness.

## Tests

Full suite green (**102/102**, 2026-11-04). Fixed: stale `language_hint` call in `test_mine_filters.py`; `migrate-repo` precheck now reports `skipped_secret` before `skipped_wrong_language`.

 Prefer `REBIRTH_EMBEDDER=hash` with Ollama 14B resident; unload Ollama before PEFT train/migrate.
