# Session handoff — 2026-11-07 (restart here)

> **Read this first** in a new agent window. Then sync [GLOBAL.md](GLOBAL.md) + [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md).
> Phases were **renumbered in execution order** on 2026-11-05 (map: [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md#phase-renumbering-2026-11-05)).

## Where we are — Phases 0–14 + 16 (offline) ✅ merged; Phase 15 local prep ✅ (2026-11-07), cloud part blocked on AWS quota

| Item | State |
|------|--------|
| Phases **0–14, 16 (offline)** | **Complete**, merged to `main` (PR #3 @ `389995b`, CI green). The 2026-11-07 Phase 15 prep work is **not committed yet** (user has not asked) |
| Live languages | **All 8** — adapters + Docker sandboxes + fixtures; fixture gate 8/8 |
| v2 LoRA adapters | `data/models/lora_<lang>_7b` for all 8 — 4-bit QLoRA @512, EOS + completion-only loss, 1 epoch. All pass the gate vs base 7B (24 shortest held-out rows) and beat or tie Ollama 14B on soft-exact |
| Eval token-cap bug | **Fixed** — `_sft_max_new_tokens` floor 64 → 256 (`SFT_MIN_NEW_TOKENS`); the 64 floor truncated additive rewrites and caused every parse failure. Old reports kept as `*_cap64.json` |
| Expanded corpus | `data/datasets/expanded/<lang>_migration.jsonl` (all CommitPackFT rows passing filters; originals untouched). Rows ≤512 tokens: Go 2.1×, C# 2.8×, Java 2.6×, Python 4.6×, Swift 1.5×, C++ 1.2×, Rust 1.0× (pool exhausted) |
| Frozen eval sets | `data/datasets/evalsets/<lang>.jsonl` — 100 random held-out rows (Rust 56, C++ 93), `training/eval_behavioral.py --eval-file`. **v2 sets** `rust_v2.jsonl` / `cpp_v2.jsonl` (100 each, old rows + new held-out full-CommitPack rows) for every Rust / C++ comparison from now on |
| Go recipe experiment | Frozen 100 rows: base 7B **5** · Ollama 14B **6** · v2 **14** · C (2.1× data) **15** · D (all-linear r16, 2 epochs) **12** (overfit). D1 (all-linear r16, 1 epoch) scored tonight → `logs/exp/wave_v3/status.jsonl` |
| MLOps (Phase 13) | MLflow store `mlruns/mlflow.db`; 8 `rebirth-<lang>` champions = v2 adapters; Go C/D/D1 all kept-champion by the gate (strict +1 soft-exact). Training/eval auto-tracked; UI via `.venv-mlflow` |
| GPU queue | systemd unit `rebirth-gpu-job` (user stops it by hand). Last job: `scripts/p15_prep_gpu.sh` (`STAGES="ev2 dpo"`, status `logs/exp/p15_prep/status.jsonl`). **Runs only when the user says go in chat.** |
| GPU hours | User starts/stops by hand: ~06:00–23:00 (Fri/Sat until 01:00) |
| Cloud | AWS $120 credit, G/VT quota (8 vCPU) **case open** — plan: [CLOUD_GPU_PLAN.md](CLOUD_GPU_PLAN.md) = Phase 15 |
| End product | One offline tool: modernize in-language **and** migrate between any of the 8 languages ([CLOUD_GPU_PLAN.md §7](CLOUD_GPU_PLAN.md#7-the-final-product)) |

## Git / GitHub

| Item | Value |
|------|--------|
| Branch | `main` (Phases 0–14 + 16 offline landed) |
| Last merge | PR #3 @ `389995b` (Phases 11–14, 16) |
| Tag | `v0.2.0-mvp` on origin (JS/TS MVP) |

## Exact next step

1. **Phase 12 — done 2026-11-06.** v3 tied/lost for Go, Python, C# → v2 stays champion everywhere; user decided no v3 for Swift/Java/JS/TS. Base-7B + v2 frozen-set refs for all 8 languages and the GGUF→Ollama test (Q4_K_M 13/99 vs PEFT 14/99) in [phases/PHASE_12_LOCAL_TUNING_V3.md](phases/PHASE_12_LOCAL_TUNING_V3.md). Rust/C++ data options: [research/RUST_CPP_DATA_SOURCES.md](research/RUST_CPP_DATA_SOURCES.md)
2. **Phase 16 (offline part) — done 2026-11-06.** All 8 7B champions served from Ollama (`rebirth-<lang>:7b`, Q4_K_M + GGUF LoRA, serving gate 8/8 pass; served 107 vs PEFT 112 soft-exact in total); `rebirth migrate` defaults to `--backend auto` via `configs/routing.yaml`; `rebirth package` / `rebirth serve status|rollback`; CI workflow (not yet run on GitHub — needs a push). Details: [phases/PHASE_16_MLOPS_CICD_PACKAGING.md](phases/PHASE_16_MLOPS_CICD_PACKAGING.md)
2. ✅ **Phase 13** done (2026-11-05) — `rebirth track backfill`, `rebirth registry list`, `rebirth mlflow-ui`; store `mlruns/` (SQLite). Promotions go through `rebirth registry gate`.
3. ✅ **Phase 14** done (2026-11-05) — DVC (`bash scripts/dvc.sh`), `dvc.yaml`, Terraform (`infra/aws`, `.tools/terraform_1.17.1`), `scripts/cloud/*`, `training/eval_vllm.py`. AWS (us-east-1): bucket + IAM applied, DVC pushed; GPU quota requested in us-east-1 (and ca-central-1 as fallback).
4. **Phase 15** — local prep **done 2026-11-07** ([PHASE_15 § Local prep](phases/PHASE_15_CLOUD_14B.md#local-prep-done-while-waiting-for-the-quota-2026-11-07)): Phase 15 corpora `data/datasets/p15/` (all 8; Rust / C++ 8K rows with full-CommitPack + tool pairs), Rust / C++ v2 eval sets, DPO tooling smoke-tested on the 7B. Cloud S0–S4 once the AWS quota is approved; `dvc push` of `pool/`, `p15/`, `dpo/` first (needs the user's go).
5. **Phase 17 stage 1 — done 2026-11-07** (started early): `rebirth migrate --to <lang>` (translate → target sandbox → retry), MultiPL-E HumanEval harness in 9 languages (`training/eval_translate.py`, oracle self-check 39/39), fixture matrix **53/56** directions pass the target project's tests ([PHASE_17 § Stage 1](phases/PHASE_17_CROSS_LANGUAGE.md#stage-1--pipeline--eval-harness-done-2026-11-07)).
6. **Phase 17 stage 2 — done 2026-11-07:** zero-shot Ollama 14B on all 56 directions × 40 HumanEval problems = **89.3 %** pass@1 (one try each). Weakest target Go 73.5 % — 42 of its 55 compile errors are missing stdlib imports; C++ best at 96.1 %. Fixed a MultiPL-E C# bug on the way (lists compared by reference → 45 problems unwinnable). Reports `data/reports/xlang/zeroshot_ollama14b*.json`, matrix in [PHASE_17 § Stage 2](phases/PHASE_17_CROSS_LANGUAGE.md#stage-2--zero-shot-baseline-done-2026-11-07). **Next (user's call):** Go import fix-up (CPU), base 7B baseline (GPU), a harder file-level eval before stage 3 data generation.
6. **18** monitoring · **19** v1.0 launch.

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

Full suite green (**179 passed**, 2026-11-07; `ruff check .` clean). Fixed: stale `language_hint` call in `test_mine_filters.py`; `migrate-repo` precheck now reports `skipped_secret` before `skipped_wrong_language`.

 Prefer `REBIRTH_EMBEDDER=hash` with Ollama 14B resident; unload Ollama before PEFT train/migrate.
