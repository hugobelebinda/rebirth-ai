# Session handoff — 2026-11-10 (restart here)

> **Read this first** in a new agent window. Then sync [GLOBAL.md](GLOBAL.md) + [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md).
> Phases were **renumbered in execution order** on 2026-11-05 (map: [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md#phase-renumbering-2026-11-05)).

## Where we are — Phases 0–15 ✅, 16 (offline) ✅, 17 stages 1–4 ✅ (2026-11-09); AWS round done, no instance running

| Item | State |
|------|--------|
| Phases **0–15, 16 (offline)** | **Complete.** Phase 15 ended at its S2 gate (2026-11-08): 14B Go / Python adapters 15 / 19 vs v2 7B 14 / 20 soft-exact per 100 → S3/S4 skipped ([PHASE_15 § Results](phases/PHASE_15_CLOUD_14B.md#results-s1s2-2026-11-08)) |
| Phase 17 | Stages 1–4 done. Shared cross-language adapter `lora_xlang_14b` (14B, trained on AWS on 14K rows of test-verified translations). Program suite, local q6_K Ollama: plain 14B **208 / 420** → adapter **265 / 420** (17 / 21 directions better, 1 worse by one problem). Cloud bf16: 215 → 274; Qwen3-Coder-30B zero-shot 282 (too big for 16 GB) ([PHASE_17 § Stage 4](phases/PHASE_17_CROSS_LANGUAGE.md#stage-4--shared-adapter--eval-done-2026-11-09-aws--local-check)) |
| Serving | Same-language: Ollama `rebirth-<lang>:7b` (v2 7B adapters, `configs/routing.yaml`). Cross-language (`migrate --to`, `--backend auto`): **`rebirth-xlang:14b-v1`** (q6_K 14B + f16 GGUF LoRA, built by `scripts/ollama_xlang.sh`), plain 14B if missing; `--backend ollama` = plain 14B (eval baseline); `REBIRTH_XLANG_MODEL` overrides |
| Adapters | `data/models/lora_<lang>_7b` (8, champions) · `lora_{go,python}_14b` (Phase 15, not promoted) · `lora_xlang_14b` (Phase 17) — all DVC-tracked; the three 14B ones are **not pushed to S3 yet** (`scripts/dvc.sh push` needs the user's go) |
| Frozen eval sets | `data/datasets/evalsets/<lang>.jsonl` (+ `rust_v2` / `cpp_v2`); translation: `data/datasets/xlang/humaneval.jsonl` (56 directions) and `programs.jsonl` (163 CodeContests programs) |
| Stage-3 data | `data/datasets/xlang/stage3/` (30,211 pairs, git-ignored, copy in S3) |
| GPU queue | systemd unit `rebirth-gpu-job` (user stops it by hand); idle since 2026-11-09 23:26. **Runs only when the user says go in chat.** |
| Cloud | g6e.xlarge in us-east-1 (4-vCPU quota). Spend ≈ **$43 of $120** (S0 + Phase 15 S1–S2 + Phase 17 stages 3–4); instance terminated 2026-11-09 16:32 EST; bucket + IAM remain. Any launch needs the user's go |
| End product | One offline tool: modernize in-language **and** migrate between any of the 8 languages ([CLOUD_GPU_PLAN.md §7](CLOUD_GPU_PLAN.md#7-the-final-product)); evidence currently favors specialists (7B same-language + 14B cross-language) over one unified 14B adapter — user decision pending |

## Git / GitHub

| Item | Value |
|------|--------|
| Branch | `main` |
| Last merge | PR #4 (Phase 15 + Phase 17 stages 1–4, 2026-11-10) |
| Tag | `v0.2.0-mvp` on origin (JS/TS MVP) |

## Exact next step (plan the day with the user)

1. **Phase 17 stage 5 — local GPU, needs a go (~5–7 h):** the adapter (`OLLAMA_MODEL=rebirth-xlang:14b-v1`) on the 35 bootstrapped directions (`eval_translate.py --suite programs --sources javascript go csharp rust swift --add-sources-from data/reports/xlang/programs_ollama14b_retry.json --retries 2`, baseline `programs_ollama14b_bootstrap.json` 343 / 511) and HumanEval (`--suite humaneval`, baseline 91.1 %) as a regression check.
2. **Decide unified vs specialists** (CLOUD_GPU_PLAN §7): the 14B same-language adapters only tied the 7B ones.
3. **Optional cloud round (~$15–25, needs a go + capacity):** Qwen3-Coder-30B-A3B-FP8 as the teacher for a second translation-data round (its zero-shot 282 / 420 ≈ our adapter) → `lora_xlang_14b` v2. Scripts: `scripts/cloud/session_p17*.sh`, `hub.sh` (teacher loading fixed: `READY_MIN`, shared-blob `drop()`).
4. **Phase 18** (monitoring + flywheel, no GPU) → **Phase 19** (installer, docs, v1.0.0).

Never start the GPU queue, launch cloud instances, or commit/push without the user's explicit go-ahead.

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

Full suite green (**194 passed**, 2026-11-10; `ruff check .` clean).

 Prefer `REBIRTH_EMBEDDER=hash` with Ollama 14B resident; unload Ollama before PEFT train/migrate.
