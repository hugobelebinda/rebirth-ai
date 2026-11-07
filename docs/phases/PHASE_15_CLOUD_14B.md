# Phase 15 — Cloud: 14B per-language adapters + DPO

> Status: **PLANNED — blocked on the AWS G/VT quota.** us-east-1 request (8 vCPU) **denied 2026-11-06** (new account, no usage history); re-requested **4 vCPU** (one g6e.xlarge) the same day. ca-central-1 case still open (fallback: g6.2xlarge L4). If refused again: resubmit after the next billing cycle, or run the same `scripts/cloud/*` over SSH on another NVIDIA provider (real money, not credits). Account setup, bucket and IAM are done.
> Previous: [PHASE_14_MLOPS_REPRO_INFRA.md](PHASE_14_MLOPS_REPRO_INFRA.md) · Next: [PHASE_16_MLOPS_CICD_PACKAGING.md](PHASE_16_MLOPS_CICD_PACKAGING.md)
> Full plan (setup guide, access rules, models, costs): **[../CLOUD_GPU_PLAN.md](../CLOUD_GPU_PLAN.md)**

## Goal

Move the per-language specialization from the 7B base to **Qwen2.5-Coder-14B-Instruct** (the model
Ollama already runs locally), at 1024 tokens in bf16, and attack the measured failure modes
(no-op outputs, partial edits) with DPO. Trained adapters come home; the instance is terminated.

## Sessions (AWS g6e.xlarge, L40S 48 GB, ~$1.86/h)

| Session | What | Est. | Gate |
|---------|------|-----:|------|
| S0 | Setup (Phase 14 scripts), 20-step 14B smoke @1024, vLLM smoke, measure step time | ~$3 | Re-estimate S2–S4 |
| S1 | vLLM baseline matrix on all 8 frozen sets: base 7B, base 14B, v2/v3 7B adapters | ~$2 | Headroom per language |
| S2 | 14B pilot: Go + Python | ~$4 | Continue only if 14B adapter ≥ 7B adapter + 3 soft-exact per 100 on both |
| S3 | 14B wave: remaining 6 languages (Python/JS-TS uncapped at 1024 tokens) | ~$12–15 | Per-language champion via the Phase 13 registry |
| S4 | DPO on 2 languages (gold ≻ own no-op/unparsable/low-overlap outputs) | ~$6 | Roll out only if it beats S3 |

Budget this phase: ~$35–40 of the $120 credit (rest → Phase 17).

## Rules

- Nothing is launched without the user's go-ahead in chat; the user launches/stops the instance (mode A).
- Every run is tracked (Phase 13) and reproducible (Phase 14); results synced home before shutdown.
- Adapters → `data/models/lora_<lang>_14b/` (+ `.gguf`); local serving is Phase 16.

## Local prep done while waiting for the quota (2026-11-07)

Everything below runs at home; it makes the paid sessions shorter and better.

| Piece | What | Used in |
|-------|------|---------|
| **Corpora** `data/datasets/p15/<lang>_migration.jsonl` | All 8 languages: expanded corpus minus rows from frozen eval-set files (leak guard). Rust / C++: 8,000 rows each with full-CommitPack modern-tier commits + compile-checked tool pairs ([research note](../research/RUST_CPP_DATA_SOURCES.md)) | S2 / S3: `--data data/datasets/p15/<lang>_migration.jsonl` |
| **Eval sets v2** `evalsets/{rust,cpp}_v2.jsonl` | 100 rows each (old 56 / 93 rows + new held-out rows) — less noisy than 56 / 93 | S1 matrix + every Rust / C++ comparison from now on (old sets stay for continuity) |
| **DPO tooling** | `training/build_dpo_pairs.py` (sample train prompts → the adapter's own outputs → gold ≻ no-op / unparsable / low-overlap / partial), `training/train_dpo.py` (TRL 1.17; SFT adapter trainable + frozen copy as reference) | S4 |
| **Cloud sync** | `scripts/cloud/sync_up.sh` skips the 0.6 GB mining pool and GGUF files | S0 |

Sizing note: Rust + C++ grow from ~4.9K to 16K rows (S3 gets ~+$2–4 at the S0-measured step time).
`params.yaml: p15.target` cuts that back in one line if S0 shows it is too slow.

### Rust / C++ on the v2 eval sets (local 7B, 2026-11-07)

Soft-exact on the v2 sets (bf16 base + PEFT, greedy; `data/reports/exp/<lang>@ev2_{v2,B_base_chat}.json`):

| Lang | Model | All 100 | Old rows | New rows | Parses |
|------|-------|--------:|---------:|---------:|-------:|
| Rust | base 7B (chat) | 6 | 5 / 56 | 1 / 44 | 100 |
| Rust | v2 adapter | **14** | 9 / 56 | 5 / 44 | 99 |
| C++ | base 7B (chat) | 8 | 8 / 93 | 0 / 7 | 100 |
| C++ | v2 adapter | **15** | 14 / 93 | 1 / 7 | 98 |

- Greedy decoding reproduces the old-set predictions exactly (56 / 56 and 93 / 93 identical), so the
  old rows needed no re-scoring.
- Paired, row by row: the adapter wins 8 rows the base misses on Rust and loses none (sign test
  p ≈ 0.008); on C++ it is 7 vs 0 (p ≈ 0.016). Before, at n = 56 / 93, the same gap could not be told
  apart from noise.
- C++ only gained 7 new rows: few full-CommitPack C++ rows are held out, fit 512 tokens and come
  from a file no adapter has trained on. A larger C++ v2 set needs more shards mined (the 60 sampled
  of 365 gave 7); not worth the time before S1.

### DPO smoke test (local 7B, Rust, 2026-11-07)

`scripts/dpo_smoke.sh` (run by `scripts/p15_prep_gpu.sh`, `STAGES="ev2 dpo"`). Purpose: run the S4 code
once at home so the cloud session does not debug it. **It worked end to end**: sample → generate →
pairs → train → eval.

| Step | Result |
|------|--------|
| Prompts | 120 Rust training rows from the p15 corpus (50 expanded, 41 generic, 15 tool, 14 modern; no held-out rows, no eval-set files) |
| v2's own outputs | 23 / 120 soft-exact. By tier: expanded 12 / 50, generic 8 / 41, modern 2 / 14, **tool 1 / 15** (v2 has never seen cargo fix / clippy edits; the S3 retrain on the p15 corpus should fix that) |
| Pairs | **97**: partial 71, low_overlap 12, noop 10, unparsable 4 (`data/datasets/dpo/rust_pairs.jsonl`) |
| Train | 4-bit 7B, seq 512, β 0.1, lr 2e-5, batch 1 × 8, 1 epoch = 13 steps, **240 s**, ~19 s / step, no OOM. Loss 0.693 → 0.667, reward margin 0 → +0.05, reward accuracy 1.0 on the last step |
| Eval on `rust_v2` | **14 / 100**, the same as v2 (same 14 rows; 89 / 100 predictions identical; no-ops 9 → 8) |

- As expected: 97 pairs and 13 small steps barely move the adapter. The point was the plumbing:
  TRL's frozen "ref" copy of the SFT adapter (checked: reference = SFT, not the base), saving
  only the "default" adapter (same 16 tensors as the SFT adapter, loads in every eval path),
  memory on 16 GB, and the eval of the result.
- For S4: ≥ 1–2K pairs per language (the 14B adapter's outputs on ~2K training prompts with
  vLLM, minutes on the L40S), 1–2 epochs, lr 5e-6–2e-5, and compare to the S3 adapter on the
  frozen sets with the strict gate. Roll it out only if it wins.
- `train_dpo.py` also runs on CPU (tiny random model, for testing without the GPU).

## Exit criteria

- [ ] S0–S3 done; per-language table base 7B / base 14B / 7B adapter / 14B adapter on frozen sets
- [ ] S4 decision recorded (DPO rolled out or not)
- [ ] All adapters + reports home; instance terminated; spend recorded in the doc
