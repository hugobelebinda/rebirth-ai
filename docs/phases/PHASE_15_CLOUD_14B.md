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

## Exit criteria

- [ ] S0–S3 done; per-language table base 7B / base 14B / 7B adapter / 14B adapter on frozen sets
- [ ] S4 decision recorded (DPO rolled out or not)
- [ ] All adapters + reports home; instance terminated; spend recorded in the doc
