# Phase 15 — Cloud: 14B per-language adapters + DPO

> Status: **DONE — stopped at the S2 gate (2026-11-08).** S0 2026-11-08 (~$1.35). S1–S2 ran 2026-11-08 18:58 → 23:31 EST (~$9): the 14B Go / Python adapters did **not** beat the v2 7B adapters by the required 3 soft-exact per 100 (Go 15 vs 14, Python 19 vs 20) → S3 (other 6 languages) and S4 (DPO) skipped; the v2 7B adapters stay the same-language champions. The same instance then ran Phase 17 stages 3–4 and was terminated 2026-11-09 16:32 EST (all results home).
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

## S0 — setup + smoke (2026-11-08)

Launched in us-east-1c after ~1.5 h of InsufficientInstanceCapacity retries (all four g6e AZs).

| Check | Result |
|-------|--------|
| Setup (`scripts/cloud/setup.sh`) | ~11 min: torch 2.9.1+cu128, transformers 5.22.1, peft 0.21.2, vLLM image, 7B + 14B weights |
| 14B LoRA smoke (`cloud_14b`, Go, 20 steps, seq 1024, batch 4 × 2) | 43 s; **~1.8 s / step** (8 rows) once warm; **peak 34.9 GB** of 46 GB |
| vLLM, 14B + smoke adapter | startup ~5 min; **100 rows in 32 s**; 13 / 100 soft-exact, 98 parses (20 steps: plumbing only) |
| vLLM, base 14B chat | 10 rows in 16 s |

Fixed during S0 (all in the repo):
- `setup.sh`: `uv pip install --index-strategy unsafe-best-match` — the pins live on PyPI + the cu128
  torch index; uv's default stops at the first index that has a package (certifi failed).
- `eval_vllm.py`: vLLM's OpenAI API takes at most 4 stop strings; the SFT plan has 5, three of them
  start with `\n###` → `minimal_stops()` sends `["\n###", "\n```"]` (same output). HTTP 4xx errors
  now show vLLM's message instead of being retried.
- `vllm_serve.sh`: compile cache mounted (`~/.cache/vllm`); `sync_up.sh`: skips `data/models/exp/`
  (4 GB of local experiment checkpoints) and the stage-3 working files; `sync_down.sh`: `--update`
  (the instance holds upload-time copies of every report — never overwrite newer local files).
- `watchdog.sh`: progress = newest write to any job / training log (not only `out.log`), so a long
  training run cannot be stopped by one 0 % GPU sample during a checkpoint save;
  `SHUTDOWN_DELAY_MIN` leaves the home side time to pull results before the stop.
- `train_dpo.py`: `--attn-impl sdpa` for the L40S (eager stays the ROCm default).

Re-estimate (measured step time scaled by each p15 corpus's mean row size, one epoch):

| Language | Rows | Est. |
|----------|-----:|-----:|
| Go | 4,690 | 0.45 h |
| Swift | 3,620 | 0.40 h |
| Rust | 8,000 | 0.89 h |
| C# | 7,640 | 0.93 h |
| C++ | 8,000 | 1.05 h |
| Java | 11,856 | 1.42 h |
| Python | 37,214 | 3.73 h |
| JS / TS | 51,509 | 4.97 h |
| **All 8** | | **~13.8 h** |

S2 ≈ 4.2 h (~$8), S3 ≈ 9.7 h (~$19), S1 < 1 h, S4 ~1–1.5 h → **~16–17 h, ~$32 for S1–S4** (plan: ~$24–27).

### S1–S4 session (`scripts/cloud/session_p15.sh`)

One job on one launch; restartable (finished trainings / evals are skipped):

1. **S1** vLLM: base 7B (chat) and the v2 7B adapters (sft) on all 10 frozen sets, then base 14B (chat).
2. **S2** 14B LoRA on Go + Python (`--data data/datasets/p15/<lang>_migration.jsonl`), eval.
   **Gate** → `data/reports/cloud/p15/gate_s2.json`; S3/S4 run only if the 14B adapter ≥ v2 7B + 3
   soft-exact per 100 rows on both (same engine), otherwise the session ends.
3. **S3** the other 6 languages, smallest corpus first, eval (Rust / C++ also on `_v2`).
4. **S4** DPO on Go + Rust: 2,000 training prompts (≤ 512 tokens, never held-out / eval-set rows) →
   the 14B adapter's own outputs → pairs → DPO (bf16, seq 1024, lr 1e-5, β 0.1, 1 epoch) → eval vs S3.

Reports: `data/reports/cloud/p15/<set>__<base7b|v2_7b|base14b|14b|14b_dpo>.json`. Adapters:
`data/models/lora_<lang>_14b[_dpo]/`. Home side: results pulled with `--update`, every adapter checked,
then the instance is terminated (if anything is missing it is stopped instead, disk kept).

## Results (S1–S2, 2026-11-08)

Soft-exact matches out of 100 rows per frozen set (Rust 56), vLLM, SFT prompt for adapters, chat
prompt for base models. Format: exact / parses / mean overlap.

| Set | base 7B | v2 7B adapter | base 14B | 14B adapter (S2) |
|-----|--------:|--------------:|---------:|-----------------:|
| C++ | 8 / 93 / 0.769 | 13 / 91 / 0.828 | 6 / 91 / 0.788 | — |
| C++ v2 | 8 / 100 / 0.773 | 14 / 98 / 0.836 | 6 / 98 / 0.793 | — |
| C# | 2 / 100 / 0.750 | 13 / 100 / 0.856 | 3 / 98 / 0.752 | — |
| Go | 5 / 100 / 0.781 | 14 / 99 / 0.855 | 6 / 99 / 0.753 | **15** / 98 / 0.862 |
| Java | 10 / 100 / 0.815 | 23 / 100 / 0.881 | 8 / 100 / 0.798 | — |
| JS / TS | 0 / 99 / 0.678 | 9 / 100 / 0.822 | 2 / 99 / 0.669 | — |
| Python | 6 / 99 / 0.757 | 20 / 99 / 0.850 | 5 / 99 / 0.732 | **19** / 100 / 0.874 |
| Rust | 5 / 56 / 0.748 | 9 / 56 / 0.799 | 6 / 56 / 0.758 | — |
| Rust v2 | 6 / 100 / 0.764 | 14 / 99 / 0.831 | 8 / 100 / 0.768 | — |
| Swift | 6 / 100 / 0.787 | 12 / 100 / 0.860 | 6 / 100 / 0.786 | — |
| **Total** | **56 / 949** | **141 / 949** | **56 / 949** | 34 / 200 |

- **The base 14B is no better than the base 7B** at same-language migration (56 = 56). The
  adapters carry the gain (×2.5), and the 14B adapter only ties the 7B one (+1 Go, −1 Python),
  with slightly higher overlap. The S2 gate (`data/reports/cloud/p15/gate_s2.json`) failed → S3/S4
  skipped, as designed.
- Same-language modernization keeps the **v2 7B adapters** (Ollama `rebirth-<lang>:7b`).
  `lora_go_14b` / `lora_python_14b` are kept (DVC), not promoted.
- The DPO rollout gate in Phase 17's session failed for lack of S4 reports (`dpo_rollout_gate.json`).
- Cost: S0 ~$1.35 + S1–S2 ~$9. Phase 17 stages 3–4 on the same instance ~$32 → **≈ $43 of the
  $120 credit** used in total.

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

- [x] S0–S2 done, S3 skipped by the S2 gate; per-language table base 7B / base 14B / 7B adapter (+ 14B adapter on the pilot languages) on the frozen sets (Results above)
- [x] S4 decision recorded: not run (S2 gate failed); DPO not rolled out
- [x] All adapters + reports home; instance terminated 2026-11-09; spend recorded (Results above)
