# Phase 17 — Cross-language migration (all 8 languages)

> Status: **PLANNED** (decided 2026-11-05; was "Phase 15" before the renumbering). Starts after the 14B adapters (Phase 15) and packaging (Phase 16).
> Previous: [PHASE_16_MLOPS_CICD_PACKAGING.md](PHASE_16_MLOPS_CICD_PACKAGING.md) · Next: [PHASE_18_MLOPS_MONITORING.md](PHASE_18_MLOPS_MONITORING.md)
> Owner decision: do it, local first, AWS GPU for the heavy stages.
> Cloud setup, access model and budget split: [../CLOUD_GPU_PLAN.md](../CLOUD_GPU_PLAN.md) — Phase 17 now uses the 14B base, a 32B teacher on g6e.xlarge (L40S 48 GB) and ~$60 of the $120 credit.

## Goal

Translate a file from one supported language to another — e.g. Python → Go — through the same
closed loop (generate → sandbox test → classify → retry) Rebirth already uses for same-language
migration. Today the tool **cannot** do this:

| Blocker today | Why |
|---------------|-----|
| Training data | CommitPackFT rows are same-language commits (file before → file after). No adapter has seen Python-in / Go-out. |
| SFT prompt | `### Objective / ### Legacy / ### Modern` — no target-language field. |
| Pipeline | `rebirth migrate --lang` picks adapter, sandbox and test runner from the **source** language; output overwrites the same file/extension; the source-language tests judge the result. No `--to` anywhere in `src/`. |

The base Qwen2.5-Coder 7B and Ollama 14B already translate reasonably in plain chat — it is just
not wired in.

## Design decisions

- **One shared cross-language adapter**, not 56. 8 languages = 56 directed pairs; the prompt names
  both sides so one run covers every direction and directions share what they learn:

  ```
  ### Source language
  python
  ### Target language
  go
  ### Source
  <code>
  ### Target
  ```
- **Zero-shot first.** Measure Ollama 14B / base 7B chat on all 56 directions before training
  anything. If it is good enough for most directions, train only for the weak ones.
- **Verification = reward.** Generated training pairs are kept only if the translation compiles in
  the target sandbox **and** behaves the same as the source (same inputs → same outputs). This is
  the rejection-sampling / best-of-N idea from Phase 11 applied to translation.
- **Sequence length ~1024.** A translation row holds the code twice (source + target), so 512
  tokens only fits small functions. Local 7800 XT: all-linear r16 QLoRA peaked at 9.2 GB @ 512 —
  1024 *may* fit but must be probed first (`training/qlora_probe.py --lengths 1024`); bf16 > 128
  crashes this GPU (see Phase 11 GPU incident). Safer on a 24 GB NVIDIA card.

## Stages and time estimate

Estimates use measured throughput (2026-11-05): ~9 s / optimizer step (8 rows @ 512, 4-bit) on the
7800 XT, ~25 min per 100-row 7B eval, a few seconds per short snippet on Ollama 14B.

| # | Stage | What | Time | Where | Needs the user? |
|---|-------|------|------|-------|-----------------|
| 1 | Pipeline | `rebirth migrate --to <lang>`; new output file with target extension; run the **target** sandbox; cross-language test strategy (I/O-equivalence harness or compile + smoke run) | **2–3 days** dev | Local | Yes |
| 2 | Zero-shot baseline | Ollama 14B on all 56 directions × ~15–20 snippets | ~6–8 h | Local | No |
| 3 | Data generation | Translate snippets from the existing per-language corpus into each target; ~20K candidates → expect 40–60 % verified → **~8–10K pairs** | ~1.5–2 days local, or ~10–20 h on AWS with vLLM | **AWS** (generation) + local (verification) | No |
| 4 | Train | One shared adapter, ~8–10K pairs, 2 epochs, seq ~1024 | ~5–8 h | **AWS** (or local if the 1024 probe passes) | No |
| 5 | Eval | 56 directions vs the stage-2 baseline | ~6–8 h | Either | No |
| 6 | Fix-up round | More data for weak directions (likely Swift, Rust), retrain | ~1 day | Either | Mostly no |

**Total ≈ 1 week calendar, ~2–3 days of hands-on work, ~60–80 GPU-hours** (fewer on AWS with batched vLLM generation).

Biggest risks: the cross-language equivalence harness for 8 languages (largest engineering piece);
sequence length on the local GPU; public parallel datasets (e.g. Rosetta Code, which has the same
tasks solved in all 8 languages) could replace part of stage 3, but licenses and quality must be
checked first.

## AWS plan (original estimate — superseded by [CLOUD_GPU_PLAN.md](../CLOUD_GPU_PLAN.md))

The user has an AWS account with **$120 credit** for all phases (G/VT quota: 8 vCPUs requested, case open 2026-11-05).

| Instance | GPU | On-demand (us-east-1, 3rd-party price lists — verify on AWS) | Hours for $100 |
|----------|-----|-------------------------------:|---------------:|
| **g6.xlarge** (default choice) | NVIDIA L4 24 GB | ~$0.80/h | ~125 h |
| g5.xlarge | NVIDIA A10G 24 GB | ~$1.01/h | ~100 h |
| g6e.xlarge | NVIDIA L40S 48 GB | ~$1.86/h | ~54 h |

Spot is ~half price but can be interrupted (fine for stage 3 with checkpointed output). Add a few
dollars/month for a ~100 GB EBS volume.

**Split (estimated ~$20–40 total, leaves room for a second round):**

| Stage | Where | Est. cost |
|-------|-------|----------:|
| 1–2 pipeline + zero-shot baseline | Local | $0 |
| 3 generation (vLLM, batched — ~10× Ollama one-at-a-time) | AWS g6.xlarge | ~$10–16 |
| 3b compile + equivalence verification (CPU) | Local (more cores than the 4 vCPU g6.xlarge) | $0 |
| 4 train shared adapter @ ~1024 | AWS g6.xlarge | ~$7–12 |
| 5 eval | Either | $0–8 |

Why NVIDIA helps here: 24 GB + mature CUDA make seq 1024 safe, **4-bit generation works** (it is
broken on ROCm), and vLLM batches generation. The same-language adapters stay local — they fit the
7800 XT.

### Before starting (account checklist)

1. **Free plan probably blocks GPU instances** (not confirmed in AWS docs; user reports say an
   upgrade to the **Paid plan** is needed). After upgrading, spend beyond the credit bills the card.
2. Request a **Service Quotas** increase now — *Running On-Demand G and VT instances* → 4–8 vCPUs
   (new accounts are usually 0; approval can take hours to days).
3. Set **AWS Budgets** alerts at ~$50 and ~$90.
4. Always stop/terminate the instance when a job finishes (scripts should self-shutdown).

## Exit criteria

- `rebirth migrate <repo> --lang python --to go` produces a Go file that passes the target sandbox
  on the fixture samples, for all 56 directions or a documented subset.
- Per-direction eval table (compile rate, equivalence rate) vs the zero-shot baseline.
- Adapter (if trained) beats zero-shot on the directions it covers; otherwise ship zero-shot.

## Sources (AWS facts, checked 2026-11-05)

- [AWS Free Tier explained (Rackspace)](https://spot.rackspace.com/blog/aws-free-tier)
- [Reddit thread on Free plan instance access](https://reddit.sentinel-team.org/posts/1rpn9gm/snapshots/2026-03-11T04%3A58%3A06.070402Z)
- [g6.xlarge pricing (DoiT)](https://www.doit.com/compute/spot/us-east-1/g6.xlarge) · [g5.xlarge (DoiT)](https://compute.doit.com/spot/us-east-1/g5.xlarge) · [g6e.xlarge (DoiT)](https://compute.doit.com/spot/us-east-1/g6e.xlarge)
