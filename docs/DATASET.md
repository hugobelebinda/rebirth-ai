# Migration datasets

> Last updated: 2026-11-01 (7B LoRA 1-epoch + stop/decode + hang notes)

## Goal

Specialize Rebirth AI on **real legacy→modern code edits**, not only hand-written toy pairs — and **use** that adapter in the agent when desired.

## What we train on

| Artifact | Rows | Role |
|----------|------|------|
| `data/datasets/js_ts_migration.jsonl` | **1339** | LoRA SFT pairs (`objective`, `legacy`, `modern`, …) |
| `data/datasets/js_ts_migration.synthetic.jsonl` | 59 | Original hand pairs (preserved) |
| `data/datasets/js_ts_migration.stats.json` | — | Mine run stats |
| `data/datasets/error_logs.jsonl` | **90** | Error classifier labels |

### Migration pair mix (current)

| Source | Count |
|--------|------:|
| `synthetic` | 59 |
| `commitpackft` (JS/TS, keyword + parse filtered) | 1280 |
| **Total** | **1339** |

Languages: ~1188 JavaScript, ~151 TypeScript.

### Upstream corpus

- Hub: [`bigcode/commitpackft`](https://huggingface.co/datasets/bigcode/commitpackft) (MIT / permissive subset)
- Local cache (gitignored): `data/cache/commitpackft/`
- Filters: migration/upgrade keywords, permissive license, size bounds, Tree-sitter parse (`--require-parse`)

## How to refresh + train

```bash
source .venv/bin/activate

# Expand classifier logs (idempotent)
python training/expand_error_logs.py
rebirth train-classifier

# Re-mine from Hub (downloads ~140MB JS+TS once into cache)
rebirth mine-dataset --max-pairs 1500 --require-parse

# 0.5B CPU (safe; slow) — proof-of-pipeline only
pip install -e ".[ml]"
python training/train_lora.py --cpu --epochs 1 --max-length 512 --batch-size 1

# 7B GPU (ROCm) — verify first; display-on-dGPU can hang the session
bash scripts/verify_rocm_torch.sh
# smoke: bash scripts/train_lora_gpu.sh
# full 1-epoch (proven): see commands under “7B full epoch” below
```

## Runtime: use the adapter

Default agent path remains **Ollama 14B** (stronger general coder). To use local PEFT:

```bash
# 0.5B adapter (default PEFT paths)
rebirth generate fixtures/js_ts_sample/src/math.js --backend peft --no-rag

# 7B adapter (explicit env)
REBIRTH_PEFT_BASE=Qwen/Qwen2.5-Coder-7B-Instruct \
REBIRTH_PEFT_ADAPTER=data/models/lora_js_ts_7b/adapter \
rebirth generate fixtures/js_ts_sample/src/math.js --backend peft --no-rag

# Closed loop
REBIRTH_GENERATOR=peft rebirth migrate fixtures/js_ts_sample \
  --seed-draft fixtures/migration_seeds/math.broken.js --backend peft --no-write
```

| Backend | Model | When |
|---------|-------|------|
| `ollama` (default) | `qwen2.5-coder:14b-instruct-q6_K` | Best agent quality / ROCm inference |
| `peft` + 0.5B | `Qwen2.5-Coder-0.5B-Instruct` + `lora_js_ts` | Pipeline proof; weak on whole-file |
| `peft` + 7B | `Qwen2.5-Coder-7B-Instruct` + `lora_js_ts_7b` | Stronger local specialization |

Implementation: `src/rebirth_ai/generate/factory.py`, `peft_client.py` (stop/decode hardened 2026-11-01).

## PEFT eval results (same 8 short synthetic pairs)

| Adapter | Exact | Mean overlap | Notes |
|---------|------:|-------------:|-------|
| 0.5B CPU LoRA | 1/8 (12.5%) | 0.523 | Junk tails / whole-file fail |
| **7B ROCm 1-epoch (r=8, seq 128)** | **5/8 (62%)** | **0.958** | Best local PEFT so far |
| 7B + stop/decode (no retrain) | 5/8 (62%) | 0.958 | Decode helps junk; not semantic misses |

**Exact-match understates 7B quality.** Remaining 3 “misses” are largely false negatives for migration usefulness:

| Case | Why exact failed | Still good? |
|------|------------------|-------------|
| Arrow callback | `[1,2,3]` vs `[1, 2, 3]` | Yes — whitespace |
| Number.isNaN | renamed `bad`→`good` | Yes for objective |
| Boolean cast | `!!flag` vs `Boolean(flag)` | Soft style disagreement |

Artifacts: `data/models/lora_js_ts/eval_smoke.json`, `data/models/lora_js_ts_7b/eval_smoke.json`.  
Re-run:

```bash
python training/eval_peft_smoke.py --n 8 --cpu   # 0.5B defaults
python training/eval_peft_smoke.py --n 8 \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter data/models/lora_js_ts_7b/adapter \
  --out data/models/lora_js_ts_7b/eval_smoke.json
```

**Verdict:** Keep default generator **Ollama 14B**. Prefer **7B PEFT** over 0.5B when using `--backend peft`. Do not treat 0.5B as production.

## AMD GPU training (ROCm PyTorch)

**Health gate:** `bash scripts/verify_rocm_torch.sh` must exit 0.

| Check | Status (2026-11-01) |
|-------|---------------------|
| Torch | `2.9.1+rocm6.4` (HIP, not NVIDIA) |
| System ROCm | 6.4.4 (`/opt/rocm` → `/opt/rocm-6.4.4`) |
| gfx1101 (7800 XT) | OK in rocminfo + PyTorch |
| HIP matmul | OK |
| bitsandbytes | Present but CUDA-oriented — use **bf16 LoRA**, not QLoRA |
| 7B smoke (5 steps, seq 128) | **PASS** → `data/models/lora_js_ts_7b_smoke` |
| 7B 1-epoch (1339 pairs, seq 128) | **PASS** → `data/models/lora_js_ts_7b` (loss ≈1.278, ~4.5 min) |
| 7B seq 512 | **OOM** |
| 7B 2-epoch / r=16 | **GPU Hang** at step 0 — aborted; keep 1-epoch adapter |

Env helpers: `scripts/rocm_env.sh`, `scripts/train_lora_gpu.sh`, `scripts/enable_amd_train_gpu.sh`.  
Empty `CUDA_VISIBLE_DEVICES=` / `HIP_VISIBLE_DEVICES=` hides GPUs — `unset` them.

### 7B full epoch (proven command)

```bash
source scripts/rocm_env.sh && source .venv/bin/activate
bash scripts/verify_rocm_torch.sh
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0
python training/train_lora.py \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --out data/models/lora_js_ts_7b \
  --batch-size 1 --grad-accum 8 --max-length 128 \
  --epochs 1 --max-steps -1 --save-steps 50 \
  --gradient-checkpointing --attn-eager
```

**Policy:** Do **not** immediately retry multi-epoch / larger-r local 7B trains with display on the 7800 XT. Prefer cloud CUDA for longer/heavier jobs, or stick with the 1-epoch adapter + decode cleanup.

### Cloud GPU prices (Hugging Face Jobs)

| Hardware | ~$/hour |
|----------|---------|
| T4 16GB | $0.40 |
| L4 24GB | $0.80 |
| A10G 24GB | $1.00–1.50 |
| A100 80GB | $2.50 |

Billed per minute. Docs: https://huggingface.co/docs/hub/jobs-pricing  
A short 7B/14B QLoRA epoch on ~1.3k pairs is often **~$1–3** on T4/L4.

## Still open / next (data track)

- Wire default PEFT paths to **7B** adapter (or document clearly in CLI help).
- Semantic / AST eval (not only exact string match).
- Held-out eval split; tag pairs by migration kind.
- 14B QLoRA on cloud matching Ollama runtime → merge (`training/export_for_ollama.md`).
- Phase 5 multi-file + GitHub PRs — **shipped** (`rebirth migrate-repo`, PR #1).
