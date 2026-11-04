# Migration datasets

> Last updated: 2026-11-03 (Phase 7: classifier **100** logs / acc **0.96**; held-out eval suite live)

## Goal

Specialize Rebirth AI on **real legacy→modern code edits**, not only hand-written toy pairs — and **use** that adapter in the agent when desired.

**v1.0 corpus target:** **≥15 000** mined migration pairs across **JS/TS, Python, Go, Java, C++, C#, Rust, Swift** (~1–2K+ filtered pairs per language). Current baseline is JS/TS-only (1339).

## What we train on

| Artifact | Rows | Role |
|----------|------|------|
| `data/datasets/js_ts_migration.jsonl` | **1339** | LoRA SFT pairs (`objective`, `legacy`, `modern`, …) |
| `data/datasets/js_ts_migration.synthetic.jsonl` | 59 | Original hand pairs (preserved) |
| `data/datasets/js_ts_migration.stats.json` | — | Mine run stats |
| `data/datasets/error_logs.jsonl` | **90** | Error classifier labels |

### Migration pair mix (current — JS/TS only)

| Source | Count |
|--------|------:|
| `synthetic` | 59 |
| `commitpackft` (JS/TS, keyword + parse filtered) | 1280 |
| **Total** | **1339** |

Languages today: ~1188 JavaScript, ~151 TypeScript.

### Multi-language corpus plan (Phases 8–10)

| Language | Dataset path (planned) | Phase | Target pairs (guide) |
|----------|------------------------|-------|----------------------:|
| JS / TS | `js_ts_migration.jsonl` | done / top-up | ≥1.3K (live) → ~2K |
| Python | `python_migration.jsonl` | 8 | ≥1.5K |
| Go | `go_migration.jsonl` | 9 | ≥1.5K |
| Java | `java_migration.jsonl` | 9 | ≥2K |
| C# | `csharp_migration.jsonl` | 9 | ≥1.5K |
| Rust | `rust_migration.jsonl` | 9 | ≥1.5K |
| C++ | `cpp_migration.jsonl` | 10 | ≥1.5K |
| Swift | `swift_migration.jsonl` | 10 | ≥1K |
| **Combined** | manifest / concat | **10 exit** | **≥15 000** |

Mining: extend `training/mine_migration_pairs.py` (or per-lang filters) over CommitPackFT / language-appropriate Hub corpora with the same keyword + parse gates. Record running totals in `*.stats.json` and this file after each mine.

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
# Defaults to 7B adapter when data/models/lora_js_ts_7b/adapter exists
rebirth generate fixtures/js_ts_sample/src/math.js --backend peft --no-rag

# Closed loop (auto hash-RAG to avoid stacking ST + 7B on the display GPU)
rebirth migrate fixtures/js_ts_sample \
  --seed-draft fixtures/migration_seeds/math.broken.js \
  --backend peft --no-write

# Safer on a shared display GPU:
# REBIRTH_PEFT_DEVICE=cpu rebirth migrate … --backend peft
```

| Backend | Model | When |
|---------|-------|------|
| `ollama` (default) | `qwen2.5-coder:14b-instruct-q6_K` | Best agent quality / ROCm inference |
| `peft` (default paths) | **7B** + `lora_js_ts_7b` if present, else 0.5B | Local specialization |
| `peft` + 0.5B override | set `REBIRTH_PEFT_*` to 0.5B paths | Pipeline proof only |
Implementation: `src/rebirth_ai/generate/factory.py`, `peft_client.py` (stop/decode hardened 2026-11-01).

## Behavioral eval (preferred — 2026-11-01)

Script: `training/eval_behavioral.py` · helpers: `src/rebirth_ai/eval/metrics.py`  
Primary metrics: **sandbox-pass**, AST parse, symbol Jaccard, **soft-exact** (whitespace-insensitive).

```bash
# Free VRAM if Ollama 14B is resident (~12GB)
ollama stop qwen2.5-coder:14b-instruct-q6_K
python training/eval_behavioral.py --backend peft --suite both --n-synth 8 \
  --out data/reports/eval_behavioral.json
python training/eval_behavioral.py --backend ollama --suite sandbox \
  --out data/reports/eval_behavioral_ollama.json
```

| Backend | Sandbox pass | Synth soft-exact | Parse | Mean symbol Jaccard |
|---------|-------------:|-----------------:|------:|--------------------:|
| **PEFT 7B** (chat one-shot) | **2/2 (100%)** | **6/8 (75%)** | 8/8 | 0.875 |
| Ollama 14B | **2/2 (100%)** | — | — | — |

Legacy strict exact-string smoke: 0.5B **1/8**, 7B **5/8** (arrow whitespace was a false fail).  
Soft-exact remaining misses: rename `bad`→`good`, `!!` vs `Boolean`.  
Note: SFT token caps can truncate whole-file repairs; sandbox suite uses **chat** style.

Artifacts: `data/reports/eval_behavioral.json`, `eval_behavioral_peft7b_sandbox.json`, `eval_behavioral_ollama.json`.

**Verdict:** Score with **sandbox-pass**, not exact string. Agent default stays **Ollama 14B**; PEFT 7B matches on fixture sandbox when chat-decoded.

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

- PEFT defaults → 7B — **done**; closed-loop math **PASS**.  
- Behavioral eval — **done** (`eval_behavioral.py`).  
- Phase 5 — **shipped**.  
- **Do not** stack Ollama 14B + PEFT 7B + ST on the 7800 XT.  
- Phase 7 held-out eval — **done** (`fixtures/eval_suite`, baseline frozen).  
- Next product work: [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md) Phases **8–11** (8 languages + **≥15K** pairs → **v1.0**).  
- Per-language mines — Phase 8 (Python), 9 (Go/Java/C#/Rust), 10 (C++/Swift + top-up to 15K).  
- Migration-kind tags / train-eval split — Phase 10.  
- Optional cloud 14B QLoRA on full multi-lang corpus — Track C (browser billing; not required).  
