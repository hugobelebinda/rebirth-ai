# Migration datasets

> Last updated: 2026-11-04 (Phase 10 complete; corpus **24 640**; next = Phase 11 per-lang LoRA — [PHASE_11_LORA_PER_LANGUAGE.md](phases/PHASE_11_LORA_PER_LANGUAGE.md))

## Goal

Specialize Rebirth AI on **real legacy→modern code edits**, not only hand-written toy pairs — and **use** that adapter in the agent when desired.

**v1.0 corpus target:** **≥15 000** mined migration pairs across **JS/TS, Python, Go, Java, C++, C#, Rust, Swift**. **Met:** running total **24 640** (`corpus_manifest.json`).

## What we train on

| Artifact | Rows | Role |
|----------|------|------|
| `data/datasets/corpus_manifest.json` | — | Combined totals (Phase 10) |
| `data/datasets/js_ts_migration.jsonl` | **9103** | JS/TS (keyword + corpus fill) |
| `data/datasets/python_migration.jsonl` | **5246** | Python |
| `data/datasets/java_migration.jsonl` | **2649** | Java |
| `data/datasets/csharp_migration.jsonl` | **1562** | C# |
| `data/datasets/go_migration.jsonl` | **1528** | Go |
| `data/datasets/swift_migration.jsonl` | **1535** | Swift |
| `data/datasets/cpp_migration.jsonl` | **1512** | C++ |
| `data/datasets/rust_migration.jsonl` | **1505** | Rust |
| `data/datasets/error_logs.jsonl` | **~105** | Error classifier labels |

### Source mix (all languages, Phase 10)

| Source | Count |
|--------|------:|
| `synthetic` | 62 |
| `commitpackft` (keyword + parse filtered) | 3182 |
| `commitpackft_corpus` (Phase 10 fill: license + quality + parse, no keyword filter) | 21396 |
| **Total** | **24640** |

History: JS/TS started at **1339** (59 synthetic + 1280 keyword-mined; the existing `lora_js_ts_7b` adapter was trained on this). Python started at **1503** in Phase 8 (scanned 56025 → kept 2459 → capped 1500 + 3 synthetic). Both were topped up in Phase 10.

Note: `js_ts_migration.jsonl` currently has **9105** valid rows vs **9103** in `corpus_manifest.json` (manifest predates 2 rows). Regenerate with `build_corpus_manifest.py` at the next corpus change.

### Train / eval splits (Phase 11)

`python training/train_lora.py --lang <id> --plan` writes `data/datasets/splits/<id>.split.json` (eval row indices, seed **1337**, 5%, corpus sha256). Training uses the train side only; the eval side is for held-out smoke eval. See [PHASE_11_LORA_PER_LANGUAGE.md](phases/PHASE_11_LORA_PER_LANGUAGE.md) §11.1.

### Multi-language corpus plan (Phases 8–10)

| Language | Dataset path (planned) | Phase | Target pairs (guide) |
|----------|------------------------|-------|----------------------:|
| JS / TS | `js_ts_migration.jsonl` | **10 done** | **9103** |
| Python | `python_migration.jsonl` | **10 done** | **5246** |
| Go | `go_migration.jsonl` | **10 done** | **1528** |
| Java | `java_migration.jsonl` | **10 done** | **2649** |
| C# | `csharp_migration.jsonl` | **10 done** | **1562** |
| Rust | `rust_migration.jsonl` | **10 done** | **1505** |
| C++ | `cpp_migration.jsonl` | **10 done** | **1512** |
| Swift | `swift_migration.jsonl` | **10 done** | **1535** |
| **Combined** | manifest / concat | **10 exit** | **≥15 000** |

Mining: extend `training/mine_migration_pairs.py` (or per-lang filters) over CommitPackFT / language-appropriate Hub corpora with the same keyword + parse gates. Record running totals in `*.stats.json` and this file after each mine.

### Phase 15 corpora + data pool (2026-11-07)

| Path | What | Tracked |
|------|------|---------|
| `data/datasets/pool/<lang>_commitpack.jsonl` | Full `bigcode/commitpack` rows (Rust: all 30 shards, C++: 60 of 365), license + size + parse + commit-quality filters, `tier` = `modern` / `generic` | DVC (`pool.dvc`) |
| `data/datasets/pool/<lang>_toolpairs.jsonl` | Tool-generated pairs, compile-checked before and after (`cargo fix`/`clippy --fix`; `clang-tidy modernize-*`), `tier` = `tool`, `checks` = fixes applied | DVC |
| `data/datasets/pool/cpp_standalone.jsonl` | Standard-library-only C++ files (tool-pair inputs only) | DVC |
| `data/datasets/p15/<lang>_migration.jsonl` | **Phase 15 training corpus**, all 8 languages: the expanded corpus minus rows from frozen eval-set files; Rust / C++ = 8,000 rows with new data (see [research/RUST_CPP_DATA_SOURCES.md](research/RUST_CPP_DATA_SOURCES.md)) | DVC stage `p15_corpus` |
| `data/datasets/evalsets/{rust,cpp}_v2.jsonl` | Bigger frozen eval sets: every old row + new held-out rows (100 each) | git (frozen) |
| `data/datasets/dpo/` | DPO prompts / preference pairs (`training/build_dpo_pairs.py`) | DVC |

```bash
python training/mine_commitpack_full.py --lang rust                    # or: --lang cpp --sample 60
python training/mine_commitpack_full.py --lang cpp --sample 60 --seed 7 --standalone
python training/make_tool_pairs.py --lang rust                          # needs rustup / clang-tidy
bash scripts/dvc.sh repro p15_corpus                                    # build_p15_corpus.py
```

### Cross-language eval problems (Phase 17, 2026-11-07)

| Path | What | Tracked |
|------|------|---------|
| `data/datasets/xlang/humaneval.jsonl` | 164 HumanEval problems: unit tests in 9 languages (MultiPL-E, MIT; Python: OpenAI HumanEval, MIT) + reference solutions in Python / C++ / Go / Java / JS (HumanEval-X, Apache-2.0) as translation sources. Built by `training/build_xlang_eval.py --download` (raw files in `data/cache/multipl_e/`, git-ignored); frozen once written | git (frozen, 2.2 MB) |
| `fixtures/xlang_oracle/humaneval.json` | Hand-written correct + wrong solutions (3–4 problems × 9 languages, incl. a list-returning C# one) — harness self-check | git |
| `fixtures/xlang_fixtures.json` | The 8 sample projects as translation sources / targets (fixture matrix) | git |

Eval only — never trained on. MBPP (MultiPL-E has it for the same languages) is the candidate
source for stage 3 training-data generation, so HumanEval stays clean.

### Upstream corpus

- Hub: [`bigcode/commitpackft`](https://huggingface.co/datasets/bigcode/commitpackft) (MIT / permissive subset); since 2026-11-07 also the full [`bigcode/commitpack`](https://huggingface.co/datasets/bigcode/commitpack) for Rust / C++ (streamed shard by shard, permissive repo licenses only)
- Local cache (gitignored): `data/cache/commitpackft/`
- Filters: migration/upgrade keywords, permissive license, size bounds, Tree-sitter parse (`--require-parse`)

## How to refresh + train

```bash
source .venv/bin/activate

# Expand classifier logs (idempotent)
python training/expand_error_logs.py
rebirth train-classifier

# Re-mine from Hub (downloads CommitPackFT once into cache)
rebirth mine-dataset --max-pairs 1500 --require-parse
python training/mine_migration_pairs.py --langs python --max-pairs 1500 --require-parse
# Phase 10 corpus fill (no keyword filter; still license+quality+parse):
python training/mine_migration_pairs.py --langs go --mode corpus --max-pairs 1500 --require-parse
python training/build_corpus_manifest.py --concat data/datasets/all_migration.jsonl

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
| `peft` (default paths) | `lora_<lang>_7b` if trained for that language; JS/TS → `lora_js_ts_7b`, else 0.5B; other langs w/o adapter → **Ollama fallback** | Local specialization |
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

- Multi-lang corpus **≥15K** — **done** (**24640**; Phase 10).  
- PEFT defaults → 7B — **done** for JS/TS; closed-loop math **PASS**.  
- Behavioral + fixture eval — **done** (JS sandbox; fixtures **8/8**).  
- **Do not** stack Ollama 14B + PEFT 7B + ST on the 7800 XT.  
- Train/eval splits + adapter metadata + per-lang PEFT routing — **done** (Phase 11.1 harness).  
- Python 7B LoRA — **done** (`lora_python_7b`; held-out results in PHASE_11 §11.3).  
- **Next:** optional JS/TS refresh, then Phase 11 waves — [PHASE_11_LORA_PER_LANGUAGE.md](phases/PHASE_11_LORA_PER_LANGUAGE.md).  
- Phase 19 / `v1.0.0` — **deferred** until LoRA waves finish.  
- Optional cloud 14B QLoRA — Track C (browser billing; per-lang fallback if local 7B insufficient).  
