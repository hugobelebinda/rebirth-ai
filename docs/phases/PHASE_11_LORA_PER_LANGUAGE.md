# Phase 11 — Per-language LoRA specialization (old Phases 12–14)

> Status: **COMPLETE (2026-11-05)** — harness ✅ · v2 recipe (4-bit QLoRA @512, EOS, completion-only loss) ✅ · adapters for all 8 languages ✅, all pass the gate vs base 7B after the eval token-cap fix. Continued in [PHASE_12_LOCAL_TUNING_V3.md](PHASE_12_LOCAL_TUNING_V3.md).  
> Numbering: this phase was "Phases 12–14" before the 2026-11-05 renumbering (wave A = old 13, wave B = old 14).  
> Depends on: Phases **0–10 complete** on `main` (8 adapters + corpus **24 640** + fixtures **8/8**)  
> Related: [PHASE_3_LORA_INFERENCE.md](PHASE_3_LORA_INFERENCE.md) (JS/TS proof), [DATASET.md](../DATASET.md), [PRODUCTION_ROADMAP.md](../PRODUCTION_ROADMAP.md)  
> Handoff: [SESSION_HANDOFF.md](../SESSION_HANDOFF.md)  
> Last docs refresh: 2026-11-04  

## Why this comes before the launch (Phase 19)

Phases 8–10 shipped **language adapters + sandboxes + mined pairs**. Generation still defaults to **stock Ollama 14B**.  
The next product work is **specialized PEFT LoRA adapters per language** (and a refreshed JS/TS adapter on the full corpus), then wire `rebirth migrate --backend peft --lang <id>` to the right adapter.

**Phase 19 (`v1.0.0` packaging) is deferred** until this phase exits (or you explicitly skip LoRA and launch on Ollama-only).

## Current baseline (as of Phase 10 close)

| Item | State |
|------|--------|
| Live adapters | `js_ts`, `python`, `go`, `java`, `csharp`, `rust`, `cpp`, `swift` |
| Corpus | **24 640** pairs — `data/datasets/corpus_manifest.json` |
| Existing LoRA | `lora_js_ts_7b` (**1339** pairs, pre-harness) · `lora_python_7b` (**4988** train rows, Phase 11) |
| Train script | `training/train_lora.py --lang <id>` (Phase 11.1 harness; no-flag defaults still JS/TS 0.5B for back-compat) |
| Proven local recipe | 7B Instruct, batch 1, grad-accum 8, max-length **128**, epochs **1**, eager attn |
| Known limits | seq 512 **OOM**; seq **256 → GPU reset** at step 1 (2026-11-04); 2-epoch / r=16 → **GPU Hang** on RX 7800 XT display GPU |

### Corpus available for training

| Lang | Dataset | Pairs | Suggested adapter out |
|------|---------|------:|------------------------|
| js_ts | `data/datasets/js_ts_migration.jsonl` | 9103 | `data/models/lora_js_ts_7b` (refresh) |
| python | `data/datasets/python_migration.jsonl` | 5246 | `data/models/lora_python_7b` |
| java | `data/datasets/java_migration.jsonl` | 2649 | `data/models/lora_java_7b` |
| csharp | `data/datasets/csharp_migration.jsonl` | 1562 | `data/models/lora_csharp_7b` |
| go | `data/datasets/go_migration.jsonl` | 1528 | `data/models/lora_go_7b` |
| swift | `data/datasets/swift_migration.jsonl` | 1535 | `data/models/lora_swift_7b` |
| cpp | `data/datasets/cpp_migration.jsonl` | 1512 | `data/models/lora_cpp_7b` |
| rust | `data/datasets/rust_migration.jsonl` | 1505 | `data/models/lora_rust_7b` |

## Phase map

```
Phase 11            Harness + JS/TS refresh + Python LoRA                 ✅
Phase 11 (wave A)   Go / Java / C# / Rust LoRAs                           ✅
Phase 11 (wave B)   C++ / Swift LoRAs + PEFT language routing + eval gates ✅
Phase 12            Local tuning v3 + eval hardening                      ← now
Phase 15            Cloud 14B adapters (replaces "Optional C")
Phase 19            Packaging + v1.0.0 (last)
```

---

## Phase 11 — Training harness + JS/TS refresh + Python

**Goal:** Make per-language training a repeatable CLI path; ship a **Python** 7B adapter; optionally refresh JS/TS on the full ~9K corpus.

### 11.1 Harness (ship first) — **SHIPPED 2026-11-04**

| Path | Role |
|------|------|
| `src/rebirth_ai/lora.py` | Lang table (`lora_target`), `normalize_lang`, content-hash split, split index, `read_meta` |
| `training/train_lora.py` | `--lang`, `--eval-frac`, `--seed`, `--plan`; trains on **train split only**; writes `rebirth_meta.json` |
| `data/datasets/splits/<lang>.split.json` | Eval row indices + seed + corpus sha256 (tracked; ~1–5 KB each) |
| `src/rebirth_ai/generate/peft_client.py` | `resolve_peft_paths(language=…)`, `lang_adapter()` (base model read from meta) |
| `src/rebirth_ai/generate/factory.py` | `get_generator(backend, language=…)` — PEFT per lang, **Ollama fallback** if that lang has no adapter |
| `src/rebirth_ai/graph/nodes.py` | `generate_code` passes `adapter_id` → factory |
| `tests/test_lora_harness.py` | 17 unit tests (registry, split stability, routing, `--plan`) |

**Split rule:** row → eval iff `sha256(seed|objective|legacy|modern)` bucket < `eval_frac` (default **0.05**, seed **1337**). Stable when rows are appended; duplicate rows never straddle train/eval. `corpus_sha256` in the index detects corpus drift.

**Routing order** (`--backend peft`): explicit args → `REBIRTH_PEFT_BASE` / `REBIRTH_PEFT_ADAPTER` → `data/models/lora_<lang>_7b/adapter` → JS/TS 7B → 0.5B. A non-JS language without its own adapter (and no env override) now **falls back to Ollama** instead of running the JS/TS adapter.

**Split sizes (seed 1337, 5%):**

| Lang | Rows | Train | Eval |
|------|-----:|------:|-----:|
| js_ts | 9105 | 8672 | 433 |
| python | 5246 | 4988 | 258 |
| java | 2649 | 2508 | 141 |
| csharp | 1562 | 1486 | 76 |
| go | 1528 | 1436 | 92 |
| swift | 1535 | 1457 | 78 |
| cpp | 1512 | 1429 | 83 |
| rust | 1505 | 1439 | 66 |

`rebirth_meta.json` (written next to `adapter/`): `language`, `base_model`, `corpus_sha256`, `corpus_rows`, `train_rows`, `eval_rows`, split seed/index, `global_steps`, `train_loss`, hyperparams, device, torch, `trained_at`. Verified via a 2-step CPU 0.5B `--smoke` run (not kept).

Original task table:


| Task | Detail | Exit |
|------|--------|------|
| Per-lang CLI | `train_lora.py --lang python` resolves data + `--out` from a table | Documented |
| Train/eval split | Hold out ~5–10% per language (seeded); write `*.train.jsonl` / `*.eval.jsonl` or index files | Reproducible |
| Adapter metadata | Save `adapter_config` + `rebirth_meta.json` (`language`, corpus hash, base model, steps, loss) | Present under `data/models/lora_<lang>_7b/` |
| PEFT path resolution | `factory` / env: `REBIRTH_PEFT_ADAPTER` or map `language → adapter dir` | `--backend peft --lang python` loads Python adapter |
| VRAM policy | Unload Ollama before GPU train; `REBIRTH_EMBEDDER=hash` for PEFT migrate | Documented in this file + OLLAMA doc |

### 11.2 JS/TS refresh (optional but recommended)

| Task | Detail | Exit |
|------|--------|------|
| Retrain / continue | 1-epoch 7B on **full** `js_ts_migration.jsonl` (~9103) with proven hyperparams | Adapter saved; no GPU hang |
| Eval | `eval_behavioral` sandbox + fixture smoke vs previous 1339-trained adapter | No regression on fixture sandbox |

### 11.3 Python LoRA

| Task | Detail | Exit |
|------|--------|------|
| Train | 7B 1-epoch on `python_migration.jsonl` (5246) | `data/models/lora_python_7b` |
| Closed-loop | `rebirth migrate fixtures/python_sample … --backend peft --lang python` | Pass ≤5 iters |
| Smoke eval | Soft-exact / sandbox on held-out Python rows or fixture | Recorded under `data/reports/` |

### 11.3 Results (2026-11-04)

**Train** — `train_lora.py --lang python --batch-size 1 --grad-accum 8 --max-length 128 --gradient-checkpointing --attn-eager` (r=8, 1 epoch): **624 steps, 1020 s (~17 min), train loss 1.121** (first logged ≈2.6). No GPU hang. Log: `logs/train_lora_python_7b.log`; meta: `data/models/lora_python_7b/rebirth_meta.json`.

**Closed loop** — `rebirth migrate fixtures/python_sample … --lang python --backend peft` → **success, 1 iteration** (loaded `lora_python_7b`). Report: `data/reports/migrate_python_sample_peft7b.json`.

**Held-out split** (24 shortest rows of the 258-row Python eval split, never trained on; `eval_behavioral.py --suite synthetic --lang python --n-synth 24`):

| Backend | Soft-exact | Parses | Mean symbol Jaccard | Mean token overlap |
|---------|-----------:|-------:|--------------------:|-------------------:|
| PEFT 7B + `lora_python_7b` | 4/24 | 20/24 | **0.910** | 0.679 |
| Ollama 14B (default) | **8/24** | **23/24** | 0.875 | **0.719** |

Reports: `data/reports/eval_heldout_python_{peft7b,ollama}.json`. Rows are real CommitPackFT edits (version bumps, import cleanups), so soft-exact is a strict bar. Remaining PEFT parse fails = over-generation cut by the token cap (seq-128 training never shows where long completions stop).

**Decode fix found by this eval:** `_clean_sft_completion` now drops a trailing lone ```` ``` ```` and cuts at leaked Qwen control tokens (`<|repo_name|>`, `<|file_sep|>`, …) which `skip_special_tokens` does not remove. PEFT went 2→4 soft-exact, 14→20 parses on the same rows.

**No-adapter ablation** (`REBIRTH_PEFT_ADAPTER=none`, same 24 rows):

| Backend | Soft-exact | Parses | Sym Jaccard | Token overlap |
|---------|-----------:|-------:|------------:|--------------:|
| 7B + `lora_python_7b` (SFT prompt) | 4/24 | 20/24 | 0.910 | 0.679 |
| 7B base, SFT prompt | 4/24 | **24/24** | **0.958** | 0.623 |
| 7B base, chat prompt | 7/24 | **24/24** | 0.917 | **0.725** |
| Ollama 14B | **8/24** | 23/24 | 0.875 | 0.719 |

Reports: `data/reports/eval_heldout_python_base7b_{sft,chat}.json`.

**Verdict:** the seq-128 adapter adds **nothing** over its own base (base+chat beats it; the adapter introduces the parse failures). Keep **Ollama default**; `lora_python_7b` = experimental, superseded once the recipe is fixed.

### Why the seq-128 recipe underperforms (measured 2026-11-04)

Qwen tokenizer, `format_example` (objective + legacy + modern), right-truncation:

| Lang | Rows | Median tok | Fits 128 | Modern partly cut @128 | **Modern fully cut @128** | Modern fully cut @256 | Fits 512 |
|------|-----:|-----------:|---------:|------------------------:|--------------------------:|----------------------:|---------:|
| python | 5246 | 168 | 27% | 50% | **23%** | 3% | 98% |
| go | 1528 | 266 | 9% | 40% | **51%** | 1.5% | 99% |
| js_ts | 9105 | 178 | 23% | 61% | **17%** | 4% | 96% |

Plus two recipe bugs independent of length:
1. **No EOS** — Qwen's tokenizer does not append `<|im_end|>`/`<|endoftext|>`, so the model never sees an example end → over-generation until the token cap.
2. **Loss on the whole sequence** (`DataCollatorForLanguageModeling`) — most gradient goes to reproducing objective + legacy, not the modern target. Also pad = `<|endoftext|>`, so an appended `<|endoftext|>` would be masked as padding; use `<|im_end|>` or a custom collator.

Fix candidates (not yet implemented): append EOS, completion-only loss (mask prompt tokens to -100), drop or left-truncate rows whose modern does not fit, seq ≥256 if VRAM allows.

### GPU limit test — Go @ seq 256 (2026-11-04) → **GPU reset**

Run: `bash scripts/train_logged.sh go 256` → `logs/train/go_seq256_20261105T041706Z/` (all logs kept; `postmortem.txt` inside).

| Time (EST) | Event (source) |
|------------|----------------|
| 23:17:06 | run start (`run.json`) |
| 23:17:31.7 | `train_begin` — weights alone **14.6 GB allocated** / 16 GB; `step_begin` 1 (`steps.jsonl`) |
| 23:17:34.9 | last telemetry: VRAM 15.3–15.6 GB, **GTT 0.7 → 2.0 GB** (spill to system RAM), GPU 98% busy, junction **55 °C** (`telemetry.jsonl`) |
| 23:17:35 | `MES failed to respond to msg=REMOVE_QUEUE` → `MES might be in unrecoverable state` → GPU reset (`kernel.log`) |
| 23:17:37 | trainer aborts, exit **134** (SIGABRT) (`exit.json`) |
| 23:17:39 | MODE1 reset **succeeded**, "VRAM is lost" — desktop survived, but every GPU client (editor, Claude session) lost its context |

**Diagnosis:** step 1 never finished. Not thermal (55 °C). Memory pressure: bf16 7B weights already fill ~91% of the card shared with the display; 256-token activations pushed allocations into GTT and the amdgpu SVM eviction path (`svm_range_deferred_list_work hogged CPU … 67 times` just before) until the MES firmware timed out. Same signature as the earlier 2-epoch r=16 hang "at step 0". The kernel also logs `SMU driver if version not matched` (driver vs firmware) on resume — may make MES less tolerant, unconfirmed.

**Conclusion:** on this card, with the display on it, **bf16 7B LoRA is capped at seq 128**. 256 is not reachable by tuning batch/grad-accum (already batch 1 + checkpointing).

Ways to get ≥256 (pick one; not yet done):
1. **Display on the iGPU** (plug the monitor into the motherboard; Raphael iGPU) → frees ~1.2–1.5 GB on the 7800 XT and a hang no longer kills the desktop. Retry 256.
2. **Smaller base** (Qwen2.5-Coder-3B bf16 ≈ 6 GB) → 512 fits locally.
3. **Cloud (Track C)** — 7B/14B at 512 on an L4/A10G, ~$1–3 per language.
4. 4-bit base (QLoRA) — bitsandbytes ROCm support on gfx1101 unverified.

### Local fix: 4-bit QLoRA + recipe v2 (2026-11-04) — **validated, not yet used for a real train**

**4-bit probe** (`training/qlora_probe.py`, logs `logs/train/qlora_probe_20261105T043200Z/`): bitsandbytes 0.50.2 ships `libbitsandbytes_rocm64.so`; self-test passes on gfx1101.

| Check | Result |
|-------|--------|
| NF4 7B weights | **5.3 GB** (vs 14.6 GB bf16) |
| LoRA step @256 / @512 (fwd+bwd+AdamW) | 1.39 s / **1.75 s**, peak **8.75 GB**, no GTT spill, 0 kernel errors |
| Real-code loss NF4 vs bf16 | **0.640 vs 0.636** → training math correct |
| NF4 `generate()` with KV cache | **garbage** (all NF4/FP4 variants) — ROCm single-token `gemv_4bit` kernel bug |
| NF4 `generate(use_cache=False)` | correct code → bug is only the cached decode path |
| int8 | fully correct incl. generation, 8.4 GB |

Training never hits the broken kernel, and inference loads adapters on the **bf16** base (`PeftClient`), so 4-bit is used for training only.

**Recipe v2** (`train_lora.py`, defaults ON; `--no-eos` / `--no-completion-only` / `--no-drop-overlong` reproduce v1): `build_features()` appends EOS `<|im_end|>` (≠ pad `<|endoftext|>`), masks prompt labels to -100, drops rows longer than `--max-length`; `DataCollatorForSeq2Seq` pads labels with -100. `--quant 4bit|8bit` loads via bitsandbytes + `prepare_model_for_kbit_training`. Recipe + feature stats recorded in `rebirth_meta.json`. 5 new unit tests.

**End-to-end validation** (`logs/train/go_seq512_20261105T043827Z/`):
- 2-step smoke `--quant 4bit --max-length 512` → exit 0, adapter + meta written.
- **Overfit test** (16 Go rows, 6 epochs, grad-accum 2, lr 5e-4): loss 0.38 → **0.05**; then loaded on the **bf16** base via `PeftClient` (SFT prompt) on those rows: **12/16 soft-exact, token overlap 0.935, 0 runaway outputs** vs base 7B **4/16, 0.721, 1 runaway** → trains on 4-bit, transfers to bf16, learns targets **and** stopping.
- Peak 9.6 GB torch / 11.5 GB card, junction 78 °C, 0 kernel errors.

**Real-run projections @512** (1.16 s per row measured):

| Lang | Train rows | Kept @512 | Dropped | Median tok | ETA (1 epoch) |
|------|-----------:|----------:|--------:|-----------:|--------------:|
| go | 1436 | 1418 | 18 | 266 | ~27 min |
| python | 4988 | 4870 | 118 | 167 | ~80–95 min |

**Before the real Go train (tomorrow):**
1. `_sft_max_new_tokens` in `peft_client.py` caps SFT decode at **256** tokens — targets trained at 512 can be ~400 tokens, so raise the cap (now that EOS is learned, a higher cap is safe).
2. Then: `bash scripts/train_logged.sh go 512 --quant 4bit` → held-out eval vs base 7B (chat + SFT) and Ollama: `eval_behavioral.py --suite synthetic --lang go --n-synth 24`, plus `rebirth migrate fixtures/go_sample … --backend peft --lang go`.
3. If Go beats base 7B: retrain Python with v2 @512, then Phase 11 (wave A).

### Hang-proof training logs

`bash scripts/train_logged.sh <lang> <max_len> [extra args]` → kept run dir `logs/train/<lang>_seq<N>_<UTC>/` with `run.json`, `train.log`, `steps.jsonl` (per-step begin/end + VRAM, fsync'd), `telemetry.jsonl` (VRAM / busy / temps / power every 2 s, fsync'd; `training/gpu_telemetry.py`), `kernel.log`, `exit.json` (absent ⇒ died mid-run). After a hard freeze + reboot: `bash scripts/gpu_hang_postmortem.sh <run dir>` (reads `journalctl -k -b -1`; journal is persistent).

**Phase 11 exit checklist**

- [x] `--lang` training path documented and used for Python (plan + full GPU train)  
- [x] `lora_python_7b` exists with metadata (`rebirth_meta.json`)  
- [x] PEFT migrate on `fixtures/python_sample` passes (1 iteration)  
- [ ] (Preferred) refreshed `lora_js_ts_7b` on full JS/TS corpus or explicit deferral note  

---

## Phase 11 (wave A) — Go, Java, C#, Rust LoRAs

**Goal:** One 7B adapter per language using the Phase 11 harness.

| Lang | Pairs | Fixture gate |
|------|------:|--------------|
| go | 1528 | `fixtures/go_sample` migrate `--backend peft` |
| java | 2649 | `fixtures/java_sample` |
| csharp | 1562 | `fixtures/csharp_sample` |
| rust | 1505 | `fixtures/rust_sample` |

**Exit checklist**

- [ ] Four adapters under `data/models/lora_{go,java,csharp,rust}_7b/`  
- [ ] Each language: fixture sandbox pass with PEFT (or documented fail + Ollama fallback)  
- [ ] Reports committed or linked under `data/reports/`  

**Note:** Smaller corpora (~1.5K) may underfit vs Python/JS. Acceptable outcomes: (a) still beat stock PEFT-none on fixture, (b) keep Ollama default for that lang and mark adapter experimental. Record the choice in this doc’s decision log when executed.

---

## Phase 11 (wave B) — C++ / Swift + routing + quality gate

**Goal:** Finish the set; make the agent pick the right adapter; gate quality.

| Task | Detail | Exit |
|------|--------|------|
| cpp + swift train | Same 7B recipe | `lora_cpp_7b`, `lora_swift_7b` |
| Adapter registry | Map all 8 ids → adapter dirs; fall back to Ollama if missing | Unit tests |
| Multi-lang PEFT eval | Extend fixture / behavioral harness `--backend peft --lang all` | Target ≥90% fixture sandbox where adapters exist |
| Docs | DATASET + GLOBAL + export notes for multi-adapter | Synced |

**Phase 11 (wave B) exit checklist**

- [x] All eight languages have a trained adapter (v2, 2026-11-05)  
- [x] `rebirth migrate --backend peft --lang <id>` resolves correctly for each trained lang (fixture migrate success in 1 iteration, all 8)  
- [x] Quality report for PEFT path written (`data/reports/gate_<lang>_v2.json`, `eval_heldout_<lang>_*`)  
- [x] Handed over to Phase 12 (local tuning v3)  

---

## Proven train command (template)

Unload Ollama first. Display-on-dGPU: prefer short 1-epoch runs; see [OLLAMA_AMD_GPU.md](../OLLAMA_AMD_GPU.md).

```bash
cd "/home/hugobele/Documents/Personal Project/Rebirth AI"
source .venv/bin/activate
pip install -e ".[ml]"
ollama stop qwen2.5-coder:14b-instruct-q6_K || true
source scripts/rocm_env.sh
bash scripts/verify_rocm_torch.sh
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0

# Dry run first: resolves paths + writes split index, no torch
python training/train_lora.py --lang python --plan

# Python 7B (--lang resolves data/out and defaults base to 7B Instruct)
python training/train_lora.py --lang python \
  --batch-size 1 --grad-accum 8 --max-length 128 \
  --epochs 1 --max-steps -1 --save-steps 50 \
  --gradient-checkpointing --attn-eager
```

Repeat with `--lang <id>` for each language in Phase 11. Then: `rebirth migrate fixtures/<id>_sample … --backend peft --lang <id>` (unload Ollama first; `REBIRTH_EMBEDDER=hash`).

### Cloud fallback (Track C)

If a language OOMs, hangs, or needs longer context / 14B base: use Hugging Face Jobs (browser billing). See [PRODUCTION_ROADMAP.md](../PRODUCTION_ROADMAP.md) § Optional Track C. Cloud is **optional** per language — does not block shipping that language’s adapter vertical (already done in Phases 8–10).

---

## What not to redo

- Do **not** re-mine the 15K corpus unless a language is too thin for useful SFT (then top-up only that lang).  
- Do **not** start Phase 19 bootstrap/tag until the LoRA phases are done or explicitly skipped.  
- Do **not** stack Ollama 14B + PEFT 7B + SentenceTransformers on the 7800 XT.  
- Do **not** jump to multi-epoch / r=16 local 7B without a non-display GPU or cloud.

## Decision log (fill as waves complete)

| Date | Decision | Notes |
|------|----------|-------|
| 2026-11-04 | Insert per-lang LoRA phases (then numbered 12–14) before launch (then Phase 11) | User: train LoRA per language next; defer launch |
| 2026-11-04 | 11.1 harness shipped | Split = index files (not copied jsonl); content-hash bucket, 5% / seed 1337 |
| 2026-11-04 | PEFT non-JS w/o adapter → Ollama | Previously silently used the JS/TS adapter on any language |
| 2026-11-04 | `--lang` defaults base to 7B | Avoid 0.5B landing in a `_7b` dir; no-flag defaults unchanged |
| 2026-11-04 | Track `lora_*_7b/*.json` | `.gitignore` negation; weights stay ignored |
| 2026-11-04 | `corpus_manifest.json` js_ts = 9103 but file has 9105 valid rows | Manifest predates 2 rows; regenerate at next corpus touch |
| 2026-11-04 | Python LoRA trained; Ollama stays default | Fixture PASS (1 iter); held-out soft-exact 4/24 vs Ollama 8/24 → adapter experimental |
| 2026-11-04 | Base-7B ablation | Python LoRA ≤ its own base (base+chat 7/24 vs adapter 4/24) → recipe is the bottleneck |
| 2026-11-04 | Go @ seq 256 → GPU MODE1 reset at step 1 | Memory pressure (14.6 GB weights + display); local bf16 7B capped at 128 |
| 2026-11-04 | Local 4-bit QLoRA instead of cloud | NF4 train math matches bf16 (loss 0.640 vs 0.636); seq 512 at 8.75 GB; cloud not needed |
| 2026-11-04 | Recipe v2 default | EOS + completion-only loss + drop overlong; overfit test 12/16 vs base 4/16, 0 runaways |
| 2026-11-04 | `eval_behavioral.py --lang` | Synthetic suite samples the language's held-out split + routes PEFT by language |
