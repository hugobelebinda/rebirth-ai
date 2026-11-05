# Phases 12–14 — Per-language LoRA specialization

> Status: **NEXT** (do **not** start Phase 11 launch until these waves finish or are explicitly deferred)  
> Depends on: Phases **0–10 complete** on `main` (8 adapters + corpus **24 640** + fixtures **8/8**)  
> Related: [PHASE_3_LORA_INFERENCE.md](PHASE_3_LORA_INFERENCE.md) (JS/TS proof), [DATASET.md](../DATASET.md), [PRODUCTION_ROADMAP.md](../PRODUCTION_ROADMAP.md)  
> Handoff: [SESSION_HANDOFF.md](../SESSION_HANDOFF.md)  
> Last docs refresh: 2026-11-04  

## Why this comes before Phase 11

Phases 8–10 shipped **language adapters + sandboxes + mined pairs**. Generation still defaults to **stock Ollama 14B**.  
The next product work is **specialized PEFT LoRA adapters per language** (and a refreshed JS/TS adapter on the full corpus), then wire `rebirth migrate --backend peft --lang <id>` to the right adapter.

**Phase 11 (`v1.0.0` packaging) is deferred** until Phases 12–14 exit (or you explicitly skip LoRA and launch on Ollama-only).

## Current baseline (as of Phase 10 close)

| Item | State |
|------|--------|
| Live adapters | `js_ts`, `python`, `go`, `java`, `csharp`, `rust`, `cpp`, `swift` |
| Corpus | **24 640** pairs — `data/datasets/corpus_manifest.json` |
| Existing LoRA | **JS/TS only** — `data/models/lora_js_ts_7b` (trained on **1339** pairs, 1-epoch bf16 seq 128) |
| Train script | `training/train_lora.py` (defaults still point at JS/TS paths) |
| Proven local recipe | 7B Instruct, batch 1, grad-accum 8, max-length **128**, epochs **1**, eager attn |
| Known limits | seq 512 **OOM**; 2-epoch / r=16 → **GPU Hang** on RX 7800 XT display GPU |

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
Phase 12  Harness + JS/TS refresh + Python LoRA
Phase 13  Go / Java / C# / Rust LoRAs
Phase 14  C++ / Swift LoRAs + PEFT language routing + eval gates
Phase 11  Packaging + v1.0.0  ← AFTER 12–14 (deferred)
Optional C  Cloud 14B QLoRA (any language / multi-lang) if local 7B is not enough
```

---

## Phase 12 — Training harness + JS/TS refresh + Python

**Goal:** Make per-language training a repeatable CLI path; ship a **Python** 7B adapter; optionally refresh JS/TS on the full ~9K corpus.

### 12.1 Harness (ship first)

| Task | Detail | Exit |
|------|--------|------|
| Per-lang CLI | `train_lora.py --lang python` resolves data + `--out` from a table | Documented |
| Train/eval split | Hold out ~5–10% per language (seeded); write `*.train.jsonl` / `*.eval.jsonl` or index files | Reproducible |
| Adapter metadata | Save `adapter_config` + `rebirth_meta.json` (`language`, corpus hash, base model, steps, loss) | Present under `data/models/lora_<lang>_7b/` |
| PEFT path resolution | `factory` / env: `REBIRTH_PEFT_ADAPTER` or map `language → adapter dir` | `--backend peft --lang python` loads Python adapter |
| VRAM policy | Unload Ollama before GPU train; `REBIRTH_EMBEDDER=hash` for PEFT migrate | Documented in this file + OLLAMA doc |

### 12.2 JS/TS refresh (optional but recommended)

| Task | Detail | Exit |
|------|--------|------|
| Retrain / continue | 1-epoch 7B on **full** `js_ts_migration.jsonl` (~9103) with proven hyperparams | Adapter saved; no GPU hang |
| Eval | `eval_behavioral` sandbox + fixture smoke vs previous 1339-trained adapter | No regression on fixture sandbox |

### 12.3 Python LoRA

| Task | Detail | Exit |
|------|--------|------|
| Train | 7B 1-epoch on `python_migration.jsonl` (5246) | `data/models/lora_python_7b` |
| Closed-loop | `rebirth migrate fixtures/python_sample … --backend peft --lang python` | Pass ≤5 iters |
| Smoke eval | Soft-exact / sandbox on held-out Python rows or fixture | Recorded under `data/reports/` |

**Phase 12 exit checklist**

- [ ] `--lang` (or explicit `--data`/`--out`) training path documented and used once for Python  
- [ ] `lora_python_7b` exists with metadata  
- [ ] PEFT migrate on `fixtures/python_sample` passes  
- [ ] (Preferred) refreshed `lora_js_ts_7b` on full JS/TS corpus or explicit deferral note  

---

## Phase 13 — Go, Java, C#, Rust LoRAs

**Goal:** One 7B adapter per language using the Phase 12 harness.

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

## Phase 14 — C++ / Swift + routing + quality gate

**Goal:** Finish the set; make the agent pick the right adapter; gate quality.

| Task | Detail | Exit |
|------|--------|------|
| cpp + swift train | Same 7B recipe | `lora_cpp_7b`, `lora_swift_7b` |
| Adapter registry | Map all 8 ids → adapter dirs; fall back to Ollama if missing | Unit tests |
| Multi-lang PEFT eval | Extend fixture / behavioral harness `--backend peft --lang all` | Target ≥90% fixture sandbox where adapters exist |
| Docs | DATASET + GLOBAL + export notes for multi-adapter | Synced |

**Phase 14 exit checklist**

- [ ] All eight languages have a trained adapter **or** an explicit “skip / Ollama-only” decision  
- [ ] `rebirth migrate --backend peft --lang <id>` resolves correctly for each trained lang  
- [ ] Quality report for PEFT path written  
- [ ] Ready for **Phase 11** launch packaging  

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

# Example: Python (after Phase 12 harness, prefer --lang python)
python training/train_lora.py \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --data data/datasets/python_migration.jsonl \
  --out data/models/lora_python_7b \
  --batch-size 1 --grad-accum 8 --max-length 128 \
  --epochs 1 --max-steps -1 --save-steps 50 \
  --gradient-checkpointing --attn-eager
```

Repeat with `--data` / `--out` (or `--lang`) for each language in Phases 12–14.

### Cloud fallback (Track C)

If a language OOMs, hangs, or needs longer context / 14B base: use Hugging Face Jobs (browser billing). See [PRODUCTION_ROADMAP.md](../PRODUCTION_ROADMAP.md) § Optional Track C. Cloud is **optional** per language — does not block shipping that language’s adapter vertical (already done in Phases 8–10).

---

## What not to redo

- Do **not** re-mine the 15K corpus unless a language is too thin for useful SFT (then top-up only that lang).  
- Do **not** start Phase 11 bootstrap/tag until Phases 12–14 are done or explicitly skipped.  
- Do **not** stack Ollama 14B + PEFT 7B + SentenceTransformers on the 7800 XT.  
- Do **not** jump to multi-epoch / r=16 local 7B without a non-display GPU or cloud.

## Decision log (fill as waves complete)

| Date | Decision | Notes |
|------|----------|-------|
| 2026-11-04 | Insert Phases 12–14 before Phase 11 | User: train LoRA per language next; defer launch |
| | | |
