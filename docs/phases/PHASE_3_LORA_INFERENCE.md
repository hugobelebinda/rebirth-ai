# Phase 3 — Code Generation & LoRA Training

> Status: **COMPLETE** (data/PEFT interlude continues in `docs/DATASET.md`)  
> Depends on: Phase 2 (RAG context available for prompts)  
> Completed: 2026-10-31 · Docs refresh: 2026-11-01

---

## 1. Objectives

- Curate synthetic `(legacy, modern)` JS/TS pairs.  
- Train a PEFT LoRA adapter on a coder base (smoke-scale locally).  
- Serve generation locally via Ollama + documented LoRA export path.  
- Expose a thin Python client used later by LangGraph.

---

## 2. Deliverables

| Path | Responsibility | Status |
|------|----------------|--------|
| `data/datasets/js_ts_migration.jsonl` | **1339** pairs (59 synthetic + CommitPackFT) | Shipped + mined |
| `training/mine_migration_pairs.py` | Hub miner + filters | Shipped |
| `docs/DATASET.md` | Dataset bible | Shipped |
| `training/train_lora.py` | PEFT training (CPU/GPU, checkpointing, save-steps) | Shipped |
| `training/eval_peft_smoke.py` | 8-pair exact/overlap smoke | Shipped |
| `training/export_for_ollama.md` | Merge / Modelfile notes | Shipped |
| `scripts/verify_rocm_torch.sh` / `train_lora_gpu.sh` | ROCm health + GPU train wrapper | Shipped |
| `src/rebirth_ai/generate/ollama_client.py` | HTTP generate API | Shipped |
| `src/rebirth_ai/generate/peft_client.py` | In-process LoRA (+ stop/decode) | Shipped |
| `src/rebirth_ai/generate/factory.py` | `ollama` \| `peft` backend select | Shipped |
| `src/rebirth_ai/generate/prompts.py` | Migration prompts + fence strip | Shipped |
| CLI `rebirth generate` / `--backend` | File → Ollama or PEFT draft | Shipped |

### Runtime model (this machine)

- Default agent: `qwen2.5-coder:14b-instruct-q6_K` on AMD ROCm (RX 7800 XT)  
- PEFT 0.5B: `data/models/lora_js_ts` — pipeline proof only  
- PEFT **7B** (preferred local adapter): `data/models/lora_js_ts_7b` — 1-epoch bf16 seq 128 on ROCm

---

## 3. Exit criteria

- [x] Dataset checked in (`js_ts_migration.jsonl`, 59 rows)  
- [x] Train script runs end-to-end on smoke subset (`--smoke`, adapter saved)  
- [x] Client returns a draft given legacy file + RAG context (live Ollama verified)  
- [x] Docs updated with commands and hyperparameters  

---

## 4. How to run

```bash
cd "/home/hugobele/Documents/Personal Project/Rebirth AI"
source .venv/bin/activate

# Live generation (uses Ollama on ROCm)
rebirth generate fixtures/js_ts_sample/src/math.js --objective "Modernize var/legacy patterns"

# Optional: ensure RAG DB exists first
REBIRTH_EMBEDDER=hash rebirth index fixtures/js_ts_sample
rebirth generate fixtures/js_ts_sample/src/math.js --rag

# LoRA smoke train (installs PEFT stack)
pip install -e ".[ml]"   # or: pip install peft datasets accelerate
python training/train_lora.py --smoke
```

### Smoke train hyperparameters

| Knob | Smoke value |
|------|-------------|
| base | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| max_steps | 2 |
| lora_r / alpha | 8 / 16 |
| batch | 1 |
| max_length | 512 |
| dtype | float32 (no bitsandbytes; AMD-friendly) |

### Sample outputs

```text
$ python -c "from rebirth_ai.generate.ollama_client import OllamaClient; ..."
MODEL qwen2.5-coder:14b-instruct-q6_K
CODE:
function add(a, b) {
  const total = a + b;
  return total;
}

$ python training/train_lora.py --smoke
Saved LoRA adapter → data/models/lora_js_ts/adapter
```

### Decision log

| Decision | Detail |
|----------|--------|
| Inference path | Ollama HTTP for agent drafts (matches deployed 14B Q6) |
| LoRA smoke base | 0.5B instruct coder — proves PEFT pipeline without 14B train cost |
| Export | Documented merge→GGUF/Modelfile; not auto-wired into Ollama yet |
| Prompting | Low temperature; output = full file; optional RAG + error class |

---

## 5. Known gaps / post-phase status (2026-11-01)

- Full **14B** LoRA matching Ollama still needs CUDA/HF Jobs (`export_for_ollama.md`).  
- Adapters under `data/models/` are gitignored — re-train or keep local.  
- PEFT CLI path **exists** (`--backend peft`); default migrate still Ollama.  
- Local multi-epoch / larger-r 7B train **hung GPU** — do not retry casually.  
- Exact-match eval understates 7B (see `docs/DATASET.md`).

---

## 6. Handoff

Phase 4 closed the LangGraph loop on Ollama. Ongoing: prefer 7B PEFT when specializing; Phase 5 for repo-scale PRs.
