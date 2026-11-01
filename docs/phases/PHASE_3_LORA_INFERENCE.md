# Phase 3 — Code Generation & LoRA Training

> Status: **COMPLETE**  
> Depends on: Phase 2 (RAG context available for prompts)  
> Completed: 2026-10-31

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
| `data/datasets/js_ts_migration.jsonl` | 59 synthetic pairs | Shipped |
| `training/train_lora.py` | PEFT training (`--smoke` supported) | Shipped |
| `training/export_for_ollama.md` | Merge / Modelfile notes | Shipped |
| `src/rebirth_ai/generate/ollama_client.py` | HTTP generate API | Shipped |
| `src/rebirth_ai/generate/prompts.py` | Migration prompts + fence strip | Shipped |
| CLI `rebirth generate` | File → Ollama draft (+ optional RAG) | Shipped |

### Runtime model (this machine)

- Default: `qwen2.5-coder:14b-instruct-q6_K` on AMD ROCm (RX 7800 XT)  
- LoRA smoke base: `Qwen/Qwen2.5-Coder-0.5B-Instruct` (local PEFT train; not the Ollama runtime)

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

## 5. Known gaps

- Full 14B LoRA matching the Ollama runtime needs CUDA/HF Jobs (see `export_for_ollama.md`).  
- Adapter artifacts under `data/models/` are largely gitignored (re-run `--smoke` locally).  
- No automatic PEFT inference path in the CLI yet — Phase 4 uses Ollama client.

---

## 6. Handoff to Phase 4

Wire `Retriever` + `OllamaClient` + `SandboxRunner` + `ErrorClassifier` into a LangGraph loop on one fixture file.
