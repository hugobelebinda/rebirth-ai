# Phase 3 — Code Generation & LoRA Training

> Status: **PLANNED**  
> Depends on: Phase 2 (RAG context available for prompts)

---

## 1. Objectives

- Curate synthetic `(legacy, modern)` JS/TS pairs.  
- Train a PEFT LoRA adapter on a coder base (Qwen2.5-Coder family preferred).  
- Serve inference via Ollama (base) + documented LoRA load/merge path.  
- Expose a thin Python client used later by LangGraph.

---

## 2. Planned deliverables

| Path | Responsibility |
|------|----------------|
| `data/datasets/js_ts_migration.jsonl` | 50–200 synthetic pairs to start |
| `training/train_lora.py` | PEFT training script (CPU-friendly defaults) |
| `training/export_for_ollama.md` | Merge / Modelfile notes |
| `src/rebirth_ai/generate/ollama_client.py` | HTTP generate API |
| `src/rebirth_ai/generate/prompts.py` | Migration prompt templates |

### Machine notes (updated 2026-10-31)

- Default inference: `qwen2.5-coder:14b-instruct-q6_K` on AMD ROCm (RX 7800 XT, 16GB, 100% VRAM).  
- Also available: `qwen2.5-coder:14b` (Q4 fallback), `deepseek-coder:6.7b`.  
- No NVIDIA GPU required for MVP; keep LoRA rank small; optional HF Jobs if local train is too slow.  
- See `docs/OLLAMA_AMD_GPU.md`.

---

## 3. Exit criteria

- [ ] Dataset checked in (or download script documented)  
- [ ] Train script runs end-to-end on a smoke subset  
- [ ] Client returns a draft given legacy file + RAG context  
- [ ] Docs updated with exact commands and hyperparameters  

---

## 4. What shipped / How to run / Sample outputs / Decision log

_Filled when Phase 3 is implemented._
