# Exporting a LoRA adapter for Ollama

> Docs refresh: 2026-11-04 — Phases 0–10 complete; Phases 12–14 will add per-lang PEFT adapters — [PHASE_12_LORA_PER_LANGUAGE.md](../docs/phases/PHASE_12_LORA_PER_LANGUAGE.md).

Phase 3+ inference for Rebirth AI uses **Ollama** (`qwen2.5-coder:14b-instruct-q6_K` on AMD ROCm) as the default runtime.

PEFT LoRA adapters trained with `training/train_lora.py` are **not** loaded by Ollama directly. Pick one path:

## A. Runtime PEFT (agent-side, no Ollama Modelfile) — **implemented**

```bash
# 0.5B (default PEFT paths)
rebirth generate path/to/file.js --backend peft --no-rag

# 7B (preferred local adapter after 2026-11-01 train)
REBIRTH_PEFT_BASE=Qwen/Qwen2.5-Coder-7B-Instruct \
REBIRTH_PEFT_ADAPTER=data/models/lora_js_ts_7b/adapter \
rebirth generate path/to/file.js --backend peft --no-rag

REBIRTH_GENERATOR=peft rebirth migrate … --backend peft
```

Uses `src/rebirth_ai/generate/peft_client.py` (stop/decode hardened).  
Default agent path remains **Ollama 14B**.

| Adapter | Path | Notes |
|---------|------|-------|
| 0.5B | `data/models/lora_js_ts/adapter` | Weak on whole-file |
| **7B** | `data/models/lora_js_ts_7b/adapter` | 1-epoch ROCm; use this for PEFT |

## B. Merge LoRA into base weights, then convert / quantize for Ollama

1. Merge:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-Coder-7B-Instruct"  # or 0.5B
adapter = "data/models/lora_js_ts_7b/adapter"
model = AutoModelForCausalLM.from_pretrained(base)
model = PeftModel.from_pretrained(model, adapter)
merged = model.merge_and_unload()
out = "data/models/lora_js_ts_7b/merged"
merged.save_pretrained(out)
AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
```

2. Convert to GGUF (llama.cpp / conversion scripts) and create an Ollama Modelfile:

```dockerfile
FROM ./rebirth-coder.gguf
PARAMETER temperature 0.1
SYSTEM You are Rebirth AI...
```

```bash
ollama create rebirth-coder -f Modelfile
```

## C. Larger production LoRA (14B matching Ollama) — **optional**

Local ROCm proved **7B bf16 LoRA** (seq 128, 1 epoch). Multi-epoch / larger-r hung the display GPU.

This track is **not required** for production-grade multi-language (see `docs/PRODUCTION_ROADMAP.md`).  
Do it only if you want a LoRA that matches the Ollama 14B brain.

### Billing (no card in the terminal)

1. Browser: [HF payment settings](https://huggingface.co/settings/billing/payment) — add card (Stripe), keep positive credits.  
2. Terminal: `hf auth login` (write token).  
3. Submit a Job; billed per minute while Starting/Running. Push adapter to the Hub before the job ends.

For a **14B-class** adapter:

- Train on Hugging Face Jobs with QLoRA / PEFT (~$1–3 for a short run on T4/L4).
- Keep the same `data/datasets/js_ts_migration.jsonl` format.
- Merge/export as above, or keep PEFT inference in-process.

## Local train reminders

```bash
bash scripts/verify_rocm_torch.sh   # must PASS
# smoke defaults: 7B, seq 128, 5 steps
bash scripts/train_lora_gpu.sh
# full 1-epoch: see docs/DATASET.md (do not casually retry 2-epoch / r=16)
```
