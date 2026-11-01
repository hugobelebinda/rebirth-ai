# Exporting a LoRA adapter for Ollama

Phase 3 inference for Rebirth AI uses **Ollama** (`qwen2.5-coder:14b-instruct-q6_K` on AMD ROCm) as the runtime.

PEFT LoRA adapters trained with `training/train_lora.py` are **not** loaded by Ollama directly. Pick one path:

## A. Runtime PEFT (agent-side, no Ollama Modelfile)

Keep serving the base model in Ollama for fast drafts, and optionally run a separate PEFT generate path in Python (`transformers` + `peft`) when a specialized adapter is required.

This is the Phase 3 default: **Ollama client for generation**, LoRA train script for producing adapters.

## B. Merge LoRA into base weights, then convert / quantize for Ollama

1. Merge:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-Coder-0.5B-Instruct"  # or your train base
adapter = "data/models/lora_js_ts/adapter"
model = AutoModelForCausalLM.from_pretrained(base)
model = PeftModel.from_pretrained(model, adapter)
merged = model.merge_and_unload()
out = "data/models/lora_js_ts/merged"
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

## C. Larger production LoRA

Local smoke uses `Qwen2.5-Coder-0.5B-Instruct`. For a 14B-class adapter matching the Ollama runtime:

- Train on NVIDIA CUDA or Hugging Face Jobs with `bitsandbytes` 4-bit + PEFT.
- Keep the same `data/datasets/js_ts_migration.jsonl` format.
- Merge/export as above, or keep PEFT inference in-process.

## Smoke train (this machine)

```bash
pip install -e ".[ml]"
python training/train_lora.py --smoke
```
