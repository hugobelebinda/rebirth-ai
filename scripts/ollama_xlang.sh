#!/usr/bin/env bash
# Package the Phase 17 shared cross-language adapter (lora_xlang_14b) as an Ollama model, applied at
# run time (v1). Superseded by v2 (the same LoRA merged before quantizing, scripts/xlang_v2_merge.sh,
# installed by `rebirth install`); `migrate --to` serves whichever version the translation route points
# rebirth-xlang:14b at (rebirth serve publish-xlang / rollback --lang xlang), or REBIRTH_XLANG_MODEL.
#   bash scripts/ollama_xlang.sh [adapter dir] [tag]
# Defaults: data/models/lora_xlang_14b/adapter (dvc pull data/models/lora_xlang_14b/adapter.dvc),
# tag rebirth-xlang:14b-v1. Needs the q6_K 14B in Ollama (ollama pull qwen2.5-coder:14b-instruct-q6_K)
# and the pinned llama.cpp checkout in .tools/llama.cpp (convert_lora_to_gguf.py).
#
# Unlike the 7B same-language adapters (raw "### Objective" template, scripts/ollama_adapter.sh), this
# adapter keeps the base model's chat template: `migrate --to` sends it the product prompt + system
# prompt, exactly as it was evaluated (program suite, q6_K + this GGUF: 265 / 420 translations
# verified vs 208 for the plain 14B, 2026-11-09).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adapter="${1:-$ROOT/data/models/lora_xlang_14b/adapter}"
tag="${2:-rebirth-xlang:14b-v1}"
from="${XLANG_FROM:-qwen2.5-coder:14b-instruct-q6_K}"
gguf="${XLANG_GGUF:-$ROOT/data/models/gguf/xlang_14b_v1.f16.gguf}"
PY="$ROOT/.venv/bin/python"
[ -f "$adapter/adapter_model.safetensors" ] || { echo "no adapter in $adapter (dvc pull?)" >&2; exit 1; }

if [ ! -f "$gguf" ] || [ "$adapter/adapter_model.safetensors" -nt "$gguf" ]; then
  # the converter only needs the base model's config + tokenizer files, not its weights
  base="$(HF_HUB_OFFLINE=0 "$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download(
    'Qwen/Qwen2.5-Coder-14B-Instruct', allow_patterns=['config.json', 'generation_config.json', 'tokenizer*']))")"
  mkdir -p "$(dirname "$gguf")"
  PYTHONPATH="$ROOT/.tools/llama.cpp/gguf-py" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" \
    "$PY" "$ROOT/.tools/llama.cpp/convert_lora_to_gguf.py" --base "$base" --outtype f16 --outfile "$gguf" "$adapter"
fi

dir="$ROOT/data/models/ollama/${tag//:/_}"
mkdir -p "$dir"
printf 'FROM %s\nADAPTER %s\n' "$from" "$gguf" > "$dir/Modelfile"   # no TEMPLATE: keep the chat template
ollama create "$tag" -f "$dir/Modelfile"
echo "ollama model $tag ready — serve it with rebirth serve publish-xlang, or REBIRTH_XLANG_MODEL=$tag"
