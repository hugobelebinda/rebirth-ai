#!/usr/bin/env bash
# Package a 7B LoRA adapter as an Ollama model (Phase 12 test → Phase 16 packaging path).
#   bash scripts/ollama_adapter.sh <lang> <q8|q4> [adapter]
# adapter: a PEFT safetensors dir (Ollama converts it) or a .gguf LoRA (llama.cpp convert_lora_to_gguf.py).
# Raw template: the adapters were trained on the plain "### Objective / ### Legacy / ### Modern" prompt,
# with no chat template, so the model must see the prompt exactly as sent.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lang="${1:?lang}" q="${2:?q8|q4}"
adapter="${3:-$ROOT/data/models/lora_${lang}_7b/adapter}"
case "$q" in
  q8) from=qwen2.5-coder:7b-instruct-q8_0 ;;
  q4) from=qwen2.5-coder:7b-instruct ;;     # Q4_K_M, the size an offline tool would ship
  *) echo "q8|q4" >&2; exit 2 ;;
esac
dir="$ROOT/data/models/ollama/rebirth-${lang}-7b-$q"
mkdir -p "$dir"
cat > "$dir/Modelfile" <<EOF
FROM $from
ADAPTER $adapter
TEMPLATE """{{ .Prompt }}"""
PARAMETER temperature 0
PARAMETER num_ctx 2048
PARAMETER stop "
### Objective"
PARAMETER stop "
### Legacy"
PARAMETER stop "
### Modern"
PARAMETER stop "
###"
PARAMETER stop "
\`\`\`"
EOF
ollama create "rebirth-${lang}-7b-$q" -f "$dir/Modelfile"
