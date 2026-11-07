#!/usr/bin/env bash
# ON the instance: serve a base model + LoRA adapters with vLLM (OpenAI API on 127.0.0.1:8000).
#   bash scripts/cloud/vllm_serve.sh Qwen/Qwen2.5-Coder-14B-Instruct go14b=data/models/lora_go_14b/adapter …
set -euo pipefail
cd ~/rebirth
base="${1:?usage: vllm_serve.sh <base model> [name=adapter_dir …]}"; shift
mods=(); for m in "$@"; do mods+=("${m%%=*}=/repo/${m#*=}"); done
docker rm -f rebirth-vllm >/dev/null 2>&1 || true
docker run -d --name rebirth-vllm --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$PWD:/repo:ro" -e HF_HUB_OFFLINE=1 \
  "${VLLM_IMAGE:-vllm/vllm-openai:latest}" --model "$base" --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" --seed 0 \
  ${mods:+--enable-lora --max-lora-rank "${MAX_LORA_RANK:-16}" --max-loras "${MAX_LORAS:-8}" --lora-modules "${mods[@]}"} >/dev/null
for _ in $(seq 1 120); do curl -sf http://127.0.0.1:8000/v1/models >/dev/null && { echo "vLLM ready: $base ${mods[*]:-}"; exit 0; }; sleep 5; done
echo "vLLM did not come up — docker logs rebirth-vllm" >&2; exit 1
