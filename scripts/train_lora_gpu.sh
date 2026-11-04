#!/usr/bin/env bash
# Hardened local AMD GPU LoRA smoke / train.
# WARNING: Display still on the 7800 XT can die if the GPU hangs (no iGPU move).
# Does NOT set HSA_OVERRIDE_GFX_VERSION (that contributed to prior instability).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/rocm_env.sh"
# shellcheck disable=SC1091
source .venv/bin/activate

# Refuse to train unless health check is green
bash "$ROOT/scripts/verify_rocm_torch.sh"

# Prefer new name; keep HIP alias for older wheels
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-$PYTORCH_ALLOC_CONF}"
# Pin discrete GPU after verify (gfx1101 was index 0)
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0

echo "PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF"
echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "count", torch.cuda.device_count())
print("device0", torch.cuda.get_device_name(0))
print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
PY

# Default: 7B LoRA smoke. seq 128 — 512 OOMs bf16 7B on 16GB + display compositor.
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
MAX_STEPS="${MAX_STEPS:-5}"
MAX_LENGTH="${MAX_LENGTH:-128}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LIMIT="${LIMIT:-64}"
OUT="${OUT:-data/models/lora_js_ts_7b_smoke}"

exec python training/train_lora.py \
  --base-model "$BASE_MODEL" \
  --out "$OUT" \
  --batch-size 1 \
  --grad-accum "$GRAD_ACCUM" \
  --max-length "$MAX_LENGTH" \
  --max-steps "$MAX_STEPS" \
  --limit "$LIMIT" \
  --gradient-checkpointing \
  --attn-eager \
  "$@"
