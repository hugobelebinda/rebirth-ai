#!/usr/bin/env bash
# Run a command in the training environment: .venv + (local AMD) ROCm env, or plain CUDA on AWS.
# Used by dvc.yaml stage commands so `dvc repro` behaves the same as the queue scripts.
#   bash scripts/with_env.sh python training/eval_behavioral.py …
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
[ -f .venv/bin/activate ] && source .venv/bin/activate
if [ -f scripts/rocm_env.sh ] && command -v rocminfo >/dev/null 2>&1 || [ -d /opt/rocm ]; then
  # shellcheck disable=SC1091
  source scripts/rocm_env.sh
  export HIP_VISIBLE_DEVICES=0 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export REBIRTH_EMBEDDER=hash PYTORCH_ALLOC_CONF=expandable_segments:True
exec "$@"
