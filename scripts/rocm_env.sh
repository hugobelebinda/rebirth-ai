#!/usr/bin/env bash
# Source this before PyTorch ROCm work.
# Matches system ROCm 6.4.4 + venv torch 2.9.1+rocm6.4
#
# IMPORTANT: do NOT export CUDA_VISIBLE_DEVICES= or HIP_VISIBLE_DEVICES=
# (empty value hides all GPUs).

if [[ -d /opt/rocm-6.4.4 ]]; then
  export ROCM_PATH=/opt/rocm-6.4.4
elif [[ -d /opt/rocm ]]; then
  export ROCM_PATH=/opt/rocm
fi

if [[ -n "${ROCM_PATH:-}" ]]; then
  export HIP_PATH="$ROCM_PATH"
  export PATH="$ROCM_PATH/bin:$PATH"
  export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
fi

# Prefer discrete RX 7800 XT (gfx1101 was device 0 after install)
# Only set if you need to pin; leave unset to see all agents.
# export HIP_VISIBLE_DEVICES=0

unset CUDA_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
# Do not set HSA_OVERRIDE_GFX_VERSION on gfx1101 unless forced.
