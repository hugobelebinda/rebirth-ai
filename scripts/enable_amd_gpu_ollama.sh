#!/usr/bin/env bash
# One-time host setup: prefer AMD ROCm (fallback Vulkan), pin discrete GPU,
# grant render access, restart Ollama. Requires sudo.
set -euo pipefail

echo "==> Rebirth AI — enable AMD GPU for Ollama"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Re-running with sudo..."
  exec sudo -E bash "$0" "$@"
fi

REAL_USER="${SUDO_USER:-hugobele}"
SERVICE_DROPIN_DIR="/etc/systemd/system/ollama.service.d"
DROPIN="${SERVICE_DROPIN_DIR}/amd-gpu.conf"

mkdir -p "${SERVICE_DROPIN_DIR}"

# Prefer ROCm for Radeon RX 7800 XT (officially supported). Keep Vulkan available
# as secondary discovery. Pin away from the Raphael iGPU when possible.
#
# GGML_VK_VISIBLE_DEVICES: discrete 7800 XT is usually Vulkan0 or Vulkan1.
# Start with unset Vulkan pin under ROCm; if ROCm fails, set OLLAMA_VULKAN=1
# and GGML_VK_VISIBLE_DEVICES to the dGPU index (see `ollama` debug logs).
cat > "${DROPIN}" <<'EOF'
[Service]
# Prefer AMD compute stacks; do not use NVIDIA CUDA on this host.
Environment="OLLAMA_LLM_LIBRARY=rocm_v7_2"
Environment="OLLAMA_VULKAN=1"
# RX 7800 XT is gfx1101 — officially listed; override only if discovery fails:
# Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
# Optional: pin ROCm device UUID from `rocminfo` (preferred over numeric index)
# Environment="ROCR_VISIBLE_DEVICES=<uuid>"
# If Vulkan is used and iGPU is selected first, pin discrete index, e.g.:
# Environment="GGML_VK_VISIBLE_DEVICES=0"
Environment="OLLAMA_KEEP_ALIVE=10m"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
EOF

echo "Wrote ${DROPIN}"

usermod -aG render,video "${REAL_USER}" || true
usermod -aG render,video ollama || true
echo "Ensured ${REAL_USER} and ollama are in render+video groups"

# Better VRAM reporting for Vulkan scheduling (harmless under ROCm)
if command -v setcap >/dev/null 2>&1; then
  setcap cap_perfmon+ep /usr/local/bin/ollama || true
fi

# Ensure ROCm addon libs are present (idempotent)
if [[ ! -d /usr/local/lib/ollama/rocm_v7_2 ]]; then
  echo "Downloading Ollama ROCm package into /usr ..."
  curl -fsSL https://ollama.com/download/ollama-linux-amd64-rocm.tar.zst | tar x -C /usr
fi

systemctl daemon-reload
systemctl restart ollama
sleep 2
systemctl --no-pager --full status ollama | head -25 || true

echo
echo "Verify GPU (as your user, new shell after group change):"
echo "  newgrp render"
echo "  ollama run qwen2.5-coder:14b-instruct-q6_K 'Say GPU_OK'"
echo "  ollama ps   # expect PROCESSOR 100% GPU"
echo
echo "Default model for Rebirth AI: qwen2.5-coder:14b-instruct-q6_K (~12GB, fits 16GB VRAM)."
echo "Do NOT use qwen2.5-coder:32b / qwen3-coder:30b as default — ~19–20GB > 16GB VRAM."
