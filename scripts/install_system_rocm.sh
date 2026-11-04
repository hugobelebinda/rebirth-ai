#!/usr/bin/env bash
# Install system ROCm 6.4.x to match venv torch 2.9.1+rocm6.4 (AMD only — no NVIDIA).
# Zorin 18 / Ubuntu 24.04 (noble). Run in a real terminal (needs sudo password).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${LOG:-/tmp/rebirth_rocm_install.log}"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Is) Rebirth AI system ROCm install ==="
echo "Log: $LOG"
echo
echo ">>> You will be prompted for your sudo password now."
echo

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Do not run as root; run as your user so groups apply correctly."
  exit 1
fi

sudo -v
echo "sudo OK"

# Keep a matching 6.4 stack for current pip torch (+rocm6.4)
AMDGPU_INSTALL_VER="${AMDGPU_INSTALL_VER:-6.4.4}"
DEB_NAME="amdgpu-install_6.4.60404-1_all.deb"
DEB_URL="https://repo.radeon.com/amdgpu-install/${AMDGPU_INSTALL_VER}/ubuntu/noble/${DEB_NAME}"
TMP_DEB="/tmp/${DEB_NAME}"

echo "Downloading ${DEB_URL}"
curl -fL --retry 3 -o "$TMP_DEB" "$DEB_URL"

echo "Installing amdgpu-install package…"
sudo apt-get install -y "$TMP_DEB"

echo "Updating apt…"
sudo apt-get update

echo "Installing ROCm userspace (usecase=rocm). This can take several minutes…"
# On Radeon desktops, amdgpu-install is the supported path.
# --no-dkms: keep existing kernel amdgpu driver (already running); install ROCm user libs.
if sudo amdgpu-install -y --usecase=rocm --no-dkms; then
  echo "amdgpu-install completed"
else
  echo "amdgpu-install failed; trying apt install rocm…"
  sudo apt-get install -y rocm || sudo apt-get install -y rocm-hip-libraries rocm-hip-runtime || true
fi

echo "Ensuring render/video membership…"
sudo usermod -aG render,video "$USER"

echo "Fixing amdgpu.ids path expected by PyTorch…"
sudo mkdir -p /opt/amdgpu/share/libdrm
if [[ -f /usr/share/libdrm/amdgpu.ids ]]; then
  sudo ln -sfn /usr/share/libdrm/amdgpu.ids /opt/amdgpu/share/libdrm/amdgpu.ids
fi

echo
echo "=== Post-install probes ==="
command -v rocminfo && rocminfo | head -40 || echo "rocminfo still missing"
ls -la /opt/rocm 2>/dev/null || ls -d /opt/rocm-* 2>/dev/null || echo "no /opt/rocm symlink yet"
ls -la /opt/amdgpu/share/libdrm/amdgpu.ids || true

echo
echo "=== Running Rebirth verify script ==="
# New groups may need newgrp / re-login; ACL on kfd should already exist
bash "$ROOT/scripts/verify_rocm_torch.sh" || true

echo
echo "=== DONE $(date -Is) ==="
echo "If verify still fails to see HIP GPUs: REBOOT, then run:"
echo "  bash \"$ROOT/scripts/verify_rocm_torch.sh\""
echo
read -r -p "Press Enter to close this window…"
