#!/usr/bin/env bash
# One-time host fix so PyTorch+ROCm can use the discrete GPU for training.
# Ollama already works; /dev/kfd is still restricted to the `render` group.
set -euo pipefail

USER_NAME="${SUDO_USER:-$USER}"

echo "Granting ${USER_NAME} access to /dev/kfd + render/video groups…"
setfacl -m "u:${USER_NAME}:rw" /dev/kfd
# Persist across reboot via udev is nicer long-term; group membership is standard:
usermod -aG render,video "${USER_NAME}"

echo "Done."
echo "  ACL now: $(getfacl -c /dev/kfd 2>/dev/null | tr '\n' ' ')"
echo "  Groups : $(id -nG "${USER_NAME}")"
echo
echo "Open a NEW terminal (or re-login) so group membership applies, then:"
echo "  source .venv/bin/activate"
echo "  python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\""
echo "  python training/train_lora.py --epochs 1 --max-length 768 --batch-size 4"
