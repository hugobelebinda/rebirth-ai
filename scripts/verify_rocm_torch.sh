#!/usr/bin/env bash
# Verify AMD ROCm + PyTorch HIP (NOT NVIDIA CUDA) before any training.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/rocm_env.sh"
# shellcheck disable=SC1091
source .venv/bin/activate

FAIL=0
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; FAIL=1; }
info() { echo "INFO  $*"; }

echo "=== Rebirth AI ROCm / PyTorch health check ==="
info "ROCM_PATH=${ROCM_PATH:-unset}"

VER="$(python -c 'import torch; print(torch.__version__)')"
HIP="$(python -c 'import torch; print(getattr(torch.version, "hip", None))')"
if [[ "$VER" == *"+rocm"* ]] && [[ "$VER" != *"+cu"* ]]; then
  pass "torch wheel is ROCm: $VER (hip=$HIP)"
else
  fail "torch wheel is not a ROCm build: $VER"
fi

if [[ -e /dev/kfd ]] && [[ -r /dev/kfd ]] && [[ -w /dev/kfd ]]; then
  pass "/dev/kfd accessible"
else
  fail "/dev/kfd not accessible — run: sudo bash scripts/enable_amd_train_gpu.sh"
fi
if id -nG | tr ' ' '\n' | grep -qx render; then
  pass "user in render group"
else
  fail "user not in render group"
fi

ROCMINFO_BIN=""
if command -v rocminfo >/dev/null 2>&1; then
  ROCMINFO_BIN="$(command -v rocminfo)"
elif [[ -x /opt/rocm-6.4.4/bin/rocminfo ]]; then
  ROCMINFO_BIN=/opt/rocm-6.4.4/bin/rocminfo
elif [[ -x /opt/rocm/bin/rocminfo ]]; then
  ROCMINFO_BIN=/opt/rocm/bin/rocminfo
fi

if [[ -n "$ROCMINFO_BIN" ]]; then
  pass "rocminfo present: $ROCMINFO_BIN"
  ROCMINFO_OUT="$("$ROCMINFO_BIN" 2>/dev/null || true)"
  echo "$ROCMINFO_OUT" | grep -E 'Marketing Name|Name:|gfx1101|Device Type' | head -25 || true
  # rocminfo can exit non-zero; don't use pipefail against it
  if echo "$ROCMINFO_OUT" | grep -q 'gfx1101'; then
    pass "rocminfo lists gfx1101"
  else
    fail "rocminfo did not list gfx1101 (RX 7800 XT)"
  fi
else
  fail "rocminfo missing — install system ROCm (scripts/install_system_rocm.sh)"
fi

if [[ -e /opt/amdgpu/share/libdrm/amdgpu.ids ]]; then
  pass "amdgpu.ids at /opt/amdgpu/share/libdrm/amdgpu.ids"
elif [[ -e /usr/share/libdrm/amdgpu.ids ]]; then
  fail "amdgpu.ids only under /usr/share/libdrm — symlink into /opt/amdgpu/share/libdrm"
else
  fail "amdgpu.ids not found"
fi

# Warn if empty visible-device vars are set (hides all GPUs)
for v in CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES; do
  if [[ -n "${!v+x}" ]] && [[ -z "${!v}" ]]; then
    fail "$v is set to empty string (hides all GPUs). unset $v"
  fi
done

python - <<'PY'
import sys
import torch

ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
print(("PASS" if ok else "FAIL") + f"  torch.cuda.is_available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
if not ok:
    sys.exit(2)
archs = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    arch = getattr(p, "gcnArchName", "?")
    archs.append(arch)
    print(f"INFO  device{i}: {torch.cuda.get_device_name(i)} arch={arch} vram={p.total_memory/1e9:.1f}GB")
if not any("gfx1101" in a for a in archs):
    print("FAIL  no gfx1101 (RX 7800 XT) visible to PyTorch")
    sys.exit(3)
print("PASS  gfx1101 present in PyTorch")
x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()
print(f"PASS  HIP matmul ok mean={float(y.mean()):.4f} alloc_mb={torch.cuda.memory_allocated()/1e6:.1f}")
PY
TORCH_RC=$?
if [[ $TORCH_RC -ne 0 ]]; then
  fail "PyTorch HIP device check failed (exit $TORCH_RC)"
fi

python - <<'PY'
try:
    import bitsandbytes
    print("INFO  bitsandbytes installed — CUDA-oriented; do not rely on it for ROCm QLoRA")
except Exception:
    print("INFO  bitsandbytes not required for ROCm bf16 LoRA")
PY

echo
if [[ $FAIL -eq 0 && $TORCH_RC -eq 0 ]]; then
  echo "=== ALL CHECKS PASSED — GPU stack healthy (AMD ROCm / HIP only) ==="
  exit 0
fi
echo "=== NOT READY FOR GPU TRAINING ==="
exit 1
