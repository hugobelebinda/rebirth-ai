# Ollama on AMD RX 7800 XT (ROCm / Vulkan)

Last verified: 2026-10-31 on this machine.

## Hardware

| Device | Role | VRAM |
|--------|------|------|
| Radeon RX 7800 XT (Navi32 / `1002:747e`) | Discrete GPU — **use this** | 16 GB |
| Raphael iGPU (`1002:164e`) | Display / weak compute | 512 MB |

CPU: Ryzen 7 7700X · System RAM: 30 GB · Ollama: 0.30.0

## Status (verified)

- Ollama already ships **ROCm v7** (`/usr/local/lib/ollama/rocm_v7_2`) and **Vulkan** (`/usr/local/lib/ollama/vulkan`).
- CUDA library folders exist but are **irrelevant** (no NVIDIA GPU).
- Systemd drop-in installed: `/etc/systemd/system/ollama.service.d/amd-gpu.conf` (`OLLAMA_LLM_LIBRARY=rocm_v7_2`).
- Server log after restart:

```text
dropping ROCm device — no rocblas support for gfx target device=ROCm1 gfx_target=gfx1036  # Raphael iGPU (expected)
inference compute library=ROCm compute=gfx1101 name=ROCm0 description="AMD Radeon RX 7800 XT"
  type=discrete total="16.0 GiB" available="15.8 GiB"
```

- Live model check (Q6_K default):

```text
qwen2.5-coder:14b-instruct-q6_K    12 GB    100% GPU    quantization Q6_K
```

## Chosen model

**Default: `qwen2.5-coder:14b-instruct-q6_K`**

| Candidate | Disk / weights | Fits 16 GB? | Role |
|-----------|----------------|-------------|------|
| `qwen2.5-coder:14b-instruct-q6_K` | ~12 GB | **Yes, fully** | **Primary for Rebirth AI** |
| `qwen2.5-coder:14b` (Q4_K_M) | ~9 GB | Yes | Faster fallback / more context headroom |
| `qwen2.5-coder:14b-instruct-q5_K_M` | ~11 GB | Yes | Mid quant option |
| `qwen2.5-coder:32b` | ~20 GB | No (spill to RAM) | Optional hard cases only |
| `qwen3-coder:30b` | ~19 GB | No (spill) | Optional experiment |
| `deepseek-r1:32b` | ~20 GB | No | Reasoning, not migration loops |

Why this tag: same 14B coder as before, higher-fidelity Q6_K quant (~12GB), still leaves ~4GB for KV cache on the 7800 XT.

## One-time hardening (needs your password)

```bash
cd "/home/hugobele/Documents/Personal Project/Rebirth AI"
bash scripts/enable_amd_gpu_ollama.sh
# then open a new terminal (or: newgrp render)
```

This writes `/etc/systemd/system/ollama.service.d/amd-gpu.conf` to prefer `OLLAMA_LLM_LIBRARY=rocm_v7_2`, keeps Vulkan enabled, restarts the service, and adds your user to `render`/`video`.

### If ROCm discovery fails

1. Confirm ROCm v7 **kernel** driver (Ollama docs): install/upgrade via AMD `amdgpu-install`.
2. Or force Vulkan only:

```ini
Environment="OLLAMA_LLM_LIBRARY="
Environment="OLLAMA_VULKAN=1"
Environment="GGML_VK_VISIBLE_DEVICES=0"
```

(Adjust the Vulkan index if the iGPU is selected — check debug logs.)

3. Optional gfx override (usually **not** needed for 7800 XT):

```ini
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
```

### Debug

```bash
# After sudo restart with debug (temporary):
sudo systemctl edit ollama   # add Environment="OLLAMA_DEBUG=1"
sudo systemctl restart ollama
journalctl -u ollama -n 100 --no-pager | rg -i 'gpu|rocm|vulkan|hip|amd'
ollama ps
```

## Project env

See `.env.example`:

```bash
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5-coder:14b-instruct-q6_K
```

