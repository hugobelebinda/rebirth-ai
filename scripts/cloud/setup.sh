#!/usr/bin/env bash
# One-time instance setup, executed ON the instance by `bash scripts/cloud/bootstrap.sh`.
# Idempotent. Records versions in logs/cloud/setup.json.
set -euo pipefail
cd ~/rebirth
mkdir -p logs/cloud
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee logs/cloud/gpu.txt
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d .venv ] || uv venv .venv --python 3.12
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -q -r requirements-cuda.txt
uv pip install -q --no-deps -e .
python - <<'PY' | tee logs/cloud/torch.txt
import torch, transformers, peft
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), transformers.__version__, peft.__version__)
PY
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
docker pull -q "$VLLM_IMAGE" >/dev/null
DIGEST=$(docker image inspect --format '{{index .RepoDigests 0}}' "$VLLM_IMAGE")
for m in ${SETUP_MODELS:-Qwen/Qwen2.5-Coder-7B-Instruct Qwen/Qwen2.5-Coder-14B-Instruct}; do
  hf download "$m" --quiet >/dev/null
done
python - "$DIGEST" <<'PY'
import json, subprocess, sys, time
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
json.dump({"t": time.strftime("%FT%T%z"), "vllm_image": sys.argv[1],
           "gpu": open("logs/cloud/gpu.txt").read().strip(), "torch": open("logs/cloud/torch.txt").read().strip(),
           "pip_freeze_lines": len(freeze.splitlines())}, open("logs/cloud/setup.json", "w"), indent=2)
PY
echo "setup ok — $(cat logs/cloud/torch.txt)"
