#!/usr/bin/env bash
# Run a per-language LoRA train with hang-proof logging.
#
#   bash scripts/train_logged.sh <lang> <max_length> [extra train_lora.py args…]
#   bash scripts/train_logged.sh go 256
#   bash scripts/train_logged.sh go 512 --recipe v3_expanded --data data/datasets/expanded/go_migration.jsonl
#
# Every run gets its own kept directory: logs/train/<lang>_seq<N>_<UTC ts>/
#   run.json        command, git sha, kernel, start time
#   train.log       full stdout/stderr (python -u)
#   steps.jsonl     per-step begin/end + VRAM, fsync'd  → last "step_begin" = where it hung
#   telemetry.jsonl GPU VRAM / busy% / temps / power every 2 s, fsync'd → when it hung
#   kernel.log      amdgpu kernel messages during the run
#   exit.json       exit code + end time (missing ⇒ the machine died mid-run)
# Then (exit 0) the run is recorded in MLflow: `rebirth track train <run dir>` (Phase 13).
# After a hard freeze + reboot: bash scripts/gpu_hang_postmortem.sh <run dir>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
lang="${1:?usage: train_logged.sh <lang> <max_length> [extra args]}"
maxlen="${2:?usage: train_logged.sh <lang> <max_length> [extra args]}"
shift 2

ts="$(date -u +%Y%m%dT%H%M%SZ)"
run="logs/train/${lang}_seq${maxlen}_${ts}"
mkdir -p "$run"
start_iso="$(date --iso-8601=seconds)"

# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source scripts/rocm_env.sh
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

if command -v ollama >/dev/null && ollama ps 2>/dev/null | tail -n +2 | grep -q .; then
  echo "Ollama has a model resident — unloading to free VRAM"
  ollama ps | tail -n +2 | awk '{print $1}' | xargs -r -n1 ollama stop
fi

# Legacy defaults — skipped when a YAML recipe is given (explicit flags would override it).
defaults=(--batch-size 1 --grad-accum 8 --epochs 1 --max-steps -1 --save-steps 50
          --gradient-checkpointing --attn-eager)
for a in "$@"; do [ "$a" = "--recipe" ] && defaults=(); done
cmd=(python -u training/train_lora.py --lang "$lang" --max-length "$maxlen"
     "${defaults[@]}" --step-log "$run/steps.jsonl" "$@")

python - "$run/run.json" "$start_iso" "${cmd[@]}" <<'PY'
import json, platform, subprocess, sys
out, start, *cmd = sys.argv[1:]
sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
json.dump({"start": start, "cmd": cmd, "git": sha, "git_dirty": dirty,
           "kernel": platform.release()}, open(out, "w"), indent=2)
PY

python -u training/gpu_telemetry.py --out "$run/telemetry.jsonl" --interval 2 &
tele_pid=$!
sync

echo "run dir: $run"
"${cmd[@]}" 2>&1 | tee "$run/train.log"
code=${PIPESTATUS[0]}

kill "$tele_pid" 2>/dev/null; wait "$tele_pid" 2>/dev/null
journalctl -k --since "$start_iso" --no-pager 2>/dev/null \
  | grep -iE 'amdgpu|drm|gpu|hang|reset|ring|timeout|page fault' > "$run/kernel.log" || true
printf '{"exit_code": %d, "end": "%s"}\n' "$code" "$(date --iso-8601=seconds)" > "$run/exit.json"
sync

# Phase 13: record the finished run in MLflow (reads the files above; never fails the run).
if [ "$code" -eq 0 ] && [ "${REBIRTH_TRACKING:-1}" != 0 ] && command -v rebirth >/dev/null; then
  rebirth track train "$run" 2>&1 | tail -3 | tee -a "$run/train.log" || true
fi
echo "exit=$code  logs kept in $run"
exit "$code"
