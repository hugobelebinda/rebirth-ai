#!/usr/bin/env bash
# Phase 15 prep on the local GPU (2026-11-07). Restartable: finished reports are skipped.
#   b) score the v2 7B adapter and the base 7B on the bigger frozen eval sets
#      data/datasets/evalsets/{rust,cpp}_v2.jsonl → data/reports/exp/<lang>@ev2_{v2,B_base_chat}.json
#   c) DPO smoke test (scripts/dpo_smoke.sh) if STAGES includes "dpo"
# Status → logs/exp/p15_prep/status.jsonl
#   systemd-run --user --unit=rebirth-gpu-job --working-directory="$PWD" bash scripts/p15_prep_gpu.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
X="logs/exp/p15_prep"
mkdir -p "$X" data/reports/exp
S="$X/status.jsonl"
BASE=Qwen/Qwen2.5-Coder-7B-Instruct
STAGES="${STAGES:-ev2}"
note() { echo "{\"t\":\"$(date -Is)\",$1}" | tee -a "$S"; }
summary() {
  python3 -c "
import json;d=json.load(open('$1'))['suites']['synthetic']
print(json.dumps({k:round(d[k],3) for k in ('n','soft_exact','parses','mean_symbol_jaccard','mean_token_overlap')}))" 2>/dev/null || echo null
}
# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source scripts/rocm_env.sh
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 REBIRTH_EMBEDDER=hash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0 PYTORCH_ALLOC_CONF=expandable_segments:True
ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | xargs -r -n1 ollama stop   # free VRAM
note "\"event\":\"job_start\",\"stages\":\"$STAGES\""

ev2() {  # lang name → data/reports/exp/<lang>@ev2_<name>.json ; name = v2 | B_base_chat
  local lang=$1 name=$2
  local out="data/reports/exp/${lang}@ev2_${name}.json"
  [ -f "$out" ] && return 0
  note "\"event\":\"eval_start\",\"lang\":\"$lang\",\"set\":\"${lang}_v2\",\"name\":\"$name\""
  if [ "$name" = v2 ]; then
    REBIRTH_PEFT_BASE=$BASE REBIRTH_PEFT_ADAPTER="data/models/lora_${lang}_7b/adapter" \
      python -u training/eval_behavioral.py --backend peft --suite synthetic --lang "$lang" \
      --eval-file "data/datasets/evalsets/${lang}_v2.jsonl" --n-synth 100 --out "$out" > "$X/eval_${lang}_ev2_$name.log" 2>&1
  else
    REBIRTH_PEFT_BASE=$BASE REBIRTH_PEFT_ADAPTER=none REBIRTH_PEFT_PROMPT_STYLE=chat \
      python -u training/eval_behavioral.py --backend peft --suite synthetic --lang "$lang" \
      --eval-file "data/datasets/evalsets/${lang}_v2.jsonl" --n-synth 100 --out "$out" > "$X/eval_${lang}_ev2_$name.log" 2>&1
  fi
  local code=$?
  note "\"event\":\"eval_end\",\"lang\":\"$lang\",\"set\":\"${lang}_v2\",\"name\":\"$name\",\"exit\":$code,\"summary\":$(summary "$out")"
}

if [[ " $STAGES " == *" ev2 "* ]]; then
  for lang in rust cpp; do ev2 "$lang" v2; ev2 "$lang" B_base_chat; done
fi
if [[ " $STAGES " == *" dpo "* ]] && [ -f scripts/dpo_smoke.sh ]; then
  bash scripts/dpo_smoke.sh
fi
note "\"event\":\"job_end\""
