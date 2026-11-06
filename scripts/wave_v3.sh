#!/usr/bin/env bash
# v3 wave: retrain per-language LoRAs on the expanded corpus (data/datasets/expanded/) and
# compare v3 vs v2 on the frozen 100-row eval sets (data/datasets/evalsets/<lang>.jsonl).
# v3 adapters go to data/models/exp/<lang>_v3 — the canonical data/models/lora_<lang>_7b
# adapters are NOT replaced (promotion is a separate, deliberate step).
#
# Restartable (stopped/started by hand: GPU hours ~06:00–23:00, see scripts/gpu_schedule.sh):
#   - finished trainings (rebirth_meta.json present) and evals (report present) are skipped
#   - trainings resume from the newest checkpoint (train_lora.py --resume; every 50 steps)
#   - optional EVAL_MIN_LEFT=45: do not start an eval (~30–45 min, not resumable) within 45 min of the
#     usual stop time — the job exits cleanly instead (default 0 = off)
#
# Step 0 (Go only): all-linear r16, 1 epoch — separates "bigger adapter" from "2nd epoch"
# in the Go recipe experiment (D = all r16 2 epochs scored below A/C).
# Env: WAVE_LANGS (default "swift csharp java python js_ts"), RECIPE (recipe name, default v3_expanded),
#      WAIT_UNIT (optional unit to wait for), EVAL_MIN_LEFT (default 0 = off).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
X="logs/exp/wave_v3"
mkdir -p "$X" data/reports/exp data/models/exp
S="$X/status.jsonl"
BASE=Qwen/Qwen2.5-Coder-7B-Instruct
LANGS=(${WAVE_LANGS:-swift csharp java python js_ts})
RECIPE="${RECIPE:-v3_expanded}"   # configs/recipes/<name>.yaml (Phase 13)
declare -A LIMIT=([python]=12000 [js_ts]=12000)   # ~3 h each at ~9 s / 8 rows
note() { echo "{\"t\":\"$(date -Is)\",$1}" | tee -a "$S"; }
# shellcheck disable=SC1091
source scripts/gpu_schedule.sh   # minutes_until_stop

if [ -n "${WAIT_UNIT:-}" ]; then
  while systemctl --user is-active --quiet "$WAIT_UNIT"; do sleep 30; done
fi
note "\"event\":\"job_start\",\"minutes_until_stop\":$(minutes_until_stop)"

# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source scripts/rocm_env.sh
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 REBIRTH_EMBEDDER=hash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0 PYTORCH_ALLOC_CONF=expandable_segments:True
ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | xargs -r -n1 ollama stop

ev() {  # lang name adapter_dir → data/reports/exp/<lang>_<name>.json
  local lang=$1 name=$2 adapter=$3
  local out="data/reports/exp/${lang}_${name}.json"
  [ -f "$out" ] && return 0
  local left; left=$(minutes_until_stop)
  if [ "$left" -lt "${EVAL_MIN_LEFT:-0}" ]; then
    note "\"event\":\"pause_for_night\",\"before\":\"eval ${lang}_${name}\",\"minutes_until_stop\":$left"
    exit 0
  fi
  note "\"event\":\"eval_start\",\"lang\":\"$lang\",\"name\":\"$name\""
  REBIRTH_PEFT_BASE=$BASE REBIRTH_PEFT_ADAPTER="$adapter" \
    python -u training/eval_behavioral.py --backend peft --suite synthetic --lang "$lang" \
    --eval-file "data/datasets/evalsets/$lang.jsonl" --n-synth 100 --out "$out" > "$X/eval_${lang}_$name.log" 2>&1
  local code=$?
  note "\"event\":\"eval_end\",\"lang\":\"$lang\",\"name\":\"$name\",\"exit\":$code,\"summary\":$(python3 -c "
import json;d=json.load(open('$out'))['suites']['synthetic']
print(json.dumps({k:round(d[k],3) for k in ('n','soft_exact','parses','mean_symbol_jaccard','mean_token_overlap')}))" 2>/dev/null || echo null)"
}

train_exp() {  # lang name extra-args… → data/models/exp/<lang>_<name>
  local lang=$1 name=$2; shift 2
  local out="data/models/exp/${lang}_$name"
  [ -f "$out/rebirth_meta.json" ] && return 0
  local limit=()
  [ -n "${LIMIT[$lang]:-}" ] && limit=(--limit "${LIMIT[$lang]}")
  note "\"event\":\"train_start\",\"lang\":\"$lang\",\"name\":\"$name\",\"resume_from\":\"$(ls -d "$out"/runs/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)\""
  bash scripts/train_logged.sh "$lang" 512 --data "data/datasets/expanded/${lang}_migration.jsonl" \
    --out "$out" --resume "${limit[@]}" "$@" >> "$X/train_${lang}_$name.console" 2>&1
  local code=$?
  local loss
  loss=$(python3 -c "import json;print(round(json.load(open('$out/rebirth_meta.json'))['train_loss'],4))" 2>/dev/null || echo null)
  note "\"event\":\"train_end\",\"lang\":\"$lang\",\"name\":\"$name\",\"exit\":$code,\"train_loss\":$loss,\"run_dir\":\"$(ls -dt logs/train/${lang}_seq512_* | head -1)\""
  return $code
}

# 0. Go: bigger adapter, 1 epoch (vs C attn r8 1ep = 15/100, D all r16 2ep = 12/100).
if [ "${SKIP_GO_D1:-0}" != 1 ]; then
  train_exp go D1_all_r16_exp_e1 --recipe v3_all_r16 \
    && ev go D1_all_r16_exp_e1 data/models/exp/go_D1_all_r16_exp_e1/adapter
fi

for lang in "${LANGS[@]}"; do
  ev "$lang" v2 "data/models/lora_${lang}_7b/adapter"
  train_exp "$lang" v3 --recipe "$RECIPE" && ev "$lang" v3 "data/models/exp/${lang}_v3/adapter"
done
note "\"event\":\"done\""
