#!/usr/bin/env bash
# Phase 12 reference scores on the frozen eval sets (data/datasets/evalsets/<lang>.jsonl).
# Order: references (2) first, Ollama test (1) last.
#   1. GGUF→Ollama adapter test: the Go v2 adapter served by Ollama on Q8_0 and Q4_K_M bases
#      (models rebirth-go-7b-q8 / rebirth-go-7b-q4, see scripts/ollama_adapter.sh), scored with
#      eval_vllm.py against Ollama's OpenAI API — compare with go_A_v2.json (PEFT, 14/99).
#      Skipped if the Ollama models do not exist.
#   2. v2 adapter scores for the languages without one, and base 7B (no adapter, chat) for all
#      languages without one → data/reports/exp/<lang>_{v2,B_base_chat}.json
# Restartable: every report that exists is skipped. Status → logs/exp/refs_p12/status.jsonl
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
X="logs/exp/refs_p12"
mkdir -p "$X" data/reports/exp
S="$X/status.jsonl"
BASE=Qwen/Qwen2.5-Coder-7B-Instruct
note() { echo "{\"t\":\"$(date -Is)\",$1}" | tee -a "$S"; }
summary() {
  python3 -c "
import json;d=json.load(open('$1'))['suites']['synthetic']
print(json.dumps({k:round(d[k],3) for k in ('n','soft_exact','parses','mean_symbol_jaccard','mean_token_overlap')}))" 2>/dev/null || echo null
}
note "\"event\":\"job_start\""

# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source scripts/rocm_env.sh
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 REBIRTH_EMBEDDER=hash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0 PYTORCH_ALLOC_CONF=expandable_segments:True
ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | xargs -r -n1 ollama stop   # free VRAM

# ---- 2. v2 + base-7B references --------------------------------------------------------------
ev() {  # lang name → data/reports/exp/<lang>_<name>.json ; name = v2 | B_base_chat
  local lang=$1 name=$2
  local out="data/reports/exp/${lang}_${name}.json"
  [ -f "$out" ] && return 0
  note "\"event\":\"eval_start\",\"lang\":\"$lang\",\"name\":\"$name\""
  if [ "$name" = v2 ]; then
    REBIRTH_PEFT_BASE=$BASE REBIRTH_PEFT_ADAPTER="data/models/lora_${lang}_7b/adapter" \
      python -u training/eval_behavioral.py --backend peft --suite synthetic --lang "$lang" \
      --eval-file "data/datasets/evalsets/$lang.jsonl" --n-synth 100 --out "$out" > "$X/eval_${lang}_$name.log" 2>&1
  else
    REBIRTH_PEFT_BASE=$BASE REBIRTH_PEFT_ADAPTER=none REBIRTH_PEFT_PROMPT_STYLE=chat \
      python -u training/eval_behavioral.py --backend peft --suite synthetic --lang "$lang" \
      --eval-file "data/datasets/evalsets/$lang.jsonl" --n-synth 100 --out "$out" > "$X/eval_${lang}_$name.log" 2>&1
  fi
  local code=$?
  [ "$code" -eq 0 ] && [ "$name" = v2 ] && bash scripts/dvc.sh commit -f -q "eval_v2@${lang}" >> "$X/dvc.log" 2>&1
  note "\"event\":\"eval_end\",\"lang\":\"$lang\",\"name\":\"$name\",\"exit\":$code,\"summary\":$(summary "$out")"
}

# Pairs per language so each finished language is a complete comparison.
for lang in python csharp; do ev "$lang" B_base_chat; done
for lang in swift java js_ts rust cpp; do ev "$lang" v2; ev "$lang" B_base_chat; done
# ---- 1. Ollama adapter test (Go) -----------------------------------------------------------
for q in q8 q4; do
  m="rebirth-go-7b-$q" out="data/reports/exp/go_O_adapter_$q.json"
  [ -f "$out" ] && continue
  ollama show "$m" > /dev/null 2>&1 || { note "\"event\":\"skip\",\"what\":\"$m\",\"why\":\"model missing\""; continue; }
  note "\"event\":\"eval_start\",\"lang\":\"go\",\"name\":\"O_adapter_$q\""
  python -u training/eval_vllm.py --server http://127.0.0.1:11434/v1 --model "$m" --style sft \
    --lang go --eval-file data/datasets/evalsets/go.jsonl --concurrency 1 --engine ollama \
    --model-label "ollama-adapter-$q" --adapter data/models/lora_go_7b/adapter \
    --out "$out" > "$X/eval_go_O_adapter_$q.log" 2>&1
  code=$?
  ollama stop "$m" > /dev/null 2>&1
  note "\"event\":\"eval_end\",\"lang\":\"go\",\"name\":\"O_adapter_$q\",\"exit\":$code,\"summary\":$(summary "$out")"
done
ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | xargs -r -n1 ollama stop

note "\"event\":\"job_end\""
