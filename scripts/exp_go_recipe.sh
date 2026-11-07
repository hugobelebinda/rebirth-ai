#!/usr/bin/env bash
# Recipe experiment on Go (cheapest language with a 2x data gain), scored on the frozen
# 100-row eval set (data/datasets/evalsets/go.jsonl) instead of the 24 shortest rows.
#
#   A  v2 adapter (attn r8, original 1.4K rows, 1 epoch)        — existing
#   B  base 7B, chat prompt                                     — reference
#   C  attn r8,  expanded 2.95K rows, 1 epoch                   — effect of more data
#   D  all-linear r16 a32, expanded, 2 epochs; eval ~epoch-1 checkpoint and final
#                                                               — capacity + epochs
#   O  Ollama 14B                                               — reference
#
# Run detached:  systemd-run --user --unit=rebirth-exp-go --working-directory="$PWD" bash scripts/exp_go_recipe.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
X="logs/exp/go_recipe_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$X" data/reports/exp data/models/exp
S="$X/status.jsonl"
EVAL=data/datasets/evalsets/go.jsonl
DATA=data/datasets/expanded/go_migration.jsonl
BASE=Qwen/Qwen2.5-Coder-7B-Instruct
note() { echo "{\"t\":\"$(date -Is)\",$1}" | tee -a "$S"; }

# Don't fight the cap-256 re-eval for the GPU.
while systemctl --user is-active --quiet rebirth-reeval-cap256; do sleep 30; done

# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source scripts/rocm_env.sh
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 REBIRTH_EMBEDDER=hash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0 PYTORCH_ALLOC_CONF=expandable_segments:True
ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | xargs -r -n1 ollama stop

ev() {  # name adapter_dir|none  → data/reports/exp/go_<name>.json
  local name=$1 adapter=$2
  local out="data/reports/exp/go_${name}.json"
  note "\"event\":\"eval_start\",\"name\":\"$name\""
  if [ "$adapter" = none ]; then
    REBIRTH_PEFT_ADAPTER=none REBIRTH_PEFT_PROMPT_STYLE=chat \
      python -u training/eval_behavioral.py --backend peft --suite synthetic --lang go \
      --eval-file "$EVAL" --n-synth 100 --out "$out" > "$X/eval_$name.log" 2>&1
  else
    REBIRTH_PEFT_BASE=$BASE REBIRTH_PEFT_ADAPTER="$adapter" \
      python -u training/eval_behavioral.py --backend peft --suite synthetic --lang go \
      --eval-file "$EVAL" --n-synth 100 --out "$out" > "$X/eval_$name.log" 2>&1
  fi
  note "\"event\":\"eval_end\",\"name\":\"$name\",\"exit\":$?,\"summary\":$(python3 -c "
import json;d=json.load(open('$out'))['suites']['synthetic']
print(json.dumps({k:round(d[k],3) for k in ('n','soft_exact','parses','mean_symbol_jaccard','mean_token_overlap')}))" 2>/dev/null || echo null)"
}

train_exp() {  # name extra-args…  → data/models/exp/go_<name>
  local name=$1; shift
  note "\"event\":\"train_start\",\"name\":\"$name\""
  bash scripts/train_logged.sh go 512 --quant 4bit --data "$DATA" --out "data/models/exp/go_$name" "$@" \
    > "$X/train_$name.console" 2>&1
  local code=$?
  note "\"event\":\"train_end\",\"name\":\"$name\",\"exit\":$code,\"run_dir\":\"$(ls -dt logs/train/go_seq512_* | head -1)\""
  return $code
}

# 0. Memory probe for the bigger LoRA (worst case: full 512-token sequences).
#    SKIP_PROBE=1 reuses a passed probe (2026-11-05: all-linear r16 @512 peak 9.2 GB).
probe=0
if [ "${SKIP_PROBE:-0}" != 1 ]; then
note "\"event\":\"probe_start\""
python -u training/qlora_probe.py --lengths 512 --steps 3 --lora-r 16 --lora-targets all \
  --log "$X/probe_all_r16.jsonl" > "$X/probe_all_r16.log" 2>&1
probe=$?
note "\"event\":\"probe_end\",\"exit\":$probe,\"peak\":$(python3 -c "
import json;r=[json.loads(l) for l in open('$X/probe_all_r16.jsonl')]
print(json.dumps([x for x in r if x.get('event')=='length_ok']))" 2>/dev/null || echo null)"
fi

ev B_base_chat none
ev A_v2 data/models/lora_go_7b/adapter

train_exp C_attn_r8_exp_e1 --lora-r 8 --lora-alpha 16 --lora-targets attn --epochs 1 \
  && ev C_attn_r8_exp_e1 data/models/exp/go_C_attn_r8_exp_e1/adapter

if [ "$probe" -eq 0 ]; then D=(--lora-r 16 --lora-alpha 32 --lora-targets all); Dn=D_all_r16_exp_e2
else D=(--lora-r 16 --lora-alpha 32 --lora-targets attn); Dn=D_attn_r16_exp_e2; fi
if train_exp "$Dn" "${D[@]}" --epochs 2; then
  out="data/models/exp/go_$Dn"
  steps=$(python3 -c "import json;print(json.load(open('$out/rebirth_meta.json'))['global_steps'])")
  half=$(ls -d "$out"/checkpoint-* | sed 's/.*checkpoint-//' | sort -n \
         | awk -v h=$((steps / 2)) '{d=$1-h; if(d<0)d=-d; if(best==""||d<bd){best=$1;bd=d}} END{print best}')
  [ -n "$half" ] && ev "${Dn}_ep1ckpt$half" "$out/checkpoint-$half"
  ev "$Dn" "$out/adapter"
fi

note "\"event\":\"ollama_start\""
REBIRTH_EMBEDDER=hash python -u training/eval_behavioral.py --backend ollama --suite synthetic --lang go \
  --eval-file "$EVAL" --n-synth 100 --out data/reports/exp/go_O_ollama.json > "$X/eval_ollama.log" 2>&1
ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | xargs -r -n1 ollama stop
note "\"event\":\"done\""
