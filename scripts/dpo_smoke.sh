#!/usr/bin/env bash
# Phase 15 S4 rehearsal: DPO on the local 7B (Rust v2 adapter), so the cloud session runs known-good code.
# Called by scripts/p15_prep_gpu.sh (STAGES="... dpo"), env already set up. Restartable per step.
#   1. 120 training prompts (no held-out rows, no eval-set files)   build_dpo_pairs.py sample
#   2. the v2 adapter's own outputs on them (tracking off)            eval_behavioral.py
#   3. preference pairs (gold ≻ own wrong output)                     build_dpo_pairs.py pairs
#   4. DPO train, 4-bit, seq 512, 1 epoch                             train_dpo.py
#   5. the DPO adapter on the frozen rust_v2 set, vs v2               eval_behavioral.py
set -uo pipefail
X="logs/exp/p15_prep"
S="$X/status.jsonl"
L=rust
BASE=Qwen/Qwen2.5-Coder-7B-Instruct
SFT="data/models/lora_${L}_7b/adapter"
P="data/datasets/dpo/${L}_prompts.jsonl"
GEN="data/reports/exp/${L}_dpo_gen.json"
PAIRS="data/datasets/dpo/${L}_pairs.jsonl"
OUT="data/models/exp/${L}_dpo_smoke"
EV="data/reports/exp/${L}@ev2_dpo_smoke.json"
note() { echo "{\"t\":\"$(date -Is)\",$1}" | tee -a "$S"; }
step() { note "\"event\":\"dpo_$1\",\"lang\":\"$L\"${2:+,$2}"; }

[ -f "$P" ] || { step sample; python training/build_dpo_pairs.py sample --lang $L --n 120 > "$X/dpo_sample.log" 2>&1 || exit 1; }
if [ ! -f "$GEN" ]; then
  step gen_start
  REBIRTH_TRACKING=0 REBIRTH_PEFT_BASE=$BASE REBIRTH_PEFT_ADAPTER=$SFT \
    python -u training/eval_behavioral.py --backend peft --suite synthetic --lang $L \
    --eval-file "$P" --n-synth 120 --out "$GEN" > "$X/dpo_gen.log" 2>&1 || { step gen_failed; exit 1; }
fi
[ -f "$PAIRS" ] || { python training/build_dpo_pairs.py pairs --lang $L --report "$GEN" > "$X/dpo_pairs.log" 2>&1 || exit 1;
                     step pairs "\"stats\":$(cat "${PAIRS%.jsonl}.stats.json" | tr -d '\n')"; }
if [ ! -f "$OUT/adapter/adapter_model.safetensors" ]; then
  step train_start
  python -u training/train_dpo.py --lang $L --pairs "$PAIRS" --sft-adapter "$SFT" --out "$OUT" \
    --lr 2e-5 --beta 0.1 --epochs 1 > "$X/dpo_train.log" 2>&1 || { step train_failed; exit 1; }
  step train_end "\"meta\":$(python3 -c "import json;m=json.load(open('$OUT/rebirth_meta.json'));print(json.dumps({k:m[k] for k in ('global_steps','train_loss','first_log','last_log','train_seconds')}))")"
fi
if [ ! -f "$EV" ]; then
  step eval_start
  REBIRTH_PEFT_BASE=$BASE REBIRTH_PEFT_ADAPTER="$OUT/adapter" \
    python -u training/eval_behavioral.py --backend peft --suite synthetic --lang $L \
    --eval-file "data/datasets/evalsets/${L}_v2.jsonl" --n-synth 100 --out "$EV" > "$X/dpo_eval.log" 2>&1
  step eval_end "\"exit\":$?"
fi
