#!/usr/bin/env bash
# Evaluate a trained per-language LoRA (recipe v2):
#   1. held-out split, adapter (SFT prompt)        → data/reports/eval_heldout_<lang>_peft7b_v2.json
#   2. held-out split, base 7B no adapter (chat)    → data/reports/eval_heldout_<lang>_base7b_chat.json
#   3. gate (adapter vs base)                       → data/reports/gate_<lang>_v2.json
#   4. fixture closed-loop migrate --backend peft   → data/reports/migrate_<lang>_sample_peft7b_v2.log
#
#   bash scripts/eval_lang_lora.sh <lang> [log dir]
# Exit code = gate exit code (0 pass, 1 fail, 2 error).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
lang="${1:?usage: eval_lang_lora.sh <lang> [log dir]}"
logdir="${2:-logs/train}"
mkdir -p "$logdir" data/reports

# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source scripts/rocm_env.sh
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 REBIRTH_EMBEDDER=hash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0
N=24
R=data/reports

# Without its adapter, --backend peft would silently fall back to Ollama — refuse instead.
if [ ! -f "data/models/lora_${lang}_7b/adapter/adapter_config.json" ]; then
  echo "{\"pass\": false, \"error\": \"no adapter at data/models/lora_${lang}_7b/adapter\"}" | tee "$R/gate_${lang}_v2.json"
  exit 2
fi

echo "[eval:$lang] adapter held-out"
python -u training/eval_behavioral.py --backend peft --suite synthetic --lang "$lang" --n-synth $N \
  --out "$R/eval_heldout_${lang}_peft7b_v2.json" > "$logdir/eval_${lang}_adapter.log" 2>&1

echo "[eval:$lang] base 7B chat held-out"
REBIRTH_PEFT_ADAPTER=none REBIRTH_PEFT_PROMPT_STYLE=chat \
python -u training/eval_behavioral.py --backend peft --suite synthetic --lang "$lang" --n-synth $N \
  --out "$R/eval_heldout_${lang}_base7b_chat.json" > "$logdir/eval_${lang}_base_chat.log" 2>&1

python training/lora_gate.py --adapter "$R/eval_heldout_${lang}_peft7b_v2.json" \
  --base "$R/eval_heldout_${lang}_base7b_chat.json" | tee "$R/gate_${lang}_v2.json"
gate=${PIPESTATUS[0]}

case "$lang" in
  go)     fx=(fixtures/go_sample --file src/mathops.go --test ./src --seed-draft fixtures/go_sample/seeds/mathops.broken.go) ;;
  rust)   fx=(fixtures/rust_sample --file src/lib.rs --seed-draft fixtures/rust_sample/seeds/lib.broken.rs) ;;
  cpp)    fx=(fixtures/cpp_sample --file src/math_ops.cpp --seed-draft fixtures/cpp_sample/seeds/math_ops.broken.cpp) ;;
  swift)  fx=(fixtures/swift_sample --file Sources/MathOps/MathOps.swift --seed-draft fixtures/swift_sample/seeds/MathOps.broken.swift) ;;
  csharp) fx=(fixtures/csharp_sample --file MathOps/Calculator.cs --seed-draft fixtures/csharp_sample/seeds/Calculator.broken.cs) ;;
  java)   fx=(fixtures/java_sample --file src/main/java/rebirth/math/MathOps.java --seed-draft fixtures/java_sample/seeds/MathOps.broken.java) ;;
  python) fx=(fixtures/python_sample --file src/math_ops.py --test tests/test_math_ops.py --seed-draft fixtures/python_sample/seeds/math_ops.broken.py) ;;
  js_ts)  fx=(fixtures/js_ts_sample --file src/math.js --test test/math.test.js --seed-draft fixtures/migration_seeds/math.broken.js) ;;
  *)      fx=() ;;
esac
if [ ${#fx[@]} -gt 0 ]; then
  echo "[eval:$lang] fixture migrate (peft)"
  rebirth migrate "${fx[@]}" --lang "$lang" --backend peft --max-iterations 5 --no-write \
    > "$R/migrate_${lang}_sample_peft7b_v2.log" 2>&1
  grep -E "status=|iterations=" "$R/migrate_${lang}_sample_peft7b_v2.log" | head -2
fi
exit "$gate"
