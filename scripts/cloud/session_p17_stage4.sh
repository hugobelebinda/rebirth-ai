#!/usr/bin/env bash
# ON the instance, called by session_p17.sh after stage 3 (user's go 2026-11-08: "the 4 steps"):
#   4   shared cross-language adapter on the 14B: stage-3 pairs balanced over the directions + a
#       few same-language rows (training/train_xlang.py), seq 2048, 2 epochs
#   4e  program-suite eval on vLLM (same 20 problems × 21 directions as eval_translate.py):
#       base 14B (product prompt + 2 retries) vs the adapter (product prompt + retries; SFT prompt)
#   4t  the two newer models on the same suite, zero-shot, same prompt + retries as the base 14B
#       (user's pick 2026-11-09, after the stage-3 teacher pilot could not load them): is the
#       14B + adapter or a newer model the better migration engine?
#   D   DPO rollout to the other 6 languages, only if S4 DPO beat the S3 adapters
#       (mean gain ≥ +2 soft-exact/100 over go + rust_v2, neither < −1); stops starting new
#       languages when the session's time budget (P17_MAX_HOURS since p17 start) runs low
# Restartable: every finished step leaves its adapter / report and is skipped.
set -uo pipefail
cd ~/rebirth
export REBIRTH_TRACKING=0 HF_HUB_OFFLINE=1 REBIRTH_EMBEDDER=hash
D=logs/cloud/p17
R=data/reports/cloud/p17
R15=data/reports/cloud/p15
X=data/datasets/xlang/stage3
P=data/datasets/dpo
B14=Qwen/Qwen2.5-Coder-14B-Instruct
XL=data/models/lora_xlang_14b
MIX=data/datasets/xlang/stage4_mix.jsonl
P17_MAX_HOURS="${P17_MAX_HOURS:-26}"
DPO_MORE="${DPO_MORE:-swift csharp cpp java python js_ts}"   # smallest corpus first
DPO_PROMPTS="${DPO_PROMPTS:-2000}"
TEACHERS="${TEACHERS:-Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 Qwen/Qwen3.6-27B-FP8}"
source scripts/cloud/hub.sh   # tag, cached, drop, fetch, free_gb, extra_for

note() { echo "{\"t\":\"$(date -Is)\",$1}" | tee -a "$D/status.jsonl"; }
hours_left() { python -c "import time; print(round($P17_MAX_HOURS - (time.time() - $(cat "$D/start_epoch" 2>/dev/null || date +%s)) / 3600, 2))"; }
enough() { python -c "import sys; sys.exit(0 if $(hours_left) >= $1 else 1)"; }
unserve() { docker rm -f rebirth-vllm >/dev/null 2>&1 || true; }
serve() {  # serve <max model len> <base> [name=adapter …]
  local len=$1; shift
  unserve
  if ! MAX_MODEL_LEN=$len bash scripts/cloud/vllm_serve.sh "$@"; then
    note "\"event\":\"vllm_failed\",\"base\":\"$1\""; docker logs --tail 60 rebirth-vllm 2>&1 | tail -n 60; unserve; return 1
  fi
}
gen() {  # gen <served model> <generate args…>
  local m=$1; shift
  python -u training/gen_xlang_pairs.py generate --backend vllm --model "$m" --max-model-len 8192 \
    --concurrency 48 --pipelines 2 --workers 3 --chunk 25 "$@"
}
have14() { [ -f "data/models/lora_$1_14b/adapter/adapter_model.safetensors" ]; }
sets_for() { case $1 in rust) echo "rust rust_v2" ;; cpp) echo "cpp cpp_v2" ;; *) echo "$1" ;; esac; }
ev() {  # ev <served model> <lang> <eval set> <label> → the p15 report table
  local out="$R15/$3__$4.json"
  [ -f "$out" ] && return 0
  python training/eval_vllm.py --model "$1" --style sft --lang "$2" \
    --eval-file "data/datasets/evalsets/$3.jsonl" --n 100 --no-track --out "$out" 2>&1 | tail -n 1
}

note "\"event\":\"stage4_start\",\"hours_left\":$(hours_left)"

# ── 4 shared adapter ───────────────────────────────────────────────────────────────────────────
if [ ! -f "$XL/adapter/adapter_model.safetensors" ] && [ -s "$X/pairs.jsonl" ]; then
  python training/train_xlang.py build-mix --pairs "$X/pairs.jsonl" --out "$MIX" \
    --max-rows "${XL_ROWS:-12000}" --same-lang "${XL_SAME:-2000}" > "$D/stage4_mix.log" 2>&1
  note "\"event\":\"stage4_train_start\",\"rows\":$(wc -l < "$MIX")"
  python -u training/train_xlang.py train --recipe cloud_14b --data "$MIX" --out "$XL" --eval-frac 0 \
    --max-length 2048 --batch-size 2 --grad-accum 4 --epochs "${XL_EPOCHS:-2}" > "$D/stage4_train.log" 2>&1
  note "\"event\":\"stage4_train_end\",\"exit\":$?"
fi

# ── 4e program-suite eval (vLLM; cached per model, so a restart only redoes what is missing) ─────
run_eval() {  # run_eval <served model> <label> [args…]
  local m=$1 label=$2; shift 2
  [ -f "$R/eval_programs_$label.json" ] && return 0
  gen "$m" --pool-file "$X/eval_pool.jsonl" --cache-dir "$X/eval/$label" --retries 2 \
    --report "$R/eval_programs_$label.json" "$@" > "$D/eval_$label.log" 2>&1
  note "\"event\":\"eval_programs\",\"label\":\"$label\",\"exit\":$?"
}
python training/train_xlang.py eval-pool --out "$X/eval_pool.jsonl" > /dev/null
mods=(); [ -f "$XL/adapter/adapter_model.safetensors" ] && mods=("xlang14b=$XL/adapter")
if serve 8192 "$B14" "${mods[@]}"; then
  run_eval "$B14" base14b
  if [ ${#mods[@]} -gt 0 ]; then
    run_eval xlang14b xlang14b_product
    run_eval xlang14b xlang14b_sft --prompt-style sft
  fi
fi
unserve
[ -n "${REBIRTH_S3_BUCKET:-}" ] && aws s3 sync "$X" "s3://$REBIRTH_S3_BUCKET/results/xlang_stage3" --only-show-errors

# ── 4t newer models on the same suite (one on disk at a time; a first load from disk takes ~12 min) ──
for m in $TEACHERS; do
  t=$(tag "$m")
  [ -f "$R/eval_programs_$t.json" ] && continue
  enough 1 || { note "\"event\":\"teacher_eval_stop\",\"why\":\"time budget\",\"hours_left\":$(hours_left)"; break; }
  for o in $TEACHERS; do [ "$o" != "$m" ] && drop "$o"; done
  note "\"event\":\"teacher_eval_start\",\"model\":\"$m\",\"free_gb\":$(free_gb)"
  fetch "$m" || continue
  READY_MIN=30 VLLM_EXTRA="$(extra_for "$m")" serve 8192 "$m" || continue
  run_eval "$m" "$t"
  unserve
done
[ -n "${REBIRTH_S3_BUCKET:-}" ] && aws s3 sync "$X" "s3://$REBIRTH_S3_BUCKET/results/xlang_stage3" --only-show-errors

# ── D DPO rollout (conditional on S4) ───────────────────────────────────────────────────────────
python - "$R15" "$R" <<'PY'
import json, sys
from pathlib import Path
r15, r = Path(sys.argv[1]), Path(sys.argv[2])
gains, rows = [], {}
for s in ("go", "rust_v2"):
    try:
        a = json.loads((r15 / f"{s}__14b_dpo.json").read_text())["suites"]["synthetic"]
        b = json.loads((r15 / f"{s}__14b.json").read_text())["suites"]["synthetic"]
    except (OSError, KeyError, ValueError) as exc:
        rows[s] = {"error": str(exc)}; continue
    g = 100.0 * (a["soft_exact"] - b["soft_exact"]) / a["n"]
    rows[s] = {"14b_dpo": a["soft_exact"], "14b": b["soft_exact"], "n": a["n"], "gain_per_100": round(g, 1)}
    gains.append(g)
ok = len(gains) == 2 and sum(gains) / 2 >= 2 and min(gains) >= -1
out = {"rule": "roll DPO out if mean gain >= +2 soft-exact/100 on go + rust_v2 and neither < -1", "sets": rows, "pass": ok}
(r / "dpo_rollout_gate.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out))
sys.exit(0 if ok else 3)
PY
gate=$?
note "\"event\":\"dpo_rollout_gate\",\"pass\":$([ $gate -eq 0 ] && echo true || echo false)"
if [ $gate -eq 0 ]; then
  for l in $DPO_MORE; do
    have14 "$l" || continue
    out="data/models/lora_${l}_14b_dpo"
    [ -f "$out/adapter/adapter_model.safetensors" ] && continue
    enough 1.5 || { note "\"event\":\"dpo_rollout_stop\",\"why\":\"time budget\",\"hours_left\":$(hours_left)"; break; }
    [ -f "$P/${l}_14b_prompts.jsonl" ] || python training/build_dpo_pairs.py sample --lang "$l" --n "$DPO_PROMPTS" \
      --max-tokens 512 --tokenizer "$B14" --out "$P/${l}_14b_prompts.jsonl" 2>&1 | tail -n 1
    if [ ! -f "$R15/${l}_dpo_gen.json" ]; then
      serve 4096 "$B14" "${l}14b=data/models/lora_${l}_14b/adapter" || continue
      python training/eval_vllm.py --model "${l}14b" --style sft --lang "$l" --eval-file "$P/${l}_14b_prompts.jsonl" \
        --n "$(wc -l < "$P/${l}_14b_prompts.jsonl")" --no-track --out "$R15/${l}_dpo_gen.json" 2>&1 | tail -n 1
      unserve
    fi
    [ -f "$P/${l}_14b_pairs.jsonl" ] || python training/build_dpo_pairs.py pairs --lang "$l" \
      --prompts "$P/${l}_14b_prompts.jsonl" --report "$R15/${l}_dpo_gen.json" --out "$P/${l}_14b_pairs.jsonl" 2>&1 | tail -n 12
    note "\"event\":\"dpo_train_start\",\"lang\":\"$l\",\"pairs\":$(wc -l < "$P/${l}_14b_pairs.jsonl")"
    python -u training/train_dpo.py --lang "$l" --pairs "$P/${l}_14b_pairs.jsonl" \
      --sft-adapter "data/models/lora_${l}_14b/adapter" --out "$out" --quant none --max-length 1024 \
      --attn-impl sdpa --lr 1e-5 --beta 0.1 --epochs 1 > "$D/dpo_train_$l.log" 2>&1
    note "\"event\":\"dpo_train_end\",\"lang\":\"$l\",\"exit\":$?"
    rm -rf "$out/runs"
    if [ -f "$out/adapter/adapter_model.safetensors" ] && serve 4096 "$B14" "${l}14bdpo=$out/adapter"; then
      for s in $(sets_for "$l"); do ev "${l}14bdpo" "$l" "$s" 14b_dpo; done
      unserve
    fi
  done
fi
note "\"event\":\"stage4_end\",\"hours_left\":$(hours_left)"
