#!/usr/bin/env bash
# ON the instance: Phase 15 S1 → S4 in one session (user's go 2026-11-08: full corpora, one launch).
#   bash scripts/cloud/run_job.sh p15 "bash scripts/cloud/session_p15.sh"
#
#   S1  vLLM baselines on the frozen sets: base 7B (chat), v2 7B adapters (sft), base 14B (chat)
#   S2  14B LoRA on Go + Python (data/datasets/p15, recipe cloud_14b), eval
#       GATE (CLOUD_GPU_PLAN §5): 14B adapter ≥ v2 7B adapter + 3 soft-exact per 100 rows on BOTH,
#       else the session ends here (the watchdog then stops the instance)
#   S3  14B LoRA on the other 6 languages (smallest first), eval (Rust / C++ also on the _v2 sets)
#   S4  DPO on $DPO_LANGS (default go rust): 2000 training prompts → the 14B adapter's own outputs
#       (vLLM) → gold ≻ own failure pairs → DPO (bf16, seq 1024) → eval vs the S3 adapter
#
# Restartable: finished trainings leave logs/cloud/p15/train_<lang>.done, finished evals are skipped
# (one report per set × model in data/reports/cloud/p15/<set>__<label>.json). Tracking is off here;
# runs and reports are recorded at home after sync_down.
set -uo pipefail
cd ~/rebirth
export REBIRTH_TRACKING=0 HF_HUB_OFFLINE=1 REBIRTH_EMBEDDER=hash
B7=Qwen/Qwen2.5-Coder-7B-Instruct
B14=Qwen/Qwen2.5-Coder-14B-Instruct
D=logs/cloud/p15
R=data/reports/cloud/p15
P=data/datasets/dpo
mkdir -p "$D" "$R" "$P"
LANGS="go python swift rust csharp cpp java js_ts"
S2_LANGS="go python"
S3_LANGS="swift rust csharp cpp java js_ts"   # smallest corpus first: more finished if time runs out
DPO_LANGS="${DPO_LANGS:-go rust}"
DPO_PROMPTS="${DPO_PROMPTS:-2000}"

note() { echo "{\"t\":\"$(date -Is)\",$1}" | tee -a "$D/status.jsonl"; }
sets_for() { case $1 in rust) echo "rust rust_v2" ;; cpp) echo "cpp cpp_v2" ;; *) echo "$1" ;; esac; }
have14() { [ -f "$D/train_$1.done" ] && [ -f "data/models/lora_$1_14b/adapter/adapter_model.safetensors" ]; }
unserve() { docker rm -f rebirth-vllm >/dev/null 2>&1 || true; }
serve() {
  unserve
  if ! bash scripts/cloud/vllm_serve.sh "$@"; then
    note "\"event\":\"vllm_failed\",\"base\":\"$1\""; docker logs --tail 40 rebirth-vllm 2>&1; unserve; return 1
  fi
}
ev() {  # ev <served model> <sft|chat> <lang> <eval set> <label>
  local out="$R/$4__$5.json"
  [ -f "$out" ] && return 0
  python training/eval_vllm.py --model "$1" --style "$2" --lang "$3" \
    --eval-file "data/datasets/evalsets/$4.jsonl" --n 100 --no-track --out "$out" 2>&1 | tail -n 1
}
train() {  # train <lang>: one epoch of the p15 corpus on the 14B base
  local l=$1
  have14 "$l" && return 0
  note "\"event\":\"train_start\",\"lang\":\"$l\""
  bash scripts/train_logged.sh "$l" 1024 --recipe cloud_14b --data "data/datasets/p15/${l}_migration.jsonl" \
    > "$D/train_$l.log" 2>&1
  local rc=$?
  note "\"event\":\"train_end\",\"lang\":\"$l\",\"exit\":$rc"
  [ "$rc" -eq 0 ] && [ -f "data/models/lora_${l}_14b/adapter/adapter_model.safetensors" ] && touch "$D/train_$l.done"
  return "$rc"
}
eval14() {  # eval14 <label suffix> <lang…>: serve the 14B with these languages' adapters, score their sets
  local suffix=$1; shift
  local mods=() l s
  for l in "$@"; do
    local dir="data/models/lora_${l}_14b${suffix:+_$suffix}/adapter"
    [ -f "$dir/adapter_model.safetensors" ] && mods+=("${l}14b${suffix}=$dir")
  done
  [ ${#mods[@]} -gt 0 ] || return 0
  serve "$B14" "${mods[@]}" || return 1
  for l in "$@"; do
    [ -f "data/models/lora_${l}_14b${suffix:+_$suffix}/adapter/adapter_model.safetensors" ] || continue
    for s in $(sets_for "$l"); do ev "${l}14b${suffix}" sft "$l" "$s" "14b${suffix:+_$suffix}"; done
  done
  unserve
}

# ── S1 baselines ───────────────────────────────────────────────────────────────────────────────
note '"event":"s1_start"'
mods=(); for l in $LANGS; do mods+=("${l}7b=data/models/lora_${l}_7b/adapter"); done
if serve "$B7" "${mods[@]}"; then
  for l in $LANGS; do for s in $(sets_for "$l"); do ev "$B7" chat "$l" "$s" base7b; ev "${l}7b" sft "$l" "$s" v2_7b; done; done
fi
if serve "$B14"; then
  for l in $LANGS; do for s in $(sets_for "$l"); do ev "$B14" chat "$l" "$s" base14b; done; done
fi
unserve
note "\"event\":\"s1_end\",\"reports\":$(ls "$R" | wc -l)"

# ── S2 pilot + gate ────────────────────────────────────────────────────────────────────────────
note '"event":"s2_start"'
for l in $S2_LANGS; do train "$l"; done
eval14 "" $S2_LANGS
python - "$R" $S2_LANGS <<'PY'
import json, sys
from pathlib import Path
R, langs = Path(sys.argv[1]), sys.argv[2:]
rows, ok = {}, True
for lang in langs:
    try:
        a = json.loads((R / f"{lang}__14b.json").read_text())["suites"]["synthetic"]
        b = json.loads((R / f"{lang}__v2_7b.json").read_text())["suites"]["synthetic"]
    except (OSError, KeyError, ValueError) as exc:
        rows[lang] = {"error": str(exc)}; ok = False; continue
    gain = 100.0 * (a["soft_exact"] - b["soft_exact"]) / a["n"]
    rows[lang] = {"14b": a["soft_exact"], "v2_7b": b["soft_exact"], "n": a["n"], "gain_per_100": round(gain, 1)}
    ok = ok and gain >= 3
out = {"rule": "14B adapter >= v2 7B adapter + 3 soft-exact per 100 rows on every pilot language",
       "langs": rows, "pass": ok}
(R / "gate_s2.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
sys.exit(0 if ok else 3)
PY
gate=$?
note "\"event\":\"s2_gate\",\"pass\":$([ $gate -eq 0 ] && echo true || echo false)"
if [ $gate -ne 0 ]; then
  note '"event":"session_end","why":"S2 gate failed — S3/S4 skipped"'
  exit 0
fi

# ── S3 wave ────────────────────────────────────────────────────────────────────────────────────
note '"event":"s3_start"'
for l in $S3_LANGS; do train "$l"; done
eval14 "" $S3_LANGS
note '"event":"s3_end"'

# ── S4 DPO ─────────────────────────────────────────────────────────────────────────────────────
note "\"event\":\"s4_start\",\"langs\":\"$DPO_LANGS\""
todo=(); for l in $DPO_LANGS; do have14 "$l" && todo+=("$l"); done
mods=()
for l in "${todo[@]}"; do
  [ -f "$P/${l}_14b_prompts.jsonl" ] || python training/build_dpo_pairs.py sample --lang "$l" --n "$DPO_PROMPTS" \
    --max-tokens 512 --tokenizer "$B14" --out "$P/${l}_14b_prompts.jsonl" 2>&1 | tail -n 1
  mods+=("${l}14b=data/models/lora_${l}_14b/adapter")
done
if [ ${#todo[@]} -gt 0 ] && serve "$B14" "${mods[@]}"; then
  for l in "${todo[@]}"; do
    n=$(wc -l < "$P/${l}_14b_prompts.jsonl")
    [ -f "$R/${l}_dpo_gen.json" ] || python training/eval_vllm.py --model "${l}14b" --style sft --lang "$l" \
      --eval-file "$P/${l}_14b_prompts.jsonl" --n "$n" --no-track --out "$R/${l}_dpo_gen.json" 2>&1 | tail -n 1
  done
fi
unserve
for l in "${todo[@]}"; do
  [ -f "$R/${l}_dpo_gen.json" ] || continue
  [ -f "$P/${l}_14b_pairs.jsonl" ] || python training/build_dpo_pairs.py pairs --lang "$l" \
    --prompts "$P/${l}_14b_prompts.jsonl" --report "$R/${l}_dpo_gen.json" --out "$P/${l}_14b_pairs.jsonl" 2>&1 | tail -n 12
  out="data/models/lora_${l}_14b_dpo"
  [ -f "$out/adapter/adapter_model.safetensors" ] && continue
  note "\"event\":\"dpo_train_start\",\"lang\":\"$l\",\"pairs\":$(wc -l < "$P/${l}_14b_pairs.jsonl")"
  python -u training/train_dpo.py --lang "$l" --pairs "$P/${l}_14b_pairs.jsonl" \
    --sft-adapter "data/models/lora_${l}_14b/adapter" --out "$out" --quant none --max-length 1024 \
    --attn-impl sdpa --lr 1e-5 --beta 0.1 --epochs 1 > "$D/dpo_train_$l.log" 2>&1
  note "\"event\":\"dpo_train_end\",\"lang\":\"$l\",\"exit\":$?"
done
eval14 dpo "${todo[@]}"
note '"event":"s4_end"'
note '"event":"session_end","why":"all stages ran"'
