#!/usr/bin/env bash
# Unattended recipe-v2 LoRA wave: Go first as a gate, then the remaining languages.
# Meant to run detached from the editor (survives a GPU reset that kills the IDE):
#
#   systemd-run --user --unit=rebirth-lora-queue --working-directory="$PWD" \
#     bash scripts/train_queue.sh
#   journalctl --user -u rebirth-lora-queue -f        # follow
#   cat logs/train/queue_*/status.md                   # human summary
#
# Per language: GPU health check → scripts/train_logged.sh <lang> 512 --quant 4bit
#   → scripts/eval_lang_lora.sh <lang> (held-out vs base 7B, gate, fixture migrate).
# Go gate FAIL ⇒ stop (don't burn hours on a recipe that does not beat its base).
# Other languages: a failure is recorded and the queue moves on.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LANGS=(${QUEUE_LANGS:-go rust cpp swift csharp java python js_ts})
GATE_LANG="${GATE_LANG:-go}"
MAXLEN="${MAXLEN:-512}"
Q="logs/train/queue_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$Q"
STATUS="$Q/status.jsonl"

note() {  # event lang key=value…
  python3 - "$STATUS" "$@" <<'PY'
import json, sys, time
path, event, lang, *kv = sys.argv[1:]
rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, "lang": lang}
rec.update(dict(x.split("=", 1) for x in kv))
with open(path, "a") as fh:
    fh.write(json.dumps(rec) + "\n"); fh.flush()
print(json.dumps(rec), flush=True)
PY
  render_md
}

render_md() {
  python3 - "$STATUS" "$Q/status.md" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
out = ["# LoRA v2 queue status", "", "| time | lang | event | detail |", "|---|---|---|---|"]
for r in rows:
    detail = ", ".join(f"{k}={v}" for k, v in r.items() if k not in {"t", "lang", "event"})
    out.append(f"| {r['t']} | {r['lang']} | {r['event']} | {detail} |")
open(sys.argv[2], "w").write("\n".join(out) + "\n")
PY
}

gpu_ok() {
  # shellcheck disable=SC1091
  ( source .venv/bin/activate && source scripts/rocm_env.sh && bash scripts/verify_rocm_torch.sh ) \
    > "$Q/gpu_check_$1.log" 2>&1
}

note queue_start all "langs=${LANGS[*]}" "maxlen=$MAXLEN" "git=$(git rev-parse --short HEAD)"

for lang in "${LANGS[@]}"; do
  if ! gpu_ok "$lang"; then
    note gpu_unhealthy "$lang" "action=wait_60s_retry"
    sleep 60
    if ! gpu_ok "$lang"; then note queue_abort "$lang" "reason=gpu_unhealthy"; exit 3; fi
  fi

  note train_start "$lang"
  bash scripts/train_logged.sh "$lang" "$MAXLEN" --quant 4bit > "$Q/train_${lang}.console" 2>&1
  code=$?
  run_dir=$(ls -dt logs/train/${lang}_seq${MAXLEN}_* 2>/dev/null | head -1)
  loss=$(python3 -c "import json;print(round(json.load(open('data/models/lora_${lang}_7b/rebirth_meta.json'))['train_loss'],4))" 2>/dev/null || echo NA)
  note train_end "$lang" "exit=$code" "train_loss=$loss" "run_dir=$run_dir"
  if [ "$code" -ne 0 ]; then
    if [ "$lang" = "$GATE_LANG" ]; then note queue_abort "$lang" "reason=gate_lang_train_failed"; exit 4; fi
    continue
  fi

  note eval_start "$lang"
  bash scripts/eval_lang_lora.sh "$lang" "$run_dir" > "$Q/eval_${lang}.console" 2>&1
  gate=$?
  verdict=$(tail -c 2000 "data/reports/gate_${lang}_v2.json" 2>/dev/null | tr -d '\n' | cut -c1-400)
  fixture=$(grep -oE "status=[a-z_]+ iterations=[0-9]+" "data/reports/migrate_${lang}_sample_peft7b_v2.log" 2>/dev/null | head -1 | tr ' ' ',')
  note eval_end "$lang" "gate_exit=$gate" "fixture=${fixture:-NA}" "gate=$verdict"
  if [ "$lang" = "$GATE_LANG" ] && [ "$gate" -ne 0 ]; then
    note queue_stop "$lang" "reason=gate_failed_see_data/reports/gate_${lang}_v2.json"
    exit 5
  fi
done

# Ollama 14B baselines on the same held-out rows (after all GPU training is done).
note ollama_baseline_start all
# shellcheck disable=SC1091
source .venv/bin/activate
for lang in "${LANGS[@]}"; do
  out="data/reports/eval_heldout_${lang}_ollama.json"
  [ -f "$out" ] && continue
  REBIRTH_EMBEDDER=hash python -u training/eval_behavioral.py --backend ollama --suite synthetic \
    --lang "$lang" --n-synth 24 --out "$out" > "$Q/ollama_${lang}.log" 2>&1 \
    || note ollama_baseline_failed "$lang"
done
ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | xargs -r -n1 ollama stop
note queue_done all
