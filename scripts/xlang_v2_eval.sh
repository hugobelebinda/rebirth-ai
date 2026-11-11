#!/usr/bin/env bash
# Phase 19 gate for the merged cross-language model (rebirth-xlang:14b-v2 = lora_xlang_14b merged into
# the bf16 14B, then q6_K; built by scripts/xlang_v2_merge.sh). Same suites, sources and settings as
# the v1 reports, so every row pairs with a v1 row:
#   A1 program suite, 21 directions (Python / C++ / Java sources)     vs programs_xlang14b_q6k.json            (v1 265 / 420)
#   A2 program suite, 35 bootstrapped directions                       vs programs_xlang14b_q6k_bootstrap.json  (v1 407 / 511)
#   B1 HumanEval, 35 directions                                        vs zeroshot_xlang14b_q6k.json            (v1 1296 / 1396)
#   B2 HumanEval, 21 bootstrapped directions                           vs zeroshot_xlang14b_q6k_bootstrap.json  (v1 768 / 833)
# Restartable (generations cached per report). Run as the stoppable GPU unit:
#   systemd-run --user --unit=rebirth-gpu-job --working-directory="$PWD" bash scripts/xlang_v2_eval.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
X=logs/exp/xlang
R=data/reports/xlang
S="$X/v2_eval_status.jsonl"
T="${XLANG_V2_SLUG:-xlang14b_v2_q6k}"
mkdir -p "$X" "$R"
note() { echo "{\"t\":\"$(date -Is)\",$1}" | tee -a "$S"; }
export REBIRTH_EMBEDDER=hash OLLAMA_MODEL="${XLANG_TAG:-rebirth-xlang:14b-v2}"
PY=".venv/bin/python -u training/eval_translate.py --backend ollama"

# stage NAME OUT LOG ARGS...: skipped when its report exists (a stopped run restarts at the first
# missing report; a half-done stage resumes from its cached generations)
stage() {
  local name=$1 out=$2 log=$3; shift 3
  if [ -f "$out" ]; then note "\"event\":\"${name}_skip\",\"report\":\"$out\""; return; fi
  note "\"event\":\"${name}_start\",\"model\":\"$OLLAMA_MODEL\""
  $PY "$@" --out "$out" > "$X/$log" 2>&1
  note "\"event\":\"${name}_end\",\"exit\":$?"
}

stage A1 "$R/programs_$T.json" v2_programs.log --suite programs --limit 20 --retries 2
stage A2 "$R/programs_${T}_bootstrap.json" v2_programs_bootstrap.log --suite programs --limit 20 --retries 2 \
  --sources javascript go csharp rust swift --add-sources-from "$R/programs_ollama14b_retry.json"
stage B1 "$R/zeroshot_$T.json" v2_humaneval_pass1.log --limit 40
stage B2 "$R/zeroshot_${T}_bootstrap.json" v2_humaneval_pass2.log --limit 40 \
  --sources rust csharp swift --add-sources-from "$R/zeroshot_ollama14b.json"
