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

note "\"event\":\"A1_start\",\"model\":\"$OLLAMA_MODEL\""
$PY --suite programs --limit 20 --retries 2 --out "$R/programs_$T.json" > "$X/v2_programs.log" 2>&1
note "\"event\":\"A1_end\",\"exit\":$?"

note "\"event\":\"A2_start\""
$PY --suite programs --limit 20 --retries 2 --sources javascript go csharp rust swift \
  --add-sources-from "$R/programs_ollama14b_retry.json" \
  --out "$R/programs_${T}_bootstrap.json" > "$X/v2_programs_bootstrap.log" 2>&1
note "\"event\":\"A2_end\",\"exit\":$?"

note "\"event\":\"B1_start\""
$PY --limit 40 --out "$R/zeroshot_$T.json" > "$X/v2_humaneval_pass1.log" 2>&1
note "\"event\":\"B1_end\",\"exit\":$?"

note "\"event\":\"B2_start\""
$PY --limit 40 --sources rust csharp swift --add-sources-from "$R/zeroshot_ollama14b.json" \
  --out "$R/zeroshot_${T}_bootstrap.json" > "$X/v2_humaneval_pass2.log" 2>&1
note "\"event\":\"B2_end\",\"exit\":$?"
