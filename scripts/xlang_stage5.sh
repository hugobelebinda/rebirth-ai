#!/usr/bin/env bash
# Phase 17 stage 5 (user's go 2026-11-10): the shared cross-language adapter (Ollama
# rebirth-xlang:14b-v1 = q6_K 14B + GGUF LoRA) on the directions not covered by stage 4, with the
# SAME sources as the plain-14B baselines (bootstrapped from the baseline reports → paired comparison):
#   A  program suite, 35 directions from JS / Go / C# / Rust / Swift   vs programs_ollama14b_bootstrap.json
#   B1 HumanEval, 35 directions from Python / JS / Go / Java / C++     vs zeroshot_ollama14b.json
#   B2 HumanEval, 21 directions from Rust / C# / Swift                 vs zeroshot_ollama14b_bootstrap.json
# Restartable (generations cached per report). Run as the stoppable GPU unit:
#   systemd-run --user --unit=rebirth-gpu-job --working-directory="$PWD" bash scripts/xlang_stage5.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
X=logs/exp/xlang
R=data/reports/xlang
S="$X/stage5_status.jsonl"
mkdir -p "$X" "$R"
note() { echo "{\"t\":\"$(date -Is)\",$1}" | tee -a "$S"; }
export REBIRTH_EMBEDDER=hash OLLAMA_MODEL="${XLANG_TAG:-rebirth-xlang:14b-v1}"
PY=".venv/bin/python -u training/eval_translate.py --backend ollama"

note "\"event\":\"A_start\",\"model\":\"$OLLAMA_MODEL\""
$PY --suite programs --limit 20 --retries 2 --sources javascript go csharp rust swift \
  --add-sources-from "$R/programs_ollama14b_retry.json" \
  --out "$R/programs_xlang14b_q6k_bootstrap.json" > "$X/stage5_programs_bootstrap.log" 2>&1
note "\"event\":\"A_end\",\"exit\":$?"

note "\"event\":\"B1_start\""
$PY --limit 40 --out "$R/zeroshot_xlang14b_q6k.json" > "$X/stage5_humaneval_pass1.log" 2>&1
note "\"event\":\"B1_end\",\"exit\":$?"

note "\"event\":\"B2_start\""
$PY --limit 40 --sources rust csharp swift --add-sources-from "$R/zeroshot_ollama14b.json" \
  --out "$R/zeroshot_xlang14b_q6k_bootstrap.json" > "$X/stage5_humaneval_pass2.log" 2>&1
note "\"event\":\"B2_end\",\"exit\":$?"
