#!/usr/bin/env bash
# Phase 17 stage 2: zero-shot translation baseline (Ollama 14B) on HumanEval, all 56 directions.
#   pass 1: 35 directions with reference sources (Python, JS, Go, Java, C++ → 8 targets)
#   pass 2: 21 directions from Rust / C# / Swift, sources = pass-1 translations that passed
# Restartable: generations are cached in <report>.gen.jsonl. LIMIT=40 problems per direction.
#   systemd-run --user --unit=rebirth-gpu-job --working-directory="$PWD" bash scripts/xlang_stage2.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LIMIT="${LIMIT:-40}"
X=logs/exp/xlang
R=data/reports/xlang
mkdir -p "$X" "$R"
S="$X/stage2_status.jsonl"
note() { echo "{\"t\":\"$(date -Is)\",$1}" | tee -a "$S"; }
export REBIRTH_EMBEDDER=hash
note "\"event\":\"pass1_start\",\"limit\":$LIMIT"
.venv/bin/python -u training/eval_translate.py --backend ollama --limit "$LIMIT" \
  --out "$R/zeroshot_ollama14b.json" > "$X/stage2_pass1.log" 2>&1
note "\"event\":\"pass1_end\",\"exit\":$?"
[ -f "$R/zeroshot_ollama14b.json" ] || exit 1
note "\"event\":\"pass2_start\""
.venv/bin/python -u training/eval_translate.py --backend ollama --limit "$LIMIT" \
  --sources rust csharp swift --add-sources-from "$R/zeroshot_ollama14b.json" \
  --out "$R/zeroshot_ollama14b_bootstrap.json" > "$X/stage2_pass2.log" 2>&1
note "\"event\":\"pass2_end\",\"exit\":$?"
