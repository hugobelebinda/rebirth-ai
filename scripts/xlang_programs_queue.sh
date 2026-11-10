#!/usr/bin/env bash
# Phase 17 (2026-11-08, user's go): program-suite follow-ups, then stage 3 data generation.
#   1. 14B with the product retry loop (2 rounds) on the 21 pass-1 directions (reuses pass-1 generations)
#   2. 14B on the 35 directions from JS / Go / C# / Rust / Swift (sources = passing translations), 2 retries
#   3. base 7B on the 21 pass-1 directions, 2 retries
#   4. stage 3: training/gen_xlang_pairs.py generate — runs until the pool is used up or the unit is stopped
# Restartable (every step caches its generations). Run as the stoppable GPU unit:
#   systemd-run --user --unit=rebirth-gpu-job --working-directory="$PWD" bash scripts/xlang_programs_queue.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
X=logs/exp/xlang
R=data/reports/xlang
S="$X/programs_queue_status.jsonl"
mkdir -p "$X" "$R"
note() { echo "{\"t\":\"$(date -Is)\",$1}" | tee -a "$S"; }
export REBIRTH_EMBEDDER=hash
PY=".venv/bin/python -u"
EV="training/eval_translate.py --suite programs --limit 20 --retries 2"

if [ ! -f "$R/programs_ollama14b_retry.gen.jsonl" ]; then
  cp "$R/programs_ollama14b.gen.jsonl" "$R/programs_ollama14b_retry.gen.jsonl"  # round 0 = pass 1
fi
note "\"event\":\"retry14b_start\""
$PY $EV --sources python cpp java --out "$R/programs_ollama14b_retry.json" > "$X/programs_retry14b.log" 2>&1
rc=$?; note "\"event\":\"retry14b_end\",\"exit\":$rc"

note "\"event\":\"bootstrap14b_start\""
$PY $EV --sources javascript go csharp rust swift --add-sources-from "$R/programs_ollama14b_retry.json" \
  --out "$R/programs_ollama14b_bootstrap.json" > "$X/programs_bootstrap14b.log" 2>&1
rc=$?; note "\"event\":\"bootstrap14b_end\",\"exit\":$rc"

note "\"event\":\"base7b_start\""
OLLAMA_MODEL=qwen2.5-coder:7b-instruct $PY $EV --sources python cpp java \
  --out "$R/programs_ollama7b.json" > "$X/programs_base7b.log" 2>&1
rc=$?; note "\"event\":\"base7b_end\",\"exit\":$rc"

note "\"event\":\"stage3_generate_start\""
$PY training/gen_xlang_pairs.py generate >> "$X/stage3_generate.log" 2>&1
rc=$?; note "\"event\":\"stage3_generate_end\",\"exit\":$rc"
