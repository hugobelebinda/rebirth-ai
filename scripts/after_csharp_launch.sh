#!/usr/bin/env bash
# One-shot (2026-11-06, user go in chat): when the C# wave ends *normally* (gate line written),
# start the Phase 12 reference job. If the wave was stopped by hand, do nothing.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
S=logs/exp/wave_v3/status.jsonl
while systemctl --user is-active --quiet rebirth-gpu-job; do sleep 60; done
if grep -q '"event":"gate","lang":"csharp"' "$S"; then
  systemctl --user reset-failed rebirth-gpu-job 2>/dev/null
  systemd-run --user --unit=rebirth-gpu-job --working-directory="$PWD" bash scripts/refs_p12.sh && echo launched
else
  echo "C# did not finish normally — not launching"; tail -2 "$S"
fi
