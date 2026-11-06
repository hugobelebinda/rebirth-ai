#!/usr/bin/env bash
# After a GPU hang / hard reboot during scripts/train_logged.sh, summarise where and when it died.
#
#   bash scripts/gpu_hang_postmortem.sh logs/train/<lang>_seq<N>_<ts>   [boot offset, default -1]
#
# Writes <run>/postmortem.txt (and prints it). Needs the persistent journal (/var/log/journal).
set -uo pipefail
run="${1:?usage: gpu_hang_postmortem.sh <run dir> [boot offset]}"
boot="${2:--1}"
out="$run/postmortem.txt"

{
  echo "== run: $run"
  [ -f "$run/exit.json" ] && echo "exit.json: $(cat "$run/exit.json")" || echo "exit.json: MISSING (process/machine died mid-run)"
  echo
  echo "== last step events (steps.jsonl)"
  tail -n 6 "$run/steps.jsonl" 2>/dev/null
  echo
  echo "== last GPU telemetry samples (telemetry.jsonl)"
  tail -n 5 "$run/telemetry.jsonl" 2>/dev/null
  echo
  echo "== peak VRAM / temps over run"
  python3 - "$run/telemetry.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip() and '"event"' not in l]
if rows:
    for k in sorted({k for r in rows for k in r if k != "t"}):
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if vals:
            print(f"{k:>18}: max={max(vals)}  last={vals[-1]}")
    print(f"{'samples':>18}: {len(rows)}  span={rows[-1]['t'] - rows[0]['t']:.0f}s  last_t={rows[-1]['t']}")
PY
  echo
  echo "== kernel log, boot $boot (amdgpu/gpu lines, last 60)"
  journalctl -k -b "$boot" --no-pager 2>/dev/null \
    | grep -iE 'amdgpu|drm|gpu|hang|reset|ring|timeout|page fault' | tail -n 60
  echo
  echo "== amdgpu devcoredump (kernel keeps it only ~5 min after a reset; root-only)"
  for dump in /sys/class/drm/card*/device/devcoredump/data; do
    [ -e "$dump" ] || continue
    if cat "$dump" > "$run/amdgpu_devcoredump.txt" 2>/dev/null; then
      echo "saved $dump → $run/amdgpu_devcoredump.txt"
    else
      echo "present but unreadable: sudo cat $dump > $run/amdgpu_devcoredump.txt"
    fi
  done
} | tee "$out"
