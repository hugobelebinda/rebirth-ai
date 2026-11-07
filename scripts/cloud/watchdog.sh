#!/usr/bin/env bash
# ON the instance: wait for job <name>; on finish (or IDLE_MIN minutes without log progress and
# an idle GPU) copy results to S3 (if REBIRTH_S3_BUCKET) and shut down (if AUTO_SHUTDOWN=1).
# Terraform sets shutdown behaviour = stop: GPU billing ends, the disk (and results) survive.
set -uo pipefail
cd ~/rebirth
name="$1"; d="logs/cloud/$name"; idle_limit=$(( ${IDLE_MIN:-30} * 60 ))
while :; do
  sleep 60
  if [ -f "$d/exit" ]; then reason="job finished (exit $(cat "$d/exit"))"; break; fi
  age=$(( $(date +%s) - $(stat -c %Y "$d/out.log" 2>/dev/null || date +%s) ))
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
  if [ "$age" -gt "$idle_limit" ] && [ "${util:-0}" -lt 5 ]; then reason="no progress for $((age / 60)) min, GPU ${util}%"; break; fi
done
echo "$(date -Is) watchdog: $reason"
if [ -n "${REBIRTH_S3_BUCKET:-}" ]; then
  aws s3 sync data/reports "s3://$REBIRTH_S3_BUCKET/results/reports" --only-show-errors || true
  aws s3 sync data/models "s3://$REBIRTH_S3_BUCKET/results/models" --exclude '*/runs/*' --only-show-errors || true
  aws s3 sync logs "s3://$REBIRTH_S3_BUCKET/results/logs" --only-show-errors || true
fi
if [ "${AUTO_SHUTDOWN:-1}" = 1 ]; then echo "$(date -Is) watchdog: shutting down"; sudo shutdown -h +1 "rebirth watchdog: $reason"; fi
