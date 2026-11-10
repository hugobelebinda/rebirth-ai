#!/usr/bin/env bash
# ON the instance: wait for job <name>; on finish (or IDLE_MIN minutes without log progress and
# an idle GPU) copy results to S3 (if REBIRTH_S3_BUCKET) and shut down (if AUTO_SHUTDOWN=1).
# Terraform sets shutdown behaviour = stop: GPU billing ends, the disk (and results) survive.
# SHUTDOWN_DELAY_MIN (default 1) leaves the home side time to sync results down and terminate.
set -uo pipefail
cd ~/rebirth
export PATH="$HOME/.local/bin:$PATH"   # aws CLI from setup.sh (uv tool)
name="$1"; d="logs/cloud/$name"; idle_limit=$(( ${IDLE_MIN:-30} * 60 ))
while :; do
  sleep 60
  if [ -f "$d/exit" ]; then reason="job finished (exit $(cat "$d/exit"))"; break; fi
  # progress = the newest write to any job log or training step log (a long training run writes
  # its own logs, not out.log; one GPU sample during a checkpoint save can read 0 %)
  newest=$(find "$d" logs/train -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
  age=$(( $(date +%s) - ${newest:-$(date +%s)} ))
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
  if [ "$age" -gt "$idle_limit" ] && [ "${util:-0}" -lt 5 ]; then reason="no progress for $((age / 60)) min, GPU ${util}%"; break; fi
done
echo "$(date -Is) watchdog: $reason"
if [ -n "${REBIRTH_S3_BUCKET:-}" ]; then
  aws s3 sync data/reports "s3://$REBIRTH_S3_BUCKET/results/reports" --only-show-errors || true
  aws s3 sync data/models "s3://$REBIRTH_S3_BUCKET/results/models" --exclude '*/runs/*' --only-show-errors || true
  aws s3 sync logs "s3://$REBIRTH_S3_BUCKET/results/logs" --only-show-errors || true
  [ -d data/datasets/dpo ] && { aws s3 sync data/datasets/dpo "s3://$REBIRTH_S3_BUCKET/results/dpo" --only-show-errors || true; }
  [ -d data/datasets/xlang/stage3 ] && { aws s3 sync data/datasets/xlang/stage3 "s3://$REBIRTH_S3_BUCKET/results/xlang_stage3" --only-show-errors || true; }
fi
if [ "${AUTO_SHUTDOWN:-1}" = 1 ]; then echo "$(date -Is) watchdog: shutting down"; sudo shutdown -h +"${SHUTDOWN_DELAY_MIN:-1}" "rebirth watchdog: $reason"; fi
