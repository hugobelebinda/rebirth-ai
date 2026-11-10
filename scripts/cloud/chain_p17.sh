#!/usr/bin/env bash
# ON the instance: when the p15 session ends, keep the instance and start the p17 session on it
# (capacity for g6e.xlarge is scarce: no stop / relaunch between the two). Started once, detached:
#   nohup setsid env REBIRTH_S3_BUCKET=<bucket> bash scripts/cloud/chain_p17.sh > logs/cloud/chain_p17.log 2>&1 &
# p15's watchdog copies results to S3, schedules a stop (+SHUTDOWN_DELAY_MIN) and exits: wait for
# that exit, cancel the stop, then start p17 + its own watchdog (which stops the instance after p17).
set -uo pipefail
cd ~/rebirth
while [ ! -f logs/cloud/p15/exit ]; do sleep 30; done
echo "$(date -Is) p15 finished (exit $(cat logs/cloud/p15/exit))"
for _ in $(seq 1 240); do pgrep -f "watchdog.sh p15" >/dev/null || break; sleep 15; done
sleep 5
sudo shutdown -c 2>/dev/null && echo "$(date -Is) p15 stop cancelled" || echo "$(date -Is) no stop to cancel"
[ -f scripts/cloud/session_p17.sh ] || { echo "$(date -Is) no session_p17.sh — stopping"; sudo shutdown -h +1; exit 1; }
mkdir -p logs/cloud/p17; rm -f logs/cloud/p17/exit
nohup setsid bash -c 'source .venv/bin/activate; export HF_HUB_OFFLINE=1 REBIRTH_EMBEDDER=hash; bash scripts/cloud/session_p17.sh; echo $? > logs/cloud/p17/exit' \
  > logs/cloud/p17/out.log 2>&1 < /dev/null &
nohup setsid env AUTO_SHUTDOWN=1 IDLE_MIN="${IDLE_MIN:-45}" SHUTDOWN_DELAY_MIN="${SHUTDOWN_DELAY_MIN:-20}" \
  REBIRTH_S3_BUCKET="${REBIRTH_S3_BUCKET:-}" bash scripts/cloud/watchdog.sh p17 > logs/cloud/p17/watchdog.log 2>&1 < /dev/null &
echo "$(date -Is) started p17"
