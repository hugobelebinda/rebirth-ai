#!/usr/bin/env bash
# Start a detached job on the instance (survives SSH disconnects) + its watchdog.
#   bash scripts/cloud/run_job.sh <name> <command…>
#   AUTO_SHUTDOWN=0 keeps the instance up after the job; IDLE_MIN=30 no-progress limit;
#   SHUTDOWN_DELAY_MIN=1 minutes between the job's end and the stop.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
name="${1:?usage: run_job.sh <name> <command…>}"; shift
cmd="$*"
[ -n "$cmd" ] || { echo "no command" >&2; exit 2; }
q=$(printf '%q' "$cmd")
remote "mkdir -p logs/cloud/$name; rm -f logs/cloud/$name/exit
  nohup setsid bash -c 'source .venv/bin/activate; export HF_HUB_OFFLINE=1 REBIRTH_EMBEDDER=hash; '$q'; echo \$? > logs/cloud/$name/exit' \
    > logs/cloud/$name/out.log 2>&1 < /dev/null &
  nohup setsid env AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-1} IDLE_MIN=${IDLE_MIN:-30} SHUTDOWN_DELAY_MIN=${SHUTDOWN_DELAY_MIN:-1} \
    REBIRTH_S3_BUCKET=${REBIRTH_S3_BUCKET:-} \
    bash scripts/cloud/watchdog.sh $name > logs/cloud/$name/watchdog.log 2>&1 < /dev/null &
  echo started $name"
