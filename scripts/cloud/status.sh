#!/usr/bin/env bash
# Progress of a cloud job:  bash scripts/cloud/status.sh <name>
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
name="${1:?usage: status.sh <name>}"
remote "echo '--- exit:' \$(cat logs/cloud/$name/exit 2>/dev/null || echo running); tail -n ${LINES_:-15} logs/cloud/$name/out.log;
  tail -n 2 logs/cloud/$name/watchdog.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader"
