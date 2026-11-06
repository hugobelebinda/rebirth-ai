#!/usr/bin/env bash
# The current GPU job. Must be restartable (skip finished stages, resume trainings) because
# GPU hours are limited (scripts/gpu_schedule.sh). Point this at whatever runs next.
#   systemd-run --user --unit=rebirth-gpu-job --working-directory="$PWD" bash scripts/gpu_job.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec bash scripts/wave_v3.sh
