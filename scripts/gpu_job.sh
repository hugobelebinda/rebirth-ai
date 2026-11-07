#!/usr/bin/env bash
# The current GPU job. Must be restartable (skip finished stages, resume trainings) because
# GPU hours are limited (scripts/gpu_schedule.sh). Point this at whatever runs next.
#   systemd-run --user --unit=rebirth-gpu-job --working-directory="$PWD" bash scripts/gpu_job.sh
#
# Now (2026-11-06): Phase 12 v3 wave, Python + C# first (largest data gains: 4.6x / 2.8x).
# Per language: v2 on the frozen set → v3 train (expanded corpus, recipe v3_expanded) → v3 on the
# frozen set → registry gate (dry-run, reported only). Override: WAVE_LANGS="swift java js_ts".
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export WAVE_LANGS="${WAVE_LANGS:-python csharp}"
exec bash scripts/wave_v3.sh
