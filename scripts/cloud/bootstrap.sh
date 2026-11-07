#!/usr/bin/env bash
# From the home PC: sync the repo up, then run setup.sh on the instance (~10–15 min first time).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
bash "$ROOT/scripts/cloud/sync_up.sh"
remote "bash scripts/cloud/setup.sh 2>&1 | tail -20"
