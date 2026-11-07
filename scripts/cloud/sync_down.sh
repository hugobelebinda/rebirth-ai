#!/usr/bin/env bash
# Pull results home, then record them in the local MLflow store (post-hoc tracking, Phase 13).
#   bash scripts/cloud/sync_down.sh       (adapters without checkpoints; reports; logs)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need_host
for p in data/reports data/models logs/train logs/cloud; do
  mkdir -p "$ROOT/$p"
  run rsync -az --info=stats1 -e "ssh ${SSH_OPTS[*]}" --exclude 'runs/' \
    "$RUSER@$HOST:~/$RDIR/$p/" "$ROOT/$p/"
done
[ "${DRY_RUN:-0}" = 1 ] || { cd "$ROOT" && .venv/bin/rebirth track backfill | tail -15; }
