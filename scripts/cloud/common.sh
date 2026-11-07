#!/usr/bin/env bash
# Shared settings for scripts/cloud/* (run from the home PC). Host = $REBIRTH_GPU_HOST or .cloud_host
# (the instance's public IP, written by you or by `terraform output -raw instance_ip > .cloud_host`).
# DRY_RUN=1 prints every remote/rsync command instead of running it.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEY="${REBIRTH_GPU_KEY:-$HOME/.ssh/rebirth-gpu}"
RUSER="${REBIRTH_GPU_USER:-ubuntu}"
RDIR="${REBIRTH_GPU_DIR:-rebirth}"            # relative to the remote home
HOST="${REBIRTH_GPU_HOST:-$(cat "$ROOT/.cloud_host" 2>/dev/null || true)}"
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=15)
need_host() { [ -n "$HOST" ] || { echo "no instance host: export REBIRTH_GPU_HOST=<ip> or write it to .cloud_host" >&2; exit 2; }; }
run() { if [ "${DRY_RUN:-0}" = 1 ]; then printf '[dry-run]'; printf ' %q' "$@"; echo; else "$@"; fi; }
remote() { need_host; run ssh "${SSH_OPTS[@]}" "$RUSER@$HOST" "cd ~/$RDIR 2>/dev/null || true; $*"; }
rsync_up() { need_host; run rsync -az --info=stats1 -e "ssh ${SSH_OPTS[*]}" "$@"; }
