#!/usr/bin/env bash
# Copy the repo (+ datasets and 7B adapters unless SYNC_DATA=0) to the instance.
#   bash scripts/cloud/sync_up.sh            SYNC_DATA=0 → code only (then `dvc pull` on the instance)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
excludes=(--exclude .venv/ --exclude .venv-mlflow/ --exclude .venv-tools/ --exclude .tools/ --exclude mlruns/
          --exclude logs/ --exclude data/cache/ --exclude .dvc/cache/ --exclude graphify-out/ --exclude 'runs/'
          --exclude infra/aws/.terraform/ --exclude '*.tfstate*' --exclude __pycache__/ --exclude .cloud_host)
[ "${SYNC_DATA:-1}" = 1 ] || excludes+=(--exclude data/datasets/expanded/ --exclude 'data/models/*/adapter/')
remote "mkdir -p ~/$RDIR"
rsync_up "${excludes[@]}" "$ROOT/" "$RUSER@$HOST:~/$RDIR/"
