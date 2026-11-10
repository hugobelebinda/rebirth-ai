#!/usr/bin/env bash
# Copy the repo (+ datasets and 7B adapters unless SYNC_DATA=0) to the instance.
#   bash scripts/cloud/sync_up.sh            SYNC_DATA=0 → code only (then `dvc pull` on the instance)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
excludes=(--exclude .venv/ --exclude .venv-mlflow/ --exclude .venv-tools/ --exclude .tools/ --exclude mlruns/
          --exclude logs/ --exclude data/cache/ --exclude .dvc/cache/ --exclude graphify-out/ --exclude 'runs/'
          --exclude infra/aws/.terraform/ --exclude '*.tfstate*' --exclude __pycache__/ --exclude .cloud_host
          --exclude data/datasets/pool/ --exclude data/models/gguf/ --exclude data/models/ollama/
          --exclude data/models/exp/ --exclude data/datasets/xlang/stage3/)
# exp/ = local experiment checkpoints (~4 GB, never read on the cloud); stage3/ = Phase 17 working files.
# Training reads data/datasets/p15 (Phase 15 corpora); the 0.6 GB mining pool stays home.
[ "${SYNC_DATA:-1}" = 1 ] || excludes+=(--exclude data/datasets/expanded/ --exclude data/datasets/p15/
                                        --exclude 'data/models/*/adapter/')
remote "mkdir -p ~/$RDIR"
rsync_up "${excludes[@]}" "$ROOT/" "$RUSER@$HOST:~/$RDIR/"
