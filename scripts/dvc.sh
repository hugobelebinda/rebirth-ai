#!/usr/bin/env bash
# DVC lives in its own venv (.venv-tools) — full DVC would downgrade python-dateutil in .venv.
#   bash scripts/dvc.sh status | repro | push | pull …
# A bare `push` / `pull` / `status -c` names every *.dvc file explicitly: plain DVC skips the
# adapter pointers under data/models/ (that tree is git-ignored except for the pointers).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -x .venv-tools/bin/dvc ] || { echo "missing .venv-tools: python3 -m venv .venv-tools && .venv-tools/bin/pip install 'dvc[s3]'" >&2; exit 2; }
all_targets() { find data -name '*.dvc' -not -path '*/runs/*' | sort; echo dvc.yaml; }
case "$*" in
  push|pull|"status -c") mapfile -t t < <(all_targets); exec .venv-tools/bin/dvc "$@" "${t[@]}" ;;
esac
exec .venv-tools/bin/dvc "$@"
