#!/usr/bin/env bash
# One-shot install of Rebirth AI on a new machine (Linux; Phase 19).
#
#   git clone https://github.com/hugobelebinda/rebirth-ai && cd rebirth-ai
#   bash scripts/bootstrap.sh                       # both packs
#   bash scripts/bootstrap.sh --pack modernize      # 7B same-language pack only (~5 GB, 8 GB GPU is enough)
#   bash scripts/bootstrap.sh --pack migrate        # 14B cross-language pack only (~12 GB, needs a 16 GB GPU)
#
# Options: --pack both|modernize|migrate · --from DIR (release files from a folder instead of Hugging
# Face) · --from-local (this repo's own build outputs) · --skip-images · --rebuild-images ·
# --skip-models · --no-embedder (use the hash retriever, no 330 MB embedding model).
#
# Steps: check tools → Python venv (.venv, CPU torch: the GPU is Ollama's) → embedding model → the 8
# sandbox images → model packs (rebirth install: download, sha256 check, ollama create) → rebirth
# doctor --strict. Re-running is safe: finished steps are skipped. Never uses sudo; prints what to
# install when a system tool is missing.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

pack=both from=() images=1 rebuild=0 models=1 embedder=1
while [ $# -gt 0 ]; do
  case "$1" in
    --pack) pack="${2:?--pack both|modernize|migrate}"; shift 2 ;;
    --from) from=(--from "${2:?--from DIR}"); shift 2 ;;
    --from-local) from=(--from-local); shift ;;
    --skip-images) images=0; shift ;;
    --rebuild-images) rebuild=1; shift ;;
    --skip-models) models=0; shift ;;
    --no-embedder) embedder=0; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (--help)" >&2; exit 2 ;;
  esac
done
case "$pack" in both|modernize|migrate) ;; *) echo "--pack both|modernize|migrate" >&2; exit 2 ;; esac

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

say "1/6 system tools"
command -v git >/dev/null || die "git missing (sudo apt install git)"
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || die "python3 missing (sudo apt install python3 python3-venv)"
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))' || die "Python ≥ 3.11 needed ($("$PY" -V))"
"$PY" -c 'import venv, ensurepip' 2>/dev/null || die "python venv module missing (sudo apt install python3-venv)"
if [ "$images" = 1 ]; then
  command -v docker >/dev/null || die "docker missing (https://docs.docker.com/engine/install/)"
  docker info >/dev/null 2>&1 || die "Docker daemon not reachable: start it, and add yourself to the docker group (sudo usermod -aG docker \$USER, then log in again)"
fi
if [ "$models" = 1 ]; then
  command -v ollama >/dev/null || die "ollama missing (https://ollama.com/download)"
  ollama list >/dev/null 2>&1 || die "Ollama server not reachable (ollama serve, or sudo systemctl start ollama)"
fi
echo "ok: git, $("$PY" -V)$([ "$images" = 1 ] && echo ", docker")$([ "$models" = 1 ] && echo ", ollama")"

say "2/6 Python environment (.venv)"
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  # CPU torch: only the small embedding model runs in Python; the LLMs run in Ollama.
  [ "${REBIRTH_TORCH:-cpu}" = cpu ] && .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/pip install -q -e .
.venv/bin/rebirth version

say "3/6 code-search embedding model"
if [ "$embedder" = 1 ]; then
  model="${REBIRTH_EMBEDDING_MODEL:-flax-sentence-embeddings/st-codesearch-distilroberta-base}"
  .venv/bin/python -c "from sentence_transformers import SentenceTransformer as S; S('$model')" >/dev/null
  echo "ok: $model cached (runs offline from now on)"
else
  echo "skipped: set REBIRTH_EMBEDDER=hash in .env"
fi
[ -f .env ] || { cp .env.example .env; echo "wrote .env from .env.example"; }

say "4/6 sandbox images (8 languages)"
if [ "$images" = 1 ]; then
  version="$(.venv/bin/python -c 'import rebirth_ai; print(rebirth_ai.__version__)')"
  for dir in docker/*_sandbox; do
    lang="$(basename "$dir" _sandbox)"; name="rebirth-ai/${lang//_/-}-sandbox"
    if [ "$rebuild" = 0 ] && docker image inspect "$name:latest" >/dev/null 2>&1; then
      echo "ok: $name:latest (exists)"
    else
      echo "building $name …"
      docker build -q -t "$name:latest" "$dir" >/dev/null
    fi
    docker tag "$name:latest" "$name:$version"
  done
else
  echo "skipped"
fi

say "5/6 model packs ($pack)"
if [ "$models" = 1 ]; then
  .venv/bin/rebirth install --pack "$pack" "${from[@]}"
else
  echo "skipped"
fi

say "6/6 rebirth doctor"
.venv/bin/rebirth doctor --strict
printf '\n\033[32mReady.\033[0m Try (bundled sample project):\n'
printf '  .venv/bin/rebirth migrate fixtures/python_sample --file src/math_ops.py --test tests/test_math_ops.py --lang python\n'
printf '  .venv/bin/rebirth migrate fixtures/python_sample --file src/math_ops.py --lang python --to go\n'
printf 'More: docs/INSTALL.md\n'
