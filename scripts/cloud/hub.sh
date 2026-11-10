# Sourced by the cloud session scripts (ON the instance): Hugging Face hub cache helpers. vLLM reads
# the same cache from docker. fetch() also needs the caller's note() and $D (log dir).
HUB="${HUB:-$HOME/.cache/huggingface/hub}"

tag() { echo "${1##*/}" | tr 'A-Z.' 'a-z_'; }
cached() { [ -d "$HUB/models--${1//\//--}/snapshots" ]; }
free_gb() { df -BG --output=avail / | tail -1 | tr -dc 0-9; }
extra_for() { case $1 in *Qwen3.6*|*Qwen3.5*) echo '--limit-mm-per-prompt {"image":0,"video":0}' ;; *) echo "" ;; esac; }

drop() {  # drop <model>: delete it from the cache, weights included
  rm -rf "$HUB/models--${1//\//--}"
  # Newer huggingface_hub keeps the weights in one shared store (hub/blobs/<xx>/<sha256>) that the
  # repo dirs only link to, so removing the repo dir frees nothing: also delete every stored file no
  # remaining repo links to (never call this during a download: its files are not linked yet).
  [ -d "$HUB/blobs" ] || return 0
  LC_ALL=C comm -23 <(find "$HUB/blobs" -mindepth 2 -type f | LC_ALL=C sort) \
    <(find "$HUB"/models--* -type l -print0 2>/dev/null | xargs -0 -r readlink -f | LC_ALL=C sort -u) \
    | xargs -r -d '\n' rm -f
}

fetch() {  # fetch <model>: download from the Hub if not cached (an FP8 ~30B needs ~35 GB free)
  cached "$1" && return 0
  [ "$(free_gb)" -ge 36 ] || { note "\"event\":\"no_disk\",\"model\":\"$1\",\"free_gb\":$(free_gb)"; return 1; }
  note "\"event\":\"download_start\",\"model\":\"$1\""
  HF_HUB_OFFLINE=0 python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])" "$1" \
    > "$D/download_$(tag "$1").log" 2>&1
  local rc=$?; note "\"event\":\"download_end\",\"model\":\"$1\",\"exit\":$rc,\"free_gb\":$(free_gb)"; return $rc
}
