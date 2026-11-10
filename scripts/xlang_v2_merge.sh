#!/usr/bin/env bash
# Build rebirth-xlang:14b-v2: the Phase 17 cross-language LoRA merged into the bf16 14B, then q6_K.
# Same weights as v1 (q6_K base + f16 LoRA applied at run time) but merged before quantizing:
# ~24 % faster generation, ~45 % faster prompt reading, one 12 GB file (2026-11-10, 7800 XT).
#   bash scripts/xlang_v2_merge.sh [adapter dir] [tag]
# Disk: ~60 GB free at the peak (merged checkpoint 30 GB + bf16 GGUF 30 GB); the base download is
# deleted shard by shard while merging. RAM: one shard (~5 GB). Steps are skipped when their output
# exists, so a failed run can be restarted.
# Needs: .tools/llama.cpp (pinned checkout) with build/bin/llama-quantize —
#   python3 -m venv .tools/buildenv && .tools/buildenv/bin/pip install cmake ninja
#   PATH=.tools/buildenv/bin:$PATH cmake -S .tools/llama.cpp -B .tools/llama.cpp/build -G Ninja \
#     -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF && cmake --build .tools/llama.cpp/build --target llama-quantize
# and sentencepiece for the converter (kept out of .venv): pip install --target .tools/pyextra sentencepiece
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
adapter="${1:-data/models/lora_xlang_14b/adapter}"
tag="${2:-rebirth-xlang:14b-v2}"
W="${XLANG_WORK:-data/models/merge_xlang_v2}"
REV="${XLANG_BASE_REV:-aedcc2d42b}"   # Qwen/Qwen2.5-Coder-14B-Instruct revision used on 2026-11-10
PY=.venv/bin/python
Q6="$W/xlang_14b_v2.q6_K.gguf"
BF="$W/xlang_14b_v2.bf16.gguf"
mkdir -p "$W"

if [ ! -f "$Q6" ] && [ ! -f "$BF" ]; then
  if [ ! -f "$W/merged/rebirth_merge.json" ]; then
    HF_HUB_OFFLINE=0 "$PY" -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-Coder-14B-Instruct', revision='$REV', local_dir='$W/base')"
    HIP_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES="" "$PY" training/merge_lora.py \
      --base "$W/base" --adapter "$adapter" --out "$W/merged" --delete-base
    rm -rf "$W/base"
  fi
  PYTHONPATH=.tools/pyextra "$PY" .tools/llama.cpp/convert_hf_to_gguf.py "$W/merged" --outtype bf16 --outfile "$BF"
  cp "$W/merged/rebirth_merge.json" "$W/rebirth_merge.json"
  rm -rf "$W/merged"
fi
[ -f "$Q6" ] || .tools/llama.cpp/build/bin/llama-quantize "$BF" "$Q6" Q6_K "$(nproc)"

# Same chat template, system prompt and license as the q6_K base v1 was built on (and evaluated with).
dir="data/models/ollama/${tag//:/_}"
mkdir -p "$dir"
ollama show --modelfile qwen2.5-coder:14b-instruct-q6_K | sed -E "s#^FROM .*#FROM $ROOT/$Q6#" > "$dir/Modelfile"
ollama create "$tag" -f "$dir/Modelfile"
echo "ollama model $tag ready; gate it with scripts/xlang_v2_eval.sh before routing --to to it"
echo "the bf16 GGUF ($BF, 30 GB) can be deleted once other quantizations (e.g. Q4_K_M) are made"
