#!/usr/bin/env bash
# Download a model's weights to D: (the 4 TB data drive).
# The WSL disk and Docker's disk are both files on C:, which is ~94% full.
#   scripts/get-model.sh Qwen/Qwen3-8B-AWQ
set -euo pipefail

REPO=${1:-Qwen/Qwen3-8B-AWQ}
DEST=${MODELS:-$HOME/models}/$(basename "$REPO")
UV=${UV:-$HOME/.local/bin/uv}
mkdir -p "$DEST" ${HF_HOME:-$HOME/models/.hf-home}

# HF_HOME on D: keeps any hub caches off C:. The Xet chunk cache is turned off so
# the weights are not written twice.
HF_HOME=${HF_HOME:-$HOME/models/.hf-home} HF_XET_CHUNK_CACHE_SIZE_BYTES=0 UV_NO_CACHE=1 \
  "$UV" tool run --from huggingface_hub hf download "$REPO" --local-dir "$DEST"
du -sh "$DEST"
