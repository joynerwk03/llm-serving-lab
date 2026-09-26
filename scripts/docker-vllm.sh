#!/usr/bin/env bash
# Run vLLM 0.30.0 in Docker on one GPU (an RTX 3090 here), on the
# inference-lab network (so Prometheus can scrape it as vllm:8000) and on
# 127.0.0.1:8000.
#
#   bash scripts/docker-vllm.sh start <label> [bind|volume] [extra vllm flags...]
#   bash scripts/docker-vllm.sh stop  <label>     # saves logs/docker-vllm-<label>.log
#
# KV cache: pinned to 84,352 tokens (KV_BYTES; 144 KiB per token for Qwen3-8B)
# so every run has the same capacity; see scripts/lib.sh.
# Weights: "bind" (default) mounts $MODELS/<model> (MODELS defaults to
# ~/models); "volume" reads a named Docker volume, qwen3-8b-awq. On WSL2 both
# loaded in ~52 s here (vLLM's loader was the limit).
# Caches (torch.compile, FlashInfer, Triton) persist in volumes so later starts
# skip the compiles.
# VLLM_USE_FLASHINFER_SAMPLER, if set in the caller's environment, is passed in.
# Optional environment:
#   MODEL_DIR, SERVED_NAME  another model (bind only), e.g. ${MODELS:-$HOME/models}/Qwen3-8B
#   DOCKER_ENV              space-separated VAR=value pairs for the container
#   DOCKER_MOUNTS           space-separated extra -v specs (e.g. traces-vllm:/traces)
#   TRITON_VOL              the Triton cache volume (default vllm-triton)
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.30.0}
cmd=$1; label=$2; shift 2

if [ "$cmd" = stop ]; then
  mkdir -p logs
  docker logs vllm > "logs/docker-vllm-$label.log" 2>&1 || true
  docker stop -t 30 vllm >/dev/null 2>&1 || true
  docker rm -f vllm >/dev/null 2>&1 || true
  for _ in $(seq 1 60); do  # wait for the GPU memory to come back
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 5000 ] && break; sleep 1
  done
  echo "stopped vllm; log in logs/docker-vllm-$label.log; GPU memory ${used:-?} MiB"
  exit 0
fi

src=${1:-bind}; shift || true
MODEL_DIR=${MODEL_DIR:-${MODELS:-$HOME/models}/Qwen3-8B-AWQ}
MNAME=$(basename "$MODEL_DIR")
case "$src" in
  volume) MOUNT=(-v qwen3-8b-awq:/models/Qwen3-8B-AWQ:ro); MNAME=Qwen3-8B-AWQ ;;
  bind)   MOUNT=(-v "$MODEL_DIR:/models/$MNAME:ro") ;;
  *) echo "weights source must be volume or bind"; exit 1 ;;
esac
for m in ${DOCKER_MOUNTS:-}; do MOUNT+=(-v "$m"); done
ENVV=()
[ -n "${VLLM_USE_FLASHINFER_SAMPLER:-}" ] && ENVV+=(-e "VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER")
for kv in ${DOCKER_ENV:-}; do ENVV+=(-e "$kv"); done

docker rm -f vllm >/dev/null 2>&1 || true
start=$(date +%s)
docker run -d --name vllm --network inference-lab --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 "${MOUNT[@]}" \
  -v vllm-cache:/root/.cache -v "${TRITON_VOL:-vllm-triton}:/root/.triton" "${ENVV[@]}" \
  "$IMAGE" "/models/$MNAME" \
  --served-model-name "${SERVED_NAME:-qwen3-8b-awq}" --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.80 --max-model-len 8192 \
  --kv-cache-memory-bytes "${KV_BYTES:-12438208512}" \
  --safetensors-load-strategy eager "$@" >/dev/null

status=timeout
for _ in $(seq 1 600); do
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then status=ready; break; fi
  if [ "$(docker inspect -f '{{.State.Running}}' vllm 2>/dev/null)" != true ]; then status=exited; break; fi
  sleep 2
done
echo "vllm ($label, weights: $src): $status after $(( $(date +%s) - start )) s"
docker logs vllm 2>&1 | grep -E "Loading weights took|Model loading took|KV cache|maximum concurrency|Using .*backend|Using FlashInfer|sampling unavailable|Pinned memory|torch.compile takes|Graph capturing finished" \
  | sed -E 's/^\([A-Za-z]+ pid=[0-9]+\) //; s/^(INFO|WARNING) [0-9-]+ [0-9:]+ /\1 /' | cut -c1-200
[ "$status" = ready ]
