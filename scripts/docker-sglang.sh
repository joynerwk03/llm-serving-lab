#!/usr/bin/env bash
# Run SGLang v0.5.20 in Docker on one GPU, on the inference-lab network (so
# Prometheus can scrape it as sglang:30000) and on 127.0.0.1:30000. Mounts the
# same weights folder as scripts/docker-vllm.sh (MODEL_DIR, default
# $MODELS/Qwen3-8B-AWQ).
#
#   bash scripts/docker-sglang.sh start <label> [extra sglang flags...]
#   bash scripts/docker-sglang.sh stop  <label>   # saves logs/docker-sglang-<label>.log
#
# MEM_FRACTION (default 0.80) is SGLang's --mem-fraction-static: the share of the
# GPU for weights plus the KV-cache pool. It is not the same knob as vLLM's
# --gpu-memory-utilization (which also covers activations and CUDA graphs), so
# compare the resulting KV capacity in tokens (vLLM: 84,352), not the flag values.
# --max-total-tokens caps SGLang's KV pool at vLLM's pinned 84,352 tokens, so
# both engines have identical capacity by construction.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${SGLANG_IMAGE:-lmsysorg/sglang:v0.5.20}
cmd=$1; label=$2; shift 2

if [ "$cmd" = stop ]; then
  mkdir -p logs
  docker logs sglang > "logs/docker-sglang-$label.log" 2>&1 || true
  docker stop -t 30 sglang >/dev/null 2>&1 || true
  docker rm -f sglang >/dev/null 2>&1 || true
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 5000 ] && break; sleep 1
  done
  echo "stopped sglang; log in logs/docker-sglang-$label.log; GPU memory ${used:-?} MiB"
  exit 0
fi

docker rm -f sglang >/dev/null 2>&1 || true
EXTRA=()   # DOCKER_MOUNTS / DOCKER_ENV: space-separated -v specs / VAR=value pairs
for m in ${DOCKER_MOUNTS:-}; do EXTRA+=(-v "$m"); done
for kv in ${DOCKER_ENV:-}; do EXTRA+=(-e "$kv"); done
start=$(date +%s)
docker run -d --name sglang --network inference-lab --gpus all --ipc=host \
  -p 127.0.0.1:30000:30000 -v "${MODEL_DIR:-${MODELS:-$HOME/models}/Qwen3-8B-AWQ}:/models/Qwen3-8B-AWQ:ro" \
  -v sglang-cache:/root/.cache -v sglang-triton:/root/.triton "${EXTRA[@]}" \
  "$IMAGE" python3 -m sglang.launch_server \
  --model-path /models/Qwen3-8B-AWQ --served-model-name qwen3-8b-awq \
  --host 0.0.0.0 --port 30000 --context-length 8192 \
  --mem-fraction-static "${MEM_FRACTION:-0.80}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS:-84352}" --enable-metrics "$@" >/dev/null

status=timeout
for _ in $(seq 1 600); do
  if curl -sf http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then status=ready; break; fi
  if [ "$(docker inspect -f '{{.State.Running}}' sglang 2>/dev/null)" != true ]; then status=exited; break; fi
  sleep 2
done
echo "sglang ($label): $status after $(( $(date +%s) - start )) s"
docker logs sglang 2>&1 | grep -iE "max_total_num_tokens|attention_backend|quantization|Load weight end|weight.*took|mem usage|cuda graph|available_gpu_mem|error|Traceback" \
  | cut -c1-220 | head -20
[ "$status" = ready ]
