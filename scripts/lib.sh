#!/usr/bin/env bash
# Shared helpers for the experiment runners (source it, don't run it).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs results
TIMELINE=${TIMELINE:-logs/timeline.txt}

mark() { echo "$(date -u +%FT%TZ) $*" | tee -a "$TIMELINE"; }

gpu_logger_start() {  # one CSV per session: clocks, power, utilization, memory
  pgrep -f "[n]vidia-smi --query-gpu=timestamp" >/dev/null && return 0
  nohup nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,power.draw,temperature.gpu,memory.used \
    --format=csv,noheader -l 1 >> logs/gpu.csv 2>/dev/null &
  mark "gpu logger started"
}

drop_caches() {  # empty the WSL2 VM's page cache (all distros share its kernel)
  sync
  docker run --rm --privileged alpine:3 sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' \
    && mark "page cache dropped"
}

wait_gpu_free() {
  local used
  for _ in $(seq 1 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 5000 ] && return 0; sleep 1
  done
  echo "GPU memory still at ${used} MiB"; return 1
}

# Every vLLM run pins its KV cache to the same size (84,352 tokens x 144 KiB
# for Qwen3-8B), so all arms of a comparison get identical capacity. Left
# automatic, vLLM's sizing moved between days on this machine, and a larger
# allocation once failed under WSL2.
export KV_BYTES=${KV_BYTES:-12438208512}
