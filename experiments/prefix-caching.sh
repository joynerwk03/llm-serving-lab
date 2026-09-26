#!/usr/bin/env bash
# Prefix caching on vs off, vLLM then SGLang, arms alternated. Workloads:
# S = a 2,000-token prefix shared by every request + 200 unique tokens;
# U = 2,200 unique tokens (the control); 128 output tokens, greedy.
# Predictions and results: FINDINGS.md, "Prefix caching".
# Run from the repo root: bash experiments/prefix-caching.sh
set -uo pipefail
source "$(dirname "$0")/../scripts/lib.sh"
O=results/prefix-caching
mkdir -p "$O"
LV="1:16 4:32 16:96"
mark "=== prefix-caching start"

run_workloads() {  # $1 label, $2 seed base; ENGINE and BACKEND come from the caller
  local label=$1 sb=$2
  # discarded warm-up, on its own seed so it can't seed the measured prefixes
  LEVELS="4:16" SEED_BASE=$((sb + 90)) INPUT_LEN=200 OUTPUT_LEN=128 PREFIX_LEN=2000 \
    bash scripts/sweep.sh "$O/warmup-$label" --temperature 0 > "logs/pc-warmup-$label.log" 2>&1
  mark "$label S"
  LEVELS="$LV" SEED_BASE=$sb INPUT_LEN=200 OUTPUT_LEN=128 PREFIX_LEN=2000 \
    bash scripts/sweep.sh "$O/S-$label" --temperature 0 > "logs/pc-S-$label.log" 2>&1
  grep -h 'prefix cache this level' "logs/pc-S-$label.log" | sed "s/^/  S $label: /" | tee -a "$TIMELINE"
  mark "$label U"
  LEVELS="$LV" SEED_BASE=$((sb + 50)) INPUT_LEN=2200 OUTPUT_LEN=128 PREFIX_LEN=0 \
    bash scripts/sweep.sh "$O/U-$label" --temperature 0 > "logs/pc-U-$label.log" 2>&1
  grep -h 'prefix cache this level' "logs/pc-U-$label.log" | sed "s/^/  U $label: /" | tee -a "$TIMELINE"
}

i=0
for arm in off on off on; do
  i=$((i + 1)); label=vllm-$arm-$i
  flags=(); [ "$arm" = off ] && flags=(--no-enable-prefix-caching)
  mark "$label start"
  if bash scripts/docker-vllm.sh start "pc-$label" bind "${flags[@]}" | tee -a "$TIMELINE"; then
    ENGINE=vllm BACKEND=vllm run_workloads "$label" $((6000 + 100 * i))
  fi
  bash scripts/docker-vllm.sh stop "pc-$label" | tee -a "$TIMELINE"
done

for arm in off on off on; do
  i=$((i + 1)); label=sglang-$arm-$i
  flags=(); [ "$arm" = off ] && flags=(--disable-radix-cache)
  mark "$label start"
  if bash scripts/docker-sglang.sh start "pc-$label" "${flags[@]}" 2>&1 | tail -3 | tee -a "$TIMELINE" \
     && curl -sf http://127.0.0.1:30000/v1/models > /dev/null; then
    ENGINE=sglang BACKEND=openai run_workloads "$label" $((6000 + 100 * i))
  fi
  bash scripts/docker-sglang.sh stop "pc-$label" | tee -a "$TIMELINE"
done
mark "=== prefix-caching done"
