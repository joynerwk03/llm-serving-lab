#!/usr/bin/env bash
# Where a batch-1 speculative-decoding step's time goes (FINDINGS.md, section
# 9). vLLM 0.30.0's V1 runner: no speculation, prompt lookup (k=4), draft model
# (Qwen3-0.6B, k=4), each with the WSL2 pinned-memory default and then with it
# on; then the V2 runner's default, for comparison. Per arm: a warm-up request,
# 3 timed, 1 profiled with the torch profiler; then the trace is broken down
# per step by scripts/analyze_sd_trace.py.
# Needs data/spec_bench_question.jsonl (experiments/speculative-decoding.sh
# downloads it) and Qwen3-0.6B next to the target.
# Run from the repo root: bash experiments/speculative-decoding-profile.sh
set -uo pipefail
source "$(dirname "$0")/../scripts/lib.sh"
O=results/speculative-decoding-profile
mkdir -p "$O" logs/traces
DRAFT='{"method":"draft_model","model":"/models/Qwen3-0.6B","num_speculative_tokens":4}'
NGRAM='{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
S=VLLM_CUSTOM_SCOPES_FOR_PROFILING=1   # the runner's phase labels in the trace
PROF='{"profiler":"torch","torch_profiler_dir":"/traces","torch_profiler_with_stack":false,"active_iterations":40}'
mark "=== speculative-decoding-profile start"

copy_traces() {  # $1 volume, $2 dest
  mkdir -p "$2"
  docker run --rm -v "$1":/t -v "$PWD/$2":/out alpine:3 \
    sh -c "cp -r /t/. /out/ && chown -R $(id -u):$(id -g) /out"
}

for arm in none none+pin ngram ngram+pin draft draft+pin v2-none; do
  label=${arm/+/-}
  case $arm in
    v2-none) env_="$S" ;;   # the default runner, pin off
    *+pin)   env_="$S VLLM_USE_V2_MODEL_RUNNER=0 VLLM_WSL2_ENABLE_PIN_MEMORY=1" ;;
    *)       env_="$S VLLM_USE_V2_MODEL_RUNNER=0" ;;
  esac
  flags=(--no-enable-prefix-caching --profiler-config "$PROF")
  case $arm in
    draft*) flags+=(--speculative-config "$DRAFT") ;;
    ngram*) flags+=(--speculative-config "$NGRAM") ;;
  esac
  mark "sd-profile $label start"
  docker volume rm -f traces-sd >/dev/null 2>&1
  if DOCKER_ENV="$env_" DOCKER_MOUNTS="traces-sd:/traces ${MODELS:-$HOME/models}/Qwen3-0.6B:/models/Qwen3-0.6B:ro" \
       bash scripts/docker-vllm.sh start "sd-profile-$label" bind "${flags[@]}" | tee -a "$TIMELINE"; then
    .venv/bin/python scripts/profile_sd.py http://127.0.0.1:8000 > "$O/$label.json" 2> "logs/sd-profile-$label.err"
    sleep 20   # let the trace flush
  fi
  bash scripts/docker-vllm.sh stop "sd-profile-$label" | tee -a "$TIMELINE"
  copy_traces traces-sd "logs/traces/sd-profile-$label"
  ms=$(python3 -c "import json; print(json.load(open('$O/$label.json'))['median_ms_per_step'])")
  python3 scripts/analyze_sd_trace.py logs/traces/sd-profile-$label/rank0.*.pt.trace.json.gz \
    --ms-per-step "$ms" --json "$O/$label-trace.json" | tee -a "$TIMELINE"
done
docker volume rm -f traces-sd >/dev/null 2>&1
mark "=== speculative-decoding-profile done"
