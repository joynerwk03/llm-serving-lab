#!/usr/bin/env bash
# Open-loop load: requests arrive at random (Poisson) times at a fixed average
# rate, whether or not earlier ones have finished. The concurrency sweep is
# closed-loop (a fixed number of clients, each waiting for its answer), which
# produces synchronized prefill waves (see FINDINGS.md).
#   RATES="1 2 3 4 5 6" N=200 SEED_BASE=900 bash scripts/poisson.sh <out_dir> [extra bench flags]
# Same workload as sweep.sh: random prompts, 256 in / 256 out, EOS ignored.
# Each rate gets its own seed (SEED_BASE + 100 + rate) so no prompt repeats.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_DIR=${MODEL_DIR:-${MODELS:-$HOME/models}/Qwen3-8B-AWQ}
NAME=${SERVED_NAME:-qwen3-8b-awq}
ENGINE=${ENGINE:-vllm}
if [ "$ENGINE" = sglang ]; then BASE_URL=${BASE_URL:-http://127.0.0.1:30000}; fi
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
SEED_BASE=${SEED_BASE:-900}
RATES=${RATES:-"1 2 3 4 5 6"}
N=${N:-200}
OUT=${1:?usage: poisson.sh <out_dir> [extra bench flags]}
shift
mkdir -p "$OUT"

for r in $RATES; do
  seed=$((SEED_BASE + 100 + r))
  echo "=== $r req/s (Poisson), $N prompts, seed $seed, $(date -u +%T) UTC ==="
  .venv/bin/vllm bench serve \
    --backend openai --base-url "$BASE_URL" \
    --model "$NAME" --tokenizer "$MODEL_DIR" \
    --dataset-name random --random-input-len 256 --random-output-len 256 --ignore-eos \
    --num-prompts "$N" --request-rate "$r" --burstiness 1.0 --seed "$seed" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$OUT" --result-filename "r${r}.json" \
    "$@"
done
