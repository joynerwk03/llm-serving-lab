#!/usr/bin/env bash
# One concurrency sweep against the running server.
# Workload: random prompts, 256 tokens in, 256 out (EOS ignored); change with
# INPUT_LEN, OUTPUT_LEN and PREFIX_LEN (a shared prefix, below).
#   SEED_BASE=100 bash scripts/sweep.sh <out_dir> [extra bench flags, e.g. --temperature 0]
#
# Seeds: every level gets its own seed (SEED_BASE + concurrency), and every sweep
# should use its own SEED_BASE. The random dataset produces the same prompts for
# the same seed, and vLLM's prefix cache (on by default) then skips their
# prefill, which flatters throughput and TTFT. Each level prints its prefix-cache
# hits so a repeat shows up.
#
# Sampling: since some version, `vllm bench serve` no longer requests greedy
# decoding; with no sampling flags the server's defaults apply (for Qwen3, from
# its generation_config.json: temperature 0.6, top-k 20, top-p 0.95).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_DIR=${MODEL_DIR:-${MODELS:-$HOME/models}/Qwen3-8B-AWQ}
NAME=${SERVED_NAME:-qwen3-8b-awq}
SEED_BASE=${SEED_BASE:-0}
# Point the same client at SGLang with ENGINE=sglang
# (BASE_URL defaults to its port). --backend vllm and openai run the same
# request function in vllm/benchmarks/lib/endpoint_request_func.py.
ENGINE=${ENGINE:-vllm}
if [ "$ENGINE" = sglang ]; then BASE_URL=${BASE_URL:-http://127.0.0.1:30000}; fi
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
BACKEND=${BACKEND:-vllm}
OUT=${1:-results/sweep}
shift || true
mkdir -p "$OUT"

counter() {  # cumulative prefix-cache hit / queried prompt tokens from /metrics
  local name v
  case "$ENGINE:$1" in
    vllm:hits)      name='vllm:prefix_cache_hits_total{' ;;
    vllm:queries)   name='vllm:prefix_cache_queries_total{' ;;
    sglang:hits)    name='sglang:cached_tokens_total{' ;;
    sglang:queries) name='sglang:prompt_tokens_total{' ;;
  esac
  v=$(curl -s "$BASE_URL/metrics" | grep -F "$name" | grep -v '^#' | awk '{s += $2} END {print s + 0}')
  echo "${v:-0}"
}

# concurrency:prompts, so each level runs at least four full waves
LEVELS=${LEVELS:-"1:16 4:32 16:64 64:256"}
# Prompt shape. PREFIX_LEN tokens are shared by every
# request at a level (one prefix per seed, so per level); INPUT_LEN are unique.
INPUT_LEN=${INPUT_LEN:-256}
OUTPUT_LEN=${OUTPUT_LEN:-256}
PREFIX_LEN=${PREFIX_LEN:-0}
for pair in $LEVELS; do
  c=${pair%%:*}
  n=${pair##*:}
  seed=$((SEED_BASE + c))
  h0=$(counter hits); q0=$(counter queries)
  echo "=== concurrency $c, $n prompts, seed $seed ==="
  .venv/bin/vllm bench serve \
    --backend "$BACKEND" --base-url "$BASE_URL" \
    --model "$NAME" --tokenizer "$MODEL_DIR" \
    --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --random-prefix-len "$PREFIX_LEN" --ignore-eos \
    --num-prompts "$n" --max-concurrency "$c" --seed "$seed" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$OUT" --result-filename "c${c}.json" \
    "$@"
  h1=$(counter hits); q1=$(counter queries)
  python3 -c "h=$h1-$h0; q=$q1-$q0; print(f'prefix cache this level: {h:.0f} hit tokens of {q:.0f} queried ({(h/q if q else 0):.1%})')"
done
