#!/usr/bin/env bash
# Speculative decoding on Qwen3-8B-AWQ, vLLM 0.30.0. Arms alternated: none,
# draft model (Qwen3-0.6B, k=4), ngram prompt lookup (k=4). Spec-Bench prompts
# (downloaded to data/ on first use), greedy, 256 output tokens, prefix caching
# off in every arm. Predictions and results: FINDINGS.md.
# Needs Qwen3-0.6B next to the target: bash scripts/get-model.sh Qwen/Qwen3-0.6B
# Run from the repo root: bash experiments/speculative-decoding.sh
set -uo pipefail
source "$(dirname "$0")/../scripts/lib.sh"
O=results/speculative-decoding
mkdir -p "$O"
DRAFT='{"method":"draft_model","model":"/models/Qwen3-0.6B","num_speculative_tokens":4}'
NGRAM='{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
LV="1:48 4:96 16:192"
mark "=== speculative-decoding start"
# Spec-Bench prompts (Apache-2.0, https://github.com/hemingkx/Spec-Bench), fetched on first use
[ -f data/spec_bench_question.jsonl ] || { mkdir -p data && curl -sSfL -o data/spec_bench_question.jsonl \
  https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl; }

spec_counters() {  # cumulative: drafts, draft tokens, accepted tokens (0 0 0 without speculation)
  curl -s http://127.0.0.1:8000/metrics | awk '
    /^vllm:spec_decode_num_drafts_total\{/ {d += $2}
    /^vllm:spec_decode_num_draft_tokens_total\{/ {t += $2}
    /^vllm:spec_decode_num_accepted_tokens_total\{/ {a += $2}
    END {printf "%d %d %d\n", d, t, a}'
}

bench() {  # $1 out dir, $2 concurrency, $3 prompts
  .venv/bin/vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 \
    --model qwen3-8b-awq --tokenizer ${MODELS:-$HOME/models}/Qwen3-8B-AWQ \
    --dataset-name spec_bench --dataset-path data/spec_bench_question.jsonl \
    --spec-bench-output-len 256 --num-prompts "$3" --max-concurrency "$2" --temperature 0 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --save-detailed --result-dir "$1" --result-filename "c$2.json"
}

i=0
for arm in none draft ngram none draft ngram; do
  i=$((i + 1)); label=$arm-$i
  flags=(--no-enable-prefix-caching)
  case $arm in
    draft) flags+=(--speculative-config "$DRAFT") ;;
    ngram) flags+=(--speculative-config "$NGRAM") ;;
  esac
  mark "sd $label start"
  # V1 runner in every arm: vLLM 0.30.0's V2 runner doesn't support ngram or
  # draft_model, so those arms fall back to V1; the baseline must match.
  if DOCKER_ENV="VLLM_USE_V2_MODEL_RUNNER=0" DOCKER_MOUNTS="${MODELS:-$HOME/models}/Qwen3-0.6B:/models/Qwen3-0.6B:ro" \
       bash scripts/docker-vllm.sh start "sd-$label" bind "${flags[@]}" | tee -a "$TIMELINE"; then
    LEVELS="4:16" SEED_BASE=$((7000 + 100 * i)) bash scripts/sweep.sh "$O/warmup-$label" --temperature 0 \
      > "logs/sd-warmup-$label.log" 2>&1
    for pair in $LV; do
      c=${pair%%:*}; n=${pair##*:}
      read -r d0 t0 a0 < <(spec_counters)
      bench "$O/$label" "$c" "$n" > "logs/sd-$label-c$c.log" 2>&1
      read -r d1 t1 a1 < <(spec_counters)
      python3 -c "d=$d1-$d0; t=$t1-$t0; a=$a1-$a0; print(f'sd $label c=$c: drafts {d}, draft tokens {t}, accepted {a}, acceptance {(a/t if t else 0):.1%}, tokens per target step {(1 + a/d if d else 1):.2f}')" \
        | tee -a "$TIMELINE"
      grep -E 'Output token throughput|Median TPOT' "logs/sd-$label-c$c.log" | sed "s/^/  /" | tee -a "$TIMELINE"
    done
  fi
  bash scripts/docker-vllm.sh stop "sd-$label" | tee -a "$TIMELINE"
done
mark "=== speculative-decoding done"
