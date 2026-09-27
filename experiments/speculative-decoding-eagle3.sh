#!/usr/bin/env bash
# Can speculation win on this machine? An EAGLE-3 head (1 layer, k=3) and vLLM's
# GPU prompt lookup (ngram_gpu, k=4) against no speculation, on the workload of
# experiments/speculative-decoding.sh (Spec-Bench, greedy, 256 tokens, prefix
# caching off, V1 runner). Arms alternated twice. Results: FINDINGS.md, section 9.
# Needs the head next to the target:
#   bash scripts/get-model.sh RedHatAI/Qwen3-8B-speculator.eagle3
# and data/spec_bench_question.jsonl (experiments/speculative-decoding.sh fetches it).
# Run from the repo root: bash experiments/speculative-decoding-eagle3.sh
set -uo pipefail
source "$(dirname "$0")/../scripts/lib.sh"
O=results/speculative-decoding-eagle3
mkdir -p "$O"
NGPU='{"method":"ngram_gpu","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
EAGLE='{"method":"eagle3","model":"/models/Qwen3-8B-speculator.eagle3","num_speculative_tokens":3}'
LV="1:48 4:96 16:192"
mark "=== speculative-decoding-eagle3 start"

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
for arm in none ngpu eagle none ngpu eagle; do
  i=$((i + 1)); label=$arm-$i
  flags=(--no-enable-prefix-caching)
  case $arm in
    ngpu)  flags+=(--speculative-config "$NGPU") ;;
    eagle) flags+=(--speculative-config "$EAGLE") ;;
  esac
  mark "sd-eagle3 $label start"
  # V1 runner in every arm, as in experiments/speculative-decoding.sh
  if DOCKER_ENV="VLLM_USE_V2_MODEL_RUNNER=0" \
     DOCKER_MOUNTS="${MODELS:-$HOME/models}/Qwen3-8B-speculator.eagle3:/models/Qwen3-8B-speculator.eagle3:ro" \
       bash scripts/docker-vllm.sh start "sd-eagle3-$label" bind "${flags[@]}" | tee -a "$TIMELINE"; then
    LEVELS="4:16" SEED_BASE=$((8000 + 100 * i)) bash scripts/sweep.sh "$O/warmup-$label" --temperature 0 \
      > "logs/sd-eagle3-warmup-$label.log" 2>&1
    for pair in $LV; do
      c=${pair%%:*}; n=${pair##*:}
      read -r d0 t0 a0 < <(spec_counters)
      bench "$O/$label" "$c" "$n" > "logs/sd-eagle3-$label-c$c.log" 2>&1
      read -r d1 t1 a1 < <(spec_counters)
      python3 -c "d=$d1-$d0; t=$t1-$t0; a=$a1-$a0; print(f'$label c=$c: drafts {d}, draft tokens {t}, accepted {a}, acceptance {(a/t if t else 0):.1%}')" \
        | tee -a "$TIMELINE"
      grep -E 'Output token throughput|Median TPOT' "logs/sd-eagle3-$label-c$c.log" | sed "s/^/  /" | tee -a "$TIMELINE"
    done
  fi
  bash scripts/docker-vllm.sh stop "sd-eagle3-$label" | tee -a "$TIMELINE"
done
mark "=== speculative-decoding-eagle3 done"
