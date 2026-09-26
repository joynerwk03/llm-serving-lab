#!/usr/bin/env bash
# What does 4-bit cost? Qwen3-8B AWQ vs BF16 on GSM8K (lm-evaluation-harness
# 0.4.13, 5-shot, greedy), same server settings, same client, then a paired
# (McNemar) test. Also a c=1 sweep level for each model's speed.
# Needs both models: scripts/get-model.sh Qwen/Qwen3-8B-AWQ and Qwen/Qwen3-8B.
# Run from the repo root: bash experiments/quantization-gsm8k.sh
set -uo pipefail
source "$(dirname "$0")/../scripts/lib.sh"
E=results/quantization
mkdir -p "$E"
UV=$HOME/.local/bin/uv
mark "=== quantization start"
if [ ! -x .venv-eval/bin/lm_eval ]; then
  "$UV" venv -q .venv-eval --python 3.12 && \
  UV_NO_CACHE=1 "$UV" pip install -q --python .venv-eval/bin/python "lm-eval[api]==0.4.13" transformers
fi
export HF_HOME=${HF_HOME:-$HOME/models/.hf-home}   # datasets cache
# identical KV for both models: 21,840 tokens (3.0 GiB); BF16 weights leave ~4 GB
export KV_BYTES=3220439040

for spec in "awq:${MODELS:-$HOME/models}/Qwen3-8B-AWQ:qwen3-8b-awq" "bf16:${MODELS:-$HOME/models}/Qwen3-8B:qwen3-8b"; do
  IFS=: read -r tag dir name <<< "$spec"
  # ONLY=awq or ONLY=bf16 runs a single arm
  [ -n "${ONLY:-}" ] && [ "$tag" != "$ONLY" ] && continue
  mark "eval-$tag start"
  MODEL_DIR=$dir SERVED_NAME=$name bash scripts/docker-vllm.sh start "eval-$tag" bind --max-model-len 4096 \
    | tee -a "$TIMELINE" || { bash scripts/docker-vllm.sh stop "eval-$tag"; continue; }
  mark "eval-$tag c=1 speed"
  MODEL_DIR=$dir SERVED_NAME=$name LEVELS="1:8" SEED_BASE=3900 \
    bash scripts/sweep.sh "$E/tput-$tag" --temperature 0 > "logs/eval-tput-$tag.log" 2>&1
  mark "eval-$tag gsm8k"
  .venv-eval/bin/lm_eval --model local-completions \
    --model_args "model=$name,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False,tokenizer=$dir" \
    --tasks gsm8k --output_path "$E/gsm8k-$tag" --log_samples > "logs/eval-gsm8k-$tag.log" 2>&1
  tail -8 "logs/eval-gsm8k-$tag.log" | tee -a "$TIMELINE"
  bash scripts/docker-vllm.sh stop "eval-$tag" | tee -a "$TIMELINE"
done
.venv/bin/python scripts/mcnemar.py "$E/gsm8k-awq" "$E/gsm8k-bf16" | tee "$E/paired.txt" | tee -a "$TIMELINE"
mark "=== quantization done"
