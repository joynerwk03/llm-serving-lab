#!/usr/bin/env bash
# Pinned memory on WSL2: does VLLM_WSL2_ENABLE_PIN_MEMORY=1 remove vLLM's
# ~1.3 ms of idle GPU per decode step? Arms alternate off/on/off/on; each arm
# gets 3 timed batch-1 requests plus a full greedy sweep. Then one torch
# profile of the "on" arm. Results: FINDINGS.md, "Pinned memory".
# Run from the repo root: bash experiments/pinned-memory.sh
set -uo pipefail
source "$(dirname "$0")/../scripts/lib.sh"
O=results/pinned-memory
mkdir -p "$O"
mark "=== pinned-memory start"
i=0
for arm in off on off on; do
  i=$((i + 1)); label=pin-$arm-$i
  env_=""; [ "$arm" = on ] && env_="VLLM_WSL2_ENABLE_PIN_MEMORY=1"
  mark "$label start"
  DOCKER_ENV="$env_" bash scripts/docker-vllm.sh start "$label" bind | tee -a "$TIMELINE"
  docker logs vllm 2>&1 | grep -i -E "pin.memory|Pinned memory" | head -2 | sed -E 's/^\([A-Za-z]+ pid=[0-9]+\) //' | cut -c1-160 | tee -a "$TIMELINE"
  .venv/bin/python - "$O/$label-timing.json" <<'PY'
import json, statistics, sys, time, urllib.request
P = ("The history of computing is a story of trading one scarce resource for "
     "another: memory for time, time for energy, energy for money. ") * 9
B = {"model": "qwen3-8b-awq", "prompt": P, "max_tokens": 64, "temperature": 0,
     "ignore_eos": True, "stream": True}
def one():
    st = []
    req = urllib.request.Request("http://127.0.0.1:8000/v1/completions", data=json.dumps(B).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            l = raw.decode().strip()
            if l.startswith("data:") and l != "data: [DONE]" and json.loads(l[5:])["choices"][0].get("text"):
                st.append(time.perf_counter())
    return statistics.median(b - a for a, b in zip(st, st[1:])) * 1000
t = [one() for _ in range(3)]
json.dump({"tpot_ms": t, "median": statistics.median(t)}, open(sys.argv[1], "w"))
print("batch-1 TPOT ms:", [round(x, 2) for x in t])
PY
  SEED_BASE=$((4000 + 100 * i)) bash scripts/sweep.sh "$O/$label" --temperature 0 > "logs/$label.sweep.log" 2>&1
  grep "prefix cache" "logs/$label.sweep.log" | tee -a "$TIMELINE"
  bash scripts/docker-vllm.sh stop "$label" | tee -a "$TIMELINE"
done

# one profile with pinned memory on
docker volume rm -f traces-pin >/dev/null 2>&1
DOCKER_ENV="VLLM_WSL2_ENABLE_PIN_MEMORY=1" DOCKER_MOUNTS="traces-pin:/traces" \
  bash scripts/docker-vllm.sh start prof-pin bind \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/traces","torch_profiler_with_stack":false,"active_iterations":40}' \
  | tee -a "$TIMELINE"
.venv/bin/python scripts/profile_decode.py vllm http://127.0.0.1:8000 > "$O/prof-pin-client.json" 2>&1
sleep 15
bash scripts/docker-vllm.sh stop prof-pin | tee -a "$TIMELINE"
mkdir -p logs/traces/pinned-on
docker run --rm -v traces-pin:/t -v "$PWD/logs/traces/pinned-on":/out alpine:3 sh -c "cp -r /t/. /out/ && chown -R $(id -u):$(id -g) /out"
mark "=== pinned-memory done"
