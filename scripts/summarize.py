#!/usr/bin/env python3
"""Turn `vllm bench serve --save-result` JSON files into one markdown table.

    python3 scripts/summarize.py results/sweep     # concurrency sweep
    python3 scripts/summarize.py results/poisson   # Poisson (request-rate) runs

"Out/req" is output tokens per completed request: 256 means EOS really was
ignored, which matters when the same client drives a different engine.
For Poisson runs, "L = λW" is Little's law: achieved req/s × mean end-to-end
latency = the average number of requests in the system.
"""
import json
import math
import pathlib
import sys

folder = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/sweep")
rows = [json.loads(f.read_text()) for f in folder.glob("*.json")]
if not rows:
    sys.exit(f"no results in {folder}")


def rate(d):
    r = d.get("request_rate")
    try:
        r = float(r)
    except (TypeError, ValueError):
        return math.inf
    return r


def g(d, k, fmt="{:.0f}"):
    v = d.get(k)
    return "n/a" if v is None else fmt.format(v)


def out_per_req(d):
    done = d.get("completed") or 0
    return f"{(d.get('total_output_tokens') or 0) / done:.0f}" if done else "n/a"


poisson = all(math.isfinite(rate(d)) for d in rows)
if poisson:
    rows.sort(key=rate)
    print("| Offered req/s | Achieved req/s | Output tok/s | TTFT median / p90 / p99 (ms) "
          "| TPOT median (ms) | E2E median (s) | L = λW | Out/req | Failed |")
    print("|---|---|---|---|---|---|---|---|---|")
    for d in rows:
        lam = d.get("request_throughput") or 0
        w = (d.get("mean_e2el_ms") or 0) / 1000
        print(f"| {rate(d):g} | {lam:.2f} | {g(d, 'output_throughput')} "
              f"| {g(d, 'median_ttft_ms')} / {g(d, 'p90_ttft_ms')} / {g(d, 'p99_ttft_ms')} "
              f"| {g(d, 'median_tpot_ms', '{:.1f}')} "
              f"| {(d.get('median_e2el_ms') or 0) / 1000:.2f} | {lam * w:.1f} "
              f"| {out_per_req(d)} | {d.get('failed', 0)} |")
    sys.exit(0)

rows.sort(key=lambda d: d.get("max_concurrency") or 0)
base = rows[0].get("output_throughput") or 0
print("| Concurrency | Prompts | Req/s | Output tok/s | vs c=1 | TTFT median / p99 (ms) "
      "| TPOT median / p99 (ms) | Out/req | Failed |")
print("|---|---|---|---|---|---|---|---|---|")
for d in rows:
    ratio = (d.get("output_throughput") or 0) / base if base else 0
    print(f"| {d.get('max_concurrency')} | {d.get('num_prompts', d.get('completed'))} "
          f"| {g(d, 'request_throughput', '{:.2f}')} | {g(d, 'output_throughput')} "
          f"| {ratio:.1f}x "
          f"| {g(d, 'median_ttft_ms')} / {g(d, 'p99_ttft_ms')} "
          f"| {g(d, 'median_tpot_ms', '{:.1f}')} / {g(d, 'p99_tpot_ms', '{:.1f}')} "
          f"| {out_per_req(d)} | {d.get('failed', 0)} |")
