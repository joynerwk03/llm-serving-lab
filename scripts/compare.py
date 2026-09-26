#!/usr/bin/env python3
"""Compare sweep results across arms, showing every replicate (not just means).

    python3 scripts/compare.py off=results/off-1,results/off-3 on=results/on-2,results/on-4

For each concurrency level: output tok/s, median TPOT and median TTFT, one
value per replicate, then each arm's mean and its difference from the first
arm's mean. A difference smaller than the spread between replicates isn't a
difference.
"""
import json
import pathlib
import sys
from statistics import mean

arms = []
for spec in sys.argv[1:]:
    name, dirs = spec.split("=", 1)
    runs = []
    for d in dirs.split(","):
        runs.append({json.loads(f.read_text()).get("max_concurrency"): json.loads(f.read_text())
                     for f in pathlib.Path(d).glob("c*.json")})
    arms.append((name, runs))
if not arms:
    sys.exit(__doc__)

levels = sorted({c for _, runs in arms for r in runs for c in r})
METRICS = [("output_throughput", "Output tok/s", "{:.0f}"),
           ("median_tpot_ms", "TPOT median (ms)", "{:.1f}"),
           ("median_ttft_ms", "TTFT median (ms)", "{:.0f}")]

for key, title, fmt in METRICS:
    print(f"\n**{title}**\n")
    print("| c | " + " | ".join(f"{n} (each run) | {n} mean" for n, _ in arms) + " | Δ vs first |")
    print("|---" * (2 * len(arms) + 2) + "|")
    for c in levels:
        cells, means = [], []
        for _, runs in arms:
            vals = [r[c][key] for r in runs if c in r and r[c].get(key) is not None]
            m = mean(vals) if vals else None
            means.append(m)
            cells.append(", ".join(fmt.format(v) for v in vals) or "n/a")
            cells.append(fmt.format(m) if m is not None else "n/a")
        deltas = [f"{(m / means[0] - 1) * 100:+.1f}%" for m in means[1:]
                  if m is not None and means[0]]
        print(f"| {c} | " + " | ".join(cells) + f" | {', '.join(deltas) or 'n/a'} |")
