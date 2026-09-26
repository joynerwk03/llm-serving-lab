#!/usr/bin/env python3
"""Break batch-1 decode steps into GPU kernel time, from a torch profiler trace.

    python3 scripts/analyze_trace.py <trace.json[.gz]> [--tpot-ms 9.0] [--json out.json]

Steps are found by the output layer (lm_head): the one large non-Marlin GEMM
that runs exactly once per forward pass. The step with the most GPU time is
the prefill and is dropped. For each decode step: GPU-busy time (kernel
intervals merged), and time per category. With --tpot-ms (the unprofiled time
per token), overhead = TPOT - GPU busy, i.e. time the GPU sits idle per token.
"""
import argparse
import gzip
import json
import re
import statistics
from collections import defaultdict

CATS = [  # first match wins
    ("marlin_gemm", r"marlin"),
    ("attention", r"flash|fmha|attn|attention|BatchDecode|BatchPrefill|decode_kernel|prefill_kernel|paged"),
    ("gemm_other", r"gemm|gemv|xmma|cutlass|sm80_|sm86_|ampere_|splitK|splitk|cublas"),
    ("norm", r"rms|norm"),
    ("rotary", r"rotary|rope"),
    ("activation", r"act_and_mul|silu|gelu"),
    ("sampling", r"argmax|sampl|topk|top_k|topp|top_p|gumbel|softmax|max_reduce|reduce_kernel"),
    ("memcpy", r"memcpy|memset|Memcpy|Memset"),
]


def cat_of(name):
    for c, pat in CATS:
        if re.search(pat, name, re.IGNORECASE):
            return c
    return "other"


ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--tpot-ms", type=float, default=None)
ap.add_argument("--json", default=None)
a = ap.parse_args()

op = gzip.open if a.trace.endswith(".gz") else open
with op(a.trace, "rt") as f:
    tr = json.load(f)
evs = tr["traceEvents"] if isinstance(tr, dict) else tr
k = [e for e in evs if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
k.sort(key=lambda e: e["ts"])
if not k:
    raise SystemExit("no GPU kernel events in trace")

# lm_head: the non-Marlin GEMM name with the largest median duration that
# occurs at least 10 times.
by_name = defaultdict(list)
for e in k:
    by_name[e["name"]].append(e["dur"])
cands = [(statistics.median(d), n) for n, d in by_name.items()
         if cat_of(n) == "gemm_other" and len(d) >= 10]
if not cands:
    raise SystemExit("could not find the lm_head GEMM")
lm_dur, lm_name = max(cands)
starts = [e["ts"] for e in k if e["name"] == lm_name]

# A step = kernels from just after one lm_head to the end of the next one.
steps, cur, idx = [], [], 0
for e in k:
    cur.append(e)
    if e["name"] == lm_name:
        steps.append(cur)
        cur = []
steps = [s for s in steps if s]


def busy(sk):
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in sk)
    tot, (s0, e0) = 0.0, iv[0]
    for s, en in iv[1:]:
        if s > e0:
            tot += e0 - s0
            s0, e0 = s, en
        else:
            e0 = max(e0, en)
    return (tot + e0 - s0) / 1000.0  # us -> ms


rows = []
for s in steps:
    cats = defaultdict(float)
    for e in s:
        cats[cat_of(e["name"])] += e["dur"] / 1000.0
    rows.append({"busy": busy(s), "span": (s[-1]["ts"] + s[-1]["dur"] - s[0]["ts"]) / 1000.0,
                 "n": len(s), "cats": cats, "kern": s})
pre = max(range(len(rows)), key=lambda i: rows[i]["busy"])
# drop the prefill step and both edges (the first and last may be partial)
dec = [r for i, r in enumerate(rows) if i not in (0, pre, len(rows) - 1)]
if len(dec) < 5:
    raise SystemExit(f"only {len(dec)} decode steps found")

med = lambda xs: statistics.median(xs)
out = {"lm_head_kernel": lm_name[:90], "lm_head_ms": lm_dur / 1000.0,
       "decode_steps": len(dec), "kernels_per_step": med([r["n"] for r in dec]),
       "gpu_busy_ms": med([r["busy"] for r in dec]),
       "profiled_step_span_ms": med([r["span"] for r in dec]),
       "by_category_ms": {c: med([r["cats"].get(c, 0.0) for r in dec])
                          for c in sorted({c for r in dec for c in r["cats"]})}}
if a.tpot_ms:
    out["unprofiled_tpot_ms"] = a.tpot_ms
    out["overhead_ms"] = a.tpot_ms - out["gpu_busy_ms"]
# Roofline references (Qwen3-8B-AWQ): Marlin weights 3.61 GB, lm_head 1.245 GB, 936 GB/s
m = out["by_category_ms"].get("marlin_gemm")
if m:
    out["marlin_bandwidth_pct_of_peak"] = 100 * (3.61e9 / (m / 1000)) / 936e9
out["lm_head_bandwidth_pct_of_peak"] = 100 * (1.245e9 / (out["lm_head_ms"] / 1000)) / 936e9
top = defaultdict(float)
for r in dec:
    for e in r["kern"]:
        top[e["name"]] += e["dur"] / 1000.0 / len(dec)
out["top_kernels_ms_per_step"] = [(n[:80], round(v, 3)) for n, v in
                                  sorted(top.items(), key=lambda kv: -kv[1])[:12]]
print(json.dumps(out, indent=1))
if a.json:
    open(a.json, "w").write(json.dumps(out, indent=1))
