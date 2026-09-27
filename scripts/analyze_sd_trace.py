#!/usr/bin/env python3
"""Split batch-1 engine steps into GPU time by phase, from a vLLM V1 torch
profiler trace, for FINDINGS.md section 9.

    python3 scripts/analyze_sd_trace.py <trace.json[.gz]> [--ms-per-step 7.7] [--json out.json]

Steps start at each of the worker's `execute_context_<n>(<tokens>)_generation_...`
ranges on the host; only pure decode steps are kept (no prefill, not empty),
and the last step, which the profiler may cut off, is dropped. Every GPU
kernel, copy and memset is tied to the host call that launched it (the
profiler's correlation id), and that call to the innermost
`gpu_model_runner: <phase>` range around it: forward (the target model),
draft (the drafter), sample, and so on. Those labels exist only with
VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 on the server. Per step: GPU-busy time
(intervals merged), busy time per phase, and host calls per phase (individual
kernel launches, CUDA-graph launches, copies, syncs).

With --ms-per-step (the unprofiled wall time per step, from profile_sd.py),
idle = that minus GPU busy: time per step the GPU waits on the host.
"""
import argparse
import bisect
import gzip
import json
import statistics
from collections import defaultdict

LAUNCH = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx",
          "cudaLaunchCooperativeKernel")
GRAPH = ("cudaGraphLaunch", "cuGraphLaunch")
COPY = ("cudaMemcpyAsync", "cudaMemcpy", "cuMemcpyAsync", "cuMemcpyHtoDAsync_v2", "cuMemcpyDtoHAsync_v2",
        "cudaMemcpy2DAsync")
SYNC = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cuStreamSynchronize",
        "cuEventSynchronize", "cuCtxSynchronize")
PREFIX = "gpu_model_runner: "


def merged(intervals):
    total, end = 0.0, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--ms-per-step", type=float, default=None)
ap.add_argument("--json", default=None)
a = ap.parse_args()

op = gzip.open if a.trace.endswith(".gz") else open
with op(a.trace, "rt") as f:
    tr = json.load(f)
evs = [e for e in (tr["traceEvents"] if isinstance(tr, dict) else tr) if e.get("ph") == "X"]

ranges = defaultdict(list)  # tid -> [(ts, end, phase)]
marks = []  # (ts, name) of each engine step
for e in evs:
    if e.get("cat") != "user_annotation":
        continue
    if e["name"].startswith(PREFIX):
        ranges[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"][len(PREFIX):]))
    elif e["name"].startswith("execute_"):
        marks.append((e["ts"], e["name"]))
marks.sort()
starts = [ts for ts, _ in marks]
if len(starts) < 4:
    raise SystemExit(f"only {len(starts)} steps found")
if not ranges:
    print("note: no gpu_model_runner phase labels (server ran without VLLM_CUSTOM_SCOPES_FOR_PROFILING=1)")


def phase_at(tid, ts):
    best = None  # innermost range containing ts
    for s, e, p in ranges.get(tid, ()):
        if s <= ts <= e and (best is None or s >= best[0]):
            best = (s, p)
    return best[1] if best else "(none)"


def step_at(ts):
    return bisect.bisect_right(starts, ts) - 1  # -1 = before the first step


runtime = {}  # correlation -> (step, phase)
calls = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # step -> phase -> counter
for e in evs:
    if e.get("cat") not in ("cuda_runtime", "cuda_driver"):
        continue
    st, ph = step_at(e["ts"]), phase_at(e["tid"], e["ts"])
    corr = (e.get("args") or {}).get("correlation")
    if corr is not None:
        runtime[corr] = (st, ph)
    n = e["name"]
    kind = ("launch" if n in LAUNCH else "graph" if n in GRAPH else "copy" if n in COPY
            else "sync" if n in SYNC else None)
    if kind:
        calls[st][ph][kind] += 1
        if kind == "sync":
            calls[st][ph]["sync_ms"] += e["dur"] / 1000

host = defaultdict(lambda: defaultdict(float))  # step -> phase -> host ms inside its ranges
for rs in ranges.values():
    for s, e, p in rs:
        host[step_at(s)][p] += (e - s) / 1000

gpu = defaultdict(lambda: defaultdict(list))  # step -> phase -> [(s, e)]
for e in evs:
    if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
        continue
    corr = (e.get("args") or {}).get("correlation")
    st, ph = runtime.get(corr, (-1, "(unmatched)"))
    gpu[st][ph].append((e["ts"], e["ts"] + e["dur"]))

steps = [i for i in range(len(starts) - 1)  # pure decode steps, not the last
         if "_context_0(" in marks[i][1] and "_generation_0(" not in marks[i][1]]
phases = sorted({p for s in steps for p in gpu[s]} | {p for s in steps for p in calls[s]})
rows = []
for s in steps:
    row = {"step": s, "host_ms": (starts[s + 1] - starts[s]) / 1000,
           "gpu_busy_ms": merged([iv for p in gpu[s] for iv in gpu[s][p]]) / 1000}
    for p in phases:
        row[f"busy:{p}"] = merged(gpu[s][p]) / 1000
        # first kernel start to last kernel end: span - busy = GPU gaps inside the phase
        row[f"span:{p}"] = (max(e for _, e in gpu[s][p]) - min(b for b, _ in gpu[s][p])) / 1000 if gpu[s][p] else 0.0
        row[f"host:{p}"] = host[s][p]
        for k in ("launch", "graph", "copy", "sync", "sync_ms"):
            row[f"{k}:{p}"] = calls[s][p][k]
    rows.append(row)

med = {k: statistics.median(r[k] for r in rows) for k in rows[0] if k != "step"}
print(f"{a.trace}\n{len(rows)} steady steps (of {len(starts)} found)")
print(f"  host time per step (profiled, inflated): {med['host_ms']:.2f} ms")
print(f"  GPU busy per step: {med['gpu_busy_ms']:.2f} ms")
if a.ms_per_step:
    print(f"  wall per step (unprofiled): {a.ms_per_step:.2f} ms -> idle {a.ms_per_step - med['gpu_busy_ms']:.2f} ms "
          f"({100 * (a.ms_per_step - med['gpu_busy_ms']) / a.ms_per_step:.0f}%)")
print("  per step, medians. GPU span = first kernel to last, so span - GPU ms = gaps inside the phase;")
print("  host ms = time inside the phase's ranges. Spans and host times are profiled, so inflated.")
print(f"  {'phase':<28}{'GPU ms':>8}{'span ms':>9}{'host ms':>9}{'launches':>10}{'graphs':>8}{'copies':>8}{'syncs':>7}{'sync ms':>9}")
for p in phases:
    print(f"  {p:<28}{med['busy:' + p]:>8.2f}{med['span:' + p]:>9.2f}{med['host:' + p]:>9.2f}{med['launch:' + p]:>10.0f}"
          f"{med['graph:' + p]:>8.0f}{med['copy:' + p]:>8.0f}{med['sync:' + p]:>7.0f}{med['sync_ms:' + p]:>9.2f}")
groups = defaultdict(list)  # by the worker's step label, e.g. generation_1(5) = 1 request, 5 tokens checked
for r in rows:
    groups[marks[r["step"]][1]].append(r)
if len(groups) > 1:
    print("  by step type (medians):")
    for name, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        m = lambda k: statistics.median(r.get(k, 0) for r in rs)
        print(f"    {name:<44} {len(rs):>4} steps  GPU {m('gpu_busy_ms'):.2f} ms  host {m('host_ms'):.2f} ms  "
              f"forward: {m('launch:forward'):.0f} launches, {m('graph:forward'):.0f} graphs")
for r in rows:
    r["label"] = marks[r["step"]][1]
if a.json:
    json.dump({"trace": a.trace, "steps": len(rows), "median": med, "ms_per_step": a.ms_per_step,
               "rows": rows}, open(a.json, "w"), indent=1)
