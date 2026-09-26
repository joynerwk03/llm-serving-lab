#!/usr/bin/env python3
"""Profile batch-1 decode on a running server (vLLM or SGLang).

    python3 scripts/profile_decode.py vllm   http://127.0.0.1:8000
    python3 scripts/profile_decode.py sglang http://127.0.0.1:30000 /traces

1. Sends one greedy streaming request unprofiled (256-ish-token prompt, 64
   output tokens, EOS ignored) three times, and reports the median time per
   output token from the stream's token timestamps. That's the true TPOT.
2. Starts the engine's torch profiler, sends the same request, stops it.
   vLLM's profiler schedule is set on its command line (--profiler-config);
   SGLang takes the output dir and activities in the /start_profile body.
"""
import json
import statistics
import sys
import time
import urllib.request

engine, base = sys.argv[1], sys.argv[2].rstrip("/")
trace_dir = sys.argv[3] if len(sys.argv) > 3 else "/traces"
PROMPT = ("The history of computing is a story of trading one scarce resource for "
          "another: memory for time, time for energy, energy for money. ") * 9
BODY = {"model": "qwen3-8b-awq", "prompt": PROMPT, "max_tokens": 64, "temperature": 0,
        "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}


def post(path, body=None, stream=False):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(base + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def timed_request():
    t0 = time.perf_counter()
    stamps, usage = [], None
    with post("/v1/completions", BODY) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("choices") and ev["choices"][0].get("text"):
                stamps.append(time.perf_counter())
            if ev.get("usage"):
                usage = ev["usage"]
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return {"ttft_ms": (stamps[0] - t0) * 1000, "tpot_ms": statistics.median(gaps) * 1000,
            "chunks": len(stamps), "usage": usage}


runs = [timed_request() for _ in range(3)]
print(json.dumps({"engine": engine, "unprofiled": runs,
                  "median_tpot_ms": statistics.median(r["tpot_ms"] for r in runs)}, indent=1))

if engine == "vllm":
    post("/start_profile").read()
else:
    post("/start_profile", {"output_dir": trace_dir, "activities": ["CPU", "GPU"],
                            "with_stack": False, "record_shapes": False}).read()
prof = timed_request()
time.sleep(1)
post("/stop_profile").read()
print(json.dumps({"profiled": prof}))
