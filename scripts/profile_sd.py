#!/usr/bin/env python3
"""Time and profile batch-1 speculative decoding on a running vLLM server
(FINDINGS.md, section 9). Run from the repo root.

    .venv/bin/python scripts/profile_sd.py http://127.0.0.1:8000

One fixed Spec-Bench prompt (#439, math reasoning), chat template applied here
as `vllm bench serve` does, greedy, 128 tokens, EOS ignored.
1. One warm-up request, then three timed ones. vLLM streams one chunk per
   engine step, so chunk timestamps give wall time per step, and tokens after
   the first chunk / steps after the first give tokens per step. The server's
   spec-decode counters (/metrics) cover the timed requests.
2. Starts the torch profiler (its schedule is on the server's command line),
   sends the same request, stops it.
"""
import json
import os
import statistics
import sys
import time
import urllib.request

from transformers import AutoTokenizer

MODELS = os.environ.get("MODELS", os.path.expanduser("~/models"))

base = sys.argv[1].rstrip("/")
QID = 439
q = next(r for r in map(json.loads, open("data/spec_bench_question.jsonl")) if r["question_id"] == QID)
tok = AutoTokenizer.from_pretrained(os.path.join(MODELS, "Qwen3-8B-AWQ"))
PROMPT = tok.apply_chat_template([{"role": "user", "content": q["turns"][0]}],
                                 add_generation_prompt=True, tokenize=False)
BODY = {"model": "qwen3-8b-awq", "prompt": PROMPT, "max_tokens": 128, "temperature": 0,
        "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}


def post(path, body=None):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(base + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def spec_counters():
    names = {"vllm:spec_decode_num_drafts_total": "drafts",
             "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
             "vllm:spec_decode_num_accepted_tokens_total": "accepted"}
    out = dict.fromkeys(names.values(), 0.0)
    with urllib.request.urlopen(base + "/metrics", timeout=60) as r:
        for line in r.read().decode().splitlines():
            name = line.split("{")[0]
            if name in names:
                out[names[name]] += float(line.rsplit(" ", 1)[1])
    return out


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
    n, steps = usage["completion_tokens"], len(stamps)
    decode_ms = (stamps[-1] - stamps[0]) * 1000
    return {"ttft_ms": (stamps[0] - t0) * 1000, "tokens": n, "chunks": steps,
            "ms_per_step": decode_ms / (steps - 1), "tokens_per_step": (n - 1) / (steps - 1),
            "ms_per_token": decode_ms / (n - 1)}


warmup = timed_request()
c0 = spec_counters()
runs = [timed_request() for _ in range(3)]
c1 = spec_counters()
delta = {k: c1[k] - c0[k] for k in c0}
med = lambda key: statistics.median(r[key] for r in runs)
out = {"question_id": QID, "category": q["category"], "prompt_tokens": len(tok(PROMPT)["input_ids"]),
       "warmup": warmup, "timed": runs, "median_ms_per_step": med("ms_per_step"),
       "median_tokens_per_step": med("tokens_per_step"), "median_ms_per_token": med("ms_per_token"),
       "spec_counters_timed": delta}

post("/start_profile").read()
out["profiled"] = timed_request()
time.sleep(1)
post("/stop_profile").read()
print(json.dumps(out, indent=1))
