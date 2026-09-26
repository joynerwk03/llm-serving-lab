#!/usr/bin/env python3
"""Determinism pilot: does the same prompt at temperature 0 give the same tokens
alone and inside a busy batch?

    python3 experiments/determinism_pilot.py run  <base_url> <out.jsonl> alone
    python3 experiments/determinism_pilot.py run  <base_url> <out.jsonl> batched
    python3 experiments/determinism_pilot.py compare <a.jsonl> <b.jsonl>

Prompts: 24 items from the public Recall or Reason question bank (the first 3
of each category), each asking for step-by-step reasoning. Get the bank with
    git clone https://github.com/joynerwk03/recall-or-reason
and point RR_BANK at its data/items.jsonl (default: ../recall-or-reason/...,
next to this repo). Greedy, 256 tokens, EOS ignored, thinking off, chat
endpoint.
- alone:   the 24 targets one at a time.
- batched: the 24 targets plus 40 background requests (other bank items) with
  varied lengths (32-400 tokens) and staggered starts (0-3 s), so the batch
  around each target changes while it generates. Fixed seed, so every batched
  run sends the same schedule.
- together: the 24 targets at once, no background. With a token budget above
  their total they share one prefill step, then decode as a fixed batch of
  24: batch size changes, but nothing is split or mixed.
"""
import json
import os
import pathlib
import random
import sys
import time
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent.parent
BANK = pathlib.Path(os.environ.get(
    "RR_BANK", HERE.parent / "recall-or-reason" / "data" / "items.jsonl"))
MODELS = os.environ.get("MODELS", os.path.expanduser("~/models"))
MODEL_DIR = os.environ.get("MODEL_DIR", f"{MODELS}/Qwen3-8B-AWQ")  # for the tokenizer


def load_items():
    rows = [json.loads(l) for l in open(BANK)]
    per = OrderedDict()
    for r in rows:
        per.setdefault(r["category"], []).append(r)
    targets = [r for c in per for r in per[c][:3]]
    others = [r for r in rows if r not in targets]
    return targets, others


def prompt_of(it):
    opts = "\n".join(f"{'ABCDEFG'[i]}) {o}" for i, o in enumerate(it["options"]))
    return (f"{it['prompt']}\n\nBackground: {it['context']}\n\nOptions:\n{opts}\n\n"
            "Reason through this step by step, then finish with a line 'Answer: <letter>'.")


def chat(base, text, max_tokens, model):
    body = {"model": model, "messages": [{"role": "user", "content": text}],
            "temperature": 0, "max_tokens": max_tokens, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"], out["usage"]["completion_tokens"]


def run(base, out_path, mode, model="qwen3-8b-awq"):
    targets, others = load_items()
    rng = random.Random(4242)
    t0 = time.time()
    rows = []
    if mode == "alone":
        for it in targets:
            text, n = chat(base, prompt_of(it), 256, model)
            rows.append({"id": it["id"], "text": text, "tokens": n})
    else:
        if mode == "together":
            jobs = [("t", it, 256, 0.0) for it in targets]
        else:
            bg = [(it, rng.randint(32, 400), rng.uniform(0, 3)) for it in others[:40]]
            jobs = [("t", it, 256, rng.uniform(0, 3)) for it in targets] + [("b", it, n, d) for it, n, d in bg]
            rng.shuffle(jobs)

        def go(job):
            kind, it, n, delay = job
            time.sleep(delay)
            text, used = chat(base, prompt_of(it), n, model)
            return kind, it["id"], text, used

        with ThreadPoolExecutor(len(jobs)) as pool:
            res = list(pool.map(go, jobs))
        order = {it["id"]: i for i, it in enumerate(targets)}
        rows = sorted(({"id": i, "text": t, "tokens": u} for k, i, t, u in res if k == "t"),
                      key=lambda r: order[r["id"]])
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{mode}: {len(rows)} targets in {time.time() - t0:.0f} s -> {out_path}")


def compare(a_path, b_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    A = {r["id"]: r for r in map(json.loads, open(a_path))}
    B = {r["id"]: r for r in map(json.loads, open(b_path))}
    div, same, answer_flips = [], 0, 0
    for i in A:
        ta, tb = tok.encode(A[i]["text"]), tok.encode(B[i]["text"])
        first = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        if first is None and len(ta) == len(tb):
            same += 1
            continue
        first = first if first is not None else min(len(ta), len(tb))
        la = A[i]["text"].rsplit("Answer:", 1)[-1].strip()[:1] if "Answer:" in A[i]["text"] else "?"
        lb = B[i]["text"].rsplit("Answer:", 1)[-1].strip()[:1] if "Answer:" in B[i]["text"] else "?"
        answer_flips += la != lb
        div.append({"id": i, "first_divergent_token": first, "answer_a": la, "answer_b": lb})
    print(json.dumps({"a": a_path, "b": b_path, "prompts": len(A), "identical": same,
                      "diverged": len(div), "answer_changed": answer_flips,
                      "divergences": sorted(div, key=lambda d: d["first_divergent_token"])}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], *(sys.argv[5:6]))
    else:
        compare(sys.argv[2], sys.argv[3])
