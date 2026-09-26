#!/usr/bin/env python3
"""Paired comparison of two lm-eval GSM8K runs (same questions), per filter.

    python3 scripts/mcnemar.py <run_a_dir> <run_b_dir>

Accuracy for each, then the discordant pairs: questions exactly one run got
right. McNemar's exact test asks whether those split evenly (two-sided
binomial, p = 0.5). Concordant questions carry no information about which is
better, which is why a paired test beats comparing two accuracies.
"""
import glob
import gzip
import json
import math
import sys


def load(d):
    # lm-eval writes samples_gsm8k_*.jsonl; the repo keeps them gzipped (12 MB raw)
    f = sorted(glob.glob(f"{d}/**/samples_gsm8k_*.jsonl*", recursive=True))
    if not f:
        sys.exit(f"no samples file under {d}")
    opener = gzip.open if f[-1].endswith(".gz") else open
    rows = [json.loads(l) for l in opener(f[-1], "rt")]
    out = {}
    for r in rows:
        for key in ("exact_match",):
            out.setdefault(r.get("filter", "none"), {})[r["doc_id"]] = r[key]
    return out


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


A, B = load(sys.argv[1]), load(sys.argv[2])
for flt in sorted(set(A) & set(B)):
    a, b = A[flt], B[flt]
    ids = sorted(set(a) & set(b))
    acc_a = sum(a[i] for i in ids) / len(ids)
    acc_b = sum(b[i] for i in ids) / len(ids)
    only_a = sum(1 for i in ids if a[i] and not b[i])
    only_b = sum(1 for i in ids if b[i] and not a[i])
    p = binom_two_sided(min(only_a, only_b), only_a + only_b)
    print(f"[{flt}] n={len(ids)}  A={acc_a:.1%}  B={acc_b:.1%}  diff={100 * (acc_a - acc_b):+.1f} pts  "
          f"only A right={only_a}  only B right={only_b}  McNemar exact p={p:.3f}")
