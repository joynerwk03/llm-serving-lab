#!/usr/bin/env python3
"""Write docker/grafana/dashboards/inference-lab.json: one dashboard, vLLM and
SGLang side by side on every panel.

Metric names are from the installed vLLM 0.30.0 (vllm/v1/metrics) and
SGLang v0.5.20 (python/sglang/srt/observability/metrics_collector.py), and
checked against each server's live /metrics.

    python3 docker/grafana/make_dashboard.py
"""
import json
import pathlib

DS = {"type": "prometheus", "uid": "prom"}
OUT = pathlib.Path(__file__).parent / "dashboards" / "inference-lab.json"


def q(expr, legend, ref):
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref, "range": True}


def quantiles(vllm_hist, sglang_hist):
    out = []
    for i, (engine, hist) in enumerate((("vLLM", vllm_hist), ("SGLang", sglang_hist))):
        for j, qv in enumerate((0.5, 0.99)):
            expr = f"histogram_quantile({qv}, sum by (le) (rate({hist}_bucket[1m])))"
            out.append(q(expr, f"{engine} p{int(qv * 100)}", "ABCD"[2 * i + j]))
    return out


PANELS = [
    ("Requests running (the batch)", "none", [
        q("sum(vllm:num_requests_running)", "vLLM", "A"),
        q("sum(sglang:num_running_reqs)", "SGLang", "B")]),
    ("Requests waiting in the queue", "none", [
        q("sum(vllm:num_requests_waiting)", "vLLM", "A"),
        q("sum(sglang:num_queue_reqs)", "SGLang", "B")]),
    ("Output tokens per second", "suffix: tok/s", [
        q("sum(rate(vllm:generation_tokens_total[30s]))", "vLLM", "A"),
        q("sum(rate(sglang:generation_tokens_total[30s]))", "SGLang", "B")]),
    ("KV cache in use", "percent", [
        q("100 * max(vllm:kv_cache_usage_perc)", "vLLM", "A"),
        q("100 * max(sglang:token_usage)", "SGLang", "B")]),
    ("Time to first token", "s", quantiles(
        "vllm:time_to_first_token_seconds", "sglang:time_to_first_token_seconds")),
    ("Time between output tokens", "s", quantiles(
        "vllm:inter_token_latency_seconds", "sglang:inter_token_latency_seconds")),
    ("Prompt tokens per second (prefill)", "suffix: tok/s", [
        q("sum(rate(vllm:prompt_tokens_total[30s]))", "vLLM", "A"),
        q("sum(rate(sglang:prompt_tokens_total[30s]))", "SGLang", "B")]),
    ("Requests finished per second", "reqps", [
        q("sum(rate(vllm:request_success_total[30s]))", "vLLM", "A"),
        q("sum(rate(sglang:num_requests_total[30s]))", "SGLang", "B")]),
]

COLORS = {"vLLM": "#3274D9", "SGLang": "#E0752D"}


def panel(i, title, unit, targets):
    overrides = [
        {"matcher": {"id": "byRegexp", "options": f"^{name}.*"},
         "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}}]}
        for name, color in COLORS.items()
    ]
    # p99 lines dashed, so p50 and p99 of one engine share a color but read apart
    overrides.append({"matcher": {"id": "byRegexp", "options": ".*p99$"},
                      "properties": [{"id": "custom.lineStyle",
                                      "value": {"fill": "dash", "dash": [10, 6]}}]})
    defaults = {"unit": unit, "min": 0,
                "custom": {"lineWidth": 2, "fillOpacity": 8, "showPoints": "never",
                           "spanNulls": False}}
    if unit == "percent":
        defaults["max"] = 100
    return {
        "id": i + 1, "type": "timeseries", "title": title, "datasource": DS,
        "gridPos": {"x": 12 * (i % 2), "y": 8 * (i // 2), "w": 12, "h": 8},
        "targets": targets,
        "fieldConfig": {"defaults": defaults, "overrides": overrides},
        "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi", "sort": "none"}},
    }


dashboard = {
    "uid": "inference-lab",
    "title": "Inference Lab: vLLM vs SGLang on one RTX 3090",
    "description": "Qwen3-8B-AWQ. vLLM 0.30.0 and SGLang v0.5.20 in Docker; only one runs at a time.",
    "tags": ["inference-lab"],
    "timezone": "browser",
    "schemaVersion": 39,
    "version": 1,
    "refresh": "5s",
    "time": {"from": "now-30m", "to": "now"},
    "panels": [panel(i, *p) for i, p in enumerate(PANELS)],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(dashboard, indent=2) + "\n")
print(f"wrote {OUT} ({len(PANELS)} panels)")
