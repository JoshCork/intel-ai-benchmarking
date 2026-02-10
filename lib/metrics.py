"""
Metrics — statistical aggregation for benchmark results.

Computes mean, median, percentiles, stddev from raw run data.
"""

from __future__ import annotations

import math
from typing import Sequence


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) from a sorted list."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    k = (p / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(sorted_values: list[float]) -> float:
    return _percentile(sorted_values, 50)


def _stddev(values: Sequence[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def compute_aggregates(metrics: list[dict]) -> dict:
    """Compute aggregate statistics from a list of per-run metric dicts.

    Each dict should have: ttft_ms, total_ms, tokens_per_sec, output_tokens.
    Only non-warmup runs are included.

    Returns dict with keys like ttft_mean, ttft_median, tps_mean, etc.
    """
    # Filter to measured runs only
    measured = [m for m in metrics if not m.get("is_warmup", False)]
    if not measured:
        return {}

    ttft_vals = sorted(m["ttft_ms"] for m in measured if m.get("ttft_ms"))
    tps_vals = sorted(m["tokens_per_sec"] for m in measured if m.get("tokens_per_sec"))
    total_vals = sorted(m["total_ms"] for m in measured if m.get("total_ms"))
    output_vals = [m["output_tokens"] for m in measured if m.get("output_tokens")]

    def _stats(sorted_vals: list[float], prefix: str) -> dict:
        if not sorted_vals:
            return {}
        m = _mean(sorted_vals)
        return {
            f"{prefix}_mean": round(m, 2),
            f"{prefix}_median": round(_median(sorted_vals), 2),
            f"{prefix}_p5": round(_percentile(sorted_vals, 5), 2),
            f"{prefix}_p95": round(_percentile(sorted_vals, 95), 2),
            f"{prefix}_stddev": round(_stddev(sorted_vals, m), 2),
        }

    result = {}
    result.update(_stats(ttft_vals, "ttft"))
    result.update(_stats(tps_vals, "tps"))
    result.update(_stats(total_vals, "total"))
    result["total_measured_runs"] = len(measured)
    result["avg_output_tokens"] = round(_mean(output_vals), 1) if output_vals else 0

    return result


def format_comparison_table(rows: list[dict]) -> str:
    """Format comparison data as a readable ASCII table."""
    if not rows:
        return "No benchmark data found."

    headers = [
        "Machine", "Codename", "GPU", "Model", "Precision",
        "Temp", "Device", "Scenario",
        "TPS Mean", "TPS Med", "TPS P95",
        "TTFT Mean", "Total Mean", "Runs",
    ]

    def _row(r: dict) -> list[str]:
        return [
            r.get("hostname", ""),
            r.get("cpu_codename", ""),
            (r.get("gpu_model", "") or "")[:30],
            (r.get("model_name", "").split("/")[-1])[:25],
            r.get("model_precision", ""),
            str(r.get("temperature", "")),
            r.get("target_device", ""),
            r.get("scenario_name", "")[:15],
            f"{r.get('tps_mean', 0):.1f}" if r.get("tps_mean") else "-",
            f"{r.get('tps_median', 0):.1f}" if r.get("tps_median") else "-",
            f"{r.get('tps_p95', 0):.1f}" if r.get("tps_p95") else "-",
            f"{r.get('ttft_mean', 0):.0f}" if r.get("ttft_mean") else "-",
            f"{r.get('total_mean', 0):.0f}" if r.get("total_mean") else "-",
            str(r.get("total_measured_runs", "")),
        ]

    data_rows = [_row(r) for r in rows]

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in data_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Format
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"

    lines = [sep, header_line, sep]
    for row in data_rows:
        lines.append(
            "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"
        )
    lines.append(sep)

    return "\n".join(lines)
