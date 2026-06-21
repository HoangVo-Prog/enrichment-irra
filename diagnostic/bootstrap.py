"""Optimized cluster bootstrap over query clusters."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from diagnostic.constants import BOOTSTRAP_METRICS


def _cluster_key(row: dict) -> tuple:
    return (
        row.get("dataset", ""),
        row.get("retriever_name", ""),
        row.get("case_id", ""),
        row.get("query_id", ""),
    )


def cluster_bootstrap(
    rows: Iterable[dict],
    metrics: Iterable[str] = BOOTSTRAP_METRICS,
    iters: int = 1000,
    seed: int = 123,
) -> list[dict]:
    rows = list(rows)
    metrics = list(metrics)
    if not rows:
        return [
            {
                "metric": metric,
                "mean": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "bootstrap_iters": iters,
                "cluster_count": 0,
                "trial_count": 0,
            }
            for metric in metrics
        ]

    cluster_index: dict[tuple, int] = {}
    sums = []
    counts = []
    for row in rows:
        key = _cluster_key(row)
        if key not in cluster_index:
            cluster_index[key] = len(cluster_index)
            sums.append(np.zeros(len(metrics), dtype=np.float64))
            counts.append(0)
        idx = cluster_index[key]
        for metric_idx, metric in enumerate(metrics):
            value = row.get(metric, 0.0)
            if value in ("", None):
                value = 0.0
            sums[idx][metric_idx] += float(value)
        counts[idx] += 1

    sum_arr = np.vstack(sums)
    count_arr = np.asarray(counts, dtype=np.float64)
    total_count = float(count_arr.sum())
    observed = sum_arr.sum(axis=0) / max(total_count, 1.0)
    cluster_count = len(cluster_index)

    rng = np.random.default_rng(seed)
    boot = np.zeros((max(iters, 0), len(metrics)), dtype=np.float64)
    if iters > 0 and cluster_count > 0:
        for i in range(iters):
            sample = rng.integers(0, cluster_count, size=cluster_count)
            sample_sums = sum_arr[sample].sum(axis=0)
            sample_counts = count_arr[sample].sum()
            boot[i, :] = sample_sums / max(sample_counts, 1.0)

    rows_out = []
    for metric_idx, metric in enumerate(metrics):
        if iters > 0:
            ci_low, ci_high = np.quantile(boot[:, metric_idx], [0.025, 0.975])
        else:
            ci_low = ci_high = observed[metric_idx]
        rows_out.append(
            {
                "metric": metric,
                "mean": float(observed[metric_idx]),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "bootstrap_iters": int(iters),
                "cluster_count": int(cluster_count),
                "trial_count": int(len(rows)),
            }
        )
    return rows_out
