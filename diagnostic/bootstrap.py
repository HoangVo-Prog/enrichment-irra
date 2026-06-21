"""Optimized cluster bootstrap over explicit diagnostic cluster units."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from diagnostic.constants import (
    BOOTSTRAP_METRICS,
    BOOTSTRAP_UNIT_CASE_QUERY,
    BOOTSTRAP_UNIT_UNIQUE_QUERY,
    CASE_QUERY_CLUSTER_COLS,
    UNIQUE_QUERY_CLUSTER_COLS,
)


def cluster_key(row: dict, cluster_columns: Sequence[str]) -> tuple:
    return tuple(row.get(column, "") for column in cluster_columns)


def group_rows_by_cluster(rows: Iterable[dict], cluster_columns: Sequence[str]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(cluster_key(row, cluster_columns), []).append(row)
    return grouped


def bootstrap_count_stats(rows: Sequence[dict], cluster_columns: Sequence[str]) -> dict[str, int]:
    unique_queries = {
        (row.get("dataset", ""), row.get("retriever_name", ""), row.get("query_id", ""))
        for row in rows
    }
    case_queries = {
        (row.get("dataset", ""), row.get("retriever_name", ""), row.get("case_id", ""), row.get("query_id", ""))
        for row in rows
    }
    clusters = {cluster_key(row, cluster_columns) for row in rows}
    return {
        "cluster_count": len(clusters),
        "unique_query_count": len(unique_queries),
        "case_query_count": len(case_queries),
        "trial_count": len(rows),
    }


def cluster_columns_for_unit(bootstrap_unit: str) -> tuple[str, ...]:
    if bootstrap_unit == BOOTSTRAP_UNIT_CASE_QUERY:
        return CASE_QUERY_CLUSTER_COLS
    if bootstrap_unit == BOOTSTRAP_UNIT_UNIQUE_QUERY:
        return UNIQUE_QUERY_CLUSTER_COLS
    raise ValueError(f"Unsupported bootstrap unit: {bootstrap_unit}")


def bootstrap_unit_label(bootstrap_unit: str) -> str:
    if bootstrap_unit == BOOTSTRAP_UNIT_CASE_QUERY:
        return "case-query-instance cluster bootstrap"
    if bootstrap_unit == BOOTSTRAP_UNIT_UNIQUE_QUERY:
        return "unique-query cluster bootstrap"
    raise ValueError(f"Unsupported bootstrap unit: {bootstrap_unit}")


def _value(row: dict, metric: str) -> float:
    value = row.get(metric, 0.0)
    if value in ("", None):
        return 0.0
    return float(value)


def _empty_rows(
    metrics: Sequence[str],
    iters: int,
    bootstrap_unit: str,
    counts: dict[str, int] | None = None,
) -> list[dict]:
    counts = counts or {
        "cluster_count": 0,
        "unique_query_count": 0,
        "case_query_count": 0,
        "trial_count": 0,
    }
    return [
        {
            "metric": metric,
            "mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "bootstrap_iters": int(iters),
            "bootstrap_unit": bootstrap_unit,
            **counts,
        }
        for metric in metrics
    ]


def cluster_bootstrap_ci(
    rows: Iterable[dict],
    metric_columns: Iterable[str] = BOOTSTRAP_METRICS,
    cluster_columns: Sequence[str] = CASE_QUERY_CLUSTER_COLS,
    bootstrap_iters: int = 1000,
    bootstrap_seed: int = 123,
    bootstrap_unit: str = BOOTSTRAP_UNIT_CASE_QUERY,
) -> list[dict]:
    rows = list(rows)
    metric_columns = list(metric_columns)
    counts = bootstrap_count_stats(rows, cluster_columns)
    if not rows:
        return _empty_rows(metric_columns, bootstrap_iters, bootstrap_unit, counts)

    cluster_index: dict[tuple, int] = {}
    sums = []
    row_counts = []
    for row in rows:
        key = cluster_key(row, cluster_columns)
        if key not in cluster_index:
            cluster_index[key] = len(cluster_index)
            sums.append(np.zeros(len(metric_columns), dtype=np.float64))
            row_counts.append(0)
        idx = cluster_index[key]
        for metric_idx, metric in enumerate(metric_columns):
            sums[idx][metric_idx] += _value(row, metric)
        row_counts[idx] += 1

    sum_arr = np.vstack(sums)
    count_arr = np.asarray(row_counts, dtype=np.float64)
    total_count = float(count_arr.sum())
    observed = sum_arr.sum(axis=0) / max(total_count, 1.0)
    cluster_count = len(cluster_index)

    rng = np.random.default_rng(bootstrap_seed)
    boot = np.zeros((max(bootstrap_iters, 0), len(metric_columns)), dtype=np.float64)
    if bootstrap_iters > 0 and cluster_count > 0:
        for i in range(bootstrap_iters):
            sample = rng.integers(0, cluster_count, size=cluster_count)
            sample_sums = sum_arr[sample].sum(axis=0)
            sample_count = count_arr[sample].sum()
            boot[i, :] = sample_sums / max(sample_count, 1.0)

    rows_out = []
    counts = {
        **counts,
        "cluster_count": int(cluster_count),
        "trial_count": int(len(rows)),
    }
    for metric_idx, metric in enumerate(metric_columns):
        if bootstrap_iters > 0:
            ci_low, ci_high = np.quantile(boot[:, metric_idx], [0.025, 0.975])
        else:
            ci_low = ci_high = observed[metric_idx]
        rows_out.append(
            {
                "metric": metric,
                "mean": float(observed[metric_idx]),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "bootstrap_iters": int(bootstrap_iters),
                "bootstrap_unit": bootstrap_unit,
                **counts,
            }
        )
    return rows_out


def cluster_bootstrap_by_unit(
    rows: Iterable[dict],
    metrics: Iterable[str] = BOOTSTRAP_METRICS,
    iters: int = 1000,
    seed: int = 123,
    bootstrap_unit: str = BOOTSTRAP_UNIT_UNIQUE_QUERY,
) -> list[dict]:
    return cluster_bootstrap_ci(
        rows,
        metric_columns=metrics,
        cluster_columns=cluster_columns_for_unit(bootstrap_unit),
        bootstrap_iters=iters,
        bootstrap_seed=seed,
        bootstrap_unit=bootstrap_unit,
    )


def cluster_bootstrap(
    rows: Iterable[dict],
    metrics: Iterable[str] = BOOTSTRAP_METRICS,
    iters: int = 1000,
    seed: int = 123,
) -> list[dict]:
    """Backward-compatible case-query-instance cluster bootstrap."""

    return cluster_bootstrap_by_unit(
        rows,
        metrics=metrics,
        iters=iters,
        seed=seed,
        bootstrap_unit=BOOTSTRAP_UNIT_CASE_QUERY,
    )
