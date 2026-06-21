import math

import numpy as np

from diagnostic.bootstrap import (
    bootstrap_count_stats,
    cluster_bootstrap,
    cluster_bootstrap_by_unit,
    cluster_columns_for_unit,
    group_rows_by_cluster,
)
from diagnostic.constants import BOOTSTRAP_METRICS


def _row(query_id, case_id, trial_id, base):
    row = {
        "dataset": "RSTPReid",
        "retriever_name": "irra",
        "case_id": case_id,
        "query_id": query_id,
        "trial_id": trial_id,
    }
    for idx, metric in enumerate(BOOTSTRAP_METRICS):
        row[metric] = float(base + idx * 0.1)
    return row


def _six_rows():
    rows = []
    for trial in range(2):
        rows.append(_row(1, "case_a", trial, 1.0 + trial))
        rows.append(_row(1, "case_b", trial, 3.0 + trial))
        rows.append(_row(2, "case_a", trial, 5.0 + trial))
    return rows


def _metric_map(rows):
    return {row["metric"]: row for row in rows}


def test_cluster_counts_for_case_query_and_unique_query():
    rows = _six_rows()

    case_counts = bootstrap_count_stats(rows, cluster_columns_for_unit("case_query"))
    unique_counts = bootstrap_count_stats(rows, cluster_columns_for_unit("unique_query"))

    assert case_counts["cluster_count"] == 3
    assert unique_counts["cluster_count"] == 2
    assert case_counts["unique_query_count"] == 2
    assert unique_counts["case_query_count"] == 3
    assert case_counts["trial_count"] == 6
    assert unique_counts["trial_count"] == 6


def test_unique_query_grouping_keeps_all_cases_together():
    rows = _six_rows()
    grouped = group_rows_by_cluster(rows, cluster_columns_for_unit("unique_query"))
    q1_key = ("RSTPReid", "irra", 1)
    q1_cases = {row["case_id"] for row in grouped[q1_key]}

    assert q1_cases == {"case_a", "case_b"}
    assert len(grouped[q1_key]) == 4


def test_variable_sized_unique_query_clusters_are_weighted_by_rows():
    rows = []
    for trial in range(2):
        rows.append(_row("A", "case_1", trial, 1.0))
        for case_id in ("case_1", "case_2", "case_3"):
            rows.append(_row("B", case_id, trial, 5.0))

    ci_rows = cluster_bootstrap_by_unit(rows, iters=10, seed=7, bootstrap_unit="unique_query")
    first = _metric_map(ci_rows)["r1_flip"]

    assert first["cluster_count"] == 2
    assert first["unique_query_count"] == 2
    assert first["case_query_count"] == 4
    assert first["trial_count"] == 8
    assert math.isclose(first["mean"], 4.0)


def test_point_estimates_identical_between_bootstrap_units():
    rows = _six_rows()
    case_rows = _metric_map(cluster_bootstrap_by_unit(rows, iters=25, seed=11, bootstrap_unit="case_query"))
    unique_rows = _metric_map(cluster_bootstrap_by_unit(rows, iters=25, seed=11, bootstrap_unit="unique_query"))

    for metric in BOOTSTRAP_METRICS:
        assert math.isclose(case_rows[metric]["mean"], unique_rows[metric]["mean"])


def test_reproducibility_for_same_seed_iters_and_unit():
    rows = _six_rows()
    first = cluster_bootstrap_by_unit(rows, iters=50, seed=123, bootstrap_unit="unique_query")
    second = cluster_bootstrap_by_unit(rows, iters=50, seed=123, bootstrap_unit="unique_query")

    assert first == second


def test_paired_delta_uses_same_resampled_clusters():
    rows = [
        {
            "dataset": "RSTPReid",
            "retriever_name": "irra",
            "case_id": "case_a",
            "query_id": "q1",
            "trial_id": 0,
            "r1_flip": 10.0,
            "rank_shift": 0.0,
            "hm_r1_flip": 1.0,
            "hm_rank_shift": 0.0,
            "delta_r1_flip": 9.0,
            "delta_rank_shift": 4.0,
            "cue_shift": 0.0,
        },
        {
            "dataset": "RSTPReid",
            "retriever_name": "irra",
            "case_id": "case_b",
            "query_id": "q2",
            "trial_id": 0,
            "r1_flip": 1.0,
            "rank_shift": 0.0,
            "hm_r1_flip": 10.0,
            "hm_rank_shift": 0.0,
            "delta_r1_flip": -9.0,
            "delta_rank_shift": -4.0,
            "cue_shift": 0.0,
        },
    ]
    out = _metric_map(cluster_bootstrap_by_unit(rows, iters=4, seed=5, bootstrap_unit="case_query"))

    rng = np.random.default_rng(5)
    cluster_delta_sums = np.asarray([9.0, -9.0])
    cluster_delta_rank_sums = np.asarray([4.0, -4.0])
    reps = []
    rank_reps = []
    for _ in range(4):
        sample = rng.integers(0, 2, size=2)
        reps.append(float(cluster_delta_sums[sample].sum() / 2.0))
        rank_reps.append(float(cluster_delta_rank_sums[sample].sum() / 2.0))
    low, high = np.quantile(np.asarray(reps), [0.025, 0.975])
    rank_low, rank_high = np.quantile(np.asarray(rank_reps), [0.025, 0.975])

    assert math.isclose(out["delta_r1_flip"]["ci_low"], low)
    assert math.isclose(out["delta_r1_flip"]["ci_high"], high)
    assert math.isclose(out["delta_rank_shift"]["ci_low"], rank_low)
    assert math.isclose(out["delta_rank_shift"]["ci_high"], rank_high)


def test_case_query_wrapper_reproduces_existing_behavior():
    rows = _six_rows()
    old = cluster_bootstrap(rows, iters=30, seed=99)
    explicit = cluster_bootstrap_by_unit(rows, iters=30, seed=99, bootstrap_unit="case_query")

    assert old == explicit
