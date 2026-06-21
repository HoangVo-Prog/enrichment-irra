"""Retrieval and cue-shift metrics."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def deterministic_order(scores: np.ndarray, image_ids: Iterable[int] | None = None) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if image_ids is None:
        image_ids = np.arange(scores.shape[0])
    image_ids = np.asarray(list(image_ids), dtype=np.int64)
    return np.lexsort((image_ids, -scores))


def retrieval_metrics(
    scores: np.ndarray,
    gallery_pids: np.ndarray,
    query_pid: int,
    image_ids: Iterable[int] | None = None,
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    gallery_pids = np.asarray(gallery_pids)
    order = deterministic_order(scores, image_ids=image_ids)
    matches = gallery_pids[order] == int(query_pid)
    num_rel = int(matches.sum())
    if num_rel == 0:
        return {
            "R1": 0.0,
            "R5": 0.0,
            "R10": 0.0,
            "AP": 0.0,
            "best_positive_rank": math.inf,
        }
    positive_positions = np.flatnonzero(matches)
    best_rank = int(positive_positions[0]) + 1
    ranks = positive_positions + 1
    precisions = np.arange(1, num_rel + 1, dtype=np.float64) / ranks
    return {
        "R1": float(best_rank <= 1),
        "R5": float(best_rank <= 5),
        "R10": float(best_rank <= 10),
        "AP": float(precisions.mean()),
        "best_positive_rank": float(best_rank),
    }


def paired_retrieval_metrics(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {
        "r1_flip": float(a["R1"] != b["R1"]),
        "rank_shift": float(abs(a["best_positive_rank"] - b["best_positive_rank"])),
        "ap_delta": float(abs(a["AP"] - b["AP"])),
    }


def cue_density(
    gallery_ids: Iterable[int],
    query_pid: int,
    gallery_pids: np.ndarray,
    cue_scores: np.ndarray,
    threshold: float,
    tau_density: float,
) -> float:
    ids = np.asarray(list(gallery_ids), dtype=np.int64)
    if ids.size == 0:
        return 0.0
    distractor_ids = ids[gallery_pids[ids] != int(query_pid)]
    if distractor_ids.size == 0:
        return 0.0
    values = sigmoid((cue_scores[distractor_ids] - threshold) / tau_density)
    return float(values.mean())


def cue_shift(
    gallery_a_ids: Iterable[int],
    gallery_b_ids: Iterable[int],
    query_pid: int,
    gallery_pids: np.ndarray,
    cue_a_scores: np.ndarray,
    cue_b_scores: np.ndarray,
    threshold_a: float,
    threshold_b: float,
    tau_density: float,
) -> dict[str, float]:
    d_a_ga = cue_density(gallery_a_ids, query_pid, gallery_pids, cue_a_scores, threshold_a, tau_density)
    d_a_gb = cue_density(gallery_b_ids, query_pid, gallery_pids, cue_a_scores, threshold_a, tau_density)
    d_b_ga = cue_density(gallery_a_ids, query_pid, gallery_pids, cue_b_scores, threshold_b, tau_density)
    d_b_gb = cue_density(gallery_b_ids, query_pid, gallery_pids, cue_b_scores, threshold_b, tau_density)
    return {
        "cue_density_a_ga": d_a_ga,
        "cue_density_a_gb": d_a_gb,
        "cue_density_b_ga": d_b_ga,
        "cue_density_b_gb": d_b_gb,
        "cue_shift": float(0.5 * ((d_a_ga - d_a_gb) + (d_b_gb - d_b_ga))),
    }


def positive_ratio(gallery_ids: Iterable[int], query_pid: int, gallery_pids: np.ndarray) -> float:
    ids = np.asarray(list(gallery_ids), dtype=np.int64)
    if ids.size == 0:
        return 0.0
    return float((gallery_pids[ids] == int(query_pid)).mean())
