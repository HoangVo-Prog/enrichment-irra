"""Cue-biased gallery construction using external CLIP cue affinities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from diagnostic.config import stable_int_seed


@dataclass
class GalleryPair:
    query_id: int
    trial_id: int
    positives: list[int]
    a_dense: list[int]
    b_dense: list[int]
    a_neutral: list[int]
    b_neutral: list[int]
    gallery_a: list[int]
    gallery_b: list[int]


def _rng(seed: int, *parts):
    return np.random.default_rng(stable_int_seed(seed, *parts))


def _sample_ranked(
    candidates: np.ndarray,
    scores: np.ndarray,
    k: int,
    seed: int,
    *parts,
    descending: bool = True,
    pool_factor: int = 2,
) -> list[int]:
    if k <= 0:
        return []
    if candidates.size < k:
        return []
    order = np.lexsort((candidates, -scores[candidates] if descending else scores[candidates]))
    ranked = candidates[order]
    pool_size = min(ranked.size, max(k, k * pool_factor))
    pool = ranked[:pool_size].copy()
    gen = _rng(seed, *parts)
    gen.shuffle(pool)
    chosen = pool[:k]
    chosen_order = np.lexsort((chosen,))
    return [int(x) for x in chosen[chosen_order]]


def _sample_random(candidates: np.ndarray, k: int, seed: int, *parts) -> list[int]:
    if k <= 0:
        return []
    if candidates.size < k:
        return []
    gen = _rng(seed, *parts)
    chosen = gen.choice(candidates, size=k, replace=False)
    return [int(x) for x in sorted(chosen.tolist())]


def construct_gallery_pair(
    query_id: int,
    query_pid: int,
    trial_id: int,
    case_id: str,
    cue_a: str,
    cue_b: str,
    gallery_pids: np.ndarray,
    cue_scores: dict[str, np.ndarray],
    gallery_size: int,
    dense_ratio: float,
    lambda_contrast: float,
    neutral_strategy: str,
    neutral_pool_factor: int,
    seed: int,
) -> tuple[GalleryPair | None, str]:
    all_ids = np.arange(len(gallery_pids), dtype=np.int64)
    positives = all_ids[gallery_pids == int(query_pid)]
    distractors = all_ids[gallery_pids != int(query_pid)]
    if positives.size == 0:
        return None, "no_positive"
    if positives.size >= gallery_size:
        return None, "too_many_positives"

    remaining = gallery_size - int(positives.size)
    dense_count = int(round(remaining * dense_ratio))
    neutral_count = remaining - dense_count
    if distractors.size < remaining:
        return None, "insufficient_distractors"

    psi_a = cue_scores[cue_a]
    psi_b = cue_scores[cue_b]
    dense_score_a = psi_a - lambda_contrast * psi_b
    dense_score_b = psi_b - lambda_contrast * psi_a

    a_dense = _sample_ranked(
        distractors,
        dense_score_a,
        dense_count,
        seed,
        case_id,
        query_id,
        trial_id,
        "a_dense",
        descending=True,
    )
    b_dense = _sample_ranked(
        distractors,
        dense_score_b,
        dense_count,
        seed,
        case_id,
        query_id,
        trial_id,
        "b_dense",
        descending=True,
    )
    if len(a_dense) != dense_count:
        return None, "insufficient_a_dense"
    if len(b_dense) != dense_count:
        return None, "insufficient_b_dense"

    a_excluded = set(int(x) for x in positives.tolist()) | set(a_dense)
    b_excluded = set(int(x) for x in positives.tolist()) | set(b_dense)
    shared_excluded = a_excluded | b_excluded
    shared_candidates = np.asarray([int(x) for x in distractors if int(x) not in shared_excluded], dtype=np.int64)

    if neutral_strategy == "random":
        shared_neutral = _sample_random(
            shared_candidates,
            neutral_count,
            seed,
            case_id,
            query_id,
            trial_id,
            "shared_neutral",
        )
    else:
        max_affinity = np.maximum(psi_a, psi_b)
        shared_neutral = _sample_ranked(
            shared_candidates,
            max_affinity,
            neutral_count,
            seed,
            case_id,
            query_id,
            trial_id,
            "shared_neutral",
            descending=False,
            pool_factor=neutral_pool_factor,
        )

    if len(shared_neutral) == neutral_count:
        a_neutral = shared_neutral
        b_neutral = shared_neutral
    else:
        a_candidates = np.asarray([int(x) for x in distractors if int(x) not in a_excluded], dtype=np.int64)
        b_candidates = np.asarray([int(x) for x in distractors if int(x) not in b_excluded], dtype=np.int64)
        if neutral_strategy == "random":
            a_neutral = _sample_random(a_candidates, neutral_count, seed, case_id, query_id, trial_id, "a_neutral")
            b_neutral = _sample_random(b_candidates, neutral_count, seed, case_id, query_id, trial_id, "b_neutral")
        else:
            max_affinity = np.maximum(psi_a, psi_b)
            a_neutral = _sample_ranked(
                a_candidates,
                max_affinity,
                neutral_count,
                seed,
                case_id,
                query_id,
                trial_id,
                "a_neutral",
                descending=False,
                pool_factor=neutral_pool_factor,
            )
            b_neutral = _sample_ranked(
                b_candidates,
                max_affinity,
                neutral_count,
                seed,
                case_id,
                query_id,
                trial_id,
                "b_neutral",
                descending=False,
                pool_factor=neutral_pool_factor,
            )
        if len(a_neutral) != neutral_count or len(b_neutral) != neutral_count:
            return None, "insufficient_neutral"

    positives_list = [int(x) for x in positives.tolist()]
    gallery_a = positives_list + a_dense + a_neutral
    gallery_b = positives_list + b_dense + b_neutral
    if len(set(gallery_a)) != gallery_size or len(set(gallery_b)) != gallery_size:
        return None, "duplicate_image_id"

    return (
        GalleryPair(
            query_id=int(query_id),
            trial_id=int(trial_id),
            positives=positives_list,
            a_dense=a_dense,
            b_dense=b_dense,
            a_neutral=a_neutral,
            b_neutral=b_neutral,
            gallery_a=gallery_a,
            gallery_b=gallery_b,
        ),
        "",
    )


def constructibility_rows(cases, selected_by_case, gallery_pids, gallery_size, dense_ratio):
    rows = []
    num_gallery = int(len(gallery_pids))
    remaining_min = gallery_size
    dense_count = int(round(max(0, gallery_size - 1) * dense_ratio))
    neutral_count = max(0, gallery_size - 1 - dense_count)
    for case in cases:
        selected = selected_by_case.get(case.case_id, [])
        min_pos = None
        min_dist = None
        for query in selected:
            pos = int((gallery_pids == int(query.pid)).sum())
            dist = int((gallery_pids != int(query.pid)).sum())
            min_pos = pos if min_pos is None else min(min_pos, pos)
            min_dist = dist if min_dist is None else min(min_dist, dist)
        if min_pos is None:
            min_pos = 0
            min_dist = 0
        constructible = min_pos > 0 and min_pos < gallery_size and min_dist >= gallery_size - min_pos
        reason = "" if constructible else "insufficient_gallery_or_queries"
        remaining_min = max(0, gallery_size - min_pos)
        dense_count = int(round(remaining_min * dense_ratio))
        neutral_count = remaining_min - dense_count
        rows.append(
            {
                "case_id": case.case_id,
                "cue_a": case.cue_a,
                "cue_b": case.cue_b,
                "num_queries": len(case.query_ids),
                "num_selected_queries": len(selected),
                "num_gallery": num_gallery,
                "min_positive_candidates": int(min_pos),
                "min_a_dense_candidates": int(min_dist),
                "min_b_dense_candidates": int(min_dist),
                "min_neutral_candidates": int(max(0, min_dist - dense_count)),
                "constructible": bool(constructible),
                "reason": reason,
            }
        )
    return rows
