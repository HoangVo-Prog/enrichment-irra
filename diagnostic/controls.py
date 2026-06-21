"""Hardness-matched control construction using IRRA scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HardnessControl:
    hm_a: list[int]
    hm_b: list[int]
    hm_a_replacement: list[int]
    hm_b_replacement: list[int]
    diagnostics: dict[str, float]


def _nearest_unused_by_score(
    target_scores: np.ndarray,
    candidate_ids: np.ndarray,
    all_scores: np.ndarray,
) -> list[int]:
    if len(target_scores) == 0:
        return []
    if candidate_ids.size < len(target_scores):
        return []
    candidate_scores = all_scores[candidate_ids]
    order = np.argsort(candidate_scores, kind="mergesort")
    sorted_ids = candidate_ids[order]
    sorted_scores = candidate_scores[order]
    used = np.zeros(sorted_ids.shape[0], dtype=bool)
    replacements: list[int] = []

    for target in target_scores:
        pos = int(np.searchsorted(sorted_scores, target))
        left = pos - 1
        right = pos
        chosen_idx = -1
        while left >= 0 or right < sorted_scores.size:
            left_dist = abs(sorted_scores[left] - target) if left >= 0 and not used[left] else np.inf
            right_dist = abs(sorted_scores[right] - target) if right < sorted_scores.size and not used[right] else np.inf
            if left_dist == np.inf and right_dist == np.inf:
                left -= 1
                right += 1
                continue
            if left_dist <= right_dist:
                chosen_idx = left
            else:
                chosen_idx = right
            break
        if chosen_idx < 0:
            return []
        used[chosen_idx] = True
        replacements.append(int(sorted_ids[chosen_idx]))
    return replacements


def _diag(prefix: str, cue_ids: list[int], control_ids: list[int], scores: np.ndarray) -> dict[str, float]:
    cue_values = scores[np.asarray(cue_ids, dtype=np.int64)] if cue_ids else np.asarray([], dtype=np.float64)
    control_values = scores[np.asarray(control_ids, dtype=np.int64)] if control_ids else np.asarray([], dtype=np.float64)
    return {
        f"{prefix}_mean_score_cue_subset": float(cue_values.mean()) if cue_values.size else 0.0,
        f"{prefix}_mean_score_control_subset": float(control_values.mean()) if control_values.size else 0.0,
        f"{prefix}_std_score_cue_subset": float(cue_values.std()) if cue_values.size else 0.0,
        f"{prefix}_std_score_control_subset": float(control_values.std()) if control_values.size else 0.0,
    }


def construct_hardness_controls(pair, query_pid: int, gallery_pids: np.ndarray, score_vector: np.ndarray) -> tuple[HardnessControl | None, str]:
    all_ids = np.arange(len(gallery_pids), dtype=np.int64)
    distractors = all_ids[gallery_pids != int(query_pid)]

    a_excluded = set(pair.positives) | set(pair.a_dense) | set(pair.a_neutral)
    b_excluded = set(pair.positives) | set(pair.b_dense) | set(pair.b_neutral)
    a_candidates = np.asarray([int(x) for x in distractors if int(x) not in a_excluded], dtype=np.int64)
    b_candidates = np.asarray([int(x) for x in distractors if int(x) not in b_excluded], dtype=np.int64)

    a_targets = score_vector[np.asarray(pair.a_dense, dtype=np.int64)] if pair.a_dense else np.asarray([], dtype=np.float64)
    b_targets = score_vector[np.asarray(pair.b_dense, dtype=np.int64)] if pair.b_dense else np.asarray([], dtype=np.float64)
    a_repl = _nearest_unused_by_score(a_targets, a_candidates, score_vector)
    b_repl = _nearest_unused_by_score(b_targets, b_candidates, score_vector)
    if len(a_repl) != len(pair.a_dense):
        return None, "insufficient_hm_a_candidates"
    if len(b_repl) != len(pair.b_dense):
        return None, "insufficient_hm_b_candidates"

    hm_a = list(pair.positives) + a_repl + list(pair.a_neutral)
    hm_b = list(pair.positives) + b_repl + list(pair.b_neutral)
    if len(set(hm_a)) != len(hm_a) or len(set(hm_b)) != len(hm_b):
        return None, "duplicate_hm_image_id"

    diagnostics = {}
    diagnostics.update(_diag("hm_a", pair.a_dense, a_repl, score_vector))
    diagnostics.update(_diag("hm_b", pair.b_dense, b_repl, score_vector))
    return HardnessControl(hm_a, hm_b, a_repl, b_repl, diagnostics), ""
