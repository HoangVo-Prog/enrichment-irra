"""IRRA cosine scoring and whole-test sanity metrics."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from diagnostic.metrics import retrieval_metrics


def resolve_score_mode(score_mode: str, has_grab: bool = False) -> str:
    if score_mode == "auto":
        return "global"
    if score_mode != "global":
        raise ValueError(f"Unsupported score mode for IRRA: {score_mode}")
    return score_mode


def score_gallery(score_vector: np.ndarray, gallery_ids: Iterable[int]) -> np.ndarray:
    ids = np.asarray(list(gallery_ids), dtype=np.int64)
    return np.asarray(score_vector[ids], dtype=np.float64)


def whole_test_metrics(query_embeddings, gallery_embeddings, query_pids, gallery_pids) -> dict[str, float]:
    import torch

    if len(query_pids) == 0 or len(gallery_pids) == 0:
        return {"R1": 0.0, "R5": 0.0, "R10": 0.0, "mAP": 0.0, "num_queries": 0, "num_gallery": 0}
    q = query_embeddings
    g = gallery_embeddings
    if not torch.is_tensor(q):
        q = torch.as_tensor(q)
    if not torch.is_tensor(g):
        g = torch.as_tensor(g)
    sim = (q.float() @ g.float().t()).cpu().numpy()
    rows = [retrieval_metrics(sim[i], gallery_pids, int(pid)) for i, pid in enumerate(query_pids)]
    return {
        "R1": float(np.mean([row["R1"] for row in rows]) * 100.0),
        "R5": float(np.mean([row["R5"] for row in rows]) * 100.0),
        "R10": float(np.mean([row["R10"] for row in rows]) * 100.0),
        "mAP": float(np.mean([row["AP"] for row in rows]) * 100.0),
        "num_queries": int(len(query_pids)),
        "num_gallery": int(len(gallery_pids)),
    }


def precompute_score_cache(
    query_embeddings,
    gallery_embeddings,
    query_ids: Iterable[int],
    device: str,
    logger: logging.Logger | None = None,
    chunk_size: int = 128,
) -> dict[int, np.ndarray]:
    import torch

    unique_query_ids = sorted(set(int(qid) for qid in query_ids))
    if not unique_query_ids:
        return {}
    if logger is not None:
        logger.info("Precomputing full score vectors for %d selected queries", len(unique_query_ids))

    q = query_embeddings.float() if torch.is_tensor(query_embeddings) else torch.as_tensor(query_embeddings).float()
    g = gallery_embeddings.float() if torch.is_tensor(gallery_embeddings) else torch.as_tensor(gallery_embeddings).float()
    use_device = torch.device(device)
    cache: dict[int, np.ndarray] = {}
    g_dev = g.to(use_device)
    for start in range(0, len(unique_query_ids), chunk_size):
        ids = unique_query_ids[start : start + chunk_size]
        q_dev = q[ids].to(use_device)
        with torch.no_grad():
            scores = q_dev @ g_dev.t()
        scores_np = scores.detach().cpu().numpy()
        for row_idx, qid in enumerate(ids):
            cache[int(qid)] = scores_np[row_idx].astype(np.float32, copy=False)
    if logger is not None:
        logger.info("Finished score precompute: score_cache_queries=%d", len(cache))
    return cache


def query_reference_rank(score_vector: np.ndarray, gallery_pids: np.ndarray, query_pid: int) -> float:
    return retrieval_metrics(score_vector, gallery_pids, query_pid)["best_positive_rank"]
