"""IRRA retriever score computation and full-test sanity metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from diagnostic.metrics import retrieval_metrics


@dataclass
class RetrieverEmbeddingCache:
    query_global: torch.Tensor
    gallery_global: torch.Tensor
    query_grab: Optional[torch.Tensor] = None
    gallery_grab: Optional[torch.Tensor] = None


def resolve_score_mode(mode: str, has_grab: bool = False, lambda_global: float = 1.0) -> str:
    if mode == "auto":
        return "global"
    if mode != "global":
        raise ValueError(f"--score_mode {mode} is not supported for IRRA diagnostics")
    return mode


def query_gallery_scores(
    cache: RetrieverEmbeddingCache,
    query_id: int,
    gallery_indices: Sequence[int],
    mode: str,
    lambda_global: float,
) -> np.ndarray:
    if mode != "global":
        raise ValueError(f"Unsupported IRRA score mode: {mode}")
    idx = torch.as_tensor(gallery_indices, dtype=torch.long)
    return (cache.query_global[int(query_id)] @ cache.gallery_global[idx].T).cpu().numpy()


def full_query_scores(
    cache: RetrieverEmbeddingCache,
    query_id: int,
    mode: str,
    lambda_global: float,
) -> np.ndarray:
    gallery_indices = np.arange(cache.gallery_global.shape[0], dtype=np.int64)
    return query_gallery_scores(cache, query_id, gallery_indices, mode, lambda_global)


def full_similarity_chunk(
    cache: RetrieverEmbeddingCache,
    start: int,
    end: int,
    mode: str,
    lambda_global: float,
) -> torch.Tensor:
    if mode != "global":
        raise ValueError(f"Unsupported IRRA score mode: {mode}")
    return cache.query_global[start:end] @ cache.gallery_global.T


def compute_full_test_metrics(
    cache: RetrieverEmbeddingCache,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    mode: str,
    lambda_global: float,
    chunk_size: int = 128,
) -> dict[str, float]:
    g_pids = np.asarray(gallery_pids, dtype=np.int64)
    rows = []
    for start in range(0, len(query_pids), chunk_size):
        end = min(start + chunk_size, len(query_pids))
        scores = full_similarity_chunk(cache, start, end, mode, lambda_global).cpu().numpy()
        for offset, row_scores in enumerate(scores):
            query_index = start + offset
            is_positive = g_pids == int(query_pids[query_index])
            if not is_positive.any():
                continue
            rows.append(retrieval_metrics(row_scores, is_positive))
    if not rows:
        raise RuntimeError("No full-test queries had gallery positives")
    return {
        "mode": mode,
        "R1": float(np.mean([row["R1"] for row in rows])),
        "R5": float(np.mean([row["R5"] for row in rows])),
        "R10": float(np.mean([row["R10"] for row in rows])),
        "mAP": float(np.mean([row["AP"] for row in rows])),
        "mean_best_positive_rank": float(np.mean([row["best_positive_rank"] for row in rows])),
        "num_queries": int(len(rows)),
    }

