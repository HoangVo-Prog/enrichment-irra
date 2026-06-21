"""Embedding extraction for the frozen IRRA retriever."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EmbeddingSet:
    embeddings: Any
    pids: Any


def _normalize(features):
    import torch.nn.functional as F

    return F.normalize(features.float(), p=2, dim=1)


def extract_text_embeddings(adapter, txt_loader) -> EmbeddingSet:
    import torch

    all_pids = []
    all_features = []
    adapter.eval()
    for pid, caption in txt_loader:
        with torch.no_grad():
            features = adapter.encode_text(caption)
        all_pids.append(pid.view(-1).cpu())
        all_features.append(_normalize(features).cpu())
    return EmbeddingSet(torch.cat(all_features, 0), torch.cat(all_pids, 0).numpy())


def extract_image_embeddings(adapter, img_loader) -> EmbeddingSet:
    import torch

    all_pids = []
    all_features = []
    adapter.eval()
    for pid, image in img_loader:
        with torch.no_grad():
            features = adapter.encode_image(image)
        all_pids.append(pid.view(-1).cpu())
        all_features.append(_normalize(features).cpu())
    return EmbeddingSet(torch.cat(all_features, 0), torch.cat(all_pids, 0).numpy())


def extract_retrieval_embeddings(adapter, split_data) -> tuple[EmbeddingSet, EmbeddingSet]:
    text = extract_text_embeddings(adapter, split_data.txt_loader)
    image = extract_image_embeddings(adapter, split_data.img_loader)
    return text, image
