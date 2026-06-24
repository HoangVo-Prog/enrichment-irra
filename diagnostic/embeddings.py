"""Embedding extraction for the frozen IRRA retriever."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from diagnostic.scoring import RetrieverEmbeddingCache


def extract_retriever_embeddings(
    adapter,
    img_loader,
    txt_loader,
    use_grab: bool = False,
    logger: logging.Logger | None = None,
) -> RetrieverEmbeddingCache:
    if use_grab:
        raise ValueError("IRRA diagnostics support only global text/image embeddings")

    query_global = []
    gallery_global = []
    adapter.eval()
    with torch.no_grad():
        for batch_index, (_pid, captions) in enumerate(txt_loader, start=1):
            feats = adapter.encode_text(captions)
            query_global.append(F.normalize(feats.float(), p=2, dim=1).cpu())
            if logger is not None and batch_index % 50 == 0:
                logger.info("Encoded IRRA text batches=%d", batch_index)
        for batch_index, (_pid, images) in enumerate(img_loader, start=1):
            feats = adapter.encode_image(images)
            gallery_global.append(F.normalize(feats.float(), p=2, dim=1).cpu())
            if logger is not None and batch_index % 50 == 0:
                logger.info("Encoded IRRA image batches=%d", batch_index)
    return RetrieverEmbeddingCache(
        query_global=torch.cat(query_global, dim=0),
        gallery_global=torch.cat(gallery_global, dim=0),
        query_grab=None,
        gallery_grab=None,
    )


def extract_retrieval_embeddings(adapter, split_data):
    cache = extract_retriever_embeddings(
        adapter,
        split_data.img_loader,
        split_data.txt_loader,
        use_grab=False,
    )
    return cache.query_global, cache.gallery_global

