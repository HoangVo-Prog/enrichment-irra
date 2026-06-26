from types import SimpleNamespace

import pytest
import torch

from datasets import target_pool as target_pool_module
from datasets.target_pool import TargetPoolManager


def args(**overrides):
    defaults = dict(
        img_size=(16, 8),
        text_length=8,
        top_m=2,
        topm_rank_space="host_global",
        topm_rank_lambda=0.5,
        recompute_level="epoch",
        recompute_interval=2,
        pool_interval=None,
        use_freeze_indices=False,
        freeze_indices=False,
        test_batch_size=4,
        target_cache_batch_size=4,
        target_query_batch_size=4,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class DummyTrainDataset:
    def __init__(self):
        self.dataset = [
            (0, 10, "a.jpg", "first caption"),
            (0, 10, "a.jpg", "second caption"),
            (1, 11, "b.jpg", "third caption"),
        ]


def test_records_deduplicate_by_image_id_and_keep_query_rows():
    manager = TargetPoolManager(DummyTrainDataset(), args())
    assert [record["image_id"] for record in manager.image_records] == [10, 11]
    assert [record["query_index"] for record in manager.query_records] == [0, 1, 2]
    assert manager.query_records[1]["caption"] == "second caption"


def test_interval_id_epoch_and_step():
    manager = TargetPoolManager(DummyTrainDataset(), args(recompute_level="epoch", recompute_interval=2))
    assert manager._interval_id(epoch=1, step=99) == 0
    assert manager._interval_id(epoch=2, step=99) == 0
    assert manager._interval_id(epoch=3, step=99) == 1
    step_manager = TargetPoolManager(DummyTrainDataset(), args(recompute_level="step", recompute_interval=3))
    assert step_manager._interval_id(epoch=1, step=1) == 0
    assert step_manager._interval_id(epoch=1, step=4) == 1
    once_manager = TargetPoolManager(DummyTrainDataset(), args(recompute_interval=-1))
    assert once_manager._interval_id(epoch=100, step=100) == 0


def test_frozen_batch_requires_index_and_slices_rank_depth():
    manager = TargetPoolManager(DummyTrainDataset(), args(use_freeze_indices=True, top_m=2))
    manager.frozen_cache = {
        "host_image_features": torch.randn(3, 4),
        "retrieval_features": torch.randn(3, 4),
        "evidence_bank": torch.randn(3, 1, 4),
        "pids": torch.tensor([0, 1, 2]),
    }
    manager.frozen_rank_indices = torch.tensor([[0, 1, 2], [1, 0, 2], [2, 1, 0]])
    with pytest.raises(KeyError):
        manager._get_frozen_cache(model=None, batch={})
    cache = manager._get_frozen_cache(model=None, batch={"index": torch.tensor([0, 2])})
    assert cache["top_indices"].shape == (2, 2)
    assert torch.equal(cache["top_indices"].cpu(), torch.tensor([[0, 1], [2, 1]]))
    assert cache["diagnostics"]["frozen_indices_used"] == 1.0
    with pytest.raises(ValueError):
        manager._get_frozen_cache(model=None, batch={"index": torch.tensor([99])})


def test_image_cache_uses_target_cache_batch_size(monkeypatch):
    seen_batches = []

    class FakeImageDataset(torch.utils.data.Dataset):
        def __init__(self, records, transform):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            record = self.records[index]
            return record["pid"], record["image_id"], torch.zeros(3, 4, 4)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def encode_target_image_cache(self, images, cache_prototypes=True):
            seen_batches.append(int(images.shape[0]))
            batch_size = images.shape[0]
            return {
                "host_image_features": torch.zeros(batch_size, 4, device=images.device),
                "retrieval_features": torch.zeros(batch_size, 4, device=images.device),
                "evidence_bank": torch.zeros(batch_size, 1, 4, device=images.device),
            }

        def finalize_target_cache(self, cache):
            return cache

    monkeypatch.setattr(target_pool_module, "_PoolImageDataset", FakeImageDataset)
    manager = TargetPoolManager(
        DummyTrainDataset(),
        args(test_batch_size=99, target_cache_batch_size=1),
    )

    manager._encode_records(FakeModel(), manager.image_records)

    assert seen_batches == [1, 1]


def test_text_cache_uses_target_query_batch_size(monkeypatch):
    seen_batches = []

    class FakeTextDataset(torch.utils.data.Dataset):
        def __init__(self, records, tokenizer, text_length, truncate):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            record = self.records[index]
            return record["pid"], record["query_index"], torch.zeros(8, dtype=torch.long)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def encode_clip_global_text(self, tokens):
            seen_batches.append(int(tokens.shape[0]))
            return torch.zeros(tokens.shape[0], 4, device=tokens.device)

    monkeypatch.setattr(target_pool_module, "_PoolTextDataset", FakeTextDataset)
    manager = TargetPoolManager(
        DummyTrainDataset(),
        args(test_batch_size=99, target_query_batch_size=2),
    )
    manager.train_dataset.tokenizer = object()
    manager.train_dataset.text_length = 8
    manager.train_dataset.truncate = True

    manager._encode_text_records(FakeModel(), "global")

    assert seen_batches == [2, 1]


def test_frozen_topk_chunks_use_target_query_batch_size(monkeypatch):
    seen_batches = []
    original_topk = target_pool_module.torch.topk

    def recording_topk(scores, *topk_args, **topk_kwargs):
        seen_batches.append(int(scores.shape[0]))
        return original_topk(scores, *topk_args, **topk_kwargs)

    monkeypatch.setattr(target_pool_module.torch, "topk", recording_topk)
    manager = TargetPoolManager(
        DummyTrainDataset(),
        args(test_batch_size=99, target_query_batch_size=2),
    )
    queries = torch.eye(5)
    images = torch.eye(5)

    ranked = manager._topk_chunks(queries, images, rank_depth=1)

    assert ranked.shape == (5, 1)
    assert torch.equal(ranked.view(-1), torch.arange(5))
    assert seen_batches == [2, 2, 1]
