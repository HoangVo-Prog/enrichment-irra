from types import SimpleNamespace

import pytest
import torch

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
