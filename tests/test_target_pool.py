import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from model.target_pool import TargetPoolManager


class FakeTrainDataset(object):
    def __init__(self):
        self.dataset = [
            (0, 10, "a.jpg", "caption a1"),
            (0, 10, "a.jpg", "caption a2"),
            (1, 11, "b.jpg", "caption b"),
            (1, 12, "c.jpg", "caption c"),
        ]
        self.transform = None


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


def make_args(**overrides):
    args = SimpleNamespace(
        recompute_interval=-1,
        recompute_level="epoch",
        use_freeze_indices=False,
        use_shared_k=False,
        pool_k_mode="static",
        pool_k=0,
        pool_k_candidates="2,3,4",
        pool_clusters=2,
        positive_ratio_max=0.5,
        pool_dist_metric="l1",
        pool_dist_threshold=0.25,
        pool_coverage_epochs=1,
        pool_seed=1,
        top_m=2,
        batch_size=2,
        test_batch_size=2,
        num_workers=0,
        text_length=77,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TargetPoolTest(unittest.TestCase):
    def test_unique_image_deduping_and_query_records(self):
        manager = TargetPoolManager(FakeTrainDataset(), make_args())
        self.assertEqual(len(manager.records), 3)
        self.assertEqual(len(manager.query_records), 4)
        self.assertEqual(manager.required_anchor_count, 2)

    def test_shared_anchor_selection_covers_identities(self):
        manager = TargetPoolManager(FakeTrainDataset(), make_args(use_shared_k=True, pool_k=3))
        anchors = manager._select_anchor_indices()
        anchor_pids = {manager.records[idx].pid for idx in anchors}
        self.assertEqual(anchor_pids, {0, 1})

    def test_static_shared_k_rejects_too_small_pool(self):
        manager = TargetPoolManager(FakeTrainDataset(), make_args(use_shared_k=True, pool_k=1))
        with self.assertRaises(ValueError):
            manager._choose_shared_k(TinyModel())

    def test_adaptive_k_satisfies_dilution_and_capacity(self):
        manager = TargetPoolManager(
            FakeTrainDataset(),
            make_args(use_shared_k=True, pool_k_mode="adaptive", pool_k_candidates="2,3,4"),
        )
        manager.cluster_assignments = torch.tensor([0, 1, 1])
        manager.cluster_distribution = torch.tensor([1.0 / 3.0, 2.0 / 3.0])
        selected, diagnostics = manager._choose_shared_k(TinyModel())
        self.assertEqual(selected, 3)
        self.assertEqual(diagnostics["pool_k_capacity_limited"], 1.0)

    def test_frozen_index_range_check(self):
        manager = TargetPoolManager(FakeTrainDataset(), make_args(use_freeze_indices=True))
        device = torch.device("cpu")
        manager.frozen_cache = {
            "host_image_features": torch.randn(3, 4),
            "retrieval_features": torch.randn(3, 4),
            "prototypes": torch.randn(3, 1, 4),
            "pids": torch.tensor([0, 1, 1]),
            "image_ids": torch.tensor([10, 11, 12]),
            "diagnostics": {},
        }
        manager.frozen_rankings = torch.tensor([[0, 1], [1, 2], [2, 1], [0, 2]])
        batch = {"index": torch.tensor([0, 5])}
        with self.assertRaises(ValueError):
            manager.get_train_cache(TinyModel().to(device), batch, epoch=1, step=0)

    def test_frozen_indices_are_gathered_by_batch_index(self):
        manager = TargetPoolManager(FakeTrainDataset(), make_args(use_freeze_indices=True))
        manager.frozen_cache = {
            "host_image_features": torch.randn(3, 4),
            "retrieval_features": torch.randn(3, 4),
            "prototypes": torch.randn(3, 1, 4),
            "pids": torch.tensor([0, 1, 1]),
            "image_ids": torch.tensor([10, 11, 12]),
            "diagnostics": {},
        }
        manager.frozen_rankings = torch.tensor([[0, 1], [1, 2], [2, 1], [0, 2]])
        cache = manager.get_train_cache(TinyModel(), {"index": torch.tensor([1, 3])}, epoch=1, step=0)
        self.assertTrue(torch.equal(cache["top_indices"], torch.tensor([[1, 2], [0, 2]])))


if __name__ == "__main__":
    unittest.main()
