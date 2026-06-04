import unittest
from types import SimpleNamespace

import torch

from model.target_enrichment import (
    RankPartQueryConditionedMixerAdapter,
    TargetPrototypeEnricher,
    build_part_prototypes,
)


def make_args(**overrides):
    args = SimpleNamespace(
        top_m=2,
        extractor_mode="global,horizontal",
        num_parts=2,
        mixer_dim=8,
        mixer_depth=1,
        mixer_hidden_part=4,
        mixer_hidden_rank=4,
        mixer_hidden_channel=16,
        mixer_hidden_readout=16,
        context_pooling="mlp",
        residual_gate="residual",
        enrich_gamma=None,
        residual_gate_hidden_dim=8,
        use_target_retrieval_loss=True,
        use_target_robust_loss=True,
        lambda_ret=1.0,
        lambda_rob=1.0,
        lambda_gain=1.0,
        gain_margin=0.1,
        tau=0.07,
        robust_hard_k=2,
        enrichment_space="global",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TargetEnrichmentTest(unittest.TestCase):
    def test_prototype_slot_counts_and_normalization(self):
        tokens = torch.randn(2, 1 + 12, 4)
        prototypes = build_part_prototypes(
            tokens,
            num_parts=2,
            grid_size=(3, 4),
            mode="global,horizontal,vertical,grid",
        )
        self.assertEqual(tuple(prototypes.shape), (2, 9, 4))
        norms = prototypes.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_invalid_grid_raises(self):
        tokens = torch.randn(2, 1 + 12, 4)
        with self.assertRaises(ValueError):
            build_part_prototypes(tokens, num_parts=2, mode="vertical")

    def test_mixer_output_shape(self):
        mixer = RankPartQueryConditionedMixerAdapter(
            feature_dim=8,
            top_m=4,
            num_slots=3,
            mixer_dim=8,
            mixer_depth=1,
            mixer_hidden_part=4,
            mixer_hidden_rank=4,
            mixer_hidden_channel=16,
            mixer_hidden_readout=16,
        )
        out = mixer(torch.randn(2, 2, 3, 8), torch.randn(2, 8))
        self.assertEqual(tuple(out.shape), (2, 8))
        self.assertIn("mixer/context_norm", mixer.last_diagnostics)

    def test_residual_gate_initializes_near_point_one(self):
        enricher = TargetPrototypeEnricher(8, make_args())
        gate_layer = enricher.gate_mlp[-1]
        gate = torch.sigmoid(gate_layer.bias)
        self.assertTrue(torch.allclose(gate, torch.tensor([0.1]), atol=1e-5))

    def test_disabled_losses_return_zero_scalar(self):
        enricher = TargetPrototypeEnricher(
            8,
            make_args(use_target_retrieval_loss=False, use_target_robust_loss=False),
        )
        pool_cache = {
            "host_image_features": torch.randn(3, 8),
            "retrieval_features": torch.randn(3, 8),
            "prototypes": torch.randn(3, 3, 8),
            "pids": torch.tensor([0, 1, 2]),
            "image_ids": torch.tensor([0, 1, 2]),
        }
        ret = enricher(
            query_features=torch.randn(2, 8),
            host_text_features=torch.randn(2, 8),
            query_pids=torch.tensor([0, 2]),
            pool_cache=pool_cache,
        )
        self.assertEqual(ret["total_loss"].dim(), 0)
        self.assertTrue(torch.allclose(ret["total_loss"], torch.tensor(0.0), atol=1e-6))

    def test_losses_are_finite_and_gradients_flow(self):
        torch.manual_seed(1)
        enricher = TargetPrototypeEnricher(8, make_args())
        pool_cache = {
            "host_image_features": torch.randn(5, 8),
            "retrieval_features": torch.randn(5, 8),
            "prototypes": torch.randn(5, 3, 8),
            "pids": torch.tensor([0, 1, 0, 2, 3]),
            "image_ids": torch.arange(5),
        }
        ret = enricher(
            query_features=torch.randn(2, 8),
            host_text_features=torch.randn(2, 8),
            query_pids=torch.tensor([0, 2]),
            pool_cache=pool_cache,
        )
        self.assertTrue(torch.isfinite(ret["target_retrieval_loss"]))
        self.assertTrue(torch.isfinite(ret["target_robust_loss"]))
        ret["total_loss"].backward()
        grads = [
            p.grad
            for p in enricher.parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        ]
        self.assertGreater(len(grads), 0)

    def test_topm_selection_is_not_label_forced(self):
        enricher = TargetPrototypeEnricher(2, make_args(top_m=1, num_parts=1, extractor_mode="global"))
        pool_cache = {
            "host_image_features": torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
            "retrieval_features": torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
            "prototypes": torch.randn(2, 1, 2),
            "pids": torch.tensor([0, 5]),
            "image_ids": torch.tensor([0, 1]),
        }
        ret = enricher(
            query_features=torch.tensor([[1.0, 0.0]]),
            host_text_features=torch.tensor([[1.0, 0.0]]),
            query_pids=torch.tensor([5]),
            pool_cache=pool_cache,
        )
        self.assertEqual(int(ret["top_indices"][0, 0].item()), 0)
        sims = ret["enriched_features"] @ torch.nn.functional.normalize(
            pool_cache["retrieval_features"].float(), p=2, dim=1
        ).t()
        self.assertEqual(tuple(sims.shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
