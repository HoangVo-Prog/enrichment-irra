from types import SimpleNamespace

import pytest
import torch

from model.target_enrichment import (
    TargetPrototypeEnricher,
    balanced_bounds,
    build_evidence_bank,
    finalize_target_evidence_cache,
    normalize_enrichment_space,
    normalize_rank_space,
    parse_extractor_mode,
    validate_target_enrichment_args,
)


def args(**overrides):
    defaults = dict(
        target_enrichment=True,
        top_m=3,
        lambda_ret=1.0,
        topm_rank_lambda=0.5,
        num_parts=2,
        context_module="mixer",
        context_pooling="mlp",
        mixer_context_pooling="mlp",
        recompute_level="epoch",
        recompute_interval=1,
        pool_interval=None,
        residual_gate="residual",
        gate_mode="residual",
        enrich_gamma=None,
        enrichment_space="global",
        topm_rank_space="host_global",
        extractor_mode="global,horizontal",
        target_relative_space="host_global",
        target_relative_num_clusters=2,
        evidence_projection="auto",
        pnp_text_only=False,
        freeze_host=False,
        use_host_loss=True,
        use_freeze_indices=False,
        freeze_indices=False,
        temperature=0.02,
        mixer_dim=8,
        mixer_depth=1,
        mixer_hidden_part=4,
        mixer_hidden_rank=4,
        mixer_hidden_channel=16,
        mixer_hidden_readout=4,
        residual_gate_hidden_dim=4,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_aliases_and_validation():
    assert normalize_enrichment_space("grab") == "retrieval"
    assert normalize_rank_space("hybrid_global_grab") == "hybrid_global_retrieval"
    assert parse_extractor_mode("global_horizontal,horizontal,grid") == ["global", "horizontal", "grid"]
    with pytest.raises(ValueError):
        validate_target_enrichment_args(args(top_m=0))
    with pytest.raises(ValueError):
        validate_target_enrichment_args(args(lambda_ret=0))
    with pytest.raises(ValueError):
        validate_target_enrichment_args(args(pool_k=32))


def test_balanced_bounds_and_evidence_shapes():
    assert balanced_bounds(2, 4) == [(0, 1), (0, 1), (1, 2), (1, 2)]
    tokens = torch.randn(2, 1 + 6, 4)
    bank = build_evidence_bank(tokens, num_parts=2, grid_size=(2, 3), mode="global,horizontal,vertical,grid")
    assert bank.shape == (2, 1 + 2 + 2 + 4, 4)
    assert torch.allclose(bank.norm(dim=-1), torch.ones_like(bank.norm(dim=-1)), atol=1e-5)
    with pytest.raises(ValueError):
        build_evidence_bank(tokens, num_parts=2, grid_size=None, mode="vertical")


def test_target_relative_finalizer_writes_cluster_fields():
    cfg = args(extractor_mode="global,cluster,cluster_residual,cluster_density,cluster_rarity", num_parts=2)
    cache = {
        "host_image_features": torch.randn(4, 5),
        "retrieval_features": torch.randn(4, 5),
        "evidence_bank": torch.zeros(4, 5, 5),
    }
    out = finalize_target_evidence_cache(cache, cfg, evidence_dim=5)
    assert out["target_evidence_finalized"] is True
    assert out["target_relative_cluster_ids"].shape == (4,)
    assert out["target_relative_cluster_counts"].shape == (4,)
    assert out["cluster_density_scalar"].shape == (4, 1)
    assert out["cluster_rarity_scalar"].shape == (4, 1)


def test_enricher_label_free_topm_and_loss():
    cfg = args(extractor_mode="global", num_parts=1, top_m=2)
    enricher = TargetPrototypeEnricher(cfg, global_dim=4, retrieval_dim=4, evidence_dim=4)
    query = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    cache = {
        "host_image_features": torch.eye(4)[:3],
        "retrieval_features": torch.eye(4)[:3],
        "evidence_bank": torch.eye(4)[:3].unsqueeze(1),
        "pids": torch.tensor([10, 11, 12]),
    }
    out_a = enricher(query, query, None, cache)
    cache_b = dict(cache)
    cache_b["pids"] = torch.tensor([99, 98, 97])
    out_b = enricher(query, query, None, cache_b)
    assert torch.equal(out_a["top_indices"], out_b["top_indices"])

    train_cache = dict(cache)
    train_cache["pids"] = torch.tensor([0, 1, 1])
    train_out = enricher(query, query, torch.tensor([0, 1]), train_cache)
    assert train_out["target_retrieval_loss"].isfinite()
    train_out["total_loss"].backward()
    assert any(p.grad is not None for p in enricher.parameters())

    missing_cache = dict(cache)
    missing_cache["pids"] = torch.tensor([5, 6, 7])
    with pytest.raises(ValueError):
        enricher(query, query, torch.tensor([0, 1]), missing_cache)


def test_gate_modes_and_normalized_output():
    residual = TargetPrototypeEnricher(args(extractor_mode="global"), 4, 4, 4)
    query = torch.randn(2, 4)
    cache = {
        "host_image_features": torch.randn(3, 4),
        "retrieval_features": torch.randn(3, 4),
        "evidence_bank": torch.randn(3, 1, 4),
        "pids": torch.tensor([0, 1, 2]),
    }
    out = residual(query, query, None, cache)
    assert torch.allclose(out["gate"], torch.full_like(out["gate"], 0.1), atol=1e-5)
    assert torch.allclose(out["enriched_features"].norm(dim=-1), torch.ones(2), atol=1e-5)

    static = TargetPrototypeEnricher(args(extractor_mode="global", residual_gate="static", gate_mode="static", enrich_gamma=0.25), 4, 4, 4)
    out_static = static(query, query, None, cache)
    assert torch.allclose(out_static["gate"], torch.full_like(out_static["gate"], 0.25))
    assert "static_gate" not in dict(static.named_buffers())
    assert sum(buffer.numel() for _name, buffer in static.named_buffers()) == 0
