import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_PROVIDERS = {
    "global",
    "horizontal",
    "vertical",
    "grid",
    "retrieval_backbone",
    "cluster",
    "cluster_residual",
    "cluster_density",
    "cluster_rarity",
}

REMOVED_TARGET_OPTIONS = {
    "use_shared_k",
    "pool_k",
    "pool_k_mode",
    "pool_k_candidates",
    "pool_coverage_epochs",
    "pool_clusters",
    "positive_ratio_max",
    "eta",
    "pool_dist_metric",
    "pool_dist_threshold",
    "epsilon",
    "use_target_retrieval_loss",
    "use_target_robust_loss",
    "hard_neg_k",
    "robust_hard_k",
    "lambda_rob",
    "lambda_gain",
    "gain_margin",
}

TARGET_RELATIVE_PROVIDERS = {
    "cluster",
    "cluster_residual",
    "cluster_density",
    "cluster_rarity",
}


def _arg(args, name, default=None):
    return getattr(args, name, default)


def normalize_enrichment_space(space):
    value = str(space).lower()
    if value == "clip_global":
        return "global"
    if value == "grab":
        return "retrieval"
    if value in {"global", "retrieval"}:
        return value
    raise ValueError(f"unsupported enrichment_space: {space}")


def normalize_rank_space(space):
    value = str(space).lower()
    if value in {"global", "clip_global"}:
        return "host_global"
    if value == "grab":
        return "retrieval"
    if value == "hybrid_global_grab":
        return "hybrid_global_retrieval"
    if value in {"host_global", "retrieval", "hybrid_global_retrieval"}:
        return value
    raise ValueError(f"unsupported topm_rank_space: {space}")


def parse_extractor_mode(mode):
    aliases = {
        "global_horizontal": ["global", "horizontal"],
        "global_vertical": ["global", "vertical"],
        "global_grid": ["global", "grid"],
    }
    tokens = []
    for raw in str(mode).split(","):
        token = raw.strip().lower()
        if token:
            tokens.extend(aliases.get(token, [token]))
    providers = []
    seen = set()
    for token in tokens:
        if token not in SUPPORTED_PROVIDERS:
            raise ValueError(f"unsupported evidence provider: {token}")
        if token not in seen:
            providers.append(token)
            seen.add(token)
    if not providers:
        raise ValueError("extractor_mode must contain at least one provider")
    return providers


def provider_slot_count(provider, num_parts):
    if provider in {"global", "retrieval_backbone", "cluster", "cluster_residual", "cluster_density", "cluster_rarity"}:
        return 1
    if provider in {"horizontal", "vertical"}:
        return int(num_parts)
    if provider == "grid":
        return int(num_parts) * int(num_parts)
    raise ValueError(f"unsupported evidence provider: {provider}")


def evidence_slot_count(providers, num_parts):
    return sum(provider_slot_count(provider, num_parts) for provider in providers)


def build_slot_index_map(providers, num_parts):
    indices = OrderedDict()
    cursor = 0
    for provider in providers:
        count = provider_slot_count(provider, num_parts)
        indices[provider] = list(range(cursor, cursor + count))
        cursor += count
    return indices


def balanced_bounds(length, num_parts):
    if length <= 0:
        raise ValueError("axis length must be positive")
    boundaries = torch.linspace(0, length, int(num_parts) + 1).round().long().tolist()
    bounds = []
    for part in range(int(num_parts)):
        start = int(boundaries[part])
        end = int(boundaries[part + 1])
        if end <= start:
            end = min(start + 1, length)
            start = max(0, end - 1)
        bounds.append((start, end))
    return bounds


def validate_target_enrichment_args(args):
    values = vars(args) if hasattr(args, "__dict__") else {}
    removed = sorted(name for name in REMOVED_TARGET_OPTIONS if name in values)
    if removed:
        raise ValueError("removed target-enrichment options are not supported: " + ", ".join(removed))
    if int(_arg(args, "top_m", 32)) < 1:
        raise ValueError("top_m must be >= 1")
    if float(_arg(args, "lambda_ret", 1.0)) <= 0:
        raise ValueError("lambda_ret must be > 0")
    if float(_arg(args, "tau", 0.015)) <= 0:
        raise ValueError("tau must be > 0")
    if int(_arg(args, "num_parts", 6)) < 1:
        raise ValueError("num_parts must be >= 1")
    rank_lambda = float(_arg(args, "topm_rank_lambda", 0.5))
    if rank_lambda < 0 or rank_lambda > 1:
        raise ValueError("topm_rank_lambda must be in [0, 1]")
    if str(_arg(args, "context_module", "mixer")).lower() != "mixer":
        raise ValueError("context_module must be mixer")
    pooling = str(_arg(args, "context_pooling", _arg(args, "mixer_context_pooling", "mlp"))).lower()
    if pooling != "mlp":
        raise ValueError("context_pooling must be mlp")
    pool_interval = _arg(args, "pool_interval", None)
    interval = int(pool_interval if pool_interval is not None else _arg(args, "recompute_interval", 1))
    if interval != -1 and interval < 1:
        raise ValueError("recompute_interval must be -1 or >= 1")
    if str(_arg(args, "recompute_level", "epoch")).lower() not in {"epoch", "step"}:
        raise ValueError("recompute_level must be epoch or step")
    gate = str(_arg(args, "gate_mode", _arg(args, "residual_gate", "residual"))).lower()
    enrich_gamma = _arg(args, "enrich_gamma", None)
    if gate == "static" and enrich_gamma is None:
        raise ValueError("residual_gate=static requires enrich_gamma")
    if gate == "residual" and enrich_gamma is not None:
        raise ValueError("residual_gate=residual forbids enrich_gamma")
    if gate not in {"residual", "static"}:
        raise ValueError("residual_gate must be residual or static")
    normalize_enrichment_space(_arg(args, "enrichment_space", "global"))
    normalize_rank_space(_arg(args, "topm_rank_space", "host_global"))
    parse_extractor_mode(_arg(args, "extractor_mode", "global,horizontal"))
    target_relative_space = str(_arg(args, "target_relative_space", "host_global")).lower()
    if target_relative_space in {"global", "clip_global"}:
        target_relative_space = "host_global"
    if target_relative_space not in {"host_global", "retrieval"}:
        raise ValueError("target_relative_space must be host_global or retrieval")
    if str(_arg(args, "evidence_projection", "auto")).lower() not in {"auto", "linear", "none"}:
        raise ValueError("evidence_projection must be auto, linear, or none")
    if bool(_arg(args, "pnp_text_only", False)):
        if not bool(_arg(args, "freeze_host", False)):
            raise ValueError("pnp_text_only requires freeze_host")
        if bool(_arg(args, "use_host_loss", True)):
            raise ValueError("pnp_text_only requires use_host_loss=false")
        if not (bool(_arg(args, "use_freeze_indices", False)) or bool(_arg(args, "freeze_indices", False))):
            raise ValueError("pnp_text_only requires use_freeze_indices")
        if normalize_enrichment_space(_arg(args, "enrichment_space", "global")) != "global":
            raise ValueError("pnp_text_only requires enrichment_space=global")


def has_target_relative_provider(args):
    providers = parse_extractor_mode(_arg(args, "extractor_mode", "global,horizontal"))
    return any(provider in TARGET_RELATIVE_PROVIDERS for provider in providers)


def _normalize(x):
    return F.normalize(x.float(), p=2, dim=-1)


def _valid_grid_size(grid_size, num_patches):
    if grid_size is None:
        return None
    height, width = int(grid_size[0]), int(grid_size[1])
    if height <= 0 or width <= 0 or height * width != int(num_patches):
        return None
    return height, width


def build_evidence_bank(token_features, num_parts=6, grid_size=None, mode="global,horizontal", retrieval_features=None):
    providers = parse_extractor_mode(mode)
    if token_features.dim() != 3:
        raise ValueError("token_features must have shape [B, 1 + patches, D]")
    tokens = token_features.float()
    cls = tokens[:, 0, :]
    patches = tokens[:, 1:, :]
    batch_size, num_patches, dim = patches.shape
    grid_hw = _valid_grid_size(grid_size, num_patches)
    patch_grid = patches.reshape(batch_size, grid_hw[0], grid_hw[1], dim) if grid_hw is not None else None
    slots = []
    for provider in providers:
        if provider == "global":
            slots.append(_normalize(cls).unsqueeze(1))
        elif provider == "horizontal":
            if num_patches <= 0:
                raise ValueError("horizontal evidence requires patch tokens")
            part_slots = []
            if patch_grid is not None:
                for start, end in balanced_bounds(grid_hw[0], num_parts):
                    part_slots.append(_normalize(patch_grid[:, start:end, :, :].mean(dim=(1, 2))))
            else:
                for start, end in balanced_bounds(num_patches, num_parts):
                    part_slots.append(_normalize(patches[:, start:end, :].mean(dim=1)))
            slots.append(torch.stack(part_slots, dim=1))
        elif provider == "vertical":
            if patch_grid is None:
                raise ValueError("vertical evidence requires a valid patch grid")
            part_slots = []
            for start, end in balanced_bounds(grid_hw[1], num_parts):
                part_slots.append(_normalize(patch_grid[:, :, start:end, :].mean(dim=(1, 2))))
            slots.append(torch.stack(part_slots, dim=1))
        elif provider == "grid":
            if patch_grid is None:
                raise ValueError("grid evidence requires a valid patch grid")
            row_bounds = balanced_bounds(grid_hw[0], num_parts)
            col_bounds = balanced_bounds(grid_hw[1], num_parts)
            part_slots = []
            for row_start, row_end in row_bounds:
                for col_start, col_end in col_bounds:
                    region = patch_grid[:, row_start:row_end, col_start:col_end, :]
                    part_slots.append(_normalize(region.mean(dim=(1, 2))))
            slots.append(torch.stack(part_slots, dim=1))
        elif provider == "retrieval_backbone":
            if retrieval_features is not None and retrieval_features.shape[-1] == dim:
                slots.append(_normalize(retrieval_features).unsqueeze(1))
            else:
                slots.append(torch.zeros(batch_size, 1, dim, dtype=tokens.dtype, device=tokens.device))
        elif provider in TARGET_RELATIVE_PROVIDERS:
            count = provider_slot_count(provider, num_parts)
            slots.append(torch.zeros(batch_size, count, dim, dtype=tokens.dtype, device=tokens.device))
        else:
            raise ValueError(f"unsupported evidence provider: {provider}")
    return torch.cat(slots, dim=1)


def _standardize_scalar(values):
    if values.numel() <= 1:
        return values - values.mean()
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def _spherical_kmeans(features, clusters):
    pool_size = features.shape[0]
    if clusters == pool_size:
        ids = torch.arange(pool_size, device=features.device, dtype=torch.long)
        counts = torch.ones(pool_size, device=features.device, dtype=torch.long)
        return ids, features, counts
    init_indices = torch.linspace(0, pool_size - 1, clusters, device=features.device).round().long()
    centroids = _normalize(features[init_indices])
    assignments = torch.full((pool_size,), -1, device=features.device, dtype=torch.long)
    for _ in range(10):
        scores = features @ centroids.t()
        new_assignments = scores.argmax(dim=1)
        if torch.equal(new_assignments, assignments):
            break
        assignments = new_assignments
        for cluster_id in range(clusters):
            mask = assignments == cluster_id
            if mask.any():
                centroids[cluster_id] = _normalize(features[mask].mean(dim=0, keepdim=True))[0]
    counts = torch.bincount(assignments, minlength=clusters).long()
    return assignments, centroids, counts


def finalize_target_evidence_cache(cache, args, evidence_dim=None):
    providers = parse_extractor_mode(_arg(args, "extractor_mode", "global,horizontal"))
    if not any(provider in TARGET_RELATIVE_PROVIDERS for provider in providers):
        cache["target_evidence_finalized"] = True
        return cache
    if "evidence_bank" not in cache:
        raise KeyError("target cache missing evidence_bank")
    evidence_bank = cache["evidence_bank"]
    evidence_dim = int(evidence_dim or evidence_bank.shape[-1])
    target_relative_space = str(_arg(args, "target_relative_space", "host_global")).lower()
    if target_relative_space in {"global", "clip_global"}:
        target_relative_space = "host_global"
    features = cache["retrieval_features"] if target_relative_space == "retrieval" else cache["host_image_features"]
    phi = _normalize(features)
    pool_size = phi.shape[0]
    if pool_size == 0:
        raise ValueError("empty target cache")
    cluster_count = min(max(1, int(_arg(args, "target_relative_num_clusters", 16))), pool_size)
    assignments, centroids, counts = _spherical_kmeans(phi, cluster_count)
    assigned_centroids = centroids[assignments]
    assigned_counts = counts[assignments]
    slot_map = build_slot_index_map(providers, int(_arg(args, "num_parts", 6)))
    projection = str(_arg(args, "evidence_projection", "auto")).lower()
    bank = evidence_bank.clone()
    if "cluster" in slot_map:
        cluster_features = _normalize(assigned_centroids)
        if cluster_features.shape[-1] == evidence_dim:
            bank[:, slot_map["cluster"], :] = cluster_features.unsqueeze(1)
        elif projection == "none":
            raise ValueError("cluster evidence dimension differs and evidence_projection=none")
        else:
            cache["cluster_features"] = cluster_features
    if "cluster_residual" in slot_map:
        residual = phi - assigned_centroids
        residual_norm = residual.norm(dim=-1, keepdim=True)
        residual = torch.where(residual_norm <= 1e-8, torch.zeros_like(residual), residual)
        residual = _normalize(residual)
        if residual.shape[-1] == evidence_dim:
            bank[:, slot_map["cluster_residual"], :] = residual.unsqueeze(1)
        elif projection == "none":
            raise ValueError("cluster residual dimension differs and evidence_projection=none")
        else:
            cache["cluster_residual_features"] = residual
    density_raw = torch.log((assigned_counts.float() / float(pool_size)).clamp_min(1e-8))
    rarity_raw = -torch.log((assigned_counts.float() / float(pool_size)).clamp_min(1e-8))
    if "cluster_density" in slot_map:
        cache["cluster_density_scalar"] = _standardize_scalar(density_raw).unsqueeze(1)
    if "cluster_rarity" in slot_map:
        cache["cluster_rarity_scalar"] = _standardize_scalar(rarity_raw).unsqueeze(1)
    cache["evidence_bank"] = bank
    cache["prototypes"] = bank
    cache["target_relative_cluster_ids"] = assignments.long()
    cache["target_relative_cluster_counts"] = assigned_counts.long()
    cache["target_evidence_finalized"] = True
    return cache


def _module_matrix_norm(module, name_contains=None):
    total = None
    for name, parameter in module.named_parameters():
        if name_contains is not None and name_contains not in name:
            continue
        if parameter.dim() < 2:
            continue
        value = parameter.detach().float().norm()
        total = value if total is None else total + value
    if total is None:
        return torch.tensor(0.0)
    return total


class _MixerBlock(nn.Module):
    def __init__(self, mixer_dim, num_ranks, num_slots, hidden_part, hidden_rank, hidden_channel):
        super().__init__()
        self.ln_slot = nn.LayerNorm(mixer_dim)
        self.slot_mlp = nn.Sequential(nn.Linear(num_slots, hidden_part), nn.GELU(), nn.Linear(hidden_part, num_slots))
        self.ln_rank = nn.LayerNorm(mixer_dim)
        self.rank_mlp = nn.Sequential(nn.Linear(num_ranks, hidden_rank), nn.GELU(), nn.Linear(hidden_rank, num_ranks))
        self.ln_channel = nn.LayerNorm(mixer_dim)
        self.channel_mlp = nn.Sequential(nn.Linear(mixer_dim, hidden_channel), nn.GELU(), nn.Linear(hidden_channel, mixer_dim))

    def forward(self, x, rank_mask=None):
        delta = self.slot_mlp(self.ln_slot(x).permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = x + delta
        if rank_mask is not None:
            x = x * rank_mask
        delta = self.rank_mlp(self.ln_rank(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + delta
        if rank_mask is not None:
            x = x * rank_mask
        x = x + self.channel_mlp(self.ln_channel(x))
        if rank_mask is not None:
            x = x * rank_mask
        return x


class RankPartQueryConditionedMixerAdapter(nn.Module):
    def __init__(self, active_dim, num_ranks, num_slots, args):
        super().__init__()
        self.active_dim = int(active_dim)
        self.num_ranks = int(num_ranks)
        self.num_slots = int(num_slots)
        mixer_dim = int(_arg(args, "mixer_dim", 256))
        self.mixer_dim = mixer_dim
        self.input_proj = nn.Linear(self.active_dim, mixer_dim)
        self.query_proj = nn.Linear(self.active_dim, mixer_dim)
        self.rank_embedding = nn.Parameter(torch.empty(1, self.num_ranks, 1, mixer_dim))
        self.slot_embedding = nn.Parameter(torch.empty(1, 1, self.num_slots, mixer_dim))
        nn.init.trunc_normal_(self.rank_embedding, std=0.02)
        nn.init.trunc_normal_(self.slot_embedding, std=0.02)
        self.input_norm = nn.LayerNorm(mixer_dim)
        self.film = nn.Sequential(nn.Linear(mixer_dim, mixer_dim), nn.GELU(), nn.Linear(mixer_dim, 2 * mixer_dim))
        self.blocks = nn.ModuleList([
            _MixerBlock(
                mixer_dim,
                self.num_ranks,
                self.num_slots,
                int(_arg(args, "mixer_hidden_part", 32)),
                int(_arg(args, "mixer_hidden_rank", 64)),
                int(_arg(args, "mixer_hidden_channel", 512)),
            )
            for _ in range(int(_arg(args, "mixer_depth", 2)))
        ])
        hidden_readout = int(_arg(args, "mixer_hidden_readout", 128))
        self.final_norm = nn.LayerNorm(mixer_dim)
        self.readout = nn.Sequential(nn.Linear(self.num_ranks * self.num_slots, hidden_readout), nn.GELU(), nn.Linear(hidden_readout, 1))
        self.output_proj = nn.Linear(mixer_dim, self.active_dim)
        self.last_diagnostics = {}

    def forward(self, query, evidence):
        if query.dim() != 2 or evidence.dim() != 4:
            raise ValueError("mixer expects query [B, D] and evidence [B, M, S, D]")
        if query.shape[0] != evidence.shape[0] or query.shape[1] != self.active_dim:
            raise ValueError("mixer query shape mismatch")
        if evidence.shape[-1] != self.active_dim or evidence.shape[2] != self.num_slots:
            raise ValueError("mixer evidence shape mismatch")
        if evidence.shape[1] > self.num_ranks:
            raise ValueError("evidence rank count exceeds configured top_m")
        actual_ranks = evidence.shape[1]
        rank_mask = None
        if actual_ranks < self.num_ranks:
            pad = evidence.new_zeros(evidence.shape[0], self.num_ranks - actual_ranks, self.num_slots, self.active_dim)
            evidence = torch.cat([evidence, pad], dim=1)
            rank_mask = evidence.new_zeros(1, self.num_ranks, 1, 1)
            rank_mask[:, :actual_ranks, :, :] = 1
        x = self.input_proj(evidence.float()) + self.rank_embedding + self.slot_embedding
        if rank_mask is not None:
            x = x * rank_mask
        q_m = self.query_proj(query.float())
        scale_raw, shift = self.film(q_m).chunk(2, dim=-1)
        scale = torch.tanh(scale_raw)
        x = self.input_norm(x) * (1 + scale[:, None, None, :]) + shift[:, None, None, :]
        if rank_mask is not None:
            x = x * rank_mask
        for block in self.blocks:
            x = block(x, rank_mask=rank_mask)
        h = self.final_norm(x)
        if rank_mask is not None:
            h = h * rank_mask
        h_flat = h.reshape(h.shape[0], self.num_ranks * self.num_slots, self.mixer_dim)
        readout_output = self.readout(h_flat.transpose(1, 2)).squeeze(-1)
        context = self.output_proj(readout_output)
        self.last_diagnostics = {
            "mixer/film_scale_mean": scale.mean().detach(),
            "mixer/film_scale_std": scale.std(unbiased=False).detach(),
            "mixer/film_shift_mean": shift.mean().detach(),
            "mixer/film_shift_std": shift.std(unbiased=False).detach(),
            "mixer/H_mean": h.mean().detach(),
            "mixer/H_std": h.std(unbiased=False).detach(),
            "mixer/H_flat_token_std": h_flat.std(unbiased=False).detach(),
            "mixer/readout_output_norm": readout_output.norm(dim=-1).mean().detach(),
            "mixer/rank_mixing_weight_norm": _module_matrix_norm(self, "rank_mlp").to(query.device),
            "mixer/part_mixing_weight_norm": _module_matrix_norm(self, "slot_mlp").to(query.device),
            "mixer/channel_mixing_weight_norm": _module_matrix_norm(self, "channel_mlp").to(query.device),
            "mixer/readout_weight_norm": _module_matrix_norm(self.readout, None).to(query.device),
        }
        return context


class TargetPrototypeEnricher(nn.Module):
    def __init__(self, args, global_dim, retrieval_dim=None, evidence_dim=None):
        super().__init__()
        validate_target_enrichment_args(args)
        self.args = args
        self.global_dim = int(global_dim)
        self.retrieval_dim = int(retrieval_dim or global_dim)
        self.evidence_dim = int(evidence_dim or global_dim)
        self.top_m = int(_arg(args, "top_m", 32))
        self.num_parts = int(_arg(args, "num_parts", 6))
        self.providers = parse_extractor_mode(_arg(args, "extractor_mode", "global,horizontal"))
        self.slot_map = build_slot_index_map(self.providers, self.num_parts)
        self.num_slots = evidence_slot_count(self.providers, self.num_parts)
        self.enrichment_space = normalize_enrichment_space(_arg(args, "enrichment_space", "global"))
        self.topm_rank_space = normalize_rank_space(_arg(args, "topm_rank_space", "host_global"))
        self.lambda_ret = float(_arg(args, "lambda_ret", 1.0))
        self.topm_rank_lambda = float(_arg(args, "topm_rank_lambda", 0.5))
        self.temperature = float(_arg(args, "temperature", 0.02))
        self.tau = float(_arg(args, "tau", 0.015))
        self.gate_mode = str(_arg(args, "gate_mode", _arg(args, "residual_gate", "residual"))).lower()
        self.gamma = _arg(args, "enrich_gamma", None)
        self.active_dim = self.global_dim if self.enrichment_space == "global" else self.retrieval_dim
        self.proto_to_global = nn.Identity() if self.evidence_dim == self.global_dim else nn.Linear(self.evidence_dim, self.global_dim)
        self.proto_to_retrieval = nn.Identity() if self.evidence_dim == self.retrieval_dim else nn.Linear(self.evidence_dim, self.retrieval_dim)
        self.raw_projectors = nn.ModuleDict()
        if "retrieval_backbone" in self.providers and self.retrieval_dim != self.evidence_dim:
            self.raw_projectors["retrieval_backbone"] = nn.Linear(self.retrieval_dim, self.evidence_dim)
        target_relative_space = str(_arg(args, "target_relative_space", "host_global")).lower()
        raw_dim = self.retrieval_dim if target_relative_space == "retrieval" else self.global_dim
        if "cluster" in self.providers and raw_dim != self.evidence_dim:
            self.raw_projectors["cluster"] = nn.Linear(raw_dim, self.evidence_dim)
        if "cluster_residual" in self.providers and raw_dim != self.evidence_dim:
            self.raw_projectors["cluster_residual"] = nn.Linear(raw_dim, self.evidence_dim)
        self.scalar_projectors = nn.ModuleDict()
        if "cluster_density" in self.providers:
            self.scalar_projectors["cluster_density"] = nn.Linear(1, self.evidence_dim)
        if "cluster_rarity" in self.providers:
            self.scalar_projectors["cluster_rarity"] = nn.Linear(1, self.evidence_dim)
        self.mixer = RankPartQueryConditionedMixerAdapter(self.active_dim, self.top_m, self.num_slots, args)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * self.active_dim, self.active_dim),
            nn.LayerNorm(self.active_dim),
            nn.GELU(),
            nn.Linear(self.active_dim, self.active_dim),
        )
        if self.gate_mode == "residual":
            hidden = int(_arg(args, "residual_gate_hidden_dim", 128))
            self.residual_gate = nn.Sequential(
                nn.Linear(3 * self.active_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            last = self.residual_gate[-1]
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, math.log(0.1 / 0.9))
        else:
            self.residual_gate = None
            self.gamma = float(self.gamma)

    def enrich_only(self, query_features, host_text_features, pool_cache, space=None, grab_text_features=None):
        out = self.forward(query_features, host_text_features, None, pool_cache, space=space, grab_text_features=grab_text_features)
        return out["enriched_features"]

    def _cache_tensor(self, pool_cache, key, device, dtype=torch.float32):
        # Target caches are detached feature banks. Moving them here keeps the
        # long-lived cache CPU-resident while preserving the same full-gallery math.
        return pool_cache[key].detach().to(device=device, dtype=dtype, non_blocking=True)

    def _gather_bank(self, bank, top_indices, trailing_shape):
        flat_indices = top_indices.reshape(-1)
        if bank.device == top_indices.device:
            gathered = bank.detach().index_select(0, flat_indices)
        else:
            # Gather large CPU banks before moving rows to GPU so evaluation does
            # not materialize the full evidence bank on the device for each task.
            gathered = bank.detach().cpu().index_select(0, flat_indices.detach().cpu())
            gathered = gathered.to(device=top_indices.device, non_blocking=True)
        return gathered.view(*top_indices.shape, *trailing_shape)

    def forward(self, query_features, host_text_features, query_pids, pool_cache, space=None, grab_text_features=None):
        space = normalize_enrichment_space(space or self.enrichment_space)
        if space != self.enrichment_space:
            raise ValueError(f"TargetPrototypeEnricher was built for {self.enrichment_space}, got {space}")
        for key in ["host_image_features", "retrieval_features", "pids"]:
            if key not in pool_cache:
                raise KeyError(f"target cache missing {key}")
        device = query_features.device
        host_images = _normalize(self._cache_tensor(pool_cache, "host_image_features", device))
        retrieval_images = _normalize(self._cache_tensor(pool_cache, "retrieval_features", device))
        evidence_bank = pool_cache.get("evidence_bank", pool_cache.get("prototypes"))
        if evidence_bank is None:
            raise KeyError("target cache missing evidence_bank/prototypes")
        evidence_bank = evidence_bank.detach()
        if evidence_bank.shape[1] != self.num_slots:
            raise ValueError(f"evidence slot count mismatch: expected {self.num_slots}, got {evidence_bank.shape[1]}")
        if any(provider in TARGET_RELATIVE_PROVIDERS for provider in self.providers) and not pool_cache.get("target_evidence_finalized", False):
            raise ValueError("target-relative evidence requires finalized target cache")
        pool_pids = self._cache_tensor(pool_cache, "pids", device, dtype=torch.long)
        q = _normalize(query_features)
        g = _normalize(host_text_features)
        top_indices = self._select_top_indices(q, g, host_images, retrieval_images, pool_cache, grab_text_features)
        gathered = self._gather_bank(evidence_bank, top_indices, (evidence_bank.shape[1], evidence_bank.shape[2])).float()
        gathered = self._apply_auxiliary_evidence(gathered, top_indices, pool_cache)
        selected = self._project_to_active_space(gathered, space)
        context = self.mixer(q, selected)
        interaction = q * context
        delta = self.fusion_mlp(torch.cat([q, context, interaction], dim=-1))
        if self.gate_mode == "static":
            gate = q.new_full((q.shape[0], 1), self.gamma)
        else:
            gate = torch.sigmoid(self.residual_gate(torch.cat([q, context, interaction], dim=-1)))
        enriched = _normalize(q + gate * delta)
        out = {"enriched_features": enriched, "top_indices": top_indices, "context": context, "gate": gate}
        diagnostics = self._diagnostics(q, enriched, context, delta, gate, top_indices, g, host_images, query_pids, pool_pids)
        diagnostics.update(self.mixer.last_diagnostics)
        diagnostics["mixer/context_norm"] = context.norm(dim=-1).mean().detach()
        diagnostics["mixer/context_delta_cosine"] = F.cosine_similarity(context, delta, dim=-1).mean().detach()
        diagnostics["mixer/output_delta_norm"] = delta.norm(dim=-1).mean().detach()
        out.update(diagnostics)
        if query_pids is not None:
            loss = self._target_retrieval_loss(enriched, retrieval_images, query_pids, pool_pids)
            out["target_retrieval_loss"] = loss
            out["total_loss"] = loss * self.lambda_ret
        return out

    def _select_top_indices(self, q, host_text, host_images, retrieval_images, pool_cache, grab_text_features):
        pool_size = host_images.shape[0]
        m_actual = min(self.top_m, pool_size)
        if "top_indices" in pool_cache:
            top_indices = pool_cache["top_indices"].to(q.device).long()
            if top_indices.dim() != 2 or top_indices.shape[0] != q.shape[0] or top_indices.shape[1] < 1:
                raise ValueError("top_indices must have shape [B, M>=1]")
            if top_indices.min() < 0 or top_indices.max() >= pool_size:
                raise ValueError("top_indices out of target-pool range")
            return top_indices[:, :m_actual]
        with torch.no_grad():
            if self.topm_rank_space == "host_global":
                scores = host_text @ host_images.t()
            elif self.topm_rank_space == "retrieval":
                if q.shape[-1] != retrieval_images.shape[-1]:
                    raise ValueError("retrieval top-M ranking dimension mismatch")
                scores = q @ retrieval_images.t()
            elif self.topm_rank_space == "hybrid_global_retrieval":
                alt_text = q if grab_text_features is None else _normalize(grab_text_features)
                alt_images = pool_cache.get("grab_image_features", pool_cache["retrieval_features"])
                alt_images = alt_images.detach().to(device=q.device, non_blocking=True)
                alt_images = _normalize(alt_images)
                if alt_text.shape[-1] != alt_images.shape[-1]:
                    raise ValueError("hybrid ranking dimension mismatch")
                scores = self.topm_rank_lambda * (host_text @ host_images.t())
                scores = scores + (1 - self.topm_rank_lambda) * (alt_text @ alt_images.t())
            else:
                raise ValueError(f"unsupported topm_rank_space: {self.topm_rank_space}")
            return torch.topk(scores, k=m_actual, dim=1, largest=True, sorted=True).indices

    def _apply_auxiliary_evidence(self, gathered, top_indices, pool_cache):
        result = gathered.clone()
        vector_keys = {
            "retrieval_backbone": "retrieval_backbone_features",
            "cluster": "cluster_features",
            "cluster_residual": "cluster_residual_features",
        }
        for provider, raw_key in vector_keys.items():
            if provider not in self.slot_map or raw_key not in pool_cache:
                continue
            raw_bank = pool_cache[raw_key]
            selected = self._gather_bank(raw_bank, top_indices, (raw_bank.shape[-1],)).float()
            if provider in self.raw_projectors:
                selected = self.raw_projectors[provider](selected)
            result[:, :, self.slot_map[provider], :] = _normalize(selected).unsqueeze(2)
        scalar_keys = {"cluster_density": "cluster_density_scalar", "cluster_rarity": "cluster_rarity_scalar"}
        for provider, scalar_key in scalar_keys.items():
            if provider not in self.slot_map:
                continue
            if scalar_key not in pool_cache:
                raise KeyError(f"target cache missing {scalar_key}")
            scalar_bank = pool_cache[scalar_key]
            selected = self._gather_bank(scalar_bank, top_indices, (1,)).float()
            projected = self.scalar_projectors[provider](selected)
            result[:, :, self.slot_map[provider], :] = _normalize(projected).unsqueeze(2)
        return result

    def _project_to_active_space(self, gathered, space):
        if space == "global":
            return _normalize(self.proto_to_global(gathered))
        if space == "retrieval":
            return _normalize(self.proto_to_retrieval(gathered))
        raise ValueError(f"unsupported enrichment space: {space}")

    def _target_retrieval_loss(self, enriched, retrieval_images, query_pids, pool_pids):
        query_pids = query_pids.to(enriched.device).view(-1)
        pool_pids = pool_pids.to(enriched.device).view(-1)
        positives = query_pids[:, None].eq(pool_pids[None, :])
        if not positives.any(dim=1).all():
            raise ValueError("every query must have at least one positive target image in the pool")
        logits = enriched @ retrieval_images.t() / max(self.tau, 1e-6)
        pos_logits = logits.masked_fill(~positives, float("-inf"))
        return -(torch.logsumexp(pos_logits, dim=1) - torch.logsumexp(logits, dim=1)).mean()

    def _diagnostics(self, q, enriched, context, delta, gate, top_indices, host_text, host_images, query_pids, pool_pids):
        diagnostics = {
            "target_raw_enriched_cosine": (q * enriched).sum(dim=-1).mean().detach(),
            "target_raw_context_cosine": F.cosine_similarity(q, _normalize(context), dim=-1).mean().detach(),
            "target_context_norm": context.norm(dim=-1).mean().detach(),
            "target_enrichment_shift_norm": (enriched - q).norm(dim=-1).mean().detach(),
            "target_residual_gate_mean": gate.mean().detach(),
            "target_residual_gate_std": gate.std(unbiased=False).detach(),
            "target_residual_gate_min": gate.min().detach(),
            "target_residual_gate_max": gate.max().detach(),
        }
        if query_pids is None or pool_pids is None:
            return diagnostics
        pool_pids = pool_pids.to(q.device).view(-1)
        query_pids = query_pids.to(q.device).view(-1)
        top_pids = pool_pids[top_indices]
        positives_pool = query_pids[:, None].eq(pool_pids[None, :])
        positives_top = top_pids.eq(query_pids[:, None])
        positive_in_pool = positives_pool.any(dim=1).float()
        positive_in_topm = positives_top.any(dim=1).float()
        num_positive_pool = positives_pool.sum(dim=1).float()
        num_positive_top = positives_top.sum(dim=1).float()
        host_scores = host_text @ host_images.t()
        selected_scores = torch.gather(host_scores, 1, top_indices)
        rank_ids = torch.arange(1, top_indices.shape[1] + 1, device=q.device).view(1, -1)
        absent_rank = torch.full((q.shape[0],), top_indices.shape[1] + 1, device=q.device, dtype=torch.long)
        first_rank = torch.where(positives_top, rank_ids.expand_as(positives_top), absent_rank[:, None]).min(dim=1).values.float()
        present = positive_in_topm.bool()
        diagnostics.update({
            "target_positive_in_pool_rate": positive_in_pool.mean().detach(),
            "target_positive_in_topm_rate": positive_in_topm.mean().detach(),
            "target_num_positive_in_pool": num_positive_pool.mean().detach(),
            "target_num_positive_in_topm": num_positive_top.mean().detach(),
            "target_host_topm_recall": (num_positive_top / num_positive_pool.clamp_min(1)).mean().detach(),
            "target_first_positive_rank": (first_rank[present].mean() if present.any() else first_rank.new_tensor(0.0)).detach(),
            "target_first_positive_rank_with_absent": first_rank.mean().detach(),
            "target_missing_topm_rate": (1 - positive_in_topm).mean().detach(),
            "target_host_top1_score": selected_scores[:, 0].mean().detach(),
            "target_host_topm_score": selected_scores.mean().detach(),
            "target_host_top1_topm_gap": (selected_scores[:, 0] - selected_scores[:, -1]).mean().detach(),
        })
        return diagnostics


def freeze_host_for_target_enrichment(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("target_enricher.")
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("freeze_host left zero trainable target_enricher parameters")
    return trainable



