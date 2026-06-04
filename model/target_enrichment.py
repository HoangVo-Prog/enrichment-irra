import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_MODE_ORDER = ("global", "horizontal", "vertical", "grid")
_ALIASES = {
    "global_horizontal": "global,horizontal",
    "global_vertical": "global,vertical",
    "global_grid": "global,grid",
}


def parse_extractor_modes(mode):
    mode = _ALIASES.get(mode, mode)
    modes = [m.strip().lower() for m in mode.split(",") if m.strip()]
    if not modes:
        raise ValueError("extractor_mode must contain at least one mode")
    unknown = [m for m in modes if m not in _MODE_ORDER]
    if unknown:
        raise ValueError("Unsupported extractor mode(s): {}".format(", ".join(unknown)))
    return [m for m in _MODE_ORDER if m in set(modes)]


def count_prototype_slots(mode, num_parts):
    if num_parts <= 0:
        raise ValueError("num_parts must be positive")
    total = 0
    for extractor in parse_extractor_modes(mode):
        if extractor == "global":
            total += 1
        elif extractor in ("horizontal", "vertical"):
            total += num_parts
        elif extractor == "grid":
            total += num_parts * num_parts
    return total


def _balanced_slices(length, parts):
    bounds = []
    for idx in range(parts):
        start = int(math.floor(float(idx) * length / parts))
        end = int(math.floor(float(idx + 1) * length / parts))
        bounds.append((start, end))
    return bounds


def _validate_grid(num_patches, grid_size):
    if grid_size is None:
        raise ValueError("grid_size is required for vertical or grid prototypes")
    grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
    if grid_h <= 0 or grid_w <= 0 or grid_h * grid_w != num_patches:
        raise ValueError(
            "grid_size {} is incompatible with {} patches".format(
                grid_size, num_patches
            )
        )
    return grid_h, grid_w


def _mean_or_zero(tokens):
    if tokens.shape[1] == 0:
        return tokens.new_zeros(tokens.shape[0], tokens.shape[-1])
    return tokens.mean(dim=1)


def build_part_prototypes(token_features, num_parts=6, grid_size=None, mode="global,horizontal"):
    """Build normalized fixed-slot visual prototypes from CLIP image tokens."""
    if token_features.dim() != 3 or token_features.shape[1] < 1:
        raise ValueError("token_features must have shape [B, 1 + patches, D]")
    if num_parts <= 0:
        raise ValueError("num_parts must be positive")

    modes = parse_extractor_modes(mode)
    global_token = token_features[:, :1, :]
    patches = token_features[:, 1:, :]
    num_patches = patches.shape[1]
    slots = []

    for extractor in modes:
        if extractor == "global":
            slots.append(global_token)
        elif extractor == "horizontal":
            if grid_size is not None and num_patches > 0:
                grid_h, grid_w = _validate_grid(num_patches, grid_size)
                grid = patches.reshape(patches.shape[0], grid_h, grid_w, patches.shape[-1])
                part_slots = []
                for start, end in _balanced_slices(grid_h, num_parts):
                    part_slots.append(_mean_or_zero(grid[:, start:end, :, :].reshape(patches.shape[0], -1, patches.shape[-1])))
                slots.append(torch.stack(part_slots, dim=1))
            else:
                part_slots = []
                for start, end in _balanced_slices(num_patches, num_parts):
                    part_slots.append(_mean_or_zero(patches[:, start:end, :]))
                slots.append(torch.stack(part_slots, dim=1))
        elif extractor == "vertical":
            grid_h, grid_w = _validate_grid(num_patches, grid_size)
            grid = patches.reshape(patches.shape[0], grid_h, grid_w, patches.shape[-1])
            part_slots = []
            for start, end in _balanced_slices(grid_w, num_parts):
                part_slots.append(_mean_or_zero(grid[:, :, start:end, :].reshape(patches.shape[0], -1, patches.shape[-1])))
            slots.append(torch.stack(part_slots, dim=1))
        elif extractor == "grid":
            grid_h, grid_w = _validate_grid(num_patches, grid_size)
            grid = patches.reshape(patches.shape[0], grid_h, grid_w, patches.shape[-1])
            part_slots = []
            for row_start, row_end in _balanced_slices(grid_h, num_parts):
                for col_start, col_end in _balanced_slices(grid_w, num_parts):
                    part_slots.append(
                        _mean_or_zero(
                            grid[:, row_start:row_end, col_start:col_end, :].reshape(
                                patches.shape[0], -1, patches.shape[-1]
                            )
                        )
                    )
            slots.append(torch.stack(part_slots, dim=1))

    prototypes = torch.cat(slots, dim=1)
    return F.normalize(prototypes.float(), p=2, dim=-1)


def _scalar_stat(value, default=None):
    if value is None:
        return default
    if value.numel() == 0:
        return value.new_tensor(0.0)
    return value.float().mean()


class _RankPartMixerBlock(nn.Module):
    def __init__(self, top_m, num_slots, dim, hidden_part, hidden_rank, hidden_channel):
        super().__init__()
        self.part_norm = nn.LayerNorm(num_slots)
        self.part_mlp = nn.Sequential(
            nn.Linear(num_slots, hidden_part),
            nn.GELU(),
            nn.Linear(hidden_part, num_slots),
        )
        self.rank_norm = nn.LayerNorm(top_m)
        self.rank_mlp = nn.Sequential(
            nn.Linear(top_m, hidden_rank),
            nn.GELU(),
            nn.Linear(hidden_rank, top_m),
        )
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, hidden_channel),
            nn.GELU(),
            nn.Linear(hidden_channel, dim),
        )

    def forward(self, x):
        part_x = x.permute(0, 1, 3, 2)
        part_x = self.part_mlp(self.part_norm(part_x)).permute(0, 1, 3, 2)
        x = x + part_x

        rank_x = x.permute(0, 2, 3, 1)
        rank_x = self.rank_mlp(self.rank_norm(rank_x)).permute(0, 3, 1, 2)
        x = x + rank_x

        x = x + self.channel_mlp(self.channel_norm(x))
        return x


class RankPartQueryConditionedMixerAdapter(nn.Module):
    def __init__(
        self,
        feature_dim,
        top_m,
        num_slots,
        mixer_dim=256,
        mixer_depth=2,
        mixer_hidden_part=128,
        mixer_hidden_rank=128,
        mixer_hidden_channel=512,
        mixer_hidden_readout=512,
        context_pooling="mlp",
    ):
        super().__init__()
        if top_m <= 0:
            raise ValueError("top_m must be positive")
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if context_pooling not in ("mlp", "late_attention", "hybrid_attention"):
            raise ValueError("Unsupported context_pooling: {}".format(context_pooling))

        self.feature_dim = feature_dim
        self.top_m = top_m
        self.num_slots = num_slots
        self.mixer_dim = mixer_dim
        self.context_pooling = context_pooling

        self.w_in = nn.Linear(feature_dim, mixer_dim)
        self.w_q = nn.Linear(feature_dim, mixer_dim)
        self.rank_embed = nn.Parameter(torch.zeros(top_m, mixer_dim))
        self.part_embed = nn.Parameter(torch.zeros(num_slots, mixer_dim))
        self.film = nn.Linear(mixer_dim, 2 * mixer_dim)
        self.blocks = nn.ModuleList(
            [
                _RankPartMixerBlock(
                    top_m,
                    num_slots,
                    mixer_dim,
                    mixer_hidden_part,
                    mixer_hidden_rank,
                    mixer_hidden_channel,
                )
                for _ in range(mixer_depth)
            ]
        )
        self.final_norm = nn.LayerNorm(mixer_dim)
        self.readout = nn.Sequential(
            nn.Linear(top_m * num_slots * mixer_dim, mixer_hidden_readout),
            nn.GELU(),
            nn.Linear(mixer_hidden_readout, mixer_dim),
        )
        self.attn_proj = nn.Linear(mixer_dim, mixer_dim)
        self.hybrid_gate = nn.Linear(3 * mixer_dim, 1)
        self.w_out = nn.Linear(mixer_dim, feature_dim)
        self.last_diagnostics = {}

        nn.init.normal_(self.rank_embed, std=0.02)
        nn.init.normal_(self.part_embed, std=0.02)

    def _pad_rank(self, prototypes):
        actual_m = prototypes.shape[1]
        if actual_m > self.top_m:
            return prototypes[:, : self.top_m, :, :]
        if actual_m == self.top_m:
            return prototypes
        pad = prototypes.new_zeros(
            prototypes.shape[0], self.top_m - actual_m, prototypes.shape[2], prototypes.shape[3]
        )
        return torch.cat([prototypes, pad], dim=1)

    def _attention_pool(self, x, query_token):
        flat = x.reshape(x.shape[0], self.top_m * self.num_slots, self.mixer_dim)
        scores = (flat * query_token.unsqueeze(1)).sum(dim=-1) / math.sqrt(float(self.mixer_dim))
        weights = torch.softmax(scores, dim=1)
        pooled = (weights.unsqueeze(-1) * flat).sum(dim=1)
        pooled = self.attn_proj(pooled)
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=1)
        slot_weights = weights.reshape(weights.shape[0], self.top_m, self.num_slots)
        diagnostics = {
            "mixer/attn_entropy": entropy.mean(),
            "mixer/attn_top1_mass": weights.max(dim=1)[0].mean(),
            "mixer/attn_rank1_mass": weights[:, : self.num_slots].sum(dim=1).mean(),
            "mixer/attn_slot0_mass": slot_weights[:, :, 0].sum(dim=1).mean(),
            "mixer/attn_non_slot0_mass": slot_weights[:, :, 1:].sum(dim=(1, 2)).mean()
            if self.num_slots > 1
            else weights.new_tensor(0.0),
            "mixer/attention_pool_output_norm": pooled.norm(dim=1).mean(),
            "mixer/attention_pool_weight_norm": self.attn_proj.weight.float().norm(),
        }
        return pooled, diagnostics

    def forward(self, prototypes, query_features):
        if prototypes.dim() != 4:
            raise ValueError("prototypes must have shape [B, M, S, D]")
        if prototypes.shape[2] != self.num_slots:
            raise ValueError(
                "Expected {} prototype slots, got {}".format(self.num_slots, prototypes.shape[2])
            )

        prototypes = self._pad_rank(prototypes.float())
        query_features = query_features.float()
        x = self.w_in(prototypes)
        x = x + self.rank_embed.view(1, self.top_m, 1, self.mixer_dim)
        x = x + self.part_embed.view(1, 1, self.num_slots, self.mixer_dim)

        q = self.w_q(query_features)
        film_scale, film_shift = self.film(q).chunk(2, dim=-1)
        x = x * (1.0 + film_scale.view(-1, 1, 1, self.mixer_dim))
        x = x + film_shift.view(-1, 1, 1, self.mixer_dim)

        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        flat = x.reshape(x.shape[0], -1)
        mlp_context = self.readout(flat)
        diagnostics = {
            "mixer/context_norm": mlp_context.norm(dim=1).mean(),
            "mixer/rank_mixing_weight_norm": torch.stack([b.rank_mlp[0].weight.float().norm() for b in self.blocks]).mean()
            if len(self.blocks) > 0
            else x.new_tensor(0.0),
            "mixer/part_mixing_weight_norm": torch.stack([b.part_mlp[0].weight.float().norm() for b in self.blocks]).mean()
            if len(self.blocks) > 0
            else x.new_tensor(0.0),
            "mixer/channel_mixing_weight_norm": torch.stack([b.channel_mlp[0].weight.float().norm() for b in self.blocks]).mean()
            if len(self.blocks) > 0
            else x.new_tensor(0.0),
            "mixer/readout_weight_norm": self.readout[0].weight.float().norm(),
            "mixer/film_scale_mean": film_scale.mean(),
            "mixer/film_scale_std": film_scale.std(unbiased=False),
            "mixer/film_shift_mean": film_shift.mean(),
            "mixer/film_shift_std": film_shift.std(unbiased=False),
            "mixer/H_mean": x.mean(),
            "mixer/H_std": x.std(unbiased=False),
            "mixer/H_flat_token_std": x.reshape(x.shape[0], -1).std(dim=1, unbiased=False).mean(),
            "mixer/readout_output_norm": mlp_context.norm(dim=1).mean(),
        }

        if self.context_pooling == "mlp":
            context = mlp_context
        else:
            attn_context, attn_diag = self._attention_pool(x, q)
            diagnostics.update(attn_diag)
            if self.context_pooling == "late_attention":
                context = attn_context
            else:
                gate = torch.sigmoid(self.hybrid_gate(torch.cat([mlp_context, attn_context, q], dim=1)))
                context = gate * attn_context + (1.0 - gate) * mlp_context
                diagnostics.update(
                    {
                        "mixer/hybrid_pool_gate_mean": gate.mean(),
                        "mixer/hybrid_pool_gate_std": gate.std(unbiased=False),
                        "mixer/hybrid_pool_gate_min": gate.min(),
                        "mixer/hybrid_pool_gate_max": gate.max(),
                        "mixer/mlp_pool_output_norm": mlp_context.norm(dim=1).mean(),
                        "mixer/hybrid_pool_gate_weight_norm": self.hybrid_gate.weight.float().norm(),
                    }
                )

        context = self.w_out(context)
        diagnostics["mixer/context_delta_cosine"] = F.cosine_similarity(
            F.normalize(context, p=2, dim=1), F.normalize(query_features, p=2, dim=1), dim=1
        ).mean()
        diagnostics["mixer/output_delta_norm"] = context.norm(dim=1).mean()
        self.last_diagnostics = diagnostics
        return context


class TargetPrototypeEnricher(nn.Module):
    def __init__(self, feature_dim, args):
        super().__init__()
        if getattr(args, "enrichment_space", "global") not in ("global", "default"):
            raise ValueError("IRRA target enrichment supports only global/default enrichment_space")

        self.feature_dim = feature_dim
        self.top_m = args.top_m
        self.num_slots = count_prototype_slots(args.extractor_mode, args.num_parts)
        self.use_target_retrieval_loss = args.use_target_retrieval_loss
        self.use_target_robust_loss = args.use_target_robust_loss
        self.lambda_ret = args.lambda_ret
        self.lambda_rob = args.lambda_rob
        self.lambda_gain = args.lambda_gain
        self.gain_margin = args.gain_margin
        self.tau = args.tau
        self.robust_hard_k = args.robust_hard_k
        self.residual_gate = args.residual_gate
        self.enrich_gamma = args.enrich_gamma

        self.mixer = RankPartQueryConditionedMixerAdapter(
            feature_dim=feature_dim,
            top_m=args.top_m,
            num_slots=self.num_slots,
            mixer_dim=args.mixer_dim,
            mixer_depth=args.mixer_depth,
            mixer_hidden_part=args.mixer_hidden_part,
            mixer_hidden_rank=args.mixer_hidden_rank,
            mixer_hidden_channel=args.mixer_hidden_channel,
            mixer_hidden_readout=args.mixer_hidden_readout,
            context_pooling=args.context_pooling,
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        if self.residual_gate == "residual":
            hidden = args.residual_gate_hidden_dim
            self.gate_mlp = nn.Sequential(
                nn.Linear(3 * feature_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.constant_(self.gate_mlp[-1].bias, math.log(0.1 / 0.9))
        elif self.residual_gate == "static":
            self.gate_mlp = None
            if self.enrich_gamma is None:
                raise ValueError("residual_gate=static requires enrich_gamma")
        else:
            raise ValueError("Unsupported residual_gate: {}".format(self.residual_gate))

    def _zero(self, like):
        return like.sum() * 0.0

    def _cache_tensor(self, cache, key, device):
        if key not in cache:
            raise KeyError("target cache missing '{}'".format(key))
        return cache[key].to(device)

    def _select_topm(self, host_text_features, pool_host_features, pool_cache):
        batch_size = host_text_features.shape[0]
        pool_size = pool_host_features.shape[0]
        k = min(self.top_m, pool_size)
        if "top_indices" in pool_cache and pool_cache["top_indices"] is not None:
            top_indices = pool_cache["top_indices"].to(host_text_features.device).long()
            if top_indices.dim() != 2 or top_indices.shape[0] != batch_size:
                raise ValueError("precomputed top_indices must have shape [B, K]")
            if top_indices.numel() > 0:
                if top_indices.min().item() < 0 or top_indices.max().item() >= pool_size:
                    raise ValueError("precomputed top_indices are out of range")
            top_indices = top_indices[:, :k]
            host_scores = F.normalize(host_text_features.float(), p=2, dim=1) @ F.normalize(
                pool_host_features.float(), p=2, dim=1
            ).t()
            return top_indices, host_scores

        host_scores = F.normalize(host_text_features.float(), p=2, dim=1) @ F.normalize(
            pool_host_features.float(), p=2, dim=1
        ).t()
        top_indices = host_scores.topk(k=k, dim=1, largest=True, sorted=True).indices
        return top_indices, host_scores

    def _target_retrieval_loss(self, enriched, retrieval_features, positive_mask):
        zero = self._zero(enriched)
        if not self.use_target_retrieval_loss:
            return zero, positive_mask.any(dim=1)
        valid = positive_mask.any(dim=1)
        if not bool(valid.any()):
            return zero, valid

        scores = enriched @ retrieval_features.t() / self.tau
        neg_inf = torch.finfo(scores.dtype).min
        pos_lse = scores.masked_fill(~positive_mask, neg_inf).logsumexp(dim=1)
        all_lse = scores.logsumexp(dim=1)
        return -(pos_lse[valid] - all_lse[valid]).mean(), valid

    def _robust_loss(self, raw_query, enriched, retrieval_features, positive_mask, top_indices):
        zero = self._zero(enriched)
        if not self.use_target_robust_loss:
            return zero, zero, zero, {
                "target_valid_robust_rate": zero.detach(),
                "target_reliable_robust_rate": zero.detach(),
            }

        raw_scores = raw_query @ retrieval_features.t()
        enriched_scores = enriched @ retrieval_features.t()
        negative_mask = ~positive_mask
        valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not bool(valid.any()):
            return zero, zero, zero, {
                "target_valid_robust_rate": valid.float().mean(),
                "target_reliable_robust_rate": valid.float().mean(),
            }

        hard_k = max(1, min(self.robust_hard_k, retrieval_features.shape[0]))
        neg_inf = torch.finfo(raw_scores.dtype).min
        hard_scores = raw_scores.masked_fill(~negative_mask, neg_inf)
        hard_indices = hard_scores.topk(k=hard_k, dim=1, largest=True, sorted=False).indices
        hard_mask = torch.zeros_like(positive_mask)
        hard_mask.scatter_(1, hard_indices, True)
        hard_mask = hard_mask & negative_mask

        raw_pos = raw_scores.masked_fill(~positive_mask, neg_inf).logsumexp(dim=1)
        raw_neg = raw_scores.masked_fill(~hard_mask, neg_inf).logsumexp(dim=1)
        enriched_pos = enriched_scores.masked_fill(~positive_mask, neg_inf).logsumexp(dim=1)
        enriched_neg = enriched_scores.masked_fill(~hard_mask, neg_inf).logsumexp(dim=1)

        raw_margin = raw_pos - raw_neg
        enriched_margin = enriched_pos - enriched_neg
        margin_gain = enriched_margin - raw_margin
        guard_loss = F.relu(raw_margin - enriched_margin)[valid].mean()

        positive_in_topm = positive_mask.gather(1, top_indices).any(dim=1)
        reliable = valid & positive_in_topm
        if bool(reliable.any()):
            gain_loss = F.relu(self.gain_margin - margin_gain)[reliable].mean()
        else:
            gain_loss = zero
        robust_loss = guard_loss + self.lambda_gain * gain_loss

        diagnostics = {
            "target_valid_robust_rate": valid.float().mean(),
            "target_raw_margin": raw_margin[valid].mean(),
            "target_enriched_margin": enriched_margin[valid].mean(),
            "target_margin_gain": margin_gain[valid].mean(),
            "target_guard_violation_rate": (raw_margin[valid] > enriched_margin[valid]).float().mean(),
            "target_reliable_robust_rate": reliable.float().mean(),
            "target_margin_gain_reliable": margin_gain[reliable].mean() if bool(reliable.any()) else zero.detach(),
            "target_gain_satisfied_rate": (margin_gain[reliable] >= self.gain_margin).float().mean()
            if bool(reliable.any())
            else zero.detach(),
        }
        return robust_loss, guard_loss, gain_loss, diagnostics

    def forward(self, query_features, host_text_features, query_pids, pool_cache, space="global"):
        if space not in ("global", "default"):
            raise ValueError("IRRA target enrichment supports only global/default enrichment_space")
        device = query_features.device
        raw_query = F.normalize(query_features.float(), p=2, dim=1)
        host_text_features = host_text_features.float()
        query_pids = query_pids.to(device).long()

        pool_host = self._cache_tensor(pool_cache, "host_image_features", device).float()
        retrieval = F.normalize(self._cache_tensor(pool_cache, "retrieval_features", device).float(), p=2, dim=1)
        prototypes = self._cache_tensor(pool_cache, "prototypes", device).float()
        pool_pids = self._cache_tensor(pool_cache, "pids", device).long()
        if prototypes.shape[1] != self.num_slots:
            raise ValueError(
                "Target cache prototype slot count {} does not match expected {}".format(
                    prototypes.shape[1], self.num_slots
                )
            )

        top_indices, host_scores = self._select_topm(host_text_features, pool_host, pool_cache)
        selected_prototypes = prototypes[top_indices]
        context = self.mixer(selected_prototypes, raw_query)
        context = F.normalize(context, p=2, dim=1)

        fusion_input = torch.cat([raw_query, context, raw_query * context], dim=1)
        delta = self.fusion(fusion_input.float())
        if self.residual_gate == "static":
            gate = raw_query.new_full((raw_query.shape[0], 1), float(self.enrich_gamma))
        else:
            gate = torch.sigmoid(self.gate_mlp(fusion_input.float()))
        enriched = F.normalize(raw_query + gate * delta, p=2, dim=1)

        positive_mask = query_pids.view(-1, 1).eq(pool_pids.view(1, -1))
        target_retrieval_loss, valid_retrieval = self._target_retrieval_loss(
            enriched, retrieval, positive_mask
        )
        robust_loss, guard_loss, gain_loss, robust_diag = self._robust_loss(
            raw_query, enriched, retrieval, positive_mask, top_indices
        )
        total_loss = self.lambda_ret * target_retrieval_loss + self.lambda_rob * robust_loss

        selected_positive = positive_mask.gather(1, top_indices)
        first_positive = selected_positive.float().argmax(dim=1).float() + 1.0
        absent_value = top_indices.shape[1] + 1.0
        first_positive = torch.where(
            selected_positive.any(dim=1), first_positive, first_positive.new_full(first_positive.shape, absent_value)
        )
        top_scores = host_scores.gather(1, top_indices)
        num_positive_in_pool = positive_mask.float().sum(dim=1)
        num_positive_in_topm = selected_positive.float().sum(dim=1)
        host_topm_recall = torch.where(
            num_positive_in_pool > 0,
            num_positive_in_topm / num_positive_in_pool.clamp_min(1.0),
            num_positive_in_pool.new_zeros(num_positive_in_pool.shape),
        )

        ret = {
            "enriched_features": enriched,
            "top_indices": top_indices,
            "target_retrieval_loss": target_retrieval_loss,
            "target_robust_loss": robust_loss,
            "target_guard_loss": guard_loss,
            "target_gain_loss": gain_loss,
            "total_loss": total_loss,
            "target_positive_in_pool_rate": positive_mask.any(dim=1).float().mean(),
            "target_positive_in_topm_rate": selected_positive.any(dim=1).float().mean(),
            "target_num_positive_in_pool": num_positive_in_pool.mean(),
            "target_num_positive_in_topm": num_positive_in_topm.mean(),
            "target_host_topm_recall": host_topm_recall.mean(),
            "target_first_positive_rank": first_positive[selected_positive.any(dim=1)].mean()
            if bool(selected_positive.any())
            else self._zero(enriched).detach(),
            "target_first_positive_rank_with_absent": first_positive.mean(),
            "target_missing_topm_rate": (~selected_positive.any(dim=1)).float().mean(),
            "target_host_top1_score": top_scores[:, 0].mean(),
            "target_host_topm_score": top_scores.mean(),
            "target_host_top1_topm_gap": (top_scores[:, 0] - top_scores[:, -1]).mean(),
            "target_raw_enriched_cosine": F.cosine_similarity(raw_query, enriched, dim=1).mean(),
            "target_raw_context_cosine": F.cosine_similarity(raw_query, context, dim=1).mean(),
            "target_context_norm": context.norm(dim=1).mean(),
            "target_enrichment_shift_norm": (enriched - raw_query).norm(dim=1).mean(),
            "target_residual_gate_mean": gate.mean(),
            "target_residual_gate_std": gate.std(unbiased=False),
            "target_residual_gate_min": gate.min(),
            "target_residual_gate_max": gate.max(),
            "target_valid_retrieval_rate": valid_retrieval.float().mean(),
        }
        ret.update(robust_diag)
        ret.update(self.mixer.last_diagnostics)
        return ret
