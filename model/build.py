from collections import OrderedDict

import torch
import torch.nn as nn

from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
from .target_enrichment import (
    TargetPrototypeEnricher,
    build_evidence_bank,
    finalize_target_evidence_cache,
    freeze_host_for_target_enrichment,
    normalize_enrichment_space,
    normalize_rank_space,
    parse_extractor_mode,
    validate_target_enrichment_args,
)


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.target_enricher = None

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def init_target_enricher(self):
        validate_target_enrichment_args(self.args)
        self.target_enricher = TargetPrototypeEnricher(
            self.args,
            global_dim=self.embed_dim,
            retrieval_dim=self.embed_dim,
            evidence_dim=self.embed_dim,
        ).float()

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_clip_global_image(self, image):
        return self.encode_image(image)

    def encode_clip_global_text(self, text):
        return self.encode_text(text)

    def encode_retrieval_image(self, image):
        return self.encode_image(image)

    def encode_retrieval_text(self, text):
        return self.encode_text(text)

    def _patch_grid_size(self):
        visual = getattr(self.base_model, "visual", None)
        if hasattr(visual, "num_y") and hasattr(visual, "num_x"):
            return int(visual.num_y), int(visual.num_x)
        return None

    def encode_target_image_cache(self, images, cache_prototypes=True):
        image_tokens = self.base_model.encode_image(images)
        host_image_features = image_tokens[:, 0, :].float()
        cache = {"host_image_features": host_image_features}
        if not cache_prototypes:
            return cache

        retrieval_features = host_image_features
        cache["retrieval_features"] = retrieval_features
        if normalize_rank_space(getattr(self.args, "topm_rank_space", "host_global")) == "hybrid_global_retrieval":
            cache["grab_image_features"] = retrieval_features

        evidence_bank = build_evidence_bank(
            token_features=image_tokens.float(),
            num_parts=getattr(self.args, "num_parts", 6),
            grid_size=self._patch_grid_size(),
            mode=getattr(self.args, "extractor_mode", "global,horizontal"),
            retrieval_features=retrieval_features,
        )
        cache["evidence_bank"] = evidence_bank
        cache["prototypes"] = evidence_bank

        providers = parse_extractor_mode(getattr(self.args, "extractor_mode", "global,horizontal"))
        if "retrieval_backbone" in providers and retrieval_features.shape[-1] != evidence_bank.shape[-1]:
            if str(getattr(self.args, "evidence_projection", "auto")).lower() == "none":
                raise ValueError("retrieval_backbone dimension differs and evidence_projection=none")
            cache["retrieval_backbone_features"] = retrieval_features
        return cache

    def finalize_target_cache(self, cache):
        return finalize_target_evidence_cache(cache, self.args, evidence_dim=self.embed_dim)

    def enrich_text_features(self, query_features, host_text_features, target_cache, alt_text_features=None):
        if self.target_enricher is None:
            raise RuntimeError("target_enricher is not initialized")
        return self.target_enricher.enrich_only(
            query_features=query_features,
            host_text_features=host_text_features,
            pool_cache=target_cache,
            space=normalize_enrichment_space(getattr(self.args, "enrichment_space", "global")),
            grab_text_features=alt_text_features,
        )

    def forward(self, batch, epoch=None, current_step=None, target_cache=None):
        ret = dict()
        caption_ids = batch['caption_ids']
        use_target = target_cache is not None and self.target_enricher is not None
        use_host_loss = (not use_target) or bool(getattr(self.args, "use_host_loss", True))
        compute_images = use_host_loss or not bool(getattr(self.args, "pnp_text_only", False)) or not use_target

        image_feats = None
        i_feats = None
        if compute_images:
            images = batch['images']
            image_feats = self.base_model.encode_image(images)
            i_feats = image_feats[:, 0, :].float()

        text_feats_all = self.base_model.encode_text(caption_ids)
        t_feats = text_feats_all[torch.arange(text_feats_all.shape[0], device=text_feats_all.device), caption_ids.argmax(dim=-1)].float()
        text_for_loss = t_feats

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        target_loss = None
        if use_target:
            space = normalize_enrichment_space(getattr(self.args, "enrichment_space", "global"))
            query_features = t_feats
            target_ret = self.target_enricher(
                query_features=query_features,
                host_text_features=t_feats,
                grab_text_features=t_feats,
                query_pids=batch['pids'],
                pool_cache=target_cache,
                space=space,
            )
            text_for_loss = target_ret["enriched_features"]
            target_loss = target_ret["total_loss"]
            ret["target_enrichment_loss"] = target_loss
            ret["target_retrieval_loss"] = target_ret["target_retrieval_loss"].detach()
            ret["_loss_grad_sources"] = {
                "target_enrichment_loss": target_loss,
                "target_retrieval_loss": target_ret["target_retrieval_loss"],
            }
            for key, value in target_ret.items():
                if key.startswith("target_") or key.startswith("mixer/"):
                    ret[key] = value

        host_losses = []
        if use_host_loss:
            if i_feats is None:
                raise ValueError("host losses require online image features")
            if 'itc' in self.current_task:
                loss = objectives.compute_itc(i_feats, text_for_loss, logit_scale)
                ret.update({'itc_loss': loss})
                host_losses.append(loss)

            if 'sdm' in self.current_task:
                loss = objectives.compute_sdm(i_feats, text_for_loss, batch['pids'], logit_scale)
                ret.update({'sdm_loss': loss})
                host_losses.append(loss)

            if 'cmpm' in self.current_task:
                loss = objectives.compute_cmpm(i_feats, text_for_loss, batch['pids'])
                ret.update({'cmpm_loss': loss})
                host_losses.append(loss)

            if 'id' in self.current_task:
                image_logits = self.classifier(i_feats.half()).float()
                text_logits = self.classifier(text_for_loss.half()).float()
                loss = objectives.compute_id(image_logits, text_logits, batch['pids']) * self.args.id_loss_weight
                ret.update({'id_loss': loss})
                host_losses.append(loss)

                image_pred = torch.argmax(image_logits, dim=1)
                text_pred = torch.argmax(text_logits, dim=1)
                ret.update({'img_acc': (image_pred == batch['pids']).float().mean()})
                ret.update({'txt_acc': (text_pred == batch['pids']).float().mean()})

            if 'mlm' in self.current_task:
                if image_feats is None:
                    raise ValueError("mlm loss requires online image tokens")
                mlm_ids = batch['mlm_ids']
                mlm_feats = self.base_model.encode_text(mlm_ids)
                x = self.cross_former(mlm_feats, image_feats, image_feats)
                mlm_labels = batch['mlm_labels'].reshape(-1)
                x = x.reshape(-1, x.shape[-1])
                masked = mlm_labels != 0
                x = self.mlm_head(x[masked])
                scores = x.float().reshape(-1, self.args.vocab_size)
                mlm_labels = mlm_labels[masked]
                loss = objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight
                ret.update({'mlm_loss': loss})
                host_losses.append(loss)
                pred = scores.max(1)[1]
                ret.update({'mlm_acc': (pred == mlm_labels).float().mean()})

        host_loss = sum(host_losses) if host_losses else t_feats.new_tensor(0.0)
        if use_target and use_host_loss:
            ret["host_loss"] = host_loss
        total_loss = host_loss if use_host_loss else t_feats.new_tensor(0.0)
        if use_target and use_host_loss:
            total_loss = float(getattr(self.args, "lambda_host", 1.0)) * host_loss
        if target_loss is not None:
            total_loss = total_loss + target_loss
        ret["loss"] = total_loss
        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    convert_weights(model)
    if bool(getattr(args, "target_enrichment", False)):
        model.init_target_enricher()
        if bool(getattr(args, "freeze_host", False)):
            freeze_host_for_target_enrichment(model)
    return model
