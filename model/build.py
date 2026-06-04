from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
from .target_enrichment import TargetPrototypeEnricher, build_part_prototypes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.target_grid_size = self._infer_visual_grid_size()

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

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

        self.target_enricher = None
        if getattr(args, "target_enrichment", False):
            self.target_enricher = TargetPrototypeEnricher(self.embed_dim, args)
            if getattr(args, "freeze_host", False):
                self._freeze_host_parameters()

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def _infer_visual_grid_size(self):
        visual = getattr(self.base_model, "visual", None)
        if hasattr(visual, "num_y") and hasattr(visual, "num_x"):
            return (visual.num_y, visual.num_x)
        return None

    def _freeze_host_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("target_enricher."):
                parameter.requires_grad = False
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        if x.dim() == 3:
            return x[:, 0, :].float()
        return x.float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    @torch.no_grad()
    def encode_target_image_cache(self, images, cache_prototypes=True):
        token_features = self.base_model.encode_image(images)
        if token_features.dim() == 2:
            token_features = token_features.unsqueeze(1)
        token_features = token_features.float()
        image_features = token_features[:, 0, :]
        ret = {
            "host_image_features": image_features,
            "retrieval_features": image_features,
        }
        if cache_prototypes:
            ret["prototypes"] = build_part_prototypes(
                token_features,
                num_parts=self.args.num_parts,
                grid_size=self.target_grid_size,
                mode=self.args.extractor_mode,
            )
        return ret

    def _zero_loss(self, reference):
        return reference.float().sum() * 0.0

    def _copy_target_outputs(self, ret, target_ret):
        for key, value in target_ret.items():
            if key in ("enriched_features", "top_indices", "total_loss"):
                continue
            ret[key] = value

    def forward(self, batch, epoch=None, current_step=None, target_cache=None):
        ret = dict()

        caption_ids = batch['caption_ids']
        target_active = getattr(self.args, "target_enrichment", False) and target_cache is not None
        use_host_loss = (not getattr(self.args, "target_enrichment", False)) or getattr(self.args, "use_host_loss", True)
        pnp_text_only = target_active and getattr(self.args, "pnp_text_only", False)

        image_feats = None
        i_feats = None
        if not pnp_text_only and "images" in batch and (use_host_loss or not target_active):
            images = batch['images']
            image_feats, text_feats = self.base_model(images, caption_ids)
            if image_feats.dim() == 3:
                i_feats = image_feats[:, 0, :].float()
            else:
                i_feats = image_feats.float()
        else:
            text_feats = self.base_model.encode_text(caption_ids)
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})
        host_loss = self._zero_loss(t_feats)

        if use_host_loss:
            if i_feats is None:
                raise ValueError("Host loss requires image features; disable pnp_text_only or host loss")

            if 'itc' in self.current_task:
                ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
                host_loss = host_loss + ret['itc_loss']
            
            if 'sdm' in self.current_task:
                ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})
                host_loss = host_loss + ret['sdm_loss']

            if 'cmpm' in self.current_task:
                ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
                host_loss = host_loss + ret['cmpm_loss']
            
            if 'id' in self.current_task:
                image_logits = self.classifier(i_feats.half()).float()
                text_logits = self.classifier(t_feats.half()).float()
                ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})
                host_loss = host_loss + ret['id_loss']

                image_pred = torch.argmax(image_logits, dim=1)
                text_pred = torch.argmax(text_logits, dim=1)

                image_precision = (image_pred == batch['pids']).float().mean()
                text_precision = (text_pred == batch['pids']).float().mean()
                ret.update({'img_acc': image_precision})
                ret.update({'txt_acc': text_precision})
            
            if 'mlm' in self.current_task:
                mlm_ids = batch['mlm_ids']

                mlm_feats = self.base_model.encode_text(mlm_ids)

                x = self.cross_former(mlm_feats, image_feats, image_feats)
                mlm_labels = batch['mlm_labels'].reshape(-1)
                x = x.reshape(-1, x.shape[-1])
                masked = mlm_labels != 0

                x = self.mlm_head(x[masked])
                scores = x.float().reshape(-1, self.args.vocab_size)
                mlm_labels = mlm_labels[masked]
                ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})
                host_loss = host_loss + ret['mlm_loss']

                pred = scores.max(1)[1]
                acc = (pred == mlm_labels).float().mean()
                ret.update({'mlm_acc': acc})

        target_enrichment_loss = self._zero_loss(t_feats)
        if target_active:
            target_ret = self.target_enricher(
                query_features=t_feats,
                host_text_features=t_feats,
                query_pids=batch["pids"],
                pool_cache=target_cache,
                space=getattr(self.args, "enrichment_space", "global"),
            )
            target_enrichment_loss = target_ret["total_loss"]
            ret["target_enrichment_loss"] = target_enrichment_loss
            self._copy_target_outputs(ret, target_ret)

        ret["host_loss"] = host_loss
        if getattr(self.args, "target_enrichment", False):
            host_weight = self.args.lambda_host if use_host_loss else 0.0
            ret["loss"] = host_weight * host_loss + target_enrichment_loss
        else:
            ret["loss"] = host_loss

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    if getattr(model, "target_enricher", None) is not None:
        model.target_enricher.float()
    return model
