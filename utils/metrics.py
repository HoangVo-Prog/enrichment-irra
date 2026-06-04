from utils.table import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    max_rank = min(max_rank, similarity.shape[1])
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader, args=None):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.args = args
        self.logger = logging.getLogger("IRRA.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids

    def _compute_target_cache(self, model):
        model = model.eval()
        device = next(model.parameters()).device
        pids, image_ids, host_features, retrieval_features, prototypes = [], [], [], [], []
        running_index = 0
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                encoded = model.encode_target_image_cache(img, cache_prototypes=True)
            batch_size = img.shape[0]
            pids.append(pid.to(device).view(-1))
            image_ids.append(torch.arange(running_index, running_index + batch_size, device=device))
            running_index += batch_size
            host_features.append(encoded["host_image_features"])
            retrieval_features.append(encoded["retrieval_features"])
            prototypes.append(encoded["prototypes"])
        return {
            "host_image_features": torch.cat(host_features, 0),
            "retrieval_features": torch.cat(retrieval_features, 0),
            "prototypes": torch.cat(prototypes, 0),
            "pids": torch.cat(pids, 0).long(),
            "image_ids": torch.cat(image_ids, 0).long(),
        }

    def _enrich_text_features(self, model, qfeats, qids, target_cache):
        args = self.args or getattr(model, "args", None)
        batch_size = getattr(args, "test_batch_size", 512)
        device = qfeats.device
        enriched = []
        with torch.no_grad():
            for start in range(0, qfeats.shape[0], batch_size):
                end = min(start + batch_size, qfeats.shape[0])
                target_ret = model.target_enricher(
                    query_features=qfeats[start:end],
                    host_text_features=qfeats[start:end],
                    query_pids=qids[start:end].to(device),
                    pool_cache=target_cache,
                    space=getattr(args, "enrichment_space", "global"),
                )
                enriched.append(target_ret["enriched_features"])
        return torch.cat(enriched, 0)

    def _scale_rows_like(self, base_scores, target_scores):
        base_min = base_scores.min(dim=1, keepdim=True)[0]
        base_max = base_scores.max(dim=1, keepdim=True)[0]
        target_min = target_scores.min(dim=1, keepdim=True)[0]
        target_max = target_scores.max(dim=1, keepdim=True)[0]
        scaled = (base_scores - base_min) / (base_max - base_min).clamp_min(1e-8)
        return scaled * (target_max - target_min) + target_min

    def _fusion_lambdas(self, args):
        raw = getattr(args, "fusion_lambdas", "0.0,0.25,0.5,0.75,1.0")
        return [float(v.strip()) for v in str(raw).split(",") if v.strip()]
    
    def eval(self, model, i2t_metric=False):

        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        similarity = qfeats @ gfeats.t()

        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        r5_idx = min(4, len(t2i_cmc) - 1)
        r10_idx = min(9, len(t2i_cmc) - 1)
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[r5_idx], t2i_cmc[r10_idx], t2i_mAP, t2i_mINP])
        return_top1 = t2i_cmc[0]

        args = self.args or getattr(model, "args", None)
        if (
            args is not None
            and getattr(args, "target_enrichment", False)
            and getattr(args, "eval_target_enrichment", False)
            and getattr(model, "target_enricher", None) is not None
        ):
            target_cache = self._compute_target_cache(model)
            enriched_qfeats = self._enrich_text_features(model, qfeats, qids, target_cache)
            target_gfeats = F.normalize(target_cache["retrieval_features"], p=2, dim=1)
            target_similarity = F.normalize(enriched_qfeats, p=2, dim=1) @ target_gfeats.t()
            target_gids = target_cache["pids"].detach().cpu()
            target_cmc, target_mAP, target_mINP, _ = rank(
                similarity=target_similarity,
                q_pids=qids,
                g_pids=target_gids,
                max_rank=10,
                get_mAP=True,
            )
            target_cmc = target_cmc.numpy()
            target_mAP = target_mAP.numpy()
            target_mINP = target_mINP.numpy()
            table.add_row(
                [
                    "t2i_target",
                    target_cmc[0],
                    target_cmc[min(4, len(target_cmc) - 1)],
                    target_cmc[min(9, len(target_cmc) - 1)],
                    target_mAP,
                    target_mINP,
                ]
            )
            return_top1 = target_cmc[0]

            if getattr(args, "eval_fusion_ablation", False):
                scaled_base = self._scale_rows_like(similarity, target_similarity)
                best = None
                for lambda_proto in self._fusion_lambdas(args):
                    fused = (1.0 - lambda_proto) * scaled_base + lambda_proto * target_similarity
                    fused_cmc, fused_mAP, fused_mINP, _ = rank(
                        similarity=fused,
                        q_pids=qids,
                        g_pids=target_gids,
                        max_rank=10,
                        get_mAP=True,
                    )
                    fused_cmc_np = fused_cmc.numpy()
                    row = (
                        lambda_proto,
                        fused_cmc_np[0],
                        fused_cmc_np[min(4, len(fused_cmc_np) - 1)],
                        fused_cmc_np[min(9, len(fused_cmc_np) - 1)],
                        fused_mAP.numpy(),
                        fused_mINP.numpy(),
                    )
                    if best is None or row[1] > best[1]:
                        best = row
                if best is not None:
                    table.add_row(["t2i_fused@{:.2f}".format(best[0]), best[1], best[2], best[3], best[4], best[5]])

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[min(4, len(i2t_cmc) - 1)], i2t_cmc[min(9, len(i2t_cmc) - 1)], i2t_mAP, i2t_mINP])
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))
        
        return return_top1
