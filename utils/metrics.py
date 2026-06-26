from prettytable import PrettyTable
import torch
import torch.nn.functional as F
import logging

from datasets.target_pool import compute_target_gallery_cache
from model.target_enrichment import normalize_enrichment_space


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        _, indices = torch.topk(similarity, k=max_rank, dim=1, largest=True, sorted=True)
    pred_labels = g_pids[indices.cpu()]
    matches = pred_labels.eq(q_pids.view(-1, 1))

    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)
    tmp_cmc = matches.cumsum(1)
    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100
    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel
    mAP = AP.mean() * 100
    return all_cmc, mAP, mINP, indices


def scale_like(source, reference, eps=1e-8):
    src_min = source.min(dim=1, keepdim=True).values
    src_max = source.max(dim=1, keepdim=True).values
    ref_min = reference.min(dim=1, keepdim=True).values
    ref_max = reference.max(dim=1, keepdim=True).values
    src = (source - src_min) / (src_max - src_min).clamp_min(eps)
    return src * (ref_max - ref_min) + ref_min


def scaled_fuse(primary, secondary, primary_weight):
    return primary_weight * primary + (1 - primary_weight) * scale_like(secondary, primary)


class Evaluator():
    def __init__(self, img_loader, txt_loader, args=None):
        self.img_loader = img_loader
        self.txt_loader = txt_loader
        self.args = args
        self.logger = logging.getLogger("IRRA.eval")
        self.last_metrics = {}
        self.last_best_task = None

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device
        qids, gids, qfeats, gfeats = [], [], [], []
        rqfeats, rgfeats = [], []
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_clip_global_text(caption)
                retrieval_text = model.encode_retrieval_text(caption)
            qids.append(pid.view(-1))
            qfeats.append(text_feat)
            rqfeats.append(retrieval_text)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)
        rqfeats = torch.cat(rqfeats, 0)

        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_clip_global_image(img)
                retrieval_img = model.encode_retrieval_image(img)
            gids.append(pid.view(-1))
            gfeats.append(img_feat)
            rgfeats.append(retrieval_img)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        rgfeats = torch.cat(rgfeats, 0)
        return qfeats, gfeats, rqfeats, rgfeats, qids, gids

    def _metric_row(self, name, similarity, qids, gids, table):
        cmc, mAP, mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        cmc, mAP, mINP = cmc.cpu().numpy(), mAP.cpu().numpy(), mINP.cpu().numpy()
        rsum = cmc[0] + cmc[4] + cmc[9]
        table.add_row([name, cmc[0], cmc[4], cmc[9], mAP, mINP, rsum])
        return {"R1": float(cmc[0]), "R5": float(cmc[4]), "R10": float(cmc[9]), "mAP": float(mAP), "mINP": float(mINP), "rSum": float(rsum)}

    def _target_scores(self, model, qfeats, rqfeats):
        target_cache, _ = compute_target_gallery_cache(model, self.img_loader)
        retrieval_images = F.normalize(target_cache["retrieval_features"].float(), p=2, dim=1)
        space = normalize_enrichment_space(getattr(model.args, "enrichment_space", "global"))
        query_bank = rqfeats if space == "retrieval" else qfeats
        chunks = []
        batch_size = int(getattr(model.args, "target_query_batch_size", getattr(model.args, "test_batch_size", 512)))
        with torch.no_grad():
            for start in range(0, query_bank.shape[0], batch_size):
                end = start + batch_size
                enriched = model.enrich_text_features(
                    query_features=query_bank[start:end],
                    host_text_features=qfeats[start:end],
                    target_cache=target_cache,
                    alt_text_features=rqfeats[start:end],
                )
                chunks.append(F.normalize(enriched, p=2, dim=1))
        enriched_queries = torch.cat(chunks, dim=0)
        return enriched_queries @ retrieval_images.t()

    def eval(self, model, i2t_metric=False, use_target_enrichment=None):
        if use_target_enrichment is None:
            use_target_enrichment = bool(getattr(model.args, "target_enrichment", False))
        qfeats, gfeats, rqfeats, rgfeats, qids, gids = self._compute_embedding(model)
        qfeats = F.normalize(qfeats, p=2, dim=1)
        gfeats = F.normalize(gfeats, p=2, dim=1)
        rqfeats = F.normalize(rqfeats, p=2, dim=1)
        rgfeats = F.normalize(rgfeats, p=2, dim=1)

        scores = {
            "global-t2i": qfeats @ gfeats.t(),
            "retrieval-t2i": rqfeats @ rgfeats.t(),
        }
        if use_target_enrichment and getattr(model, "target_enricher", None) is not None:
            target_score = self._target_scores(model, qfeats, rqfeats)
            scores["target+proto(1)-t2i"] = target_score
            for base_name in ["global-t2i", "retrieval-t2i"]:
                base_prefix = base_name.replace("-t2i", "")
                for i in range(0, 11):
                    proto_lambda = i / 10.0
                    scores[f"{base_prefix}+proto({proto_lambda:g})-t2i"] = (1 - proto_lambda) * scale_like(scores[base_name], target_score) + proto_lambda * target_score

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP", "rSum"])
        metrics = {}
        for name, similarity in scores.items():
            metrics[name] = self._metric_row(name, similarity, qids, gids, table)

        if i2t_metric:
            self._metric_row("global-i2t", scores["global-t2i"].t(), gids, qids, table)

        for column in ["R1", "R5", "R10", "mAP", "mINP", "rSum"]:
            table.custom_format[column] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))

        best_task, best_values = max(metrics.items(), key=lambda item: item[1]["R1"])
        best_r1 = best_values["R1"]
        self.last_metrics = {
            f"eval/{task}/{metric}": value
            for task, row in metrics.items()
            for metric, value in row.items()
        }
        self.last_best_task = best_task
        return best_r1
