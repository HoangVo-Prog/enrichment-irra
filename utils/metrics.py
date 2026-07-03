import logging
import re

from prettytable import PrettyTable
import torch
import torch.nn.functional as F

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
    inp = [
        tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.0)
        for i, match_row in enumerate(matches)
    ]
    mINP = torch.cat(inp).mean() * 100
    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel
    mAP = AP.mean() * 100
    return all_cmc, mAP, mINP, indices


def get_metrics(similarity, qids, gids, name, return_indices=False):
    cmc, mAP, mINP, indices = rank(
        similarity=similarity,
        q_pids=qids,
        g_pids=gids,
        max_rank=10,
        get_mAP=True,
    )
    cmc, mAP, mINP = cmc.cpu().numpy(), mAP.cpu().numpy(), mINP.cpu().numpy()
    row = [
        name,
        cmc[0],
        cmc[4],
        cmc[9],
        mAP,
        mINP,
        cmc[0] + cmc[4] + cmc[9],
    ]
    if return_indices:
        return row, indices
    return row


def _metric_task_name(task):
    task = str(task).replace("+", "_plus_")
    task = task.replace("(", "_").replace(")", "")
    task = task.replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_/-]+", "_", task).strip("_")


def _row_to_eval_metrics(row):
    task = _metric_task_name(row[0])
    return {
        f"eval/{task}/R1": float(row[1]),
        f"eval/{task}/R5": float(row[2]),
        f"eval/{task}/R10": float(row[3]),
        f"eval/{task}/mAP": float(row[4]),
        f"eval/{task}/mINP": float(row[5]),
        f"eval/{task}/rSum": float(row[6]) if len(row) > 6 else 0.0,
    }


def _scale_scores_like(scores, reference, eps=1e-12):
    score_min = scores.min(dim=1, keepdim=True).values
    score_max = scores.max(dim=1, keepdim=True).values
    ref_min = reference.min(dim=1, keepdim=True).values
    ref_max = reference.max(dim=1, keepdim=True).values

    score_range = (score_max - score_min).clamp_min(eps)
    ref_range = ref_max - ref_min
    return (scores - score_min) / score_range * ref_range + ref_min


def _prototype_lambdas():
    return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _format_lambda(value):
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return "{:.2f}".format(value).rstrip("0").rstrip(".")


def _ablation_lambda_from_key(key):
    match = re.search(r"\(([-+]?\d*\.?\d+)\)$", str(key))
    return float(match.group(1)) if match else 0.0


def _clear_cuda_cache_if_available():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _log_cuda_memory(logger, label):
    if not torch.cuda.is_available():
        return
    device = torch.cuda.current_device()
    mib = 1024.0 ** 2
    logger.info(
        "CUDA memory %s: allocated=%.1fMiB reserved=%.1fMiB max_allocated=%.1fMiB",
        label,
        torch.cuda.memory_allocated(device) / mib,
        torch.cuda.memory_reserved(device) / mib,
        torch.cuda.max_memory_allocated(device) / mib,
    )


def _log_cache_tensors(logger, label, cache):
    if not isinstance(cache, dict):
        return
    keys = (
        "host_image_features",
        "retrieval_features",
        "grab_image_features",
        "evidence_bank",
        "prototypes",
        "pids",
    )
    parts = []
    for key in keys:
        value = cache.get(key)
        if torch.is_tensor(value):
            parts.append(
                "{} shape={} device={} dtype={}".format(
                    key,
                    tuple(value.shape),
                    value.device,
                    value.dtype,
                )
            )
    if parts:
        logger.info("%s cache tensors: %s", label, "; ".join(parts))


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


class Evaluator():
    def __init__(self, img_loader, txt_loader, args=None):
        self.img_loader = img_loader
        self.txt_loader = txt_loader
        self.args = args
        self.logger = logging.getLogger("IRRA.eval")
        self.last_metrics = {}
        self.last_best_task = None

    def _active_args(self, model):
        if self.args is not None:
            return self.args
        args = getattr(_unwrap_model(model), "args", None)
        self.args = args
        return args

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device
        qids, gids, qfeats, gfeats = [], [], [], []
        rqfeats, rgfeats = [], []
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.inference_mode():
                text_feat = model.encode_clip_global_text(caption)
                retrieval_text = model.encode_retrieval_text(caption)
            qids.append(pid.view(-1))
            # Evaluation embeddings are reused only for scoring; CPU storage avoids
            # keeping full query/gallery banks live on GPU across ablation rows.
            qfeats.append(text_feat.detach().cpu())
            rqfeats.append(retrieval_text.detach().cpu())
        qids = torch.cat(qids, 0).cpu()
        qfeats = torch.cat(qfeats, 0).cpu()
        rqfeats = torch.cat(rqfeats, 0).cpu()

        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.inference_mode():
                img_feat = model.encode_clip_global_image(img)
                retrieval_img = model.encode_retrieval_image(img)
            gids.append(pid.view(-1))
            gfeats.append(img_feat.detach().cpu())
            rgfeats.append(retrieval_img.detach().cpu())
        gids = torch.cat(gids, 0).cpu()
        gfeats = torch.cat(gfeats, 0).cpu()
        rgfeats = torch.cat(rgfeats, 0).cpu()
        return qfeats, gfeats, rqfeats, rgfeats, qids, gids

    def _iter_base_tasks(self, sims_global, sims_retrieval):
        yield "global", sims_global
        yield "retrieval", sims_retrieval

    def _target_scores(self, model, qfeats, rqfeats):
        args = self._active_args(model)
        target_cache, _ = compute_target_gallery_cache(model, self.img_loader)
        _log_cache_tensors(self.logger, "Evaluation target gallery", target_cache)
        retrieval_images = F.normalize(target_cache["retrieval_features"].detach().cpu().float(), p=2, dim=1)
        space = normalize_enrichment_space(getattr(args, "enrichment_space", "global"))
        query_bank = rqfeats if space == "retrieval" else qfeats
        chunks = []
        device = next(model.parameters()).device
        batch_size = int(getattr(args, "target_query_batch_size", getattr(args, "test_batch_size", 512)))
        with torch.inference_mode():
            for start in range(0, query_bank.shape[0], batch_size):
                end = start + batch_size
                query_chunk = query_bank[start:end].to(device)
                host_chunk = qfeats[start:end].to(device)
                alt_chunk = rqfeats[start:end].to(device)
                enriched = model.enrich_text_features(
                    query_features=query_chunk,
                    host_text_features=host_chunk,
                    target_cache=target_cache,
                    alt_text_features=alt_chunk,
                )
                chunks.append(F.normalize(enriched, p=2, dim=1).detach().cpu())
                del query_chunk, host_chunk, alt_chunk, enriched
        enriched_queries = torch.cat(chunks, dim=0).cpu()
        sims = enriched_queries @ retrieval_images.t()
        del target_cache, retrieval_images, enriched_queries, chunks
        _clear_cuda_cache_if_available()
        _log_cuda_memory(self.logger, "after target-aware scoring")
        return sims

    def _iter_eval_tasks(self, sims_global, sims_retrieval, sims_target):
        base_tasks = list(self._iter_base_tasks(sims_global, sims_retrieval))
        for task_name, task_scores in base_tasks:
            yield task_name, task_scores

        if sims_target is None:
            return

        yield "target+proto(1)", sims_target
        for proto_lambda in _prototype_lambdas():
            proto_value = _format_lambda(proto_lambda)
            for base_name, base_scores in base_tasks:
                fused_name = "{}+proto({})".format(base_name, proto_value)
                scaled_base_scores = _scale_scores_like(base_scores, sims_target)
                yield fused_name, (
                    (1.0 - proto_lambda) * scaled_base_scores
                    + proto_lambda * sims_target
                )

    def eval(self, model, i2t_metric=False, use_target_enrichment=None):
        args = self._active_args(model)
        if use_target_enrichment is None:
            use_target_enrichment = bool(getattr(args, "target_enrichment", False))

        self.logger.info("Starting evaluation feature extraction")
        _log_cuda_memory(self.logger, "before evaluation")
        qfeats, gfeats, rqfeats, rgfeats, qids, gids = self._compute_embedding(model)
        qfeats = F.normalize(qfeats, p=2, dim=1)
        gfeats = F.normalize(gfeats, p=2, dim=1)
        rqfeats = F.normalize(rqfeats, p=2, dim=1)
        rgfeats = F.normalize(rgfeats, p=2, dim=1)

        sims_global = qfeats @ gfeats.t()
        sims_retrieval = rqfeats @ rgfeats.t()
        self.logger.info(
            "Evaluation similarity matrices ready on CPU: queries={} gallery={}".format(
                qfeats.shape[0],
                gfeats.shape[0],
            )
        )
        _log_cuda_memory(self.logger, "after evaluation feature extraction")
        sims_target = None
        if use_target_enrichment and getattr(_unwrap_model(model), "target_enricher", None) is not None:
            sims_target = self._target_scores(model, qfeats, rqfeats)

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP", "rSum"])
        eval_metrics = {}
        rows_by_task = {}
        best_task = None
        best_row = None
        best_ablation_task = None
        best_ablation_row = None

        total_tasks = 2
        if sims_target is not None:
            total_tasks += 1 + len(_prototype_lambdas()) * 2
        self.logger.info("Scoring %d evaluation tasks sequentially", total_tasks)
        for task_idx, (key, similarity) in enumerate(self._iter_eval_tasks(sims_global, sims_retrieval, sims_target), 1):
            self.logger.debug("Evaluation task %d/%d start: %s", task_idx, total_tasks, key)
            row = get_metrics(similarity, qids, gids, "{}-t2i".format(key), False)
            table.add_row(row)
            rows_by_task[key] = row
            eval_metrics.update(_row_to_eval_metrics(row))
            if i2t_metric:
                i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(
                    similarity=similarity.t(),
                    q_pids=gids,
                    g_pids=qids,
                    max_rank=10,
                    get_mAP=True,
                )
                i2t_cmc, i2t_mAP, i2t_mINP = (
                    i2t_cmc.cpu().numpy(),
                    i2t_mAP.cpu().numpy(),
                    i2t_mINP.cpu().numpy(),
                )
                i2t_row = [
                    "{}-i2t".format(key),
                    i2t_cmc[0],
                    i2t_cmc[4],
                    i2t_cmc[9],
                    i2t_mAP,
                    i2t_mINP,
                    i2t_cmc[0] + i2t_cmc[4] + i2t_cmc[9],
                ]
                table.add_row(i2t_row)
                eval_metrics.update(_row_to_eval_metrics(i2t_row))

            if best_row is None or row[1] > best_row[1]:
                best_task = key
                best_row = row
            if "+proto(" in key and (
                best_ablation_row is None or row[1] > best_ablation_row[1]
            ):
                best_ablation_task = key
                best_ablation_row = row
            self.logger.debug("Evaluation task %d/%d end: %s", task_idx, total_tasks, key)

        if best_row is not None:
            top1 = float(best_row[1])
        else:
            top1 = 0.0

        if best_ablation_row is not None:
            eval_metrics["eval/ablation_best_R1"] = float(best_ablation_row[1])
            eval_metrics["eval/ablation_best_R5"] = float(best_ablation_row[2])
            eval_metrics["eval/ablation_best_R10"] = float(best_ablation_row[3])
            eval_metrics["eval/ablation_best_mAP"] = float(best_ablation_row[4])
            eval_metrics["eval/ablation_best_mINP"] = float(best_ablation_row[5])
            eval_metrics["eval/ablation_best_rSum"] = float(best_ablation_row[6])
            eval_metrics["eval/ablation_best_lambda"] = _ablation_lambda_from_key(
                best_ablation_task
            )

        target_key = "global+proto(1)"
        if "global" in rows_by_task and target_key in rows_by_task:
            global_row = rows_by_task["global"]
            target_row = rows_by_task[target_key]
            eval_metrics["eval/delta_R1_target_vs_global"] = float(target_row[1] - global_row[1])
            eval_metrics["eval/delta_R5_target_vs_global"] = float(target_row[2] - global_row[2])
            eval_metrics["eval/delta_R10_target_vs_global"] = float(target_row[3] - global_row[3])
            eval_metrics["eval/delta_mAP_target_vs_global"] = float(target_row[4] - global_row[4])
            eval_metrics["eval/delta_mINP_target_vs_global"] = float(target_row[5] - global_row[5])
            eval_metrics["eval/delta_rSum_target_vs_global"] = float(target_row[6] - global_row[6])

        self.last_metrics = eval_metrics
        self.last_best_task = best_task

        for column in ["R1", "R5", "R10", "mAP", "mINP", "rSum"]:
            table.custom_format[column] = lambda f, v: "{:.2f}".format(v)
        self.logger.info('\n' + str(table))
        self.logger.info('\n' + "best R1 = " + str(top1))
        if best_task is not None:
            self.logger.info("best R1 row = {}".format(best_task))
        _clear_cuda_cache_if_available()
        _log_cuda_memory(self.logger, "after evaluation")
        return top1
