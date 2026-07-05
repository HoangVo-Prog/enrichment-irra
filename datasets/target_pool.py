import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model.target_enrichment import normalize_enrichment_space, normalize_rank_space
from utils.reproducibility import seed_worker, seeded_generator


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _loader_batch_size(args, primary_name, fallback_name="test_batch_size", default=1):
    value = getattr(args, primary_name, None)
    if value is None:
        value = getattr(args, fallback_name, default)
    return max(1, int(value))


class _PoolImageDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from utils.iotools import read_image

        record = self.records[index]
        image = read_image(record["img_path"])
        if self.transform is not None:
            image = self.transform(image)
        return record["pid"], record["image_id"], image


class _PoolTextDataset(Dataset):
    def __init__(self, records, tokenizer, text_length, truncate):
        self.records = records
        self.tokenizer = tokenizer
        self.text_length = text_length
        self.truncate = truncate

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from datasets.bases import tokenize

        record = self.records[index]
        caption = tokenize(
            record["caption"],
            tokenizer=self.tokenizer,
            text_length=self.text_length,
            truncate=self.truncate,
        )
        return record["pid"], record["query_index"], caption


class TargetPoolManager:
    def __init__(self, train_dataset, args, logger=None):
        self.train_dataset = train_dataset
        self.args = args
        self.seed = int(getattr(args, "seed", 1))
        self.logger = logger or logging.getLogger("IRRA.target_pool")
        source_records = getattr(train_dataset, "dataset", train_dataset)
        self.image_records = self._build_image_records(source_records)
        self.query_records = self._build_query_records(source_records)
        self.transform = self._build_transform(args.img_size)
        self.cache = None
        self.cache_interval_id = None
        self.cache_returned = False
        self.frozen_cache = None
        self.frozen_rank_indices = None
        self.frozen_returned = False
        self.frozen_index_depth = None

        if self.logger is not None:
            self.logger.info(
                "Target enrichment will select top-M directly from the full training set."
            )
        if getattr(args, "use_freeze_indices", False) and self.logger is not None:
            if not getattr(args, "freeze_host", False):
                self.logger.warning(
                    "--use_freeze_indices is intended for a frozen host. If host weights "
                    "change, the precomputed retrieval ranking will become stale."
                )

    @staticmethod
    def _build_image_records(records):
        by_image = {}
        for pid, image_id, img_path, _caption in records:
            image_id = int(image_id)
            if image_id not in by_image:
                by_image[image_id] = {"pid": int(pid), "image_id": image_id, "img_path": img_path}
        return list(by_image.values())

    @staticmethod
    def _build_query_records(records):
        query_records = []
        for query_index, (pid, _image_id, _img_path, caption) in enumerate(records):
            query_records.append({"pid": int(pid), "query_index": int(query_index), "caption": caption})
        return query_records

    @staticmethod
    def _build_transform(img_size):
        height, width = img_size
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

        def transform(image):
            image = image.resize((width, height))
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            return (tensor - mean) / std

        return transform

    def _recompute_interval(self):
        pool_interval = getattr(self.args, "pool_interval", None)
        return int(pool_interval if pool_interval is not None else getattr(self.args, "recompute_interval", 1))

    def _interval_id(self, epoch, step):
        level = str(getattr(self.args, "recompute_level", "epoch")).lower()
        interval = self._recompute_interval()
        if interval == -1:
            return 0
        if interval < 1:
            raise ValueError("--recompute_interval must be -1 or a positive integer")
        unit = int(step if level == "step" else epoch)
        return (unit - 1) // interval

    def get_train_cache(self, model, batch, epoch, step):
        if bool(getattr(self.args, "use_freeze_indices", False)):
            return self._get_frozen_cache(model, batch)
        interval_id = self._interval_id(epoch, step)
        needs_rebuild = self.cache is None or (self._recompute_interval() != -1 and interval_id != self.cache_interval_id)
        if needs_rebuild:
            self.cache = self._encode_records(model, self.image_records)
            self.cache_interval_id = interval_id
            self.cache_returned = False
            if self.logger is not None:
                self.logger.info(
                    "Target-pool full training cache built: interval={} images={} top_m={}".format(
                        interval_id,
                        len(self.image_records),
                        getattr(self.args, "top_m", 1),
                    )
                )
        diagnostics = {
            "pool_interval_id": float(interval_id),
            "pool_interval_reused": 1.0 if self.cache_returned else 0.0,
            "pool_cache_size": float(len(self.image_records)),
        }
        self.cache_returned = True
        cache = dict(self.cache)
        cache["diagnostics"] = diagnostics
        return cache

    def _get_frozen_cache(self, model, batch):
        if self.frozen_cache is None or self.frozen_rank_indices is None:
            self._build_frozen_cache(model)
        if "index" not in batch:
            raise KeyError("frozen-index target enrichment requires batch['index']")
        query_indices = batch["index"].detach().long().cpu()
        if query_indices.min() < 0 or query_indices.max() >= self.frozen_rank_indices.shape[0]:
            raise ValueError("batch query index out of frozen-rank range")
        rank_depth = self.frozen_index_depth
        depth = min(int(getattr(self.args, "top_m", 32)), rank_depth)
        top_indices = self.frozen_rank_indices[query_indices, :depth].to(self.frozen_cache["host_image_features"].device)
        diagnostics = {
            "pool_interval_id": 0.0,
            "pool_interval_reused": 1.0 if self.frozen_returned else 0.0,
            "pool_cache_size": float(len(self.image_records)),
            "frozen_indices_used": 1.0,
            "frozen_index_depth": float(rank_depth),
        }
        self.frozen_returned = True
        cache = dict(self.frozen_cache)
        cache["top_indices"] = top_indices
        cache["diagnostics"] = diagnostics
        return cache

    def _build_frozen_cache(self, model):
        cache = self._encode_records(model, self.image_records)
        rank_space = normalize_rank_space(getattr(self.args, "topm_rank_space", "host_global"))
        host_images = F.normalize(cache["host_image_features"].float(), p=2, dim=-1)
        retrieval_images = F.normalize(cache["retrieval_features"].float(), p=2, dim=-1)
        rank_depth = min(int(getattr(self.args, "top_m", 32)), host_images.shape[0])
        if rank_depth < 1:
            raise ValueError("Frozen index depth must be positive")

        query_features = self._encode_text_records(model, "global")
        if rank_space == "host_global":
            rank_indices = self._topk_chunks(query_features, host_images, rank_depth)
        elif rank_space == "retrieval":
            retrieval_queries = self._encode_text_records(model, "retrieval")
            rank_indices = self._topk_chunks(retrieval_queries, retrieval_images, rank_depth)
        elif rank_space == "hybrid_global_retrieval":
            retrieval_queries = self._encode_text_records(model, "retrieval")
            rank_indices = self._hybrid_topk_chunks(query_features, host_images, retrieval_queries, retrieval_images, rank_depth)
        else:
            raise ValueError(f"unsupported topm_rank_space: {rank_space}")
        self.frozen_cache = cache
        self.frozen_rank_indices = rank_indices.cpu()
        self.frozen_index_depth = rank_depth
        self.frozen_returned = False
        if self.logger is not None:
            self.logger.info(
                "Frozen host retrieval index built: queries={} gallery={} rank_depth={}".format(
                    self.frozen_rank_indices.shape[0],
                    len(self.image_records),
                    rank_depth,
                )
            )

    def _topk_chunks(self, queries, images, rank_depth):
        chunks = []
        chunk_size = min(
            _loader_batch_size(self.args, "target_query_batch_size"),
            max(1, queries.shape[0]),
        )
        for start in range(0, queries.shape[0], chunk_size):
            scores = F.normalize(queries[start:start + chunk_size].float(), p=2, dim=-1) @ images.t()
            chunks.append(torch.topk(scores, k=rank_depth, dim=1, largest=True, sorted=True).indices.cpu())
        return torch.cat(chunks, dim=0)

    def _hybrid_topk_chunks(self, global_queries, global_images, retrieval_queries, retrieval_images, rank_depth):
        chunks = []
        chunk_size = min(
            _loader_batch_size(self.args, "target_query_batch_size"),
            max(1, global_queries.shape[0]),
        )
        weight = float(getattr(self.args, "topm_rank_lambda", 0.5))
        for start in range(0, global_queries.shape[0], chunk_size):
            end = start + chunk_size
            global_scores = F.normalize(global_queries[start:end].float(), p=2, dim=-1) @ global_images.t()
            retrieval_scores = F.normalize(retrieval_queries[start:end].float(), p=2, dim=-1) @ retrieval_images.t()
            scores = weight * global_scores + (1 - weight) * retrieval_scores
            chunks.append(torch.topk(scores, k=rank_depth, dim=1, largest=True, sorted=True).indices.cpu())
        return torch.cat(chunks, dim=0)

    def _encode_records(self, model, records):
        if not records:
            raise ValueError("empty image pool")
        module = _unwrap_model(model)
        device = next(module.parameters()).device
        was_training = module.training
        module.eval()
        dataset = _PoolImageDataset(records, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=min(_loader_batch_size(self.args, "target_cache_batch_size"), max(1, len(records))),
            shuffle=False,
            num_workers=getattr(self.args, "num_workers", 0),
            worker_init_fn=seed_worker,
            generator=seeded_generator(self.seed + 801),
        )
        chunks = []
        with torch.no_grad():
            for _pid, _image_id, images in loader:
                images = images.to(device)
                chunks.append(module.encode_target_image_cache(images, cache_prototypes=True))
        if was_training:
            module.train()
        cache = self._concat_cache_chunks(chunks, device)
        cache["image_ids"] = torch.tensor([record["image_id"] for record in records], device=device, dtype=torch.long)
        cache["pids"] = torch.tensor([record["pid"] for record in records], device=device, dtype=torch.long)
        cache = module.finalize_target_cache(cache)
        return cache

    @staticmethod
    def _concat_cache_chunks(chunks, device):
        cache = {}
        keys = set(key for chunk in chunks for key in chunk.keys())
        for key in keys:
            values = [chunk[key] for chunk in chunks if key in chunk]
            if torch.is_tensor(values[0]):
                cache[key] = torch.cat([value.detach().to(device) for value in values], dim=0)
        return cache

    def _encode_text_records(self, model, space):
        if not self.query_records:
            raise ValueError("empty query pool")
        module = _unwrap_model(model)
        device = next(module.parameters()).device
        tokenizer = getattr(self.train_dataset, "tokenizer", None)
        if tokenizer is None:
            from utils.simple_tokenizer import SimpleTokenizer
            tokenizer = SimpleTokenizer()
        dataset = _PoolTextDataset(
            self.query_records,
            tokenizer=tokenizer,
            text_length=getattr(self.train_dataset, "text_length", getattr(self.args, "text_length", 77)),
            truncate=getattr(self.train_dataset, "truncate", True),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(_loader_batch_size(self.args, "target_query_batch_size"), max(1, len(self.query_records))),
            shuffle=False,
            num_workers=getattr(self.args, "num_workers", 0),
            worker_init_fn=seed_worker,
            generator=seeded_generator(self.seed + 901),
        )
        was_training = module.training
        module.eval()
        chunks = []
        with torch.no_grad():
            for _pid, _query_index, tokens in loader:
                tokens = tokens.to(device)
                if normalize_enrichment_space(space) == "retrieval":
                    chunks.append(module.encode_retrieval_text(tokens))
                else:
                    chunks.append(module.encode_clip_global_text(tokens))
        if was_training:
            module.train()
        return torch.cat(chunks, dim=0)


def compute_target_gallery_cache(model, img_loader):
    module = _unwrap_model(model)
    device = next(module.parameters()).device
    was_training = module.training
    module.eval()
    chunks = []
    gids = []
    with torch.no_grad():
        for pid, images in img_loader:
            images = images.to(device)
            chunks.append(module.encode_target_image_cache(images, cache_prototypes=True))
            gids.append(pid.view(-1))
    if was_training:
        module.train()
    cache = TargetPoolManager._concat_cache_chunks(chunks, device)
    cache["pids"] = torch.cat(gids, dim=0).to(device).long()
    cache["image_ids"] = torch.arange(cache["pids"].shape[0], device=device, dtype=torch.long)
    cache = module.finalize_target_cache(cache)
    return cache, torch.cat(gids, dim=0)
