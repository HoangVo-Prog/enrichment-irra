import logging
import math
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class TargetImageRecord(object):
    def __init__(self, pid, image_id, img_path):
        self.pid = int(pid)
        self.image_id = int(image_id)
        self.img_path = img_path


class TargetQueryRecord(object):
    def __init__(self, index, pid, image_id, img_path, caption):
        self.index = int(index)
        self.pid = int(pid)
        self.image_id = int(image_id)
        self.img_path = img_path
        self.caption = caption


class TargetImageDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from utils.iotools import read_image

        record = self.records[index]
        image = read_image(record.img_path)
        if self.transform is not None:
            image = self.transform(image)
        return record.pid, record.image_id, image


class TargetTextDataset(Dataset):
    def __init__(self, records, text_length=77, truncate=True):
        self.records = records
        self.text_length = text_length
        self.truncate = truncate
        from utils.simple_tokenizer import SimpleTokenizer

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from datasets.bases import tokenize

        record = self.records[index]
        tokens = tokenize(
            record.caption,
            tokenizer=self.tokenizer,
            text_length=self.text_length,
            truncate=self.truncate,
        )
        return record.index, record.pid, tokens


def _collate_images(batch):
    pids, image_ids, images = zip(*batch)
    return torch.tensor(pids), torch.tensor(image_ids), torch.stack(images)


def _collate_text(batch):
    indices, pids, captions = zip(*batch)
    return torch.tensor(indices), torch.tensor(pids), torch.stack(captions)


def _parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if value is None or value == "":
        return []
    return [int(v.strip()) for v in str(value).split(",") if v.strip()]


def _model_device(model):
    return next(model.parameters()).device


def _as_float(value, device):
    return torch.tensor(float(value), device=device)


class TargetPoolManager(object):
    def __init__(self, train_dataset, args, logger=None):
        self.train_dataset = train_dataset
        self.args = args
        self.logger = logger or logging.getLogger("IRRA.target_pool")
        self.records, self.query_records = self._build_records(train_dataset)
        if len(self.records) == 0:
            raise ValueError("target enrichment requires at least one training image")

        self.pid_to_record_indices = defaultdict(list)
        for idx, record in enumerate(self.records):
            self.pid_to_record_indices[record.pid].append(idx)

        self.coverage_counts = defaultdict(int)
        self.full_cache = None
        self.shared_cache = None
        self.descriptor_cache = None
        self.frozen_cache = None
        self.frozen_rankings = None
        self.cluster_assignments = None
        self.cluster_distribution = None
        self.interval_id = 0
        self.last_epoch_refresh = None
        self.last_step_refresh = None
        self.shared_diagnostics = {}
        self.rng = random.Random(getattr(args, "pool_seed", 1))

    def _build_records(self, train_dataset):
        if not hasattr(train_dataset, "dataset"):
            raise ValueError("TargetPoolManager expects an ImageTextDataset-like object")
        unique = {}
        query_records = []
        for index, item in enumerate(train_dataset.dataset):
            pid, image_id, img_path, caption = item
            if int(image_id) not in unique:
                unique[int(image_id)] = TargetImageRecord(pid, image_id, img_path)
            query_records.append(TargetQueryRecord(index, pid, image_id, img_path, caption))
        records = [unique[key] for key in sorted(unique.keys())]
        return records, query_records

    @property
    def capacity(self):
        return len(self.records)

    @property
    def required_anchor_count(self):
        return len(self.pid_to_record_indices)

    def _should_refresh(self, cache, epoch, step):
        if cache is None:
            return True
        interval = int(getattr(self.args, "recompute_interval", -1))
        if interval < 0:
            return False
        level = getattr(self.args, "recompute_level", "epoch")
        if level == "step":
            if self.last_step_refresh is None:
                return True
            return int(step) - int(self.last_step_refresh) >= interval
        if self.last_epoch_refresh is None:
            return True
        return int(epoch) - int(self.last_epoch_refresh) >= interval

    def _mark_refresh(self, epoch, step):
        self.interval_id += 1
        self.last_epoch_refresh = int(epoch)
        self.last_step_refresh = int(step)

    def _cache_diagnostics(self, cache, diagnostics):
        cache["diagnostics"] = diagnostics
        return cache

    def _encode_records(self, model, records, cache_prototypes=True):
        device = _model_device(model)
        dataset = TargetImageDataset(records, transform=getattr(self.train_dataset, "transform", None))
        loader = DataLoader(
            dataset,
            batch_size=getattr(self.args, "test_batch_size", getattr(self.args, "batch_size", 64)),
            shuffle=False,
            num_workers=getattr(self.args, "num_workers", 0),
            collate_fn=_collate_images,
        )
        was_training = model.training
        model.eval()
        pids, image_ids, host_features, retrieval_features, prototypes = [], [], [], [], []
        with torch.no_grad():
            for batch_pids, batch_image_ids, images in loader:
                images = images.to(device)
                encoded = model.encode_target_image_cache(images, cache_prototypes=cache_prototypes)
                pids.append(batch_pids.to(device))
                image_ids.append(batch_image_ids.to(device))
                host_features.append(encoded["host_image_features"].detach())
                retrieval_features.append(encoded["retrieval_features"].detach())
                if cache_prototypes:
                    prototypes.append(encoded["prototypes"].detach())
        if was_training:
            model.train()

        cache = {
            "host_image_features": torch.cat(host_features, dim=0),
            "retrieval_features": torch.cat(retrieval_features, dim=0),
            "pids": torch.cat(pids, dim=0).long(),
            "image_ids": torch.cat(image_ids, dim=0).long(),
        }
        if cache_prototypes:
            cache["prototypes"] = torch.cat(prototypes, dim=0)
        return cache

    def _ensure_descriptor_cache(self, model):
        if self.descriptor_cache is None:
            self.descriptor_cache = self._encode_records(model, self.records, cache_prototypes=False)
        return self.descriptor_cache

    def _compute_clusters(self, model):
        descriptor_cache = self._ensure_descriptor_cache(model)
        features = F.normalize(descriptor_cache["host_image_features"].float(), p=2, dim=1)
        num_clusters = max(1, min(int(getattr(self.args, "pool_clusters", 32)), features.shape[0]))
        if num_clusters == 1:
            assignments = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
        else:
            init_idx = torch.linspace(0, features.shape[0] - 1, steps=num_clusters, device=features.device).long()
            centers = features[init_idx].clone()
            assignments = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
            for _ in range(10):
                distances = 1.0 - features @ centers.t()
                assignments = distances.argmin(dim=1)
                new_centers = []
                for cluster_id in range(num_clusters):
                    members = features[assignments == cluster_id]
                    if members.numel() == 0:
                        new_centers.append(centers[cluster_id])
                    else:
                        new_centers.append(F.normalize(members.mean(dim=0), p=2, dim=0))
                centers = torch.stack(new_centers, dim=0)
        counts = torch.bincount(assignments, minlength=num_clusters).float()
        self.cluster_assignments = assignments.cpu()
        self.cluster_distribution = (counts / counts.sum().clamp_min(1.0)).cpu()

    def _distribution_distance(self, observed, expected):
        metric = getattr(self.args, "pool_dist_metric", "l1")
        observed = observed.float()
        expected = expected.float()
        if metric == "js":
            eps = 1e-8
            midpoint = 0.5 * (observed + expected)
            js = 0.5 * (observed * torch.log((observed + eps) / (midpoint + eps))).sum()
            js = js + 0.5 * (expected * torch.log((expected + eps) / (midpoint + eps))).sum()
            return float(js.item())
        return float(torch.abs(observed - expected).sum().item())

    def _choose_shared_k(self, model):
        required = self.required_anchor_count
        capacity = self.capacity
        diagnostics = {
            "pool_k_valid": float(required),
            "pool_k_dilute": float(required),
            "pool_k_dist": 0.0,
            "pool_k_cover": 0.0,
            "pool_adaptive_fallback": 0.0,
            "pool_k_capacity_limited": 0.0,
        }

        if getattr(self.args, "pool_k_mode", "static") == "static":
            selected = int(getattr(self.args, "pool_k", 0))
            if selected <= 0:
                selected = capacity
            if selected < required:
                raise ValueError(
                    "pool_k={} cannot include one anchor for each of {} train identities".format(
                        selected, required
                    )
                )
            selected = min(selected, capacity)
            diagnostics["pool_k_valid_target"] = float(required)
            diagnostics["pool_k_dist_target"] = float(selected)
            return selected, diagnostics

        positive_ratio_max = float(getattr(self.args, "positive_ratio_max", 0.5))
        k_valid = max(1, required)
        k_dilute = int(math.ceil(required / max(positive_ratio_max, 1e-8)))
        horizon = max(1, int(getattr(self.args, "pool_coverage_epochs", 1)))
        k_cover = int(math.ceil(float(capacity) / horizon))

        if self.cluster_assignments is None:
            self._compute_clusters(model)
        candidates = sorted(set(_parse_int_list(getattr(self.args, "pool_k_candidates", ""))))
        threshold = float(getattr(self.args, "pool_dist_threshold", 0.25))
        k_dist = 0
        dist_found = 0.0
        for candidate in candidates:
            if candidate < k_valid or candidate > capacity:
                continue
            expected = self.cluster_distribution
            quotas = torch.floor(expected * candidate)
            while quotas.sum() < candidate:
                deficits = expected * candidate - quotas
                quotas[deficits.argmax()] += 1
            observed = quotas / quotas.sum().clamp_min(1.0)
            dist = self._distribution_distance(observed, expected)
            if dist <= threshold:
                k_dist = candidate
                dist_found = 1.0
                break
        if k_dist == 0:
            k_dist = candidates[-1] if candidates else k_valid
            diagnostics["pool_adaptive_fallback"] = 1.0

        selected = max(k_valid, k_dilute, k_cover, k_dist)
        if selected > capacity:
            if capacity < k_valid:
                raise ValueError("Not enough unique images to satisfy shared-K identity anchors")
            selected = capacity
            diagnostics["pool_k_capacity_limited"] = 1.0

        diagnostics.update(
            {
                "pool_k_valid": float(k_valid),
                "pool_k_dilute": float(k_dilute),
                "pool_k_dist": float(k_dist),
                "pool_k_cover": float(k_cover),
                "pool_k_valid_target": float(k_valid),
                "pool_k_dilute_target": float(k_dilute),
                "pool_k_dist_target": float(k_dist),
                "pool_k_cover_target": float(k_cover),
                "pool_k_dist_found": float(dist_found),
            }
        )
        return selected, diagnostics

    def _record_cluster(self, record_index):
        if self.cluster_assignments is None:
            return 0
        return int(self.cluster_assignments[record_index].item())

    def _select_anchor_indices(self):
        anchors = []
        for pid in sorted(self.pid_to_record_indices.keys()):
            candidates = self.pid_to_record_indices[pid]
            best = min(
                candidates,
                key=lambda idx: (
                    self.coverage_counts[self.records[idx].image_id],
                    self.records[idx].image_id,
                ),
            )
            anchors.append(best)
        return anchors

    def _select_shared_records(self, model):
        if self.cluster_assignments is None:
            self._compute_clusters(model)
        selected_k, diagnostics = self._choose_shared_k(model)
        anchor_indices = self._select_anchor_indices()
        if selected_k < len(anchor_indices):
            raise ValueError("selected shared-K pool cannot fit required identity anchors")

        selected = list(anchor_indices)
        selected_set = set(selected)
        num_clusters = int(self.cluster_distribution.shape[0]) if self.cluster_distribution is not None else 1
        cluster_counts = torch.zeros(num_clusters)
        for idx in selected:
            cluster_counts[self._record_cluster(idx)] += 1

        expected = self.cluster_distribution if self.cluster_distribution is not None else torch.ones(1)
        all_candidates = [idx for idx in range(len(self.records)) if idx not in selected_set]
        all_candidates.sort(
            key=lambda idx: (
                self.coverage_counts[self.records[idx].image_id],
                self.records[idx].image_id,
            )
        )
        while len(selected) < selected_k and all_candidates:
            target_counts = expected * selected_k
            deficits = target_counts - cluster_counts
            cluster_order = deficits.argsort(descending=True).tolist()
            picked = None
            for cluster_id in cluster_order:
                for candidate in all_candidates:
                    if self._record_cluster(candidate) == int(cluster_id):
                        picked = candidate
                        break
                if picked is not None:
                    break
            if picked is None:
                picked = all_candidates[0]
            all_candidates.remove(picked)
            selected.append(picked)
            selected_set.add(picked)
            cluster_counts[self._record_cluster(picked)] += 1

        observed = cluster_counts / cluster_counts.sum().clamp_min(1.0)
        dist = self._distribution_distance(observed, expected)
        selected_records = [self.records[idx] for idx in selected]
        self.rng.shuffle(selected_records)
        for record in selected_records:
            self.coverage_counts[record.image_id] += 1

        diagnostics.update(
            {
                "pool_shared_k_used": 1.0,
                "pool_selected_k": float(selected_k),
                "pool_final_pool_size": float(len(selected_records)),
                "pool_num_required_positives": float(len(anchor_indices)),
                "pool_num_inserted_positives": float(len(anchor_indices)),
                "pool_positive_ratio": float(len(anchor_indices)) / max(1.0, float(len(selected_records))),
                "pool_cluster_distribution_distance": float(dist),
                "pool_missing_positive_count": 0.0,
                "pool_cluster_shortage_count": float(max(0, selected_k - len(selected_records))),
                "pool_coverage_horizon": float(getattr(self.args, "pool_coverage_epochs", 1)),
                "pool_coverage_seen": float(sum(1 for v in self.coverage_counts.values() if v > 0)),
                "pool_coverage_remaining": float(
                    self.capacity - sum(1 for v in self.coverage_counts.values() if v > 0)
                ),
            }
        )
        return selected_records, diagnostics

    def _base_diagnostics(self, device, reused):
        return {
            "pool_interval_id": _as_float(self.interval_id, device),
            "pool_interval_reused": _as_float(1.0 if reused else 0.0, device),
            "pool_shared_k_used": _as_float(0.0, device),
            "pool_selected_k": _as_float(self.capacity, device),
            "pool_final_pool_size": _as_float(self.capacity, device),
        }

    def _tensor_diagnostics(self, diagnostics, device):
        return {key: _as_float(value, device) for key, value in diagnostics.items()}

    def _build_full_cache(self, model, epoch, step):
        self._mark_refresh(epoch, step)
        cache = self._encode_records(model, self.records, cache_prototypes=True)
        device = cache["host_image_features"].device
        diagnostics = self._base_diagnostics(device, reused=False)
        return self._cache_diagnostics(cache, diagnostics)

    def _build_shared_cache(self, model, epoch, step):
        self._mark_refresh(epoch, step)
        records, diagnostics = self._select_shared_records(model)
        cache = self._encode_records(model, records, cache_prototypes=True)
        device = cache["host_image_features"].device
        base = self._base_diagnostics(device, reused=False)
        base.update(self._tensor_diagnostics(diagnostics, device))
        return self._cache_diagnostics(cache, base)

    def _ensure_frozen_cache(self, model):
        if self.frozen_cache is not None and self.frozen_rankings is not None:
            return
        self.frozen_cache = self._encode_records(model, self.records, cache_prototypes=True)
        device = self.frozen_cache["host_image_features"].device
        gallery = F.normalize(self.frozen_cache["host_image_features"].float(), p=2, dim=1)
        rank_depth = min(
            max(int(getattr(self.args, "pool_k", 0)), int(getattr(self.args, "top_m", 1))),
            gallery.shape[0],
        )

        text_dataset = TargetTextDataset(
            self.query_records,
            text_length=getattr(self.args, "text_length", 77),
            truncate=True,
        )
        text_loader = DataLoader(
            text_dataset,
            batch_size=getattr(self.args, "test_batch_size", getattr(self.args, "batch_size", 64)),
            shuffle=False,
            num_workers=getattr(self.args, "num_workers", 0),
            collate_fn=_collate_text,
        )
        was_training = model.training
        model.eval()
        rankings = []
        with torch.no_grad():
            for _, _, caption_ids in text_loader:
                caption_ids = caption_ids.to(device)
                text_features = F.normalize(model.encode_text(caption_ids).float(), p=2, dim=1)
                rankings.append((text_features @ gallery.t()).topk(rank_depth, dim=1).indices.cpu())
        if was_training:
            model.train()
        self.frozen_rankings = torch.cat(rankings, dim=0)
        diagnostics = self._base_diagnostics(device, reused=False)
        diagnostics.update(
            {
                "frozen_indices_used": _as_float(1.0, device),
                "frozen_index_depth": _as_float(rank_depth, device),
            }
        )
        self.frozen_cache["diagnostics"] = diagnostics

    def get_train_cache(self, model, batch, epoch, step):
        if getattr(self.args, "use_freeze_indices", False):
            self._ensure_frozen_cache(model)
            if "index" not in batch:
                raise KeyError("frozen-index target enrichment requires batch['index']")
            indices = batch["index"].detach().cpu().long()
            if indices.numel() > 0:
                if indices.min().item() < 0 or indices.max().item() >= self.frozen_rankings.shape[0]:
                    raise ValueError("batch index is out of frozen ranking range")
            cache = dict(self.frozen_cache)
            cache["top_indices"] = self.frozen_rankings[indices].to(_model_device(model))
            device = cache["host_image_features"].device
            diagnostics = dict(cache.get("diagnostics", {}))
            diagnostics["pool_interval_reused"] = _as_float(1.0, device)
            cache["diagnostics"] = diagnostics
            return cache

        if getattr(self.args, "use_shared_k", False):
            refresh = self._should_refresh(self.shared_cache, epoch, step)
            if refresh:
                self.shared_cache = self._build_shared_cache(model, epoch, step)
            else:
                device = self.shared_cache["host_image_features"].device
                diagnostics = dict(self.shared_cache.get("diagnostics", {}))
                diagnostics["pool_interval_reused"] = _as_float(1.0, device)
                self.shared_cache["diagnostics"] = diagnostics
            return self.shared_cache

        refresh = self._should_refresh(self.full_cache, epoch, step)
        if refresh:
            self.full_cache = self._build_full_cache(model, epoch, step)
        else:
            device = self.full_cache["host_image_features"].device
            diagnostics = dict(self.full_cache.get("diagnostics", {}))
            diagnostics["pool_interval_reused"] = _as_float(1.0, device)
            self.full_cache["diagnostics"] = diagnostics
        return self.full_cache

    def build_eval_cache(self, model, img_loader):
        device = _model_device(model)
        was_training = model.training
        model.eval()
        pids, image_ids, host_features, retrieval_features, prototypes = [], [], [], [], []
        running_index = 0
        with torch.no_grad():
            for batch_pids, images in img_loader:
                images = images.to(device)
                encoded = model.encode_target_image_cache(images, cache_prototypes=True)
                batch_size = images.shape[0]
                pids.append(batch_pids.to(device))
                image_ids.append(torch.arange(running_index, running_index + batch_size, device=device))
                running_index += batch_size
                host_features.append(encoded["host_image_features"].detach())
                retrieval_features.append(encoded["retrieval_features"].detach())
                prototypes.append(encoded["prototypes"].detach())
        if was_training:
            model.train()
        cache = {
            "host_image_features": torch.cat(host_features, dim=0),
            "retrieval_features": torch.cat(retrieval_features, dim=0),
            "prototypes": torch.cat(prototypes, dim=0),
            "pids": torch.cat(pids, dim=0).long(),
            "image_ids": torch.cat(image_ids, dim=0).long(),
            "diagnostics": {},
        }
        return cache
