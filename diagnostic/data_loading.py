"""IRRA split loading and normalized diagnostic records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QueryRecord:
    query_id: int
    text: str
    pid: int


@dataclass(frozen=True)
class GalleryRecord:
    image_id: int
    path: str
    pid: int


@dataclass
class SplitData:
    dataset: Any
    query_records: list[QueryRecord]
    gallery_records: list[GalleryRecord]
    query_pids: Any
    gallery_pids: Any
    gallery_paths: list[str]
    img_loader: Any
    txt_loader: Any
    num_classes: int


SplitMetadata = SplitData


def _dataset_factory():
    from datasets.cuhkpedes import CUHKPEDES
    from datasets.icfgpedes import ICFGPEDES
    from datasets.rstpreid import RSTPReid

    return {
        "CUHK-PEDES": CUHKPEDES,
        "ICFG-PEDES": ICFGPEDES,
        "RSTPReid": RSTPReid,
    }


def _direct_metadata(args: Any, split: str) -> SplitData:
    import json
    import os.path as op
    import numpy as np

    specs = {
        "CUHK-PEDES": ("CUHK-PEDES", "reid_raw.json", "file_path"),
        "ICFG-PEDES": ("ICFG-PEDES", "ICFG-PEDES.json", "file_path"),
        "RSTPReid": ("RSTPReid", "data_captions.json", "img_path"),
    }
    dataset_dir, anno_name, path_key = specs[args.dataset_name]
    dataset_root = op.join(args.root_dir, dataset_dir)
    img_dir = op.join(dataset_root, "imgs")
    anno_path = op.join(dataset_root, anno_name)
    if not op.exists(anno_path):
        raise RuntimeError(f"'{anno_path}' is not available")
    with open(anno_path, "r", encoding="utf-8") as handle:
        annotations = json.load(handle)

    selected = [anno for anno in annotations if anno.get("split") == split]
    train_ids = {int(anno["id"]) for anno in annotations if anno.get("split") == "train"}
    query_records: list[QueryRecord] = []
    gallery_records: list[GalleryRecord] = []
    for image_id, anno in enumerate(selected):
        pid = int(anno["id"])
        img_path = op.join(img_dir, anno[path_key])
        gallery_records.append(GalleryRecord(image_id=image_id, path=img_path, pid=pid))
        for caption in anno.get("captions", []):
            query_records.append(QueryRecord(query_id=len(query_records), text=str(caption), pid=pid))

    return SplitData(
        dataset=None,
        query_records=query_records,
        gallery_records=gallery_records,
        query_pids=np.asarray([r.pid for r in query_records], dtype=np.int64),
        gallery_pids=np.asarray([r.pid for r in gallery_records], dtype=np.int64),
        gallery_paths=[r.path for r in gallery_records],
        img_loader=None,
        txt_loader=None,
        num_classes=len(train_ids),
    )


def load_split(args: Any, split: str, metadata_only: bool = False) -> SplitData:
    import numpy as np

    if metadata_only:
        return _direct_metadata(args, split)

    try:
        factory = _dataset_factory()
        dataset = factory[args.dataset_name](root=args.root_dir, verbose=False)
        ds = dataset.val if split == "val" else dataset.test

        query_records = [
            QueryRecord(query_id=i, text=str(text), pid=int(pid))
            for i, (pid, text) in enumerate(zip(ds["caption_pids"], ds["captions"]))
        ]
        gallery_records = [
            GalleryRecord(image_id=i, path=str(path), pid=int(pid))
            for i, (pid, path) in enumerate(zip(ds["image_pids"], ds["img_paths"]))
        ]

        query_pids = np.asarray([r.pid for r in query_records], dtype=np.int64)
        gallery_pids = np.asarray([r.pid for r in gallery_records], dtype=np.int64)
        gallery_paths = [r.path for r in gallery_records]

        from torch.utils.data import DataLoader
        from datasets.bases import ImageDataset, TextDataset
        from datasets.build import build_transforms

        transform = build_transforms(img_size=args.img_size, is_train=False)
        img_set = ImageDataset(ds["image_pids"], ds["img_paths"], transform)
        txt_set = TextDataset(ds["caption_pids"], ds["captions"], text_length=args.text_length)
        img_loader = DataLoader(
            img_set,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        txt_loader = DataLoader(
            txt_set,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        return SplitData(
            dataset=dataset,
            query_records=query_records,
            gallery_records=gallery_records,
            query_pids=query_pids,
            gallery_pids=gallery_pids,
            gallery_paths=gallery_paths,
            img_loader=img_loader,
            txt_loader=txt_loader,
            num_classes=len(dataset.train_id_container),
        )
    except ModuleNotFoundError as exc:
        if exc.name != "torchvision":
            raise
        return _load_split_without_torchvision(args, split)


def load_split_metadata(repo_args: Any, split: str) -> SplitMetadata:
    return load_split(repo_args, split, metadata_only=True)


class _EvalTransform:
    def __init__(self, img_size):
        import numpy as np

        self.height, self.width = int(img_size[0]), int(img_size[1])
        self.mean = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.std = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def __call__(self, image):
        import numpy as np
        import torch

        image = image.resize((self.width, self.height))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - self.mean) / self.std
        array = array.transpose(2, 0, 1)
        return torch.from_numpy(array)


class _LocalImageDataset:
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image

        record = self.records[index]
        image = Image.open(record.path).convert("RGB")
        return record.pid, self.transform(image)


class _LocalTextDataset:
    def __init__(self, records, text_length):
        self.records = records
        self.text_length = text_length
        from utils.simple_tokenizer import SimpleTokenizer

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch

        record = self.records[index]
        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self.tokenizer.encode(record.text) + [eot_token]
        result = torch.zeros(self.text_length, dtype=torch.long)
        if len(tokens) > self.text_length:
            tokens = tokens[: self.text_length]
            tokens[-1] = eot_token
        result[: len(tokens)] = torch.tensor(tokens)
        return record.pid, result


def _load_split_without_torchvision(args: Any, split: str) -> SplitData:
    from torch.utils.data import DataLoader

    data = _direct_metadata(args, split)
    transform = _EvalTransform(args.img_size)
    data.img_loader = DataLoader(
        _LocalImageDataset(data.gallery_records, transform),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    data.txt_loader = DataLoader(
        _LocalTextDataset(data.query_records, args.text_length),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return data


def empty_split() -> SplitData:
    import numpy as np

    return SplitData(
        dataset=None,
        query_records=[],
        gallery_records=[],
        query_pids=np.asarray([], dtype=np.int64),
        gallery_pids=np.asarray([], dtype=np.int64),
        gallery_paths=[],
        img_loader=None,
        txt_loader=None,
        num_classes=0,
    )
