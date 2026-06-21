"""Argument parsing, config loading, and deterministic helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from types import SimpleNamespace
from typing import Any

from diagnostic.constants import (
    BOOTSTRAP_UNITS,
    CUE_SCORER_NAMES,
    DATASET_NAMES,
    NEUTRAL_STRATEGIES,
    RETRIEVER_NAMES,
    SCORE_MODES,
    SPLIT_NAMES,
)


IRRA_DEFAULTS = {
    "local_rank": 0,
    "name": "diagnostic",
    "output_dir": "logs",
    "log_period": 100,
    "eval_period": 1,
    "val_dataset": "test",
    "resume": False,
    "resume_ckpt_file": "",
    "pretrain_choice": "ViT-B/16",
    "temperature": 0.02,
    "img_aug": False,
    "cmt_depth": 4,
    "masked_token_rate": 0.8,
    "masked_token_unchanged_rate": 0.1,
    "lr_factor": 5.0,
    "MLM": False,
    "loss_names": "sdm+id+mlm",
    "mlm_loss_weight": 1.0,
    "id_loss_weight": 1.0,
    "img_size": (384, 128),
    "stride_size": 16,
    "text_length": 77,
    "vocab_size": 49408,
    "optimizer": "Adam",
    "lr": 1e-5,
    "bias_lr_factor": 2.0,
    "momentum": 0.9,
    "weight_decay": 4e-5,
    "weight_decay_bias": 0.0,
    "alpha": 0.9,
    "beta": 0.999,
    "num_epoch": 60,
    "milestones": (20, 50),
    "gamma": 0.1,
    "warmup_factor": 0.1,
    "warmup_epochs": 5,
    "warmup_method": "linear",
    "lrscheduler": "cosine",
    "target_lr": 0,
    "power": 0.9,
    "dataset_name": "CUHK-PEDES",
    "sampler": "random",
    "num_instance": 4,
    "root_dir": "./data",
    "batch_size": 128,
    "test_batch_size": 512,
    "num_workers": 8,
    "training": False,
    "distributed": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cue-swap diagnostic for a frozen IRRA text-based person search "
            "retriever using an off-the-shelf CLIP cue scorer."
        )
    )
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--split", choices=SPLIT_NAMES, default="test")
    parser.add_argument("--retriever_name", choices=RETRIEVER_NAMES, default="irra")
    parser.add_argument("--retriever_config", required=True)
    parser.add_argument("--retriever_checkpoint", required=True)
    parser.add_argument("--cue_scorer", choices=CUE_SCORER_NAMES, default="off_the_shelf_clip")
    parser.add_argument("--clip_model_name", default="ViT-B/16")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--cases_file", default=None)
    parser.add_argument("--auto_cases", action="store_true")
    parser.add_argument("--cue_vocab_file", default=None)
    parser.add_argument("--gallery_size", type=int, default=500)
    parser.add_argument("--dense_ratio", type=float, default=0.9)
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score_mode", choices=SCORE_MODES, default="auto")
    parser.add_argument("--lambda_global", type=float, default=1.0)
    parser.add_argument("--lambda_contrast", type=float, default=0.8)
    parser.add_argument("--cue_threshold_quantile", type=float, default=0.75)
    parser.add_argument("--tau_density", type=float, default=0.02)
    parser.add_argument("--min_pair_cue_shift", type=float, default=0.0)
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=123)
    parser.add_argument("--bootstrap_unit", choices=BOOTSTRAP_UNITS, default="unique_query")
    parser.add_argument("--enable_random_control", action="store_true")
    parser.add_argument("--save_galleries", action="store_true")
    parser.add_argument("--save_image_paths", action="store_true")
    parser.add_argument("--max_queries_per_case", type=int, default=None)
    parser.add_argument("--min_queries_per_auto_case", type=int, default=30)
    parser.add_argument("--max_auto_cases", type=int, default=None)
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--neutral_strategy", choices=NEUTRAL_STRATEGIES, default="low_affinity")
    parser.add_argument("--neutral_pool_factor", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def _parse_tuple(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple):
        return tuple(int(x) for x in value)
    if isinstance(value, list):
        return tuple(int(x) for x in value)
    if isinstance(value, str):
        text = value.strip().strip("()[]")
        parts = [p.strip() for p in text.replace("x", ",").split(",") if p.strip()]
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    raise ValueError(f"Cannot parse img_size={value!r} as a height,width tuple")


def load_yaml_config(path: str) -> dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Retriever config must contain a mapping: {path}")
    return data


def load_irra_config(args: argparse.Namespace) -> SimpleNamespace:
    merged = dict(IRRA_DEFAULTS)
    merged.update(load_yaml_config(args.retriever_config))
    merged["dataset_name"] = args.dataset
    merged["training"] = False
    merged["val_dataset"] = args.split
    merged["output_dir"] = os.path.dirname(os.path.abspath(args.retriever_checkpoint)) or "."
    if args.root_dir is not None:
        merged["root_dir"] = args.root_dir
    if args.test_batch_size is not None:
        merged["test_batch_size"] = args.test_batch_size
        merged["batch_size"] = args.test_batch_size
    if args.num_workers is not None:
        merged["num_workers"] = args.num_workers
    merged["img_size"] = _parse_tuple(merged.get("img_size", (384, 128)))
    if isinstance(merged.get("milestones"), list):
        merged["milestones"] = tuple(merged["milestones"])
    return SimpleNamespace(**merged)


def validate_args(args: argparse.Namespace) -> None:
    if args.gallery_size <= 0:
        raise ValueError("--gallery_size must be positive")
    if not 0.0 <= args.dense_ratio <= 1.0:
        raise ValueError("--dense_ratio must be in [0, 1]")
    if args.num_trials <= 0:
        raise ValueError("--num_trials must be positive")
    if not 0.0 <= args.cue_threshold_quantile <= 1.0:
        raise ValueError("--cue_threshold_quantile must be in [0, 1]")
    if args.tau_density <= 0:
        raise ValueError("--tau_density must be positive")
    if args.bootstrap_iters < 0:
        raise ValueError("--bootstrap_iters must be non-negative")
    if args.neutral_pool_factor <= 0:
        raise ValueError("--neutral_pool_factor must be positive")
    if args.enable_random_control:
        raise NotImplementedError("--enable_random_control is accepted but not implemented")
    if not args.dry_run:
        if not os.path.exists(args.retriever_config):
            raise FileNotFoundError(f"Retriever config not found: {args.retriever_config}")
        if not os.path.exists(args.retriever_checkpoint):
            raise FileNotFoundError(f"Retriever checkpoint not found: {args.retriever_checkpoint}")


def resolve_device(device_text: str, dry_run: bool = False) -> str:
    if dry_run:
        return device_text
    if device_text.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"
    return device_text


def set_deterministic(seed: int, include_torch: bool = True) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    if not include_torch:
        return
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def stable_int_seed(*parts: Any, modulo: int = 2**32) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) % modulo


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if hasattr(value, "__dict__"):
        return json_safe(vars(value))
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return str(value)


def config_payload(args: argparse.Namespace, irra_args: SimpleNamespace | None, resolved_device: str) -> dict[str, Any]:
    return {
        "bootstrap_unit": args.bootstrap_unit,
        "diagnostic_args": json_safe(vars(args)),
        "irra_config": json_safe(vars(irra_args)) if irra_args is not None else {},
        "resolved_device": resolved_device,
        "scientific_note": (
            "Uses an off-the-shelf CLIP cue scorer for external cue-affinity "
            "scores and a frozen IRRA retriever for retrieval sensitivity."
        ),
    }


def dump_json_payload(payload: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
