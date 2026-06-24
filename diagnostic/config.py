"""CLI parsing, IRRA config loading, and deterministic diagnostic helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch

from diagnostic.constants import BOOTSTRAP_UNITS, CUE_SCORERS, DATASET_NAMES, RETRIEVER_NAMES


IRRA_DEFAULTS: Dict[str, Any] = {
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
        description="Generalized cue-swap diagnostic for frozen IRRA retrieval sensitivity"
    )
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--retriever_name", choices=RETRIEVER_NAMES, required=True)
    parser.add_argument("--retriever_config", type=Path, required=True)
    parser.add_argument("--retriever_checkpoint", type=Path, required=True)
    parser.add_argument("--cue_scorer", choices=CUE_SCORERS, default="off_the_shelf_clip")
    parser.add_argument("--clip_model_name", default="ViT-B/16")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--root_dir", type=Path, default=None)
    parser.add_argument("--cases_file", type=Path, default=None)
    parser.add_argument("--auto_cases", action="store_true")
    parser.add_argument("--cue_vocab_file", type=Path, default=None)
    parser.add_argument("--gallery_size", type=int, default=500)
    parser.add_argument("--dense_ratio", type=float, default=0.5)
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score_mode", choices=["auto", "global"], default="auto")
    parser.add_argument("--lambda_global", type=float, default=0.68)
    parser.add_argument("--lambda_contrast", type=float, default=0.5)
    parser.add_argument("--cue_threshold_quantile", type=float, default=0.75)
    parser.add_argument("--tau_density", type=float, default=0.02)
    parser.add_argument("--min_pair_cue_shift", type=float, default=0.0)
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=123)
    parser.add_argument(
        "--bootstrap_unit",
        choices=BOOTSTRAP_UNITS,
        default="unique_query",
        help=(
            "Cluster unit for confidence intervals: case_query preserves the original "
            "case-query-instance bootstrap, unique_query resamples underlying query_id "
            "clusters, both writes both analyses and uses unique_query as primary."
        ),
    )
    parser.add_argument("--enable_random_control", action="store_true")
    parser.add_argument("--save_galleries", action="store_true")
    parser.add_argument("--save_image_paths", action="store_true")
    parser.add_argument("--max_queries_per_case", type=int, default=None)
    parser.add_argument("--min_queries_per_auto_case", type=int, default=10)
    parser.add_argument("--max_auto_cases", type=int, default=None)
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--neutral_strategy", choices=["low_affinity", "random"], default="low_affinity")
    parser.add_argument("--neutral_pool_factor", type=int, default=5)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _parse_tuple(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple):
        return tuple(int(x) for x in value)
    if isinstance(value, list):
        return tuple(int(x) for x in value)
    if isinstance(value, str):
        numbers = [int(part) for part in re.findall(r"\d+", value)]
        if len(numbers) == 2:
            return numbers[0], numbers[1]
    raise ValueError(f"Could not parse img_size from config value: {value!r}")


def validate_args(args: argparse.Namespace) -> None:
    if args.cases_file is not None and args.auto_cases:
        raise ValueError("Provide either --cases_file or --auto_cases, not both.")
    if args.cases_file is None:
        args.auto_cases = True
    for path_arg in ("retriever_config", "retriever_checkpoint", "cases_file", "cue_vocab_file"):
        path = getattr(args, path_arg)
        if path is not None and not Path(path).exists():
            raise FileNotFoundError(f"--{path_arg} does not exist: {path}")
    if args.gallery_size <= 0:
        raise ValueError("--gallery_size must be positive")
    if not 0.0 <= args.dense_ratio <= 1.0:
        raise ValueError("--dense_ratio must be in [0, 1]")
    if args.num_trials <= 0:
        raise ValueError("--num_trials must be positive")
    if not 0.0 <= args.lambda_global <= 1.0:
        raise ValueError("--lambda_global must be in [0, 1]")
    if args.tau_density <= 0:
        raise ValueError("--tau_density must be positive")
    if not 0.0 <= args.cue_threshold_quantile <= 1.0:
        raise ValueError("--cue_threshold_quantile must be in [0, 1]")
    if args.min_pair_cue_shift < 0:
        raise ValueError("--min_pair_cue_shift must be non-negative")
    if args.bootstrap_iters <= 0:
        raise ValueError("--bootstrap_iters must be positive")
    if args.neutral_pool_factor <= 0:
        raise ValueError("--neutral_pool_factor must be positive")
    if args.enable_random_control:
        raise NotImplementedError(
            "--enable_random_control is reserved for a future random replacement control. "
            "The mandatory hardness-matched control is always enabled."
        )


def load_yaml_config(path: Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.load(handle, Loader=yaml.FullLoader)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Retriever config must be a YAML mapping: {path}")
    return dict(loaded)


def load_irra_config(args: argparse.Namespace) -> SimpleNamespace:
    cfg = dict(IRRA_DEFAULTS)
    cfg.update(load_yaml_config(args.retriever_config))
    cfg["dataset_name"] = args.dataset
    cfg["training"] = False
    cfg["val_dataset"] = args.split
    cfg["output_dir"] = str(args.output_dir)
    cfg["distributed"] = False
    if args.root_dir is not None:
        cfg["root_dir"] = str(args.root_dir)
    if args.test_batch_size is not None:
        cfg["test_batch_size"] = args.test_batch_size
        cfg["batch_size"] = args.test_batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    cfg["img_size"] = _parse_tuple(cfg.get("img_size", (384, 128)))
    cfg["milestones"] = tuple(cfg.get("milestones", (20, 50)))
    return SimpleNamespace(**cfg)


def load_repo_args(args: argparse.Namespace) -> SimpleNamespace:
    return load_irra_config(args)


def set_deterministic(seed: int, include_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if not include_torch:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return (int(base_seed) + int(digest[:8], 16)) % (2**32)


def stable_int_seed(*parts: Any, modulo: int = 2**32) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) % modulo


def resolve_device(device_name: str, dry_run: bool = False) -> torch.device:
    device = torch.device(device_name)
    if dry_run:
        return device
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return device


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, SimpleNamespace):
        return {key: jsonable(value) for key, value in vars(obj).items()}
    if isinstance(obj, dict):
        return {str(key): jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


json_safe = jsonable


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)


def config_payload(args: argparse.Namespace, irra_args: SimpleNamespace | None, resolved_device: Any) -> dict[str, Any]:
    return {
        "args": vars(args),
        "repo_args": irra_args if irra_args is not None else {},
        "resolved_device": str(resolved_device),
        "bootstrap_unit": args.bootstrap_unit,
        "scientific_note": (
            "Uses an off-the-shelf CLIP cue scorer for external cue-affinity scores "
            "and a frozen IRRA retriever for retrieval sensitivity."
        ),
    }

