"""Frozen IRRA retriever adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RetrieverAdapter:
    name: str
    model: Any
    device: Any
    has_grab: bool = False

    def eval(self):
        self.model.eval()
        return self

    def encode_text(self, batch):
        import torch

        tokens = batch[1] if isinstance(batch, (tuple, list)) else batch
        tokens = tokens.to(self.device, non_blocking=True)
        with torch.no_grad():
            return self.model.encode_text(tokens)

    def encode_image(self, batch):
        import torch

        images = batch[1] if isinstance(batch, (tuple, list)) else batch
        images = images.to(self.device, non_blocking=True)
        with torch.no_grad():
            return self.model.encode_image(images)


def _state_dict_from_checkpoint(checkpoint: Any) -> Any:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict", "net"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_retriever(args: Any, num_classes: int, checkpoint_path: str, device: str) -> RetrieverAdapter:
    import logging
    import torch
    from model import build_model
    from utils.checkpoint import Checkpointer, load_state_dict

    logger = logging.getLogger("diagnostic.retriever")
    model = build_model(args, num_classes=num_classes)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        logger.info("Loading IRRA checkpoint through official Checkpointer")
        Checkpointer(model).load(checkpoint_path)
    else:
        logger.info("Loading IRRA checkpoint through official load_state_dict helper")
        load_state_dict(model, _state_dict_from_checkpoint(checkpoint))

    torch_device = torch.device(device)
    if torch_device.type == "cpu":
        model.float()
    model.to(torch_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return RetrieverAdapter(name="irra", model=model, device=torch_device, has_grab=False)
