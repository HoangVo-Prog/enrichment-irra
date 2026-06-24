"""Independent off-the-shelf CLIP cue scorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from diagnostic.constants import DEFAULT_PROMPT_TEMPLATES


@dataclass
class CueScorerOutput:
    cues: list[str]
    affinities: dict[str, np.ndarray]
    prompts_by_cue: dict[str, list[str]]
    image_features_shape: tuple[int, int]

    @property
    def scores(self) -> dict[str, np.ndarray]:
        return self.affinities

    @property
    def prompts(self) -> dict[str, list[str]]:
        return self.prompts_by_cue


class OffTheShelfCLIPCueScorer:
    """Separate CLIP scorer loaded from original CLIP weights, not IRRA checkpoint."""

    def __init__(
        self,
        model_name: str,
        repo_args_or_device,
        device=None,
        logger=None,
        image_size: tuple[int, int] | None = None,
        stride_size: int | None = None,
        prompt_templates: Iterable[str] = DEFAULT_PROMPT_TEMPLATES,
    ) -> None:
        import torch
        from model.clip_model import build_CLIP_from_openai_pretrained

        self.model_name = model_name
        if device is None:
            self.device = torch.device(repo_args_or_device)
            if image_size is None or stride_size is None:
                raise ValueError("image_size and stride_size are required when repo args are not provided")
        else:
            self.device = torch.device(device)
            image_size = tuple(getattr(repo_args_or_device, "img_size", image_size or (384, 128)))
            stride_size = int(getattr(repo_args_or_device, "stride_size", stride_size or 16))
        self.image_size = tuple(image_size)
        self.stride_size = int(stride_size)
        self.prompt_templates = tuple(prompt_templates)
        if logger is not None:
            logger.info("Loading off-the-shelf CLIP cue scorer: %s", model_name)
        self.model, _ = build_CLIP_from_openai_pretrained(model_name, self.image_size, self.stride_size)
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _pool_image_features(self, features):
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim == 3:
            return features[:, 0, :]
        return features

    def _pool_text_features(self, features, tokens):
        import torch

        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim == 3:
            return features[torch.arange(features.shape[0], device=features.device), tokens.argmax(dim=-1)]
        return features

    def encode_gallery_images(self, img_loader, logger=None):
        import torch
        import torch.nn.functional as F

        features = []
        for batch_idx, (_pid, image) in enumerate(img_loader):
            image = image.to(self.device, non_blocking=True)
            with torch.no_grad():
                encoded = self._pool_image_features(self.model.encode_image(image))
                encoded = F.normalize(encoded.float(), p=2, dim=1)
            features.append(encoded.cpu())
            if logger is not None and (batch_idx + 1) % 50 == 0:
                logger.info("CLIP cue image batches encoded: %d", batch_idx + 1)
        return torch.cat(features, 0)

    def encode_cues(self, cues: Sequence[str], text_length: int):
        import torch
        import torch.nn.functional as F
        from utils.simple_tokenizer import SimpleTokenizer

        tokenizer = SimpleTokenizer()
        cue_features = []
        cue_prompt_map = {}
        for cue in cues:
            prompts = [template.format(cue=cue) for template in self.prompt_templates]
            cue_prompt_map[cue] = prompts
            tokens = torch.stack([self._tokenize(prompt, tokenizer, text_length) for prompt in prompts], dim=0).to(self.device)
            with torch.no_grad():
                encoded = self._pool_text_features(self.model.encode_text(tokens), tokens)
                encoded = F.normalize(encoded.float(), p=2, dim=1)
                encoded = F.normalize(encoded.mean(dim=0, keepdim=True), p=2, dim=1)
            cue_features.append(encoded.cpu())
        return torch.cat(cue_features, 0), cue_prompt_map

    def _tokenize(self, caption: str, tokenizer, text_length: int):
        import torch

        sot_token = tokenizer.encoder["<|startoftext|>"]
        eot_token = tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]
        result = torch.zeros(text_length, dtype=torch.long)
        if len(tokens) > text_length:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        result[: len(tokens)] = torch.tensor(tokens)
        return result

    def score(self, cues: Sequence[str], img_loader, text_length: int, logger=None) -> CueScorerOutput:
        import torch

        cues = list(dict.fromkeys(cues))
        if logger is not None:
            logger.info("Scoring %d cues with off-the-shelf CLIP model %s", len(cues), self.model_name)
        image_features = self.encode_gallery_images(img_loader, logger=logger)
        cue_features, prompt_map = self.encode_cues(cues, text_length=text_length)
        with torch.no_grad():
            matrix = image_features.float() @ cue_features.float().t()
        matrix_np = matrix.numpy()
        scores = {cue: matrix_np[:, idx].astype(np.float32, copy=False) for idx, cue in enumerate(cues)}
        return CueScorerOutput(
            cues=cues,
            affinities=scores,
            prompts_by_cue=prompt_map,
            image_features_shape=(int(image_features.shape[0]), int(image_features.shape[1])),
        )


def threshold_rows(
    scores: dict[str, np.ndarray],
    prompts: dict[str, list[str]],
    quantile: float,
) -> tuple[dict[str, float], list[dict]]:
    thresholds: dict[str, float] = {}
    rows = []
    for cue, values in sorted(scores.items()):
        values = np.asarray(values, dtype=np.float64)
        threshold = float(np.quantile(values, quantile)) if values.size else 0.0
        thresholds[cue] = threshold
        rows.append(
            {
                "cue": cue,
                "threshold": threshold,
                "quantile": quantile,
                "mean": float(values.mean()) if values.size else 0.0,
                "std": float(values.std()) if values.size else 0.0,
                "min": float(values.min()) if values.size else 0.0,
                "max": float(values.max()) if values.size else 0.0,
                "num_gallery": int(values.size),
                "prompts_json": str(prompts.get(cue, [])),
            }
        )
    return thresholds, rows
