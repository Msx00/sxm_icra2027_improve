"""Atomic manifests and image writers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def write_json_atomic(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_rgb(path: str | Path, tensor: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = tensor.detach().float().cpu()
    if value.ndim == 4:
        value = value[0]
    array = (
        value.permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0
    ).round().astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def save_mask(path: str | Path, tensor: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = tensor.detach().cpu()
    while value.ndim > 2:
        value = value[0]
    array = value.bool().numpy().astype(np.uint8) * 255
    Image.fromarray(array, mode="L").save(path)


def save_heatmap(
    path: str | Path,
    tensor: torch.Tensor,
    maximum: float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = tensor.detach().float().cpu()
    while value.ndim > 2:
        value = value[0]
    array = value.numpy()
    finite = np.isfinite(array)
    if maximum is None:
        maximum = float(np.quantile(array[finite], 0.99)) if finite.any() else 1.0
    maximum = max(float(maximum), 1.0e-8)
    normalized = np.where(finite, np.clip(array / maximum, 0.0, 1.0), 0.0)
    Image.fromarray((normalized * 255.0).round().astype(np.uint8), mode="L").save(path)


def save_label_map(path: str | Path, tensor: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = tensor.detach().cpu()
    while value.ndim > 2:
        value = value[0]
    labels = value.numpy().astype(np.uint8)
    palette = np.asarray(
        [
            [40, 40, 40],
            [245, 196, 48],
            [50, 180, 90],
            [55, 125, 220],
            [220, 70, 70],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(palette[np.clip(labels, 0, len(palette) - 1)], mode="RGB").save(path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, device: torch.device) -> dict:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=True)
    return payload if isinstance(payload, dict) else {"model": payload}
