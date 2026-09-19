"""Small rigid-geometry helpers kept separate for testability."""

from __future__ import annotations

import torch


def invert_rigid_transform(transform: torch.Tensor) -> torch.Tensor:
    """Invert batched 4x4 rigid transforms without a generic matrix inverse."""
    if transform.ndim == 2:
        transform = transform.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    if transform.ndim != 3 or transform.shape[-2:] != (4, 4):
        raise ValueError("transform must have shape [4,4] or [B,4,4]")
    rotation = transform[:, :3, :3]
    translation = transform[:, :3, 3:4]
    result = torch.eye(
        4, device=transform.device, dtype=transform.dtype
    ).unsqueeze(0).repeat(transform.shape[0], 1, 1)
    result[:, :3, :3] = rotation.transpose(1, 2)
    result[:, :3, 3:4] = -rotation.transpose(1, 2) @ translation
    return result[0] if squeeze else result

