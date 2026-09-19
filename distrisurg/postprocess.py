"""Auditable inference-time repair for tiny disconnected synthesis regions."""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch


@torch.no_grad()
def repair_small_synthesis_regions(
    prediction_rgb: torch.Tensor,
    warped_rgb: torch.Tensor,
    trusted_mask: torch.Tensor,
    max_area_ratio: float,
    radius: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace only tiny untrusted components using Telea RGB inpainting.

    Large connected regions are deliberately left to the learned synthesis
    expert. The returned mask records every changed pixel for auditing.
    """
    if prediction_rgb.ndim != 4 or prediction_rgb.shape[1] != 3:
        raise ValueError("prediction_rgb must have shape [B, 3, H, W]")
    if warped_rgb.shape != prediction_rgb.shape:
        raise ValueError("warped_rgb must match prediction_rgb")
    if trusted_mask.shape != prediction_rgb[:, :1].shape:
        raise ValueError("trusted_mask must have shape [B, 1, H, W]")
    if not 0.0 <= max_area_ratio <= 1.0:
        raise ValueError("max_area_ratio must be in [0, 1]")
    if radius <= 0.0:
        raise ValueError("radius must be positive")

    repaired = prediction_rgb.clone()
    repair_mask = torch.zeros_like(trusted_mask, dtype=torch.bool)
    if max_area_ratio == 0.0:
        return repaired, repair_mask

    height, width = prediction_rgb.shape[-2:]
    max_area = max(1, math.ceil(max_area_ratio * height * width))
    for batch_index in range(prediction_rgb.shape[0]):
        untrusted = (~trusted_mask[batch_index, 0].bool()).detach().cpu().numpy()
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            untrusted.astype(np.uint8), connectivity=8
        )
        selected = np.zeros((height, width), dtype=np.uint8)
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area <= max_area:
                selected[labels == component] = 255
        if not selected.any():
            continue

        warp = warped_rgb[batch_index].detach().float().clamp(0.0, 1.0)
        warp_u8 = (
            warp.permute(1, 2, 0).cpu().numpy() * 255.0
        ).round().astype(np.uint8)
        filled_u8 = cv2.inpaint(warp_u8, selected, radius, cv2.INPAINT_TELEA)
        filled = torch.from_numpy(filled_u8).permute(2, 0, 1).to(
            device=repaired.device, dtype=repaired.dtype
        ) / 255.0
        selected_tensor = torch.from_numpy(selected > 0).to(repaired.device)
        repaired[batch_index] = torch.where(
            selected_tensor.unsqueeze(0), filled, repaired[batch_index]
        )
        repair_mask[batch_index, 0] = selected_tensor
    return repaired, repair_mask
