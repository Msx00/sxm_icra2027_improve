"""Appearance, geometry, routing and calibration objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from distrisurg.config import ExperimentConfig


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(1)
    if mask.shape[1] == 1 and value.shape[1] != 1:
        mask = mask.expand(-1, value.shape[1], -1, -1)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def charbonnier(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1.0e-3,
) -> torch.Tensor:
    return masked_mean(
        torch.sqrt((prediction - target).square() + epsilon * epsilon), mask
    )


def image_gradients(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = value[..., :, 1:] - value[..., :, :-1]
    dy = value[..., 1:, :] - value[..., :-1, :]
    return dx, dy


def gradient_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    prediction_dx, prediction_dy = image_gradients(prediction)
    target_dx, target_dy = image_gradients(target)
    mask_x = mask[..., :, 1:] & mask[..., :, :-1]
    mask_y = mask[..., 1:, :] & mask[..., :-1, :]
    return charbonnier(prediction_dx, target_dx, mask_x) + charbonnier(
        prediction_dy, target_dy, mask_y
    )


def seam_band(mask: torch.Tensor, width: int) -> torch.Tensor:
    kernel = max(1, int(width))
    if kernel % 2 == 0:
        kernel += 1
    value = mask.float()
    dilation = F.max_pool2d(value, kernel, stride=1, padding=kernel // 2)
    erosion = -F.max_pool2d(-value, kernel, stride=1, padding=kernel // 2)
    return (dilation - erosion) > 0.0


class DistriSurgLoss(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        prediction = output["target_rgb"]
        target = batch["target_rgb"]
        tissue = ~batch.get(
            "target_tool_mask",
            torch.zeros_like(output["hole_mask"]),
        ).bool()
        hole = output["hole_mask"].bool() & tissue
        visible = output["trusted_mask"].bool() & tissue
        seam = seam_band(output["hole_mask"].bool(), self.config.loss.seam_width) & tissue
        loss_hole = charbonnier(prediction, target, hole)
        loss_visible = charbonnier(prediction, target, visible)
        loss_seam = charbonnier(prediction, target, seam)
        loss_gradient = gradient_loss(prediction, target, hole | seam)

        source_depth_gt = batch.get("source_depth_gt", batch["source_depth"])
        source_valid_gt = batch.get(
            "source_depth_valid_gt", batch["source_depth_valid"]
        ).bool()
        depth_error = (
            output["source_depth_mean"] - source_depth_gt
        ).abs()
        depth_sigma = output["source_depth_sigma"].clamp_min(1.0e-3)
        loss_depth_nll = masked_mean(
            depth_error / depth_sigma + torch.log(depth_sigma), source_valid_gt
        )
        target_depth_valid = batch.get(
            "target_depth_valid", torch.zeros_like(output["target_depth"], dtype=torch.bool)
        ).bool()
        if target_depth_valid.any():
            target_depth_error = (
                output["target_depth"] - batch["target_depth"]
            ).abs() / batch["target_depth"].clamp_min(1.0)
            loss_target_depth = masked_mean(target_depth_error, target_depth_valid)
        else:
            loss_target_depth = prediction.sum() * 0.0

        if "cycle_source_rgb" in output:
            cycle_mask = (
                output["cycle_source_valid"].bool()
                & source_valid_gt
            )
            loss_cycle = charbonnier(
                output["cycle_source_rgb"], batch["source_rgb"], cycle_mask
            )
        else:
            loss_cycle = prediction.sum() * 0.0

        pseudo_synthesis = (
            (output["support"] < self.config.model.trusted_support)
            | (
                output["render_variance"]
                > self.config.model.trusted_variance_mm2
            )
            | (
                output["collision_entropy"]
                > self.config.model.trusted_entropy
            )
        ).float().detach()
        loss_router = F.binary_cross_entropy(
            output["synthesis_gate"].clamp(1.0e-5, 1.0 - 1.0e-5),
            pseudo_synthesis,
        )

        pixel_error = (prediction - target).abs().mean(dim=1, keepdim=True)
        risk_scale = 0.01 + 0.49 * output["risk"].clamp(0.0, 1.0)
        loss_uncertainty = masked_mean(
            pixel_error / risk_scale + torch.log(risk_scale), hole
        )
        weights = self.config.loss
        total = (
            weights.hole * loss_hole
            + weights.visible * loss_visible
            + weights.seam * loss_seam
            + weights.gradient * loss_gradient
            + weights.depth_nll * (loss_depth_nll + loss_target_depth)
            + weights.cycle * loss_cycle
            + weights.router * loss_router
            + weights.uncertainty * loss_uncertainty
        )
        known_drift = masked_mean(
            (prediction - output["warped_rgb"]).abs(), output["trusted_mask"]
        )
        return {
            "total": total,
            "hole": loss_hole,
            "visible": loss_visible,
            "seam": loss_seam,
            "gradient": loss_gradient,
            "depth_nll": loss_depth_nll,
            "target_depth": loss_target_depth,
            "cycle": loss_cycle,
            "router": loss_router,
            "uncertainty": loss_uncertainty,
            "known_drift": known_drift,
        }
