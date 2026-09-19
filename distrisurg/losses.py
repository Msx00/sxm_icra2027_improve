"""Appearance, geometry, routing and calibration objectives."""

from __future__ import annotations

from pathlib import Path

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


def gaussian_blur(value: torch.Tensor, sigma: float) -> torch.Tensor:
    """Differentiable channel-wise Gaussian blur with an FP32 compute path."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = max(1, int(3.0 * sigma + 0.5))
    coordinates = torch.arange(
        -radius, radius + 1, device=value.device, dtype=torch.float32
    )
    kernel = torch.exp(-0.5 * (coordinates / float(sigma)).square())
    kernel = kernel / kernel.sum()
    channels = value.shape[1]
    horizontal = kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1)
    with torch.autocast(device_type=value.device.type, enabled=False):
        result = F.conv2d(
            F.pad(value.float(), (radius, radius, 0, 0), mode="replicate"),
            horizontal,
            groups=channels,
        )
        result = F.conv2d(
            F.pad(result, (0, 0, radius, radius), mode="replicate"),
            vertical,
            groups=channels,
        )
    return result


def multi_scale_low_frequency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sigmas: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0),
) -> torch.Tensor:
    """Match low-frequency appearance only where completion meets the seam."""
    if not sigmas:
        raise ValueError("at least one low-frequency scale is required")
    mask = mask.bool()
    losses = []
    for sigma in sigmas:
        # Blur the support as well, retaining a soft boundary weighting without
        # allowing the much larger exterior region to dominate the objective.
        soft_mask = gaussian_blur(mask.float(), sigma).clamp(0.0, 1.0)
        losses.append(
            charbonnier(
                gaussian_blur(prediction, sigma),
                gaussian_blur(target, sigma),
                soft_mask,
            )
        )
    return torch.stack(losses).mean()


class VGG16FeatureExtractor(nn.Module):
    """Frozen multi-stage VGG16 features loaded strictly from a local file."""

    def __init__(self, weights_path: str) -> None:
        super().__init__()
        path = Path(weights_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"local VGG16 perceptual weights not found: {path}; "
                "set loss.perceptual=0 or provide loss.perceptual_weights_path"
            )
        try:
            from torchvision.models import vgg16
        except ImportError as error:
            raise RuntimeError(
                "torchvision is required when loss.perceptual > 0"
            ) from error
        model = vgg16(weights=None)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state_dict, dict):
            raise ValueError(f"invalid VGG16 checkpoint at {path}")
        model.load_state_dict(state_dict, strict=True)
        self.features = model.features
        self.stage_ends = frozenset((3, 8, 15, 22))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "VGG16FeatureExtractor":
        # There are no stochastic/norm layers in VGG16 features, but pinning
        # eval mode makes the frozen-backbone contract explicit.
        return super().train(False)

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for index, layer in enumerate(self.features):
            value = layer(value)
            if index in self.stage_ends:
                outputs.append(value)
            if index >= max(self.stage_ends):
                break
        return outputs


class MaskedPerceptualLoss(nn.Module):
    def __init__(self, feature_extractor: nn.Module) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor.requires_grad_(False).eval()
        self.register_buffer(
            "mean", torch.tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)
        )

    def train(self, mode: bool = True) -> "MaskedPerceptualLoss":
        super().train(mode)
        self.feature_extractor.eval()
        return self

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            prediction_normalized = (prediction.float() - self.mean) / self.std
            target_normalized = (target.float() - self.mean) / self.std
            prediction_features = self.feature_extractor(prediction_normalized)
            with torch.no_grad():
                target_features = self.feature_extractor(target_normalized)
            losses = []
            for predicted, expected in zip(prediction_features, target_features):
                resized_mask = F.interpolate(
                    mask.float(), size=predicted.shape[-2:], mode="area"
                )
                losses.append(masked_mean((predicted - expected).abs(), resized_mask))
            if not losses:
                raise RuntimeError("perceptual feature extractor returned no features")
            return torch.stack(losses).mean()


class DistriSurgLoss(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.perceptual_loss: MaskedPerceptualLoss | None = None
        if config.loss.perceptual > 0:
            self.perceptual_loss = MaskedPerceptualLoss(
                VGG16FeatureExtractor(config.loss.perceptual_weights_path)
            )

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
        appearance_mask = hole | seam
        loss_low_frequency = (
            multi_scale_low_frequency_loss(prediction, target, appearance_mask)
            if self.config.loss.low_frequency > 0
            else prediction.sum() * 0.0
        )
        loss_perceptual = (
            self.perceptual_loss(prediction, target, appearance_mask)
            if self.perceptual_loss is not None
            else prediction.sum() * 0.0
        )

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
        # Use logits in FP32. Probability-form BCE has singular gradients at
        # 0/1 and caused AMP overflow followed by CUDA Loss.cu assertions.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            loss_router = F.binary_cross_entropy_with_logits(
                output["synthesis_logits"].float(),
                pseudo_synthesis.float(),
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
            + weights.low_frequency * loss_low_frequency
            + weights.perceptual * loss_perceptual
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
            "low_frequency": loss_low_frequency,
            "perceptual": loss_perceptual,
            "depth_nll": loss_depth_nll,
            "target_depth": loss_target_depth,
            "cycle": loss_cycle,
            "router": loss_router,
            "uncertainty": loss_uncertainty,
            "known_drift": known_drift,
        }
