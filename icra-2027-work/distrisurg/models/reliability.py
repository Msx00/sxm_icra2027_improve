"""RGB-D reliability, completion and surface feature encoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from distrisurg.config import DSSConfig, ModelConfig


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class ResidualConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.layers(value)


class DepthReliabilityEncoder(nn.Module):
    """Predict depth distribution, normals, confidence and transport features."""

    def __init__(self, model: ModelConfig, dss: DSSConfig) -> None:
        super().__init__()
        channels = model.reliability_channels
        self.model_config = model
        self.dss_config = dss
        self.stem = nn.Conv2d(5, channels, 5, padding=2)
        self.full = nn.Sequential(ResidualConv(channels), ResidualConv(channels))
        self.down = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            ResidualConv(channels * 2),
            nn.Conv2d(channels * 2, channels * 2, 3, stride=2, padding=1),
            ResidualConv(channels * 2),
        )
        self.context = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            ResidualConv(channels),
        )
        output_channels = model.source_feature_channels + 1 + 1 + 1 + 3
        self.head = nn.Conv2d(channels, output_channels, 3, padding=1)
        with torch.no_grad():
            # Start close to observed depth, moderate observed confidence, and
            # low rather than maximal uncertainty.
            self.head.bias.zero_()
            sigma_index = model.source_feature_channels + 1
            confidence_index = model.source_feature_channels + 2
            self.head.bias[sigma_index] = -2.0
            self.head.bias[confidence_index] = 0.5

    @staticmethod
    def _depth_scale(depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        scales = []
        for sample_depth, sample_valid in zip(depth, valid):
            values = sample_depth[sample_valid]
            if values.numel():
                scale = values.median()
            else:
                scale = depth.new_tensor(100.0)
            scales.append(scale.clamp_min(1.0))
        return torch.stack(scales).reshape(-1, 1, 1, 1)

    def forward(
        self,
        rgb: torch.Tensor,
        sparse_depth: torch.Tensor,
        valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [B,3,H,W]")
        if sparse_depth.shape != rgb[:, :1].shape or valid.shape != sparse_depth.shape:
            raise ValueError("depth and valid must have shape [B,1,H,W]")
        valid = valid.bool() & torch.isfinite(sparse_depth) & (sparse_depth > 0)
        depth_scale = self._depth_scale(sparse_depth, valid)
        normalized = torch.where(
            valid,
            sparse_depth / depth_scale,
            torch.zeros_like(sparse_depth),
        )
        input_value = torch.cat(
            (rgb, normalized.clamp(0.0, 5.0), valid.float()), dim=1
        )
        full = self.full(self.stem(input_value))
        context = self.down(full)
        context = F.interpolate(
            self.context(context), size=full.shape[-2:], mode="bilinear", align_corners=False
        )
        fused = self.fuse(torch.cat((full, context), dim=1))
        raw = self.head(fused)
        feature_channels = self.model_config.source_feature_channels
        source_features = raw[:, :feature_channels]
        cursor = feature_channels
        raw_depth = raw[:, cursor : cursor + 1]
        cursor += 1
        raw_sigma = raw[:, cursor : cursor + 1]
        cursor += 1
        raw_confidence = raw[:, cursor : cursor + 1]
        cursor += 1
        raw_normal = raw[:, cursor : cursor + 3]

        completion = (F.softplus(raw_depth) + 0.05) * depth_scale
        residual_ratio = self.model_config.max_depth_residual_ratio * torch.tanh(raw_depth)
        observed = sparse_depth * (1.0 + residual_ratio)
        depth_mean = torch.where(valid, observed, completion).clamp(
            self.dss_config.min_depth_mm, self.dss_config.max_depth_mm
        )
        sigma_fraction = torch.sigmoid(raw_sigma)
        depth_sigma = self.dss_config.min_sigma_mm + sigma_fraction * (
            self.dss_config.max_sigma_mm - self.dss_config.min_sigma_mm
        )
        confidence = torch.sigmoid(raw_confidence)
        confidence = torch.where(
            valid,
            0.5 + 0.5 * confidence,
            self.model_config.invalid_depth_confidence_scale * confidence,
        )
        normal_prior = torch.zeros_like(raw_normal)
        normal_prior[:, 2] = 1.0
        normal = F.normalize(raw_normal + normal_prior, dim=1, eps=1.0e-6)
        return {
            "source_features": source_features,
            "depth_mean": depth_mean,
            "depth_sigma": depth_sigma,
            "depth_scale": depth_scale,
            "confidence": confidence,
            "normal": normal,
            "observed_valid": valid,
            "render_valid": torch.isfinite(depth_mean) & (depth_mean > 0),
        }

