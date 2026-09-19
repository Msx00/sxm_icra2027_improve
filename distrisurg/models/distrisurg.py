"""End-to-end DistriSurg network."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from distrisurg.config import ExperimentConfig
from distrisurg.geometry import DistributionalSurfaceSplat, invert_rigid_transform

from .reliability import DepthReliabilityEncoder
from .router import TransportExpert, VisibilityRouter
from .uffc import UncertaintyConditionedUFFC


def spatially_consistent_trusted_mask(
    raw_trusted: torch.Tensor,
    render_valid: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """Suppress isolated synthesis islands without inventing warp support.

    Binary opening is applied to the *untrusted* mask, so coherent holes and
    disocclusions remain routed to synthesis while salt-and-pepper islands
    smaller than the renderer footprint return to geometric transport. Pixels
    with no completed DSS support can never become trusted.
    """
    if kernel_size <= 1:
        return raw_trusted.bool() & render_valid.bool()
    untrusted = (~raw_trusted.bool()).float()
    padding = kernel_size // 2
    eroded = -F.max_pool2d(
        -untrusted, kernel_size=kernel_size, stride=1, padding=padding
    )
    coherent_untrusted = F.max_pool2d(
        eroded, kernel_size=kernel_size, stride=1, padding=padding
    ) > 0.5
    return render_valid.bool() & ~coherent_untrusted


class DistriSurg(nn.Module):
    """Distributional transport plus visibility-constrained Fourier completion."""

    CONDITION_CHANNELS = 6

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.reliability = DepthReliabilityEncoder(config.model, config.dss)
        self.renderer = DistributionalSurfaceSplat(config.dss)
        rendered_channels = 3 + config.model.source_feature_channels
        network_input_channels = rendered_channels + self.CONDITION_CHANNELS
        self.transport = TransportExpert(
            network_input_channels, config.model.base_channels
        )
        # RGB, normalized depth and log-risk are predicted jointly.
        self.synthesis = UncertaintyConditionedUFFC(
            network_input_channels,
            output_channels=5,
            base_channels=config.model.base_channels,
            condition_channels=self.CONDITION_CHANNELS,
            bottleneck_blocks=config.model.synthesis_bottleneck_blocks,
        )
        self.router = VisibilityRouter(
            config.model, condition_channels=self.CONDITION_CHANNELS
        )

    def _condition_maps(
        self,
        render: dict[str, torch.Tensor],
        depth_scale: torch.Tensor,
    ) -> torch.Tensor:
        temperature_squared = self.config.dss.visibility_temperature_mm ** 2
        variance_normalized = render["variance"] / (
            render["variance"] + temperature_squared
        )
        depth_normalized = (render["depth"] / depth_scale).clamp(0.0, 5.0) / 5.0
        return torch.cat(
            (
                render["coverage"].clamp(0.0, 1.0),
                render["confidence"].clamp(0.0, 1.0),
                variance_normalized.clamp(0.0, 1.0),
                render["collision_entropy"].clamp(0.0, 1.0),
                depth_normalized,
                render["valid_mask"].float(),
            ),
            dim=1,
        )

    def forward(
        self,
        source_rgb: torch.Tensor,
        source_depth: torch.Tensor,
        source_depth_valid: torch.Tensor,
        source_intrinsics: torch.Tensor,
        target_intrinsics: torch.Tensor,
        source_to_target: torch.Tensor,
        return_cycle: bool = False,
    ) -> dict[str, torch.Tensor]:
        reliability = self.reliability(
            source_rgb, source_depth, source_depth_valid
        )
        source_features = torch.cat(
            (source_rgb, reliability["source_features"]), dim=1
        )
        render_valid = (
            reliability["render_valid"]
            if self.config.ablation.depth_completion
            else reliability["observed_valid"]
        )
        depth_sigma = (
            reliability["depth_sigma"]
            if self.config.ablation.distributional_depth
            else torch.full_like(
                reliability["depth_sigma"], self.config.dss.min_sigma_mm
            )
        )
        render = self.renderer(
            source_features,
            reliability["depth_mean"],
            depth_sigma,
            render_valid,
            reliability["confidence"],
            source_intrinsics,
            target_intrinsics,
            source_to_target,
            reliability["normal"]
            if self.config.ablation.anisotropic_footprint
            else None,
        )
        warped_rgb = render["features"][:, :3].clamp(0.0, 1.0)
        condition_maps = self._condition_maps(render, reliability["depth_scale"])
        network_input = torch.cat((render["features"], condition_maps), dim=1)
        transport_rgb = self.transport(network_input, warped_rgb)
        synthesis_raw = self.synthesis(network_input, condition_maps)
        synthesis_rgb = torch.sigmoid(synthesis_raw[:, :3])
        synthesis_depth = (
            F.softplus(synthesis_raw[:, 3:4]) + 0.05
        ) * reliability["depth_scale"]
        risk = torch.sigmoid(synthesis_raw[:, 4:5])
        route = self.router(
            condition_maps,
            render["support"],
            render["variance"],
            render["collision_entropy"],
            render["valid_mask"],
        )
        gate = route["synthesis_gate"]
        if not self.config.ablation.physics_router:
            gate = (~render["valid_mask"].bool()).float()
        mixed_rgb = (1.0 - gate) * transport_rgb + gate * synthesis_rgb
        mixed_depth = (1.0 - gate) * render["depth"] + gate * synthesis_depth
        raw_trusted = route["trusted_mask"]
        trusted = spatially_consistent_trusted_mask(
            raw_trusted,
            render["valid_mask"],
            self.config.model.trust_cleanup_kernel,
        )
        if self.config.ablation.hard_composition:
            target_rgb = torch.where(trusted.expand_as(mixed_rgb), warped_rgb, mixed_rgb)
            target_depth = torch.where(trusted, render["depth"], mixed_depth)
            risk = torch.where(trusted, torch.zeros_like(risk), risk)
            effective_gate = torch.where(trusted, torch.zeros_like(gate), gate)
        else:
            target_rgb = mixed_rgb
            target_depth = mixed_depth
            effective_gate = gate
        output = {
            "target_rgb": target_rgb,
            "target_depth": target_depth,
            "risk": risk,
            "hole_mask": ~trusted,
            "warped_rgb": warped_rgb,
            "transport_rgb": transport_rgb,
            "synthesis_rgb": synthesis_rgb,
            "synthesis_gate": effective_gate,
            "raw_synthesis_gate": gate,
            "synthesis_logits": route["synthesis_logits"],
            "physics_prior": route["physics_prior"],
            "trusted_mask": trusted,
            "raw_trusted_mask": raw_trusted,
            "support": render["support"],
            "coverage": render["coverage"],
            "render_confidence": render["confidence"],
            "render_depth": render["depth"],
            "completed_valid_mask": render["valid_mask"],
            "render_variance": render["variance"],
            "collision_entropy": render["collision_entropy"],
            "source_depth_mean": reliability["depth_mean"],
            "source_depth_sigma": reliability["depth_sigma"],
            "source_confidence": reliability["confidence"],
            "source_normal": reliability["normal"],
            "depth_scale": reliability["depth_scale"],
        }
        if return_cycle and self.config.ablation.rgbd_cycle:
            inverse = invert_rigid_transform(source_to_target)
            cycle_sigma = (
                self.config.dss.min_sigma_mm
                + risk.detach() * self.config.dss.max_sigma_mm
            )
            cycle = self.renderer(
                target_rgb,
                target_depth,
                cycle_sigma,
                torch.isfinite(target_depth) & (target_depth > 0),
                (1.0 - risk).clamp(0.05, 1.0),
                target_intrinsics,
                source_intrinsics,
                inverse,
            )
            output["cycle_source_rgb"] = cycle["features"]
            output["cycle_source_depth"] = cycle["depth"]
            output["cycle_source_valid"] = cycle["valid_mask"]
        return output

    @torch.no_grad()
    def renderer_only(
        self,
        source_rgb: torch.Tensor,
        source_depth: torch.Tensor,
        source_depth_valid: torch.Tensor,
        source_intrinsics: torch.Tensor,
        target_intrinsics: torch.Tensor,
        source_to_target: torch.Tensor,
        sigma_mm: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        sigma = torch.full_like(source_depth, float(sigma_mm))
        confidence = source_depth_valid.float()
        return self.renderer(
            source_rgb,
            source_depth,
            sigma,
            source_depth_valid,
            confidence,
            source_intrinsics,
            target_intrinsics,
            source_to_target,
        )
