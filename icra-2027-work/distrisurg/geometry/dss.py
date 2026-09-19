"""Distributional Surface Splatting (DSS).

The renderer treats source depth as a per-pixel distribution.  It projects a
small deterministic set of sigma points, derives the image-plane footprint
from the projection Jacobian, applies a soft nearest-surface visibility term,
and exposes support/variance/collision statistics to the completion network.

The integer scatter destinations are necessarily piecewise constant, while
the sub-pixel kernels, depth values, confidence, features and visibility
weights remain differentiable.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import nn

from distrisurg.config import DSSConfig


class DistributionalSurfaceSplat(nn.Module):
    def __init__(self, config: DSSConfig | dict[str, Any]) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = DSSConfig(**config)
        self.config = config
        self.register_buffer(
            "sigma_offsets",
            torch.tensor(config.sigma_samples, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sigma_weights",
            torch.tensor(config.sigma_weights, dtype=torch.float32),
            persistent=False,
        )

    def extra_repr(self) -> str:
        values = asdict(self.config)
        return ", ".join(
            f"{key}={values[key]}"
            for key in ("radius", "base_footprint_px", "visibility_temperature_mm")
        )

    @staticmethod
    def _validate(
        features: torch.Tensor,
        depth_mean: torch.Tensor,
        depth_sigma: torch.Tensor,
        valid: torch.Tensor,
        confidence: torch.Tensor,
        source_intrinsics: torch.Tensor,
        target_intrinsics: torch.Tensor,
        source_to_target: torch.Tensor,
    ) -> None:
        if features.ndim != 4:
            raise ValueError("features must have shape [B,C,H,W]")
        expected = (features.shape[0], 1, features.shape[2], features.shape[3])
        for name, value in (
            ("depth_mean", depth_mean),
            ("depth_sigma", depth_sigma),
            ("valid", valid),
            ("confidence", confidence),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
        batch = features.shape[0]
        if tuple(source_intrinsics.shape) != (batch, 3, 3):
            raise ValueError("source_intrinsics must have shape [B,3,3]")
        if tuple(target_intrinsics.shape) != (batch, 3, 3):
            raise ValueError("target_intrinsics must have shape [B,3,3]")
        if tuple(source_to_target.shape) != (batch, 4, 4):
            raise ValueError("source_to_target must have shape [B,4,4]")

    def _projection(
        self,
        depth: torch.Tensor,
        sigma: torch.Tensor,
        valid: torch.Tensor,
        confidence: torch.Tensor,
        source_k: torch.Tensor,
        target_k: torch.Tensor,
        transform: torch.Tensor,
        normal: torch.Tensor | None,
        sample_offset: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        height, width = depth.shape
        device = depth.device
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        flat_depth = depth.reshape(-1)
        flat_sigma = sigma.reshape(-1)
        flat_valid = valid.reshape(-1)
        flat_confidence = confidence.reshape(-1)
        source_index = torch.nonzero(flat_valid, as_tuple=False).squeeze(1)
        if source_index.numel() == 0:
            empty = depth.new_empty((0,), dtype=torch.float32)
            return {
                "source_index": source_index,
                "u": empty,
                "v": empty,
                "z": empty,
                "sigma_u": empty,
                "sigma_v": empty,
                "confidence": empty,
                "sample_weight": empty,
            }

        source_depth = flat_depth[source_index]
        source_sigma = flat_sigma[source_index]
        z_source = source_depth + sample_offset * source_sigma
        z_source = z_source.clamp(
            min=self.config.min_depth_mm,
            max=self.config.max_depth_mm,
        )
        u_source = xx.reshape(-1)[source_index]
        v_source = yy.reshape(-1)[source_index]
        ray_x = (u_source - source_k[0, 2]) / source_k[0, 0]
        ray_y = (v_source - source_k[1, 2]) / source_k[1, 1]
        ray = torch.stack((ray_x, ray_y, torch.ones_like(ray_x)), dim=0)
        point_source = ray * z_source.unsqueeze(0)
        rotation = transform[:3, :3]
        point_target = rotation @ point_source + transform[:3, 3:4]
        x_target, y_target, z_target = point_target.unbind(0)
        finite = torch.isfinite(point_target).all(dim=0) & (z_target > 1.0e-4)
        u_target = target_k[0, 0] * x_target / z_target + target_k[0, 2]
        v_target = target_k[1, 1] * y_target / z_target + target_k[1, 2]
        finite &= (
            (u_target > -self.config.radius - 1)
            & (u_target < width + self.config.radius)
            & (v_target > -self.config.radius - 1)
            & (v_target < height + self.config.radius)
        )
        source_index = source_index[finite]
        u_target = u_target[finite]
        v_target = v_target[finite]
        z_target = z_target[finite]
        source_sigma = source_sigma[finite]

        # Analytic image-plane Jacobian d(u,v)/d(source depth).
        ray_target = rotation @ ray[:, finite]
        dx_dz, dy_dz, dz_dz = ray_target.unbind(0)
        x_target = x_target[finite]
        y_target = y_target[finite]
        z_squared = z_target.square().clamp_min(1.0e-8)
        du_dz = target_k[0, 0] * (dx_dz * z_target - x_target * dz_dz) / z_squared
        dv_dz = target_k[1, 1] * (dy_dz * z_target - y_target * dz_dz) / z_squared

        if normal is None:
            tilt = torch.zeros_like(z_target)
        else:
            normal_z = normal[2].reshape(-1)[source_index].abs().clamp(0.0, 1.0)
            tilt = 1.0 - normal_z
        base = self.config.base_footprint_px
        normal_scale = self.config.normal_footprint_px * tilt
        sigma_u = (base + du_dz.abs() * source_sigma + normal_scale).clamp(
            min=0.35, max=self.config.max_footprint_px
        )
        sigma_v = (base + dv_dz.abs() * source_sigma + normal_scale).clamp(
            min=0.35, max=self.config.max_footprint_px
        )
        return {
            "source_index": source_index,
            "u": u_target,
            "v": v_target,
            "z": z_target,
            "sigma_u": sigma_u,
            "sigma_v": sigma_v,
            "confidence": flat_confidence[source_index].clamp(0.0, 1.0),
            "sample_weight": torch.full_like(z_target, sample_weight),
        }

    def _contributions(
        self,
        projection: dict[str, torch.Tensor],
        height: int,
        width: int,
    ):
        if projection["u"].numel() == 0:
            return
        base_u = torch.floor(projection["u"]).long()
        base_v = torch.floor(projection["v"]).long()
        radius = self.config.radius
        for offset_v in range(-radius, radius + 1):
            for offset_u in range(-radius, radius + 1):
                pixel_u = base_u + offset_u
                pixel_v = base_v + offset_v
                inside = (
                    (pixel_u >= 0)
                    & (pixel_u < width)
                    & (pixel_v >= 0)
                    & (pixel_v < height)
                )
                if not inside.any():
                    continue
                selected = torch.nonzero(inside, as_tuple=False).squeeze(1)
                du = (projection["u"][selected] - pixel_u[selected]) / projection[
                    "sigma_u"
                ][selected]
                dv = (projection["v"][selected] - pixel_v[selected]) / projection[
                    "sigma_v"
                ][selected]
                spatial = torch.exp(-0.5 * (du.square() + dv.square()))
                keep = spatial > 1.0e-5
                if not keep.any():
                    continue
                selected = selected[keep]
                yield {
                    "target_index": (
                        pixel_v[selected] * width + pixel_u[selected]
                    ).long(),
                    "source_index": projection["source_index"][selected],
                    "spatial": spatial[keep],
                    "z": projection["z"][selected],
                    "confidence": projection["confidence"][selected],
                    "sample_weight": projection["sample_weight"][selected],
                }

    def _render_one(
        self,
        features: torch.Tensor,
        depth_mean: torch.Tensor,
        depth_sigma: torch.Tensor,
        valid: torch.Tensor,
        confidence: torch.Tensor,
        source_k: torch.Tensor,
        target_k: torch.Tensor,
        transform: torch.Tensor,
        normal: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        channels, height, width = features.shape
        pixel_count = height * width
        depth_sigma = depth_sigma.clamp(
            self.config.min_sigma_mm, self.config.max_sigma_mm
        )
        valid = (
            valid.bool()
            & torch.isfinite(depth_mean)
            & (depth_mean >= self.config.min_depth_mm)
            & (depth_mean <= self.config.max_depth_mm)
        )
        projections = [
            self._projection(
                depth_mean[0],
                depth_sigma[0],
                valid[0],
                confidence[0],
                source_k,
                target_k,
                transform,
                normal,
                offset,
                weight,
            )
            for offset, weight in zip(self.sigma_offsets, self.sigma_weights)
        ]

        nearest = torch.full(
            (pixel_count,),
            float("inf"),
            device=features.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            for projection in projections:
                for contribution in self._contributions(projection, height, width):
                    nearest.scatter_reduce_(
                        0,
                        contribution["target_index"],
                        contribution["z"].detach(),
                        reduce="amin",
                        include_self=True,
                    )

        support = torch.zeros_like(nearest)
        depth_sum = torch.zeros_like(nearest)
        depth_square_sum = torch.zeros_like(nearest)
        weighted_log_sum = torch.zeros_like(nearest)
        contribution_count = torch.zeros_like(nearest)
        feature_sum = torch.zeros(
            (channels, pixel_count),
            device=features.device,
            dtype=torch.float32,
        )
        flat_features = features.float().reshape(channels, -1)
        temperature = self.config.visibility_temperature_mm
        for projection in projections:
            for contribution in self._contributions(projection, height, width):
                target_index = contribution["target_index"]
                delta = (contribution["z"] - nearest[target_index]).clamp_min(0.0)
                visibility = torch.exp(-delta / temperature)
                weight = (
                    contribution["spatial"]
                    * contribution["confidence"]
                    * contribution["sample_weight"]
                    * visibility
                ).float()
                nonzero = weight > 1.0e-8
                if not nonzero.any():
                    continue
                target_index = target_index[nonzero]
                source_index = contribution["source_index"][nonzero]
                z = contribution["z"][nonzero].float()
                weight = weight[nonzero]
                support.scatter_add_(0, target_index, weight)
                depth_sum.scatter_add_(0, target_index, weight * z)
                depth_square_sum.scatter_add_(0, target_index, weight * z.square())
                weighted_log_sum.scatter_add_(
                    0, target_index, weight * torch.log(weight.clamp_min(1.0e-12))
                )
                contribution_count.scatter_add_(
                    0, target_index, torch.ones_like(weight)
                )
                feature_sum.scatter_add_(
                    1,
                    target_index.unsqueeze(0).expand(channels, -1),
                    flat_features[:, source_index] * weight.unsqueeze(0),
                )

        safe_support = support.clamp_min(1.0e-8)
        expected_depth = depth_sum / safe_support
        variance = (depth_square_sum / safe_support - expected_depth.square()).clamp_min(0.0)
        entropy = torch.log(safe_support) - weighted_log_sum / safe_support
        entropy = entropy.clamp_min(0.0)
        normalized_entropy = torch.where(
            contribution_count > 1,
            entropy / torch.log(contribution_count.clamp_min(2.0)),
            torch.zeros_like(entropy),
        ).clamp(0.0, 1.0)
        # Entropy alone also reacts to normal interpolation.  Multiplying by
        # depth dispersion turns it into a collision-of-surfaces statistic.
        collision_entropy = normalized_entropy * (
            1.0 - torch.exp(-variance / (temperature * temperature))
        )
        coverage = 1.0 - torch.exp(-support)
        confidence_target = coverage * torch.exp(
            -variance / (4.0 * temperature * temperature)
        ) * torch.exp(-collision_entropy)
        valid_target = support >= self.config.min_support
        valid_float = valid_target.float()
        rendered = (
            feature_sum / safe_support.unsqueeze(0)
        ) * valid_float.unsqueeze(0)
        expected_depth = expected_depth * valid_float
        variance = variance * valid_float
        collision_entropy = collision_entropy * valid_float
        confidence_target = confidence_target * valid_float
        return {
            "features": rendered.reshape(channels, height, width),
            "depth": expected_depth.reshape(1, height, width),
            "support": support.reshape(1, height, width),
            "coverage": coverage.reshape(1, height, width),
            "variance": variance.reshape(1, height, width),
            "collision_entropy": collision_entropy.reshape(1, height, width),
            "confidence": confidence_target.reshape(1, height, width),
            "valid_mask": valid_target.reshape(1, height, width),
        }

    def forward(
        self,
        features: torch.Tensor,
        depth_mean: torch.Tensor,
        depth_sigma: torch.Tensor,
        valid: torch.Tensor,
        confidence: torch.Tensor,
        source_intrinsics: torch.Tensor,
        target_intrinsics: torch.Tensor,
        source_to_target: torch.Tensor,
        normal: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate(
            features,
            depth_mean,
            depth_sigma,
            valid,
            confidence,
            source_intrinsics,
            target_intrinsics,
            source_to_target,
        )
        if normal is not None and tuple(normal.shape) != (
            features.shape[0],
            3,
            features.shape[2],
            features.shape[3],
        ):
            raise ValueError("normal must have shape [B,3,H,W]")
        dtype = features.dtype
        outputs = [
            self._render_one(
                features[index],
                depth_mean[index].float(),
                depth_sigma[index].float(),
                valid[index],
                confidence[index].float(),
                source_intrinsics[index].float(),
                target_intrinsics[index].float(),
                source_to_target[index].float(),
                None if normal is None else normal[index].float(),
            )
            for index in range(features.shape[0])
        ]
        keys = outputs[0].keys()
        result = {key: torch.stack([value[key] for value in outputs]) for key in keys}
        result["features"] = result["features"].to(dtype=dtype)
        return result
