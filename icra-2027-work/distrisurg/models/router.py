"""Physics-prior transport/synthesis mixture-of-experts routing."""

from __future__ import annotations

import torch
from torch import nn

from distrisurg.config import ModelConfig


class VisibilityRouter(nn.Module):
    """Learn a bounded residual over an interpretable synthesis prior."""

    def __init__(self, config: ModelConfig, condition_channels: int = 6) -> None:
        super().__init__()
        self.config = config
        hidden = max(16, config.base_channels // 2)
        self.learned_residual = nn.Sequential(
            nn.Conv2d(condition_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.learned_residual[-1].weight)
        nn.init.zeros_(self.learned_residual[-1].bias)

    def forward(
        self,
        condition: torch.Tensor,
        support: torch.Tensor,
        variance: torch.Tensor,
        collision_entropy: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # condition order: coverage, confidence, normalized variance,
        # collision entropy, normalized depth, valid mask.
        confidence = condition[:, 1:2].clamp(0.0, 1.0)
        variance_normalized = condition[:, 2:3].clamp(0.0, 1.0)
        collision = condition[:, 3:4].clamp(0.0, 1.0)
        physics_prior = (
            0.65 * (1.0 - confidence)
            + 0.20 * variance_normalized
            + 0.15 * collision
        ).clamp(0.0, 1.0)
        physics_prior = torch.where(
            valid_mask.bool(), physics_prior, torch.ones_like(physics_prior)
        )
        residual = self.config.router_residual_scale * torch.tanh(
            self.learned_residual(condition)
        )
        synthesis_gate = (physics_prior + residual).clamp(0.0, 1.0)
        trusted = (
            valid_mask.bool()
            & (support >= self.config.trusted_support)
            & (variance <= self.config.trusted_variance_mm2)
            & (collision_entropy <= self.config.trusted_entropy)
        )
        synthesis_gate = torch.where(
            trusted, torch.zeros_like(synthesis_gate), synthesis_gate
        )
        return {
            "synthesis_gate": synthesis_gate,
            "physics_prior": physics_prior,
            "trusted_mask": trusted,
        }


class TransportExpert(nn.Module):
    """Small residual color corrector for geometrically transported content."""

    def __init__(self, input_channels: int, base_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 5, padding=2),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, 3, 3, padding=1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self, condition: torch.Tensor, warped_rgb: torch.Tensor
    ) -> torch.Tensor:
        residual = 0.15 * torch.tanh(self.network(condition))
        return (warped_rgb + residual).clamp(0.0, 1.0)

