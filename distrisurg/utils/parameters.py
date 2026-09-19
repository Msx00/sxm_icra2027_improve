"""Reusable model parameter accounting."""

from __future__ import annotations

from torch import nn


def parameter_count(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def distrisurg_parameter_report(model: nn.Module) -> dict[str, int]:
    """Count the complete model and its named learnable branches."""
    return {
        "Total parameters": parameter_count(model),
        "Trainable parameters": parameter_count(model, trainable_only=True),
        "DepthReliabilityEncoder": parameter_count(model.reliability),
        "TransportExpert": parameter_count(model.transport),
        "UncertaintyConditionedUFFC": parameter_count(model.synthesis),
        "VisibilityRouter": parameter_count(model.router),
    }
