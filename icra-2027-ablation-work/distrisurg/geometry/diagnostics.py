"""Diagnostic taxonomy for projection defects.

The labels are deliberately named diagnostics rather than ground truth: with a
single source view, true disocclusion cannot be separated perfectly from a
catastrophic depth-completion failure.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


LABELS = {
    0: "reliable_or_unclassified",
    1: "local_sampling_gap",
    2: "depth_completion_recoverable",
    3: "unsupported_or_disoccluded",
    4: "surface_collision_risk",
}


def projection_defect_taxonomy(
    raw_valid: torch.Tensor,
    completed_valid: torch.Tensor,
    variance: torch.Tensor,
    collision_entropy: torch.Tensor,
    variance_threshold: float,
    entropy_threshold: float,
) -> torch.Tensor:
    if raw_valid.shape != completed_valid.shape:
        raise ValueError("raw and completed masks must have equal shapes")
    raw_valid = raw_valid.bool()
    completed_valid = completed_valid.bool()
    raw_hole = ~raw_valid
    local_support = F.max_pool2d(
        raw_valid.float(), kernel_size=3, stride=1, padding=1
    ) > 0
    sampling = raw_hole & local_support
    depth_recoverable = raw_hole & completed_valid & ~sampling
    unsupported = raw_hole & ~completed_valid & ~sampling
    collision = raw_valid & (
        (variance > variance_threshold)
        | (collision_entropy > entropy_threshold)
    )
    labels = torch.zeros_like(raw_valid, dtype=torch.uint8)
    labels = torch.where(sampling, torch.ones_like(labels), labels)
    labels = torch.where(depth_recoverable, torch.full_like(labels, 2), labels)
    labels = torch.where(unsupported, torch.full_like(labels, 3), labels)
    labels = torch.where(collision, torch.full_like(labels, 4), labels)
    return labels


def taxonomy_ratios(labels: torch.Tensor) -> dict[str, float]:
    total = labels.numel()
    return {
        f"diagnostic_{name}_ratio": float((labels == identifier).sum().item() / total)
        for identifier, name in LABELS.items()
    }
