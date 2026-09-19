"""Differentiable geometry modules."""

from .dss import DistributionalSurfaceSplat
from .diagnostics import LABELS, projection_defect_taxonomy, taxonomy_ratios
from .reprojection import invert_rigid_transform

__all__ = [
    "DistributionalSurfaceSplat",
    "invert_rigid_transform",
    "projection_defect_taxonomy",
    "taxonomy_ratios",
    "LABELS",
]
