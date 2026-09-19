"""Mask-aware image, seam and uncertainty metrics with scene-level bootstrap."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .losses import seam_band


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mask = mask.float().expand_as(prediction)
    return ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1.0)


def masked_psnr(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> float:
    mse = masked_mse(prediction, target, mask)
    return float((-10.0 * torch.log10(mse.clamp_min(1.0e-8))).item())


def masked_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    window: int = 11,
) -> float:
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
        mask = mask.unsqueeze(0)
    padding = window // 2
    mean_x = F.avg_pool2d(prediction, window, 1, padding)
    mean_y = F.avg_pool2d(target, window, 1, padding)
    var_x = F.avg_pool2d(prediction.square(), window, 1, padding) - mean_x.square()
    var_y = F.avg_pool2d(target.square(), window, 1, padding) - mean_y.square()
    covariance = F.avg_pool2d(prediction * target, window, 1, padding) - mean_x * mean_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1)
        * (var_x + var_y + c2)
    ).clamp_min(1.0e-8)
    expanded = mask.float().expand_as(score)
    return float((score * expanded).sum().div(expanded.sum().clamp_min(1.0)).item())


def uncertainty_metrics(
    risk: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    bins: int = 10,
) -> dict[str, float]:
    error = (prediction - target).abs().mean(dim=1, keepdim=True)
    selected_risk = risk[mask].detach().float().cpu().numpy()
    selected_error = error[mask].detach().float().cpu().numpy()
    if selected_risk.size == 0:
        return {"risk_ece": float("nan"), "risk_aurc": float("nan")}
    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selected = (selected_risk >= lower) & (
            selected_risk <= upper if upper == 1.0 else selected_risk < upper
        )
        if selected.any():
            ece += selected.mean() * abs(
                float(selected_risk[selected].mean())
                - float(selected_error[selected].mean())
            )
    order = np.argsort(selected_risk)
    ordered_error = selected_error[order]
    cumulative = np.cumsum(ordered_error) / np.arange(1, ordered_error.size + 1)
    aurc = float(cumulative.mean())
    return {"risk_ece": float(ece), "risk_aurc": aurc}


def frame_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    raw_valid: torch.Tensor,
    trusted: torch.Tensor,
    warped_rgb: torch.Tensor,
    risk: torch.Tensor,
    tissue_mask: torch.Tensor | None = None,
    seam_width: int = 7,
) -> dict[str, float]:
    if tissue_mask is None:
        tissue_mask = torch.ones_like(raw_valid, dtype=torch.bool)
    tissue_mask = tissue_mask.bool()
    full = tissue_mask
    hole = (~raw_valid.bool()) & tissue_mask
    visible = raw_valid.bool() & tissue_mask
    seam = seam_band(~raw_valid.bool(), seam_width) & tissue_mask
    result = {
        "psnr": masked_psnr(prediction, target, full),
        "ssim": masked_ssim(prediction, target, full),
        "hole_psnr": masked_psnr(prediction, target, hole),
        "hole_ssim": masked_ssim(prediction, target, hole),
        "visible_psnr": masked_psnr(prediction, target, visible),
        "visible_ssim": masked_ssim(prediction, target, visible),
        "seam_psnr": masked_psnr(prediction, target, seam),
        "seam_ssim": masked_ssim(prediction, target, seam),
        "known_drift": float(
            ((prediction - warped_rgb).abs() * trusted.float()).sum()
            .div(trusted.float().sum().clamp_min(1.0) * prediction.shape[1])
            .item()
        ),
        "raw_overlap": float(raw_valid.float().mean().item()),
        "raw_hole_ratio": float((~raw_valid.bool()).float().mean().item()),
        "trusted_ratio": float(trusted.float().mean().item()),
    }
    result.update(uncertainty_metrics(risk, prediction, target, hole))
    return result


def scene_bootstrap_summary(
    rows: Iterable[dict],
    iterations: int = 2000,
    seed: int = 6666,
) -> dict[str, dict[str, float]]:
    rows = list(rows)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scene"])].append(row)
    if not grouped:
        return {}
    metric_keys = [
        key
        for key, value in rows[0].items()
        if key not in {"scene", "frame_id", "sample_id"}
        and isinstance(value, (int, float))
    ]
    scene_values: dict[str, dict[str, float]] = {}
    for scene, scene_rows in grouped.items():
        scene_values[scene] = {
            key: float(np.nanmean([float(row[key]) for row in scene_rows]))
            for key in metric_keys
        }
    names = sorted(scene_values)
    generator = np.random.default_rng(seed)
    summary: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        values = np.asarray([scene_values[name][key] for name in names], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if not finite.size:
            summary[key] = {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
            continue
        samples = generator.choice(finite, size=(iterations, finite.size), replace=True)
        boot = samples.mean(axis=1)
        summary[key] = {
            "mean": float(finite.mean()),
            "ci95_low": float(np.quantile(boot, 0.025)),
            "ci95_high": float(np.quantile(boot, 0.975)),
        }
    return summary
