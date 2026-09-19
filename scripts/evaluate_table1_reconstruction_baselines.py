#!/usr/bin/env python3
"""Evaluate Table-I reconstruction baselines on the fixed 1,392-frame split.

Every rendered frame is retained, including degenerate predictions, so every
method uses the same test denominator and no output is filtered post hoc.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageStat

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRIMARY_RUN = PROJECT_ROOT / "results" / "validation" / "distrisurg"
COMPARISON_ROOT = Path("/home/data/mashixing/dataset_8tb/iMed/comparison/task2")
METHOD_LAYOUTS = {
    "PR-ENDO": ("SurgicalGaussian/PR-ENDO/output_imed_nvs", "test/ours_40000/renders"),
    "EndoGS": ("EndoGS/output/imed_nvs", "point_cloud/iteration_60000/render"),
    "EndoGaussian": ("EndoGaussian/output/imed_nvs", "test/ours_30000/renders"),
}


def masked_psnr(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = mask.float().expand_as(prediction)
    mse = ((prediction - target).square() * expanded).sum() / expanded.sum().clamp_min(1.0)
    return float((-10.0 * torch.log10(mse.clamp_min(1.0e-8))).item())


def masked_ssim(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, window: int = 11) -> float:
    padding = window // 2
    mean_x = F.avg_pool2d(prediction, window, 1, padding)
    mean_y = F.avg_pool2d(target, window, 1, padding)
    var_x = F.avg_pool2d(prediction.square(), window, 1, padding) - mean_x.square()
    var_y = F.avg_pool2d(target.square(), window, 1, padding) - mean_y.square()
    covariance = F.avg_pool2d(prediction * target, window, 1, padding) - mean_x * mean_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2)
    ).clamp_min(1.0e-8)
    expanded = mask.float().expand_as(score)
    return float((score * expanded).sum().div(expanded.sum().clamp_min(1.0)).item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-run", type=Path, default=PRIMARY_RUN)
    parser.add_argument("--comparison-root", type=Path, default=COMPARISON_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/validation/table1_equal_n")
    parser.add_argument("--methods", nargs="+", choices=tuple(METHOD_LAYOUTS), default=list(METHOD_LAYOUTS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def image_paths(directory: Path) -> list[Path]:
    paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}) if directory.is_dir() else []
    if not paths:
        raise FileNotFoundError(f"no images found in {directory}")
    return paths


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_mask(path: Path, size: tuple[int, int], *, invert: bool = False, missing_value: bool | None = None) -> torch.Tensor:
    if not path.is_file():
        if missing_value is None:
            raise FileNotFoundError(path)
        return torch.full((1, size[1], size[0]), missing_value, dtype=torch.bool)
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array < 128 if invert else array >= 128).unsqueeze(0)


def is_degenerate(path: Path) -> bool:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        extrema, stat = rgb.getextrema(), ImageStat.Stat(rgb)
    dynamic_range = max(high for _, high in extrema) - min(low for low, _ in extrema)
    return dynamic_range <= 1 or max(stat.stddev) <= 0.5


def sample_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "sample_sd": float(array.std(ddof=1))}


def load_protocol(primary_run: Path) -> tuple[Path, dict[str, list[str]]]:
    manifest_path = primary_run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("completed"):
        raise RuntimeError(f"incomplete primary validation run: {manifest_path}")
    by_scene: dict[str, list[str]] = defaultdict(list)
    for record in manifest["selected_frames"]:
        scene, stem = str(record["sample_id"]).split("/", 1)
        by_scene[scene].append(stem)
    count = sum(map(len, by_scene.values()))
    if count != 1392 or len(by_scene) != 7:
        raise ValueError(f"expected seven scenes and 1,392 frames; got {len(by_scene)} and {count}")
    return Path(manifest["data_root"]), dict(sorted(by_scene.items()))


def evaluate_method(method: str, comparison_root: Path, primary_run: Path, data_root: Path, by_scene: dict[str, list[str]], device: torch.device, batch_size: int, scalar_lpips: torch.nn.Module, spatial_lpips: torch.nn.Module) -> tuple[dict, list[dict]]:
    root_suffix, render_suffix = METHOD_LAYOUTS[method]
    records: list[dict] = []
    for scene, stems in by_scene.items():
        render_dir = comparison_root / root_suffix / scene / render_suffix
        renders = image_paths(render_dir)
        if len(renders) != len(stems):
            raise ValueError(f"{method}/{scene}: expected {len(stems)} renders, got {len(renders)} in {render_dir}")
        for render, stem in zip(renders, stems):
            target = primary_run / scene / "targets" / f"{stem}.png"
            hole = primary_run / scene / "hole_mask" / f"{stem}.png"
            tool = data_root / scene / "endoscope1/toolL" / f"{stem}.png"
            if not target.is_file() or not hole.is_file():
                raise FileNotFoundError(f"missing target or hole mask for {scene}/{stem}")
            records.append({"method": method, "scene": scene, "frame_id": stem, "sample_id": f"{scene}/{stem}", "render_path": str(render), "target_path": str(target), "hole_path": str(hole), "tool_path": str(tool), "degenerate": is_degenerate(render)})

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        predictions, targets, tissue_masks, hole_masks = [], [], [], []
        for record in batch:
            target = load_rgb(Path(record["target_path"]))
            height, width = target.shape[-2:]
            prediction = load_rgb(Path(record["render_path"]), (width, height))
            tissue = load_mask(Path(record["tool_path"]), (width, height), invert=True, missing_value=True)
            hole = load_mask(Path(record["hole_path"]), (width, height)) & tissue
            predictions.append(prediction); targets.append(target)
            tissue_masks.append(tissue); hole_masks.append(hole)

        prediction_batch = torch.stack(predictions).to(device, non_blocking=True)
        target_batch = torch.stack(targets).to(device, non_blocking=True)
        tissue_batch = torch.stack(tissue_masks).to(device, non_blocking=True)
        hole_batch = torch.stack(hole_masks).to(device, non_blocking=True)
        with torch.inference_mode():
            full_lpips = scalar_lpips(prediction_batch * tissue_batch.float(), target_batch * tissue_batch.float(), normalize=True).reshape(-1)
            spatial = spatial_lpips(prediction_batch, target_batch, normalize=True)
            hole_lpips = (spatial * hole_batch.float()).flatten(1).sum(1).div(hole_batch.flatten(1).sum(1).clamp_min(1).float())
        for index, record in enumerate(batch):
            prediction, target = prediction_batch[index:index + 1], target_batch[index:index + 1]
            tissue, hole = tissue_batch[index:index + 1], hole_batch[index:index + 1]
            record.update({"psnr": masked_psnr(prediction, target, tissue), "ssim": masked_ssim(prediction, target, tissue), "lpips": float(full_lpips[index]), "hole_psnr": masked_psnr(prediction, target, hole), "hole_ssim": masked_ssim(prediction, target, hole), "hole_lpips": float(hole_lpips[index])})
        print(f"{method}: {min(start + batch_size, len(records))}/{len(records)}", flush=True)

    metric_names = ("psnr", "ssim", "lpips", "hole_psnr", "hole_ssim", "hole_lpips")
    summary = {"method": method, "count": len(records), "scene_count": len(by_scene), "degenerate_count": sum(bool(r["degenerate"]) for r in records), "metrics": {name: sample_summary([float(r[name]) for r in records]) for name in metric_names}}
    return summary, records


def main() -> None:
    args = parse_args()
    primary_run, comparison_root, output_dir = args.primary_run.resolve(), args.comparison_root.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root, by_scene = load_protocol(primary_run)
    import lpips  # type: ignore
    device = torch.device(args.device)
    scalar_lpips = lpips.LPIPS(net="alex").eval().to(device)
    spatial_lpips = lpips.LPIPS(net="alex", spatial=True).eval().to(device)
    summaries, rows = {}, []
    for method in args.methods:
        summary, method_rows = evaluate_method(method, comparison_root, primary_run, data_root, by_scene, device, args.batch_size, scalar_lpips, spatial_lpips)
        summaries[method] = summary; rows.extend(method_rows)
    protocol = {"name": "Table I equal-N reconstruction-baseline evaluation", "primary_run": str(primary_run), "data_root": str(data_root), "scenes": {s: len(v) for s, v in by_scene.items()}, "count_per_method": sum(map(len, by_scene.values())), "aggregation": "frame-wise mean and sample standard deviation", "inclusion": "all frames retained, including degenerate predictions", "full_mask": "non-tool tissue mask", "hole_mask": "non-tool tissue intersected with saved raw-DSS hole mask", "lpips": "AlexNet LPIPS; [0,1] inputs normalized internally to [-1,1]; spatial map averaged for H-LPIPS"}
    payload = {"protocol": protocol, "methods": summaries}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = ["method", "scene", "frame_id", "sample_id", "degenerate", "psnr", "ssim", "lpips", "hole_psnr", "hole_ssim", "hole_lpips", "render_path", "target_path", "hole_path", "tool_path"]
    with (output_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
