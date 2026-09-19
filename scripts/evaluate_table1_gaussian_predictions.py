#!/usr/bin/env python3
"""Build the Table-I Gaussian mapping and recompute canonical PSNR/SSIM.

The metric definitions intentionally mirror
``icra-2027/scripts/evaluate_table1_reconstruction_baselines.py``:

* full support: non-tool tissue;
* H support: non-tool tissue intersected with the primary-run hole mask;
* SSIM: 11x11 ``avg_pool2d`` with zero padding;
* aggregation: frame-wise mean and sample standard deviation (ddof=1).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageStat


DEFAULT_PRIMARY = Path(
    "/home/data20tb/shixingma/icra-2027/results/validation/distrisurg"
)
DEFAULT_TASK2 = Path("/home/data/mashixing/dataset_8tb/iMed/comparison/task2")
METHODS = (
    "Deform3DGS",
    "Endo-4DGS",
    "Free-SurGS",
    "SurgicalGS",
    "StereoSurGS (FS)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-run", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--task2-root", type=Path, default=DEFAULT_TASK2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/data/mashixing/table1_gaussian_canonical"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=32)
    parser.add_argument("--mapping-only", action="store_true")
    return parser.parse_args()


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_mask(
    path: Path,
    size: tuple[int, int],
    *,
    invert: bool = False,
    missing_value: bool | None = None,
) -> torch.Tensor:
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


def load_protocol(primary_run: Path) -> tuple[Path, list[tuple[str, str]], dict[str, list[str]]]:
    manifest_path = primary_run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("completed"):
        raise RuntimeError(f"incomplete primary validation run: {manifest_path}")
    samples: list[tuple[str, str]] = []
    by_scene: dict[str, list[str]] = defaultdict(list)
    for record in manifest["selected_frames"]:
        scene, stem = str(record["sample_id"]).split("/", 1)
        samples.append((scene, stem))
        by_scene[scene].append(stem)
    if len(samples) != 1392 or len(by_scene) != 7:
        raise ValueError(f"expected 7 scenes / 1392 frames, got {len(by_scene)} / {len(samples)}")
    return Path(manifest["data_root"]), samples, dict(by_scene)


def build_mapping(
    primary_run: Path,
    task2_root: Path,
    samples: list[tuple[str, str]],
    by_scene: dict[str, list[str]],
) -> list[dict[str, str]]:
    qualified = task2_root / "test_runs/qualified_methods/tasks"
    method_paths: dict[str, dict[tuple[str, str], Path]] = {}
    expected = set(samples)

    for method in METHODS[:-1]:
        paths: dict[tuple[str, str], Path] = {}
        for scene, stems in by_scene.items():
            report_path = qualified / method / scene / "metrics.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            views = report.get("per_view", [])
            if len(views) != len(stems):
                raise ValueError(f"{method}/{scene}: {len(views)} views != {len(stems)} protocol frames")
            for stem, view in zip(stems, views):
                witness = view.get("mask")
                if witness and Path(witness).stem != stem:
                    raise ValueError(f"ordinal/frame mismatch: {method}/{scene}/{stem}: {witness}")
                paths[(scene, stem)] = Path(view["render"]).resolve()
        if set(paths) != expected:
            raise ValueError(f"{method}: sample-set mismatch")
        method_paths[method] = paths

    stereo_root = task2_root / "task2_sdu/output/task2_sdu_gs200000"
    stereo_paths: dict[tuple[str, str], Path] = {}
    for scene in by_scene:
        manifest_path = stereo_root / scene / "prediction_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("pipeline") != "per_scene_foundation_stereo_surgicalgs":
            raise ValueError(f"unexpected StereoSurGS pipeline: {manifest_path}")
        for record in manifest["renders"]:
            stem = f"frame_{int(record['frame_id']):06d}"
            stereo_paths[(scene, stem)] = Path(record["output"]).resolve()
    if set(stereo_paths) != expected:
        raise ValueError("StereoSurGS (FS): sample-set mismatch")
    method_paths[METHODS[-1]] = stereo_paths

    data_root, _, _ = load_protocol(primary_run)
    rows: list[dict[str, str]] = []
    for method in METHODS:
        for scene, stem in samples:
            render = method_paths[method][(scene, stem)]
            target = (primary_run / scene / "targets" / f"{stem}.png").resolve()
            hole = (primary_run / scene / "hole_mask" / f"{stem}.png").resolve()
            tool = (data_root / scene / "endoscope1/toolL" / f"{stem}.png").resolve()
            for required in (render, target, hole):
                if not required.is_file():
                    raise FileNotFoundError(required)
            rows.append(
                {
                    "method": method,
                    "sample_id": f"{scene}/{stem}",
                    "render_path": str(render),
                    "target_path": str(target),
                    "hole_path": str(hole),
                    "tool_path": str(tool),
                }
            )
    return rows


def write_mapping(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "sample_id", "render_path", "target_path", "hole_path", "tool_path"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_vectors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    full_mask: torch.Tensor,
    hole_mask: torch.Tensor,
    window: int = 11,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized per-frame form of the paper evaluator's exact equations."""
    reduce = (1, 2, 3)
    squared_error = (prediction - target).square()

    def psnr_for(mask: torch.Tensor) -> torch.Tensor:
        expanded = mask.float().expand_as(prediction)
        mse = (squared_error * expanded).sum(reduce) / expanded.sum(reduce).clamp_min(1.0)
        return -10.0 * torch.log10(mse.clamp_min(1.0e-8))

    psnr = psnr_for(full_mask)
    hole_psnr = psnr_for(hole_mask)

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
    def ssim_for(mask: torch.Tensor) -> torch.Tensor:
        expanded = mask.float().expand_as(score)
        return (score * expanded).sum(reduce) / expanded.sum(reduce).clamp_min(1.0)

    ssim = ssim_for(full_mask)
    hole_ssim = ssim_for(hole_mask)
    return psnr, ssim, hole_psnr, hole_ssim


def load_sample(job: tuple[str, dict[str, dict[str, str]]]) -> dict:
    sample_id, methods = job
    scene, stem = sample_id.split("/", 1)
    first = methods[METHODS[0]]
    target = load_rgb(Path(first["target_path"]))
    height, width = target.shape[-2:]
    size = (width, height)
    tissue = load_mask(Path(first["tool_path"]), size, invert=True, missing_value=True)
    hole = load_mask(Path(first["hole_path"]), size) & tissue
    predictions, degenerates = [], []
    for method in METHODS:
        record = methods[method]
        render = Path(record["render_path"])
        predictions.append(load_rgb(render, size))
        degenerates.append(is_degenerate(render))
    return {
        "sample_id": sample_id,
        "scene": scene,
        "frame_id": stem,
        "methods": methods,
        "prediction": torch.stack(predictions),
        "target": target,
        "tissue": tissue,
        "hole": hole,
        "degenerate": degenerates,
    }


def evaluate(
    mapping: list[dict[str, str]],
    samples: list[tuple[str, str]],
    batch_size: int,
    workers: int,
) -> list[dict]:
    indexed: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in mapping:
        indexed[row["sample_id"]][row["method"]] = row
    jobs = [(f"{scene}/{stem}", indexed[f"{scene}/{stem}"]) for scene, stem in samples]
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        loaded = pool.map(load_sample, jobs)
        batch: list[dict] = []
        for sample in loaded:
            batch.append(sample)
            if len(batch) < batch_size:
                continue
            rows.extend(evaluate_batch(batch))
            print(f"evaluated {len(rows) // len(METHODS)}/{len(samples)} frames", flush=True)
            batch.clear()
        if batch:
            rows.extend(evaluate_batch(batch))
            print(f"evaluated {len(rows) // len(METHODS)}/{len(samples)} frames", flush=True)
    return rows


def evaluate_batch(batch: list[dict]) -> list[dict]:
    method_count = len(METHODS)
    prediction = torch.cat([sample["prediction"] for sample in batch], dim=0)
    target = torch.stack([sample["target"] for sample in batch]).repeat_interleave(method_count, 0)
    tissue = torch.stack([sample["tissue"] for sample in batch]).repeat_interleave(method_count, 0)
    hole = torch.stack([sample["hole"] for sample in batch]).repeat_interleave(method_count, 0)
    with torch.inference_mode():
        psnr, ssim, hole_psnr, hole_ssim = metric_vectors(
            prediction, target, tissue, hole
        )
    result: list[dict] = []
    offset = 0
    for sample in batch:
        for method_index, method in enumerate(METHODS):
            source = sample["methods"][method]
            index = offset + method_index
            result.append(
                {
                    "method": method,
                    "scene": sample["scene"],
                    "frame_id": sample["frame_id"],
                    "sample_id": sample["sample_id"],
                    "degenerate": sample["degenerate"][method_index],
                    "psnr": float(psnr[index]),
                    "ssim": float(ssim[index]),
                    "hole_psnr": float(hole_psnr[index]),
                    "hole_ssim": float(hole_ssim[index]),
                    "render_path": source["render_path"],
                    "target_path": source["target_path"],
                    "hole_path": source["hole_path"],
                    "tool_path": source["tool_path"],
                }
            )
        offset += method_count
    return result


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    primary_run, task2_root, output_dir = (
        args.primary_run.resolve(),
        args.task2_root.resolve(),
        args.output_dir.resolve(),
    )
    data_root, samples, by_scene = load_protocol(primary_run)
    mapping = build_mapping(primary_run, task2_root, samples, by_scene)
    mapping_path = output_dir / "prediction_mapping.csv"
    write_mapping(mapping_path, mapping)
    print(f"mapping: {mapping_path} ({len(mapping)} rows)", flush=True)
    if args.mapping_only:
        return

    rows = evaluate(mapping, samples, args.batch_size, args.workers)
    metric_names = ("psnr", "ssim", "hole_psnr", "hole_ssim")
    summaries = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        summaries[method] = {
            "method": method,
            "count": len(selected),
            "scene_count": len({row["scene"] for row in selected}),
            "degenerate_count": sum(bool(row["degenerate"]) for row in selected),
            "metrics": {
                name: sample_summary([float(row[name]) for row in selected])
                for name in metric_names
            },
        }
    payload = {
        "protocol": {
            "name": "Table I canonical Gaussian PSNR/SSIM evaluation",
            "primary_run": str(primary_run),
            "data_root": str(data_root),
            "scenes": {scene: len(stems) for scene, stems in by_scene.items()},
            "count_per_method": len(samples),
            "aggregation": "frame-wise mean and sample standard deviation",
            "inclusion": "all frames retained, including degenerate predictions",
            "full_mask": "non-tool tissue mask",
            "hole_mask": "non-tool tissue intersected with saved raw-DSS hole mask",
            "ssim": "11x11 torch.nn.functional.avg_pool2d with padding=5",
        },
        "methods": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = [
        "method", "scene", "frame_id", "sample_id", "degenerate",
        "psnr", "ssim", "hole_psnr", "hole_ssim",
        "render_path", "target_path", "hole_path", "tool_path",
    ]
    with (output_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
