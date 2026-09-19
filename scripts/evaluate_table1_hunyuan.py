#!/usr/bin/env python3
"""Evaluate Hunyuan completion predictions with the fixed Table-I protocol.

The evaluator deliberately reuses the metric helpers from
``evaluate_table1_reconstruction_baselines.py``.  Quality metrics are computed
against the primary validation run's targets, non-tool tissue masks, and saved
raw-DSS hole masks.  The generic-completion warp and inpaint mask are used only
to audit known-region preservation.

All 1,392 predictions must be present exactly once.  Missing, duplicated, or
unexpected samples are fatal rather than being silently filtered.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from evaluate_table1_reconstruction_baselines import (
    is_degenerate,
    load_mask,
    load_protocol,
    load_rgb,
    masked_psnr,
    masked_ssim,
    sample_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_RUN = PROJECT_ROOT / "results" / "validation" / "distrisurg"
PREDICTION_ROOT = PROJECT_ROOT / "results" / "validation" / "hunyuan_completion" / "parts"
INPUT_LIST = (
    PROJECT_ROOT
    / "results"
    / "validation"
    / "generic_inpainting"
    / "comparison_methods"
    / "zeroshot_inputs.jsonl"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "validation" / "hunyuan_completion"
EXPECTED_SCENES = 7
EXPECTED_SAMPLES = 1392
METRIC_NAMES = ("psnr", "ssim", "lpips", "hole_psnr", "hole_ssim", "hole_lpips")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-run", type=Path, default=PRIMARY_RUN)
    parser.add_argument("--prediction-root", type=Path, default=PREDICTION_ROOT)
    parser.add_argument(
        "--prediction-pattern",
        default="gpu*/inference/*/*_completed.png",
        help="Glob relative to --prediction-root; parent directory must be the scene name.",
    )
    parser.add_argument("--input-list", type=Path, default=INPUT_LIST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--method-name", default="Hunyuan-DiT (in-domain adapted)")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def sample_id(scene: str, stem: str) -> str:
    return f"{scene}/{stem}"


def expected_sample_ids(by_scene: dict[str, list[str]]) -> set[str]:
    return {sample_id(scene, stem) for scene, stems in by_scene.items() for stem in stems}


def describe_set_mismatch(label: str, actual: set[str], expected: set[str]) -> str:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    return (
        f"{label} sample set does not match the Table-I protocol: "
        f"expected={len(expected)}, actual={len(actual)}, "
        f"missing={len(missing)} {missing[:8]}, "
        f"unexpected={len(unexpected)} {unexpected[:8]}"
    )


def load_completion_inputs(path: Path, expected: set[str]) -> dict[str, dict[str, Path]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: dict[str, dict[str, Path]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            scene = str(payload.get("scene", "")).strip()
            stem = str(payload.get("name", "")).strip()
            if not stem and payload.get("warped_rgb"):
                stem = Path(str(payload["warped_rgb"])).stem
            if not scene or not stem:
                raise ValueError(f"{path}:{line_number}: missing scene or name")
            identifier = sample_id(scene, stem)
            if identifier in records:
                raise ValueError(f"{path}:{line_number}: duplicate sample {identifier}")
            try:
                warped_rgb = Path(payload["warped_rgb"]).expanduser().resolve()
                inpaint_mask = Path(payload["inpaint_mask"]).expanduser().resolve()
            except KeyError as error:
                raise ValueError(f"{path}:{line_number}: missing {error.args[0]}") from error
            if not warped_rgb.is_file() or not inpaint_mask.is_file():
                raise FileNotFoundError(
                    f"{path}:{line_number}: missing warp or inpaint mask for {identifier}: "
                    f"{warped_rgb}, {inpaint_mask}"
                )
            records[identifier] = {
                "warped_rgb": warped_rgb,
                "inpaint_mask": inpaint_mask,
            }
    actual = set(records)
    if actual != expected:
        raise ValueError(describe_set_mismatch("completion-input", actual, expected))
    return records


def discover_predictions(root: Path, pattern: str, expected: set[str]) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    predictions: dict[str, Path] = {}
    paths = sorted(path.resolve() for path in root.glob(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"no predictions matched {root / pattern}")
    suffix = "_completed"
    for path in paths:
        stem = path.stem
        if not stem.endswith(suffix):
            raise ValueError(f"prediction does not end with {suffix!r}: {path}")
        identifier = sample_id(path.parent.name, stem[: -len(suffix)])
        if identifier in predictions:
            raise ValueError(f"duplicate prediction for {identifier}: {predictions[identifier]} and {path}")
        predictions[identifier] = path
    actual = set(predictions)
    if actual != expected:
        raise ValueError(describe_set_mismatch("prediction", actual, expected))
    return predictions


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def known_region_diagnostics(
    prediction: torch.Tensor,
    warped_rgb: torch.Tensor,
    inpaint_mask: torch.Tensor,
) -> dict[str, int | float]:
    """Return exact 8-bit preservation diagnostics outside the inpaint mask."""
    known = ~inpaint_mask.bool()
    known_pixel_count = int(known.sum().item())
    if known_pixel_count == 0:
        raise ValueError("inpaint mask leaves no known-region pixels")
    known_channels = known.expand_as(prediction)
    difference_8bit = (prediction - warped_rgb).abs().mul(255.0)
    selected = difference_8bit.masked_select(known_channels)
    changed_pixels = ((difference_8bit.amax(dim=0, keepdim=True) > 0.5) & known).sum()
    return {
        "known_pixel_count": known_pixel_count,
        "known_changed_pixel_count": int(changed_pixels.item()),
        "known_changed_pixel_fraction": float(changed_pixels.item() / known_pixel_count),
        "known_mae_8bit": float(selected.mean().item()),
        "known_max_abs_8bit": float(selected.max().item()),
    }


def build_records(
    primary_run: Path,
    data_root: Path,
    by_scene: dict[str, list[str]],
    predictions: dict[str, Path],
    completion_inputs: dict[str, dict[str, Path]],
    method_name: str,
) -> list[dict]:
    records: list[dict] = []
    for scene, stems in by_scene.items():
        for stem in stems:
            identifier = sample_id(scene, stem)
            prediction = predictions[identifier]
            target = primary_run / scene / "targets" / f"{stem}.png"
            hole = primary_run / scene / "hole_mask" / f"{stem}.png"
            tool = data_root / scene / "endoscope1" / "toolL" / f"{stem}.png"
            warped_rgb = completion_inputs[identifier]["warped_rgb"]
            inpaint_mask = completion_inputs[identifier]["inpaint_mask"]
            required = {
                "prediction": prediction,
                "target": target,
                "hole": hole,
                "warped_rgb": warped_rgb,
                "inpaint_mask": inpaint_mask,
            }
            absent = {name: str(path) for name, path in required.items() if not path.is_file()}
            if absent:
                raise FileNotFoundError(f"missing files for {identifier}: {absent}")
            target_size = image_size(target)
            mismatched = {
                name: image_size(path)
                for name, path in required.items()
                if image_size(path) != target_size
            }
            if mismatched:
                raise ValueError(
                    f"image-size mismatch for {identifier}; target={target_size}, mismatched={mismatched}"
                )
            records.append(
                {
                    "method": method_name,
                    "scene": scene,
                    "frame_id": stem,
                    "sample_id": identifier,
                    "degenerate": is_degenerate(prediction),
                    "prediction_path": str(prediction),
                    "target_path": str(target),
                    "hole_path": str(hole),
                    "tool_path": str(tool),
                    "warped_rgb_path": str(warped_rgb),
                    "inpaint_mask_path": str(inpaint_mask),
                }
            )
    if len(records) != EXPECTED_SAMPLES:
        raise ValueError(f"expected {EXPECTED_SAMPLES} records, built {len(records)}")
    return records


def evaluate(
    records: list[dict],
    device: torch.device,
    batch_size: int,
    scalar_lpips: torch.nn.Module,
    spatial_lpips: torch.nn.Module,
) -> None:
    if batch_size < 1:
        raise ValueError(f"batch size must be positive, got {batch_size}")
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        tissue_masks: list[torch.Tensor] = []
        hole_masks: list[torch.Tensor] = []
        for record in batch:
            target = load_rgb(Path(record["target_path"]))
            height, width = target.shape[-2:]
            size = (width, height)
            prediction = load_rgb(Path(record["prediction_path"]), size)
            tissue = load_mask(Path(record["tool_path"]), size, invert=True, missing_value=True)
            hole = load_mask(Path(record["hole_path"]), size) & tissue
            warped_rgb = load_rgb(Path(record["warped_rgb_path"]), size)
            input_hole = load_mask(Path(record["inpaint_mask_path"]), size)
            record.update(known_region_diagnostics(prediction, warped_rgb, input_hole))
            predictions.append(prediction)
            targets.append(target)
            tissue_masks.append(tissue)
            hole_masks.append(hole)

        prediction_batch = torch.stack(predictions).to(device, non_blocking=True)
        target_batch = torch.stack(targets).to(device, non_blocking=True)
        tissue_batch = torch.stack(tissue_masks).to(device, non_blocking=True)
        hole_batch = torch.stack(hole_masks).to(device, non_blocking=True)
        with torch.inference_mode():
            full_lpips = scalar_lpips(
                prediction_batch * tissue_batch.float(),
                target_batch * tissue_batch.float(),
                normalize=True,
            ).reshape(-1)
            spatial = spatial_lpips(prediction_batch, target_batch, normalize=True)
            hole_lpips = (
                (spatial * hole_batch.float())
                .flatten(1)
                .sum(1)
                .div(hole_batch.flatten(1).sum(1).clamp_min(1).float())
            )
        for index, record in enumerate(batch):
            prediction = prediction_batch[index : index + 1]
            target = target_batch[index : index + 1]
            tissue = tissue_batch[index : index + 1]
            hole = hole_batch[index : index + 1]
            record.update(
                {
                    "psnr": masked_psnr(prediction, target, tissue),
                    "ssim": masked_ssim(prediction, target, tissue),
                    "lpips": float(full_lpips[index].item()),
                    "hole_psnr": masked_psnr(prediction, target, hole),
                    "hole_ssim": masked_ssim(prediction, target, hole),
                    "hole_lpips": float(hole_lpips[index].item()),
                }
            )
        completed = min(start + batch_size, len(records))
        method = str(batch[0]["method"]) if batch else "prediction"
        print(f"{method}: {completed}/{len(records)}", flush=True)


def metric_summary(records: list[dict]) -> dict[str, dict[str, float]]:
    return {
        name: sample_summary([float(record[name]) for record in records])
        for name in METRIC_NAMES
    }


def preservation_summary(records: list[dict]) -> dict[str, int | float | dict[str, float]]:
    total_known_pixels = sum(int(record["known_pixel_count"]) for record in records)
    total_changed_pixels = sum(int(record["known_changed_pixel_count"]) for record in records)
    if total_known_pixels <= 0:
        raise ValueError("no known pixels were available for preservation auditing")
    weighted_channel_error = sum(
        float(record["known_mae_8bit"]) * int(record["known_pixel_count"]) * 3
        for record in records
    )
    return {
        "definition": "prediction versus generic-completion warped_rgb outside inpaint_mask",
        "quantization": "absolute RGB error in 8-bit intensity units; a pixel is changed if any channel differs by >0.5",
        "known_pixel_count": total_known_pixels,
        "changed_known_pixel_count": total_changed_pixels,
        "changed_known_pixel_fraction": float(total_changed_pixels / total_known_pixels),
        "frames_with_changed_known_pixels": sum(
            int(record["known_changed_pixel_count"] > 0) for record in records
        ),
        "global_known_mae_8bit": float(weighted_channel_error / (total_known_pixels * 3)),
        "global_known_max_abs_8bit": max(float(record["known_max_abs_8bit"]) for record in records),
        "per_frame_known_mae_8bit": sample_summary(
            [float(record["known_mae_8bit"]) for record in records]
        ),
        "per_frame_changed_pixel_fraction": sample_summary(
            [float(record["known_changed_pixel_fraction"]) for record in records]
        ),
    }


def write_outputs(
    output_dir: Path,
    payload: dict,
    records: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False)
    (output_dir / "metrics.json").write_text(encoded + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(encoded + "\n", encoding="utf-8")
    fields = [
        "method",
        "scene",
        "frame_id",
        "sample_id",
        "degenerate",
        *METRIC_NAMES,
        "known_pixel_count",
        "known_changed_pixel_count",
        "known_changed_pixel_fraction",
        "known_mae_8bit",
        "known_max_abs_8bit",
        "prediction_path",
        "target_path",
        "hole_path",
        "tool_path",
        "warped_rgb_path",
        "inpaint_mask_path",
    ]
    with (output_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    primary_run = args.primary_run.expanduser().resolve()
    prediction_root = args.prediction_root.expanduser().resolve()
    input_list = args.input_list.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    data_root, by_scene = load_protocol(primary_run)
    if len(by_scene) != EXPECTED_SCENES:
        raise ValueError(f"expected {EXPECTED_SCENES} scenes, got {len(by_scene)}")
    expected = expected_sample_ids(by_scene)
    if len(expected) != EXPECTED_SAMPLES:
        raise ValueError(f"expected {EXPECTED_SAMPLES} unique samples, got {len(expected)}")
    completion_inputs = load_completion_inputs(input_list, expected)
    predictions = discover_predictions(prediction_root, args.prediction_pattern, expected)
    records = build_records(
        primary_run,
        data_root,
        by_scene,
        predictions,
        completion_inputs,
        args.method_name,
    )

    import lpips  # type: ignore

    device = torch.device(args.device)
    scalar_lpips = lpips.LPIPS(net="alex").eval().to(device)
    spatial_lpips = lpips.LPIPS(net="alex", spatial=True).eval().to(device)
    evaluate(records, device, args.batch_size, scalar_lpips, spatial_lpips)

    per_scene = {
        scene: {
            "count": len(scene_records),
            "metrics": metric_summary(scene_records),
        }
        for scene, scene_records in sorted(
            (
                (scene, [record for record in records if record["scene"] == scene])
                for scene in by_scene
            ),
            key=lambda item: item[0],
        )
    }
    summary = {
        "method": args.method_name,
        "count": len(records),
        "scene_count": len(by_scene),
        "degenerate_count": sum(bool(record["degenerate"]) for record in records),
        "metrics": metric_summary(records),
        "known_region_preservation": preservation_summary(records),
        "per_scene": per_scene,
    }
    protocol = {
        "name": "Table I fixed-N Hunyuan completion evaluation",
        "primary_run": str(primary_run),
        "prediction_root": str(prediction_root),
        "prediction_pattern": args.prediction_pattern,
        "completion_input_list": str(input_list),
        "checkpoint": str(args.checkpoint.expanduser().resolve()) if args.checkpoint else None,
        "data_root": str(data_root),
        "scenes": {scene: len(stems) for scene, stems in by_scene.items()},
        "count": len(records),
        "aggregation": "frame-wise mean and sample standard deviation (ddof=1)",
        "inclusion": "all protocol frames retained, including degenerate predictions",
        "full_mask": "primary-run non-tool tissue mask",
        "hole_mask": "primary-run non-tool tissue intersected with saved raw-DSS hole mask",
        "lpips": "AlexNet LPIPS; [0,1] inputs normalized internally to [-1,1]; spatial map averaged for H-LPIPS",
        "known_region_audit": "generic-completion warped_rgb outside its binary inpaint_mask; not used for quality metrics",
    }
    payload = {"protocol": protocol, "methods": {args.method_name: summary}}
    write_outputs(output_dir, payload, records)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
