#!/usr/bin/env python3
"""Uniformly evaluate Table-I image-completion baselines on 1,392 frames.

LDM/Stable Diffusion 1.5, LaMa, LCM-LoRA ``final``, and (optionally)
Hunyuan-DiT are mapped by exact ``scene/frame`` identifiers and evaluated with
the primary run's targets, non-tool tissue masks, and saved raw-DSS hole masks.
All methods use the same metric implementation and frame-wise aggregation.

The Hunyuan method additionally audits preservation of the generic completion
input outside its inpaint mask.  Missing, duplicated, or unexpected samples
are fatal; no prediction is filtered post hoc.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

import evaluate_table1_hunyuan as hunyuan_eval
from evaluate_table1_reconstruction_baselines import (
    load_mask,
    load_protocol,
    load_rgb,
    masked_psnr,
    masked_ssim,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_RUN = PROJECT_ROOT / "results" / "validation" / "distrisurg"
GENERIC_ROOT = (
    PROJECT_ROOT
    / "results"
    / "validation"
    / "generic_inpainting"
    / "comparison_methods"
)
INPUT_LIST = GENERIC_ROOT / "zeroshot_inputs.jsonl"
SUPERVISED_LORA_ROOT = PROJECT_ROOT / "results" / "validation" / "supervised_lora"
HUNYUAN_ROOT = PROJECT_ROOT / "results" / "validation" / "hunyuan_completion" / "parts"
OUTPUT_DIR = PROJECT_ROOT / "results" / "validation" / "table1_completion_equal_n"


@dataclass(frozen=True)
class MethodSpec:
    key: str
    display_name: str
    root: Path
    pattern: str
    scene_parent: int
    filename_suffix: str = ""
    audit_known_region: bool = False


METHOD_ALIASES = {
    "ldm": "sd15",
    "sd15": "sd15",
    "lama": "lama",
    "lcm": "lcm_lora",
    "lcm_lora": "lcm_lora",
    "hunyuan": "hunyuan",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-run", type=Path, default=PRIMARY_RUN)
    parser.add_argument("--input-list", type=Path, default=INPUT_LIST)
    parser.add_argument("--sd15-root", type=Path, default=GENERIC_ROOT / "predictions" / "sd15")
    parser.add_argument("--lama-root", type=Path, default=GENERIC_ROOT / "predictions" / "lama")
    parser.add_argument("--lcm-root", type=Path, default=SUPERVISED_LORA_ROOT)
    parser.add_argument("--hunyuan-root", type=Path, default=HUNYUAN_ROOT)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_ALIASES),
        default=["sd15", "lama", "lcm_lora"],
        help="Default excludes Hunyuan until its 1,392 predictions are complete.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--hunyuan-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check exact sample mapping and image dimensions without loading LPIPS.",
    )
    return parser.parse_args()


def method_specs(args: argparse.Namespace) -> dict[str, MethodSpec]:
    return {
        "sd15": MethodSpec(
            key="sd15",
            display_name="LDM",
            root=args.sd15_root.expanduser().resolve(),
            pattern="*/*.png",
            scene_parent=0,
        ),
        "lama": MethodSpec(
            key="lama",
            display_name="LaMa",
            root=args.lama_root.expanduser().resolve(),
            pattern="*/*.png",
            scene_parent=0,
        ),
        "lcm_lora": MethodSpec(
            key="lcm_lora",
            display_name="LCM-LoRA",
            root=args.lcm_root.expanduser().resolve(),
            pattern="*/results/endo1_supervised_rank32_step2500/lcm/final/*.png",
            scene_parent=4,
        ),
        "hunyuan": MethodSpec(
            key="hunyuan",
            display_name="Hunyuan-DiT (in-domain adapted)",
            root=args.hunyuan_root.expanduser().resolve(),
            pattern="gpu*/inference/*/*_completed.png",
            scene_parent=0,
            filename_suffix="_completed",
            audit_known_region=True,
        ),
    }


def selected_method_keys(names: list[str]) -> list[str]:
    keys: list[str] = []
    for name in names:
        key = METHOD_ALIASES[name]
        if key in keys:
            raise ValueError(f"method selected more than once through aliases: {name} -> {key}")
        keys.append(key)
    return keys


def discover_predictions(spec: MethodSpec, expected: set[str]) -> dict[str, Path]:
    if not spec.root.is_dir():
        raise FileNotFoundError(spec.root)
    paths = sorted(path.resolve() for path in spec.root.glob(spec.pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"{spec.display_name}: no predictions matched {spec.root / spec.pattern}")
    predictions: dict[str, Path] = {}
    for path in paths:
        stem = path.stem
        if spec.filename_suffix:
            if not stem.endswith(spec.filename_suffix):
                raise ValueError(
                    f"{spec.display_name}: filename does not end with "
                    f"{spec.filename_suffix!r}: {path}"
                )
            stem = stem[: -len(spec.filename_suffix)]
        try:
            scene = path.parents[spec.scene_parent].name
        except IndexError as error:
            raise ValueError(f"{spec.display_name}: cannot extract scene from {path}") from error
        identifier = hunyuan_eval.sample_id(scene, stem)
        if identifier in predictions:
            raise ValueError(
                f"{spec.display_name}: duplicate prediction for {identifier}: "
                f"{predictions[identifier]} and {path}"
            )
        predictions[identifier] = path
    actual = set(predictions)
    if actual != expected:
        raise ValueError(
            hunyuan_eval.describe_set_mismatch(spec.display_name, actual, expected)
        )
    return predictions


def evaluate_method(
    spec: MethodSpec,
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
            if spec.audit_known_region:
                warped_rgb = load_rgb(Path(record["warped_rgb_path"]), size)
                input_hole = load_mask(Path(record["inpaint_mask_path"]), size)
                record.update(
                    hunyuan_eval.known_region_diagnostics(prediction, warped_rgb, input_hole)
                )
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
        print(
            f"{spec.display_name}: {min(start + batch_size, len(records))}/{len(records)}",
            flush=True,
        )


def summarize_method(spec: MethodSpec, records: list[dict]) -> dict:
    per_scene: dict[str, dict] = {}
    for scene in sorted({str(record["scene"]) for record in records}):
        scene_records = [record for record in records if record["scene"] == scene]
        per_scene[scene] = {
            "count": len(scene_records),
            "metrics": hunyuan_eval.metric_summary(scene_records),
        }
    summary = {
        "method_key": spec.key,
        "method": spec.display_name,
        "count": len(records),
        "scene_count": len(per_scene),
        "degenerate_count": sum(bool(record["degenerate"]) for record in records),
        "metrics": hunyuan_eval.metric_summary(records),
        "per_scene": per_scene,
    }
    if spec.audit_known_region:
        summary["known_region_preservation"] = hunyuan_eval.preservation_summary(records)
    return summary


def write_outputs(output_dir: Path, payload: dict, records: list[dict]) -> None:
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
        *hunyuan_eval.METRIC_NAMES,
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
    input_list = args.input_list.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    data_root, by_scene = load_protocol(primary_run)
    if len(by_scene) != hunyuan_eval.EXPECTED_SCENES:
        raise ValueError(
            f"expected {hunyuan_eval.EXPECTED_SCENES} scenes, got {len(by_scene)}"
        )
    expected = hunyuan_eval.expected_sample_ids(by_scene)
    if len(expected) != hunyuan_eval.EXPECTED_SAMPLES:
        raise ValueError(
            f"expected {hunyuan_eval.EXPECTED_SAMPLES} samples, got {len(expected)}"
        )
    completion_inputs = hunyuan_eval.load_completion_inputs(input_list, expected)
    specs = method_specs(args)
    method_keys = selected_method_keys(args.methods)

    records_by_method: dict[str, list[dict]] = {}
    validation: dict[str, dict] = {}
    for key in method_keys:
        spec = specs[key]
        predictions = discover_predictions(spec, expected)
        records = hunyuan_eval.build_records(
            primary_run,
            data_root,
            by_scene,
            predictions,
            completion_inputs,
            spec.display_name,
        )
        records_by_method[key] = records
        validation[key] = {
            "method": spec.display_name,
            "root": str(spec.root),
            "pattern": spec.pattern,
            "count": len(records),
            "scene_count": len({record["scene"] for record in records}),
            "dimensions": "640x512 RGB predictions; exact target-size equality checked",
        }

    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "primary_run": str(primary_run),
                    "input_list": str(input_list),
                    "expected_count": len(expected),
                    "methods": validation,
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    import lpips  # type: ignore

    device = torch.device(args.device)
    scalar_lpips = lpips.LPIPS(net="alex").eval().to(device)
    spatial_lpips = lpips.LPIPS(net="alex", spatial=True).eval().to(device)
    summaries: dict[str, dict] = {}
    all_records: list[dict] = []
    for key in method_keys:
        spec = specs[key]
        records = records_by_method[key]
        evaluate_method(spec, records, device, args.batch_size, scalar_lpips, spatial_lpips)
        summaries[spec.display_name] = summarize_method(spec, records)
        all_records.extend(records)

    layouts = {
        specs[key].display_name: {
            "root": str(specs[key].root),
            "pattern": specs[key].pattern,
        }
        for key in method_keys
    }
    protocol = {
        "name": "Table I equal-N image-completion evaluation",
        "primary_run": str(primary_run),
        "completion_input_list": str(input_list),
        "hunyuan_checkpoint": (
            str(args.hunyuan_checkpoint.expanduser().resolve())
            if args.hunyuan_checkpoint
            else None
        ),
        "data_root": str(data_root),
        "scenes": {scene: len(stems) for scene, stems in by_scene.items()},
        "count_per_method": len(expected),
        "prediction_layouts": layouts,
        "aggregation": "frame-wise mean and sample standard deviation (ddof=1)",
        "inclusion": "all protocol frames retained, including degenerate predictions",
        "full_mask": "primary-run non-tool tissue mask",
        "hole_mask": "primary-run non-tool tissue intersected with saved raw-DSS hole mask",
        "lpips": "AlexNet LPIPS; [0,1] inputs normalized internally to [-1,1]; spatial map averaged for H-LPIPS",
        "known_region_audit": "Hunyuan only: generic warped_rgb outside its binary inpaint_mask; not used for quality metrics",
    }
    payload = {"protocol": protocol, "methods": summaries}
    write_outputs(output_dir, payload, all_records)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
