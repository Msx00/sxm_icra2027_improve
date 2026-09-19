#!/usr/bin/env python3
"""Zero-shot dataset89 inference with strict raw-overlap filtering."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distrisurg.config import load_config
from distrisurg.data import StereoEndoscopyDataset, read_scene_list
from distrisurg.metrics import frame_metrics, scene_bootstrap_summary
from distrisurg.geometry import projection_defect_taxonomy, taxonomy_ratios
from distrisurg.models import DistriSurg
from distrisurg.postprocess import repair_small_synthesis_regions
from distrisurg.utils.io import (
    load_checkpoint,
    save_heatmap,
    save_label_map,
    save_mask,
    save_rgb,
    write_json_atomic,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/train.yaml"))
    parser.add_argument("--ablation-config", action="append", default=[])
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output", default=str(ROOT / "outputs/dataset89"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scenes-file", default="")
    parser.add_argument("--scenes", nargs="*", default=[])
    parser.add_argument("--split-name", default="dataset89")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--renderer-only", action="store_true")
    parser.add_argument("--sigma-mm", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--small-hole-max-area-ratio",
        type=float,
        default=0.0,
        help="repair untrusted components up to this image-area ratio; 0 disables",
    )
    parser.add_argument("--small-hole-inpaint-radius", type=float, default=3.0)
    return parser.parse_args()


def move_tensors(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().item())


def save_row_images(
    output: Path,
    scene: str,
    frame: int,
    prediction: dict[str, torch.Tensor],
    raw: dict[str, torch.Tensor],
    target: torch.Tensor,
    taxonomy: torch.Tensor,
) -> None:
    name = f"frame_{frame:06d}.png"
    base = output / scene
    save_rgb(base / "renders" / name, prediction["target_rgb"])
    save_rgb(base / "warps" / name, prediction["warped_rgb"])
    save_rgb(base / "targets" / name, target)
    save_mask(base / "raw_valid_mask" / name, raw["valid_mask"])
    save_mask(base / "trusted_mask" / name, prediction["trusted_mask"])
    if "small_hole_repair_mask" in prediction:
        save_mask(
            base / "small_hole_repair_mask" / name,
            prediction["small_hole_repair_mask"],
        )
    if "raw_trusted_mask" in prediction:
        save_mask(
            base / "raw_trusted_mask" / name,
            prediction["raw_trusted_mask"],
        )
    save_mask(base / "hole_mask" / name, ~raw["valid_mask"])
    save_heatmap(base / "support" / name, prediction["support"])
    save_heatmap(base / "variance" / name, prediction["render_variance"])
    save_heatmap(
        base / "collision_entropy" / name,
        prediction["collision_entropy"],
        maximum=1.0,
    )
    save_label_map(base / "projection_taxonomy" / name, taxonomy)
    save_heatmap(base / "risk" / name, prediction["risk"], maximum=1.0)
    save_heatmap(
        base / "synthesis_gate" / name,
        prediction["synthesis_gate"],
        maximum=1.0,
    )
    if "raw_synthesis_gate" in prediction:
        save_heatmap(
            base / "raw_synthesis_gate" / name,
            prediction["raw_synthesis_gate"],
            maximum=1.0,
        )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = arguments()
    if not 0.0 <= args.small_hole_max_area_ratio <= 1.0:
        raise ValueError("--small-hole-max-area-ratio must be in [0, 1]")
    if args.small_hole_inpaint_radius <= 0.0:
        raise ValueError("--small-hole-inpaint-radius must be positive")
    config = load_config(args.config, args.set, args.ablation_config)
    if not args.renderer_only and not args.checkpoint:
        raise ValueError("--checkpoint is required unless --renderer-only is selected")
    if args.scenes_file and args.scenes:
        raise ValueError("choose either --scenes-file or --scenes")
    selected = None
    if args.scenes_file:
        selected = read_scene_list(args.scenes_file)
    elif args.scenes:
        selected = args.scenes
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    dataset = StereoEndoscopyDataset(
        config.data.eval_root,
        height=config.data.height,
        width=config.data.width,
        scene_names=selected,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = DistriSurg(config).to(device).eval()
    checkpoint_payload: dict = {}
    if args.checkpoint:
        checkpoint_payload = load_checkpoint(args.checkpoint, model, device)
        train_scenes = set(checkpoint_payload.get("train_scenes", []))
        overlap = train_scenes & set(dataset.scenes)
        if overlap:
            raise RuntimeError(f"checkpoint training/evaluation scene leakage: {sorted(overlap)}")
    inference_config = config.to_dict()
    checkpoint_config = checkpoint_payload.get("config", {})
    checkpoint_ablation = (
        checkpoint_config.get("ablation", {})
        if isinstance(checkpoint_config, dict)
        else {}
    )
    inference_ablation = inference_config["ablation"]
    ablation_mismatches = {
        key: {
            "checkpoint_training": checkpoint_ablation.get(key),
            "inference": inference_ablation.get(key),
        }
        for key in sorted(set(checkpoint_ablation) | set(inference_ablation))
        if checkpoint_ablation.get(key) != inference_ablation.get(key)
    }
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"non-empty output exists; pass --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "DSS renderer" if args.renderer_only else "DistriSurg",
        "split": args.split_name,
        "protocol": (
            f"{args.split_name}; right-to-left inference; raw DSS valid ratio "
            "strictly greater than threshold"
        ),
        "data_root": str(Path(config.data.eval_root).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "checkpoint_step": checkpoint_payload.get("step"),
        "checkpoint_training_ablation": checkpoint_ablation,
        "checkpoint_inference_ablation_mismatches": ablation_mismatches,
        "checkpoint_train_scenes": checkpoint_payload.get("train_scenes", []),
        "scenes": dataset.scenes,
        "min_overlap": config.data.min_overlap,
        "small_hole_repair": {
            "method": "telea_from_warped_rgb",
            "max_area_ratio": args.small_hole_max_area_ratio,
            "radius": args.small_hole_inpaint_radius,
        },
        "config": inference_config,
        "selected_frames": [],
        "skipped_frames": [],
        "completed": False,
        "started_at_unix": time.time(),
    }
    write_json_atomic(output / "run_manifest.json", manifest)
    rows: list[dict] = []
    with torch.inference_mode():
        for index, raw_batch in enumerate(loader):
            if args.max_frames > 0 and index >= args.max_frames:
                break
            batch = move_tensors(raw_batch, device)
            scene = str(batch["scene"][0])
            frame = int(batch["frame_id"].item())
            raw_render = model.renderer_only(
                batch["source_rgb"],
                batch["source_depth"],
                batch["source_depth_valid"],
                batch["source_intrinsics"],
                batch["target_intrinsics"],
                batch["source_to_target"],
                sigma_mm=args.sigma_mm,
            )
            overlap = scalar(raw_render["valid_mask"].float().mean())
            sample_id = f"{scene}/frame_{frame:06d}"
            if not overlap > config.data.min_overlap:
                manifest["skipped_frames"].append(
                    {"sample_id": sample_id, "raw_overlap": overlap}
                )
                print(f"skip {sample_id}: overlap={overlap:.4f}", flush=True)
                continue
            if args.renderer_only:
                prediction = {
                    "target_rgb": raw_render["features"].clamp(0.0, 1.0),
                    "warped_rgb": raw_render["features"].clamp(0.0, 1.0),
                    "trusted_mask": raw_render["valid_mask"],
                    "support": raw_render["support"],
                    "render_variance": raw_render["variance"],
                    "collision_entropy": raw_render["collision_entropy"],
                    "risk": 1.0 - raw_render["confidence"],
                    "synthesis_gate": (~raw_render["valid_mask"]).float(),
                    "completed_valid_mask": raw_render["valid_mask"],
                }
            else:
                prediction = model(
                    batch["source_rgb"],
                    batch["source_depth"],
                    batch["source_depth_valid"],
                    batch["source_intrinsics"],
                    batch["target_intrinsics"],
                    batch["source_to_target"],
                )
            repaired_rgb, repair_mask = repair_small_synthesis_regions(
                prediction["target_rgb"],
                prediction["warped_rgb"],
                prediction["trusted_mask"],
                max_area_ratio=args.small_hole_max_area_ratio,
                radius=args.small_hole_inpaint_radius,
            )
            prediction["target_rgb"] = repaired_rgb
            prediction["small_hole_repair_mask"] = repair_mask
            tissue = ~batch["target_tool_mask"].bool()
            taxonomy = projection_defect_taxonomy(
                raw_render["valid_mask"],
                prediction["completed_valid_mask"],
                prediction["render_variance"],
                prediction["collision_entropy"],
                config.model.trusted_variance_mm2,
                config.model.trusted_entropy,
            )
            values = frame_metrics(
                prediction["target_rgb"],
                batch["target_rgb"],
                raw_render["valid_mask"],
                prediction["trusted_mask"],
                prediction["warped_rgb"],
                prediction["risk"],
                tissue,
                seam_width=config.loss.seam_width,
            )
            row = {"scene": scene, "frame_id": frame, "sample_id": sample_id, **values}
            row["small_hole_repair_ratio"] = scalar(repair_mask.float().mean())
            row.update(taxonomy_ratios(taxonomy))
            rows.append(row)
            manifest["selected_frames"].append(
                {"sample_id": sample_id, "raw_overlap": overlap}
            )
            save_row_images(
                output,
                scene,
                frame,
                prediction,
                raw_render,
                batch["target_rgb"],
                taxonomy,
            )
            print(
                f"[{len(rows):04d}] {sample_id} overlap={overlap:.4f} "
                f"PSNR={values['psnr']:.3f} hole={values['hole_psnr']:.3f}",
                flush=True,
            )
    if not rows:
        raise RuntimeError("no frame passed the strict overlap threshold")
    summary = scene_bootstrap_summary(rows)
    write_csv(output / "frame_metrics.csv", rows)
    write_json_atomic(output / "metrics.json", {"summary": summary, "frames": rows})
    manifest["completed"] = True
    manifest["completed_at_unix"] = time.time()
    manifest["selected_count"] = len(rows)
    manifest["skipped_count"] = len(manifest["skipped_frames"])
    write_json_atomic(output / "run_manifest.json", manifest)
    concise = {key: round(value["mean"], 5) for key, value in summary.items()}
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
