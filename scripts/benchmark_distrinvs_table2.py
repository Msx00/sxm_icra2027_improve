#!/usr/bin/env python3
"""Benchmark Table-II DistriNVS model latency on the frozen EndoVis pairs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distrisurg.config import load_config
from distrisurg.data import StereoEndoscopyDataset
from distrisurg.models import DistriSurg
from distrisurg.utils.io import load_checkpoint, write_json_atomic


DEFAULT_DATA = (
    "/home/data/mashixing/dataset_18tb/"
    "icra2027-diffusion-zero-test-dataset/endovis-final-more-data"
)
DEFAULT_MANIFEST = ROOT / (
    "results/zeroshot/endovis_final_more_275/"
    "distrinvs_soft_train_hard_guard/run_manifest.json"
)
DEFAULT_CHECKPOINT = ROOT / (
    "checkpoints/ablations/no_hard_composition/seed_6666/step_0010000.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "results/zeroshot/endovis_final_more_275/timing/distrinvs.json"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/distrisurg_dataset89.yaml"))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--selection-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--expected-count", type=int, default=275)
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args()


def model_inputs(sample: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    keys = (
        "source_rgb",
        "source_depth",
        "source_depth_valid",
        "source_intrinsics",
        "target_intrinsics",
        "source_to_target",
    )
    return tuple(sample[key].unsqueeze(0).to(device) for key in keys)


def main() -> None:
    args = arguments()
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the Table-II latency benchmark")

    overrides = [
        f"data.eval_root={DEFAULT_DATA}",
        "data.min_overlap=0.7",
        "ablation.hard_composition=true",
        *args.set,
    ]
    config = load_config(args.config, overrides, [])
    device = torch.device(args.device)

    selection_path = Path(args.selection_manifest).expanduser().resolve()
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_ids = [row["sample_id"] for row in selection_payload["selected_frames"]]
    if len(selected_ids) != args.expected_count or len(set(selected_ids)) != len(selected_ids):
        raise RuntimeError(
            f"expected {args.expected_count} unique selected frames, got {len(selected_ids)}"
        )

    dataset = StereoEndoscopyDataset(
        config.data.eval_root,
        height=config.data.height,
        width=config.data.width,
    )
    dataset_index = {
        f"{sample.scene}/frame_{sample.frame_id:06d}": index
        for index, sample in enumerate(dataset.samples)
    }
    missing = [sample_id for sample_id in selected_ids if sample_id not in dataset_index]
    if missing:
        raise RuntimeError(f"selected samples missing from dataset: {missing[:8]}")

    model = DistriSurg(config).to(device).eval()
    load_checkpoint(args.checkpoint, model, device)

    first = model_inputs(dataset[dataset_index[selected_ids[0]]], device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(*first)
        torch.cuda.synchronize(device)

        records = []
        for ordinal, sample_id in enumerate(selected_ids):
            inputs = model_inputs(dataset[dataset_index[sample_id]], device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            prediction = model(*inputs)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            records.append({
                "sample_id": sample_id,
                "seconds": seconds,
            })
            if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(selected_ids):
                print(f"[{ordinal + 1:03d}/{len(selected_ids)}] {seconds:.6f} s", flush=True)
            del prediction, inputs

    values = [row["seconds"] for row in records]
    payload = {
        "method": "DistriNVS",
        "protocol": "single-frame GPU model latency; model loaded once; disk I/O excluded",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "precision": "float32",
        "batch_size": 1,
        "warmup_frames": args.warmup,
        "count": len(values),
        "mean_seconds": statistics.fmean(values),
        "sample_sd_seconds": statistics.stdev(values),
        "selection_manifest": str(selection_path),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "config": config.to_dict(),
        "frames": records,
    }
    write_json_atomic(Path(args.output).expanduser().resolve(), payload)
    print(
        f"DistriNVS: {payload['mean_seconds']:.6f} +- "
        f"{payload['sample_sd_seconds']:.6f} s/frame"
    )


if __name__ == "__main__":
    main()
