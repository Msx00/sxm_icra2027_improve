#!/usr/bin/env python3
"""Benchmark official big-LaMa forward latency on a JSONL pair manifest."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
LAMA_REPO = ROOT / "third_party/lama"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LAMA_REPO))

from common.io import load_pair, load_samples
from common.mask import binary_mask
from saicinpainting.training.trainers import load_checkpoint


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(ROOT / "weights/big-lama"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--expected-count", type=int, default=275)
    return parser.parse_args()


def prepare(sample, device: torch.device) -> dict[str, torch.Tensor]:
    image, mask = load_pair(sample)
    image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
    mask_array = np.asarray(binary_mask(mask), dtype=np.float32).copy() / 255.0
    return {
        "image": torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device),
        "mask": torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).to(device),
    }


def main() -> None:
    args = arguments()
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Table-II latency benchmark")

    samples = load_samples(args.input)
    if len(samples) != args.expected_count:
        raise RuntimeError(f"expected {args.expected_count} samples, got {len(samples)}")

    model_root = Path(args.model).expanduser().resolve()
    with (model_root / "config.yaml").open("r", encoding="utf-8") as handle:
        train_config = OmegaConf.create(yaml.safe_load(handle))
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = "noop"
    checkpoint = model_root / "models/best.ckpt"
    model = load_checkpoint(
        train_config,
        str(checkpoint),
        strict=False,
        map_location="cpu",
    )
    model.freeze()
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    warm_batch = prepare(samples[0], device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model({key: value for key, value in warm_batch.items()})
        torch.cuda.synchronize(device)

        records = []
        for ordinal, sample in enumerate(samples):
            batch = prepare(sample, device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = model(batch)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            records.append({
                "sample_id": f"{sample.scene}/{sample.name}",
                "seconds": seconds,
            })
            if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(samples):
                print(f"[{ordinal + 1:03d}/{len(samples)}] {seconds:.6f} s", flush=True)
            del result, batch

    values = [row["seconds"] for row in records]
    payload = {
        "method": "LaMa",
        "protocol": "single-frame GPU model latency; model loaded once; disk I/O excluded",
        "gpu": torch.cuda.get_device_name(device),
        "precision": "float32",
        "batch_size": 1,
        "warmup_frames": args.warmup,
        "count": len(values),
        "mean_seconds": statistics.fmean(values),
        "sample_sd_seconds": statistics.stdev(values),
        "input": str(Path(args.input).expanduser().resolve()),
        "model": str(model_root),
        "checkpoint": str(checkpoint),
        "frames": records,
    }
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(
        f"LaMa: {payload['mean_seconds']:.6f} +- "
        f"{payload['sample_sd_seconds']:.6f} s/frame"
    )


if __name__ == "__main__":
    main()
