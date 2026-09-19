#!/usr/bin/env python3
"""Compute Table-I LPIPS metrics for a saved DistriNVS inference run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
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
    mask = array < 128 if invert else array >= 128
    return torch.from_numpy(mask).unsqueeze(0)


def sample_sd(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_root = Path(manifest["data_root"])
    samples = [str(row["sample_id"]) for row in manifest["selected_frames"]]
    if not manifest.get("completed") or not samples:
        raise RuntimeError(f"incomplete or empty run: {manifest_path}")

    import lpips  # type: ignore

    device = torch.device(args.device)
    model = lpips.LPIPS(net="alex").eval().to(device)
    spatial_model = lpips.LPIPS(net="alex", spatial=True).eval().to(device)
    full_values: list[float] = []
    hole_values: list[float] = []

    for start in range(0, len(samples), args.batch_size):
        batch_samples = samples[start : start + args.batch_size]
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        full_masks: list[torch.Tensor] = []
        hole_masks: list[torch.Tensor] = []
        for sample_id in batch_samples:
            scene, stem = sample_id.split("/")
            filename = f"{stem}.png"
            prediction = load_rgb(run_dir / scene / "renders" / filename)
            target = load_rgb(run_dir / scene / "targets" / filename)
            height, width = prediction.shape[-2:]
            if target.shape != prediction.shape:
                raise ValueError(f"shape mismatch for {sample_id}")
            tissue = load_mask(
                data_root / scene / "endoscope1" / "toolL" / filename,
                (width, height),
                invert=True,
                missing_value=True,
            )
            hole = load_mask(run_dir / scene / "hole_mask" / filename, (width, height))
            predictions.append(prediction)
            targets.append(target)
            full_masks.append(tissue)
            hole_masks.append(tissue & hole)

        prediction_batch = torch.stack(predictions).to(device, non_blocking=True)
        target_batch = torch.stack(targets).to(device, non_blocking=True)
        full_batch = torch.stack(full_masks).to(device, non_blocking=True).float()
        hole_batch = torch.stack(hole_masks).to(device, non_blocking=True).float()
        with torch.inference_mode():
            # Inputs are loaded in [0,1]; normalize=True applies LPIPS's required
            # conversion to [-1,1] before AlexNet feature extraction.
            full = model(prediction_batch * full_batch, target_batch * full_batch, normalize=True).reshape(-1)
            spatial = spatial_model(prediction_batch, target_batch, normalize=True)
            hole = (spatial * hole_batch).flatten(1).sum(1).div(
                hole_batch.flatten(1).sum(1).clamp_min(1.0)
            )
        full_values.extend(float(value) for value in full.cpu())
        hole_values.extend(float(value) for value in hole.cpu())
        print(f"{min(start + args.batch_size, len(samples))}/{len(samples)}", flush=True)

    result = {
        "protocol": (
            "Table I; full LPIPS uses zeroed non-tissue pixels and scalar AlexNet LPIPS; "
            "H-LPIPS averages the spatial AlexNet LPIPS map over the tissue/raw-DSS-hole "
            "intersection; [0,1] inputs are normalized internally to [-1,1]"
        ),
        "count": len(samples),
        "lpips": {"mean": float(np.mean(full_values)), "sample_sd": sample_sd(full_values)},
        "hole_lpips": {"mean": float(np.mean(hole_values)), "sample_sd": sample_sd(hole_values)},
        "frames": [
            {"sample_id": sample_id, "lpips": full, "hole_lpips": hole}
            for sample_id, full, hole in zip(samples, full_values, hole_values)
        ],
    }
    output_path = run_dir / "table1_lpips.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)
    print(json.dumps({key: value for key, value in result.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
