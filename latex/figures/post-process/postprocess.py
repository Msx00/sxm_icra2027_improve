#!/usr/bin/env python3
"""Conservative reference-free enhancement for endoscopic images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_IMAGES = ("frame_000299_423.png", "frame_000299_752.png")


def robust_luminance_stretch(
    image_bgr: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """Stretch luminance independently while excluding black padding."""
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    luminance = ycrcb[:, :, 0]
    valid = np.max(image_bgr, axis=2) > 5
    samples = luminance[valid] if np.any(valid) else luminance.reshape(-1)
    low, high = np.percentile(samples, [low_percentile, high_percentile])

    if high > low + 1e-6:
        stretched = (luminance - low) * (245.0 / (high - low)) + 5.0
        ycrcb[:, :, 0] = np.where(valid, np.clip(stretched, 0, 255), luminance)

    return cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def enhance_saturation(image_bgr: np.ndarray, factor: float = 1.05) -> np.ndarray:
    """Apply a small reference-free saturation increase."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def unsharp_luminance(
    image_bgr: np.ndarray,
    sigma: float = 0.9,
    amount: float = 0.15,
    threshold: float = 2.0,
) -> np.ndarray:
    """Sharpen luminance mildly without amplifying low-amplitude noise."""
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    luminance = ycrcb[:, :, 0].astype(np.float32)
    blurred = cv2.GaussianBlur(luminance, (0, 0), sigma)
    detail = luminance - blurred
    detail[np.abs(detail) < threshold] = 0.0
    ycrcb[:, :, 0] = np.clip(luminance + amount * detail, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def enhance(image_bgr: np.ndarray) -> np.ndarray:
    output = robust_luminance_stretch(image_bgr)
    output = enhance_saturation(output)
    return unsharp_luminance(output)


def process_image(input_path: Path, output_dir: Path) -> Path:
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(input_path)

    output_path = output_dir / f"{input_path.stem}_enhanced.png"
    if not cv2.imwrite(str(output_path), enhance(image)):
        raise OSError(f"Failed to write {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        default=[script_dir / name for name in DEFAULT_IMAGES],
        help="input images; defaults to the two frame_000299 images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="directory for *_enhanced.png outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in args.images:
        output_path = process_image(image_path.resolve(), args.output_dir.resolve())
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
