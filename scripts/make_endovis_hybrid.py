#!/usr/bin/env python3
"""Create a fixed-weight DistriNVS--LaMa hybrid for diagnostic evaluation.

This is a separate candidate output and must not replace pure DistriNVS in the
paper. LaMa is an additional expert. The default 0.5 weight is a fixed
diagnostic setting, not fitted on EndoVis targets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(image), 0, 255).astype(np.uint8)).save(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--distrinvs-root", type=Path, required=True)
    p.add_argument("--lama-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--distrinvs-weight", type=float, default=0.5)
    args = p.parse_args()
    if not 0.0 <= args.distrinvs_weight <= 1.0:
        raise ValueError("--distrinvs-weight must be in [0, 1]")
    for dpath in sorted(args.distrinvs_root.glob("*/renders/*.png")):
        scene, name = dpath.parents[1].name, dpath.name
        lpath = args.lama_root / scene / name
        if not lpath.is_file():
            raise FileNotFoundError(lpath)
        d = rgb(dpath)
        l = rgb(lpath)
        if d.shape != l.shape:
            raise ValueError(f"shape mismatch: {dpath} vs {lpath}")
        w = args.distrinvs_weight
        save(args.output / scene / "renders" / name, w * d + (1.0 - w) * l)


if __name__ == "__main__":
    main()
