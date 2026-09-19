from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from PIL import Image

@dataclass(frozen=True)
class Sample:
    name: str
    image: Path
    mask: Path
    target: Path | None = None
    scene: str = "default"

def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()

def samples_from_manifest(path: Path) -> Iterator[Sample]:
    base = path.resolve().parent
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_value = row.get("warped_rgb", row.get("image"))
            mask_value = row.get("inpaint_mask", row.get("mask"))
            if not image_value or not mask_value:
                raise KeyError(f"{path}:{number}: warped_rgb/image and inpaint_mask/mask required")
            image = _resolve(image_value, base)
            target_value = row.get("target_rgb", row.get("target", row.get("ground_truth")))
            yield Sample(str(row.get("name", image.stem)), image, _resolve(mask_value, base),
                         _resolve(target_value, base) if target_value else None,
                         str(row.get("scene", row.get("sequence", "default"))))

def discover_samples(root: Path) -> Iterator[Sample]:
    root = root.resolve()
    if (root / "warped_rgb.png").is_file() and (root / "inpaint_mask.png").is_file():
        target = root / "target_rgb.png"
        yield Sample("completed_rgb", root / "warped_rgb.png", root / "inpaint_mask.png",
                     target if target.is_file() else None, root.name)
        return
    for image in sorted(root.glob("**/warped_rgb/*")):
        if not image.is_file():
            continue
        frame_root = image.parent.parent
        mask = frame_root / "inpaint_mask" / image.name
        if not mask.is_file():
            raise FileNotFoundError(f"Missing matching mask: {mask}")
        target = next((frame_root / key / image.name for key in ("target_rgb", "ground_truth", "gt")
                       if (frame_root / key / image.name).is_file()), None)
        yield Sample(image.stem, image, mask, target, frame_root.parent.name)

def load_samples(value: str) -> list[Sample]:
    path = Path(value)
    samples = list(samples_from_manifest(path) if path.is_file() else discover_samples(path))
    if not samples:
        raise FileNotFoundError(f"No input pairs found under {path}")
    for sample in samples:
        if not sample.image.is_file() or not sample.mask.is_file():
            raise FileNotFoundError(f"Missing input for {sample.name}: {sample.image}, {sample.mask}")
    return samples

def load_pair(sample: Sample) -> tuple[Image.Image, Image.Image]:
    image = Image.open(sample.image).convert("RGB")
    mask = Image.open(sample.mask).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    return image, mask

def output_path(root: Path, sample: Sample, method: str) -> Path:
    if os.environ.get("ICRA_METHOD_FIRST_OUTPUT") == "1":
        directory = root / method / sample.scene
    else:
        directory = root / sample.scene / method
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{sample.name}.png"
