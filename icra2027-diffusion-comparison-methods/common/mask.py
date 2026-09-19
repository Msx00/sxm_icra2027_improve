from __future__ import annotations
import numpy as np
from PIL import Image

def binary_mask(mask: Image.Image, threshold: int = 127) -> Image.Image:
    array = np.asarray(mask.convert("L"), dtype=np.uint8)
    return Image.fromarray(np.where(array > threshold, 255, 0).astype(np.uint8), "L")

def composite_known(generated: Image.Image, warped: Image.Image, mask: Image.Image) -> Image.Image:
    """White selects generated pixels; black preserves the geometric warp."""
    generated = generated.convert("RGB").resize(warped.size, Image.Resampling.LANCZOS)
    return Image.composite(generated, warped.convert("RGB"), binary_mask(mask))

