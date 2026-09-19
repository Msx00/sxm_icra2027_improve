from __future__ import annotations
import math
import numpy as np
from PIL import Image

def load_rgb(path, size=None):
    image = Image.open(path).convert("RGB")
    if size and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0

def load_mask(path, size):
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 127

def psnr(prediction, target, mask=None):
    squared = (prediction - target) ** 2
    if mask is not None:
        if not np.any(mask): return float("nan")
        mse = float(squared[mask].mean())
    else: mse = float(squared.mean())
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)

def ssim(prediction, target, mask=None):
    from skimage.metrics import structural_similarity
    _, score_map = structural_similarity(target, prediction, channel_axis=2,
        data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
        full=True)
    per_pixel = score_map.mean(axis=2) if score_map.ndim == 3 else score_map
    if mask is not None:
        return float(per_pixel[mask].mean()) if np.any(mask) else float("nan")
    return float(per_pixel.mean())

