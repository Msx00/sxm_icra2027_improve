#!/usr/bin/env python3
"""Run the trained no-hard-composition model and visualize DSS outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties, findfont, fontManager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image


DEFAULT_ROOT = Path("/home/data/mashixing/dataset_18tb/icra-2027")
DEFAULT_DATA = Path("/home/data/mashixing/dataset_8tb/iMed/datasets/task2-nvs")
DEFAULT_SCENE = "session_004_scene_2_tool_3"
DEFAULT_FRAME = 428


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--frame", type=int, default=DEFAULT_FRAME)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_plotting() -> str:
    font_dir = Path("/home/data/mashixing/.local/share/fonts/msttcorefonts")
    for name in ("Times.TTF", "Timesbd.TTF", "Timesi.TTF", "Timesbi.TTF"):
        path = font_dir / name
        if path.is_file():
            fontManager.addfont(path)
    try:
        font_path = findfont(
            FontProperties(family="Times New Roman"), fallback_to_default=False
        )
        family = "Times New Roman"
    except ValueError:
        font_path = findfont(FontProperties(family="DejaVu Sans"))
        family = "DejaVu Sans"
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 7.0,
            "axes.titlesize": 7.2,
            "axes.titleweight": "regular",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return font_path


def chw(value: torch.Tensor) -> np.ndarray:
    tensor = value.detach().float().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    return tensor.numpy()


def scalar(value: torch.Tensor) -> np.ndarray:
    array = chw(value)
    if array.shape[0] != 1:
        raise ValueError(f"expected one channel, got {array.shape}")
    return array[0]


def rgb(value: torch.Tensor) -> np.ndarray:
    array = chw(value)
    if array.shape[0] != 3:
        raise ValueError(f"expected RGB, got {array.shape}")
    return np.clip(np.moveaxis(array, 0, -1), 0.0, 1.0)


def robust_range(array: np.ndarray, valid: np.ndarray, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    values = array[valid & np.isfinite(array)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [low, high]).astype(float)
    if hi <= lo:
        hi = lo + 1.0e-8
    return lo, hi


def feature_pca(features: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    channels, height, width = features.shape
    values = np.moveaxis(features, 0, -1).reshape(-1, channels)
    mask = valid.reshape(-1)
    training = values[mask]
    mean = training.mean(axis=0, keepdims=True)
    centered = training - mean
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, np.argsort(eigenvalues)[::-1][:3]]
    for index in range(3):
        anchor = int(np.argmax(np.abs(basis[:, index])))
        if basis[anchor, index] < 0:
            basis[:, index] *= -1
    projected = ((values - mean) @ basis).reshape(height, width, 3)
    output = np.zeros_like(projected, dtype=np.float32)
    ranges: list[list[float]] = []
    for index in range(3):
        lo, hi = robust_range(projected[..., index], valid)
        output[..., index] = np.clip((projected[..., index] - lo) / (hi - lo), 0.0, 1.0)
        ranges.append([lo, hi])
    output[~valid] = 0.0
    return output, {"component_ranges": ranges}


def show_rgb(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(image, interpolation="nearest")
    ax.set_title(title, pad=2.5)
    ax.set_axis_off()


def show_scalar(
    ax: plt.Axes,
    image: np.ndarray,
    title: str,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    valid: np.ndarray | None = None,
    colorbar_label: str = "",
) -> None:
    cmap_object = mpl.colormaps[cmap].copy()
    cmap_object.set_bad("#15191E")
    displayed = np.ma.masked_where(~valid, image) if valid is not None else image
    artist = ax.imshow(
        displayed,
        cmap=cmap_object,
        norm=Normalize(vmin=vmin, vmax=vmax),
        interpolation="nearest",
    )
    ax.set_title(title, pad=2.5)
    ax.set_axis_off()
    color_axis = inset_axes(
        ax,
        width="72%",
        height="5%",
        loc="lower center",
        borderpad=0.55,
    )
    colorbar = ax.figure.colorbar(artist, cax=color_axis, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=5.2, length=1.6, width=0.45, pad=1.0)
    colorbar.outline.set_linewidth(0.4)
    if colorbar_label:
        colorbar.set_label(colorbar_label, fontsize=5.4, labelpad=0.5)


def save_panel(path: Path, array: np.ndarray, *, cmap: str | None = None, vmin: float = 0.0, vmax: float = 1.0, valid: np.ndarray | None = None) -> None:
    if cmap is None:
        image = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    else:
        normalized = np.clip((array - vmin) / max(vmax - vmin, 1.0e-8), 0.0, 1.0)
        image = (mpl.colormaps[cmap](normalized)[..., :3] * 255.0).round().astype(np.uint8)
        if valid is not None:
            image[~valid] = np.asarray([21, 25, 30], dtype=np.uint8)
    Image.fromarray(image).save(path)


def load_exact_sample(data_root: Path, scene_name: str, frame: int, height: int, width: int) -> dict[str, object]:
    """Load exactly one stereo pair without scanning unrelated derived PNGs."""
    from distrisurg.data.scared import (
        read_fixed_source_to_target,
        read_intrinsics,
        read_pose_pairs,
        scale_intrinsics,
    )

    scene = data_root / scene_name
    name = f"frame_{frame:06d}"
    source_path = scene / "endoscope2" / "L" / f"{name}.png"
    target_path = scene / "endoscope1" / "L" / f"{name}.png"
    depth_path = scene / "endoscope2" / "depthL" / f"{name}.npy"
    for path in (source_path, target_path, depth_path, scene / "K.txt"):
        if not path.is_file():
            raise FileNotFoundError(path)

    def load_rgb(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            original_size = (image.height, image.width)
            resized = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous(), original_size

    source_rgb, source_hw = load_rgb(source_path)
    target_rgb, target_hw = load_rgb(target_path)
    depth_array = np.squeeze(np.load(depth_path, allow_pickle=False)).astype(np.float32)
    if depth_array.ndim != 2:
        raise ValueError(f"depth must be 2-D, got {depth_array.shape}")
    valid_array = np.isfinite(depth_array) & (depth_array > 0)
    clean_depth = np.where(valid_array, depth_array, 0.0)
    source_depth = torch.from_numpy(clean_depth).unsqueeze(0).unsqueeze(0)
    source_valid = torch.from_numpy(valid_array.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    if depth_array.shape != (height, width):
        source_depth = F.interpolate(source_depth, (height, width), mode="nearest")
        source_valid = F.interpolate(source_valid, (height, width), mode="nearest")

    intrinsics = read_intrinsics(scene / "K.txt")
    source_k = scale_intrinsics(intrinsics["K2_L"], source_hw, (height, width))
    target_k = scale_intrinsics(intrinsics["K1_L"], target_hw, (height, width))
    pose_pairs = scene / "pose_pairs.txt"
    if pose_pairs.is_file():
        transforms = read_pose_pairs(pose_pairs)
        if frame not in transforms:
            raise KeyError(f"frame {frame} missing from {pose_pairs}")
        transform = transforms[frame]
    else:
        transform = read_fixed_source_to_target(scene / "pose.txt")
    return {
        "source_rgb": source_rgb,
        "source_depth": source_depth[0],
        "source_depth_valid": source_valid[0] > 0.5,
        "target_rgb": target_rgb,
        "source_intrinsics": torch.from_numpy(source_k.copy()),
        "target_intrinsics": torch.from_numpy(target_k.copy()),
        "source_to_target": torch.from_numpy(transform.copy()),
        "source_path": str(source_path),
        "target_path": str(target_path),
    }


def main() -> None:
    args = arguments()
    root = args.project_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from distrisurg.config import load_config
    from distrisurg.models import DistriSurg
    from distrisurg.utils.io import load_checkpoint

    config_path = root / "configs/distrisurg_dataset89.yaml"
    ablation_path = root / "configs/ablations/a6_no_hard_composition.yaml"
    checkpoint = root / "checkpoints/ablations/no_hard_composition/seed_6666/step_0010000.pt"
    for path in (config_path, ablation_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = load_config(
        config_path,
        overrides=[f"data.eval_root={data_root}", "data.min_overlap=0.0"],
        extra_paths=[ablation_path],
    )
    if config.ablation.hard_composition:
        raise RuntimeError("expected the no-hard-composition configuration")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu if needed")
    sample = load_exact_sample(
        data_root, args.scene, args.frame, config.data.height, config.data.width
    )

    model = DistriSurg(config).to(device).eval()
    payload = load_checkpoint(checkpoint, model, device)
    if args.scene in set(payload.get("train_scenes", [])):
        raise RuntimeError("selected validation scene occurs in checkpoint training scenes")
    tensor_names = (
        "source_rgb",
        "source_depth",
        "source_depth_valid",
        "source_intrinsics",
        "target_intrinsics",
        "source_to_target",
    )
    tensors = {
        name: sample[name].unsqueeze(0).to(device, non_blocking=True)
        for name in tensor_names
    }
    with torch.inference_mode():
        reliability = model.reliability(
            tensors["source_rgb"],
            tensors["source_depth"],
            tensors["source_depth_valid"],
        )
        rendered = model.renderer(
            torch.cat((tensors["source_rgb"], reliability["source_features"]), dim=1),
            reliability["depth_mean"],
            reliability["depth_sigma"],
            reliability["render_valid"],
            reliability["confidence"],
            tensors["source_intrinsics"],
            tensors["target_intrinsics"],
            tensors["source_to_target"],
            reliability["normal"],
        )
        prediction = model(*[tensors[name] for name in tensor_names])

    if not torch.allclose(rendered["features"][:, :3], prediction["warped_rgb"], atol=1.0e-6, rtol=0.0):
        raise RuntimeError("direct DSS output disagrees with full-model warped RGB")
    valid = scalar(rendered["valid_mask"]).astype(bool)
    support = scalar(rendered["support"])
    if not np.array_equal(valid, support >= config.dss.min_support):
        raise RuntimeError("valid mask disagrees with the configured support threshold")

    source = rgb(tensors["source_rgb"])
    warped = rgb(rendered["features"][:, :3])
    features = chw(rendered["features"][:, 3:])
    feature_image, feature_metadata = feature_pca(features, valid)
    depth = scalar(rendered["depth"])
    coverage = scalar(rendered["coverage"])
    variance = scalar(rendered["variance"])
    entropy = scalar(rendered["collision_entropy"])
    confidence = scalar(rendered["confidence"])

    depth_range = robust_range(depth, valid)
    support_range = (0.0, robust_range(support, valid)[1])
    variance_range = (0.0, robust_range(variance, valid)[1])
    ranges = {
        "expected_depth_mm": list(depth_range),
        "support": list(support_range),
        "coverage": [0.0, 1.0],
        "depth_variance_mm2": list(variance_range),
        "collision_entropy": [0.0, 1.0],
        "geometry_confidence": [0.0, 1.0],
    }

    font_path = configure_plotting()
    fig, axes = plt.subplots(2, 5, figsize=(7.16, 3.32))
    plt.subplots_adjust(left=0.018, right=0.992, top=0.925, bottom=0.055, wspace=0.075, hspace=0.20)
    titles = [
        r"(a) Source RGB $\mathbf{I}_s$",
        r"(b) Warped RGB $\widetilde{\mathbf{I}}_t$",
        r"(c) Warped features $\widetilde{\mathbf{F}}_t$ (PCA)",
        r"(d) Expected depth $\bar{D}_t$",
        r"(e) Valid mask $M^{\mathrm{dss}}$",
    ]
    show_rgb(axes[0, 0], source, titles[0])
    show_rgb(axes[0, 1], warped, titles[1])
    show_rgb(axes[0, 2], feature_image, titles[2])
    show_scalar(axes[0, 3], depth, titles[3], cmap="cividis", vmin=depth_range[0], vmax=depth_range[1], valid=valid, colorbar_label="mm")
    show_scalar(axes[0, 4], valid.astype(np.float32), titles[4], cmap="gray", vmin=0.0, vmax=1.0)
    show_scalar(axes[1, 0], support, r"(f) Support mass $S$", cmap="viridis", vmin=support_range[0], vmax=support_range[1])
    show_scalar(axes[1, 1], coverage, r"(g) Coverage $C$", cmap="viridis", vmin=0.0, vmax=1.0)
    show_scalar(axes[1, 2], variance, r"(h) Depth variance $V$", cmap="magma", vmin=variance_range[0], vmax=variance_range[1], valid=valid, colorbar_label=r"mm$^2$")
    show_scalar(axes[1, 3], entropy, r"(i) Collision entropy $\mathcal{H}$", cmap="magma", vmin=0.0, vmax=1.0, valid=valid)
    show_scalar(axes[1, 4], confidence, r"(j) Geometry confidence $\kappa$", cmap="viridis", vmin=0.0, vmax=1.0)

    base = output_dir / "dss_outputs_visualization"
    fig.savefig(base.with_suffix(".pdf"), dpi=600, metadata={"Creator": "Python/matplotlib"})
    fig.savefig(base.with_suffix(".svg"), dpi=600, metadata={"Creator": "Python/matplotlib"})
    fig.savefig(base.with_suffix(".png"), dpi=600)
    tiff_path = base.with_suffix(".tiff")
    fig.savefig(tiff_path, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    with Image.open(tiff_path) as image:
        rgb_tiff = image.convert("RGB")
    rgb_tiff.save(tiff_path, compression="tiff_lzw", dpi=(600, 600))

    panels = output_dir / "dss_output_panels"
    panels.mkdir(exist_ok=True)
    save_panel(panels / "source_rgb.png", source)
    save_panel(panels / "warped_rgb.png", warped)
    save_panel(panels / "warped_features_pca.png", feature_image)
    save_panel(panels / "expected_depth.png", depth, cmap="cividis", vmin=depth_range[0], vmax=depth_range[1], valid=valid)
    save_panel(panels / "valid_mask.png", valid.astype(np.float32), cmap="gray")
    save_panel(panels / "support.png", support, cmap="viridis", vmin=support_range[0], vmax=support_range[1])
    save_panel(panels / "coverage.png", coverage, cmap="viridis")
    save_panel(panels / "depth_variance.png", variance, cmap="magma", vmin=variance_range[0], vmax=variance_range[1], valid=valid)
    save_panel(panels / "collision_entropy.png", entropy, cmap="magma", valid=valid)
    save_panel(panels / "geometry_confidence.png", confidence, cmap="viridis")

    np.savez_compressed(
        output_dir / "dss_outputs_raw.npz",
        source_rgb=source.astype(np.float32),
        warped_rgb=warped.astype(np.float32),
        warped_features=features.astype(np.float32),
        warped_features_pca=feature_image.astype(np.float32),
        expected_depth=depth.astype(np.float32),
        support=support.astype(np.float32),
        coverage=coverage.astype(np.float32),
        depth_variance=variance.astype(np.float32),
        collision_entropy=entropy.astype(np.float32),
        geometry_confidence=confidence.astype(np.float32),
        valid_mask=valid.astype(np.uint8),
    )
    manifest = {
        "sample_id": f"{args.scene}/frame_{args.frame:06d}",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "configuration": "no_hard_composition",
        "hard_composition": bool(config.ablation.hard_composition),
        "source_path": sample["source_path"],
        "target_path": sample["target_path"],
        "image_size_hw": [config.data.height, config.data.width],
        "raw_hole_ratio": float(1.0 - valid.mean()),
        "valid_ratio": float(valid.mean()),
        "min_support": float(config.dss.min_support),
        "display_ranges": ranges,
        "feature_pca": feature_metadata,
        "font_file": font_path,
        "image_integrity": {
            "crop": "none",
            "brightness_contrast_gamma": "none",
            "pseudo_color": "scalar maps only; mappings and ranges recorded above",
            "raw_arrays": "dss_outputs_raw.npz",
        },
    }
    (output_dir / "dss_outputs_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
