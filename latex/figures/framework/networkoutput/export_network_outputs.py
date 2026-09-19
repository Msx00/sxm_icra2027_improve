#!/usr/bin/env python3
"""Run one no-hard-composition sample and export six network outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = Path("/home/data/mashixing/dataset_18tb/icra-2027")
DATA_ROOT = Path("/home/data/mashixing/dataset_8tb/iMed/datasets/task2-nvs")
SOURCE_IMAGE = DATA_ROOT / (
    "session_004_scene_2_tool_3/endoscope2/L/frame_000428.png"
)
CHECKPOINT = PROJECT_ROOT / (
    "checkpoints/ablations/no_hard_composition/seed_6666/step_0010000.pt"
)
OUTPUT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--source-image", type=Path, default=SOURCE_IMAGE)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def add_batch(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.unsqueeze(0).to(device, non_blocking=True)


def chw_rgb(value: torch.Tensor) -> np.ndarray:
    value = value.detach().float().cpu()
    if value.ndim == 4:
        value = value[0]
    return value.permute(1, 2, 0).clamp(0, 1).numpy()


def hw_scalar(value: torch.Tensor) -> np.ndarray:
    value = value.detach().float().cpu()
    while value.ndim > 2:
        value = value[0]
    return value.numpy()


def save_rgb(path: Path, value: torch.Tensor) -> None:
    array = np.rint(chw_rgb(value) * 255).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def load_rgb(path: Path, height: int, width: int) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_depth(path: Path, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    array = np.squeeze(np.load(path, allow_pickle=False)).astype(np.float32)
    valid = np.isfinite(array) & (array > 0)
    clean = np.where(valid, array, 0.0)
    depth = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0)
    mask = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    if array.shape != (height, width):
        depth = F.interpolate(depth, (height, width), mode="nearest")
        mask = F.interpolate(mask, (height, width), mode="nearest")
    return depth[0], mask[0] > 0.5


def colorize_depth(
    path: Path,
    depth: np.ndarray,
    vmin: float,
    vmax: float,
    valid: np.ndarray,
) -> None:
    normalized = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
    rgb = matplotlib.colormaps["viridis"](normalized)[..., :3]
    rgb[~valid] = 0
    Image.fromarray(np.rint(rgb * 255).astype(np.uint8), mode="RGB").save(path)


def main() -> None:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    source_image = args.source_image.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (source_image, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from distrisurg.config import load_config
    from distrisurg.data.scared import (
        read_fixed_source_to_target,
        read_intrinsics,
        read_pose_pairs,
        scale_intrinsics,
    )
    from distrisurg.models import DistriSurg
    from distrisurg.utils.io import load_checkpoint

    scene = source_image.parents[2].name
    frame_id = int(source_image.stem.removeprefix("frame_"))
    config = load_config(
        root / "configs/train.yaml",
        overrides=[f"data.eval_root={data_root}", "data.min_overlap=0.0"],
    )
    if config.ablation.hard_composition:
        raise RuntimeError("hard composition must be disabled")

    scene_root = source_image.parents[2]
    target_image = scene_root / "endoscope1" / "L" / source_image.name
    source_depth_path = scene_root / "endoscope2" / "depthL" / f"frame_{frame_id:06d}.npy"
    for path in (target_image, source_depth_path, scene_root / "K.txt"):
        if not path.is_file():
            raise FileNotFoundError(path)
    with Image.open(source_image) as image:
        source_hw = (image.height, image.width)
    with Image.open(target_image) as image:
        target_hw = (image.height, image.width)
    intrinsics = read_intrinsics(scene_root / "K.txt")
    pose_pairs = scene_root / "pose_pairs.txt"
    if pose_pairs.is_file():
        poses = read_pose_pairs(pose_pairs)
        if frame_id not in poses:
            raise KeyError(f"frame {frame_id} missing from {pose_pairs}")
        source_to_target = poses[frame_id]
    else:
        source_to_target = read_fixed_source_to_target(scene_root / "pose.txt")
    source_depth, source_depth_valid = load_depth(
        source_depth_path, config.data.height, config.data.width
    )
    sample = {
        "source_rgb": load_rgb(source_image, config.data.height, config.data.width),
        "source_depth": source_depth,
        "source_depth_valid": source_depth_valid,
        "source_intrinsics": torch.from_numpy(
            scale_intrinsics(
                intrinsics["K2_L"], source_hw, (config.data.height, config.data.width)
            ).copy()
        ),
        "target_intrinsics": torch.from_numpy(
            scale_intrinsics(
                intrinsics["K1_L"], target_hw, (config.data.height, config.data.width)
            ).copy()
        ),
        "source_to_target": torch.from_numpy(source_to_target.copy()),
        "source_path": str(source_image),
        "target_path": str(target_image),
    }

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = DistriSurg(config).to(device).eval()
    payload = load_checkpoint(checkpoint, model, device)

    synthesis_raw: list[torch.Tensor] = []
    hook = model.synthesis.register_forward_hook(
        lambda _module, _inputs, result: synthesis_raw.append(result)
    )
    names = (
        "source_rgb",
        "source_depth",
        "source_depth_valid",
        "source_intrinsics",
        "target_intrinsics",
        "source_to_target",
    )
    tensors = {name: add_batch(sample[name], device) for name in names}
    with torch.inference_mode():
        prediction = model(*(tensors[name] for name in names))
    hook.remove()
    if len(synthesis_raw) != 1:
        raise RuntimeError(f"expected one synthesis output, found {len(synthesis_raw)}")

    raw = synthesis_raw[0]
    synthesis_depth = (F.softplus(raw[:, 3:4]) + 0.05) * prediction["depth_scale"]
    render_depth = prediction["render_depth"]
    fused_depth = prediction["target_depth"]
    gate = prediction["synthesis_gate"]
    expected_rgb = (1 - gate) * prediction["transport_rgb"] + gate * prediction["synthesis_rgb"]
    expected_depth = (1 - gate) * render_depth + gate * synthesis_depth
    rgb_error = float((expected_rgb - prediction["target_rgb"]).abs().max())
    depth_error = float((expected_depth - fused_depth).abs().max())
    if max(rgb_error, depth_error) > 1e-5:
        raise RuntimeError(
            f"soft-fusion check failed: rgb={rgb_error:.3e}, depth={depth_error:.3e}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_outputs = {
        "01_I_t_syn.png": prediction["synthesis_rgb"],
        "02_I_t_trans.png": prediction["transport_rgb"],
        "03_I_t_hat.png": prediction["target_rgb"],
    }
    for filename, tensor in rgb_outputs.items():
        save_rgb(output_dir / filename, tensor)

    depth_outputs = {
        "04_D_t_hat.png": fused_depth,
        "05_D_t_bar.png": render_depth,
        "06_D_t_syn.png": synthesis_depth,
    }
    depth_arrays = {name: hw_scalar(tensor) for name, tensor in depth_outputs.items()}
    valid_masks = {
        name: np.isfinite(array) & (array > 0)
        for name, array in depth_arrays.items()
    }
    pooled = np.concatenate(
        [array[valid_masks[name]] for name, array in depth_arrays.items()]
    )
    vmin, vmax = np.quantile(pooled, [0.01, 0.99]).astype(float)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    for filename, array in depth_arrays.items():
        colorize_depth(output_dir / filename, array, vmin, vmax, valid_masks[filename])
        np.save(output_dir / filename.replace(".png", ".npy"), array.astype(np.float32))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
    })
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.8), constrained_layout=True)
    rgb_panels = [
        (prediction["synthesis_rgb"], r"$\mathbf{I}_t^{\mathrm{syn}}$"),
        (prediction["transport_rgb"], r"$\mathbf{I}_t^{\mathrm{trans}}$"),
        (prediction["target_rgb"], r"$\widehat{\mathbf{I}}_t$"),
    ]
    for axis, (tensor, title) in zip(axes[0], rgb_panels):
        axis.imshow(chw_rgb(tensor))
        axis.set_title(title)
        axis.axis("off")
    depth_panels = [
        (depth_arrays["04_D_t_hat.png"], valid_masks["04_D_t_hat.png"], r"$\widehat{\mathbf{D}}_t$"),
        (depth_arrays["05_D_t_bar.png"], valid_masks["05_D_t_bar.png"], r"$\bar{D}_t$"),
        (depth_arrays["06_D_t_syn.png"], valid_masks["06_D_t_syn.png"], r"$\mathbf{D}_t^{\mathrm{syn}}$"),
    ]
    depth_cmap = matplotlib.colormaps["viridis"].copy()
    depth_cmap.set_bad("black")
    image = None
    for axis, (array, valid, title) in zip(axes[1], depth_panels):
        image = axis.imshow(
            np.ma.masked_where(~valid, array), cmap=depth_cmap, vmin=vmin, vmax=vmax
        )
        axis.set_facecolor("black")
        axis.set_title(title)
        axis.axis("off")
    colorbar = fig.colorbar(image, ax=axes[1].tolist(), fraction=0.022, pad=0.015)
    colorbar.set_label("Depth (mm)")
    fig.savefig(output_dir / "network_outputs_panel.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "network_outputs_panel.pdf", bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "scene": scene,
        "frame_id": frame_id,
        "source_path": str(sample["source_path"]),
        "target_path": str(sample["target_path"]),
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(payload.get("step", 10000)),
        "hard_composition": bool(config.ablation.hard_composition),
        "post_processing": False,
        "depth_display_range_mm": [vmin, vmax],
        "soft_fusion_rgb_max_abs_error": rgb_error,
        "soft_fusion_depth_max_abs_error": depth_error,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
