#!/usr/bin/env python3
"""Export a traceable single-frame no-hard-composition inference sequence.

The archived validation warp belongs to ``session_004_scene_2_tool_3`` and is
therefore paired with that scene's source RGB, depth, calibration, and pose.
The user-requested ``tool_4`` RGB is exported separately as a training-domain
appearance example; it is never presented as the cause of the ``tool_3`` warp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = Path("/home/data/mashixing/dataset_18tb/icra-2027")
DEFAULT_DATA_ROOT = Path(
    "/home/data/mashixing/dataset_8tb/iMed/datasets/task2-nvs"
)
DEFAULT_REQUESTED_SOURCE = DEFAULT_DATA_ROOT / (
    "session_004_scene_2_tool_4/endoscope2/L/frame_000002.png"
)
DEFAULT_ARCHIVED_WARP = DEFAULT_PROJECT_ROOT / (
    "outputs/ablations_validation_no_post/no_hard_composition/seed_6666/"
    "session_004_scene_2_tool_3/warps/frame_000002.png"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--scene", default="session_004_scene_2_tool_3")
    parser.add_argument("--frame", type=int, default=2)
    parser.add_argument("--requested-source", type=Path, default=DEFAULT_REQUESTED_SOURCE)
    parser.add_argument("--archived-warp", type=Path, default=DEFAULT_ARCHIVED_WARP)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "intermediate_assets")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batched(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.unsqueeze(0).to(device, non_blocking=True)


def rgb_array(value: torch.Tensor) -> np.ndarray:
    tensor = value.detach().float().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"expected CHW RGB tensor, got {tuple(tensor.shape)}")
    return (
        tensor.permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0
    ).round().astype(np.uint8)


def scalar_array(value: torch.Tensor) -> np.ndarray:
    tensor = value.detach().float().cpu()
    while tensor.ndim > 2:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"expected scalar image, got {tuple(tensor.shape)}")
    return tensor.numpy()


def save_rgb(path: Path, value: torch.Tensor) -> np.ndarray:
    array = rgb_array(value)
    Image.fromarray(array, mode="RGB").save(path)
    return array


def save_mask(path: Path, value: torch.Tensor) -> np.ndarray:
    array = scalar_array(value).astype(bool)
    Image.fromarray(array.astype(np.uint8) * 255, mode="L").save(path)
    return array


def colorize(
    path: Path,
    value: torch.Tensor,
    *,
    cmap: str = "magma",
    minimum: float | None = None,
    maximum: float | None = None,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    array = scalar_array(value)
    finite = np.isfinite(array)
    if valid is not None:
        finite &= valid
    values = array[finite]
    if minimum is None:
        minimum = float(np.quantile(values, 0.01)) if values.size else 0.0
    if maximum is None:
        maximum = float(np.quantile(values, 0.99)) if values.size else 1.0
    if maximum <= minimum:
        maximum = minimum + 1.0e-8
    normalized = np.clip((array - minimum) / (maximum - minimum), 0.0, 1.0)
    normalized = np.where(np.isfinite(normalized), normalized, 0.0)
    rgba = matplotlib.colormaps[cmap](normalized)
    rgb = (rgba[..., :3] * 255.0).round().astype(np.uint8)
    if valid is not None:
        rgb[~valid] = np.asarray([16, 20, 26], dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)
    return array.astype(np.float32), float(minimum), float(maximum)


def save_normal(path: Path, value: torch.Tensor) -> np.ndarray:
    tensor = value.detach().float().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = tensor.permute(1, 2, 0).numpy()
    image = (np.clip((array + 1.0) / 2.0, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(image, mode="RGB").save(path)
    return array.astype(np.float32)


def main() -> None:
    args = arguments()
    root = args.project_root.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from distrisurg.config import load_config
    from distrisurg.data import StereoEndoscopyDataset
    from distrisurg.models import DistriSurg
    from distrisurg.utils.io import load_checkpoint

    config = load_config(
        root / "configs/distrisurg_dataset89.yaml",
        overrides=[
            f"data.eval_root={args.data_root.expanduser().resolve()}",
            "data.min_overlap=0.0",
        ],
        extra_paths=[root / "configs/ablations/a6_no_hard_composition.yaml"],
    )
    if config.ablation.hard_composition:
        raise RuntimeError("the requested checkpoint must have hard_composition=false")
    checkpoint = root / (
        "checkpoints/ablations/no_hard_composition/seed_6666/step_0010000.pt"
    )
    requested_source = args.requested_source.expanduser().resolve()
    archived_warp = args.archived_warp.expanduser().resolve()
    for path in (checkpoint, requested_source, archived_warp):
        if not path.is_file():
            raise FileNotFoundError(path)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dataset = StereoEndoscopyDataset(
        config.data.eval_root,
        height=config.data.height,
        width=config.data.width,
        scene_names=[args.scene],
    )
    matches = [
        i
        for i, sample in enumerate(dataset.samples)
        if sample.scene == args.scene and sample.frame_id == args.frame
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one matched frame, found {len(matches)}")
    sample = dataset[matches[0]]

    model = DistriSurg(config).to(device).eval()
    payload = load_checkpoint(checkpoint, model, device)
    train_scenes = set(payload.get("train_scenes", []))
    if args.scene in train_scenes:
        raise RuntimeError(f"held-out visualization scene leaked into training: {args.scene}")

    names = (
        "source_rgb",
        "source_depth",
        "source_depth_valid",
        "source_intrinsics",
        "target_intrinsics",
        "source_to_target",
    )
    tensors = {name: batched(sample[name], device) for name in names}
    with torch.inference_mode():
        prediction = model(
            tensors["source_rgb"],
            tensors["source_depth"],
            tensors["source_depth_valid"],
            tensors["source_intrinsics"],
            tensors["target_intrinsics"],
            tensors["source_to_target"],
        )

    reconstructed = (
        (1.0 - prediction["synthesis_gate"]) * prediction["transport_rgb"]
        + prediction["synthesis_gate"] * prediction["synthesis_rgb"]
    )
    fusion_error = float((reconstructed - prediction["target_rgb"]).abs().max().item())
    if fusion_error > 1.0e-6:
        raise RuntimeError(f"soft-fusion identity failed: {fusion_error:.3e}")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    actual_source = save_rgb(output / "01_matched_source_rgb.png", tensors["source_rgb"])
    warp = save_rgb(output / "02_dss_warp.png", prediction["warped_rgb"])
    transport = save_rgb(output / "09_transport_rgb.png", prediction["transport_rgb"])
    synthesis = save_rgb(output / "10_synthesis_rgb.png", prediction["synthesis_rgb"])
    fused = save_rgb(output / "12_soft_fused_rgb.png", prediction["target_rgb"])
    target = save_rgb(output / "14_target_rgb_reference.png", sample["target_rgb"])

    requested_image = Image.open(requested_source).convert("RGB").resize(
        (config.data.width, config.data.height), Image.Resampling.BILINEAR
    )
    requested_array = np.asarray(requested_image, dtype=np.uint8)
    requested_image.save(output / "00_requested_tool4_train_exemplar.png")

    valid = scalar_array(prediction["completed_valid_mask"]).astype(bool)
    source_valid = scalar_array(tensors["source_depth_valid"]).astype(bool)
    arrays: dict[str, np.ndarray] = {
        "requested_tool4_train_exemplar": requested_array,
        "matched_source_rgb": actual_source,
        "dss_warp": warp,
        "transport_rgb": transport,
        "synthesis_rgb": synthesis,
        "soft_fused_rgb": fused,
        "target_rgb_reference": target,
    }
    display_ranges: dict[str, list[float]] = {}

    for filename, key, cmap, low, high, mask in (
        ("03_source_depth_mean.png", "source_depth_mean", "viridis", None, None, source_valid),
        ("04_source_depth_sigma.png", "source_depth_sigma", "magma", 0.0, None, source_valid),
        ("05_dss_support.png", "support", "magma", 0.0, None, None),
        ("06_dss_variance.png", "render_variance", "magma", 0.0, None, valid),
        ("07_collision_entropy.png", "collision_entropy", "magma", 0.0, 1.0, valid),
        ("08_physics_prior.png", "physics_prior", "magma", 0.0, 1.0, None),
        ("11_synthesis_gate.png", "synthesis_gate", "magma", 0.0, 1.0, None),
        ("13_predicted_risk.png", "risk", "magma", 0.0, 1.0, None),
        ("15_render_depth.png", "render_depth", "viridis", None, None, valid),
        ("16_source_confidence.png", "source_confidence", "magma", 0.0, 1.0, None),
        ("17_render_confidence.png", "render_confidence", "magma", 0.0, 1.0, None),
    ):
        array, vmin, vmax = colorize(
            output / filename,
            prediction[key],
            cmap=cmap,
            minimum=low,
            maximum=high,
            valid=mask,
        )
        arrays[key] = array
        display_ranges[key] = [vmin, vmax]

    arrays["source_normal"] = save_normal(
        output / "18_source_normal.png", prediction["source_normal"]
    )
    arrays["completed_valid_mask"] = save_mask(
        output / "19_completed_valid_mask.png", prediction["completed_valid_mask"]
    ).astype(np.uint8)
    arrays["trusted_mask"] = save_mask(
        output / "20_trusted_mask.png", prediction["trusted_mask"]
    ).astype(np.uint8)

    archived_array = np.asarray(Image.open(archived_warp).convert("RGB"), dtype=np.uint8)
    if archived_array.shape != warp.shape:
        raise ValueError(
            f"archived warp shape {archived_array.shape} != inferred warp {warp.shape}"
        )
    warp_max_error = int(np.abs(archived_array.astype(np.int16) - warp.astype(np.int16)).max())
    if warp_max_error != 0:
        raise RuntimeError(f"inferred warp differs from archived PNG by {warp_max_error} levels")

    np.savez_compressed(output / "intermediates_raw.npz", **arrays)
    manifest = {
        "model": "DistriNVS / no_hard_composition",
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(payload.get("step", 10000)),
        "hard_composition": bool(config.ablation.hard_composition),
        "post_processing": False,
        "matched_sample": f"{args.scene}/frame_{args.frame:06d}",
        "matched_source": str(sample["source_path"]),
        "archived_warp": str(archived_warp),
        "archived_warp_sha256": sha256(archived_warp),
        "requested_tool4_image": str(requested_source),
        "requested_tool4_sha256": sha256(requested_source),
        "requested_tool4_role": (
            "training-domain appearance example only; not paired with the tool_3 warp"
        ),
        "train_scene_contains_requested_tool4": "session_004_scene_2_tool_4" in train_scenes,
        "soft_fusion_identity_max_abs_error": fusion_error,
        "archived_warp_max_u8_error": warp_max_error,
        "display_ranges": display_ranges,
        "gate_quantiles": np.quantile(
            scalar_array(prediction["synthesis_gate"]), [0.0, 0.5, 0.9, 0.99, 1.0]
        ).tolist(),
        "asset_note": (
            "Heatmaps are display-normalized as documented by display_ranges; "
            "raw float arrays are preserved in intermediates_raw.npz."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
