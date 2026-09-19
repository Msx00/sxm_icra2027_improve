#!/usr/bin/env python3
"""Export exact same-frame DSS and soft-fusion evidence for Figure 3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VALIDATION_ROOT = Path(
    "/home/data/mashixing/dataset_8tb/iMed/datasets/task2-nvs"
)
EXPECTED_RAW_OVERLAP = 0.4231201112270355


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=SCRIPT_DIR.parents[2],
        help="DistriNVS repository root.",
    )
    parser.add_argument("--scene", default="session_004_scene_2_tool_3")
    parser.add_argument("--frame", type=int, default=2)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_VALIDATION_ROOT,
        help="Prepared iMED stereo-pair validation root.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "dss_evidence_data.npz")
    return parser.parse_args()


def batched_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.unsqueeze(0).to(device, non_blocking=True)


def main() -> None:
    args = arguments()
    root = args.project_root.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from distrisurg.config import load_config
    from distrisurg.data import StereoEndoscopyDataset
    from distrisurg.models import DistriSurg
    from distrisurg.utils.io import load_checkpoint

    config_path = root / "configs" / "train.yaml"
    checkpoint_path = (
        root
        / "checkpoints"
        / "ablations"
        / "no_hard_composition"
        / "seed_6666"
        / "step_0010000.pt"
    )
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    config = load_config(
        config_path,
        overrides=[f"data.eval_root={data_root}", "data.min_overlap=0.0"],
    )
    if config.ablation.hard_composition:
        raise ValueError("Figure 3 must use the no-hard/soft-fusion configuration")

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
        index
        for index, sample in enumerate(dataset.samples)
        if sample.scene == args.scene and sample.frame_id == args.frame
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one sample for {args.scene}/frame_{args.frame:06d}, got {len(matches)}"
        )
    sample = dataset[matches[0]]

    model = DistriSurg(config).to(device).eval()
    payload = load_checkpoint(checkpoint_path, model, device)
    train_scenes = set(payload.get("train_scenes", []))
    if args.scene in train_scenes:
        raise RuntimeError(f"evaluation scene appears in checkpoint training scenes: {args.scene}")

    tensor_names = (
        "source_rgb",
        "source_depth",
        "source_depth_valid",
        "source_intrinsics",
        "target_intrinsics",
        "source_to_target",
    )
    values = {name: batched_tensor(sample[name], device) for name in tensor_names}
    with torch.inference_mode():
        raw_render = model.renderer_only(
            values["source_rgb"],
            values["source_depth"],
            values["source_depth_valid"],
            values["source_intrinsics"],
            values["target_intrinsics"],
            values["source_to_target"],
        )
        prediction = model(
            values["source_rgb"],
            values["source_depth"],
            values["source_depth_valid"],
            values["source_intrinsics"],
            values["target_intrinsics"],
            values["source_to_target"],
        )

    raw_overlap = float(raw_render["valid_mask"].float().mean().item())
    if not np.isclose(raw_overlap, EXPECTED_RAW_OVERLAP, atol=1.0e-6, rtol=0.0):
        raise ValueError(
            f"raw overlap {raw_overlap:.9f} differs from archived validation "
            f"value {EXPECTED_RAW_OVERLAP:.9f}"
        )
    support = prediction["support"][0, 0].detach().float().cpu().numpy()
    valid = prediction["completed_valid_mask"][0, 0].detach().bool().cpu().numpy()
    gate = prediction["synthesis_gate"][0, 0].detach().float().cpu().numpy()
    raw_gate = prediction["raw_synthesis_gate"][0, 0].detach().float().cpu().numpy()
    warp = prediction["warped_rgb"][0].detach().float().cpu().permute(1, 2, 0).numpy()

    expected_shape = (config.data.height, config.data.width)
    if support.shape != expected_shape or valid.shape != expected_shape or gate.shape != expected_shape:
        raise ValueError(
            f"unexpected shapes: support={support.shape}, valid={valid.shape}, gate={gate.shape}"
        )
    if not np.isfinite(support).all() or not np.isfinite(gate).all():
        raise ValueError("support and gate must be finite")
    expected_valid = support >= config.dss.min_support
    if not np.array_equal(valid, expected_valid):
        mismatch = float(np.mean(valid != expected_valid))
        raise ValueError(f"validity disagrees with S >= S_min at {mismatch:.6%} of pixels")
    if float(gate.min()) < -1.0e-6 or float(gate.max()) > 1.0 + 1.0e-6:
        raise ValueError(f"gate outside [0, 1]: [{gate.min()}, {gate.max()}]")
    if not np.allclose(gate, raw_gate, atol=1.0e-7, rtol=0.0):
        raise ValueError("no-hard output gate differs from the router's effective gate")

    transport = prediction["transport_rgb"]
    synthesis = prediction["synthesis_rgb"]
    reconstructed = (1.0 - prediction["synthesis_gate"]) * transport + prediction[
        "synthesis_gate"
    ] * synthesis
    fusion_error = float((reconstructed - prediction["target_rgb"]).abs().max().item())
    if fusion_error > 1.0e-6:
        raise ValueError(f"soft-fusion identity failed: max error={fusion_error:.3e}")

    warp_u8 = (np.clip(warp, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        warped_rgb=warp_u8,
        support=support.astype(np.float32),
        completed_valid_mask=valid.astype(np.uint8),
        synthesis_gate=gate.astype(np.float32),
        sample_id=np.asarray(f"{args.scene}/frame_{args.frame:06d}"),
        checkpoint_tag=np.asarray(
            "ablations/no_hard_composition/seed_6666/step_0010000.pt"
        ),
        hard_composition=np.asarray(False),
        min_support=np.asarray(config.dss.min_support, dtype=np.float32),
        raw_overlap=np.asarray(raw_overlap, dtype=np.float32),
        fusion_identity_max_error=np.asarray(fusion_error, dtype=np.float32),
    )
    quantiles = np.quantile(gate, [0.0, 0.5, 0.9, 0.99, 1.0])
    print(f"wrote {args.output}")
    print(f"sample={args.scene}/frame_{args.frame:06d}")
    print(f"raw_overlap={raw_overlap:.6f}; raw_hole_ratio={1.0 - raw_overlap:.6f}")
    print(f"valid_ratio={valid.mean():.6f}; gate_quantiles={quantiles.tolist()}")
    print(f"soft_fusion_max_error={fusion_error:.3e}")


if __name__ == "__main__":
    main()
