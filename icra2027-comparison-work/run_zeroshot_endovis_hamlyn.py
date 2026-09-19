#!/usr/bin/env python3
"""Prepare and evaluate zero-shot inpainting baselines on EndoVis/Hamlyn."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path("/home/data/mashixing/dataset_18tb/icra2027-diffusion-zero-test-dataset")
DEFAULT_GEOMETRY_ROOT = Path("/home/data/mashixing/dataset_8tb/iMed/comparison/task2-icra/cross_endo_rendering_surpvised")
DEFAULT_OUTPUT = DEFAULT_GEOMETRY_ROOT / "results/zeroshot"
METHOD_REQUIREMENTS = {
    "sd15": ROOT / "weights/sd15_inpainting/model_index.json",
    "lama": ROOT / "weights/big-lama/models/best.ckpt",
    "mat": ROOT / "weights/Places_512_FullData.pkl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenes-file", type=Path, default=None)
    parser.add_argument("--geometry-root", type=Path, default=DEFAULT_GEOMETRY_ROOT)
    parser.add_argument("--prepare-python", default="/home/data/mashixing/miniconda3/envs/foundation_stereo/bin/python3.11")
    parser.add_argument("--methods", nargs="+", default=["available"],
                        choices=["available", "sd15", "lama", "mat"],
                        help="'available' runs each method whose checkpoint is installed")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--min-valid-ratio", type=float, default=0.5,
        help="Keep only frames whose valid_mask ratio is strictly above this value",
    )
    parser.add_argument("--small-hole-max-area", type=int, default=0)
    parser.add_argument("--seam-kernel", type=int, default=1)
    parser.add_argument("--overwrite-warps", action="store_true")
    parser.add_argument("--overwrite-inference", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_scenes(path: Path) -> list[str]:
    scenes = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        name = raw.split("#", 1)[0].strip()
        if not name:
            continue
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid scene name in {path}: {name!r}")
        scenes.append(name)
    if not scenes or len(scenes) != len(set(scenes)):
        raise RuntimeError(f"Scene list is empty or contains duplicates: {path}")
    return scenes


def resolve_scene(data: Path, name: str) -> Path:
    candidates = (data / name, data / "endovis" / name, data / "hamlyn" / name)
    matches = [path for path in candidates if path.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one directory for {name!r}; checked: "
            + ", ".join(str(path) for path in candidates)
        )
    return matches[0]


def reusable_warp(path: Path, scene: Path, args: argparse.Namespace) -> bool:
    if args.overwrite_warps or not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        recorded_scene = Path(payload["scene"]).expanduser().resolve()
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        payload.get("completed") is True
        and recorded_scene == scene.resolve()
        and payload.get("height") == args.height
        and payload.get("width") == args.width
        and payload.get("seam_kernel") == args.seam_kernel
        and payload.get("small_hole_max_area") == args.small_hole_max_area
        and payload.get("min_valid_ratio") == args.min_valid_ratio
        and payload.get("mask_strategy")
        == "valid_mask->fill_small_enclosed_holes->invert->small_dilation"
    )


def choose_methods(requested: list[str]) -> list[str]:
    if "available" in requested:
        if len(requested) != 1:
            raise ValueError("Use --methods available alone, or list explicit methods")
        selected = [name for name, path in METHOD_REQUIREMENTS.items() if path.is_file()]
        missing = [name for name, path in METHOD_REQUIREMENTS.items() if not path.is_file()]
        if missing:
            print("Skipping methods without checkpoints: " + ", ".join(missing))
    else:
        selected = list(dict.fromkeys(requested))
        missing = [name for name in selected if not METHOD_REQUIREMENTS[name].is_file()]
        if missing:
            details = ", ".join(f"{name}: {METHOD_REQUIREMENTS[name]}" for name in missing)
            raise FileNotFoundError(f"Requested checkpoint(s) missing: {details}")
    if not selected:
        raise RuntimeError("No runnable comparison method was found")
    return selected


def run(command: list[object], env: dict[str, str], dry_run: bool) -> None:
    print("$ " + " ".join(shlex.quote(str(value)) for value in command), flush=True)
    if not dry_run:
        subprocess.run([str(value) for value in command], cwd=ROOT, env=env, check=True)


def target_for(scene: Path, frame_id: int) -> Path:
    target = scene / "endoscope1/L" / f"frame_{frame_id:06d}.png"
    if not target.is_file():
        raise FileNotFoundError(f"Target image missing: {target}")
    return target.resolve()


def manifest_rows(scene_name: str, scene: Path, warp_root: Path) -> list[dict[str, str]]:
    path = warp_root / "warp_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("completed"):
        raise RuntimeError(f"Incomplete warp manifest: {path}")
    rows = []
    for frame in payload.get("frames", []):
        frame_id = int(frame["frame_id"])
        image, mask = Path(frame["warped_rgb"]), Path(frame["inpaint_mask"])
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"Prepared input missing for {scene_name}/frame_{frame_id:06d}")
        rows.append({"scene": scene_name, "name": f"frame_{frame_id:06d}",
                     "warped_rgb": str(image.resolve()), "inpaint_mask": str(mask.resolve()),
                     "target_rgb": str(target_for(scene, frame_id))})
    if not rows:
        raise RuntimeError(f"No prepared frames in {path}")
    return rows


def main() -> None:
    args = parse_args()
    data_root, output_root = args.data_root.expanduser().resolve(), args.output_root.expanduser().resolve()
    geometry_root = args.geometry_root.expanduser().resolve()
    scenes_file = (args.scenes_file or data_root / "zeroshot_scenes.txt").expanduser().resolve()
    prepare_script = geometry_root / "prepare_scene.py"
    for path in (data_root, geometry_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in (scenes_file, prepare_script, ROOT / "run_all.py", ROOT / "evaluate_all.py"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.height <= 0 or args.width <= 0 or args.height % 8 or args.width % 8:
        raise ValueError("height and width must be positive multiples of 8")
    if not 0.0 <= args.min_valid_ratio < 1.0:
        raise ValueError("min-valid-ratio must satisfy 0 <= value < 1")

    scenes, methods = read_scenes(scenes_file), choose_methods(args.methods)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    env["ICRA_METHOD_FIRST_OUTPUT"] = "1"
    comparison_root = output_root / "comparison_methods"
    predictions_root = comparison_root / "predictions"
    manifest_path = comparison_root / "zeroshot_inputs.jsonl"
    metrics_path = comparison_root / "metrics/metrics.csv"
    print(f"Scenes: {len(scenes)}\nMethods: {', '.join(methods)}\nGPU: {args.gpu_id}\nOutputs: {output_root}")

    rows: list[dict[str, str]] = []
    for index, scene_name in enumerate(scenes, 1):
        scene = resolve_scene(data_root, scene_name)
        warp_root = output_root / scene_name / "warps"
        pose_exists = (scene / "pose_pairs.txt").is_file() or (scene / "pose.txt").is_file()
        required = [scene / "K.txt", scene / "endoscope1/L",
                    scene / "endoscope2/L", scene / "endoscope2/depthL"]
        missing = [str(path) for path in required if not path.exists()]
        if not pose_exists:
            missing.append(str(scene / "pose_pairs.txt|pose.txt"))
        if missing:
            raise FileNotFoundError(f"Incomplete scene {scene_name}: {missing}")
        print(f"\n[{index}/{len(scenes)}] Prepare {scene_name}", flush=True)
        warp_manifest = warp_root / "warp_manifest.json"
        if reusable_warp(warp_manifest, scene, args):
            print(f"Reusing corrected warp: {warp_root}")
        else:
            command: list[object] = [args.prepare_python, prepare_script, "--scene", scene,
                "--output", warp_root, "--height", args.height, "--width", args.width,
                "--confidence-threshold", 0.2, "--close-kernel", 3,
                "--seam-kernel", args.seam_kernel,
                "--small-hole-max-area", args.small_hole_max_area,
                "--min-valid-ratio", args.min_valid_ratio]
            if args.max_frames > 0:
                command += ["--max-frames", args.max_frames]
            if warp_manifest.exists() or args.overwrite_warps:
                command.append("--overwrite")
            run(command, env, args.dry_run)
        if not args.dry_run:
            rows.extend(manifest_rows(scene_name, scene, warp_root))

    if not args.dry_run:
        comparison_root.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        temporary.replace(manifest_path)
        config = {"protocol": "zero-shot; no training or checkpoint selection on EndoVis/Hamlyn",
                  "data_root": str(data_root), "scenes_file": str(scenes_file), "scenes": scenes,
                  "methods": methods, "gpu_id": args.gpu_id, "frames": len(rows),
                  "height": args.height, "width": args.width,
                  "min_valid_ratio": args.min_valid_ratio,
                  "small_hole_max_area": args.small_hole_max_area,
                  "seam_kernel": args.seam_kernel}
        (comparison_root / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    inference: list[object] = [sys.executable, ROOT / "run_all.py", "--input", manifest_path,
                               "--output", predictions_root, "--methods", *methods]
    if args.overwrite_inference:
        inference.append("--overwrite")
    run(inference, env, args.dry_run)
    if not args.skip_evaluation:
        evaluation: list[object] = ["conda", "run", "--no-capture-output", "-n", "icra2027-inpaint",
            "python", ROOT / "evaluate_all.py", "--input", manifest_path, "--results", predictions_root,
            "--methods", *methods, "--output", metrics_path, "--method-first"]
        if args.lpips:
            evaluation.append("--lpips")
        run(evaluation, env, args.dry_run)
    print(f"\n{'Dry run validated' if args.dry_run else 'Completed'}: {comparison_root}")


if __name__ == "__main__":
    main()
