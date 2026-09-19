#!/usr/bin/env python3
"""Train DistriSurg on the frozen iMED scene split."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distrisurg.config import load_config
from distrisurg.data import StereoEndoscopyDataset, discover_scenes, read_scene_list
from distrisurg.losses import DistriSurgLoss
from distrisurg.models import DistriSurg
from distrisurg.utils.io import load_checkpoint, write_json_atomic


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/distrisurg_dataset89.yaml"))
    parser.add_argument("--ablation-config", action="append", default=[])
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", default=str(ROOT / "checkpoints/distrisurg"))
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0, help="debug-only dataset cap")
    parser.add_argument(
        "--allow-eval-data-training",
        action="store_true",
        help="explicitly bypass the dataset89 leakage guard (never use for paper results)",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_split(config, allow_eval: bool) -> list[str]:
    train_root = Path(config.data.train_root).expanduser().resolve()
    eval_root = Path(config.data.eval_root).expanduser().resolve()
    scene_file = Path(config.data.train_scenes_file).expanduser().resolve()
    for path in (train_root, eval_root, scene_file):
        if not path.exists():
            raise FileNotFoundError(path)
    train_names = read_scene_list(scene_file)
    eval_names = {path.name for path in discover_scenes(eval_root)}
    overlap = sorted(set(train_names) & eval_names)
    suspicious = [name for name in train_names if name.startswith("endovis_dataset_8_") or name.startswith("endovis_dataset_9_")]
    unsafe = train_root == eval_root or bool(overlap) or bool(suspicious)
    if config.train.forbid_eval_scene_training and unsafe and not allow_eval:
        raise RuntimeError(
            "training/evaluation leakage detected: "
            f"same_root={train_root == eval_root}, overlap={overlap}, suspicious={suspicious}. "
            "Use a clean split; the bypass flag invalidates zero-shot claims."
        )
    return train_names


def move_tensors(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def corrupt_depth(batch: dict, probability: float) -> dict:
    valid_gt = batch["source_depth_valid"].bool()
    depth_gt = batch["source_depth"]
    drop = (torch.rand_like(depth_gt) < probability) & valid_gt
    # Add contiguous missing islands so depth completion is not reduced to
    # independent pixel denoising.
    seeds = (torch.rand_like(depth_gt) < probability * 0.02).float()
    blobs = torch.nn.functional.max_pool2d(
        seeds, kernel_size=9, stride=1, padding=4
    ) > 0
    drop |= blobs & valid_gt
    valid_input = valid_gt & ~drop
    result = dict(batch)
    result["source_depth_gt"] = depth_gt
    result["source_depth_valid_gt"] = valid_gt
    result["source_depth"] = torch.where(valid_input, depth_gt, torch.zeros_like(depth_gt))
    result["source_depth_valid"] = valid_input
    return result


def save_checkpoint(
    output: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config,
    train_scenes: list[str],
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"step_{step:07d}.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
            "train_scenes": train_scenes,
        },
        temporary,
    )
    temporary.replace(path)
    latest = output / "latest.pt"
    latest_tmp = output / "latest.pt.tmp"
    torch.save(torch.load(path, map_location="cpu", weights_only=False), latest_tmp)
    latest_tmp.replace(latest)
    return path


def main() -> None:
    args = arguments()
    config = load_config(args.config, args.set, args.ablation_config)
    set_seed(config.train.seed)
    train_scenes = validate_split(config, args.allow_eval_data_training)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    dataset = StereoEndoscopyDataset(
        config.data.train_root,
        height=config.data.height,
        width=config.data.width,
        scene_names=train_scenes,
    )
    if args.max_samples > 0:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.data.num_workers > 0,
    )
    model = DistriSurg(config).to(device)
    criterion = DistriSurgLoss(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    start_step = 0
    if args.resume:
        payload = load_checkpoint(args.resume, model, device)
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload.get("step", 0))
    use_amp = config.train.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "DistriSurg",
        "train_root": str(Path(config.data.train_root).resolve()),
        "train_scenes_file": str(Path(config.data.train_scenes_file).resolve()),
        "train_scenes": train_scenes,
        "eval_root_forbidden": str(Path(config.data.eval_root).resolve()),
        "zero_shot_eval_scene_reads": 0,
        "config": config.to_dict(),
        "started_at_unix": time.time(),
        "completed": False,
    }
    write_json_atomic(output / "training_manifest.json", manifest)
    log_path = output / "train_log.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = start_step
    micro_step = 0
    while global_step < config.train.steps:
        for raw_batch in loader:
            batch = corrupt_depth(move_tensors(raw_batch, device), config.train.depth_dropout)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.float32,
                enabled=use_amp,
            ):
                prediction = model(
                    batch["source_rgb"],
                    batch["source_depth"],
                    batch["source_depth_valid"],
                    batch["source_intrinsics"],
                    batch["target_intrinsics"],
                    batch["source_to_target"],
                    return_cycle=config.loss.cycle > 0,
                )
                losses = criterion(prediction, batch)
                loss = losses["total"] / config.train.accumulation_steps
            scaler.scale(loss).backward()
            micro_step += 1
            if micro_step % config.train.accumulation_steps:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            values = {
                key: float(value.detach().float().item()) for key, value in losses.items()
            }
            row = {"step": global_step, **values, "time_unix": time.time()}
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            if global_step % config.train.log_every == 0 or global_step == 1:
                print(
                    f"step={global_step:07d} total={values['total']:.5f} "
                    f"hole={values['hole']:.5f} cycle={values['cycle']:.5f} "
                    f"depth={values['depth_nll']:.5f}",
                    flush=True,
                )
            if global_step % config.train.save_every == 0:
                path = save_checkpoint(
                    output, global_step, model, optimizer, config, train_scenes
                )
                print(f"saved {path}", flush=True)
            if global_step >= config.train.steps:
                break
    final = save_checkpoint(output, global_step, model, optimizer, config, train_scenes)
    manifest["completed"] = True
    manifest["completed_at_unix"] = time.time()
    manifest["final_checkpoint"] = str(final)
    write_json_atomic(output / "training_manifest.json", manifest)
    print(f"training complete: {final}")


if __name__ == "__main__":
    main()
