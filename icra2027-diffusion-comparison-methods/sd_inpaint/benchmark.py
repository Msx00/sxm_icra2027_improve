#!/usr/bin/env python3
"""Benchmark SD1.5/LDM inpainting latency on a JSONL pair manifest."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io import load_pair, load_samples
from common.mask import binary_mask


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(ROOT / "weights/sd15_inpainting"))
    parser.add_argument(
        "--prompt",
        default="realistic surgical endoscopy image, natural tissue, photorealistic",
    )
    parser.add_argument(
        "--negative-prompt",
        default="text, watermark, border, cartoon, illustration, instruments",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--expected-count", type=int, default=275)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Table-II latency benchmark")

    from diffusers import StableDiffusionInpaintPipeline

    samples = load_samples(args.input)
    if len(samples) != args.expected_count:
        raise RuntimeError(f"expected {args.expected_count} samples, got {len(samples)}")
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    ).to(args.device)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    def prepare(sample):
        image, mask = load_pair(sample)
        work_size = (args.size, args.size)
        return (
            image.resize(work_size, Image.Resampling.LANCZOS),
            binary_mask(mask).resize(work_size, Image.Resampling.NEAREST),
        )

    def infer(image, mask, seed):
        generator = torch.Generator(device=args.device).manual_seed(seed)
        return pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

    warm_image, warm_mask = prepare(samples[0])
    for _ in range(args.warmup):
        infer(warm_image, warm_mask, args.seed)
    torch.cuda.synchronize()

    records = []
    for ordinal, sample in enumerate(samples):
        image, mask = prepare(sample)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = infer(image, mask, args.seed)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        records.append({
            "sample_id": f"{sample.scene}/{sample.name}",
            "seconds": seconds,
        })
        if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(samples):
            print(f"[{ordinal + 1:03d}/{len(samples)}] {seconds:.6f} s", flush=True)
        del result

    values = [row["seconds"] for row in records]
    payload = {
        "method": "LDM",
        "protocol": "single-frame GPU model latency; model loaded once; disk I/O excluded",
        "gpu": torch.cuda.get_device_name(),
        "precision": "float16",
        "batch_size": 1,
        "warmup_frames": args.warmup,
        "count": len(values),
        "mean_seconds": statistics.fmean(values),
        "sample_sd_seconds": statistics.stdev(values),
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "input": str(Path(args.input).expanduser().resolve()),
        "model": str(Path(args.model).expanduser().resolve()),
        "frames": records,
    }
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(
        f"LDM: {payload['mean_seconds']:.6f} +- "
        f"{payload['sample_sd_seconds']:.6f} s/frame"
    )


if __name__ == "__main__":
    main()
