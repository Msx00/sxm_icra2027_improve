#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.cli import inference_parser
from common.io import load_pair, load_samples, output_path
from common.mask import binary_mask, composite_known

def main() -> None:
    parser = inference_parser("Stable Diffusion 1.5 zero-shot inpainting")
    parser.add_argument("--model", default=str(ROOT / "weights/sd15_inpainting"))
    parser.add_argument("--prompt", default="realistic surgical endoscopy image, natural tissue, photorealistic")
    parser.add_argument("--negative-prompt", default="text, watermark, border, cartoon, illustration, instruments")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    from diffusers import StableDiffusionInpaintPipeline
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu only for diagnostics")
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model, torch_dtype=dtype, use_safetensors=True,
        variant="fp16", safety_checker=None, requires_safety_checker=False,
    ).to(args.device)
    pipe.enable_attention_slicing()
    for sample in load_samples(args.input):
        destination = output_path(Path(args.output), sample, "sd15")
        if destination.exists() and not args.overwrite:
            continue
        image, mask = load_pair(sample)
        work_size = (args.size, args.size)
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        generated = pipe(prompt=args.prompt, negative_prompt=args.negative_prompt,
            image=image.resize(work_size, Image.Resampling.LANCZOS),
            mask_image=binary_mask(mask).resize(work_size, Image.Resampling.NEAREST),
            num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
            generator=generator).images[0]
        composite_known(generated, image, mask).save(destination)
        print(destination)

if __name__ == "__main__":
    main()
