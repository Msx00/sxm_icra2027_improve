#!/usr/bin/env python3
from __future__ import annotations
import os, subprocess, sys, tempfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.cli import inference_parser
from common.io import load_pair, load_samples, output_path
from common.mask import binary_mask, composite_known

def main() -> None:
    parser = inference_parser("Official big-LaMa zero-shot inpainting adapter")
    parser.add_argument("--repo", default=str(ROOT / "third_party/lama"))
    parser.add_argument("--model", default=str(ROOT / "weights/big-lama"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    repo, model = Path(args.repo).resolve(), Path(args.model).resolve()
    predict = repo / "bin/predict.py"
    if not predict.is_file() or not (model / "config.yaml").is_file():
        raise FileNotFoundError("LaMa code/weights missing; run scripts/setup_lama.sh")
    pending = []
    for sample in load_samples(args.input):
        destination = output_path(Path(args.output), sample, "lama")
        if not destination.exists() or args.overwrite:
            pending.append((sample, destination))
    if not pending: return
    with tempfile.TemporaryDirectory(prefix="lama_") as value:
        temporary = Path(value); result_dir = temporary / "out"
        for index, (sample, _) in enumerate(pending):
            image, mask = load_pair(sample); stem = f"sample_{index:08d}"
            image.save(temporary / f"{stem}.png")
            binary_mask(mask).save(temporary / f"{stem}_mask.png")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(repo) + os.pathsep + environment.get("PYTHONPATH", "")
        subprocess.run([sys.executable, str(predict), f"model.path={model}",
            f"indir={temporary}", f"outdir={result_dir}", f"device={args.device}",
            "dataset.img_suffix=.png"], cwd=repo, check=True, env=environment)
        for index, (sample, destination) in enumerate(pending):
            candidate = result_dir / f"sample_{index:08d}_mask.png"
            if not candidate.is_file():
                matches = sorted(result_dir.glob(f"sample_{index:08d}*.png"))
                if not matches: raise RuntimeError(f"LaMa output missing for {sample.name}")
                candidate = matches[0]
            image, mask = load_pair(sample)
            composite_known(Image.open(candidate), image, mask).save(destination)
            print(destination)

if __name__ == "__main__":
    main()
