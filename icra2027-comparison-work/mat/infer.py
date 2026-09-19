#!/usr/bin/env python3
from __future__ import annotations
import subprocess, sys, tempfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.cli import inference_parser
from common.io import load_pair, load_samples, output_path
from common.mask import binary_mask, composite_known

def main() -> None:
    parser = inference_parser("Official MAT Places2 inpainting adapter")
    parser.add_argument("--repo", default=str(ROOT / "third_party/MAT"))
    parser.add_argument("--model", default=str(ROOT / "weights/Places_512_FullData.pkl"))
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()
    repo, model = Path(args.repo).resolve(), Path(args.model).resolve()
    generator = repo / "generate_image.py"
    if not generator.is_file() or not model.is_file():
        raise FileNotFoundError("MAT code/weights missing; run scripts/setup_mat.sh")
    pending=[]
    for sample in load_samples(args.input):
        destination=output_path(Path(args.output),sample,"mat")
        if not destination.exists() or args.overwrite: pending.append((sample,destination))
    if not pending: return
    with tempfile.TemporaryDirectory(prefix="mat_") as value:
        temporary=Path(value); image_dir,mask_dir,out_dir=temporary/"images",temporary/"masks",temporary/"out"
        image_dir.mkdir(); mask_dir.mkdir(); out_dir.mkdir(); size=(args.resolution,args.resolution)
        for index,(sample,_) in enumerate(pending):
            image,mask=load_pair(sample); name=f"sample_{index:08d}.png"
            image.resize(size,Image.Resampling.LANCZOS).save(image_dir/name)
            binary_mask(mask).resize(size,Image.Resampling.NEAREST).save(mask_dir/name)
        subprocess.run([sys.executable,str(generator),"--network",str(model),"--dpath",str(image_dir),
            "--mpath",str(mask_dir),"--resolution",str(args.resolution),"--outdir",str(out_dir)],cwd=repo,check=True)
        for index,(sample,destination) in enumerate(pending):
            candidate=out_dir/f"sample_{index:08d}.png"
            if not candidate.is_file(): raise RuntimeError(f"MAT output missing for {sample.name}")
            image,mask=load_pair(sample); composite_known(Image.open(candidate),image,mask).save(destination)
            print(destination)

if __name__ == "__main__":
    main()
