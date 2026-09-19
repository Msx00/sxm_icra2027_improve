#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common.io import load_samples
from common.metrics import load_mask, load_rgb, psnr, ssim

def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline outputs against manifest targets")
    parser.add_argument("--input", required=True, help="same input directory/JSONL used for inference")
    parser.add_argument("--results", default="results")
    parser.add_argument("--methods", nargs="+", default=["sd15", "lama", "mat"])
    parser.add_argument("--output", default="metrics.csv")
    parser.add_argument("--lpips", action="store_true", help="also compute AlexNet LPIPS (GPU if available)")
    parser.add_argument("--method-first", action="store_true",
                        help="read results/<method>/<scene>/<frame>.png")
    args = parser.parse_args()
    lpips_model = None
    if args.lpips:
        import torch, lpips
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_model = lpips.LPIPS(net="alex", spatial=True).to(device).eval()
    rows = []
    for sample in load_samples(args.input):
        if sample.target is None:
            raise ValueError(f"No target_rgb/target/ground_truth for {sample.name}")
        target = load_rgb(sample.target)
        size = (target.shape[1], target.shape[0])
        mask = load_mask(sample.mask, size)
        for method in args.methods:
            if args.method_first:
                result = Path(args.results) / method / sample.scene / f"{sample.name}.png"
            else:
                result = Path(args.results) / sample.scene / method / f"{sample.name}.png"
            if not result.is_file():
                print(f"warning: missing {result}", file=sys.stderr); continue
            prediction = load_rgb(result, size)
            row = {"scene":sample.scene, "frame":sample.name, "method":method,
                "psnr":psnr(prediction,target), "ssim":ssim(prediction,target),
                "psnr_hole":psnr(prediction,target,mask), "ssim_hole":ssim(prediction,target,mask),
                "hole_fraction":float(mask.mean())}
            if lpips_model is not None:
                import torch
                p = torch.from_numpy(prediction).permute(2,0,1).unsqueeze(0).to(device)*2-1
                t = torch.from_numpy(target).permute(2,0,1).unsqueeze(0).to(device)*2-1
                with torch.no_grad(): spatial = lpips_model(p,t).squeeze().detach().cpu().numpy()
                if spatial.shape != mask.shape:
                    spatial = np.asarray(Image.fromarray(spatial.astype(np.float32), mode="F").resize(size))
                row["lpips"] = float(spatial.mean())
                row["lpips_hole"] = float(spatial[mask].mean()) if np.any(mask) else float("nan")
            rows.append(row)
    if not rows: raise RuntimeError("No predictions evaluated")
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for method in args.methods:
        selected = [row for row in rows if row["method"] == method]
        if selected:
            keys = ("psnr","ssim","psnr_hole","ssim_hole","lpips","lpips_hole")
            summary[method] = {key:float(np.nanmean([row[key] for row in selected]))
                               for key in keys if key in selected[0]}
    destination.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(destination); print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
