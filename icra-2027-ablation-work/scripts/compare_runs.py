#!/usr/bin/env python3
"""Create a compact paper table from completed DistriSurg/ablation runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = (
    "psnr",
    "ssim",
    "hole_psnr",
    "hole_ssim",
    "seam_psnr",
    "seam_ssim",
    "known_drift",
    "risk_ece",
    "risk_aurc",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=Path("comparison_table"))
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.labels and len(args.labels) != len(args.runs):
        raise ValueError("--labels must match the number of runs")
    rows = []
    for index, run in enumerate(args.runs):
        metrics_path = run / "metrics.json"
        manifest_path = run / "run_manifest.json"
        if not metrics_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"incomplete run: {run}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["summary"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        label = args.labels[index] if args.labels else f"{manifest['method']}:{run.name}"
        row = {"method": label, "frames": manifest.get("selected_count", 0)}
        for key in METRICS:
            value = metrics.get(key, {})
            row[key] = value.get("mean", float("nan"))
            row[f"{key}_ci95_low"] = value.get("ci95_low", float("nan"))
            row[f"{key}_ci95_high"] = value.get("ci95_high", float("nan"))
        rows.append(row)
    csv_path = args.output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    headers = ["Method", "Frames", "PSNR", "SSIM", "Hole PSNR", "Hole SSIM", "Seam PSNR", "Known drift", "Risk AURC"]
    keys = ["method", "frames", "psnr", "ssim", "hole_psnr", "hole_ssim", "seam_psnr", "known_drift", "risk_aurc"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for row in rows:
        cells = []
        for key in keys:
            value = row[key]
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path} and {markdown_path}")


if __name__ == "__main__":
    main()

