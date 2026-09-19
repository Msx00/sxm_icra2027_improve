#!/usr/bin/env python3
"""Merge disjoint completed inference runs and recompute aggregate metrics."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distrisurg.metrics import scene_bootstrap_summary
from distrisurg.utils.io import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("parts", type=Path, nargs="+")
    args = parser.parse_args()

    manifests = []
    rows = []
    seen_samples: set[str] = set()
    seen_scenes: set[str] = set()
    args.output.mkdir(parents=True, exist_ok=True)

    for part in args.parts:
        manifest = json.loads((part / "run_manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((part / "metrics.json").read_text(encoding="utf-8"))
        if not manifest.get("completed"):
            raise RuntimeError(f"incomplete part: {part}")
        part_scenes = set(manifest["scenes"])
        overlap = seen_scenes & part_scenes
        if overlap:
            raise RuntimeError(f"duplicate scenes: {sorted(overlap)}")
        seen_scenes |= part_scenes
        for row in metrics["frames"]:
            sample_id = str(row["sample_id"])
            if sample_id in seen_samples:
                raise RuntimeError(f"duplicate sample: {sample_id}")
            seen_samples.add(sample_id)
            rows.append(row)
        for scene in part_scenes:
            shutil.copytree(part / scene, args.output / scene, dirs_exist_ok=True)
        manifests.append(manifest)

    rows.sort(key=lambda row: (str(row["scene"]), int(row["frame_id"])))
    summary = scene_bootstrap_summary(rows)
    write_json_atomic(args.output / "metrics.json", {"summary": summary, "frames": rows})
    with (args.output / "frame_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = dict(manifests[0])
    manifest["scenes"] = sorted(seen_scenes)
    manifest["selected_frames"] = sorted(
        (item for value in manifests for item in value["selected_frames"]),
        key=lambda item: str(item["sample_id"]),
    )
    manifest["skipped_frames"] = sorted(
        (item for value in manifests for item in value["skipped_frames"]),
        key=lambda item: str(item["sample_id"]),
    )
    manifest["selected_count"] = len(rows)
    manifest["skipped_count"] = len(manifest["skipped_frames"])
    manifest["started_at_unix"] = min(value["started_at_unix"] for value in manifests)
    manifest["completed_at_unix"] = max(value["completed_at_unix"] for value in manifests)
    manifest["completed"] = True
    manifest["merged_from"] = [str(part.resolve()) for part in args.parts]
    write_json_atomic(args.output / "run_manifest.json", manifest)
    print(json.dumps({key: round(value["mean"], 6) for key, value in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
