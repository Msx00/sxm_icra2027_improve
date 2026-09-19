#!/usr/bin/env python3
"""Validate and summarize the five Table-II per-frame timing benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMING = ROOT / "results/zeroshot/endovis_final_more_275/timing"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-dir", type=Path, default=DEFAULT_TIMING)
    parser.add_argument("--expected-full-count", type=int, default=275)
    parser.add_argument(
        "--expected-timing-count",
        type=int,
        default=0,
        help="Expected shared timing subset size; 0 accepts the Hunyuan file size.",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_id(row: dict) -> str:
    if row.get("sample_id"):
        return str(row["sample_id"])
    return f"{row['scene']}/{row['name']}"


def summarize(method: str, records: list[dict], sources: list[Path]) -> dict:
    ids = [sample_id(row) for row in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{method} contains duplicate sample identifiers")
    values = [float(row["seconds"]) for row in records]
    mean_seconds = statistics.fmean(values)
    sample_sd_seconds = statistics.stdev(values)
    mean_ms = 1000.0 * mean_seconds
    sample_sd_ms = 1000.0 * sample_sd_seconds
    return {
        "method": method,
        "count": len(values),
        "mean_seconds": mean_seconds,
        "sample_sd_seconds": sample_sd_seconds,
        "mean_ms": mean_ms,
        "sample_sd_ms": sample_sd_ms,
        "min_seconds": min(values),
        "max_seconds": max(values),
        "two_decimal_seconds_text": (
            f"{mean_seconds:.2f}\\pm{sample_sd_seconds:.2f}"
        ),
        "two_decimal_ms_text": f"{mean_ms:.2f}\\pm{sample_sd_ms:.2f}",
        "sources": [str(path.resolve()) for path in sources],
        "sample_ids": ids,
        "seconds_by_id": dict(zip(ids, values)),
    }


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = arguments()
    timing_dir = args.timing_dir.expanduser().resolve()
    regular = {
        "LDM": timing_dir / "ldm.json",
        "LaMa": timing_dir / "lama.json",
        "LCM-LoRA": timing_dir / "lcm_lora.json",
        "DistriNVS": timing_dir / "distrinvs.json",
    }
    regular_records = {}
    full_reference = None
    for method, path in regular.items():
        records = load(path)["frames"]
        ids = [sample_id(row) for row in records]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{method} contains duplicate sample identifiers")
        if len(ids) != args.expected_full_count:
            raise RuntimeError(
                f"{method}: expected {args.expected_full_count} full-set frames, "
                f"got {len(ids)}"
            )
        if full_reference is None:
            full_reference = set(ids)
        elif set(ids) != full_reference:
            raise RuntimeError(
                f"{method}: full-set sample identifiers differ from LDM"
            )
        regular_records[method] = records

    single_hunyuan = timing_dir / "hunyuan.json"
    hunyuan_paths = (
        [single_hunyuan]
        if single_hunyuan.is_file()
        else sorted(timing_dir.glob("hunyuan_shard*.json"))
    )
    if not hunyuan_paths:
        raise FileNotFoundError(
            "no hunyuan.json or hunyuan_shard*.json timing files found"
        )
    hunyuan_records = []
    for path in hunyuan_paths:
        hunyuan_records.extend(load(path)["frames"])
    hunyuan_records.sort(key=lambda row: int(row["global_index"]))
    timing_ids = [sample_id(row) for row in hunyuan_records]
    if len(timing_ids) != len(set(timing_ids)):
        raise RuntimeError("Hunyuan-DiT contains duplicate sample identifiers")
    if args.expected_timing_count and len(timing_ids) != args.expected_timing_count:
        raise RuntimeError(
            f"Hunyuan-DiT: expected {args.expected_timing_count} timing frames, "
            f"got {len(timing_ids)}"
        )
    missing = set(timing_ids) - full_reference
    if missing:
        raise RuntimeError(
            f"Hunyuan-DiT timing samples are outside the frozen set: "
            f"{sorted(missing)[:5]}"
        )

    timing_reference = set(timing_ids)
    summaries = {}
    for method, path in regular.items():
        selected = [
            row for row in regular_records[method]
            if sample_id(row) in timing_reference
        ]
        if len(selected) != len(timing_ids):
            raise RuntimeError(
                f"{method}: expected {len(timing_ids)} shared timing frames, "
                f"got {len(selected)}"
            )
        summaries[method] = summarize(method, selected, [path])
    summaries["Hunyuan-DiT"] = summarize(
        "Hunyuan-DiT", hunyuan_records, hunyuan_paths
    )

    order = ["LDM", "LaMa", "LCM-LoRA", "Hunyuan-DiT", "DistriNVS"]
    concise = {
        method: {
            key: summaries[method][key]
            for key in (
                "count",
                "mean_seconds",
                "sample_sd_seconds",
                "mean_ms",
                "sample_sd_ms",
                "min_seconds",
                "max_seconds",
                "two_decimal_seconds_text",
                "two_decimal_ms_text",
                "sources",
            )
        }
        for method in order
    }
    summary_payload = {
        "protocol": {
            "full_test_pairs": args.expected_full_count,
            "timing_pairs": len(timing_ids),
            "selection": "Hunyuan global indices; uniformly strided over the frozen list",
            "batch_size": 1,
            "unit": "s/frame",
            "dispersion": "sample standard deviation",
        },
        "methods": concise,
    }
    write_json(timing_dir / "summary.json", summary_payload)

    with (timing_dir / "frame_timings.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", *order])
        for identifier in timing_ids:
            writer.writerow([
                identifier,
                *[summaries[method]["seconds_by_id"][identifier] for method in order],
            ])

    rows = [
        f"{method} & ${concise[method]['two_decimal_seconds_text']}$ \\\\"
        for method in order
    ]
    (timing_dir / "table_rows.tex").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    for method in order:
        print(
            f"{method}: {concise[method]['mean_seconds']:.6f} +- "
            f"{concise[method]['sample_sd_seconds']:.6f} s/frame"
        )


if __name__ == "__main__":
    main()
