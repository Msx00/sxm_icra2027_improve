#!/usr/bin/env python3
"""Evaluate five methods on the fixed 275-pair EndoVis Table-II protocol.

The completed DistriNVS primary run defines the sample IDs, targets, and
raw-DSS hole masks. Every method must have exactly the same 275 scene/frame
IDs. Full metrics use all pixels (EndoVis has no tool mask); H metrics use the
saved primary-run hole mask. Implementations match the Table-I evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from evaluate_table1_reconstruction_baselines import (
    is_degenerate,
    load_mask,
    load_rgb,
    masked_psnr,
    masked_ssim,
    sample_summary,
)

EXPECTED_SAMPLES, EXPECTED_SCENES = 275, 10
METRICS = ("psnr", "ssim", "lpips", "hole_psnr", "hole_ssim", "hole_lpips")
LOWER_IS_BETTER = {"lpips", "hole_lpips"}


@dataclass(frozen=True)
class MethodSpec:
    key: str
    name: str
    latex: str
    root: Path
    pattern: str
    scene_parent: int
    suffix: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary-run", type=Path, required=True)
    p.add_argument("--ldm-root", type=Path, required=True)
    p.add_argument("--lama-root", type=Path, required=True)
    p.add_argument("--lcm-root", type=Path, required=True)
    p.add_argument("--hunyuan-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ldm-pattern", default="*/*.png")
    p.add_argument("--lama-pattern", default="*/*.png")
    p.add_argument(
        "--lcm-pattern",
        default="*/results/endo1_supervised_rank32_step2500/lcm/final/*.png",
    )
    p.add_argument("--hunyuan-pattern", default="gpu*/inference/*/*_completed.png")
    p.add_argument("--distrinvs-pattern", default="*/renders/*.png")
    p.add_argument("--ldm-scene-parent", type=int, default=0)
    p.add_argument("--lama-scene-parent", type=int, default=0)
    p.add_argument("--lcm-scene-parent", type=int, default=4)
    p.add_argument("--hunyuan-scene-parent", type=int, default=0)
    p.add_argument("--distrinvs-scene-parent", type=int, default=1)
    p.add_argument("--ldm-filename-suffix", default="")
    p.add_argument("--lama-filename-suffix", default="")
    p.add_argument("--lcm-filename-suffix", default="")
    p.add_argument("--hunyuan-filename-suffix", default="_completed")
    p.add_argument("--distrinvs-filename-suffix", default="")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--validate-only", action="store_true",
        help="Validate all sample mappings and sizes without loading LPIPS.",
    )
    return p.parse_args()


def specs(args: argparse.Namespace, primary: Path) -> list[MethodSpec]:
    return [
        MethodSpec(
            "ldm", "LDM", r"LDM~\cite{rombach2022ldm}",
            args.ldm_root.expanduser().resolve(), args.ldm_pattern,
            args.ldm_scene_parent, args.ldm_filename_suffix,
        ),
        MethodSpec(
            "lama", "LaMa", r"LaMa~\cite{suvorov2022lama}",
            args.lama_root.expanduser().resolve(), args.lama_pattern,
            args.lama_scene_parent, args.lama_filename_suffix,
        ),
        MethodSpec(
            "lcm_lora", "LCM-LoRA", r"LCM-LoRA~\cite{luo2023lcmlora}",
            args.lcm_root.expanduser().resolve(), args.lcm_pattern,
            args.lcm_scene_parent, args.lcm_filename_suffix,
        ),
        MethodSpec(
            "hunyuan", "Hunyuan-DiT",
            r"Hunyuan-DiT$^\dagger$~\cite{li2024hunyuandit}",
            args.hunyuan_root.expanduser().resolve(), args.hunyuan_pattern,
            args.hunyuan_scene_parent, args.hunyuan_filename_suffix,
        ),
        MethodSpec(
            "distrinvs", "DistriNVS (soft fusion; primary)",
            "DistriNVS (soft fusion; primary)", primary,
            args.distrinvs_pattern, args.distrinvs_scene_parent,
            args.distrinvs_filename_suffix,
        ),
    ]


def make_id(scene: str, stem: str) -> str:
    if not scene or not stem or "/" in scene or "/" in stem:
        raise ValueError(f"invalid scene/frame ID: {scene!r}/{stem!r}")
    return f"{scene}/{stem}"


def mismatch(label: str, actual: set[str], expected: set[str]) -> str:
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    return (
        f"{label}: expected={len(expected)}, actual={len(actual)}, "
        f"missing={len(missing)} {missing[:8]}, "
        f"unexpected={len(extra)} {extra[:8]}"
    )


def load_protocol(primary: Path) -> tuple[dict[str, Any], list[str]]:
    path = primary / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("completed"):
        raise RuntimeError(f"incomplete primary run: {path}")
    selected = manifest.get("selected_frames")
    if not isinstance(selected, list):
        raise ValueError(f"{path}: selected_frames is not a list")
    min_overlap = manifest.get("min_overlap")
    if not isinstance(min_overlap, (int, float)) or not math.isclose(
        float(min_overlap), 0.70, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{path}: min_overlap must equal 0.70, got {min_overlap!r}")
    if int(manifest.get("skipped_count", -1)) != 0:
        raise ValueError(f"{path}: skipped_count must be zero")
    skipped = manifest.get("skipped_frames")
    if not isinstance(skipped, list) or skipped:
        raise ValueError(f"{path}: skipped_frames must be an empty list")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{path}: config must be a mapping")
    ablation = config.get("ablation")
    if not isinstance(ablation, dict) or ablation.get("hard_composition") is not False:
        raise ValueError(
            f"{path}: config.ablation.hard_composition must be false"
        )
    repair = manifest.get("small_hole_repair")
    if (
        not isinstance(repair, dict)
        or not isinstance(repair.get("max_area_ratio"), (int, float))
        or float(repair["max_area_ratio"]) != 0.0
    ):
        raise ValueError(
            f"{path}: small_hole_repair.max_area_ratio must equal zero"
        )
    ids: list[str] = []
    for index, record in enumerate(selected):
        if not isinstance(record, dict) or "sample_id" not in record:
            raise ValueError(f"{path}: malformed selected_frames[{index}]")
        raw_overlap = record.get("raw_overlap")
        if (
            not isinstance(raw_overlap, (int, float))
            or not math.isfinite(float(raw_overlap))
            or float(raw_overlap) <= 0.70
        ):
            raise ValueError(
                f"{path}: selected_frames[{index}].raw_overlap must be > 0.70, "
                f"got {raw_overlap!r}"
            )
        parts = str(record["sample_id"]).split("/")
        if len(parts) != 2:
            raise ValueError(f"{path}: expected scene/frame: {record['sample_id']!r}")
        ids.append(make_id(*parts))
    duplicate = sorted(k for k, n in Counter(ids).items() if n > 1)
    if duplicate:
        raise ValueError(f"{path}: duplicate sample IDs: {duplicate[:8]}")
    if len(ids) != EXPECTED_SAMPLES:
        raise ValueError(f"expected {EXPECTED_SAMPLES} samples, got {len(ids)}")
    if int(manifest.get("selected_count", -1)) != EXPECTED_SAMPLES:
        raise ValueError(f"{path}: selected_count is not {EXPECTED_SAMPLES}")
    scenes = {item.split("/", 1)[0] for item in ids}
    if len(scenes) != EXPECTED_SCENES:
        raise ValueError(f"expected {EXPECTED_SCENES} scenes, got {sorted(scenes)}")
    declared = manifest.get("scenes")
    if not isinstance(declared, list) or set(map(str, declared)) != scenes:
        raise ValueError(f"{path}: scenes disagree with selected_frames")
    return manifest, ids


def canonical_files(
    primary: Path, directory: str, expected: set[str]
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(primary.glob(f"*/{directory}/*.png")):
        if not path.is_file():
            continue
        identifier = make_id(path.parent.parent.name, path.stem)
        if identifier in result:
            raise ValueError(f"duplicate {directory}: {identifier}")
        result[identifier] = path
    if set(result) != expected:
        raise ValueError(mismatch(f"canonical {directory}", set(result), expected))
    return result


def predictions(spec: MethodSpec, expected: set[str]) -> dict[str, Path]:
    if spec.scene_parent < 0:
        raise ValueError(f"{spec.key}: negative scene-parent")
    if not spec.root.is_dir():
        raise FileNotFoundError(spec.root)
    paths = sorted(path for path in spec.root.glob(spec.pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"{spec.name}: no match for {spec.root / spec.pattern}")
    result: dict[str, Path] = {}
    for path in paths:
        stem = path.stem
        if spec.suffix:
            if not stem.endswith(spec.suffix):
                raise ValueError(f"{spec.name}: wrong suffix: {path}")
            stem = stem[: -len(spec.suffix)]
        try:
            scene = path.parents[spec.scene_parent].name
        except IndexError as error:
            raise ValueError(f"{spec.name}: bad scene-parent for {path}") from error
        identifier = make_id(scene, stem)
        if identifier in result:
            raise ValueError(f"{spec.name}: duplicate prediction: {identifier}")
        result[identifier] = path
    if set(result) != expected:
        raise ValueError(mismatch(spec.name, set(result), expected))
    return result


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as image:
        return image.size


def target_sizes(
    ids: list[str], targets: dict[str, Path], holes: dict[str, Path]
) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for identifier in ids:
        size = image_size(targets[identifier])
        if image_size(holes[identifier]) != size:
            raise ValueError(f"target/hole size mismatch: {identifier}")
        result[identifier] = size
    return result


def build_records(
    spec: MethodSpec,
    ids: list[str],
    preds: dict[str, Path],
    targets: dict[str, Path],
    holes: dict[str, Path],
    sizes: dict[str, tuple[int, int]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for identifier in ids:
        scene, stem = identifier.split("/", 1)
        prediction = preds[identifier]
        if image_size(prediction) != sizes[identifier]:
            raise ValueError(
                f"{spec.name}: prediction/target size mismatch: {identifier}"
            )
        result.append({
            "method_key": spec.key,
            "method": spec.name,
            "scene": scene,
            "frame_id": stem,
            "sample_id": identifier,
            "degenerate": is_degenerate(prediction),
            "prediction_path": str(prediction),
            "target_path": str(targets[identifier]),
            "hole_path": str(holes[identifier]),
        })
    if len(result) != EXPECTED_SAMPLES:
        raise AssertionError("record count changed after exact-set validation")
    return result


def evaluate(
    spec: MethodSpec,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    scalar_lpips: torch.nn.Module,
    spatial_lpips: torch.nn.Module,
) -> None:
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        pred_list, target_list, full_list, hole_list = [], [], [], []
        for record in batch:
            pred = load_rgb(Path(record["prediction_path"]))
            target = load_rgb(Path(record["target_path"]))
            if pred.shape != target.shape:
                raise ValueError(f"tensor shape mismatch: {record['sample_id']}")
            height, width = target.shape[-2:]
            full = torch.ones((1, height, width), dtype=torch.bool)
            hole = load_mask(Path(record["hole_path"]), (width, height))
            hole_count = int(hole.sum())
            if hole_count == 0:
                raise ValueError(f"empty canonical hole: {record['sample_id']}")
            record["full_pixel_count"] = height * width
            record["hole_pixel_count"] = hole_count
            pred_list.append(pred)
            target_list.append(target)
            full_list.append(full)
            hole_list.append(hole)
        pred_b = torch.stack(pred_list).to(device, non_blocking=True)
        target_b = torch.stack(target_list).to(device, non_blocking=True)
        full_b = torch.stack(full_list).to(device, non_blocking=True)
        hole_b = torch.stack(hole_list).to(device, non_blocking=True)
        with torch.inference_mode():
            lpips_full = scalar_lpips(
                pred_b * full_b.float(), target_b * full_b.float(), normalize=True
            ).reshape(-1)
            spatial = spatial_lpips(pred_b, target_b, normalize=True)
            lpips_hole = (
                (spatial * hole_b.float()).flatten(1).sum(1)
                / hole_b.flatten(1).sum(1).clamp_min(1).float()
            )
        for i, record in enumerate(batch):
            pred, target = pred_b[i:i + 1], target_b[i:i + 1]
            full, hole = full_b[i:i + 1], hole_b[i:i + 1]
            record.update({
                "psnr": masked_psnr(pred, target, full),
                "ssim": masked_ssim(pred, target, full),
                "lpips": float(lpips_full[i].item()),
                "hole_psnr": masked_psnr(pred, target, hole),
                "hole_ssim": masked_ssim(pred, target, hole),
                "hole_lpips": float(lpips_hole[i].item()),
            })
        print(
            f"{spec.name}: {min(start + batch_size, len(records))}/{len(records)}",
            flush=True,
        )


def summarize(spec: MethodSpec, records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        metric: sample_summary([float(record[metric]) for record in records])
        for metric in METRICS
    }
    if not all(
        math.isfinite(value)
        for summary in metrics.values()
        for value in summary.values()
    ):
        raise ValueError(f"{spec.name}: non-finite aggregate")
    scenes = Counter(str(record["scene"]) for record in records)
    return {
        "method_key": spec.key,
        "method": spec.name,
        "count": len(records),
        "scene_count": len(scenes),
        "scene_counts": dict(sorted(scenes.items())),
        "degenerate_count": sum(bool(r["degenerate"]) for r in records),
        "metrics": metrics,
    }


def best_methods(summaries: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        values = {
            key: float(value["metrics"][metric]["mean"])
            for key, value in summaries.items()
        }
        optimum = min(values.values()) if metric in LOWER_IS_BETTER else max(values.values())
        result[metric] = {
            "direction": "lower" if metric in LOWER_IS_BETTER else "higher",
            "best_mean": optimum,
            "method_keys": [
                key for key, value in values.items()
                if math.isclose(value, optimum, rel_tol=1e-12, abs_tol=1e-12)
            ],
        }
    return result


def metric_tex(summary: dict[str, float], bold: bool) -> str:
    body = f"{summary['mean']:.2f}\\pm{summary['sample_sd']:.2f}"
    if bold:
        body = rf"\mathbf{{{body}}}"
    return f"${body}$"


def table_rows(
    methods: list[MethodSpec],
    summaries: dict[str, dict[str, Any]],
    best: dict[str, dict[str, Any]],
) -> str:
    lines = [
        r"% Generated by evaluate_table2_endovis_completion.py; do not edit.",
        r"% EndoVis 8+9 combined; frame-wise mean $\pm$ sample SD (ddof=1).",
    ]
    for spec in methods:
        if spec.key == "distrinvs":
            lines.append(r"\midrule")
        summary = summaries[spec.key]
        values = [
            metric_tex(
                summary["metrics"][metric],
                spec.key in best[metric]["method_keys"],
            )
            for metric in METRICS
        ]
        lines.append(
            f"{spec.latex} & {summary['count']:,} & "
            + " & ".join(values)
            + r" \\"
        )
    return "\n".join(lines) + "\n"


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_outputs(
    output: Path,
    payload: dict[str, Any],
    records: list[dict[str, Any]],
    rows: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    atomic_text(output / "summary.json", encoded)
    atomic_text(output / "metrics.json", encoded)
    atomic_text(output / "table_rows.tex", rows)
    fields = [
        "method_key", "method", "scene", "frame_id", "sample_id", "degenerate",
        *METRICS, "full_pixel_count", "hole_pixel_count",
        "prediction_path", "target_path", "hole_path",
    ]
    destination = output / "frame_metrics.csv"
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fields})
    temporary.replace(destination)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    primary = args.primary_run.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    manifest, ids = load_protocol(primary)
    expected = set(ids)
    targets = canonical_files(primary, "targets", expected)
    holes = canonical_files(primary, "hole_mask", expected)
    sizes = target_sizes(ids, targets, holes)
    methods = specs(args, primary)
    records_by_method: dict[str, list[dict[str, Any]]] = {}
    layouts: dict[str, dict[str, Any]] = {}
    for spec in methods:
        records = build_records(
            spec, ids, predictions(spec, expected), targets, holes, sizes
        )
        records_by_method[spec.key] = records
        layouts[spec.key] = {
            "method": spec.name,
            "root": str(spec.root),
            "pattern": spec.pattern,
            "scene_parent": spec.scene_parent,
            "filename_suffix": spec.suffix,
            "count": len(records),
            "scene_count": len({r["scene"] for r in records}),
            "dimensions": "exact canonical-target equality checked",
        }
    if args.validate_only:
        print(json.dumps({
            "status": "validated",
            "primary_run": str(primary),
            "expected_count": len(expected),
            "scene_count": len({item.split("/", 1)[0] for item in expected}),
            "methods": layouts,
        }, indent=2, ensure_ascii=False))
        return

    import lpips  # type: ignore
    device = torch.device(args.device)
    scalar_lpips = lpips.LPIPS(net="alex").eval().to(device)
    spatial_lpips = lpips.LPIPS(net="alex", spatial=True).eval().to(device)
    summaries: dict[str, dict[str, Any]] = {}
    all_records: list[dict[str, Any]] = []
    for spec in methods:
        records = records_by_method[spec.key]
        evaluate(
            spec, records, device, args.batch_size, scalar_lpips, spatial_lpips
        )
        summaries[spec.key] = summarize(spec, records)
        all_records.extend(records)
    best = best_methods(summaries)
    protocol = {
        "name": "Table II fixed-N EndoVis zero-shot evaluation",
        "primary_run": str(primary),
        "primary_manifest": str(primary / "run_manifest.json"),
        "primary_manifest_method": manifest.get("method"),
        "primary_checkpoint": manifest.get("checkpoint"),
        "data_root": manifest.get("data_root"),
        "scenes": dict(sorted(Counter(x.split("/", 1)[0] for x in ids).items())),
        "count_per_method": len(ids),
        "overlap_rule": (
            "run_manifest min_overlap=0.70 and every selected raw_overlap>0.70; "
            "no frames skipped"
        ),
        "composition": (
            "direct soft-fusion output; config.ablation.hard_composition=false"
        ),
        "postprocess": "small-hole repair disabled (max_area_ratio=0)",
        "prediction_layouts": layouts,
        "aggregation": (
            "Datasets 8 and 9 combined; frame-wise mean and sample SD (ddof=1)"
        ),
        "inclusion": "all 275 manifest frames, including degenerate predictions",
        "full_mask": "all pixels; no EndoVis target tool mask",
        "hole_mask": "primary-run raw-DSS hole mask without tool intersection",
        "psnr_ssim": "Table-I masked_psnr and masked_ssim implementations",
        "lpips": (
            "AlexNet LPIPS; [0,1] normalized internally to [-1,1]; "
            "spatial LPIPS averaged over the hole for H-LPIPS"
        ),
        "ranking": "unrounded Combined means; LPIPS metrics are lower-is-better",
    }
    payload = {"protocol": protocol, "methods": summaries, "best": best}
    write_outputs(output, payload, all_records, table_rows(methods, summaries, best))
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
