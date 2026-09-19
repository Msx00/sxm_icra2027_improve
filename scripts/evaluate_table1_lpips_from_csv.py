#!/usr/bin/env python3
"""Compute canonical Table-I LPIPS metrics from explicit frame mappings.

Each input CSV must contain ``method``, ``sample_id``, ``target_path``,
``hole_path``, and ``tool_path`` plus either ``render_path`` or the equivalent
``prediction_path`` column.  Multiple CSVs may be supplied (for example, one
per method or shard), but duplicate method/sample pairs are rejected.

The evaluator enforces the fixed seven-scene, 1,392-frame Table-I denominator
for every selected method and requires the exact sample-ID set to agree across
methods.  Full-frame LPIPS is scalar AlexNet LPIPS on pairs whose non-tissue
pixels have been zeroed.  H-LPIPS is the spatial AlexNet LPIPS map averaged
over the intersection of the canonical raw-DSS hole mask and non-tool tissue.
Images are loaded in [0, 1] and LPIPS performs its required normalization to
[-1, 1] via ``normalize=True``.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError


EXPECTED_SAMPLES = 1392
EXPECTED_SCENES = 7
REQUIRED_COLUMNS = {"method", "sample_id", "target_path", "hole_path", "tool_path"}
PREDICTION_COLUMNS = ("render_path", "prediction_path")


@dataclass(frozen=True)
class FrameMapping:
    method: str
    sample_id: str
    scene: str
    render_path: Path
    target_path: Path
    hole_path: Path
    tool_path: Path | None
    source_csv: Path
    source_row: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mapping_csv",
        type=Path,
        nargs="+",
        help="One or more frame-mapping CSV files.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional exact method-name filter (default: all methods in the CSVs).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel PIL loaders per method (default: 8).",
    )
    return parser.parse_args()


def resolve_csv_path(value: str, source_csv: Path, *, allow_empty: bool = False) -> Path | None:
    cleaned = value.strip()
    if not cleaned:
        if allow_empty:
            return None
        raise ValueError(f"empty required path in {source_csv}")
    path = Path(cleaned).expanduser()
    if not path.is_absolute():
        path = source_csv.parent / path
    return path.resolve()


def parse_sample_id(value: str, source_csv: Path, row_number: int) -> tuple[str, str]:
    sample_id = value.strip().strip("/")
    parts = sample_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"{source_csv}:{row_number}: sample_id must be 'scene/frame', got {value!r}"
        )
    return sample_id, parts[0]


def choose_prediction_path(row: dict[str, str], source_csv: Path, row_number: int) -> Path:
    values = {
        column: row.get(column, "").strip()
        for column in PREDICTION_COLUMNS
        if column in row
    }
    populated = {column: value for column, value in values.items() if value}
    if not populated:
        raise ValueError(
            f"{source_csv}:{row_number}: missing both render_path and prediction_path"
        )
    resolved = {
        column: resolve_csv_path(value, source_csv)
        for column, value in populated.items()
    }
    unique = {str(path) for path in resolved.values()}
    if len(unique) != 1:
        raise ValueError(
            f"{source_csv}:{row_number}: render_path and prediction_path disagree: {resolved}"
        )
    return next(iter(resolved.values()))  # type: ignore[return-value]


def read_mappings(csv_paths: list[Path]) -> dict[str, dict[str, FrameMapping]]:
    by_method: dict[str, dict[str, FrameMapping]] = {}
    for input_path in csv_paths:
        source_csv = input_path.expanduser().resolve()
        if not source_csv.is_file():
            raise FileNotFoundError(source_csv)
        with source_csv.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing = REQUIRED_COLUMNS - fieldnames
            if missing:
                raise ValueError(
                    f"{source_csv}: missing required columns: {sorted(missing)}"
                )
            if not any(column in fieldnames for column in PREDICTION_COLUMNS):
                raise ValueError(
                    f"{source_csv}: requires render_path or prediction_path"
                )
            for row_number, row in enumerate(reader, start=2):
                method = row.get("method", "").strip()
                if not method:
                    raise ValueError(f"{source_csv}:{row_number}: empty method")
                sample_id, scene = parse_sample_id(
                    row.get("sample_id", ""), source_csv, row_number
                )
                mapping = FrameMapping(
                    method=method,
                    sample_id=sample_id,
                    scene=scene,
                    render_path=choose_prediction_path(row, source_csv, row_number),
                    target_path=resolve_csv_path(  # type: ignore[arg-type]
                        row.get("target_path", ""), source_csv
                    ),
                    hole_path=resolve_csv_path(  # type: ignore[arg-type]
                        row.get("hole_path", ""), source_csv
                    ),
                    tool_path=resolve_csv_path(
                        row.get("tool_path", ""), source_csv, allow_empty=True
                    ),
                    source_csv=source_csv,
                    source_row=row_number,
                )
                method_records = by_method.setdefault(method, {})
                if sample_id in method_records:
                    previous = method_records[sample_id]
                    raise ValueError(
                        f"duplicate {method}/{sample_id}: "
                        f"{previous.source_csv}:{previous.source_row} and "
                        f"{source_csv}:{row_number}"
                    )
                method_records[sample_id] = mapping
    if not by_method:
        raise ValueError("no frame mappings found")
    return by_method


def select_and_validate(
    by_method: dict[str, dict[str, FrameMapping]], requested: list[str] | None
) -> tuple[list[str], list[str]]:
    if requested is None:
        methods = sorted(by_method)
    else:
        methods = []
        for method in requested:
            if method in methods:
                raise ValueError(f"method selected more than once: {method}")
            if method not in by_method:
                raise ValueError(
                    f"selected method {method!r} not found; available: {sorted(by_method)}"
                )
            methods.append(method)
    if not methods:
        raise ValueError("no methods selected")

    reference_ids: set[str] | None = None
    for method in methods:
        sample_ids = set(by_method[method])
        scenes = {mapping.scene for mapping in by_method[method].values()}
        if len(sample_ids) != EXPECTED_SAMPLES or len(scenes) != EXPECTED_SCENES:
            raise ValueError(
                f"{method}: expected {EXPECTED_SAMPLES} samples in {EXPECTED_SCENES} "
                f"scenes, got {len(sample_ids)} samples in {len(scenes)} scenes"
            )
        if reference_ids is None:
            reference_ids = sample_ids
        elif sample_ids != reference_ids:
            missing = sorted(reference_ids - sample_ids)
            unexpected = sorted(sample_ids - reference_ids)
            raise ValueError(
                f"{method}: sample-ID set differs from the first selected method; "
                f"missing={len(missing)} {missing[:5]}, "
                f"unexpected={len(unexpected)} {unexpected[:5]}"
            )
    assert reference_ids is not None
    return methods, sorted(reference_ids)


def open_rgb(
    path: Path,
    *,
    expected_size: tuple[int, int] | None = None,
    resize_if_needed: bool = False,
) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with Image.open(path) as image:
            image.load()
            image = image.convert("RGB")
            if expected_size is not None and image.size != expected_size:
                if not resize_if_needed:
                    raise ValueError(
                        f"image-size mismatch for {path}: got {image.size}, "
                        f"expected {expected_size}"
                    )
                image = image.resize(expected_size, Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"unreadable RGB image: {path}: {error}") from error
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def open_mask(
    path: Path | None,
    expected_size: tuple[int, int],
    *,
    invert: bool = False,
    missing_is_true: bool = False,
) -> torch.Tensor:
    if path is None or not path.is_file():
        if missing_is_true:
            return torch.ones((1, expected_size[1], expected_size[0]), dtype=torch.bool)
        raise FileNotFoundError(path)
    try:
        with Image.open(path) as image:
            image.load()
            image = image.convert("L")
            if image.size != expected_size:
                # Dataset tool masks are stored at 1280x1024 while the fixed
                # Table-I targets are 640x512.  Match the paper evaluators by
                # resizing binary masks with nearest-neighbour sampling.
                image = image.resize(expected_size, Image.Resampling.NEAREST)
            array = np.asarray(image, dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"unreadable mask image: {path}: {error}") from error
    mask = array < 128 if invert else array >= 128
    return torch.from_numpy(mask).unsqueeze(0)


def sample_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all():
        raise ValueError("metric values must contain at least two finite observations")
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
    }


def load_evaluation_inputs(
    job: tuple[str, FrameMapping],
) -> tuple[FrameMapping, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load one mapped frame; ThreadPoolExecutor preserves submitted order."""
    method, mapping = job
    try:
        target = open_rgb(mapping.target_path)
        height, width = target.shape[-2:]
        prediction = open_rgb(
            mapping.render_path,
            expected_size=(width, height),
            resize_if_needed=True,
        )
        tissue = open_mask(
            mapping.tool_path,
            (width, height),
            invert=True,
            missing_is_true=True,
        )
        hole = open_mask(mapping.hole_path, (width, height)) & tissue
    except Exception as error:
        raise type(error)(
            f"{method}/{mapping.sample_id} ({mapping.source_csv}:"
            f"{mapping.source_row}): {error}"
        ) from error
    if not bool(tissue.any()):
        raise ValueError(f"{method}/{mapping.sample_id}: empty non-tool tissue mask")
    if not bool(hole.any()):
        raise ValueError(f"{method}/{mapping.sample_id}: empty canonical H mask")
    return mapping, prediction, target, tissue, hole


def evaluate_method(
    method: str,
    mappings: dict[str, FrameMapping],
    ordered_ids: list[str],
    device: torch.device,
    batch_size: int,
    workers: int,
    scalar_model: torch.nn.Module,
    spatial_model: torch.nn.Module,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(ordered_ids), batch_size):
            batch_ids = ordered_ids[start : start + batch_size]
            loaded = list(
                pool.map(
                    load_evaluation_inputs,
                    [(method, mappings[sample_id]) for sample_id in batch_ids],
                )
            )
            batch_mappings = [item[0] for item in loaded]
            predictions = [item[1] for item in loaded]
            targets = [item[2] for item in loaded]
            tissue_masks = [item[3] for item in loaded]
            hole_masks = [item[4] for item in loaded]
            expected_tensor_shape = tuple(targets[0].shape)
            for mapping, target in zip(batch_mappings[1:], targets[1:]):
                if tuple(target.shape) != expected_tensor_shape:
                    raise ValueError(
                        f"{method}: mixed target sizes within one batch; "
                        f"{mapping.sample_id} has {tuple(target.shape)}, "
                        f"expected {expected_tensor_shape}. Use a batch size of 1 "
                        "for mixed-resolution mappings."
                    )

            prediction_batch = torch.stack(predictions).to(device, non_blocking=True)
            target_batch = torch.stack(targets).to(device, non_blocking=True)
            tissue_batch = torch.stack(tissue_masks).to(device, non_blocking=True).float()
            hole_batch = torch.stack(hole_masks).to(device, non_blocking=True).float()
            with torch.inference_mode():
                scalar = scalar_model(
                    prediction_batch * tissue_batch,
                    target_batch * tissue_batch,
                    normalize=True,
                ).reshape(-1)
                spatial = spatial_model(
                    prediction_batch,
                    target_batch,
                    normalize=True,
                )
                if spatial.shape[-2:] != hole_batch.shape[-2:]:
                    raise ValueError(
                        f"{method}: spatial LPIPS shape {tuple(spatial.shape[-2:])} "
                        f"does not match mask shape {tuple(hole_batch.shape[-2:])}"
                    )
                hole = (spatial * hole_batch).flatten(1).sum(1).div(
                    hole_batch.flatten(1).sum(1)
                )
            scalar_values = scalar.detach().cpu().tolist()
            hole_values = hole.detach().cpu().tolist()
            for mapping, lpips_value, hole_value, tissue, h_mask in zip(
                batch_mappings,
                scalar_values,
                hole_values,
                tissue_masks,
                hole_masks,
            ):
                records.append(
                    {
                        "method": method,
                        "scene": mapping.scene,
                        "sample_id": mapping.sample_id,
                        "lpips": float(lpips_value),
                        "hole_lpips": float(hole_value),
                        "tissue_pixel_count": int(tissue.sum().item()),
                        "hole_pixel_count": int(h_mask.sum().item()),
                        "tool_mask_missing": mapping.tool_path is None
                        or not mapping.tool_path.is_file(),
                        "render_path": str(mapping.render_path),
                        "target_path": str(mapping.target_path),
                        "hole_path": str(mapping.hole_path),
                        "tool_path": str(mapping.tool_path) if mapping.tool_path else "",
                        "source_csv": str(mapping.source_csv),
                        "source_row": mapping.source_row,
                    }
                )
            print(
                f"{method}: {min(start + batch_size, len(ordered_ids))}/"
                f"{len(ordered_ids)}",
                flush=True,
            )

    summary = {
        "count": len(records),
        "scene_count": len({record["scene"] for record in records}),
        "missing_tool_mask_count": sum(
            bool(record["tool_mask_missing"]) for record in records
        ),
        "metrics": {
            "lpips": sample_summary([float(record["lpips"]) for record in records]),
            "hole_lpips": sample_summary(
                [float(record["hole_lpips"]) for record in records]
            ),
        },
    }
    return summary, records


def atomic_write_text(path: Path, contents: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def write_outputs(output_dir: Path, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_dir / "summary.json",
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )
    fields = [
        "method",
        "scene",
        "sample_id",
        "lpips",
        "hole_lpips",
        "tissue_pixel_count",
        "hole_pixel_count",
        "tool_mask_missing",
        "render_path",
        "target_path",
        "hole_path",
        "tool_path",
        "source_csv",
        "source_row",
    ]
    temporary = output_dir / "frame_metrics.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_dir / "frame_metrics.csv")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError(f"batch size must be positive, got {args.batch_size}")
    if args.workers < 1:
        raise ValueError(f"workers must be positive, got {args.workers}")
    mappings = read_mappings(args.mapping_csv)
    methods, ordered_ids = select_and_validate(mappings, args.methods)

    import lpips  # type: ignore

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    scalar_model = lpips.LPIPS(net="alex").eval().to(device)
    spatial_model = lpips.LPIPS(net="alex", spatial=True).eval().to(device)

    summaries: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for method in methods:
        summary, rows = evaluate_method(
            method,
            mappings[method],
            ordered_ids,
            device,
            args.batch_size,
            args.workers,
            scalar_model,
            spatial_model,
        )
        summaries[method] = summary
        all_rows.extend(rows)

    payload = {
        "protocol": {
            "name": "Table I canonical LPIPS-only evaluation",
            "mapping_csvs": [
                str(path.expanduser().resolve()) for path in args.mapping_csv
            ],
            "methods": methods,
            "count_per_method": EXPECTED_SAMPLES,
            "scene_count": EXPECTED_SCENES,
            "sample_ids_identical_across_methods": True,
            "aggregation": "frame-wise mean and sample standard deviation (ddof=1)",
            "prediction_resize": "bilinear to target size only when necessary",
            "target_resize": "disabled; mismatches are fatal",
            "mask_resize": "nearest-neighbour to target size when necessary",
            "full_mask": "inverse binary tool mask; absent tool mask means all tissue",
            "hole_mask": "non-tool tissue intersected with canonical raw-DSS hole mask",
            "lpips": (
                "scalar AlexNet LPIPS on tissue-zeroed pairs; [0,1] inputs "
                "normalized internally to [-1,1]"
            ),
            "hole_lpips": (
                "spatial AlexNet LPIPS map averaged over the canonical H mask; "
                "[0,1] inputs normalized internally to [-1,1]"
            ),
        },
        "methods": summaries,
    }
    write_outputs(args.output_dir.expanduser().resolve(), payload, all_rows)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
