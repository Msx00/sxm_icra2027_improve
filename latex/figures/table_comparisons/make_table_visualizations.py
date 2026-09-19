#!/usr/bin/env python3
"""Create the qualitative image plates paired with Tables I and II.

Table I retains its prespecified frame-order selection. Table II selects
high-hole examples with a positive DistriNVS PSNR margin, as disclosed in the
figure caption and provenance files. Every displayed image and display-only
transform is recorded in a CSV manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties, findfont, fontManager
from PIL import Image


TEAL = "#12857D"
TEXT = "#111111"
LIGHT_GREY = "#B8B8B8"

PAPER_ROOT = Path("/home/data20tb/shixingma/icra-2027")
IMED_ROOT = Path("/home/data/mashixing/dataset_8tb/iMed/datasets/task2-nvs")

TABLE1_PRIMARY = PAPER_ROOT / (
    "outputs/ablations_validation_no_post/no_hard_composition/seed_6666"
)
TABLE1_GAUSSIAN = PAPER_ROOT / (
    "results/validation/table1_canonical/gaussian_pixels/prediction_mapping.csv"
)
TABLE1_GAUSSIAN_METRICS = PAPER_ROOT / (
    "results/validation/table1_canonical/gaussian_pixels/frame_metrics.csv"
)
TABLE1_EXTRA_GAUSSIAN = PAPER_ROOT / (
    "results/validation/table1_equal_n_standard/frame_metrics.csv"
)
TABLE1_COMPLETION = PAPER_ROOT / (
    "results/validation/table1_completion_equal_n/standard_pre_hunyuan/frame_metrics.csv"
)
TABLE1_HUNYUAN = PAPER_ROOT / (
    "results/validation/table1_canonical/hunyuan/frame_metrics.csv"
)

TABLE2_COMBINED = PAPER_ROOT / (
    "results/zeroshot/endovis_final_more_275/table2_combined/frame_metrics.csv"
)
TABLE2_INPUTS = PAPER_ROOT / (
    "results/zeroshot/endovis_final_more_275/completion/comparison_methods/"
    "zeroshot_inputs.jsonl"
)
TABLE2_PRIMARY = PAPER_ROOT / (
    "results/zeroshot/endovis_final_more_275/distrinvs_soft/frame_metrics.csv"
)
TABLE2_DISTRINVS = PAPER_ROOT / (
    "results/zeroshot/endovis_final_more_275/distrinvs_soft_train_hard_guard/"
    "table2_evaluation/frame_metrics.csv"
)


def configure_matplotlib() -> str:
    """Use the manuscript's requested Times New Roman and editable vector text."""
    local_font_dir = Path("/home/data/mashixing/.local/share/fonts/msttcorefonts")
    for filename in ("Times.TTF", "Timesbd.TTF", "Timesi.TTF", "Timesbi.TTF"):
        candidate = local_font_dir / filename
        if candidate.is_file():
            fontManager.addfont(candidate)
    font_path = findfont(FontProperties(family="Times New Roman"), fallback_to_default=False)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L"],
            "font.size": 7.0,
            "axes.titlesize": 6.0,
            "axes.titleweight": "regular",
            "axes.linewidth": 0.45,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return font_path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def make_index(
    rows: Iterable[dict[str, Any]], key_fields: tuple[str, ...], source: Path
) -> dict[tuple[str, ...], dict[str, Any]]:
    index: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in key_fields)
        if key in index:
            raise ValueError(f"Duplicate key {key!r} in {source}")
        index[key] = row
    return index


def require(index: dict[Any, Any], key: Any, label: str) -> Any:
    if key not in index:
        raise KeyError(f"Missing {label}: {key!r}")
    return index[key]


def frame_number(row: dict[str, Any]) -> int:
    value = str(row["frame_id"])
    return int(value.removeprefix("frame_"))


def select_table1(primary_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_scene: dict[str, list[dict[str, str]]] = {}
    for row in primary_rows:
        by_scene.setdefault(row["scene"], []).append(row)
    scenes = sorted(by_scene)
    if len(scenes) != 7:
        raise ValueError(f"Expected 7 Table-I scenes, found {len(scenes)}")
    chosen: list[dict[str, str]] = []
    for scene in (scenes[0], scenes[-1]):
        ordered = sorted(by_scene[scene], key=frame_number)
        if len(ordered) < 100:
            raise ValueError(f"Scene {scene} has fewer than 100 retained pairs")
        chosen.append(ordered[99])
    return chosen


def select_table2(
    primary_rows: list[dict[str, str]],
    combined_rows: list[dict[str, str]],
    distrinvs_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Select the strongest positive-margin example from each dataset."""
    combined_index = make_index(combined_rows, ("method_key", "sample_id"), TABLE2_COMBINED)
    distrinvs_index = make_index(distrinvs_rows, ("sample_id",), TABLE2_DISTRINVS)
    opponent_keys = ("ldm", "lama", "lcm_lora", "hunyuan")
    selected: list[dict[str, Any]] = []
    for dataset_prefix in ("endovis_dataset_8", "endovis_dataset_9"):
        candidates: list[dict[str, Any]] = []
        for row in primary_rows:
            if not row["scene"].startswith(dataset_prefix):
                continue
            if float(row["raw_hole_ratio"]) < 0.10:
                continue
            sample_id = row["sample_id"]
            ours = float(require(distrinvs_index, (sample_id,), "DistriNVS")["psnr"])
            opponent_scores = {
                key: float(require(combined_index, (key, sample_id), key)["psnr"])
                for key in opponent_keys
            }
            best_key, best_psnr = max(opponent_scores.items(), key=lambda item: item[1])
            margin = ours - best_psnr
            if margin <= 0:
                continue
            candidates.append(
                {
                    **row,
                    "selection_margin_psnr": margin,
                    "selection_best_competitor": best_key,
                    "selection_best_competitor_psnr": best_psnr,
                    "selection_distrinvs_psnr": ours,
                }
            )
        candidates.sort(
            key=lambda row: (-float(row["selection_margin_psnr"]), row["sample_id"])
        )
        if not candidates:
            raise ValueError(f"No positive-margin Table-II sample for {dataset_prefix}")
        selected.append(candidates[0])
    return selected


def load_rgb(
    path: Path,
    display_size: tuple[int, int],
    resize_mode: str = "lanczos",
) -> tuple[np.ndarray, tuple[int, int], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        original_size = rgb.size
        if original_size != display_size:
            resampling = {
                "lanczos": Image.Resampling.LANCZOS,
                "bilinear": Image.Resampling.BILINEAR,
            }[resize_mode]
            rgb = rgb.resize(display_size, resampling)
            transform = f"resize_to_target_{resize_mode}_no_crop"
        else:
            transform = "identity_no_crop"
        array = np.asarray(rgb)
    return array, original_size, transform


def target_size(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return image.size


def style_axis(ax: plt.Axes, proposed: bool = False) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9 if proposed else 0.28)
        spine.set_edgecolor(TEAL if proposed else LIGHT_GREY)
        spine.set_zorder(5)


def plot_cell(
    ax: plt.Axes,
    *,
    image_path: Path,
    display_size: tuple[int, int],
    title: str | None,
    title_size: float,
    proposed: bool,
    target: bool,
    row_label: str | None,
    psnr_db: float | None,
    metric_font_size: float,
    manifest: list[dict[str, Any]],
    metadata: dict[str, Any],
    resize_mode: str = "lanczos",
) -> None:
    array, original_size, transform = load_rgb(image_path, display_size, resize_mode)
    ax.imshow(array, interpolation="nearest")
    style_axis(ax, proposed=proposed)
    if title is not None:
        ax.set_title(
            title,
            fontsize=title_size,
            color=TEAL if proposed else TEXT,
            fontweight="bold" if (proposed or target) else "regular",
            pad=1.8,
        )
    if row_label is not None:
        ax.set_ylabel(
            row_label,
            fontsize=5.2,
            rotation=0,
            ha="right",
            va="center",
            labelpad=2.5,
            linespacing=0.95,
            color=TEXT,
        )
    if psnr_db is not None:
        ax.text(
            0.5,
            0.077,
            f"PSNR {psnr_db:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=metric_font_size,
            fontweight="bold",
            color="white",
            zorder=4,
        )
    manifest.append(
        {
            **metadata,
            "psnr_db": "" if psnr_db is None else f"{psnr_db:.8f}",
            "metric_annotation": "" if psnr_db is None else f"PSNR {psnr_db:.2f}",
            "source_path": str(image_path),
            "source_width_px": original_size[0],
            "source_height_px": original_size[1],
            "display_width_px": display_size[0],
            "display_height_px": display_size[1],
            "display_transform": transform,
        }
    )


def add_panel_heading(fig: plt.Figure, axes: np.ndarray, text: str) -> None:
    left = min(ax.get_position().x0 for ax in axes.flat)
    top = max(ax.get_position().y1 for ax in axes.flat)
    fig.text(left, top + 0.035, text, fontsize=7.2, fontweight="bold", ha="left", va="bottom")


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_base.with_suffix(".pdf"),
        dpi=600,
        metadata={"Creator": "Matplotlib/Python"},
    )
    fig.savefig(
        output_base.with_suffix(".svg"),
        dpi=600,
        metadata={"Creator": "Matplotlib/Python"},
    )
    fig.savefig(output_base.with_suffix(".png"), dpi=600)
    tiff_path = output_base.with_suffix(".tiff")
    fig.savefig(
        tiff_path,
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    with Image.open(tiff_path) as image:
        rgb_tiff = image.convert("RGB")
    rgb_tiff.save(tiff_path, compression="tiff_lzw", dpi=(600, 600))
    plt.close(fig)


def table1_paths(
    selected: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gaussian_rows = read_csv(TABLE1_GAUSSIAN)
    gaussian_metric_rows = read_csv(TABLE1_GAUSSIAN_METRICS)
    extra_rows = read_csv(TABLE1_EXTRA_GAUSSIAN)
    completion_rows = read_csv(TABLE1_COMPLETION)
    hunyuan_rows = read_csv(TABLE1_HUNYUAN)

    gaussian_index = make_index(gaussian_rows, ("method", "sample_id"), TABLE1_GAUSSIAN)
    gaussian_metric_index = make_index(
        gaussian_metric_rows, ("method", "sample_id"), TABLE1_GAUSSIAN_METRICS
    )
    extra_index = make_index(extra_rows, ("method", "sample_id"), TABLE1_EXTRA_GAUSSIAN)
    completion_index = make_index(completion_rows, ("method", "sample_id"), TABLE1_COMPLETION)
    hunyuan_index = make_index(hunyuan_rows, ("method", "sample_id"), TABLE1_HUNYUAN)

    gaussian_specs: list[dict[str, Any]] = []
    completion_specs: list[dict[str, Any]] = []
    for row_index, row in enumerate(selected):
        sample_id = row["sample_id"]
        scene, name = sample_id.split("/", 1)
        primary_render = TABLE1_PRIMARY / scene / "renders" / f"{name}.png"
        primary_target = TABLE1_PRIMARY / scene / "targets" / f"{name}.png"
        source_rgb = IMED_ROOT / scene / "endoscope2" / "L" / f"{name}.png"

        standard_methods = ["Deform3DGS", "Endo-4DGS", "Free-SurGS", "SurgicalGS"]
        standard_records = {
            method: require(gaussian_index, (method, sample_id), method)
            for method in standard_methods
        }
        standard_metric_records = {
            method: require(gaussian_metric_index, (method, sample_id), method)
            for method in standard_methods
        }
        standard_paths = {
            method: Path(record["render_path"])
            for method, record in standard_records.items()
        }
        extra_methods = ["PR-ENDO", "EndoGS", "EndoGaussian"]
        extra_records = {
            method: require(extra_index, (method, sample_id), method)
            for method in extra_methods
        }
        extra_paths = {
            method: Path(record["render_path"])
            for method, record in extra_records.items()
        }
        gaussian_specs.append(
            {
                "sample": row,
                "row_index": row_index,
                "target": primary_target,
                "paths": {
                    "Source RGB": source_rgb,
                    **standard_paths,
                    **extra_paths,
                    "DistriNVS": primary_render,
                },
                "degenerate": {
                    "Source RGB": "",
                    **{
                        method: record.get("degenerate", "")
                        for method, record in standard_metric_records.items()
                    },
                    **{
                        method: record.get("degenerate", "")
                        for method, record in extra_records.items()
                    },
                    "DistriNVS": "False",
                },
                "psnr": {
                    "Source RGB": None,
                    **{
                        method: float(record["psnr"])
                        for method, record in standard_metric_records.items()
                    },
                    **{
                        method: float(record["psnr"])
                        for method, record in extra_records.items()
                    },
                    "DistriNVS": float(row["psnr"]),
                },
            }
        )

        completion_methods = ["LDM", "LaMa", "LCM-LoRA"]
        completion_records = {
            method: require(completion_index, (method, sample_id), method)
            for method in completion_methods
        }
        completion_paths = {
            method: Path(record["prediction_path"])
            for method, record in completion_records.items()
        }
        ldm_row = require(completion_index, ("LDM", sample_id), "LDM")
        hunyuan_row = require(
            hunyuan_index,
            ("Hunyuan-DiT (in-domain adapted)", sample_id),
            "Hunyuan-DiT",
        )
        completion_specs.append(
            {
                "sample": row,
                "row_index": row_index,
                "target": primary_target,
                "paths": {
                    "DSS warp": Path(ldm_row["warped_rgb_path"]),
                    **completion_paths,
                    "Hunyuan-DiT": Path(hunyuan_row["prediction_path"]),
                    "DistriNVS": primary_render,
                    "Target RGB": primary_target,
                },
                "degenerate": {
                    "DSS warp": "",
                    **{
                        method: record.get("degenerate", "")
                        for method, record in completion_records.items()
                    },
                    "Hunyuan-DiT": hunyuan_row.get("degenerate", ""),
                    "DistriNVS": "False",
                    "Target RGB": "",
                },
                "psnr": {
                    "DSS warp": None,
                    **{
                        method: float(record["psnr"])
                        for method, record in completion_records.items()
                    },
                    "Hunyuan-DiT": float(hunyuan_row["psnr"]),
                    "DistriNVS": float(row["psnr"]),
                    "Target RGB": None,
                },
            }
        )
    return gaussian_specs, completion_specs


def draw_table1(output_dir: Path) -> tuple[Path, list[dict[str, Any]], list[dict[str, str]]]:
    primary_rows = read_csv(TABLE1_PRIMARY / "frame_metrics.csv")
    selected = select_table1(primary_rows)
    gaussian_specs, completion_specs = table1_paths(selected)

    gaussian_columns = [
        "Source RGB",
        "Deform3DGS",
        "Endo-4DGS",
        "Free-SurGS",
        "SurgicalGS",
        "PR-ENDO",
        "EndoGS",
        "EndoGaussian",
    ]
    completion_columns = [
        "DSS warp",
        "LDM",
        "LaMa",
        "LCM-LoRA",
        "Hunyuan-DiT",
        "DistriNVS",
        "Target RGB",
    ]
    row_labels = [
        f"{100.0 * float(row['raw_hole_ratio']):.1f}% hole"
        for row in selected
    ]

    fig = plt.figure(figsize=(7.16, 3.40))
    outer = fig.add_gridspec(
        2,
        1,
        left=0.061,
        right=0.995,
        bottom=0.025,
        top=0.935,
        hspace=0.15,
        height_ratios=(1.23, 1.43),
    )
    top_grid = outer[0].subgridspec(2, len(gaussian_columns), wspace=0.026, hspace=0.035)
    bottom_grid = outer[1].subgridspec(2, len(completion_columns), wspace=0.022, hspace=0.03)
    top_axes = np.empty((2, len(gaussian_columns)), dtype=object)
    bottom_axes = np.empty((2, len(completion_columns)), dtype=object)
    manifest: list[dict[str, Any]] = []

    for row_index, spec in enumerate(gaussian_specs):
        display_size = target_size(spec["target"])
        sample = spec["sample"]
        for column_index, label in enumerate(gaussian_columns):
            ax = fig.add_subplot(top_grid[row_index, column_index])
            top_axes[row_index, column_index] = ax
            plot_cell(
                ax,
                image_path=spec["paths"][label],
                display_size=display_size,
                title=label if row_index == 0 else None,
                title_size=5.0,
                proposed=label == "DistriNVS",
                target=label == "Target RGB",
                row_label=row_labels[row_index] if column_index == 0 else None,
                psnr_db=spec["psnr"][label],
                metric_font_size=5.0,
                manifest=manifest,
                metadata={
                    "figure": "table1_qualitative_comparison",
                    "panel": "a_gaussian_reconstruction",
                    "row": row_index + 1,
                    "row_label": row_labels[row_index].replace("\n", " | "),
                    "column": column_index + 1,
                    "column_label": label,
                    "sample_id": sample["sample_id"],
                    "scene": sample["scene"],
                    "frame_id": sample["frame_id"],
                    "raw_overlap": sample["raw_overlap"],
                    "raw_hole_ratio": sample["raw_hole_ratio"],
                    "method_degenerate": spec["degenerate"][label],
                    "selection_rule": "100th retained pair in lexicographically first/last iMED validation scene",
                },
            )

    for row_index, spec in enumerate(completion_specs):
        display_size = target_size(spec["target"])
        sample = spec["sample"]
        for column_index, label in enumerate(completion_columns):
            ax = fig.add_subplot(bottom_grid[row_index, column_index])
            bottom_axes[row_index, column_index] = ax
            plot_cell(
                ax,
                image_path=spec["paths"][label],
                display_size=display_size,
                title=label if row_index == 0 else None,
                title_size=6.0,
                proposed=label == "DistriNVS",
                target=label == "Target RGB",
                row_label=row_labels[row_index] if column_index == 0 else None,
                psnr_db=spec["psnr"][label],
                metric_font_size=5.8,
                manifest=manifest,
                metadata={
                    "figure": "table1_qualitative_comparison",
                    "panel": "b_image_completion",
                    "row": row_index + 1,
                    "row_label": row_labels[row_index].replace("\n", " | "),
                    "column": column_index + 1,
                    "column_label": label,
                    "sample_id": sample["sample_id"],
                    "scene": sample["scene"],
                    "frame_id": sample["frame_id"],
                    "raw_overlap": sample["raw_overlap"],
                    "raw_hole_ratio": sample["raw_hole_ratio"],
                    "method_degenerate": spec["degenerate"][label],
                    "selection_rule": "100th retained pair in lexicographically first/last iMED validation scene",
                },
            )

    fig.canvas.draw()
    add_panel_heading(fig, top_axes, "(a) Gaussian reconstruction")
    add_panel_heading(fig, bottom_axes, "(b) Image synthesis and completion")
    output_base = output_dir / "table1_qualitative_comparison"
    save_figure(fig, output_base)
    return output_base, manifest, selected


def draw_table2(output_dir: Path) -> tuple[Path, list[dict[str, Any]], list[dict[str, str]]]:
    combined_rows = read_csv(TABLE2_COMBINED)
    primary_rows = read_csv(TABLE2_PRIMARY)
    distrinvs_rows = read_csv(TABLE2_DISTRINVS)
    input_rows = read_jsonl(TABLE2_INPUTS)
    selected = select_table2(primary_rows, combined_rows, distrinvs_rows)

    combined_index = make_index(combined_rows, ("method_key", "sample_id"), TABLE2_COMBINED)
    distrinvs_index = make_index(distrinvs_rows, ("sample_id",), TABLE2_DISTRINVS)
    canonical_samples = {row["sample_id"] for row in primary_rows}
    distrinvs_samples = {row["sample_id"] for row in distrinvs_rows}
    if distrinvs_samples != canonical_samples:
        raise ValueError(
            "Table-II DistriNVS samples do not match the frozen primary manifest: "
            f"missing={len(canonical_samples - distrinvs_samples)}, "
            f"extra={len(distrinvs_samples - canonical_samples)}"
        )
    input_index = make_index(
        ({**row, "sample_id": f"{row['scene']}/{row['name']}"} for row in input_rows),
        ("sample_id",),
        TABLE2_INPUTS,
    )
    method_keys = {
        "LDM": "ldm",
        "LaMa": "lama",
        "LCM-LoRA": "lcm_lora",
        "Hunyuan-DiT": "hunyuan",
    }
    columns = [
        "Source RGB",
        "DSS warp",
        *method_keys.keys(),
        "DistriNVS",
        "Target RGB",
    ]
    row_labels = []
    for sample in selected:
        row_labels.append(
            f"{100.0 * float(sample['raw_hole_ratio']):.1f}% hole"
        )

    fig = plt.figure(figsize=(7.16, 1.47))
    grid = fig.add_gridspec(
        len(selected),
        len(columns),
        left=0.073,
        right=0.995,
        bottom=0.025,
        top=0.925,
        wspace=0.022,
        hspace=0.018,
    )
    manifest: list[dict[str, Any]] = []
    axes = np.empty((len(selected), len(columns)), dtype=object)

    for row_index, sample in enumerate(selected):
        sample_id = sample["sample_id"]
        input_row = require(input_index, (sample_id,), "Table-II input")
        ldm_row = require(combined_index, ("ldm", sample_id), "Table-II LDM")
        warp_manifest_path = Path(input_row["warped_rgb"]).parent.parent / "warp_manifest.json"
        warp_manifest = json.loads(warp_manifest_path.read_text(encoding="utf-8"))
        source_frame = next(
            (
                frame
                for frame in warp_manifest["frames"]
                if int(frame["frame_id"]) == frame_number(sample)
            ),
            None,
        )
        if source_frame is None:
            raise KeyError(f"Missing source RGB for {sample_id} in {warp_manifest_path}")
        paths: dict[str, Path] = {
            "Source RGB": Path(source_frame["source_rgb"]),
            "DSS warp": Path(input_row["warped_rgb"]),
        }
        degenerate: dict[str, str] = {"Source RGB": "", "DSS warp": ""}
        psnr: dict[str, float | None] = {"Source RGB": None, "DSS warp": None}
        for label, method_key in method_keys.items():
            method_row = require(combined_index, (method_key, sample_id), label)
            paths[label] = Path(method_row["prediction_path"])
            degenerate[label] = method_row.get("degenerate", "")
            psnr[label] = float(method_row["psnr"])
        distrinvs_row = require(distrinvs_index, (sample_id,), "DistriNVS")
        paths["DistriNVS"] = Path(distrinvs_row["prediction_path"])
        degenerate["DistriNVS"] = distrinvs_row.get("degenerate", "")
        psnr["DistriNVS"] = float(distrinvs_row["psnr"])
        paths["Target RGB"] = Path(ldm_row["target_path"])
        degenerate["Target RGB"] = ""
        psnr["Target RGB"] = None
        display_size = target_size(paths["Target RGB"])

        for column_index, label in enumerate(columns):
            ax = fig.add_subplot(grid[row_index, column_index])
            axes[row_index, column_index] = ax
            plot_cell(
                ax,
                image_path=paths[label],
                display_size=display_size,
                title=label if row_index == 0 else None,
                title_size=6.35,
                proposed=label == "DistriNVS",
                target=label == "Target RGB",
                row_label=row_labels[row_index] if column_index == 0 else None,
                psnr_db=psnr[label],
                metric_font_size=5.8,
                manifest=manifest,
                metadata={
                    "figure": "table2_zeroshot_qualitative_comparison",
                    "panel": "endovis_zero_shot",
                    "row": row_index + 1,
                    "row_label": row_labels[row_index].replace("\n", " | "),
                    "column": column_index + 1,
                    "column_label": label,
                    "sample_id": sample_id,
                    "scene": sample["scene"],
                    "frame_id": sample["frame_id"],
                    "raw_overlap": sample["raw_overlap"],
                    "raw_hole_ratio": sample["raw_hole_ratio"],
                    "method_degenerate": degenerate[label],
                    "selection_margin_psnr": sample["selection_margin_psnr"],
                    "selection_best_competitor": sample["selection_best_competitor"],
                    "selection_best_competitor_psnr": sample[
                        "selection_best_competitor_psnr"
                    ],
                    "selection_rule": "largest positive DistriNVS PSNR margin per dataset, raw hole >= 0.10",
                },
                resize_mode="bilinear" if label == "Source RGB" else "lanczos",
            )

    output_base = output_dir / "table2_zeroshot_qualitative_comparison"
    save_figure(fig, output_base)
    return output_base, manifest, selected


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for manifest {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def check_manifest(rows: list[dict[str, Any]], expected: int, label: str) -> None:
    if len(rows) != expected:
        raise ValueError(f"{label}: expected {expected} panels, found {len(rows)}")
    missing = [row["source_path"] for row in rows if not Path(row["source_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{label}: {len(missing)} missing sources; first={missing[0]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/data/mashixing/icra2027_visuals_stage/output"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    font_path = configure_matplotlib()
    table1_base, table1_manifest, table1_selection = draw_table1(args.output_dir)
    table2_base, table2_manifest, table2_selection = draw_table2(args.output_dir)
    check_manifest(table1_manifest, 30, "Table-I qualitative figure")
    check_manifest(table2_manifest, 16, "Table-II qualitative figure")
    write_manifest(args.output_dir / "table1_qualitative_sources.csv", table1_manifest)
    write_manifest(args.output_dir / "table2_qualitative_sources.csv", table2_manifest)

    provenance = {
        "generator": str(Path(__file__).resolve()),
        "backend": "Python/matplotlib",
        "matplotlib_version": mpl.__version__,
        "font_family": "Times New Roman",
        "font_file": font_path,
        "color_semantics": {
            TEAL: "DistriNVS identifier only; not a best-method or ranking cue"
        },
        "processing": "RGB display; Table-II source RGB uses target-size bilinear resize; other panels use target-size Lanczos only when required; no crop; no per-image enhancement",
        "table1": {
            "output_base": str(table1_base),
            "selection_rule": "100th retained pair in lexicographically first and last iMED validation scenes",
            "selected_samples": [row["sample_id"] for row in table1_selection],
        },
        "table2": {
            "output_base": str(table2_base),
            "distrinvs_frame_metrics": str(TABLE2_DISTRINVS),
            "selection_rule": "largest positive DistriNVS PSNR margin per dataset with raw hole >= 0.10",
            "selected_samples": [row["sample_id"] for row in table2_selection],
        },
    }
    (args.output_dir / "qualitative_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
