#!/usr/bin/env python3
"""Build the DSS evidence and soft-fusion routing figure used by the manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


SCRIPT_DIR = Path(__file__).resolve().parent
EXPECTED_SAMPLE = "session_004_scene_2_tool_3/frame_000002"
EXPECTED_RAW_OVERLAP = 0.4231201112270355
S_MIN = 0.03
TIMES_NEW_ROMAN_FILENAMES = (
    "Times.TTF",
    "Timesbd.TTF",
    "Timesi.TTF",
    "Timesbi.TTF",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=SCRIPT_DIR / "dss_evidence_data.npz",
        help="Exact same-frame DSS evidence and effective soft-fusion gate.",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument(
        "--font-dir",
        type=Path,
        default=Path.home() / ".local" / "share" / "fonts" / "msttcorefonts",
        help=(
            "Directory containing Times New Roman regular, bold, italic, "
            "and bold-italic TTF files."
        ),
    )
    return parser.parse_args()


def configure_times_new_roman(font_dir: Path) -> None:
    """Register genuine Times New Roman faces and reject font substitution."""
    font_paths = [font_dir / filename for filename in TIMES_NEW_ROMAN_FILENAMES]
    missing = [str(path) for path in font_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"required Times New Roman font files are missing: {missing}"
        )
    for path in font_paths:
        font_manager.fontManager.addfont(path)
        family = font_manager.FontProperties(fname=path).get_name()
        if family != "Times New Roman":
            raise ValueError(
                f"{path} identifies as {family!r}, not 'Times New Roman'"
            )

    resolved = font_manager.findfont(
        font_manager.FontProperties(family="Times New Roman"),
        fallback_to_default=False,
    )
    if font_manager.FontProperties(fname=resolved).get_name() != "Times New Roman":
        raise RuntimeError(f"Times New Roman resolved unexpectedly to {resolved}")

    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.sf": "Times New Roman",
            "mathtext.cal": "Times New Roman:italic",
            "mathtext.tt": "Times New Roman",
            "mathtext.fallback": None,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_panel_frame(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#262626")
        spine.set_linewidth(0.45)


def main() -> None:
    args = arguments()
    configure_times_new_roman(args.font_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = np.load(args.cache)
    required = {
        "warped_rgb",
        "support",
        "completed_valid_mask",
        "synthesis_gate",
        "sample_id",
        "hard_composition",
        "min_support",
        "raw_overlap",
    }
    missing = sorted(required.difference(cache.files))
    if missing:
        raise KeyError(f"cache is missing required arrays: {missing}")
    sample_id = str(np.asarray(cache["sample_id"]).item())
    if sample_id != EXPECTED_SAMPLE:
        raise ValueError(f"unexpected sample: {sample_id}")
    if bool(np.asarray(cache["hard_composition"]).item()):
        raise ValueError("Figure 3 requires the no-hard/soft-fusion checkpoint")
    min_support = float(np.asarray(cache["min_support"]).item())
    if not np.isclose(min_support, S_MIN, atol=1.0e-8, rtol=0.0):
        raise ValueError(f"cache S_min={min_support} differs from figure S_min={S_MIN}")
    raw_overlap = float(np.asarray(cache["raw_overlap"]).item())
    if not np.isclose(raw_overlap, EXPECTED_RAW_OVERLAP, atol=1.0e-6, rtol=0.0):
        raise ValueError(
            f"cache raw overlap {raw_overlap:.9f} differs from expected "
            f"{EXPECTED_RAW_OVERLAP:.9f}"
        )

    warp = np.asarray(cache["warped_rgb"], dtype=np.uint8)
    support = np.squeeze(cache["support"]).astype(np.float32)
    valid = np.squeeze(cache["completed_valid_mask"]).astype(bool)
    gate = np.squeeze(cache["synthesis_gate"]).astype(np.float32)
    expected = support >= S_MIN
    if (
        support.shape != (512, 640)
        or valid.shape != support.shape
        or gate.shape != support.shape
        or warp.shape != (512, 640, 3)
    ):
        raise ValueError(
            "unexpected cached shapes: "
            f"warp={warp.shape}, support={support.shape}, mask={valid.shape}, gate={gate.shape}"
        )
    if not np.isfinite(support).all() or not np.isfinite(gate).all():
        raise ValueError("support and gate must be finite")
    if not np.array_equal(valid, expected):
        mismatch = float(np.mean(valid != expected))
        raise ValueError(f"cached mask disagrees with S >= S_min at {mismatch:.6%} of pixels")
    if float(gate.min()) < -1.0e-6 or float(gate.max()) > 1.0 + 1.0e-6:
        raise ValueError(f"gate outside [0, 1]: [{gate.min()}, {gate.max()}]")

    vmax = float(np.quantile(support[np.isfinite(support)], 0.99))
    # Four aligned evidence maps are shown as one single-column image plate.
    # The narrow inter-panel gaps make pixelwise spatial correspondences easy
    # to compare without merging the framed panels visually.
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(3.45, 1.15),
        constrained_layout=False,
    )

    axes[0].imshow(warp)
    axes[0].set_title(r"(a) Warp $I_t^w$", fontsize=5.4, pad=1.2)

    axes[1].imshow(support, cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest")
    axes[1].contour(
        expected.astype(np.uint8),
        levels=[0.5],
        colors=["#4DD0E1"],
        linewidths=0.45,
    )
    axes[1].set_title(r"(b) Support $S$", fontsize=5.4, pad=1.2)

    axes[2].imshow(valid, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title(r"(c) Validity $M^{\mathrm{dss}}$", fontsize=5.4, pad=1.2)

    gate_image = axes[3].imshow(
        gate,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[3].set_title(r"(d) Gate $g$", fontsize=5.4, pad=1.2)

    for ax in axes:
        add_panel_frame(ax)
    figure.subplots_adjust(
        left=0.004,
        right=0.996,
        bottom=0.140,
        top=0.870,
        wspace=0.018,
    )
    gate_position = axes[3].get_position()
    colorbar_axis = figure.add_axes(
        [
            gate_position.x0 + 0.03 * gate_position.width,
            gate_position.y0 - 0.050,
            0.94 * gate_position.width,
            0.025,
        ]
    )
    colorbar = figure.colorbar(
        gate_image,
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=[0.0, 0.5, 1.0],
    )
    colorbar.ax.tick_params(labelsize=4.5, length=1.3, width=0.4, pad=0.7)
    colorbar.outline.set_linewidth(0.45)

    stem = args.output_dir / "distrinvs_dss_evidence"
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.01)
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.01)
    figure.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.01,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)
    print(f"wrote {stem}.{{svg,pdf,png,tiff}}")
    print(f"raw overlap={raw_overlap:.6f}; raw hole ratio={1.0 - raw_overlap:.6f}")
    print(f"S_min={S_MIN:.2f}; valid ratio={valid.mean():.6f}; display vmax(q99)={vmax:.6f}")
    print(
        f"gate range=[{gate.min():.6f}, {gate.max():.6f}]; "
        f"mean={gate.mean():.6f}"
    )


if __name__ == "__main__":
    main()
