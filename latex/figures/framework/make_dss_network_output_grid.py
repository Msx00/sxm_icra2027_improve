#!/usr/bin/env python3
"""Create a compact 2 x 8 DSS/network-output image plate."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont, fontManager
from PIL import Image


ROOT = Path(__file__).resolve().parent
FRAMEWORK_DIR = ROOT / "dss_output_panels"
NETWORK_DIR = ROOT / "networkoutput"
OUTPUT_STEM = ROOT / "dss_network_output_grid"
EXPORT_DPI = 720
PANEL_SIZE = (640, 512)
# The full-resolution panels are embedded at about 738 ppi in the assembled
# PDF.  Paper submission processors downsample images above 450 ppi and can
# mishandle Matplotlib's 1-bit indexed encoding of an exactly binary mask.
# Pre-resampling only the validity mask to about 300 ppi introduces 8-bit edge
# levels while preserving its binary regions and prevents a second resampling.
VALID_MASK_EMBED_SIZE = (260, 208)
VALID_MASK_FILENAME = "valid_mask.png"


def configure_fonts() -> str:
    font_dir = Path("/home/data/mashixing/.local/share/fonts/msttcorefonts")
    for filename in ("Times.TTF", "Timesbd.TTF", "Timesi.TTF", "Timesbi.TTF"):
        path = font_dir / filename
        if path.is_file():
            fontManager.addfont(path)
    font_path = findfont(
        FontProperties(family="Times New Roman"), fallback_to_default=False
    )
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 6.4,
            "axes.titlesize": 6.4,
            "axes.titleweight": "regular",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": EXPORT_DPI,
            "image.interpolation": "none",
            "image.resample": False,
        }
    )
    return font_path


def panel_specs() -> list[tuple[Path, str]]:
    dss = FRAMEWORK_DIR
    net = NETWORK_DIR
    return [
        (dss / "source_rgb.png", "(a) Source RGB\n" + r"$\mathbf{I}_s\in[0,1]^{3\times H\times W}$"),
        (dss / "warped_rgb.png", "(b) Warped RGB\n" + r"$\mathbf{I}_t^w\in[0,1]^{3\times H\times W}$"),
        (dss / "warped_features_pca.png", "(c) Warped features\n" + r"$\mathbf{F}_t^w\in\mathbb{R}^{C_f\times H\times W}$"),
        (dss / "expected_depth.png", "(d) Expected depth\n" + r"$\bar{D}_t\in\mathbb{R}_+^{1\times H\times W}$"),
        (dss / "support.png", "(e) Raw support\n" + r"$S\in\mathbb{R}_+^{1\times H\times W}$"),
        (dss / "coverage.png", "(f) Coverage\n" + r"$C\in[0,1]^{1\times H\times W}$"),
        (dss / "depth_variance.png", "(g) Depth variance\n" + r"$V\in\mathbb{R}_+^{1\times H\times W}$"),
        (dss / "collision_entropy.png", "(h) Collision ambiguity\n" + r"$\mathcal{H}\in[0,1]^{1\times H\times W}$"),
        (dss / "geometry_confidence.png", "(i) Target confidence\n" + r"$\kappa\in[0,1]^{1\times H\times W}$"),
        (dss / "valid_mask.png", "(j) Render validity\n" + r"$\mathcal{M}^{\rm dss}\in\{0,1\}^{1\times H\times W}$"),
        (net / "01_I_t_syn.png", "(k) Synthesis RGB\n" + r"$\mathbf{I}_t^{\rm syn}\in[0,1]^{3\times H\times W}$"),
        (net / "02_I_t_trans.png", "(l) Transport RGB\n" + r"$\mathbf{I}_t^{\rm trans}\in[0,1]^{3\times H\times W}$"),
        (net / "03_I_t_hat.png", "(m) Fused RGB\n" + r"$\widehat{\mathbf{I}}_t\in[0,1]^{3\times H\times W}$"),
        (net / "06_D_t_syn.png", "(n) Synthesis depth\n" + r"$\mathbf{D}_t^{\rm syn}\in\mathbb{R}_+^{1\times H\times W}$"),
        (net / "04_D_t_hat.png", "(o) Fused depth\n" + r"$\widehat{\mathbf{D}}_t\in\mathbb{R}_+^{1\times H\times W}$"),
        (dss / "target_frame_000428.png", "(p) Target RGB\n" + r"$\mathbf{I}_t\in[0,1]^{3\times H\times W}$"),
    ]


def prepare_panel(path: Path, image: Image.Image) -> Image.Image:
    """Return an RGB panel that remains robust under PDF submission fixups."""
    panel = image.convert("RGB")
    if panel.size != PANEL_SIZE:
        panel = panel.resize(PANEL_SIZE, Image.Resampling.LANCZOS)

    if path.name == VALID_MASK_FILENAME:
        mask = panel.convert("L")
        low, high = mask.getextrema()
        if low == high:
            raise ValueError(f"Render-validity mask is constant: {path}")
        panel = mask.resize(
            VALID_MASK_EMBED_SIZE, Image.Resampling.LANCZOS
        ).convert("RGB")

    return panel


def main() -> None:
    font_path = configure_fonts()
    specs = panel_specs()
    paths = [path.resolve() for path, _ in specs]
    if len(paths) != 16 or len(set(paths)) != 16:
        raise RuntimeError("The plate must contain exactly 16 distinct image files.")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing panels: " + ", ".join(map(str, missing)))

    # Double-column width with compact gutters modeled on the manuscript's
    # qualitative comparison plate. Text remains vector-editable in the PDF.
    fig, axes = plt.subplots(2, 8, figsize=(7.08, 1.86), squeeze=False)
    for axis, (path, title) in zip(axes.flat, specs):
        with Image.open(path) as image:
            panel = prepare_panel(path, image)
            # ``none`` passes the original raster to vector backends instead of
            # reducing each panel to the default 100-dpi axes resolution.
            axis.imshow(panel, interpolation="none", resample=False)
        axis.set_title(title, pad=1.7)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(
        left=0.002,
        right=0.998,
        bottom=0.006,
        top=0.968,
        wspace=0.018,
        hspace=0.245,
    )
    fig.savefig(
        OUTPUT_STEM.with_suffix(".pdf"),
        dpi=EXPORT_DPI,
        bbox_inches="tight",
        pad_inches=0.005,
    )
    fig.savefig(
        OUTPUT_STEM.with_suffix(".png"),
        dpi=EXPORT_DPI,
        bbox_inches="tight",
        pad_inches=0.005,
    )
    plt.close(fig)

    manifest = {
        "layout": "2 rows x 8 columns",
        "standard_embedded_panel_size_wh": list(PANEL_SIZE),
        "render_validity_embedded_panel_size_wh": list(VALID_MASK_EMBED_SIZE),
        "render_validity_export_policy": (
            "Pre-resampled with Lanczos to approximately 300 ppi so the PDF "
            "uses 8-bit image data rather than a 1-bit indexed mask."
        ),
        "font": "Times New Roman",
        "font_file": font_path,
        "excluded_semantic_duplicate": str((NETWORK_DIR / "05_D_t_bar.png").resolve()),
        "panels": [
            {"label": chr(ord("a") + index), "title": title, "source": str(path.resolve())}
            for index, (path, title) in enumerate(specs)
        ],
    }
    OUTPUT_STEM.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
