#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

# ============================================================
# Input / Output
# ============================================================
csv_path = Path(
    "/home/data/mashixing/dataset_18tb/icra-2027/latex/figures/"
    "hole_ratio/binned_metrics.csv"
)

png_path = Path(
    "/home/data/mashixing/dataset_18tb/icra-2027/latex/figures/"
    "hole_ratio/hole_ratio_boxplot.png"
)

pdf_path = Path(
    "/home/data/mashixing/dataset_18tb/icra-2027/latex/figures/"
    "hole_ratio/hole_ratio_boxplot.pdf"
)

# ============================================================
# Load data
#
# Expected columns:
# method, bin, n, metric, mean, ci95
# ============================================================
df = pd.read_csv(csv_path)

# Recover SD from stored 95% CI:
#
# ci95 = 1.96 * SD / sqrt(n)
#
# therefore:
#
# SD = ci95 * sqrt(n) / 1.96
#
df["sd"] = df["ci95"] * np.sqrt(df["n"]) / 1.96


# ============================================================
# Plot settings
# ============================================================
bin_order = [
    "0–10",
    "10–20",
    "20–30",
    "30–40",
    ">40",
]

methods = [
    "DSS warp",
    "LaMa",
    "LCM-LoRA",
    "DistriNVS (SF)",
]

colors = {
    "DSS warp": "#1f77b4",        # blue
    "LaMa": "#ff7f0e",            # orange
    "LCM-LoRA": "#2ca02c",        # green
    "DistriNVS (SF)": "#d62728",  # red
}


# ============================================================
# Matplotlib global style
# ============================================================
plt.rcParams.update({

    # --------------------------------------------------------
    # Font
    # --------------------------------------------------------
    "font.family": "Times New Roman",

    "font.size": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10.5,

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------
    "axes.linewidth": 1.0,

    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,

    "xtick.major.size": 4,
    "ytick.major.size": 4,

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------
    "figure.dpi": 220,

    # --------------------------------------------------------
    # PDF font embedding
    # --------------------------------------------------------
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Approximate boxplot statistics
# ============================================================
#
# Current CSV only contains:
# mean, CI95, n
#
# It does NOT contain true:
# Q1, median, Q3
#
# We therefore assume an approximately Gaussian distribution.
#
# For a Gaussian:
#
# Q1 = mean - 0.67449 * SD
# Q3 = mean + 0.67449 * SD
#
# Then standard boxplot whiskers are:
#
# lower = Q1 - 1.5 * IQR
# upper = Q3 + 1.5 * IQR
#
z_q = 0.67448975


# ============================================================
# X-axis positions
# ============================================================
x_centers = np.arange(len(bin_order), dtype=float)

# Four methods within each source-unobserved-ratio group
offsets = {
    "DSS warp": -0.27,
    "LaMa": -0.09,
    "LCM-LoRA": 0.09,
    "DistriNVS (SF)": 0.27,
}

box_width = 0.14


# ============================================================
# Create figure
# ============================================================
fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.8, 4.15),
    constrained_layout=True,
)

fig.patch.set_facecolor("white")


specs = [
    (
        "psnr",
        "PSNR (dB) ↑",
        "(a)",
    ),
    (
        "lpips",
        "LPIPS ↓",
        "(b)",
    ),
]


# ============================================================
# Draw panels
# ============================================================
for ax, (metric, ylabel, panel) in zip(axes, specs):

    ax.set_facecolor("white")

    # --------------------------------------------------------
    # No background grid
    # --------------------------------------------------------
    ax.grid(False)

    # --------------------------------------------------------
    # Draw all four methods
    # --------------------------------------------------------
    for method in methods:

        sub = df[
            (df["method"] == method)
            & (df["metric"] == metric)
        ].copy()

        sub["bin"] = pd.Categorical(
            sub["bin"],
            categories=bin_order,
            ordered=True,
        )

        sub = sub.sort_values("bin")

        # ----------------------------------------------------
        # Draw one box for every bin
        # ----------------------------------------------------
        for i, row in enumerate(sub.itertuples()):

            mu = float(row.mean)
            sd = float(row.sd)

            # Gaussian approximation of quartiles
            q1 = mu - z_q * sd
            q3 = mu + z_q * sd

            iqr = q3 - q1

            # Standard 1.5×IQR whiskers
            whislo = q1 - 1.5 * iqr
            whishi = q3 + 1.5 * iqr

            pos = x_centers[i] + offsets[method]

            stats = [{
                "med": mu,
                "q1": q1,
                "q3": q3,
                "whislo": whislo,
                "whishi": whishi,
                "fliers": [],
            }]

            artists = ax.bxp(
                stats,
                positions=[pos],
                widths=box_width,
                showfliers=False,
                patch_artist=True,
                manage_ticks=False,
                zorder=3,
            )

            # ------------------------------------------------
            # Box
            # ------------------------------------------------
            for box in artists["boxes"]:

                box.set_facecolor(colors[method])
                box.set_edgecolor(colors[method])

                box.set_alpha(0.58)
                box.set_linewidth(1.3)

            # ------------------------------------------------
            # Median
            # ------------------------------------------------
            for median in artists["medians"]:

                median.set_color("white")
                median.set_linewidth(1.8)

            # ------------------------------------------------
            # Whiskers
            # ------------------------------------------------
            for whisker in artists["whiskers"]:

                whisker.set_color(colors[method])
                whisker.set_linewidth(1.15)

            # ------------------------------------------------
            # Whisker caps
            # ------------------------------------------------
            for cap in artists["caps"]:

                cap.set_color(colors[method])
                cap.set_linewidth(1.15)

            # ------------------------------------------------
            # Mean marker
            #
            # Here mean = median in the Gaussian approximation.
            # Diamond marker mainly improves visual readability.
            # ------------------------------------------------
            ax.scatter(
                pos,
                mu,
                s=19,
                marker="D",
                facecolor=colors[method],
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
            )

    # ========================================================
    # X axis
    # ========================================================
    ax.set_xticks(
        x_centers,
        bin_order,
    )

    ax.set_xlabel(
        "Source-unobserved ratio (%)"
    )

    # ========================================================
    # Y axis
    # ========================================================
    ax.set_ylabel(
        ylabel
    )

    # Slight horizontal margin
    ax.margins(x=0.05)

    # ========================================================
    # Full rectangular border
    # ========================================================
    for side in [
        "top",
        "right",
        "bottom",
        "left",
    ]:

        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_color("black")

    # Tick style
    ax.tick_params(
        axis="both",
        direction="out",
        width=1.0,
    )

    # ========================================================
    # Panel label
    # ========================================================
    ax.text(
        -0.11,
        1.02,
        panel,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="bottom",
    )


# ============================================================
# Y-axis limits
#
# Use approximated whisker ranges
# ============================================================
for ax, metric in zip(
    axes,
    ["psnr", "lpips"],
):

    sub = df[
        df["metric"] == metric
    ].copy()

    q1 = (
        sub["mean"]
        - z_q * sub["sd"]
    )

    q3 = (
        sub["mean"]
        + z_q * sub["sd"]
    )

    iqr = q3 - q1

    lo = (
        q1
        - 1.5 * iqr
    ).min()

    hi = (
        q3
        + 1.5 * iqr
    ).max()

    if metric == "psnr":
        margin = 0.35
    else:
        margin = 0.012

    ax.set_ylim(
        lo - margin,
        hi + margin,
    )


# ============================================================
# Shared legend
# ============================================================
legend_handles = [

    Patch(
        facecolor=colors[method],
        edgecolor=colors[method],
        alpha=0.58,
        label=method,
    )

    for method in methods
]


fig.legend(
    handles=legend_handles,

    loc="upper center",

    bbox_to_anchor=(
        0.5,
        1.06,
    ),

    ncol=4,

    frameon=False,

    columnspacing=1.8,

    handlelength=1.5,
)


# ============================================================
# Save
# ============================================================
png_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    png_path,
    dpi=400,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()


print(
    f"Saved PNG: {png_path}"
)

print(
    f"Saved PDF: {pdf_path}"
)