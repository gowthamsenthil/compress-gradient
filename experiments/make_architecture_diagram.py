"""Render the experiment-setup architecture diagram.

Produces results/fig0_architecture.png:

  - 12 workers arranged in a ring, color-coded by bandwidth tier.
  - Each worker shows its tier index, bandwidth, and latency.
  - Inset: MLP architecture (784-128-128-10) with parameter counts.
  - Inset: per-round timeline showing local compute -> compress -> ring all-reduce -> apply.
  - Annotation: the round-time formula T_round = max_i T_i.

Self-contained matplotlib; no other dependencies beyond what the project already uses.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "fig0_architecture.png"

# -- tier definition (must match run_bound_refinements.py 3tier) -------------
TIERS = [
    {"name": "Fast",   "bw": 10.0, "lat_us": 5,  "count": 6, "color": "#2ca02c",
     "real": "NVLink / intra-rack 100GbE"},
    {"name": "Medium", "bw":  5.0, "lat_us": 20, "count": 3, "color": "#1f77b4",
     "real": "Cross-rack DC Ethernet"},
    {"name": "Slow",   "bw":  1.0, "lat_us": 50, "count": 3, "color": "#d62728",
     "real": "Cross-region cloud"},
]
N = sum(t["count"] for t in TIERS)  # 12


def _worker_colors():
    """Return list of N tier-color strings, deterministically ordered."""
    out = []
    for t in TIERS:
        out.extend([t["color"]] * t["count"])
    return out


def _worker_labels():
    """Return list of N (tier_short, bw_text) tuples."""
    out = []
    for t in TIERS:
        for _ in range(t["count"]):
            out.append((t["name"][0], f"{t['bw']:.0f} Gbps"))
    return out


# ---------------------------------------------------------------------------

def draw_ring(ax):
    """Draw the N-worker ring topology."""
    R = 3.0
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, N, endpoint=False)
    xs = R * np.cos(angles)
    ys = R * np.sin(angles)
    colors = _worker_colors()
    labels = _worker_labels()

    # Ring edges (drawn first so node patches sit on top)
    for i in range(N):
        j = (i + 1) % N
        ax.annotate(
            "", xy=(xs[j], ys[j]), xytext=(xs[i], ys[i]),
            arrowprops=dict(
                arrowstyle="->", color="#666666", lw=1.4,
                connectionstyle="arc3,rad=0.10",
                shrinkA=18, shrinkB=18,
            ),
            zorder=1,
        )

    # Worker nodes
    for i, (x, y, c, (tier_short, bw_txt)) in enumerate(zip(xs, ys, colors, labels)):
        circle = plt.Circle((x, y), 0.42, facecolor=c, edgecolor="black",
                            linewidth=1.2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y + 0.05, f"W{i}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=4)
        ax.text(x, y - 0.18, tier_short, ha="center", va="center",
                fontsize=8, color="white", zorder=4)
        # Bandwidth tag outside the ring
        rx, ry = x * 1.20, y * 1.20
        ax.text(rx, ry, bw_txt, ha="center", va="center",
                fontsize=8, color=c, fontweight="bold", zorder=2)

    # Center label
    ax.text(0, 0.45, "Ring all-reduce",
            ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0, 0.05,
            r"$T_{round} = \max_i \, T_i$",
            ha="center", va="center", fontsize=12)
    ax.text(0, -0.45, "(slowest worker dictates pace)",
            ha="center", va="center", fontsize=9, style="italic", color="#444")

    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-4.4, 4.4)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_mlp_inset(ax):
    """Draw the MLP architecture inset (784 -> 128 -> 128 -> 10)."""
    layers = [
        ("Input\n784",            "#cccccc"),
        ("FC1+ReLU\n100,480 prm", "#88c0d0"),
        ("FC2+ReLU\n16,512 prm",  "#81a1c1"),
        ("Output\n1,290 prm",     "#5e81ac"),
    ]
    y0 = 0.82
    h = 0.12
    box_w = 0.20
    gap = 0.025
    total_w = len(layers) * box_w + (len(layers) - 1) * gap
    x0 = 0.5 - total_w / 2
    for i, (lab, c) in enumerate(layers):
        x = x0 + i * (box_w + gap)
        rect = FancyBboxPatch((x, y0 - h), box_w, h,
                              boxstyle="round,pad=0.005",
                              facecolor=c, edgecolor="black", linewidth=0.8,
                              transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y0 - h / 2, lab,
                ha="center", va="center", fontsize=8, transform=ax.transAxes)
        if i < len(layers) - 1:
            ax.annotate("",
                        xy=(x + box_w + gap, y0 - h / 2),
                        xytext=(x + box_w, y0 - h / 2),
                        xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", lw=1, color="#444"))
    ax.text(0.5, y0 + 0.025, "Per-worker MLP replica  (784-128-128-10)",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, fontweight="bold")
    ax.text(0.5, y0 - h - 0.04,
            "118,282 params  |  3.78 Mb / gradient (32-bit floats)",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color="#333", style="italic")


def draw_round_pipeline(ax):
    """Draw the per-round pipeline as a horizontal swimlane below the ring."""
    stages = [
        ("Local\nforward+\nbackward",   "#fde9c4"),
        ("Top-K\ncompress\n(+ EF)", "#f4cae4"),
        ("Ring\nall-reduce",          "#cbd5e8"),
        ("Apply\navg grad",  "#b3e2cd"),
    ]
    y0 = 0.30
    h = 0.20
    box_w = 0.21
    gap = 0.015
    total_w = len(stages) * box_w + (len(stages) - 1) * gap
    x0 = 0.5 - total_w / 2
    for i, (lab, c) in enumerate(stages):
        x = x0 + i * (box_w + gap)
        rect = FancyBboxPatch((x, y0 - h), box_w, h,
                              boxstyle="round,pad=0.005",
                              facecolor=c, edgecolor="black", linewidth=0.8,
                              transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y0 - h / 2, lab,
                ha="center", va="center", fontsize=8, transform=ax.transAxes)
        if i < len(stages) - 1:
            ax.annotate("",
                        xy=(x + box_w + gap, y0 - h / 2),
                        xytext=(x + box_w, y0 - h / 2),
                        xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", lw=1, color="#444"))
    ax.text(0.5, y0 + 0.040, "One training round (per worker)",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, fontweight="bold")
    ax.text(0.5, y0 - h - 0.04,
            r"Simulator advances clock by  $T_{round} = \frac{2(N-1)}{N} G \, \max_i \frac{r_i}{B_i} \, + \, 2(N-1)\max_i \alpha_i$",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color="#333")


def draw_legend_and_specs(ax):
    """Draw the tier legend + experimental hyperparameters."""
    # Tier legend
    handles = []
    for t in TIERS:
        handles.append(mpatches.Patch(
            color=t["color"],
            label=f"{t['name']:<7s} {t['count']}x  |  {t['bw']:>4.0f} Gbps  |  "
                  f"{t['lat_us']:>2d} us  |  {t['real']}",
        ))
    leg = ax.legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(0.02, 0.98), frameon=True,
                    title="Bandwidth tiers (3-tier cluster)",
                    fontsize=9, title_fontsize=10)
    leg.get_frame().set_edgecolor("#888")
    leg._legend_box.align = "left"

    # Hyperparams text box
    text = (
        "Training hyperparameters\n"
        "  Dataset:  MNIST (IID shards)\n"
        "  Model:    MLP 784-128-128-10\n"
        "  Optim:    SGD,  lr = 0.05\n"
        "  Batch:    64\n"
        "  Rounds:   200-300\n"
        "  Error feedback:  enabled (Top-K)\n"
        "  Recalibration:   every K = 25 rounds\n"
        "  Workers:  N = 12"
    )
    ax.text(0.02, 0.55, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8",
                      edgecolor="#888"))
    ax.axis("off")


# ---------------------------------------------------------------------------

def main():
    fig = plt.figure(figsize=(15.5, 9.5))
    # Layout: left half = ring; right half = stacked panels (mlp, pipeline, legend).
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0],
                          height_ratios=[0.9, 1.05, 1.05],
                          hspace=0.10, wspace=0.05)
    ax_ring   = fig.add_subplot(gs[:, 0])
    ax_mlp    = fig.add_subplot(gs[0, 1])
    ax_pipe   = fig.add_subplot(gs[1, 1])
    ax_specs  = fig.add_subplot(gs[2, 1])

    draw_ring(ax_ring)
    ax_mlp.axis("off")
    draw_mlp_inset(ax_mlp)
    ax_pipe.axis("off")
    draw_round_pipeline(ax_pipe)
    draw_legend_and_specs(ax_specs)

    fig.suptitle(
        "Experimental setup: 12-worker 3-tier ring all-reduce with "
        "per-worker compression",
        fontsize=14, fontweight="bold", y=0.995,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
