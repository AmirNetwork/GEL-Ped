# Author: Amir Ghorbani
"""Publication figures for the geometry-guarded GEL-Ped revision."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


COLORS = {
    "navy": "#17324D",
    "blue": "#2F6BFF",
    "teal": "#00989D",
    "orange": "#E67E22",
    "red": "#C73E3A",
    "grey": "#6B7785",
    "light": "#EEF3F8",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.2,
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def box(ax, xy, width, height, text, *, color, fill="white", fontsize=9.5):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.6,
        edgecolor=color,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["navy"],
    )


def arrow(ax, start, end, *, color=COLORS["grey"], width=1.4):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=width,
            color=color,
            connectionstyle="arc3,rad=0",
        )
    )


def architecture(destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(
        ax,
        (0.02, 0.38),
        0.15,
        0.24,
        "0.8-s observed history\npositions + velocities",
        color=COLORS["navy"],
        fill="#F7F9FC",
    )
    box(
        ax,
        (0.22, 0.66),
        0.20,
        0.22,
        "Interaction graph\nshared edge encoder\n+ attention pooling",
        color=COLORS["blue"],
        fill="#EDF3FF",
    )
    box(
        ax,
        (0.22, 0.12),
        0.20,
        0.22,
        "Prospective geometry\ntime, clearance,\npassing side",
        color=COLORS["orange"],
        fill="#FFF5E9",
    )
    box(
        ax,
        (0.47, 0.66),
        0.17,
        0.22,
        "Graph residual\nexpert",
        color=COLORS["blue"],
        fill="#EDF3FF",
    )
    box(
        ax,
        (0.47, 0.12),
        0.17,
        0.22,
        "Geometry residual\nexpert",
        color=COLORS["orange"],
        fill="#FFF5E9",
    )
    box(
        ax,
        (0.69, 0.38),
        0.15,
        0.24,
        "Causal guard\n2-s rolling\ndisagreement",
        color=COLORS["red"],
        fill="#FFF0EF",
    )
    box(
        ax,
        (0.88, 0.38),
        0.10,
        0.24,
        "Current velocity\n+ guarded\ncorrection",
        color=COLORS["teal"],
        fill="#ECFAFA",
        fontsize=9,
    )
    arrow(ax, (0.17, 0.51), (0.22, 0.76))
    arrow(ax, (0.17, 0.49), (0.22, 0.23))
    arrow(ax, (0.42, 0.77), (0.47, 0.77), color=COLORS["blue"])
    arrow(ax, (0.42, 0.23), (0.47, 0.23), color=COLORS["orange"])
    arrow(ax, (0.64, 0.77), (0.72, 0.61), color=COLORS["blue"])
    arrow(ax, (0.64, 0.23), (0.72, 0.39), color=COLORS["orange"])
    arrow(ax, (0.84, 0.50), (0.88, 0.50), color=COLORS["teal"])
    ax.text(
        0.555,
        0.95,
        "learn a correction, not the complete motion",
        ha="center",
        va="center",
        color=COLORS["navy"],
        fontsize=10,
        fontstyle="italic",
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def evidence(processed: Path, destination: Path) -> None:
    summary = pd.read_csv(processed / "causal_anchor_summary.csv")
    run_seed = pd.read_csv(processed / "causal_anchor_run_seed_mean.csv")
    gates = pd.read_csv(processed / "causal_router_realized_weights.csv")
    diagnostics = pd.read_csv(processed / "causal_buffer_diagnostics.csv")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    ax = axes[0, 0]
    regimes = ["familiar corridor", "altered geometry", "crossing topology"]
    models = [
        "constant velocity",
        "direct MLP",
        "graph interaction network",
        "matched direct control",
        "GEL-Ped",
    ]
    labels = ["CV", "Direct MLP", "Graph", "Matched direct", "GEL-Ped"]
    palette = ["#ADB5BD", "#7A8794", COLORS["blue"], COLORS["orange"], COLORS["teal"]]
    x = np.arange(len(regimes))
    width = 0.15
    for index, (model, label, color) in enumerate(zip(models, labels, palette)):
        part = summary[summary.model.eq(model)].set_index("dataset").loc[regimes]
        ax.bar(
            x + (index - 2) * width,
            part.mean_rmse_mps,
            width,
            yerr=part.run_sd_rmse_mps / np.sqrt(part.runs),
            capsize=2,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xticks(x, ["Familiar", "Altered", "Crossing"])
    ax.set_ylabel("Vector RMSE (m/s)")
    ax.set_title("A  Complete-run prediction error")
    ax.grid(axis="y", color="#D7DEE7", linewidth=0.6)
    ax.legend(ncol=3, frameon=False, loc="upper left")

    ax = axes[0, 1]
    cross = run_seed[run_seed.dataset.eq("crossing topology")]
    pivot = cross.pivot(index="run", columns="model", values="vector_rmse_mps")
    difference = pivot["GEL-Ped"] - pivot["graph interaction network"]
    order = np.argsort(difference.to_numpy())
    values = difference.to_numpy()[order]
    colors = np.where(values < 0, COLORS["teal"], COLORS["red"])
    ax.bar(np.arange(len(values)) + 1, values * 1000, color=colors, width=0.72)
    ax.axhline(0, color=COLORS["navy"], linewidth=0.8)
    ax.set_xlabel("Crossing run (ordered)")
    ax.set_ylabel("GEL-Ped $-$ graph RMSE (mm/s)")
    ax.set_title("B  GEL-Ped improves all 13 crossing runs")
    ax.grid(axis="y", color="#D7DEE7", linewidth=0.6)

    ax = axes[1, 0]
    gate_order = ["calibration", *regimes]
    gate_labels = ["Calibration", "Familiar", "Altered", "Crossing"]
    for index, dataset in enumerate(gate_order):
        values = gates.loc[gates.dataset.eq(dataset), "mean_g"].to_numpy()
        jitter = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=26,
            color=COLORS["teal"] if dataset != "crossing topology" else COLORS["red"],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        ax.hlines(np.mean(values), index - 0.22, index + 0.22, color=COLORS["navy"], linewidth=2)
    ax.axhline(0.70, linestyle="--", color=COLORS["grey"], linewidth=1, label="nominal graph weight")
    ax.set_xticks(range(4), gate_labels)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("Realised graph weight $g$")
    ax.set_title("C  The guard activates under topology shift")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="y", color="#D7DEE7", linewidth=0.6)

    ax = axes[1, 1]
    crossing = diagnostics[diagnostics.dataset.eq("crossing topology")].copy()
    crossing["bin"] = pd.qcut(crossing.disagreement_mps, 10, duplicates="drop")
    binned = crossing.groupby("bin", observed=True).agg(
        disagreement=("disagreement_mps", "mean"),
        error_gap=("in_support_minus_transfer_mse", "mean"),
        error_se=("in_support_minus_transfer_mse", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
    )
    ax.errorbar(
        binned.disagreement,
        binned.error_gap * 1000,
        yerr=binned.error_se * 1000,
        marker="o",
        markersize=4.5,
        color=COLORS["red"],
        linewidth=1.5,
        capsize=2,
    )
    ax.axhline(0, color=COLORS["navy"], linewidth=0.8)
    ax.set_xlabel("Expert disagreement (m/s)")
    ax.set_ylabel("Graph $-$ geometry MSE ($10^{-3}$ m$^2$/s$^2$)")
    ax.set_title("D  Disagreement identifies when geometry is safer")
    ax.grid(color="#D7DEE7", linewidth=0.6)

    for ax in axes.flat:
        ax.tick_params(length=3, color=COLORS["grey"])
    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    fig.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    style()
    project = Path(__file__).resolve().parents[1]
    destination = project / "manuscript/overleaf_trc/figures"
    processed = project / "data/processed"
    architecture(destination / "gel_ped_guard_architecture.png")
    evidence(processed, destination / "causal_revision_results.png")


if __name__ == "__main__":
    main()
