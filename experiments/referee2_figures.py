# Author: Amir Ghorbani
"""Publication figures for the instability-aware GEL-Ped revision."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "manuscript" / "overleaf_trc" / "figures"

COLORS = {
    "graph": "#315A8C",
    "primary": "#C44E52",
    "coarse": "#4C956C",
    "median": "#8172B2",
    "cv": "#777777",
    "ink": "#20242A",
    "soft": "#E9EEF4",
}

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "STIXGeneral", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 320,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _box(ax, xy, width, height, text, *, face, edge, fontsize=8.5):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.15,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        linespacing=1.25,
    )
    return patch


def _arrow(ax, start, end, *, color="#5F6772", width=1.05, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=10,
            linewidth=width,
            color=color,
            connectionstyle="arc3,rad=0",
        )
    )


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.25))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    _box(
        ax,
        (0.015, 0.38),
        0.15,
        0.24,
        "Past 0.8 s\npositions and velocities",
        face="#F5F7FA",
        edge="#687383",
    )
    _box(
        ax,
        (0.205, 0.38),
        0.18,
        0.24,
        "Shared causal state\nroute, persistence,\npersonal space, closing",
        face="#EDF3F8",
        edge=COLORS["graph"],
    )
    _arrow(ax, (0.165, 0.50), (0.205, 0.50))

    _box(
        ax,
        (0.43, 0.66),
        0.20,
        0.19,
        "Five graph residual learners\nindependent initialisations",
        face="#E8F0FA",
        edge=COLORS["graph"],
    )
    _box(
        ax,
        (0.43, 0.15),
        0.20,
        0.19,
        "Bounded boosted residual\ncoarse behavioural fallback",
        face="#EAF5EE",
        edge=COLORS["coarse"],
    )
    _arrow(ax, (0.385, 0.52), (0.43, 0.755))
    _arrow(ax, (0.385, 0.48), (0.43, 0.245))

    _box(
        ax,
        (0.68, 0.64),
        0.14,
        0.23,
        "Graph mean\n$\\bar{\\mathbf{v}}^{G}$\n\nmember spread $e_i$",
        face="#F2F6FB",
        edge=COLORS["graph"],
    )
    _arrow(ax, (0.63, 0.755), (0.68, 0.755))
    _box(
        ax,
        (0.68, 0.12),
        0.14,
        0.23,
        "Fallback\n$\\widehat{\\mathbf{v}}^{C}$",
        face="#F1F8F3",
        edge=COLORS["coarse"],
    )
    _arrow(ax, (0.63, 0.245), (0.68, 0.235))

    _box(
        ax,
        (0.67, 0.405),
        0.16,
        0.14,
        "Past-only local guard\n$g_i=\\exp[-(E_i-\\tau)_+/S]$",
        face="#FFF3F1",
        edge=COLORS["primary"],
        fontsize=8.0,
    )
    _arrow(ax, (0.75, 0.64), (0.75, 0.545), color=COLORS["primary"])

    _box(
        ax,
        (0.855, 0.38),
        0.13,
        0.24,
        "GEL-Ped\n$g_i\\bar{\\mathbf{v}}^{G}$\n$+(1-g_i)\\widehat{\\mathbf{v}}^{C}$",
        face="#FDECE9",
        edge=COLORS["primary"],
        fontsize=8.5,
    )
    _arrow(ax, (0.83, 0.475), (0.855, 0.50), color=COLORS["primary"])
    _arrow(ax, (0.82, 0.235), (0.90, 0.38), color=COLORS["coarse"])
    _arrow(ax, (0.82, 0.755), (0.94, 0.62), color=COLORS["graph"])

    ax.text(
        0.51,
        0.955,
        "Training: seven complete runs; no frame-level split",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.745,
        0.035,
        "Leave-one-run-out calibration fixes $\\tau$ and $S$; $g_i=1$ inside the envelope",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#4D5560",
    )
    fig.savefig(OUT / "gel_ped_instability_architecture.pdf", bbox_inches="tight")
    fig.savefig(OUT / "gel_ped_instability_architecture.png", bbox_inches="tight")
    plt.close(fig)


def evidence() -> None:
    metrics = pd.read_csv(PROCESSED / "referee2_julich_metrics.csv")
    diagnostics = pd.read_csv(PROCESSED / "referee2_julich_diagnostics.csv")
    rollouts = pd.read_csv(PROCESSED / "referee2_rollout_metrics.csv")

    graph = "five-member graph ensemble"
    primary = "GEL-Ped instability guard"
    median = "coordinate-median graph ensemble"
    coarse = "anchored coarse HGB"
    cv = "constant velocity"
    model_specs = [
        (graph, "Graph ensemble", COLORS["graph"], "o"),
        (primary, "GEL-Ped", COLORS["primary"], "D"),
        (coarse, "Coarse HGB", COLORS["coarse"], "s"),
        (median, "Median ensemble", COLORS["median"], "^"),
        (cv, "Constant velocity", COLORS["cv"], "x"),
    ]
    regime_order = [
        "familiar corridor",
        "altered geometry",
        "crossing topology",
        "untouched bottleneck",
        "untouched unidirectional corridor",
    ]
    regime_labels = ["Familiar", "Altered", "Crossing", "Bottleneck", "Unidirectional"]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    ax = axes[0, 0]
    x = np.arange(len(regime_order))
    offsets = np.linspace(-0.24, 0.24, len(model_specs))
    for offset, (name, label, color, marker) in zip(offsets, model_specs, strict=True):
        values = []
        low = []
        high = []
        for regime in regime_order:
            run_values = metrics.loc[
                metrics.dataset.eq(regime) & metrics.model.eq(name), "vector_rmse_mps"
            ].to_numpy()
            values.append(run_values.mean())
            low.append(run_values.mean() - run_values.min())
            high.append(run_values.max() - run_values.mean())
        ax.errorbar(
            x + offset,
            values,
            yerr=np.vstack((low, high)),
            fmt=marker,
            markersize=4.2,
            color=color,
            capsize=2,
            elinewidth=0.8,
            label=label,
        )
    ax.set_xticks(x, regime_labels, rotation=18, ha="right")
    ax.set_ylabel("Complete-run RMSE (m/s)")
    ax.set_title("A  Accuracy across controlled regimes")
    ax.grid(axis="y", color="#D8DDE4", linewidth=0.55)
    ax.legend(ncol=2, frameon=False, loc="upper left")

    ax = axes[0, 1]
    crossing = metrics[metrics.dataset.eq("crossing topology")]
    paired = crossing.pivot(index="run", columns="model", values="vector_rmse_mps")
    difference = (paired[primary] - paired[graph]).sort_values()
    colors = np.where(difference < 0, COLORS["primary"], "#777777")
    ax.barh(
        np.arange(len(difference)),
        1000.0 * difference.to_numpy(),
        color=colors,
        alpha=0.88,
    )
    ax.set_yticks(np.arange(len(difference)), [item.replace("crossing_90_", "") for item in difference.index])
    ax.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax.axvline(1000.0 * difference.mean(), color=COLORS["graph"], linestyle="--", linewidth=1.0, label="mean")
    ax.axvline(1000.0 * difference.median(), color=COLORS["coarse"], linestyle=":", linewidth=1.2, label="median")
    ax.set_xlabel("GEL-Ped minus graph RMSE (mm/s)")
    ax.set_title("B  Every crossing run improves; typical gain is smaller")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#D8DDE4", linewidth=0.55)

    ax = axes[1, 0]
    d = diagnostics[
        diagnostics.dataset.eq("crossing topology")
        & diagnostics.diagnostic.eq("epistemic stratum")
    ]
    strata = ["inside", "0--1 IQR", "1--2 IQR", ">2 IQR"]
    aggregate = d.groupby("stratum").agg(
        samples=("samples", "sum"),
        graph_sse=("graph_sse", "sum"),
        guard_sse=("guard_sse", "sum"),
        median_sse=("median_sse", "sum"),
    )
    positions = np.arange(len(strata))
    width = 0.23
    for index, (column, label, color) in enumerate(
        (
            ("graph_sse", "Graph ensemble", COLORS["graph"]),
            ("median_sse", "Median ensemble", COLORS["median"]),
            ("guard_sse", "GEL-Ped", COLORS["primary"]),
        )
    ):
        values = [np.sqrt(aggregate.loc[item, column] / aggregate.loc[item, "samples"]) for item in strata]
        ax.bar(positions + (index - 1) * width, values, width=width, label=label, color=color, alpha=0.9)
    counts = [int(aggregate.loc[item, "samples"]) for item in strata]
    ax.set_xticks(positions, [f"{item}\n$n$={count:,}" for item, count in zip(strata, counts, strict=True)])
    ax.set_ylabel("Sample RMSE (m/s)")
    ax.set_title("C  Benefit concentrates in the instability tail")
    ax.grid(axis="y", color="#D8DDE4", linewidth=0.55)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    run_curve = (
        rollouts.groupby(["run", "horizon_s", "model"])["mean_displacement_error_m"]
        .mean()
        .reset_index()
    )
    rollout_specs = [
        (graph, "Graph ensemble", COLORS["graph"], "o"),
        (primary, "GEL-Ped", COLORS["primary"], "D"),
        (coarse, "Coarse HGB", COLORS["coarse"], "s"),
        (cv, "Constant velocity", COLORS["cv"], "x"),
    ]
    for name, label, color, marker in rollout_specs:
        selected = run_curve[run_curve.model.eq(name)]
        summary = selected.groupby("horizon_s").mean_displacement_error_m.agg(["mean", "min", "max"])
        horizon = summary.index.to_numpy()
        ax.plot(horizon, summary["mean"], color=color, marker=marker, linewidth=1.5, markersize=4, label=label)
        ax.fill_between(horizon, summary["min"], summary["max"], color=color, alpha=0.08, linewidth=0)
    ax.set_xlabel("Recursive horizon (s)")
    ax.set_ylabel("Mean displacement error (m)")
    ax.set_title("D  Recursive error: mean and complete-run range")
    ax.grid(color="#D8DDE4", linewidth=0.55)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout(pad=1.1)
    fig.savefig(OUT / "referee2_evidence.pdf", bbox_inches="tight")
    fig.savefig(OUT / "referee2_evidence.png", bbox_inches="tight")
    plt.close(fig)


def context() -> None:
    """Combine the licensed experiment photographs with the final chronology."""

    corridor = plt.imread(ROOT / "figures" / "experiment_bidirectional_overview.jpg")
    crossing = plt.imread(ROOT / "figures" / "experiment_crossing90_overview.png")
    fig = plt.figure(figsize=(10.0, 6.0))
    grid = fig.add_gridspec(2, 5, height_ratios=(3.15, 1.0), hspace=0.08, wspace=0.035)
    ax_a = fig.add_subplot(grid[0, : 3])
    ax_b = fig.add_subplot(grid[0, 3:])
    for ax, image_data, panel, label in (
        (ax_a, corridor, "A", "Bidirectional corridor"),
        (ax_b, crossing, "B", "Perpendicular crossing"),
    ):
        ax.imshow(image_data)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(
            0.03,
            0.94,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=13,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#183B59", edgecolor="none", alpha=0.92),
        )
        ax.text(
            0.5,
            0.04,
            label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color="white",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.30", facecolor="black", edgecolor="none", alpha=0.68),
        )

    stages = [
        ("7 runs", "fit + LORO\ncalibration", "#315A8C"),
        ("5 + 3 runs", "same / altered\ncorridor", "#4C956C"),
        ("13 runs", "crossing-informed\nstress test", "#C44E52"),
        ("8 + 4 runs", "additional corridor /\nbottleneck", "#D17C24"),
        ("5 scenes", "clean ETH/UCY\nLOSO test", "#8172B2"),
    ]
    for index, (headline, subtitle, color) in enumerate(stages):
        ax = fig.add_subplot(grid[1, index])
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        _box(ax, (0.04, 0.10), 0.92, 0.78, f"{headline}\n{subtitle}", face="#F4F6F9", edge=color, fontsize=8.8)
        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=(1.05, 0.5),
                xytext=(0.95, 0.5),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#687383", lw=1.0),
            )
    fig.savefig(OUT / "experimental_context_referee2.pdf", bbox_inches="tight")
    fig.savefig(OUT / "experimental_context_referee2.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    evidence()
    context()
