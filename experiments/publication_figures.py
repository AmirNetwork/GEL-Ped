# Author: Amir Ghorbani
"""Create editorial figures for the GEL-Ped manuscript.

Author: Amir Ghorbani
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from pedgeom.calibration import build_velocity_samples, fit_tensor_geometry_model
from pedgeom.datasets import load_julich_crossing_trajectory


INK = "#16324F"
TEAL = "#008C95"
BLUE = "#2563EB"
RED = "#DC2626"
GOLD = "#D97706"
GRAY = "#64748B"
LIGHT = "#E8EEF5"


def _project() -> Path:
    return Path(__file__).resolve().parents[1]


def latin_modern_family() -> str:
    """Register Latin Modern from common TeX installations without invoking LaTeX."""

    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "__missing__"))
        / "Programs/MiKTeX/fonts/opentype/public/lm",
        Path(os.environ.get("APPDATA", "__missing__"))
        / "MiKTeX/fonts/opentype/public/lm",
        Path("/usr/share/texmf/fonts/opentype/public/lm"),
        Path("/usr/share/texlive/texmf-dist/fonts/opentype/public/lm"),
    ]
    for directory in candidates:
        regular = directory / "lmroman10-regular.otf"
        if regular.exists():
            for font in directory.glob("lmroman10-*.otf"):
                font_manager.fontManager.addfont(font)
            return font_manager.FontProperties(fname=regular).get_name()
    return "STIXGeneral"


def publication_style() -> None:
    """Match figure typography to the manuscript's Latin Modern text."""

    family = latin_modern_family()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 10.5,
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
        }
    )


def experiment_context(project: Path) -> None:
    corridor = mpimg.imread(project / "figures" / "experiment_bidirectional_overview.jpg")
    crossing = mpimg.imread(project / "figures" / "experiment_crossing90_overview.png")
    figure = plt.figure(figsize=(12.5, 7.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(4.3, 1.4))
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    for axis, image, label, title in zip(
        axes,
        (corridor, crossing),
        ("A", "B"),
        ("Bidirectional corridor", "Perpendicular crossing"),
        strict=True,
    ):
        axis.imshow(image, aspect="auto")
        axis.set_axis_off()
        axis.text(
            0.02,
            0.95,
            label,
            transform=axis.transAxes,
            va="top",
            color="white",
            fontsize=14,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": INK, "edgecolor": "none"},
        )
        axis.text(
            0.5,
            0.04,
            title,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            color="white",
            fontsize=13,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": (0, 0, 0, 0.65), "edgecolor": "none"},
        )
    flow = figure.add_subplot(grid[1, :])
    flow.set_axis_off()
    boxes = [
        (0.02, "7 runs", "calibrate", BLUE),
        (0.265, "5 runs", "held-out corridor", TEAL),
        (0.51, "3 runs", "altered geometry", GOLD),
        (0.755, "13 runs", "external crossing", RED),
    ]
    for index, (x, count, label, color) in enumerate(boxes):
        box = FancyBboxPatch(
            (x, 0.15),
            0.205,
            0.68,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            transform=flow.transAxes,
            facecolor=LIGHT,
            edgecolor=color,
            linewidth=2,
        )
        flow.add_patch(box)
        flow.text(x + 0.1025, 0.57, count, ha="center", va="center", color=INK, fontsize=16, fontweight="bold")
        flow.text(x + 0.1025, 0.32, label, ha="center", va="center", color=INK, fontsize=11)
        if index < len(boxes) - 1:
            flow.add_patch(
                FancyArrowPatch(
                    (x + 0.21, 0.49),
                    (boxes[index + 1][0] - 0.006, 0.49),
                    transform=flow.transAxes,
                    arrowstyle="-|>",
                    mutation_scale=15,
                    color=GRAY,
                    linewidth=1.5,
                )
            )
    flow.text(0.5, 0.02, "parameters frozen before every test stage", ha="center", va="bottom", color=GRAY, fontsize=10)
    figure.savefig(project / "figures" / "experimental_context_and_protocol.png", dpi=260, bbox_inches="tight")
    plt.close(figure)


def tensor_schematic(project: Path) -> None:
    figure, axis = plt.subplots(figsize=(11.5, 5.8))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 6)
    axis.set_aspect("equal")
    axis.axis("off")
    pedestrian = Circle((2.0, 3.0), 0.33, facecolor=TEAL, edgecolor=INK, linewidth=1.5)
    axis.add_patch(pedestrian)
    axis.arrow(2.0, 3.0, 2.0, 0, width=0.035, head_width=0.25, head_length=0.3, color=BLUE, length_includes_head=True)
    axis.arrow(2.0, 3.0, 0, 1.45, width=0.025, head_width=0.22, head_length=0.25, color=GOLD, length_includes_head=True)
    axis.text(3.15, 2.62, r"goal axis $e$", va="center", ha="center", color=INK, fontsize=12)
    axis.text(2.0, 4.72, r"lateral axis $\ell$", ha="center", color=INK, fontsize=12)
    neighbours = [(0.8, 4.1), (0.9, 2.0), (3.1, 4.2), (3.3, 1.75)]
    for n, (x, y) in enumerate(neighbours):
        axis.add_patch(Circle((x, y), 0.23, facecolor=LIGHT, edgecolor=GRAY, linewidth=1.2))
        axis.add_patch(FancyArrowPatch((x, y), (1.75, 2.95), arrowstyle="->", mutation_scale=12, color=GRAY, alpha=0.8))
    axis.text(2.0, 0.65, "persistence  +  goal  +  neighbour  +  closing  +  wall fields", ha="center", color=INK, fontsize=11)
    axis.add_patch(FancyArrowPatch((4.6, 3.0), (5.55, 3.0), arrowstyle="-|>", mutation_scale=18, color=GRAY, linewidth=1.8))
    for y, title, details, color in (
        (4.15, "LONGITUDINAL CHANNEL", "5 projected features\n" + r"$\beta_{\parallel}$: forward progress", BLUE),
        (1.85, "LATERAL CHANNEL", "5 projected features\n" + r"$\beta_{\perp}$: steering response", GOLD),
    ):
        box = FancyBboxPatch((5.55, y - 0.75), 2.95, 1.5, boxstyle="round,pad=0.08", facecolor=LIGHT, edgecolor=color, linewidth=2)
        axis.add_patch(box)
        axis.text(7.025, y + 0.25, title, ha="center", va="center", color=INK, fontsize=9.8, fontweight="bold")
        axis.text(7.025, y - 0.28, details, ha="center", va="center", color=INK, fontsize=10)
    axis.add_patch(FancyArrowPatch((8.55, 4.15), (9.35, 3.25), arrowstyle="-|>", mutation_scale=16, color=BLUE, linewidth=1.8))
    axis.add_patch(FancyArrowPatch((8.55, 1.85), (9.35, 2.75), arrowstyle="-|>", mutation_scale=16, color=GOLD, linewidth=1.8))
    output = FancyBboxPatch((9.45, 2.05), 2.25, 1.9, boxstyle="round,pad=0.1", facecolor="white", edgecolor=TEAL, linewidth=2.4)
    axis.add_patch(output)
    axis.text(10.575, 3.35, "PREDICTED VELOCITY", ha="center", va="center", color=INK, fontsize=11, fontweight="bold")
    axis.text(10.575, 2.72, r"$\widehat{v}=v_{\parallel}e+v_{\perp}\ell$", ha="center", va="center", color=INK, fontsize=14)
    axis.text(10.575, 2.3, "10 fitted coefficients", ha="center", va="center", color=GRAY, fontsize=10)
    axis.text(6.0, 5.65, "Goal-frame response tensors: forward persistence vs lateral adaptation", ha="center", color=INK, fontsize=15, fontweight="bold")
    figure.savefig(project / "figures" / "tensor_model_schematic.png", dpi=260, bbox_inches="tight")
    plt.close(figure)


def external_forecast_case(project: Path) -> None:
    splits = json.loads((project / "data" / "splits.json").read_text(encoding="utf-8"))
    corridor = project / "data" / "raw" / "2013bidirectional"
    training = [build_velocity_samples(corridor / f"{run}.txt", 0.8) for run in splits["calibration"]]
    tensor = fit_tensor_geometry_model(training)
    sample = build_velocity_samples(
        project / "data" / "raw" / "2013crossing90" / "trajectories" / "crossing_90_d_1.txt",
        0.8,
        wall_y_bounds=None,
        loader=load_julich_crossing_trajectory,
    )
    counts = pd.Series(sample.frame).value_counts()
    frame = int(counts.index[0])
    selected = np.flatnonzero(sample.frame == frame)
    frame_positions = sample.position[selected]
    centre = np.median(frame_positions, axis=0)
    distance_to_centre = np.linalg.norm(frame_positions - centre, axis=1)
    selected = selected[np.argsort(distance_to_centre)[:60]]
    position = sample.position[selected]
    target = sample.target[selected]
    persistence = sample.features[selected, 0, :]
    tensor_prediction = tensor.predict(sample, speed_cap=100.0)[selected]
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharex=True, sharey=True)
    for axis, prediction, title in (
        (axes[0], persistence, "Constant velocity"),
        (axes[1], tensor_prediction, "Tensor response"),
    ):
        axis.scatter(position[:, 0], position[:, 1], s=22, color=INK, alpha=0.78, label="current position")
        axis.quiver(position[:, 0], position[:, 1], target[:, 0], target[:, 1], color=TEAL, angles="xy", scale_units="xy", scale=2.0, width=0.0065, alpha=0.88, label="observed future")
        axis.quiver(position[:, 0], position[:, 1], prediction[:, 0], prediction[:, 1], color=RED, angles="xy", scale_units="xy", scale=2.0, width=0.0052, alpha=0.78, label="prediction")
        error = np.sqrt(np.mean(np.sum((prediction - target) ** 2, axis=1)))
        axis.set_title(f"{title}\nframe RMSE = {error:.3f} m/s", color=INK, fontweight="bold")
        axis.set_xlabel("x (m)")
        axis.set_xlim(position[:, 0].min() - 0.45, position[:, 0].max() + 0.45)
        axis.set_ylim(position[:, 1].min() - 0.45, position[:, 1].max() + 0.45)
        axis.grid(alpha=0.16)
        axis.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("y (m)")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.93))
    figure.suptitle("Unseen 90-degree crossing: zoomed interaction region", y=0.995, color=INK, fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(project / "figures" / "external_crossing_forecast_snapshot.png", dpi=260, bbox_inches="tight")
    plt.close(figure)


def evidence_summary(project: Path) -> None:
    if (project / "data" / "processed" / "major_revision_metrics.csv").exists():
        print("preserving major-revision publication_evidence_summary.png", flush=True)
        return
    metrics = pd.read_csv(project / "data" / "processed" / "data_driven_benchmark_metrics.csv")
    summary = pd.read_csv(project / "data" / "processed" / "data_driven_benchmark_summary.csv")
    statistics = json.loads((project / "data" / "processed" / "benchmark_comparison_statistics.json").read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.1), gridspec_kw={"width_ratios": (1.55, 1.0)})
    left = axes[0]
    y = 0
    ticks = []
    labels = []
    for dataset in ("corridor held-out", "crossing external"):
        pivot = metrics[metrics["dataset"] == dataset].pivot(index="run", columns="model", values="vector_rmse_mps")
        for run, row in pivot.iterrows():
            left.plot([row["constant_velocity"], row["tensor_geometry"]], [y, y], color=LIGHT, linewidth=3, zorder=1)
            left.scatter(row["constant_velocity"], y, color=GRAY, s=34, zorder=2)
            left.scatter(row["tensor_geometry"], y, color=TEAL, s=46, zorder=3)
            y += 1
        ticks.append(y - (len(pivot) + 1) / 2)
        labels.append(dataset)
        y += 1
    left.set_yticks(ticks, labels)
    left.set_xlabel("run vector RMSE (m/s)  ← lower is better")
    left.set_title("Every run improves over constant velocity", color=INK, fontweight="bold")
    left.grid(axis="x", alpha=0.18)
    left.scatter([], [], color=GRAY, label="constant velocity")
    left.scatter([], [], color=TEAL, label="tensor geometry")
    left.legend(frameon=False, loc="lower right")

    right = axes[1]
    counts = statistics["fitted_parameter_counts"]
    ext = summary[summary["dataset"] == "crossing external"].set_index("model")
    names = ["isotropic_social_force", "decoupled_geometry", "tensor_geometry", "invariant_ridge", "interaction_mlp"]
    labels_map = {"isotropic_social_force": "isotropic", "decoupled_geometry": "decoupled", "tensor_geometry": "tensor", "invariant_ridge": "ridge", "interaction_mlp": "MLP"}
    colors = {"tensor_geometry": TEAL, "interaction_mlp": RED, "invariant_ridge": BLUE, "isotropic_social_force": GRAY, "decoupled_geometry": GOLD}
    for name in names:
        x = counts[name]
        value = ext.loc[name, "vector_rmse_mps"]
        right.scatter(x, value, s=95 if name == "tensor_geometry" else 60, color=colors[name], zorder=3)
        offset = {"tensor_geometry": (5, -2), "invariant_ridge": (5, 8)}.get(name, (5, 5))
        right.annotate(labels_map[name], (x, value), xytext=offset, textcoords="offset points", fontsize=10, color=INK)
    right.set_xscale("log")
    right.set_xlabel("fitted parameters (log scale)")
    right.set_ylabel("external crossing RMSE (m/s)")
    right.set_title("Transfer accuracy versus complexity", color=INK, fontweight="bold")
    right.grid(alpha=0.18)
    right.annotate("10 coefficients\n≈ MLP external RMSE", xy=(10, ext.loc["tensor_geometry", "vector_rmse_mps"]), xytext=(60, 0.365), arrowprops={"arrowstyle": "->", "color": TEAL}, color=INK, fontsize=10)
    figure.tight_layout()
    figure.savefig(project / "figures" / "publication_evidence_summary.png", dpi=260, bbox_inches="tight")
    plt.close(figure)


def graphical_abstract(project: Path) -> None:
    """Create a compact journal graphical abstract from sourced photos and measured results."""
    corridor = mpimg.imread(project / "figures" / "experiment_bidirectional_overview.jpg")
    crossing = mpimg.imread(project / "figures" / "experiment_crossing90_overview.png")
    figure = plt.figure(figsize=(14.2, 5.2), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.15, 1.35, 1.15))

    photos = figure.add_subplot(grid[0, 0])
    photos.set_axis_off()
    photos.imshow(corridor, extent=(0, 1, 1.05, 2.0), aspect="auto")
    photos.imshow(crossing, extent=(0, 1, 0.0, 0.95), aspect="auto")
    photos.set_xlim(0, 1)
    photos.set_ylim(0, 2)
    photos.text(0.04, 1.91, "CALIBRATE: COUNTERFLOW", color="white", fontsize=9.5, fontweight="bold", va="top", bbox={"facecolor": (0, 0, 0, 0.67), "edgecolor": "none", "pad": 3})
    photos.text(0.04, 0.86, r"TEST: $90^{\circ}$ CROSSING", color="white", fontsize=9.5, fontweight="bold", va="top", bbox={"facecolor": (0, 0, 0, 0.67), "edgecolor": "none", "pad": 3})
    photos.set_title("Real controlled trajectories", color=INK, fontweight="bold", pad=8)

    model = figure.add_subplot(grid[0, 1])
    model.set_xlim(0, 10)
    model.set_ylim(0, 10)
    model.set_axis_off()
    model.add_patch(Circle((2.0, 5.0), 0.48, facecolor=TEAL, edgecolor=INK, linewidth=1.5))
    model.arrow(2.0, 5.0, 2.3, 0, width=0.055, head_width=0.38, head_length=0.42, color=BLUE, length_includes_head=True)
    model.arrow(2.0, 5.0, 0, 2.0, width=0.045, head_width=0.34, head_length=0.38, color=GOLD, length_includes_head=True)
    for x, y in ((0.7, 6.4), (0.9, 3.5), (3.2, 6.6), (3.4, 3.3)):
        model.add_patch(Circle((x, y), 0.30, facecolor=LIGHT, edgecolor=GRAY, linewidth=1.0))
        model.add_patch(FancyArrowPatch((x, y), (1.7, 4.9), arrowstyle="->", mutation_scale=10, color=GRAY, alpha=0.8))
    for y, title, color in ((6.7, "LONGITUDINAL\npersistence", BLUE), (3.3, "LATERAL\nadaptation", GOLD)):
        model.add_patch(FancyBboxPatch((5.2, y - 0.85), 3.8, 1.7, boxstyle="round,pad=0.08", facecolor=LIGHT, edgecolor=color, linewidth=2))
        model.text(7.1, y, title, ha="center", va="center", color=INK, fontsize=11, fontweight="bold")
    model.add_patch(FancyArrowPatch((4.35, 5.0), (5.1, 5.0), arrowstyle="-|>", mutation_scale=16, color=GRAY, linewidth=1.5))
    model.text(5.0, 8.9, "Goal-frame tensor response", ha="center", color=INK, fontsize=14, fontweight="bold")
    model.text(5.0, 1.0, r"10 coefficients $\bullet$ past-only route frame", ha="center", color=INK, fontsize=10.5)

    result = figure.add_subplot(grid[0, 2])
    result.set_axis_off()
    result.set_xlim(0, 1)
    result.set_ylim(0, 1)
    cards = [
        (0.76, "5 / 5", "held-out corridor runs improve", TEAL),
        (0.50, "3 / 3", "altered geometries improve", GOLD),
        (0.24, "13 / 13", "external crossing runs improve", RED),
    ]
    for y, value, label, color in cards:
        result.add_patch(FancyBboxPatch((0.05, y - 0.095), 0.9, 0.19, boxstyle="round,pad=0.012", facecolor=LIGHT, edgecolor=color, linewidth=2))
        result.text(0.23, y, value, ha="center", va="center", color=INK, fontsize=16, fontweight="bold")
        result.text(0.62, y, label, ha="center", va="center", color=INK, fontsize=10)
    result.text(0.5, 0.94, "Frozen transfer evidence", ha="center", color=INK, fontsize=14, fontweight="bold")
    result.text(0.5, 0.055, r"crossing RMSE  $0.413\rightarrow0.369$ m/s", ha="center", va="center", color=INK, fontsize=11.5, fontweight="bold")

    figure.savefig(project / "figures" / "graphical_abstract.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    project = _project()
    publication_style()
    experiment_context(project)
    tensor_schematic(project)
    external_forecast_case(project)
    evidence_summary(project)
    graphical_abstract(project)


if __name__ == "__main__":
    main()
