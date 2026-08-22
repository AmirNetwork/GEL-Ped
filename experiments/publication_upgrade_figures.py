# Author: Amir Ghorbani
"""Publication figures for GEL-Ped.

Author: Amir Ghorbani
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from major_revision_analysis import build_batch, fit_models
from pedgeom.calibration import velocity_metrics


NAVY = "#17324D"
BLUE = "#2563EB"
TEAL = "#0F8B8D"
ORANGE = "#E07A00"
RED = "#D64545"
PURPLE = "#775DA6"
GREY = "#68778D"
LIGHT = "#EDF2F7"


def project_root() -> Path:
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


def style() -> None:
    family = latin_modern_family()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 10.5,
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": NAVY,
            "axes.labelcolor": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "figure.facecolor": "white",
        }
    )


def architecture_figure(project: Path) -> None:
    figure, axis = plt.subplots(figsize=(13.0, 5.2))
    axis.set_xlim(0, 13)
    axis.set_ylim(0, 5.2)
    axis.axis("off")

    def box(x, y, width, height, text, color, subtitle="", text_color="white"):
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=color, edgecolor=color, linewidth=1.5,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height * 0.62, text, ha="center", va="center",
                  color=text_color, fontsize=12, fontweight="bold")
        if subtitle:
            axis.text(x + width / 2, y + height * 0.28, subtitle, ha="center", va="center",
                      color=text_color, fontsize=8.2)

    def arrow(x1, y1, x2, y2, color=GREY):
        axis.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                       mutation_scale=14, linewidth=1.8, color=color))

    box(0.25, 1.85, 1.85, 1.45, "Observed state", NAVY,
        "motion | neighbours | walls")
    box(2.55, 1.85, 2.05, 1.45, "Goal-aligned fields", BLUE,
        "progress  |  lateral avoidance")
    box(5.25, 3.2, 2.5, 1.2, "Direct neural expert", ORANGE,
        "strong matched-scene learner")
    box(5.25, 0.65, 2.5, 1.45, "Tensor-residual expert", TEAL,
        "10-coefficient prior + gated correction")
    box(8.55, 1.85, 2.1, 1.45, "Calibrated blend", RED,
        "weight from run-wise CV")
    box(11.25, 1.85, 1.5, 1.45, "Forecast", NAVY,
        "future velocity")

    arrow(2.1, 2.58, 2.55, 2.58)
    arrow(4.6, 2.66, 5.18, 3.6, ORANGE)
    arrow(4.6, 2.45, 5.18, 1.38, TEAL)
    arrow(7.75, 3.72, 8.48, 2.92, ORANGE)
    arrow(7.75, 1.38, 8.48, 2.25, TEAL)
    arrow(10.65, 2.58, 11.25, 2.58, RED)

    axis.text(6.5, 4.72, "two complementary experts", color=NAVY,
              ha="center", fontsize=10.5, fontweight="bold")
    axis.text(6.5, 0.2, "the structured branch learns only what the tensor does not explain", color=TEAL,
              ha="center", fontsize=10.5, fontweight="bold")
    axis.text(0.25, 4.88, "GEL-Ped: geometry-encoded dual-expert prediction", color=NAVY,
              fontsize=18, fontweight="bold")
    axis.text(0.25, 4.52,
              "Run-wise calibration combines neural accuracy with a structured, data-efficient prior.",
              color=GREY, fontsize=11)
    figure.tight_layout()
    figure.savefig(project / "figures" / "tensor_residual_architecture.png", dpi=300,
                   bbox_inches="tight")
    plt.close(figure)


def _run_bootstrap(values: np.ndarray, seed: int = 20260722) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, (0.025, 0.975)))


def performance_figure(project: Path) -> None:
    metrics = pd.read_csv(project / "data" / "processed" / "major_revision_metrics.csv")
    statistics = json.loads(
        (project / "data" / "processed" / "small_sample_and_contact_statistics.json").read_text()
    )
    models = [
        "constant_velocity", "social_force", "unrestricted_ten_scalar",
        "interaction_mlp", "goal_stable_neural", "gel_ped",
    ]
    labels = [
        "Constant velocity", "Social Force", "Linear", "Direct NN",
        "Goal-stable", "GEL-Ped",
    ]
    colors = ["#9AA7B7", "#6B7B8C", BLUE, ORANGE, PURPLE, TEAL]
    datasets = ["corridor held-out", "altered geometry", "crossing external"]
    titles = ["A  Familiar corridor", "B  Altered corridor geometry", "C  Perpendicular crossing"]

    figure = plt.figure(figsize=(13.0, 8.2))
    grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 0.92), hspace=0.42, wspace=0.28)
    for column, (dataset, title) in enumerate(zip(datasets, titles, strict=True)):
        axis = figure.add_subplot(grid[0, column])
        subset = metrics[(metrics.dataset == dataset) & metrics.model.isin(models)]
        aggregate = subset.groupby("model").vector_rmse_mps.agg(["mean", "sem"]).reindex(models)
        x = np.arange(len(models))
        axis.bar(x, aggregate["mean"], color=colors, width=0.72, zorder=2)
        axis.errorbar(x, aggregate["mean"], yerr=aggregate["sem"], fmt="none",
                      ecolor=NAVY, capsize=3, linewidth=1.2, zorder=3)
        for position, value in zip(x, aggregate["mean"], strict=True):
            axis.text(position, value + 0.006, f"{value:.3f}", ha="center", va="bottom",
                      fontsize=8.5, color=NAVY, fontweight="bold")
        axis.set_xticks(x, labels, fontsize=7.3, rotation=25, ha="right")
        axis.set_ylabel("Run-balanced velocity RMSE (m/s)" if column == 0 else "")
        axis.set_title(title, loc="left")
        axis.grid(axis="y", alpha=0.18, zorder=0)
        axis.set_ylim(0, 0.46 if dataset == "crossing external" else 0.27)

    axis = figure.add_subplot(grid[1, :])
    held = metrics[metrics.dataset != "calibration"].pivot(
        index=["dataset", "run"], columns="model", values="vector_rmse_mps"
    )
    comparators = ["constant_velocity", "social_force", "unrestricted_ten_scalar"]
    comparator_labels = ["Constant velocity", "Social Force", "Unrestricted linear"]
    comparator_colors = ["#9AA7B7", "#6B7B8C", BLUE]
    y_positions = np.arange(len(comparators))[::-1]
    for y, comparator, label, color in zip(
        y_positions, comparators, comparator_labels, comparator_colors, strict=True
    ):
        improvement = 100.0 * (
            held[comparator] - held.gel_ped
        ) / held[comparator]
        low, high = _run_bootstrap(improvement.to_numpy())
        mean = improvement.mean()
        jitter = np.linspace(-0.09, 0.09, len(improvement))
        dataset_colors = held.index.get_level_values(0).map(
            {"corridor held-out": BLUE, "altered geometry": ORANGE, "crossing external": TEAL}
        )
        axis.scatter(improvement, y + jitter, c=dataset_colors, s=28, alpha=0.68,
                     edgecolor="white", linewidth=0.4, zorder=2)
        axis.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt="D", color=color,
                      capsize=4, markersize=7, linewidth=2.2, zorder=4)
        adjusted = statistics["held_out_pooled_relative_rmse"][comparator][
            "holm_adjusted_randomization_p"
        ]
        p_label = "<0.001" if adjusted < 0.001 else f"={adjusted:.3f}"
        axis.text(high + 0.7, y, f"mean {mean:.1f}%  |  Holm p{p_label}",
                  va="center", fontsize=9, color=NAVY)
    axis.axvline(0, color=NAVY, linewidth=1.1)
    axis.set_yticks(y_positions, comparator_labels)
    axis.set_xlabel("RMSE reduction from GEL-Ped (%)")
    axis.set_title("D  Run-level gains over operational and linear reference models",
                   loc="left")
    axis.grid(axis="x", alpha=0.2)
    axis.text(0.99, -0.24,
              "Dots: complete experimental runs (blue corridor, orange altered, teal crossing); diamonds: mean and 95% run-bootstrap interval",
              transform=axis.transAxes, ha="right", fontsize=8.3, color=GREY)
    figure.suptitle("Accuracy gains are large against operational baselines and persist across topology",
                    fontsize=16, fontweight="bold", color=NAVY, y=0.99)
    figure.savefig(project / "figures" / "primary_performance_summary.png", dpi=300,
                   bbox_inches="tight")
    plt.close(figure)


def diagnostics_figure(project: Path) -> None:
    processed = project / "data" / "processed"
    metrics = pd.read_csv(processed / "major_revision_metrics.csv")
    support = pd.read_csv(processed / "residual_support_diagnostics.csv")
    seed_summary = pd.read_csv(processed / "neural_seed_sensitivity_summary.csv")
    coefficients = pd.read_csv(processed / "coefficient_stability_summary.csv")

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))

    ablations = [
        "tensor_response", "unrestricted_ten_scalar", "interaction_mlp",
        "tensor_residual_ungated", "tensor_residual", "gel_ped",
    ]
    labels = ["Tensor\nprior", "Unrestricted\nlinear", "Direct\nnetwork",
              "Residual\n(no gate)", "Tensor\nresidual", "GEL-Ped"]
    for dataset, color, marker in (
        ("corridor held-out", BLUE, "o"),
        ("altered geometry", ORANGE, "D"),
        ("crossing external", TEAL, "s"),
    ):
        subset = metrics[(metrics.dataset == dataset) & metrics.model.isin(ablations)]
        values = subset.groupby("model").vector_rmse_mps.mean().reindex(ablations)
        relative = 100.0 * (values / values.loc["tensor_response"] - 1.0)
        axes[0, 0].plot(np.arange(len(ablations)), relative, color=color, marker=marker,
                        linewidth=2, markersize=6, label=dataset)
    axes[0, 0].axhline(0, color=GREY, linewidth=1)
    axes[0, 0].set(xticks=np.arange(len(ablations)), xticklabels=labels,
                   ylabel="RMSE change from tensor prior (%)",
                   title="A  Component and capacity ablation")
    axes[0, 0].legend(frameon=False, fontsize=8.5)

    aggregate = support.groupby("dataset").agg(
        mean_gate=("mean_residual_gate", "mean"),
        shrunk=("fraction_residual_shrunk", "mean"),
    ).reindex(["calibration", "corridor held-out", "altered geometry", "crossing external"])
    x = np.arange(len(aggregate))
    axes[0, 1].bar(x - 0.18, aggregate.mean_gate, 0.36, color=TEAL, label="mean gate")
    axes[0, 1].bar(x + 0.18, aggregate.shrunk, 0.36, color=RED, label="fraction shrunk")
    axes[0, 1].set(xticks=x, xticklabels=["calibration", "held-out\ncorridor", "altered\ngeometry", "crossing"],
                   ylim=(0, 1.08), title="B  Data-support gate is observable")
    axes[0, 1].legend(frameon=False, fontsize=8.5)

    ordered = coefficients.copy()
    ordered["label"] = ordered["channel"].str[0].str.upper() + ": " + ordered["feature"]
    y = np.arange(len(ordered))
    axes[1, 0].errorbar(
        ordered.coefficient, y,
        xerr=np.vstack((ordered.coefficient - ordered.bootstrap_95_low,
                        ordered.bootstrap_95_high - ordered.coefficient)),
        fmt="o", capsize=3, color=TEAL,
    )
    axes[1, 0].axvline(0, color=GREY, linewidth=1)
    axes[1, 0].set(yticks=y, yticklabels=ordered.label, title="C  Tensor coefficients and run-bootstrap intervals")
    axes[1, 0].invert_yaxis()

    datasets = ["corridor held-out", "altered geometry", "crossing external"]
    y = np.arange(len(datasets))
    for offset, model, color, label in (
        (-0.11, "interaction_mlp", ORANGE, "direct network"),
        (0.11, "gel_ped", TEAL, "GEL-Ped"),
    ):
        subset = seed_summary[seed_summary.model == model].set_index("dataset").reindex(datasets)
        axes[1, 1].errorbar(
            subset.median_rmse_mps, y + offset,
            xerr=np.vstack((subset.median_rmse_mps - subset.minimum_rmse_mps,
                            subset.maximum_rmse_mps - subset.median_rmse_mps)),
            fmt="o", color=color, capsize=3, label=label,
        )
    axes[1, 1].set(yticks=y, yticklabels=["held-out corridor", "altered geometry", "crossing"],
                   xlabel="Run-balanced RMSE across five seeds (m/s)",
                   title="D  Optimization-seed sensitivity")
    axes[1, 1].legend(frameon=False, fontsize=8.5)

    for axis in axes.flat:
        axis.grid(alpha=0.18)
        axis.tick_params(labelsize=8.8)
    figure.suptitle("What GEL-Ped gains, when it defers, and how stable it is",
                    fontsize=16, fontweight="bold", color=NAVY)
    figure.tight_layout()
    figure.savefig(project / "figures" / "hybrid_diagnostics.png", dpi=300,
                   bbox_inches="tight")
    plt.close(figure)


def data_efficiency_figure(project: Path) -> None:
    processed = project / "data" / "processed"
    summary = pd.read_csv(processed / "data_efficiency_summary.csv")
    metrics = pd.read_csv(processed / "data_efficiency_metrics.csv")
    statistics = json.loads((processed / "data_efficiency_statistics.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for model, color, label, marker in (
        ("direct_network", ORANGE, "direct neural model", "o"),
        ("gel_ped", TEAL, "GEL-Ped", "D"),
    ):
        data = summary[summary.model == model].sort_values("calibration_runs")
        axes[0].plot(data.calibration_runs, data.mean_rmse_mps, color=color, marker=marker,
                     linewidth=2.4, markersize=7, label=label)
        axes[0].fill_between(data.calibration_runs, data.minimum_rmse_mps,
                             data.maximum_rmse_mps, color=color, alpha=0.13)
    axes[0].set(xlabel="Site-specific fitting runs", ylabel="Pooled held-out RMSE (m/s)",
                xticks=list(range(1, 8)),
                title="A  Fixed-design estimation curve over every run subset")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    subset = metrics.groupby(["calibration_runs", "subset", "model"], as_index=False).agg(
        rmse=("vector_rmse_mps", "mean")
    ).pivot(index=["calibration_runs", "subset"], columns="model", values="rmse")
    run_counts = list(range(1, 8))
    for position, run_count in enumerate(run_counts):
        data = subset.loc[run_count]
        improvement = 100.0 * (data.direct_network - data.gel_ped) / data.direct_network
        axes[1].scatter(np.full(len(improvement), position), improvement, color=TEAL,
                        s=40, alpha=0.75, edgecolor="white", linewidth=0.5)
        axes[1].scatter(position, improvement.mean(), color=NAVY, marker="D", s=55, zorder=4)
        axes[1].text(position, improvement.max() + 0.8, f"{improvement.mean():.1f}%",
                     ha="center", color=NAVY, fontsize=9, fontweight="bold")
    axes[1].axhline(0, color=GREY, linewidth=1)
    axes[1].set(xticks=np.arange(len(run_counts)), xticklabels=run_counts,
                xlabel="Site-specific fitting runs",
                ylabel="GEL-Ped RMSE reduction from direct network (%)",
                title="B  Gain across distinct calibration subsets")
    total_subsets = sum(statistics[str(count)]["subsets_total"] for count in run_counts)
    total_wins = sum(statistics[str(count)]["subsets_gel_ped_lower"] for count in run_counts)
    axes[1].text(
        0.02, 0.94,
        f"All {total_subsets} subsets enumerated; GEL-Ped lower in {total_wins}/{total_subsets}",
        transform=axes[1].transAxes, color=NAVY, fontsize=9, va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
    )
    axes[1].grid(axis="y", alpha=0.2)
    figure.suptitle("Structured guidance improves fixed-design estimation across all subsets",
                    fontsize=16, fontweight="bold", color=NAVY)
    figure.tight_layout()
    figure.savefig(project / "figures" / "data_efficiency.png", dpi=300,
                   bbox_inches="tight")
    plt.close(figure)


def horizon_transfer_figure(project: Path) -> None:
    processed = project / "data" / "processed"
    metrics = pd.read_csv(processed / "neural_horizon_sensitivity_metrics.csv")
    crossing = metrics[metrics.dataset == "crossing external"]
    summary = crossing.groupby(["horizon_s", "model"], as_index=False).agg(
        rmse=("vector_rmse_mps", "mean")
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    model_style = (
        ("direct_network", ORANGE, "direct neural model", "o"),
        ("goal_stable_neural", PURPLE, "2024 goal-stable hybrid", "s"),
        ("gel_ped", TEAL, "GEL-Ped", "D"),
    )
    for model, color, label, marker in model_style:
        data = summary[summary.model == model].sort_values("horizon_s")
        axes[0].plot(
            data.horizon_s, data.rmse, color=color, marker=marker,
            linewidth=2.4, markersize=7, label=label,
        )
    axes[0].set(
        xlabel="Forecast horizon (s)", ylabel="Run-balanced crossing RMSE (m/s)",
        xticks=[0.2, 0.4, 0.8, 1.2],
        title="A  Frozen transfer to perpendicular crossing flow",
    )
    axes[0].legend(frameon=False, fontsize=8.8)
    axes[0].grid(alpha=0.2)

    pivot = crossing.pivot(
        index=["horizon_s", "run"], columns="model", values="vector_rmse_mps"
    )
    for position, horizon in enumerate((0.2, 0.4, 0.8, 1.2)):
        data = pivot.loc[horizon]
        reduction = 100.0 * (
            data.direct_network - data.gel_ped
        ) / data.direct_network
        low, high = _run_bootstrap(reduction.to_numpy())
        mean = float(reduction.mean())
        jitter = np.linspace(-0.10, 0.10, len(reduction))
        axes[1].scatter(
            np.full(len(reduction), position) + jitter, reduction,
            color=TEAL, s=31, alpha=0.66, edgecolor="white", linewidth=0.4,
        )
        axes[1].errorbar(
            position, mean, yerr=[[mean - low], [high - mean]], fmt="D",
            color=NAVY, capsize=4, linewidth=2.0, markersize=6.5,
        )
        axes[1].text(
            position, high + 0.35, f"{mean:.2f}%", ha="center",
            color=NAVY, fontsize=9, fontweight="bold",
        )
    axes[1].axhline(0, color=GREY, linewidth=1)
    axes[1].set(
        xticks=np.arange(4), xticklabels=["0.2", "0.4", "0.8", "1.2"],
        xlabel="Forecast horizon (s)",
        ylabel="RMSE reduction from direct neural model (%)",
        title="B  Complete-run improvement at every horizon",
    )
    axes[1].grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Directional structure transfers beyond a single forecast horizon",
        fontsize=16, fontweight="bold", color=NAVY,
    )
    figure.tight_layout()
    figure.savefig(
        project / "figures" / "cross_horizon_neural_transfer.png", dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def forecast_snapshot(project: Path) -> None:
    config = json.loads((project / "configs" / "major_revision.json").read_text())
    splits = json.loads((project / "data" / "splits.json").read_text())
    selected = json.loads((project / "data" / "processed" / "selected_hyperparameters.json").read_text())
    corridor = project / "data" / "raw" / "2013bidirectional"
    training = [build_batch(corridor / f"{run}.txt", config, "entry") for run in splits["calibration"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        models = fit_models(training, selected, config)
    source = project / "data" / "raw" / "2013crossing90" / "trajectories" / "crossing_90_d_7.txt"
    batch = build_batch(source, config, "entry", crossing=True)
    comparisons = [
        ("constant_velocity", "Constant velocity", "#9AA7B7"),
        ("interaction_mlp", "Direct network", ORANGE),
        ("gel_ped", "GEL-Ped", TEAL),
    ]
    predictions = {
        key: models[key].model.predict(batch, speed_cap=models[key].speed_cap)
        for key, _, _ in comparisons
    }
    # Use a reproducible, representative positive-transfer frame rather than the
    # visually most favourable frame.  Among frames with at least 20 pedestrians,
    # select the frame whose GEL-Ped-over-network gain is closest to the median
    # positive gain.
    candidates = []
    for candidate, count in pd.Series(batch.frame).value_counts().items():
        if count < 20:
            continue
        rows = batch.frame == candidate
        direct_rmse = velocity_metrics(batch.target[rows], predictions["interaction_mlp"][rows])[
            "vector_rmse_mps"
        ]
        ensemble_rmse = velocity_metrics(
            batch.target[rows], predictions["gel_ped"][rows]
        )["vector_rmse_mps"]
        gain = direct_rmse - ensemble_rmse
        if gain > 0:
            candidates.append((int(candidate), gain))
    median_gain = float(np.median([gain for _, gain in candidates]))
    frame = min(candidates, key=lambda item: abs(item[1] - median_gain))[0]
    selected_rows = batch.frame == frame
    position = batch.position[selected_rows]
    target = batch.target[selected_rows]
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 5.0), sharex=True, sharey=True)
    for axis, (key, title, color) in zip(axes, comparisons, strict=True):
        prediction = predictions[key][selected_rows]
        rmse = velocity_metrics(target, prediction)["vector_rmse_mps"]
        axis.scatter(position[:, 0], position[:, 1], s=22, color=NAVY, alpha=0.65, zorder=3)
        axis.quiver(position[:, 0], position[:, 1], target[:, 0], target[:, 1],
                    color="#20A4A8", angles="xy", scale_units="xy", scale=2.5,
                    width=0.008, alpha=0.9, label="observed")
        axis.quiver(position[:, 0], position[:, 1], prediction[:, 0], prediction[:, 1],
                    color=color, angles="xy", scale_units="xy", scale=2.5,
                    width=0.006, alpha=0.9, label="predicted")
        axis.set_title(f"{title}\nframe RMSE {rmse:.3f} m/s")
        axis.set_aspect("equal")
        axis.grid(alpha=0.16)
        axis.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[-1].legend(frameon=False, loc="upper right", fontsize=8.5)
    figure.suptitle("Unseen crossing flow: observed and predicted short-horizon motion",
                    fontsize=15, fontweight="bold", color=NAVY, y=0.98)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    figure.savefig(project / "figures" / "external_crossing_hybrid_snapshot.png", dpi=300,
                   bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    project = project_root()
    style()
    architecture_figure(project)
    performance_figure(project)
    diagnostics_figure(project)
    data_efficiency_figure(project)
    horizon_transfer_figure(project)
    forecast_snapshot(project)
    print("publication-upgrade figures complete", flush=True)


if __name__ == "__main__":
    main()
