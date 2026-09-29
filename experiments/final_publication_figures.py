# Author: Amir Ghorbani
"""Generate the compact final figure set for shift-aware GEL-Ped."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from reviewer_figures import BLUE, GREY, NAVY, ORANGE, PURPLE, RED, TEAL, style


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def node(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    color: str,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=color,
        facecolor="white",
    )
    axis.add_patch(box)
    axis.text(
        x + width / 2,
        y + 0.62 * height,
        title,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=NAVY,
    )
    axis.text(
        x + width / 2,
        y + 0.28 * height,
        subtitle,
        ha="center",
        va="center",
        fontsize=8.3,
        color=GREY,
    )


def arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.35,
            color=NAVY,
            shrinkA=3,
            shrinkB=3,
        )
    )


def architecture(project: Path) -> None:
    figure, axis = plt.subplots(figsize=(13.5, 5.1))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(
        0.02,
        0.94,
        "GEL-Ped: geometry-encoded residual learning with unlabeled shift routing",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
    )
    axis.text(
        0.02,
        0.885,
        "A kinematic anchor reduces the learning burden; expert disagreement decides how the correction is supplied.",
        fontsize=9.5,
        color=GREY,
    )

    node(axis, 0.02, 0.38, 0.15, 0.25, "Recent trajectories", "positions and velocities", NAVY)
    node(axis, 0.22, 0.38, 0.16, 0.25, "Local geometry", "route frame + crowd fields", BLUE)
    node(axis, 0.43, 0.38, 0.14, 0.25, "Velocity anchor", "observed velocity", PURPLE)
    node(axis, 0.62, 0.60, 0.17, 0.21, "Boosted residual", "accurate in support", ORANGE)
    node(axis, 0.62, 0.19, 0.17, 0.21, "Neural residual", "smooth under shift", TEAL)
    node(axis, 0.84, 0.38, 0.14, 0.25, "Routed forecast", "anchor + selected residual", RED)
    arrow(axis, (0.17, 0.505), (0.22, 0.505))
    arrow(axis, (0.38, 0.505), (0.43, 0.505))
    arrow(axis, (0.57, 0.52), (0.62, 0.69))
    arrow(axis, (0.57, 0.49), (0.62, 0.29))
    arrow(axis, (0.79, 0.70), (0.84, 0.55))
    arrow(axis, (0.79, 0.29), (0.84, 0.46))

    axis.text(
        0.705,
        0.50,
        r"$d=$ mean expert disagreement",
        ha="center",
        fontsize=9,
        color=NAVY,
    )
    axis.text(
        0.705,
        0.455,
        r"$g=\exp[-(d-\tau)_+/\mathrm{IQR}]$",
        ha="center",
        fontsize=10,
        color=NAVY,
    )
    axis.text(
        0.705,
        0.09,
        r"The threshold $\tau$ and IQR come only from seven calibration runs; no target labels are needed at deployment.",
        ha="center",
        fontsize=8.7,
        color=GREY,
    )
    figure.savefig(project / "figures/gel_ped_architecture.png", dpi=320, bbox_inches="tight")
    plt.close(figure)


def final_results(project: Path) -> None:
    processed = project / "data/processed"
    metrics = pd.read_csv(processed / "shift_routed_upgrade_metrics.csv")
    router = pd.read_csv(processed / "shift_routed_router_diagnostics.csv")
    rollout = pd.read_csv(processed / "autoregressive_rollout_run_horizon.csv")
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.5))

    regimes = ["corridor held-out", "altered geometry", "crossing topology"]
    models = ["direct_mlp", "direct_routed_control", "gel_ped"]
    labels = ["Direct MLP", "Direct routed control", "GEL-Ped"]
    colors = [GREY, ORANGE, TEAL]
    x = np.arange(len(regimes))
    for index, (model, label, color) in enumerate(zip(models, labels, colors, strict=True)):
        frame = metrics[metrics.model.eq(model)]
        summary = frame.groupby("dataset").vector_rmse_mps.agg(["mean", "sem"]).reindex(regimes)
        axes[0, 0].bar(
            x + (index - 1) * 0.25,
            summary["mean"],
            0.23,
            yerr=summary["sem"],
            capsize=3,
            color=color,
            label=label,
        )
    axes[0, 0].set(
        title="A  Accuracy across complete-run test regimes",
        ylabel="velocity RMSE (m/s)",
        xticks=x,
        xticklabels=["familiar\ncorridor", "altered\ngeometry", "perpendicular\ncrossing"],
    )
    axes[0, 0].legend(frameon=False, fontsize=8.4)

    crossing = metrics[metrics.dataset.eq("crossing topology")].pivot(
        index="run", columns="model", values="vector_rmse_mps"
    )
    reduction = 100.0 * (
        crossing["direct_routed_control"] - crossing["gel_ped"]
    ) / crossing["direct_routed_control"]
    order = np.argsort(reduction.to_numpy())
    axes[0, 1].bar(
        np.arange(len(reduction)), reduction.to_numpy()[order], color=TEAL, width=0.72
    )
    axes[0, 1].axhline(reduction.mean(), color=NAVY, linewidth=1.5, linestyle="--")
    axes[0, 1].text(
        12.35,
        reduction.mean() + 0.18,
        f"mean {reduction.mean():.1f}%",
        ha="right",
        color=NAVY,
        fontsize=8.5,
    )
    axes[0, 1].set(
        title="B  Every crossing run improves",
        xlabel="crossing runs, ordered by reduction",
        ylabel="RMSE reduction from routed control (%)",
        xticks=[],
    )

    routed = router[router.model.eq("gel_ped")]
    categories = ["calibration", *regimes]
    category_labels = ["calibration", "familiar", "altered", "crossing"]
    category_colors = [NAVY, BLUE, PURPLE, TEAL]
    for index, (category, color) in enumerate(
        zip(categories, category_colors, strict=True)
    ):
        values = routed.loc[routed.dataset.eq(category), "disagreement_mps"].to_numpy()
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.zeros(1)
        axes[1, 0].scatter(
            index + jitter,
            values,
            s=34,
            color=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.4,
        )
    threshold = float(routed.calibration_threshold_mps.iloc[0])
    axes[1, 0].axhline(threshold, color=RED, linestyle="--", linewidth=1.4, label="calibration fence")
    axes[1, 0].set(
        title="C  Unlabeled disagreement detects topology shift",
        ylabel="mean expert disagreement (m/s)",
        xticks=np.arange(4),
        xticklabels=category_labels,
    )
    axes[1, 0].legend(frameon=False, fontsize=8.4)

    rollout_models = ["direct_mlp", "direct_routed_control", "gel_ped", "tensor_prior"]
    rollout_labels = ["Direct MLP", "Direct routed control", "GEL-Ped", "Tensor prior"]
    rollout_colors = [GREY, ORANGE, TEAL, BLUE]
    for model, label, color in zip(
        rollout_models, rollout_labels, rollout_colors, strict=True
    ):
        frame = rollout[rollout.model.eq(model)]
        summary = frame.groupby("horizon_s").mean_displacement_error_m.agg(["mean", "sem"])
        axes[1, 1].errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["sem"],
            marker="o",
            markersize=4,
            linewidth=2,
            capsize=2,
            color=color,
            label=label,
        )
    axes[1, 1].set(
        title="D  Gains compound during 3.2 s rollout",
        xlabel="rollout horizon (s)",
        ylabel="mean displacement error (m)",
    )
    axes[1, 1].legend(frameon=False, fontsize=8.0)
    figure.tight_layout(w_pad=2.2, h_pad=2.4)
    figure.savefig(project / "figures/final_results.png", dpi=320, bbox_inches="tight")
    plt.close(figure)


def data_efficiency(project: Path) -> None:
    processed = project / "data/processed"
    summary = pd.read_csv(processed / "shift_routed_data_efficiency_summary.csv")
    regimes = pd.read_csv(processed / "shift_routed_data_efficiency_per_regime.csv")
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    model_styles = {
        "constant_velocity": ("Constant velocity", GREY),
        "direct_mlp": ("Direct MLP", ORANGE),
        "gel_ped": ("GEL-Ped", TEAL),
    }
    for model, (label, color) in model_styles.items():
        frame = summary[summary.model.eq(model)].sort_values("calibration_runs")
        axes[0].plot(
            frame.calibration_runs,
            frame.mean_rmse_mps,
            marker="o",
            linewidth=2.1,
            color=color,
            label=label,
        )
        if model != "constant_velocity":
            axes[0].fill_between(
                frame.calibration_runs,
                frame.minimum_rmse_mps,
                frame.maximum_rmse_mps,
                color=color,
                alpha=0.09,
            )
    axes[0].set(
        title="A  Calibration-data efficiency",
        xlabel="complete calibration runs",
        ylabel="regime-balanced RMSE (m/s)",
        xticks=range(1, 8),
    )
    axes[0].legend(frameon=False)

    pivot = regimes.pivot(
        index=["calibration_runs", "dataset"], columns="model", values="mean_rmse_mps"
    ).reset_index()
    pivot["reduction"] = 100.0 * (pivot.direct_mlp - pivot.gel_ped) / pivot.direct_mlp
    regime_styles = {
        "corridor held-out": ("familiar corridor", BLUE),
        "altered geometry": ("altered geometry", PURPLE),
        "crossing topology": ("crossing", TEAL),
    }
    for dataset, (label, color) in regime_styles.items():
        frame = pivot[pivot.dataset.eq(dataset)].sort_values("calibration_runs")
        axes[1].plot(
            frame.calibration_runs,
            frame.reduction,
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
    axes[1].axhline(0.0, color=GREY, linewidth=1)
    axes[1].set(
        title="B  Reduction from the direct MLP",
        xlabel="complete calibration runs",
        ylabel="RMSE reduction (%)",
        xticks=range(1, 8),
    )
    axes[1].legend(frameon=False)
    figure.tight_layout(w_pad=2.3)
    figure.savefig(project / "figures/data_efficiency.png", dpi=320, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    project = project_root()
    style()
    architecture(project)
    final_results(project)
    data_efficiency(project)
    print("final publication figures complete", flush=True)


if __name__ == "__main__":
    main()
