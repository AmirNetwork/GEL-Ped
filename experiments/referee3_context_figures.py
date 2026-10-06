# Author: Amir Ghorbani
"""Build the result-independent GEL-Ped architecture and data-role figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "context"
OUT = ROOT / "manuscript" / "overleaf_trc" / "figures"

INK = "#20242A"
BLUE = "#174A6E"
ORANGE = "#D86B27"
GREEN = "#2A8C73"
PURPLE = "#7768AE"
GREY = "#6D7278"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def _box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    face: str,
    edge: str,
    fontsize: float = 8.1,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.016,rounding_size=0.016",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.1,
        )
    )
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.22,
    )


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GREY,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.05,
            color=color,
            connectionstyle="arc3,rad=0",
        )
    )


def architecture() -> None:
    """Draw the causal forecast, monitor, and fallback path."""

    figure, axis = plt.subplots(figsize=(10.0, 4.05))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")

    _box(
        axis,
        (0.015, 0.38),
        0.15,
        0.24,
        "Past 0.8 s\npositions and velocities",
        face="#F6F7F9",
        edge=GREY,
    )
    _box(
        axis,
        (0.205, 0.38),
        0.18,
        0.24,
        "Causal local state\npersistence, personal space,\nforward and closing motion",
        face="#EDF3F8",
        edge=BLUE,
        fontsize=7.8,
    )
    _arrow(axis, (0.165, 0.50), (0.205, 0.50))

    _box(
        axis,
        (0.43, 0.66),
        0.20,
        0.19,
        "Independently trained\ngraph interaction models",
        face="#E8F0FA",
        edge=BLUE,
    )
    _box(
        axis,
        (0.43, 0.15),
        0.20,
        0.19,
        "Feature-based HGB residual\nwith a 3 m/s output cap",
        face="#EAF5EE",
        edge=GREEN,
    )
    _arrow(axis, (0.385, 0.52), (0.43, 0.755))
    _arrow(axis, (0.385, 0.48), (0.43, 0.245))

    _box(
        axis,
        (0.67, 0.66),
        0.15,
        0.19,
        "Graph mean\n$\\bar{\\mathbf{v}}^{G}$\n+ member spread",
        face="#F2F6FB",
        edge=BLUE,
    )
    _arrow(axis, (0.63, 0.755), (0.67, 0.755))
    _box(
        axis,
        (0.67, 0.15),
        0.15,
        0.19,
        "Fallback forecast\n$\\widehat{\\mathbf{v}}^{F}$",
        face="#F1F8F3",
        edge=GREEN,
    )
    _arrow(axis, (0.63, 0.245), (0.67, 0.245))

    _box(
        axis,
        (0.655, 0.405),
        0.18,
        0.14,
        "1.6-s local score\nvs. calibration envelope",
        face="#FFF2EC",
        edge=ORANGE,
        fontsize=7.8,
    )
    _arrow(axis, (0.745, 0.66), (0.745, 0.545), color=ORANGE)

    _box(
        axis,
        (0.86, 0.38),
        0.125,
        0.24,
        "GEL-Ped\n\n$g\\bar{\\mathbf{v}}^{G}$\n$+(1-g)\\widehat{\\mathbf{v}}^{F}$",
        face="#FDEDE5",
        edge=ORANGE,
        fontsize=8.0,
    )
    _arrow(axis, (0.835, 0.475), (0.86, 0.50), color=ORANGE)
    _arrow(axis, (0.82, 0.245), (0.91, 0.38), color=GREEN)
    _arrow(axis, (0.82, 0.755), (0.95, 0.62), color=BLUE)

    axis.text(
        0.51,
        0.95,
        "Complete-run fitting and calibration; no frame-level split",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        0.745,
        0.045,
        "$g=1$ inside the envelope; only an exceedance invokes the fallback",
        ha="center",
        va="center",
        fontsize=7.7,
        color=GREY,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / "gel_ped_instability_architecture.pdf", bbox_inches="tight")
    plt.close(figure)


def empirical_context() -> None:
    """Combine licensed experiment images with the frozen data chronology."""

    corridor = plt.imread(ASSETS / "bidirectional_corridor.jpg")
    crossing = plt.imread(ASSETS / "perpendicular_crossing.png")
    figure = plt.figure(figsize=(10.0, 5.8))
    grid = figure.add_gridspec(
        2,
        4,
        height_ratios=(3.25, 1.0),
        hspace=0.09,
        wspace=0.04,
    )
    left = figure.add_subplot(grid[0, :2])
    right = figure.add_subplot(grid[0, 2:])
    for axis, image, panel, label in (
        (left, corridor, "A", "Bidirectional corridor"),
        (right, crossing, "B", "Perpendicular crossing"),
    ):
        axis.imshow(image)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.text(
            0.03,
            0.94,
            panel,
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=12,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": BLUE,
                "edgecolor": "none",
                "alpha": 0.93,
            },
        )
        axis.text(
            0.5,
            0.04,
            label,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            color="white",
            fontsize=9.5,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "black",
                "edgecolor": "none",
                "alpha": 0.67,
            },
        )

    stages = (
        ("DEVELOP", "7 corridor runs\nfit + LORO calibration", BLUE),
        ("LOCKED TEST", "5 familiar + 3 altered\ncomplete runs", GREEN),
        ("STRESS", "13 crossing runs\ndevelopment-informed", ORANGE),
        ("CONFIRM", "11 T-junction runs\nuntouched one-shot", PURPLE),
    )
    for index, (phase, detail, color) in enumerate(stages):
        axis = figure.add_subplot(grid[1, index])
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.axis("off")
        _box(
            axis,
            (0.035, 0.08),
            0.93,
            0.82,
            f"{phase}\n{detail}",
            face="#F5F6F8",
            edge=color,
            fontsize=8.0,
        )
        if index < len(stages) - 1:
            axis.annotate(
                "",
                xy=(1.05, 0.50),
                xytext=(0.95, 0.50),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": GREY, "lw": 1.0},
            )
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / "experimental_context_referee3.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    architecture()
    empirical_context()


if __name__ == "__main__":
    main()
