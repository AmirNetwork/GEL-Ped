# Author: Amir Ghorbani
"""Create transparent raw and rescaled spacetime-style field visualisations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pedgeom import GeometricGradientParameters, simulate_geometric_gradient
from pedgeom.visualization import (
    normalized_geometric_elevation,
    potential_grid,
    weak_field_time_factor_deviation,
)


def corridor_wall_points(length=10.0, width=4.0, spacing=0.25):
    x = np.arange(0.0, length + spacing / 2.0, spacing)
    return np.vstack((np.column_stack((x, np.zeros_like(x))), np.column_stack((x, width * np.ones_like(x)))))


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parameters = GeometricGradientParameters()
    initial_positions = np.array([[6.0, 2.5], [3.0, 2.5]])
    destinations = np.array([[0.0, 2.0], [10.0, 2.0]])
    walls = corridor_wall_points()
    result = simulate_geometric_gradient(
        initial_positions,
        destinations,
        duration=4.0,
        step=0.1,
        parameters=parameters,
        obstacles=walls,
    )
    selected_times = [0.0, 1.0, 2.0, 3.0]
    selected_frames = [int(round(time / 0.1)) for time in selected_times]

    figure = plt.figure(figsize=(15, 7.7))
    for column, (time, frame) in enumerate(zip(selected_times, selected_frames, strict=True)):
        grid_x, grid_y, potential = potential_grid(
            0,
            result.positions[frame],
            destinations,
            walls,
            parameters.potential,
            x_limits=(0.0, 10.0),
            y_limits=(0.0, 4.0),
            resolution=(85, 42),
        )
        elevation = normalized_geometric_elevation(potential)
        delta_f = weak_field_time_factor_deviation(potential)

        surface_axis = figure.add_subplot(2, 4, column + 1, projection="3d")
        surface_axis.plot_surface(
            grid_x,
            grid_y,
            elevation,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            alpha=0.95,
        )
        surface_axis.set(
            xlabel="x",
            ylabel="y",
            zlabel="H",
            title=f"t={time:.0f} s",
            zlim=(0.0, 1.0),
        )
        surface_axis.view_init(elev=30, azim=-125)

        contour_axis = figure.add_subplot(2, 4, column + 5)
        contour = contour_axis.contourf(grid_x, grid_y, elevation, levels=18, cmap="viridis")
        contour_axis.plot(
            result.positions[: frame + 1, 0, 0],
            result.positions[: frame + 1, 0, 1],
            color="#dc2626",
            linewidth=1.8,
        )
        contour_axis.scatter(
            result.positions[frame, :, 0],
            result.positions[frame, :, 1],
            c=["#dc2626", "#22c55e"],
            edgecolors="white",
            s=32,
        )
        contour_axis.set(
            xlabel="x (m)",
            ylabel="y (m)" if column == 0 else "",
            xlim=(0.0, 10.0),
            ylim=(0.0, 4.0),
        )
        contour_axis.text(
            0.02,
            0.03,
            f"raw Δf: {delta_f.min():.1e} to {delta_f.max():.1e}",
            transform=contour_axis.transAxes,
            fontsize=7,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 2},
        )
        if column == 3:
            figure.colorbar(contour, ax=contour_axis, fraction=0.046, label="H")

    figure.suptitle(
        "Effective spacetime field for the red pedestrian\n"
        "H is a dimensionless rescaled potential elevation; raw Δf is reported without amplification",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    output = project / "figures" / "spacetime_field_sequence.png"
    figure.savefig(output, dpi=220)
    plt.close(figure)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
