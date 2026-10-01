# Author: Amir Ghorbani
"""Validate the canonical reconstruction against the manuscript's Table 2."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pedgeom.manuscript_data import (
    OBSERVATION_TIMES,
    OBSERVED_POSITIONS,
    OBSERVED_SPEEDS,
    manuscript_errors,
)
from pedgeom import simulate_geometric_gradient


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    positions = np.array([[6.0, 2.5], [3.0, 2.5]])
    destinations = np.array([[0.0, 2.0], [10.0, 2.0]])
    canonical = simulate_geometric_gradient(
        positions, destinations, duration=4.0, step=0.1
    )
    position_rmse, speed_rmse = manuscript_errors(canonical)

    output = project / "data" / "processed" / "canonical_model_validation.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "position_rmse_m",
                "speed_rmse_mps",
                "minimum_separation_m",
                "maximum_speed_mps",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "position_rmse_m": position_rmse,
                "speed_rmse_mps": speed_rmse,
                "minimum_separation_m": canonical.minimum_separation,
                "maximum_speed_mps": np.linalg.norm(canonical.velocities, axis=2).max(),
            }
        )

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].plot(
        canonical.positions[:, 0, 0],
        canonical.positions[:, 0, 1],
        color="#dc2626",
        label="canonical red",
    )
    axes[0].plot(
        canonical.positions[:, 1, 0],
        canonical.positions[:, 1, 1],
        color="#059669",
        label="canonical green",
    )
    axes[0].scatter(
        OBSERVED_POSITIONS[:, 0, 0],
        OBSERVED_POSITIONS[:, 0, 1],
        facecolors="none",
        edgecolors="#7f1d1d",
        marker="o",
        label="Table 2 red",
    )
    axes[0].scatter(
        OBSERVED_POSITIONS[:, 1, 0],
        OBSERVED_POSITIONS[:, 1, 1],
        facecolors="none",
        edgecolors="#065f46",
        marker="s",
        label="Table 2 green",
    )
    axes[0].set(
        xlabel="x (m)",
        ylabel="y (m)",
        title=f"Trajectory reproduction (RMSE {position_rmse:.3f} m)",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    speeds = np.linalg.norm(canonical.velocities, axis=2)
    axes[1].plot(canonical.times, speeds[:, 0], color="#dc2626", label="canonical red")
    axes[1].plot(canonical.times, speeds[:, 1], color="#059669", label="canonical green")
    axes[1].scatter(
        OBSERVATION_TIMES,
        OBSERVED_SPEEDS[:, 0],
        color="#7f1d1d",
        s=18,
        marker="o",
    )
    axes[1].scatter(
        OBSERVATION_TIMES,
        OBSERVED_SPEEDS[:, 1],
        color="#065f46",
        s=18,
        marker="s",
    )
    axes[1].set(
        xlabel="time (s)",
        ylabel="speed (m/s)",
        title=f"Speed reproduction (RMSE {speed_rmse:.3f} m/s)",
        ylim=(0.0, 1.6),
    )
    axes[1].grid(alpha=0.2)
    figure.suptitle("Canonical geometric-gradient reconstruction")
    figure.tight_layout()
    figure.savefig(project / "figures" / "canonical_model_validation.png", dpi=220)
    plt.close(figure)

    print(
        f"position_rmse={position_rmse:.4f} m speed_rmse={speed_rmse:.4f} m/s "
        f"minimum_separation={canonical.minimum_separation:.4f} m"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
