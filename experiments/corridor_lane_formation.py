# Author: Amir Ghorbani
"""Test whether anisotropic geometry produces bidirectional lane segregation."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pedgeom import (
    AnisotropicParameters,
    simulate_avm_periodic_corridor,
    simulate_periodic_corridor,
)
from pedgeom.datasets import lane_order_parameter


def nonoverlapping_positions(rng, agents, length, width, minimum_distance=0.45):
    accepted = []
    attempts = 0
    while len(accepted) < agents:
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError("could not construct nonoverlapping initial state")
        candidate = np.array(
            [rng.uniform(-length / 2.0, length / 2.0), rng.uniform(0.25, width - 0.25)]
        )
        if all(np.linalg.norm(candidate - existing) >= minimum_distance for existing in accepted):
            accepted.append(candidate)
    return np.asarray(accepted)


def order_history(result, directions, width):
    frames = np.arange(len(result.times))
    data = pd.DataFrame(
        {
            "frame": np.repeat(frames, len(directions)),
            "y_m": result.positions[:, :, 1].reshape(-1),
            "direction": np.tile(directions.astype(int), len(frames)),
        }
    )
    return lane_order_parameter(
        data,
        y_min=0.0,
        y_max=width,
        bins=8,
        frame_stride=20,
        minimum_bin_count=3,
    )


def periodic_minimum_separation(result, length):
    minimum = float("inf")
    for positions in result.positions:
        for first in range(len(positions)):
            delta = positions[first + 1 :] - positions[first]
            if len(delta) == 0:
                continue
            delta[:, 0] = (delta[:, 0] + length / 2.0) % length - length / 2.0
            minimum = min(minimum, float(np.linalg.norm(delta, axis=1).min()))
    return minimum


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(20260722)
    agents = 60
    length = 10.0
    width = 4.0
    positions = nonoverlapping_positions(rng, agents, length, width)
    directions = np.ones(agents)
    directions[agents // 2 :] = -1.0
    rng.shuffle(directions)

    base = AnisotropicParameters()
    isotropic = simulate_periodic_corridor(
        positions,
        directions,
        duration=25.0,
        step=0.05,
        corridor_length=length,
        corridor_width=width,
        parameters=replace(base, metric_strength=0.0),
    )
    anisotropic = simulate_periodic_corridor(
        positions,
        directions,
        duration=25.0,
        step=0.05,
        corridor_length=length,
        corridor_width=width,
        parameters=base,
    )
    avm = simulate_avm_periodic_corridor(
        positions,
        directions,
        duration=25.0,
        step=0.05,
        corridor_length=length,
        corridor_width=width,
    )
    isotropic_order = order_history(isotropic, directions, width)
    anisotropic_order = order_history(anisotropic, directions, width)
    avm_order = order_history(avm, directions, width)

    rows = []
    for name, result, order in [
        ("isotropic_restriction", isotropic, isotropic_order),
        ("anisotropic_pilot", anisotropic, anisotropic_order),
        ("avm_baseline", avm, avm_order),
    ]:
        rows.append(
            {
                "model": name,
                "initial_lane_order": float(order["lane_order"].iloc[:5].mean()),
                "final_lane_order": float(order["lane_order"].iloc[-5:].mean()),
                "mean_lane_order": float(order["lane_order"].mean()),
                "minimum_separation_m": periodic_minimum_separation(result, length),
                "mean_speed_mps": float(np.linalg.norm(result.velocities, axis=2).mean()),
            }
        )

    output = project / "data" / "processed" / "corridor_lane_formation.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(
        isotropic_order["frame"] * 0.05,
        isotropic_order["lane_order"],
        label="isotropic restriction",
        color="#6b7280",
    )
    axes[0].plot(
        anisotropic_order["frame"] * 0.05,
        anisotropic_order["lane_order"],
        label="anisotropic pilot",
        color="#059669",
    )
    axes[0].plot(
        avm_order["frame"] * 0.05,
        avm_order["lane_order"],
        label="AVM baseline",
        color="#7c3aed",
    )
    axes[0].set(
        xlabel="time (s)",
        ylabel="directional segregation",
        title="Lane-order evolution",
        ylim=(0.0, 1.0),
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    final = avm.positions[-1]
    colors = np.where(directions > 0, "#2563eb", "#dc2626")
    axes[1].scatter(final[:, 0], final[:, 1], c=colors, s=24, alpha=0.8)
    axes[1].set(
        xlabel="periodic x (m)",
        ylabel="y (m)",
        title="Uncalibrated AVM baseline at 25 s",
        xlim=(-length / 2.0, length / 2.0),
        ylim=(0.0, width),
    )
    axes[1].grid(alpha=0.2)
    figure.suptitle("Controlled bidirectional-corridor stress test")
    figure.tight_layout()
    figure.savefig(project / "figures" / "corridor_lane_formation.png", dpi=220)
    plt.close(figure)

    for row in rows:
        print(row)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
