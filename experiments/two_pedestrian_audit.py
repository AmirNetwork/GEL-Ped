"""Compare the paper update, corrected ODE, and anisotropic pilot."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pedgeom import LegacyParameters, simulate_anisotropic, simulate_corrected, simulate_legacy_reset


OBSERVATION_TIMES = np.arange(0.0, 4.01, 0.5)
OBSERVED_POSITIONS = np.array(
    [
        [[6.00, 2.50], [3.00, 2.50]],
        [[5.35, 2.45], [3.64, 2.45]],
        [[4.85, 2.39], [4.14, 2.41]],
        [[4.83, 2.32], [4.16, 2.40]],
        [[4.80, 2.16], [4.19, 2.51]],
        [[4.44, 1.87], [4.55, 2.79]],
        [[3.79, 1.92], [5.19, 2.78]],
        [[3.14, 2.02], [5.82, 2.66]],
        [[2.48, 2.08], [6.46, 2.54]],
    ]
)
OBSERVED_SPEEDS = np.array(
    [
        [1.30, 1.30],
        [1.30, 1.30],
        [0.51, 0.50],
        [0.20, 0.00],
        [0.51, 0.41],
        [1.26, 1.26],
        [1.33, 1.32],
        [1.30, 1.32],
        [1.40, 1.32],
    ]
)


def interpolate_history(times, values):
    output = np.empty((len(OBSERVATION_TIMES), *values.shape[1:]))
    for agent in range(values.shape[1]):
        for component in range(values.shape[2]):
            output[:, agent, component] = np.interp(
                OBSERVATION_TIMES, times, values[:, agent, component]
            )
    return output


def errors(result):
    predicted_positions = interpolate_history(result.times, result.positions)
    speed_history = np.linalg.norm(result.velocities, axis=2)[..., None]
    predicted_speeds = interpolate_history(result.times, speed_history)[..., 0]
    position_rmse = float(np.sqrt(np.mean((predicted_positions - OBSERVED_POSITIONS) ** 2)))
    speed_rmse = float(np.sqrt(np.mean((predicted_speeds - OBSERVED_SPEEDS) ** 2)))
    return position_rmse, speed_rmse


def plot_results(results, project):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    colors = ["#6b7280", "#2563eb", "#dc2626", "#059669"]
    for result, color in zip(results, colors, strict=True):
        if result.model_name == "inferred_linear_persistent_ode":
            continue
        axes[0].plot(
            result.positions[:, 0, 0],
            result.positions[:, 0, 1],
            color=color,
            label=result.model_name,
        )
        axes[0].plot(
            result.positions[:, 1, 0], result.positions[:, 1, 1], color=color, linestyle="--"
        )
        mean_speed = np.linalg.norm(result.velocities, axis=2).mean(axis=1)
        axes[1].plot(result.times, mean_speed, color=color, label=result.model_name)

    axes[0].scatter(
        OBSERVED_POSITIONS[:, 0, 0],
        OBSERVED_POSITIONS[:, 0, 1],
        marker="o",
        facecolors="none",
        edgecolors="black",
        label="paper Table 2",
        zorder=10,
    )
    axes[0].scatter(
        OBSERVED_POSITIONS[:, 1, 0],
        OBSERVED_POSITIONS[:, 1, 1],
        marker="s",
        facecolors="none",
        edgecolors="black",
        zorder=10,
    )
    axes[1].scatter(OBSERVATION_TIMES, OBSERVED_SPEEDS.mean(axis=1), color="black", s=20)
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Head-on trajectories")
    axes[1].set(xlabel="time (s)", ylabel="mean speed (m/s)", title="Speed response")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)
    axes[1].set_ylim(bottom=0.0)
    unstable = next(result for result in results if result.model_name == "inferred_linear_persistent_ode")
    unstable_speed = float(np.linalg.norm(unstable.velocities, axis=2).max())
    axes[1].text(
        0.03,
        0.92,
        f"Persistent ODE omitted from plot: max speed {unstable_speed:.1f} m/s",
        transform=axes[1].transAxes,
        fontsize=8,
        color="#dc2626",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    figure.suptitle("Submitted results versus stable executable reconstructions")
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(project / "figures" / "two_pedestrian_audit.png", dpi=220)
    plt.close(figure)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    output = project / "data" / "processed" / "two_pedestrian_summary.csv"

    positions = np.array([[6.0, 2.5], [3.0, 2.5]])
    destinations = np.array([[0.0, 2.0], [10.0, 2.0]])
    initial_velocities = np.array([[-1.3, 0.0], [1.3, 0.0]])

    inferred_parameters = LegacyParameters(destination_form="linear")
    as_printed = replace(
        simulate_legacy_reset(positions, destinations, duration=4.0, step=0.1),
        model_name="as_printed_reciprocal_reset",
    )
    inferred_reset = replace(
        simulate_legacy_reset(
            positions,
            destinations,
            duration=4.0,
            step=0.1,
            parameters=inferred_parameters,
        ),
        model_name="inferred_linear_reset",
    )
    inferred_persistent = replace(
        simulate_corrected(
            positions,
            initial_velocities,
            destinations,
            duration=4.0,
            step=0.1,
            parameters=inferred_parameters,
        ),
        model_name="inferred_linear_persistent_ode",
    )
    results = [
        as_printed,
        inferred_reset,
        inferred_persistent,
        simulate_anisotropic(
            positions, initial_velocities, destinations, duration=4.0, step=0.02
        ),
    ]

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "minimum_separation_m",
                "maximum_speed_mps",
                "position_rmse_m",
                "speed_rmse_mps",
                "final_x_agent_0_m",
                "final_y_agent_0_m",
                "final_x_agent_1_m",
                "final_y_agent_1_m",
            ],
        )
        writer.writeheader()
        for result in results:
            final = result.positions[-1]
            position_rmse, speed_rmse = errors(result)
            writer.writerow(
                {
                    "model": result.model_name,
                    "minimum_separation_m": result.minimum_separation,
                    "maximum_speed_mps": float(np.linalg.norm(result.velocities, axis=2).max()),
                    "position_rmse_m": position_rmse,
                    "speed_rmse_mps": speed_rmse,
                    "final_x_agent_0_m": final[0, 0],
                    "final_y_agent_0_m": final[0, 1],
                    "final_x_agent_1_m": final[1, 0],
                    "final_y_agent_1_m": final[1, 1],
                }
            )

    plot_results(results, project)

    for result in results:
        maximum_speed = float(np.linalg.norm(result.velocities, axis=2).max())
        position_rmse, speed_rmse = errors(result)
        print(
            f"{result.model_name:18s} min_sep={result.minimum_separation:.3f} m "
            f"max_speed={maximum_speed:.3f} m/s position_rmse={position_rmse:.3f} m "
            f"speed_rmse={speed_rmse:.3f} m/s"
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
