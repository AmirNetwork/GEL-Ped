"""Demonstrate the timestep dependence introduced by resetting velocity."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pedgeom import LegacyParameters, simulate_legacy_reset


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    positions = np.array([[6.0, 2.5], [3.0, 2.5]])
    destinations = np.array([[0.0, 2.0], [10.0, 2.0]])
    parameters = LegacyParameters(destination_form="linear")
    steps = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])
    rows = []

    for step in steps:
        result = simulate_legacy_reset(
            positions, destinations, duration=1.0, step=float(step), parameters=parameters
        )
        displacement = np.linalg.norm(result.positions[-1] - positions, axis=1).mean()
        mean_speed = np.linalg.norm(result.velocities[-1], axis=1).mean()
        rows.append(
            {
                "timestep_s": step,
                "displacement_after_1s_m": displacement,
                "reported_final_speed_mps": mean_speed,
            }
        )

    output = project / "data" / "processed" / "timestep_sensitivity.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    axis.plot(
        [row["timestep_s"] for row in rows],
        [row["displacement_after_1s_m"] for row in rows],
        "o-",
        label="reset simulation",
    )
    reference = rows[1]["displacement_after_1s_m"] * steps / steps[1]
    axis.plot(steps, reference, "--", label="linear in timestep")
    axis.set(
        xlabel="scene timestep (s)",
        ylabel="mean displacement after 1 s (m)",
        title="Velocity reset prevents timestep convergence",
    )
    axis.grid(which="both", alpha=0.25)
    axis.set_xlim(left=0.0)
    axis.set_ylim(bottom=0.0)
    axis.legend()
    figure.tight_layout()
    figure.savefig(project / "figures" / "timestep_sensitivity.png", dpi=220)
    plt.close(figure)

    for row in rows:
        print(row)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
