# Author: Amir Ghorbani
"""Summarize every run in the Jülich bidirectional-corridor archive."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pedgeom.datasets import add_motion_features, lane_order_parameter, load_julich_trajectory


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    raw = project / "data" / "raw" / "2013bidirectional"
    metadata = json.loads((raw / "metadata.json").read_text(encoding="utf-8"))
    run_metadata = {
        item["run_name"]: item for item in metadata["experiment"]["run"]
    }
    rows = []

    for source in sorted(raw.glob("bi_corr_*.txt")):
        data = add_motion_features(load_julich_trajectory(source))
        order = lane_order_parameter(data)
        plausible_speeds = data.loc[data["speed_mps"].between(0.0, 3.0), "speed_mps"]
        occupancy = data.groupby("frame").size()
        info = run_metadata[source.stem]
        parameters = info["parameter"]
        rows.append(
            {
                "run": source.stem,
                "pedestrians": data["pedestrian_id"].nunique(),
                "duration_s": (data["frame"].max() - data["frame"].min()) / 25.0,
                "mean_occupancy": occupancy.mean(),
                "maximum_occupancy": occupancy.max(),
                "median_speed_mps": plausible_speeds.median(),
                "mean_lane_order": order["lane_order"].mean(),
                "entrance_width_m": float(parameters["entrance width [m]"]),
                "corridor_length_m": float(parameters["corridor length [m]"]),
                "exit_width_m": float(parameters["exit width [m]"]),
                "participant_information": parameters["participant information"],
                "screen_information": parameters["screen information"],
            }
        )
        print(f"summarized {source.name}")

    catalog = pd.DataFrame(rows)
    output = project / "data" / "processed" / "empirical_run_catalog.csv"
    catalog.to_csv(output, index=False)

    comparable = catalog[
        (catalog["corridor_length_m"] == 22.0) & (catalog["exit_width_m"] == 5.0)
    ].copy()
    informed = comparable["participant_information"] != "none"
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for mask, label, color in [
        (~informed, "no routing instruction", "#2563eb"),
        (informed, "assigned exits", "#dc2626"),
    ]:
        axes[0].scatter(
            comparable.loc[mask, "entrance_width_m"],
            comparable.loc[mask, "mean_lane_order"],
            label=label,
            color=color,
        )
        axes[1].scatter(
            comparable.loc[mask, "mean_occupancy"],
            comparable.loc[mask, "median_speed_mps"],
            label=label,
            color=color,
        )
    axes[0].set(
        xlabel="entrance width (m)",
        ylabel="mean lane order",
        title="Directional segregation",
        ylim=(0.0, 1.0),
    )
    axes[1].set(
        xlabel="mean tracked occupancy",
        ylabel="median finite-difference speed (m/s)",
        title="Speed versus occupancy",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("Jülich bidirectional corridor: run-level empirical targets")
    figure.tight_layout()
    figure.savefig(project / "figures" / "empirical_run_catalog.png", dpi=220)
    plt.close(figure)
    print(catalog.to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
