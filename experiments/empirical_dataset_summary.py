# Author: Amir Ghorbani
"""Create the first reproducible summary of a held-out empirical dataset."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pedgeom.datasets import add_motion_features, lane_order_parameter, load_julich_trajectory


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    source = project / "data" / "raw" / "2013bidirectional" / "bi_corr_400_a_03.txt"
    data = add_motion_features(load_julich_trajectory(source))
    order = lane_order_parameter(data)

    plausible_speeds = data.loc[data["speed_mps"].between(0.0, 3.0), "speed_mps"]
    direction_counts = (
        data.drop_duplicates("pedestrian_id")["direction"].value_counts().to_dict()
    )
    summary = {
        "run": source.stem,
        "rows": int(len(data)),
        "pedestrians": int(data["pedestrian_id"].nunique()),
        "duration_s": float((data["frame"].max() - data["frame"].min()) / 25.0),
        "x_range_m": [float(data["x_m"].min()), float(data["x_m"].max())],
        "y_range_m": [float(data["y_m"].min()), float(data["y_m"].max())],
        "direction_positive_count": int(direction_counts.get(1, 0)),
        "direction_negative_count": int(direction_counts.get(-1, 0)),
        "median_speed_mps": float(plausible_speeds.median()),
        "speed_iqr_mps": [
            float(plausible_speeds.quantile(0.25)),
            float(plausible_speeds.quantile(0.75)),
        ],
        "mean_lane_order": float(order["lane_order"].mean()),
        "lane_order_iqr": [
            float(order["lane_order"].quantile(0.25)),
            float(order["lane_order"].quantile(0.75)),
        ],
    }

    output = project / "data" / "processed" / "empirical_dataset_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rng = np.random.default_rng(20260722)
    identifiers = data["pedestrian_id"].unique()
    sample = rng.choice(identifiers, size=min(70, len(identifiers)), replace=False)
    for identifier in sample:
        trajectory = data[data["pedestrian_id"] == identifier]
        color = "#2563eb" if trajectory["direction"].iat[0] > 0 else "#dc2626"
        axes[0].plot(trajectory["x_m"], trajectory["y_m"], color=color, alpha=0.35, linewidth=0.8)

    axes[0].set(
        xlabel="x (m)",
        ylabel="y (m)",
        title="Sample measured trajectories",
        xlim=(-5.8, 4.8),
        ylim=(0.0, 4.8),
    )
    axes[0].grid(alpha=0.2)
    axes[1].plot(order["frame"] / 25.0, order["lane_order"], color="#059669")
    axes[1].axhline(order["lane_order"].mean(), color="black", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="time (s)",
        ylabel="directional segregation",
        title="Empirical lane-order diagnostic",
        ylim=(0.0, 1.0),
    )
    axes[1].grid(alpha=0.2)
    figure.suptitle("Jülich bidirectional corridor: bi_corr_400_a_03")
    figure.tight_layout()
    figure.savefig(project / "figures" / "empirical_dataset_summary.png", dpi=220)
    plt.close(figure)

    print(json.dumps(summary, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
