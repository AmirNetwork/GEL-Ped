# Author: Amir Ghorbani
"""Quantify high-frequency coordinate variation in the Juelich trajectories.

The audit compares the released head tracks with a past/future-neutral
Savitzky--Golay smooth used only for sensitivity analysis.  It reports both
pointwise coordinate differences and their effect on 0.4-s displacement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from pedgeom.datasets import load_julich_crossing_trajectory, load_julich_trajectory
from major_revision_analysis import load_inputs, project_root


def audit_run(path: Path, *, crossing: bool) -> dict[str, float | str | int]:
    loader = load_julich_crossing_trajectory if crossing else load_julich_trajectory
    raw = loader(path)
    smooth = raw.copy()
    for _, indices in smooth.groupby("pedestrian_id", sort=False).groups.items():
        index = np.asarray(list(indices))
        count = len(index)
        window = min(11, count if count % 2 == 1 else count - 1)
        if window >= 5:
            smooth.loc[index, "x_m"] = savgol_filter(
                smooth.loc[index, "x_m"], window, 2
            )
            smooth.loc[index, "y_m"] = savgol_filter(
                smooth.loc[index, "y_m"], window, 2
            )

    point_delta = np.linalg.norm(
        raw[["x_m", "y_m"]].to_numpy() - smooth[["x_m", "y_m"]].to_numpy(),
        axis=1,
    )
    future_raw = raw[["pedestrian_id", "frame", "x_m", "y_m"]].copy()
    future_smooth = smooth[["pedestrian_id", "frame", "x_m", "y_m"]].copy()
    future_raw["frame"] -= 10
    future_smooth["frame"] -= 10
    future_raw = future_raw.rename(columns={"x_m": "future_x", "y_m": "future_y"})
    future_smooth = future_smooth.rename(
        columns={"x_m": "future_x", "y_m": "future_y"}
    )
    key = ["pedestrian_id", "frame"]
    raw_pair = raw.merge(future_raw, on=key, how="inner")
    smooth_pair = smooth.merge(future_smooth, on=key, how="inner")
    raw_disp = raw_pair[["future_x", "future_y"]].to_numpy() - raw_pair[
        ["x_m", "y_m"]
    ].to_numpy()
    smooth_disp = smooth_pair[["future_x", "future_y"]].to_numpy() - smooth_pair[
        ["x_m", "y_m"]
    ].to_numpy()
    displacement_delta = np.linalg.norm(raw_disp - smooth_disp, axis=1)
    return {
        "run": path.stem,
        "dataset": "crossing topology" if crossing else "corridor",
        "points": len(point_delta),
        "median_coordinate_delta_m": float(np.median(point_delta)),
        "q90_coordinate_delta_m": float(np.quantile(point_delta, 0.90)),
        "rms_coordinate_delta_m": float(np.sqrt(np.mean(point_delta**2))),
        "median_04s_displacement_delta_m": float(np.median(displacement_delta)),
        "q90_04s_displacement_delta_m": float(
            np.quantile(displacement_delta, 0.90)
        ),
    }


def main() -> None:
    project = project_root()
    _, splits = load_inputs(project)
    corridor = project / "data/raw/2013bidirectional"
    crossing = project / "data/raw/2013crossing90/trajectories"
    corridor_runs = (
        splits["calibration"]
        + splits["held_out_validation"]
        + splits["held_out_geometry_stress"]
    )
    rows = [audit_run(corridor / f"{run}.txt", crossing=False) for run in corridor_runs]
    rows.extend(
        audit_run(path, crossing=True)
        for path in sorted(crossing.glob("crossing_90_[de]_*.txt"))
    )
    result = pd.DataFrame(rows)
    result.to_csv(project / "data/processed/measurement_noise_audit.csv", index=False)
    print(result.groupby("dataset").mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
