"""Loaders and descriptive metrics for empirical pedestrian trajectories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TRAJECTORY_COLUMNS = ["pedestrian_id", "frame", "x_cm", "y_cm", "height_cm"]


def load_julich_trajectory(path: str | Path, fps: float = 25.0) -> pd.DataFrame:
    """Load a Jülich PeTrack text trajectory and convert coordinates to SI units."""

    source = Path(path)
    data = pd.read_csv(source, sep=r"\s+", header=None, names=TRAJECTORY_COLUMNS)
    data = data.assign(
        time_s=data["frame"] / fps,
        x_m=data["x_cm"] / 100.0,
        y_m=data["y_cm"] / 100.0,
        height_m=data["height_cm"] / 100.0,
    )
    return data[
        ["pedestrian_id", "frame", "time_s", "x_m", "y_m", "height_m"]
    ].sort_values(["pedestrian_id", "frame"], ignore_index=True)


def add_motion_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add trajectory direction and finite-difference velocity estimates."""

    result = data.copy()
    groups = result.groupby("pedestrian_id", sort=False)
    first_x = groups["x_m"].transform("first")
    last_x = groups["x_m"].transform("last")
    result["direction"] = np.where(last_x >= first_x, 1, -1).astype(np.int8)
    dt = groups["time_s"].diff()
    result["vx_mps"] = groups["x_m"].diff() / dt
    result["vy_mps"] = groups["y_m"].diff() / dt
    result["speed_mps"] = np.hypot(result["vx_mps"], result["vy_mps"])
    return result


def lane_order_parameter(
    data: pd.DataFrame,
    y_min: float = 0.0,
    y_max: float = 4.0,
    bins: int = 8,
    frame_stride: int = 25,
    minimum_bin_count: int = 5,
) -> pd.DataFrame:
    """Compute a binned directional-segregation index in [0, 1].

    Zero indicates locally balanced counterflow and one indicates locally unidirectional
    lanes. Frames are subsampled for an inexpensive descriptive diagnostic.
    """

    selected_frames = np.sort(data["frame"].unique())[::frame_stride]
    selected = data[data["frame"].isin(selected_frames)].copy()
    edges = np.linspace(y_min, y_max, bins + 1)
    selected["y_bin"] = pd.cut(selected["y_m"], edges, labels=False, include_lowest=True)
    grouped = (
        selected.dropna(subset=["y_bin"])
        .groupby(["frame", "y_bin"], observed=True)["direction"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grouped = grouped[grouped["count"] >= minimum_bin_count]
    grouped["weighted_order"] = grouped["mean"].abs() * grouped["count"]
    numerator = grouped.groupby("frame")["weighted_order"].sum()
    denominator = grouped.groupby("frame")["count"].sum()
    order = (numerator / denominator).rename("lane_order").reset_index()
    return order

