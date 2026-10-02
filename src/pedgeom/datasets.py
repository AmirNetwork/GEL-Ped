# Author: Amir Ghorbani
"""Loaders and descriptive metrics for empirical pedestrian trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


TRAJECTORY_COLUMNS = ["pedestrian_id", "frame", "x_cm", "y_cm", "height_cm"]
GoalMethod = Literal["endpoint", "entry", "prior_velocity"]


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


def resample_trajectory_to_fps(
    data: pd.DataFrame,
    *,
    target_fps: float = 10.0,
) -> pd.DataFrame:
    """Linearly resample trajectories onto a common integer-frame time grid.

    The target grid is anchored at time zero, so target frame ``k`` always
    represents ``k / target_fps`` seconds for every pedestrian.  Each
    pedestrian is evaluated only at grid times inside its own observed time
    support; ``numpy.interp`` therefore never extrapolates.  The returned
    schema is directly consumable by :func:`add_motion_features` and
    ``build_velocity_samples(loader=...)``.
    """

    if target_fps <= 0.0:
        raise ValueError("target_fps must be positive")
    required = {"pedestrian_id", "time_s", "x_m", "y_m"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"trajectory data are missing required columns: {sorted(missing)}")
    columns = ["pedestrian_id", "frame", "time_s", "x_m", "y_m", "height_m"]
    if data.empty:
        return pd.DataFrame(columns=columns)

    blocks: list[pd.DataFrame] = []
    tolerance = 1e-10
    for pedestrian_id, group in data.groupby("pedestrian_id", sort=False):
        ordered = group.sort_values("time_s", kind="stable")
        source_time = ordered["time_s"].to_numpy(float)
        if not np.all(np.isfinite(source_time)):
            raise ValueError("trajectory times must be finite")
        if len(source_time) > 1 and np.any(np.diff(source_time) <= 0.0):
            raise ValueError("each pedestrian must have strictly increasing times")

        first_frame = int(np.ceil(source_time[0] * target_fps - tolerance))
        last_frame = int(np.floor(source_time[-1] * target_fps + tolerance))
        if last_frame < first_frame:
            continue
        target_frame = np.arange(first_frame, last_frame + 1, dtype=int)
        target_time = target_frame.astype(float) / target_fps
        # Guard explicitly against floating-point tolerance ever admitting an
        # out-of-support grid point.
        inside = (target_time >= source_time[0] - tolerance) & (
            target_time <= source_time[-1] + tolerance
        )
        target_frame = target_frame[inside]
        target_time = target_time[inside]
        if len(target_frame) == 0:
            continue

        if "height_m" in ordered:
            source_height = ordered["height_m"].to_numpy(float)
            height = np.interp(target_time, source_time, source_height)
        else:
            height = np.full(len(target_time), np.nan, dtype=float)
        blocks.append(
            pd.DataFrame(
                {
                    "pedestrian_id": [pedestrian_id] * len(target_frame),
                    "frame": target_frame,
                    "time_s": target_time,
                    "x_m": np.interp(
                        target_time, source_time, ordered["x_m"].to_numpy(float)
                    ),
                    "y_m": np.interp(
                        target_time, source_time, ordered["y_m"].to_numpy(float)
                    ),
                    "height_m": height,
                }
            )
        )

    if not blocks:
        return pd.DataFrame(columns=columns)
    result = pd.concat(blocks, ignore_index=True)
    result["pedestrian_id"] = result["pedestrian_id"].astype(data["pedestrian_id"].dtype)
    return result.sort_values(["pedestrian_id", "frame"], ignore_index=True)


def load_resampled_julich_trajectory(
    path: str | Path,
    source_fps: float = 16.0,
    target_fps: float = 10.0,
) -> pd.DataFrame:
    """Load centimetre-format Juelich data and resample it to an exact time grid."""

    if source_fps <= 0.0:
        raise ValueError("source_fps must be positive")
    source = load_julich_trajectory(path, fps=source_fps)
    return resample_trajectory_to_fps(source, target_fps=target_fps)


def load_julich_crossing_trajectory(path: str | Path, fps: float = 25.0) -> pd.DataFrame:
    """Load the newer PeTrack crossing format whose coordinates are already in metres."""

    source = Path(path)
    columns = [
        "pedestrian_id",
        "frame",
        "x_m",
        "y_m",
        "height_m",
        "rotation_rad",
        "marker_id",
        "flag",
    ]
    data = pd.read_csv(source, sep=r"\s+", comment="#", header=None, names=columns)
    data["time_s"] = data["frame"] / fps
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


def add_goal_directions(
    data: pd.DataFrame,
    minimum_displacement: float = 1.0,
    cardinal_routes: bool = True,
) -> pd.DataFrame:
    """Assign a route direction from endpoints without using future path curvature.

    Controlled corridor and crossing experiments use axis-aligned entrances and exits.
    Quantizing to the dominant cardinal direction uses endpoints only to recover that known
    route class, rather than leaking the detailed future trajectory into the forecast.
    """

    result = data.copy()
    groups = result.groupby("pedestrian_id", sort=False)
    dx = groups["x_m"].transform("last") - groups["x_m"].transform("first")
    dy = groups["y_m"].transform("last") - groups["y_m"].transform("first")
    norm = np.hypot(dx, dy)
    valid = norm >= minimum_displacement
    if cardinal_routes:
        horizontal = np.abs(dx) >= np.abs(dy)
        result["goal_x"] = np.where(valid & horizontal, np.sign(dx), 0.0)
        result["goal_y"] = np.where(valid & ~horizontal, np.sign(dy), 0.0)
    else:
        result["goal_x"] = np.divide(dx, norm, out=np.zeros_like(dx), where=valid)
        result["goal_y"] = np.divide(dy, norm, out=np.zeros_like(dy), where=valid)
    result["goal_valid"] = valid
    result["goal_ready_frame"] = groups["frame"].transform("first") if "frame" in result else 0
    return result


def _assign_unit_directions(
    result: pd.DataFrame,
    dx: pd.Series,
    dy: pd.Series,
    valid: pd.Series,
    cardinal_routes: bool,
) -> pd.DataFrame:
    """Assign unit or cardinal directions to an existing trajectory table."""

    norm = np.hypot(dx, dy)
    if cardinal_routes:
        horizontal = np.abs(dx) >= np.abs(dy)
        result["goal_x"] = np.where(valid & horizontal, np.sign(dx), 0.0)
        result["goal_y"] = np.where(valid & ~horizontal, np.sign(dy), 0.0)
    else:
        result["goal_x"] = np.divide(dx, norm, out=np.zeros_like(dx), where=valid)
        result["goal_y"] = np.divide(dy, norm, out=np.zeros_like(dy), where=valid)
    result["goal_valid"] = valid
    return result


def add_entry_goal_directions(
    data: pd.DataFrame,
    history_frames: int = 10,
    minimum_displacement: float = 0.10,
    cardinal_routes: bool = True,
) -> pd.DataFrame:
    """Infer a fixed route class from the first 0.4 s of each observed trajectory.

    The direction is available once ``history_frames`` have elapsed and uses no sample after
    the prediction time. This is the primary leakage-free route estimator for the controlled
    axis-aligned experiments.
    """

    if history_frames <= 0:
        raise ValueError("history_frames must be positive")
    result = data.copy()
    records = []
    for pedestrian_id, group in result.groupby("pedestrian_id", sort=False):
        ordered = group.sort_values("frame")
        first = ordered.iloc[0]
        candidates = ordered[ordered["frame"] >= int(first["frame"]) + history_frames]
        if candidates.empty:
            records.append((pedestrian_id, np.nan, np.nan, np.inf))
            continue
        ready = candidates.iloc[0]
        records.append(
            (
                pedestrian_id,
                float(ready["x_m"] - first["x_m"]),
                float(ready["y_m"] - first["y_m"]),
                int(ready["frame"]),
            )
        )
    route = pd.DataFrame(
        records,
        columns=["pedestrian_id", "route_dx", "route_dy", "goal_ready_frame"],
    )
    result = result.merge(route, on="pedestrian_id", how="left", validate="many_to_one")
    norm = np.hypot(result["route_dx"], result["route_dy"])
    valid = (norm >= minimum_displacement) & (result["frame"] >= result["goal_ready_frame"])
    result = _assign_unit_directions(
        result,
        result["route_dx"],
        result["route_dy"],
        valid,
        cardinal_routes,
    )
    return result.drop(columns=["route_dx", "route_dy"])


def add_prior_velocity_goal_directions(
    data: pd.DataFrame,
    history_frames: int = 10,
    minimum_displacement: float = 0.08,
    cardinal_routes: bool = False,
) -> pd.DataFrame:
    """Use only the displacement ending at the prediction time as the route direction."""

    if history_frames <= 0:
        raise ValueError("history_frames must be positive")
    result = data.copy()
    previous = result[["pedestrian_id", "frame", "x_m", "y_m"]].copy()
    previous["frame"] += history_frames
    previous = previous.rename(columns={"x_m": "route_previous_x", "y_m": "route_previous_y"})
    result = result.merge(previous, on=["pedestrian_id", "frame"], how="left", validate="one_to_one")
    dx = result["x_m"] - result["route_previous_x"]
    dy = result["y_m"] - result["route_previous_y"]
    valid = np.hypot(dx, dy) >= minimum_displacement
    result = _assign_unit_directions(result, dx, dy, valid, cardinal_routes)
    result["goal_ready_frame"] = result["frame"]
    return result.drop(columns=["route_previous_x", "route_previous_y"])


def add_prediction_goal_directions(
    data: pd.DataFrame,
    method: GoalMethod,
    history_frames: int = 10,
    cardinal_routes: bool | None = None,
) -> pd.DataFrame:
    """Dispatch the documented route-information alternatives used in the ablation."""

    if method == "endpoint":
        return add_goal_directions(
            data,
            cardinal_routes=True if cardinal_routes is None else cardinal_routes,
        )
    if method == "entry":
        return add_entry_goal_directions(
            data,
            history_frames=history_frames,
            cardinal_routes=True if cardinal_routes is None else cardinal_routes,
        )
    if method == "prior_velocity":
        return add_prior_velocity_goal_directions(
            data,
            history_frames=history_frames,
            cardinal_routes=False if cardinal_routes is None else cardinal_routes,
        )
    raise ValueError(f"unknown goal-direction method: {method}")


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
