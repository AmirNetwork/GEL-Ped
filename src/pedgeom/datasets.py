# Author: Amir Ghorbani
"""Loaders and descriptive metrics for empirical pedestrian trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from zipfile import ZipFile

import numpy as np
import pandas as pd


TRAJECTORY_COLUMNS = ["pedestrian_id", "frame", "x_cm", "y_cm", "height_cm"]
GoalMethod = Literal["endpoint", "entry", "prior_velocity"]


def _parse_julich_rows(source) -> pd.DataFrame:
    data = pd.read_csv(source, sep=r"\s+", header=None, comment="#")
    if data.shape[1] != len(TRAJECTORY_COLUMNS):
        raise ValueError(
            "Jülich trajectory rows must contain exactly five columns; "
            f"found {data.shape[1]}"
        )
    data.columns = TRAJECTORY_COLUMNS
    try:
        data = data.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Jülich trajectory columns must be numeric") from error
    if data.empty:
        raise ValueError("Jülich trajectory must contain at least one row")
    if not np.all(np.isfinite(data.to_numpy(float))):
        raise ValueError("Jülich trajectory columns must be finite")
    for name in ("pedestrian_id", "frame"):
        values = data[name].to_numpy(float)
        if not np.all(values == np.round(values)):
            raise ValueError(f"{name} must contain integers")
        data[name] = values.astype(np.int64)
    if data.duplicated(["pedestrian_id", "frame"]).any():
        raise ValueError("Jülich trajectory contains duplicate pedestrian/frame rows")
    return data


def _normalise_julich_trajectory(data: pd.DataFrame, fps: float) -> pd.DataFrame:
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    data = data.assign(
        time_s=data["frame"] / fps,
        x_m=data["x_cm"] / 100.0,
        y_m=data["y_cm"] / 100.0,
        height_m=data["height_cm"] / 100.0,
    )
    return data[
        ["pedestrian_id", "frame", "time_s", "x_m", "y_m", "height_m"]
    ].sort_values(["pedestrian_id", "frame"], ignore_index=True)


def load_julich_trajectory(path: str | Path, fps: float = 25.0) -> pd.DataFrame:
    """Load a Jülich PeTrack text trajectory and convert coordinates to SI units."""

    source = Path(path)
    data = _parse_julich_rows(source)
    return _normalise_julich_trajectory(data, fps)


def load_julich_archive_trajectory(
    archive_path: str | Path,
    member: str,
    *,
    fps: float = 16.0,
) -> pd.DataFrame:
    """Load one flat PeTrack trajectory member directly from an official ZIP.

    Reading the member in place avoids extracting an untouched evaluation
    archive.  Directory components are rejected so the caller must identify an
    exact top-level member from the frozen archive registry.
    """

    if Path(member).name != member or not member.lower().endswith(".txt"):
        raise ValueError("member must be a flat .txt archive entry")
    with ZipFile(Path(archive_path)) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError as error:
            raise ValueError(f"archive member not found: {member}") from error
        if info.is_dir():
            raise ValueError(f"archive member is a directory: {member}")
        with archive.open(info) as stream:
            data = _parse_julich_rows(stream)
    return _normalise_julich_trajectory(data, fps)


def resample_trajectory_to_fps(
    data: pd.DataFrame,
    *,
    target_fps: float = 10.0,
    source_fps: float = 16.0,
    maximum_gap_factor: float = 1.5,
) -> pd.DataFrame:
    """Causally project trajectories onto a common integer-frame time grid.

    The target grid is anchored at time zero, so target frame ``k`` always
    represents ``k / target_fps`` seconds for every pedestrian.  At each grid
    time, position and height are linearly projected from the two most recent
    source observations at or before that time.  Thus no observation after the
    forecast origin is used.  A target is dropped unless both source samples
    are consecutive enough and the latest sample is recent enough; this avoids
    bridging internal tracking gaps.  Targets outside observed support and
    targets without two prior observations are also dropped.

    ``maximum_gap_factor`` multiplies the nominal source interval.  The frozen
    16-Hz protocol uses 1.5, allowing harmless timestamp tolerance while still
    rejecting a missing frame, whose gap is two nominal intervals.
    """

    if target_fps <= 0.0:
        raise ValueError("target_fps must be positive")
    if source_fps <= 0.0:
        raise ValueError("source_fps must be positive")
    if not np.isfinite(maximum_gap_factor) or maximum_gap_factor < 1.0:
        raise ValueError("maximum_gap_factor must be finite and at least one")
    required = {"pedestrian_id", "time_s", "x_m", "y_m"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"trajectory data are missing required columns: {sorted(missing)}")
    columns = ["pedestrian_id", "frame", "time_s", "x_m", "y_m", "height_m"]
    if data.empty:
        return pd.DataFrame(columns=columns)

    blocks: list[pd.DataFrame] = []
    tolerance = 1e-10
    maximum_gap_s = maximum_gap_factor / source_fps
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
        if len(target_frame) == 0 or len(source_time) < 2:
            continue

        latest = np.searchsorted(source_time, target_time + tolerance, side="right") - 1
        previous = latest - 1
        enough_history = previous >= 0
        safe_previous = np.maximum(previous, 0)
        safe_latest = np.maximum(latest, 0)
        pair_gap = source_time[safe_latest] - source_time[safe_previous]
        sample_age = target_time - source_time[safe_latest]
        valid = (
            enough_history
            & (pair_gap > tolerance)
            & (pair_gap <= maximum_gap_s + tolerance)
            & (sample_age >= -tolerance)
            & (sample_age <= maximum_gap_s + tolerance)
        )
        target_frame = target_frame[valid]
        target_time = target_time[valid]
        previous = previous[valid]
        latest = latest[valid]
        pair_gap = pair_gap[valid]
        if len(target_frame) == 0:
            continue

        fraction = (target_time - source_time[previous]) / pair_gap

        def causal_projection(column: str) -> np.ndarray:
            source_values = ordered[column].to_numpy(float)
            return source_values[previous] + fraction * (
                source_values[latest] - source_values[previous]
            )

        if "height_m" in ordered:
            height = causal_projection("height_m")
        else:
            height = np.full(len(target_time), np.nan, dtype=float)
        blocks.append(
            pd.DataFrame(
                {
                    "pedestrian_id": [pedestrian_id] * len(target_frame),
                    "frame": target_frame,
                    "time_s": target_time,
                    "x_m": causal_projection("x_m"),
                    "y_m": causal_projection("y_m"),
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
    return resample_trajectory_to_fps(
        source,
        target_fps=target_fps,
        source_fps=source_fps,
    )


def load_resampled_julich_archive_trajectory(
    archive_path: str | Path,
    member: str,
    *,
    source_fps: float = 16.0,
    target_fps: float = 10.0,
) -> pd.DataFrame:
    """Load and exactly resample one PeTrack member without extracting its ZIP."""

    source = load_julich_archive_trajectory(
        archive_path,
        member,
        fps=source_fps,
    )
    return resample_trajectory_to_fps(
        source,
        target_fps=target_fps,
        source_fps=source_fps,
    )


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
    frame_step = groups["frame"].diff()
    dt = groups["time_s"].diff().where(frame_step == 1)
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
