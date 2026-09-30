# Author: Amir Ghorbani
"""Naturalistic ETH/UCY data adapter for the protocol-matched 0.4-s task."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pedgeom.calibration import FEATURE_NAMES, VelocitySamples, _interaction_features


def load_eth_ucy(path: str | Path) -> pd.DataFrame:
    """Load the four-column ETH/UCY text representation used by Trajectron++."""

    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=("frame", "pedestrian_id", "x_m", "y_m"),
    )
    frame["frame"] = frame["frame"].astype(int)
    frame["pedestrian_id"] = frame["pedestrian_id"].astype(int)
    return frame.sort_values(["frame", "pedestrian_id"]).reset_index(drop=True)


def build_eth_ucy_samples(
    path: str | Path,
    *,
    step_frames: int = 10,
    history_steps: int = 2,
    maximum_samples: int = 6000,
    neighbour_cutoff: float = 3.0,
    maximum_neighbours: int = 8,
    speed_outlier_threshold: float = 3.0,
    seed: int = 20260722,
) -> VelocitySamples:
    """Reconstruct a 0.4-s task using only current and past observations.

    ETH/UCY positions are sampled at 2.5 Hz (ten raw frame units).  Recent
    velocity and route direction use ``history_steps`` observed intervals,
    while the target is always the next single interval.
    """

    path = Path(path)
    data = load_eth_ucy(path)
    history_frames = history_steps * step_frames
    current = data.copy()
    past = data.rename(columns={"x_m": "past_x", "y_m": "past_y"}).copy()
    past["frame"] += history_frames
    future = data.rename(columns={"x_m": "future_x", "y_m": "future_y"}).copy()
    future["frame"] -= step_frames
    eligible = current.merge(past, on=["pedestrian_id", "frame"]).merge(
        future, on=["pedestrian_id", "frame"]
    )
    history_dt = 0.4 * history_steps
    eligible["previous_vx"] = (eligible["x_m"] - eligible["past_x"]) / history_dt
    eligible["previous_vy"] = (eligible["y_m"] - eligible["past_y"]) / history_dt
    eligible["target_vx"] = (eligible["future_x"] - eligible["x_m"]) / 0.4
    eligible["target_vy"] = (eligible["future_y"] - eligible["y_m"]) / 0.4
    previous_speed = np.hypot(eligible["previous_vx"], eligible["previous_vy"])
    target_speed = np.hypot(eligible["target_vx"], eligible["target_vy"])
    eligible = eligible[
        (previous_speed <= speed_outlier_threshold)
        & (target_speed <= speed_outlier_threshold)
        & (previous_speed > 0.05)
    ].copy()
    speed = np.hypot(eligible["previous_vx"], eligible["previous_vy"])
    eligible["goal_x"] = eligible["previous_vx"] / speed
    eligible["goal_y"] = eligible["previous_vy"] / speed

    motion = data.merge(past, on=["pedestrian_id", "frame"], how="left")
    motion["vx"] = (motion["x_m"] - motion["past_x"]) / history_dt
    motion["vy"] = (motion["y_m"] - motion["past_y"]) / history_dt
    motion_by_frame = {
        frame: group.set_index("pedestrian_id")
        for frame, group in motion.groupby("frame", sort=False)
    }
    all_by_frame = {frame: group for frame, group in data.groupby("frame", sort=False)}

    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    id_blocks: list[np.ndarray] = []
    frame_blocks: list[np.ndarray] = []
    position_blocks: list[np.ndarray] = []
    neighbour_blocks: list[np.ndarray] = []
    occupancy_blocks: list[np.ndarray] = []
    raw_blocks: list[np.ndarray] = []
    for frame, group in eligible.groupby("frame", sort=True):
        observed = all_by_frame[frame]
        observed_motion = motion_by_frame[frame]
        valid_ids = observed_motion.index[
            observed_motion[["vx", "vy"]].notna().all(axis=1)
        ]
        observed = observed[observed["pedestrian_id"].isin(valid_ids)].copy()
        if observed.empty:
            continue
        group = group[group["pedestrian_id"].isin(observed["pedestrian_id"])].copy()
        if group.empty:
            continue
        index_by_id = pd.Series(np.arange(len(observed)), index=observed["pedestrian_id"])
        row_indices = index_by_id.loc[group["pedestrian_id"]].to_numpy()
        positions = observed[["x_m", "y_m"]].to_numpy(float)
        velocities = observed_motion.loc[
            observed["pedestrian_id"], ["vx", "vy"]
        ].to_numpy(float)
        velocity_norm = np.linalg.norm(velocities, axis=1, keepdims=True)
        goals = np.divide(
            velocities,
            velocity_norm,
            out=np.tile(np.array([[1.0, 0.0]]), (len(velocities), 1)),
            where=velocity_norm > 0.05,
        )
        isotropic, forward, closing, wall, neighbour_count = _interaction_features(
            positions,
            velocities,
            goals,
            interaction_range=0.35,
            radius=0.25,
            wall_range=0.25,
            wall_y_bounds=None,
            neighbour_cutoff=neighbour_cutoff,
        )
        count = len(group)
        features = np.zeros((count, len(FEATURE_NAMES), 2), dtype=float)
        features[:, 0] = group[["previous_vx", "previous_vy"]].to_numpy(float)
        features[:, 1] = group[["goal_x", "goal_y"]].to_numpy(float)
        features[:, 2] = isotropic[row_indices]
        features[:, 3] = forward[row_indices]
        features[:, 4] = closing[row_indices]
        features[:, 5] = wall[row_indices]
        feature_blocks.append(features)
        target_blocks.append(group[["target_vx", "target_vy"]].to_numpy(float))
        id_blocks.append(group["pedestrian_id"].to_numpy())
        frame_blocks.append(np.full(count, frame, dtype=int))
        position_blocks.append(group[["x_m", "y_m"]].to_numpy(float))
        neighbour_blocks.append(neighbour_count[row_indices].astype(float))
        occupancy_blocks.append(np.full(count, len(observed), dtype=float))

        delta = positions[None, :, :] - positions[:, None, :]
        relative_velocity = velocities[None, :, :] - velocities[:, None, :]
        distance = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(distance, np.inf)
        raw = np.zeros((count, maximum_neighbours, 5), dtype=float)
        for local_index, scene_index in enumerate(row_indices):
            ordered = np.argsort(distance[scene_index])
            ordered = ordered[distance[scene_index, ordered] < neighbour_cutoff]
            ordered = ordered[:maximum_neighbours]
            raw[local_index, : len(ordered), :2] = delta[scene_index, ordered]
            raw[local_index, : len(ordered), 2:4] = relative_velocity[
                scene_index, ordered
            ]
            raw[local_index, : len(ordered), 4] = 1.0
        raw_blocks.append(raw)

    arrays = (
        np.concatenate(feature_blocks),
        np.concatenate(target_blocks),
        np.concatenate(id_blocks),
        np.concatenate(frame_blocks),
        np.concatenate(position_blocks),
        np.concatenate(neighbour_blocks),
        np.concatenate(occupancy_blocks),
        np.concatenate(raw_blocks),
    )
    if len(arrays[1]) > maximum_samples:
        selected = np.sort(
            np.random.default_rng(seed).choice(
                len(arrays[1]), maximum_samples, replace=False
            )
        )
        arrays = tuple(array[selected] for array in arrays)
    return VelocitySamples(
        run=path.stem,
        features=arrays[0],
        target=arrays[1],
        pedestrian_id=arrays[2],
        frame=arrays[3],
        position=arrays[4],
        neighbour_count=arrays[5],
        occupancy=arrays[6],
        goal_method=f"past-{0.4 * history_steps:.1f}s",
        raw_neighbours=arrays[7],
    )

