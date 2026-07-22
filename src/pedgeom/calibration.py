"""Empirical one-step calibration for geometric pedestrian velocity fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from pedgeom.datasets import add_motion_features, load_julich_trajectory


FEATURE_NAMES = ("persistence", "goal", "isotropic", "forward", "closing", "wall")


@dataclass(frozen=True)
class VelocitySamples:
    """Vector features and future velocity targets sampled from one experimental run."""

    run: str
    features: np.ndarray  # observations x features x spatial dimensions
    target: np.ndarray  # observations x spatial dimensions
    pedestrian_id: np.ndarray
    frame: np.ndarray

    def select_features(self, names: tuple[str, ...]) -> np.ndarray:
        indices = [FEATURE_NAMES.index(name) for name in names]
        return self.features[:, indices, :]


@dataclass(frozen=True)
class FittedVelocityModel:
    names: tuple[str, ...]
    coefficients: np.ndarray

    def predict(self, samples: VelocitySamples, speed_cap: float = 2.5) -> np.ndarray:
        selected = samples.select_features(self.names)
        prediction = np.einsum("nkc,k->nc", selected, self.coefficients)
        speed = np.linalg.norm(prediction, axis=1)
        scale = np.minimum(1.0, speed_cap / np.maximum(speed, 1e-12))
        return prediction * scale[:, None]


def _interaction_features(
    positions: np.ndarray,
    velocities: np.ndarray,
    directions: np.ndarray,
    interaction_range: float,
    radius: float,
    wall_range: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return isotropic, forward-weighted, closing, and wall repulsion fields."""

    delta = positions[None, :, :] - positions[:, None, :]  # i -> j
    distance = np.linalg.norm(delta, axis=2)
    valid = (distance > 1e-9) & (distance < 3.0)
    unit_to_neighbor = np.divide(
        delta,
        distance[:, :, None],
        out=np.zeros_like(delta),
        where=valid[:, :, None],
    )
    magnitude = np.exp(-(distance - 2.0 * radius) / interaction_range) * valid
    away = -unit_to_neighbor
    isotropic = np.sum(magnitude[:, :, None] * away, axis=1)

    cos_ahead = unit_to_neighbor[:, :, 0] * directions[:, None]
    forward_weight = np.clip(cos_ahead, 0.0, 1.0) ** 2
    forward = np.sum((magnitude * forward_weight)[:, :, None] * away, axis=1)

    relative_velocity = velocities[None, :, :] - velocities[:, None, :]
    distance_rate = np.sum(relative_velocity * unit_to_neighbor, axis=2)
    closing_speed = np.clip(-distance_rate, 0.0, 3.0)
    closing = np.sum((magnitude * closing_speed)[:, :, None] * away, axis=1)

    y = positions[:, 1]
    wall = np.zeros_like(positions)
    wall[:, 1] = np.exp(-y / wall_range) - np.exp(-(4.0 - y) / wall_range)
    return isotropic, forward, closing, wall


def build_velocity_samples(
    source: str | Path,
    interaction_range: float = 0.35,
    horizon_frames: int = 10,
    frame_stride: int = 10,
    maximum_samples: int = 6000,
    radius: float = 0.25,
    wall_range: float = 0.25,
    seed: int = 20260722,
) -> VelocitySamples:
    """Construct observed-state to future-velocity samples without trajectory leakage."""

    source = Path(source)
    data = add_motion_features(load_julich_trajectory(source))
    current = data[["pedestrian_id", "frame", "x_m", "y_m", "direction"]].copy()
    previous = current[["pedestrian_id", "frame", "x_m", "y_m"]].copy()
    previous["frame"] += horizon_frames
    previous = previous.rename(columns={"x_m": "previous_x", "y_m": "previous_y"})
    future = current[["pedestrian_id", "frame", "x_m", "y_m"]].copy()
    future["frame"] -= horizon_frames
    future = future.rename(columns={"x_m": "future_x", "y_m": "future_y"})
    eligible = current.merge(previous, on=["pedestrian_id", "frame"], how="inner").merge(
        future, on=["pedestrian_id", "frame"], how="inner"
    )
    dt = horizon_frames / 25.0
    eligible["previous_vx"] = (eligible["x_m"] - eligible["previous_x"]) / dt
    eligible["previous_vy"] = (eligible["y_m"] - eligible["previous_y"]) / dt
    eligible["target_vx"] = (eligible["future_x"] - eligible["x_m"]) / dt
    eligible["target_vy"] = (eligible["future_y"] - eligible["y_m"]) / dt

    first_frame = int(eligible["frame"].min())
    eligible = eligible[(eligible["frame"] - first_frame) % frame_stride == 0]
    previous_speed = np.hypot(eligible["previous_vx"], eligible["previous_vy"])
    target_speed = np.hypot(eligible["target_vx"], eligible["target_vy"])
    eligible = eligible[(previous_speed <= 3.0) & (target_speed <= 3.0)]

    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    id_blocks: list[np.ndarray] = []
    frame_blocks: list[np.ndarray] = []
    all_by_frame = {frame: group for frame, group in current.groupby("frame", sort=False)}

    for frame, group in eligible.groupby("frame", sort=True):
        observed = all_by_frame[frame]
        index_by_id = pd.Series(np.arange(len(observed)), index=observed["pedestrian_id"])
        row_indices = index_by_id.loc[group["pedestrian_id"]].to_numpy()
        positions = observed[["x_m", "y_m"]].to_numpy(float)

        observed_motion = data[data["frame"].eq(frame)].set_index("pedestrian_id")
        velocities = observed_motion.loc[observed["pedestrian_id"], ["vx_mps", "vy_mps"]].to_numpy(float)
        velocities = np.nan_to_num(velocities, nan=0.0, posinf=0.0, neginf=0.0)
        directions = observed["direction"].to_numpy(float)
        isotropic, forward, closing, wall = _interaction_features(
            positions, velocities, directions, interaction_range, radius, wall_range
        )

        count = len(group)
        features = np.zeros((count, len(FEATURE_NAMES), 2), dtype=float)
        features[:, 0, :] = group[["previous_vx", "previous_vy"]].to_numpy(float)
        features[:, 1, 0] = group["direction"].to_numpy(float)
        features[:, 2, :] = isotropic[row_indices]
        features[:, 3, :] = forward[row_indices]
        features[:, 4, :] = closing[row_indices]
        features[:, 5, :] = wall[row_indices]
        feature_blocks.append(features)
        target_blocks.append(group[["target_vx", "target_vy"]].to_numpy(float))
        id_blocks.append(group["pedestrian_id"].to_numpy())
        frame_blocks.append(np.full(count, frame, dtype=int))

    features = np.concatenate(feature_blocks)
    targets = np.concatenate(target_blocks)
    pedestrian_ids = np.concatenate(id_blocks)
    frames = np.concatenate(frame_blocks)
    if len(targets) > maximum_samples:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(targets), maximum_samples, replace=False))
        features = features[selected]
        targets = targets[selected]
        pedestrian_ids = pedestrian_ids[selected]
        frames = frames[selected]
    return VelocitySamples(source.stem, features, targets, pedestrian_ids, frames)


def fit_velocity_model(
    batches: list[VelocitySamples], names: tuple[str, ...]
) -> FittedVelocityModel:
    """Fit nonnegative, run-balanced coefficients by bounded least squares."""

    design_parts = []
    target_parts = []
    for batch in batches:
        design = batch.select_features(names).transpose(0, 2, 1).reshape(-1, len(names))
        target = batch.target.reshape(-1)
        weight = 1.0 / np.sqrt(len(target))
        design_parts.append(design * weight)
        target_parts.append(target * weight)
    result = lsq_linear(
        np.concatenate(design_parts),
        np.concatenate(target_parts),
        bounds=(np.zeros(len(names)), np.full(len(names), 5.0)),
        lsmr_tol="auto",
    )
    return FittedVelocityModel(names, result.x)


def velocity_metrics(target: np.ndarray, prediction: np.ndarray, horizon_s: float = 0.4) -> dict[str, float]:
    """Compute interpretable vector, speed, direction, and displacement errors."""

    error = prediction - target
    vector_error = np.linalg.norm(error, axis=1)
    target_speed = np.linalg.norm(target, axis=1)
    prediction_speed = np.linalg.norm(prediction, axis=1)
    moving = (target_speed > 0.2) & (prediction_speed > 0.05)
    cosine = np.sum(target[moving] * prediction[moving], axis=1) / (
        target_speed[moving] * prediction_speed[moving]
    )
    angular = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    baseline_ss = np.sum((target - target.mean(axis=0)) ** 2)
    return {
        "vector_rmse_mps": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        "displacement_mae_m": float(horizon_s * np.mean(vector_error)),
        "speed_mae_mps": float(np.mean(np.abs(prediction_speed - target_speed))),
        "median_angle_deg": float(np.median(angular)) if len(angular) else np.nan,
        "vector_r2": float(1.0 - np.sum(error**2) / baseline_ss),
        "samples": float(len(target)),
    }
