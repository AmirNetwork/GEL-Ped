# Author: Amir Ghorbani
"""Empirical one-step calibration for geometric pedestrian velocity fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, lsq_linear

from pedgeom.datasets import (
    GoalMethod,
    add_motion_features,
    add_prediction_goal_directions,
    load_julich_trajectory,
)


FEATURE_NAMES = ("persistence", "goal", "isotropic", "forward", "closing", "wall")


@dataclass(frozen=True)
class VelocitySamples:
    """Vector features and future velocity targets sampled from one experimental run."""

    run: str
    features: np.ndarray  # observations x features x spatial dimensions
    target: np.ndarray  # observations x spatial dimensions
    pedestrian_id: np.ndarray
    frame: np.ndarray
    position: np.ndarray | None = None
    neighbour_count: np.ndarray | None = None
    occupancy: np.ndarray | None = None
    goal_method: str = "entry"

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


@dataclass(frozen=True)
class DirectionSpeedModel:
    """Geometric direction forecast with separately calibrated speed persistence."""

    field_model: FittedVelocityModel
    speed_coefficients: np.ndarray

    def predict(self, samples: VelocitySamples, speed_cap: float = 2.5) -> np.ndarray:
        field = self.field_model.predict(samples, speed_cap=100.0)
        field_speed = np.linalg.norm(field, axis=1)
        previous_speed = np.linalg.norm(samples.features[:, 0, :], axis=1)
        design = np.column_stack((field_speed, previous_speed, np.ones(len(field))))
        speed = np.clip(design @ self.speed_coefficients, 0.0, speed_cap)
        direction = np.divide(
            field,
            field_speed[:, None],
            out=np.zeros_like(field),
            where=field_speed[:, None] > 1e-12,
        )
        return speed[:, None] * direction


@dataclass(frozen=True)
class TensorGeometryModel:
    """Goal-frame tensor prior implementing manuscript Eqs. (6)-(7)."""

    parallel_indices: tuple[int, ...]
    lateral_indices: tuple[int, ...]
    parallel_coefficients: np.ndarray
    lateral_coefficients: np.ndarray

    def predict(self, samples: VelocitySamples, speed_cap: float = 2.5) -> np.ndarray:
        goal = samples.features[:, 1, :]
        lateral = np.column_stack((-goal[:, 1], goal[:, 0]))
        along_features = np.einsum("nfc,nc->nf", samples.features, goal)
        lateral_features = np.einsum("nfc,nc->nf", samples.features, lateral)
        along = along_features[:, self.parallel_indices] @ self.parallel_coefficients
        across = lateral_features[:, self.lateral_indices] @ self.lateral_coefficients
        prediction = along[:, None] * goal + across[:, None] * lateral
        speed = np.linalg.norm(prediction, axis=1)
        scale = np.minimum(1.0, speed_cap / np.maximum(speed, 1e-12))
        return prediction * scale[:, None]


@dataclass(frozen=True)
class SocialForceResponseModel:
    """Observed-state Social Force reference in manuscript Eq. (12)."""

    relaxation_time: float
    desired_speed: float
    pedestrian_acceleration: float
    wall_acceleration: float
    horizon_s: float = 0.4

    def predict(self, samples: VelocitySamples, speed_cap: float = 2.5) -> np.ndarray:
        previous = samples.features[:, 0, :]
        goal = samples.features[:, 1, :]
        pedestrian_field = samples.features[:, 2, :]
        wall_field = samples.features[:, 5, :]
        acceleration = (
            (self.desired_speed * goal - previous) / self.relaxation_time
            + self.pedestrian_acceleration * pedestrian_field
            + self.wall_acceleration * wall_field
        )
        prediction = previous + self.horizon_s * acceleration
        speed = np.linalg.norm(prediction, axis=1)
        scale = np.minimum(1.0, speed_cap / np.maximum(speed, 1e-12))
        return prediction * scale[:, None]


def _interaction_features(
    positions: np.ndarray,
    velocities: np.ndarray,
    goal_directions: np.ndarray,
    interaction_range: float,
    radius: float,
    wall_range: float,
    wall_y_bounds: tuple[float, float] | None,
    neighbour_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return isotropic, forward-weighted, closing, and wall repulsion fields."""

    delta = positions[None, :, :] - positions[:, None, :]  # i -> j
    distance = np.linalg.norm(delta, axis=2)
    valid = (distance > 1e-9) & (distance < neighbour_cutoff)
    unit_to_neighbor = np.divide(
        delta,
        distance[:, :, None],
        out=np.zeros_like(delta),
        where=valid[:, :, None],
    )
    magnitude = np.exp(-(distance - 2.0 * radius) / interaction_range) * valid
    away = -unit_to_neighbor
    isotropic = np.sum(magnitude[:, :, None] * away, axis=1)

    cos_ahead = np.sum(unit_to_neighbor * goal_directions[:, None, :], axis=2)
    forward_weight = np.clip(cos_ahead, 0.0, 1.0) ** 2
    forward = np.sum((magnitude * forward_weight)[:, :, None] * away, axis=1)

    relative_velocity = velocities[None, :, :] - velocities[:, None, :]
    distance_rate = np.sum(relative_velocity * unit_to_neighbor, axis=2)
    closing_speed = np.clip(-distance_rate, 0.0, 3.0)
    closing = np.sum((magnitude * closing_speed)[:, :, None] * away, axis=1)

    wall = np.zeros_like(positions)
    if wall_y_bounds is not None:
        lower, upper = wall_y_bounds
        y = positions[:, 1]
        wall[:, 1] = np.exp(-(y - lower) / wall_range) - np.exp(-(upper - y) / wall_range)
    return isotropic, forward, closing, wall, valid.sum(axis=1)


def build_velocity_samples(
    source: str | Path,
    interaction_range: float = 0.35,
    horizon_frames: int = 10,
    frame_stride: int = 10,
    maximum_samples: int = 6000,
    radius: float = 0.25,
    wall_range: float = 0.25,
    wall_y_bounds: tuple[float, float] | None = (0.0, 4.0),
    neighbour_cutoff: float = 3.0,
    goal_method: GoalMethod = "entry",
    cardinal_routes: bool | None = None,
    speed_outlier_threshold: float = 3.0,
    loader=load_julich_trajectory,
    seed: int = 20260722,
) -> VelocitySamples:
    """Construct manuscript Eqs. (1)-(5) without trajectory or endpoint leakage."""

    source = Path(source)
    data = add_motion_features(loader(source))
    data = add_prediction_goal_directions(
        data,
        method=goal_method,
        history_frames=horizon_frames,
        cardinal_routes=cardinal_routes,
    )
    current_all = data[
        ["pedestrian_id", "frame", "x_m", "y_m", "goal_x", "goal_y"]
    ].copy()
    current = current_all[data["goal_valid"].to_numpy()].copy()
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
    eligible = eligible[
        (previous_speed <= speed_outlier_threshold)
        & (target_speed <= speed_outlier_threshold)
    ]

    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    id_blocks: list[np.ndarray] = []
    frame_blocks: list[np.ndarray] = []
    position_blocks: list[np.ndarray] = []
    neighbour_blocks: list[np.ndarray] = []
    occupancy_blocks: list[np.ndarray] = []
    all_by_frame = {frame: group for frame, group in current_all.groupby("frame", sort=False)}

    for frame, group in eligible.groupby("frame", sort=True):
        observed = all_by_frame[frame]
        index_by_id = pd.Series(np.arange(len(observed)), index=observed["pedestrian_id"])
        row_indices = index_by_id.loc[group["pedestrian_id"]].to_numpy()
        positions = observed[["x_m", "y_m"]].to_numpy(float)

        observed_motion = data[data["frame"].eq(frame)].set_index("pedestrian_id")
        velocities = observed_motion.loc[observed["pedestrian_id"], ["vx_mps", "vy_mps"]].to_numpy(float)
        velocities = np.nan_to_num(velocities, nan=0.0, posinf=0.0, neginf=0.0)
        goal_directions = observed[["goal_x", "goal_y"]].to_numpy(float)
        isotropic, forward, closing, wall, neighbour_count = _interaction_features(
            positions,
            velocities,
            goal_directions,
            interaction_range,
            radius,
            wall_range,
            wall_y_bounds,
            neighbour_cutoff,
        )

        count = len(group)
        features = np.zeros((count, len(FEATURE_NAMES), 2), dtype=float)
        features[:, 0, :] = group[["previous_vx", "previous_vy"]].to_numpy(float)
        features[:, 1, :] = group[["goal_x", "goal_y"]].to_numpy(float)
        features[:, 2, :] = isotropic[row_indices]
        features[:, 3, :] = forward[row_indices]
        features[:, 4, :] = closing[row_indices]
        features[:, 5, :] = wall[row_indices]
        feature_blocks.append(features)
        target_blocks.append(group[["target_vx", "target_vy"]].to_numpy(float))
        id_blocks.append(group["pedestrian_id"].to_numpy())
        frame_blocks.append(np.full(count, frame, dtype=int))
        position_blocks.append(group[["x_m", "y_m"]].to_numpy(float))
        neighbour_blocks.append(neighbour_count[row_indices].astype(float))
        occupancy_blocks.append(np.full(count, len(observed), dtype=float))

    features = np.concatenate(feature_blocks)
    targets = np.concatenate(target_blocks)
    pedestrian_ids = np.concatenate(id_blocks)
    frames = np.concatenate(frame_blocks)
    sample_positions = np.concatenate(position_blocks)
    neighbour_counts = np.concatenate(neighbour_blocks)
    occupancies = np.concatenate(occupancy_blocks)
    if len(targets) > maximum_samples:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(targets), maximum_samples, replace=False))
        features = features[selected]
        targets = targets[selected]
        pedestrian_ids = pedestrian_ids[selected]
        frames = frames[selected]
        sample_positions = sample_positions[selected]
        neighbour_counts = neighbour_counts[selected]
        occupancies = occupancies[selected]
    return VelocitySamples(
        source.stem,
        features,
        targets,
        pedestrian_ids,
        frames,
        sample_positions,
        neighbour_counts,
        occupancies,
        goal_method,
    )


def fit_velocity_model(
    batches: list[VelocitySamples],
    names: tuple[str, ...],
    coefficient_bounds: tuple[float, float] = (0.0, 5.0),
) -> FittedVelocityModel:
    """Fit run-balanced coefficients by bounded least squares."""

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
        bounds=(
            np.full(len(names), coefficient_bounds[0]),
            np.full(len(names), coefficient_bounds[1]),
        ),
        lsmr_tol="auto",
    )
    return FittedVelocityModel(names, result.x)


def fit_social_force_response_model(
    batches: list[VelocitySamples], horizon_s: float = 0.4
) -> SocialForceResponseModel:
    """Calibrate a four-parameter Social Force response on complete runs.

    The pedestrian and wall fields carry the standard exponential-distance form.
    The objective gives every experimental run equal total weight.
    """

    def unpack(parameters: np.ndarray) -> SocialForceResponseModel:
        return SocialForceResponseModel(*parameters, horizon_s=horizon_s)

    def residual(parameters: np.ndarray) -> np.ndarray:
        model = unpack(parameters)
        pieces = []
        for batch in batches:
            error = model.predict(batch, speed_cap=100.0) - batch.target
            pieces.append(error.reshape(-1) / np.sqrt(2.0 * len(batch.target)))
        return np.concatenate(pieces)

    result = least_squares(
        residual,
        x0=np.array([0.5, 1.3, 0.5, 0.5]),
        bounds=(
            np.array([0.05, 0.2, 0.0, 0.0]),
            np.array([10.0, 3.0, 5.0, 5.0]),
        ),
        x_scale="jac",
    )
    return unpack(result.x)


def fit_direction_speed_model(
    batches: list[VelocitySamples],
    names: tuple[str, ...] = ("persistence", "goal", "isotropic", "forward", "wall"),
) -> DirectionSpeedModel:
    """Fit the interaction direction field and a run-balanced scalar speed model."""

    field_model = fit_velocity_model(batches, names)
    design_parts = []
    target_parts = []
    for batch in batches:
        field = field_model.predict(batch, speed_cap=100.0)
        design = np.column_stack(
            (
                np.linalg.norm(field, axis=1),
                np.linalg.norm(batch.features[:, 0, :], axis=1),
                np.ones(len(batch.target)),
            )
        )
        target = np.linalg.norm(batch.target, axis=1)
        weight = 1.0 / np.sqrt(len(target))
        design_parts.append(design * weight)
        target_parts.append(target * weight)
    result = lsq_linear(
        np.concatenate(design_parts),
        np.concatenate(target_parts),
        bounds=(np.zeros(3), np.array([2.0, 2.0, 2.0])),
        lsmr_tol="auto",
    )
    return DirectionSpeedModel(field_model, result.x)


def fit_tensor_geometry_model(
    batches: list[VelocitySamples],
    parallel_names: tuple[str, ...] = (
        "persistence",
        "goal",
        "isotropic",
        "forward",
        "closing",
    ),
    lateral_names: tuple[str, ...] = (
        "persistence",
        "isotropic",
        "forward",
        "closing",
        "wall",
    ),
    coefficient_bounds: tuple[float, float] = (-5.0, 5.0),
) -> TensorGeometryModel:
    """Fit a sparse diagonal mobility tensor in each pedestrian's goal frame."""

    parallel_indices = tuple(FEATURE_NAMES.index(name) for name in parallel_names)
    lateral_indices = tuple(FEATURE_NAMES.index(name) for name in lateral_names)
    parallel_design = []
    lateral_design = []
    parallel_target = []
    lateral_target = []
    for batch in batches:
        goal = batch.features[:, 1, :]
        lateral = np.column_stack((-goal[:, 1], goal[:, 0]))
        along_features = np.einsum("nfc,nc->nf", batch.features, goal)
        across_features = np.einsum("nfc,nc->nf", batch.features, lateral)
        weight = 1.0 / np.sqrt(len(batch.target))
        parallel_design.append(along_features[:, parallel_indices] * weight)
        lateral_design.append(across_features[:, lateral_indices] * weight)
        parallel_target.append(np.sum(batch.target * goal, axis=1) * weight)
        lateral_target.append(np.sum(batch.target * lateral, axis=1) * weight)
    parallel_result = lsq_linear(
        np.concatenate(parallel_design),
        np.concatenate(parallel_target),
        bounds=(
            np.full(len(parallel_indices), coefficient_bounds[0]),
            np.full(len(parallel_indices), coefficient_bounds[1]),
        ),
    )
    lateral_result = lsq_linear(
        np.concatenate(lateral_design),
        np.concatenate(lateral_target),
        bounds=(
            np.full(len(lateral_indices), coefficient_bounds[0]),
            np.full(len(lateral_indices), coefficient_bounds[1]),
        ),
    )
    return TensorGeometryModel(
        parallel_indices,
        lateral_indices,
        parallel_result.x,
        lateral_result.x,
    )


def velocity_metrics(target: np.ndarray, prediction: np.ndarray, horizon_s: float = 0.4) -> dict[str, float]:
    """Compute manuscript Eq. (13) and the declared secondary metrics."""

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


def proximity_metrics(
    samples: VelocitySamples,
    prediction: np.ndarray,
    horizon_s: float = 0.4,
    contact_distance: float = 0.4,
    comparison_distance: float = 2.0,
) -> dict[str, float]:
    """Measure short-horizon false contacts among simultaneously sampled pedestrians."""

    if samples.position is None:
        raise ValueError("sample positions are required for proximity metrics")
    predicted_positions = samples.position + horizon_s * prediction
    observed_positions = samples.position + horizon_s * samples.target
    false_contacts = 0
    predicted_contacts = 0
    compared_pairs = 0
    for frame in np.unique(samples.frame):
        selected = samples.frame == frame
        if np.count_nonzero(selected) < 2:
            continue
        current = samples.position[selected]
        predicted = predicted_positions[selected]
        observed = observed_positions[selected]
        for first in range(len(current) - 1):
            current_distance = np.linalg.norm(current[first + 1 :] - current[first], axis=1)
            relevant = current_distance < comparison_distance
            if not np.any(relevant):
                continue
            predicted_distance = np.linalg.norm(
                predicted[first + 1 :][relevant] - predicted[first], axis=1
            )
            observed_distance = np.linalg.norm(
                observed[first + 1 :][relevant] - observed[first], axis=1
            )
            predicted_contact = predicted_distance < contact_distance
            predicted_contacts += int(np.count_nonzero(predicted_contact))
            false_contacts += int(
                np.count_nonzero(predicted_contact & (observed_distance >= contact_distance))
            )
            compared_pairs += int(np.count_nonzero(relevant))
    denominator = max(compared_pairs, 1)
    return {
        "predicted_contact_rate": predicted_contacts / denominator,
        "false_contact_rate": false_contacts / denominator,
        "compared_pairs": float(compared_pairs),
    }
