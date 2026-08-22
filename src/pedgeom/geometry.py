# Author: Amir Ghorbani
"""Anisotropic local metric construction for the revised pilot."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .models import AnisotropicParameters

FloatArray = NDArray[np.float64]


def _unit(vector: FloatArray, fallback: FloatArray | None = None) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-12:
        return vector / norm
    if fallback is None:
        return np.array([1.0, 0.0])
    return fallback.copy()


def _logistic(value: float) -> float:
    clipped = float(np.clip(value, -40.0, 40.0))
    return 1.0 / (1.0 + np.exp(-clipped))


def local_metric(
    agent_index: int,
    positions: FloatArray,
    velocities: FloatArray,
    destinations: FloatArray,
    parameters: AnisotropicParameters,
) -> tuple[FloatArray, float]:
    """Construct a symmetric positive-definite, interaction-dependent metric.

    The rank-one updates penalize travel through predicted encounter directions. A
    small lateral tilt breaks the head-on symmetry and represents a passing-side
    convention. This is a pilot field construction, not yet the final learned metric.
    """

    position = positions[agent_index]
    goal = _unit(destinations[agent_index] - position)
    preferred_right = np.array([goal[1], -goal[0]])
    metric = np.eye(2)
    maximum_risk = 0.0

    for other_index, other_position in enumerate(positions):
        if other_index == agent_index:
            continue
        separation = position - other_position
        distance = max(float(np.linalg.norm(separation)), 1e-9)
        relative_velocity = velocities[agent_index] - velocities[other_index]
        radial_rate = float(np.dot(separation, relative_velocity) / distance)
        approaching = _logistic(-radial_rate / parameters.closing_speed_scale)

        predicted = separation + parameters.anticipation_horizon * relative_velocity
        predicted_distance = float(np.linalg.norm(predicted))
        clearance = min(distance, predicted_distance) - 2.0 * parameters.radius
        proximity = float(np.exp(-max(clearance, 0.0) / parameters.interaction_range))
        risk = approaching * proximity

        encounter_axis = _unit(
            separation + parameters.lateral_tilt * distance * preferred_right,
            fallback=separation / distance,
        )
        metric += parameters.metric_strength * risk * np.outer(encounter_axis, encounter_axis)
        maximum_risk = max(maximum_risk, risk)

    return metric, maximum_risk


def geometric_direction(
    agent_index: int,
    positions: FloatArray,
    velocities: FloatArray,
    destinations: FloatArray,
    parameters: AnisotropicParameters,
) -> tuple[FloatArray, float, FloatArray]:
    """Return the local Riemannian steepest-descent direction and risk."""

    goal = _unit(destinations[agent_index] - positions[agent_index])
    metric, risk = local_metric(
        agent_index, positions, velocities, destinations, parameters
    )
    direction = _unit(np.linalg.solve(metric, goal), fallback=goal)
    return direction, risk, metric


def collision_free_speed(
    agent_index: int,
    positions: FloatArray,
    velocities: FloatArray,
    direction: FloatArray,
    desired_speed: float,
    radius: float,
    time_headway: float,
    safety_margin: float = 0.0,
    periodic_length: float | None = None,
) -> float:
    """Bound speed using forward free distance between circular pedestrians."""

    allowed_speed = desired_speed
    effective_diameter = 2.0 * radius + safety_margin
    for other_index, other_position in enumerate(positions):
        if other_index == agent_index:
            continue
        offset = other_position - positions[agent_index]
        if periodic_length is not None:
            offset[0] = (offset[0] + periodic_length / 2.0) % periodic_length - (
                periodic_length / 2.0
            )
        forward = float(np.dot(offset, direction))
        if forward <= 0.0:
            continue
        lateral_squared = max(float(np.dot(offset, offset)) - forward**2, 0.0)
        if lateral_squared >= effective_diameter**2:
            continue
        intersection = np.sqrt(effective_diameter**2 - lateral_squared)
        free_distance = max(forward - intersection, 0.0)
        candidate = free_distance / time_headway
        allowed_speed = min(allowed_speed, candidate)
    return float(allowed_speed)
