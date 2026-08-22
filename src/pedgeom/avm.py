# Author: Amir Ghorbani
"""Anticipation Velocity Model baseline following Xu, Chraibi, and Seyfried."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .models import AVMParameters

FloatArray = NDArray[np.float64]


def _unit(vector: FloatArray, fallback: FloatArray) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else fallback.copy()


def nearest_periodic_offsets(
    agent_index: int, positions: FloatArray, periodic_length: float | None
) -> FloatArray:
    offsets = positions - positions[agent_index]
    if periodic_length is not None:
        offsets[:, 0] = (offsets[:, 0] + periodic_length / 2.0) % periodic_length - (
            periodic_length / 2.0
        )
    return offsets


def desired_avm_direction(
    agent_index: int,
    positions: FloatArray,
    velocities: FloatArray,
    current_directions: FloatArray,
    preferred_direction: FloatArray,
    parameters: AVMParameters,
    periodic_length: float | None = None,
) -> FloatArray:
    """Calculate the AVM anticipatory direction for one pedestrian."""

    offsets = nearest_periodic_offsets(agent_index, positions, periodic_length)
    preferred = _unit(preferred_direction, np.array([1.0, 0.0]))
    perpendicular = np.array([-preferred[1], preferred[0]])
    avoidance = np.zeros(2)
    predicted_positions = positions + parameters.prediction_time * velocities
    predicted_offsets = predicted_positions - predicted_positions[agent_index]
    if periodic_length is not None:
        predicted_offsets[:, 0] = (
            predicted_offsets[:, 0] + periodic_length / 2.0
        ) % periodic_length - periodic_length / 2.0

    for other in range(len(positions)):
        if other == agent_index:
            continue
        distance = float(np.linalg.norm(offsets[other]))
        if distance <= 1e-12 or distance > parameters.interaction_cutoff:
            continue
        line_of_sight = offsets[other] / distance
        if float(np.dot(preferred, line_of_sight)) < -0.2:
            continue
        predicted_distance = max(
            float(np.linalg.norm(predicted_offsets[other])), 2.0 * parameters.radius
        )
        directional_weight = parameters.repulsion_strength * (
            1.0 + (1.0 - float(np.dot(preferred, current_directions[other]))) / 2.0
        )
        impact = directional_weight * np.exp(
            (2.0 * parameters.radius - predicted_distance) / parameters.interaction_range
        )
        lateral_position = float(np.dot(predicted_offsets[other], perpendicular))
        side = -1.0 if lateral_position >= 0.0 else 1.0
        avoidance += impact * side * perpendicular

    return _unit(preferred + avoidance, preferred)


def avm_speed(
    agent_index: int,
    positions: FloatArray,
    direction: FloatArray,
    parameters: AVMParameters,
    periodic_length: float | None = None,
) -> float:
    """Calculate the collision-free speed-headway branch of AVM."""

    offsets = nearest_periodic_offsets(agent_index, positions, periodic_length)
    perpendicular = np.array([-direction[1], direction[0]])
    minimum_distance = float("inf")
    for other in range(len(positions)):
        if other == agent_index:
            continue
        distance = float(np.linalg.norm(offsets[other]))
        if distance <= 1e-12:
            return 0.0
        unit_offset = offsets[other] / distance
        in_front = float(np.dot(direction, unit_offset)) >= 0.0
        crossing_path = abs(float(np.dot(perpendicular, unit_offset))) <= (
            2.0 * parameters.radius / distance
        )
        if in_front and crossing_path:
            minimum_distance = min(minimum_distance, distance)

    if not np.isfinite(minimum_distance):
        return parameters.desired_speed
    headway = max(minimum_distance - 2.0 * parameters.radius, 0.0)
    return min(parameters.desired_speed, headway / parameters.time_gap)
