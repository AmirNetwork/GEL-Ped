# Author: Amir Ghorbani
"""Potential and analytic gradient used by the submitted model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .models import LegacyParameters

FloatArray = NDArray[np.float64]


def _soft_distance(vector: FloatArray, softening: float) -> float:
    return float(np.sqrt(np.dot(vector, vector) + softening**2))


def destination_potential_gradient(
    position: FloatArray,
    destination: FloatArray,
    strength: float,
    softening: float,
) -> tuple[float, FloatArray]:
    """Return U=A/r and its Cartesian gradient."""

    offset = position - destination
    distance = _soft_distance(offset, softening)
    potential = strength / distance
    gradient = -strength * offset / distance**3
    return potential, gradient


def linear_destination_potential_gradient(
    position: FloatArray,
    destination: FloatArray,
    acceleration: float,
    softening: float,
) -> tuple[float, FloatArray]:
    """Return U=a*r, which yields the constant acceleration used in Section 6.1."""

    offset = position - destination
    distance = _soft_distance(offset, softening)
    potential = acceleration * distance
    gradient = acceleration * offset / distance
    return potential, gradient


def exponential_repulsion_gradient(
    position: FloatArray,
    source: FloatArray,
    strength: float,
    interaction_range: float,
    clearance: float,
    softening: float,
) -> tuple[float, FloatArray]:
    """Return A exp(-(r-clearance)/range) and its Cartesian gradient."""

    offset = position - source
    distance = _soft_distance(offset, softening)
    potential = strength * np.exp(-(distance - clearance) / interaction_range)
    gradient = -(potential / interaction_range) * offset / distance
    return float(potential), gradient


def total_potential_gradient(
    agent_index: int,
    positions: FloatArray,
    destinations: FloatArray,
    obstacles: FloatArray | None,
    parameters: LegacyParameters,
) -> tuple[float, FloatArray]:
    """Evaluate Equation 23 in Cartesian coordinates for one pedestrian."""

    position = positions[agent_index]
    if parameters.destination_form == "reciprocal":
        potential, gradient = destination_potential_gradient(
            position,
            destinations[agent_index],
            parameters.destination_strength,
            parameters.softening,
        )
    elif parameters.destination_form == "linear":
        potential, gradient = linear_destination_potential_gradient(
            position,
            destinations[agent_index],
            parameters.destination_strength,
            parameters.softening,
        )
    else:  # pragma: no cover - protected by the parameter type
        raise ValueError(f"unknown destination form: {parameters.destination_form}")

    for other_index, other_position in enumerate(positions):
        if other_index == agent_index:
            continue
        value, derivative = exponential_repulsion_gradient(
            position,
            other_position,
            parameters.pedestrian_strength,
            parameters.pedestrian_range,
            2.0 * parameters.radius,
            parameters.softening,
        )
        potential += value
        gradient += derivative

    if obstacles is not None:
        for obstacle in obstacles:
            value, derivative = exponential_repulsion_gradient(
                position,
                obstacle,
                parameters.obstacle_strength,
                parameters.obstacle_range,
                parameters.radius,
                parameters.softening,
            )
            potential += value
            gradient += derivative

    return float(potential), gradient


def accelerations(
    positions: FloatArray,
    destinations: FloatArray,
    obstacles: FloatArray | None,
    parameters: LegacyParameters,
) -> FloatArray:
    """Newtonian acceleration a=-grad(U) implied by the corrected derivation."""

    result = np.empty_like(positions, dtype=float)
    for index in range(len(positions)):
        _, gradient = total_potential_gradient(
            index, positions, destinations, obstacles, parameters
        )
        result[index] = -gradient
    return result
