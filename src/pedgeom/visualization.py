# Author: Amir Ghorbani
"""Metric-field and spacetime-style visualisation helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .models import LegacyParameters
from .potential import total_potential_gradient

FloatArray = NDArray[np.float64]


def potential_grid(
    agent_index: int,
    positions: FloatArray,
    destinations: FloatArray,
    obstacles: FloatArray | None,
    parameters: LegacyParameters,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    resolution: tuple[int, int] = (100, 50),
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate an agent-specific potential over a Cartesian grid."""

    x = np.linspace(*x_limits, resolution[0])
    y = np.linspace(*y_limits, resolution[1])
    grid_x, grid_y = np.meshgrid(x, y)
    values = np.empty_like(grid_x)
    evaluation_positions = positions.copy()
    for row in range(len(y)):
        for column in range(len(x)):
            evaluation_positions[agent_index] = [grid_x[row, column], grid_y[row, column]]
            values[row, column] = total_potential_gradient(
                agent_index,
                evaluation_positions,
                destinations,
                obstacles,
                parameters,
            )[0]
    return grid_x, grid_y, values


def normalized_geometric_elevation(potential: FloatArray) -> FloatArray:
    """Return a robust dimensionless elevation used only for visualisation."""

    lower, upper = np.quantile(potential, [0.02, 0.98])
    scale = max(float(upper - lower), 1e-12)
    return np.clip((potential - lower) / scale, 0.0, 1.0)


def weak_field_time_factor_deviation(
    potential: FloatArray, reference_speed: float = 299_792_458.0
) -> FloatArray:
    """Return the stable weak-field value `f-1`, approximately U/c_ref^2.

    Returning the deviation avoids floating-point cancellation when the physical
    reference speed makes `f` numerically indistinguishable from one.
    """

    return potential / reference_speed**2
