# Author: Amir Ghorbani
"""Digitised simulation results reported in manuscript Table 2."""

import numpy as np

OBSERVATION_TIMES = np.arange(0.0, 4.01, 0.5)
OBSERVED_POSITIONS = np.array(
    [
        [[6.00, 2.50], [3.00, 2.50]],
        [[5.35, 2.45], [3.64, 2.45]],
        [[4.85, 2.39], [4.14, 2.41]],
        [[4.83, 2.32], [4.16, 2.40]],
        [[4.80, 2.16], [4.19, 2.51]],
        [[4.44, 1.87], [4.55, 2.79]],
        [[3.79, 1.92], [5.19, 2.78]],
        [[3.14, 2.02], [5.82, 2.66]],
        [[2.48, 2.08], [6.46, 2.54]],
    ]
)
OBSERVED_SPEEDS = np.array(
    [
        [1.30, 1.30],
        [1.30, 1.30],
        [0.51, 0.50],
        [0.20, 0.00],
        [0.51, 0.41],
        [1.26, 1.26],
        [1.33, 1.32],
        [1.30, 1.32],
        [1.40, 1.32],
    ]
)


def interpolate_history(times, values):
    output = np.empty((len(OBSERVATION_TIMES), *values.shape[1:]))
    for agent in range(values.shape[1]):
        for component in range(values.shape[2]):
            output[:, agent, component] = np.interp(
                OBSERVATION_TIMES, times, values[:, agent, component]
            )
    return output


def manuscript_errors(result):
    predicted_positions = interpolate_history(result.times, result.positions)
    speed_history = np.linalg.norm(result.velocities, axis=2)[..., None]
    predicted_speeds = interpolate_history(result.times, speed_history)[..., 0]
    position_rmse = float(np.sqrt(np.mean((predicted_positions - OBSERVED_POSITIONS) ** 2)))
    speed_rmse = float(np.sqrt(np.mean((predicted_speeds - OBSERVED_SPEEDS) ** 2)))
    return position_rmse, speed_rmse
