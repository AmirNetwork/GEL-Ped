# Author: Amir Ghorbani
import numpy as np

from pedgeom.geometry import collision_free_speed, geometric_direction, local_metric
from pedgeom.models import AnisotropicParameters


def test_metric_is_symmetric_positive_definite():
    positions = np.array([[0.0, 0.0], [1.0, 0.1], [-0.2, 0.8]])
    velocities = np.array([[1.0, 0.0], [-0.5, 0.0], [0.0, -0.2]])
    destinations = np.array([[5.0, 0.0], [-5.0, 0.0], [0.0, -5.0]])
    metric, risk = local_metric(
        0, positions, velocities, destinations, AnisotropicParameters()
    )
    np.testing.assert_allclose(metric, metric.T, atol=1e-12)
    assert np.linalg.eigvalsh(metric).min() >= 1.0 - 1e-12
    assert risk >= 0.0


def test_isolated_agent_moves_toward_goal():
    positions = np.array([[0.0, 0.0]])
    velocities = np.array([[0.0, 0.0]])
    destinations = np.array([[3.0, 4.0]])
    direction, risk, metric = geometric_direction(
        0, positions, velocities, destinations, AnisotropicParameters()
    )
    np.testing.assert_allclose(direction, np.array([0.6, 0.8]), atol=1e-12)
    np.testing.assert_allclose(metric, np.eye(2), atol=1e-12)
    assert risk == 0.0


def test_collision_free_speed_stops_before_close_oncoming_agent():
    positions = np.array([[0.0, 0.0], [0.5, 0.0]])
    velocities = np.array([[1.0, 0.0], [-1.0, 0.0]])
    speed = collision_free_speed(
        0,
        positions,
        velocities,
        np.array([1.0, 0.0]),
        desired_speed=1.35,
        radius=0.2,
        time_headway=0.5,
        safety_margin=0.05,
    )
    assert speed <= 0.1 + 1e-12
