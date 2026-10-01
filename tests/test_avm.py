# Author: Amir Ghorbani
import numpy as np

from pedgeom.avm import avm_speed, desired_avm_direction
from pedgeom.models import AVMParameters


def test_isolated_avm_agent_keeps_direction_and_free_speed():
    positions = np.array([[0.0, 0.0]])
    velocities = np.array([[1.0, 0.0]])
    directions = np.array([[1.0, 0.0]])
    parameters = AVMParameters()
    desired = desired_avm_direction(
        0, positions, velocities, directions, directions[0], parameters
    )
    speed = avm_speed(0, positions, desired, parameters)
    np.testing.assert_allclose(desired, [1.0, 0.0])
    assert speed == parameters.desired_speed


def test_avm_headway_reduces_speed():
    positions = np.array([[0.0, 0.0], [0.5, 0.0]])
    parameters = AVMParameters(radius=0.2, time_gap=0.5)
    speed = avm_speed(0, positions, np.array([1.0, 0.0]), parameters)
    np.testing.assert_allclose(speed, 0.2)
