# Author: Amir Ghorbani
import numpy as np

from pedgeom.models import LegacyParameters
from pedgeom.potential import (
    destination_potential_gradient,
    linear_destination_potential_gradient,
    total_potential_gradient,
)


def finite_difference(function, point, step=1e-6):
    result = np.empty(2)
    for axis in range(2):
        delta = np.zeros(2)
        delta[axis] = step
        result[axis] = (function(point + delta) - function(point - delta)) / (2.0 * step)
    return result


def test_destination_gradient_matches_finite_difference():
    destination = np.array([3.0, -0.5])
    position = np.array([1.2, 2.4])
    value, gradient = destination_potential_gradient(position, destination, 13.5, 0.05)
    numerical = finite_difference(
        lambda x: destination_potential_gradient(x, destination, 13.5, 0.05)[0],
        position,
    )
    assert value > 0.0
    np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-7)


def test_total_gradient_matches_finite_difference():
    parameters = LegacyParameters()
    positions = np.array([[1.0, 1.2], [2.1, 1.5]])
    destinations = np.array([[5.0, 1.0], [0.0, 1.0]])
    _, analytic = total_potential_gradient(0, positions, destinations, None, parameters)

    def value(point):
        changed = positions.copy()
        changed[0] = point
        return total_potential_gradient(0, changed, destinations, None, parameters)[0]

    numerical = finite_difference(value, positions[0])
    np.testing.assert_allclose(analytic, numerical, rtol=2e-5, atol=2e-6)


def test_printed_destination_is_repulsive_but_inferred_linear_form_is_attractive():
    position = np.array([2.0, 0.0])
    destination = np.array([0.0, 0.0])
    _, reciprocal_gradient = destination_potential_gradient(position, destination, 13.5, 0.05)
    _, linear_gradient = linear_destination_potential_gradient(position, destination, 13.5, 0.05)
    reciprocal_acceleration = -reciprocal_gradient
    linear_acceleration = -linear_gradient
    assert reciprocal_acceleration[0] > 0.0
    assert linear_acceleration[0] < 0.0
