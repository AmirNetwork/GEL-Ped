# Author: Amir Ghorbani
import numpy as np

from pedgeom.visualization import (
    normalized_geometric_elevation,
    weak_field_time_factor_deviation,
)


def test_geometric_elevation_is_bounded():
    potential = np.arange(100, dtype=float).reshape(10, 10)
    elevation = normalized_geometric_elevation(potential)
    assert elevation.min() >= 0.0
    assert elevation.max() <= 1.0


def test_time_factor_deviation_preserves_tiny_values():
    potential = np.array([1.0, 10.0])
    deviation = weak_field_time_factor_deviation(potential)
    assert np.all(deviation > 0.0)
    np.testing.assert_allclose(deviation[1] / deviation[0], 10.0)
