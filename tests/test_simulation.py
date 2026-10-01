# Author: Amir Ghorbani
import numpy as np

from pedgeom import (
    simulate_anisotropic,
    simulate_corrected,
    simulate_geometric_gradient,
    simulate_legacy_reset,
    simulate_periodic_corridor,
)


def scene():
    positions = np.array([[6.0, 2.5], [3.0, 2.5]])
    velocities = np.array([[-1.3, 0.0], [1.3, 0.0]])
    destinations = np.array([[0.0, 2.0], [10.0, 2.0]])
    return positions, velocities, destinations


def test_output_shapes_and_finite_values():
    positions, velocities, destinations = scene()
    results = [
        simulate_legacy_reset(positions, destinations, 0.2, 0.1),
        simulate_corrected(positions, velocities, destinations, 0.2, 0.1),
        simulate_anisotropic(positions, velocities, destinations, 0.2, 0.02),
    ]
    for result in results:
        assert result.positions.shape == result.velocities.shape
        assert result.positions.shape[1:] == (2, 2)
        assert np.isfinite(result.positions).all()
        assert np.isfinite(result.velocities).all()


def test_reset_update_has_quadratic_timestep_dependence():
    positions, _, destinations = scene()
    coarse = simulate_legacy_reset(positions, destinations, 0.1, 0.1)
    fine = simulate_legacy_reset(positions, destinations, 0.1, 0.05)
    coarse_displacement = np.linalg.norm(coarse.positions[-1] - positions, axis=1).mean()
    fine_displacement = np.linalg.norm(fine.positions[-1] - positions, axis=1).mean()
    # Resetting velocity every substep makes total displacement shrink approximately
    # linearly with dt for fixed physical duration, demonstrating non-convergence.
    assert coarse_displacement > 1.7 * fine_displacement


def test_periodic_corridor_stays_inside_domain():
    positions = np.array([[-0.9, 0.2], [0.9, 1.8]])
    directions = np.array([-1.0, 1.0])
    result = simulate_periodic_corridor(
        positions,
        directions,
        duration=0.2,
        step=0.02,
        corridor_length=2.0,
        corridor_width=2.0,
    )
    assert np.all(result.positions[:, :, 0] >= -1.0)
    assert np.all(result.positions[:, :, 0] < 1.0)
    assert np.all(result.positions[:, :, 1] > 0.0)
    assert np.all(result.positions[:, :, 1] < 2.0)


def test_canonical_model_is_stable_under_timestep_halving():
    positions, _, destinations = scene()
    coarse = simulate_geometric_gradient(positions, destinations, 1.0, 0.02)
    fine = simulate_geometric_gradient(positions, destinations, 1.0, 0.01)
    np.testing.assert_allclose(coarse.positions[-1], fine.positions[-1], atol=0.02)
