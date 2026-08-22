# Author: Amir Ghorbani
import numpy as np

from pedgeom.benchmarks import (
    GoalStableRegressor,
    InvariantRegressor,
    fit_tensor_residual_mlp,
    invariant_design,
    local_target,
    tensor_scalar_design,
)
from pedgeom.calibration import FEATURE_NAMES, VelocitySamples, fit_tensor_geometry_model


class _FixedLocalEstimator:
    def predict(self, design: np.ndarray) -> np.ndarray:
        return np.tile(np.array([-0.4, 0.2]), (len(design), 1))


def test_goal_aligned_features_are_rotation_invariant() -> None:
    features = np.zeros((1, len(FEATURE_NAMES), 2))
    features[0, 0] = [0.8, 0.2]
    features[0, 1] = [1.0, 0.0]
    target = np.array([[0.7, 0.1]])
    first = VelocitySamples("x", features, target, np.array([1]), np.array([1]))

    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated = VelocitySamples(
        "y",
        features @ rotation.T,
        target @ rotation.T,
        np.array([1]),
        np.array([1]),
    )
    assert np.allclose(invariant_design(first), invariant_design(rotated))
    assert np.allclose(local_target(first), local_target(rotated))
    assert np.allclose(tensor_scalar_design(first), tensor_scalar_design(rotated))


def test_tensor_residual_model_is_rotation_equivariant() -> None:
    rng = np.random.default_rng(23)
    features = rng.normal(size=(180, len(FEATURE_NAMES), 2))
    features[:, 1, :] = [1.0, 0.0]
    target = 0.75 * features[:, 0, :] + 0.15 * np.tanh(features[:, 3, :])
    samples = VelocitySamples("first", features, target, np.arange(180), np.arange(180))
    tensor = fit_tensor_geometry_model([samples])
    model = fit_tensor_residual_mlp([samples], tensor, hidden_layers=(8,), max_iter=8)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated = VelocitySamples(
        "rotated",
        features @ rotation.T,
        target @ rotation.T,
        np.arange(180),
        np.arange(180),
    )
    assert np.allclose(model.predict(rotated), model.predict(samples) @ rotation.T)


def test_goal_stable_reference_enforces_progress_and_rotation_equivariance() -> None:
    features = np.zeros((2, len(FEATURE_NAMES), 2))
    features[:, 1, :] = [1.0, 0.0]
    samples = VelocitySamples(
        "base", features, np.zeros((2, 2)), np.arange(2), np.arange(2)
    )
    model = GoalStableRegressor(InvariantRegressor(_FixedLocalEstimator()))
    prediction = model.predict(samples)
    assert np.allclose(prediction, [[0.0, 0.2], [0.0, 0.2]])

    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated = VelocitySamples(
        "rotated",
        features @ rotation.T,
        np.zeros((2, 2)),
        np.arange(2),
        np.arange(2),
    )
    assert np.allclose(model.predict(rotated), prediction @ rotation.T)
