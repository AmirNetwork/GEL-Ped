# Author: Amir Ghorbani
import numpy as np

from pedgeom.benchmarks import (
    AVMRegressor,
    DisagreementRoutedRegressor,
    GoalStableRegressor,
    InvariantRegressor,
    fit_linear_skip,
    fit_full_tensor_geometry_model,
    fit_residual_on_base_mlp,
    fit_tensor_residual_mlp,
    invariant_design,
    local_target,
    raw_neighbour_design,
    TTCResponseRegressor,
    calibrate_disagreement_router,
    tensor_scalar_design,
)
from pedgeom.calibration import FEATURE_NAMES, VelocitySamples, fit_tensor_geometry_model


class _FixedLocalEstimator:
    def predict(self, design: np.ndarray) -> np.ndarray:
        return np.tile(np.array([-0.4, 0.2]), (len(design), 1))


class _SecondFixedLocalEstimator:
    def predict(self, design: np.ndarray) -> np.ndarray:
        return np.tile(np.array([0.6, -0.1]), (len(design), 1))


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


def test_generic_residual_and_linear_skip_are_rotation_equivariant() -> None:
    rng = np.random.default_rng(29)
    features = rng.normal(size=(160, len(FEATURE_NAMES), 2))
    features[:, 1, :] = [1.0, 0.0]
    target = 0.65 * features[:, 0, :] + 0.2 * np.tanh(features[:, 4, :])
    samples = VelocitySamples("first", features, target, np.arange(160), np.arange(160))
    base = InvariantRegressor(_FixedLocalEstimator())
    residual = fit_residual_on_base_mlp(
        [samples], base, hidden_layers=(8,), max_iter=8, seed=29
    )
    direct = fit_residual_on_base_mlp(
        [samples], base, hidden_layers=(8,), max_iter=8, seed=31
    )
    additive = fit_linear_skip([samples], direct, penalty=0.1)

    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated = VelocitySamples(
        "rotated",
        features @ rotation.T,
        target @ rotation.T,
        np.arange(160),
        np.arange(160),
    )
    assert np.allclose(residual.predict(rotated), residual.predict(samples) @ rotation.T)
    assert np.allclose(additive.predict(rotated), additive.predict(samples) @ rotation.T)


def test_raw_neighbour_design_is_rotation_invariant() -> None:
    features = np.zeros((2, len(FEATURE_NAMES), 2))
    features[:, 0] = [[0.8, 0.1], [0.5, -0.1]]
    features[:, 1] = [1.0, 0.0]
    raw = np.zeros((2, 8, 5))
    raw[:, 0, :] = [0.6, 0.2, -0.1, 0.3, 1.0]
    samples = VelocitySamples(
        "base", features, np.zeros((2, 2)), np.arange(2), np.arange(2),
        raw_neighbours=raw,
    )
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated_raw = raw.copy()
    rotated_raw[:, :, :2] = raw[:, :, :2] @ rotation.T
    rotated_raw[:, :, 2:4] = raw[:, :, 2:4] @ rotation.T
    rotated = VelocitySamples(
        "rotated",
        features @ rotation.T,
        np.zeros((2, 2)),
        np.arange(2),
        np.arange(2),
        raw_neighbours=rotated_raw,
    )
    assert np.allclose(raw_neighbour_design(samples), raw_neighbour_design(rotated))


def test_mechanism_baselines_are_rotation_equivariant() -> None:
    features = np.zeros((2, len(FEATURE_NAMES), 2))
    features[:, 0] = [[0.8, 0.1], [0.5, -0.1]]
    features[:, 1] = [1.0, 0.0]
    raw = np.zeros((2, 8, 5))
    raw[:, 0, :] = [0.6, 0.2, -0.4, -0.1, 1.0]
    samples = VelocitySamples(
        "base", features, np.zeros((2, 2)), np.arange(2), np.arange(2),
        raw_neighbours=raw,
    )
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated_raw = raw.copy()
    rotated_raw[:, :, :2] = raw[:, :, :2] @ rotation.T
    rotated_raw[:, :, 2:4] = raw[:, :, 2:4] @ rotation.T
    rotated = VelocitySamples(
        "rotated", features @ rotation.T, np.zeros((2, 2)), np.arange(2), np.arange(2),
        raw_neighbours=rotated_raw,
    )
    models = (
        TTCResponseRegressor(np.array([0.8, 0.2, -0.4, 0.1])),
        AVMRegressor(1.3, 0.5, 2.0, 0.3, 0.8, 0.3),
    )
    for model in models:
        assert np.allclose(model.predict(rotated), model.predict(samples) @ rotation.T)


def test_full_tensor_geometry_is_rotation_equivariant() -> None:
    rng = np.random.default_rng(37)
    features = rng.normal(size=(120, len(FEATURE_NAMES), 2))
    features[:, 1] = [1.0, 0.0]
    target = 0.7 * features[:, 0] + 0.1 * features[:, 3]
    samples = VelocitySamples("base", features, target, np.arange(120), np.arange(120))
    model = fit_full_tensor_geometry_model([samples], penalty=1e-6)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated = VelocitySamples(
        "rotated", features @ rotation.T, target @ rotation.T, np.arange(120), np.arange(120)
    )
    assert np.allclose(model.predict(rotated), model.predict(samples) @ rotation.T)


def test_disagreement_router_is_calibrated_and_rotation_equivariant() -> None:
    features = np.zeros((12, len(FEATURE_NAMES), 2))
    features[:, 1] = [1.0, 0.0]
    samples = VelocitySamples(
        "base", features, np.zeros((12, 2)), np.arange(12), np.arange(12)
    )
    first = InvariantRegressor(_FixedLocalEstimator())
    second = InvariantRegressor(_SecondFixedLocalEstimator())
    router = calibrate_disagreement_router([samples] * 4, first, second)
    assert router.in_support_weight(samples) == 1.0

    forced_blend = DisagreementRoutedRegressor(first, second, threshold=0.0, scale=1.0)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated = VelocitySamples(
        "rotated",
        features @ rotation.T,
        np.zeros((12, 2)),
        np.arange(12),
        np.arange(12),
    )
    assert np.allclose(
        forced_blend.predict(rotated), forced_blend.predict(samples) @ rotation.T
    )
