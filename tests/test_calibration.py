# Author: Amir Ghorbani
import numpy as np

from pedgeom.calibration import (
    FEATURE_NAMES,
    VelocitySamples,
    fit_direction_speed_model,
    fit_social_force_response_model,
    fit_tensor_geometry_model,
    fit_velocity_model,
    velocity_metrics,
)
from pedgeom.datasets import add_goal_directions


def test_bounded_fit_recovers_nonnegative_linear_field() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(200, len(FEATURE_NAMES), 2))
    target = 0.7 * features[:, 0, :] + 1.2 * features[:, 1, :]
    samples = VelocitySamples("synthetic", features, target, np.arange(200), np.arange(200))
    model = fit_velocity_model([samples], ("persistence", "goal"))
    assert np.allclose(model.coefficients, [0.7, 1.2], atol=1e-8)


def test_perfect_prediction_metrics() -> None:
    target = np.array([[1.0, 0.0], [0.5, 0.2]])
    metrics = velocity_metrics(target, target)
    assert metrics["vector_rmse_mps"] == 0.0
    assert metrics["displacement_mae_m"] == 0.0
    assert metrics["vector_r2"] == 1.0


def test_goal_direction_is_two_dimensional() -> None:
    import pandas as pd

    data = pd.DataFrame(
        {
            "pedestrian_id": [1, 1, 2, 2],
            "x_m": [0.0, 0.0, 2.0, 0.0],
            "y_m": [0.0, 2.0, 1.0, 1.0],
        }
    )
    result = add_goal_directions(data)
    assert np.allclose(result.loc[result["pedestrian_id"] == 1, ["goal_x", "goal_y"]], [0, 1])
    assert np.allclose(result.loc[result["pedestrian_id"] == 2, ["goal_x", "goal_y"]], [-1, 0])


def test_direction_speed_model_preserves_field_direction() -> None:
    features = np.zeros((30, len(FEATURE_NAMES), 2))
    features[:, 0, 0] = 0.5
    features[:, 1, 0] = 1.0
    target = np.column_stack((np.full(30, 0.8), np.zeros(30)))
    samples = VelocitySamples("synthetic", features, target, np.arange(30), np.arange(30))
    model = fit_direction_speed_model([samples], ("persistence", "goal"))
    prediction = model.predict(samples)
    assert np.allclose(prediction[:, 1], 0.0)
    assert np.allclose(np.linalg.norm(prediction, axis=1), 0.8)


def test_tensor_geometry_is_rotation_equivariant() -> None:
    rng = np.random.default_rng(12)
    features = rng.normal(size=(100, len(FEATURE_NAMES), 2))
    features[:, 1, :] = [1.0, 0.0]
    target = 0.8 * features[:, 0, :] + 0.2 * features[:, 3, :]
    first = VelocitySamples("first", features, target, np.arange(100), np.arange(100))
    model = fit_tensor_geometry_model([first])
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated = VelocitySamples(
        "rotated",
        features @ rotation.T,
        target @ rotation.T,
        np.arange(100),
        np.arange(100),
    )
    assert np.allclose(model.predict(rotated), model.predict(first) @ rotation.T)


def test_social_force_fit_recovers_relaxation_response() -> None:
    features = np.zeros((120, len(FEATURE_NAMES), 2))
    features[:, 0, 0] = np.linspace(0.2, 1.2, 120)
    features[:, 1, 0] = 1.0
    target = features[:, 0, :] + 0.4 * (
        (1.4 * features[:, 1, :] - features[:, 0, :]) / 0.5
    )
    samples = VelocitySamples("synthetic", features, target, np.arange(120), np.arange(120))
    model = fit_social_force_response_model([samples])
    assert np.allclose(model.predict(samples, speed_cap=100.0), target, atol=1e-5)
