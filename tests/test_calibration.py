import numpy as np

from pedgeom.calibration import FEATURE_NAMES, VelocitySamples, fit_velocity_model, velocity_metrics


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
