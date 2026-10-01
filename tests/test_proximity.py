# Author: Amir Ghorbani
import numpy as np

from pedgeom.calibration import FEATURE_NAMES, VelocitySamples, proximity_metrics


def test_proximity_metric_detects_false_contact() -> None:
    features = np.zeros((2, len(FEATURE_NAMES), 2))
    target = np.zeros((2, 2))
    samples = VelocitySamples(
        "pair",
        features,
        target,
        np.array([1, 2]),
        np.array([0, 0]),
        np.array([[0.0, 0.0], [0.6, 0.0]]),
    )
    prediction = np.array([[0.5, 0.0], [-0.5, 0.0]])
    metrics = proximity_metrics(samples, prediction)
    assert metrics["predicted_contact_rate"] == 1.0
    assert metrics["false_contact_rate"] == 1.0
