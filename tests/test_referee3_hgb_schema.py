# Author: Amir Ghorbani
"""Referee-3 feature-schema and complete-run weighting checks for HGB."""

from __future__ import annotations

import numpy as np

import pedgeom.benchmarks as benchmarks
from pedgeom.calibration import FEATURE_NAMES, FittedVelocityModel, VelocitySamples


def _batch(run: str, count: int, wall_value: float) -> VelocitySamples:
    features = np.zeros((count, len(FEATURE_NAMES), 2), dtype=float)
    features[:, 0, 0] = 1.0
    features[:, 1, 0] = 1.0
    features[:, 5, :] = wall_value
    return VelocitySamples(
        run=run,
        features=features,
        target=np.column_stack((np.ones(count), np.zeros(count))),
        pedestrian_id=np.arange(count),
        frame=np.arange(count),
    )


class _CapturingEstimator:
    def __init__(self) -> None:
        self.design = None
        self.target = None
        self.sample_weight = None

    def fit(self, design, target, sample_weight=None):
        self.design = np.asarray(design)
        self.target = np.asarray(target)
        self.sample_weight = (
            None if sample_weight is None else np.asarray(sample_weight)
        )
        return self

    def predict(self, design):
        return np.zeros((len(design), 2), dtype=float)


def test_referee3_hgb_is_wall_free_and_equal_weights_complete_runs(monkeypatch) -> None:
    estimator = _CapturingEstimator()
    monkeypatch.setattr(
        benchmarks,
        "_hist_gradient_boosting_estimator",
        lambda **_kwargs: estimator,
    )
    batches = [_batch("short", 2, 10.0), _batch("long", 4, 20.0)]
    base = FittedVelocityModel(("persistence",), np.array([1.0]))

    model = benchmarks.fit_gradient_boosted_residual(
        batches,
        base,
        include_wall=False,
        run_balanced=True,
    )

    assert estimator.design.shape == (6, 12)
    np.testing.assert_allclose(estimator.sample_weight[:2].sum(), 1.0)
    np.testing.assert_allclose(estimator.sample_weight[2:].sum(), 1.0)
    assert model.include_wall is False
    assert model._design(batches[0]).shape == (2, 12)


def test_legacy_hgb_defaults_keep_wall_columns_and_origin_weights(monkeypatch) -> None:
    estimator = _CapturingEstimator()
    monkeypatch.setattr(
        benchmarks,
        "_hist_gradient_boosting_estimator",
        lambda **_kwargs: estimator,
    )
    batch = _batch("legacy", 3, 5.0)
    base = FittedVelocityModel(("persistence",), np.array([1.0]))

    model = benchmarks.fit_gradient_boosted_residual([batch], base)

    assert estimator.design.shape == (3, 14)
    assert estimator.sample_weight is None
    assert model.include_wall is True
    assert model._design(batch).shape == (3, 14)
