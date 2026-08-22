# Author: Amir Ghorbani
"""Fair data-driven baselines using the same invariant interaction features and splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pedgeom.calibration import TensorGeometryModel, VelocitySamples
from pedgeom.calibration import FEATURE_NAMES


def _local_basis(samples: VelocitySamples) -> tuple[np.ndarray, np.ndarray]:
    goal = samples.features[:, 1, :]
    lateral = np.column_stack((-goal[:, 1], goal[:, 0]))
    return goal, lateral


def invariant_design(samples: VelocitySamples) -> np.ndarray:
    """Rotate every vector feature into a pedestrian's goal-aligned frame."""

    goal, lateral = _local_basis(samples)
    along = np.einsum("nfc,nc->nf", samples.features, goal)
    across = np.einsum("nfc,nc->nf", samples.features, lateral)
    return np.concatenate((along, across), axis=1)


def local_target(samples: VelocitySamples) -> np.ndarray:
    return local_target_from_vectors(samples, samples.target)


def local_target_from_vectors(samples: VelocitySamples, vectors: np.ndarray) -> np.ndarray:
    goal, lateral = _local_basis(samples)
    return np.column_stack(
        (np.sum(vectors * goal, axis=1), np.sum(vectors * lateral, axis=1))
    )


def tensor_scalar_design(samples: VelocitySamples) -> np.ndarray:
    """Return the ten scalar projections used by the structured tensor model."""

    goal, lateral = _local_basis(samples)
    along = np.einsum("nfc,nc->nf", samples.features, goal)
    across = np.einsum("nfc,nc->nf", samples.features, lateral)
    parallel = tuple(
        FEATURE_NAMES.index(name)
        for name in ("persistence", "goal", "isotropic", "forward", "closing")
    )
    perpendicular = tuple(
        FEATURE_NAMES.index(name)
        for name in ("persistence", "isotropic", "forward", "closing", "wall")
    )
    return np.concatenate((along[:, parallel], across[:, perpendicular]), axis=1)


@dataclass
class InvariantRegressor:
    """Direct goal-frame neural expert in manuscript Eq. (10)."""

    estimator: RegressorMixin
    design_builder: Callable[[VelocitySamples], np.ndarray] = invariant_design

    def predict(self, samples: VelocitySamples, speed_cap: float = 2.5) -> np.ndarray:
        local = self.estimator.predict(self.design_builder(samples))
        goal, lateral = _local_basis(samples)
        prediction = local[:, 0, None] * goal + local[:, 1, None] * lateral
        speed = np.linalg.norm(prediction, axis=1)
        scale = np.minimum(1.0, speed_cap / np.maximum(speed, 1e-12))
        return prediction * scale[:, None]


@dataclass
class GoalStableRegressor:
    """One-step goal-stable neural reference with positive route progress.

    For a positive-definite goal-directed dynamical system, velocity has a
    non-negative projection on the attraction direction. At the study's single
    forecast window, projecting a matched invariant neural prediction onto that
    half space is the least-squares realization of the defining stability constraint.
    This adapts the dynamical prior of Wang et al. (ICRA 2024) to the no-endpoint,
    fixed-information protocol; it is not their long-horizon Transformer.
    """

    base: InvariantRegressor

    def predict(self, samples: VelocitySamples, speed_cap: float = 2.5) -> np.ndarray:
        local = np.asarray(
            self.base.estimator.predict(self.base.design_builder(samples)), dtype=float
        ).copy()
        local[:, 0] = np.maximum(local[:, 0], 0.0)
        goal, lateral = _local_basis(samples)
        prediction = local[:, 0, None] * goal + local[:, 1, None] * lateral
        speed = np.linalg.norm(prediction, axis=1)
        scale = np.minimum(1.0, speed_cap / np.maximum(speed, 1e-12))
        return prediction * scale[:, None]


@dataclass
class TensorResidualRegressor:
    """Structured GEL-Ped expert implementing manuscript Eqs. (8)-(9)."""

    tensor: TensorGeometryModel
    estimator: RegressorMixin
    support_scaler: StandardScaler
    support_threshold: float
    gate_strength: float = 4.0

    def _design(self, samples: VelocitySamples) -> np.ndarray:
        tensor_local = local_target_from_vectors(
            samples, self.tensor.predict(samples, speed_cap=100.0)
        )
        return np.column_stack((invariant_design(samples), tensor_local))

    def support_score(self, samples: VelocitySamples) -> np.ndarray:
        standardized = self.support_scaler.transform(self._design(samples))
        return np.sqrt(np.mean(standardized**2, axis=1))

    def residual_gate(self, samples: VelocitySamples) -> np.ndarray:
        excess = np.maximum(
            self.support_score(samples) / max(self.support_threshold, 1e-12) - 1.0,
            0.0,
        )
        return 1.0 / (1.0 + self.gate_strength * excess**2)

    def predict(self, samples: VelocitySamples, speed_cap: float = 2.5) -> np.ndarray:
        tensor_prediction = self.tensor.predict(samples, speed_cap=100.0)
        residual_local = self.estimator.predict(self._design(samples))
        residual_local *= self.residual_gate(samples)[:, None]
        goal, lateral = _local_basis(samples)
        residual = residual_local[:, 0, None] * goal + residual_local[:, 1, None] * lateral
        prediction = tensor_prediction + residual
        speed = np.linalg.norm(prediction, axis=1)
        scale = np.minimum(1.0, speed_cap / np.maximum(speed, 1e-12))
        return prediction * scale[:, None]


@dataclass
class GELPedRegressor:
    """Complete GEL-Ped predictor: calibration-selected blend in manuscript Eq. (11)."""

    direct: object
    structured: object
    direct_weight: float

    def predict(self, samples: VelocitySamples, speed_cap: float = 2.5) -> np.ndarray:
        direct_prediction = self.direct.predict(samples, speed_cap=100.0)
        structured_prediction = self.structured.predict(samples, speed_cap=100.0)
        prediction = (
            self.direct_weight * direct_prediction
            + (1.0 - self.direct_weight) * structured_prediction
        )
        speed = np.linalg.norm(prediction, axis=1)
        scale = np.minimum(1.0, speed_cap / np.maximum(speed, 1e-12))
        return prediction * scale[:, None]


# Backward-compatible import for earlier notebooks and archived experiment scripts.
BlendedRegressor = GELPedRegressor


def fit_invariant_ridge(batches: list[VelocitySamples], penalty: float = 1.0) -> InvariantRegressor:
    design = np.concatenate([invariant_design(batch) for batch in batches])
    target = np.concatenate([local_target(batch) for batch in batches])
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=penalty))
    estimator.fit(design, target)
    return InvariantRegressor(estimator)


def fit_unrestricted_tensor_features(
    batches: list[VelocitySamples], penalty: float = 0.0
) -> InvariantRegressor:
    """Fit both velocity channels from all ten tensor projections (22 parameters)."""

    design = np.concatenate([tensor_scalar_design(batch) for batch in batches])
    target = np.concatenate([local_target(batch) for batch in batches])
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=penalty))
    estimator.fit(design, target)
    return InvariantRegressor(estimator, tensor_scalar_design)


def fit_interaction_mlp(
    batches: list[VelocitySamples],
    hidden_layers: tuple[int, ...] = (64, 64),
    alpha: float = 1e-3,
    max_iter: int = 150,
    early_stopping: bool = False,
    seed: int = 20260722,
) -> InvariantRegressor:
    """Fit an interaction MLP using a configuration selected by run-wise validation."""

    design = np.concatenate([invariant_design(batch) for batch in batches])
    target = np.concatenate([local_target(batch) for batch in batches])
    estimator = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="tanh",
            solver="adam",
            alpha=alpha,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=seed,
        ),
    )
    estimator.fit(design, target)
    return InvariantRegressor(estimator)


def fit_tensor_residual_mlp(
    batches: list[VelocitySamples],
    tensor: TensorGeometryModel,
    hidden_layers: tuple[int, ...] = (64, 64),
    alpha: float = 1e-3,
    max_iter: int = 150,
    seed: int = 20260722,
    support_quantile: float = 0.95,
    gate_strength: float = 4.0,
) -> TensorResidualRegressor:
    """Fit a nonlinear correction around a frozen tensor response.

    Both the correction and its support threshold use calibration observations only.
    Beyond that support, the correction is smoothly shrunk toward the tensor prior.
    """

    design_parts = []
    residual_parts = []
    for batch in batches:
        tensor_prediction = tensor.predict(batch, speed_cap=100.0)
        tensor_local = local_target_from_vectors(batch, tensor_prediction)
        design_parts.append(np.column_stack((invariant_design(batch), tensor_local)))
        residual_parts.append(local_target(batch) - tensor_local)
    design = np.concatenate(design_parts)
    residual = np.concatenate(residual_parts)
    estimator = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="tanh",
            solver="adam",
            alpha=alpha,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=False,
            random_state=seed,
        ),
    )
    estimator.fit(design, residual)
    support_scaler = StandardScaler().fit(design)
    standardized = support_scaler.transform(design)
    support_distance = np.sqrt(np.mean(standardized**2, axis=1))
    threshold = float(np.quantile(support_distance, support_quantile))
    return TensorResidualRegressor(
        tensor=tensor,
        estimator=estimator,
        support_scaler=support_scaler,
        support_threshold=threshold,
        gate_strength=gate_strength,
    )


def fit_interaction_mlp_with_run_stopping(
    batches: list[VelocitySamples],
    validation: VelocitySamples,
    hidden_layers: tuple[int, ...] = (64, 64),
    alpha: float = 1e-3,
    maximum_epochs: int = 60,
    patience: int = 8,
    seed: int = 20260722,
) -> tuple[InvariantRegressor, int, list[float]]:
    """Stop an MLP using one complete calibration run as the validation unit."""

    design = np.concatenate([invariant_design(batch) for batch in batches])
    target = np.concatenate([local_target(batch) for batch in batches])
    scaler = StandardScaler().fit(design)
    scaled_design = scaler.transform(design)
    validation_design = scaler.transform(invariant_design(validation))
    validation_target = local_target(validation)
    regressor = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="tanh",
        solver="adam",
        alpha=alpha,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=1,
        early_stopping=False,
        random_state=seed,
    )
    best_epoch = 1
    best_loss = float("inf")
    best_coefs: list[np.ndarray] | None = None
    best_intercepts: list[np.ndarray] | None = None
    epochs_without_improvement = 0
    history: list[float] = []
    for epoch in range(1, maximum_epochs + 1):
        regressor.partial_fit(scaled_design, target)
        prediction = regressor.predict(validation_design)
        loss = float(np.sqrt(np.mean(np.sum((prediction - validation_target) ** 2, axis=1))))
        history.append(loss)
        if loss < best_loss - 1e-5:
            best_loss = loss
            best_epoch = epoch
            best_coefs = [coefficient.copy() for coefficient in regressor.coefs_]
            best_intercepts = [intercept.copy() for intercept in regressor.intercepts_]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    if best_coefs is None or best_intercepts is None:
        raise RuntimeError("MLP validation failed to produce a finite checkpoint")
    regressor.coefs_ = best_coefs
    regressor.intercepts_ = best_intercepts
    estimator = make_pipeline(scaler, regressor)
    return InvariantRegressor(estimator), best_epoch, history
