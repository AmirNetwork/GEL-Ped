# Author: Amir Ghorbani
"""Fair data-driven baselines using the same invariant interaction features and splits."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import least_squares, lsq_linear
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
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


def raw_neighbour_design(samples: VelocitySamples) -> np.ndarray:
    """Encode up to eight individual neighbours without interaction summaries."""

    if samples.raw_neighbours is None:
        raise ValueError("raw-neighbour states were not retained for this sample batch")
    goal, lateral = _local_basis(samples)
    raw = samples.raw_neighbours
    relative_position = raw[:, :, :2]
    relative_velocity = raw[:, :, 2:4]
    present = raw[:, :, 4:5]
    position_local = np.stack(
        (
            np.einsum("nkc,nc->nk", relative_position, goal),
            np.einsum("nkc,nc->nk", relative_position, lateral),
        ),
        axis=2,
    )
    velocity_local = np.stack(
        (
            np.einsum("nkc,nc->nk", relative_velocity, goal),
            np.einsum("nkc,nc->nk", relative_velocity, lateral),
        ),
        axis=2,
    )
    focal = invariant_design(samples)[:, [0, 6, 5, 11]]
    neighbours = np.concatenate((position_local, velocity_local, present), axis=2)
    return np.column_stack((focal, neighbours.reshape(len(samples.target), -1)))


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
class FullTensorGeometryRegressor:
    """Run-balanced full local tensor mapping over the ten geometric projections."""

    coefficients: np.ndarray  # ten scalar geometry inputs x two local outputs

    def predict(self, samples: VelocitySamples, speed_cap: float = 2.5) -> np.ndarray:
        local = tensor_scalar_design(samples) @ self.coefficients
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
class ResidualOnBaseRegressor:
    """Matched residual MLP around any frozen velocity predictor.

    This class is used for the reviewer-requested attribution study.  It has the
    same 12 invariant inputs and two base-prediction inputs as the GEL-Ped
    residual branch, but it does not assume that the base is a tensor model.
    """

    base: object
    estimator: RegressorMixin

    def _design(self, samples: VelocitySamples) -> np.ndarray:
        base_local = local_target_from_vectors(
            samples, self.base.predict(samples, speed_cap=100.0)
        )
        return np.column_stack((invariant_design(samples), base_local))

    def predict(self, samples: VelocitySamples, speed_cap: float = 2.5) -> np.ndarray:
        base_prediction = self.base.predict(samples, speed_cap=100.0)
        residual_local = self.estimator.predict(self._design(samples))
        goal, lateral = _local_basis(samples)
        residual = residual_local[:, 0, None] * goal + residual_local[:, 1, None] * lateral
        prediction = base_prediction + residual
        speed = np.linalg.norm(prediction, axis=1)
        scale = np.minimum(1.0, speed_cap / np.maximum(speed, 1e-12))
        return prediction * scale[:, None]


@dataclass
class AdditiveRegressor:
    """Add a calibrated invariant linear skip to a frozen nonlinear model."""

    base: object
    correction: InvariantRegressor

    def predict(self, samples: VelocitySamples, speed_cap: float = 2.5) -> np.ndarray:
        prediction = self.base.predict(samples, speed_cap=100.0) + self.correction.predict(
            samples, speed_cap=100.0
        )
        speed = np.linalg.norm(prediction, axis=1)
        scale = np.minimum(1.0, speed_cap / np.maximum(speed, 1e-12))
        return prediction * scale[:, None]


def _ttc_field(samples: VelocitySamples, radius: float = 0.25) -> np.ndarray:
    """Collision-normal field from positive pairwise contact times."""

    if samples.raw_neighbours is None:
        raise ValueError("raw-neighbour states are required for TTC prediction")
    raw = samples.raw_neighbours
    relative_position = raw[:, :, :2]
    relative_velocity = raw[:, :, 2:4]
    present = raw[:, :, 4] > 0.5
    a = np.sum(relative_velocity**2, axis=2)
    b = 2.0 * np.sum(relative_position * relative_velocity, axis=2)
    c = np.sum(relative_position**2, axis=2) - (2.0 * radius) ** 2
    discriminant = b**2 - 4.0 * a * c
    valid = present & (a > 1e-9) & (discriminant >= 0.0)
    root = np.full_like(a, np.inf)
    candidate = np.divide(
        -b - np.sqrt(np.maximum(discriminant, 0.0)),
        2.0 * a,
        out=np.full_like(a, np.inf),
        where=a > 1e-9,
    )
    root[valid & (candidate > 0.0) & (candidate <= 3.0)] = candidate[
        valid & (candidate > 0.0) & (candidate <= 3.0)
    ]
    distance = np.linalg.norm(relative_position, axis=2)
    away = np.divide(
        -relative_position,
        distance[:, :, None],
        out=np.zeros_like(relative_position),
        where=distance[:, :, None] > 1e-9,
    )
    weight = np.where(np.isfinite(root), np.exp(-root), 0.0)
    return np.sum(weight[:, :, None] * away, axis=1)


@dataclass
class TTCResponseRegressor:
    """Low-parameter time-to-collision response benchmark."""

    coefficients: np.ndarray

    def predict(self, samples: VelocitySamples, speed_cap: float = 2.5) -> np.ndarray:
        vectors = np.stack(
            (
                samples.features[:, 0],
                samples.features[:, 1],
                _ttc_field(samples),
                samples.features[:, 5],
            ),
            axis=1,
        )
        prediction = np.einsum("nfc,f->nc", vectors, self.coefficients)
        speed = np.linalg.norm(prediction, axis=1)
        scale = np.minimum(1.0, speed_cap / np.maximum(speed, 1e-12))
        return prediction * scale[:, None]


@dataclass
class AVMRegressor:
    """Sample-wise Anticipation Velocity Model using individual neighbour states."""

    desired_speed: float
    time_gap: float
    repulsion_strength: float
    interaction_range: float
    prediction_time: float
    turning_time: float
    radius: float = 0.25
    horizon_s: float = 0.4

    def predict(self, samples: VelocitySamples, speed_cap: float = 2.5) -> np.ndarray:
        if samples.raw_neighbours is None:
            raise ValueError("raw-neighbour states are required for AVM prediction")
        raw = samples.raw_neighbours
        position = raw[:, :, :2]
        relative_velocity = raw[:, :, 2:4]
        present = raw[:, :, 4] > 0.5
        goal, lateral = _local_basis(samples)
        predicted = position + self.prediction_time * relative_velocity
        distance = np.linalg.norm(position, axis=2)
        predicted_distance = np.maximum(np.linalg.norm(predicted, axis=2), 2.0 * self.radius)
        forward = np.einsum("nkc,nc->nk", position, goal)
        lateral_position = np.einsum("nkc,nc->nk", predicted, lateral)
        active = present & (distance <= 3.0) & (forward >= -0.2 * distance)
        side = np.where(lateral_position >= 0.0, -1.0, 1.0)
        impact = (
            self.repulsion_strength
            * np.exp((2.0 * self.radius - predicted_distance) / self.interaction_range)
            * active
        )
        avoidance = np.sum(impact * side, axis=1)
        desired_direction = goal + avoidance[:, None] * lateral
        desired_direction /= np.maximum(
            np.linalg.norm(desired_direction, axis=1, keepdims=True), 1e-12
        )
        previous = samples.features[:, 0]
        previous_direction = np.divide(
            previous,
            np.linalg.norm(previous, axis=1, keepdims=True),
            out=goal.copy(),
            where=np.linalg.norm(previous, axis=1, keepdims=True) > 1e-9,
        )
        turn_fraction = 1.0 - np.exp(-self.horizon_s / self.turning_time)
        direction = (1.0 - turn_fraction) * previous_direction + turn_fraction * desired_direction
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)

        along = np.einsum("nkc,nc->nk", position, direction)
        across = np.abs(
            np.einsum(
                "nkc,nc->nk",
                position,
                np.column_stack((-direction[:, 1], direction[:, 0])),
            )
        )
        obstructing = present & (along > 0.0) & (across <= 2.0 * self.radius)
        headway = np.where(obstructing, along, np.inf).min(axis=1)
        free_speed = np.where(
            np.isfinite(headway),
            np.maximum(headway - 2.0 * self.radius, 0.0) / self.time_gap,
            self.desired_speed,
        )
        speed = np.minimum(self.desired_speed, free_speed)
        prediction = speed[:, None] * direction
        cap_scale = np.minimum(
            1.0, speed_cap / np.maximum(np.linalg.norm(prediction, axis=1), 1e-12)
        )
        return prediction * cap_scale[:, None]


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


@dataclass
class DisagreementRoutedRegressor:
    """Route between an in-support expert and a transfer expert without labels.

    The score is the mean Euclidean disagreement between the two predictions over
    the current observation buffer.  Scores inside the calibration envelope use
    the in-support expert.  Above the robust upper fence, its weight decays
    exponentially over one calibration interquartile range.
    """

    in_support: object
    transfer: object
    threshold: float
    scale: float

    def routing_score(self, samples: VelocitySamples) -> float:
        in_support_prediction = self.in_support.predict(samples, speed_cap=100.0)
        transfer_prediction = self.transfer.predict(samples, speed_cap=100.0)
        return float(
            np.mean(np.linalg.norm(in_support_prediction - transfer_prediction, axis=1))
        )

    def in_support_weight(self, samples: VelocitySamples) -> float:
        excess = max(self.routing_score(samples) - self.threshold, 0.0)
        return float(np.exp(-excess / max(self.scale, 1e-12)))

    def predict(self, samples: VelocitySamples, speed_cap: float = 2.5) -> np.ndarray:
        in_support_prediction = self.in_support.predict(samples, speed_cap=100.0)
        transfer_prediction = self.transfer.predict(samples, speed_cap=100.0)
        weight = self.in_support_weight(samples)
        prediction = weight * in_support_prediction + (1.0 - weight) * transfer_prediction
        speed = np.linalg.norm(prediction, axis=1)
        cap_scale = np.minimum(1.0, speed_cap / np.maximum(speed, 1e-12))
        return prediction * cap_scale[:, None]


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


def fit_full_tensor_geometry_model(
    batches: list[VelocitySamples], penalty: float = 0.0
) -> FullTensorGeometryRegressor:
    """Fit a 20-coefficient full tensor prior with equal total run weight."""

    design_parts = []
    target_parts = []
    for batch in batches:
        weight = 1.0 / np.sqrt(len(batch.target))
        design_parts.append(tensor_scalar_design(batch) * weight)
        target_parts.append(local_target(batch) * weight)
    design = np.concatenate(design_parts)
    target = np.concatenate(target_parts)
    if penalty > 0.0:
        gram = design.T @ design + penalty * np.eye(design.shape[1])
        coefficients = np.linalg.solve(gram, design.T @ target)
    else:
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return FullTensorGeometryRegressor(coefficients)


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


def _hist_gradient_boosting_estimator(
    *,
    max_iter: int = 200,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1.0,
    learning_rate: float = 0.05,
    seed: int = 20260722,
) -> MultiOutputRegressor:
    """Construct the two-output tabular learner used by the routed model."""

    return MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            learning_rate=learning_rate,
            random_state=seed,
        )
    )


def fit_gradient_boosted_direct(
    batches: list[VelocitySamples],
    *,
    max_iter: int = 200,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1.0,
    learning_rate: float = 0.05,
    seed: int = 20260722,
) -> InvariantRegressor:
    """Fit a strong tabular control directly to local future velocity."""

    design = np.concatenate([invariant_design(batch) for batch in batches])
    target = np.concatenate([local_target(batch) for batch in batches])
    estimator = _hist_gradient_boosting_estimator(
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        learning_rate=learning_rate,
        seed=seed,
    )
    estimator.fit(design, target)
    return InvariantRegressor(estimator)


def fit_gradient_boosted_residual(
    batches: list[VelocitySamples],
    base: object,
    *,
    max_iter: int = 200,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1.0,
    learning_rate: float = 0.05,
    seed: int = 20260722,
) -> ResidualOnBaseRegressor:
    """Fit a boosted residual around a frozen kinematic or geometric base."""

    design_parts = []
    residual_parts = []
    for batch in batches:
        base_local = local_target_from_vectors(
            batch, base.predict(batch, speed_cap=100.0)
        )
        design_parts.append(np.column_stack((invariant_design(batch), base_local)))
        residual_parts.append(local_target(batch) - base_local)
    estimator = _hist_gradient_boosting_estimator(
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        learning_rate=learning_rate,
        seed=seed,
    )
    estimator.fit(np.concatenate(design_parts), np.concatenate(residual_parts))
    return ResidualOnBaseRegressor(base=base, estimator=estimator)


def calibrate_disagreement_router(
    batches: list[VelocitySamples],
    in_support: object,
    transfer: object,
) -> DisagreementRoutedRegressor:
    """Calibrate a Tukey upper-fence router from unlabeled calibration runs."""

    scores = []
    for batch in batches:
        first = in_support.predict(batch, speed_cap=100.0)
        second = transfer.predict(batch, speed_cap=100.0)
        scores.append(float(np.mean(np.linalg.norm(first - second, axis=1))))
    first_quartile, third_quartile = np.quantile(scores, (0.25, 0.75))
    interquartile_range = float(third_quartile - first_quartile)
    threshold = float(third_quartile + 1.5 * interquartile_range)
    return DisagreementRoutedRegressor(
        in_support=in_support,
        transfer=transfer,
        threshold=threshold,
        scale=max(interquartile_range, 1e-12),
    )


def fit_raw_neighbour_mlp(
    batches: list[VelocitySamples],
    hidden_layers: tuple[int, ...] = (112, 112),
    alpha: float = 1e-2,
    max_iter: int = 60,
    seed: int = 20260722,
) -> InvariantRegressor:
    """Fit a matched MLP to sorted raw neighbour states and focal motion."""

    design = np.concatenate([raw_neighbour_design(batch) for batch in batches])
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
            early_stopping=False,
            random_state=seed,
        ),
    )
    estimator.fit(design, target)
    return InvariantRegressor(estimator, raw_neighbour_design)


def fit_ttc_response_model(batches: list[VelocitySamples]) -> TTCResponseRegressor:
    """Fit persistence, goal, TTC, and wall weights with equal run influence."""

    design_parts = []
    target_parts = []
    for batch in batches:
        vectors = np.stack(
            (
                batch.features[:, 0],
                batch.features[:, 1],
                _ttc_field(batch),
                batch.features[:, 5],
            ),
            axis=1,
        )
        weight = 1.0 / np.sqrt(2.0 * len(batch.target))
        design_parts.append(vectors.transpose(0, 2, 1).reshape(-1, 4) * weight)
        target_parts.append(batch.target.reshape(-1) * weight)
    result = lsq_linear(
        np.concatenate(design_parts),
        np.concatenate(target_parts),
        bounds=(np.array([0.0, 0.0, -5.0, -5.0]), np.array([2.0, 2.0, 0.0, 5.0])),
    )
    return TTCResponseRegressor(result.x)


def fit_avm_model(batches: list[VelocitySamples]) -> AVMRegressor:
    """Calibrate six AVM parameters by run-balanced nonlinear least squares."""

    fitting_batches = []
    for batch in batches:
        indices = np.linspace(
            0, len(batch.target) - 1, min(750, len(batch.target)), dtype=int
        )

        def optional(values: np.ndarray | None) -> np.ndarray | None:
            return None if values is None else values[indices]

        fitting_batches.append(
            replace(
                batch,
                features=batch.features[indices],
                target=batch.target[indices],
                pedestrian_id=batch.pedestrian_id[indices],
                frame=batch.frame[indices],
                position=optional(batch.position),
                neighbour_count=optional(batch.neighbour_count),
                occupancy=optional(batch.occupancy),
                raw_neighbours=optional(batch.raw_neighbours),
            )
        )

    def unpack(values: np.ndarray) -> AVMRegressor:
        return AVMRegressor(*values)

    def residual(values: np.ndarray) -> np.ndarray:
        model = unpack(values)
        return np.concatenate(
            [
                (model.predict(batch, speed_cap=100.0) - batch.target).reshape(-1)
                / np.sqrt(2.0 * len(batch.target))
                for batch in fitting_batches
            ]
        )

    result = least_squares(
        residual,
        x0=np.array([1.35, 0.5, 2.0, 0.3, 0.8, 0.3]),
        bounds=(
            np.array([0.5, 0.1, 0.0, 0.1, 0.1, 0.05]),
            np.array([2.5, 2.0, 8.0, 1.5, 2.0, 2.0]),
        ),
        x_scale="jac",
        max_nfev=40,
    )
    return unpack(result.x)


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


def fit_residual_on_base_mlp(
    batches: list[VelocitySamples],
    base: object,
    hidden_layers: tuple[int, ...] = (64, 64),
    alpha: float = 1e-3,
    max_iter: int = 150,
    seed: int = 20260722,
) -> ResidualOnBaseRegressor:
    """Fit a capacity-matched residual MLP around a frozen base predictor."""

    design_parts = []
    residual_parts = []
    for batch in batches:
        base_local = local_target_from_vectors(
            batch, base.predict(batch, speed_cap=100.0)
        )
        design_parts.append(np.column_stack((invariant_design(batch), base_local)))
        residual_parts.append(local_target(batch) - base_local)
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
    return ResidualOnBaseRegressor(base=base, estimator=estimator)


def fit_linear_skip(
    batches: list[VelocitySamples],
    base: object,
    penalty: float = 1.0,
) -> AdditiveRegressor:
    """Fit an unrestricted 10-feature linear correction to a frozen predictor."""

    design = np.concatenate([tensor_scalar_design(batch) for batch in batches])
    residual = np.concatenate(
        [
            local_target(batch)
            - local_target_from_vectors(
                batch, base.predict(batch, speed_cap=100.0)
            )
            for batch in batches
        ]
    )
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=penalty))
    estimator.fit(design, residual)
    correction = InvariantRegressor(estimator, tensor_scalar_design)
    return AdditiveRegressor(base=base, correction=correction)


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
