# Author: Amir Ghorbani
"""Causal runtime-assurance utilities for GEL-Ped referee-3 analyses.

The functions in this module are deliberately independent of the existing
experiment scripts.  They provide small, deterministic building blocks for a
time-indexed guard, tail-risk evaluation, ensemble-error decomposition, and a
causal delayed-feedback comparator.  No function consults future observations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Hashable, Literal, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, roc_auc_score


GateKind = Literal["exponential", "linear", "logistic"]


def _one_dimensional(name: str, values: ArrayLike, n: int | None = None) -> NDArray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if n is not None and len(array) != n:
        raise ValueError(f"{name} must contain {n} entries")
    return array


def _float_vector(name: str, values: ArrayLike, n: int | None = None) -> NDArray[np.float64]:
    array = np.asarray(_one_dimensional(name, values, n), dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _hashable(value: object, name: str) -> Hashable:
    try:
        hash(value)
    except TypeError as error:
        raise ValueError(f"{name} entries must be hashable") from error
    return value  # type: ignore[return-value]


def _same_label(left: object, right: object) -> bool:
    """Return scalar equality for scene labels and reject ambiguous arrays."""

    result = left == right
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    raise ValueError("scene identifiers must have scalar equality")


def causal_trailing_time_mean(
    values: ArrayLike,
    pedestrian_ids: ArrayLike,
    frames: ArrayLike,
    *,
    frames_per_second: float,
    window_seconds: float,
    scene_ids: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute a causal trailing-time mean for each pedestrian.

    The current value and earlier values with
    ``current_time - window_seconds <= time <= current_time`` are included.
    Input order defines the causal processing order.  Frames must be strictly
    increasing for a pedestrian within a contiguous scene block.  Whenever the
    scene identifier changes, *all* histories are cleared; reusing a pedestrian
    identifier in a new scene therefore cannot leak its earlier state.

    ``values`` may be one-dimensional or have arbitrary trailing dimensions.
    Frames are converted to seconds as ``frame / frames_per_second``.
    """

    data = np.asarray(values, dtype=float)
    if data.ndim == 0:
        raise ValueError("values must have a row dimension")
    if not np.all(np.isfinite(data)):
        raise ValueError("values must contain only finite values")
    n = data.shape[0]
    pedestrians = _one_dimensional("pedestrian_ids", pedestrian_ids, n)
    frame_values = _float_vector("frames", frames, n)
    if scene_ids is None:
        scenes = np.zeros(n, dtype=np.int8)
    else:
        scenes = _one_dimensional("scene_ids", scene_ids, n)
    if not np.isfinite(frames_per_second) or frames_per_second <= 0:
        raise ValueError("frames_per_second must be positive and finite")
    if not np.isfinite(window_seconds) or window_seconds < 0:
        raise ValueError("window_seconds must be non-negative and finite")
    if n == 0:
        return np.empty_like(data, dtype=float)

    output = np.empty_like(data, dtype=float)
    histories: dict[Hashable, deque[tuple[float, NDArray[np.float64]]]] = {}
    running_sums: dict[Hashable, NDArray[np.float64]] = {}
    last_frames: dict[Hashable, float] = {}
    current_scene: object | None = None

    for row in range(n):
        scene = _hashable(scenes[row].item() if hasattr(scenes[row], "item") else scenes[row], "scene_ids")
        if row == 0 or not _same_label(scene, current_scene):
            histories.clear()
            running_sums.clear()
            last_frames.clear()
            current_scene = scene

        raw_pedestrian = pedestrians[row]
        pedestrian = _hashable(
            raw_pedestrian.item() if hasattr(raw_pedestrian, "item") else raw_pedestrian,
            "pedestrian_ids",
        )
        frame = float(frame_values[row])
        if pedestrian in last_frames and frame <= last_frames[pedestrian]:
            raise ValueError(
                "frames must be strictly increasing for each pedestrian within a scene"
            )
        last_frames[pedestrian] = frame
        time = frame / frames_per_second
        cutoff = time - window_seconds

        history = histories.setdefault(pedestrian, deque())
        if pedestrian not in running_sums:
            running_sums[pedestrian] = np.zeros_like(data[row], dtype=float)
        running_sum = running_sums[pedestrian]
        while history and history[0][0] < cutoff - 1e-12:
            _, expired = history.popleft()
            running_sum -= expired

        current_value = np.array(data[row], dtype=float, copy=True)
        history.append((time, current_value))
        running_sum += current_value
        output[row] = running_sum / len(history)

    return output


def finite_sample_upper_quantile(values: ArrayLike, probability: float) -> float:
    """Return the conservative finite-sample upper quantile.

    For ``n`` observations the selected one-based order-statistic rank is
    ``ceil(probability * (n + 1))``, clipped to ``[1, n]``.  This convention is
    explicit so calibration does not depend on a library interpolation default.
    """

    sample = _float_vector("values", values)
    if len(sample) == 0:
        raise ValueError("values must not be empty")
    if not np.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("probability must lie strictly between zero and one")
    rank = int(np.ceil(probability * (len(sample) + 1)))
    rank = min(max(rank, 1), len(sample))
    return float(np.partition(sample, rank - 1)[rank - 1])


def one_sided_gate_weights(
    scores: ArrayLike,
    *,
    threshold: float,
    scale: float,
    kind: GateKind = "exponential",
) -> NDArray[np.float64]:
    """Map scores to graph-expert weights that equal one inside an envelope.

    ``threshold`` defines the calibration envelope.  For positive normalised
    excess ``x = (score - threshold) / scale``, the three outside-envelope
    forms are ``exp(-x)``, ``max(0, 1-x)``, and ``2/(1+exp(x))``.  The logistic
    form is scaled to meet the identity branch continuously at ``x=0``.
    """

    score = np.asarray(scores, dtype=float)
    if not np.all(np.isfinite(score)):
        raise ValueError("scores must contain only finite values")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    if kind not in {"exponential", "linear", "logistic"}:
        raise ValueError(f"unknown gate kind: {kind}")

    excess = np.maximum((score - threshold) / scale, 0.0)
    if kind == "exponential":
        outside = np.exp(-excess)
    elif kind == "linear":
        outside = np.maximum(1.0 - excess, 0.0)
    else:
        # Computing from exp(-x) avoids overflow for very large positive x.
        exp_negative = np.exp(-excess)
        outside = 2.0 * exp_negative / (1.0 + exp_negative)
    return np.where(score <= threshold, 1.0, outside).astype(float, copy=False)


def empirical_upper_cvar(errors: ArrayLike, level: float) -> float:
    """Return empirical upper-tail CVaR with fractional boundary mass.

    The largest ``(1-level) * n`` observations are averaged.  When that number
    is non-integer, the next boundary observation contributes fractionally.
    This is deterministic and avoids unstable threshold-tie conventions.
    """

    values = _float_vector("errors", errors)
    if len(values) == 0:
        raise ValueError("errors must not be empty")
    if np.any(values < 0):
        raise ValueError("errors must be non-negative")
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must lie strictly between zero and one")

    descending = np.sort(values)[::-1]
    tail_mass = (1.0 - level) * len(descending)
    complete = int(np.floor(tail_mass))
    fraction = tail_mass - complete
    numerator = float(np.sum(descending[:complete]))
    if fraction > 1e-15:
        numerator += fraction * float(descending[complete])
    return numerator / tail_mass


def per_run_tail_metrics(
    errors: ArrayLike,
    run_ids: ArrayLike,
    *,
    large_error_threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute tail-risk metrics separately for each run.

    Quantiles use NumPy's linear empirical interpolation.  CVaR uses
    :func:`empirical_upper_cvar`.  ``large_error_rate`` is the strict rate
    ``P(error > large_error_threshold)``.
    """

    error = _float_vector("errors", errors)
    if np.any(error < 0):
        raise ValueError("errors must be non-negative")
    runs = _one_dimensional("run_ids", run_ids, len(error))
    if len(error) == 0:
        raise ValueError("errors must not be empty")
    if not np.isfinite(large_error_threshold) or large_error_threshold < 0:
        raise ValueError("large_error_threshold must be non-negative and finite")

    rows: list[dict[str, object]] = []
    for run in pd.unique(runs):
        mask = runs == run
        current = error[mask]
        rows.append(
            {
                "run_id": run,
                "n": int(len(current)),
                "mean_error": float(np.mean(current)),
                "q90": float(np.quantile(current, 0.90, method="linear")),
                "q95": float(np.quantile(current, 0.95, method="linear")),
                "q99": float(np.quantile(current, 0.99, method="linear")),
                "cvar95": empirical_upper_cvar(current, 0.95),
                "cvar99": empirical_upper_cvar(current, 0.99),
                "large_error_rate": float(np.mean(current > large_error_threshold)),
            }
        )
    return pd.DataFrame(rows)


def binary_detection_metrics(labels: ArrayLike, scores: ArrayLike) -> dict[str, float]:
    """Return AUROC and average precision (AUPRC) for binary detection.

    Both classes must be present.  Average precision is used for AUPRC because
    it is the standard step-wise summary for an imbalanced rare-event target.
    """

    label = _one_dimensional("labels", labels)
    score = _float_vector("scores", scores, len(label))
    if len(label) == 0:
        raise ValueError("labels must not be empty")
    if not np.all(np.isin(label, [0, 1, False, True])):
        raise ValueError("labels must be binary")
    binary = label.astype(int)
    if len(np.unique(binary)) != 2:
        raise ValueError("labels must contain both classes")
    return {
        "n": float(len(binary)),
        "positive_rate": float(np.mean(binary)),
        "auroc": float(roc_auc_score(binary, score)),
        "auprc": float(average_precision_score(binary, score)),
    }


def top_fraction_binary_labels(scores: ArrayLike, fraction: float) -> NDArray[np.int64]:
    """Label exactly the highest-scoring fraction of a finite sample.

    The number of positives is ``ceil(fraction * n)``, clipped so that at least
    one positive and one negative remain.  A stable descending sort provides a
    deterministic index-order tie break.  This rank convention avoids the
    variable event prevalence produced by thresholding at an interpolated
    quantile when several observations share the boundary value.
    """

    score = _float_vector("scores", scores)
    if len(score) < 2:
        raise ValueError("scores must contain at least two observations")
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("fraction must lie strictly between zero and one")
    positive_count = min(max(int(np.ceil(fraction * len(score))), 1), len(score) - 1)
    # Negating before a stable ascending sort retains input order within ties.
    selected = np.argsort(-score, kind="stable")[:positive_count]
    labels = np.zeros(len(score), dtype=np.int64)
    labels[selected] = 1
    return labels


@dataclass(frozen=True)
class RiskRetentionCurve:
    """Discrete risk-retention curve sorted from lowest to highest score."""

    retention: NDArray[np.float64]
    risk: NDArray[np.float64]
    aurc: float
    order: NDArray[np.int64]


def risk_retention_curve(losses: ArrayLike, scores: ArrayLike) -> RiskRetentionCurve:
    """Return a tie-invariant retained-set risk curve and its discrete area.

    Forecasts are retained from lower to higher score.  All observations with
    the same score enter together because the score cannot order them.  Risk is
    reported at the end of each score group and AURC is the right-continuous
    area weighted by that group's share of observations.  Thus a constant
    score has AURC equal to overall risk rather than an arbitrary value induced
    by input order.  ``order`` still contains every row in stable sorted order.
    """

    loss = _float_vector("losses", losses)
    score = _float_vector("scores", scores, len(loss))
    if len(loss) == 0:
        raise ValueError("losses must not be empty")
    if np.any(loss < 0):
        raise ValueError("losses must be non-negative")
    order = np.argsort(score, kind="stable")
    ordered_loss = loss[order]
    ordered_score = score[order]
    group_ends = np.flatnonzero(
        np.r_[ordered_score[1:] != ordered_score[:-1], True]
    )
    counts = group_ends.astype(float) + 1.0
    risk = np.cumsum(ordered_loss)[group_ends] / counts
    retention = counts / len(loss)
    group_sizes = np.diff(np.r_[0, counts])
    return RiskRetentionCurve(
        retention=retention,
        risk=risk,
        aurc=float(np.sum(risk * group_sizes) / len(loss)),
        order=order.astype(np.int64, copy=False),
    )


@dataclass(frozen=True)
class EnsembleErrorDecomposition:
    """Exact squared-error decomposition for an ensemble of vector forecasts."""

    ensemble_mean: NDArray[np.float64]
    shared_squared_error: NDArray[np.float64]
    between_member_variance: NDArray[np.float64]
    mean_member_squared_error: NDArray[np.float64]
    shared_mse: float
    between_member_variance_mean: float
    member_mse: float


def ensemble_error_decomposition(
    member_predictions: ArrayLike,
    targets: ArrayLike,
) -> EnsembleErrorDecomposition:
    """Decompose member MSE into shared error and between-member variance.

    ``member_predictions`` has shape ``(members, samples, dimensions)`` and
    ``targets`` has shape ``(samples, dimensions)``.  For every sample the
    following identity holds up to floating-point roundoff::

        mean_m ||p_m - y||^2
          = ||mean_m p_m - y||^2 + mean_m ||p_m - mean_m p_m||^2.

    The first term is called ``shared_squared_error`` because every member's
    average prediction shares it; the second is the detectable between-member
    component.  These are squared vector errors, not RMSE values.
    """

    predictions = np.asarray(member_predictions, dtype=float)
    target = np.asarray(targets, dtype=float)
    if predictions.ndim != 3:
        raise ValueError("member_predictions must have shape (members, samples, dimensions)")
    if target.ndim != 2:
        raise ValueError("targets must have shape (samples, dimensions)")
    if predictions.shape[0] < 1 or predictions.shape[1] < 1:
        raise ValueError("at least one member and one sample are required")
    if predictions.shape[1:] != target.shape:
        raise ValueError("prediction sample/dimension shape must match targets")
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(target)):
        raise ValueError("predictions and targets must be finite")

    ensemble_mean = np.mean(predictions, axis=0)
    shared = np.sum((ensemble_mean - target) ** 2, axis=1)
    between = np.mean(np.sum((predictions - ensemble_mean[None, :, :]) ** 2, axis=2), axis=0)
    member_error = np.mean(
        np.sum((predictions - target[None, :, :]) ** 2, axis=2),
        axis=0,
    )
    return EnsembleErrorDecomposition(
        ensemble_mean=ensemble_mean,
        shared_squared_error=shared,
        between_member_variance=between,
        mean_member_squared_error=member_error,
        shared_mse=float(np.mean(shared)),
        between_member_variance_mean=float(np.mean(between)),
        member_mse=float(np.mean(member_error)),
    )


def permute_weights_within_runs(
    weights: ArrayLike,
    run_ids: ArrayLike,
    *,
    seed: int,
) -> NDArray:
    """Permute weight rows independently within each run.

    The multiset and marginal distribution of weights in every run are exactly
    preserved while their forecast-level alignment is broken.  Run order is
    first-appearance order and NumPy's ``PCG64`` generator makes a given seed
    reproducible.
    """

    weight = np.asarray(weights)
    if weight.ndim == 0:
        raise ValueError("weights must have a row dimension")
    runs = _one_dimensional("run_ids", run_ids, weight.shape[0])
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    generator = np.random.default_rng(int(seed))
    output = np.array(weight, copy=True)
    for run in pd.unique(runs):
        indices = np.flatnonzero(runs == run)
        output[indices] = weight[indices[generator.permutation(len(indices))]]
    return output


@dataclass(frozen=True)
class CausalExpertWeightResult:
    """Weights available at each forecast origin and prior labels consumed."""

    weights: NDArray[np.float64]
    updates_applied: NDArray[np.int64]


def causal_delayed_two_expert_weights(
    losses: ArrayLike,
    frames: ArrayLike,
    *,
    frames_per_second: float,
    label_delay_seconds: float,
    learning_rate: float,
    fixed_share: float = 0.0,
    stream_ids: ArrayLike | None = None,
    scene_ids: ArrayLike | None = None,
    initial_weights: Sequence[float] = (0.5, 0.5),
) -> CausalExpertWeightResult:
    """Compute causal exponential/fixed-share weights for two experts.

    ``losses[row]`` contains the two realised losses for the forecast issued at
    ``frames[row]``.  That loss is queued and can update future weights only at
    origins whose frame is at least
    ``frames[row] + label_delay_seconds * frames_per_second``.  Updates due at
    the current origin are applied *before* returning its weights; the current
    row's own loss is queued afterwards and can never affect its own forecast.

    Each ``stream_id`` maintains independent weights and delayed-label queue.
    A scene transition clears every stream, queue, and counter.  Frames must be
    strictly increasing for a stream within a contiguous scene block.

    Setting ``fixed_share=0`` gives standard exponential weighting.  For two
    experts, positive fixed share applies the transition
    ``[(1-a)w0+a*w1, a*w0+(1-a)w1]`` after every loss update.
    """

    loss = np.asarray(losses, dtype=float)
    if loss.ndim != 2 or loss.shape[1] != 2:
        raise ValueError("losses must have shape (origins, 2)")
    if not np.all(np.isfinite(loss)) or np.any(loss < 0):
        raise ValueError("losses must be finite and non-negative")
    n = loss.shape[0]
    frame_values = _float_vector("frames", frames, n)
    streams = (
        np.zeros(n, dtype=np.int8)
        if stream_ids is None
        else _one_dimensional("stream_ids", stream_ids, n)
    )
    scenes = (
        np.zeros(n, dtype=np.int8)
        if scene_ids is None
        else _one_dimensional("scene_ids", scene_ids, n)
    )
    if not np.isfinite(frames_per_second) or frames_per_second <= 0:
        raise ValueError("frames_per_second must be positive and finite")
    if not np.isfinite(label_delay_seconds) or label_delay_seconds < 0:
        raise ValueError("label_delay_seconds must be non-negative and finite")
    if not np.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("learning_rate must be non-negative and finite")
    if not np.isfinite(fixed_share) or not 0 <= fixed_share <= 1:
        raise ValueError("fixed_share must lie in [0, 1]")
    initial = np.asarray(initial_weights, dtype=float)
    if initial.shape != (2,) or not np.all(np.isfinite(initial)) or np.any(initial <= 0):
        raise ValueError("initial_weights must contain two positive finite values")
    initial = initial / np.sum(initial)

    output = np.empty((n, 2), dtype=float)
    update_counts = np.empty(n, dtype=np.int64)
    weights_by_stream: dict[Hashable, NDArray[np.float64]] = {}
    queues: dict[Hashable, deque[tuple[float, NDArray[np.float64]]]] = {}
    counts: dict[Hashable, int] = {}
    last_frames: dict[Hashable, float] = {}
    current_scene: object | None = None
    delay_frames = label_delay_seconds * frames_per_second

    for row in range(n):
        raw_scene = scenes[row]
        scene = _hashable(
            raw_scene.item() if hasattr(raw_scene, "item") else raw_scene,
            "scene_ids",
        )
        if row == 0 or not _same_label(scene, current_scene):
            weights_by_stream.clear()
            queues.clear()
            counts.clear()
            last_frames.clear()
            current_scene = scene

        raw_stream = streams[row]
        stream = _hashable(
            raw_stream.item() if hasattr(raw_stream, "item") else raw_stream,
            "stream_ids",
        )
        frame = float(frame_values[row])
        if stream in last_frames and frame <= last_frames[stream]:
            raise ValueError("frames must be strictly increasing for each stream within a scene")
        last_frames[stream] = frame

        current_weights = weights_by_stream.setdefault(stream, initial.copy())
        queue = queues.setdefault(stream, deque())
        counts.setdefault(stream, 0)
        while queue and queue[0][0] <= frame + 1e-12:
            _, observed_loss = queue.popleft()
            log_weight = np.log(current_weights) - learning_rate * observed_loss
            log_weight -= np.max(log_weight)
            updated = np.exp(log_weight)
            updated /= np.sum(updated)
            if fixed_share:
                updated = np.array(
                    [
                        (1.0 - fixed_share) * updated[0] + fixed_share * updated[1],
                        fixed_share * updated[0] + (1.0 - fixed_share) * updated[1],
                    ]
                )
            current_weights[:] = updated
            counts[stream] += 1

        output[row] = current_weights
        update_counts[row] = counts[stream]
        queue.append((frame + delay_frames, np.array(loss[row], copy=True)))

    return CausalExpertWeightResult(weights=output, updates_applied=update_counts)


__all__ = [
    "CausalExpertWeightResult",
    "EnsembleErrorDecomposition",
    "RiskRetentionCurve",
    "binary_detection_metrics",
    "causal_delayed_two_expert_weights",
    "causal_trailing_time_mean",
    "empirical_upper_cvar",
    "ensemble_error_decomposition",
    "finite_sample_upper_quantile",
    "one_sided_gate_weights",
    "per_run_tail_metrics",
    "permute_weights_within_runs",
    "risk_retention_curve",
    "top_fraction_binary_labels",
]
