# Author: Amir Ghorbani
"""Analyse referee-3 member predictions without retraining or test tuning.

The analysis is intentionally forecast-level but all uncertainty summaries are
formed within complete runs before regime aggregation.  Crossing is labelled
development-informed.  The same code is later used once, unchanged, for the
locked T-junction evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, ttest_1samp
from sklearn.ensemble import HistGradientBoostingRegressor

from pedgeom.runtime_guard import (
    binary_detection_metrics,
    causal_delayed_two_expert_weights,
    causal_trailing_time_mean,
    ensemble_error_decomposition,
    finite_sample_upper_quantile,
    one_sided_gate_weights,
    per_run_tail_metrics,
    permute_weights_within_runs,
    risk_retention_curve,
)


@dataclass(frozen=True)
class Bundle:
    path: Path
    run: str
    role: str
    target: np.ndarray
    frame: np.ndarray
    origin_time_s: np.ndarray
    pedestrian_id: np.ndarray
    position: np.ndarray
    prior_velocity: np.ndarray
    neighbour_count: np.ndarray
    occupancy: np.ndarray
    members_raw: np.ndarray
    fallback_raw: np.ndarray
    constant_velocity: np.ndarray
    mahalanobis: np.ndarray


@dataclass(frozen=True)
class DrawCalibration:
    indices: tuple[int, ...]
    tau: float
    scale: float
    disagreement_tau: float
    disagreement_scale: float
    mahalanobis_tau: float
    mahalanobis_scale: float
    speed_tau: float
    speed_scale: float
    fixed_graph_weight: float


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _scalar(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def load_bundle(path: Path) -> Bundle:
    with np.load(path, allow_pickle=False) as data:
        return Bundle(
            path=path,
            run=_scalar(data["run"]),
            role=_scalar(data["role"]),
            target=np.array(data["target"], dtype=float),
            frame=np.array(data["frame"], dtype=np.int64),
            origin_time_s=np.array(data["origin_time_s"], dtype=float),
            pedestrian_id=np.array(data["pedestrian_id"]),
            position=np.array(data["position"], dtype=float),
            prior_velocity=np.array(data["prior_velocity"], dtype=float),
            neighbour_count=np.array(data["neighbour_count"], dtype=float),
            occupancy=np.array(data["occupancy"], dtype=float),
            members_raw=np.array(data["members_raw"], dtype=float),
            fallback_raw=np.array(data["fallback_raw"], dtype=float),
            constant_velocity=np.array(data["constant_velocity"], dtype=float),
            mahalanobis=np.array(data["mahalanobis"], dtype=float),
        )


def cap_vectors(vectors: np.ndarray, limit: float) -> np.ndarray:
    speed = np.linalg.norm(vectors, axis=-1)
    scale = np.minimum(1.0, limit / np.maximum(speed, 1e-12))
    return vectors * scale[..., None]


def selected_predictions(
    bundle: Bundle, indices: tuple[int, ...], speed_cap: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    members = cap_vectors(bundle.members_raw[np.asarray(indices)], speed_cap)
    graph = np.mean(members, axis=0)
    fallback = cap_vectors(bundle.fallback_raw, speed_cap)
    return members, graph, fallback


def _rolling(bundle: Bundle, values: np.ndarray, window_seconds: float) -> np.ndarray:
    # Explicit seconds make the calculation invariant to each archive's native
    # frame rate and to the 16 -> 10 Hz resampling used by the untouched test.
    return causal_trailing_time_mean(
        values,
        bundle.pedestrian_id,
        bundle.origin_time_s,
        frames_per_second=1.0,
        window_seconds=window_seconds,
    )


def signals(
    bundle: Bundle,
    members: np.ndarray,
    graph: np.ndarray,
    fallback: np.ndarray,
    window_seconds: float,
) -> dict[str, np.ndarray]:
    instantaneous_spread = np.sqrt(
        np.mean(np.sum((members - graph[None, :, :]) ** 2, axis=2), axis=0)
    )
    disagreement = np.linalg.norm(graph - fallback, axis=1)
    speed = np.linalg.norm(bundle.prior_velocity, axis=1)
    return {
        "spread": _rolling(bundle, instantaneous_spread, window_seconds),
        "disagreement": _rolling(bundle, disagreement, window_seconds),
        "mahalanobis": _rolling(bundle, bundle.mahalanobis, window_seconds),
        "speed": _rolling(bundle, speed, window_seconds),
    }


def _threshold_scale(values: np.ndarray) -> tuple[float, float]:
    first, third = np.quantile(values, [0.25, 0.75])
    return finite_sample_upper_quantile(values, 0.9), max(float(third - first), 1e-6)


def _run_balanced_rmse(
    targets: list[np.ndarray], predictions: list[np.ndarray]
) -> float:
    values = [
        np.sqrt(np.mean(np.sum((prediction - target) ** 2, axis=1)))
        for target, prediction in zip(targets, predictions, strict=True)
    ]
    return float(np.mean(values))


def calibrate_draw(
    oof: list[Bundle],
    indices: tuple[int, ...],
    config: dict,
) -> DrawCalibration:
    score_parts: dict[str, list[np.ndarray]] = {
        "spread": [],
        "disagreement": [],
        "mahalanobis": [],
        "speed": [],
    }
    graphs: list[np.ndarray] = []
    fallbacks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    speed_cap = float(config["experts"]["fallback_prediction_speed_cap_mps"])
    window = float(config["guard"]["trailing_window_seconds"])
    for bundle in oof:
        members, graph, fallback = selected_predictions(bundle, indices, speed_cap)
        current = signals(bundle, members, graph, fallback, window)
        for name in score_parts:
            score_parts[name].append(current[name])
        graphs.append(graph)
        fallbacks.append(fallback)
        targets.append(bundle.target)
    limits = {
        name: _threshold_scale(np.concatenate(parts))
        for name, parts in score_parts.items()
    }
    step = float(config["controls"]["fixed_blend_grid_step"])
    candidates = np.arange(0.0, 1.0 + step / 2.0, step)
    losses = [
        _run_balanced_rmse(
            targets,
            [
                weight * graph + (1.0 - weight) * fallback
                for graph, fallback in zip(graphs, fallbacks, strict=True)
            ],
        )
        for weight in candidates
    ]
    return DrawCalibration(
        indices=indices,
        tau=limits["spread"][0],
        scale=limits["spread"][1],
        disagreement_tau=limits["disagreement"][0],
        disagreement_scale=limits["disagreement"][1],
        mahalanobis_tau=limits["mahalanobis"][0],
        mahalanobis_scale=limits["mahalanobis"][1],
        speed_tau=limits["speed"][0],
        speed_scale=limits["speed"][1],
        fixed_graph_weight=float(candidates[int(np.argmin(losses))]),
    )


def draw_indices(
    pool_size: int, size: int, count: int, seed: int, *, include_primary: bool
) -> list[tuple[int, ...]]:
    if size > pool_size:
        raise ValueError(f"ensemble size {size} exceeds member pool {pool_size}")
    generator = np.random.default_rng(seed + 10_000 * size)
    output: list[tuple[int, ...]] = []
    if include_primary:
        output.append(tuple(range(size)))
    seen = set(output)
    while len(output) < count:
        candidate = tuple(sorted(generator.choice(pool_size, size=size, replace=False)))
        if candidate not in seen:
            seen.add(candidate)
            output.append(candidate)
    return output


def _prediction_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.linalg.norm(prediction - target, axis=1)
    squared = np.sum((prediction - target) ** 2, axis=1)
    tail = per_run_tail_metrics(error, np.zeros(len(error), dtype=int)).iloc[0]
    return {
        "rmse": float(np.sqrt(np.mean(squared))),
        "mae_vector": float(np.mean(error)),
        "q90": float(tail["q90"]),
        "q95": float(tail["q95"]),
        "q99": float(tail["q99"]),
        "cvar95": float(tail["cvar95"]),
        "cvar99": float(tail["cvar99"]),
        "large_error_rate": float(tail["large_error_rate"]),
    }


def _site_online_weights(
    bundle: Bundle,
    graph: np.ndarray,
    fallback: np.ndarray,
    *,
    learning_rate: float,
    fixed_share: float,
) -> tuple[np.ndarray, np.ndarray]:
    graph_loss = np.sum((graph - bundle.target) ** 2, axis=1)
    fallback_loss = np.sum((fallback - bundle.target) ** 2, axis=1)
    unique_frames = np.unique(bundle.origin_time_s)
    frame_loss = np.array(
        [
            [
                np.mean(graph_loss[bundle.origin_time_s == frame]),
                np.mean(fallback_loss[bundle.origin_time_s == frame]),
            ]
            for frame in unique_frames
        ]
    )
    result = causal_delayed_two_expert_weights(
        frame_loss,
        unique_frames,
        frames_per_second=1.0,
        label_delay_seconds=0.4,
        learning_rate=learning_rate,
        fixed_share=fixed_share,
    )
    lookup = {frame: index for index, frame in enumerate(unique_frames)}
    index = np.array([lookup[frame] for frame in bundle.origin_time_s])
    return result.weights[index, 0], result.updates_applied[index]


def _pedestrian_online_weights(
    bundle: Bundle,
    graph: np.ndarray,
    fallback: np.ndarray,
    *,
    learning_rate: float,
    fixed_share: float,
) -> tuple[np.ndarray, np.ndarray]:
    losses = np.column_stack(
        (
            np.sum((graph - bundle.target) ** 2, axis=1),
            np.sum((fallback - bundle.target) ** 2, axis=1),
        )
    )
    result = causal_delayed_two_expert_weights(
        losses,
        bundle.origin_time_s,
        frames_per_second=1.0,
        label_delay_seconds=0.4,
        learning_rate=learning_rate,
        fixed_share=fixed_share,
        stream_ids=bundle.pedestrian_id,
    )
    return result.weights[:, 0], result.updates_applied


def choose_online_rule(
    oof: list[Bundle], calibration: DrawCalibration, config: dict
) -> tuple[str, float, float, pd.DataFrame]:
    rows: list[dict] = []
    cap = float(config["experts"]["fallback_prediction_speed_cap_mps"])
    for scope in ("site", "pedestrian"):
        for learning_rate in config["controls"]["online_experts"]["learning_rates"]:
            for fixed_share in config["controls"]["online_experts"]["fixed_share_rates"]:
                run_values = []
                for bundle in oof:
                    _, graph, fallback = selected_predictions(bundle, calibration.indices, cap)
                    function = (
                        _site_online_weights if scope == "site" else _pedestrian_online_weights
                    )
                    weight, _ = function(
                        bundle,
                        graph,
                        fallback,
                        learning_rate=float(learning_rate),
                        fixed_share=float(fixed_share),
                    )
                    prediction = weight[:, None] * graph + (1.0 - weight[:, None]) * fallback
                    run_values.append(_prediction_metrics(bundle.target, prediction)["rmse"])
                rows.append(
                    {
                        "scope": scope,
                        "learning_rate": float(learning_rate),
                        "fixed_share": float(fixed_share),
                        "oof_run_macro_rmse": float(np.mean(run_values)),
                    }
                )
    table = pd.DataFrame(rows).sort_values(
        ["oof_run_macro_rmse", "scope", "learning_rate", "fixed_share"]
    )
    best = table.iloc[0]
    return (
        str(best["scope"]),
        float(best["learning_rate"]),
        float(best["fixed_share"]),
        table,
    )


def _optimal_graph_weight(
    graph: np.ndarray, fallback: np.ndarray, target: np.ndarray
) -> np.ndarray:
    difference = graph - fallback
    denominator = np.sum(difference**2, axis=1)
    numerator = np.sum(difference * (target - fallback), axis=1)
    return np.clip(
        np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, 0.5),
            where=denominator > 1e-12,
        ),
        0.0,
        1.0,
    )


def fit_learned_gate(
    oof: list[Bundle], calibration: DrawCalibration, config: dict
) -> HistGradientBoostingRegressor:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    cap = float(config["experts"]["fallback_prediction_speed_cap_mps"])
    window = float(config["guard"]["trailing_window_seconds"])
    for bundle in oof:
        members, graph, fallback = selected_predictions(bundle, calibration.indices, cap)
        current = signals(bundle, members, graph, fallback, window)
        features.append(
            np.column_stack(
                (
                    current["spread"],
                    current["disagreement"],
                    current["mahalanobis"],
                    current["speed"],
                    bundle.neighbour_count,
                    bundle.occupancy,
                )
            )
        )
        targets.append(_optimal_graph_weight(graph, fallback, bundle.target))
    model = HistGradientBoostingRegressor(
        max_iter=150,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        learning_rate=0.05,
        l2_regularization=2.0,
        random_state=int(config["replication"]["member_seed_base"]),
    )
    return model.fit(np.concatenate(features), np.concatenate(targets))


def _partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    ranked = [rankdata(value) for value in (x, y, control)]
    design = np.column_stack((np.ones(len(x)), ranked[2]))
    x_residual = ranked[0] - design @ np.linalg.lstsq(design, ranked[0], rcond=None)[0]
    y_residual = ranked[1] - design @ np.linalg.lstsq(design, ranked[1], rcond=None)[0]
    return float(pearsonr(x_residual, y_residual).statistic)


def _evaluate_primary_draw(
    oof: list[Bundle],
    tests: list[Bundle],
    calibration: DrawCalibration,
    config: dict,
    output: Path,
) -> None:
    cap = float(config["experts"]["fallback_prediction_speed_cap_mps"])
    window = float(config["guard"]["trailing_window_seconds"])
    learned_gate = fit_learned_gate(oof, calibration, config)
    online_scope, online_eta, online_share, online_grid = choose_online_rule(
        oof, calibration, config
    )
    online_grid.to_csv(output / "referee3_online_calibration.csv", index=False)
    metric_rows: list[dict] = []
    forecast_rows: list[pd.DataFrame] = []
    detection_rows: list[dict] = []
    risk_rows: list[dict] = []
    stratum_rows: list[dict] = []
    decomposition_rows: list[dict] = []
    targeting_rows: list[dict] = []
    correlation_rows: list[dict] = []
    firing_rows: list[dict] = []
    permutation_count = int(config["controls"]["within_run_gate_permutations"])
    permutation_seed = int(config["controls"]["permutation_seed"])

    for bundle in oof + tests:
        members, graph, fallback = selected_predictions(bundle, calibration.indices, cap)
        current = signals(bundle, members, graph, fallback, window)
        weights = {
            "GEL-Ped": one_sided_gate_weights(
                current["spread"],
                threshold=calibration.tau,
                scale=calibration.scale,
                kind=config["guard"]["primary_shape"],
            ),
            "disagreement guard": one_sided_gate_weights(
                current["disagreement"],
                threshold=calibration.disagreement_tau,
                scale=calibration.disagreement_scale,
            ),
            "Mahalanobis guard": one_sided_gate_weights(
                current["mahalanobis"],
                threshold=calibration.mahalanobis_tau,
                scale=calibration.mahalanobis_scale,
            ),
            "speed guard": one_sided_gate_weights(
                current["speed"],
                threshold=calibration.speed_tau,
                scale=calibration.speed_scale,
            ),
        }
        learned_features = np.column_stack(
            (
                current["spread"],
                current["disagreement"],
                current["mahalanobis"],
                current["speed"],
                bundle.neighbour_count,
                bundle.occupancy,
            )
        )
        weights["learned matched gate"] = np.clip(
            learned_gate.predict(learned_features), 0.0, 1.0
        )
        online_function = (
            _site_online_weights if online_scope == "site" else _pedestrian_online_weights
        )
        online_weight, online_updates = online_function(
            bundle,
            graph,
            fallback,
            learning_rate=online_eta,
            fixed_share=online_share,
        )
        weights["delayed-error online gate"] = online_weight
        predictions = {
            "graph ensemble": graph,
            "feature-based HGB fallback": fallback,
            "calibration-fixed blend": calibration.fixed_graph_weight * graph
            + (1.0 - calibration.fixed_graph_weight) * fallback,
        }
        for name, weight in weights.items():
            predictions[name] = weight[:, None] * graph + (1.0 - weight[:, None]) * fallback
        graph_error = np.linalg.norm(graph - bundle.target, axis=1)
        fallback_error = np.linalg.norm(fallback - bundle.target, axis=1)
        oracle = np.where(
            (graph_error <= fallback_error)[:, None], graph, fallback
        )
        predictions["future-label oracle"] = oracle
        mean_g = float(np.mean(weights["GEL-Ped"]))
        predictions["mean-g fixed diagnostic"] = mean_g * graph + (1.0 - mean_g) * fallback

        for name, prediction in predictions.items():
            metric_rows.append(
                {
                    "role": bundle.role,
                    "run": bundle.run,
                    "model": name,
                    "n": len(bundle.target),
                    **_prediction_metrics(bundle.target, prediction),
                }
            )
        firing_rows.append(
            {
                "role": bundle.role,
                "run": bundle.run,
                "n": len(bundle.target),
                "fraction_spread_above_tau": float(
                    np.mean(current["spread"] > calibration.tau)
                ),
                "mean_graph_weight": mean_g,
                "q10_graph_weight": float(np.quantile(weights["GEL-Ped"], 0.1)),
                "tau": calibration.tau,
                "scale": calibration.scale,
            }
        )

        threshold = np.quantile(graph_error, 0.95)
        high_error = (graph_error >= threshold).astype(int)
        for signal_name, signal in {
            **current,
            "neighbour density": bundle.neighbour_count / (np.pi * 3.0**2),
        }.items():
            values = binary_detection_metrics(high_error, signal)
            detection_rows.append(
                {
                    "role": bundle.role,
                    "run": bundle.run,
                    "signal": signal_name,
                    **values,
                }
            )
            curve = risk_retention_curve(graph_error**2, signal)
            risk_rows.append(
                {
                    "role": bundle.role,
                    "run": bundle.run,
                    "signal": signal_name,
                    "aurc_squared_error": curve.aurc,
                }
            )
        correlation_rows.append(
            {
                "role": bundle.role,
                "run": bundle.run,
                "partial_spearman_spread_error_given_speed": _partial_spearman(
                    current["spread"], graph_error, current["speed"]
                ),
            }
        )

        normalized = (current["spread"] - calibration.tau) / calibration.scale
        bins = np.digitize(normalized, [0.0, 1.0, 2.0])
        labels = ("inside", "0--1 IQR", "1--2 IQR", ">2 IQR")
        primary_permuted = permute_weights_within_runs(
            weights["GEL-Ped"], np.repeat(bundle.run, len(bundle.target)), seed=permutation_seed
        )
        stratum_predictions = {
            "graph ensemble": graph,
            "GEL-Ped": predictions["GEL-Ped"],
            "feature-based HGB fallback": fallback,
            "calibration-fixed blend": predictions["calibration-fixed blend"],
            "permuted-g diagnostic": primary_permuted[:, None] * graph
            + (1.0 - primary_permuted[:, None]) * fallback,
            "future-label oracle": oracle,
        }
        for code, label in enumerate(labels):
            mask = bins == code
            if not np.any(mask):
                continue
            for name, prediction in stratum_predictions.items():
                stratum_rows.append(
                    {
                        "role": bundle.role,
                        "run": bundle.run,
                        "stratum": label,
                        "model": name,
                        "n": int(np.sum(mask)),
                        "share": float(np.mean(mask)),
                        "rmse": float(
                            np.sqrt(
                                np.mean(
                                    np.sum(
                                        (prediction[mask] - bundle.target[mask]) ** 2,
                                        axis=1,
                                    )
                                )
                            )
                        ),
                    }
                )
        decomposition = ensemble_error_decomposition(members, bundle.target)
        for code, label in enumerate(labels):
            mask = bins == code
            if not np.any(mask):
                continue
            decomposition_rows.append(
                {
                    "role": bundle.role,
                    "run": bundle.run,
                    "stratum": label,
                    "n": int(np.sum(mask)),
                    "shared_mse": float(
                        np.mean(decomposition.shared_squared_error[mask])
                    ),
                    "between_member_variance": float(
                        np.mean(decomposition.between_member_variance[mask])
                    ),
                    "member_mse": float(
                        np.mean(decomposition.mean_member_squared_error[mask])
                    ),
                }
            )

        if bundle.role != "calibration":
            observed_rmse = _prediction_metrics(bundle.target, predictions["GEL-Ped"])[
                "rmse"
            ]
            for permutation in range(permutation_count):
                permuted = permute_weights_within_runs(
                    weights["GEL-Ped"],
                    np.repeat(bundle.run, len(bundle.target)),
                    seed=permutation_seed + 1 + permutation,
                )
                prediction = permuted[:, None] * graph + (1.0 - permuted[:, None]) * fallback
                targeting_rows.append(
                    {
                        "role": bundle.role,
                        "run": bundle.run,
                        "permutation": permutation,
                        "observed_gel_rmse": observed_rmse,
                        "permuted_rmse": _prediction_metrics(bundle.target, prediction)["rmse"],
                    }
                )

        if bundle.role != "calibration":
            forecast_rows.append(
                pd.DataFrame(
                    {
                        "role": bundle.role,
                        "run": bundle.run,
                        "frame": bundle.frame,
                        "origin_time_s": bundle.origin_time_s,
                        "pedestrian_id": bundle.pedestrian_id,
                        "target_x": bundle.target[:, 0],
                        "target_y": bundle.target[:, 1],
                        "graph_x": graph[:, 0],
                        "graph_y": graph[:, 1],
                        "fallback_x": fallback[:, 0],
                        "fallback_y": fallback[:, 1],
                        "gel_x": predictions["GEL-Ped"][:, 0],
                        "gel_y": predictions["GEL-Ped"][:, 1],
                        "spread": current["spread"],
                        "graph_weight": weights["GEL-Ped"],
                        "mahalanobis": current["mahalanobis"],
                        "graph_fallback_disagreement": current["disagreement"],
                        "speed_mps": current["speed"],
                        "neighbour_count": bundle.neighbour_count,
                        "occupancy": bundle.occupancy,
                        "online_updates": online_updates,
                    }
                )
            )

    pd.DataFrame(metric_rows).to_csv(output / "referee3_primary_run_metrics.csv", index=False)
    pd.DataFrame(detection_rows).to_csv(output / "referee3_detection_metrics.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(output / "referee3_risk_retention.csv", index=False)
    pd.DataFrame(stratum_rows).to_csv(output / "referee3_targeting_by_stratum.csv", index=False)
    pd.DataFrame(decomposition_rows).to_csv(output / "referee3_error_decomposition.csv", index=False)
    pd.DataFrame(targeting_rows).to_csv(output / "referee3_permuted_gate.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(output / "referee3_speed_partial_correlation.csv", index=False)
    pd.DataFrame(firing_rows).to_csv(output / "referee3_firing_rates.csv", index=False)
    pd.concat(forecast_rows, ignore_index=True).to_csv(
        output / "referee3_forecasts_primary.csv.gz", index=False, compression="gzip"
    )
    (output / "referee3_online_selection.json").write_text(
        json.dumps(
            {
                "scope": online_scope,
                "learning_rate": online_eta,
                "fixed_share": online_share,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _draw_analysis(
    oof: list[Bundle], tests: list[Bundle], config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool_size = min(bundle.members_raw.shape[0] for bundle in oof + tests)
    seed = int(config["replication"]["ensemble_draw_seed"])
    cap = float(config["experts"]["fallback_prediction_speed_cap_mps"])
    window = float(config["guard"]["trailing_window_seconds"])
    metric_rows: list[dict] = []
    calibration_rows: list[dict] = []
    for size in config["replication"]["ensemble_sizes"]:
        count = int(config["replication"]["draws_per_size"])
        draws = draw_indices(
            pool_size,
            int(size),
            count,
            seed,
            include_primary=int(size) == int(config["replication"]["primary_ensemble_size"]),
        )
        for draw_index, indices in enumerate(draws):
            calibration = calibrate_draw(oof, indices, config)
            calibration_rows.append(
                {
                    "ensemble_size": size,
                    "draw": draw_index,
                    "member_indices": ";".join(map(str, indices)),
                    "tau": calibration.tau,
                    "scale": calibration.scale,
                    "fixed_graph_weight": calibration.fixed_graph_weight,
                }
            )
            for bundle in tests:
                members, graph, fallback = selected_predictions(bundle, indices, cap)
                current = signals(bundle, members, graph, fallback, window)
                weight = one_sided_gate_weights(
                    current["spread"],
                    threshold=calibration.tau,
                    scale=calibration.scale,
                    kind=config["guard"]["primary_shape"],
                )
                guard = weight[:, None] * graph + (1.0 - weight[:, None]) * fallback
                graph_metrics = _prediction_metrics(bundle.target, graph)
                guard_metrics = _prediction_metrics(bundle.target, guard)
                member_rmse = np.sqrt(
                    np.mean(
                        np.sum((members - bundle.target[None, :, :]) ** 2, axis=2),
                        axis=1,
                    )
                )
                median = float(np.median(member_rmse))
                mad = float(np.median(np.abs(member_rmse - median)))
                scaled_mad = 1.4826 * mad
                divergent = int(np.sum(member_rmse > median + 3.0 * max(scaled_mad, 1e-12)))
                metric_rows.append(
                    {
                        "role": bundle.role,
                        "run": bundle.run,
                        "ensemble_size": size,
                        "draw": draw_index,
                        "member_indices": ";".join(map(str, indices)),
                        "graph_rmse": graph_metrics["rmse"],
                        "guard_rmse": guard_metrics["rmse"],
                        "guard_minus_graph_rmse": guard_metrics["rmse"]
                        - graph_metrics["rmse"],
                        "graph_q95": graph_metrics["q95"],
                        "guard_q95": guard_metrics["q95"],
                        "guard_minus_graph_q95": guard_metrics["q95"]
                        - graph_metrics["q95"],
                        "graph_cvar95": graph_metrics["cvar95"],
                        "guard_cvar95": guard_metrics["cvar95"],
                        "guard_minus_graph_cvar95": guard_metrics["cvar95"]
                        - graph_metrics["cvar95"],
                        "graph_large_error_rate": graph_metrics["large_error_rate"],
                        "guard_large_error_rate": guard_metrics["large_error_rate"],
                        "mean_graph_weight": float(np.mean(weight)),
                        "firing_rate": float(np.mean(current["spread"] > calibration.tau)),
                        "divergent_members": divergent,
                    }
                )
            print(f"analysed M={size}, draw {draw_index + 1}/{len(draws)}", flush=True)
    return pd.DataFrame(metric_rows), pd.DataFrame(calibration_rows)


def _calibration_bootstrap(
    oof: list[Bundle], calibration: DrawCalibration, config: dict
) -> pd.DataFrame:
    cap = float(config["experts"]["fallback_prediction_speed_cap_mps"])
    window = float(config["guard"]["trailing_window_seconds"])
    by_run = []
    for bundle in oof:
        members, graph, fallback = selected_predictions(bundle, calibration.indices, cap)
        by_run.append(signals(bundle, members, graph, fallback, window)["spread"])
    generator = np.random.default_rng(int(config["endpoints"]["bootstrap_seed"]))
    rows = []
    for replicate in range(int(config["endpoints"]["bootstrap_replicates"])):
        selected = generator.integers(0, len(by_run), size=len(by_run))
        values = np.concatenate([by_run[index] for index in selected])
        tau, scale = _threshold_scale(values)
        rows.append({"replicate": replicate, "tau": tau, "scale": scale})
    return pd.DataFrame(rows)


def _window_shape_sensitivity(
    oof: list[Bundle], tests: list[Bundle], indices: tuple[int, ...], config: dict
) -> pd.DataFrame:
    cap = float(config["experts"]["fallback_prediction_speed_cap_mps"])
    rows: list[dict] = []
    windows = {1: 0.0, 3: 0.8, 5: 1.6, 10: 3.6}
    for origins, window in windows.items():
        score_parts = []
        for bundle in oof:
            members, graph, fallback = selected_predictions(bundle, indices, cap)
            score_parts.append(signals(bundle, members, graph, fallback, window)["spread"])
        tau, scale = _threshold_scale(np.concatenate(score_parts))
        for shape in config["guard"]["shape_sensitivities"] + [
            config["guard"]["primary_shape"]
        ]:
            for bundle in tests:
                members, graph, fallback = selected_predictions(bundle, indices, cap)
                score = signals(bundle, members, graph, fallback, window)["spread"]
                weight = one_sided_gate_weights(
                    score, threshold=tau, scale=scale, kind=shape
                )
                prediction = weight[:, None] * graph + (1.0 - weight[:, None]) * fallback
                rows.append(
                    {
                        "role": bundle.role,
                        "run": bundle.run,
                        "origins": origins,
                        "window_seconds": window,
                        "shape": shape,
                        "tau": tau,
                        "scale": scale,
                        "rmse": _prediction_metrics(bundle.target, prediction)["rmse"],
                        "mean_graph_weight": float(np.mean(weight)),
                    }
                )
    return pd.DataFrame(rows)


def _equivalence(metrics: pd.DataFrame, margin: float) -> pd.DataFrame:
    rows = []
    primary = metrics[metrics["ensemble_size"].eq(5) & metrics["draw"].eq(0)]
    for role, group in primary.groupby("role", sort=False):
        difference = group["guard_minus_graph_rmse"].to_numpy(float)
        lower = ttest_1samp(difference, -margin, alternative="greater")
        upper = ttest_1samp(difference, margin, alternative="less")
        rows.append(
            {
                "role": role,
                "n_runs": len(difference),
                "mean_difference": float(np.mean(difference)),
                "margin": margin,
                "p_lower": float(lower.pvalue),
                "p_upper": float(upper.pvalue),
                "tost_p": float(max(lower.pvalue, upper.pvalue)),
                "equivalent_at_0.05": bool(max(lower.pvalue, upper.pvalue) < 0.05),
            }
        )
    return pd.DataFrame(rows)


def analyse(args: argparse.Namespace) -> None:
    project = project_root()
    config = json.loads((project / "configs/referee3_lock.json").read_text())
    output = project / "data/processed"
    bundle_dir = output / "referee3_predictions"
    oof = [load_bundle(path) for path in sorted(bundle_dir.glob("oof_*.npz"))]
    tests = [load_bundle(path) for path in sorted(bundle_dir.glob("test_*.npz"))]
    if len(oof) != 7 or not tests:
        raise RuntimeError("prediction bundles are incomplete; run referee3_generate_predictions.py")
    pool_size = min(bundle.members_raw.shape[0] for bundle in oof + tests)
    required = max(config["replication"]["ensemble_sizes"])
    if pool_size < required:
        raise RuntimeError(f"member pool {pool_size} is smaller than required M={required}")
    primary = tuple(int(value) for value in config["replication"]["primary_member_indices"])
    calibration = calibrate_draw(oof, primary, config)
    _evaluate_primary_draw(oof, tests, calibration, config, output)
    draw_metrics, draw_calibration = _draw_analysis(oof, tests, config)
    draw_metrics.to_csv(output / "referee3_draw_run_metrics.csv", index=False)
    draw_calibration.to_csv(output / "referee3_draw_calibration.csv", index=False)
    _calibration_bootstrap(oof, calibration, config).to_csv(
        output / "referee3_calibration_bootstrap.csv", index=False
    )
    _window_shape_sensitivity(oof, tests, primary, config).to_csv(
        output / "referee3_window_shape_sensitivity.csv", index=False
    )
    _equivalence(
        draw_metrics, float(config["endpoints"]["equivalence_absolute_margin_mps"])
    ).to_csv(output / "referee3_equivalence.csv", index=False)
    (output / "referee3_primary_calibration.json").write_text(
        json.dumps(calibration.__dict__, indent=2), encoding="utf-8"
    )
    print("referee-3 analysis complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    analyse(parser.parse_args())


if __name__ == "__main__":
    main()
