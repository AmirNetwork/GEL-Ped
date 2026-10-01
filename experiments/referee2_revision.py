# Author: Amir Ghorbani
"""Confirmatory analyses requested in the second referee report.

The script implements the specification frozen in ``configs/referee2_lock.json``.
All guard parameters are estimated from complete-run out-of-fold corridor
predictions before any test-regime result is evaluated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr

from pedgeom.benchmarks import (
    EnsembleRegressor,
    fit_gradient_boosted_residual,
    invariant_design,
)
from pedgeom.calibration import VelocitySamples, build_velocity_samples, velocity_metrics
from pedgeom.datasets import load_julich_trajectory
from pedgeom.external import build_eth_ucy_samples
from pedgeom.graph_baseline import fit_graph_interaction_network

from causal_anchor_revision import (
    anchor,
    build_julich,
    conformal_quantile,
    fit_hgb,
    load_inputs,
    project_root,
    settings,
)


MODEL_NAMES = (
    "constant velocity",
    "single graph network",
    "five-member graph ensemble",
    "coordinate-median graph ensemble",
    "trimmed graph ensemble",
    "anchored coarse HGB",
    "prospective HGB",
    "global disagreement guard",
    "local geometry guard",
    "local coarse-HGB guard",
    "local CV guard",
    "graph-ensemble-dispersion guard",
    "Mahalanobis OOD guard",
    "learned mixture-of-experts gate",
    "GEL-Ped reliability guard",
    "GEL-Ped instability guard",
)


@dataclass(frozen=True)
class GateCalibration:
    nominal_weight: float
    disagreement_threshold: float
    disagreement_scale: float
    epistemic_threshold: float
    epistemic_scale: float
    mahalanobis_threshold: float
    mahalanobis_scale: float


@dataclass
class BaseModels:
    graph_members: tuple[object, ...]
    graph_ensemble: EnsembleRegressor
    geometry: object
    coarse: object
    constant_velocity: object


def _rolling_by_pedestrian(
    values: np.ndarray, samples: VelocitySamples, window: int = 5
) -> np.ndarray:
    """Past-only rolling mean for each pedestrian independently."""

    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    pedestrian_id = np.asarray(samples.pedestrian_id)
    frame = np.asarray(samples.frame)
    for identity in np.unique(pedestrian_id):
        indices = np.flatnonzero(pedestrian_id == identity)
        indices = indices[np.argsort(frame[indices], kind="stable")]
        ordered = values[indices]
        cumulative = np.concatenate(([0.0], np.cumsum(ordered)))
        for position, index in enumerate(indices):
            start = max(0, position - window + 1)
            result[index] = (cumulative[position + 1] - cumulative[start]) / (
                position - start + 1
            )
    return result


def _rolling_scene_mean(
    values: np.ndarray, samples: VelocitySamples, window: int = 5
) -> np.ndarray:
    """Past-only scene-wide score used by the original guard."""

    frame = np.asarray(samples.frame)
    unique = np.unique(frame)
    frame_values = np.array([np.mean(values[frame == item]) for item in unique])
    rolling = np.empty_like(frame_values)
    for index in range(len(unique)):
        start = max(0, index - window + 1)
        rolling[index] = np.mean(frame_values[start : index + 1])
    lookup = dict(zip(unique, rolling, strict=True))
    return np.array([lookup[item] for item in frame], dtype=float)


def _gate_weight(
    score: np.ndarray, threshold: float, scale: float, nominal: float
) -> np.ndarray:
    excess = np.maximum((score - threshold) / max(scale, 1e-9), 0.0)
    return nominal * np.exp(-excess)


def _combined_weight(
    disagreement: np.ndarray,
    epistemic: np.ndarray,
    calibration: GateCalibration,
) -> np.ndarray:
    disagreement_excess = np.maximum(
        (disagreement - calibration.disagreement_threshold)
        / max(calibration.disagreement_scale, 1e-9),
        0.0,
    )
    epistemic_excess = np.maximum(
        (epistemic - calibration.epistemic_threshold)
        / max(calibration.epistemic_scale, 1e-9),
        0.0,
    )
    return calibration.nominal_weight * np.exp(
        -np.maximum(disagreement_excess, epistemic_excess)
    )


def _blend(first: np.ndarray, second: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return weight[:, None] * first + (1.0 - weight[:, None]) * second


def _optimal_weight(
    first: np.ndarray, second: np.ndarray, target: np.ndarray
) -> np.ndarray:
    direction = first - second
    denominator = np.sum(direction**2, axis=1)
    numerator = np.sum((target - second) * direction, axis=1)
    return np.clip(
        np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, 0.5),
            where=denominator > 1e-10,
        ),
        0.0,
        1.0,
    )


def _fit_mahalanobis(batches: list[VelocitySamples]) -> tuple[np.ndarray, np.ndarray]:
    design = np.concatenate([invariant_design(batch) for batch in batches])
    estimator = LedoitWolf().fit(design)
    return estimator.location_, estimator.precision_


def _mahalanobis_score(
    samples: VelocitySamples, location: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    centred = invariant_design(samples) - location
    return np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", centred, precision, centred), 0.0))


def _fit_models(
    batches: list[VelocitySamples],
    seeds: list[int],
    epochs: int,
    hgb: dict,
    prospective: dict,
) -> BaseModels:
    members = tuple(
        fit_graph_interaction_network(
            batches,
            seed=seed,
            epochs=epochs,
            residual=True,
        )
        for seed in seeds
    )
    geometry = fit_hgb(
        batches,
        prospective,
        hgb,
        seeds[0],
        residual=True,
    )
    coarse = fit_gradient_boosted_residual(
        batches,
        anchor(),
        seed=seeds[0],
        **hgb,
    )
    return BaseModels(
        graph_members=members,
        graph_ensemble=EnsembleRegressor(members),
        geometry=geometry,
        coarse=coarse,
        constant_velocity=anchor(),
    )


def _base_predictions(
    models: BaseModels, samples: VelocitySamples
) -> dict[str, np.ndarray]:
    member = models.graph_ensemble.member_predictions(samples, speed_cap=100.0)
    ensemble = np.mean(member, axis=0)
    median = np.median(member, axis=0)
    distance_from_median = np.linalg.norm(member - median[None, :, :], axis=2)
    retained = np.argsort(distance_from_median, axis=0, kind="stable")[:3]
    trimmed = np.take_along_axis(member, retained[:, :, None], axis=0).mean(axis=0)
    return {
        "members": member,
        "single": member[0],
        "ensemble": ensemble,
        "median": median,
        "trimmed": trimmed,
        "epistemic": np.sqrt(
            np.mean(np.sum((member - ensemble[None, :, :]) ** 2, axis=2), axis=0)
        ),
        "geometry": models.geometry.predict(samples, speed_cap=100.0),
        "coarse": models.coarse.predict(samples, speed_cap=100.0),
        "cv": models.constant_velocity.predict(samples, speed_cap=100.0),
    }


def _rows_from_fold(
    validation: VelocitySamples,
    predictions: dict[str, np.ndarray],
    location: np.ndarray,
    precision: np.ndarray,
) -> pd.DataFrame:
    ensemble = predictions["ensemble"]
    geometry = predictions["geometry"]
    coarse = predictions["coarse"]
    constant = predictions["cv"]
    raw_geometry = np.linalg.norm(ensemble - geometry, axis=1)
    raw_coarse = np.linalg.norm(ensemble - coarse, axis=1)
    raw_cv = np.linalg.norm(ensemble - constant, axis=1)
    epistemic = predictions["epistemic"]
    data = pd.DataFrame(
        {
            "run": validation.run,
            "frame": validation.frame,
            "pedestrian_id": validation.pedestrian_id,
            "target_x": validation.target[:, 0],
            "target_y": validation.target[:, 1],
            "graph_x": ensemble[:, 0],
            "graph_y": ensemble[:, 1],
            "geometry_x": geometry[:, 0],
            "geometry_y": geometry[:, 1],
            "coarse_x": coarse[:, 0],
            "coarse_y": coarse[:, 1],
            "cv_x": constant[:, 0],
            "cv_y": constant[:, 1],
            "local_geometry_disagreement": _rolling_by_pedestrian(
                raw_geometry, validation
            ),
            "global_geometry_disagreement": _rolling_scene_mean(
                raw_geometry, validation
            ),
            "local_coarse_disagreement": _rolling_by_pedestrian(
                raw_coarse, validation
            ),
            "local_cv_disagreement": _rolling_by_pedestrian(raw_cv, validation),
            "local_epistemic": _rolling_by_pedestrian(epistemic, validation),
            "mahalanobis": _mahalanobis_score(validation, location, precision),
            "neighbour_count": validation.neighbour_count,
            "occupancy": validation.occupancy,
            "speed_mps": np.linalg.norm(validation.features[:, 0], axis=1),
        }
    )
    target = validation.target
    data["optimal_geometry_weight"] = _optimal_weight(ensemble, geometry, target)
    return data


def make_oof(
    training: list[VelocitySamples],
    seeds: list[int],
    epochs: int,
    hgb: dict,
    prospective: dict,
    checkpoint: Path,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for held_out, validation in enumerate(training):
        fitting = [batch for index, batch in enumerate(training) if index != held_out]
        fold_seeds = [seed + 10000 * held_out for seed in seeds]
        models = _fit_models(fitting, fold_seeds, epochs, hgb, prospective)
        location, precision = _fit_mahalanobis(fitting)
        rows.append(
            _rows_from_fold(
                validation,
                _base_predictions(models, validation),
                location,
                precision,
            )
        )
        pd.concat(rows, ignore_index=True).to_csv(checkpoint, index=False)
        print(
            f"OOF calibration fold {held_out + 1}/{len(training)}: {validation.run}",
            flush=True,
        )
    return pd.concat(rows, ignore_index=True)


def _matrix(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    return frame[[f"{prefix}_x", f"{prefix}_y"]].to_numpy(float)


def _run_balanced_rmse(
    frame: pd.DataFrame, prediction: np.ndarray
) -> float:
    error = np.sum((prediction - _matrix(frame, "target")) ** 2, axis=1)
    run = frame["run"].to_numpy()
    return float(np.mean([np.sqrt(np.mean(error[run == item])) for item in np.unique(run)]))


def _fixed_weight(frame: pd.DataFrame, fallback: str) -> float:
    graph = _matrix(frame, "graph")
    second = _matrix(frame, fallback)
    candidates = np.linspace(0.0, 1.0, 21)
    losses = [
        _run_balanced_rmse(frame, weight * graph + (1.0 - weight) * second)
        for weight in candidates
    ]
    return float(candidates[int(np.argmin(losses))])


def _threshold_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    first, third = np.quantile(values, (0.25, 0.75))
    return conformal_quantile(values, 0.90), max(float(third - first), 1e-6)


def calibrate(oof: pd.DataFrame) -> tuple[dict[str, GateCalibration], object]:
    epistemic_threshold, epistemic_scale = _threshold_scale(
        oof["local_epistemic"].to_numpy()
    )
    mahalanobis_threshold, mahalanobis_scale = _threshold_scale(
        oof["mahalanobis"].to_numpy()
    )
    calibrations: dict[str, GateCalibration] = {}
    for fallback in ("geometry", "coarse", "cv"):
        disagreement_threshold, disagreement_scale = _threshold_scale(
            oof[f"local_{fallback}_disagreement"].to_numpy()
        )
        calibrations[fallback] = GateCalibration(
            nominal_weight=_fixed_weight(oof, fallback),
            disagreement_threshold=disagreement_threshold,
            disagreement_scale=disagreement_scale,
            epistemic_threshold=epistemic_threshold,
            epistemic_scale=epistemic_scale,
            mahalanobis_threshold=mahalanobis_threshold,
            mahalanobis_scale=mahalanobis_scale,
        )
    feature_names = [
        "local_geometry_disagreement",
        "local_epistemic",
        "mahalanobis",
        "neighbour_count",
        "occupancy",
        "speed_mps",
    ]
    learned_gate = HistGradientBoostingRegressor(
        max_iter=150,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        learning_rate=0.05,
        l2_regularization=2.0,
        random_state=20260722,
    ).fit(oof[feature_names], oof["optimal_geometry_weight"])
    learned_gate.feature_names_ = feature_names
    return calibrations, learned_gate


def calibration_payload(calibrations: dict[str, GateCalibration]) -> dict:
    return {
        name: {
            "nominal_weight": value.nominal_weight,
            "disagreement_threshold": value.disagreement_threshold,
            "disagreement_scale": value.disagreement_scale,
            "epistemic_threshold": value.epistemic_threshold,
            "epistemic_scale": value.epistemic_scale,
            "mahalanobis_threshold": value.mahalanobis_threshold,
            "mahalanobis_scale": value.mahalanobis_scale,
        }
        for name, value in calibrations.items()
    }


def _all_predictions(
    samples: VelocitySamples,
    base: dict[str, np.ndarray],
    calibrations: dict[str, GateCalibration],
    learned_gate: object,
    location: np.ndarray,
    precision: np.ndarray,
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    graph = base["ensemble"]
    geometry = base["geometry"]
    coarse = base["coarse"]
    constant = base["cv"]
    raw_geometry = np.linalg.norm(graph - geometry, axis=1)
    local_geometry = _rolling_by_pedestrian(raw_geometry, samples)
    global_geometry = _rolling_scene_mean(raw_geometry, samples)
    local_coarse = _rolling_by_pedestrian(
        np.linalg.norm(graph - coarse, axis=1), samples
    )
    local_cv = _rolling_by_pedestrian(
        np.linalg.norm(graph - constant, axis=1), samples
    )
    local_epistemic = _rolling_by_pedestrian(base["epistemic"], samples)
    mahalanobis = _mahalanobis_score(samples, location, precision)
    geometry_calibration = calibrations["geometry"]
    coarse_calibration = calibrations["coarse"]
    cv_calibration = calibrations["cv"]
    weight_global = _gate_weight(
        global_geometry,
        geometry_calibration.disagreement_threshold,
        geometry_calibration.disagreement_scale,
        geometry_calibration.nominal_weight,
    )
    weight_geometry = _gate_weight(
        local_geometry,
        geometry_calibration.disagreement_threshold,
        geometry_calibration.disagreement_scale,
        geometry_calibration.nominal_weight,
    )
    weight_coarse = _gate_weight(
        local_coarse,
        coarse_calibration.disagreement_threshold,
        coarse_calibration.disagreement_scale,
        coarse_calibration.nominal_weight,
    )
    weight_cv = _gate_weight(
        local_cv,
        cv_calibration.disagreement_threshold,
        cv_calibration.disagreement_scale,
        cv_calibration.nominal_weight,
    )
    weight_epistemic = _gate_weight(
        local_epistemic,
        geometry_calibration.epistemic_threshold,
        geometry_calibration.epistemic_scale,
        geometry_calibration.nominal_weight,
    )
    weight_mahalanobis = _gate_weight(
        mahalanobis,
        geometry_calibration.mahalanobis_threshold,
        geometry_calibration.mahalanobis_scale,
        geometry_calibration.nominal_weight,
    )
    weight_combined = _combined_weight(
        local_geometry, local_epistemic, geometry_calibration
    )
    weight_instability = _gate_weight(
        local_epistemic,
        coarse_calibration.epistemic_threshold,
        coarse_calibration.epistemic_scale,
        1.0,
    )
    learned_features = pd.DataFrame(
        {
            "local_geometry_disagreement": local_geometry,
            "local_epistemic": local_epistemic,
            "mahalanobis": mahalanobis,
            "neighbour_count": samples.neighbour_count,
            "occupancy": samples.occupancy,
            "speed_mps": np.linalg.norm(samples.features[:, 0], axis=1),
        }
    )
    weight_learned = np.clip(learned_gate.predict(learned_features), 0.0, 1.0)
    predictions = {
        "constant velocity": constant,
        "single graph network": base["single"],
        "five-member graph ensemble": graph,
        "coordinate-median graph ensemble": base["median"],
        "trimmed graph ensemble": base["trimmed"],
        "anchored coarse HGB": coarse,
        "prospective HGB": geometry,
        "global disagreement guard": _blend(graph, geometry, weight_global),
        "local geometry guard": _blend(graph, geometry, weight_geometry),
        "local coarse-HGB guard": _blend(graph, coarse, weight_coarse),
        "local CV guard": _blend(graph, constant, weight_cv),
        "graph-ensemble-dispersion guard": _blend(
            graph, geometry, weight_epistemic
        ),
        "Mahalanobis OOD guard": _blend(graph, geometry, weight_mahalanobis),
        "learned mixture-of-experts gate": _blend(
            graph, geometry, weight_learned
        ),
        "GEL-Ped reliability guard": _blend(graph, geometry, weight_combined),
        "GEL-Ped instability guard": _blend(
            graph, coarse, weight_instability
        ),
    }
    weights = {
        "global disagreement guard": weight_global,
        "local geometry guard": weight_geometry,
        "local coarse-HGB guard": weight_coarse,
        "local CV guard": weight_cv,
        "graph-ensemble-dispersion guard": weight_epistemic,
        "Mahalanobis OOD guard": weight_mahalanobis,
        "learned mixture-of-experts gate": weight_learned,
        "GEL-Ped reliability guard": weight_combined,
        "GEL-Ped instability guard": weight_instability,
    }
    signals = {
        "local_epistemic": local_epistemic,
        "local_geometry_disagreement": local_geometry,
        "mahalanobis": mahalanobis,
    }
    return predictions, weights, signals


def evaluate(
    datasets: dict[str, list[VelocitySamples]],
    models: BaseModels,
    calibrations: dict[str, GateCalibration],
    learned_gate: object,
    location: np.ndarray,
    precision: np.ndarray,
    horizon_seconds: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    horizon_seconds = horizon_seconds or {}
    metric_rows: list[dict] = []
    weight_rows: list[dict] = []
    seed_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    for dataset, batches in datasets.items():
        horizon = horizon_seconds.get(dataset, 0.4)
        for batch in batches:
            base = _base_predictions(models, batch)
            predictions, weights, signals = _all_predictions(
                batch,
                base,
                calibrations,
                learned_gate,
                location,
                precision,
            )
            for name, prediction in predictions.items():
                metric_rows.append(
                    {
                        "dataset": dataset,
                        "run": batch.run,
                        "model": name,
                        "horizon_s": horizon,
                        **velocity_metrics(batch.target, prediction, horizon_s=horizon),
                    }
                )
            for name, value in weights.items():
                weight_rows.append(
                    {
                        "dataset": dataset,
                        "run": batch.run,
                        "model": name,
                        "mean_graph_weight": float(np.mean(value)),
                        "q10_graph_weight": float(np.quantile(value, 0.10)),
                        "q90_graph_weight": float(np.quantile(value, 0.90)),
                    }
                )
            for member_index, member_prediction in enumerate(base["members"]):
                seed_rows.append(
                    {
                        "dataset": dataset,
                        "run": batch.run,
                        "member": member_index + 1,
                        **velocity_metrics(
                            batch.target, member_prediction, horizon_s=horizon
                        ),
                    }
                )
            calibration = calibrations["coarse"]
            normalized = (
                signals["local_epistemic"] - calibration.epistemic_threshold
            ) / max(calibration.epistemic_scale, 1e-9)
            bin_code = np.digitize(normalized, [0.0, 1.0, 2.0])
            labels = ("inside", "0--1 IQR", "1--2 IQR", ">2 IQR")
            graph_error = np.sum(
                (predictions["five-member graph ensemble"] - batch.target) ** 2,
                axis=1,
            )
            guard_error = np.sum(
                (predictions["GEL-Ped instability guard"] - batch.target) ** 2,
                axis=1,
            )
            median_error = np.sum(
                (predictions["coordinate-median graph ensemble"] - batch.target)
                ** 2,
                axis=1,
            )
            for code, label in enumerate(labels):
                selected = bin_code == code
                if not np.any(selected):
                    continue
                diagnostic_rows.append(
                    {
                        "dataset": dataset,
                        "run": batch.run,
                        "diagnostic": "epistemic stratum",
                        "stratum": label,
                        "samples": int(np.sum(selected)),
                        "graph_sse": float(np.sum(graph_error[selected])),
                        "guard_sse": float(np.sum(guard_error[selected])),
                        "median_sse": float(np.sum(median_error[selected])),
                        "mean_score": float(
                            np.mean(signals["local_epistemic"][selected])
                        ),
                        "mean_neighbour_count": float(
                            np.mean(batch.neighbour_count[selected])
                        ),
                        "mean_occupancy": float(np.mean(batch.occupancy[selected])),
                        "mean_speed_mps": float(
                            np.mean(np.linalg.norm(batch.features[selected, 0], axis=1))
                        ),
                    }
                )
            graph_advantage = graph_error - guard_error
            for variable_name, variable in {
                "neighbour_count": batch.neighbour_count,
                "occupancy": batch.occupancy,
                "speed_mps": np.linalg.norm(batch.features[:, 0], axis=1),
                "graph_minus_guard_squared_error": graph_advantage,
            }.items():
                correlation = spearmanr(
                    signals["local_epistemic"], variable, nan_policy="omit"
                ).statistic
                diagnostic_rows.append(
                    {
                        "dataset": dataset,
                        "run": batch.run,
                        "diagnostic": "run-level Spearman correlation",
                        "stratum": variable_name,
                        "samples": len(batch.target),
                        "rho": float(correlation),
                    }
                )
        print(f"evaluated {dataset}", flush=True)
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(weight_rows),
        pd.DataFrame(seed_rows),
        pd.DataFrame(diagnostic_rows),
    )


def untouched_tordeux(project: Path, config: dict) -> dict[str, list[VelocitySamples]]:
    root = project / "data/raw/tordeux2017_untouched"
    loader = partial(load_julich_trajectory, fps=16.0)

    def build(path: Path) -> VelocitySamples:
        return build_velocity_samples(
            path,
            interaction_range=config["interaction_range_m"],
            horizon_frames=6,
            history_frames=13,
            frame_stride=6,
            frames_per_second=16.0,
            maximum_samples=4000,
            radius=config["pedestrian_radius_m"],
            wall_range=config["wall_range_m"],
            wall_y_bounds=None,
            neighbour_cutoff=config["neighbour_cutoff_m"],
            maximum_neighbours=8,
            goal_method="prior_velocity",
            cardinal_routes=False,
            speed_outlier_threshold=config["speed_outlier_threshold_mps"],
            loader=loader,
            seed=config["seed"],
        )

    return {
        "untouched unidirectional corridor": [
            build(path)
            for path in sorted((root / "unidirectional_corridor").glob("*.txt"))
        ],
        "untouched bottleneck": [
            build(path) for path in sorted((root / "bottleneck").glob("*.txt"))
        ],
    }


def run_julich(args: argparse.Namespace) -> None:
    project = project_root()
    processed = project / "data/processed"
    config, splits = load_inputs(project)
    _, hgb, prospective = settings(project)
    training_map, datasets = build_julich(
        project, config, splits, history_frames=20
    )
    training = list(training_map.values())
    seeds = [config["seed"] + 997 * index for index in range(5)]
    oof_path = processed / "referee2_oof_calibration.csv"
    if args.reuse_oof and oof_path.exists():
        oof = pd.read_csv(oof_path)
    else:
        oof = make_oof(
            training,
            seeds,
            args.crossfit_epochs,
            hgb,
            prospective,
            oof_path,
        )
    calibrations, learned_gate = calibrate(oof)
    (processed / "referee2_guard_calibration.json").write_text(
        json.dumps(calibration_payload(calibrations), indent=2), encoding="utf-8"
    )
    models = _fit_models(training, seeds, args.final_epochs, hgb, prospective)
    location, precision = _fit_mahalanobis(training)
    untouched = untouched_tordeux(project, config)
    combined = datasets | untouched
    horizon = {name: 0.375 for name in untouched}
    metrics, weights, seed_metrics, diagnostics = evaluate(
        combined,
        models,
        calibrations,
        learned_gate,
        location,
        precision,
        horizon,
    )
    metrics.to_csv(processed / "referee2_julich_metrics.csv", index=False)
    weights.to_csv(processed / "referee2_julich_weights.csv", index=False)
    seed_metrics.to_csv(processed / "referee2_graph_member_metrics.csv", index=False)
    diagnostics.to_csv(processed / "referee2_julich_diagnostics.csv", index=False)


def _external_batches(
    directory: Path,
    *,
    maximum_samples: int,
    seed: int,
) -> list[VelocitySamples]:
    return [
        build_eth_ucy_samples(
            path,
            history_steps=2,
            maximum_samples=maximum_samples,
            seed=seed + index,
        )
        for index, path in enumerate(sorted(directory.glob("*.txt")))
    ]


def run_eth_ucy(args: argparse.Namespace) -> None:
    project = project_root()
    processed = project / "data/processed"
    raw = project / "tmp/Trajectron-plus-plus/experiments/pedestrians/raw"
    _, hgb, prospective = settings(project)
    config, _ = load_inputs(project)
    seeds = [config["seed"] + 997 * index for index in range(5)]
    metric_parts: list[pd.DataFrame] = []
    weight_parts: list[pd.DataFrame] = []
    calibration_records: dict[str, dict] = {}
    for scene in ("eth", "hotel", "univ", "zara1", "zara2"):
        source = raw / scene
        training = _external_batches(
            source / "train", maximum_samples=3000, seed=config["seed"]
        )
        validation = _external_batches(
            source / "val", maximum_samples=2000, seed=config["seed"] + 100
        )
        test = _external_batches(
            source / "test", maximum_samples=6000, seed=config["seed"] + 200
        )
        models = _fit_models(training, seeds, args.external_epochs, hgb, prospective)
        location, precision = _fit_mahalanobis(training)
        validation_rows = [
            _rows_from_fold(
                batch,
                _base_predictions(models, batch),
                location,
                precision,
            )
            for batch in validation
        ]
        validation_oof = pd.concat(validation_rows, ignore_index=True)
        calibrations, learned_gate = calibrate(validation_oof)
        calibration_records[scene] = calibration_payload(calibrations)
        metrics, weights, _, diagnostics = evaluate(
            {scene.upper(): test},
            models,
            calibrations,
            learned_gate,
            location,
            precision,
        )
        metric_parts.append(metrics)
        weight_parts.append(weights)
        diagnostics.to_csv(
            processed / f"referee2_{scene}_diagnostics.csv", index=False
        )
        print(f"completed ETH/UCY leave-one-scene-out: {scene}", flush=True)
    pd.concat(metric_parts, ignore_index=True).to_csv(
        processed / "referee2_eth_ucy_metrics.csv", index=False
    )
    pd.concat(weight_parts, ignore_index=True).to_csv(
        processed / "referee2_eth_ucy_weights.csv", index=False
    )
    (processed / "referee2_eth_ucy_calibration.json").write_text(
        json.dumps(calibration_records, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("julich", "ethucy", "all"), default="all")
    parser.add_argument("--crossfit-epochs", type=int, default=25)
    parser.add_argument("--final-epochs", type=int, default=45)
    parser.add_argument("--external-epochs", type=int, default=25)
    parser.add_argument("--reuse-oof", action="store_true")
    args = parser.parse_args()
    if args.stage in {"julich", "all"}:
        run_julich(args)
    if args.stage in {"ethucy", "all"}:
        run_eth_ucy(args)


if __name__ == "__main__":
    main()
