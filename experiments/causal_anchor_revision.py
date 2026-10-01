# Author: Amir Ghorbani
"""Major reviewer revision: causal routing, factorial attribution, and external data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pedgeom.benchmarks import (
    CausalBufferRouter,
    FixedBlendRegressor,
    fit_avm_model,
    fit_gradient_boosted_residual,
    fit_interaction_mlp,
    fit_residual_on_base_mlp,
    fit_ttc_response_model,
    invariant_design,
    local_target,
    local_target_from_vectors,
    prospective_interaction_state,
)
from pedgeom.calibration import (
    FittedVelocityModel,
    VelocitySamples,
    build_velocity_samples,
    fit_anisotropic_social_force_response_model,
    velocity_metrics,
)
from pedgeom.datasets import load_julich_crossing_trajectory, load_julich_trajectory
from pedgeom.external import build_eth_ucy_samples
from pedgeom.graph_baseline import fit_graph_interaction_network
from pedgeom.statistics import exact_paired_randomization_pvalue, run_bootstrap_interval


def project_root() -> Path:
    """Return the repository root for a source checkout."""

    return Path(__file__).resolve().parents[1]


def load_inputs(project: Path) -> tuple[dict, dict]:
    """Load the frozen experimental configuration and complete-run split."""

    config = json.loads(
        (project / "configs" / "major_revision.json").read_text(encoding="utf-8")
    )
    splits = json.loads(
        (project / "data" / "splits.json").read_text(encoding="utf-8")
    )
    return config, splits


@dataclass
class ProspectiveRegressor:
    """Prospective-geometry expert expressed in the pedestrian's local frame."""

    estimator: object
    parameters: dict
    residual: bool

    def _design(self, samples: VelocitySamples) -> np.ndarray:
        design = np.column_stack(
            (
                invariant_design(samples),
                prospective_interaction_state(samples, **self.parameters),
            )
        )
        if self.residual:
            anchor_local = local_target_from_vectors(samples, samples.features[:, 0])
            design = np.column_stack((design, anchor_local))
        return design

    def predict(self, samples: VelocitySamples, speed_cap: float = 2.5) -> np.ndarray:
        local = self.estimator.predict(self._design(samples))
        if self.residual:
            local += local_target_from_vectors(samples, samples.features[:, 0])
        goal = samples.features[:, 1]
        lateral = np.column_stack((-goal[:, 1], goal[:, 0]))
        prediction = local[:, :1] * goal + local[:, 1:] * lateral
        speed = np.linalg.norm(prediction, axis=1)
        scale = np.minimum(1.0, speed_cap / np.maximum(speed, 1e-12))
        return prediction * scale[:, None]


def fit_hgb(
    batches: list[VelocitySamples],
    parameters: dict,
    settings: dict,
    seed: int,
    *,
    residual: bool,
) -> ProspectiveRegressor:
    """Fit the tree-based prospective expert used in the primary model."""

    model = ProspectiveRegressor(None, parameters, residual)
    design = np.concatenate([model._design(batch) for batch in batches])
    target = np.concatenate([local_target(batch) for batch in batches])
    if residual:
        target -= np.concatenate(
            [local_target_from_vectors(batch, batch.features[:, 0]) for batch in batches]
        )
    estimator = MultiOutputRegressor(
        HistGradientBoostingRegressor(random_state=seed, **settings)
    )
    estimator.fit(design, target)
    return ProspectiveRegressor(estimator, parameters, residual)


def fit_mlp(
    batches: list[VelocitySamples],
    parameters: dict,
    selected: dict,
    seed: int,
    *,
    residual: bool,
) -> ProspectiveRegressor:
    """Fit the matched MLP used only in the two-by-two attribution check."""

    model = ProspectiveRegressor(None, parameters, residual)
    design = np.concatenate([model._design(batch) for batch in batches])
    target = np.concatenate([local_target(batch) for batch in batches])
    if residual:
        target -= np.concatenate(
            [local_target_from_vectors(batch, batch.features[:, 0]) for batch in batches]
        )
        epochs = selected["residual_epoch"]
    else:
        epochs = selected["mlp_configuration"]["selected_epoch"]
    config = selected["mlp_configuration"]
    estimator = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=tuple(config["hidden_layers"]),
            activation="tanh",
            solver="adam",
            alpha=config["alpha"],
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=epochs,
            early_stopping=False,
            random_state=seed,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(design, target)
    return ProspectiveRegressor(estimator, parameters, residual)


def anchor() -> FittedVelocityModel:
    return FittedVelocityModel(("persistence",), np.array([1.0]))


def build_batch(
    source: Path,
    config: dict,
    *,
    history_frames: int,
    crossing: bool = False,
    speed_threshold: float | None = None,
    neighbour_cutoff: float | None = None,
    maximum_neighbours: int = 8,
) -> VelocitySamples:
    return build_velocity_samples(
        source,
        interaction_range=config["interaction_range_m"],
        horizon_frames=config["horizon_frames"],
        history_frames=history_frames,
        frame_stride=config["frame_stride"],
        maximum_samples=config["maximum_samples_per_run"],
        radius=config["pedestrian_radius_m"],
        wall_range=config["wall_range_m"],
        wall_y_bounds=None,
        neighbour_cutoff=(
            config["neighbour_cutoff_m"]
            if neighbour_cutoff is None
            else neighbour_cutoff
        ),
        maximum_neighbours=maximum_neighbours,
        goal_method=config["primary_goal_method"],
        speed_outlier_threshold=(
            config["speed_outlier_threshold_mps"]
            if speed_threshold is None
            else speed_threshold
        ),
        loader=load_julich_crossing_trajectory if crossing else load_julich_trajectory,
        seed=config["seed"],
    )


def build_julich(
    project: Path,
    config: dict,
    splits: dict,
    *,
    history_frames: int,
    speed_threshold: float | None = None,
    neighbour_cutoff: float | None = None,
    maximum_neighbours: int = 8,
) -> tuple[dict[str, VelocitySamples], dict[str, list[VelocitySamples]]]:
    corridor = project / "data/raw/2013bidirectional"
    crossing = project / "data/raw/2013crossing90/trajectories"
    keyword = {
        "history_frames": history_frames,
        "speed_threshold": speed_threshold,
        "neighbour_cutoff": neighbour_cutoff,
        "maximum_neighbours": maximum_neighbours,
    }
    training = {
        run: build_batch(corridor / f"{run}.txt", config, **keyword)
        for run in splits["calibration"]
    }
    datasets = {
        "familiar corridor": [
            build_batch(corridor / f"{run}.txt", config, **keyword)
            for run in splits["held_out_validation"]
        ],
        "altered geometry": [
            build_batch(corridor / f"{run}.txt", config, **keyword)
            for run in splits["held_out_geometry_stress"]
        ],
        "crossing topology": [
            build_batch(source, config, crossing=True, **keyword)
            for source in sorted(crossing.glob("crossing_90_[de]_*.txt"))
        ],
    }
    return training, datasets


def settings(project: Path) -> tuple[dict, dict, dict]:
    processed = project / "data/processed"
    selected = json.loads((processed / "selected_hyperparameters.json").read_text())
    prospective = json.loads((processed / "prospective_selection.json").read_text())
    hgb = {
        key: value
        for key, value in prospective["backbone_settings"].items()
    }
    prospective_parameters = {
        key: prospective[key]
        for key in ("horizon_s", "time_scale_s", "clearance_scale_m")
    }
    return selected, hgb, prospective_parameters


def fit_pair(
    batches: list[VelocitySamples],
    selected: dict,
    hgb: dict,
    prospective_parameters: dict,
    seed: int,
    *,
    residual: bool,
) -> tuple[object, object]:
    """Fit prospective boosted and coarse neural experts with matched targets."""

    first = fit_hgb(
        batches, prospective_parameters, hgb, seed, residual=residual
    )
    mlp = selected["mlp_configuration"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if residual:
            second = fit_residual_on_base_mlp(
                batches,
                anchor(),
                hidden_layers=tuple(mlp["hidden_layers"]),
                alpha=mlp["alpha"],
                max_iter=selected["residual_epoch"],
                seed=seed,
            )
        else:
            second = fit_interaction_mlp(
                batches,
                hidden_layers=tuple(mlp["hidden_layers"]),
                alpha=mlp["alpha"],
                max_iter=mlp["selected_epoch"],
                seed=seed,
            )
    return first, second


def fit_guard_pair(
    batches: list[VelocitySamples],
    hgb: dict,
    prospective_parameters: dict,
    seed: int,
    *,
    residual: bool,
    graph_epochs: int = 25,
) -> tuple[object, object]:
    """Fit a learned graph backbone and anchored geometry transfer expert."""

    graph = fit_graph_interaction_network(
        batches, seed=seed, epochs=graph_epochs, residual=residual
    )
    geometry = fit_hgb(
        batches, prospective_parameters, hgb, seed, residual=residual
    )
    return graph, geometry


def frame_scores(router: CausalBufferRouter, batch: VelocitySamples) -> np.ndarray:
    scores = router.sample_scores(batch)
    frame = pd.DataFrame({"frame": batch.frame, "score": scores})
    return frame.groupby("frame", sort=True)["score"].first().to_numpy()


def conformal_quantile(values: np.ndarray, coverage: float) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    rank = min(len(values), int(np.ceil((len(values) + 1) * coverage)))
    return float(values[max(rank - 1, 0)])


def calibrate_router_crossfit(
    training: list[VelocitySamples],
    selected: dict,
    hgb: dict,
    prospective_parameters: dict,
    seed: int,
    *,
    residual: bool,
    graph_guard: bool = False,
    graph_epochs: int = 25,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Estimate the router only from leave-one-complete-run-out predictions."""

    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []
    fold_models: list[tuple[VelocitySamples, object, object]] = []
    all_scores: list[np.ndarray] = []
    for held_out, validation in enumerate(training):
        fitting = [batch for index, batch in enumerate(training) if index != held_out]
        if graph_guard:
            first, second = fit_guard_pair(
                fitting,
                hgb,
                prospective_parameters,
                seed + held_out,
                residual=residual,
                graph_epochs=graph_epochs,
            )
        else:
            first, second = fit_pair(
                fitting,
                selected,
                hgb,
                prospective_parameters,
                seed + held_out,
                residual=residual,
            )
        provisional = CausalBufferRouter(first, second, np.inf, 1.0, buffer_frames=5)
        scores = frame_scores(provisional, validation)
        all_scores.append(scores)
        fold_models.append((validation, first, second))
        fold_rows.extend(
            {
                "run": validation.run,
                "frame_index": index,
                "disagreement_mps": float(score),
            }
            for index, score in enumerate(scores)
        )

    pooled = np.concatenate(all_scores)
    threshold = conformal_quantile(pooled, 0.90)
    q25, q75 = np.quantile(pooled, (0.25, 0.75))
    scale = max(float(q75 - q25), 1e-6)
    candidates = {
        "conformal-80": conformal_quantile(pooled, 0.80),
        "conformal-90": threshold,
        "conformal-95": conformal_quantile(pooled, 0.95),
        "Tukey-1.5": float(q75 + 1.5 * (q75 - q25)),
    }
    fixed_grid = np.linspace(0.0, 1.0, 21)
    fixed_losses = {weight: [] for weight in fixed_grid}
    for validation, first, second in fold_models:
        first_prediction = first.predict(validation, speed_cap=100.0)
        second_prediction = second.predict(validation, speed_cap=100.0)
        for weight in fixed_grid:
            prediction = weight * first_prediction + (1.0 - weight) * second_prediction
            fixed_losses[weight].append(
                velocity_metrics(validation.target, prediction)["vector_rmse_mps"]
            )
    best_fixed = min(fixed_losses, key=lambda value: np.mean(fixed_losses[value]))

    sensitivity_rows = []
    routed_errors: list[np.ndarray] = []
    for validation, first, second in fold_models:
        first_prediction = first.predict(validation, speed_cap=100.0)
        second_prediction = second.predict(validation, speed_cap=100.0)
        for label, candidate_threshold in candidates.items():
            candidate = CausalBufferRouter(
                first,
                second,
                candidate_threshold,
                scale,
                buffer_frames=5,
                base_weight=float(best_fixed),
            )
            prediction = candidate.predict(validation, speed_cap=100.0)
            sensitivity_rows.append(
                {
                    "rule": label,
                    "run": validation.run,
                    "vector_rmse_mps": velocity_metrics(
                        validation.target, prediction
                    )["vector_rmse_mps"],
                    "mean_in_support_weight": float(
                        np.mean(candidate.in_support_weights(validation))
                    ),
                }
            )
        router = CausalBufferRouter(
            first,
            second,
            threshold,
            scale,
            buffer_frames=5,
            base_weight=float(best_fixed),
        )
        routed = router.predict(validation, speed_cap=100.0)
        routed_errors.append(np.linalg.norm(routed - validation.target, axis=1))
        weights = router.in_support_weights(validation)
        score = router.sample_scores(validation)
        first_squared = np.sum((first_prediction - validation.target) ** 2, axis=1)
        second_squared = np.sum((second_prediction - validation.target) ** 2, axis=1)
        for index in range(len(validation.target)):
            prediction_rows.append(
                {
                    "run": validation.run,
                    "frame": int(validation.frame[index]),
                    "sample": index,
                    "disagreement_mps": float(score[index]),
                    "in_support_weight": float(weights[index]),
                    "in_support_minus_transfer_squared_error": float(
                        first_squared[index] - second_squared[index]
                    ),
                    "routed_radial_error_mps": float(
                        np.linalg.norm(routed[index] - validation.target[index])
                    ),
                }
            )
    payload = {
        "calibration": "leave-one-complete-run-out",
        "buffer_frames": 5,
        "buffer_seconds": 2.0,
        "threshold_rule": "finite-sample 90% upper conformal quantile",
        "threshold_mps": threshold,
        "scale_mps": scale,
        "nominal_in_support_weight": float(best_fixed),
        "oof_90_radial_error_mps": conformal_quantile(
            np.concatenate(routed_errors), 0.90
        ),
    }
    return payload, pd.DataFrame(prediction_rows), pd.DataFrame(sensitivity_rows)


def evaluate_models(
    datasets: dict[str, list[VelocitySamples]], models: dict[str, object], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for dataset, batches in datasets.items():
        for batch in batches:
            for model_name, model in models.items():
                prediction = model.predict(batch, speed_cap=100.0)
                rows.append(
                    {
                        "seed": seed,
                        "dataset": dataset,
                        "run": batch.run,
                        "model": model_name,
                        "samples": len(batch.target),
                        **velocity_metrics(batch.target, prediction),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame()


def conflict_state(batch: VelocitySamples, parameters: dict) -> pd.DataFrame:
    state = prospective_interaction_state(batch, **parameters)
    raw = batch.raw_neighbours
    position = raw[:, :, :2]
    velocity = raw[:, :, 2:4]
    present = raw[:, :, 4] > 0.5
    speed_squared = np.sum(velocity**2, axis=2)
    approach = np.sum(position * velocity, axis=2)
    tca = np.divide(
        -approach,
        speed_squared,
        out=np.full_like(approach, np.inf),
        where=speed_squared > 1e-9,
    )
    active = present & (approach < 0.0) & (tca > 0.0) & (tca <= parameters["horizon_s"])
    projected = position + np.where(active, tca, 0.0)[:, :, None] * velocity
    clearance = np.maximum(np.linalg.norm(projected, axis=2) - 0.5, 0.0)
    tca[~active] = np.inf
    clearance[~active] = np.inf
    return pd.DataFrame(
        {
            "risk": state[:, 0],
            "minimum_tca_s": np.min(tca, axis=1),
            "minimum_clearance_m": np.min(clearance, axis=1),
        }
    )


def stratified_evaluation(
    datasets: dict[str, list[VelocitySamples]],
    models: dict[str, object],
    prospective_parameters: dict,
) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset, batches in datasets.items():
        risk_all = np.concatenate(
            [conflict_state(batch, prospective_parameters)["risk"] for batch in batches]
        )
        positive = risk_all[risk_all > 0.0]
        risk_edges = np.quantile(positive, (0.0, 1 / 3, 2 / 3, 1.0)) if len(positive) else None
        for batch in batches:
            state = conflict_state(batch, prospective_parameters)
            if risk_edges is None:
                risk_label = np.full(len(state), "no projected conflict", dtype=object)
            else:
                risk_label = np.full(len(state), "no projected conflict", dtype=object)
                active = state["risk"].to_numpy() > 0.0
                risk_label[active] = pd.cut(
                    state.loc[active, "risk"],
                    bins=np.unique(risk_edges),
                    labels=("low", "medium", "high")[: len(np.unique(risk_edges)) - 1],
                    include_lowest=True,
                ).astype(str)
            tca = state["minimum_tca_s"].to_numpy()
            tca_label = np.select(
                (tca < 1.0, tca < 2.0, np.isfinite(tca)),
                ("<1 s", "1-2 s", "2-3 s"),
                default="none",
            )
            clearance = state["minimum_clearance_m"].to_numpy()
            clearance_label = np.select(
                (clearance < 0.25, clearance < 0.75, np.isfinite(clearance)),
                ("<0.25 m", "0.25-0.75 m", ">=0.75 m"),
                default="none",
            )
            for model_name, model in models.items():
                prediction = model.predict(batch, speed_cap=100.0)
                squared = np.sum((prediction - batch.target) ** 2, axis=1)
                for variable, labels in (
                    ("prospective risk", risk_label),
                    ("time to closest approach", tca_label),
                    ("closest clearance", clearance_label),
                ):
                    for level in np.unique(labels):
                        mask = labels == level
                        rows.append(
                            {
                                "dataset": dataset,
                                "run": batch.run,
                                "stratum": variable,
                                "level": level,
                                "model": model_name,
                                "samples": int(mask.sum()),
                                "vector_rmse_mps": float(np.sqrt(np.mean(squared[mask]))),
                            }
                        )
    return pd.DataFrame(rows)


def buffer_correlation(
    dataset: str,
    batches: list[VelocitySamples],
    router: CausalBufferRouter,
) -> pd.DataFrame:
    rows = []
    for batch in batches:
        first = router.in_support.predict(batch, speed_cap=100.0)
        second = router.transfer.predict(batch, speed_cap=100.0)
        score = router.sample_scores(batch)
        first_error = np.sum((first - batch.target) ** 2, axis=1)
        second_error = np.sum((second - batch.target) ** 2, axis=1)
        routed_prediction = router.predict(batch, speed_cap=100.0)
        frames = np.unique(batch.frame)
        for start in range(0, len(frames), router.buffer_frames):
            selected_frames = frames[start : start + router.buffer_frames]
            mask = np.isin(batch.frame, selected_frames)
            rows.append(
                {
                    "dataset": dataset,
                    "run": batch.run,
                    "buffer": start // router.buffer_frames,
                    "disagreement_mps": float(np.mean(score[mask])),
                    "in_support_minus_transfer_mse": float(
                        np.mean(first_error[mask] - second_error[mask])
                    ),
                    "routed_rmse_mps": float(
                        np.sqrt(
                            np.mean(
                                np.sum(
                                    (
                                        routed_prediction[mask] - batch.target[mask]
                                    )
                                    ** 2,
                                    axis=1,
                                )
                            )
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def collision_screen(
    datasets: dict[str, list[VelocitySamples]],
    models: dict[str, object],
    *,
    margin_m: float = 0.8,
    horizon_s: float = 1.0,
) -> pd.DataFrame:
    rows: list[dict] = []

    def unsafe(batch: VelocitySamples, focal: np.ndarray) -> np.ndarray:
        raw = batch.raw_neighbours
        position = raw[:, :, :2]
        present = raw[:, :, 4] > 0.5
        neighbour_velocity = batch.features[:, None, 0, :] + raw[:, :, 2:4]
        relative_velocity = neighbour_velocity - focal[:, None, :]
        speed_squared = np.sum(relative_velocity**2, axis=2)
        time = np.divide(
            -np.sum(position * relative_velocity, axis=2),
            speed_squared,
            out=np.zeros_like(speed_squared),
            where=speed_squared > 1e-9,
        )
        time = np.clip(time, 0.0, horizon_s)
        distance = np.linalg.norm(position + time[:, :, None] * relative_velocity, axis=2)
        distance[~present] = np.inf
        return np.min(distance, axis=1) < margin_m

    for dataset, batches in datasets.items():
        for batch in batches:
            truth = unsafe(batch, batch.target)
            for model_name, model in models.items():
                prediction = model.predict(batch, speed_cap=100.0)
                flagged = unsafe(batch, prediction)
                negative = ~truth
                positive = truth
                rows.append(
                    {
                        "dataset": dataset,
                        "run": batch.run,
                        "model": model_name,
                        "margin_m": margin_m,
                        "true_conflicts": int(positive.sum()),
                        "false_alarm_rate": float(np.mean(flagged[negative]))
                        if np.any(negative)
                        else np.nan,
                        "miss_rate": float(np.mean(~flagged[positive]))
                        if np.any(positive)
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def paired_statistics(metrics: pd.DataFrame, primary: str, comparators: list[str]) -> dict:
    averaged = (
        metrics.groupby(["dataset", "run", "model"], sort=False)["vector_rmse_mps"]
        .mean()
        .reset_index()
    )
    output: dict[str, dict] = {}
    all_p: list[tuple[str, str, float]] = []
    for dataset, frame in averaged.groupby("dataset", sort=False):
        pivot = frame.pivot(index="run", columns="model", values="vector_rmse_mps")
        output[dataset] = {}
        for comparator in comparators:
            if primary not in pivot or comparator not in pivot:
                continue
            difference = (pivot[primary] - pivot[comparator]).dropna().to_numpy()
            p_value = exact_paired_randomization_pvalue(difference)
            all_p.append((dataset, comparator, p_value))
            output[dataset][comparator] = {
                "runs": len(difference),
                "mean_difference_mps": float(np.mean(difference)),
                "relative_reduction_percent": float(
                    -100.0 * np.mean(difference) / pivot[comparator].mean()
                ),
                "bootstrap_95_ci_mps": run_bootstrap_interval(difference),
                "exact_p": p_value,
                "runs_lower": int(np.sum(difference < 0.0)),
            }
    ordered = sorted(all_p, key=lambda item: item[2])
    adjusted: dict[tuple[str, str], float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (dataset, comparator, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p_value))
        adjusted[(dataset, comparator)] = running
    for dataset, comparator, _ in all_p:
        output[dataset][comparator]["holm_adjusted_p_across_reported_tests"] = adjusted[
            (dataset, comparator)
        ]
    return output


def external_eth(
    project: Path,
    selected: dict,
    hgb: dict,
    prospective_parameters: dict,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    source = project / "tmp/Trajectron-plus-plus/experiments/pedestrians/raw/eth"
    if not source.exists():
        return pd.DataFrame(), {"status": "raw source unavailable"}
    training = [
        build_eth_ucy_samples(path, history_steps=2, seed=seed)
        for path in sorted((source / "train").glob("*.txt"))
    ]
    test = [build_eth_ucy_samples(source / "test/biwi_eth.txt", history_steps=2, seed=seed)]
    calibration, _, _ = calibrate_router_crossfit(
        training,
        selected,
        hgb,
        prospective_parameters,
        seed,
        residual=True,
        graph_guard=True,
    )
    first, second = fit_guard_pair(
        training,
        hgb,
        prospective_parameters,
        seed,
        residual=True,
        graph_epochs=45,
    )
    router = CausalBufferRouter(
        first,
        second,
        calibration["threshold_mps"],
        calibration["scale_mps"],
        buffer_frames=5,
        base_weight=calibration["nominal_in_support_weight"],
    )
    direct = fit_interaction_mlp(
        training,
        hidden_layers=tuple(selected["mlp_configuration"]["hidden_layers"]),
        alpha=selected["mlp_configuration"]["alpha"],
        max_iter=selected["mlp_configuration"]["selected_epoch"],
        seed=seed,
    )
    metrics, _ = evaluate_models(
        {"ETH naturalistic": test},
        {
            "constant velocity": anchor(),
            "direct MLP": direct,
            "graph interaction network": first,
            "GEL-Ped": router,
        },
        seed,
    )
    provenance = {
        "source_repository": "https://github.com/StanfordASL/Trajectron-plus-plus",
        "source_commit": "1031c7bd1a444273af378c1ec1dcca907ba59830",
        "split": "official eth train/test folders",
        "test_file": "biwi_eth.txt",
        "history_s": 0.8,
        "prediction_s": 0.4,
        "raw_data_redistributed": False,
    }
    return metrics, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--calibration-graph-epochs", type=int, default=25)
    parser.add_argument("--graph-epochs", type=int, default=45)
    parser.add_argument("--reuse-calibration", action="store_true")
    parser.add_argument("--reuse-five-seed-metrics", action="store_true")
    parser.add_argument("--skip-external", action="store_true")
    args = parser.parse_args()
    project = project_root()
    processed = project / "data/processed"
    config, splits = load_inputs(project)
    selected, hgb, prospective_parameters = settings(project)

    # A longer observation interval is fixed before any test regime is read.
    history_frames = 20
    training_map, datasets = build_julich(
        project, config, splits, history_frames=history_frames
    )
    training = list(training_map.values())
    if args.reuse_calibration:
        stored = json.loads(
            (processed / "causal_router_calibration.json").read_text()
        )
        residual_calibration = stored["anchored"]
        direct_calibration = stored["direct"]
        for payload in (residual_calibration, direct_calibration):
            if "nominal_in_support_weight" not in payload:
                payload["nominal_in_support_weight"] = payload.pop(
                    "best_fixed_weight_calibration"
                )
    else:
        residual_calibration, residual_oof, fence_sensitivity = calibrate_router_crossfit(
            training,
            selected,
            hgb,
            prospective_parameters,
            config["seed"],
            residual=True,
            graph_guard=True,
            graph_epochs=args.calibration_graph_epochs,
        )
        direct_calibration, direct_oof, direct_fence_sensitivity = (
            calibrate_router_crossfit(
                training,
                selected,
                hgb,
                prospective_parameters,
                config["seed"],
                residual=False,
                graph_guard=True,
                graph_epochs=args.calibration_graph_epochs,
            )
        )
        (processed / "causal_router_calibration.json").write_text(
            json.dumps(
                {"anchored": residual_calibration, "direct": direct_calibration},
                indent=2,
            )
        )
        pd.concat(
            (
                residual_oof.assign(model="anchored"),
                direct_oof.assign(model="direct"),
            ),
            ignore_index=True,
        ).to_csv(processed / "causal_router_oof_predictions.csv", index=False)
        pd.concat(
            (
                fence_sensitivity.assign(model="anchored"),
                direct_fence_sensitivity.assign(model="direct"),
            ),
            ignore_index=True,
        ).to_csv(processed / "causal_router_fence_sensitivity.csv", index=False)

    all_metrics: list[pd.DataFrame] = []
    primary_models: dict[str, object] | None = None
    seeds = [config["seed"] + 997 * index for index in range(args.seeds)]
    for seed in seeds:
        graph_residual, prospective_residual = fit_guard_pair(
            training,
            hgb,
            prospective_parameters,
            seed,
            residual=True,
            graph_epochs=args.graph_epochs,
        )
        graph_direct, prospective_direct = fit_guard_pair(
            training,
            hgb,
            prospective_parameters,
            seed,
            residual=False,
            graph_epochs=args.graph_epochs,
        )
        mlp = selected["mlp_configuration"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            coarse_residual = fit_residual_on_base_mlp(
                training,
                anchor(),
                hidden_layers=tuple(mlp["hidden_layers"]),
                alpha=mlp["alpha"],
                max_iter=selected["residual_epoch"],
                seed=seed,
            )
            coarse_direct = fit_interaction_mlp(
                training,
                hidden_layers=tuple(mlp["hidden_layers"]),
                alpha=mlp["alpha"],
                max_iter=mlp["selected_epoch"],
                seed=seed,
            )
        gel = CausalBufferRouter(
            graph_residual,
            prospective_residual,
            residual_calibration["threshold_mps"],
            residual_calibration["scale_mps"],
            buffer_frames=5,
            base_weight=residual_calibration["nominal_in_support_weight"],
        )
        matched = CausalBufferRouter(
            graph_direct,
            prospective_direct,
            direct_calibration["threshold_mps"],
            direct_calibration["scale_mps"],
            buffer_frames=5,
            base_weight=direct_calibration["nominal_in_support_weight"],
        )
        models = {
            "constant velocity": anchor(),
            "direct MLP": coarse_direct,
            "graph interaction network": graph_residual,
            "anchored coarse MLP": coarse_residual,
            "matched direct control": matched,
            "GEL-Ped": gel,
        }
        if primary_models is None:
            primary_models = models | {
                "geometry transfer expert (g=0)": prospective_residual,
                "calibration-best fixed g": FixedBlendRegressor(
                    graph_residual,
                    prospective_residual,
                    residual_calibration["nominal_in_support_weight"],
                ),
            }
        seed_metrics, _ = evaluate_models(datasets, models, seed)
        all_metrics.append(seed_metrics)
        print(f"completed stochastic seed {seed}", flush=True)

    if args.reuse_five_seed_metrics:
        metrics = pd.read_csv(processed / "causal_anchor_metrics.csv")
    else:
        mechanical = {
            "anisotropic Social Force": fit_anisotropic_social_force_response_model(
                training
            ),
            "TTC response": fit_ttc_response_model(training),
            "Anticipation Velocity Model": fit_avm_model(training),
        }
        mechanical_metrics, _ = evaluate_models(datasets, mechanical, seeds[0])
        all_metrics.append(mechanical_metrics)
        metrics = pd.concat(all_metrics, ignore_index=True)
        metrics.to_csv(processed / "causal_anchor_metrics.csv", index=False)

    seed_mean = (
        metrics.groupby(["dataset", "run", "model"], sort=False)
        .agg(
            seeds=("seed", "nunique"),
            vector_rmse_mps=("vector_rmse_mps", "mean"),
            seed_sd_rmse_mps=("vector_rmse_mps", "std"),
            displacement_mae_m=("displacement_mae_m", "mean"),
        )
        .reset_index()
    )
    seed_mean.to_csv(processed / "causal_anchor_run_seed_mean.csv", index=False)
    summary = (
        seed_mean.groupby(["dataset", "model"], sort=False)
        .agg(
            runs=("run", "nunique"),
            mean_rmse_mps=("vector_rmse_mps", "mean"),
            run_sd_rmse_mps=("vector_rmse_mps", "std"),
            mean_seed_sd_rmse_mps=("seed_sd_rmse_mps", "mean"),
            displacement_mae_m=("displacement_mae_m", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(processed / "causal_anchor_summary.csv", index=False)
    statistics = paired_statistics(
        metrics,
        "GEL-Ped",
        [
            "constant velocity",
            "direct MLP",
            "graph interaction network",
            "anchored coarse MLP",
            "matched direct control",
        ],
    )
    (processed / "causal_anchor_statistics.json").write_text(
        json.dumps(statistics, indent=2)
    )

    assert primary_models is not None
    primary_gel = primary_models["GEL-Ped"]
    # Full router ablation, including a label-using oracle reported only as a bound.
    ablation_models = {
        name: model
        for name, model in primary_models.items()
        if name
        in {
            "anchored coarse MLP",
            "graph interaction network",
            "GEL-Ped",
            "geometry transfer expert (g=0)",
            "calibration-best fixed g",
        }
    }
    ablation, _ = evaluate_models(datasets, ablation_models, seeds[0])
    oracle_rows = []
    first = primary_gel.in_support
    second = primary_gel.transfer
    for dataset, batches in datasets.items():
        for weight in np.linspace(0.0, 1.0, 21):
            model = FixedBlendRegressor(first, second, float(weight))
            values = [
                velocity_metrics(batch.target, model.predict(batch, speed_cap=100.0))[
                    "vector_rmse_mps"
                ]
                for batch in batches
            ]
            oracle_rows.append(
                {
                    "dataset": dataset,
                    "weight": weight,
                    "mean_rmse_mps": np.mean(values),
                }
            )
    oracle = pd.DataFrame(oracle_rows)
    oracle = oracle.loc[oracle.groupby("dataset")["mean_rmse_mps"].idxmin()]
    ablation.to_csv(processed / "causal_router_ablation.csv", index=False)
    oracle.to_csv(processed / "causal_router_oracle_fixed.csv", index=False)

    # Same-learner 2 x 2 attribution: feature state x learner family.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        prospective_mlp = fit_mlp(
            training,
            prospective_parameters,
            selected,
            seeds[0],
            residual=True,
        )
    coarse_hgb = fit_gradient_boosted_residual(
        training, anchor(), seed=seeds[0], **hgb
    )
    factorial_models = {
        "coarse + MLP": primary_models["anchored coarse MLP"],
        "prospective + MLP": prospective_mlp,
        "coarse + HGB": coarse_hgb,
        "prospective + HGB": primary_gel.transfer,
    }
    factorial, _ = evaluate_models(datasets, factorial_models, seeds[0])
    factorial.to_csv(processed / "causal_factorial_2x2.csv", index=False)

    strata = stratified_evaluation(
        datasets,
        {
            "direct MLP": primary_models["direct MLP"],
            "graph interaction network": primary_models["graph interaction network"],
            "anchored coarse MLP": primary_models["anchored coarse MLP"],
            "GEL-Ped": primary_models["GEL-Ped"],
        },
        prospective_parameters,
    )
    strata.to_csv(processed / "causal_conflict_strata.csv", index=False)

    correlations = []
    for dataset, batches in {"calibration OOF": training, **datasets}.items():
        if dataset == "calibration OOF":
            continue
        frame = buffer_correlation(dataset, batches, primary_gel)
        correlations.append(frame)
    buffer_frame = pd.concat(correlations, ignore_index=True)
    buffer_frame.to_csv(processed / "causal_buffer_diagnostics.csv", index=False)
    correlation_summary = []
    for dataset, frame in buffer_frame.groupby("dataset", sort=False):
        rho, p_value = spearmanr(
        frame["disagreement_mps"], frame["in_support_minus_transfer_mse"]
        )
        correlation_summary.append(
            {
                "dataset": dataset,
                "buffers": len(frame),
                "spearman_rho": rho,
                "p_value_descriptive_only": p_value,
            }
        )
    pd.DataFrame(correlation_summary).to_csv(
        processed / "causal_buffer_correlation.csv", index=False
    )

    gate_rows = []
    for dataset, batches in {"calibration": training, **datasets}.items():
        for batch in batches:
            weights = primary_gel.in_support_weights(batch)
            gate_rows.append(
                {
                    "dataset": dataset,
                    "run": batch.run,
                    "mean_g": float(np.mean(weights)),
                    "median_g": float(np.median(weights)),
                    "q05_g": float(np.quantile(weights, 0.05)),
                    "q95_g": float(np.quantile(weights, 0.95)),
                    "fraction_g_below_half": float(np.mean(weights < 0.5)),
                }
            )
    pd.DataFrame(gate_rows).to_csv(
        processed / "causal_router_realized_weights.csv", index=False
    )

    screen = collision_screen(
        datasets,
        {
            "constant velocity": anchor(),
            "direct MLP": primary_models["direct MLP"],
            "graph interaction network": primary_models["graph interaction network"],
            "GEL-Ped": primary_models["GEL-Ped"],
        },
    )
    screen.to_csv(processed / "causal_collision_screen.csv", index=False)

    # A simple, fully calibration-only prediction set.  Its radius is the
    # finite-sample 90% quantile of leave-one-run-out GEL-Ped errors; the table
    # below reports held-out coverage without retuning the radius.
    uncertainty_rows = []
    radius = float(residual_calibration["oof_90_radial_error_mps"])
    for dataset, batches in datasets.items():
        for batch in batches:
            error = np.linalg.norm(
                primary_gel.predict(batch, speed_cap=100.0) - batch.target,
                axis=1,
            )
            uncertainty_rows.append(
                {
                    "dataset": dataset,
                    "run": batch.run,
                    "samples": len(error),
                    "calibrated_radius_mps": radius,
                    "empirical_coverage": float(np.mean(error <= radius)),
                    "mean_radial_error_mps": float(np.mean(error)),
                }
            )
    pd.DataFrame(uncertainty_rows).to_csv(
        processed / "causal_uncertainty_coverage.csv", index=False
    )

    # Deployment-facing encoding sensitivity.  The fitted weights and router
    # are frozen; only the available observation history, neighbour envelope,
    # or evaluation speed filter changes.  This avoids selecting a model on a
    # held-out regime while exposing dependence on preprocessing choices.
    encoding_variants = {
        "primary: 0.8 s, 3 m, 8 neighbours, <3 m/s": {
            "history_frames": history_frames,
        },
        "short history: 0.4 s": {"history_frames": 10},
        "wider field: 5 m, 12 neighbours": {
            "history_frames": history_frames,
            "neighbour_cutoff": 5.0,
            "maximum_neighbours": 12,
        },
        "all observed speeds": {
            "history_frames": history_frames,
            "speed_threshold": 100.0,
        },
    }
    sensitivity_rows = []
    for label, options in encoding_variants.items():
        _, variant_datasets = build_julich(project, config, splits, **options)
        variant, _ = evaluate_models(
            variant_datasets,
            {
                "graph interaction network": primary_models[
                    "graph interaction network"
                ],
                "GEL-Ped": primary_gel,
            },
            seeds[0],
        )
        sensitivity_rows.append(variant.assign(encoding=label))
    pd.concat(sensitivity_rows, ignore_index=True).to_csv(
        processed / "causal_encoding_sensitivity.csv", index=False
    )

    if not args.skip_external:
        external_metrics, provenance = external_eth(
            project,
            selected,
            hgb,
            prospective_parameters,
            seeds[0],
        )
        external_metrics.to_csv(processed / "external_eth_metrics.csv", index=False)
        (processed / "external_eth_provenance.json").write_text(
            json.dumps(provenance, indent=2)
        )

    print(summary.to_string(index=False), flush=True)
    print("causal anchor revision complete", flush=True)


if __name__ == "__main__":
    main()
