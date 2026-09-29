# Author: Amir Ghorbani
"""Reviewer-requested attribution, confound, mechanism, and subgroup analyses."""

from __future__ import annotations

import json
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.exceptions import ConvergenceWarning

from pedgeom.benchmarks import (
    GELPedRegressor,
    fit_avm_model,
    fit_linear_skip,
    fit_raw_neighbour_mlp,
    fit_residual_on_base_mlp,
    fit_ttc_response_model,
)
from pedgeom.calibration import (
    FEATURE_NAMES,
    VelocitySamples,
    build_velocity_samples,
    fit_anisotropic_social_force_response_model,
    velocity_metrics,
)
from pedgeom.datasets import (
    load_julich_crossing_trajectory,
    load_julich_trajectory,
)
from pedgeom.statistics import exact_paired_randomization_pvalue, run_bootstrap_interval

from major_revision_analysis import (
    ModelSpec,
    build_batch,
    fit_models,
    load_inputs,
    mlp_parameter_count,
    project_root,
)


def copy_batch(
    batch: VelocitySamples,
    *,
    features: np.ndarray | None = None,
) -> VelocitySamples:
    return replace(batch, features=batch.features.copy() if features is None else features)


def without_wall(batch: VelocitySamples) -> VelocitySamples:
    features = batch.features.copy()
    features[:, FEATURE_NAMES.index("wall"), :] = 0.0
    return copy_batch(batch, features=features)


def model_parameters(model: object, default: int) -> int:
    """Return a documented parameter count for compact benchmark tables."""

    return default


def run_metrics(
    datasets: dict[str, list[VelocitySamples]],
    models: dict[str, ModelSpec],
) -> pd.DataFrame:
    rows = []
    for dataset, batches in datasets.items():
        for batch in batches:
            for name, spec in models.items():
                prediction = spec.model.predict(batch, speed_cap=spec.speed_cap)
                rows.append(
                    {
                        "dataset": dataset,
                        "run": batch.run,
                        "model": name,
                        "parameters": spec.parameters,
                        "samples": len(batch.target),
                        **velocity_metrics(batch.target, prediction),
                    }
                )
            print(f"reviewer analysis: {dataset} / {batch.run}", flush=True)
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["dataset", "model"], sort=False)
        .agg(
            parameters=("parameters", "first"),
            runs=("run", "nunique"),
            mean_rmse_mps=("vector_rmse_mps", "mean"),
            sd_rmse_mps=("vector_rmse_mps", "std"),
            mean_displacement_mae_m=("displacement_mae_m", "mean"),
            mean_speed_mae_mps=("speed_mae_mps", "mean"),
            mean_angle_deg=("median_angle_deg", "mean"),
        )
        .reset_index()
    )


def paired_statistics(
    metrics: pd.DataFrame,
    primary: str,
    comparators: tuple[str, ...],
    seed: int,
) -> dict:
    result: dict[str, dict] = {}
    for dataset, subset in metrics.groupby("dataset", sort=False):
        pivot = subset.pivot(index="run", columns="model", values="vector_rmse_mps")
        result[dataset] = {}
        for comparator in comparators:
            if comparator not in pivot or primary not in pivot:
                continue
            difference = (pivot[primary] - pivot[comparator]).dropna().to_numpy()
            result[dataset][comparator] = {
                "runs": len(difference),
                "mean_primary_minus_comparator_mps": float(np.mean(difference)),
                "run_bootstrap_95_ci_mps": run_bootstrap_interval(
                    difference, repetitions=20000, seed=seed
                ),
                "exact_paired_randomization_p": exact_paired_randomization_pvalue(difference),
                "runs_primary_lower": int(np.sum(difference < 0.0)),
                "minimum_attainable_two_sided_p": float(2.0 ** (1 - len(difference))),
            }
    return result


def build_datasets(project: Path, config: dict, splits: dict) -> tuple[dict, dict]:
    corridor = project / "data" / "raw" / "2013bidirectional"
    crossing = project / "data" / "raw" / "2013crossing90" / "trajectories"
    method = config["primary_goal_method"]
    training = {
        run: build_batch(corridor / f"{run}.txt", config, method)
        for run in splits["calibration"]
    }
    datasets = {
        "corridor held-out": [
            build_batch(corridor / f"{run}.txt", config, method)
            for run in splits["held_out_validation"]
        ],
        "altered geometry": [
            build_batch(corridor / f"{run}.txt", config, method)
            for run in splits["held_out_geometry_stress"]
        ],
        "crossing topology": [
            build_batch(source, config, method, crossing=True)
            for source in sorted(crossing.glob("crossing_90_[de]_*.txt"))
        ],
    }
    return training, datasets


def attribution_models(
    training: list[VelocitySamples], selected: dict, config: dict
) -> dict[str, ModelSpec]:
    models = fit_models(training, selected, config)
    print("fitted primary direct/tensor/GEL-Ped models", flush=True)
    hidden = tuple(selected["mlp_configuration"]["hidden_layers"])
    alpha = selected["mlp_configuration"]["alpha"]
    epochs = selected["residual_epoch"]
    cap = selected["speed_cap_mps"]
    direct_parameters = mlp_parameter_count(hidden)
    residual_parameters = mlp_parameter_count(hidden, inputs=14)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        constant_residual = fit_residual_on_base_mlp(
            training,
            models["constant_velocity"].model,
            hidden_layers=hidden,
            alpha=alpha,
            max_iter=epochs,
            seed=config["seed"],
        )
        print("fitted constant-velocity residual", flush=True)
        linear_residual = fit_residual_on_base_mlp(
            training,
            models["unrestricted_ten_scalar"].model,
            hidden_layers=hidden,
            alpha=alpha,
            max_iter=epochs,
            seed=config["seed"],
        )
        print("fitted unrestricted-linear residual", flush=True)
        social_residual = fit_residual_on_base_mlp(
            training,
            models["social_force"].model,
            hidden_layers=hidden,
            alpha=alpha,
            max_iter=epochs,
            seed=config["seed"],
        )
        print("fitted Social Force residual", flush=True)
        raw_neighbour = fit_raw_neighbour_mlp(
            training,
            hidden_layers=(112, 112),
            alpha=alpha,
            max_iter=selected["mlp_configuration"]["selected_epoch"],
            seed=config["seed"],
        )
        print("fitted raw-neighbour MLP", flush=True)
    direct = models["interaction_mlp"].model
    direct_linear_skip = fit_linear_skip(training, direct, penalty=0.01)
    anisotropic_sfm = fit_anisotropic_social_force_response_model(training)
    print("fitted anisotropic Social Force model", flush=True)
    ttc = fit_ttc_response_model(training)
    print("fitted TTC-response model", flush=True)
    avm = fit_avm_model(training)
    print("fitted Anticipation Velocity Model", flush=True)
    weight = selected["direct_blend_weight"]
    return {
        "constant_velocity": models["constant_velocity"],
        "social_force_isotropic": models["social_force"],
        "social_force_anisotropic": ModelSpec(anisotropic_sfm, cap, 6),
        "anticipation_velocity_model": ModelSpec(avm, cap, 6),
        "time_to_collision_response": ModelSpec(ttc, cap, 4),
        "unrestricted_linear": models["unrestricted_ten_scalar"],
        "direct_mlp": models["interaction_mlp"],
        "raw_neighbour_mlp": ModelSpec(raw_neighbour, cap, 17922),
        "direct_mlp_linear_skip": ModelSpec(
            direct_linear_skip, cap, direct_parameters + 22
        ),
        "constant_residual": ModelSpec(
            constant_residual, cap, residual_parameters
        ),
        "linear_residual": ModelSpec(
            linear_residual, cap, residual_parameters + 22
        ),
        "social_force_residual": ModelSpec(
            social_residual, cap, residual_parameters + 4
        ),
        "tensor_prior": models["tensor_response"],
        "tensor_residual": models["tensor_residual"],
        "gel_ped": models["gel_ped"],
        "direct_plus_constant_residual": ModelSpec(
            GELPedRegressor(direct, constant_residual, weight),
            cap,
            direct_parameters + residual_parameters,
        ),
        "direct_plus_linear_residual": ModelSpec(
            GELPedRegressor(direct, linear_residual, weight),
            cap,
            direct_parameters + residual_parameters + 22,
        ),
        "direct_plus_social_residual": ModelSpec(
            GELPedRegressor(direct, social_residual, weight),
            cap,
            direct_parameters + residual_parameters + 4,
        ),
    }


def wall_confound_analysis(
    training: dict[str, VelocitySamples],
    datasets: dict[str, list[VelocitySamples]],
    selected: dict,
    config: dict,
    processed: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    zero_training = [without_wall(batch) for batch in training.values()]
    zero_datasets = {
        name: [without_wall(batch) for batch in batches] for name, batches in datasets.items()
    }
    no_wall_models = fit_models(zero_training, selected, config)
    headline = {
        "constant_velocity": no_wall_models["constant_velocity"],
        "social_force": no_wall_models["social_force"],
        "unrestricted_linear": no_wall_models["unrestricted_ten_scalar"],
        "direct_mlp": no_wall_models["interaction_mlp"],
        "tensor_prior": no_wall_models["tensor_response"],
        "tensor_residual": no_wall_models["tensor_residual"],
        "gel_ped": no_wall_models["gel_ped"],
    }
    refit = run_metrics(zero_datasets, headline)
    refit["condition"] = "retrained_without_wall"
    original = fit_models(list(training.values()), selected, config)
    inference_models = {
        "direct_mlp": original["interaction_mlp"],
        "tensor_prior": original["tensor_response"],
        "gel_ped": original["gel_ped"],
    }
    corridor_only = {
        name: batches for name, batches in zero_datasets.items() if name != "crossing topology"
    }
    inference = run_metrics(corridor_only, inference_models)
    inference["condition"] = "wall_zeroed_at_inference"
    all_metrics = pd.concat((refit, inference), ignore_index=True)
    all_metrics.to_csv(processed / "reviewer_wall_confound_metrics.csv", index=False)
    wall_summary = (
        all_metrics.groupby(["condition", "dataset", "model"], sort=False)
        .agg(
            runs=("run", "nunique"),
            mean_rmse_mps=("vector_rmse_mps", "mean"),
            sd_rmse_mps=("vector_rmse_mps", "std"),
        )
        .reset_index()
    )
    wall_summary.to_csv(processed / "reviewer_wall_confound_summary.csv", index=False)
    return all_metrics, wall_summary


def minimum_ttc(batch: VelocitySamples, radius: float = 0.25) -> np.ndarray:
    if batch.raw_neighbours is None:
        return np.full(len(batch.target), np.inf)
    raw = batch.raw_neighbours
    position = raw[:, :, :2]
    velocity = raw[:, :, 2:4]
    present = raw[:, :, 4] > 0.5
    a = np.sum(velocity**2, axis=2)
    b = 2.0 * np.sum(position * velocity, axis=2)
    c = np.sum(position**2, axis=2) - (2.0 * radius) ** 2
    discriminant = b**2 - 4.0 * a * c
    valid = present & (a > 1e-9) & (discriminant >= 0.0)
    root = np.divide(
        -b - np.sqrt(np.maximum(discriminant, 0.0)),
        2.0 * a,
        out=np.full_like(a, np.inf),
        where=a > 1e-9,
    )
    root[~valid | (root <= 0.0)] = np.inf
    return root.min(axis=1)


def stratified_errors(
    datasets: dict[str, list[VelocitySamples]], models: dict[str, ModelSpec]
) -> pd.DataFrame:
    rows = []
    for dataset, batches in datasets.items():
        occupancy_values = np.concatenate(
            [batch.occupancy for batch in batches if batch.occupancy is not None]
        )
        occupancy_edges = np.unique(np.quantile(occupancy_values, [0.0, 1 / 3, 2 / 3, 1.0]))
        for batch in batches:
            neighbour = np.asarray(batch.neighbour_count)
            occupancy = np.asarray(batch.occupancy)
            ttc = minimum_ttc(batch)
            bins = {
                "neighbours": np.select(
                    [neighbour <= 1, neighbour <= 3], ["0-1", "2-3"], default="4+"
                ),
                "occupancy": pd.cut(
                    occupancy,
                    bins=occupancy_edges,
                    labels=[f"Q{i + 1}" for i in range(len(occupancy_edges) - 1)],
                    include_lowest=True,
                    duplicates="drop",
                ).astype(str),
                "ttc": np.select(
                    [ttc < 1.0, ttc < 2.0, np.isfinite(ttc)],
                    ["<1 s", "1-2 s", ">=2 s"],
                    default="no predicted contact",
                ),
            }
            for model_name, spec in models.items():
                prediction = spec.model.predict(batch, speed_cap=spec.speed_cap)
                squared = np.sum((prediction - batch.target) ** 2, axis=1)
                for variable, labels in bins.items():
                    for label in np.unique(labels):
                        mask = labels == label
                        if not np.any(mask):
                            continue
                        rows.append(
                            {
                                "dataset": dataset,
                                "run": batch.run,
                                "stratum": variable,
                                "level": label,
                                "model": model_name,
                                "samples": int(mask.sum()),
                                "vector_rmse_mps": float(np.sqrt(np.mean(squared[mask]))),
                            }
                        )
    return pd.DataFrame(rows)


def smoothed_loader(base_loader):
    def load(path):
        data = base_loader(path)
        result = data.copy()
        for _, indices in result.groupby("pedestrian_id", sort=False).groups.items():
            index = np.asarray(list(indices))
            count = len(index)
            window = min(11, count if count % 2 == 1 else count - 1)
            if window < 5:
                continue
            result.loc[index, "x_m"] = savgol_filter(result.loc[index, "x_m"], window, 2)
            result.loc[index, "y_m"] = savgol_filter(result.loc[index, "y_m"], window, 2)
        return result

    return load


def build_smoothed_batch(
    source: Path, config: dict, *, crossing: bool = False
) -> VelocitySamples:
    loader = load_julich_crossing_trajectory if crossing else load_julich_trajectory
    return build_velocity_samples(
        source,
        interaction_range=config["interaction_range_m"],
        horizon_frames=config["horizon_frames"],
        frame_stride=config["frame_stride"],
        maximum_samples=config["maximum_samples_per_run"],
        radius=config["pedestrian_radius_m"],
        wall_range=config["wall_range_m"],
        wall_y_bounds=None if crossing else (0.0, 4.0),
        neighbour_cutoff=config["neighbour_cutoff_m"],
        goal_method=config["primary_goal_method"],
        speed_outlier_threshold=config["speed_outlier_threshold_mps"],
        loader=smoothed_loader(loader),
        seed=config["seed"],
    )


def smoothing_sensitivity(
    project: Path, config: dict, splits: dict, selected: dict, processed: Path
) -> pd.DataFrame:
    corridor = project / "data" / "raw" / "2013bidirectional"
    crossing = project / "data" / "raw" / "2013crossing90" / "trajectories"
    training = [
        build_smoothed_batch(corridor / f"{run}.txt", config)
        for run in splits["calibration"]
    ]
    models = fit_models(training, selected, config)
    selected_models = {
        "direct_mlp": models["interaction_mlp"],
        "tensor_prior": models["tensor_response"],
        "gel_ped": models["gel_ped"],
    }
    datasets = {
        "corridor held-out": [
            build_smoothed_batch(corridor / f"{run}.txt", config)
            for run in splits["held_out_validation"]
        ],
        "altered geometry": [
            build_smoothed_batch(corridor / f"{run}.txt", config)
            for run in splits["held_out_geometry_stress"]
        ],
        "crossing topology": [
            build_smoothed_batch(source, config, crossing=True)
            for source in sorted(crossing.glob("crossing_90_[de]_*.txt"))
        ],
    }
    metrics = run_metrics(datasets, selected_models)
    metrics.to_csv(processed / "reviewer_smoothing_sensitivity_metrics.csv", index=False)
    summarize(metrics).to_csv(
        processed / "reviewer_smoothing_sensitivity_summary.csv", index=False
    )
    return metrics


def route_accuracy_and_error(
    project: Path,
    config: dict,
    splits: dict,
    datasets: dict[str, list[VelocitySamples]],
    models: dict[str, ModelSpec],
    processed: Path,
) -> pd.DataFrame:
    corridor = project / "data" / "raw" / "2013bidirectional"
    crossing = project / "data" / "raw" / "2013crossing90" / "trajectories"
    sources = {
        **{
            run: (corridor / f"{run}.txt", False)
            for run in (*splits["held_out_validation"], *splits["held_out_geometry_stress"])
        },
        **{source.stem: (source, True) for source in crossing.glob("crossing_90_[de]_*.txt")},
    }
    rows = []
    for dataset, batches in datasets.items():
        for entry in batches:
            source, crossing_flag = sources[entry.run]
            endpoint = build_batch(
                source, config, "endpoint", crossing=crossing_flag
            )
            endpoint_frame = pd.DataFrame(
                {
                    "pedestrian_id": endpoint.pedestrian_id,
                    "frame": endpoint.frame,
                    "endpoint_x": endpoint.features[:, 1, 0],
                    "endpoint_y": endpoint.features[:, 1, 1],
                }
            )
            entry_frame = pd.DataFrame(
                {
                    "pedestrian_id": entry.pedestrian_id,
                    "frame": entry.frame,
                    "entry_x": entry.features[:, 1, 0],
                    "entry_y": entry.features[:, 1, 1],
                    "sample_index": np.arange(len(entry.target)),
                }
            )
            merged = entry_frame.merge(endpoint_frame, on=["pedestrian_id", "frame"])
            dot = merged["entry_x"] * merged["endpoint_x"] + merged["entry_y"] * merged["endpoint_y"]
            agreement = dot > 0.99
            for model_name, spec in models.items():
                prediction = spec.model.predict(entry, speed_cap=spec.speed_cap)
                squared = np.sum((prediction - entry.target) ** 2, axis=1)
                aligned = squared[merged["sample_index"].to_numpy()]
                for label, mask in (("agrees", agreement), ("disagrees", ~agreement)):
                    if not np.any(mask):
                        continue
                    rows.append(
                        {
                            "dataset": dataset,
                            "run": entry.run,
                            "route_classification": label,
                            "samples": int(np.sum(mask)),
                            "sample_agreement_rate": float(np.mean(agreement)),
                            "model": model_name,
                            "vector_rmse_mps": float(np.sqrt(np.mean(aligned[mask]))),
                        }
                    )
    results = pd.DataFrame(rows)
    results.to_csv(processed / "reviewer_route_accuracy_error.csv", index=False)
    return results


def seed_summary(processed: Path) -> pd.DataFrame:
    metrics = pd.read_csv(processed / "neural_seed_sensitivity_metrics.csv")
    summary = (
        metrics.groupby(["seed", "dataset", "model"], sort=False)["vector_rmse_mps"]
        .mean()
        .reset_index()
        .groupby(["dataset", "model"], sort=False)["vector_rmse_mps"]
        .agg(seed_mean_rmse_mps="mean", seed_sd_rmse_mps="std", minimum="min", maximum="max")
        .reset_index()
    )
    summary.to_csv(processed / "reviewer_neural_seed_mean_sd.csv", index=False)
    return summary


def main() -> None:
    project = project_root()
    processed = project / "data" / "processed"
    config, splits = load_inputs(project)
    selected = json.loads(
        (processed / "selected_hyperparameters.json").read_text(encoding="utf-8")
    )
    training, datasets = build_datasets(project, config, splits)
    models = attribution_models(list(training.values()), selected, config)
    metrics = run_metrics(datasets, models)
    metrics.to_csv(processed / "reviewer_attribution_metrics.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(processed / "reviewer_attribution_summary.csv", index=False)
    statistics = paired_statistics(
        metrics,
        "gel_ped",
        (
            "direct_mlp",
            "unrestricted_linear",
            "linear_residual",
            "direct_plus_linear_residual",
            "direct_mlp_linear_skip",
            "raw_neighbour_mlp",
            "social_force_anisotropic",
            "anticipation_velocity_model",
            "time_to_collision_response",
        ),
        config["seed"],
    )
    (processed / "reviewer_attribution_statistics.json").write_text(
        json.dumps(statistics, indent=2), encoding="utf-8"
    )
    wall_confound_analysis(training, datasets, selected, config, processed)
    subgroup_models = {
        "direct_mlp": models["direct_mlp"],
        "tensor_prior": models["tensor_prior"],
        "gel_ped": models["gel_ped"],
    }
    strata = stratified_errors(datasets, subgroup_models)
    strata.to_csv(processed / "reviewer_stratified_errors.csv", index=False)
    route_accuracy_and_error(
        project, config, splits, datasets, subgroup_models, processed
    )
    smoothing_sensitivity(project, config, splits, selected, processed)
    seed_summary(processed)
    print("reviewer revision analysis complete", flush=True)


if __name__ == "__main__":
    main()
