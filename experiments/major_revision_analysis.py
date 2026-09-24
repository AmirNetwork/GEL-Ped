# Author: Amir Ghorbani
"""Analyses added for the major-revision round.

This entry point resolves route-information leakage, performs run-wise inner model
selection, expands baseline/feature ablations, quantifies small-sample uncertainty and
coefficient stability, documents distribution shift, and measures model-head and full
scene feature-processing time.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning

from pedgeom.benchmarks import (
    GELPedRegressor,
    GoalStableRegressor,
    fit_interaction_mlp,
    fit_interaction_mlp_with_run_stopping,
    fit_invariant_ridge,
    fit_tensor_residual_mlp,
    fit_unrestricted_tensor_features,
)
from pedgeom.calibration import (
    FEATURE_NAMES,
    FittedVelocityModel,
    VelocitySamples,
    _interaction_features,
    build_velocity_samples,
    fit_direction_speed_model,
    fit_social_force_response_model,
    fit_tensor_geometry_model,
    fit_velocity_model,
    proximity_metrics,
    velocity_metrics,
)
from pedgeom.datasets import (
    add_motion_features,
    add_prediction_goal_directions,
    load_julich_crossing_trajectory,
    load_julich_trajectory,
)
from pedgeom.statistics import (
    exact_paired_randomization_pvalue,
    exact_sign_test_pvalue,
    hierarchical_rmse_difference_interval,
    run_bootstrap_interval,
)


PARALLEL_NAMES = ("persistence", "goal", "isotropic", "forward", "closing")
LATERAL_NAMES = ("persistence", "isotropic", "forward", "closing", "wall")


@dataclass(frozen=True)
class ModelSpec:
    model: object
    speed_cap: float
    parameters: int


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_inputs(project: Path) -> tuple[dict, dict]:
    config = json.loads((project / "configs" / "major_revision.json").read_text(encoding="utf-8"))
    splits = json.loads((project / "data" / "splits.json").read_text(encoding="utf-8"))
    return config, splits


def build_batch(
    source: Path,
    config: dict,
    goal_method: str,
    *,
    crossing: bool = False,
    interaction_range: float | None = None,
    neighbour_cutoff: float | None = None,
) -> VelocitySamples:
    return build_velocity_samples(
        source,
        interaction_range=(
            config["interaction_range_m"] if interaction_range is None else interaction_range
        ),
        horizon_frames=config["horizon_frames"],
        frame_stride=config["frame_stride"],
        maximum_samples=config["maximum_samples_per_run"],
        radius=config["pedestrian_radius_m"],
        wall_range=config["wall_range_m"],
        wall_y_bounds=None if crossing else (0.0, 4.0),
        neighbour_cutoff=(
            config["neighbour_cutoff_m"] if neighbour_cutoff is None else neighbour_cutoff
        ),
        goal_method=goal_method,
        speed_outlier_threshold=config["speed_outlier_threshold_mps"],
        loader=load_julich_crossing_trajectory if crossing else load_julich_trajectory,
        seed=config["seed"],
    )


def mlp_parameter_count(hidden_layers: tuple[int, ...], inputs: int = 12, outputs: int = 2) -> int:
    sizes = (inputs, *hidden_layers, outputs)
    return int(sum(left * right + right for left, right in zip(sizes[:-1], sizes[1:], strict=True)))


def subsample_batch(batch: VelocitySamples, maximum_samples: int) -> VelocitySamples:
    """Deterministically thin a batch for inner MLP selection only."""

    if len(batch.target) <= maximum_samples:
        return batch
    indices = np.linspace(0, len(batch.target) - 1, maximum_samples, dtype=int)

    def optional(values: np.ndarray | None) -> np.ndarray | None:
        return None if values is None else values[indices]

    return VelocitySamples(
        batch.run,
        batch.features[indices],
        batch.target[indices],
        batch.pedestrian_id[indices],
        batch.frame[indices],
        optional(batch.position),
        optional(batch.neighbour_count),
        optional(batch.occupancy),
        batch.goal_method,
    )


def select_hyperparameters(
    training: dict[str, VelocitySamples],
    config: dict,
    processed: Path,
) -> dict:
    """Leave one complete calibration run out for every data-dependent choice."""

    rows: list[dict] = []
    runs = list(training)
    mlp_training = {
        run: subsample_batch(batch, config["mlp_inner_maximum_samples_per_run"])
        for run, batch in training.items()
    }
    for held_out in runs:
        inner = [training[run] for run in runs if run != held_out]
        validation = training[held_out]
        tensor = fit_tensor_geometry_model(inner)
        for cap in config["speed_cap_candidates_mps"]:
            rows.append(
                {
                    "selection": "speed_cap",
                    "candidate": f"{cap:g}",
                    "held_out_run": held_out,
                    "vector_rmse_mps": velocity_metrics(
                        validation.target, tensor.predict(validation, speed_cap=cap)
                    )["vector_rmse_mps"],
                }
            )
        for penalty in config["ridge_penalties"]:
            ridge = fit_invariant_ridge(inner, penalty=penalty)
            rows.append(
                {
                    "selection": "ridge_penalty",
                    "candidate": f"{penalty:g}",
                    "held_out_run": held_out,
                    "vector_rmse_mps": velocity_metrics(
                        validation.target, ridge.predict(validation, speed_cap=100.0)
                    )["vector_rmse_mps"],
                }
            )
        for index, candidate in enumerate(config["mlp_candidates"]):
            mlp, best_epoch, history = fit_interaction_mlp_with_run_stopping(
                [mlp_training[run] for run in runs if run != held_out],
                mlp_training[held_out],
                hidden_layers=tuple(candidate["hidden_layers"]),
                alpha=candidate["alpha"],
                maximum_epochs=candidate["maximum_epochs"],
                patience=config["mlp_run_stopping_patience"],
                seed=config["seed"],
            )
            rows.append(
                {
                    "selection": "mlp_configuration",
                    "candidate": str(index),
                    "held_out_run": held_out,
                    "best_epoch": best_epoch,
                    "epochs_trained": len(history),
                    "vector_rmse_mps": velocity_metrics(
                        mlp_training[held_out].target,
                        mlp.predict(mlp_training[held_out], speed_cap=100.0),
                    )["vector_rmse_mps"],
                }
            )
            print(
                f"run-stopped MLP {index + 1}/{len(config['mlp_candidates'])}, "
                f"validation run {held_out}",
                flush=True,
            )

    results = pd.DataFrame(rows)
    results.to_csv(processed / "model_selection_inner_run_cv.csv", index=False)
    means = results.groupby(["selection", "candidate"])["vector_rmse_mps"].mean()
    selected_cap = float(means.xs("speed_cap").idxmin())
    selected_ridge = float(means.xs("ridge_penalty").idxmin())
    selected_mlp_index = int(means.xs("mlp_configuration").idxmin())
    selected_epochs = results[
        (results["selection"] == "mlp_configuration")
        & (results["candidate"] == str(selected_mlp_index))
    ]["best_epoch"]
    selected_mlp_epoch = int(np.median(selected_epochs))
    mlp_configuration = dict(config["mlp_candidates"][selected_mlp_index])
    mlp_configuration["selected_epoch"] = selected_mlp_epoch
    for held_out in runs:
        inner = [mlp_training[run] for run in runs if run != held_out]
        validation = mlp_training[held_out]
        tensor = fit_tensor_geometry_model(inner)
        for epoch in config["residual_epoch_candidates"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                hybrid = fit_tensor_residual_mlp(
                    inner,
                    tensor,
                    hidden_layers=tuple(mlp_configuration["hidden_layers"]),
                    alpha=mlp_configuration["alpha"],
                    max_iter=epoch,
                    seed=config["seed"],
                    support_quantile=config["residual_support_quantile"],
                    gate_strength=0.0,
                )
            for gate_strength in config["residual_gate_strength_candidates"]:
                candidate = replace(hybrid, gate_strength=float(gate_strength))
                rows.append(
                    {
                        "selection": "residual_epoch_gate",
                        "candidate": f"{epoch}|{gate_strength:g}",
                        "held_out_run": held_out,
                        "best_epoch": epoch,
                        "epochs_trained": epoch,
                        "vector_rmse_mps": velocity_metrics(
                            validation.target,
                            candidate.predict(validation, speed_cap=100.0),
                        )["vector_rmse_mps"],
                    }
                )
        print(f"tensor-residual epoch/gate search, validation run {held_out}", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(processed / "model_selection_inner_run_cv.csv", index=False)
    residual_means = results[results["selection"] == "residual_epoch_gate"].groupby(
        "candidate"
    )["vector_rmse_mps"].mean()
    selected_residual_candidate = residual_means.idxmin()
    epoch_text, gate_text = selected_residual_candidate.split("|")
    selected_residual_epoch = int(epoch_text)
    selected_residual_gate_strength = float(gate_text)
    for held_out in runs:
        inner = [mlp_training[run] for run in runs if run != held_out]
        validation = mlp_training[held_out]
        tensor = fit_tensor_geometry_model(inner)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            direct = fit_interaction_mlp(
                inner,
                hidden_layers=tuple(mlp_configuration["hidden_layers"]),
                alpha=mlp_configuration["alpha"],
                max_iter=mlp_configuration["selected_epoch"],
                seed=config["seed"],
            )
            structured = fit_tensor_residual_mlp(
                inner,
                tensor,
                hidden_layers=tuple(mlp_configuration["hidden_layers"]),
                alpha=mlp_configuration["alpha"],
                max_iter=selected_residual_epoch,
                seed=config["seed"],
                support_quantile=config["residual_support_quantile"],
                gate_strength=selected_residual_gate_strength,
            )
        direct_prediction = direct.predict(validation, speed_cap=100.0)
        structured_prediction = structured.predict(validation, speed_cap=100.0)
        for weight in config["blend_weight_candidates"]:
            prediction = weight * direct_prediction + (1.0 - weight) * structured_prediction
            rows.append(
                {
                    "selection": "direct_blend_weight",
                    "candidate": f"{weight:g}",
                    "held_out_run": held_out,
                    "vector_rmse_mps": velocity_metrics(validation.target, prediction)[
                        "vector_rmse_mps"
                    ],
                }
            )
        print(f"dual-expert blend search, validation run {held_out}", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(processed / "model_selection_inner_run_cv.csv", index=False)
    blend_means = results[results["selection"] == "direct_blend_weight"].groupby(
        "candidate"
    )["vector_rmse_mps"].mean()
    selected_blend_weight = float(blend_means.idxmin())
    selected = {
        "speed_cap_mps": selected_cap,
        "ridge_penalty": selected_ridge,
        "mlp_configuration_index": selected_mlp_index,
        "mlp_configuration": mlp_configuration,
        "residual_epoch": selected_residual_epoch,
        "residual_gate_strength": selected_residual_gate_strength,
        "direct_blend_weight": selected_blend_weight,
        "criterion": (
            "ridge and cap: lowest mean leave-one-calibration-run-out RMSE; "
            "MLP: lowest mean complete-run cross-validation RMSE, with the epoch count "
            "set to the median early-stopped epoch across calibration folds; residual "
            "epoch, candidate residual-shrinkage strength, and convex expert weight: lowest mean "
            "complete-run cross-validation RMSE"
        ),
        "independent_unit": "complete experimental run for every selection decision",
    }
    (processed / "selected_hyperparameters.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    return selected


def fit_models(
    training: list[VelocitySamples],
    selected: dict,
    config: dict,
) -> dict[str, ModelSpec]:
    cap = selected["speed_cap_mps"]
    tensor = fit_tensor_geometry_model(training)
    mlp_config = selected["mlp_configuration"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp = fit_interaction_mlp(
            training,
            hidden_layers=tuple(mlp_config["hidden_layers"]),
            alpha=mlp_config["alpha"],
            max_iter=mlp_config["selected_epoch"],
            early_stopping=False,
            seed=config["seed"],
        )
        tensor_residual = fit_tensor_residual_mlp(
            training,
            tensor,
            hidden_layers=tuple(mlp_config["hidden_layers"]),
            alpha=mlp_config["alpha"],
            max_iter=selected["residual_epoch"],
            seed=config["seed"],
            support_quantile=config["residual_support_quantile"],
            gate_strength=selected["residual_gate_strength"],
        )
    return {
        "constant_velocity": ModelSpec(
            FittedVelocityModel(("persistence",), np.array([1.0])), cap, 0
        ),
        "social_force": ModelSpec(fit_social_force_response_model(training), cap, 4),
        "isotropic_constrained": ModelSpec(
            fit_velocity_model(training, ("persistence", "goal", "isotropic", "wall")),
            cap,
            4,
        ),
        "isotropic_unconstrained": ModelSpec(
            fit_velocity_model(
                training,
                ("persistence", "goal", "isotropic", "wall"),
                coefficient_bounds=(-5.0, 5.0),
            ),
            cap,
            4,
        ),
        "shared_goal_frame": ModelSpec(
            fit_velocity_model(
                training,
                FEATURE_NAMES,
                coefficient_bounds=(-5.0, 5.0),
            ),
            cap,
            6,
        ),
        "decoupled_scalar_geometry": ModelSpec(fit_direction_speed_model(training), cap, 8),
        "tensor_persistence_only": ModelSpec(
            fit_tensor_geometry_model(
                training,
                parallel_names=("persistence",),
                lateral_names=("persistence",),
            ),
            cap,
            2,
        ),
        "tensor_no_isotropic": ModelSpec(
            fit_tensor_geometry_model(
                training,
                parallel_names=("persistence", "goal", "forward", "closing"),
                lateral_names=("persistence", "forward", "closing", "wall"),
            ),
            cap,
            8,
        ),
        "tensor_no_forward": ModelSpec(
            fit_tensor_geometry_model(
                training,
                parallel_names=("persistence", "goal", "isotropic", "closing"),
                lateral_names=("persistence", "isotropic", "closing", "wall"),
            ),
            cap,
            8,
        ),
        "tensor_no_closing": ModelSpec(
            fit_tensor_geometry_model(
                training,
                parallel_names=("persistence", "goal", "isotropic", "forward"),
                lateral_names=("persistence", "isotropic", "forward", "wall"),
            ),
            cap,
            8,
        ),
        "tensor_no_wall": ModelSpec(
            fit_tensor_geometry_model(
                training,
                parallel_names=PARALLEL_NAMES,
                lateral_names=("persistence", "isotropic", "forward", "closing"),
            ),
            cap,
            9,
        ),
        "tensor_response": ModelSpec(tensor, cap, 10),
        "tensor_without_cap": ModelSpec(tensor, 100.0, 10),
        "unrestricted_ten_scalar": ModelSpec(
            fit_unrestricted_tensor_features(training, penalty=0.0), cap, 22
        ),
        "invariant_ridge": ModelSpec(
            fit_invariant_ridge(training, penalty=selected["ridge_penalty"]), cap, 26
        ),
        "interaction_mlp": ModelSpec(
            mlp,
            cap,
            mlp_parameter_count(tuple(mlp_config["hidden_layers"])),
        ),
        "goal_stable_neural": ModelSpec(
            GoalStableRegressor(mlp),
            cap,
            mlp_parameter_count(tuple(mlp_config["hidden_layers"])),
        ),
        "tensor_residual": ModelSpec(
            tensor_residual,
            cap,
            10 + mlp_parameter_count(tuple(mlp_config["hidden_layers"]), inputs=14),
        ),
        "tensor_residual_ungated": ModelSpec(
            replace(tensor_residual, gate_strength=0.0),
            cap,
            10 + mlp_parameter_count(tuple(mlp_config["hidden_layers"]), inputs=14),
        ),
        "gel_ped": ModelSpec(
            GELPedRegressor(mlp, tensor_residual, selected["direct_blend_weight"]),
            cap,
            mlp_parameter_count(tuple(mlp_config["hidden_layers"]))
            + 10
            + mlp_parameter_count(tuple(mlp_config["hidden_layers"]), inputs=14),
        ),
    }


def evaluate_models(
    datasets: dict[str, list[VelocitySamples]],
    models: dict[str, ModelSpec],
    processed: Path,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    support_rows: list[dict] = []
    errors: dict[tuple[str, str, str], np.ndarray] = {}
    for dataset, batches in datasets.items():
        for batch in batches:
            print(f"evaluating {dataset}: {batch.run}", flush=True)
            for name, spec in models.items():
                prediction = spec.model.predict(batch, speed_cap=spec.speed_cap)
                if name == "tensor_residual":
                    gate = spec.model.residual_gate(batch)
                    support_rows.append(
                        {
                            "dataset": dataset,
                            "run": batch.run,
                            "mean_residual_gate": float(np.mean(gate)),
                            "median_residual_gate": float(np.median(gate)),
                            "fraction_residual_shrunk": float(np.mean(gate < 0.999999)),
                            "fraction_gate_below_half": float(np.mean(gate < 0.5)),
                        }
                    )
                errors[(dataset, batch.run, name)] = np.sum((prediction - batch.target) ** 2, axis=1)
                rows.append(
                    {
                        "dataset": dataset,
                        "run": batch.run,
                        "model": name,
                        "parameters": spec.parameters,
                        "speed_cap_mps": spec.speed_cap,
                        **velocity_metrics(batch.target, prediction),
                        **proximity_metrics(batch, prediction),
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(processed / "major_revision_metrics.csv", index=False)
    pd.DataFrame(support_rows).to_csv(processed / "residual_support_diagnostics.csv", index=False)
    summary = metrics.groupby(["dataset", "model"], sort=False).agg(
        parameters=("parameters", "first"),
        vector_rmse_mps=("vector_rmse_mps", "mean"),
        sem_vector_rmse_mps=("vector_rmse_mps", "sem"),
        displacement_mae_m=("displacement_mae_m", "mean"),
        speed_mae_mps=("speed_mae_mps", "mean"),
        median_angle_deg=("median_angle_deg", "mean"),
        vector_r2=("vector_r2", "mean"),
        false_contact_rate=("false_contact_rate", "mean"),
    )
    summary.to_csv(processed / "major_revision_summary.csv")

    primary_model = "gel_ped"
    comparator_names = (
        "constant_velocity",
        "social_force",
        "tensor_response",
        "unrestricted_ten_scalar",
        "invariant_ridge",
        "interaction_mlp",
        "goal_stable_neural",
        "tensor_residual",
    )
    confirmatory_comparators = (
        "constant_velocity",
        "social_force",
        "unrestricted_ten_scalar",
        "interaction_mlp",
        "goal_stable_neural",
    )
    comparisons: dict[str, dict] = {}
    for dataset in datasets:
        subset = metrics[metrics["dataset"] == dataset]
        pivot = subset.pivot(index="run", columns="model", values="vector_rmse_mps")
        comparisons[dataset] = {}
        raw_p_values: dict[str, float] = {}
        for comparator in comparator_names:
            difference = pivot[primary_model] - pivot[comparator]
            hierarchical_a = {
                run: errors[(dataset, run, primary_model)] for run in pivot.index
            }
            hierarchical_b = {run: errors[(dataset, run, comparator)] for run in pivot.index}
            randomization_p = exact_paired_randomization_pvalue(difference.to_numpy())
            raw_p_values[comparator] = randomization_p
            comparisons[dataset][comparator] = {
                "mean_rmse_difference_primary_minus_comparator_mps": float(difference.mean()),
                "mean_relative_rmse_reduction_percent": float(
                    np.mean(100.0 * (pivot[comparator] - pivot[primary_model]) / pivot[comparator])
                ),
                "run_bootstrap_95_ci_mps": run_bootstrap_interval(
                    difference.to_numpy(),
                    repetitions=config["run_bootstrap_repetitions"],
                    seed=config["seed"],
                ),
                "hierarchical_run_and_forecast_bootstrap_95_ci_mps": (
                    hierarchical_rmse_difference_interval(
                        hierarchical_a,
                        hierarchical_b,
                        repetitions=config["hierarchical_bootstrap_repetitions"],
                        seed=config["seed"],
                    )
                ),
                "exact_paired_randomization_p": randomization_p,
                "exact_sign_test_p": exact_sign_test_pvalue(difference.to_numpy()),
                "runs_primary_lower": int((difference < 0.0).sum()),
                "runs_total": int(len(difference)),
            }

        ordered = sorted(confirmatory_comparators, key=raw_p_values.get)
        running_max = 0.0
        adjusted: dict[str, float] = {}
        for rank, comparator in enumerate(ordered):
            candidate = min(1.0, (len(ordered) - rank) * raw_p_values[comparator])
            running_max = max(running_max, candidate)
            adjusted[comparator] = running_max
        for comparator, value in adjusted.items():
            comparisons[dataset][comparator]["holm_adjusted_randomization_p"] = value

        contact = subset.pivot(index="run", columns="model", values="false_contact_rate")
        for name in (
            "constant_velocity",
            "social_force",
            primary_model,
            "interaction_mlp",
            "goal_stable_neural",
        ):
            comparisons[dataset][f"false_contact_{name}"] = {
                "mean": float(contact[name].mean()),
                "run_bootstrap_95_ci": run_bootstrap_interval(
                    contact[name].to_numpy(),
                    repetitions=config["run_bootstrap_repetitions"],
                    seed=config["seed"],
                ),
            }
        contact_difference = contact[primary_model] - contact["constant_velocity"]
        comparisons[dataset]["false_contact_primary_minus_constant"] = {
            "mean_absolute_difference": float(contact_difference.mean()),
            "run_bootstrap_95_ci": run_bootstrap_interval(
                contact_difference.to_numpy(),
                repetitions=config["run_bootstrap_repetitions"],
                seed=config["seed"],
            ),
        }
    pooled = metrics[metrics["dataset"] != "calibration"].pivot(
        index=["dataset", "run"], columns="model", values="vector_rmse_mps"
    )
    pooled_results: dict[str, dict] = {}
    pooled_p_values: dict[str, float] = {}
    for comparator in comparator_names:
        relative_difference = 100.0 * (
            pooled[primary_model] - pooled[comparator]
        ) / pooled[comparator]
        randomization_p = exact_paired_randomization_pvalue(relative_difference.to_numpy())
        pooled_p_values[comparator] = randomization_p
        pooled_results[comparator] = {
            "mean_relative_rmse_difference_percent": float(relative_difference.mean()),
            "run_bootstrap_95_ci_percent": run_bootstrap_interval(
                relative_difference.to_numpy(),
                repetitions=config["run_bootstrap_repetitions"],
                seed=config["seed"],
            ),
            "exact_paired_randomization_p": randomization_p,
            "exact_sign_test_p": exact_sign_test_pvalue(relative_difference.to_numpy()),
            "runs_primary_lower": int((relative_difference < 0.0).sum()),
            "runs_total": int(len(relative_difference)),
        }
    ordered = sorted(confirmatory_comparators, key=pooled_p_values.get)
    running_max = 0.0
    for rank, comparator in enumerate(ordered):
        candidate = min(1.0, (len(ordered) - rank) * pooled_p_values[comparator])
        running_max = max(running_max, candidate)
        pooled_results[comparator]["holm_adjusted_randomization_p"] = running_max
    comparisons["held_out_pooled_relative_rmse"] = pooled_results
    (processed / "small_sample_and_contact_statistics.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )
    return metrics, comparisons


def neural_seed_sensitivity(
    training: list[VelocitySamples],
    datasets: dict[str, list[VelocitySamples]],
    selected: dict,
    config: dict,
    processed: Path,
) -> pd.DataFrame:
    """Repeat the learned heads across prespecified seeds without test-set selection."""

    tensor = fit_tensor_geometry_model(training)
    mlp_config = selected["mlp_configuration"]
    rows: list[dict] = []
    for offset in range(config["neural_seed_repetitions"]):
        seed = config["seed"] + offset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            direct = fit_interaction_mlp(
                training,
                hidden_layers=tuple(mlp_config["hidden_layers"]),
                alpha=mlp_config["alpha"],
                max_iter=mlp_config["selected_epoch"],
                seed=seed,
            )
            hybrid = fit_tensor_residual_mlp(
                training,
                tensor,
                hidden_layers=tuple(mlp_config["hidden_layers"]),
                alpha=mlp_config["alpha"],
                max_iter=selected["residual_epoch"],
                seed=seed,
                support_quantile=config["residual_support_quantile"],
                gate_strength=selected["residual_gate_strength"],
            )
            ensemble = GELPedRegressor(
                direct, hybrid, selected["direct_blend_weight"]
            )
        for dataset, batches in datasets.items():
            if dataset == "calibration":
                continue
            for batch in batches:
                for name, model in (
                    ("interaction_mlp", direct),
                    ("tensor_residual", hybrid),
                    ("gel_ped", ensemble),
                ):
                    rows.append(
                        {
                            "seed": seed,
                            "dataset": dataset,
                            "run": batch.run,
                            "model": name,
                            **velocity_metrics(batch.target, model.predict(batch, speed_cap=100.0)),
                        }
                    )
        print(f"neural seed sensitivity {offset + 1}/{config['neural_seed_repetitions']}", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(processed / "neural_seed_sensitivity_metrics.csv", index=False)
    by_seed = results.groupby(["seed", "dataset", "model"], as_index=False).agg(
        run_balanced_rmse_mps=("vector_rmse_mps", "mean")
    )
    summary = by_seed.groupby(["dataset", "model"], as_index=False).agg(
        median_rmse_mps=("run_balanced_rmse_mps", "median"),
        minimum_rmse_mps=("run_balanced_rmse_mps", "min"),
        maximum_rmse_mps=("run_balanced_rmse_mps", "max"),
    )
    summary.to_csv(processed / "neural_seed_sensitivity_summary.csv", index=False)
    return summary


def route_information_ablation(
    project: Path,
    config: dict,
    splits: dict,
    selected_cap: float,
    processed: Path,
) -> pd.DataFrame:
    corridor = project / "data" / "raw" / "2013bidirectional"
    crossing = project / "data" / "raw" / "2013crossing90" / "trajectories"
    rows: list[dict] = []
    for method in config["route_sensitivity_methods"]:
        print(f"route-information ablation: {method}", flush=True)
        training = [
            build_batch(corridor / f"{run}.txt", config, method)
            for run in splits["calibration"]
        ]
        model = fit_tensor_geometry_model(training)
        evaluations = {
            "corridor held-out": [
                build_batch(corridor / f"{run}.txt", config, method)
                for run in splits["held_out_validation"]
            ],
            "crossing external": [
                build_batch(source, config, method, crossing=True)
                for source in sorted(crossing.glob("crossing_90_[de]_*.txt"))
            ],
        }
        for dataset, batches in evaluations.items():
            for batch in batches:
                rows.append(
                    {
                        "goal_method": method,
                        "dataset": dataset,
                        "run": batch.run,
                        **velocity_metrics(
                            batch.target, model.predict(batch, speed_cap=selected_cap)
                        ),
                    }
                )
    results = pd.DataFrame(rows)
    results.to_csv(processed / "route_information_ablation.csv", index=False)
    results.groupby(["goal_method", "dataset"], sort=False).agg(
        vector_rmse_mps=("vector_rmse_mps", "mean"),
        sem_vector_rmse_mps=("vector_rmse_mps", "sem"),
        displacement_mae_m=("displacement_mae_m", "mean"),
        runs=("run", "nunique"),
    ).to_csv(processed / "route_information_ablation_summary.csv")
    return results


def field_sensitivity(
    project: Path,
    config: dict,
    splits: dict,
    selected_cap: float,
    processed: Path,
) -> pd.DataFrame:
    raw = project / "data" / "raw" / "2013bidirectional"
    rows: list[dict] = []
    candidates = [
        (interaction_range, config["neighbour_cutoff_m"], "interaction_range")
        for interaction_range in config["interaction_range_sensitivity_m"]
    ] + [
        (config["interaction_range_m"], cutoff, "neighbour_cutoff")
        for cutoff in config["neighbour_cutoff_sensitivity_m"]
    ]
    for interaction_range, cutoff, sensitivity in candidates:
        batches = {
            run: build_batch(
                raw / f"{run}.txt",
                config,
                config["primary_goal_method"],
                interaction_range=interaction_range,
                neighbour_cutoff=cutoff,
            )
            for run in splits["calibration"]
        }
        for held_out in splits["calibration"]:
            model = fit_tensor_geometry_model(
                [batch for run, batch in batches.items() if run != held_out]
            )
            rows.append(
                {
                    "sensitivity": sensitivity,
                    "interaction_range_m": interaction_range,
                    "neighbour_cutoff_m": cutoff,
                    "held_out_run": held_out,
                    "vector_rmse_mps": velocity_metrics(
                        batches[held_out].target,
                        model.predict(batches[held_out], speed_cap=selected_cap),
                    )["vector_rmse_mps"],
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(processed / "field_scale_and_cutoff_sensitivity.csv", index=False)
    return results


def _channel_arrays(
    batch: VelocitySamples, names: tuple[str, ...], channel: str
) -> tuple[np.ndarray, np.ndarray]:
    goal = batch.features[:, 1, :]
    lateral = np.column_stack((-goal[:, 1], goal[:, 0]))
    basis = goal if channel == "parallel" else lateral
    projected = np.einsum("nfc,nc->nf", batch.features, basis)
    indices = [FEATURE_NAMES.index(name) for name in names]
    target = np.sum(batch.target * basis, axis=1)
    return projected[:, indices], target


def coefficient_stability(
    training: dict[str, VelocitySamples],
    config: dict,
    processed: Path,
) -> tuple[pd.DataFrame, dict]:
    full = fit_tensor_geometry_model(list(training.values()))
    fold_rows: list[dict] = []
    for held_out in training:
        model = fit_tensor_geometry_model(
            [batch for run, batch in training.items() if run != held_out]
        )
        for channel, names, coefficients in (
            ("parallel", PARALLEL_NAMES, model.parallel_coefficients),
            ("lateral", LATERAL_NAMES, model.lateral_coefficients),
        ):
            for name, coefficient in zip(names, coefficients, strict=True):
                fold_rows.append(
                    {
                        "held_out_run": held_out,
                        "channel": channel,
                        "feature": name,
                        "coefficient": coefficient,
                    }
                )
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(processed / "coefficient_leave_one_run_out.csv", index=False)

    units = {
        "persistence": "dimensionless (m/s per m/s)",
        "goal": "m/s per unit goal vector",
        "isotropic": "m/s per unit dimensionless field",
        "forward": "m/s per unit dimensionless field",
        "closing": "dimensionless (m/s per m/s closing field)",
        "wall": "m/s per unit dimensionless field",
    }
    rng = np.random.default_rng(config["seed"])
    result_rows: list[dict] = []
    diagnostics: dict[str, dict] = {}
    for channel, names, full_coefficients in (
        ("parallel", PARALLEL_NAMES, full.parallel_coefficients),
        ("lateral", LATERAL_NAMES, full.lateral_coefficients),
    ):
        matrices = {}
        all_design = []
        all_target = []
        for run, batch in training.items():
            design, target = _channel_arrays(batch, names, channel)
            matrices[run] = (
                design.T @ design / len(design),
                design.T @ target / len(design),
            )
            all_design.append(design)
            all_target.append(target)
        design = np.concatenate(all_design)
        target = np.concatenate(all_target)
        scales = design.std(axis=0)
        active = scales > 1e-12
        standardized = (design[:, active] - design[:, active].mean(axis=0)) / scales[active]
        correlation = np.corrcoef(standardized, rowvar=False)
        off_diagonal = correlation - np.eye(np.count_nonzero(active))
        diagnostics[channel] = {
            "standardized_design_condition_number": float(np.linalg.cond(standardized)),
            "correlation_matrix_condition_number": float(np.linalg.cond(correlation)),
            "maximum_absolute_predictor_correlation": float(np.max(np.abs(off_diagonal))),
            "predictor_correlation_matrix": correlation.tolist(),
            "correlation_feature_order": [
                name for name, included in zip(names, active, strict=True) if included
            ],
            "constant_predictors_excluded_from_correlation": [
                name for name, included in zip(names, active, strict=True) if not included
            ],
        }
        run_names = list(training)
        bootstrap = np.empty((5000, len(names)), dtype=float)
        for replicate in range(len(bootstrap)):
            selected_runs = rng.choice(run_names, size=len(run_names), replace=True)
            xtx = sum((matrices[run][0] for run in selected_runs), start=np.zeros((len(names), len(names))))
            xty = sum((matrices[run][1] for run in selected_runs), start=np.zeros(len(names)))
            bootstrap[replicate] = np.clip(
                np.linalg.lstsq(xtx, xty, rcond=None)[0],
                config["tensor_coefficient_bounds"][0],
                config["tensor_coefficient_bounds"][1],
            )
        target_scale = max(float(target.std()), 1e-12)
        standardized_effect = full_coefficients * design.std(axis=0) / target_scale
        for index, name in enumerate(names):
            fold = folds[(folds["channel"] == channel) & (folds["feature"] == name)][
                "coefficient"
            ]
            low, high = np.quantile(bootstrap[:, index], (0.025, 0.975))
            result_rows.append(
                {
                    "channel": channel,
                    "feature": name,
                    "coefficient": full_coefficients[index],
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "leave_one_run_out_min": fold.min(),
                    "leave_one_run_out_max": fold.max(),
                    "leave_one_run_out_sign_retention": np.mean(
                        np.sign(fold) == np.sign(full_coefficients[index])
                    ),
                    "standardized_effect": standardized_effect[index],
                    "units": units[name],
                }
            )
    results = pd.DataFrame(result_rows)
    results.to_csv(processed / "coefficient_stability_summary.csv", index=False)
    diagnostics["bounds"] = {
        "lower": config["tensor_coefficient_bounds"][0],
        "upper": config["tensor_coefficient_bounds"][1],
        "minimum_distance_of_fitted_coefficient_to_bound": float(
            np.min(5.0 - np.abs(np.r_[full.parallel_coefficients, full.lateral_coefficients]))
        ),
    }
    (processed / "coefficient_design_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    return results, diagnostics


def distribution_shift_summary(
    project: Path,
    datasets: dict[str, list[VelocitySamples]],
    splits: dict,
    processed: Path,
) -> pd.DataFrame:
    raw_corridor = project / "data" / "raw" / "2013bidirectional"
    raw_crossing = project / "data" / "raw" / "2013crossing90" / "trajectories"
    pedestrian_totals: dict[str, int] = {}
    for run in (*splits["calibration"], *splits["held_out_validation"], *splits["held_out_geometry_stress"]):
        trajectory = load_julich_trajectory(raw_corridor / f"{run}.txt")
        pedestrian_totals[run] = int(trajectory["pedestrian_id"].nunique())
    for source in sorted(raw_crossing.glob("crossing_90_[de]_*.txt")):
        trajectory = load_julich_crossing_trajectory(source)
        pedestrian_totals[source.stem] = int(trajectory["pedestrian_id"].nunique())

    rows = []
    for dataset, batches in datasets.items():
        for batch in batches:
            previous_speed = np.linalg.norm(batch.features[:, 0, :], axis=1)
            rows.append(
                {
                    "dataset": dataset,
                    "run": batch.run,
                    "total_pedestrians": pedestrian_totals[batch.run],
                    "mean_simultaneous_occupancy": float(np.mean(batch.occupancy)),
                    "median_speed_mps": float(np.median(previous_speed)),
                    "mean_neighbours_within_3m": float(np.mean(batch.neighbour_count)),
                    "mean_local_density_per_m2": float(
                        np.mean(batch.neighbour_count) / (math.pi * 3.0**2)
                    ),
                    "forecasts": len(batch.target),
                }
            )
    run_table = pd.DataFrame(rows)
    run_table.to_csv(processed / "distribution_shift_by_run.csv", index=False)
    summary = run_table.groupby("dataset", sort=False).agg(
        runs=("run", "nunique"),
        mean_total_pedestrians=("total_pedestrians", "mean"),
        median_total_pedestrians=("total_pedestrians", "median"),
        mean_occupancy=("mean_simultaneous_occupancy", "mean"),
        median_speed_mps=("median_speed_mps", "median"),
        mean_neighbours_within_3m=("mean_neighbours_within_3m", "mean"),
        mean_local_density_per_m2=("mean_local_density_per_m2", "mean"),
        forecasts=("forecasts", "sum"),
    )
    summary.to_csv(processed / "distribution_shift_summary.csv")
    return summary


def total_memory_gb() -> float | None:
    if os.name != "nt":
        return None
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.total_physical / 1024**3
    except (AttributeError, OSError):
        return None


def _prepared_scene_states(
    source: Path,
    crossing: bool,
    config: dict,
    maximum_frames: int = 80,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    loader = load_julich_crossing_trajectory if crossing else load_julich_trajectory
    data = add_prediction_goal_directions(
        add_motion_features(loader(source)),
        method=config["primary_goal_method"],
        history_frames=config["horizon_frames"],
    )
    states = []
    grouped = [(frame, group) for frame, group in data[data["goal_valid"]].groupby("frame")]
    grouped.sort(key=lambda item: len(item[1]), reverse=True)
    for _, group in grouped[:maximum_frames]:
        positions = group[["x_m", "y_m"]].to_numpy(float)
        velocities = np.nan_to_num(group[["vx_mps", "vy_mps"]].to_numpy(float))
        goals = group[["goal_x", "goal_y"]].to_numpy(float)
        states.append((positions, velocities, goals))
    return states


def timing_analysis(
    project: Path,
    config: dict,
    model: object,
    representative_batch: VelocitySamples,
    processed: Path,
) -> dict:
    for _ in range(config["timing_warmup_repetitions"]):
        model.predict(representative_batch)
    head_times = []
    for _ in range(config["timing_head_repetitions"]):
        start = time.perf_counter_ns()
        model.predict(representative_batch)
        head_times.append(time.perf_counter_ns() - start)

    corridor_source = project / "data" / "raw" / "2013bidirectional" / "bi_corr_400_b_06.txt"
    crossing_source = (
        project
        / "data"
        / "raw"
        / "2013crossing90"
        / "trajectories"
        / "crossing_90_d_7.txt"
    )
    scene_results = {}
    for dataset, source, crossing in (
        ("corridor", corridor_source, False),
        ("crossing", crossing_source, True),
    ):
        states = _prepared_scene_states(source, crossing, config)

        def process(state: tuple[np.ndarray, np.ndarray, np.ndarray]) -> int:
            positions, velocities, goals = state
            isotropic, forward, closing, wall, _ = _interaction_features(
                positions,
                velocities,
                goals,
                config["interaction_range_m"],
                config["pedestrian_radius_m"],
                config["wall_range_m"],
                None if crossing else (0.0, 4.0),
                config["neighbour_cutoff_m"],
            )
            features = np.zeros((len(positions), len(FEATURE_NAMES), 2))
            features[:, 0] = velocities
            features[:, 1] = goals
            features[:, 2] = isotropic
            features[:, 3] = forward
            features[:, 4] = closing
            features[:, 5] = wall
            samples = VelocitySamples(
                dataset,
                features,
                np.zeros_like(positions),
                np.arange(len(positions)),
                np.zeros(len(positions), dtype=int),
            )
            model.predict(samples)
            return len(positions)

        for index in range(config["timing_warmup_repetitions"]):
            process(states[index % len(states)])
        elapsed = []
        occupancies = []
        for index in range(config["timing_scene_repetitions"]):
            start = time.perf_counter_ns()
            occupancy = process(states[index % len(states)])
            elapsed.append(time.perf_counter_ns() - start)
            occupancies.append(occupancy)
        elapsed_ms = np.asarray(elapsed) / 1e6
        scene_results[dataset] = {
            "mean_scene_ms": float(elapsed_ms.mean()),
            "p95_scene_ms": float(np.quantile(elapsed_ms, 0.95)),
            "mean_occupancy": float(np.mean(occupancies)),
            "mean_ms_per_pedestrian": float(
                np.mean(elapsed_ms / np.asarray(occupancies, dtype=float))
            ),
            "included_steps": [
                "dense pairwise neighbour discovery",
                "isotropic, forward, closing, and wall field construction",
                "goal-frame projection",
                "tensor head",
                "radial speed cap",
            ],
            "excluded_steps": ["video tracking", "disk I/O", "route-estimator update"],
        }
    head_array = np.asarray(head_times, dtype=float)
    result = {
        "environment": {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "memory_gb": total_memory_gb(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "operating_system": platform.platform(),
        },
        "model_head_precomputed_features": {
            "batch_size": len(representative_batch.target),
            "repetitions": config["timing_head_repetitions"],
            "warmup_repetitions": config["timing_warmup_repetitions"],
            "mean_batch_ms": float(head_array.mean() / 1e6),
            "mean_microseconds_per_pedestrian": float(
                head_array.mean() / 1e3 / len(representative_batch.target)
            ),
            "interpretation": "amortized vectorized arithmetic; not end-to-end latency",
        },
        "full_observed_scene_processing": scene_results,
    }
    (processed / "computational_timing.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def make_figures(
    project: Path,
    metrics: pd.DataFrame,
    route_results: pd.DataFrame,
    coefficients: pd.DataFrame,
    timing: dict,
) -> None:
    figures = project / "figures"
    colors = {"corridor held-out": "#2563eb", "crossing external": "#0f766e"}
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))

    route_summary = route_results.groupby(["goal_method", "dataset"])["vector_rmse_mps"].agg(
        ["mean", "sem"]
    )
    methods = ["entry", "endpoint", "prior_velocity"]
    x = np.arange(len(methods))
    for offset, dataset in zip((-0.16, 0.16), colors, strict=True):
        values = route_summary.xs(dataset, level="dataset").reindex(methods)
        axes[0, 0].bar(
            x + offset,
            values["mean"],
            0.31,
            yerr=values["sem"],
            capsize=3,
            color=colors[dataset],
            label=dataset,
        )
    axes[0, 0].set(
        xticks=x,
        xticklabels=["past-only\nentry", "endpoint\n(sensitivity)", "prior velocity\ncontinuous"],
        ylabel="run-balanced RMSE (m/s)",
        title="A  Route information",
    )
    axes[0, 0].legend(frameon=False, fontsize=9)

    ablations = [
        "tensor_persistence_only",
        "tensor_no_isotropic",
        "tensor_no_forward",
        "tensor_no_closing",
        "tensor_no_wall",
        "tensor_response",
        "tensor_without_cap",
    ]
    labels = ["persistence\nonly", "- isotropic", "- forward", "- closing", "- wall", "full", "no cap"]
    for dataset, color, marker in (
        ("corridor held-out", "#2563eb", "o"),
        ("crossing external", "#0f766e", "s"),
    ):
        subset = metrics[(metrics["dataset"] == dataset) & metrics["model"].isin(ablations)]
        aggregate = subset.groupby("model")["vector_rmse_mps"].agg(["mean", "sem"]).reindex(ablations)
        axes[0, 1].errorbar(
            np.arange(len(ablations)),
            aggregate["mean"],
            yerr=aggregate["sem"],
            marker=marker,
            capsize=3,
            color=color,
            label=dataset,
        )
    axes[0, 1].set(
        xticks=np.arange(len(ablations)),
        xticklabels=labels,
        ylabel="run-balanced RMSE (m/s)",
        title="B  Structural ablations",
    )
    axes[0, 1].legend(frameon=False, fontsize=9)

    ordered = coefficients.copy()
    ordered["label"] = ordered["channel"].str[0].str.upper() + ": " + ordered["feature"]
    y = np.arange(len(ordered))
    axes[1, 0].errorbar(
        ordered["coefficient"],
        y,
        xerr=np.vstack(
            (
                ordered["coefficient"] - ordered["bootstrap_95_low"],
                ordered["bootstrap_95_high"] - ordered["coefficient"],
            )
        ),
        fmt="o",
        capsize=3,
        color="#7c3aed",
    )
    axes[1, 0].axvline(0.0, color="#64748b", linewidth=1)
    axes[1, 0].set(yticks=y, yticklabels=ordered["label"], title="C  Coefficient stability")
    axes[1, 0].invert_yaxis()

    timing_names = ["head only", "corridor scene", "crossing scene"]
    timing_values = [
        timing["model_head_precomputed_features"]["mean_microseconds_per_pedestrian"] / 1000.0,
        timing["full_observed_scene_processing"]["corridor"]["mean_ms_per_pedestrian"],
        timing["full_observed_scene_processing"]["crossing"]["mean_ms_per_pedestrian"],
    ]
    axes[1, 1].bar(timing_names, timing_values, color=["#94a3b8", "#2563eb", "#0f766e"])
    axes[1, 1].set(yscale="log", ylabel="ms per pedestrian (log scale)", title="D  Timing scope matters")
    axes[1, 1].text(
        0.02,
        0.96,
        "Scene timing includes neighbour search + fields + tensor head",
        transform=axes[1, 1].transAxes,
        va="top",
        fontsize=9,
    )
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(labelsize=9)
    figure.suptitle("Major-revision diagnostics: leakage, structure, stability, and cost", fontsize=15)
    figure.tight_layout()
    figure.savefig(figures / "major_revision_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    primary = metrics[metrics["model"].isin(
        ["constant_velocity", "isotropic_constrained", "tensor_response", "invariant_ridge", "interaction_mlp"]
    )]
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    for dataset, color in colors.items():
        pivot = primary[primary["dataset"] == dataset].pivot(
            index="run", columns="model", values="vector_rmse_mps"
        )
        for row_index, (_, row) in enumerate(pivot.iterrows()):
            y_value = row_index + (0 if dataset == "corridor held-out" else len(pivot) * 0.03)
            axes[0].plot(
                [row["constant_velocity"], row["tensor_response"]],
                [y_value, y_value],
                color=color,
                alpha=0.55,
                linewidth=2,
            )
            axes[0].scatter(row["constant_velocity"], y_value, facecolor="white", edgecolor=color, s=30)
            axes[0].scatter(row["tensor_response"], y_value, color=color, s=34)
    axes[0].set(
        xlabel="run RMSE (m/s)",
        ylabel="paired experimental runs",
        yticks=[],
        title="A  Every primary run shifts left",
    )

    external = primary[primary["dataset"] == "crossing external"]
    frontier = external.groupby("model").agg(
        rmse=("vector_rmse_mps", "mean"),
        sem=("vector_rmse_mps", "sem"),
        parameters=("parameters", "first"),
    )
    palette = {
        "constant_velocity": "#94a3b8",
        "isotropic_constrained": "#64748b",
        "tensor_response": "#0f766e",
        "invariant_ridge": "#2563eb",
        "interaction_mlp": "#dc2626",
    }
    for name, row in frontier.iterrows():
        x_value = max(float(row["parameters"]), 0.5)
        axes[1].errorbar(
            x_value,
            row["rmse"],
            yerr=row["sem"],
            fmt="o",
            capsize=4,
            color=palette[name],
            markersize=8 if name == "tensor_response" else 6,
        )
        axes[1].annotate(
            name.replace("_", " "),
            (x_value, row["rmse"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8.5,
        )
    axes[1].set(
        xscale="log",
        xlabel="fitted parameters (log scale; constant velocity shown at 0.5)",
        ylabel="external run-balanced RMSE (m/s)",
        title="B  External error-complexity plane (mean +/- SE)",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.tick_params(labelsize=9.5)
    figure.suptitle("Consistency and the external error-complexity frontier", fontsize=15)
    figure.tight_layout()
    figure.savefig(figures / "publication_evidence_summary.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def development_chronology(processed: Path, selected: dict, config: dict) -> None:
    rows = [
        ("Complete-run partitions", "Archive metadata and design rationale", "Before any model fitting"),
        ("Primary past-only route estimator", "Calibration runs; endpoint retained only as sensitivity", "Before all test stages"),
        ("Interaction scale", "Calibration-run sensitivity; 0.8 m inherited and declared", "Before all test stages"),
        ("Neighbour cutoff", "3 m fixed a priori; calibration-run sensitivity reported", "Before all test stages"),
        ("Tensor specification and bounds", "Calibration runs only; bounds [-5, 5]", "Before all test stages"),
        ("Speed cap", "Leave-one-calibration-run-out CV", "Before all test stages"),
        ("Ridge penalty", "Leave-one-calibration-run-out CV", "Before all test stages"),
        (
            "MLP architecture and regularization",
            "Fixed a priori; 64-64 tanh network with L2=0.001",
            "Before all test stages",
        ),
        (
            "MLP iteration count",
            "One prespecified complete calibration run (bi_corr_400_b_09)",
            "Before all test stages",
        ),
        ("Ordinary corridor evaluation", "Five untouched runs", "No model selection"),
        ("Altered-geometry evaluation", "Three reserved runs", "After ordinary evaluation; no refitting"),
        ("Crossing evaluation", "Thirteen external runs", "No coefficient or architecture changes"),
    ]
    table = pd.DataFrame(rows, columns=["item", "selected_using", "frozen_before"])
    table["analysis_id"] = config["analysis_id"]
    table.to_csv(processed / "development_chronology.csv", index=False)
    (processed / "major_revision_run_manifest.json").write_text(
        json.dumps(
            {
                "analysis_id": config["analysis_id"],
                "entry_point": "experiments/major_revision_analysis.py",
                "configuration": "configs/major_revision.json",
                "selected_hyperparameters": selected,
                "random_seed": config["seed"],
                "primary_route_information": (
                    "dominant cardinal displacement during first 10 observed frames; predictions "
                    "begin only once this past-only direction is available"
                ),
                "topology_specific_inputs": {
                    "corridor": "two-wall boundary field and past-only route direction",
                    "crossing": "no two-wall field; past-only route direction",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    project = project_root()
    processed = project / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    config, splits = load_inputs(project)
    corridor = project / "data" / "raw" / "2013bidirectional"
    crossing = project / "data" / "raw" / "2013crossing90" / "trajectories"
    primary_method = config["primary_goal_method"]

    training = {
        run: build_batch(corridor / f"{run}.txt", config, primary_method)
        for run in splits["calibration"]
    }
    selected = select_hyperparameters(training, config, processed)
    print("selected hyperparameters", selected, flush=True)
    models = fit_models(list(training.values()), selected, config)
    social_force = models["social_force"].model
    tensor = models["tensor_response"].model
    (processed / "fitted_primary_model_parameters.json").write_text(
        json.dumps(
            {
                "social_force": {
                    "relaxation_time_s": social_force.relaxation_time,
                    "desired_speed_mps": social_force.desired_speed,
                    "pedestrian_acceleration_mps2": social_force.pedestrian_acceleration,
                    "wall_acceleration_mps2": social_force.wall_acceleration,
                },
                "tensor_parallel_coefficients": tensor.parallel_coefficients.tolist(),
                "tensor_lateral_coefficients": tensor.lateral_coefficients.tolist(),
                "tensor_parallel_features": list(PARALLEL_NAMES),
                "tensor_lateral_features": list(LATERAL_NAMES),
                "residual_network": selected["mlp_configuration"],
                "residual_selected_epoch": selected["residual_epoch"],
                "support_quantile": config["residual_support_quantile"],
                "gate_strength": selected["residual_gate_strength"],
                "direct_expert_blend_weight": selected["direct_blend_weight"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    datasets = {
        "calibration": list(training.values()),
        "corridor held-out": [
            build_batch(corridor / f"{run}.txt", config, primary_method)
            for run in splits["held_out_validation"]
        ],
        "altered geometry": [
            build_batch(corridor / f"{run}.txt", config, primary_method)
            for run in splits["held_out_geometry_stress"]
        ],
        "crossing external": [
            build_batch(source, config, primary_method, crossing=True)
            for source in sorted(crossing.glob("crossing_90_[de]_*.txt"))
        ],
    }
    metrics, _ = evaluate_models(datasets, models, processed, config)
    neural_seed_sensitivity(list(training.values()), datasets, selected, config, processed)
    route_results = route_information_ablation(
        project, config, splits, selected["speed_cap_mps"], processed
    )
    field_sensitivity(project, config, splits, selected["speed_cap_mps"], processed)
    coefficients, _ = coefficient_stability(training, config, processed)
    distribution_shift_summary(project, datasets, splits, processed)
    timing = timing_analysis(
        project,
        config,
        models["tensor_response"].model,
        datasets["corridor held-out"][0],
        processed,
    )
    make_figures(project, metrics, route_results, coefficients, timing)
    development_chronology(processed, selected, config)
    print("major-revision analysis complete", flush=True)


if __name__ == "__main__":
    main()
