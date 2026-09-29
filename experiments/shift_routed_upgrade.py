# Author: Amir Ghorbani
"""Calibration-only upgrade to shift-aware, dual-backbone GEL-Ped."""

from __future__ import annotations

import json
import warnings

from joblib import Parallel, delayed, parallel_config
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from pedgeom.benchmarks import (
    calibrate_disagreement_router,
    fit_gradient_boosted_direct,
    fit_gradient_boosted_residual,
    fit_residual_on_base_mlp,
)
from pedgeom.calibration import FittedVelocityModel, velocity_metrics
from pedgeom.statistics import exact_paired_randomization_pvalue, run_bootstrap_interval

from major_revision_analysis import ModelSpec, fit_models, load_inputs, project_root
from reviewer_revision_analysis import build_datasets, run_metrics, summarize


HGB_CANDIDATES = (
    {
        "max_iter": 150,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "learning_rate": 0.05,
    },
    {
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "learning_rate": 0.05,
    },
    {
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
        "learning_rate": 0.05,
    },
    {
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 10.0,
        "learning_rate": 0.05,
    },
)


def constant_velocity_base() -> FittedVelocityModel:
    """Return the zero-fit kinematic anchor used by both residual experts."""

    return FittedVelocityModel(("persistence",), np.array([1.0]))


def evaluate_hgb_candidate(
    validation_index: int,
    candidate_index: int,
    training: list,
    seed: int,
) -> list[dict]:
    """Evaluate one boosted-tree setting on one complete calibration run."""

    fit_batches = [
        batch for index, batch in enumerate(training) if index != validation_index
    ]
    validation = training[validation_index]
    candidate = HGB_CANDIDATES[candidate_index]
    direct = fit_gradient_boosted_direct(fit_batches, seed=seed, **candidate)
    residual = fit_gradient_boosted_residual(
        fit_batches, constant_velocity_base(), seed=seed, **candidate
    )
    rows = []
    for model_name, model in (("direct", direct), ("anchored_residual", residual)):
        rows.append(
            {
                "validation_run": validation.run,
                "candidate": candidate_index,
                "model": model_name,
                **candidate,
                "vector_rmse_mps": velocity_metrics(
                    validation.target, model.predict(validation, speed_cap=100.0)
                )["vector_rmse_mps"],
            }
        )
    return rows


def select_hgb_settings(training: list, seed: int) -> tuple[pd.DataFrame, dict]:
    """Select direct and residual tree settings by leave-one-run-out CV."""

    tasks = [
        delayed(evaluate_hgb_candidate)(validation, candidate, training, seed)
        for validation in range(len(training))
        for candidate in range(len(HGB_CANDIDATES))
    ]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        groups = Parallel(n_jobs=3, verbose=5)(tasks)
    results = pd.DataFrame([row for group in groups for row in group])
    means = results.groupby(["model", "candidate"])["vector_rmse_mps"].mean()
    selected = {
        model: int(means.loc[model].idxmin())
        for model in ("direct", "anchored_residual")
    }
    return results, selected


def router_rows(dataset: str, batches: list, models: dict[str, ModelSpec]) -> list[dict]:
    """Record the unlabeled score and deployed mixture weight for every run."""

    rows = []
    for batch in batches:
        for model_name in ("direct_routed_control", "gel_ped"):
            router = models[model_name].model
            rows.append(
                {
                    "dataset": dataset,
                    "run": batch.run,
                    "model": model_name,
                    "disagreement_mps": router.routing_score(batch),
                    "calibration_threshold_mps": router.threshold,
                    "calibration_iqr_mps": router.scale,
                    "in_support_weight": router.in_support_weight(batch),
                }
            )
    return rows


def paired_statistics(metrics: pd.DataFrame, seed: int) -> dict:
    """Compute complete-run comparisons for the final routed model."""

    output: dict[str, dict] = {}
    for dataset, frame in metrics.groupby("dataset", sort=False):
        pivot = frame.pivot(index="run", columns="model", values="vector_rmse_mps")
        output[dataset] = {}
        for comparator in (
            "direct_mlp",
            "hgb_direct",
            "direct_routed_control",
            "fixed_blend_gel_ped",
        ):
            difference = (pivot["gel_ped"] - pivot[comparator]).to_numpy()
            output[dataset][comparator] = {
                "runs": len(difference),
                "mean_gel_minus_comparator_mps": float(np.mean(difference)),
                "relative_reduction_percent": float(
                    -100.0 * np.mean(difference) / pivot[comparator].mean()
                ),
                "run_bootstrap_95_ci_mps": run_bootstrap_interval(
                    difference, repetitions=20000, seed=seed
                ),
                "exact_paired_randomization_p": exact_paired_randomization_pvalue(
                    difference
                ),
                "runs_gel_lower": int(np.sum(difference < 0.0)),
                "minimum_attainable_two_sided_p": float(2.0 ** (1 - len(difference))),
            }
    return output


def fit_routed_models(
    training: list,
    selected: dict,
    config: dict,
    selection_payload: dict,
) -> dict[str, ModelSpec]:
    """Fit the final GEL-Ped model and its architecture-matched direct control."""

    fixed = fit_models(training, selected, config)
    base = constant_velocity_base()
    direct_settings = {
        key: value
        for key, value in selection_payload["direct"].items()
        if key != "candidate"
    }
    residual_settings = {
        key: value
        for key, value in selection_payload["anchored_residual"].items()
        if key != "candidate"
    }
    direct_hgb = fit_gradient_boosted_direct(
        training, seed=config["seed"], **direct_settings
    )
    residual_hgb = fit_gradient_boosted_residual(
        training, base, seed=config["seed"], **residual_settings
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        residual_mlp = fit_residual_on_base_mlp(
            training,
            base,
            hidden_layers=tuple(selected["mlp_configuration"]["hidden_layers"]),
            alpha=selected["mlp_configuration"]["alpha"],
            max_iter=selected["residual_epoch"],
            seed=config["seed"],
        )
    direct_router = calibrate_disagreement_router(
        training, direct_hgb, fixed["interaction_mlp"].model
    )
    gel_ped = calibrate_disagreement_router(training, residual_hgb, residual_mlp)
    cap = selected["speed_cap_mps"]
    return {
        "constant_velocity": fixed["constant_velocity"],
        "social_force": fixed["social_force"],
        "direct_mlp": fixed["interaction_mlp"],
        "hgb_direct": ModelSpec(direct_hgb, cap, -1),
        "direct_routed_control": ModelSpec(direct_router, cap, -1),
        "kinematic_residual_hgb": ModelSpec(residual_hgb, cap, -1),
        "kinematic_residual_mlp": ModelSpec(residual_mlp, cap, 18690),
        "fixed_blend_gel_ped": fixed["gel_ped"],
        "tensor_prior": fixed["tensor_response"],
        "gel_ped": ModelSpec(gel_ped, cap, -1),
    }


def main() -> None:
    project = project_root()
    processed = project / "data" / "processed"
    config, splits = load_inputs(project)
    selected = json.loads(
        (processed / "selected_hyperparameters.json").read_text(encoding="utf-8")
    )
    training_map, datasets = build_datasets(project, config, splits)
    training = list(training_map.values())

    selection_rows, chosen = select_hgb_settings(training, config["seed"])
    selection_rows.to_csv(processed / "shift_routed_hgb_selection.csv", index=False)
    selection_payload = {
        "criterion": "lowest mean leave-one-complete-calibration-run-out velocity RMSE",
        "direct": {
            "candidate": chosen["direct"],
            **HGB_CANDIDATES[chosen["direct"]],
        },
        "anchored_residual": {
            "candidate": chosen["anchored_residual"],
            **HGB_CANDIDATES[chosen["anchored_residual"]],
        },
    }
    (processed / "shift_routed_hgb_selection.json").write_text(
        json.dumps(selection_payload, indent=2), encoding="utf-8"
    )

    models = fit_routed_models(training, selected, config, selection_payload)
    for auxiliary in ("constant_velocity", "social_force", "tensor_prior"):
        models.pop(auxiliary)
    metrics = run_metrics(datasets, models)
    metrics.to_csv(processed / "shift_routed_upgrade_metrics.csv", index=False)
    summarize(metrics).to_csv(processed / "shift_routed_upgrade_summary.csv", index=False)
    diagnostics = pd.DataFrame(
        [
            row
            for dataset, batches in {
                "calibration": training,
                **datasets,
            }.items()
            for row in router_rows(dataset, batches, models)
        ]
    )
    diagnostics.to_csv(processed / "shift_routed_router_diagnostics.csv", index=False)
    statistics = paired_statistics(metrics, config["seed"])
    (processed / "shift_routed_upgrade_statistics.json").write_text(
        json.dumps(statistics, indent=2), encoding="utf-8"
    )
    print(json.dumps(selection_payload, indent=2), flush=True)
    print(summarize(metrics).to_string(index=False), flush=True)
    print("shift-routed GEL-Ped upgrade complete", flush=True)


if __name__ == "__main__":
    main()
