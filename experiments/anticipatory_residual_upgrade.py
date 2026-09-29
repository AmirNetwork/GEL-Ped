# Author: Amir Ghorbani
"""Prospective-interaction residual without a separate analytic force model."""

from __future__ import annotations

from dataclasses import dataclass
import json
import warnings

from joblib import Parallel, delayed, parallel_config
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pedgeom.benchmarks import (
    calibrate_disagreement_router,
    invariant_design,
    local_target,
    local_target_from_vectors,
    prospective_interaction_state,
)
from pedgeom.calibration import FittedVelocityModel, VelocitySamples, velocity_metrics
from pedgeom.statistics import exact_paired_randomization_pvalue, run_bootstrap_interval

from major_revision_analysis import ModelSpec, load_inputs, project_root
from reviewer_revision_analysis import build_datasets, run_metrics, summarize
from shift_routed_upgrade import HGB_CANDIDATES, fit_routed_models


ANTICIPATION_CANDIDATES = (
    {"horizon_s": 1.5, "time_scale_s": 0.6, "clearance_scale_m": 0.35},
    {"horizon_s": 2.0, "time_scale_s": 0.9, "clearance_scale_m": 0.45},
    {"horizon_s": 2.5, "time_scale_s": 1.2, "clearance_scale_m": 0.55},
    {"horizon_s": 3.0, "time_scale_s": 1.5, "clearance_scale_m": 0.70},
)


def kinematic_anchor() -> FittedVelocityModel:
    return FittedVelocityModel(("persistence",), np.array([1.0]))


def prospective_design(samples: VelocitySamples, parameters: dict) -> np.ndarray:
    """Add prospective conflict moments to the established local geometry state."""

    return np.column_stack(
        (
            invariant_design(samples),
            prospective_interaction_state(samples, **parameters),
        )
    )


@dataclass
class ProspectiveRegressor:
    estimator: object
    parameters: dict
    residual: bool

    def _design(self, samples: VelocitySamples) -> np.ndarray:
        design = prospective_design(samples, self.parameters)
        if self.residual:
            anchor_local = local_target_from_vectors(
                samples, samples.features[:, 0]
            )
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


def hgb_estimator(settings: dict, seed: int) -> MultiOutputRegressor:
    return MultiOutputRegressor(
        HistGradientBoostingRegressor(random_state=seed, **settings)
    )


def fit_hgb(
    batches: list[VelocitySamples],
    parameters: dict,
    settings: dict,
    seed: int,
    *,
    residual: bool,
) -> ProspectiveRegressor:
    model = ProspectiveRegressor(None, parameters, residual)
    design = np.concatenate([model._design(batch) for batch in batches])
    target = np.concatenate([local_target(batch) for batch in batches])
    if residual:
        target -= np.concatenate(
            [local_target_from_vectors(batch, batch.features[:, 0]) for batch in batches]
        )
    estimator = hgb_estimator(settings, seed)
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


def selection_task(
    held_out: int,
    candidate: int,
    training: list[VelocitySamples],
    settings: dict,
    seed: int,
) -> dict:
    fit_batches = [batch for index, batch in enumerate(training) if index != held_out]
    validation = training[held_out]
    parameters = ANTICIPATION_CANDIDATES[candidate]
    model = fit_hgb(fit_batches, parameters, settings, seed, residual=True)
    return {
        "validation_run": validation.run,
        "candidate": candidate,
        **parameters,
        "vector_rmse_mps": velocity_metrics(
            validation.target, model.predict(validation, speed_cap=100.0)
        )["vector_rmse_mps"],
    }


def backbone_selection_task(
    held_out: int,
    candidate: int,
    training: list[VelocitySamples],
    parameters: dict,
    seed: int,
) -> dict:
    """Select prospective learner capacity on complete calibration runs only."""

    fit_batches = [batch for index, batch in enumerate(training) if index != held_out]
    validation = training[held_out]
    settings = HGB_CANDIDATES[candidate]
    model = fit_hgb(fit_batches, parameters, settings, seed, residual=True)
    return {
        "validation_run": validation.run,
        "candidate": candidate,
        **settings,
        "vector_rmse_mps": velocity_metrics(
            validation.target, model.predict(validation, speed_cap=100.0)
        )["vector_rmse_mps"],
    }


def paired_statistics(metrics: pd.DataFrame, seed: int) -> dict:
    output = {}
    for dataset, frame in metrics.groupby("dataset", sort=False):
        pivot = frame.pivot(index="run", columns="model", values="vector_rmse_mps")
        output[dataset] = {}
        for comparator in (
            "direct_mlp",
            "direct_routed_control",
            "gel_ped_v1",
            "prospective_direct_router",
            "prospective_transfer_control",
        ):
            difference = (
                pivot["gel_ped_prospective_transfer"] - pivot[comparator]
            ).to_numpy()
            output[dataset][comparator] = {
                "mean_difference_mps": float(np.mean(difference)),
                "relative_reduction_percent": float(
                    -100.0 * np.mean(difference) / pivot[comparator].mean()
                ),
                "run_bootstrap_95_ci_mps": run_bootstrap_interval(
                    difference, repetitions=20000, seed=seed
                ),
                "exact_paired_randomization_p": exact_paired_randomization_pvalue(
                    difference
                ),
                "runs_lower": int(np.sum(difference < 0.0)),
            }
    return output


def router_diagnostics(dataset: str, batches: list[VelocitySamples], routers: dict) -> list[dict]:
    rows = []
    for batch in batches:
        for name, router in routers.items():
            rows.append(
                {
                    "dataset": dataset,
                    "run": batch.run,
                    "model": name,
                    "disagreement_mps": router.routing_score(batch),
                    "calibration_threshold_mps": router.threshold,
                    "calibration_iqr_mps": router.scale,
                    "in_support_weight": router.in_support_weight(batch),
                }
            )
    return rows


def main() -> None:
    project = project_root()
    processed = project / "data" / "processed"
    config, splits = load_inputs(project)
    selected = json.loads(
        (processed / "selected_hyperparameters.json").read_text(encoding="utf-8")
    )
    previous_selection = json.loads(
        (processed / "shift_routed_hgb_selection.json").read_text(encoding="utf-8")
    )
    training_map, datasets = build_datasets(project, config, splits)
    training = list(training_map.values())
    hgb_settings = {
        key: value
        for key, value in previous_selection["anchored_residual"].items()
        if key != "candidate"
    }

    tasks = [
        delayed(selection_task)(held_out, candidate, training, hgb_settings, config["seed"])
        for held_out in range(len(training))
        for candidate in range(len(ANTICIPATION_CANDIDATES))
    ]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        selection = pd.DataFrame(Parallel(n_jobs=3, verbose=5)(tasks))
    selection.to_csv(processed / "prospective_selection.csv", index=False)
    means = selection.groupby("candidate")["vector_rmse_mps"].mean()
    chosen_index = int(means.idxmin())
    parameters = ANTICIPATION_CANDIDATES[chosen_index]

    backbone_tasks = [
        delayed(backbone_selection_task)(
            held_out, candidate, training, parameters, config["seed"]
        )
        for held_out in range(len(training))
        for candidate in range(len(HGB_CANDIDATES))
    ]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        backbone_selection = pd.DataFrame(
            Parallel(n_jobs=3, verbose=5)(backbone_tasks)
        )
    backbone_selection.to_csv(
        processed / "prospective_backbone_selection.csv", index=False
    )
    backbone_means = backbone_selection.groupby("candidate")["vector_rmse_mps"].mean()
    chosen_backbone = int(backbone_means.idxmin())
    hgb_settings = HGB_CANDIDATES[chosen_backbone]
    (processed / "prospective_selection.json").write_text(
        json.dumps(
            {
                "criterion": "lowest leave-one-complete-calibration-run-out RMSE",
                "candidate": chosen_index,
                **parameters,
                "mean_validation_rmse_mps": float(means.loc[chosen_index]),
                "backbone_candidate": chosen_backbone,
                "backbone_settings": hgb_settings,
                "backbone_mean_validation_rmse_mps": float(
                    backbone_means.loc[chosen_backbone]
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    direct_hgb = fit_hgb(training, parameters, hgb_settings, config["seed"], residual=False)
    residual_hgb = fit_hgb(training, parameters, hgb_settings, config["seed"], residual=True)
    direct_mlp = fit_mlp(training, parameters, selected, config["seed"], residual=False)
    residual_mlp = fit_mlp(training, parameters, selected, config["seed"], residual=True)
    direct_router = calibrate_disagreement_router(training, direct_hgb, direct_mlp)
    prospective_router = calibrate_disagreement_router(
        training, residual_hgb, residual_mlp
    )
    previous_models = fit_routed_models(training, selected, config, previous_selection)
    prospective_transfer_control = calibrate_disagreement_router(
        training, direct_hgb, previous_models["direct_mlp"].model
    )
    prospective_transfer = calibrate_disagreement_router(
        training, residual_hgb, previous_models["kinematic_residual_mlp"].model
    )
    cap = selected["speed_cap_mps"]
    models = {
        "direct_mlp": previous_models["direct_mlp"],
        "direct_routed_control": previous_models["direct_routed_control"],
        "gel_ped_v1": previous_models["gel_ped"],
        "prospective_direct_mlp": ModelSpec(direct_mlp, cap, -1),
        "prospective_direct_router": ModelSpec(direct_router, cap, -1),
        "prospective_transfer_control": ModelSpec(
            prospective_transfer_control, cap, -1
        ),
        "prospective_residual_hgb": ModelSpec(residual_hgb, cap, -1),
        "prospective_residual_mlp": ModelSpec(residual_mlp, cap, -1),
        "gel_ped_prospective": ModelSpec(prospective_router, cap, -1),
        "gel_ped_prospective_transfer": ModelSpec(prospective_transfer, cap, -1),
    }
    metrics = run_metrics(datasets, models)
    metrics.to_csv(processed / "prospective_upgrade_metrics.csv", index=False)
    summarize(metrics).to_csv(processed / "prospective_upgrade_summary.csv", index=False)
    routers = {
        "prospective_direct_control": prospective_transfer_control,
        "gel_ped_anticipatory": prospective_transfer,
    }
    diagnostics = pd.DataFrame(
        [
            row
            for dataset, batches in {"calibration": training, **datasets}.items()
            for row in router_diagnostics(dataset, batches, routers)
        ]
    )
    diagnostics.to_csv(processed / "prospective_router_diagnostics.csv", index=False)
    statistics = paired_statistics(metrics, config["seed"])
    statistics["selection"] = {
        "candidate": chosen_index,
        **parameters,
        "direct_router_threshold": direct_router.threshold,
        "gel_router_threshold": prospective_router.threshold,
        "prospective_transfer_threshold": prospective_transfer.threshold,
        "prospective_transfer_control_threshold": prospective_transfer_control.threshold,
    }
    (processed / "prospective_upgrade_statistics.json").write_text(
        json.dumps(statistics, indent=2), encoding="utf-8"
    )
    print(means, flush=True)
    print(backbone_means, flush=True)
    print(summarize(metrics).to_string(index=False), flush=True)
    print(json.dumps(statistics, indent=2), flush=True)


if __name__ == "__main__":
    main()
