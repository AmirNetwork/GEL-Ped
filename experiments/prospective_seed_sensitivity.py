# Author: Amir Ghorbani
"""Five-seed sensitivity for anticipatory GEL-Ped and its matched control."""

from __future__ import annotations

from copy import deepcopy
import json

from joblib import Parallel, delayed, parallel_config
import pandas as pd

from pedgeom.benchmarks import calibrate_disagreement_router

from anticipatory_residual_upgrade import fit_hgb
from major_revision_analysis import ModelSpec, load_inputs, project_root
from reviewer_revision_analysis import build_datasets, run_metrics
from shift_routed_seed_sensitivity import SEEDS
from shift_routed_upgrade import fit_routed_models


def evaluate_seed(seed, training, datasets, selected, config, selection, parameters):
    seeded = deepcopy(config)
    seeded["seed"] = seed
    previous = fit_routed_models(training, selected, seeded, selection)
    residual_settings = {
        key: value for key, value in selection["anchored_residual"].items()
        if key != "candidate"
    }
    direct_settings = {
        key: value for key, value in selection["direct"].items()
        if key != "candidate"
    }
    prospective_residual = fit_hgb(
        training, parameters, residual_settings, seed, residual=True
    )
    prospective_direct = fit_hgb(
        training, parameters, direct_settings, seed, residual=False
    )
    gel = calibrate_disagreement_router(
        training, prospective_residual, previous["kinematic_residual_mlp"].model
    )
    direct = calibrate_disagreement_router(
        training, prospective_direct, previous["direct_mlp"].model
    )
    cap = selected["speed_cap_mps"]
    models = {
        "direct_mlp": previous["direct_mlp"],
        "prospective_direct_control": ModelSpec(direct, cap, -1),
        "gel_ped_anticipatory": ModelSpec(gel, cap, -1),
    }
    frame = run_metrics(datasets, models)
    frame.insert(0, "seed", seed)
    return frame


def main() -> None:
    project = project_root()
    processed = project / "data" / "processed"
    config, splits = load_inputs(project)
    selected = json.loads(
        (processed / "selected_hyperparameters.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        (processed / "shift_routed_hgb_selection.json").read_text(encoding="utf-8")
    )
    prospective = json.loads(
        (processed / "prospective_selection.json").read_text(encoding="utf-8")
    )
    parameters = {
        key: prospective[key]
        for key in ("horizon_s", "time_scale_s", "clearance_scale_m")
    }
    training_map, datasets = build_datasets(project, config, splits)
    training = list(training_map.values())
    tasks = [
        delayed(evaluate_seed)(
            seed, training, datasets, selected, config, selection, parameters
        )
        for seed in SEEDS
    ]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        frames = Parallel(n_jobs=3, verbose=5)(tasks)
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(processed / "prospective_seed_sensitivity_metrics.csv", index=False)
    run_balanced = (
        metrics.groupby(["seed", "dataset", "model"], sort=False)["vector_rmse_mps"]
        .mean().reset_index()
    )
    summary = (
        run_balanced.groupby(["dataset", "model"], sort=False)["vector_rmse_mps"]
        .agg(mean_rmse_mps="mean", sd_across_seeds_mps="std", minimum="min", maximum="max")
        .reset_index()
    )
    summary.to_csv(processed / "prospective_seed_sensitivity_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
