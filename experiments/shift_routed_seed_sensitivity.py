# Author: Amir Ghorbani
"""Five-seed sensitivity check for the final shift-aware GEL-Ped model."""

from __future__ import annotations

import json
from copy import deepcopy

from joblib import Parallel, delayed, parallel_config
import pandas as pd

from major_revision_analysis import load_inputs, project_root
from reviewer_revision_analysis import build_datasets, run_metrics
from shift_routed_upgrade import fit_routed_models


SEEDS = (20260722, 20260723, 20260724, 20260725, 20260726)
MODELS = ("direct_mlp", "direct_routed_control", "gel_ped")


def evaluate_seed(
    seed: int,
    training: list,
    datasets: dict,
    selected: dict,
    config: dict,
    selection_payload: dict,
) -> pd.DataFrame:
    """Refit every stochastic component and evaluate complete test runs."""

    seeded_config = deepcopy(config)
    seeded_config["seed"] = seed
    models = fit_routed_models(training, selected, seeded_config, selection_payload)
    metrics = run_metrics(datasets, {name: models[name] for name in MODELS})
    metrics.insert(0, "seed", seed)
    return metrics


def main() -> None:
    project = project_root()
    processed = project / "data" / "processed"
    config, splits = load_inputs(project)
    selected = json.loads(
        (processed / "selected_hyperparameters.json").read_text(encoding="utf-8")
    )
    selection_payload = json.loads(
        (processed / "shift_routed_hgb_selection.json").read_text(encoding="utf-8")
    )
    training_map, datasets = build_datasets(project, config, splits)
    training = list(training_map.values())

    tasks = [
        delayed(evaluate_seed)(
            seed, training, datasets, selected, config, selection_payload
        )
        for seed in SEEDS
    ]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        frames = Parallel(n_jobs=3, verbose=5)(tasks)
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(processed / "shift_routed_seed_sensitivity_metrics.csv", index=False)

    run_balanced = (
        metrics.groupby(["seed", "dataset", "model"], sort=False)["vector_rmse_mps"]
        .mean()
        .reset_index()
    )
    summary = (
        run_balanced.groupby(["dataset", "model"], sort=False)["vector_rmse_mps"]
        .agg(mean_rmse_mps="mean", sd_across_seeds_mps="std", min_rmse_mps="min", max_rmse_mps="max")
        .reset_index()
    )
    summary.to_csv(processed / "shift_routed_seed_sensitivity_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
