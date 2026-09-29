# Author: Amir Ghorbani
"""Calibration-size sensitivity for the final shift-routed GEL-Ped model."""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import warnings

from joblib import Parallel, delayed, parallel_config
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from pedgeom.benchmarks import (
    calibrate_disagreement_router,
    fit_gradient_boosted_residual,
    fit_interaction_mlp,
    fit_residual_on_base_mlp,
)
from pedgeom.calibration import velocity_metrics

from data_efficiency_retuned import CANDIDATES
from major_revision_analysis import build_batch, subsample_batch
from shift_routed_upgrade import constant_velocity_base


def evaluate_subset(
    size: int,
    subset_number: int,
    indices: tuple[int, ...],
    training: list,
    evaluation: dict,
    config: dict,
    mlp_selection: dict,
    hgb_settings: dict,
) -> list[dict]:
    """Fit and evaluate one fixed complete-run calibration subset."""

    batches = [training[index] for index in indices]
    direct_choice = CANDIDATES[mlp_selection[str(size)]["direct_network"]["candidate"]]
    residual_choice = CANDIDATES[mlp_selection[str(size)]["gel_ped"]["candidate"]]
    base = constant_velocity_base()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        direct = fit_interaction_mlp(
            batches,
            hidden_layers=(128, 128),
            alpha=direct_choice["alpha"],
            max_iter=direct_choice["direct_epochs"],
            seed=config["seed"],
        )
        neural_residual = fit_residual_on_base_mlp(
            batches,
            base,
            hidden_layers=(128, 128),
            alpha=residual_choice["alpha"],
            max_iter=residual_choice["residual_epochs"],
            seed=config["seed"],
        )
    if size >= 4:
        boosted_residual = fit_gradient_boosted_residual(
            batches, base, seed=config["seed"], **hgb_settings
        )
        gel_ped = calibrate_disagreement_router(
            batches, boosted_residual, neural_residual
        )
        routing_mode = "calibration-envelope router"
    else:
        gel_ped = neural_residual
        routing_mode = "neural residual fallback"

    rows = []
    for dataset, test_batches in evaluation.items():
        for batch in test_batches:
            for model_name, model in (
                ("constant_velocity", base),
                ("direct_mlp", direct),
                ("gel_ped", gel_ped),
            ):
                rows.append(
                    {
                        "calibration_runs": size,
                        "subset": subset_number,
                        "training_run_names": ";".join(
                            training[index].run for index in indices
                        ),
                        "samples_per_training_run": 750,
                        "routing_mode": routing_mode,
                        "dataset": dataset,
                        "run": batch.run,
                        "model": model_name,
                        **velocity_metrics(
                            batch.target, model.predict(batch, speed_cap=100.0)
                        ),
                    }
                )
    return rows


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    processed = project / "data" / "processed"
    config = json.loads((project / "configs" / "major_revision.json").read_text())
    splits = json.loads((project / "data" / "splits.json").read_text())
    mlp_selection = json.loads(
        (processed / "data_efficiency_size_specific_selection.json").read_text()
    )
    hgb_selection = json.loads(
        (processed / "shift_routed_hgb_selection.json").read_text()
    )
    hgb_settings = {
        key: value
        for key, value in hgb_selection["anchored_residual"].items()
        if key != "candidate"
    }
    corridor = project / "data" / "raw" / "2013bidirectional"
    crossing = project / "data" / "raw" / "2013crossing90" / "trajectories"
    training = [
        subsample_batch(build_batch(corridor / f"{run}.txt", config, "entry"), 750)
        for run in splits["calibration"]
    ]
    evaluation = {
        "corridor held-out": [
            build_batch(corridor / f"{run}.txt", config, "entry")
            for run in splits["held_out_validation"]
        ],
        "altered geometry": [
            build_batch(corridor / f"{run}.txt", config, "entry")
            for run in splits["held_out_geometry_stress"]
        ],
        "crossing topology": [
            build_batch(source, config, "entry", crossing=True)
            for source in sorted(crossing.glob("crossing_90_[de]_*.txt"))
        ],
    }

    tasks = [
        delayed(evaluate_subset)(
            size,
            subset_number,
            indices,
            training,
            evaluation,
            config,
            mlp_selection,
            hgb_settings,
        )
        for size in range(1, 8)
        for subset_number, indices in enumerate(combinations(range(7), size), start=1)
    ]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        groups = Parallel(n_jobs=3, verbose=10)(tasks)
    metrics = pd.DataFrame([row for group in groups for row in group])
    metrics.to_csv(processed / "shift_routed_data_efficiency_metrics.csv", index=False)

    per_subset_regime = (
        metrics.groupby(
            ["calibration_runs", "subset", "dataset", "model"], sort=False
        )["vector_rmse_mps"]
        .mean()
        .rename("run_balanced_rmse_mps")
        .reset_index()
    )
    per_regime = (
        per_subset_regime.groupby(
            ["calibration_runs", "dataset", "model"], sort=False
        )["run_balanced_rmse_mps"]
        .agg(
            mean_rmse_mps="mean",
            sd_across_subsets_mps="std",
            subset_configurations="count",
        )
        .reset_index()
    )
    per_regime.to_csv(
        processed / "shift_routed_data_efficiency_per_regime.csv", index=False
    )
    regime_balanced = (
        per_subset_regime.groupby(
            ["calibration_runs", "subset", "model"], sort=False
        )["run_balanced_rmse_mps"]
        .mean()
        .rename("regime_balanced_rmse_mps")
        .reset_index()
    )
    summary = (
        regime_balanced.groupby(["calibration_runs", "model"], sort=False)[
            "regime_balanced_rmse_mps"
        ]
        .agg(
            mean_rmse_mps="mean",
            sd_across_subsets_mps="std",
            minimum_rmse_mps="min",
            maximum_rmse_mps="max",
            subset_configurations="count",
        )
        .reset_index()
    )
    summary.to_csv(processed / "shift_routed_data_efficiency_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
