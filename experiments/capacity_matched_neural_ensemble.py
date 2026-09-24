# Author: Amir Ghorbani
"""Capacity-matched direct-neural ensemble control for GEL-Ped.

The control averages two independently initialized direct experts. It uses the
same selected branch architecture and almost the same total parameter count as
GEL-Ped, but contains no geometry-encoded expert. No test result is used to
choose its seeds, weight, architecture, or training duration.
"""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from major_revision_analysis import build_batch, mlp_parameter_count
from pedgeom.benchmarks import GELPedRegressor, fit_interaction_mlp
from pedgeom.calibration import proximity_metrics, velocity_metrics
from pedgeom.statistics import exact_paired_randomization_pvalue, run_bootstrap_interval


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    processed = project / "data" / "processed"
    config = json.loads((project / "configs" / "major_revision.json").read_text())
    splits = json.loads((project / "data" / "splits.json").read_text())
    selected = json.loads((processed / "selected_hyperparameters.json").read_text())
    corridor = project / "data" / "raw" / "2013bidirectional"
    crossing = project / "data" / "raw" / "2013crossing90" / "trajectories"

    training = [
        build_batch(corridor / f"{run}.txt", config, "entry")
        for run in splits["calibration"]
    ]
    mlp = selected["mlp_configuration"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        experts = [
            fit_interaction_mlp(
                training,
                hidden_layers=tuple(mlp["hidden_layers"]),
                alpha=mlp["alpha"],
                max_iter=mlp["selected_epoch"],
                seed=config["seed"] + offset,
            )
            for offset in (0, 1)
        ]
    ensemble = GELPedRegressor(experts[0], experts[1], 0.5)
    parameter_count = 2 * mlp_parameter_count(tuple(mlp["hidden_layers"]))

    datasets = {
        "corridor held-out": [
            build_batch(corridor / f"{run}.txt", config, "entry")
            for run in splits["held_out_validation"]
        ],
        "altered geometry": [
            build_batch(corridor / f"{run}.txt", config, "entry")
            for run in splits["held_out_geometry_stress"]
        ],
        "crossing external": [
            build_batch(source, config, "entry", crossing=True)
            for source in sorted(crossing.glob("crossing_90_[de]_*.txt"))
        ],
    }
    rows: list[dict] = []
    for dataset, batches in datasets.items():
        for batch in batches:
            prediction = ensemble.predict(batch, speed_cap=selected["speed_cap_mps"])
            rows.append(
                {
                    "dataset": dataset,
                    "run": batch.run,
                    "model": "direct_neural_ensemble",
                    "parameters": parameter_count,
                    **velocity_metrics(batch.target, prediction),
                    **proximity_metrics(batch, prediction),
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(processed / "capacity_matched_ensemble_metrics.csv", index=False)
    summary = metrics.groupby("dataset", as_index=False).agg(
        parameters=("parameters", "first"),
        vector_rmse_mps=("vector_rmse_mps", "mean"),
        displacement_mae_m=("displacement_mae_m", "mean"),
        median_angle_deg=("median_angle_deg", "mean"),
        false_contact_rate=("false_contact_rate", "mean"),
        runs=("run", "nunique"),
    )
    summary.to_csv(processed / "capacity_matched_ensemble_summary.csv", index=False)

    gel = pd.read_csv(processed / "major_revision_metrics.csv")
    gel = gel[(gel.model == "gel_ped") & (gel.dataset != "calibration")][
        ["dataset", "run", "vector_rmse_mps"]
    ].rename(columns={"vector_rmse_mps": "gel_ped_rmse_mps"})
    paired = metrics.merge(gel, on=["dataset", "run"], validate="one_to_one")
    statistics: dict[str, dict] = {}
    for dataset in [*datasets, "held-out pooled"]:
        data = paired if dataset == "held-out pooled" else paired[paired.dataset == dataset]
        difference = data.gel_ped_rmse_mps - data.vector_rmse_mps
        relative = 100.0 * difference / data.vector_rmse_mps
        statistics[dataset] = {
            "mean_gel_ped_minus_ensemble_mps": float(difference.mean()),
            "mean_relative_difference_percent": float(relative.mean()),
            "run_bootstrap_95_ci_mps": run_bootstrap_interval(
                difference.to_numpy(), repetitions=20_000, seed=config["seed"]
            ),
            "exact_paired_randomization_p": exact_paired_randomization_pvalue(
                difference.to_numpy()
            ),
            "runs_gel_ped_lower": int(np.sum(difference < 0.0)),
            "runs_total": int(len(difference)),
        }
    (processed / "capacity_matched_ensemble_statistics.json").write_text(
        json.dumps(statistics, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(statistics, indent=2), flush=True)


if __name__ == "__main__":
    main()
