# Author: Amir Ghorbani
"""Frozen neural comparison across short operational forecast horizons."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from major_revision_analysis import build_batch
from pedgeom.benchmarks import (
    GELPedRegressor,
    GoalStableRegressor,
    fit_interaction_mlp,
    fit_tensor_residual_mlp,
)
from pedgeom.calibration import fit_tensor_geometry_model, velocity_metrics
from pedgeom.statistics import exact_paired_randomization_pvalue, run_bootstrap_interval


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    config = json.loads((project / "configs" / "major_revision.json").read_text())
    splits = json.loads((project / "data" / "splits.json").read_text())
    selected = json.loads(
        (project / "data" / "processed" / "selected_hyperparameters.json").read_text()
    )
    corridor = project / "data" / "raw" / "2013bidirectional"
    crossing = project / "data" / "raw" / "2013crossing90" / "trajectories"
    mlp = selected["mlp_configuration"]
    rows: list[dict] = []
    for horizon_frames in (5, 10, 20, 30):
        horizon_config = dict(config)
        horizon_config["horizon_frames"] = horizon_frames
        horizon_config["frame_stride"] = horizon_frames
        horizon_s = horizon_frames / config["frames_per_second"]
        training = [
            build_batch(corridor / f"{run}.txt", horizon_config, "entry")
            for run in splits["calibration"]
        ]
        evaluation = {
            "corridor held-out": [
                build_batch(corridor / f"{run}.txt", horizon_config, "entry")
                for run in splits["held_out_validation"]
            ],
            "altered geometry": [
                build_batch(corridor / f"{run}.txt", horizon_config, "entry")
                for run in splits["held_out_geometry_stress"]
            ],
            "crossing external": [
                build_batch(source, horizon_config, "entry", crossing=True)
                for source in sorted(crossing.glob("crossing_90_[de]_*.txt"))
            ],
        }
        tensor = fit_tensor_geometry_model(training)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            direct = fit_interaction_mlp(
                training,
                hidden_layers=tuple(mlp["hidden_layers"]),
                alpha=mlp["alpha"],
                max_iter=mlp["selected_epoch"],
                seed=config["seed"],
            )
            structured = fit_tensor_residual_mlp(
                training,
                tensor,
                hidden_layers=tuple(mlp["hidden_layers"]),
                alpha=mlp["alpha"],
                max_iter=selected["residual_epoch"],
                seed=config["seed"],
                support_quantile=config["residual_support_quantile"],
                gate_strength=selected["residual_gate_strength"],
            )
        models = {
            "direct_network": direct,
            "goal_stable_neural": GoalStableRegressor(direct),
            "gel_ped": GELPedRegressor(
                direct, structured, selected["direct_blend_weight"]
            ),
        }
        for dataset, batches in evaluation.items():
            for batch in batches:
                for name, model in models.items():
                    rows.append(
                        {
                            "horizon_s": horizon_s,
                            "dataset": dataset,
                            "run": batch.run,
                            "model": name,
                            **velocity_metrics(
                                batch.target,
                                model.predict(batch, speed_cap=100.0),
                                horizon_s,
                            ),
                        }
                    )
        print(f"neural horizon sensitivity: {horizon_s:.1f} s", flush=True)

    results = pd.DataFrame(rows)
    processed = project / "data" / "processed"
    results.to_csv(processed / "neural_horizon_sensitivity_metrics.csv", index=False)
    summary = results.groupby(["horizon_s", "dataset", "model"], as_index=False).agg(
        run_balanced_rmse_mps=("vector_rmse_mps", "mean"),
        displacement_mae_m=("displacement_mae_m", "mean"),
        runs=("run", "nunique"),
    )
    summary.to_csv(processed / "neural_horizon_sensitivity_summary.csv", index=False)

    statistics: dict[str, dict] = {}
    pivot = results.pivot(
        index=["horizon_s", "dataset", "run"],
        columns="model",
        values="vector_rmse_mps",
    )
    for horizon_s in sorted(results.horizon_s.unique()):
        for dataset in results.dataset.unique():
            data = pivot.loc[(horizon_s, dataset)]
            difference = (
                data.gel_ped - data.direct_network
            ).to_numpy()
            key = f"{horizon_s:.1f}s::{dataset}"
            statistics[key] = {
                "mean_difference_gel_ped_minus_direct_mps": float(difference.mean()),
                "mean_relative_reduction_percent": float(
                    np.mean(
                        100.0
                        * (data.direct_network - data.gel_ped)
                        / data.direct_network
                    )
                ),
                "run_bootstrap_95_ci_mps": run_bootstrap_interval(
                    difference, repetitions=20_000, seed=config["seed"]
                ),
                "exact_paired_randomization_p": exact_paired_randomization_pvalue(
                    difference
                ),
                "runs_gel_ped_lower": int((difference < 0.0).sum()),
                "runs_total": int(len(difference)),
            }
    for dataset in results.dataset.unique():
        keys = [key for key in statistics if key.endswith(f"::{dataset}")]
        ordered = sorted(keys, key=lambda key: statistics[key]["exact_paired_randomization_p"])
        running_max = 0.0
        for rank, key in enumerate(ordered):
            raw = statistics[key]["exact_paired_randomization_p"]
            adjusted = min(1.0, (len(ordered) - rank) * raw)
            running_max = max(running_max, adjusted)
            statistics[key]["holm_adjusted_p_across_horizons"] = running_max
    (processed / "neural_horizon_sensitivity_statistics.json").write_text(
        json.dumps(statistics, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
