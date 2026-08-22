# Author: Amir Ghorbani
"""Calibration-run data-efficiency comparison for the direct network and GEL-Ped."""

from __future__ import annotations

from itertools import combinations
import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from major_revision_analysis import build_batch
from pedgeom.benchmarks import GELPedRegressor, fit_interaction_mlp, fit_tensor_residual_mlp
from pedgeom.calibration import fit_tensor_geometry_model, velocity_metrics
from pedgeom.statistics import exact_paired_randomization_pvalue, run_bootstrap_interval


def selected_subsets(run_count: int, total: int = 7) -> list[tuple[int, ...]]:
    """Enumerate every calibration-run subset to avoid favorable subset selection."""

    return list(combinations(range(total), run_count))


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    config = json.loads((project / "configs" / "major_revision.json").read_text())
    splits = json.loads((project / "data" / "splits.json").read_text())
    selected = json.loads(
        (project / "data" / "processed" / "selected_hyperparameters.json").read_text()
    )
    corridor = project / "data" / "raw" / "2013bidirectional"
    crossing = project / "data" / "raw" / "2013crossing90" / "trajectories"
    training = [
        build_batch(corridor / f"{run}.txt", config, "entry")
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
        "crossing external": [
            build_batch(source, config, "entry", crossing=True)
            for source in sorted(crossing.glob("crossing_90_[de]_*.txt"))
        ],
    }
    mlp = selected["mlp_configuration"]
    rows: list[dict] = []
    for run_count in range(1, 8):
        subsets = selected_subsets(run_count)
        for subset_number, indices in enumerate(subsets, start=1):
            batches = [training[index] for index in indices]
            tensor = fit_tensor_geometry_model(batches)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                direct = fit_interaction_mlp(
                    batches,
                    hidden_layers=tuple(mlp["hidden_layers"]),
                    alpha=mlp["alpha"],
                    max_iter=mlp["selected_epoch"],
                    seed=config["seed"],
                )
                structured = fit_tensor_residual_mlp(
                    batches,
                    tensor,
                    hidden_layers=tuple(mlp["hidden_layers"]),
                    alpha=mlp["alpha"],
                    max_iter=selected["residual_epoch"],
                    seed=config["seed"],
                    support_quantile=config["residual_support_quantile"],
                    gate_strength=config["residual_gate_strength"],
                )
            gel_ped = GELPedRegressor(
                direct, structured, selected["direct_blend_weight"]
            )
            for dataset, test_batches in evaluation.items():
                for batch in test_batches:
                    for name, model in (
                        ("direct_network", direct),
                        ("gel_ped", gel_ped),
                    ):
                        rows.append(
                            {
                                "calibration_runs": run_count,
                                "subset": subset_number,
                                "training_run_names": ";".join(
                                    training[index].run for index in indices
                                ),
                                "dataset": dataset,
                                "run": batch.run,
                                "model": name,
                                **velocity_metrics(
                                    batch.target, model.predict(batch, speed_cap=100.0)
                                ),
                            }
                        )
            print(
                f"data efficiency: {run_count} runs, subset {subset_number}/{len(subsets)}",
                flush=True,
            )
    results = pd.DataFrame(rows)
    processed = project / "data" / "processed"
    results.to_csv(processed / "data_efficiency_metrics.csv", index=False)
    subset_means = results.groupby(
        ["calibration_runs", "subset", "model"], as_index=False
    ).agg(run_balanced_rmse_mps=("vector_rmse_mps", "mean"))
    summary = subset_means.groupby(["calibration_runs", "model"], as_index=False).agg(
        mean_rmse_mps=("run_balanced_rmse_mps", "mean"),
        minimum_rmse_mps=("run_balanced_rmse_mps", "min"),
        maximum_rmse_mps=("run_balanced_rmse_mps", "max"),
        subset_configurations=("subset", "nunique"),
    )
    summary.to_csv(processed / "data_efficiency_summary.csv", index=False)
    paired_statistics = {}
    pivot = subset_means.pivot(
        index=["calibration_runs", "subset"], columns="model", values="run_balanced_rmse_mps"
    )
    for run_count in range(1, 8):
        data = pivot.loc[run_count]
        relative = 100.0 * (
            data.gel_ped - data.direct_network
        ) / data.direct_network
        paired_statistics[str(run_count)] = {
            "mean_relative_rmse_difference_percent": float(relative.mean()),
            "subset_bootstrap_95_ci_percent": run_bootstrap_interval(
                relative.to_numpy(), repetitions=20_000, seed=config["seed"]
            ),
            "exact_paired_randomization_p": (
                exact_paired_randomization_pvalue(relative.to_numpy())
                if len(relative) <= 24
                else None
            ),
            "subsets_gel_ped_lower": int((relative < 0.0).sum()),
            "subsets_total": int(len(relative)),
            "interpretation": (
                "descriptive robustness over overlapping calibration subsets; "
                "subsets are not independent test experiments"
            ),
        }
    (processed / "data_efficiency_statistics.json").write_text(
        json.dumps(paired_statistics, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
