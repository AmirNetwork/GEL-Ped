# Author: Amir Ghorbani
"""Complete-run inference for the autoregressive rollout experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pedgeom.statistics import exact_paired_randomization_pvalue, run_bootstrap_interval


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    processed = project / "data" / "processed"
    metrics = pd.read_csv(processed / "autoregressive_rollout_run_horizon.csv")
    pivot = metrics.pivot(
        index=["run", "horizon_s"], columns="model", values="mean_displacement_error_m"
    )
    results = {}
    for horizon in sorted(metrics["horizon_s"].unique()):
        frame = pivot.xs(horizon, level="horizon_s")
        results[f"{horizon:.1f}"] = {}
        for primary, comparator in (
            ("gel_ped", "direct_mlp"),
            ("gel_ped", "constant_velocity"),
            ("tensor_prior", "direct_mlp"),
            ("tensor_prior", "constant_velocity"),
        ):
            difference = (frame[primary] - frame[comparator]).to_numpy()
            results[f"{horizon:.1f}"][f"{primary}_minus_{comparator}"] = {
                "mean_difference_m": float(difference.mean()),
                "run_bootstrap_95_ci_m": run_bootstrap_interval(
                    difference, repetitions=20000, seed=20260722
                ),
                "exact_paired_randomization_p": exact_paired_randomization_pvalue(difference),
                "runs_primary_lower": int((difference < 0.0).sum()),
                "runs": len(difference),
            }
    (processed / "autoregressive_rollout_statistics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
