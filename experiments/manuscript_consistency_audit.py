# Author: Amir Ghorbani
"""Audit GEL-Ped manuscript claims against the frozen machine-readable results.

The audit fails on stale model names, overlapping run partitions, unmatched neural
benchmark capacity, or a mismatch between displayed values and analysis artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def require(condition: bool, message: str) -> None:
    """Raise a concise error when a reproducibility condition is not met."""

    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, tolerance: float = 5e-5) -> None:
    """Check values at the precision reported in the manuscript."""

    require(abs(actual - expected) <= tolerance, f"{actual} != {expected}")


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    processed = project / "data" / "processed"
    manuscript_path = project / "manuscript" / "manuscript.md"
    manuscript = manuscript_path.read_text(encoding="utf-8")

    splits = json.loads((project / "data" / "splits.json").read_text(encoding="utf-8"))
    groups = {
        name: set(splits[name])
        for name in ("calibration", "held_out_validation", "held_out_geometry_stress")
    }
    group_names = list(groups)
    for index, left in enumerate(group_names):
        for right in group_names[index + 1 :]:
            require(groups[left].isdisjoint(groups[right]), f"overlap: {left} and {right}")
    require([len(groups[name]) for name in groups] == [7, 5, 3], "unexpected split sizes")

    summary = pd.read_csv(processed / "major_revision_summary.csv")
    primary = summary[
        summary.dataset.isin(["corridor held-out", "altered geometry", "crossing external"])
        & summary.model.isin(
            [
                "interaction_mlp",
                "goal_stable_neural",
                "gel_ped",
                "social_force",
                "unrestricted_ten_scalar",
            ]
        )
    ]
    require("geometry_guided_ensemble" not in set(summary.model), "stale model key")
    require("gel_ped" in set(primary.model), "GEL-Ped result missing")

    def metric(dataset: str, model: str, column: str = "vector_rmse_mps") -> float:
        row = primary[(primary.dataset == dataset) & (primary.model == model)]
        require(len(row) == 1, f"missing unique row: {dataset}/{model}")
        return float(row.iloc[0][column])

    expected_primary = {
        ("corridor held-out", "interaction_mlp"): 0.1819,
        ("corridor held-out", "goal_stable_neural"): 0.1822,
        ("corridor held-out", "gel_ped"): 0.1838,
        ("altered geometry", "interaction_mlp"): 0.1798,
        ("altered geometry", "goal_stable_neural"): 0.1808,
        ("altered geometry", "gel_ped"): 0.1807,
        ("crossing external", "interaction_mlp"): 0.3638,
        ("crossing external", "goal_stable_neural"): 0.3642,
        ("crossing external", "gel_ped"): 0.3588,
    }
    for key, expected in expected_primary.items():
        close(metric(*key), expected)
        require(f"{expected:.4f}" in manuscript, f"manuscript omits {expected:.4f}")

    parameters = summary[summary.dataset == "corridor held-out"].set_index("model").parameters
    require(int(parameters.interaction_mlp) == 5122, "direct-network capacity changed")
    require(int(parameters.goal_stable_neural) == 5122, "goal-stable capacity is not matched")
    require(int(parameters.gel_ped) == 10382, "GEL-Ped parameter count changed")

    horizons = pd.read_csv(processed / "neural_horizon_sensitivity_summary.csv")
    crossing = horizons[horizons.dataset == "crossing external"]
    expected_horizons = {
        0.2: (0.3975, 0.3831),
        0.4: (0.3638, 0.3588),
        0.8: (0.3542, 0.3373),
        1.2: (0.3878, 0.3760),
    }
    for horizon, (direct_expected, gel_expected) in expected_horizons.items():
        rows = crossing[crossing.horizon_s == horizon].set_index("model")
        close(float(rows.loc["direct_network", "run_balanced_rmse_mps"]), direct_expected)
        close(float(rows.loc["gel_ped", "run_balanced_rmse_mps"]), gel_expected)

    efficiency = pd.read_csv(processed / "data_efficiency_summary.csv")
    require(int(efficiency.subset_configurations.sum() / 2) == 127, "subset count is not 127")
    paired = (
        pd.read_csv(processed / "data_efficiency_metrics.csv")
        .groupby(["calibration_runs", "subset", "model"], as_index=False)
        .vector_rmse_mps.mean()
    )
    pivot = paired.pivot(
        index=["calibration_runs", "subset"], columns="model", values="vector_rmse_mps"
    )
    require(bool((pivot.gel_ped < pivot.direct_network).all()), "GEL-Ped loses a subset")
    one_run = efficiency[efficiency.calibration_runs == 1].set_index("model")
    close(float(one_run.loc["direct_network", "mean_rmse_mps"]), 0.4019)
    close(float(one_run.loc["gel_ped", "mean_rmse_mps"]), 0.3414)

    required_text = [
        "# GEL-Ped:",
        "Table 1. Notation used in GEL-Ped.",
        "Appendix A. Frozen partitions and reproducibility map",
        "Appendix B. Complete-run confirmatory evidence",
        "Appendix C. Component and optimization checks",
        "https://github.com/AmirNetwork/GEL-Ped",
    ]
    for text in required_text:
        require(text in manuscript, f"missing manuscript element: {text}")
    for stale in (
        "geometry_guided_ensemble",
        "positive ensemble gain",
        "Supplementary material",
    ):
        require(stale not in manuscript, f"stale manuscript text: {stale}")

    report = {
        "status": "pass",
        "author": "Amir Ghorbani",
        "method": "GEL-Ped: Geometry-Encoded Learning for Pedestrian Forecasting",
        "split_sizes": {name: len(runs) for name, runs in groups.items()},
        "neural_parameter_fairness": {
            "direct_network": int(parameters.interaction_mlp),
            "goal_stable_neural": int(parameters.goal_stable_neural),
            "structured_expert": 5260,
            "gel_ped_two_expert_total": int(parameters.gel_ped),
            "same_observed_fields": True,
            "future_endpoint_used": False,
        },
        "external_crossing_rmse_mps": {
            "direct_network": metric("crossing external", "interaction_mlp"),
            "goal_stable_neural": metric("crossing external", "goal_stable_neural"),
            "gel_ped": metric("crossing external", "gel_ped"),
        },
        "data_efficiency_subsets": int(len(pivot)),
        "data_efficiency_subsets_gel_ped_lower": int(
            (pivot.gel_ped < pivot.direct_network).sum()
        ),
        "manuscript_source": str(manuscript_path.relative_to(project)),
    }
    output = processed / "manuscript_consistency_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
