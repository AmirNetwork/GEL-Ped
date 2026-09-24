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
    manuscript_present = manuscript_path.exists()
    manuscript = manuscript_path.read_text(encoding="utf-8") if manuscript_present else ""

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
                "tensor_residual",
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
        ("corridor held-out", "interaction_mlp"): 0.1812,
        ("corridor held-out", "goal_stable_neural"): 0.1817,
        ("corridor held-out", "gel_ped"): 0.1803,
        ("altered geometry", "interaction_mlp"): 0.1792,
        ("altered geometry", "goal_stable_neural"): 0.1802,
        ("altered geometry", "gel_ped"): 0.1774,
        ("crossing external", "interaction_mlp"): 0.3680,
        ("crossing external", "goal_stable_neural"): 0.3685,
        ("crossing external", "gel_ped"): 0.3572,
    }
    for key, expected in expected_primary.items():
        close(metric(*key), expected)
        if manuscript_present:
            require(f"{expected:.4f}" in manuscript, f"manuscript omits {expected:.4f}")

    parameters = summary[summary.dataset == "corridor held-out"].set_index("model").parameters
    require(int(parameters.interaction_mlp) == 18434, "direct-network capacity changed")
    require(int(parameters.goal_stable_neural) == 18434, "goal-stable capacity is not matched")
    require(int(parameters.tensor_residual) == 18700, "structured capacity changed")
    require(int(parameters.gel_ped) == 37134, "GEL-Ped parameter count changed")
    ensemble = pd.read_csv(processed / "capacity_matched_ensemble_summary.csv")
    require(bool((ensemble.parameters == 36868).all()), "direct ensemble is not capacity matched")
    ensemble_crossing = float(
        ensemble.loc[
            ensemble.dataset == "crossing external", "vector_rmse_mps"
        ].iloc[0]
    )
    close(ensemble_crossing, 0.3626)
    if manuscript_present:
        require("0.3626" in manuscript, "manuscript omits ensemble crossing result")

    selected = json.loads(
        (processed / "selected_hyperparameters.json").read_text(encoding="utf-8")
    )
    require(
        selected["mlp_configuration"]["hidden_layers"] == [128, 128],
        "unexpected selected architecture",
    )
    close(float(selected["residual_gate_strength"]), 0.0, tolerance=1e-12)
    if manuscript_present:
        require("128--128" in manuscript, "selected architecture missing from manuscript")
        require("selected zero shrinkage" in manuscript, "inactive candidate shrinkage not disclosed")

    horizons = pd.read_csv(processed / "neural_horizon_sensitivity_summary.csv")
    crossing = horizons[horizons.dataset == "crossing external"]
    expected_horizons = {
        0.2: (0.3933, 0.3887),
        0.4: (0.3680, 0.3572),
        0.8: (0.3441, 0.3332),
        1.2: (0.3760, 0.3710),
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
    close(float(one_run.loc["direct_network", "mean_rmse_mps"]), 0.4069)
    close(float(one_run.loc["gel_ped", "mean_rmse_mps"]), 0.3469)
    if manuscript_present:
        require("at most 750" in manuscript, "low-data observation cap is not disclosed")

    required_text = [
        "# GEL-Ped:",
        "Table 1. Notation used in GEL-Ped.",
        "Appendix A. Frozen partitions and reproducibility map",
        "Appendix B. Complete-run confirmatory evidence",
        "Appendix C. Component and optimization checks",
        "https://github.com/AmirNetwork/GEL-Ped",
    ]
    if manuscript_present:
        for text in required_text:
            require(text in manuscript, f"missing manuscript element: {text}")
        for stale in (
            "geometry_guided_ensemble",
            "positive ensemble gain",
            "Supplementary material",
            "support-gated residual expert",
            "uses an observable support gate",
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
            "direct_neural_ensemble": int(ensemble.parameters.iloc[0]),
            "structured_expert": int(parameters.tensor_residual),
            "gel_ped_two_expert_total": int(parameters.gel_ped),
            "same_observed_fields": True,
            "future_endpoint_used": False,
        },
        "external_crossing_rmse_mps": {
            "direct_network": metric("crossing external", "interaction_mlp"),
            "goal_stable_neural": metric("crossing external", "goal_stable_neural"),
            "direct_neural_ensemble": ensemble_crossing,
            "gel_ped": metric("crossing external", "gel_ped"),
        },
        "data_efficiency_subsets": int(len(pivot)),
        "data_efficiency_subsets_gel_ped_lower": int(
            (pivot.gel_ped < pivot.direct_network).sum()
        ),
        "manuscript_source": (
            str(manuscript_path.relative_to(project)) if manuscript_present else None
        ),
    }
    output = processed / "manuscript_consistency_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
