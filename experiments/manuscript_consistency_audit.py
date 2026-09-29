# Author: Amir Ghorbani
"""Audit the final GEL-Ped manuscript against machine-readable result files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, tolerance: float = 5e-5) -> None:
    require(abs(actual - expected) <= tolerance, f"{actual} != {expected}")


def metric(frame: pd.DataFrame, filters: dict[str, object], column: str) -> float:
    selected = frame
    for name, value in filters.items():
        selected = selected[selected[name] == value]
    require(len(selected) == 1, f"expected one row for {filters}, found {len(selected)}")
    return float(selected.iloc[0][column])


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    processed = project / "data" / "processed"
    manuscript_path = project / "manuscript" / "manuscript.md"
    manuscript = (
        manuscript_path.read_text(encoding="utf-8") if manuscript_path.exists() else None
    )

    splits = json.loads((project / "data" / "splits.json").read_text(encoding="utf-8"))
    split_names = ("calibration", "held_out_validation", "held_out_geometry_stress")
    groups = {name: set(splits[name]) for name in split_names}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            require(groups[left].isdisjoint(groups[right]), f"overlap: {left}/{right}")
    require([len(groups[name]) for name in split_names] == [7, 5, 3], "split sizes")

    primary = pd.read_csv(processed / "shift_routed_upgrade_summary.csv")
    expected = {
        ("corridor held-out", "direct_mlp"): 0.1812,
        ("corridor held-out", "direct_routed_control"): 0.1762,
        ("corridor held-out", "gel_ped"): 0.1751,
        ("altered geometry", "direct_mlp"): 0.1792,
        ("altered geometry", "direct_routed_control"): 0.1701,
        ("altered geometry", "gel_ped"): 0.1690,
        ("crossing topology", "direct_mlp"): 0.3680,
        ("crossing topology", "direct_routed_control"): 0.3682,
        ("crossing topology", "gel_ped"): 0.3511,
    }
    primary_values: dict[str, float] = {}
    for (dataset, model), displayed in expected.items():
        value = metric(primary, {"dataset": dataset, "model": model}, "mean_rmse_mps")
        close(value, displayed)
        if manuscript is not None:
            require(f"{displayed:.4f}" in manuscript, f"manuscript omits {displayed:.4f}")
        primary_values[f"{dataset}/{model}"] = value

    statistics = json.loads(
        (processed / "shift_routed_upgrade_statistics.json").read_text(encoding="utf-8")
    )
    crossing = statistics["crossing topology"]["direct_routed_control"]
    close(crossing["relative_reduction_percent"], 4.623, tolerance=0.001)
    require(crossing["runs_gel_lower"] == 13, "crossing run consistency")
    close(crossing["exact_paired_randomization_p"], 0.000244140625, tolerance=1e-12)

    router = pd.read_csv(processed / "shift_routed_router_diagnostics.csv")
    gel_router = router[router.model == "gel_ped"]
    fence = float(gel_router.calibration_threshold_mps.iloc[0])
    close(fence, 0.0634733222, tolerance=1e-8)
    crossing_scores = gel_router[gel_router.dataset == "crossing topology"]
    familiar_scores = gel_router[gel_router.dataset == "corridor held-out"]
    require((crossing_scores.disagreement_mps > fence).all(), "crossing router separation")
    require((familiar_scores.disagreement_mps < fence).all(), "familiar router separation")

    wall = pd.read_csv(processed / "shift_routed_wall_check_summary.csv")
    wall_values = {
        model: metric(
            wall,
            {"dataset": "crossing topology", "model": model},
            "mean_rmse_mps",
        )
        for model in ("direct_mlp", "direct_routed_control", "gel_ped")
    }
    close(wall_values["direct_mlp"], 0.3635)
    close(wall_values["gel_ped"], 0.3481)
    require(wall_values["gel_ped"] < wall_values["direct_routed_control"], "wall check")

    efficiency = pd.read_csv(processed / "shift_routed_data_efficiency_summary.csv")
    efficiency_values: dict[str, float] = {}
    for count in range(1, 8):
        direct = metric(
            efficiency,
            {"calibration_runs": count, "model": "direct_mlp"},
            "mean_rmse_mps",
        )
        gel = metric(
            efficiency,
            {"calibration_runs": count, "model": "gel_ped"},
            "mean_rmse_mps",
        )
        require(gel < direct, f"low-data direct comparison at {count} runs")
        efficiency_values[f"gel_ped_{count}_runs"] = gel
    two_constant = metric(
        efficiency,
        {"calibration_runs": 2, "model": "constant_velocity"},
        "mean_rmse_mps",
    )
    require(efficiency_values["gel_ped_2_runs"] < two_constant, "two-run persistence")
    require(int(efficiency.subset_configurations.max()) == 35, "subset enumeration")

    rollout = pd.read_csv(processed / "autoregressive_rollout_summary.csv")
    rollout_values: dict[str, float] = {}
    for horizon, direct_expected, gel_expected in (
        (2.0, 0.7962, 0.6606),
        (3.2, 1.5283, 1.2239),
    ):
        direct = metric(rollout, {"model": "direct_mlp", "horizon_s": horizon}, "mean_m")
        gel = metric(rollout, {"model": "gel_ped", "horizon_s": horizon}, "mean_m")
        close(direct, direct_expected)
        close(gel, gel_expected)
        require(gel < direct, f"rollout ranking at {horizon} s")
        rollout_values[f"direct_{horizon:.1f}s"] = direct
        rollout_values[f"gel_ped_{horizon:.1f}s"] = gel

    seeds = pd.read_csv(processed / "shift_routed_seed_sensitivity_summary.csv")
    seed_crossing = seeds[seeds.dataset == "crossing topology"]
    seed_direct = metric(seed_crossing, {"model": "direct_mlp"}, "mean_rmse_mps")
    seed_gel = metric(seed_crossing, {"model": "gel_ped"}, "mean_rmse_mps")
    require(seed_gel < seed_direct, "five-seed crossing ranking")

    if manuscript is not None:
        for required in (
            "# GEL-Ped: Shift-aware",
            "direct routed control",
            "cross-topology test",
            "Appendix A. Compact reproducibility record",
            "https://github.com/AmirNetwork/GEL-Ped",
        ):
            require(required in manuscript, f"missing manuscript element: {required}")
        for stale in (
            "geometry_guided_ensemble",
            "goal-stable hybrid principle",
            "Supplementary material",
            "crossing external",
            "Table B.",
            "Table C.",
        ):
            require(stale not in manuscript, f"stale manuscript text: {stale}")

    for figure in (
        "experimental_context_and_protocol.png",
        "gel_ped_architecture.png",
        "final_results.png",
        "data_efficiency.png",
    ):
        require((project / "figures" / figure).exists(), f"missing figure: {figure}")

    report = {
        "status": "pass",
        "author": "Amir Ghorbani",
        "method": "GEL-Ped: shift-aware geometry-encoded residual learning",
        "split_sizes": {name: len(groups[name]) for name in split_names},
        "primary_rmse_mps": primary_values,
        "matched_control_crossing_reduction_percent": crossing[
            "relative_reduction_percent"
        ],
        "wall_free_crossing_rmse_mps": wall_values,
        "five_seed_crossing_mean_rmse_mps": {
            "direct_mlp": seed_direct,
            "gel_ped": seed_gel,
        },
        "autoregressive_displacement_error_m": rollout_values,
        "manuscript_source": (
            str(manuscript_path.relative_to(project)) if manuscript is not None else None
        ),
    }
    output = processed / "manuscript_consistency_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
