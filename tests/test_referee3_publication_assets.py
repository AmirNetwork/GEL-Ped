# Author: Amir Ghorbani
"""Synthetic-only tests for the referee-3 publication asset builder."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from referee3_publication_assets import (  # noqa: E402
    AssetInputs,
    AssetSchemaError,
    HELDOUT_ENDPOINTS,
    PLANNER_POLICIES,
    PRIMARY_MODELS,
    STRATUM_ORDER,
    build_assets,
)


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _synthetic_inputs(tmp_path: Path) -> AssetInputs:
    roles = ("familiar corridor", "crossing topology")
    runs = ("run_a", "run_b")

    draw_rows = []
    for role_index, role in enumerate(roles):
        for size in (3, 5):
            for endpoint_index, endpoint in enumerate(("rmse", "q95", "cvar95")):
                effect = -0.002 * (endpoint_index + 1) - 0.0001 * role_index
                draw_rows.append(
                    {
                        "role": role,
                        "ensemble_size": size,
                        "endpoint": endpoint,
                        "n_runs": len(runs),
                        "n_draws": 4,
                        "mean_guard_minus_graph": effect,
                        "fraction_draws_with_improvement": 0.75,
                        "crossed_bootstrap_q025": effect - 0.001,
                        "crossed_bootstrap_q975": effect + 0.001,
                    }
                )

    primary_rows = []
    for role_index, role in enumerate(roles):
        for run_index, run in enumerate(runs):
            base = 0.31 + 0.02 * role_index + 0.005 * run_index
            for model_index, model in enumerate(PRIMARY_MODELS):
                adjustment = 0.0 if model_index == 0 else -0.001 * model_index
                primary_rows.append(
                    {
                        "role": role,
                        "run": run,
                        "model": model,
                        "n": 40,
                        "evaluation_status": "locked-evaluation",
                        "rmse": base + adjustment,
                        "q95": base + 0.20 + 2.0 * adjustment,
                        "cvar95": base + 0.28 + 2.5 * adjustment,
                        "large_error_rate": 0.11 + adjustment,
                    }
                )

    targeting_rows = []
    decomposition_rows = []
    permuted_rows = []
    for role_index, role in enumerate(roles):
        for run_index, run in enumerate(runs):
            for stratum_index, stratum in enumerate(STRATUM_ORDER):
                graph = 0.25 + 0.04 * stratum_index + 0.005 * role_index
                gel = graph - 0.002 * stratum_index
                share = (0.65, 0.18, 0.10, 0.07)[stratum_index]
                for model, value in (("graph ensemble", graph), ("GEL-Ped", gel)):
                    targeting_rows.append(
                        {
                            "role": role,
                            "run": run,
                            "stratum": stratum,
                            "model": model,
                            "n": 20,
                            "share": share,
                            "rmse": value + 0.001 * run_index,
                        }
                    )
                shared = 0.05 + 0.01 * stratum_index
                between = 0.005 + 0.004 * stratum_index
                decomposition_rows.append(
                    {
                        "role": role,
                        "run": run,
                        "stratum": stratum,
                        "n": 20,
                        "shared_mse": shared,
                        "between_member_variance": between,
                        "member_mse": shared + between,
                        "identity_residual": 0.0,
                    }
                )
                for permutation in range(3):
                    permuted_rows.append(
                        {
                            "role": role,
                            "run": run,
                            "stratum": stratum,
                            "n": 20,
                            "permutation": permutation,
                            "observed_gel_rmse": gel,
                            "permuted_rmse": gel + 0.001 * (stratum_index + 1),
                        }
                    )

    heldout_rows = []
    for run_index, run in enumerate(("t01", "t02", "t03")):
        for model, adjustment in (("graph", 0.0), ("guard", -0.01)):
            heldout_rows.append(
                {
                    "run": run,
                    "role": "untouched T-junction",
                    "model": model,
                    "n": 50,
                    "rmse": 0.4 + 0.01 * run_index + adjustment,
                    "q95": 0.7 + 0.01 * run_index + adjustment,
                    "cvar95": 0.8 + 0.01 * run_index + adjustment,
                    "rate_error_gt_0_5": 0.2 + adjustment,
                }
            )
    heldout_inference_rows = []
    graph_values = (0.41, 0.71, 0.81, 0.20)
    for endpoint, graph_value in zip(HELDOUT_ENDPOINTS, graph_values, strict=True):
        effect = -0.01
        heldout_inference_rows.append(
            {
                "endpoint": endpoint,
                "n_runs": 3,
                "graph_run_macro": graph_value,
                "guard_run_macro": graph_value + effect,
                "mean_guard_minus_graph": effect,
                "median_guard_minus_graph": effect,
                "relative_mean_difference_percent": 100.0 * effect / graph_value,
                "wins_guard": 3,
                "ties": 0,
                "ci95_lower": -0.015,
                "ci95_upper": -0.005,
                "exact_two_sided_p": 0.02,
                "holm_adjusted_p": 0.04,
                "reject_holm_0_05": True,
                "inference_unit": "complete physical run",
                "small_run_caution": "small number of complete runs",
            }
        )

    planner_rows = []
    for role_index, role in enumerate(roles):
        for run_index, run in enumerate(runs):
            for policy_index, policy in enumerate(PLANNER_POLICIES):
                planner_rows.append(
                    {
                        "role": role,
                        "run": run,
                        "policy": policy,
                        "completed": True,
                        "failure": False,
                        "steps": 20,
                        "elapsed_seconds": 8.0 + policy_index,
                        "travel_time_seconds": 8.0 + policy_index,
                        "route_length_m": 5.0,
                        "travelled_distance_m": 5.2,
                        "progress_fraction": 1.0,
                        "personal_space_intrusions": max(0, 2 - policy_index),
                        "intrusion_rate": 0.10 - 0.03 * policy_index,
                        "minimum_separation_m": 0.48 + 0.04 * policy_index,
                        "stop_count": policy_index,
                        "stop_frequency": 0.02 * policy_index,
                        "infeasible_count": 0,
                    }
                )

    transformer_rows = []
    for role in roles:
        for model_index, model in enumerate(
            ("transformer ensemble", "feature-based HGB fallback", "guarded transformer")
        ):
            transformer_rows.append(
                {
                    "regime": role,
                    "model": model,
                    "runs": 2,
                    "rmse_mps": 0.30 - 0.002 * model_index,
                    "q95_mps": 0.50 - 0.003 * model_index,
                    "cvar95_mps": 0.60 - 0.004 * model_index,
                    "guard_firing_rate": 0.1,
                    "mean_transformer_weight": 0.9,
                }
            )

    latency_rows = []
    for role in roles:
        for run in runs:
            for stage_index, stage in enumerate(
                ("five-member reduction", "instantaneous guard and blend")
            ):
                latency_rows.append(
                    {
                        "role": role,
                        "run": run,
                        "stage": stage,
                        "scope": (
                            "post-prediction CPU timing; excludes neural and HGB forward passes"
                        ),
                        "pedestrians": 12,
                        "warmup": 2,
                        "repeats": 5,
                        "mean_ms": 0.1 + 0.1 * stage_index,
                        "median_ms": 0.09 + 0.1 * stage_index,
                        "p95_ms": 0.12 + 0.1 * stage_index,
                        "minimum_ms": 0.08,
                        "maximum_ms": 0.23,
                    }
                )

    base = tmp_path / "inputs"
    return AssetInputs(
        draw_inference=_write_csv(base / "draw.csv", draw_rows),
        primary_metrics=_write_csv(base / "primary.csv", primary_rows),
        targeting=_write_csv(base / "targeting.csv", targeting_rows),
        decomposition=_write_csv(base / "decomposition.csv", decomposition_rows),
        permuted_gate=_write_csv(base / "permuted.csv", permuted_rows),
        heldout_metrics=_write_csv(base / "heldout.csv", heldout_rows),
        heldout_inference=_write_csv(base / "heldout_inference.csv", heldout_inference_rows),
        planner_metrics=_write_csv(base / "planner.csv", planner_rows),
        transformer_summary=_write_csv(base / "transformer.csv", transformer_rows),
        planner_latency=_write_csv(base / "latency.csv", latency_rows),
        training_manifest=_write_json(
            base / "training.json",
            {"fits": [{"parameters": 13203}, {"parameters": 13203}]},
        ),
        transformer_manifest=_write_json(
            base / "transformer.json",
            {
                "seeds": [1, 2, 3, 4, 5],
                "fits": [{"parameter_count": 8120}, {"parameter_count": 8120}],
            },
        ),
        config=_write_json(
            base / "config.json", {"replication": {"primary_ensemble_size": 5}}
        ),
    )


def test_builds_complete_vector_asset_set_from_synthetic_outputs(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path)
    outputs = build_assets(inputs, tmp_path / "paper")
    repeated = build_assets(inputs, tmp_path / "paper_repeat")

    assert len(outputs.figures) == 5
    assert len(outputs.tables) == 7
    assert all(path.is_file() and path.read_bytes().startswith(b"%PDF") for path in outputs.figures)
    assert all("\\label{tab:r3-" in path.read_text(encoding="utf-8") for path in outputs.tables)
    assert "\\label{fig:r3-heldout-tjunction}" in outputs.snippets.read_text(
        encoding="utf-8"
    )
    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert manifest["scope"].startswith("presentation-only")
    assert len(manifest["outputs"]) == 13
    assert {
        path.name: path.read_bytes() for path in [*outputs.tables, *outputs.figures]
    } == {
        path.name: path.read_bytes() for path in [*repeated.tables, *repeated.figures]
    }


def test_missing_analysis_column_fails_loudly(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path)
    primary = pd.read_csv(inputs.primary_metrics).drop(columns="q95")
    primary.to_csv(inputs.primary_metrics, index=False)

    with pytest.raises(AssetSchemaError, match="missing required columns.*q95"):
        build_assets(inputs, tmp_path / "paper")


def test_heldout_model_contract_rejects_unregistered_name(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path)
    heldout = pd.read_csv(inputs.heldout_metrics)
    heldout.loc[heldout["model"].eq("guard"), "model"] = "GEL-Ped"
    heldout.to_csv(inputs.heldout_metrics, index=False)

    with pytest.raises(AssetSchemaError, match="frozen values"):
        build_assets(inputs, tmp_path / "paper")
