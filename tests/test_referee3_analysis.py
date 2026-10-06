# Author: Amir Ghorbani
from pathlib import Path

import numpy as np
import pandas as pd

from referee3_analyse import (
    Bundle,
    _crossed_draw_run_inference,
    _equal_run_sample_weights,
    _equivalence,
    _evaluate_primary_draw,
    _resolve_declared_bundle_path,
    _site_online_weights,
    _stable_run_seed,
    calibrate_draw,
    cap_vectors,
    draw_indices,
    selected_predictions,
    signals,
)


def _bundle(run: str = "r1", role: str = "calibration") -> Bundle:
    n = 12
    target = np.column_stack((np.ones(n), np.zeros(n)))
    member_offsets = np.linspace(-0.1, 0.1, 25)
    members = np.stack(
        [target + np.array([offset, 0.0]) for offset in member_offsets], axis=0
    )
    fallback = np.column_stack((np.full(n, 0.9), np.zeros(n)))
    return Bundle(
        path=Path("synthetic.npz"),
        run=run,
        role=role,
        target=target,
        frame=np.tile(np.arange(0, 60, 10), 2),
        origin_time_s=np.tile(np.arange(6) * 0.4, 2),
        pedestrian_id=np.repeat([1, 2], 6),
        position=np.zeros((n, 2)),
        prior_velocity=target.copy(),
        neighbour_count=np.ones(n),
        occupancy=np.full(n, 2.0),
        members_raw=members,
        fallback_raw=fallback,
        constant_velocity=target.copy(),
        mahalanobis=np.linspace(0.0, 1.0, n),
    )


def _config() -> dict:
    return {
        "experts": {"fallback_prediction_speed_cap_mps": 3.0},
        "guard": {"trailing_window_seconds": 1.6},
        "controls": {"fixed_blend_grid_step": 0.1},
    }


def test_vector_cap_and_member_selection_are_deterministic() -> None:
    vectors = np.array([[4.0, 0.0], [0.0, 2.0]])
    assert np.allclose(cap_vectors(vectors, 3.0), [[3.0, 0.0], [0.0, 2.0]])
    members, graph, fallback = selected_predictions(_bundle(), (0, 1, 2), 3.0)
    assert members.shape == (3, 12, 2)
    assert graph.shape == fallback.shape == (12, 2)


def test_signals_use_explicit_seconds_for_each_pedestrian() -> None:
    bundle = _bundle()
    members, graph, fallback = selected_predictions(bundle, (0, 1, 2, 3, 4), 3.0)
    result = signals(bundle, members, graph, fallback, window_seconds=1.6)
    assert set(result) == {"spread", "disagreement", "mahalanobis", "speed"}
    assert all(value.shape == (12,) for value in result.values())
    assert np.all(np.isfinite(result["spread"]))


def test_manifest_path_allows_only_the_frozen_prediction_archive(tmp_path: Path) -> None:
    project = tmp_path / "project"
    archive = project / "data/processed/referee3_predictions"
    archive.mkdir(parents=True)
    bundle = archive / "test_run.npz"
    bundle.touch()

    accepted = _resolve_declared_bundle_path(
        project, "data/processed/referee3_predictions/test_run.npz"
    )
    assert accepted == bundle

    outside = project / "data/processed/other/test_run.npz"
    outside.parent.mkdir(parents=True)
    outside.touch()
    with np.testing.assert_raises_regex(RuntimeError, "leaves the prediction archive"):
        _resolve_declared_bundle_path(
            project, "data/processed/referee3_predictions/../other/test_run.npz"
        )


def test_manifest_path_accepts_allowlisted_relocated_archive(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression for the production junction without requiring link privileges."""

    project = tmp_path / "project"
    lexical_root = project / "data/processed/referee3_predictions"
    lexical_path = lexical_root / "test_run.npz"
    lexical_root.mkdir(parents=True)
    lexical_path.touch()
    relocated_root = tmp_path / "large-volume/referee3_predictions"
    relocated_path = relocated_root / "test_run.npz"
    relocated_root.mkdir(parents=True)
    relocated_path.touch()

    original_resolve = Path.resolve

    def resolve(path: Path, strict: bool = False) -> Path:
        if path == lexical_root:
            return relocated_root
        if path == lexical_path:
            return relocated_path
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve)
    accepted = _resolve_declared_bundle_path(
        project, "data/processed/referee3_predictions/test_run.npz"
    )
    assert accepted == lexical_path


def test_draws_are_unique_and_primary_is_first() -> None:
    draws = draw_indices(25, 5, 20, 17, include_primary=True)
    assert draws[0] == (0, 1, 2, 3, 4)
    assert len(draws) == len(set(draws)) == 20
    assert all(len(draw) == 5 for draw in draws)


def test_calibration_and_site_online_rule_are_finite() -> None:
    bundles = [_bundle("r1"), _bundle("r2")]
    calibration = calibrate_draw(bundles, (0, 1, 2, 3, 4), _config())
    assert calibration.tau >= 0.0
    assert calibration.scale > 0.0
    bundle = bundles[0]
    _, graph, fallback = selected_predictions(bundle, calibration.indices, 3.0)
    weight, updates = _site_online_weights(
        bundle, graph, fallback, learning_rate=1.0, fixed_share=0.01
    )
    assert weight.shape == updates.shape == (12,)
    assert np.all((weight >= 0.0) & (weight <= 1.0))


def test_learned_gate_sample_weights_give_each_complete_run_equal_mass() -> None:
    weights = _equal_run_sample_weights([np.zeros((2, 1)), np.zeros((8, 1))])
    assert np.sum(weights[:2]) == np.sum(weights[2:])
    assert np.allclose(weights[:2], 0.5)
    assert np.allclose(weights[2:], 0.125)


def test_run_specific_random_streams_are_stable_and_independent() -> None:
    assert _stable_run_seed(10, "run-a", 3) == _stable_run_seed(10, "run-a", 3)
    assert _stable_run_seed(10, "run-a", 3) != _stable_run_seed(10, "run-b", 3)
    assert _stable_run_seed(10, "run-a", 3) != _stable_run_seed(10, "run-a", 4)


def test_crossed_draw_run_summary_retains_both_replication_axes() -> None:
    rows = []
    values = {("a", 0): 1.0, ("a", 1): 3.0, ("b", 0): 5.0, ("b", 1): 7.0}
    endpoints = (
        "rmse",
        "q90",
        "q95",
        "q99",
        "cvar95",
        "cvar99",
        "large_error_rate",
    )
    for (run, draw), value in values.items():
        row = {"role": "familiar", "ensemble_size": 5, "run": run, "draw": draw}
        row.update({f"guard_minus_graph_{name}": value for name in endpoints})
        rows.append(row)
    summary = _crossed_draw_run_inference(
        pd.DataFrame(rows), repetitions=50, seed=123
    )
    rmse = summary.loc[summary["endpoint"].eq("rmse")].iloc[0]
    assert rmse["n_runs"] == 2
    assert rmse["n_draws"] == 2
    assert rmse["mean_guard_minus_graph"] == 4.0
    assert rmse["fraction_draws_with_improvement"] == 0.0


def test_equivalence_reports_both_frozen_margin_scales() -> None:
    table = pd.DataFrame(
        {
            "role": ["familiar"] * 4,
            "run": ["a", "b", "c", "d"],
            "ensemble_size": [5] * 4,
            "draw": [0] * 4,
            "graph_rmse": [1.0] * 4,
            "guard_minus_graph_rmse": [-0.0001, 0.0001, -0.00005, 0.00005],
        }
    )
    result = _equivalence(
        table,
        absolute_margin=0.002,
        relative_margin=0.01,
        primary_ensemble_size=5,
    )
    assert set(result["margin_type"]) == {
        "absolute_mps",
        "relative_to_run_graph_rmse",
    }
    assert result["equivalent_at_0.05"].all()
    assert np.all(result["ci90_lower"] < result["ci90_upper"])


def test_primary_analysis_synthetic_dry_run_writes_complete_outputs(tmp_path: Path) -> None:
    config = {
        "experts": {"fallback_prediction_speed_cap_mps": 3.0},
        "guard": {"trailing_window_seconds": 1.6, "primary_shape": "exponential"},
        "replication": {"member_seed_base": 17},
        "controls": {
            "fixed_blend_grid_step": 0.1,
            "within_run_gate_permutations": 3,
            "permutation_seed": 23,
            "online_experts": {
                "label_delay_seconds": 0.4,
                "learning_rates": [0.5],
                "fixed_share_rates": [0.0],
                "cold_start_seconds": 0.8,
            },
        },
    }
    oof = [_bundle("cal-a"), _bundle("cal-b")]
    test_bundle = _bundle("test-a", role="familiar")
    calibration = calibrate_draw(oof, (0, 1, 2, 3, 4), config)
    _evaluate_primary_draw(oof, [test_bundle], calibration, config, tmp_path)
    required = {
        "referee3_primary_run_metrics.csv",
        "referee3_detection_metrics.csv",
        "referee3_risk_retention_curve.csv",
        "referee3_permuted_gate.csv",
        "referee3_online_cold_start_metrics.csv",
        "referee3_activation_by_track_age.csv",
        "referee3_forecasts_primary.csv.gz",
    }
    assert required <= {path.name for path in tmp_path.iterdir()}
    detection = pd.read_csv(tmp_path / "referee3_detection_metrics.csv")
    test_detection = detection.loc[detection["run"].eq("test-a")]
    assert set(test_detection["positive_count"]) == {1}
    targeting = pd.read_csv(tmp_path / "referee3_permuted_gate.csv")
    assert set(targeting["stratum"]) >= {"all"}
    assert targeting["permutation"].nunique() == 3
