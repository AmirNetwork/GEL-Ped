from pathlib import Path

import numpy as np

from referee3_analyse import (
    Bundle,
    _site_online_weights,
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
