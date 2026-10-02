# Author: Amir Ghorbani
import numpy as np
import pytest

from pedgeom.runtime_guard import (
    binary_detection_metrics,
    causal_delayed_two_expert_weights,
    causal_trailing_time_mean,
    empirical_upper_cvar,
    ensemble_error_decomposition,
    finite_sample_upper_quantile,
    one_sided_gate_weights,
    per_run_tail_metrics,
    permute_weights_within_runs,
    risk_retention_curve,
)


def test_trailing_time_mean_is_causal_time_based_and_pedestrian_keyed() -> None:
    values = np.array([1.0, 10.0, 3.0, 5.0, 14.0])
    pedestrians = np.array([1, 2, 1, 1, 2])
    frames = np.array([0, 0, 2, 4, 4])
    result = causal_trailing_time_mean(
        values,
        pedestrians,
        frames,
        frames_per_second=2.0,
        window_seconds=1.0,
    )
    # At exactly one second the boundary is retained; at two seconds it expires.
    assert np.allclose(result, [1.0, 10.0, 2.0, 4.0, 14.0])


def test_trailing_time_mean_strictly_resets_when_scene_changes() -> None:
    result = causal_trailing_time_mean(
        np.array([1.0, 3.0, 100.0, 200.0, 9.0]),
        np.array([7, 7, 7, 7, 7]),
        np.array([0, 1, 0, 1, 5]),
        frames_per_second=1.0,
        window_seconds=10.0,
        scene_ids=np.array(["A", "A", "B", "B", "A"]),
    )
    # Returning to label A starts another scene block; no old A history survives.
    assert np.allclose(result, [1.0, 2.0, 100.0, 150.0, 9.0])


def test_trailing_time_mean_supports_vector_values_and_rejects_bad_order() -> None:
    result = causal_trailing_time_mean(
        np.array([[1.0, 2.0], [3.0, 6.0]]),
        np.array([1, 1]),
        np.array([0, 1]),
        frames_per_second=1.0,
        window_seconds=2.0,
    )
    assert np.allclose(result, [[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(ValueError, match="strictly increasing"):
        causal_trailing_time_mean(
            [1.0, 2.0],
            [1, 1],
            [3, 3],
            frames_per_second=1.0,
            window_seconds=1.0,
        )


def test_finite_sample_upper_quantile_uses_declared_order_statistic() -> None:
    values = np.arange(1.0, 20.0)
    # ceil(0.9 * (19 + 1)) = 18.
    assert finite_sample_upper_quantile(values, 0.9) == 18.0
    # The clipped rank makes a high finite-sample quantile equal the maximum.
    assert finite_sample_upper_quantile(np.arange(1.0, 11.0), 0.9) == 10.0
    with pytest.raises(ValueError, match="strictly between"):
        finite_sample_upper_quantile(values, 1.0)


def test_one_sided_gate_forms_are_identity_inside_and_continuous_at_threshold() -> None:
    scores = np.array([0.0, 1.0, 2.0, 3.0])
    exponential = one_sided_gate_weights(scores, threshold=1.0, scale=1.0)
    linear = one_sided_gate_weights(scores, threshold=1.0, scale=1.0, kind="linear")
    logistic = one_sided_gate_weights(scores, threshold=1.0, scale=1.0, kind="logistic")
    assert np.allclose(exponential, [1.0, 1.0, np.exp(-1), np.exp(-2)])
    assert np.allclose(linear, [1.0, 1.0, 0.0, 0.0])
    assert np.allclose(logistic[:2], 1.0)
    assert np.allclose(logistic[2:], 2.0 / (1.0 + np.exp([1.0, 2.0])))
    assert np.all((0.0 <= logistic) & (logistic <= 1.0))


def test_tail_metrics_are_per_run_and_cvar_uses_fractional_tail_mass() -> None:
    errors = np.concatenate([np.arange(10) / 10.0, np.array([0.1, 0.2, 0.3, 0.4])])
    runs = np.array(["A"] * 10 + ["B"] * 4)
    table = per_run_tail_metrics(errors, runs, large_error_threshold=0.5)
    first = table.set_index("run_id").loc["A"]
    assert first["n"] == 10
    assert first["q90"] == pytest.approx(0.81)
    assert first["q95"] == pytest.approx(0.855)
    assert first["q99"] == pytest.approx(0.891)
    # Top 5% and 1% empirical mass are both fractional mass of the maximum.
    assert first["cvar95"] == pytest.approx(0.9)
    assert first["cvar99"] == pytest.approx(0.9)
    assert first["large_error_rate"] == pytest.approx(0.4)
    assert empirical_upper_cvar([0.0, 1.0, 2.0, 3.0], 0.5) == pytest.approx(2.5)


def test_binary_detection_metrics_for_perfect_ranking() -> None:
    result = binary_detection_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result["auroc"] == 1.0
    assert result["auprc"] == 1.0
    assert result["positive_rate"] == 0.5
    with pytest.raises(ValueError, match="both classes"):
        binary_detection_metrics([0, 0], [0.1, 0.2])


def test_risk_retention_curve_and_discrete_aurc_are_exact() -> None:
    curve = risk_retention_curve(losses=[4.0, 1.0, 2.0], scores=[3.0, 1.0, 2.0])
    expected_risk = np.array([1.0, 1.5, 7.0 / 3.0])
    assert np.array_equal(curve.order, [1, 2, 0])
    assert np.allclose(curve.retention, [1 / 3, 2 / 3, 1.0])
    assert np.allclose(curve.risk, expected_risk)
    assert curve.aurc == pytest.approx(np.mean(expected_risk))


def test_ensemble_error_decomposition_satisfies_exact_vector_identity() -> None:
    predictions = np.array(
        [
            [[0.0, 0.0], [2.0, 0.0]],
            [[2.0, 0.0], [4.0, 0.0]],
        ]
    )
    targets = np.array([[1.0, 0.0], [1.0, 0.0]])
    result = ensemble_error_decomposition(predictions, targets)
    assert np.allclose(result.ensemble_mean, [[1.0, 0.0], [3.0, 0.0]])
    assert np.allclose(result.shared_squared_error, [0.0, 4.0])
    assert np.allclose(result.between_member_variance, [1.0, 1.0])
    assert np.allclose(result.mean_member_squared_error, [1.0, 5.0])
    assert np.allclose(
        result.mean_member_squared_error,
        result.shared_squared_error + result.between_member_variance,
        atol=1e-12,
    )
    assert result.member_mse == pytest.approx(
        result.shared_mse + result.between_member_variance_mean
    )


def test_within_run_permutation_is_deterministic_and_never_crosses_runs() -> None:
    weights = np.array([0.1, 0.2, 0.3, 10.0, 20.0, 30.0])
    runs = np.array(["A", "A", "A", "B", "B", "B"])
    first = permute_weights_within_runs(weights, runs, seed=42)
    second = permute_weights_within_runs(weights, runs, seed=42)
    assert np.array_equal(first, second)
    assert sorted(first[:3]) == [0.1, 0.2, 0.3]
    assert sorted(first[3:]) == [10.0, 20.0, 30.0]
    assert not np.array_equal(first, weights)


def test_delayed_exponential_weights_use_only_labels_available_at_origin() -> None:
    losses = np.tile([0.0, 1.0], (4, 1))
    result = causal_delayed_two_expert_weights(
        losses,
        frames=[0, 1, 2, 3],
        frames_per_second=1.0,
        label_delay_seconds=2.0,
        learning_rate=np.log(2.0),
    )
    assert np.allclose(result.weights[0], [0.5, 0.5])
    assert np.allclose(result.weights[1], [0.5, 0.5])
    assert np.allclose(result.weights[2], [2 / 3, 1 / 3])
    assert np.allclose(result.weights[3], [0.8, 0.2])
    assert np.array_equal(result.updates_applied, [0, 0, 1, 2])


def test_delayed_weights_are_stream_keyed_and_scene_reset_is_strict() -> None:
    result = causal_delayed_two_expert_weights(
        losses=np.array(
            [
                [0.0, 1.0],  # A, pedestrian p
                [1.0, 0.0],  # A, pedestrian q
                [0.0, 1.0],  # A, p after its first label
                [1.0, 0.0],  # A, q after its first label
                [0.0, 1.0],  # B, p: reset despite reused identifier
            ]
        ),
        frames=[0, 0, 1, 1, 0],
        frames_per_second=1.0,
        label_delay_seconds=1.0,
        learning_rate=np.log(2.0),
        stream_ids=["p", "q", "p", "q", "p"],
        scene_ids=["A", "A", "A", "A", "B"],
    )
    assert np.allclose(result.weights[2], [2 / 3, 1 / 3])
    assert np.allclose(result.weights[3], [1 / 3, 2 / 3])
    assert np.allclose(result.weights[4], [0.5, 0.5])
    assert result.updates_applied[4] == 0


def test_fixed_share_matches_two_expert_transition_and_current_loss_is_never_used() -> None:
    result = causal_delayed_two_expert_weights(
        losses=np.array([[0.0, 1.0], [0.0, 1.0]]),
        frames=[0, 1],
        frames_per_second=1.0,
        label_delay_seconds=0.0,
        learning_rate=np.log(2.0),
        fixed_share=0.1,
    )
    # Row zero remains prior-weighted although its delay is zero; its label is
    # processed only before the next forecast origin.
    assert np.allclose(result.weights[0], [0.5, 0.5])
    assert np.allclose(result.weights[1], [0.6333333333333333, 0.3666666666666667])
    assert np.array_equal(result.updates_applied, [0, 1])


def test_delayed_weights_reject_non_increasing_frames_within_stream() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        causal_delayed_two_expert_weights(
            losses=np.ones((2, 2)),
            frames=[2, 2],
            frames_per_second=1.0,
            label_delay_seconds=1.0,
            learning_rate=1.0,
            stream_ids=["p", "p"],
        )
