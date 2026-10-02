# Author: Amir Ghorbani
import numpy as np
import pytest

from pedgeom.calibration import FEATURE_NAMES, VelocitySamples
from pedgeom.graph_baseline import _graph_arrays
from pedgeom.transformer_baseline import (
    _InteractionTransformer,
    _prepare_batches,
    _transformer_arrays,
    fit_transformer_interaction_network,
)


def _make_samples(
    *,
    run: str = "synthetic",
    n: int = 24,
    seed: int = 7,
    with_neighbours: bool = True,
) -> VelocitySamples:
    rng = np.random.default_rng(seed)
    features = rng.normal(scale=0.25, size=(n, len(FEATURE_NAMES), 2))
    features[:, 0] += np.array([0.8, 0.0])
    features[:, 1] = np.array([1.0, 0.0])
    features[:, 5] = 0.0
    raw = np.zeros((n, 4, 5), dtype=float)
    if with_neighbours:
        raw[:, 0, :4] = rng.normal(scale=0.3, size=(n, 4))
        raw[:, 0, 0] += 0.8
        raw[:, 0, 4] = 1.0
        raw[::2, 1, :4] = rng.normal(scale=0.2, size=(len(raw[::2]), 4))
        raw[::2, 1, 1] += 0.6
        raw[::2, 1, 4] = 1.0
    local_residual = np.column_stack(
        (
            0.08 * np.tanh(raw[:, 0, 0]),
            -0.05 * np.tanh(raw[:, 0, 1]),
        )
    )
    target = features[:, 0] + local_residual
    return VelocitySamples(
        run,
        features,
        target,
        np.arange(n),
        np.arange(n),
        raw_neighbours=raw,
    )


def _fit(samples: VelocitySamples, *, seed: int = 11, epochs: int = 2):
    return fit_transformer_interaction_network(
        [samples],
        width=8,
        num_heads=2,
        num_layers=1,
        epochs=epochs,
        batch_size=12,
        seed=seed,
        num_threads=1,
    )


def test_transformer_uses_exact_graph_protocol_wall_free_arrays() -> None:
    samples = _make_samples()
    node, edge, mask = _transformer_arrays(samples)
    expected_node, expected_edge, expected_mask = _graph_arrays(samples, include_wall=False)
    assert node.shape[1] == 10
    assert np.array_equal(node, expected_node)
    assert np.array_equal(edge, expected_edge)
    assert np.array_equal(mask, expected_mask)


def test_run_balancing_gives_complete_runs_equal_total_weight() -> None:
    first = _make_samples(run="short", n=8, seed=1)
    second = _make_samples(run="long", n=24, seed=2)
    prepared = _prepare_batches([first, second], residual=True)
    first_mass = np.sum(prepared.weights[:8])
    second_mass = np.sum(prepared.weights[8:])
    assert first_mass == pytest.approx(second_mass)
    assert np.mean(prepared.weights) == pytest.approx(1.0)


def test_predict_parameter_count_and_speed_cap() -> None:
    samples = _make_samples()
    model = _fit(samples)
    prediction = model.predict(samples, speed_cap=0.4)
    assert prediction.shape == (len(samples.target), 2)
    assert model.parameter_count > 0
    assert np.all(np.linalg.norm(prediction, axis=1) <= 0.4 + 1e-7)
    with pytest.raises(ValueError, match="speed_cap"):
        model.predict(samples, speed_cap=0.0)


def test_masked_attention_is_invariant_to_neighbour_slot_permutation() -> None:
    samples = _make_samples()
    model = _fit(samples)
    permuted_raw = samples.raw_neighbours[:, [1, 0, 3, 2], :]
    permuted = VelocitySamples(
        "permuted",
        samples.features,
        samples.target,
        samples.pedestrian_id,
        samples.frame,
        raw_neighbours=permuted_raw,
    )
    assert np.allclose(model.predict(samples), model.predict(permuted), atol=2e-6)


def test_goal_frame_transformer_is_rotation_equivariant() -> None:
    samples = _make_samples()
    model = _fit(samples)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated_raw = samples.raw_neighbours.copy()
    rotated_raw[:, :, :2] = samples.raw_neighbours[:, :, :2] @ rotation.T
    rotated_raw[:, :, 2:4] = samples.raw_neighbours[:, :, 2:4] @ rotation.T
    rotated = VelocitySamples(
        "rotated",
        samples.features @ rotation.T,
        samples.target @ rotation.T,
        samples.pedestrian_id,
        samples.frame,
        raw_neighbours=rotated_raw,
    )
    assert np.allclose(model.predict(rotated), model.predict(samples) @ rotation.T, atol=2e-6)


def test_fitting_is_deterministic_for_fixed_seed() -> None:
    samples = _make_samples()
    first = _fit(samples, seed=29)
    second = _fit(samples, seed=29)
    assert np.array_equal(first.predict(samples), second.predict(samples))
    for first_parameter, second_parameter in zip(
        first.network.parameters(), second.network.parameters(), strict=True
    ):
        assert np.array_equal(
            first_parameter.detach().numpy(), second_parameter.detach().numpy()
        )


def test_complete_run_validation_early_stopping_and_weight_averaging() -> None:
    training = _make_samples(run="train", seed=5)
    validation = _make_samples(run="validation", seed=6)
    model = fit_transformer_interaction_network(
        [training],
        validation_batches=[validation],
        width=8,
        num_heads=2,
        epochs=10,
        batch_size=12,
        seed=37,
        early_stopping_patience=1,
        minimum_improvement=1e6,
        weight_average_epochs=2,
        num_threads=1,
    )
    assert model.epochs_trained == 2
    assert model.best_epoch == 0
    assert len(model.validation_history) == model.epochs_trained
    # The selected checkpoint is the first epoch, so only that causally prior
    # state is available to the requested trailing average.
    assert model.averaged_epochs == 1


def test_empty_neighbourhood_is_finite_because_summary_token_is_unmasked() -> None:
    samples = _make_samples(with_neighbours=False)
    model = _fit(samples, epochs=1)
    assert np.all(np.isfinite(model.predict(samples)))


def test_transformer_hyperparameter_validation() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _InteractionTransformer(node_features=10, width=10, num_heads=4)
