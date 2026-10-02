# Author: Amir Ghorbani
"""Lightweight infrastructure tests for referee-3 prediction generation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

from pedgeom.calibration import FEATURE_NAMES, VelocitySamples

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import referee3_generate_predictions as pipeline  # noqa: E402


def _batch(run: str, sample_count: int) -> VelocitySamples:
    features = np.zeros((sample_count, len(FEATURE_NAMES), 2), dtype=float)
    features[:, 0, 0] = 1.0
    features[:, 1, 0] = 1.0
    frame = np.arange(sample_count, dtype=int) * 4
    return VelocitySamples(
        run=run,
        features=features,
        target=np.column_stack((np.ones(sample_count), np.zeros(sample_count))),
        pedestrian_id=np.arange(sample_count, dtype=int),
        frame=frame,
        position=np.zeros((sample_count, 2), dtype=float),
        neighbour_count=np.zeros(sample_count, dtype=float),
        occupancy=np.ones(sample_count, dtype=float),
        goal_method="prior_velocity",
        raw_neighbours=np.zeros((sample_count, 2, 5), dtype=float),
        origin_time_s=frame / 10.0,
    )


def _config() -> dict:
    return {
        "replication": {
            "member_pool_size": 3,
            "member_seed_base": 11,
            "member_seed_stride": 7,
            "graph_width": 4,
            "training_epochs": 1,
        }
    }


def test_cache_signature_is_deterministic_and_tracks_variable_length_runs() -> None:
    options = {"width": 4, "epochs": 1, "threads_per_worker": 1}
    training = [_batch("train", 3)]
    first = pipeline._pool_cache_signature(
        training, [_batch("short", 2), _batch("long", 5)], options
    )
    repeated = pipeline._pool_cache_signature(
        training, [_batch("short", 2), _batch("long", 5)], options
    )
    changed_length = pipeline._pool_cache_signature(
        training, [_batch("short", 2), _batch("long", 6)], options
    )
    changed_order = pipeline._pool_cache_signature(
        training, [_batch("long", 5), _batch("short", 2)], options
    )

    assert first == repeated
    assert first != changed_length
    assert first != changed_order


def test_single_process_worker_preserves_variable_prediction_lengths(monkeypatch) -> None:
    class FakeModel:
        parameter_count = 17

        def __init__(self, seed: int) -> None:
            self.seed = seed

        def predict(self, batch: VelocitySamples, speed_cap: float) -> np.ndarray:
            assert speed_cap == 100.0
            return np.full((len(batch.target), 2), self.seed / 100.0)

    def fake_fit(_training, *, seed, **_kwargs):
        return FakeModel(seed)

    monkeypatch.setattr(pipeline, "fit_graph_interaction_network", fake_fit)
    evaluation = [_batch("short", 2), _batch("long", 5)]
    pipeline._worker_initialise(
        [_batch("train", 3)],
        evaluation,
        {
            "width": 4,
            "epochs": 1,
            "threads_per_worker": 1,
            "residual": True,
            "include_wall": False,
            "raw_prediction_speed_cap_mps": 100.0,
        },
    )
    seed, predictions, seconds, parameters = pipeline._fit_member_and_predict(13)

    assert seed == 13
    assert [prediction.shape for prediction in predictions] == [(2, 2), (5, 2)]
    assert seconds >= 0.0
    assert parameters == 17


def test_bundle_validates_shapes_and_records_member_identity(monkeypatch) -> None:
    captured = {}

    def fake_atomic_savez(path: Path, **arrays) -> None:
        captured["path"] = path
        captured.update(arrays)

    monkeypatch.setattr(pipeline, "_atomic_savez", fake_atomic_savez)
    batch = _batch("run", 4)
    pipeline._write_bundle(
        Path("bundle.npz"),
        batch,
        members=np.zeros((2, 4, 2)),
        fallback=np.zeros((4, 2)),
        constant_velocity=np.zeros((4, 2)),
        mahalanobis=np.zeros(4),
        member_seeds=[101, 202],
        role="test",
        fit_role="all calibration runs",
    )

    np.testing.assert_array_equal(captured["member_seeds"], [101, 202])
    assert captured["members_raw"].shape == (2, 4, 2)
    assert captured["origin_time_s"].shape == (4,)
    with pytest.raises(ValueError, match="must have shape"):
        pipeline._write_bundle(
            Path("bad.npz"),
            batch,
            members=np.zeros((2, 3, 2)),
            fallback=np.zeros((4, 2)),
            constant_velocity=np.zeros((4, 2)),
            mahalanobis=np.zeros(4),
            member_seeds=[101, 202],
            role="test",
            fit_role="all calibration runs",
        )


def test_member_seed_generation_rejects_empty_or_duplicate_pools() -> None:
    assert pipeline._member_seeds(_config()) == [11, 18, 25]
    with pytest.raises(ValueError, match="at least one"):
        pipeline._member_seeds(_config(), 0)
    duplicate = _config()
    duplicate["replication"]["member_seed_stride"] = 0
    with pytest.raises(ValueError, match="unique"):
        pipeline._member_seeds(duplicate)


def test_member_pool_reuses_valid_cache_and_rebuilds_corrupt_entry(
    monkeypatch, tmp_path: Path
) -> None:
    fitted: list[int] = []

    class FakeModel:
        parameter_count = 17

        def __init__(self, seed: int) -> None:
            self.seed = seed

        def predict(self, batch: VelocitySamples, speed_cap: float) -> np.ndarray:
            assert speed_cap == 100.0
            return np.full((len(batch.target), 2), self.seed / 100.0)

    def fake_fit(_training, *, seed, **_kwargs):
        fitted.append(seed)
        return FakeModel(seed)

    monkeypatch.setattr(pipeline, "fit_graph_interaction_network", fake_fit)
    training = [_batch("train", 3)]
    evaluation = [_batch("short", 2), _batch("long", 5)]
    prefix = tmp_path / "members" / "fold"

    first_manifest: list[dict] = []
    first = pipeline._predict_member_pool(
        training,
        evaluation,
        [11, 18],
        cache_prefix=prefix,
        config=_config(),
        workers=1,
        manifest_rows=first_manifest,
    )
    assert fitted == [11, 18]
    assert [values.shape for values in first] == [(2, 2, 2), (2, 5, 2)]
    assert len(first_manifest) == 2

    second_manifest: list[dict] = []
    second = pipeline._predict_member_pool(
        training,
        evaluation,
        [11, 18],
        cache_prefix=prefix,
        config=_config(),
        workers=1,
        manifest_rows=second_manifest,
    )
    assert fitted == [11, 18]
    for first_run, second_run in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_run, second_run)

    corrupt = pipeline._member_cache_path(prefix, 18)
    corrupt.write_bytes(b"not a NumPy archive")
    pipeline._predict_member_pool(
        training,
        evaluation,
        [11, 18],
        cache_prefix=prefix,
        config=_config(),
        workers=1,
        manifest_rows=[],
    )
    assert fitted == [11, 18, 18]


def test_parallel_branch_passes_only_file_backed_manifest_to_spawned_workers(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        parameter_count = 19

        def __init__(self, seed: int) -> None:
            self.seed = seed

        def predict(self, batch: VelocitySamples, speed_cap: float) -> np.ndarray:
            return np.full((len(batch.target), 2), self.seed / speed_cap)

    def fake_fit(_training, *, seed, **_kwargs):
        return FakeModel(seed)

    class ImmediateFuture:
        def __init__(self, function, *args) -> None:
            self.value = function(*args)

        def result(self):
            return self.value

    class FakeExecutor:
        def __init__(self, *, max_workers, initializer, initargs) -> None:
            captured["max_workers"] = max_workers
            captured["initializer"] = initializer
            captured["initargs"] = initargs
            assert isinstance(initargs[0], str)
            assert isinstance(initargs[1], dict)
            assert not any(isinstance(value, VelocitySamples) for value in initargs)
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            # Drop mmap references before TemporaryDirectory removes the files
            # in this same-process executor stand-in.
            pipeline._worker_initialise([], [], {})

        def submit(self, function, *args):
            return ImmediateFuture(function, *args)

    monkeypatch.setattr(pipeline, "fit_graph_interaction_network", fake_fit)
    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(pipeline, "as_completed", lambda futures: futures)
    training = [_batch("train", 3)]
    evaluation = [_batch("short", 2), _batch("long", 5)]
    predictions = pipeline._predict_member_pool(
        training,
        evaluation,
        [11, 18],
        cache_prefix=tmp_path / "parallel" / "fold",
        config=_config(),
        workers=2,
        manifest_rows=[],
    )

    assert captured["max_workers"] == 2
    assert captured["initializer"] is pipeline._worker_initialise_file_backed
    assert [values.shape for values in predictions] == [(2, 2, 2), (2, 5, 2)]
    assert not list((tmp_path / "parallel").glob(".*_workers_*"))


def test_target_blind_inference_cache_is_invariant_to_heldout_labels(
    monkeypatch, tmp_path: Path
) -> None:
    fitted: list[int] = []

    class FakeModel:
        parameter_count = 23

        def predict(self, batch: VelocitySamples, speed_cap: float) -> np.ndarray:
            assert np.count_nonzero(batch.target) == 0
            return batch.features[:, 0] / speed_cap

    def fake_fit(_training, *, seed, **_kwargs):
        fitted.append(seed)
        return FakeModel()

    monkeypatch.setattr(pipeline, "fit_graph_interaction_network", fake_fit)
    evaluation = _batch("heldout", 4)
    changed_labels = replace(evaluation, target=evaluation.target + 999.0)
    prefix = tmp_path / "blind" / "final_pool"

    first = pipeline._predict_member_pool(
        [_batch("train", 3)],
        [evaluation],
        [11],
        cache_prefix=prefix,
        config=_config(),
        workers=1,
        manifest_rows=[],
        evaluation_target_blind=True,
    )
    second = pipeline._predict_member_pool(
        [_batch("train", 3)],
        [changed_labels],
        [11],
        cache_prefix=prefix,
        config=_config(),
        workers=1,
        manifest_rows=[],
        evaluation_target_blind=True,
    )

    assert fitted == [11]
    np.testing.assert_array_equal(first[0], second[0])


def test_non_neural_prediction_path_never_exposes_evaluation_labels(
    monkeypatch,
) -> None:
    class FakeFallback:
        def predict(self, batch: VelocitySamples, speed_cap: float) -> np.ndarray:
            assert np.count_nonzero(batch.target) == 0
            return batch.features[:, 0] / speed_cap

    monkeypatch.setattr(
        pipeline,
        "settings",
        lambda _project: ({}, {}, {}),
    )
    monkeypatch.setattr(
        pipeline,
        "fit_gradient_boosted_residual",
        lambda *_args, **_kwargs: FakeFallback(),
    )
    monkeypatch.setattr(
        pipeline,
        "_fit_mahalanobis",
        lambda _training: (np.zeros(10), np.eye(10)),
    )

    def fake_mahalanobis(batch, _location, _precision):
        assert np.count_nonzero(batch.target) == 0
        return np.arange(len(batch.target), dtype=float)

    monkeypatch.setattr(pipeline, "_mahalanobis", fake_mahalanobis)
    config = _config()
    config["experts"] = {"fallback_prediction_speed_cap_mps": 3.0}
    first = _batch("heldout", 4)
    changed = replace(first, target=first.target + 999.0)

    original = pipeline._fit_non_neural([_batch("train", 3)], [first], config)
    perturbed = pipeline._fit_non_neural([_batch("train", 3)], [changed], config)

    for original_group, perturbed_group in zip(original, perturbed, strict=True):
        np.testing.assert_array_equal(original_group[0], perturbed_group[0])
