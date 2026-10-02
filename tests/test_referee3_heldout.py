# Author: Amir Ghorbani
"""Firewall tests for the untouched referee-3 T-junction path."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from pedgeom.calibration import FEATURE_NAMES, VelocitySamples

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import referee3_generate_heldout as heldout  # noqa: E402
import referee3_generate_predictions as pipeline  # noqa: E402


def _archive_registry(path: Path) -> dict:
    with ZipFile(path) as archive:
        members = []
        for info in archive.infolist():
            rows = sum(1 for line in archive.read(info).splitlines() if line.strip())
            members.append(
                {
                    "name": info.filename,
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                    "rows": rows,
                    "columns": 5,
                }
            )
    return {
        "analysis_freeze_commit": "freeze",
        "source_url": "https://example.invalid/archive.zip",
        "archive_bytes": path.stat().st_size,
        "archive_sha256": heldout._sha256(path),
        "trajectory_schema": [
            "pedestrian_id",
            "frame",
            "x_cm",
            "y_cm",
            "z_cm",
        ],
        "source_fps": 16.0,
        "target_fps": 10.0,
        "resampling": {"method": "past-only"},
        "members": members,
    }


def _write_archive(path: Path) -> dict:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "run-a.txt",
            "1 0 0 0 170\n1 1 6.25 0 170\n1 2 12.5 0 170\n",
        )
    return _archive_registry(path)


def _config() -> dict:
    return {
        "task": {
            "history_seconds": 0.8,
            "prediction_seconds": 0.4,
            "origin_spacing_seconds": 0.4,
            "interaction_range_m": 0.8,
            "pedestrian_radius_m": 0.25,
            "neighbour_cutoff_m": 3.0,
            "past_speed_quality_limit_mps": 3.0,
            "maximum_neighbours": 8,
        },
        "replication": {"member_seed_base": 101},
    }


def _batch() -> VelocitySamples:
    frame = np.array([8], dtype=int)
    return VelocitySamples(
        run="run-a",
        features=np.zeros((1, len(FEATURE_NAMES), 2)),
        target=np.zeros((1, 2)),
        pedestrian_id=np.array([1]),
        frame=frame,
        position=np.zeros((1, 2)),
        neighbour_count=np.zeros(1),
        occupancy=np.ones(1),
        goal_method="prior_velocity",
        raw_neighbours=np.zeros((1, 8, 5)),
        origin_time_s=frame / 10.0,
    )


def test_archive_inventory_validates_exact_hash_members_and_schema(tmp_path) -> None:
    archive_path = tmp_path / "archive.zip"
    registry = _write_archive(archive_path)

    inventory = heldout.inspect_registered_archive(archive_path, registry)

    assert inventory["archive_sha256"] == registry["archive_sha256"]
    assert inventory["members"] == registry["members"]
    assert inventory["performance_analysis_run"] is False
    wrong_hash = dict(registry, archive_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        heldout.inspect_registered_archive(archive_path, wrong_hash)
    wrong_members = dict(registry, members=[])
    with pytest.raises(RuntimeError, match="member order/set"):
        heldout.inspect_registered_archive(archive_path, wrong_members)


def test_heldout_builder_uses_exact_locked_protocol(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_build(source, **kwargs):
        captured["source"] = source
        captured.update(kwargs)
        return _batch()

    monkeypatch.setattr(heldout, "build_velocity_samples", fake_build)
    registry = {"source_fps": 16.0, "target_fps": 10.0}

    result = heldout._build_heldout_batch(
        tmp_path / "archive.zip",
        "run-a.txt",
        config=_config(),
        registry=registry,
        seed=123,
    )

    assert result.run == "run-a"
    assert captured["history_frames"] == 8
    assert captured["horizon_frames"] == 4
    assert captured["frame_stride"] == 4
    assert captured["frames_per_second"] == 10.0
    assert captured["maximum_samples"] is None
    assert captured["interaction_range"] == 0.8
    assert captured["goal_method"] == "prior_velocity"
    assert captured["wall_y_bounds"] is None
    assert captured["filter_target_speed"] is False
    assert captured["past_speed_outlier_threshold"] == 3.0


def test_one_shot_guard_refuses_any_existing_final_directory(tmp_path) -> None:
    expected = tmp_path / heldout.FINAL_DIRECTORY_NAME
    assert heldout._assert_one_shot_available(tmp_path) == expected
    expected.mkdir()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        heldout._assert_one_shot_available(tmp_path)


def test_bad_archive_aborts_before_any_training(monkeypatch, tmp_path) -> None:
    archive_path = tmp_path / "not-registered.zip"
    _write_archive(archive_path)
    called = False

    def forbidden_training(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("training must not start")

    monkeypatch.setattr(heldout, "build_julich_fit_batches", forbidden_training)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        heldout.execute_once(archive_path, tmp_path / "work", workers=1)
    assert called is False


def test_deployed_sentinel_is_label_blind_and_feature_specific(tmp_path) -> None:
    sentinel_path = tmp_path / "sentinel.npz"
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    batch = _batch()
    seeds = [11, 18]
    predictions = np.arange(4, dtype=np.float32).reshape(2, 1, 2)
    pipeline._write_deployed_sentinel(
        sentinel_path,
        batch,
        predictions,
        seeds,
        config_path=config_path,
    )
    graph_path = heldout.project_root() / "src/pedgeom/graph_baseline.py"

    changed_labels = replace(batch, target=batch.target + 999.0)
    loaded = heldout.verify_deployed_sentinel(
        sentinel_path,
        changed_labels,
        seeds,
        config_path=config_path,
        graph_path=graph_path,
    )
    np.testing.assert_array_equal(loaded, predictions)

    changed_features = replace(batch, features=batch.features + 1.0)
    with pytest.raises(RuntimeError, match="input fingerprint"):
        heldout.verify_deployed_sentinel(
            sentinel_path,
            changed_features,
            seeds,
            config_path=config_path,
            graph_path=graph_path,
        )
