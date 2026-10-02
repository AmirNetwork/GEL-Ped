# Author: Amir Ghorbani
"""Synthetic checks for exact-horizon trajectory resampling."""

from __future__ import annotations

from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from pedgeom.calibration import build_velocity_samples
import pedgeom.datasets as datasets_module
from pedgeom.datasets import (
    add_motion_features,
    load_resampled_julich_archive_trajectory,
    load_resampled_julich_trajectory,
    resample_trajectory_to_fps,
)


def _source_trajectory() -> pd.DataFrame:
    blocks = []
    for pedestrian_id, first_frame, last_frame, y_offset in (
        (7, 1, 15, 0.0),
        (11, 4, 12, 1.0),
    ):
        frame = np.arange(first_frame, last_frame + 1)
        time_s = frame / 16.0
        blocks.append(
            pd.DataFrame(
                {
                    "pedestrian_id": pedestrian_id,
                    "frame": frame,
                    "time_s": time_s,
                    "x_m": 2.0 * time_s,
                    "y_m": y_offset - time_s,
                    "height_m": 1.70 + 0.01 * pedestrian_id,
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


def test_resampling_uses_only_past_samples_and_stays_inside_support() -> None:
    source = _source_trajectory()
    result = resample_trajectory_to_fps(source, target_fps=10.0)

    first = result[result["pedestrian_id"] == 7]
    second = result[result["pedestrian_id"] == 11]
    np.testing.assert_array_equal(first["frame"], np.arange(2, 10))
    np.testing.assert_allclose(first["time_s"], np.arange(2, 10) / 10.0)
    np.testing.assert_array_equal(second["frame"], np.arange(4, 8))
    np.testing.assert_allclose(second["time_s"], np.arange(4, 8) / 10.0)

    support = source.groupby("pedestrian_id")["time_s"].agg(["min", "max"])
    for pedestrian_id, group in result.groupby("pedestrian_id"):
        assert group["time_s"].min() >= support.loc[pedestrian_id, "min"]
        assert group["time_s"].max() <= support.loc[pedestrian_id, "max"]
        assert np.all(np.diff(group["frame"]) > 0)
        np.testing.assert_allclose(group["x_m"], 2.0 * group["time_s"])


def test_resampled_loader_preserves_schema_and_identifier_dtype(monkeypatch) -> None:
    source = _source_trajectory()
    monkeypatch.setattr(
        datasets_module,
        "load_julich_trajectory",
        lambda _path, fps: source.assign(time_s=source["frame"] / fps),
    )
    result = load_resampled_julich_trajectory("unused.txt", source_fps=16.0, target_fps=10.0)

    assert list(result.columns) == [
        "pedestrian_id",
        "frame",
        "time_s",
        "x_m",
        "y_m",
        "height_m",
    ]
    assert result["pedestrian_id"].dtype == source["pedestrian_id"].dtype
    np.testing.assert_allclose(result["time_s"], result["frame"] / 10.0)


def test_ten_hz_grid_supports_exact_point_eight_and_point_four_second_horizons() -> None:
    blocks = []
    for pedestrian_id, y_m in ((1, 0.0), (2, 1.0)):
        # Long enough to isolate exact 0.8/0.4-s compatibility from the
        # builder's separate goal-history eligibility requirement.
        source_frame = np.arange(0, 49)
        source_time = source_frame / 16.0
        blocks.append(
            pd.DataFrame(
                {
                    "pedestrian_id": pedestrian_id,
                    "frame": source_frame,
                    "time_s": source_time,
                    "x_m": source_time,
                    "y_m": y_m,
                    "height_m": 1.75,
                }
            )
        )
    resampled = resample_trajectory_to_fps(pd.concat(blocks), target_fps=10.0)
    samples = build_velocity_samples(
        "unused.txt",
        interaction_range=0.8,
        horizon_frames=4,
        history_frames=8,
        frame_stride=4,
        frames_per_second=10.0,
        maximum_samples=None,
        wall_y_bounds=None,
        goal_method="prior_velocity",
        cardinal_routes=False,
        speed_outlier_threshold=None,
        filter_target_speed=False,
        loader=lambda _: resampled,
    )

    assert samples.origin_time_s is not None
    np.testing.assert_allclose(samples.origin_time_s, samples.frame / 10.0)
    np.testing.assert_allclose(samples.features[:, 0, 0], 1.0)
    np.testing.assert_allclose(samples.target[:, 0], 1.0)
    np.testing.assert_allclose(samples.features[:, 0, 1], 0.0)
    np.testing.assert_allclose(samples.target[:, 1], 0.0)


def test_archive_member_loader_resamples_in_place_without_extrapolation(tmp_path) -> None:
    archive_path = tmp_path / "trajectories.zip"
    rows = [
        f"7 {frame} {200.0 * frame / 16.0:.6f} {-100.0 * frame / 16.0:.6f} 170"
        for frame in range(1, 16)
    ]
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("run.txt", "\n".join(rows) + "\n")

    result = load_resampled_julich_archive_trajectory(
        archive_path,
        "run.txt",
        source_fps=16.0,
        target_fps=10.0,
    )

    np.testing.assert_array_equal(result["frame"], np.arange(2, 10))
    np.testing.assert_allclose(result["time_s"], np.arange(2, 10) / 10.0)
    np.testing.assert_allclose(result["x_m"], 2.0 * result["time_s"])
    np.testing.assert_allclose(result["y_m"], -result["time_s"])
    with pytest.raises(ValueError, match="flat .txt"):
        load_resampled_julich_archive_trajectory(archive_path, "../run.txt")


def test_causal_resampling_does_not_bridge_a_missing_source_frame() -> None:
    source_frame = np.array([0, 1, 2, 3, 6, 7, 8, 9])
    source_time = source_frame / 16.0
    source = pd.DataFrame(
        {
            "pedestrian_id": 1,
            "frame": source_frame,
            "time_s": source_time,
            "x_m": source_time,
            "y_m": 0.0,
            "height_m": 1.7,
        }
    )

    result = resample_trajectory_to_fps(source, target_fps=10.0, source_fps=16.0)

    assert 3 not in set(result["frame"])
    assert 4 not in set(result["frame"])
    assert 5 in set(result["frame"])
    motion = add_motion_features(result)
    resumed = motion[motion["frame"] == 5].iloc[0]
    assert np.isnan(resumed["vx_mps"])
    assert np.isnan(resumed["vy_mps"])


def test_causal_resampling_is_invariant_to_observations_after_grid_time() -> None:
    source = _source_trajectory()
    changed = source.copy()
    changed.loc[changed["time_s"] > 0.4, ["x_m", "y_m"]] += 1000.0

    original = resample_trajectory_to_fps(source, target_fps=10.0, source_fps=16.0)
    perturbed = resample_trajectory_to_fps(changed, target_fps=10.0, source_fps=16.0)
    early = original["time_s"] <= 0.4

    np.testing.assert_allclose(
        original.loc[early, ["x_m", "y_m"]],
        perturbed.loc[early, ["x_m", "y_m"]],
    )


def test_archive_loader_rejects_wrong_width_and_duplicate_origins(tmp_path) -> None:
    wrong_width = tmp_path / "wrong.zip"
    with ZipFile(wrong_width, "w") as archive:
        archive.writestr("run.txt", "1 0 0 0 170 99\n")
    with pytest.raises(ValueError, match="exactly five"):
        load_resampled_julich_archive_trajectory(wrong_width, "run.txt")

    duplicate = tmp_path / "duplicate.zip"
    with ZipFile(duplicate, "w") as archive:
        archive.writestr("run.txt", "1 0 0 0 170\n1 0 1 0 170\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_resampled_julich_archive_trajectory(duplicate, "run.txt")
