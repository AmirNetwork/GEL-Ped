# Author: Amir Ghorbani
"""Synthetic checks for exact-horizon trajectory resampling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pedgeom.calibration import build_velocity_samples
import pedgeom.datasets as datasets_module
from pedgeom.datasets import (
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


def test_resampling_uses_exact_common_times_without_extrapolation() -> None:
    source = _source_trajectory()
    result = resample_trajectory_to_fps(source, target_fps=10.0)

    first = result[result["pedestrian_id"] == 7]
    second = result[result["pedestrian_id"] == 11]
    np.testing.assert_array_equal(first["frame"], np.arange(1, 10))
    np.testing.assert_allclose(first["time_s"], np.arange(1, 10) / 10.0)
    np.testing.assert_array_equal(second["frame"], np.arange(3, 8))
    np.testing.assert_allclose(second["time_s"], np.arange(3, 8) / 10.0)

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
        source_frame = np.arange(0, 33)
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
