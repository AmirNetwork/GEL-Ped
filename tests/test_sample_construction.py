# Author: Amir Ghorbani
"""Focused tests for causal sample construction and representation parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pedgeom.calibration import build_velocity_samples
import pedgeom.external as external_module
from pedgeom.external import build_eth_ucy_samples


def _controlled_loader(_: Path) -> pd.DataFrame:
    rows = []
    first_x = (0.0, 1.0, 2.0, 6.0, 7.0, 8.0)
    second_x = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    for pedestrian_id, x_values, y in (
        (1, first_x, 0.0),
        (2, second_x, 1.0),
    ):
        for frame, x_m in enumerate(x_values):
            rows.append(
                {
                    "pedestrian_id": pedestrian_id,
                    "frame": frame,
                    "time_s": float(frame),
                    "x_m": x_m,
                    "y_m": y,
                    "height_m": 1.7,
                }
            )
    return pd.DataFrame(rows).sort_values(["pedestrian_id", "frame"])


def _build_controlled(**overrides):
    parameters = {
        "interaction_range": 0.8,
        "horizon_frames": 1,
        "history_frames": 1,
        "frame_stride": 1,
        "frames_per_second": 1.0,
        "maximum_samples": None,
        "wall_y_bounds": None,
        "goal_method": "prior_velocity",
        "cardinal_routes": False,
        "loader": _controlled_loader,
    }
    parameters.update(overrides)
    return build_velocity_samples("controlled.txt", **parameters)


def _eth_ucy_frame() -> pd.DataFrame:
    rows = []
    for pedestrian_id, offset in ((1, 0.0), (2, 1.0)):
        for step, frame in enumerate((0, 10, 20, 30, 40)):
            rows.append((frame, pedestrian_id, offset + 0.4 * step, 0.2 * pedestrian_id))
    return pd.DataFrame(rows, columns=("frame", "pedestrian_id", "x_m", "y_m"))


def _patch_eth_ucy_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _eth_ucy_frame()
    monkeypatch.setattr(external_module, "load_eth_ucy", lambda _: frame.copy())


def test_velocity_samples_have_explicit_origin_times_and_full_origin_mode() -> None:
    full = _build_controlled(speed_outlier_threshold=100.0)
    capped = _build_controlled(speed_outlier_threshold=100.0, maximum_samples=3)

    assert len(full.target) > len(capped.target) == 3
    assert full.origin_time_s is not None
    assert np.array_equal(full.origin_time_s, full.frame.astype(float))
    assert np.array_equal(capped.origin_time_s, capped.frame.astype(float))


def test_target_speed_filter_can_be_disabled_without_disabling_past_qc() -> None:
    filtered = _build_controlled(speed_outlier_threshold=3.0)
    inclusive = _build_controlled(
        speed_outlier_threshold=3.0,
        filter_target_speed=False,
    )

    filtered_origins = set(zip(filtered.pedestrian_id, filtered.frame, strict=True))
    inclusive_origins = set(zip(inclusive.pedestrian_id, inclusive.frame, strict=True))
    assert (1, 2) not in filtered_origins
    assert (1, 2) in inclusive_origins
    # The next origin still fails the independently retained causal past-speed QC.
    assert (1, 3) not in inclusive_origins


def test_prior_velocity_uses_one_declared_history_and_future_only_as_label() -> None:
    rows = []
    for pedestrian_id, y_m in ((1, 0.0), (2, 1.0)):
        for frame in range(13):
            rows.append(
                {
                    "pedestrian_id": pedestrian_id,
                    "frame": frame,
                    "time_s": frame / 10.0,
                    "x_m": frame / 10.0,
                    "y_m": y_m,
                    "height_m": 1.7,
                }
            )
    source = pd.DataFrame(rows)

    def build(frame: pd.DataFrame):
        return build_velocity_samples(
            "exact-history.txt",
            interaction_range=0.8,
            horizon_frames=4,
            history_frames=8,
            frame_stride=1,
            frames_per_second=10.0,
            maximum_samples=None,
            wall_y_bounds=None,
            goal_method="prior_velocity",
            cardinal_routes=False,
            speed_outlier_threshold=None,
            filter_target_speed=False,
            loader=lambda _: frame,
        )

    baseline = build(source)
    assert np.min(baseline.frame) == 8

    changed_future = source.copy()
    changed_future.loc[changed_future["frame"] > 8, "x_m"] += 2.0
    changed = build(changed_future)
    baseline_origin = np.flatnonzero(baseline.frame == 8)
    changed_origin = np.flatnonzero(changed.frame == 8)
    np.testing.assert_allclose(
        baseline.features[baseline_origin],
        changed.features[changed_origin],
    )
    assert not np.allclose(
        baseline.target[baseline_origin],
        changed.target[changed_origin],
    )


def test_eth_ucy_interaction_range_is_explicit_and_changes_only_field_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_eth_ucy_loader(monkeypatch)
    source = Path("scene.txt")
    short = build_eth_ucy_samples(
        source,
        history_steps=1,
        maximum_samples=None,
        interaction_range=0.35,
    )
    legacy = build_eth_ucy_samples(
        source,
        history_steps=1,
        maximum_samples=None,
    )
    long = build_eth_ucy_samples(
        source,
        history_steps=1,
        maximum_samples=None,
        interaction_range=0.8,
    )

    assert np.array_equal(short.frame, long.frame)
    assert np.allclose(short.target, long.target)
    assert np.allclose(short.features, legacy.features)
    assert not np.allclose(short.features[:, 2:5], long.features[:, 2:5])
    assert np.allclose(short.raw_neighbours, long.raw_neighbours)
    assert np.allclose(short.origin_time_s, short.frame / 25.0)


def test_eth_ucy_filters_and_full_origin_mode_are_independently_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_eth_ucy_loader(monkeypatch)
    source = Path("scene.txt")
    full = build_eth_ucy_samples(
        source,
        history_steps=1,
        maximum_samples=None,
        filter_target_speed=False,
        minimum_past_speed=None,
    )
    capped = build_eth_ucy_samples(
        source,
        history_steps=1,
        maximum_samples=2,
        filter_target_speed=False,
        minimum_past_speed=None,
    )

    assert len(full.target) > len(capped.target) == 2
    assert full.origin_time_s is not None
    assert np.allclose(full.origin_time_s, full.frame / 25.0)


def test_eth_ucy_target_filter_can_be_disabled_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _eth_ucy_frame()
    frame.loc[
        (frame["pedestrian_id"] == 1) & (frame["frame"] == 30), "x_m"
    ] += 1.6
    monkeypatch.setattr(external_module, "load_eth_ucy", lambda _: frame.copy())
    filtered = build_eth_ucy_samples(
        "scene.txt",
        history_steps=1,
        maximum_samples=None,
    )
    inclusive = build_eth_ucy_samples(
        "scene.txt",
        history_steps=1,
        maximum_samples=None,
        filter_target_speed=False,
    )

    filtered_origins = set(zip(filtered.pedestrian_id, filtered.frame, strict=True))
    inclusive_origins = set(zip(inclusive.pedestrian_id, inclusive.frame, strict=True))
    assert (1, 20) not in filtered_origins
    assert (1, 20) in inclusive_origins
    assert (1, 30) not in inclusive_origins
