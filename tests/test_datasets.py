# Author: Amir Ghorbani
from pathlib import Path

import numpy as np

from pedgeom.datasets import (
    add_entry_goal_directions,
    add_motion_features,
    add_prior_velocity_goal_directions,
    lane_order_parameter,
    load_julich_trajectory,
)


def test_loader_and_motion_features(tmp_path: Path):
    source = tmp_path / "trajectory.txt"
    source.write_text(
        "1 0 0 100 175\n1 1 4 100 175\n2 0 100 200 180\n2 1 96 200 180\n",
        encoding="utf-8",
    )
    data = add_motion_features(load_julich_trajectory(source, fps=25.0))
    assert list(data.columns[:6]) == [
        "pedestrian_id",
        "frame",
        "time_s",
        "x_m",
        "y_m",
        "height_m",
    ]
    np.testing.assert_allclose(data["height_m"].to_numpy(), [1.75, 1.75, 1.8, 1.8])
    assert data.drop_duplicates("pedestrian_id")["direction"].tolist() == [1, -1]
    np.testing.assert_allclose(data.dropna()["speed_mps"].to_numpy(), [1.0, 1.0])


def test_lane_order_is_one_for_separated_directions():
    rows = []
    for pedestrian in range(10):
        direction = 1 if pedestrian < 5 else -1
        y = 0.25 if direction == 1 else 1.25
        rows.append(
            {
                "pedestrian_id": pedestrian,
                "frame": 0,
                "x_m": float(pedestrian),
                "y_m": y,
                "direction": direction,
            }
        )
    import pandas as pd

    order = lane_order_parameter(
        pd.DataFrame(rows), y_min=0.0, y_max=2.0, bins=2, frame_stride=1
    )
    assert order["lane_order"].iat[0] == 1.0


def test_entry_goal_uses_only_initial_history() -> None:
    import pandas as pd

    data = pd.DataFrame(
        {
            "pedestrian_id": [1, 1, 1, 1],
            "frame": [0, 10, 20, 30],
            "x_m": [0.0, 1.0, 1.0, -4.0],
            "y_m": [0.0, 0.1, 3.0, 5.0],
        }
    )
    result = add_entry_goal_directions(data, history_frames=10)
    assert not bool(result.loc[result["frame"] == 0, "goal_valid"].iat[0])
    assert np.allclose(result.loc[result["frame"] >= 10, ["goal_x", "goal_y"]], [1.0, 0.0])


def test_prior_velocity_goal_is_continuous_and_past_only() -> None:
    import pandas as pd

    data = pd.DataFrame(
        {
            "pedestrian_id": [1, 1, 1],
            "frame": [0, 10, 20],
            "x_m": [0.0, 0.3, 0.6],
            "y_m": [0.0, 0.4, 0.8],
        }
    )
    result = add_prior_velocity_goal_directions(data, history_frames=10)
    assert not bool(result.loc[result["frame"] == 0, "goal_valid"].iat[0])
    np.testing.assert_allclose(
        result.loc[result["frame"] >= 10, ["goal_x", "goal_y"]],
        np.tile([0.6, 0.8], (2, 1)),
    )
