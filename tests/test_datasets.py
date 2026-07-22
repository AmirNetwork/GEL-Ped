from pathlib import Path

import numpy as np

from pedgeom.datasets import add_motion_features, lane_order_parameter, load_julich_trajectory


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

