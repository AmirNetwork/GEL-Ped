import numpy as np

from pedgeom.calibration import VelocitySamples
from pedgeom.graph_baseline import _graph_arrays, fit_graph_interaction_network


def _samples() -> VelocitySamples:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(20, 6, 2))
    features[:, 1] = np.array([1.0, 0.0])
    features[:, 5] = 0.0
    raw = np.zeros((20, 8, 5))
    return VelocitySamples(
        "synthetic",
        features,
        rng.normal(size=(20, 2)),
        np.repeat(np.arange(4), 5),
        np.tile(np.arange(5), 4),
        raw_neighbours=raw,
    )


def test_wall_free_graph_schema_removes_both_zero_columns() -> None:
    samples = _samples()
    with_wall, _, _ = _graph_arrays(samples, include_wall=True)
    wall_free, _, _ = _graph_arrays(samples, include_wall=False)
    assert with_wall.shape == (20, 12)
    assert wall_free.shape == (20, 10)
    assert np.allclose(wall_free, np.delete(with_wall, (5, 11), axis=1))


def test_wall_free_setting_is_retained_for_inference() -> None:
    samples = _samples()
    model = fit_graph_interaction_network(
        [samples], width=8, epochs=1, batch_size=10, include_wall=False
    )
    assert model.include_wall is False
    assert model.predict(samples).shape == (20, 2)
