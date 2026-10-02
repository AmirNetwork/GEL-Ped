# Author: Amir Ghorbani
import numpy as np
import pytest

from pedgeom.reactive_planner import (
    PlannerParameters,
    ReplayFrame,
    adaptive_clearances,
    benchmark_callable,
    benchmark_model_stack,
    candidate_velocities,
    choose_reactive_velocity,
    closest_approach_distances,
    derive_replay_route,
    replay_episode,
)


def _parameters(**overrides: float) -> PlannerParameters:
    values = {
        "replan_seconds": 0.4,
        "lookahead_seconds": 0.8,
        "robot_radius_m": 0.3,
        "pedestrian_radius_m": 0.25,
        "nominal_speed_mps": 1.0,
        "fixed_clearance_m": 0.55,
        "adaptive_extra_clearance_m": 0.35,
        "candidate_speeds_mps": (0.0, 0.5, 1.0),
        "candidate_heading_offsets_deg": (-45.0, 0.0, 45.0),
        "intrusion_distance_m": 0.55,
        "stop_speed_mps": 0.1,
    }
    values.update(overrides)
    return PlannerParameters(**values)


def test_candidate_lattice_has_one_stop_and_goal_aligned_nominal_velocity() -> None:
    candidates = candidate_velocities([0.0, 0.0], [2.0, 0.0], _parameters())
    assert len(candidates.velocities) == 7
    assert np.count_nonzero(candidates.speeds == 0.0) == 1
    assert any(np.allclose(velocity, [1.0, 0.0]) for velocity in candidates.velocities)


def test_closest_approach_distance_is_exact_for_head_on_motion() -> None:
    distance = closest_approach_distances(
        [0.0, 0.0],
        [[1.0, 0.0]],
        [[2.0, 0.0]],
        [[-1.0, 0.0]],
        horizon_seconds=1.0,
    )
    assert distance.shape == (1, 1)
    assert distance[0, 0] == pytest.approx(0.0)


def test_reactive_choice_prefers_straight_nominal_progress_when_clear() -> None:
    decision = choose_reactive_velocity(
        [0.0, 0.0],
        [2.0, 0.0],
        np.empty((0, 2)),
        np.empty((0, 2)),
        0.55,
        _parameters(),
    )
    assert decision.feasible
    assert decision.speed_mps == 1.0
    assert decision.heading_offset_deg == 0.0
    assert np.allclose(decision.velocity, [1.0, 0.0])


def test_adaptive_clearance_expands_only_when_graph_is_downweighted() -> None:
    clearance = adaptive_clearances([1.0, 0.5, 0.0], _parameters())
    assert np.allclose(clearance, [0.55, 0.725, 0.9])


def test_route_uses_minor_first_frame_axis_and_no_future_positions() -> None:
    positions = np.array([[-2.0, -0.2], [0.0, 0.0], [2.0, 0.2]])
    start, goal = derive_replay_route(positions, padding_m=0.55)
    axis = (goal - start) / np.linalg.norm(goal - start)
    # The major cloud direction is nearly horizontal, so the cross-flow route
    # must be nearly vertical.
    assert abs(axis[1]) > 0.9
    assert np.linalg.norm(goal - start) >= 1.1


def test_replay_reports_completion_progress_and_stop_metrics() -> None:
    parameters = _parameters(
        robot_radius_m=0.05,
        candidate_speeds_mps=(0.0, 1.0),
        candidate_heading_offsets_deg=(0.0,),
    )
    empty = np.empty((0, 2))
    frames = [
        ReplayFrame(float(index), empty, empty, empty, np.empty(0))
        for index in range(3)
    ]
    result = replay_episode(
        frames,
        [0.0, 0.0],
        [0.8, 0.0],
        parameters,
        adaptive_margin=False,
    )
    assert result.completed
    assert not result.failure
    assert result.steps == 2
    assert result.travel_time_seconds == pytest.approx(0.8)
    assert result.progress_fraction == 1.0
    assert result.personal_space_intrusions == 0
    assert result.stop_count == 0


def test_replay_counts_realised_intrusion_without_claiming_reciprocal_response() -> None:
    parameters = _parameters(
        fixed_clearance_m=0.01,
        intrusion_distance_m=0.55,
        candidate_speeds_mps=(1.0,),
        candidate_heading_offsets_deg=(0.0,),
    )
    frame = ReplayFrame(
        0.0,
        np.array([[0.2, 0.0]]),
        np.array([[0.0, 0.0]]),
        np.array([[0.0, 0.0]]),
        np.array([1.0]),
    )
    result = replay_episode(
        [frame],
        [0.0, 0.0],
        [2.0, 0.0],
        parameters,
        adaptive_margin=False,
    )
    assert result.personal_space_intrusions == 1
    assert result.minimum_separation_m == pytest.approx(0.0)
    assert result.failure


class _FixedPredictor:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, samples: object, speed_cap: float = 2.5) -> np.ndarray:
        count = len(samples.target)
        return np.full((count, 2), min(self.value, speed_cap))


class _Samples:
    target = np.zeros((5, 2))


def test_latency_helpers_return_finite_stage_specific_summaries() -> None:
    simple = benchmark_callable(lambda: np.arange(10) ** 2, warmup=1, repeats=3)
    assert simple["repeats"] == 3
    assert simple["median_ms"] >= 0.0
    rows = benchmark_model_stack(
        [_FixedPredictor(0.2), _FixedPredictor(0.4)],
        _FixedPredictor(0.1),
        _Samples(),
        threshold=0.05,
        scale=0.1,
        speed_cap=3.0,
        warmup=1,
        repeats=2,
    )
    assert [row["stage"] for row in rows] == [
        "all ensemble members",
        "feature-based fallback",
        "guard post-processing",
        "end-to-end guarded stack",
    ]
    assert all(row["pedestrians"] == 5 for row in rows)
    assert all(row["median_ms"] >= 0.0 for row in rows)
