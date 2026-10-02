# Author: Amir Ghorbani
"""Lightweight replay planner for pedestrian-forecast operational audits.

The planner is deliberately modest: it searches a frozen discrete set of robot
speeds and heading offsets, assumes constant forecast velocity over a short
look-ahead, and evaluates realised clearance against recorded pedestrian
motion.  It does not model reciprocal pedestrian response and must not be
described as a closed-loop human-robot simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pedgeom.runtime_guard import one_sided_gate_weights


@dataclass(frozen=True)
class PlannerParameters:
    """Frozen reactive-planner parameters expressed in SI units."""

    replan_seconds: float
    lookahead_seconds: float
    robot_radius_m: float
    pedestrian_radius_m: float
    nominal_speed_mps: float
    fixed_clearance_m: float
    adaptive_extra_clearance_m: float
    candidate_speeds_mps: tuple[float, ...]
    candidate_heading_offsets_deg: tuple[float, ...]
    intrusion_distance_m: float
    stop_speed_mps: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PlannerParameters":
        """Build and validate parameters from ``referee3_lock.json``."""

        result = cls(
            replan_seconds=float(values["replan_seconds"]),
            lookahead_seconds=float(values["lookahead_seconds"]),
            robot_radius_m=float(values["robot_radius_m"]),
            pedestrian_radius_m=float(values["pedestrian_radius_m"]),
            nominal_speed_mps=float(values["nominal_speed_mps"]),
            fixed_clearance_m=float(values["fixed_clearance_m"]),
            adaptive_extra_clearance_m=float(values["adaptive_extra_clearance_m"]),
            candidate_speeds_mps=tuple(float(value) for value in values["candidate_speeds_mps"]),
            candidate_heading_offsets_deg=tuple(
                float(value) for value in values["candidate_heading_offsets_deg"]
            ),
            intrusion_distance_m=float(values["intrusion_distance_m"]),
            stop_speed_mps=float(values["stop_speed_mps"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        positive = {
            "replan_seconds": self.replan_seconds,
            "lookahead_seconds": self.lookahead_seconds,
            "robot_radius_m": self.robot_radius_m,
            "pedestrian_radius_m": self.pedestrian_radius_m,
            "nominal_speed_mps": self.nominal_speed_mps,
            "fixed_clearance_m": self.fixed_clearance_m,
            "intrusion_distance_m": self.intrusion_distance_m,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            not np.isfinite(self.adaptive_extra_clearance_m)
            or self.adaptive_extra_clearance_m < 0
        ):
            raise ValueError("adaptive_extra_clearance_m must be non-negative and finite")
        if not np.isfinite(self.stop_speed_mps) or self.stop_speed_mps < 0:
            raise ValueError("stop_speed_mps must be non-negative and finite")
        if not self.candidate_speeds_mps or not self.candidate_heading_offsets_deg:
            raise ValueError("candidate speed and heading sets must not be empty")
        if any(not np.isfinite(value) or value < 0 for value in self.candidate_speeds_mps):
            raise ValueError("candidate speeds must be non-negative and finite")
        if any(not np.isfinite(value) for value in self.candidate_heading_offsets_deg):
            raise ValueError("candidate headings must be finite")


@dataclass(frozen=True)
class CandidateSet:
    velocities: NDArray[np.float64]
    speeds: NDArray[np.float64]
    heading_offsets_deg: NDArray[np.float64]


@dataclass(frozen=True)
class ReactiveDecision:
    velocity: NDArray[np.float64]
    speed_mps: float
    heading_offset_deg: float
    predicted_min_separation_m: float
    clearance_margin_m: float
    feasible: bool
    candidate_index: int


@dataclass(frozen=True)
class ReplayFrame:
    """One paired replay origin; arrays contain only pedestrians present now."""

    frame: float
    pedestrian_positions: NDArray[np.float64]
    forecast_velocities: NDArray[np.float64]
    realised_velocities: NDArray[np.float64]
    graph_weights: NDArray[np.float64]


@dataclass(frozen=True)
class ReplayTraceStep:
    frame: float
    robot_x: float
    robot_y: float
    velocity_x: float
    velocity_y: float
    speed_mps: float
    predicted_min_separation_m: float
    realised_min_separation_m: float
    clearance_margin_m: float
    feasible: bool
    intrusion: bool
    stopped: bool


@dataclass(frozen=True)
class ReplayResult:
    completed: bool
    failure: bool
    steps: int
    elapsed_seconds: float
    travel_time_seconds: float
    route_length_m: float
    travelled_distance_m: float
    progress_fraction: float
    personal_space_intrusions: int
    intrusion_rate: float
    minimum_separation_m: float
    stop_count: int
    stop_frequency: float
    infeasible_count: int
    trace: tuple[ReplayTraceStep, ...]


def _points(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def candidate_velocities(
    robot_position: ArrayLike,
    goal_position: ArrayLike,
    parameters: PlannerParameters,
) -> CandidateSet:
    """Construct the frozen velocity lattice relative to the current goal bearing."""

    parameters.validate()
    robot = np.asarray(robot_position, dtype=float)
    goal = np.asarray(goal_position, dtype=float)
    if robot.shape != (2,) or goal.shape != (2,) or not np.all(np.isfinite([robot, goal])):
        raise ValueError("robot_position and goal_position must be finite 2-vectors")
    delta = goal - robot
    if np.linalg.norm(delta) <= 1e-12:
        return CandidateSet(np.zeros((1, 2)), np.zeros(1), np.zeros(1))
    base_angle = float(np.arctan2(delta[1], delta[0]))
    velocities: list[np.ndarray] = []
    speeds: list[float] = []
    offsets: list[float] = []
    zero_added = False
    for speed in parameters.candidate_speeds_mps:
        if speed <= 1e-12:
            if not zero_added:
                velocities.append(np.zeros(2))
                speeds.append(0.0)
                offsets.append(0.0)
                zero_added = True
            continue
        for offset in parameters.candidate_heading_offsets_deg:
            angle = base_angle + np.deg2rad(offset)
            velocities.append(speed * np.array([np.cos(angle), np.sin(angle)]))
            speeds.append(speed)
            offsets.append(offset)
    if not velocities:
        raise ValueError("candidate set contains no usable velocity")
    return CandidateSet(
        velocities=np.asarray(velocities),
        speeds=np.asarray(speeds),
        heading_offsets_deg=np.asarray(offsets),
    )


def closest_approach_distances(
    robot_position: ArrayLike,
    robot_velocities: ArrayLike,
    pedestrian_positions: ArrayLike,
    pedestrian_velocities: ArrayLike,
    *,
    horizon_seconds: float,
) -> NDArray[np.float64]:
    """Exact constant-velocity closest distances for every candidate/pedestrian."""

    robot = np.asarray(robot_position, dtype=float)
    candidates = _points("robot_velocities", robot_velocities)
    pedestrians = _points("pedestrian_positions", pedestrian_positions)
    pedestrian_velocity = _points("pedestrian_velocities", pedestrian_velocities)
    if robot.shape != (2,) or not np.all(np.isfinite(robot)):
        raise ValueError("robot_position must be a finite 2-vector")
    if pedestrians.shape != pedestrian_velocity.shape:
        raise ValueError("pedestrian positions and velocities must have matching shapes")
    if not np.isfinite(horizon_seconds) or horizon_seconds < 0:
        raise ValueError("horizon_seconds must be non-negative and finite")
    if len(pedestrians) == 0:
        return np.empty((len(candidates), 0), dtype=float)
    offset = pedestrians[None, :, :] - robot[None, None, :]
    relative_velocity = pedestrian_velocity[None, :, :] - candidates[:, None, :]
    speed_squared = np.sum(relative_velocity**2, axis=2)
    time = np.divide(
        -np.sum(offset * relative_velocity, axis=2),
        speed_squared,
        out=np.zeros_like(speed_squared),
        where=speed_squared > 1e-12,
    )
    time = np.clip(time, 0.0, horizon_seconds)
    return np.linalg.norm(offset + time[:, :, None] * relative_velocity, axis=2)


def adaptive_clearances(
    graph_weights: ArrayLike,
    parameters: PlannerParameters,
) -> NDArray[np.float64]:
    """Expand centre-to-centre clearance where the graph expert is downweighted."""

    weights = np.asarray(graph_weights, dtype=float)
    if weights.ndim != 1 or not np.all(np.isfinite(weights)):
        raise ValueError("graph_weights must be a finite vector")
    if np.any((weights < 0.0) | (weights > 1.0)):
        raise ValueError("graph_weights must lie in [0, 1]")
    return parameters.fixed_clearance_m + parameters.adaptive_extra_clearance_m * (
        1.0 - weights
    )


def choose_reactive_velocity(
    robot_position: ArrayLike,
    goal_position: ArrayLike,
    pedestrian_positions: ArrayLike,
    forecast_velocities: ArrayLike,
    clearances_m: float | ArrayLike,
    parameters: PlannerParameters,
) -> ReactiveDecision:
    """Select the safest progress-making candidate under the frozen lattice.

    Feasible candidates satisfy every pedestrian-specific clearance.  Among
    them, the planner maximises goal-directed progress and then prefers a
    smaller turn.  If none is feasible, it maximises worst clearance margin
    and then prefers lower speed.  This deterministic rule has no learned or
    test-tuned cost weights.
    """

    robot = np.asarray(robot_position, dtype=float)
    goal = np.asarray(goal_position, dtype=float)
    pedestrian = _points("pedestrian_positions", pedestrian_positions)
    forecast = _points("forecast_velocities", forecast_velocities)
    if pedestrian.shape != forecast.shape:
        raise ValueError("pedestrian positions and forecasts must have matching shapes")
    clearance = np.asarray(clearances_m, dtype=float)
    if clearance.ndim == 0:
        clearance = np.full(len(pedestrian), float(clearance))
    if clearance.shape != (len(pedestrian),):
        raise ValueError("clearances_m must be scalar or one value per pedestrian")
    if not np.all(np.isfinite(clearance)) or np.any(clearance < 0):
        raise ValueError("clearances_m must be finite and non-negative")

    candidate = candidate_velocities(robot, goal, parameters)
    distances = closest_approach_distances(
        robot,
        candidate.velocities,
        pedestrian,
        forecast,
        horizon_seconds=parameters.lookahead_seconds,
    )
    if len(pedestrian):
        margins = distances - clearance[None, :]
        minimum_distance = np.min(distances, axis=1)
        minimum_margin = np.min(margins, axis=1)
        feasible = np.all(margins >= 0.0, axis=1)
    else:
        minimum_distance = np.full(len(candidate.velocities), np.inf)
        minimum_margin = np.full(len(candidate.velocities), np.inf)
        feasible = np.ones(len(candidate.velocities), dtype=bool)

    direction = goal - robot
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-12 else np.zeros(2)
    progress = candidate.velocities @ direction
    feasible_indices = np.flatnonzero(feasible)
    if len(feasible_indices):
        selected = max(
            feasible_indices,
            key=lambda index: (
                progress[index],
                -abs(candidate.heading_offsets_deg[index]),
                candidate.speeds[index],
                -int(index),
            ),
        )
    else:
        selected = max(
            range(len(candidate.velocities)),
            key=lambda index: (
                minimum_margin[index],
                -candidate.speeds[index],
                progress[index],
                -abs(candidate.heading_offsets_deg[index]),
                -int(index),
            ),
        )
    return ReactiveDecision(
        velocity=np.array(candidate.velocities[selected], copy=True),
        speed_mps=float(candidate.speeds[selected]),
        heading_offset_deg=float(candidate.heading_offsets_deg[selected]),
        predicted_min_separation_m=float(minimum_distance[selected]),
        clearance_margin_m=float(minimum_margin[selected]),
        feasible=bool(feasible[selected]),
        candidate_index=int(selected),
    )


def derive_replay_route(
    first_frame_positions: ArrayLike,
    *,
    padding_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Derive a deterministic cross-flow route from the first observed frame.

    The route follows the lower-variance spatial axis and spans the observed
    pedestrians plus the fixed clearance padding.  Only the first frame is
    used, so future pedestrian positions cannot define the route.
    """

    positions = _points("first_frame_positions", first_frame_positions)
    if len(positions) == 0:
        raise ValueError("at least one first-frame pedestrian is required")
    if not np.isfinite(padding_m) or padding_m <= 0:
        raise ValueError("padding_m must be positive and finite")
    centre = np.median(positions, axis=0)
    if len(positions) == 1 or np.allclose(positions, positions[0]):
        axis = np.array([1.0, 0.0])
    else:
        covariance = np.cov((positions - centre).T, bias=True)
        _, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, 0]
        nonzero = np.flatnonzero(np.abs(axis) > 1e-12)
        if len(nonzero) and axis[nonzero[0]] < 0:
            axis = -axis
    projection = (positions - centre) @ axis
    start = centre + (float(np.min(projection)) - padding_m) * axis
    goal = centre + (float(np.max(projection)) + padding_m) * axis
    return start, goal


def _validate_replay_frame(frame: ReplayFrame) -> None:
    positions = _points("pedestrian_positions", frame.pedestrian_positions)
    forecast = _points("forecast_velocities", frame.forecast_velocities)
    realised = _points("realised_velocities", frame.realised_velocities)
    weights = np.asarray(frame.graph_weights, dtype=float)
    if positions.shape != forecast.shape or positions.shape != realised.shape:
        raise ValueError("replay position and velocity arrays must match")
    if weights.shape != (len(positions),) or not np.all(np.isfinite(weights)):
        raise ValueError("each replay pedestrian needs one finite graph weight")
    if np.any((weights < 0.0) | (weights > 1.0)):
        raise ValueError("replay graph weights must lie in [0, 1]")


def replay_episode(
    frames: Sequence[ReplayFrame],
    start_position: ArrayLike,
    goal_position: ArrayLike,
    parameters: PlannerParameters,
    *,
    adaptive_margin: bool,
) -> ReplayResult:
    """Run one paired open-loop crowd replay for a forecast policy."""

    if not frames:
        raise ValueError("frames must not be empty")
    for frame in frames:
        _validate_replay_frame(frame)
    robot = np.asarray(start_position, dtype=float).copy()
    goal = np.asarray(goal_position, dtype=float)
    if robot.shape != (2,) or goal.shape != (2,) or not np.all(np.isfinite([robot, goal])):
        raise ValueError("start_position and goal_position must be finite 2-vectors")
    route_length = float(np.linalg.norm(goal - robot))
    if route_length <= 1e-12:
        raise ValueError("start and goal must be distinct")

    trace: list[ReplayTraceStep] = []
    travelled = 0.0
    elapsed = 0.0
    for frame in frames:
        if np.linalg.norm(goal - robot) <= parameters.robot_radius_m:
            break
        clearance: float | np.ndarray
        if adaptive_margin:
            clearance = adaptive_clearances(frame.graph_weights, parameters)
        else:
            clearance = parameters.fixed_clearance_m
        decision = choose_reactive_velocity(
            robot,
            goal,
            frame.pedestrian_positions,
            frame.forecast_velocities,
            clearance,
            parameters,
        )
        realised_distance = closest_approach_distances(
            robot,
            decision.velocity[None, :],
            frame.pedestrian_positions,
            frame.realised_velocities,
            horizon_seconds=parameters.replan_seconds,
        )
        realised_minimum = (
            float(np.min(realised_distance)) if realised_distance.size else float("inf")
        )
        intrusion = realised_minimum < parameters.intrusion_distance_m
        stopped = decision.speed_mps <= parameters.stop_speed_mps
        trace.append(
            ReplayTraceStep(
                frame=float(frame.frame),
                robot_x=float(robot[0]),
                robot_y=float(robot[1]),
                velocity_x=float(decision.velocity[0]),
                velocity_y=float(decision.velocity[1]),
                speed_mps=decision.speed_mps,
                predicted_min_separation_m=decision.predicted_min_separation_m,
                realised_min_separation_m=realised_minimum,
                clearance_margin_m=decision.clearance_margin_m,
                feasible=decision.feasible,
                intrusion=intrusion,
                stopped=stopped,
            )
        )
        displacement = parameters.replan_seconds * decision.velocity
        robot += displacement
        travelled += float(np.linalg.norm(displacement))
        elapsed += parameters.replan_seconds

    completed = bool(np.linalg.norm(goal - robot) <= parameters.robot_radius_m)
    steps = len(trace)
    intrusions = sum(step.intrusion for step in trace)
    stops = sum(step.stopped for step in trace)
    infeasible = sum(not step.feasible for step in trace)
    minimum = min(
        (step.realised_min_separation_m for step in trace),
        default=float("inf"),
    )
    progress = float(np.clip(1.0 - np.linalg.norm(goal - robot) / route_length, 0.0, 1.0))
    return ReplayResult(
        completed=completed,
        failure=not completed,
        steps=steps,
        elapsed_seconds=elapsed,
        travel_time_seconds=elapsed if completed else float("nan"),
        route_length_m=route_length,
        travelled_distance_m=travelled,
        progress_fraction=progress,
        personal_space_intrusions=int(intrusions),
        intrusion_rate=float(intrusions / steps) if steps else 0.0,
        minimum_separation_m=float(minimum),
        stop_count=int(stops),
        stop_frequency=float(stops / steps) if steps else 0.0,
        infeasible_count=int(infeasible),
        trace=tuple(trace),
    )


def benchmark_callable(
    function: Callable[[], Any],
    *,
    warmup: int = 10,
    repeats: int = 100,
) -> dict[str, float | int]:
    """Benchmark a CPU callable with warm-up and robust latency summaries."""

    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    for _ in range(warmup):
        function()
    durations = np.empty(repeats, dtype=float)
    for index in range(repeats):
        start = perf_counter_ns()
        function()
        durations[index] = (perf_counter_ns() - start) / 1e6
    return {
        "warmup": warmup,
        "repeats": repeats,
        "mean_ms": float(np.mean(durations)),
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.quantile(durations, 0.95, method="linear")),
        "minimum_ms": float(np.min(durations)),
        "maximum_ms": float(np.max(durations)),
    }


def benchmark_model_stack(
    member_models: Sequence[Any],
    fallback_model: Any,
    samples: Any,
    *,
    threshold: float,
    scale: float,
    speed_cap: float,
    warmup: int = 5,
    repeats: int = 50,
) -> list[dict[str, float | int | str]]:
    """Benchmark members, fallback, guard post-processing, and end-to-end stack.

    Models need only expose ``predict(samples, speed_cap=...)``.  This utility
    is intended for trained-model objects retained by a full experimental run;
    unlike array-only timing, it measures actual predictor inference.
    """

    if not member_models:
        raise ValueError("at least one member model is required")

    def member_prediction() -> np.ndarray:
        return np.stack(
            [model.predict(samples, speed_cap=speed_cap) for model in member_models]
        )

    def fallback_prediction() -> np.ndarray:
        return np.asarray(fallback_model.predict(samples, speed_cap=speed_cap))

    cached_members = member_prediction()
    cached_fallback = fallback_prediction()

    def guard_postprocess() -> np.ndarray:
        ensemble = np.mean(cached_members, axis=0)
        score = np.sqrt(
            np.mean(
                np.sum((cached_members - ensemble[None, :, :]) ** 2, axis=2),
                axis=0,
            )
        )
        weight = one_sided_gate_weights(score, threshold=threshold, scale=scale)
        return weight[:, None] * ensemble + (1.0 - weight[:, None]) * cached_fallback

    def end_to_end() -> np.ndarray:
        members = member_prediction()
        fallback = fallback_prediction()
        ensemble = np.mean(members, axis=0)
        score = np.sqrt(
            np.mean(np.sum((members - ensemble[None, :, :]) ** 2, axis=2), axis=0)
        )
        weight = one_sided_gate_weights(score, threshold=threshold, scale=scale)
        return weight[:, None] * ensemble + (1.0 - weight[:, None]) * fallback

    rows = []
    for stage, function in (
        ("all ensemble members", member_prediction),
        ("feature-based fallback", fallback_prediction),
        ("guard post-processing", guard_postprocess),
        ("end-to-end guarded stack", end_to_end),
    ):
        rows.append(
            {
                "stage": stage,
                "pedestrians": int(len(cached_fallback)),
                **benchmark_callable(function, warmup=warmup, repeats=repeats),
            }
        )
    return rows


__all__ = [
    "CandidateSet",
    "PlannerParameters",
    "ReactiveDecision",
    "ReplayFrame",
    "ReplayResult",
    "ReplayTraceStep",
    "adaptive_clearances",
    "benchmark_callable",
    "benchmark_model_stack",
    "candidate_velocities",
    "choose_reactive_velocity",
    "closest_approach_distances",
    "derive_replay_route",
    "replay_episode",
]
