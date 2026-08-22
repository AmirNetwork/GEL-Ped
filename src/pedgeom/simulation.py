# Author: Amir Ghorbani
"""Simulation entry points for legacy, corrected, and revised pilot models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .avm import avm_speed, desired_avm_direction
from .geometry import collision_free_speed, geometric_direction
from .integrators import rk4_step
from .models import (
    AVMParameters,
    AnisotropicParameters,
    GeometricGradientParameters,
    LegacyParameters,
)
from .potential import accelerations

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SimulationResult:
    times: FloatArray
    positions: FloatArray
    velocities: FloatArray
    model_name: str

    @property
    def minimum_separation(self) -> float:
        if self.positions.shape[1] < 2:
            return float("inf")
        minimum = float("inf")
        for frame in self.positions:
            for first in range(len(frame)):
                for second in range(first + 1, len(frame)):
                    minimum = min(minimum, float(np.linalg.norm(frame[first] - frame[second])))
        return minimum


def _validate_scene(
    initial_positions: FloatArray,
    initial_velocities: FloatArray,
    destinations: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    positions = np.asarray(initial_positions, dtype=float)
    velocities = np.asarray(initial_velocities, dtype=float)
    goals = np.asarray(destinations, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (agents, 2)")
    if velocities.shape != positions.shape or goals.shape != positions.shape:
        raise ValueError("velocities and destinations must match positions")
    return positions.copy(), velocities.copy(), goals.copy()


def _time_grid(duration: float, step: float) -> FloatArray:
    count = int(round(duration / step))
    if not np.isclose(count * step, duration):
        raise ValueError("duration must be an integer multiple of step")
    return np.linspace(0.0, duration, count + 1)


def simulate_corrected(
    initial_positions: FloatArray,
    initial_velocities: FloatArray,
    destinations: FloatArray,
    duration: float,
    step: float,
    parameters: LegacyParameters = LegacyParameters(),
    obstacles: FloatArray | None = None,
) -> SimulationResult:
    """Integrate the corrected persistent-velocity ODE with RK4."""

    positions, velocities, goals = _validate_scene(
        initial_positions, initial_velocities, destinations
    )
    times = _time_grid(duration, step)
    position_history = np.empty((len(times), *positions.shape))
    velocity_history = np.empty_like(position_history)
    state = np.stack((positions, velocities), axis=0)
    position_history[0], velocity_history[0] = positions, velocities

    def derivative(current: FloatArray) -> FloatArray:
        current_positions, current_velocities = current
        current_accelerations = accelerations(
            current_positions, goals, obstacles, parameters
        )
        return np.stack((current_velocities, current_accelerations), axis=0)

    for frame in range(1, len(times)):
        state = rk4_step(state, step, derivative)
        position_history[frame], velocity_history[frame] = state

    return SimulationResult(times, position_history, velocity_history, "corrected_ode")


def simulate_legacy_reset(
    initial_positions: FloatArray,
    destinations: FloatArray,
    duration: float,
    step: float,
    parameters: LegacyParameters = LegacyParameters(),
    obstacles: FloatArray | None = None,
) -> SimulationResult:
    """Implement the paper's stated velocity reset at every scene update.

    Each interval is solved from zero velocity and its displacement is multiplied by
    the paper's scale factor. Reported velocity is displacement divided by the scene
    timestep, because persistent ODE velocity is discarded by construction.
    """

    zero_velocities = np.zeros_like(initial_positions, dtype=float)
    positions, _, goals = _validate_scene(initial_positions, zero_velocities, destinations)
    times = _time_grid(duration, step)
    position_history = np.empty((len(times), *positions.shape))
    velocity_history = np.zeros_like(position_history)
    position_history[0] = positions

    for frame in range(1, len(times)):
        local_state = np.stack((positions, np.zeros_like(positions)), axis=0)

        def derivative(current: FloatArray) -> FloatArray:
            current_positions, current_velocities = current
            current_accelerations = accelerations(
                current_positions, goals, obstacles, parameters
            )
            return np.stack((current_velocities, current_accelerations), axis=0)

        integrated = rk4_step(local_state, step, derivative)
        displacement = (
            integrated[0] - positions
        ) * parameters.reset_displacement_scale
        positions = positions + displacement
        position_history[frame] = positions
        velocity_history[frame] = displacement / step

    return SimulationResult(times, position_history, velocity_history, "legacy_reset")


def simulate_geometric_gradient(
    initial_positions: FloatArray,
    destinations: FloatArray,
    duration: float,
    step: float,
    parameters: GeometricGradientParameters = GeometricGradientParameters(),
    obstacles: FloatArray | None = None,
    initial_velocities: FloatArray | None = None,
) -> SimulationResult:
    """Simulate the canonical first-order geometric potential model.

    The update is `v* = -mobility_time * grad(U)`. With zero relaxation this is
    the direct decision rule implicit in the manuscript's per-step velocity reset,
    but it remains well-defined when the numerical integration step changes.
    """

    if initial_velocities is None:
        initial_velocities = np.zeros_like(initial_positions, dtype=float)
    positions, velocities, goals = _validate_scene(
        initial_positions, initial_velocities, destinations
    )
    times = _time_grid(duration, step)
    position_history = np.empty((len(times), *positions.shape))
    velocity_history = np.empty_like(position_history)
    position_history[0], velocity_history[0] = positions, velocities

    for frame in range(1, len(times)):
        target_velocities = parameters.mobility_time * accelerations(
            positions, goals, obstacles, parameters.potential
        )
        target_speeds = np.linalg.norm(target_velocities, axis=1)
        over_limit = target_speeds > parameters.maximum_speed
        if np.any(over_limit):
            target_velocities[over_limit] *= (
                parameters.maximum_speed / target_speeds[over_limit]
            )[:, None]
        if parameters.velocity_relaxation_time > 0.0:
            blend = min(step / parameters.velocity_relaxation_time, 1.0)
            velocities = velocities + blend * (target_velocities - velocities)
        else:
            velocities = target_velocities
        positions = positions + step * velocities
        position_history[frame], velocity_history[frame] = positions, velocities

    return SimulationResult(
        times, position_history, velocity_history, "canonical_geometric_gradient"
    )


def simulate_anisotropic(
    initial_positions: FloatArray,
    initial_velocities: FloatArray,
    destinations: FloatArray,
    duration: float,
    step: float,
    parameters: AnisotropicParameters = AnisotropicParameters(),
) -> SimulationResult:
    """Integrate the calibration-ready anisotropic geometric pilot."""

    positions, velocities, goals = _validate_scene(
        initial_positions, initial_velocities, destinations
    )
    times = _time_grid(duration, step)
    position_history = np.empty((len(times), *positions.shape))
    velocity_history = np.empty_like(position_history)
    position_history[0], velocity_history[0] = positions, velocities

    for frame in range(1, len(times)):
        accelerations_now = np.empty_like(positions)
        for agent in range(len(positions)):
            direction, risk, _ = geometric_direction(
                agent, positions, velocities, goals, parameters
            )
            target_speed = parameters.desired_speed / (
                1.0 + parameters.speed_sensitivity * risk
            )
            target_speed = collision_free_speed(
                agent,
                positions,
                velocities,
                direction,
                target_speed,
                parameters.radius,
                parameters.collision_time_headway,
                parameters.safety_margin,
            )
            target_velocity = target_speed * direction
            accelerations_now[agent] = (
                target_velocity - velocities[agent]
            ) / parameters.relaxation_time

        velocities = velocities + step * accelerations_now
        speeds = np.linalg.norm(velocities, axis=1)
        over_limit = speeds > parameters.maximum_speed
        if np.any(over_limit):
            velocities[over_limit] *= (
                parameters.maximum_speed / speeds[over_limit]
            )[:, None]
        positions = positions + step * velocities
        position_history[frame], velocity_history[frame] = positions, velocities

    return SimulationResult(times, position_history, velocity_history, "anisotropic_pilot")


def simulate_periodic_corridor(
    initial_positions: FloatArray,
    directions: FloatArray,
    duration: float,
    step: float,
    corridor_length: float,
    corridor_width: float,
    parameters: AnisotropicParameters = AnisotropicParameters(),
) -> SimulationResult:
    """Run a controlled bidirectional-flow diagnostic with periodic x boundaries.

    This is a mesoscopic stress test for lane formation, not a reconstruction of the
    Jülich inflow/outflow protocol. Opposing streams have preferred x directions and
    interact through the anisotropic local metric.
    """

    positions = np.asarray(initial_positions, dtype=float).copy()
    flow_directions = np.asarray(directions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (agents, 2)")
    if flow_directions.shape != (len(positions),):
        raise ValueError("directions must have one value per agent")
    if not np.all(np.isin(flow_directions, [-1.0, 1.0])):
        raise ValueError("directions must be -1 or 1")

    velocities = np.column_stack(
        (flow_directions * parameters.desired_speed, np.zeros(len(positions)))
    )
    times = _time_grid(duration, step)
    position_history = np.empty((len(times), *positions.shape))
    velocity_history = np.empty_like(position_history)
    position_history[0], velocity_history[0] = positions, velocities

    for frame in range(1, len(times)):
        goals = positions + np.column_stack(
            (100.0 * flow_directions, np.zeros(len(positions)))
        )
        target_velocities = np.empty_like(positions)
        for agent in range(len(positions)):
            local_positions = positions.copy()
            wrapped_dx = positions[:, 0] - positions[agent, 0]
            wrapped_dx = (wrapped_dx + corridor_length / 2.0) % corridor_length - (
                corridor_length / 2.0
            )
            local_positions[:, 0] = positions[agent, 0] + wrapped_dx
            local_goals = goals.copy()
            local_goals[agent] = local_positions[agent] + np.array(
                [100.0 * flow_directions[agent], 0.0]
            )
            direction, risk, _ = geometric_direction(
                agent, local_positions, velocities, local_goals, parameters
            )
            lower_wall = np.exp(-positions[agent, 1] / parameters.wall_range)
            upper_wall = np.exp(
                -(corridor_width - positions[agent, 1]) / parameters.wall_range
            )
            wall_bias = parameters.wall_strength * (lower_wall - upper_wall)
            direction = direction + np.array([0.0, wall_bias])
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            target_speed = parameters.desired_speed / (
                1.0 + parameters.speed_sensitivity * risk
            )
            target_speed = collision_free_speed(
                agent,
                positions,
                velocities,
                direction,
                target_speed,
                parameters.radius,
                parameters.collision_time_headway,
                parameters.safety_margin,
                periodic_length=corridor_length,
            )
            target_velocities[agent] = target_speed * direction

        # The dense-flow diagnostic uses a first-order velocity update. A relaxed
        # second-order update can overshoot a collision-free speed constraint.
        velocities = target_velocities
        speeds = np.linalg.norm(velocities, axis=1)
        over_limit = speeds > parameters.maximum_speed
        if np.any(over_limit):
            velocities[over_limit] *= (
                parameters.maximum_speed / speeds[over_limit]
            )[:, None]
        positions = positions + step * velocities
        positions[:, 0] = ((positions[:, 0] + corridor_length / 2.0) % corridor_length) - (
            corridor_length / 2.0
        )
        positions[:, 1] = np.clip(positions[:, 1], 0.01, corridor_width - 0.01)
        position_history[frame], velocity_history[frame] = positions, velocities

    return SimulationResult(
        times, position_history, velocity_history, "anisotropic_periodic_corridor"
    )


def simulate_avm_periodic_corridor(
    initial_positions: FloatArray,
    directions: FloatArray,
    duration: float,
    step: float,
    corridor_length: float,
    corridor_width: float,
    parameters: AVMParameters = AVMParameters(),
) -> SimulationResult:
    """Run the AVM baseline in the same controlled periodic corridor."""

    positions = np.asarray(initial_positions, dtype=float).copy()
    flow_directions = np.asarray(directions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (agents, 2)")
    if flow_directions.shape != (len(positions),):
        raise ValueError("directions must have one value per agent")
    preferred = np.column_stack((flow_directions, np.zeros(len(positions))))
    current_directions = preferred.copy()
    velocities = parameters.desired_speed * current_directions
    times = _time_grid(duration, step)
    position_history = np.empty((len(times), *positions.shape))
    velocity_history = np.empty_like(position_history)
    position_history[0], velocity_history[0] = positions, velocities

    for frame in range(1, len(times)):
        desired_directions = np.empty_like(current_directions)
        for agent in range(len(positions)):
            desired = desired_avm_direction(
                agent,
                positions,
                velocities,
                current_directions,
                preferred[agent],
                parameters,
                periodic_length=corridor_length,
            )
            lower_wall = np.exp(-positions[agent, 1] / parameters.wall_range)
            upper_wall = np.exp(
                -(corridor_width - positions[agent, 1]) / parameters.wall_range
            )
            desired = desired + np.array(
                [0.0, parameters.wall_strength * (lower_wall - upper_wall)]
            )
            desired_directions[agent] = desired / max(float(np.linalg.norm(desired)), 1e-12)

        current_directions += (
            step / parameters.turning_time
        ) * (desired_directions - current_directions)
        current_directions /= np.maximum(
            np.linalg.norm(current_directions, axis=1, keepdims=True), 1e-12
        )
        speeds = np.array(
            [
                avm_speed(
                    agent,
                    positions,
                    current_directions[agent],
                    parameters,
                    periodic_length=corridor_length,
                )
                for agent in range(len(positions))
            ]
        )
        velocities = speeds[:, None] * current_directions
        positions = positions + step * velocities
        positions[:, 0] = ((positions[:, 0] + corridor_length / 2.0) % corridor_length) - (
            corridor_length / 2.0
        )
        positions[:, 1] = np.clip(positions[:, 1], 0.01, corridor_width - 0.01)
        position_history[frame], velocity_history[frame] = positions, velocities

    return SimulationResult(times, position_history, velocity_history, "avm_periodic_corridor")
