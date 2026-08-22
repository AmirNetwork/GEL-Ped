# Author: Amir Ghorbani
"""Parameter objects with explicit SI units."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class LegacyParameters:
    """Parameters for the submitted potential model.

    For the printed reciprocal potential, `destination_strength` needs units m^3/s^2.
    For the inferred linear potential, it has the table's units m/s^2. The latter is
    included because the speed formula in Section 6.1 can only be recovered from a
    constant-magnitude destination acceleration, not the printed reciprocal potential.
    """

    destination_strength: float = 13.5
    destination_form: Literal["reciprocal", "linear"] = "reciprocal"
    pedestrian_strength: float = 10.0
    obstacle_strength: float = 10.0
    pedestrian_range: float = 0.3
    obstacle_range: float = 0.2
    radius: float = 0.2
    softening: float = 0.05
    reset_displacement_scale: float = 2.0


@dataclass(frozen=True)
class AnisotropicParameters:
    """Parameters for the first anisotropic geometric pilot."""

    desired_speed: float = 1.35
    relaxation_time: float = 0.35
    interaction_range: float = 0.8
    metric_strength: float = 8.0
    anticipation_horizon: float = 1.2
    closing_speed_scale: float = 0.25
    lateral_tilt: float = 0.45
    speed_sensitivity: float = 1.5
    radius: float = 0.2
    maximum_speed: float = 2.0
    wall_range: float = 0.35
    wall_strength: float = 1.5
    collision_time_headway: float = 0.5
    safety_margin: float = 0.03


@dataclass(frozen=True)
class AVMParameters:
    """Parameters for the published Anticipation Velocity Model baseline."""

    radius: float = 0.2
    desired_speed: float = 1.35
    time_gap: float = 0.5
    repulsion_strength: float = 2.0
    interaction_range: float = 0.3
    turning_time: float = 0.3
    prediction_time: float = 0.8
    interaction_cutoff: float = 3.0
    wall_range: float = 0.35
    wall_strength: float = 1.5


@dataclass(frozen=True)
class GeometricGradientParameters:
    """Canonical reconstruction of the manuscript's implemented movement rule.

    The potential creates a social-force-like acceleration field. `mobility_time`
    converts that field into an instantaneous desired velocity, which is the
    mathematically explicit form of resetting velocity at each decision update.
    """

    mobility_time: float = 0.1
    maximum_speed: float = 1.5
    velocity_relaxation_time: float = 0.0
    potential: LegacyParameters = field(
        default_factory=lambda: LegacyParameters(destination_form="linear")
    )
