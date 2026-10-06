# Author: Amir Ghorbani
"""GEL-Ped instability-triggered pedestrian-forecast safeguard."""

__version__ = "1.0.0"

from .benchmarks import GELPedRegressor, GoalStableRegressor, TensorResidualRegressor
from .calibration import TensorGeometryModel, VelocitySamples

from .models import (
    AVMParameters,
    AnisotropicParameters,
    GeometricGradientParameters,
    LegacyParameters,
)
from .simulation import (
    SimulationResult,
    simulate_anisotropic,
    simulate_avm_periodic_corridor,
    simulate_corrected,
    simulate_geometric_gradient,
    simulate_legacy_reset,
    simulate_periodic_corridor,
)

__all__ = [
    "__version__",
    "GELPedRegressor",
    "GoalStableRegressor",
    "TensorGeometryModel",
    "TensorResidualRegressor",
    "VelocitySamples",
    "AnisotropicParameters",
    "AVMParameters",
    "LegacyParameters",
    "GeometricGradientParameters",
    "SimulationResult",
    "simulate_anisotropic",
    "simulate_avm_periodic_corridor",
    "simulate_corrected",
    "simulate_geometric_gradient",
    "simulate_legacy_reset",
    "simulate_periodic_corridor",
]
