# Author: Amir Ghorbani
"""Small explicit integrators used to expose timestep effects."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Derivative = Callable[[FloatArray], FloatArray]


def rk4_step(state: FloatArray, step: float, derivative: Derivative) -> FloatArray:
    k1 = derivative(state)
    k2 = derivative(state + 0.5 * step * k1)
    k3 = derivative(state + 0.5 * step * k2)
    k4 = derivative(state + step * k3)
    return state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
