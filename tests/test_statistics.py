# Author: Amir Ghorbani
import numpy as np

from pedgeom.statistics import exact_paired_randomization_pvalue, exact_sign_test_pvalue


def test_exact_sign_test_for_five_same_direction_effects() -> None:
    differences = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
    assert exact_sign_test_pvalue(differences) == 0.0625


def test_exact_randomization_is_symmetric() -> None:
    differences = np.array([-0.2, -0.1, 0.1, 0.2])
    assert exact_paired_randomization_pvalue(differences) == 1.0
