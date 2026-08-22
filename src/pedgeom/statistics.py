# Author: Amir Ghorbani
"""Small-sample uncertainty tools that preserve complete experimental runs."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def exact_paired_randomization_pvalue(differences: np.ndarray) -> float:
    """Two-sided exact sign-flip test for the mean paired run difference."""

    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    if len(values) > 24:
        raise ValueError("exact enumeration is limited to 24 paired runs")
    observed = abs(float(values.mean()))
    permutations = 1 << len(values)
    extreme = 0
    comparison_tolerance = max(1e-12, observed * 1e-12)
    bit_positions = np.arange(len(values), dtype=np.uint64)
    for start in range(0, permutations, 100_000):
        stop = min(start + 100_000, permutations)
        indices = np.arange(start, stop, dtype=np.uint64)[:, None]
        signs = 2.0 * ((indices >> bit_positions) & 1).astype(float) - 1.0
        permuted = signs @ values / len(values)
        extreme += int(
            np.count_nonzero(np.abs(permuted) >= observed - comparison_tolerance)
        )
    return extreme / permutations


def exact_sign_test_pvalue(differences: np.ndarray) -> float:
    """Two-sided exact sign test after dropping numerical ties."""

    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values) & (np.abs(values) > 1e-15)]
    if not len(values):
        return 1.0
    positives = int(np.count_nonzero(values > 0.0))
    return float(binomtest(positives, len(values), 0.5, alternative="two-sided").pvalue)


def run_bootstrap_interval(
    values: np.ndarray,
    repetitions: int = 20_000,
    seed: int = 20260722,
) -> tuple[float, float]:
    """Percentile interval for a mean with the run as the independent unit."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(repetitions, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high)


def hierarchical_rmse_difference_interval(
    squared_errors_a: dict[str, np.ndarray],
    squared_errors_b: dict[str, np.ndarray],
    repetitions: int = 2_000,
    seed: int = 20260722,
) -> tuple[float, float]:
    """Resample runs and paired observations within runs for mean RMSE(A)-RMSE(B)."""

    runs = sorted(set(squared_errors_a) & set(squared_errors_b))
    if not runs:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for replicate in range(repetitions):
        selected_runs = rng.choice(runs, size=len(runs), replace=True)
        run_differences = []
        for run in selected_runs:
            first = np.asarray(squared_errors_a[run], dtype=float)
            second = np.asarray(squared_errors_b[run], dtype=float)
            if len(first) != len(second):
                raise ValueError(f"paired errors differ in length for run {run}")
            indices = rng.integers(0, len(first), size=len(first))
            run_differences.append(
                np.sqrt(first[indices].mean()) - np.sqrt(second[indices].mean())
            )
        estimates[replicate] = np.mean(run_differences)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)
