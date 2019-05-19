"""Confidence interval helper."""

import math
import scipy.stats

from typing import Tuple


Z = scipy.stats.norm.ppf(0.05 / 2.0)
ZZ = Z * Z


def wilson_score(accuracy: float, total: int) -> Tuple[float, float]:
    """Computes Wilson score 95% confidence intervals.

    Args:
        accuracy: task accuracy.
        total: number of observations.

    Returns:
        A (lower bound, upper bound) tuple.
    """
    a1 = 1.0 / (1.0 + ZZ / total)
    a2 = accuracy + ZZ / (2 * total)
    a3 = Z * math.sqrt(
        accuracy * (1.0 - accuracy) / total + ZZ / (4 * total * total)
    )
    lower = a1 * (a2 + a3)
    upper = a1 * (a2 - a3)
    return (lower, upper)
