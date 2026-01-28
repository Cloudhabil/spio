"""
Brahim's Calculator

Deterministic computation functions for Sovereign PIO.
All outputs are determined solely by inputs - no randomness.
"""

from math import log, pi
from .constants import PHI


def D(x: float) -> float:
    """
    Dimension from value. DETERMINISTIC.

    Maps a value to its corresponding dimension in PHI-space.

    Args:
        x: Input value (must be > 0)

    Returns:
        Dimension value

    Example:
        >>> D(PHI)
        -1.0
        >>> D(1/PHI)
        1.0
    """
    if x <= 0:
        raise ValueError("D(x) requires x > 0")
    return -log(x) / log(PHI)


def Theta(x: float) -> float:
    """
    Phase from value. DETERMINISTIC.

    Converts a value to its phase angle.

    Args:
        x: Input value

    Returns:
        Phase angle in radians
    """
    return 2 * pi * x


def Energy(x: float) -> float:
    """
    Energy is ALWAYS 2*PI. PROVEN.

    The energy function demonstrates conservation:
    E(x) = PHI^D(x) * Theta(x) = 2*PI for all valid x

    Args:
        x: Input value (must be > 0)

    Returns:
        Energy value (always 2*PI)
    """
    if x <= 0:
        raise ValueError("Energy(x) requires x > 0")
    return (PHI ** D(x)) * Theta(x)


def x_from_D(d: float) -> float:
    """
    Value from dimension. INVERSE of D(x).

    Recovers the original value from a dimension.

    Args:
        d: Dimension value

    Returns:
        Original value x such that D(x) = d

    Example:
        >>> x_from_D(1.0)
        0.6180339887498949  # 1/PHI
    """
    return 1 / (PHI ** d)


def lucas(n: int) -> int:
    """
    Lucas number L(n). DETERMINISTIC.

    Lucas numbers: 2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322...
    (Using 1-indexed dimensions, L(1)=1, L(2)=3, ...)

    Args:
        n: Index (1-based for dimensions)

    Returns:
        Lucas number at index n
    """
    if n < 1:
        raise ValueError("lucas(n) requires n >= 1")

    # Pre-computed for dimensions 1-12
    lucas_seq = [1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322]

    if n <= len(lucas_seq):
        return lucas_seq[n - 1]

    # Compute dynamically for larger n
    a, b = 2, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def npu_bandwidth(n_parallel: int) -> float:
    """
    NPU bandwidth follows PHI saturation. MEASURED AND PROVEN.

    BW(N) = 7.20 * (1 - e^(-N/PHI))

    Args:
        n_parallel: Number of parallel operations

    Returns:
        Bandwidth in GB/s
    """
    from math import exp
    BW_MAX = 7.20
    return BW_MAX * (1 - exp(-n_parallel / PHI))


def optimal_parallelism() -> int:
    """
    Returns the optimal parallelism for NPU operations.

    Based on measured saturation point k = PHI, optimal N = 16.

    Returns:
        Optimal number of parallel operations
    """
    return 16
