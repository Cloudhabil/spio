"""
Brahim's Calculator

Deterministic computation functions for Sovereign PIO.
All outputs are determined solely by inputs - no randomness.

Real domain:
    D(x) = -ln(x) / ln(φ)       x ∈ R, x > 0
    Θ(x) = 2πx
    E(x) = φ^D(x) · Θ(x) = 2π  (energy conservation)

Complex domain (Phase 4):
    D_k(z) = -(ln|z| + i(arg(z) + 2πk)) / ln(φ)    k ∈ Z
    E_k(z) = φ^D_k(z) · Θ(z) = 2π                   (branch-independent)
    |E_k(z)| = 2π for all z ≠ 0, all k               (energy conserved)

Key result — for z = i (Axiom A6):
    D_k(i) = -i(π/2 + 2πk) / ln(φ)
    Re(D_k(i)) = 0 for all k   (purely imaginary)
    Branch spacing: 2π / ln(φ) ≈ 13.0472
"""

import cmath
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


# ---------------------------------------------------------------------------
# Complex Extensions — D_k(z) multi-branch dimension
# ---------------------------------------------------------------------------

def D_complex(z: complex, k: int = 0) -> complex:
    """
    Complex dimension with branch index k.

    D_k(z) = -(ln|z| + i(arg(z) + 2πk)) / ln(φ)

    For z = i, k = 0:
        D_0(i) = -i(π/2) / ln(φ) ≈ -3.2618i

    Properties:
        - Re(D_k(i)) = 0 for all k  (purely imaginary)
        - φ^D_k(z) = 1/z for all k  (branch-independent)
        - |E_k(z)| = 2π for all k   (energy conserved)

    Proof of φ^D_k = 1/z:
        φ^D_k(z) = exp(D_k · ln(φ))
                  = exp(-(ln(z) + 2πik) / ln(φ) · ln(φ))
                  = exp(-(ln(z) + 2πik))
                  = exp(-ln(z)) · exp(-2πik)
                  = (1/z) · 1     [k integer → exp(-2πik) = 1]

    Args:
        z: Complex input (z ≠ 0)
        k: Branch index (integer, default 0 = principal branch)

    Returns:
        Complex dimension value
    """
    if z == 0:
        raise ValueError("D_complex(z) requires z != 0")
    ln_z_k = cmath.log(z) + 2j * pi * k
    return -ln_z_k / log(PHI)


def Theta_complex(z: complex) -> complex:
    """
    Complex phase.

    Θ(z) = 2πz

    Args:
        z: Complex input

    Returns:
        Complex phase value
    """
    return 2 * pi * z


def Energy_complex(z: complex, k: int = 0) -> complex:
    """
    Complex energy with branch index k.

    E_k(z) = φ^D_k(z) · Θ(z)

    This equals 2π for all z ≠ 0, all k ∈ Z.

    Proof:
        φ^D_k(z) = 1/z  (see D_complex docstring)
        E_k(z) = (1/z) · 2πz = 2π  ∎

    In floating point, |E_k(z)| ≈ 2π within machine epsilon.

    Args:
        z: Complex input (z ≠ 0)
        k: Branch index (integer)

    Returns:
        Complex energy value (2π within floating-point tolerance)
    """
    if z == 0:
        raise ValueError("Energy_complex(z) requires z != 0")
    d = D_complex(z, k)
    phi_d = cmath.exp(d * log(PHI))
    return phi_d * Theta_complex(z)


def branch_spectrum(z: complex, k_max: int = 6) -> list[dict]:
    """
    Compute D_k(z) for k ∈ [-k_max, k_max].

    Returns the full branch spectrum proving |E_k(z)| = 2π for all k.

    For z = i (Axiom A6):
        D_k(i) = -i(π/2 + 2πk) / ln(φ)
        All branches purely imaginary (Re = 0).
        Branch spacing = 2π / ln(φ) ≈ 13.0472 (BRANCH_SPACING constant).

    Args:
        z: Complex input (z ≠ 0)
        k_max: Maximum |k| (default 6, yields 13 branches)

    Returns:
        List of dicts: {k, dimension, real, imag, energy_magnitude}
    """
    results = []
    for k in range(-k_max, k_max + 1):
        d = D_complex(z, k)
        e = Energy_complex(z, k)
        results.append({
            "k": k,
            "dimension": d,
            "real": d.real,
            "imag": d.imag,
            "energy_magnitude": abs(e),
        })
    return results


def spectral_dimensions(z: complex, k_max: int = 6) -> list[int]:
    """
    Map complex branches to real silicon dimensions (1-12).

    For each branch k, compute:
        spectral_dim = (floor(|Im(D_k(z))|) mod 12) + 1

    This maps the countably infinite branch spectrum onto
    the 12 physical dimensions, cycling through all silicon
    targets as k increases.

    Args:
        z: Complex input (z ≠ 0)
        k_max: Maximum |k| (default 6)

    Returns:
        List of silicon dimensions (1-12) for each branch
    """
    dims = []
    for k in range(-k_max, k_max + 1):
        d = D_complex(z, k)
        dim = int(abs(d.imag)) % 12 + 1
        dims.append(dim)
    return dims
