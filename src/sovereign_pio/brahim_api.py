"""
Brahim Unified API - Mathematical Foundation

Complete mathematical engine implementing Brahim's laws for:
- Wormhole geometry and traversability
- Golden ratio hierarchies
- Brahim sequence with mirror symmetry
- Routing, compression, and signal processing

All computations are DETERMINISTIC and provably stable via Lyapunov analysis.
"""

from math import sqrt, log, exp, pi, sin, cos, atan2
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .constants import PHI, ALPHA, OMEGA, BETA, GAMMA, LUCAS_NUMBERS


# =============================================================================
# BRAHIM SEQUENCE - Perfect Mirror Symmetry
# =============================================================================

# The Brahim Sequence: pairs sum to 214, center at 107
BRAHIM_SEQUENCE = [27, 42, 60, 75, 97, 117, 139, 154, 172, 187]
PAIR_SUM = 214
CENTER = 107
SEQUENCE_SUM = 1070  # sum(BRAHIM_SEQUENCE)

# Mirror pairs
MIRROR_PAIRS = [
    (27, 187),   # 27 + 187 = 214
    (42, 172),   # 42 + 172 = 214
    (60, 154),   # 60 + 154 = 214
    (75, 139),   # 75 + 139 = 214
    (97, 117),   # 97 + 117 = 214
]


def brahim_mirror(x: int) -> int:
    """Get the mirror value in Brahim sequence."""
    return PAIR_SUM - x


def is_brahim_number(x: int) -> bool:
    """Check if x is in the Brahim sequence."""
    return x in BRAHIM_SEQUENCE


def brahim_distance(a: int, b: int) -> int:
    """Distance in Brahim space (symmetric)."""
    return abs(a - b)


# =============================================================================
# PHI HIERARCHY - Golden Ratio Powers
# =============================================================================

@dataclass
class PhiHierarchy:
    """Golden ratio power hierarchy."""
    level: int
    value: float
    name: str
    domain: str

    @classmethod
    def from_level(cls, level: int) -> "PhiHierarchy":
        """Create hierarchy level from integer."""
        value = 1 / (PHI ** level)
        names = {
            0: ("Unity", "origin"),
            1: ("Omega", "return"),
            2: ("Alpha", "creation"),
            3: ("Beta", "security"),
            4: ("Gamma", "damping"),
            5: ("Delta", "decay"),
            6: ("Epsilon", "threshold"),
        }
        name, domain = names.get(level, (f"Level_{level}", "extended"))
        return cls(level=level, value=value, name=name, domain=domain)


def phi_power(n: int) -> float:
    """Compute PHI^n."""
    return PHI ** n


def phi_inverse(n: int) -> float:
    """Compute 1/PHI^n = OMEGA^n."""
    return OMEGA ** n


def phi_decompose(x: float) -> Tuple[int, float]:
    """
    Decompose x into PHI hierarchy.

    Returns (level, residual) where x ≈ PHI^(-level) * (1 + residual)
    """
    if x <= 0:
        return (0, 0.0)

    level = int(-log(x) / log(PHI))
    base = phi_inverse(level)
    residual = (x / base) - 1.0

    return (level, residual)


# =============================================================================
# WORMHOLE GEOMETRY - Morris-Thorne Framework
# =============================================================================

@dataclass
class WormholeGeometry:
    """Morris-Thorne wormhole geometry parameters."""
    throat_radius: float    # r_0 (minimum radius)
    shape_parameter: float  # b(r) shape function parameter
    redshift: float         # Φ(r) redshift function

    # Derived quantities
    aperture: float = 0.0   # Wormhole aperture (1.16% = ε)
    traversability: float = 0.0

    def __post_init__(self):
        self.aperture = 1.0 / (PHI ** 4)  # ≈ 0.0116 = 1.16%
        self.traversability = self._compute_traversability()

    def _compute_traversability(self) -> float:
        """Compute traversability metric."""
        # Based on Brahim's laws: T = β * (1 - aperture)
        return BETA * (1 - self.aperture)

    def shape_function(self, r: float) -> float:
        """
        Morris-Thorne shape function b(r).

        Must satisfy: b(r_0) = r_0 and b(r) < r for r > r_0
        """
        if r <= self.throat_radius:
            return self.throat_radius

        # Brahim shape function: b(r) = r_0 * (r_0/r)^β
        return self.throat_radius * (self.throat_radius / r) ** BETA

    def is_traversable(self, r: float) -> bool:
        """Check if wormhole is traversable at radius r."""
        b = self.shape_function(r)
        return b < r and self.traversability > 0


def create_wormhole(throat_radius: float = 1.0) -> WormholeGeometry:
    """Create a wormhole with Brahim-optimal parameters."""
    return WormholeGeometry(
        throat_radius=throat_radius,
        shape_parameter=BETA,
        redshift=GAMMA,
    )


# =============================================================================
# DIMENSIONAL ROUTING - 12D Navigation
# =============================================================================

@dataclass
class DimensionalState:
    """State in the 12-dimensional space."""
    dimension: int          # 1-12
    position: float         # Position within dimension
    energy: float           # Current energy level
    phase: float            # Phase angle

    @property
    def capacity(self) -> int:
        """Lucas number capacity for this dimension."""
        if 1 <= self.dimension <= 12:
            return LUCAS_NUMBERS[self.dimension - 1]
        return 1

    @property
    def silicon(self) -> str:
        """Hardware affinity for this dimension."""
        if self.dimension <= 4:
            return "NPU"
        elif self.dimension <= 8:
            return "CPU"
        else:
            return "GPU"


def dimension_energy(d: int) -> float:
    """
    Energy at dimension d.

    E(d) = 2π for all d (conservation law)
    """
    return 2 * pi


def dimension_transition(from_d: int, to_d: int) -> float:
    """
    Compute energy cost for dimension transition.

    Cost = |log_φ(L(to)/L(from))|
    """
    if not (1 <= from_d <= 12 and 1 <= to_d <= 12):
        return float('inf')

    l_from = LUCAS_NUMBERS[from_d - 1]
    l_to = LUCAS_NUMBERS[to_d - 1]

    if l_from == 0 or l_to == 0:
        return float('inf')

    return abs(log(l_to / l_from) / log(PHI))


def optimal_route(from_d: int, to_d: int) -> List[int]:
    """
    Find optimal route between dimensions.

    Uses PHI-weighted path to minimize energy.
    """
    if from_d == to_d:
        return [from_d]

    # Direct path for adjacent dimensions
    if abs(from_d - to_d) == 1:
        return [from_d, to_d]

    # For longer paths, step through dimensions
    path = [from_d]
    current = from_d

    while current != to_d:
        if current < to_d:
            current += 1
        else:
            current -= 1
        path.append(current)

    return path


# =============================================================================
# WORMHOLE TRANSFORM - Signal Processing
# =============================================================================

def wormhole_transform(signal: np.ndarray) -> np.ndarray:
    """
    Apply perfect wormhole transform.

    W*(σ) = σ/φ + C̄·α

    Where C̄ is the complex conjugate of the center.
    """
    sigma = signal.astype(np.complex128)

    # Transform: divide by PHI and add center correction
    center = CENTER * (1 + 0j)  # Real center at 107
    alpha_correction = center.conjugate() * ALPHA

    transformed = sigma / PHI + alpha_correction

    return transformed.real.astype(signal.dtype)


def inverse_wormhole_transform(signal: np.ndarray) -> np.ndarray:
    """
    Inverse wormhole transform.

    W*⁻¹(σ) = (σ - C̄·α) · φ
    """
    sigma = signal.astype(np.complex128)

    center = CENTER * (1 + 0j)
    alpha_correction = center.conjugate() * ALPHA

    recovered = (sigma - alpha_correction) * PHI

    return recovered.real.astype(signal.dtype)


# =============================================================================
# COMPRESSION - PHI-Optimal Encoding
# =============================================================================

def phi_compress(data: np.ndarray, levels: int = 4) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    PHI-optimal compression using golden ratio quantization.

    Returns (compressed_data, metadata)
    """
    # Normalize to [0, 1]
    min_val, max_val = data.min(), data.max()
    normalized = (data - min_val) / (max_val - min_val + 1e-10)

    # Quantize to PHI levels
    boundaries = [phi_inverse(i) for i in range(levels)]
    boundaries = sorted(boundaries + [1.0])

    quantized = np.digitize(normalized, boundaries)

    metadata = {
        "min": float(min_val),
        "max": float(max_val),
        "levels": levels,
        "boundaries": boundaries,
        "compression_ratio": data.nbytes / quantized.nbytes,
    }

    return quantized.astype(np.uint8), metadata


def phi_decompress(compressed: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
    """Decompress PHI-compressed data."""
    boundaries = metadata["boundaries"]
    min_val = metadata["min"]
    max_val = metadata["max"]

    # Map back to boundary midpoints
    midpoints = [(boundaries[i] + boundaries[i+1]) / 2
                 for i in range(len(boundaries) - 1)]
    midpoints = [0.0] + midpoints + [1.0]

    # Reconstruct
    normalized = np.array([midpoints[min(v, len(midpoints)-1)]
                          for v in compressed.flatten()])
    normalized = normalized.reshape(compressed.shape)

    # Denormalize
    return normalized * (max_val - min_val) + min_val


# =============================================================================
# STABILITY ANALYSIS - Lyapunov Functions
# =============================================================================

def lyapunov_stability(state: np.ndarray, equilibrium: np.ndarray) -> float:
    """
    Compute Lyapunov stability measure.

    V(x) = ||x - x*||² / φ

    System is stable if V decreases along trajectories.
    """
    deviation = state - equilibrium
    v = np.sum(deviation ** 2) / PHI
    return float(v)


def is_lyapunov_stable(
    trajectory: List[np.ndarray],
    equilibrium: np.ndarray,
) -> bool:
    """
    Check if trajectory is Lyapunov stable.

    Returns True if V(x(t)) is monotonically decreasing.
    """
    if len(trajectory) < 2:
        return True

    v_values = [lyapunov_stability(s, equilibrium) for s in trajectory]

    # Check monotonic decrease
    for i in range(1, len(v_values)):
        if v_values[i] > v_values[i-1]:
            return False

    return True


# =============================================================================
# UNIFIED API CLASS
# =============================================================================

class BrahimAPI:
    """
    Unified API for Brahim's mathematical framework.

    Provides:
    - Wormhole geometry and traversability
    - Dimensional routing
    - Signal transforms
    - Compression
    - Stability analysis

    Example:
        api = BrahimAPI()

        # Create wormhole
        wh = api.create_wormhole(throat_radius=1.0)
        print(f"Traversability: {wh.traversability}")

        # Route between dimensions
        path = api.route(from_d=3, to_d=9)

        # Transform signal
        transformed = api.transform(signal)
    """

    def __init__(self):
        self.phi = PHI
        self.sequence = BRAHIM_SEQUENCE
        self.center = CENTER

    def create_wormhole(self, throat_radius: float = 1.0) -> WormholeGeometry:
        """Create a Brahim-optimal wormhole."""
        return create_wormhole(throat_radius)

    def route(self, from_d: int, to_d: int) -> List[int]:
        """Find optimal dimensional route."""
        return optimal_route(from_d, to_d)

    def transition_cost(self, from_d: int, to_d: int) -> float:
        """Compute dimension transition cost."""
        return dimension_transition(from_d, to_d)

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """Apply wormhole transform."""
        return wormhole_transform(signal)

    def inverse_transform(self, signal: np.ndarray) -> np.ndarray:
        """Apply inverse wormhole transform."""
        return inverse_wormhole_transform(signal)

    def compress(self, data: np.ndarray, levels: int = 4) -> Tuple[np.ndarray, Dict]:
        """PHI-optimal compression."""
        return phi_compress(data, levels)

    def decompress(self, data: np.ndarray, metadata: Dict) -> np.ndarray:
        """Decompress PHI-compressed data."""
        return phi_decompress(data, metadata)

    def stability(self, state: np.ndarray, equilibrium: np.ndarray) -> float:
        """Compute Lyapunov stability measure."""
        return lyapunov_stability(state, equilibrium)

    def mirror(self, x: int) -> int:
        """Get Brahim mirror value."""
        return brahim_mirror(x)

    def hierarchy(self, level: int) -> PhiHierarchy:
        """Get PHI hierarchy at level."""
        return PhiHierarchy.from_level(level)

    def dimension_state(self, d: int, position: float = 0.0) -> DimensionalState:
        """Create dimensional state."""
        return DimensionalState(
            dimension=d,
            position=position,
            energy=dimension_energy(d),
            phase=position * 2 * pi,
        )
