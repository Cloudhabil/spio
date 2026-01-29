"""
Wormhole Core - Mathematical Bridge Architecture

Ported from: CLI-main/src/core/brahim_wormhole_engine.py
             CLI-main/src/gpia/wormhole/wormhole_router.py

This is the MATHEMATICAL UNIFICATION LANGUAGE - not a DSL but a
transformation framework that bridges SPIO with hardware routing.

Key Features:
- Morris-Thorne traversable wormhole geometry
- Golden ratio stability (Lyapunov analysis)
- Wormhole transform for O(1) compression
- Error detection via mirror symmetry
- Packet routing with wormhole shortcuts
"""

import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# FUNDAMENTAL CONSTANTS (Brahim's Calculator)
# =============================================================================

PHI: float = (1 + math.sqrt(5)) / 2          # 1.618033988749895
PHI_INV: float = 1 / PHI                      # 0.618033988749895
ALPHA: float = 1 / PHI**2                     # 0.381966011250105
BETA: float = 1 / PHI**3                      # 0.236067977499790
GAMMA: float = 1 / PHI**4                     # 0.145898033750315

# Brahim Sequence (Corrected - full mirror symmetry)
BRAHIM_SEQUENCE: Tuple[int, ...] = (27, 42, 60, 75, 97, 117, 139, 154, 172, 187)

# Sequence constants
PAIR_SUM: int = 214           # Each mirror pair sums to this
CENTER: int = 107             # C = S/2 (critical line)
DIMENSION: int = 10           # Sequence length

# Derived constants
CENTROID: List[float] = [b / PAIR_SUM for b in BRAHIM_SEQUENCE]
EQUILIBRIUM_RADIUS: float = (CENTER / PAIR_SUM) * PHI  # ~0.809


# =============================================================================
# ENUMS
# =============================================================================

class WormholeState(Enum):
    """State of the wormhole."""
    STABLE = "stable"
    EVOLVING = "evolving"
    COLLAPSED = "collapsed"
    TRAVERSABLE = "traversable"


class RouteType(Enum):
    """Type of routing path."""
    DIRECT = "direct"           # Single wormhole hop
    BRIDGE = "bridge"           # Cross-region wormhole
    CASCADE = "cascade"         # Multiple wormhole hops
    STANDARD = "standard"       # Non-wormhole path
    BLOCKED = "blocked"         # Safety blocked route


class SafetyLevel(Enum):
    """ASIOS safety classification."""
    SAFE = "safe"
    NOMINAL = "nominal"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    BLOCKED = "blocked"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WormholeGeometry:
    """Geometric properties of the wormhole."""
    throat_radius: float
    shape_at_throat: float
    flare_out: float
    asymptotic_flatness: float
    is_valid: bool

    def __str__(self) -> str:
        return (f"WormholeGeometry(r0={self.throat_radius:.4f}, "
                f"b(r0)={self.shape_at_throat:.4f}, "
                f"b'(r0)={self.flare_out:.4f}, valid={self.is_valid})")


@dataclass
class TraversabilityResult:
    """Result of traversability analysis."""
    nec_violated: bool
    nec_factor: float
    exotic_matter_required: bool
    is_traversable: bool
    stress_energy: Dict[str, float] = field(default_factory=dict)


@dataclass
class StabilityResult:
    """Result of Lyapunov stability analysis."""
    eigenvalues: List[float]
    lyapunov_exponents: List[float]
    spectral_abscissa: float
    is_stable: bool
    stability_class: str


@dataclass
class WormholeTransformResult:
    """Result of wormhole transform operation."""
    input_vector: List[float]
    output_vector: List[float]
    compression_ratio: float
    distance_to_centroid: float
    iterations: int = 1


@dataclass
class ErrorCheckResult:
    """Result of error detection via mirror symmetry."""
    is_valid: bool
    corrupted_pairs: List[Tuple[int, int, int]]  # (index, expected, actual)
    error_magnitude: int
    recoverable: bool


@dataclass
class RoutingResult:
    """Result of a wormhole routing operation."""
    query: str
    packets: List[Any]
    route_type: RouteType
    hops: int
    latency_ms: float
    safety_level: SafetyLevel
    wormholes_used: int
    path: List[str]
    similarity_scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return len(self.packets) > 0 and self.safety_level != SafetyLevel.BLOCKED

    @property
    def compression_ratio(self) -> float:
        if self.hops == 0:
            return 1.0
        theoretical_hops = self.hops + self.wormholes_used * 3
        return self.hops / theoretical_hops


# =============================================================================
# BRAHIM WORMHOLE ENGINE
# =============================================================================

class BrahimWormholeEngine:
    """
    Core engine for Brahim Wormhole computations.

    The engine provides:
    1. Geometry calculations (shape function, flare-out)
    2. Traversability analysis (NEC violation)
    3. Stability analysis (Lyapunov)
    4. Transform operations (routing, compression)
    5. Error detection (mirror symmetry)
    6. Evolution simulation

    Example:
        engine = BrahimWormholeEngine()
        geom = engine.analyze_geometry(r0=1.0)
        trav = engine.check_traversability(r0=1.0)
        result = engine.transform([1, 2, 3, 4, 5])
    """

    def __init__(self, throat_radius: float = 1.0):
        self.r0 = throat_radius
        self.state = WormholeState.STABLE
        self.creation_time = time.time()
        self.evolution_history: List[Dict] = []
        self._lock = threading.Lock()

        # Verify constants on initialization
        self._verify_constants()

    def _verify_constants(self) -> None:
        """Verify fundamental constant relationships."""
        # Check alpha + beta = 1/phi
        identity_error = abs(ALPHA + BETA - PHI_INV)
        if identity_error > 1e-14:
            raise ValueError("Constant identity violated: alpha + beta != 1/phi")

        # Check sequence closure
        for i in range(DIMENSION // 2):
            pair_sum = BRAHIM_SEQUENCE[i] + BRAHIM_SEQUENCE[DIMENSION - 1 - i]
            if pair_sum != PAIR_SUM:
                raise ValueError(f"Sequence closure violated at pair {i}")

    # =========================================================================
    # GEOMETRY
    # =========================================================================

    def shape_function(self, r: float, r0: Optional[float] = None) -> float:
        """
        Compute the shape function b(r).

        b(r) = r0 * (r0/r)^alpha * exp(-beta * (r - r0) / r0)
        """
        if r0 is None:
            r0 = self.r0

        if r <= 0:
            raise ValueError("Radial coordinate must be positive")

        return r0 * (r0 / r)**ALPHA * math.exp(-BETA * (r - r0) / r0)

    def shape_derivative(self, r: float, r0: Optional[float] = None, h: float = 1e-8) -> float:
        """
        Compute the derivative of the shape function b'(r).

        At the throat: b'(r0) = -(alpha + beta) = -1/phi
        """
        if r0 is None:
            r0 = self.r0

        # Central difference
        return (self.shape_function(r + h, r0) - self.shape_function(r - h, r0)) / (2 * h)

    def analyze_geometry(self, r0: Optional[float] = None) -> WormholeGeometry:
        """
        Analyze the wormhole geometry.

        Checks:
        1. Throat condition: b(r0) = r0
        2. Flare-out condition: b'(r0) < 1
        3. Asymptotic flatness: b(r)/r -> 0 as r -> infinity
        """
        if r0 is None:
            r0 = self.r0

        # Compute at throat
        b_throat = self.shape_function(r0, r0)
        b_prime = self.shape_derivative(r0, r0)

        # Asymptotic behavior
        r_far = 100 * r0
        b_far = self.shape_function(r_far, r0)
        asymptotic = b_far / r_far

        # Validity checks
        throat_ok = abs(b_throat - r0) < 1e-10
        flare_ok = b_prime < 1
        asymptotic_ok = asymptotic < 0.01

        return WormholeGeometry(
            throat_radius=r0,
            shape_at_throat=b_throat,
            flare_out=b_prime,
            asymptotic_flatness=asymptotic,
            is_valid=throat_ok and flare_ok and asymptotic_ok
        )

    # =========================================================================
    # TRAVERSABILITY
    # =========================================================================

    def nec_factor(self, r: float, r0: Optional[float] = None) -> float:
        """
        Compute the Null Energy Condition (NEC) factor.

        NEC factor = (b/r - b')
        At throat: NEC = phi (golden ratio)
        """
        if r0 is None:
            r0 = self.r0

        b = self.shape_function(r, r0)
        b_prime = self.shape_derivative(r, r0)

        return b / r - b_prime

    def check_traversability(self, r0: Optional[float] = None) -> TraversabilityResult:
        """
        Check if the wormhole is traversable.

        Traversability requires NEC violation near throat (exotic matter).
        """
        if r0 is None:
            r0 = self.r0

        # Check NEC at throat
        nec_throat = self.nec_factor(r0 * 1.001, r0)

        # NEC profile
        stress_energy = {}
        for r_mult in [1.01, 1.1, 1.5, 2.0, 3.0]:
            r = r0 * r_mult
            stress_energy[f"r={r_mult}r0"] = self.nec_factor(r, r0)

        # Traversability determination
        nec_violated = nec_throat > 0

        return TraversabilityResult(
            nec_violated=nec_violated,
            nec_factor=nec_throat,
            exotic_matter_required=nec_violated,
            is_traversable=nec_violated,
            stress_energy=stress_energy
        )

    # =========================================================================
    # STABILITY
    # =========================================================================

    def analyze_stability(self) -> StabilityResult:
        """
        Perform Lyapunov stability analysis.

        The stability matrix has golden ratio structure:
        J = [[-alpha,  beta ],
             [ beta,  -alpha]]

        Eigenvalues: lambda = -alpha +/- beta
                   = {-gamma, -1/phi}

        Both negative -> stable
        """
        # Compute eigenvalues analytically
        eigenvalues = [
            -ALPHA + BETA,  # -gamma
            -ALPHA - BETA,  # -1/phi
        ]

        real_parts = eigenvalues
        lyapunov = sorted(real_parts, reverse=True)
        spectral_abscissa = max(real_parts)

        # Stability classification
        if spectral_abscissa < -1e-10:
            stability_class = "asymptotically_stable"
            is_stable = True
        elif abs(spectral_abscissa) < 1e-10:
            stability_class = "marginally_stable"
            is_stable = True
        else:
            stability_class = "unstable"
            is_stable = False

        return StabilityResult(
            eigenvalues=eigenvalues,
            lyapunov_exponents=lyapunov,
            spectral_abscissa=spectral_abscissa,
            is_stable=is_stable,
            stability_class=stability_class
        )

    # =========================================================================
    # WORMHOLE TRANSFORM
    # =========================================================================

    def transform(self, x: List[float], iterations: int = 1) -> WormholeTransformResult:
        """
        Apply the wormhole transform.

        W(x) = x/phi + C_bar * alpha

        Properties:
        - Compression ratio: 1/phi per iteration
        - Fixed point: centroid C_bar
        - Invertible
        """
        x = list(x)

        # Resize to 10 dimensions
        if len(x) < DIMENSION:
            x = x + [0.0] * (DIMENSION - len(x))
        elif len(x) > DIMENSION:
            x = x[:DIMENSION]

        original_x = x.copy()

        # Apply transform iteratively
        for _ in range(iterations):
            x = [xi / PHI + CENTROID[i] * ALPHA for i, xi in enumerate(x)]

        # Compute metrics
        dist_before = math.sqrt(sum((a - b)**2 for a, b in zip(original_x, CENTROID)))
        dist_after = math.sqrt(sum((a - b)**2 for a, b in zip(x, CENTROID)))
        compression = dist_after / dist_before if dist_before > 0 else 0

        return WormholeTransformResult(
            input_vector=original_x,
            output_vector=x,
            compression_ratio=compression,
            distance_to_centroid=dist_after,
            iterations=iterations
        )

    def inverse_transform(self, w: List[float]) -> List[float]:
        """
        Apply the inverse wormhole transform.

        W^-1(w) = (w - C_bar * alpha) * phi
        """
        w = list(w)
        return [(wi - CENTROID[i] * ALPHA) * PHI for i, wi in enumerate(w)]

    def route(self, source: List[float], max_hops: int = 10,
              convergence_threshold: float = 0.01) -> List[List[float]]:
        """
        Route a packet using wormhole transform.

        Each hop compresses by 1/phi until convergence to centroid.
        """
        path = [list(source)]

        # Resize
        if len(path[0]) < DIMENSION:
            path[0] = path[0] + [0.0] * (DIMENSION - len(path[0]))

        for _ in range(max_hops):
            current = path[-1]
            next_pos = [ci / PHI + CENTROID[i] * ALPHA for i, ci in enumerate(current)]
            path.append(next_pos)

            # Check convergence
            dist = math.sqrt(sum((a - b)**2 for a, b in zip(next_pos, CENTROID)))
            if dist < convergence_threshold:
                break

        return path

    # =========================================================================
    # COMPRESSION
    # =========================================================================

    def compress(self, data: List[float], levels: int = 5) -> Dict[str, Any]:
        """
        Hierarchical compression using golden ratio.

        Each level compresses by 1/phi (0.618).
        """
        result = {
            "original_size": len(data),
            "levels": []
        }

        current = list(data)

        for level in range(levels):
            # Downsample by golden ratio
            new_size = max(1, int(len(current) * PHI_INV))

            # Averaging compression
            if new_size < len(current):
                ratio = len(current) / new_size
                compressed = []
                for i in range(new_size):
                    start = int(i * ratio)
                    end = int((i + 1) * ratio)
                    compressed.append(sum(current[start:end]) / max(end - start, 1))
            else:
                compressed = current

            result["levels"].append({
                "level": level,
                "size": len(compressed),
                "compression_ratio": len(compressed) / result["original_size"],
                "data": compressed
            })

            current = compressed

        result["final_size"] = len(current)
        result["total_compression"] = result["final_size"] / result["original_size"]

        return result

    # =========================================================================
    # ERROR DETECTION
    # =========================================================================

    def detect_errors(self, sequence: List[int]) -> ErrorCheckResult:
        """
        Detect errors using mirror symmetry.

        For a valid Brahim sequence, all mirror pairs sum to 214.
        """
        if len(sequence) != DIMENSION:
            return ErrorCheckResult(
                is_valid=False,
                corrupted_pairs=[(-1, DIMENSION, len(sequence))],
                error_magnitude=abs(len(sequence) - DIMENSION),
                recoverable=False
            )

        corrupted = []
        total_error = 0

        for i in range(DIMENSION // 2):
            j = DIMENSION - 1 - i
            actual_sum = sequence[i] + sequence[j]

            if actual_sum != PAIR_SUM:
                error = actual_sum - PAIR_SUM
                corrupted.append((i, PAIR_SUM, actual_sum))
                total_error += abs(error)

        return ErrorCheckResult(
            is_valid=len(corrupted) == 0,
            corrupted_pairs=corrupted,
            error_magnitude=total_error,
            recoverable=len(corrupted) == 1
        )

    def correct_error(self, sequence: List[int], pair_index: int,
                      correct_first: bool = True) -> List[int]:
        """
        Correct a single error using mirror symmetry.
        """
        corrected = list(sequence)
        i = pair_index
        j = DIMENSION - 1 - pair_index

        if correct_first:
            corrected[j] = PAIR_SUM - corrected[i]
        else:
            corrected[i] = PAIR_SUM - corrected[j]

        return corrected

    # =========================================================================
    # EVOLUTION
    # =========================================================================

    def evolve(self, time_steps: int = 100, dt: float = 0.1) -> List[Dict]:
        """
        Simulate wormhole evolution over time.

        The throat radius evolves according to:
        dr/dt = -beta * (r - r_eq)
        """
        r = self.r0
        history = []

        for t in range(time_steps):
            state = {
                "time": t * dt,
                "throat_radius": r,
                "distance_to_equilibrium": abs(r - EQUILIBRIUM_RADIUS),
                "is_stable": r > 0.1,
                "nec_factor": self.nec_factor(max(r * 1.01, 0.11), r) if r > 0.1 else 0
            }
            history.append(state)

            # Evolve
            dr = -BETA * (r - EQUILIBRIUM_RADIUS) * dt
            r = max(0.1, r + dr)

        self.evolution_history = history
        return history

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate(self) -> Dict[str, bool]:
        """Run full validation suite."""
        geom = self.analyze_geometry()
        trav = self.check_traversability()
        stab = self.analyze_stability()
        err_check = self.detect_errors(list(BRAHIM_SEQUENCE))

        return {
            "geometry_valid": geom.is_valid,
            "traversable": trav.is_traversable,
            "stable": stab.is_stable,
            "sequence_valid": err_check.is_valid,
            "identity_alpha_plus_beta": abs(ALPHA + BETA - PHI_INV) < 1e-14,
            "all_valid": (geom.is_valid and trav.is_traversable and
                         stab.is_stable and err_check.is_valid),
        }

    def get_constants(self) -> Dict[str, float]:
        """Return all fundamental constants."""
        return {
            "phi": PHI,
            "phi_inv": PHI_INV,
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "pair_sum": PAIR_SUM,
            "center": CENTER,
            "dimension": DIMENSION,
            "equilibrium_radius": EQUILIBRIUM_RADIUS,
        }


# =============================================================================
# WORMHOLE TRANSFORM (Standalone)
# =============================================================================

class WormholeTransform:
    """Standalone wormhole transform operations."""

    @staticmethod
    def forward(x: List[float], iterations: int = 1) -> List[float]:
        """Apply forward wormhole transform."""
        x = list(x)
        if len(x) < DIMENSION:
            x = x + [0.0] * (DIMENSION - len(x))
        elif len(x) > DIMENSION:
            x = x[:DIMENSION]

        for _ in range(iterations):
            x = [xi / PHI + CENTROID[i] * ALPHA for i, xi in enumerate(x)]

        return x

    @staticmethod
    def inverse(w: List[float]) -> List[float]:
        """Apply inverse wormhole transform."""
        return [(wi - CENTROID[i] * ALPHA) * PHI for i, wi in enumerate(w)]

    @staticmethod
    def compression_ratio(iterations: int) -> float:
        """Get compression ratio after N iterations."""
        return PHI_INV ** iterations


# =============================================================================
# WORMHOLE ROUTER
# =============================================================================

class WormholeRouter:
    """
    Routes queries through wormhole shortcuts.

    Features:
    - O(1) routing for cached queries
    - BETA-compressed distances via wormholes
    - Safety integration
    """

    def __init__(self, safety_checker: Optional[Callable[[str], SafetyLevel]] = None):
        self.safety_checker = safety_checker or self._default_safety_check
        self._cache: Dict[str, Tuple[RoutingResult, float]] = {}
        self._wormholes: Dict[str, List[str]] = {}
        self._packets: Dict[str, Any] = {}
        self._lock = threading.Lock()

        # Metrics
        self.total_queries = 0
        self.wormhole_hits = 0
        self.cache_hits = 0

    def _default_safety_check(self, query: str) -> SafetyLevel:
        """Default safety check."""
        unsafe_patterns = ["hack", "exploit", "bypass", "inject"]
        query_lower = query.lower()

        for pattern in unsafe_patterns:
            if pattern in query_lower:
                return SafetyLevel.CAUTION

        return SafetyLevel.SAFE

    def add_packet(self, packet_id: str, data: Any, embedding: List[float] = None) -> None:
        """Add a packet to the router."""
        with self._lock:
            self._packets[packet_id] = {
                "id": packet_id,
                "data": data,
                "embedding": embedding or [],
            }

    def create_wormhole(self, source_id: str, target_id: str) -> bool:
        """Create a wormhole connection between packets."""
        with self._lock:
            if source_id not in self._packets or target_id not in self._packets:
                return False

            if source_id not in self._wormholes:
                self._wormholes[source_id] = []
            if target_id not in self._wormholes[source_id]:
                self._wormholes[source_id].append(target_id)

            return True

    def route(self, query: str, k: int = 5, use_cache: bool = True) -> RoutingResult:
        """Route a query through the packet network."""
        start_time = time.time()
        self.total_queries += 1

        # Check cache
        cache_key = f"{query}:{k}"
        if use_cache and cache_key in self._cache:
            cached_result, cached_time = self._cache[cache_key]
            if time.time() - cached_time < 60:
                self.cache_hits += 1
                return cached_result

        # Safety check
        safety_level = self.safety_checker(query)
        if safety_level == SafetyLevel.BLOCKED:
            return RoutingResult(
                query=query,
                packets=[],
                route_type=RouteType.BLOCKED,
                hops=0,
                latency_ms=(time.time() - start_time) * 1000,
                safety_level=SafetyLevel.BLOCKED,
                wormholes_used=0,
                path=[],
                similarity_scores=[],
                metadata={"blocked_reason": "Safety violation"},
            )

        # Simple term matching
        matching_packets = []
        query_terms = set(query.lower().split())

        for packet_id, packet in self._packets.items():
            data_str = str(packet["data"]).lower()
            data_terms = set(data_str.split())
            overlap = len(query_terms & data_terms)

            if overlap > 0:
                score = overlap / len(query_terms) if query_terms else 0
                matching_packets.append((packet, score))

        # Sort by score
        matching_packets.sort(key=lambda x: x[1], reverse=True)
        matching_packets = matching_packets[:k]

        # Count wormholes used
        wormholes_used = 0
        path = [p["id"] for p, _ in matching_packets]
        for i in range(len(path) - 1):
            if path[i] in self._wormholes and path[i + 1] in self._wormholes[path[i]]:
                wormholes_used += 1

        if wormholes_used > 0:
            self.wormhole_hits += 1

        # Determine route type
        if wormholes_used == 0:
            route_type = RouteType.STANDARD
        elif wormholes_used == 1:
            route_type = RouteType.DIRECT
        else:
            route_type = RouteType.CASCADE

        result = RoutingResult(
            query=query,
            packets=[p for p, _ in matching_packets],
            route_type=route_type,
            hops=len(path),
            latency_ms=(time.time() - start_time) * 1000,
            safety_level=safety_level,
            wormholes_used=wormholes_used,
            path=path,
            similarity_scores=[s for _, s in matching_packets],
        )

        # Update cache
        if use_cache:
            self._cache[cache_key] = (result, time.time())

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            "total_queries": self.total_queries,
            "wormhole_hits": self.wormhole_hits,
            "cache_hits": self.cache_hits,
            "packets": len(self._packets),
            "wormholes": sum(len(v) for v in self._wormholes.values()),
            "wormhole_hit_rate": self.wormhole_hits / max(self.total_queries, 1),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_engine: Optional[BrahimWormholeEngine] = None
_router: Optional[WormholeRouter] = None


def create_engine(throat_radius: float = 1.0) -> BrahimWormholeEngine:
    """Create a new wormhole engine instance."""
    return BrahimWormholeEngine(throat_radius)


def create_router(safety_checker: Callable = None) -> WormholeRouter:
    """Create a new wormhole router instance."""
    return WormholeRouter(safety_checker)


def get_engine() -> BrahimWormholeEngine:
    """Get global wormhole engine instance."""
    global _engine
    if _engine is None:
        _engine = BrahimWormholeEngine()
    return _engine


def quick_transform(x: List[float], iterations: int = 1) -> List[float]:
    """Quick wormhole transform without creating engine."""
    return WormholeTransform.forward(x, iterations)


def verify_sequence(sequence: List[int]) -> bool:
    """Quick sequence verification."""
    if len(sequence) != DIMENSION:
        return False
    return all(
        sequence[i] + sequence[DIMENSION - 1 - i] == PAIR_SUM
        for i in range(DIMENSION // 2)
    )
