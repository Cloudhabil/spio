"""
Physics Extension - Wormhole Physics and Einstein Field Equations

Ported from: CLI-main/src/core/wormhole_physics.py

Complete mathematical framework for traversable wormholes:
- Morris-Thorne metric tensor
- Einstein Field Equations
- Traversability conditions (NEC violation)
- Lyapunov stability analysis
- Brahim algebraic-continuous unification
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# =============================================================================
# FUNDAMENTAL CONSTANTS (Brahim-Golden Ratio Derived)
# =============================================================================

PHI = (1 + math.sqrt(5)) / 2                    # Golden ratio
PHI_INV = 1 / PHI                               # 1/phi
BETA = math.sqrt(5) - 2                         # 1/phi^3
GENESIS = 2 / 901                               # Critical density
LAMBDA_DECAY = 0.0219                           # Genesis-derived decay rate

# Physical constants (normalized units where G = c = hbar = 1)
G_NEWTON = 1.0
C_LIGHT = 1.0
HBAR = 1.0
PLANCK_LENGTH = 1.0

# Brahim sequence for algebraic wormholes
BRAHIM_SEQUENCE = [27, 42, 60, 75, 97, 117, 139, 154, 172, 187]
BRAHIM_PAIR_SUM = 214
BRAHIM_SUM = 214
BRAHIM_CENTER = 107


# =============================================================================
# ENUMS
# =============================================================================

class WormholeType(Enum):
    """Classification of wormhole geometries."""
    MORRIS_THORNE = "morris_thorne"
    THIN_SHELL = "thin_shell"
    TRAVERSABLE = "traversable"
    QUANTUM = "quantum"
    BRAHIM = "brahim"


class StabilityClass(Enum):
    """Lyapunov stability classification."""
    STABLE = "stable"
    UNSTABLE = "unstable"
    MARGINALLY_STABLE = "marginally_stable"
    SADDLE = "saddle"


class EnergyCondition(Enum):
    """Energy condition types."""
    NULL = "null"
    WEAK = "weak"
    STRONG = "strong"
    DOMINANT = "dominant"


class JunctionStatus(Enum):
    """Junction condition verification status."""
    CONTINUOUS = "continuous"
    C1_SMOOTH = "c1_smooth"
    C2_SMOOTH = "c2_smooth"
    DISCONTINUOUS = "discontinuous"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MetricTensor:
    """
    Morris-Thorne wormhole metric tensor components.

    ds^2 = -e^(2*Phi)*dt^2 + dr^2/(1 - b(r)/r) + r^2*(d_theta^2 + sin^2(theta)*d_phi^2)
    """
    r: float
    theta: float = math.pi / 2
    phi_redshift: float = 0.0
    b_shape: float = 0.0

    @property
    def g_tt(self) -> float:
        """Time-time component."""
        return -math.exp(2 * self.phi_redshift)

    @property
    def g_rr(self) -> float:
        """Radial-radial component."""
        if self.r <= self.b_shape:
            return float('inf')
        return 1.0 / (1.0 - self.b_shape / self.r)

    @property
    def g_theta_theta(self) -> float:
        """Theta-theta component."""
        return self.r ** 2

    @property
    def g_phi_phi(self) -> float:
        """Phi-phi component."""
        return (self.r * math.sin(self.theta)) ** 2

    @property
    def determinant(self) -> float:
        """Metric determinant sqrt(-g)."""
        return abs(self.g_tt * self.g_rr * self.g_theta_theta * self.g_phi_phi) ** 0.5

    def to_matrix(self) -> List[List[float]]:
        """Return 4x4 metric tensor matrix."""
        g = [[0.0] * 4 for _ in range(4)]
        g[0][0] = self.g_tt
        g[1][1] = self.g_rr
        g[2][2] = self.g_theta_theta
        g[3][3] = self.g_phi_phi
        return g


@dataclass
class StressEnergyTensor:
    """
    Stress-energy tensor for exotic matter.

    T^mu_nu = diag(-rho, p_r, p_t, p_t)
    """
    rho: float
    p_r: float
    p_t: float

    @property
    def is_exotic(self) -> bool:
        """Check if matter is exotic (NEC violated)."""
        return self.rho < 0 or (self.rho + self.p_r) < 0

    @property
    def nec_value(self) -> float:
        """NEC quantity: rho + p_r (negative = violation)."""
        return self.rho + self.p_r

    @property
    def wec_value(self) -> float:
        """WEC quantity: rho (negative = violation)."""
        return self.rho

    @property
    def sec_value(self) -> float:
        """SEC quantity: rho + p_r + 2*p_t."""
        return self.rho + self.p_r + 2 * self.p_t

    def to_matrix(self) -> List[List[float]]:
        """Return 4x4 stress-energy tensor."""
        T = [[0.0] * 4 for _ in range(4)]
        T[0][0] = -self.rho
        T[1][1] = self.p_r
        T[2][2] = self.p_t
        T[3][3] = self.p_t
        return T


@dataclass
class ShapeFunction:
    """
    Wormhole shape function b(r) with flare-out verification.

    Constraints for traversability:
        1. b(r_0) = r_0 at throat
        2. b(r) < r for r > r_0
        3. b'(r_0) < 1 (flare-out)
        4. b(r)/r -> 0 as r -> infinity (asymptotic flatness)
    """
    r_throat: float
    alpha_param: float = PHI_INV
    lambda_param: float = BETA

    def b(self, r: float) -> float:
        """Compute shape function b(r)."""
        if r < self.r_throat:
            return r

        ratio = self.r_throat / r
        decay = math.exp(-self.lambda_param * (r - self.r_throat) / self.r_throat)
        return self.r_throat * (ratio ** self.alpha_param) * decay

    def b_prime(self, r: float, epsilon: float = 1e-8) -> float:
        """Numerical derivative of b(r)."""
        return (self.b(r + epsilon) - self.b(r - epsilon)) / (2 * epsilon)

    def flare_out_condition(self, r: float) -> float:
        """Flare-out condition: b(r) - r * b'(r) > 0."""
        return self.b(r) - r * self.b_prime(r)

    def is_flare_out_satisfied(self, r: float) -> bool:
        """Check if flare-out condition is met at r."""
        return self.flare_out_condition(r) > 0

    def verify_throat(self) -> Dict[str, Any]:
        """Verify all throat conditions."""
        r0 = self.r_throat
        b_r0 = self.b(r0)
        b_prime_r0 = self.b_prime(r0)

        return {
            "r_throat": r0,
            "b_at_throat": b_r0,
            "throat_condition": abs(b_r0 - r0) < 1e-6,
            "b_prime_at_throat": b_prime_r0,
            "flare_out_satisfied": b_prime_r0 < 1,
            "flare_out_value": self.flare_out_condition(r0),
        }

    def verify_asymptotic(self, r_far: float = 100.0) -> Dict[str, Any]:
        """Verify asymptotic flatness."""
        ratio = self.b(r_far) / r_far
        return {
            "r_test": r_far,
            "b_over_r": ratio,
            "asymptotically_flat": ratio < 0.01,
        }


@dataclass
class JunctionConditions:
    """
    Israel junction conditions for wormhole throat.

    Continuity conditions across throat surface.
    """
    u_minus: float
    u_plus: float
    du_dn_minus: float = 0.0
    du_dn_plus: float = 0.0
    d2u_dn2_minus: float = 0.0
    d2u_dn2_plus: float = 0.0

    @property
    def field_jump(self) -> float:
        """[u] = u+ - u-."""
        return self.u_plus - self.u_minus

    @property
    def derivative_jump(self) -> float:
        """[du/dn]."""
        return self.du_dn_plus - self.du_dn_minus

    @property
    def second_derivative_jump(self) -> float:
        """[d2u/dn2]."""
        return self.d2u_dn2_plus - self.d2u_dn2_minus

    def verify(self, tolerance: float = 1e-6) -> Dict[str, Any]:
        """Verify all junction conditions."""
        return {
            "field_continuous": abs(self.field_jump) < tolerance,
            "c1_smooth": abs(self.derivative_jump) < tolerance,
            "c2_smooth": abs(self.second_derivative_jump) < tolerance,
            "field_jump": self.field_jump,
            "derivative_jump": self.derivative_jump,
            "status": self._get_status(tolerance),
        }

    def _get_status(self, tol: float) -> JunctionStatus:
        """Determine junction status."""
        if abs(self.field_jump) > tol:
            return JunctionStatus.DISCONTINUOUS
        if abs(self.derivative_jump) > tol:
            return JunctionStatus.CONTINUOUS
        if abs(self.second_derivative_jump) > tol:
            return JunctionStatus.C1_SMOOTH
        return JunctionStatus.C2_SMOOTH


@dataclass
class LyapunovAnalysis:
    """
    Lyapunov stability analysis for wormhole equilibrium.

    For equilibrium point x*:
        V(x) = Lyapunov function (positive definite)
        dV/dt <= 0 along trajectories
    """
    equilibrium: List[float]
    jacobian: List[List[float]]
    eigenvalues: List[complex] = field(default_factory=list)

    def __post_init__(self):
        """Compute eigenvalues after initialization."""
        if self.jacobian and len(self.jacobian) > 0:
            self.eigenvalues = self._compute_eigenvalues()

    def _compute_eigenvalues(self) -> List[complex]:
        """Compute eigenvalues of 2x2 Jacobian."""
        if len(self.jacobian) != 2:
            return []
        a, b = self.jacobian[0]
        c, d = self.jacobian[1]
        trace = a + d
        det = a * d - b * c
        discriminant = trace ** 2 - 4 * det
        if discriminant >= 0:
            sqrt_disc = math.sqrt(discriminant)
            return [
                complex((trace + sqrt_disc) / 2, 0),
                complex((trace - sqrt_disc) / 2, 0)
            ]
        else:
            sqrt_disc = math.sqrt(-discriminant)
            return [
                complex(trace / 2, sqrt_disc / 2),
                complex(trace / 2, -sqrt_disc / 2)
            ]

    @property
    def stability_class(self) -> StabilityClass:
        """Classify stability based on eigenvalues."""
        if not self.eigenvalues:
            return StabilityClass.MARGINALLY_STABLE

        real_parts = [e.real for e in self.eigenvalues]

        if all(r < -1e-10 for r in real_parts):
            return StabilityClass.STABLE
        elif all(r > 1e-10 for r in real_parts):
            return StabilityClass.UNSTABLE
        elif any(r > 1e-10 for r in real_parts) and any(r < -1e-10 for r in real_parts):
            return StabilityClass.SADDLE
        else:
            return StabilityClass.MARGINALLY_STABLE

    @property
    def spectral_abscissa(self) -> float:
        """Maximum real part of eigenvalues."""
        if not self.eigenvalues:
            return 0.0
        return max(e.real for e in self.eigenvalues)

    @property
    def is_stable(self) -> bool:
        """Quick stability check."""
        return self.stability_class == StabilityClass.STABLE


@dataclass
class BrahimWormhole:
    """
    Algebraic wormhole using Brahim sequence.

    Standard mirror: M(x) = S - x = 214 - x
    Fixed point C = 107

    Wormhole transform (bypasses singularity):
        W(x) = C + (x - C)/phi
        W^-1(x) = C + (x - C) * phi
    """
    sequence: List[int] = field(default_factory=lambda: BRAHIM_SEQUENCE.copy())

    @property
    def sum_S(self) -> int:
        """Brahim sum S = 214."""
        return BRAHIM_SUM

    @property
    def center_C(self) -> float:
        """Singularity center C = 107."""
        return BRAHIM_CENTER

    def mirror(self, x: float) -> float:
        """Standard mirror transform M(x) = S - x."""
        return self.sum_S - x

    def wormhole_forward(self, x: float) -> float:
        """Wormhole transform W(x) = C + (x - C)/phi."""
        return self.center_C + (x - self.center_C) / PHI

    def wormhole_inverse(self, x: float) -> float:
        """Inverse wormhole W^-1(x) = C + (x - C) * phi."""
        return self.center_C + (x - self.center_C) * PHI

    def combined_transform(self, x: float) -> float:
        """Combined Brahim-Wormhole transform bypassing singularity."""
        m_x = self.mirror(x)
        if m_x in self.sequence:
            return m_x

        w_x = self.wormhole_forward(x)
        m_w = self.mirror(w_x)
        return self.wormhole_inverse(m_w)

    def throat_location(self) -> float:
        """Wormhole throat in Brahim space: W(S) = C * phi."""
        return self.center_C * PHI

    def verify_closure(self) -> Dict[str, Any]:
        """Verify sequence is closed under transforms."""
        results = []
        for b in self.sequence:
            m_b = self.mirror(b)
            in_seq = m_b in self.sequence
            results.append({
                "element": b,
                "mirror": m_b,
                "closed": in_seq,
            })

        return {
            "all_closed": all(r["closed"] for r in results),
            "elements": results,
            "throat": self.throat_location(),
        }


# =============================================================================
# CORE PHYSICS ENGINES
# =============================================================================

class TraversabilityEngine:
    """
    Engine for enforcing traversability conditions.

    Morris-Thorne conditions:
        1. No horizon: e^(2*Phi) finite everywhere
        2. Throat geometry: b(r_0) = r_0, b'(r_0) < 1
        3. Flare-out: (b - b'r)/b^2 > 0
        4. Exotic matter: rho + p_r < 0 (NEC violation)
    """

    def __init__(self, r_throat: float = 1.0):
        self.r_throat = r_throat
        self.shape = ShapeFunction(r_throat=r_throat)

    def compute_exotic_matter_density(
        self,
        r: float,
        phi_func: Callable[[float], float] = None
    ) -> StressEnergyTensor:
        """Compute required exotic matter from Einstein equations."""
        b_r = self.shape.b(r)
        b_prime = self.shape.b_prime(r)

        if phi_func is None:
            phi_prime = 0.0
        else:
            epsilon = 1e-8
            phi_prime = (phi_func(r + epsilon) - phi_func(r - epsilon)) / (2 * epsilon)

        rho = b_prime / (8 * math.pi * G_NEWTON * r**2)

        term1 = b_r / r**3
        term2 = 2 * (1 - b_r/r) * phi_prime / r if r > b_r else 0
        p_r = (term1 - term2) / (8 * math.pi * G_NEWTON)

        p_t = -rho / 2

        return StressEnergyTensor(rho=rho, p_r=p_r, p_t=p_t)

    def verify_nec_violation(self, r_range: Tuple[float, float] = None) -> Dict[str, Any]:
        """Verify NEC is violated in throat region."""
        if r_range is None:
            r_range = (self.r_throat, self.r_throat * 3)

        r_values = [r_range[0] + i * (r_range[1] - r_range[0]) / 49 for i in range(50)]
        violations = []

        for r in r_values:
            T = self.compute_exotic_matter_density(r)
            violations.append({
                "r": r,
                "rho": T.rho,
                "p_r": T.p_r,
                "nec_value": T.nec_value,
                "violated": T.nec_value < 0,
            })

        throat_region = [v for v in violations if v["r"] < self.r_throat * 1.5]

        return {
            "throat_nec_violated": all(v["violated"] for v in throat_region),
            "min_nec_value": min(v["nec_value"] for v in violations),
            "violation_profile": violations,
            "traversable": all(v["violated"] for v in throat_region),
        }

    def minimum_exotic_matter(self) -> Dict[str, Any]:
        """Calculate minimum exotic matter required."""
        r_values = [self.r_throat + i * (self.r_throat * 9) / 99 for i in range(100)]
        dr = r_values[1] - r_values[0]

        total_exotic = 0.0
        for r in r_values:
            T = self.compute_exotic_matter_density(r)
            if T.rho < 0:
                total_exotic += T.rho * 4 * math.pi * r**2 * dr

        return {
            "total_exotic_mass": total_exotic,
            "throat_radius": self.r_throat,
            "exotic_per_throat_area": total_exotic / (4 * math.pi * self.r_throat**2),
        }


class EinsteinFieldEquations:
    """
    Full Einstein Field Equations for wormhole spacetime.

    G_mu_nu = 8*pi*G * T_mu_nu
    """

    def __init__(self, shape: ShapeFunction):
        self.shape = shape

    def ricci_tensor_components(self, r: float, phi: float = 0.0) -> Dict[str, float]:
        """Compute Ricci tensor components R_mu_nu."""
        b_r = self.shape.b(r)
        b_prime = self.shape.b_prime(r)

        if r <= b_r:
            return {"R_tt": float('inf'), "R_rr": float('inf'),
                    "R_theta_theta": float('inf'), "R_phi_phi": float('inf')}

        factor = 1 - b_r / r

        R_tt = 0.0
        R_rr = (b_prime * r - b_r) / (2 * r**2 * factor) if factor > 0 else float('inf')
        R_theta_theta = -factor + (b_r - b_prime * r) / (2 * r)
        R_phi_phi = R_theta_theta * math.sin(math.pi/2)**2

        return {
            "R_tt": R_tt,
            "R_rr": R_rr,
            "R_theta_theta": R_theta_theta,
            "R_phi_phi": R_phi_phi,
        }

    def ricci_scalar(self, r: float) -> float:
        """Compute Ricci scalar R = g^mu_nu R_mu_nu."""
        R_comp = self.ricci_tensor_components(r)
        b_r = self.shape.b(r)

        if r <= b_r:
            return float('inf')

        g_tt_inv = -1.0
        g_rr_inv = 1 - b_r / r
        g_theta_inv = 1 / r**2
        g_phi_inv = 1 / (r**2 * math.sin(math.pi/2)**2)

        R = (g_tt_inv * R_comp["R_tt"] +
             g_rr_inv * R_comp["R_rr"] +
             g_theta_inv * R_comp["R_theta_theta"] +
             g_phi_inv * R_comp["R_phi_phi"])

        return R

    def einstein_tensor(self, r: float) -> Dict[str, float]:
        """Compute Einstein tensor G_mu_nu = R_mu_nu - (1/2)*g_mu_nu*R."""
        R_comp = self.ricci_tensor_components(r)
        R_scalar = self.ricci_scalar(r)

        metric = MetricTensor(r=r, phi_redshift=0.0, b_shape=self.shape.b(r))

        G_tt = R_comp["R_tt"] - 0.5 * metric.g_tt * R_scalar
        G_rr = R_comp["R_rr"] - 0.5 * metric.g_rr * R_scalar
        G_theta = R_comp["R_theta_theta"] - 0.5 * metric.g_theta_theta * R_scalar
        G_phi = R_comp["R_phi_phi"] - 0.5 * metric.g_phi_phi * R_scalar

        return {
            "G_tt": G_tt,
            "G_rr": G_rr,
            "G_theta_theta": G_theta,
            "G_phi_phi": G_phi,
        }

    def solve_for_stress_energy(self, r: float) -> StressEnergyTensor:
        """Solve EFE for stress-energy: T_mu_nu = G_mu_nu / (8*pi*G)."""
        G = self.einstein_tensor(r)
        factor = 1.0 / (8 * math.pi * G_NEWTON)

        rho = -G["G_tt"] * factor
        p_r = G["G_rr"] * factor / MetricTensor(r=r, b_shape=self.shape.b(r)).g_rr
        p_t = G["G_theta_theta"] * factor / r**2

        return StressEnergyTensor(rho=rho, p_r=p_r, p_t=p_t)


class StabilityAnalyzer:
    """
    Lyapunov stability analysis for wormhole equilibria.

    Analyzes stability of:
        1. Throat radius equilibrium
        2. Energy functional critical points
        3. Dynamical system fixed points
    """

    def __init__(self):
        self.analyses: List[LyapunovAnalysis] = []

    def analyze_throat_stability(
        self,
        shape: ShapeFunction,
        perturbation: float = 0.01
    ) -> LyapunovAnalysis:
        """Analyze stability of throat radius."""
        r0 = shape.r_throat
        dr = perturbation * r0

        def V(r):
            return r - shape.b(r)

        V_second = (V(r0 + dr) - 2*V(r0) + V(r0 - dr)) / dr**2

        jacobian = [
            [0, 1],
            [-V_second, -LAMBDA_DECAY]
        ]

        equilibrium = [r0, 0]

        analysis = LyapunovAnalysis(
            equilibrium=equilibrium,
            jacobian=jacobian,
        )

        self.analyses.append(analysis)
        return analysis


class UnificationEngine:
    """
    Unifies algebraic (Brahim) and continuous (Hamiltonian) wormhole frameworks.

    Correspondence Principle:
        Discrete Brahim wormhole <-> Quantized continuous energy

    Key identity:
        E[psi] = GENESIS when psi lies on "critical line"
        Brahim center C = 107 maps to E = GENESIS
    """

    def __init__(self):
        self.brahim = BrahimWormhole()
        self.genesis = GENESIS

    def algebraic_to_continuous(self, x: int) -> float:
        """Map Brahim discrete value to continuous energy density."""
        C = self.brahim.center_C

        if x <= 0:
            return 0.0

        ratio_factor = (x / C) ** BETA
        decay_factor = math.exp(-abs(x - C) / (C * PHI))

        return self.genesis * ratio_factor * decay_factor

    def continuous_to_algebraic(self, rho: float) -> int:
        """Map continuous energy density to nearest Brahim value."""
        C = self.brahim.center_C

        best_x = int(C)
        best_diff = abs(self.algebraic_to_continuous(int(C)) - rho)

        for x in range(1, BRAHIM_SUM):
            diff = abs(self.algebraic_to_continuous(x) - rho)
            if diff < best_diff:
                best_diff = diff
                best_x = x

        return best_x

    def correspondence_map(self) -> Dict[str, Any]:
        """Full correspondence between algebraic and continuous."""
        mapping = []

        for b in self.brahim.sequence:
            rho = self.algebraic_to_continuous(b)
            mapping.append({
                "brahim_value": b,
                "energy_density": rho,
                "ratio_to_genesis": rho / self.genesis,
                "mirror_value": self.brahim.mirror(b),
            })

        center_rho = self.algebraic_to_continuous(int(self.brahim.center_C))
        throat_rho = self.algebraic_to_continuous(int(self.brahim.throat_location()))

        return {
            "sequence_mapping": mapping,
            "center": {
                "brahim": self.brahim.center_C,
                "energy": center_rho,
                "is_genesis": abs(center_rho - self.genesis) < self.genesis * 0.1,
            },
            "throat": {
                "brahim": self.brahim.throat_location(),
                "energy": throat_rho,
            },
            "genesis_constant": self.genesis,
            "compression_factor": BETA,
        }

    def verify_correspondence(self) -> Dict[str, Any]:
        """Verify the algebraic-continuous correspondence principle."""
        center_rho = self.algebraic_to_continuous(int(self.brahim.center_C))
        center_verified = abs(center_rho - self.genesis) < self.genesis * 0.2

        x_test = 150
        w_x = self.brahim.wormhole_forward(x_test)
        rho_x = self.algebraic_to_continuous(x_test)
        rho_w = self.algebraic_to_continuous(int(w_x))

        energy_ratio = rho_w / rho_x if rho_x > 0 else 0
        transform_verified = abs(energy_ratio - PHI_INV) < 0.3

        return {
            "center_genesis_correspondence": center_verified,
            "center_energy": center_rho,
            "genesis": self.genesis,
            "wormhole_energy_scaling": energy_ratio,
            "expected_scaling": PHI_INV,
            "transform_verified": transform_verified,
            "overall_verified": center_verified and transform_verified,
        }


# =============================================================================
# INTEGRATED WORMHOLE PHYSICS SYSTEM
# =============================================================================

class WormholePhysicsSystem:
    """
    Complete wormhole physics system integrating all components.

    Components:
        1. TraversabilityEngine - NEC violation enforcement
        2. EinsteinFieldEquations - Metric tensor evolution
        3. StabilityAnalyzer - Lyapunov stability
        4. UnificationEngine - Algebraic-continuous bridge
    """

    def __init__(self, r_throat: float = 1.0):
        self.r_throat = r_throat
        self.shape = ShapeFunction(r_throat=r_throat)

        self.traversability = TraversabilityEngine(r_throat=r_throat)
        self.einstein = EinsteinFieldEquations(shape=self.shape)
        self.stability = StabilityAnalyzer()
        self.unification = UnificationEngine()

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def full_wormhole_analysis(self) -> Dict[str, Any]:
        """Complete wormhole physics analysis."""
        throat_check = self.shape.verify_throat()
        asymptotic_check = self.shape.verify_asymptotic()

        nec_analysis = self.traversability.verify_nec_violation()
        exotic_matter = self.traversability.minimum_exotic_matter()

        stability_analysis = self.stability.analyze_throat_stability(self.shape)

        correspondence = self.unification.verify_correspondence()

        junction = JunctionConditions(
            u_minus=self.shape.b(self.r_throat - 0.01),
            u_plus=self.shape.b(self.r_throat + 0.01),
            du_dn_minus=self.shape.b_prime(self.r_throat - 0.01),
            du_dn_plus=self.shape.b_prime(self.r_throat + 0.01),
        )
        junction_check = junction.verify()

        return {
            "session_id": self.session_id,
            "throat_radius": self.r_throat,
            "geometry": {
                "shape_function": {
                    "throat_verified": throat_check["throat_condition"],
                    "flare_out_satisfied": throat_check["flare_out_satisfied"],
                    "asymptotically_flat": asymptotic_check["asymptotically_flat"],
                },
            },
            "traversability": {
                "nec_violated": nec_analysis["traversable"],
                "min_nec_value": nec_analysis["min_nec_value"],
                "exotic_mass_required": exotic_matter["total_exotic_mass"],
            },
            "stability": {
                "class": stability_analysis.stability_class.value,
                "is_stable": stability_analysis.is_stable,
                "spectral_abscissa": stability_analysis.spectral_abscissa,
                "eigenvalues": [(e.real, e.imag) for e in stability_analysis.eigenvalues],
            },
            "junction": {
                "status": junction_check["status"].value,
                "field_continuous": junction_check["field_continuous"],
                "c1_smooth": junction_check["c1_smooth"],
            },
            "unification": {
                "algebraic_continuous_verified": correspondence["overall_verified"],
                "center_genesis_map": correspondence["center_genesis_correspondence"],
                "wormhole_scaling": correspondence["wormhole_energy_scaling"],
            },
            "wormhole_valid": (
                throat_check["throat_condition"] and
                throat_check["flare_out_satisfied"] and
                nec_analysis["traversable"] and
                stability_analysis.is_stable
            ),
        }

    def create_traversable_wormhole(
        self,
        r_throat: float = None,
        verify: bool = True
    ) -> Dict[str, Any]:
        """Create and verify a traversable wormhole configuration."""
        if r_throat is not None:
            self.r_throat = r_throat
            self.shape = ShapeFunction(r_throat=r_throat)
            self.traversability = TraversabilityEngine(r_throat=r_throat)
            self.einstein = EinsteinFieldEquations(shape=self.shape)

        wormhole = {
            "type": WormholeType.TRAVERSABLE.value,
            "throat_radius": self.r_throat,
            "shape_parameters": {
                "alpha": self.shape.alpha_param,
                "lambda": self.shape.lambda_param,
            },
            "metric_at_throat": MetricTensor(
                r=self.r_throat * 1.01,
                phi_redshift=0.0,
                b_shape=self.shape.b(self.r_throat * 1.01),
            ).to_matrix(),
            "stress_energy_at_throat": {
                "rho": self.einstein.solve_for_stress_energy(self.r_throat * 1.01).rho,
                "p_r": self.einstein.solve_for_stress_energy(self.r_throat * 1.01).p_r,
                "p_t": self.einstein.solve_for_stress_energy(self.r_throat * 1.01).p_t,
            },
        }

        if verify:
            analysis = self.full_wormhole_analysis()
            wormhole["verification"] = analysis
            wormhole["valid"] = analysis["wormhole_valid"]

        return wormhole


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_physics_system: Optional[WormholePhysicsSystem] = None


def verify_wormhole_traversability(r_throat: float = 1.0) -> Dict[str, Any]:
    """Quick verification of wormhole traversability."""
    system = WormholePhysicsSystem(r_throat=r_throat)
    return system.full_wormhole_analysis()


def create_brahim_wormhole() -> BrahimWormhole:
    """Create algebraic Brahim wormhole."""
    return BrahimWormhole()


def create_physics_system(r_throat: float = 1.0) -> WormholePhysicsSystem:
    """Create a new physics system instance."""
    return WormholePhysicsSystem(r_throat=r_throat)


def get_physics_system() -> WormholePhysicsSystem:
    """Get global physics system instance."""
    global _physics_system
    if _physics_system is None:
        _physics_system = WormholePhysicsSystem()
    return _physics_system


__all__ = [
    # Constants
    "PHI", "PHI_INV", "BETA", "GENESIS", "LAMBDA_DECAY",
    "BRAHIM_SEQUENCE", "BRAHIM_SUM", "BRAHIM_CENTER",
    # Enums
    "WormholeType", "StabilityClass", "EnergyCondition", "JunctionStatus",
    # Data Classes
    "MetricTensor", "StressEnergyTensor", "ShapeFunction",
    "JunctionConditions", "LyapunovAnalysis", "BrahimWormhole",
    # Engines
    "TraversabilityEngine", "EinsteinFieldEquations",
    "StabilityAnalyzer", "UnificationEngine",
    "WormholePhysicsSystem",
    # Factory
    "verify_wormhole_traversability", "create_brahim_wormhole",
    "create_physics_system", "get_physics_system",
]
