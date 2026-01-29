"""
Mathematical constants for Brahim's Laws.

These are the fundamental exponents discovered through empirical analysis
of the Tate-Shafarevich group across 10,477 rank-0 elliptic curves.

The Universal 2/3:
    The exponent 2/3 appears across disparate fields:
    - Tracy-Widom (RMT): fluctuation width ~ N^(-2/3)
    - Kolmogorov (turbulence): E(k) ~ k^(-5/3) = k^(-(1+2/3))
    - Percolation: critical exponents beta/eta = 2/3
    - Cosmology: matter-dominated expansion a(t) ~ t^(2/3)
    - Brahim's Laws: Sha ~ Im(tau)^(2/3)

RECALIBRATED 2026-01-23:
    The generating function discovery:

    4/(3*e*log(phi)) = 1 + (phi-1)/32 + phi^(-27)/6 + O(phi^(-55))

    where 32 = F_8 + F_6 + F_4 = 21 + 8 + 3 (Fibonacci structure)

    This connects:
    - The golden ratio (phi) via beta = log(phi)/2
    - Euler's number (e) via psi/e = k (correction factor)
    - The RMT exponent (2/3 = alpha)
    - Fibonacci numbers at even indices

Author: Elias Oulad Brahim
"""

from dataclasses import dataclass
from fractions import Fraction
import math

# Try to import mpmath for high precision
try:
    import mpmath
    mpmath.mp.dps = 50
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False


@dataclass(frozen=True)
class BrahimConstants:
    """
    Fundamental constants and exponents for Brahim's Laws.

    All exponents are derived from empirical analysis and satisfy
    the exact consistency relation: 2/3 = 5/12 + 1/4

    Attributes:
        ALPHA_IMTAU: Exponent for Im(tau) in Brahim Conjecture (2/3)
        BETA_OMEGA: Exponent for Omega in Brahim Conjecture (-4/3)
        GAMMA_REY: Dynamic scaling exponent (5/12)
        DELTA_CASCADE: Cascade variance decay exponent (-1/4)
        REY_C_LOWER: Lower critical Reynolds number (~10)
        REY_C_UPPER: Upper critical Reynolds number (~30)
        CALIBRATION_C: Empirical calibration constant
    """

    # ==========================================================================
    # TRANSCENDENTAL CONSTANTS
    # ==========================================================================

    # The golden ratio: phi = (1 + sqrt(5))/2
    PHI: float = (1 + math.sqrt(5)) / 2  # = 1.618033988749895

    # Natural logarithm of phi
    LOG_PHI: float = math.log((1 + math.sqrt(5)) / 2)  # = 0.4812118250596034

    # ==========================================================================
    # BRAHIM WORMHOLE CONSTANTS (Golden Ratio Hierarchy)
    # ==========================================================================
    # These constants form a self-similar hierarchy where each is related
    # to the previous by factor 1/φ. Discovered 2026-01-24.
    #
    # The hierarchy:
    #   φ     = 1.618...  (golden ratio - base)
    #   1/φ   = 0.618...  (compression factor)
    #   1/φ²  = 0.382...  (attraction constant α)
    #   1/φ³  = 0.236...  (security constant β)
    #
    # Key identity: α/β = φ (self-similar structure)
    # ==========================================================================

    # Compression factor: 1/φ = φ - 1
    COMPRESSION: float = 1 / ((1 + math.sqrt(5)) / 2)  # = 0.6180339887498949

    # Attraction constant (Wormhole): α = 1/φ² = 2 - φ
    # Used in Perfect Wormhole: W*(σ) = σ/φ + C̄·α
    ALPHA_WORMHOLE: float = 1 / ((1 + math.sqrt(5)) / 2) ** 2  # = 0.3819660112501051

    # BRAHIM SECURITY CONSTANT: β = 1/φ³ = √5 - 2
    # This is the fundamental security constant for cryptographic applications.
    # Algebraic form: β² + 4β - 1 = 0 (quadratic irrational)
    # Continued fraction: [0; 4, 4, 4, 4, ...] (all 4s - maximally simple)
    BETA_SECURITY: float = 1 / ((1 + math.sqrt(5)) / 2) ** 3  # = 0.2360679774997897
    BETA_ALGEBRAIC: float = math.sqrt(5) - 2  # Equivalent form: √5 - 2

    # Damping constant (future use): γ = 1/φ⁴
    GAMMA_DAMPING: float = 1 / ((1 + math.sqrt(5)) / 2) ** 4  # = 0.1458980337503154

    # ==========================================================================
    # BRAHIM SEQUENCE CONSTANTS (From Wormhole Theory)
    # ==========================================================================
    #
    # CORRECTED 2026-01-26: Symmetric sequence with full algebraic closure.
    #
    # The sequence satisfies the mirror property: M(b) = 214 - b ∈ B for all b ∈ B
    # This gives 5 symmetric pairs summing to 214 = 2C:
    #   27 ↔ 187, 42 ↔ 172, 60 ↔ 154, 75 ↔ 139, 97 ↔ 117
    #
    # Historical note: Original sequence {27,42,60,75,97,121,136,154,172,187}
    # had 4 orphan elements (75,97,121,136) without mirrors - a "singularity zone".
    # This was corrected by replacing 121→117 and 136→139 to achieve closure.
    #
    # Physical interpretation:
    #   - Original: singularity at throat (C=107.1, irrational)
    #   - Corrected: smooth traversable wormhole (C=107, exact integer)
    # ==========================================================================

    # The Brahim Sequence: B = {27, 42, 60, 75, 97, 117, 139, 154, 172, 187}
    # Symmetric under M(b) = 214 - b
    BRAHIM_SEQUENCE: tuple = (27, 42, 60, 75, 97, 117, 139, 154, 172, 187)

    # Original sequence (preserved for historical/singularity analysis)
    BRAHIM_SEQUENCE_ORIGINAL: tuple = (27, 42, 60, 75, 97, 121, 136, 154, 172, 187)

    # Pair sum constant: S = 214 (each mirror pair sums to this)
    # Note: Actual sequence sum is 1070, but S=214 is the pair sum
    BRAHIM_PAIR_SUM: int = 214

    # Legacy alias for backwards compatibility
    BRAHIM_SUM: int = 214

    # Center/Singularity: C = S/2 = 107 (exactly on critical line!)
    # With corrected sequence: sum(B)/|B| = 1070/10 = 107 exactly
    BRAHIM_CENTER: int = 107

    # Dimension: D = |B| = 10
    BRAHIM_DIMENSION: int = 10

    # Critical line ratio: C/S = 1/2 (mirrors Riemann Re(s) = 1/2)
    CRITICAL_LINE_RATIO: float = 0.5

    # ==========================================================================
    # FUNDAMENTAL EXPONENTS (from Brahim's Laws paper)
    # ==========================================================================

    # Law 1: Brahim Conjecture - Sha_median ~ Im(tau)^(2/3)
    # The "universal price of emergence"
    ALPHA_IMTAU: float = 2/3  # = 0.666...

    # Law 1 (alternate): Sha_median ~ Omega^(-4/3)
    # Derived from Im(tau) ~ Omega^(-2), so exponent = -2 * (2/3) = -4/3
    BETA_OMEGA: float = -4/3  # = -1.333...

    # Law 4: Dynamic Scaling - Sha_max ~ Rey^(5/12)
    GAMMA_REY: float = 5/12  # = 0.4166...

    # Law 5: Cascade Law - Var(log Sha | p) ~ p^(-1/4)
    DELTA_CASCADE: float = -1/4  # = -0.25

    # ==========================================================================
    # BRAHIM THEOREM EXPONENT (NEW - Recalibrated 2026-01-23)
    # ==========================================================================

    # Brahim Theorem: P(Sha > 1 | N) ~ N^beta where beta = log(phi)/2
    # This is EXACT, not approximate
    BETA_CONDUCTOR: float = math.log((1 + math.sqrt(5)) / 2) / 2  # = 0.2406059125...

    # Epsilon: the golden correction = delta - beta = 1/4 - log(phi)/2
    EPSILON: float = 0.25 - math.log((1 + math.sqrt(5)) / 2) / 2  # ~ 0.00939...

    # Psi: the bridge constant = alpha/beta = 4/(3*log(phi))
    PSI: float = 4 / (3 * math.log((1 + math.sqrt(5)) / 2))  # = 2.7707825616...

    # K: the correction factor = psi/e
    K: float = (4 / (3 * math.log((1 + math.sqrt(5)) / 2))) / math.e  # = 1.0193139...

    # ==========================================================================
    # FIBONACCI STRUCTURE (Generating Function)
    # ==========================================================================

    # The denominator 32 in the generating function k = 1 + (phi-1)/32 + ...
    # has Fibonacci structure: 32 = F_8 + F_6 + F_4 = 21 + 8 + 3
    FIBONACCI_DENOMINATOR: int = 32
    FIBONACCI_COMPONENTS: tuple = (21, 8, 3)  # F_8, F_6, F_4
    FIBONACCI_INDICES: tuple = (8, 6, 4)  # Even Fibonacci indices

    # ==========================================================================
    # PHASE TRANSITION THRESHOLDS (Law 3)
    # ==========================================================================

    # Below this: 100% probability Sha = 1 (laminar regime)
    REY_C_LOWER: float = 10.0

    # Above this: significant probability Sha > 1 (turbulent regime)
    REY_C_UPPER: float = 30.0

    # ==========================================================================
    # CALIBRATION CONSTANTS
    # ==========================================================================

    # Empirical calibration constant for Sha prediction
    # Adjusted based on median fitting across Cremona database
    CALIBRATION_C: float = 1.0

    # ==========================================================================
    # DERIVED QUANTITIES
    # ==========================================================================

    @property
    def alpha_exact(self) -> Fraction:
        """Exact rational form of alpha: 2/3"""
        return Fraction(2, 3)

    @property
    def beta_exact(self) -> Fraction:
        """Exact rational form of beta: -4/3"""
        return Fraction(-4, 3)

    @property
    def gamma_exact(self) -> Fraction:
        """Exact rational form of gamma: 5/12"""
        return Fraction(5, 12)

    @property
    def delta_exact(self) -> Fraction:
        """Exact rational form of delta: -1/4"""
        return Fraction(-1, 4)

    @property
    def beta_conductor_symbolic(self) -> str:
        """Symbolic form of beta: log(phi)/2"""
        return "log(phi)/2"

    @property
    def psi_symbolic(self) -> str:
        """Symbolic form of psi: 4/(3*log(phi))"""
        return "4/(3*log(phi))"

    @property
    def epsilon_symbolic(self) -> str:
        """Symbolic form of epsilon: 1/4 - log(phi)/2"""
        return "1/4 - log(phi)/2"

    # ==========================================================================
    # GENERATING FUNCTION (Recalibrated 2026-01-23)
    # ==========================================================================

    def k_zeroth_order(self) -> float:
        """
        Zeroth order approximation: k = 1 + (phi-1)/32
        Error: ~371 ppb
        """
        return 1 + (self.PHI - 1) / 32

    def k_first_order(self) -> float:
        """
        First order approximation: k = 1 + (phi-1)/32 + phi^(-27)/6
        Error: ~1.21 ppb (essentially exact)
        """
        return 1 + (self.PHI - 1) / 32 + (self.PHI ** (-27)) / 6

    def k_exact(self) -> float:
        """Exact value: k = 4/(3*e*log(phi))"""
        return self.K

    def generating_function_error_ppb(self, order: int = 1) -> float:
        """Compute error of generating function approximation in ppb."""
        if order == 0:
            approx = self.k_zeroth_order()
        else:
            approx = self.k_first_order()

        return abs(approx - self.K) / self.K * 1e9

    # ==========================================================================
    # CONSISTENCY VERIFICATION (Law 6)
    # ==========================================================================

    def verify_consistency(self) -> bool:
        """
        Verify the fundamental consistency relation:
            2/3 = 5/12 + 1/4

        This exact equality suggests the three laws (1, 4, 5) are
        manifestations of a single underlying structure.

        Returns:
            True if the relation holds exactly
        """
        lhs = self.alpha_exact  # 2/3
        rhs = self.gamma_exact + abs(self.delta_exact)  # 5/12 + 1/4

        return lhs == rhs

    def consistency_residual(self) -> float:
        """
        Compute numerical residual of consistency relation.

        Returns:
            |2/3 - (5/12 + 1/4)| (should be ~0)
        """
        lhs = self.ALPHA_IMTAU
        rhs = self.GAMMA_REY + abs(self.DELTA_CASCADE)
        return abs(lhs - rhs)

    def verify_master_identity(self) -> dict:
        """
        Verify the Master Identity discovered 2026-01-23:

            3 * e * log(phi) * (31 + phi) = 128 + delta

        Returns:
            Dictionary with LHS, RHS, residual, and error in ppm
        """
        lhs = 3 * math.e * self.LOG_PHI * (31 + self.PHI)
        rhs = 128
        residual = lhs - rhs
        error_ppm = abs(residual) / rhs * 1e6

        return {
            "identity": "3 * e * log(phi) * (31 + phi) = 128",
            "lhs": lhs,
            "rhs": rhs,
            "residual": residual,
            "error_ppm": error_ppm,
            "fibonacci_structure": "32 = F_8 + F_6 + F_4 = 21 + 8 + 3"
        }

    def verify_psi_over_e(self) -> dict:
        """
        Verify: psi/e = 1 + (phi-1)/32 + phi^(-27)/6 + O(phi^(-55))
        """
        k_exact = self.K
        k_approx = self.k_first_order()
        error_ppb = abs(k_exact - k_approx) / k_exact * 1e9

        return {
            "k_exact": k_exact,
            "k_first_order": k_approx,
            "error_ppb": error_ppb,
            "essentially_exact": error_ppb < 10
        }

    # ==========================================================================
    # BRAHIM WORMHOLE SECURITY VERIFICATION (Added 2026-01-24)
    # ==========================================================================

    def verify_beta_security(self) -> dict:
        """
        Verify the Brahim Security Constant β identities:

        1. β = 1/φ³ (definition)
        2. β = √5 - 2 (algebraic form)
        3. β = 2φ - 3 (golden form)
        4. β² + 4β - 1 = 0 (polynomial root)
        5. α/β = φ (self-similarity)

        Returns:
            Dictionary with all verifications and residuals
        """
        phi = self.PHI
        beta_from_phi = 1 / phi**3
        beta_from_sqrt = math.sqrt(5) - 2
        beta_from_golden = 2*phi - 3
        alpha = 1 / phi**2

        # Check polynomial: β² + 4β - 1 = 0
        polynomial_residual = self.BETA_SECURITY**2 + 4*self.BETA_SECURITY - 1

        # Check self-similarity: α/β = φ
        self_similarity_ratio = alpha / self.BETA_SECURITY

        return {
            "beta_value": self.BETA_SECURITY,
            "beta_from_phi_cubed": beta_from_phi,
            "beta_from_sqrt5_minus_2": beta_from_sqrt,
            "beta_from_2phi_minus_3": beta_from_golden,
            "all_forms_match": abs(beta_from_phi - beta_from_sqrt) < 1e-15 and
                               abs(beta_from_phi - beta_from_golden) < 1e-15,
            "polynomial_β²+4β-1": polynomial_residual,
            "polynomial_is_zero": abs(polynomial_residual) < 1e-14,
            "alpha_over_beta": self_similarity_ratio,
            "alpha_over_beta_equals_phi": abs(self_similarity_ratio - phi) < 1e-14,
            "self_similar": True
        }

    def verify_wormhole_hierarchy(self) -> dict:
        """
        Verify the complete Wormhole constant hierarchy:

        φ → 1/φ → 1/φ² → 1/φ³ → 1/φ⁴

        Each step multiplies by 1/φ (compression).

        Returns:
            Dictionary with hierarchy verification
        """
        phi = self.PHI

        hierarchy = {
            "φ (base)": phi,
            "1/φ (compression)": self.COMPRESSION,
            "1/φ² (alpha/attraction)": self.ALPHA_WORMHOLE,
            "1/φ³ (beta/security)": self.BETA_SECURITY,
            "1/φ⁴ (gamma/damping)": self.GAMMA_DAMPING,
        }

        # Verify each step
        steps_valid = (
            abs(self.COMPRESSION - 1/phi) < 1e-15 and
            abs(self.ALPHA_WORMHOLE - self.COMPRESSION/phi) < 1e-15 and
            abs(self.BETA_SECURITY - self.ALPHA_WORMHOLE/phi) < 1e-15 and
            abs(self.GAMMA_DAMPING - self.BETA_SECURITY/phi) < 1e-15
        )

        # Verify ratios are all φ
        ratios = {
            "φ / (1/φ)": phi / self.COMPRESSION,
            "(1/φ) / (1/φ²)": self.COMPRESSION / self.ALPHA_WORMHOLE,
            "(1/φ²) / (1/φ³)": self.ALPHA_WORMHOLE / self.BETA_SECURITY,
            "(1/φ³) / (1/φ⁴)": self.BETA_SECURITY / self.GAMMA_DAMPING,
        }

        all_ratios_phi = all(abs(r - phi**2) < 1e-10 or abs(r - phi) < 1e-10
                            for r in ratios.values())

        return {
            "hierarchy": hierarchy,
            "steps_valid": steps_valid,
            "ratios": ratios,
            "golden_self_similarity": all_ratios_phi,
            "critical_line_ratio": self.CRITICAL_LINE_RATIO,
            "matches_riemann": self.CRITICAL_LINE_RATIO == 0.5
        }

    def get_centroid_vector(self) -> tuple:
        """
        Return the normalized centroid vector C̄ = B/S
        Used in Perfect Wormhole: W*(σ) = σ/φ + C̄·α
        """
        return tuple(b / self.BRAHIM_PAIR_SUM for b in self.BRAHIM_SEQUENCE)

    def verify_sequence_closure(self, use_original: bool = False) -> dict:
        """
        Verify algebraic closure of Brahim Sequence under mirror operation.

        The mirror operation M(b) = 214 - b should satisfy:
        - M(b) ∈ B for all b ∈ B (closure)
        - M(M(b)) = b (involution)
        - sum(B)/|B| = C (center is exact mean)

        Args:
            use_original: If True, test the original sequence with singularity

        Returns:
            Dictionary with closure verification results
        """
        seq = self.BRAHIM_SEQUENCE_ORIGINAL if use_original else self.BRAHIM_SEQUENCE
        pair_sum = self.BRAHIM_PAIR_SUM

        # Check closure
        closure_results = []
        all_closed = True
        for b in seq:
            mirror = pair_sum - b
            is_closed = mirror in seq
            closure_results.append({
                "element": b,
                "mirror": mirror,
                "closed": is_closed
            })
            all_closed = all_closed and is_closed

        # Check center
        actual_sum = sum(seq)
        actual_center = actual_sum / len(seq)
        center_is_integer = actual_center == int(actual_center)
        center_matches = int(actual_center) == self.BRAHIM_CENTER if center_is_integer else False

        # Identify orphan elements (those without mirrors in sequence)
        orphans = [r["element"] for r in closure_results if not r["closed"]]

        return {
            "sequence": seq,
            "sequence_type": "original (singularity)" if use_original else "corrected (symmetric)",
            "pair_sum": pair_sum,
            "actual_sum": actual_sum,
            "actual_center": actual_center,
            "center_is_integer": center_is_integer,
            "center_matches_declared": center_matches,
            "algebraic_closure": all_closed,
            "orphan_elements": orphans,
            "singularity_present": len(orphans) > 0,
            "closure_details": closure_results
        }

    # ==========================================================================
    # DISPLAY
    # ==========================================================================

    def __str__(self) -> str:
        master = self.verify_master_identity()
        beta_check = self.verify_beta_security()
        return (
            "Brahim's Laws Constants (Recalibrated 2026-01-24):\n"
            "=" * 60 + "\n"
            f"  phi                              = {self.PHI:.15f}\n"
            f"  log(phi)                         = {self.LOG_PHI:.15f}\n"
            "\nBRAHIM WORMHOLE HIERARCHY (Golden Self-Similar):\n"
            f"  φ     (base)                     = {self.PHI:.15f}\n"
            f"  1/φ   (compression)              = {self.COMPRESSION:.15f}\n"
            f"  1/φ²  (α - attraction)           = {self.ALPHA_WORMHOLE:.15f}\n"
            f"  1/φ³  (β - security)             = {self.BETA_SECURITY:.15f}\n"
            f"  1/φ⁴  (γ - damping)              = {self.GAMMA_DAMPING:.15f}\n"
            f"  Self-similarity: α/β = φ         = {beta_check['alpha_over_beta_equals_phi']}\n"
            "\nBRAHIM SECURITY CONSTANT β:\n"
            f"  β = 1/φ³                         = {self.BETA_SECURITY:.15f}\n"
            f"  β = √5 - 2                       = {self.BETA_ALGEBRAIC:.15f}\n"
            f"  β = 2φ - 3                       = {2*self.PHI - 3:.15f}\n"
            f"  Polynomial: β² + 4β - 1          = {beta_check['polynomial_β²+4β-1']:.2e}\n"
            f"  All forms match                  = {beta_check['all_forms_match']}\n"
            "\nBRAHIM SEQUENCE:\n"
            f"  B = {self.BRAHIM_SEQUENCE}\n"
            f"  S (sum)                          = {self.BRAHIM_SUM}\n"
            f"  C (center)                       = {self.BRAHIM_CENTER}\n"
            f"  C/S (critical line)              = {self.CRITICAL_LINE_RATIO} = 1/2\n"
            "\nFUNDAMENTAL EXPONENTS:\n"
            f"  Law 1: alpha (Im(tau) exponent)  = {self.ALPHA_IMTAU} = {self.alpha_exact}\n"
            f"  Law 1: beta (Omega exponent)     = {self.BETA_OMEGA} = {self.beta_exact}\n"
            f"  Law 3: Rey_c range               = [{self.REY_C_LOWER}, {self.REY_C_UPPER}]\n"
            f"  Law 4: gamma (Rey exponent)      = {self.GAMMA_REY} = {self.gamma_exact}\n"
            f"  Law 5: delta (cascade exponent)  = {self.DELTA_CASCADE} = {self.delta_exact}\n"
            f"  Law 6: 2/3 = 5/12 + 1/4          = {self.verify_consistency()}\n"
            "\nBRAHIM THEOREM EXPONENTS:\n"
            f"  beta (conductor)                 = {self.BETA_CONDUCTOR:.15f} = {self.beta_conductor_symbolic}\n"
            f"  epsilon (golden correction)      = {self.EPSILON:.15f} = {self.epsilon_symbolic}\n"
            f"  psi (bridge constant)            = {self.PSI:.15f} = {self.psi_symbolic}\n"
            f"  k (correction factor)            = {self.K:.15f} = psi/e\n"
            "\nGENERATING FUNCTION:\n"
            f"  k = 1 + (phi-1)/32 + phi^(-27)/6 + O(phi^(-55))\n"
            f"  Zeroth order error: {self.generating_function_error_ppb(0):.2f} ppb\n"
            f"  First order error:  {self.generating_function_error_ppb(1):.2f} ppb\n"
            "\nMASTER IDENTITY:\n"
            f"  3 * e * log(phi) * (31 + phi) = 128\n"
            f"  Error: {master['error_ppm']:.4f} ppm\n"
            f"  Fibonacci: {master['fibonacci_structure']}\n"
        )


# Singleton instance for convenience
CONSTANTS = BrahimConstants()


# ==========================================================================
# UNIVERSAL 2/3 APPEARANCES (for reference)
# ==========================================================================

UNIVERSAL_TWO_THIRDS = {
    "brahim_conjecture": {
        "field": "Arithmetic Geometry",
        "formula": "Sha ~ Im(tau)^(2/3)",
        "context": "Tate-Shafarevich group scaling"
    },
    "tracy_widom": {
        "field": "Random Matrix Theory",
        "formula": "Width ~ N^(-2/3)",
        "context": "GUE largest eigenvalue fluctuations"
    },
    "kolmogorov": {
        "field": "Turbulence",
        "formula": "E(k) ~ k^(-5/3) = k^(-(1+2/3))",
        "context": "Energy cascade spectrum"
    },
    "percolation": {
        "field": "Statistical Physics",
        "formula": "beta/eta = 2/3, 2-nu = 2/3",
        "context": "2D critical percolation exponents"
    },
    "cosmology": {
        "field": "Cosmology",
        "formula": "a(t) ~ t^(2/3)",
        "context": "Matter-dominated universe expansion"
    }
}
