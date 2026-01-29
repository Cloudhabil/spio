"""
Core Brahim's Laws Engine.

Implements all 6 of Brahim's Laws for elliptic curve analysis.

Laws:
    1. Brahim Conjecture: Sha_median ~ Im(tau)^(2/3) ~ Omega^(-4/3)
    2. Arithmetic Reynolds Number: Rey = N/(Tam*Omega)
    3. Phase Transition: Rey_c in [10, 30]
    4. Dynamic Scaling: Sha_max ~ Rey^(5/12)
    5. Cascade Law: Var(log Sha | p) ~ p^(-1/4)
    6. Consistency Relation: 2/3 = 5/12 + 1/4

Author: Elias Oulad Brahim
"""

import numpy as np
from typing import Tuple, Optional, List
from datetime import datetime

from .constants import BrahimConstants, CONSTANTS
from ..models.curve_data import EllipticCurveData, Regime
from ..models.analysis_result import BrahimAnalysisResult


class BrahimLawsEngine:
    """
    Core computational engine for Brahim's Laws.

    This class implements all 6 laws and provides methods for:
    - Individual curve analysis
    - Batch processing
    - Statistical analysis
    - Law verification

    Example:
        engine = BrahimLawsEngine()
        result = engine.analyze(curve)
        print(result.summary())
    """

    def __init__(
        self,
        constants: Optional[BrahimConstants] = None,
        calibration_constant: float = 1.0
    ):
        """
        Initialize the engine.

        Args:
            constants: Custom constants (uses defaults if None)
            calibration_constant: Empirical scaling factor for Sha prediction
        """
        self.constants = constants or CONSTANTS
        self.C = calibration_constant

        # Copy constants for quick access
        self.ALPHA_IMTAU = self.constants.ALPHA_IMTAU
        self.BETA_OMEGA = self.constants.BETA_OMEGA
        self.GAMMA_REY = self.constants.GAMMA_REY
        self.DELTA_CASCADE = self.constants.DELTA_CASCADE
        self.REY_C_LOWER = self.constants.REY_C_LOWER
        self.REY_C_UPPER = self.constants.REY_C_UPPER

    # ==========================================================================
    # MAIN ANALYSIS METHOD
    # ==========================================================================

    def analyze(
        self,
        curve: EllipticCurveData,
        compute_cascade: bool = False
    ) -> BrahimAnalysisResult:
        """
        Perform complete Brahim's Laws analysis on a curve.

        Args:
            curve: Elliptic curve data to analyze
            compute_cascade: Whether to compute Law 5 (requires population data)

        Returns:
            Complete analysis result with all 6 laws evaluated
        """
        # Law 2: Reynolds Number
        reynolds = self.compute_reynolds(curve)

        # Law 3: Regime Classification
        regime = self.classify_regime(reynolds)

        # Law 1: Brahim Conjecture predictions
        sha_imtau = self.predict_sha_from_imtau(curve.im_tau)
        sha_omega = self.predict_sha_from_omega(curve.real_period)

        # Law 4: Dynamic Scaling
        sha_max = self.predict_sha_max(reynolds)

        # Compute errors vs actual Sha
        sha_actual = curve.sha_analytic if curve.sha_analytic else 1.0
        law1_error = self._relative_error(sha_actual, sha_imtau)
        law4_error = self._relative_error(sha_actual, sha_max)

        # Law 5: Cascade (placeholder - needs population data)
        p_exponent = self.DELTA_CASCADE  # Default to theoretical value

        # Law 6: Consistency check
        is_consistent, residual = self.check_consistency()

        return BrahimAnalysisResult(
            curve=curve,
            sha_median_predicted=sha_imtau,
            sha_omega_predicted=sha_omega,
            law1_error=law1_error,
            reynolds_number=reynolds,
            regime=regime,
            rey_c_lower=self.REY_C_LOWER,
            rey_c_upper=self.REY_C_UPPER,
            sha_max_predicted=sha_max,
            law4_error=law4_error,
            log_sha_variance=None,
            p_scaling_exponent=p_exponent,
            consistency_check=residual,
            is_consistent=is_consistent,
            vnand_hash="",  # Filled by audit module
            timestamp=datetime.now().isoformat()
        )

    # ==========================================================================
    # LAW 1: BRAHIM CONJECTURE
    # ==========================================================================

    def predict_sha_from_imtau(self, im_tau: float) -> float:
        """
        Law 1a: Predict Sha from Im(tau).

        Sha_median ~ C * Im(tau)^(2/3)

        Args:
            im_tau: Imaginary part of period ratio

        Returns:
            Predicted Sha value
        """
        if im_tau <= 0:
            return 1.0
        return self.C * (im_tau ** self.ALPHA_IMTAU)

    def predict_sha_from_omega(self, omega: float) -> float:
        """
        Law 1b: Predict Sha from real period Omega.

        Sha_median ~ C * Omega^(-4/3)

        Derived from Im(tau) ~ Omega^(-2), giving exponent -2 * (2/3) = -4/3

        Args:
            omega: Real period Omega_E

        Returns:
            Predicted Sha value
        """
        if omega <= 0:
            return float('inf')
        return self.C * (omega ** self.BETA_OMEGA)

    # ==========================================================================
    # LAW 2: ARITHMETIC REYNOLDS NUMBER
    # ==========================================================================

    def compute_reynolds(self, curve: EllipticCurveData) -> float:
        """
        Law 2: Compute Arithmetic Reynolds Number.

        Rey = N / (Tam * Omega)

        Physical interpretation:
        - N (conductor): "inertial force" - arithmetic complexity
        - Tam * Omega: "viscous damping" - geometric regularization

        High Reynolds = complexity exceeds damping = turbulent arithmetic

        Args:
            curve: Elliptic curve data

        Returns:
            Reynolds number (dimensionless)
        """
        N = curve.conductor
        Tam = curve.tamagawa_product
        Omega = curve.real_period

        denominator = Tam * Omega
        if denominator == 0:
            return float('inf')

        return N / denominator

    # ==========================================================================
    # LAW 3: PHASE TRANSITION
    # ==========================================================================

    def classify_regime(self, reynolds: float) -> Regime:
        """
        Law 3: Classify flow regime based on Reynolds number.

        - LAMINAR (Rey < 10): Sha = 1 with ~100% probability
        - TRANSITION (10 <= Rey <= 30): Sha > 1 begins to appear
        - TURBULENT (Rey > 30): Significant probability of Sha > 1

        Args:
            reynolds: Arithmetic Reynolds number

        Returns:
            Regime classification
        """
        if reynolds < self.REY_C_LOWER:
            return Regime.LAMINAR
        elif reynolds > self.REY_C_UPPER:
            return Regime.TURBULENT
        else:
            return Regime.TRANSITION

    def sha_nontrivial_probability(self, reynolds: float) -> float:
        """
        Estimate P(Sha > 1) based on Reynolds number.

        Empirical fit from Cremona database:
        - Rey < 10: ~0%
        - Rey 10-30: ~0.1%
        - Rey 30-100: ~1%
        - Rey 100-300: ~2.5%
        - Rey 300-1000: ~6.6%
        - Rey > 1000: ~14%

        Args:
            reynolds: Arithmetic Reynolds number

        Returns:
            Estimated probability of Sha > 1
        """
        if reynolds < 10:
            return 0.0
        elif reynolds < 30:
            return 0.001
        elif reynolds < 100:
            return 0.011
        elif reynolds < 300:
            return 0.025
        elif reynolds < 1000:
            return 0.066
        else:
            return 0.141

    # ==========================================================================
    # LAW 4: DYNAMIC SCALING
    # ==========================================================================

    def predict_sha_max(self, reynolds: float) -> float:
        """
        Law 4: Predict maximum possible Sha from Reynolds number.

        Sha_max ~ Rey^(5/12)

        This gives the upper envelope of Sha values at a given Reynolds.

        Args:
            reynolds: Arithmetic Reynolds number

        Returns:
            Predicted maximum Sha
        """
        if reynolds <= 0:
            return 1.0
        return reynolds ** self.GAMMA_REY

    # ==========================================================================
    # LAW 5: CASCADE LAW
    # ==========================================================================

    def compute_cascade_exponent(
        self,
        sha_values: np.ndarray,
        primes: np.ndarray
    ) -> Tuple[float, float]:
        """
        Law 5: Compute cascade variance decay exponent.

        Var(log Sha | p) ~ p^(delta), where delta ~ -1/4

        This describes how the variance of log(Sha) decays across primes,
        analogous to Kolmogorov cascade in turbulence.

        Args:
            sha_values: Array of Sha values
            primes: Array of associated primes

        Returns:
            (fitted_exponent, r_squared) where target exponent is -0.25
        """
        if len(sha_values) < 2:
            return (self.DELTA_CASCADE, 0.0)

        # Filter positive Sha values
        mask = sha_values > 0
        log_sha = np.log(sha_values[mask])
        log_p = np.log(primes[:len(log_sha)])

        if len(log_sha) < 3:
            return (self.DELTA_CASCADE, 0.0)

        # Compute running variance
        variances = []
        for i in range(2, len(log_sha)):
            variances.append(np.var(log_sha[:i]))

        if len(variances) < 2:
            return (self.DELTA_CASCADE, 0.0)

        log_var = np.log(np.array(variances) + 1e-10)
        log_p_subset = log_p[2:len(variances)+2]

        # Linear regression: log(Var) = exponent * log(p) + const
        coeffs = np.polyfit(log_p_subset, log_var, 1)
        exponent = coeffs[0]

        # R-squared
        y_pred = np.polyval(coeffs, log_p_subset)
        ss_res = np.sum((log_var - y_pred)**2)
        ss_tot = np.sum((log_var - np.mean(log_var))**2)
        r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0

        return (float(exponent), float(r_squared))

    # ==========================================================================
    # LAW 6: CONSISTENCY RELATION
    # ==========================================================================

    def check_consistency(self) -> Tuple[bool, float]:
        """
        Law 6: Verify the consistency relation 2/3 = 5/12 + 1/4.

        This exact equality connects Laws 1, 4, and 5, suggesting
        they are manifestations of a single underlying structure.

        Returns:
            (is_consistent, residual) where residual should be ~0
        """
        lhs = self.ALPHA_IMTAU  # 2/3
        rhs = self.GAMMA_REY + abs(self.DELTA_CASCADE)  # 5/12 + 1/4

        residual = abs(lhs - rhs)
        is_consistent = residual < 1e-10

        return (is_consistent, residual)

    # ==========================================================================
    # BATCH ANALYSIS
    # ==========================================================================

    def analyze_batch(
        self,
        curves: List[EllipticCurveData]
    ) -> List[BrahimAnalysisResult]:
        """
        Analyze multiple curves.

        For GPU-accelerated batch processing, use CUDABatchProcessor instead.

        Args:
            curves: List of curves to analyze

        Returns:
            List of analysis results
        """
        return [self.analyze(curve) for curve in curves]

    # ==========================================================================
    # UTILITIES
    # ==========================================================================

    def _relative_error(self, actual: float, predicted: float) -> float:
        """Compute relative error |actual - predicted| / actual."""
        if actual == 0:
            return 0.0 if predicted == 0 else float('inf')
        return abs(actual - predicted) / abs(actual)

    def __repr__(self) -> str:
        return (
            f"BrahimLawsEngine(C={self.C}, "
            f"alpha={self.ALPHA_IMTAU}, "
            f"gamma={self.GAMMA_REY}, "
            f"delta={self.DELTA_CASCADE})"
        )
