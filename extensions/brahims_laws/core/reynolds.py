"""
Reynolds Number Analysis Module.

Specialized analysis of the Arithmetic Reynolds Number, which governs
the phase transition between laminar and turbulent arithmetic behavior.

Physical Analogy:
    In fluid dynamics, Re = rho*v*L/mu governs laminar-turbulent transition.
    In arithmetic, Rey = N/(Tam*Omega) plays an analogous role:
    - N (conductor) ~ inertial force
    - Tam*Omega ~ viscous damping
    - High Rey = turbulent arithmetic (Sha > 1 likely)

Author: Elias Oulad Brahim
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .constants import CONSTANTS
from ..models.curve_data import EllipticCurveData, Regime


@dataclass
class ReynoldsStatistics:
    """Statistics from Reynolds number analysis."""
    total_curves: int
    laminar_count: int
    transition_count: int
    turbulent_count: int
    laminar_percent: float
    transition_percent: float
    turbulent_percent: float
    mean_reynolds: float
    median_reynolds: float
    std_reynolds: float
    min_reynolds: float
    max_reynolds: float
    sha_nontrivial_by_regime: Dict[str, float]


class ReynoldsAnalyzer:
    """
    Specialized analysis of Arithmetic Reynolds Number.

    Provides methods for:
    - Single curve Reynolds computation
    - Batch vectorized computation
    - Regime classification
    - Statistical analysis
    - Phase transition visualization data

    Example:
        analyzer = ReynoldsAnalyzer()
        rey = analyzer.compute(N=37, Tam=1, Omega=0.7254)
        regime = analyzer.classify(rey)
        stats = analyzer.statistics(curves)
    """

    # Critical thresholds
    REY_CRITICAL_LOWER: float = CONSTANTS.REY_C_LOWER  # 10
    REY_CRITICAL_UPPER: float = CONSTANTS.REY_C_UPPER  # 30

    def __init__(
        self,
        critical_lower: Optional[float] = None,
        critical_upper: Optional[float] = None
    ):
        """
        Initialize analyzer with optional custom thresholds.

        Args:
            critical_lower: Lower critical Reynolds (default 10)
            critical_upper: Upper critical Reynolds (default 30)
        """
        if critical_lower is not None:
            self.REY_CRITICAL_LOWER = critical_lower
        if critical_upper is not None:
            self.REY_CRITICAL_UPPER = critical_upper

    # ==========================================================================
    # CORE COMPUTATION
    # ==========================================================================

    @staticmethod
    def compute(N: int, Tam: int, Omega: float) -> float:
        """
        Compute Reynolds number from components.

        Rey = N / (Tam * Omega)

        Args:
            N: Conductor
            Tam: Tamagawa product
            Omega: Real period

        Returns:
            Reynolds number
        """
        denominator = Tam * Omega
        if denominator == 0:
            return float('inf')
        return N / denominator

    def compute_from_curve(self, curve: EllipticCurveData) -> float:
        """Compute Reynolds from curve object."""
        return self.compute(
            curve.conductor,
            curve.tamagawa_product,
            curve.real_period
        )

    # ==========================================================================
    # CLASSIFICATION
    # ==========================================================================

    def classify(self, reynolds: float) -> Regime:
        """
        Classify Reynolds number into flow regime.

        Args:
            reynolds: Arithmetic Reynolds number

        Returns:
            LAMINAR, TRANSITION, or TURBULENT
        """
        if reynolds < self.REY_CRITICAL_LOWER:
            return Regime.LAMINAR
        elif reynolds > self.REY_CRITICAL_UPPER:
            return Regime.TURBULENT
        return Regime.TRANSITION

    # ==========================================================================
    # BATCH COMPUTATION (VECTORIZED)
    # ==========================================================================

    def batch_compute(self, curves: List[EllipticCurveData]) -> np.ndarray:
        """
        Vectorized Reynolds computation for batch of curves.

        Args:
            curves: List of curves

        Returns:
            NumPy array of Reynolds numbers
        """
        N = np.array([c.conductor for c in curves], dtype=np.float64)
        Tam = np.array([c.tamagawa_product for c in curves], dtype=np.float64)
        Omega = np.array([c.real_period for c in curves], dtype=np.float64)

        denominator = Tam * Omega
        # Handle division by zero
        denominator = np.where(denominator == 0, np.inf, denominator)

        return N / denominator

    def batch_classify(self, reynolds_values: np.ndarray) -> np.ndarray:
        """
        Vectorized regime classification.

        Args:
            reynolds_values: Array of Reynolds numbers

        Returns:
            Array of regime codes (0=laminar, 1=transition, 2=turbulent)
        """
        regimes = np.ones_like(reynolds_values, dtype=np.int32)  # Default: transition
        regimes[reynolds_values < self.REY_CRITICAL_LOWER] = 0  # Laminar
        regimes[reynolds_values > self.REY_CRITICAL_UPPER] = 2  # Turbulent
        return regimes

    # ==========================================================================
    # STATISTICAL ANALYSIS
    # ==========================================================================

    def statistics(self, curves: List[EllipticCurveData]) -> ReynoldsStatistics:
        """
        Compute comprehensive Reynolds statistics for a population.

        Args:
            curves: List of curves to analyze

        Returns:
            ReynoldsStatistics dataclass with all metrics
        """
        if not curves:
            return ReynoldsStatistics(
                total_curves=0,
                laminar_count=0, transition_count=0, turbulent_count=0,
                laminar_percent=0, transition_percent=0, turbulent_percent=0,
                mean_reynolds=0, median_reynolds=0, std_reynolds=0,
                min_reynolds=0, max_reynolds=0,
                sha_nontrivial_by_regime={}
            )

        reynolds_values = self.batch_compute(curves)
        regimes = self.batch_classify(reynolds_values)

        # Filter out infinities for statistics
        finite_mask = np.isfinite(reynolds_values)
        finite_reynolds = reynolds_values[finite_mask]

        n = len(curves)

        # Regime counts
        laminar_count = int(np.sum(regimes == 0))
        transition_count = int(np.sum(regimes == 1))
        turbulent_count = int(np.sum(regimes == 2))

        # Sha > 1 rate by regime
        sha_nontrivial = {}
        for regime_code, regime_name in [(0, 'laminar'), (1, 'transition'), (2, 'turbulent')]:
            regime_mask = regimes == regime_code
            if np.sum(regime_mask) > 0:
                regime_curves = [c for c, m in zip(curves, regime_mask) if m]
                nontrivial = sum(1 for c in regime_curves if c.sha_analytic and c.sha_analytic > 1)
                sha_nontrivial[regime_name] = nontrivial / len(regime_curves) if regime_curves else 0.0
            else:
                sha_nontrivial[regime_name] = 0.0

        return ReynoldsStatistics(
            total_curves=n,
            laminar_count=laminar_count,
            transition_count=transition_count,
            turbulent_count=turbulent_count,
            laminar_percent=laminar_count / n * 100 if n > 0 else 0,
            transition_percent=transition_count / n * 100 if n > 0 else 0,
            turbulent_percent=turbulent_count / n * 100 if n > 0 else 0,
            mean_reynolds=float(np.mean(finite_reynolds)) if len(finite_reynolds) > 0 else 0,
            median_reynolds=float(np.median(finite_reynolds)) if len(finite_reynolds) > 0 else 0,
            std_reynolds=float(np.std(finite_reynolds)) if len(finite_reynolds) > 0 else 0,
            min_reynolds=float(np.min(finite_reynolds)) if len(finite_reynolds) > 0 else 0,
            max_reynolds=float(np.max(finite_reynolds)) if len(finite_reynolds) > 0 else 0,
            sha_nontrivial_by_regime=sha_nontrivial
        )

    # ==========================================================================
    # PHASE TRANSITION ANALYSIS
    # ==========================================================================

    def phase_transition_bins(
        self,
        curves: List[EllipticCurveData],
        bins: Optional[List[Tuple[float, float]]] = None
    ) -> Dict[str, Dict]:
        """
        Compute Sha > 1 probability by Reynolds bins.

        This recreates Table 2 from the Brahim's Laws paper.

        Args:
            curves: List of curves
            bins: Custom bins as [(low, high), ...] or None for defaults

        Returns:
            Dictionary with bin statistics
        """
        if bins is None:
            bins = [
                (0, 3),
                (3, 10),
                (10, 30),
                (30, 100),
                (100, 300),
                (300, 1000),
                (1000, float('inf'))
            ]

        reynolds_values = self.batch_compute(curves)
        results = {}

        for low, high in bins:
            bin_name = f"{low}-{high}" if high != float('inf') else f">{low}"

            # Find curves in this bin
            if high == float('inf'):
                mask = reynolds_values >= low
            else:
                mask = (reynolds_values >= low) & (reynolds_values < high)

            bin_curves = [c for c, m in zip(curves, mask) if m]
            count = len(bin_curves)

            if count > 0:
                sha_nontrivial = sum(1 for c in bin_curves if c.sha_analytic and c.sha_analytic > 1)
                prob = sha_nontrivial / count
            else:
                sha_nontrivial = 0
                prob = 0.0

            results[bin_name] = {
                "count": count,
                "sha_nontrivial": sha_nontrivial,
                "probability": prob,
                "range": (low, high)
            }

        return results

    # ==========================================================================
    # UTILITIES
    # ==========================================================================

    def regime_color(self, regime: Regime) -> str:
        """Get display color for regime."""
        colors = {
            Regime.LAMINAR: "green",
            Regime.TRANSITION: "yellow",
            Regime.TURBULENT: "red"
        }
        return colors.get(regime, "white")

    def __repr__(self) -> str:
        return (
            f"ReynoldsAnalyzer(critical_range="
            f"[{self.REY_CRITICAL_LOWER}, {self.REY_CRITICAL_UPPER}])"
        )
