"""
Feature Extractor for Elliptic Curve ML Models.

Transforms curve invariants into feature vectors for Sha prediction.

Key insight from Brahim's Laws:
- Sha scales with Im(tau)^(2/3) and Omega^(-4/3)
- Reynolds number Rey = N/(Tam*Omega) predicts regime
- Phase transitions occur at Rey ~ 10-30

We engineer features to capture these relationships.

Author: Elias Oulad Brahim
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from ..models.curve_data import EllipticCurveData
from ..core.constants import CONSTANTS


@dataclass
class CurveFeatures:
    """Feature vector for a single curve."""

    # Raw invariants
    conductor: float
    rank: int
    torsion_order: int
    tamagawa_product: float
    real_period: float
    im_tau: float

    # Derived features (Brahim's Laws inspired)
    log_conductor: float
    log_period: float
    log_im_tau: float
    reynolds_number: float
    log_reynolds: float

    # Scaling law features
    im_tau_scaled: float      # Im(tau)^(2/3)
    omega_scaled: float       # Omega^(-4/3)
    reynolds_scaled: float    # Rey^(5/12)

    # Regime indicators
    is_laminar: int           # Rey < 10
    is_transition: int        # 10 <= Rey <= 30
    is_turbulent: int         # Rey > 30

    # Interaction features
    conductor_period_ratio: float
    rank_reynolds_interaction: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML models."""
        return np.array([
            self.log_conductor,
            self.rank,
            self.torsion_order,
            np.log1p(self.tamagawa_product),
            self.log_period,
            self.log_im_tau,
            self.log_reynolds,
            self.im_tau_scaled,
            self.omega_scaled,
            self.reynolds_scaled,
            self.is_laminar,
            self.is_transition,
            self.is_turbulent,
            self.conductor_period_ratio,
            self.rank_reynolds_interaction,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        """Return feature names for interpretability."""
        return [
            "log_conductor",
            "rank",
            "torsion_order",
            "log_tamagawa",
            "log_period",
            "log_im_tau",
            "log_reynolds",
            "im_tau_scaled",
            "omega_scaled",
            "reynolds_scaled",
            "is_laminar",
            "is_transition",
            "is_turbulent",
            "conductor_period_ratio",
            "rank_reynolds_interaction",
        ]


class CurveFeatureExtractor:
    """
    Extract ML features from elliptic curve data.

    Features are designed based on Brahim's Laws scaling relationships:
    - Log transforms for multiplicative quantities
    - Scaling exponents (2/3, -4/3, 5/12) applied
    - Regime classification as categorical

    Example:
        extractor = CurveFeatureExtractor()
        features = extractor.extract(curve)
        X = extractor.extract_batch(curves)
    """

    def __init__(
        self,
        include_raw: bool = True,
        include_scaled: bool = True,
        include_regime: bool = True,
        include_interactions: bool = True,
        eps: float = 1e-10
    ):
        """
        Initialize feature extractor.

        Args:
            include_raw: Include raw invariants (log-transformed)
            include_scaled: Include Brahim's Laws scaled features
            include_regime: Include regime indicator features
            include_interactions: Include interaction features
            eps: Small constant for numerical stability
        """
        self.include_raw = include_raw
        self.include_scaled = include_scaled
        self.include_regime = include_regime
        self.include_interactions = include_interactions
        self.eps = eps

        # Cache constants
        self.alpha = CONSTANTS.ALPHA_IMTAU      # 2/3
        self.beta = CONSTANTS.BETA_OMEGA        # -4/3
        self.gamma = CONSTANTS.GAMMA_REY        # 5/12
        self.rey_c_lower = CONSTANTS.REY_C_LOWER
        self.rey_c_upper = CONSTANTS.REY_C_UPPER

    def extract(self, curve: EllipticCurveData) -> CurveFeatures:
        """
        Extract features from a single curve.

        Args:
            curve: Elliptic curve data

        Returns:
            CurveFeatures dataclass
        """
        # Safe values
        conductor = max(curve.conductor, 1)
        period = max(curve.real_period, self.eps)
        im_tau = max(curve.im_tau, self.eps)
        tamagawa = max(curve.tamagawa_product, 1)

        # Reynolds number
        reynolds = conductor / (tamagawa * period + self.eps)

        # Log transforms
        log_conductor = np.log(conductor)
        log_period = np.log(period)
        log_im_tau = np.log(im_tau)
        log_reynolds = np.log(max(reynolds, self.eps))

        # Scaled features (Brahim's Laws exponents)
        im_tau_scaled = np.power(im_tau, self.alpha)          # Im(tau)^(2/3)
        omega_scaled = np.power(period, self.beta)            # Omega^(-4/3)
        reynolds_scaled = np.power(max(reynolds, self.eps), self.gamma)  # Rey^(5/12)

        # Regime indicators
        is_laminar = 1 if reynolds < self.rey_c_lower else 0
        is_turbulent = 1 if reynolds > self.rey_c_upper else 0
        is_transition = 1 if not (is_laminar or is_turbulent) else 0

        # Interaction features
        conductor_period_ratio = log_conductor - log_period
        rank_reynolds_interaction = curve.rank * log_reynolds

        return CurveFeatures(
            conductor=conductor,
            rank=curve.rank,
            torsion_order=curve.torsion_order,
            tamagawa_product=tamagawa,
            real_period=period,
            im_tau=im_tau,
            log_conductor=log_conductor,
            log_period=log_period,
            log_im_tau=log_im_tau,
            reynolds_number=reynolds,
            log_reynolds=log_reynolds,
            im_tau_scaled=im_tau_scaled,
            omega_scaled=omega_scaled,
            reynolds_scaled=reynolds_scaled,
            is_laminar=is_laminar,
            is_transition=is_transition,
            is_turbulent=is_turbulent,
            conductor_period_ratio=conductor_period_ratio,
            rank_reynolds_interaction=rank_reynolds_interaction,
        )

    def extract_batch(
        self,
        curves: List[EllipticCurveData]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from multiple curves.

        Args:
            curves: List of curve data

        Returns:
            (X, feature_names) where X is shape (n_curves, n_features)
        """
        features_list = [self.extract(c) for c in curves]
        X = np.vstack([f.to_array() for f in features_list])
        return X, CurveFeatures.feature_names()

    def extract_targets(
        self,
        curves: List[EllipticCurveData],
        log_transform: bool = True
    ) -> np.ndarray:
        """
        Extract Sha values as prediction targets.

        Args:
            curves: List of curves
            log_transform: Whether to log-transform Sha (recommended)

        Returns:
            Array of Sha values
        """
        sha_values = np.array([
            c.sha_analytic if c.sha_analytic else 1
            for c in curves
        ], dtype=np.float32)

        if log_transform:
            return np.log(sha_values + self.eps)
        return sha_values

    def extract_dataset(
        self,
        curves: List[EllipticCurveData],
        log_sha: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract complete dataset for training.

        Args:
            curves: List of curves with known Sha
            log_sha: Log-transform target

        Returns:
            (X, y, feature_names)
        """
        X, names = self.extract_batch(curves)
        y = self.extract_targets(curves, log_transform=log_sha)
        return X, y, names

    def get_feature_importance_baseline(self) -> Dict[str, str]:
        """
        Get expected feature importance based on Brahim's Laws.

        Returns:
            Dict mapping feature names to their theoretical importance
        """
        return {
            "im_tau_scaled": "HIGH - Law 1: Sha ~ Im(tau)^(2/3)",
            "omega_scaled": "HIGH - Law 1: Sha ~ Omega^(-4/3)",
            "reynolds_scaled": "HIGH - Law 4: Sha_max ~ Rey^(5/12)",
            "log_reynolds": "HIGH - Primary predictor of regime",
            "is_turbulent": "MEDIUM - Turbulent regime has higher Sha",
            "is_laminar": "MEDIUM - Laminar regime has Sha=1",
            "log_conductor": "MEDIUM - Contributes to Reynolds",
            "rank": "MEDIUM - Rank affects BSD formula",
            "log_period": "MEDIUM - Part of Reynolds denominator",
            "torsion_order": "LOW - Indirect effect",
            "log_tamagawa": "LOW - Part of Reynolds denominator",
            "is_transition": "LOW - Intermediate regime",
            "conductor_period_ratio": "LOW - Interaction term",
            "rank_reynolds_interaction": "LOW - Interaction term",
            "log_im_tau": "LOW - Redundant with im_tau_scaled",
        }
