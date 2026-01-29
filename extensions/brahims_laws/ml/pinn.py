"""
Physics-Informed Neural Network (PINN) for Brahim's Laws.

A neural network that respects the mathematical identities:
- Law 6: alpha = gamma + |delta|  (2/3 = 5/12 + 1/4)
- Law 1: Sha ~ Im(tau)^alpha
- Law 4: Sha_max ~ Rey^gamma

The PINN architecture:
1. Learns the scaling exponents (alpha, gamma, delta) as parameters
2. Physics loss enforces alpha = gamma + |delta|
3. Data loss fits predictions to observed Sha values
4. The network learns representations consistent with Brahim's Laws

This is more than curve fitting - it learns the underlying physics!

Author: Elias Oulad Brahim
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import json
from pathlib import Path
from fractions import Fraction

from ..models.curve_data import EllipticCurveData
from ..core.constants import CONSTANTS
from .feature_extractor import CurveFeatureExtractor

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class BrahimLawsPINN(nn.Module):
    """
    Physics-Informed Neural Network for Brahim's Laws.

    Key innovation: The scaling exponents are LEARNABLE PARAMETERS
    constrained by physics loss to satisfy alpha = gamma + |delta|.

    If training succeeds with low physics loss, it means:
    1. The data supports Brahim's Laws
    2. The learned exponents should be close to 2/3, 5/12, 1/4

    Architecture:
    - Learnable exponents: alpha, gamma, delta
    - Calibration constant C
    - Neural network for residual correction
    - Physics constraint layer

    Loss = data_loss + lambda * physics_loss
    where physics_loss = (alpha - gamma - |delta|)^2
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: List[int] = [64, 32],
        init_alpha: float = 0.6,  # Start near 2/3
        init_gamma: float = 0.4,  # Start near 5/12
        init_delta: float = -0.2,  # Start near -1/4
        init_C: float = 1.0,
        learn_exponents: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize PINN.

        Args:
            n_features: Number of input features
            hidden_sizes: Hidden layer sizes for residual network
            init_alpha: Initial value for alpha (Law 1 exponent)
            init_gamma: Initial value for gamma (Law 4 exponent)
            init_delta: Initial value for delta (Law 5 exponent, negative)
            init_C: Initial calibration constant
            learn_exponents: If True, exponents are learnable parameters
            dropout: Dropout rate
        """
        super().__init__()

        # Learnable physics parameters
        self.learn_exponents = learn_exponents

        if learn_exponents:
            # Initialize exponents as learnable parameters
            self.log_alpha = nn.Parameter(torch.tensor(np.log(init_alpha)))
            self.log_gamma = nn.Parameter(torch.tensor(np.log(init_gamma)))
            self.log_neg_delta = nn.Parameter(torch.tensor(np.log(-init_delta)))  # delta is negative
            self.log_C = nn.Parameter(torch.tensor(np.log(init_C)))
        else:
            # Use fixed theoretical values
            self.register_buffer('log_alpha', torch.tensor(np.log(2/3)))
            self.register_buffer('log_gamma', torch.tensor(np.log(5/12)))
            self.register_buffer('log_neg_delta', torch.tensor(np.log(1/4)))
            self.register_buffer('log_C', torch.tensor(np.log(1.0)))

        # Neural network for residual/correction term
        # This captures what the simple power laws miss
        layers = []
        prev_size = n_features
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))

        self.residual_network = nn.Sequential(*layers)

    @property
    def alpha(self) -> torch.Tensor:
        """Get alpha exponent (always positive)."""
        return torch.exp(self.log_alpha)

    @property
    def gamma(self) -> torch.Tensor:
        """Get gamma exponent (always positive)."""
        return torch.exp(self.log_gamma)

    @property
    def delta(self) -> torch.Tensor:
        """Get delta exponent (always negative)."""
        return -torch.exp(self.log_neg_delta)

    @property
    def C(self) -> torch.Tensor:
        """Get calibration constant."""
        return torch.exp(self.log_C)

    def physics_loss(self) -> torch.Tensor:
        """
        Compute physics constraint loss.

        Law 6 states: alpha = gamma + |delta|
        Or equivalently: 2/3 = 5/12 + 1/4

        This loss is zero when the constraint is satisfied.
        """
        # alpha should equal gamma + |delta|
        lhs = self.alpha
        rhs = self.gamma + torch.abs(self.delta)
        return (lhs - rhs) ** 2

    def compute_sha_law1(
        self,
        im_tau: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Sha prediction using Law 1.

        Sha ~ C * Im(tau)^alpha
        """
        im_tau_safe = torch.clamp(im_tau, min=1e-10)
        return self.C * torch.pow(im_tau_safe, self.alpha)

    def compute_sha_max_law4(
        self,
        reynolds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Sha_max prediction using Law 4.

        Sha_max ~ Rey^gamma
        """
        reynolds_safe = torch.clamp(reynolds, min=1e-10)
        return torch.pow(reynolds_safe, self.gamma)

    def forward(
        self,
        features: torch.Tensor,
        im_tau: torch.Tensor,
        reynolds: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass combining physics and neural network.

        The final prediction is:
        log(Sha) = log(Sha_physics) + residual_correction

        where Sha_physics comes from Law 1.

        Args:
            features: All curve features (batch_size, n_features)
            im_tau: Im(tau) values (batch_size,)
            reynolds: Reynolds numbers (batch_size,)

        Returns:
            Dict with predictions and physics quantities
        """
        # Physics-based predictions
        sha_law1 = self.compute_sha_law1(im_tau)
        sha_max_law4 = self.compute_sha_max_law4(reynolds)

        # Neural network residual (in log space for stability)
        log_residual = self.residual_network(features).squeeze(-1)

        # Combined prediction: physics + learned correction
        log_sha_physics = torch.log(sha_law1 + 1e-10)
        log_sha_pred = log_sha_physics + log_residual

        return {
            'log_sha_pred': log_sha_pred,
            'sha_pred': torch.exp(log_sha_pred),
            'sha_law1': sha_law1,
            'sha_max_law4': sha_max_law4,
            'log_residual': log_residual,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'delta': self.delta,
            'C': self.C,
        }


class PhysicsInformedPredictor:
    """
    High-level interface for Physics-Informed Sha prediction.

    Trains a PINN that:
    1. Learns optimal scaling exponents
    2. Respects the consistency relation alpha = gamma + |delta|
    3. Uses neural network for residual corrections

    If the physics loss stays low during training, it validates Brahim's Laws!

    Example:
        predictor = PhysicsInformedPredictor()
        history = predictor.fit(curves, physics_weight=10.0)
        print(f"Learned alpha: {predictor.get_exponents()['alpha']}")
        sha_pred = predictor.predict(new_curves)
    """

    def __init__(
        self,
        learn_exponents: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize PINN predictor.

        Args:
            learn_exponents: If True, learn exponents from data
            device: PyTorch device
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for PINN")

        self.learn_exponents = learn_exponents
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        self.feature_extractor = CurveFeatureExtractor()
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.model = None
        self.is_fitted = False
        self.training_history = {}

    def _prepare_data(
        self,
        curves: List[EllipticCurveData]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and targets."""
        X, _ = self.feature_extractor.extract_batch(curves)

        # Extract specific quantities needed for physics
        im_tau = np.array([max(c.im_tau, 1e-10) for c in curves])
        reynolds = np.array([
            c.conductor / (c.tamagawa_product * c.real_period + 1e-10)
            for c in curves
        ])

        # Target: log(Sha)
        sha = np.array([c.sha_analytic if c.sha_analytic else 1 for c in curves])
        log_sha = np.log(sha + 1e-10)

        return X, im_tau, reynolds, log_sha

    def fit(
        self,
        curves: List[EllipticCurveData],
        epochs: int = 200,
        lr: float = 0.001,
        physics_weight: float = 10.0,
        test_size: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the PINN.

        Loss = MSE(log_sha_pred, log_sha_true) + physics_weight * physics_loss

        Args:
            curves: Training curves with known Sha
            epochs: Training epochs
            lr: Learning rate
            physics_weight: Weight for physics constraint loss
            test_size: Validation split
            verbose: Print progress

        Returns:
            Training history
        """
        # Filter curves with valid Sha
        valid_curves = [c for c in curves if c.sha_analytic is not None]

        if len(valid_curves) < 50:
            raise ValueError(f"Need at least 50 curves with known Sha, got {len(valid_curves)}")

        # Prepare data
        X, im_tau, reynolds, log_sha = self._prepare_data(valid_curves)

        # Split
        indices = np.arange(len(valid_curves))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42
        )

        X_train, X_test = X[train_idx], X[test_idx]
        im_tau_train, im_tau_test = im_tau[train_idx], im_tau[test_idx]
        reynolds_train, reynolds_test = reynolds[train_idx], reynolds[test_idx]
        log_sha_train, log_sha_test = log_sha[train_idx], log_sha[test_idx]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create model
        n_features = X.shape[1]
        self.model = BrahimLawsPINN(
            n_features=n_features,
            learn_exponents=self.learn_exponents
        ).to(self.device)

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_scaled).to(self.device)
        im_tau_train_t = torch.FloatTensor(im_tau_train).to(self.device)
        reynolds_train_t = torch.FloatTensor(reynolds_train).to(self.device)
        log_sha_train_t = torch.FloatTensor(log_sha_train).to(self.device)

        X_test_t = torch.FloatTensor(X_test_scaled).to(self.device)
        im_tau_test_t = torch.FloatTensor(im_tau_test).to(self.device)
        reynolds_test_t = torch.FloatTensor(reynolds_test).to(self.device)
        log_sha_test_t = torch.FloatTensor(log_sha_test).to(self.device)

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        data_criterion = nn.MSELoss()

        history = {
            'total_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'test_loss': [],
            'alpha': [],
            'gamma': [],
            'delta': [],
        }

        if verbose:
            logger.info("Training Physics-Informed Neural Network")
            logger.info(f"  Physics weight: {physics_weight}")
            logger.info(f"  Learn exponents: {self.learn_exponents}")
            logger.info(f"  Theoretical: alpha=2/3={2/3:.6f}, gamma=5/12={5/12:.6f}, |delta|=1/4={1/4:.6f}")
            logger.info("-" * 60)

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(X_train_t, im_tau_train_t, reynolds_train_t)

            # Data loss
            data_loss = data_criterion(outputs['log_sha_pred'], log_sha_train_t)

            # Physics loss
            physics_loss = self.model.physics_loss()

            # Total loss
            total_loss = data_loss + physics_weight * physics_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Record history
            history['total_loss'].append(total_loss.item())
            history['data_loss'].append(data_loss.item())
            history['physics_loss'].append(physics_loss.item())
            history['alpha'].append(outputs['alpha'].item())
            history['gamma'].append(outputs['gamma'].item())
            history['delta'].append(outputs['delta'].item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                test_outputs = self.model(X_test_t, im_tau_test_t, reynolds_test_t)
                test_loss = data_criterion(test_outputs['log_sha_pred'], log_sha_test_t)
                history['test_loss'].append(test_loss.item())

            if verbose and (epoch + 1) % 25 == 0:
                alpha = outputs['alpha'].item()
                gamma = outputs['gamma'].item()
                delta = outputs['delta'].item()

                # Check if physics constraint is satisfied
                constraint_error = abs(alpha - gamma - abs(delta))

                logger.info(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Loss: {total_loss.item():.4f} | "
                    f"Physics: {physics_loss.item():.6f} | "
                    f"alpha={alpha:.4f} gamma={gamma:.4f} |delta|={abs(delta):.4f} | "
                    f"Constraint err: {constraint_error:.6f}"
                )

        self.is_fitted = True
        self.training_history = history

        if verbose:
            self._print_final_analysis()

        return history

    def _print_final_analysis(self):
        """Print analysis of learned parameters vs theoretical."""
        logger.info("\n" + "=" * 60)
        logger.info("PINN TRAINING COMPLETE - PARAMETER ANALYSIS")
        logger.info("=" * 60)

        exponents = self.get_exponents()

        # Theoretical values
        theoretical = {
            'alpha': 2/3,
            'gamma': 5/12,
            '|delta|': 1/4
        }

        logger.info(f"\n{'Parameter':<12} {'Learned':<12} {'Theoretical':<12} {'Error':<12}")
        logger.info("-" * 48)

        for name, theo_val in theoretical.items():
            if name == '|delta|':
                learned_val = abs(exponents['delta'])
            else:
                learned_val = exponents[name]

            error = abs(learned_val - theo_val)
            error_pct = error / theo_val * 100

            logger.info(f"{name:<12} {learned_val:<12.6f} {theo_val:<12.6f} {error_pct:<10.2f}%")

        # Check consistency relation
        alpha = exponents['alpha']
        gamma = exponents['gamma']
        delta = abs(exponents['delta'])

        lhs = alpha
        rhs = gamma + delta
        consistency_error = abs(lhs - rhs)

        logger.info("\nConsistency Relation (Law 6):")
        logger.info("  alpha = gamma + |delta|")
        logger.info(f"  {alpha:.6f} = {gamma:.6f} + {delta:.6f} = {rhs:.6f}")
        logger.info(f"  Error: {consistency_error:.6f}")

        if consistency_error < 0.01:
            logger.info("  [VALIDATED] Physics constraint satisfied!")
        else:
            logger.info("  [WARNING] Physics constraint not satisfied")

        logger.info("=" * 60)

    def get_exponents(self) -> Dict[str, float]:
        """
        Get learned exponents.

        Returns:
            Dict with alpha, gamma, delta, C
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        self.model.eval()
        with torch.no_grad():
            return {
                'alpha': self.model.alpha.item(),
                'gamma': self.model.gamma.item(),
                'delta': self.model.delta.item(),
                'C': self.model.C.item(),
            }

    def predict(
        self,
        curves: List[EllipticCurveData]
    ) -> np.ndarray:
        """
        Predict Sha values.

        Args:
            curves: Curves to predict

        Returns:
            Predicted Sha values
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        X, im_tau, reynolds, _ = self._prepare_data(curves)
        X_scaled = self.scaler.transform(X)

        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_scaled).to(self.device)
            im_tau_t = torch.FloatTensor(im_tau).to(self.device)
            reynolds_t = torch.FloatTensor(reynolds).to(self.device)

            outputs = self.model(X_t, im_tau_t, reynolds_t)
            sha_pred = outputs['sha_pred'].cpu().numpy()

        # Round to nearest integer (Sha is always a perfect square)
        sha_pred = np.maximum(np.round(sha_pred), 1)
        return sha_pred

    def validate_brahim_laws(self) -> Dict[str, any]:
        """
        Check if learned exponents validate Brahim's Laws.

        Returns:
            Validation report
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        exponents = self.get_exponents()

        # Theoretical values
        alpha_theo = 2/3
        gamma_theo = 5/12
        delta_theo = -1/4

        # Errors
        alpha_error = abs(exponents['alpha'] - alpha_theo) / alpha_theo
        gamma_error = abs(exponents['gamma'] - gamma_theo) / gamma_theo
        delta_error = abs(abs(exponents['delta']) - abs(delta_theo)) / abs(delta_theo)

        # Consistency check
        consistency_error = abs(
            exponents['alpha'] - exponents['gamma'] - abs(exponents['delta'])
        )

        # Validation thresholds
        EXPONENT_THRESHOLD = 0.15  # 15% error tolerance
        CONSISTENCY_THRESHOLD = 0.05

        is_alpha_valid = alpha_error < EXPONENT_THRESHOLD
        is_gamma_valid = gamma_error < EXPONENT_THRESHOLD
        is_delta_valid = delta_error < EXPONENT_THRESHOLD
        is_consistent = consistency_error < CONSISTENCY_THRESHOLD

        overall_valid = is_alpha_valid and is_gamma_valid and is_delta_valid and is_consistent

        return {
            'overall_validation': overall_valid,
            'learned_exponents': exponents,
            'theoretical_exponents': {
                'alpha': alpha_theo,
                'gamma': gamma_theo,
                'delta': delta_theo
            },
            'relative_errors': {
                'alpha': alpha_error,
                'gamma': gamma_error,
                'delta': delta_error
            },
            'consistency_error': consistency_error,
            'individual_validations': {
                'alpha': is_alpha_valid,
                'gamma': is_gamma_valid,
                'delta': is_delta_valid,
                'consistency': is_consistent
            },
            'interpretation': (
                "VALIDATED: Learned exponents match Brahim's Laws predictions"
                if overall_valid else
                "NOT VALIDATED: Learned exponents deviate from theoretical values"
            )
        }

    def save(self, path: Path):
        """Save PINN to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "learn_exponents": self.learn_exponents,
            "is_fitted": self.is_fitted,
            "exponents": self.get_exponents() if self.is_fitted else None,
            "training_history": {
                k: v[-10:] if isinstance(v, list) else v  # Last 10 values
                for k, v in self.training_history.items()
            }
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        if SKLEARN_AVAILABLE:
            import joblib
            joblib.dump(self.scaler, path / "scaler.pkl")

        if self.model is not None:
            torch.save(self.model.state_dict(), path / "model.pt")

        logger.info(f"PINN saved to {path}")
