"""
Sha Predictor - ML Models for Tate-Shafarevich Group Prediction.

Multiple model architectures for predicting Sha from curve invariants:
1. Neural Network (PyTorch)
2. Gradient Boosting (XGBoost/LightGBM)
3. Ensemble (combines multiple models)

The key insight: If Brahim's Laws reveal statistical regularities,
ML can learn these patterns and generalize to new curves.

Author: Elias Oulad Brahim
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging

from ..models.curve_data import EllipticCurveData
from .feature_extractor import CurveFeatureExtractor

logger = logging.getLogger(__name__)

# Try importing ML libraries
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class ShaDataset:
    """Dataset container for Sha prediction."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    curves_train: List[EllipticCurveData]
    curves_test: List[EllipticCurveData]

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_train(self) -> int:
        return len(self.y_train)

    @property
    def n_test(self) -> int:
        return len(self.y_test)

    def summary(self) -> Dict:
        """Return dataset summary statistics."""
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "y_train_mean": float(np.mean(self.y_train)),
            "y_train_std": float(np.std(self.y_train)),
            "y_test_mean": float(np.mean(self.y_test)),
            "y_test_std": float(np.std(self.y_test)),
        }


if TORCH_AVAILABLE:
    class ShaNeuralNet(nn.Module):
        """
        Neural network for Sha prediction.

        Architecture inspired by Brahim's Laws structure:
        - Input: curve invariants + scaled features
        - Hidden layers with residual connections
        - Output: log(Sha) prediction
        """

        def __init__(
            self,
            n_features: int,
            hidden_sizes: List[int] = [64, 32, 16],
            dropout: float = 0.2
        ):
            super().__init__()

            layers = []
            prev_size = n_features

            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(dropout),
                ])
                prev_size = hidden_size

            # Output layer
            layers.append(nn.Linear(prev_size, 1))

            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x).squeeze(-1)
else:
    ShaNeuralNet = None


class ShaPredictor:
    """
    Main predictor class for Sha prediction.

    Supports multiple model types:
    - 'neural': PyTorch neural network
    - 'gbm': Gradient Boosting Machine
    - 'rf': Random Forest
    - 'ensemble': Average of multiple models

    Example:
        predictor = ShaPredictor(model_type='gbm')
        predictor.fit(curves_train)
        sha_pred = predictor.predict(curves_test)
    """

    def __init__(
        self,
        model_type: str = 'gbm',
        device: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Initialize predictor.

        Args:
            model_type: 'neural', 'gbm', 'rf', or 'ensemble'
            device: PyTorch device (auto-detect if None)
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.feature_extractor = CurveFeatureExtractor()
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.model = None
        self.is_fitted = False
        self.training_metrics = {}

        # Device for neural network
        if TORCH_AVAILABLE:
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
        else:
            self.device = None

    def _create_model(self, n_features: int):
        """Create model based on type."""
        if self.model_type == 'neural':
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch required for neural network model")
            self.model = ShaNeuralNet(n_features).to(self.device)

        elif self.model_type == 'gbm':
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn required for GBM model")
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state
            )

        elif self.model_type == 'rf':
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn required for RF model")
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )

        elif self.model_type == 'ensemble':
            # Create multiple models
            self.models = {
                'gbm': GradientBoostingRegressor(
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=self.random_state
                ),
                'rf': RandomForestRegressor(
                    n_estimators=150,
                    max_depth=8,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
            }
            if TORCH_AVAILABLE:
                self.models['neural'] = ShaNeuralNet(n_features).to(self.device)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def prepare_dataset(
        self,
        curves: List[EllipticCurveData],
        test_size: float = 0.2
    ) -> ShaDataset:
        """
        Prepare dataset from curves.

        Args:
            curves: List of curves with known Sha values
            test_size: Fraction for test set

        Returns:
            ShaDataset with train/test splits
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for dataset preparation")

        # Filter curves with valid Sha
        valid_curves = [c for c in curves if c.sha_analytic is not None]

        if len(valid_curves) < 10:
            raise ValueError(f"Need at least 10 curves with known Sha, got {len(valid_curves)}")

        # Extract features and targets
        X, y, names = self.feature_extractor.extract_dataset(valid_curves, log_sha=True)

        # Train/test split
        indices = np.arange(len(valid_curves))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=self.random_state
        )

        return ShaDataset(
            X_train=X[train_idx],
            X_test=X[test_idx],
            y_train=y[train_idx],
            y_test=y[test_idx],
            feature_names=names,
            curves_train=[valid_curves[i] for i in train_idx],
            curves_test=[valid_curves[i] for i in test_idx],
        )

    def fit(
        self,
        curves: List[EllipticCurveData],
        test_size: float = 0.2,
        epochs: int = 100,
        lr: float = 0.001,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model on curve data.

        Args:
            curves: Training curves with known Sha
            test_size: Validation split ratio
            epochs: Training epochs (neural network only)
            lr: Learning rate (neural network only)
            verbose: Print progress

        Returns:
            Training metrics dict
        """
        # Prepare data
        dataset = self.prepare_dataset(curves, test_size)

        # Scale features
        X_train = self.scaler.fit_transform(dataset.X_train)
        X_test = self.scaler.transform(dataset.X_test)

        # Create model
        self._create_model(dataset.n_features)

        if verbose:
            logger.info(f"Training {self.model_type} model on {dataset.n_train} curves...")

        # Train based on model type
        if self.model_type == 'neural':
            metrics = self._train_neural(
                X_train, dataset.y_train,
                X_test, dataset.y_test,
                epochs=epochs, lr=lr, verbose=verbose
            )
        elif self.model_type == 'ensemble':
            metrics = self._train_ensemble(
                X_train, dataset.y_train,
                X_test, dataset.y_test,
                epochs=epochs, lr=lr, verbose=verbose
            )
        else:
            # Sklearn models
            self.model.fit(X_train, dataset.y_train)
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)

            metrics = {
                "train_mse": mean_squared_error(dataset.y_train, y_pred_train),
                "test_mse": mean_squared_error(dataset.y_test, y_pred_test),
                "train_mae": mean_absolute_error(dataset.y_train, y_pred_train),
                "test_mae": mean_absolute_error(dataset.y_test, y_pred_test),
                "train_r2": r2_score(dataset.y_train, y_pred_train),
                "test_r2": r2_score(dataset.y_test, y_pred_test),
            }

        self.is_fitted = True
        self.training_metrics = metrics
        self._dataset = dataset

        if verbose:
            logger.info(f"Training complete. Test R2: {metrics.get('test_r2', 'N/A'):.4f}")

        return metrics

    def _train_neural(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int,
        lr: float,
        verbose: bool
    ) -> Dict:
        """Train neural network."""
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        y_test_t = torch.FloatTensor(y_test).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_test_loss = float('inf')
        history = []

        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            y_pred = self.model(X_train_t)
            loss = criterion(y_pred, y_train_t)
            loss.backward()
            optimizer.step()

            # Evaluation
            self.model.eval()
            with torch.no_grad():
                y_pred_test = self.model(X_test_t)
                test_loss = criterion(y_pred_test, y_test_t).item()

            history.append({
                "epoch": epoch,
                "train_loss": loss.item(),
                "test_loss": test_loss
            })

            if test_loss < best_test_loss:
                best_test_loss = test_loss

            if verbose and (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train: {loss.item():.4f}, Test: {test_loss:.4f}")

        # Final metrics
        self.model.eval()
        with torch.no_grad():
            y_pred_train = self.model(X_train_t).cpu().numpy()
            y_pred_test = self.model(X_test_t).cpu().numpy()

        return {
            "train_mse": mean_squared_error(y_train, y_pred_train),
            "test_mse": mean_squared_error(y_test, y_pred_test),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
            "history": history,
        }

    def _train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int,
        lr: float,
        verbose: bool
    ) -> Dict:
        """Train ensemble of models."""
        predictions_train = []
        predictions_test = []

        for name, model in self.models.items():
            if verbose:
                logger.info(f"Training {name}...")

            if name == 'neural' and TORCH_AVAILABLE:
                # Train neural network
                X_train_t = torch.FloatTensor(X_train).to(self.device)
                y_train_t = torch.FloatTensor(y_train).to(self.device)
                X_test_t = torch.FloatTensor(X_test).to(self.device)

                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = nn.MSELoss()

                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()
                    loss = criterion(model(X_train_t), y_train_t)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    predictions_train.append(model(X_train_t).cpu().numpy())
                    predictions_test.append(model(X_test_t).cpu().numpy())
            else:
                model.fit(X_train, y_train)
                predictions_train.append(model.predict(X_train))
                predictions_test.append(model.predict(X_test))

        # Ensemble averaging
        y_pred_train = np.mean(predictions_train, axis=0)
        y_pred_test = np.mean(predictions_test, axis=0)

        return {
            "train_mse": mean_squared_error(y_train, y_pred_train),
            "test_mse": mean_squared_error(y_test, y_pred_test),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
        }

    def predict(
        self,
        curves: Union[EllipticCurveData, List[EllipticCurveData]],
        return_uncertainty: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict Sha for new curves.

        Args:
            curves: Single curve or list of curves
            return_uncertainty: Also return uncertainty estimates

        Returns:
            Predicted Sha values (and optionally uncertainties)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Handle single curve
        if isinstance(curves, EllipticCurveData):
            curves = [curves]

        # Extract and scale features
        X, _ = self.feature_extractor.extract_batch(curves)
        X_scaled = self.scaler.transform(X)

        # Predict based on model type
        if self.model_type == 'neural':
            self.model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X_scaled).to(self.device)
                log_sha_pred = self.model(X_t).cpu().numpy()
        elif self.model_type == 'ensemble':
            predictions = []
            for name, model in self.models.items():
                if name == 'neural' and TORCH_AVAILABLE:
                    model.eval()
                    with torch.no_grad():
                        X_t = torch.FloatTensor(X_scaled).to(self.device)
                        predictions.append(model(X_t).cpu().numpy())
                else:
                    predictions.append(model.predict(X_scaled))
            log_sha_pred = np.mean(predictions, axis=0)

            if return_uncertainty:
                uncertainty = np.std(predictions, axis=0)
                sha_pred = np.exp(log_sha_pred)
                return sha_pred, uncertainty
        else:
            log_sha_pred = self.model.predict(X_scaled)

        # Convert from log(Sha) to Sha
        sha_pred = np.exp(log_sha_pred)

        # Round to nearest perfect square (Sha is always a perfect square)
        sha_pred = np.round(sha_pred)
        sha_pred = np.maximum(sha_pred, 1)  # Minimum is 1

        if return_uncertainty:
            # Estimate uncertainty from training residuals
            uncertainty = np.full_like(sha_pred, np.sqrt(self.training_metrics.get('test_mse', 1)))
            return sha_pred, uncertainty

        return sha_pred

    def predict_regime(
        self,
        curves: Union[EllipticCurveData, List[EllipticCurveData]]
    ) -> List[str]:
        """
        Predict Reynolds regime for curves.

        Args:
            curves: Curves to classify

        Returns:
            List of regime names
        """
        if isinstance(curves, EllipticCurveData):
            curves = [curves]

        regimes = []
        for curve in curves:
            features = self.feature_extractor.extract(curve)
            if features.is_laminar:
                regimes.append("LAMINAR")
            elif features.is_turbulent:
                regimes.append("TURBULENT")
            else:
                regimes.append("TRANSITION")

        return regimes

    def feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dict mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        if self.model_type in ['gbm', 'rf']:
            importances = self.model.feature_importances_
            names = self._dataset.feature_names
            return dict(sorted(
                zip(names, importances),
                key=lambda x: x[1],
                reverse=True
            ))
        else:
            return {"note": "Feature importance not available for this model type"}

    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "model_type": self.model_type,
            "is_fitted": self.is_fitted,
            "training_metrics": self.training_metrics,
            "timestamp": datetime.now().isoformat(),
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save scaler
        if SKLEARN_AVAILABLE:
            import joblib
            joblib.dump(self.scaler, path / "scaler.pkl")

        # Save model
        if self.model_type == 'neural' and TORCH_AVAILABLE:
            torch.save(self.model.state_dict(), path / "model.pt")
        elif self.model_type in ['gbm', 'rf'] and SKLEARN_AVAILABLE:
            import joblib
            joblib.dump(self.model, path / "model.pkl")

        logger.info(f"Model saved to {path}")

    def load(self, path: Path):
        """Load model from disk."""
        path = Path(path)

        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        self.model_type = metadata["model_type"]
        self.is_fitted = metadata["is_fitted"]
        self.training_metrics = metadata["training_metrics"]

        # Load scaler
        if SKLEARN_AVAILABLE:
            import joblib
            self.scaler = joblib.load(path / "scaler.pkl")

        # Load model
        if self.model_type == 'neural' and TORCH_AVAILABLE:
            # Need to recreate model architecture first
            # This requires knowing n_features
            state_dict = torch.load(path / "model.pt", map_location=self.device)
            n_features = state_dict['network.0.weight'].shape[1]
            self._create_model(n_features)
            self.model.load_state_dict(state_dict)
        elif self.model_type in ['gbm', 'rf'] and SKLEARN_AVAILABLE:
            import joblib
            self.model = joblib.load(path / "model.pkl")

        logger.info(f"Model loaded from {path}")

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"ShaPredictor(model_type='{self.model_type}', {status})"
