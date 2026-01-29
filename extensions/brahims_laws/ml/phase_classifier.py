"""
Phase Classifier for Reynolds Regime Prediction.

Classification problem: Predict whether a curve is in
LAMINAR, TRANSITION, or TURBULENT regime from invariants alone.

Key insight: If we can predict regime without computing Reynolds directly,
it validates that curve invariants encode the "arithmetic turbulence" structure.

Author: Elias Oulad Brahim
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
import json
from pathlib import Path

from ..models.curve_data import EllipticCurveData, Regime
from .feature_extractor import CurveFeatureExtractor

logger = logging.getLogger(__name__)

# Try imports
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
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class ClassificationMetrics:
    """Metrics for phase classification."""
    accuracy: float
    f1_macro: float
    f1_per_class: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: str


class PhaseClassifierNet(nn.Module):
    """
    Neural network for 3-class phase classification.

    Architecture:
    - Input: curve invariants (excluding Reynolds!)
    - Hidden layers with batch norm and dropout
    - Output: 3-class softmax (laminar, transition, turbulent)
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: List[int] = [128, 64, 32],
        dropout: float = 0.3
    ):
        super().__init__()

        layers = []
        prev_size = n_features

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, 3)  # 3 classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class PhaseClassifier:
    """
    Classifier for predicting Reynolds regime from curve invariants.

    The key challenge: Can we predict laminar/transition/turbulent
    WITHOUT explicitly computing Reynolds number?

    If yes, it means the regime is encoded in the raw invariants,
    validating Brahim's Laws structure.

    Example:
        classifier = PhaseClassifier(model_type='neural')
        classifier.fit(curves)
        regimes = classifier.predict(new_curves)
        probs = classifier.predict_proba(new_curves)
    """

    REGIME_NAMES = ['LAMINAR', 'TRANSITION', 'TURBULENT']

    def __init__(
        self,
        model_type: str = 'neural',
        exclude_reynolds: bool = True,
        device: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Initialize classifier.

        Args:
            model_type: 'neural', 'gbm', or 'rf'
            exclude_reynolds: If True, don't use Reynolds as feature (harder but more meaningful)
            device: PyTorch device
            random_state: Random seed
        """
        self.model_type = model_type
        self.exclude_reynolds = exclude_reynolds
        self.random_state = random_state
        self.feature_extractor = CurveFeatureExtractor()
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.label_encoder = LabelEncoder() if SKLEARN_AVAILABLE else None
        self.model = None
        self.is_fitted = False

        if TORCH_AVAILABLE:
            self.device = torch.device(
                device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
            )
        else:
            self.device = None

    def _extract_features_for_classification(
        self,
        curves: List[EllipticCurveData]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features, optionally excluding Reynolds-derived features.
        """
        X, names = self.feature_extractor.extract_batch(curves)

        if self.exclude_reynolds:
            # Remove Reynolds-related features to make the problem harder
            # but more scientifically meaningful
            exclude_indices = [
                i for i, name in enumerate(names)
                if 'reynolds' in name.lower() or name in ['is_laminar', 'is_transition', 'is_turbulent']
            ]
            keep_indices = [i for i in range(len(names)) if i not in exclude_indices]
            X = X[:, keep_indices]
            names = [names[i] for i in keep_indices]

        return X, names

    def _get_regime_labels(self, curves: List[EllipticCurveData]) -> np.ndarray:
        """Extract regime labels from curves."""
        labels = []
        for curve in curves:
            features = self.feature_extractor.extract(curve)
            if features.is_laminar:
                labels.append('LAMINAR')
            elif features.is_turbulent:
                labels.append('TURBULENT')
            else:
                labels.append('TRANSITION')
        return np.array(labels)

    def fit(
        self,
        curves: List[EllipticCurveData],
        test_size: float = 0.2,
        epochs: int = 100,
        lr: float = 0.001,
        class_weights: Optional[str] = 'balanced',
        verbose: bool = True
    ) -> ClassificationMetrics:
        """
        Train the phase classifier.

        Args:
            curves: Training curves
            test_size: Validation split
            epochs: Training epochs (neural only)
            lr: Learning rate
            class_weights: 'balanced' to handle class imbalance
            verbose: Print progress

        Returns:
            Classification metrics
        """
        # Extract features and labels
        X, self.feature_names = self._extract_features_for_classification(curves)
        y = self._get_regime_labels(curves)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size,
            random_state=self.random_state, stratify=y_encoded
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if verbose:
            logger.info(f"Training {self.model_type} phase classifier")
            logger.info(f"  Features: {len(self.feature_names)} (Reynolds excluded: {self.exclude_reynolds})")
            logger.info(f"  Train size: {len(y_train)}, Test size: {len(y_test)}")
            unique, counts = np.unique(y_train, return_counts=True)
            for u, c in zip(unique, counts):
                logger.info(f"    {self.REGIME_NAMES[u]}: {c}")

        # Train model
        if self.model_type == 'neural':
            metrics = self._train_neural(
                X_train_scaled, y_train,
                X_test_scaled, y_test,
                epochs=epochs, lr=lr,
                class_weights=class_weights,
                verbose=verbose
            )
        else:
            metrics = self._train_sklearn(
                X_train_scaled, y_train,
                X_test_scaled, y_test,
                class_weights=class_weights,
                verbose=verbose
            )

        self.is_fitted = True
        return metrics

    def _train_neural(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
        epochs: int, lr: float,
        class_weights: Optional[str],
        verbose: bool
    ) -> ClassificationMetrics:
        """Train neural network classifier."""
        n_features = X_train.shape[1]
        self.model = PhaseClassifierNet(n_features).to(self.device)

        # Compute class weights
        if class_weights == 'balanced':
            class_counts = np.bincount(y_train)
            weights = 1.0 / class_counts
            weights = weights / weights.sum() * len(class_counts)
            weight_tensor = torch.FloatTensor(weights).to(self.device)
        else:
            weight_tensor = None

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_test_t = torch.FloatTensor(X_test).to(self.device)

        best_f1 = 0
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            logits = self.model(X_train_t)
            loss = criterion(logits, y_train_t)
            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % 20 == 0:
                self.model.eval()
                with torch.no_grad():
                    test_logits = self.model(X_test_t)
                    test_preds = test_logits.argmax(dim=-1).cpu().numpy()
                    f1 = f1_score(y_test, test_preds, average='macro')
                    if f1 > best_f1:
                        best_f1 = f1
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}, F1: {f1:.4f}")

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            test_logits = self.model(X_test_t)
            y_pred = test_logits.argmax(dim=-1).cpu().numpy()

        return self._compute_metrics(y_test, y_pred)

    def _train_sklearn(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
        class_weights: Optional[str],
        verbose: bool
    ) -> ClassificationMetrics:
        """Train sklearn classifier."""
        if self.model_type == 'gbm':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                random_state=self.random_state
            )
        else:  # rf
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight=class_weights,
                random_state=self.random_state,
                n_jobs=-1
            )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        return self._compute_metrics(y_test, y_pred)

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> ClassificationMetrics:
        """Compute classification metrics."""
        accuracy = np.mean(y_true == y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_per = f1_score(y_true, y_pred, average=None)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=self.REGIME_NAMES
        )

        return ClassificationMetrics(
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_per_class={name: f1_per[i] for i, name in enumerate(self.REGIME_NAMES)},
            confusion_matrix=cm,
            classification_report=report
        )

    def predict(
        self,
        curves: List[EllipticCurveData]
    ) -> List[str]:
        """
        Predict regime for curves.

        Args:
            curves: Curves to classify

        Returns:
            List of regime names
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        X, _ = self._extract_features_for_classification(curves)
        X_scaled = self.scaler.transform(X)

        if self.model_type == 'neural':
            self.model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X_scaled).to(self.device)
                logits = self.model(X_t)
                y_pred = logits.argmax(dim=-1).cpu().numpy()
        else:
            y_pred = self.model.predict(X_scaled)

        return [self.REGIME_NAMES[i] for i in y_pred]

    def predict_proba(
        self,
        curves: List[EllipticCurveData]
    ) -> np.ndarray:
        """
        Predict regime probabilities.

        Args:
            curves: Curves to classify

        Returns:
            Array of shape (n_curves, 3) with probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        X, _ = self._extract_features_for_classification(curves)
        X_scaled = self.scaler.transform(X)

        if self.model_type == 'neural':
            self.model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X_scaled).to(self.device)
                probs = self.model.predict_proba(X_t).cpu().numpy()
        else:
            probs = self.model.predict_proba(X_scaled)

        return probs

    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance for tree-based models."""
        if self.model_type in ['gbm', 'rf'] and hasattr(self.model, 'feature_importances_'):
            return dict(sorted(
                zip(self.feature_names, self.model.feature_importances_),
                key=lambda x: x[1],
                reverse=True
            ))
        return {}

    def save(self, path: Path):
        """Save classifier to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "model_type": self.model_type,
            "exclude_reynolds": self.exclude_reynolds,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        if SKLEARN_AVAILABLE:
            import joblib
            joblib.dump(self.scaler, path / "scaler.pkl")
            joblib.dump(self.label_encoder, path / "label_encoder.pkl")

            if self.model_type in ['gbm', 'rf']:
                joblib.dump(self.model, path / "model.pkl")

        if self.model_type == 'neural' and TORCH_AVAILABLE:
            torch.save(self.model.state_dict(), path / "model.pt")

        logger.info(f"Classifier saved to {path}")
