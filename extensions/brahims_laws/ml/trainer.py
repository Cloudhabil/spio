"""
Training Pipeline for Sha Prediction Models.

Orchestrates the full ML workflow:
1. Data collection from LMFDB
2. Feature extraction
3. Model training with cross-validation
4. Evaluation and model selection
5. Model persistence

Author: Elias Oulad Brahim
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

from ..models.curve_data import EllipticCurveData
from ..data.lmfdb_client import LMFDBClient
from .sha_predictor import ShaPredictor, ShaDataset
from .feature_extractor import CurveFeatureExtractor

logger = logging.getLogger(__name__)

# Try sklearn imports
try:
    from sklearn.model_selection import cross_val_score, KFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ShaModelTrainer:
    """
    End-to-end trainer for Sha prediction models.

    Handles data collection, training, evaluation, and persistence.

    Example:
        trainer = ShaModelTrainer()
        trainer.collect_data(n_curves=10000)
        results = trainer.train_all_models()
        trainer.save_best_model("models/sha_predictor")
    """

    def __init__(
        self,
        output_dir: Path = Path("models/sha_predictor"),
        random_state: int = 42
    ):
        """
        Initialize trainer.

        Args:
            output_dir: Directory for saving models and results
            random_state: Random seed
        """
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.curves: List[EllipticCurveData] = []
        self.training_results: Dict[str, Dict] = {}
        self.best_model: Optional[ShaPredictor] = None
        self.best_model_name: Optional[str] = None

    def collect_data(
        self,
        n_curves: int = 5000,
        ranks: List[int] = [0, 1, 2],
        conductor_max: int = 500000,
        include_nontrivial_sha: bool = True,
        verbose: bool = True
    ) -> int:
        """
        Collect training data from LMFDB.

        Args:
            n_curves: Target number of curves
            ranks: Ranks to include
            conductor_max: Maximum conductor
            include_nontrivial_sha: Ensure some curves have Sha > 1
            verbose: Print progress

        Returns:
            Number of curves collected
        """
        client = LMFDBClient()

        if not client.test_connection():
            logger.warning("LMFDB not available, using cached data if present")
            return 0

        curves_per_rank = n_curves // len(ranks)
        all_curves = []

        for rank in ranks:
            if verbose:
                logger.info(f"Fetching rank {rank} curves...")

            curves = client.fetch_batch(
                rank=rank,
                conductor_max=conductor_max,
                limit=curves_per_rank
            )
            all_curves.extend(curves)

            if verbose:
                logger.info(f"  Got {len(curves)} curves")

        # Filter for curves with known Sha
        self.curves = [c for c in all_curves if c.sha_analytic is not None]

        if verbose:
            n_nontrivial = sum(1 for c in self.curves if c.sha_analytic and c.sha_analytic > 1)
            logger.info(f"Total: {len(self.curves)} curves with known Sha")
            logger.info(f"  Nontrivial Sha (>1): {n_nontrivial}")

        return len(self.curves)

    def load_data(self, filepath: Path) -> int:
        """
        Load training data from file.

        Args:
            filepath: Path to JSON/JSONL file

        Returns:
            Number of curves loaded
        """
        from ..data.cremona_loader import CremonaLoader

        loader = CremonaLoader()
        filepath = Path(filepath)

        if filepath.suffix == '.jsonl':
            self.curves = list(loader.load_from_jsonl(filepath))
        else:
            self.curves = loader.load_from_json(filepath)

        # Filter for known Sha
        self.curves = [c for c in self.curves if c.sha_analytic is not None]

        logger.info(f"Loaded {len(self.curves)} curves with known Sha")
        return len(self.curves)

    def train_model(
        self,
        model_type: str,
        test_size: float = 0.2,
        epochs: int = 100,
        verbose: bool = True
    ) -> Dict:
        """
        Train a single model type.

        Args:
            model_type: 'neural', 'gbm', 'rf', or 'ensemble'
            test_size: Validation split
            epochs: Training epochs (neural only)
            verbose: Print progress

        Returns:
            Training metrics
        """
        if len(self.curves) < 100:
            raise ValueError(f"Need at least 100 curves, got {len(self.curves)}")

        if verbose:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_type.upper()} model")
            logger.info(f"{'='*50}")

        predictor = ShaPredictor(
            model_type=model_type,
            random_state=self.random_state
        )

        metrics = predictor.fit(
            self.curves,
            test_size=test_size,
            epochs=epochs,
            verbose=verbose
        )

        # Store results
        self.training_results[model_type] = {
            "metrics": metrics,
            "predictor": predictor,
            "timestamp": datetime.now().isoformat()
        }

        # Track best model
        test_r2 = metrics.get('test_r2', -float('inf'))
        if self.best_model is None or test_r2 > self.training_results.get(
            self.best_model_name, {}
        ).get('metrics', {}).get('test_r2', -float('inf')):
            self.best_model = predictor
            self.best_model_name = model_type

        return metrics

    def train_all_models(
        self,
        test_size: float = 0.2,
        epochs: int = 100,
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Train all available model types and compare.

        Args:
            test_size: Validation split
            epochs: Training epochs
            verbose: Print progress

        Returns:
            Dict of model_type -> metrics
        """
        model_types = ['gbm', 'rf']

        # Add neural if PyTorch available
        try:
            import torch
            model_types.append('neural')
        except ImportError:
            pass

        # Add ensemble
        model_types.append('ensemble')

        results = {}
        for model_type in model_types:
            try:
                metrics = self.train_model(
                    model_type,
                    test_size=test_size,
                    epochs=epochs,
                    verbose=verbose
                )
                results[model_type] = metrics
            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
                results[model_type] = {"error": str(e)}

        # Summary
        if verbose:
            self._print_comparison(results)

        return results

    def cross_validate(
        self,
        model_type: str = 'gbm',
        n_folds: int = 5,
        verbose: bool = True
    ) -> Dict:
        """
        Perform k-fold cross-validation.

        Args:
            model_type: Model type to validate
            n_folds: Number of folds
            verbose: Print progress

        Returns:
            Cross-validation results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for cross-validation")

        extractor = CurveFeatureExtractor()
        X, y, names = extractor.extract_dataset(self.curves, log_sha=True)

        # Create base estimator
        if model_type == 'gbm':
            from sklearn.ensemble import GradientBoostingRegressor
            estimator = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                random_state=self.random_state
            )
        elif model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Cross-validation not supported for {model_type}")

        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Cross-validate
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        # R2 scores
        r2_scores = cross_val_score(estimator, X_scaled, y, cv=kfold, scoring='r2')

        # MSE scores (negative because sklearn maximizes)
        mse_scores = -cross_val_score(
            estimator, X_scaled, y, cv=kfold,
            scoring='neg_mean_squared_error'
        )

        results = {
            "model_type": model_type,
            "n_folds": n_folds,
            "r2_mean": float(np.mean(r2_scores)),
            "r2_std": float(np.std(r2_scores)),
            "r2_scores": r2_scores.tolist(),
            "mse_mean": float(np.mean(mse_scores)),
            "mse_std": float(np.std(mse_scores)),
        }

        if verbose:
            logger.info(f"\nCross-validation results ({model_type}, {n_folds} folds):")
            logger.info(f"  R2: {results['r2_mean']:.4f} (+/- {results['r2_std']:.4f})")
            logger.info(f"  MSE: {results['mse_mean']:.4f} (+/- {results['mse_std']:.4f})")

        return results

    def _print_comparison(self, results: Dict[str, Dict]):
        """Print model comparison table."""
        logger.info(f"\n{'='*60}")
        logger.info("MODEL COMPARISON")
        logger.info(f"{'='*60}")
        logger.info(f"{'Model':<15} {'Test R2':<12} {'Test MSE':<12} {'Test MAE':<12}")
        logger.info(f"{'-'*60}")

        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1].get('test_r2', -999),
            reverse=True
        )

        for model_type, metrics in sorted_models:
            if 'error' in metrics:
                logger.info(f"{model_type:<15} ERROR: {metrics['error']}")
            else:
                r2 = metrics.get('test_r2', 0)
                mse = metrics.get('test_mse', 0)
                mae = metrics.get('test_mae', 0)
                marker = " <-- BEST" if model_type == self.best_model_name else ""
                logger.info(f"{model_type:<15} {r2:<12.4f} {mse:<12.4f} {mae:<12.4f}{marker}")

        logger.info(f"{'='*60}")

    def save_best_model(self, path: Optional[Path] = None) -> Path:
        """
        Save the best performing model.

        Args:
            path: Save path (uses output_dir if None)

        Returns:
            Path where model was saved
        """
        if self.best_model is None:
            raise RuntimeError("No model trained yet")

        path = Path(path) if path else self.output_dir / self.best_model_name
        self.best_model.save(path)

        # Save training summary
        summary = {
            "best_model": self.best_model_name,
            "n_training_curves": len(self.curves),
            "all_results": {
                k: v.get('metrics', {})
                for k, v in self.training_results.items()
            },
            "timestamp": datetime.now().isoformat()
        }

        with open(path / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Best model ({self.best_model_name}) saved to {path}")
        return path

    def load_model(self, path: Path) -> ShaPredictor:
        """
        Load a trained model.

        Args:
            path: Path to saved model

        Returns:
            Loaded predictor
        """
        predictor = ShaPredictor()
        predictor.load(path)
        return predictor

    def analyze_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from best model.

        Returns:
            Feature importance dict
        """
        if self.best_model is None:
            raise RuntimeError("No model trained yet")

        return self.best_model.feature_importance()

    def generate_report(self) -> str:
        """
        Generate training report as markdown.

        Returns:
            Markdown formatted report
        """
        lines = [
            "# Sha Prediction Model Training Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Dataset",
            "",
            f"- **Total curves:** {len(self.curves)}",
            f"- **Curves with Sha > 1:** {sum(1 for c in self.curves if c.sha_analytic and c.sha_analytic > 1)}",
            "",
            "## Model Comparison",
            "",
            "| Model | Test R2 | Test MSE | Test MAE |",
            "|-------|---------|----------|----------|",
        ]

        for model_type, result in sorted(
            self.training_results.items(),
            key=lambda x: x[1].get('metrics', {}).get('test_r2', -999),
            reverse=True
        ):
            metrics = result.get('metrics', {})
            if 'error' not in metrics:
                r2 = metrics.get('test_r2', 0)
                mse = metrics.get('test_mse', 0)
                mae = metrics.get('test_mae', 0)
                best = " **BEST**" if model_type == self.best_model_name else ""
                lines.append(f"| {model_type}{best} | {r2:.4f} | {mse:.4f} | {mae:.4f} |")

        lines.extend([
            "",
            "## Feature Importance",
            "",
        ])

        if self.best_model and self.best_model_name in ['gbm', 'rf']:
            importance = self.best_model.feature_importance()
            lines.append("| Feature | Importance |")
            lines.append("|---------|------------|")
            for feat, imp in list(importance.items())[:10]:
                lines.append(f"| {feat} | {imp:.4f} |")

        lines.extend([
            "",
            "## Interpretation",
            "",
            "Based on Brahim's Laws, we expect high importance for:",
            "- `im_tau_scaled`: Sha ~ Im(tau)^(2/3) (Law 1)",
            "- `reynolds_scaled`: Sha_max ~ Rey^(5/12) (Law 4)",
            "- `is_turbulent`: Turbulent regime predicts Sha > 1 (Law 3)",
            "",
        ])

        return "\n".join(lines)
