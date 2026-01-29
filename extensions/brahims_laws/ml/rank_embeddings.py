"""
Rank-Aware Embeddings for Elliptic Curves.

Learn vector representations that capture how rank affects Sha behavior.

Key insight from Brahim's Laws:
- Higher rank curves have more "stabilization" against Sha > 1
- Rank 0 curves are most likely to have nontrivial Sha
- The embedding should capture this rank-dependent structure

Architecture:
- Curve encoder: maps invariants to embedding space
- Rank embedding: learnable vectors per rank
- Fusion: combines curve + rank embeddings
- Output heads: predict Sha, regime, etc.

Author: Elias Oulad Brahim
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import json
from pathlib import Path

from ..models.curve_data import EllipticCurveData
from .feature_extractor import CurveFeatureExtractor

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
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


class CurveDataset(Dataset):
    """PyTorch dataset for elliptic curves."""

    def __init__(
        self,
        features: np.ndarray,
        ranks: np.ndarray,
        sha_values: np.ndarray,
        regimes: np.ndarray
    ):
        self.features = torch.FloatTensor(features)
        self.ranks = torch.LongTensor(ranks)
        self.sha_values = torch.FloatTensor(sha_values)
        self.regimes = torch.LongTensor(regimes)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'rank': self.ranks[idx],
            'sha': self.sha_values[idx],
            'regime': self.regimes[idx]
        }


class RankAwareEncoder(nn.Module):
    """
    Neural network that learns rank-aware curve embeddings.

    Components:
    1. Curve encoder: MLP that maps invariants to latent space
    2. Rank embeddings: Learnable vectors for each rank (0, 1, 2, 3+)
    3. Fusion layer: Combines curve and rank embeddings
    4. Multi-task heads: Predict Sha and regime simultaneously

    The rank embeddings capture the "stabilization effect":
    - Rank 0 embedding should be "closer" to high-Sha region
    - Higher ranks should embed "closer" to Sha=1 region
    """

    def __init__(
        self,
        n_features: int,
        embedding_dim: int = 64,
        rank_embedding_dim: int = 16,
        max_rank: int = 4,
        hidden_sizes: List[int] = [128, 64],
        dropout: float = 0.2
    ):
        """
        Initialize encoder.

        Args:
            n_features: Number of input features
            embedding_dim: Curve embedding dimension
            rank_embedding_dim: Rank embedding dimension
            max_rank: Maximum rank to embed (higher ranks clamped)
            hidden_sizes: Hidden layer sizes for curve encoder
            dropout: Dropout rate
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.rank_embedding_dim = rank_embedding_dim
        self.max_rank = max_rank

        # Curve encoder: invariants -> embedding
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
        layers.append(nn.Linear(prev_size, embedding_dim))

        self.curve_encoder = nn.Sequential(*layers)

        # Rank embeddings: learnable vectors for each rank
        self.rank_embeddings = nn.Embedding(max_rank + 1, rank_embedding_dim)

        # Fusion layer: combine curve and rank embeddings
        fusion_input_dim = embedding_dim + rank_embedding_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output heads
        # Sha prediction head (regression)
        self.sha_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Regime classification head (3 classes)
        self.regime_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

        # Sha > 1 probability head (binary)
        self.sha_nontrivial_head = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def encode_curves(self, features: torch.Tensor) -> torch.Tensor:
        """Encode curve features to embedding space."""
        return self.curve_encoder(features)

    def get_rank_embedding(self, ranks: torch.Tensor) -> torch.Tensor:
        """Get rank embeddings, clamping ranks above max."""
        clamped_ranks = torch.clamp(ranks, 0, self.max_rank)
        return self.rank_embeddings(clamped_ranks)

    def fuse_embeddings(
        self,
        curve_emb: torch.Tensor,
        rank_emb: torch.Tensor
    ) -> torch.Tensor:
        """Fuse curve and rank embeddings."""
        combined = torch.cat([curve_emb, rank_emb], dim=-1)
        return self.fusion(combined)

    def forward(
        self,
        features: torch.Tensor,
        ranks: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with all predictions.

        Args:
            features: Curve features (batch_size, n_features)
            ranks: Curve ranks (batch_size,)

        Returns:
            Dict with embeddings and predictions
        """
        # Encode
        curve_emb = self.encode_curves(features)
        rank_emb = self.get_rank_embedding(ranks)

        # Fuse
        fused_emb = self.fuse_embeddings(curve_emb, rank_emb)

        # Predict
        sha_pred = self.sha_head(fused_emb).squeeze(-1)
        regime_logits = self.regime_head(fused_emb)
        sha_nontrivial_prob = self.sha_nontrivial_head(fused_emb).squeeze(-1)

        return {
            'curve_embedding': curve_emb,
            'rank_embedding': rank_emb,
            'fused_embedding': fused_emb,
            'sha_pred': sha_pred,
            'regime_logits': regime_logits,
            'sha_nontrivial_prob': sha_nontrivial_prob,
        }

    def get_embedding(
        self,
        features: torch.Tensor,
        ranks: torch.Tensor
    ) -> torch.Tensor:
        """Get final fused embedding for a curve."""
        curve_emb = self.encode_curves(features)
        rank_emb = self.get_rank_embedding(ranks)
        return self.fuse_embeddings(curve_emb, rank_emb)


class RankAwareEmbedder:
    """
    High-level interface for learning rank-aware embeddings.

    Trains a multi-task model that learns:
    1. Curve embeddings from invariants
    2. Rank embeddings that capture stabilization effect
    3. Joint representations for Sha and regime prediction

    Example:
        embedder = RankAwareEmbedder()
        embedder.fit(curves)
        embeddings = embedder.get_embeddings(new_curves)
        similarity = embedder.compute_similarity(curve_a, curve_b)
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        rank_embedding_dim: int = 16,
        device: Optional[str] = None
    ):
        """
        Initialize embedder.

        Args:
            embedding_dim: Curve embedding dimension
            rank_embedding_dim: Rank embedding dimension
            device: PyTorch device
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for rank embeddings")

        self.embedding_dim = embedding_dim
        self.rank_embedding_dim = rank_embedding_dim
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        self.feature_extractor = CurveFeatureExtractor()
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.model = None
        self.is_fitted = False

    def _prepare_data(
        self,
        curves: List[EllipticCurveData]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features, ranks, sha values, and regimes."""
        X, _ = self.feature_extractor.extract_batch(curves)

        ranks = np.array([c.rank for c in curves])
        sha_values = np.array([
            np.log(c.sha_analytic + 1e-10) if c.sha_analytic else 0
            for c in curves
        ])

        # Compute regimes
        regimes = []
        for c in curves:
            features = self.feature_extractor.extract(c)
            if features.is_laminar:
                regimes.append(0)
            elif features.is_turbulent:
                regimes.append(2)
            else:
                regimes.append(1)
        regimes = np.array(regimes)

        return X, ranks, sha_values, regimes

    def fit(
        self,
        curves: List[EllipticCurveData],
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.001,
        sha_weight: float = 1.0,
        regime_weight: float = 0.5,
        nontrivial_weight: float = 0.5,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the rank-aware embedding model.

        Multi-task loss:
        L = sha_weight * L_sha + regime_weight * L_regime + nontrivial_weight * L_nontrivial

        Args:
            curves: Training curves
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            sha_weight: Weight for Sha regression loss
            regime_weight: Weight for regime classification loss
            nontrivial_weight: Weight for Sha>1 binary classification
            verbose: Print progress

        Returns:
            Training history
        """
        # Prepare data
        X, ranks, sha_values, regimes = self._prepare_data(curves)
        X_scaled = self.scaler.fit_transform(X)

        # Create dataset
        dataset = CurveDataset(X_scaled, ranks, sha_values, regimes)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Create model
        n_features = X.shape[1]
        self.model = RankAwareEncoder(
            n_features=n_features,
            embedding_dim=self.embedding_dim,
            rank_embedding_dim=self.rank_embedding_dim
        ).to(self.device)

        # Optimizers and losses
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        sha_criterion = nn.MSELoss()
        regime_criterion = nn.CrossEntropyLoss()
        nontrivial_criterion = nn.BCELoss()

        history = {'total_loss': [], 'sha_loss': [], 'regime_loss': [], 'nontrivial_loss': []}

        for epoch in range(epochs):
            epoch_losses = []

            self.model.train()
            for batch in dataloader:
                features = batch['features'].to(self.device)
                batch_ranks = batch['rank'].to(self.device)
                batch_sha = batch['sha'].to(self.device)
                batch_regime = batch['regime'].to(self.device)

                # Forward pass
                outputs = self.model(features, batch_ranks)

                # Compute losses
                sha_loss = sha_criterion(outputs['sha_pred'], batch_sha)
                regime_loss = regime_criterion(outputs['regime_logits'], batch_regime)

                # Sha > 1 binary target
                sha_nontrivial_target = (batch_sha > 0).float()  # log(sha) > 0 means sha > 1
                nontrivial_loss = nontrivial_criterion(
                    outputs['sha_nontrivial_prob'],
                    sha_nontrivial_target
                )

                # Total loss
                total_loss = (
                    sha_weight * sha_loss +
                    regime_weight * regime_loss +
                    nontrivial_weight * nontrivial_loss
                )

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_losses.append(total_loss.item())

            # Record history
            avg_loss = np.mean(epoch_losses)
            history['total_loss'].append(avg_loss)

            if verbose and (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        self.is_fitted = True

        if verbose:
            self._analyze_rank_embeddings()

        return history

    def _analyze_rank_embeddings(self):
        """Analyze learned rank embeddings."""
        logger.info("\nRank Embedding Analysis:")
        logger.info("-" * 40)

        with torch.no_grad():
            for rank in range(self.model.max_rank + 1):
                emb = self.model.rank_embeddings(torch.tensor([rank]).to(self.device))
                norm = torch.norm(emb).item()
                logger.info(f"Rank {rank}: ||embedding|| = {norm:.4f}")

            # Compute pairwise distances
            all_emb = self.model.rank_embeddings.weight  # (max_rank+1, dim)
            distances = torch.cdist(all_emb.unsqueeze(0), all_emb.unsqueeze(0)).squeeze()

            logger.info("\nRank embedding distances:")
            for i in range(min(3, self.model.max_rank + 1)):
                for j in range(i + 1, min(4, self.model.max_rank + 1)):
                    logger.info(f"  d(rank {i}, rank {j}) = {distances[i, j].item():.4f}")

    def get_embeddings(
        self,
        curves: List[EllipticCurveData]
    ) -> np.ndarray:
        """
        Get fused embeddings for curves.

        Args:
            curves: Curves to embed

        Returns:
            Embedding array of shape (n_curves, embedding_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        X, ranks, _, _ = self._prepare_data(curves)
        X_scaled = self.scaler.transform(X)

        self.model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(X_scaled).to(self.device)
            ranks_t = torch.LongTensor(ranks).to(self.device)
            embeddings = self.model.get_embedding(features, ranks_t)
            return embeddings.cpu().numpy()

    def get_rank_embeddings(self) -> Dict[int, np.ndarray]:
        """
        Get learned rank embeddings.

        Returns:
            Dict mapping rank -> embedding vector
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        self.model.eval()
        with torch.no_grad():
            embeddings = {}
            for rank in range(self.model.max_rank + 1):
                emb = self.model.rank_embeddings(torch.tensor([rank]).to(self.device))
                embeddings[rank] = emb.cpu().numpy().flatten()
            return embeddings

    def compute_similarity(
        self,
        curve_a: EllipticCurveData,
        curve_b: EllipticCurveData
    ) -> float:
        """
        Compute cosine similarity between two curves in embedding space.

        Args:
            curve_a: First curve
            curve_b: Second curve

        Returns:
            Cosine similarity (-1 to 1)
        """
        embeddings = self.get_embeddings([curve_a, curve_b])
        emb_a, emb_b = embeddings[0], embeddings[1]

        cos_sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
        return float(cos_sim)

    def predict_sha(
        self,
        curves: List[EllipticCurveData]
    ) -> np.ndarray:
        """Predict Sha values using the embedding model."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        X, ranks, _, _ = self._prepare_data(curves)
        X_scaled = self.scaler.transform(X)

        self.model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(X_scaled).to(self.device)
            ranks_t = torch.LongTensor(ranks).to(self.device)
            outputs = self.model(features, ranks_t)
            log_sha = outputs['sha_pred'].cpu().numpy()
            return np.exp(log_sha)

    def predict_regime(
        self,
        curves: List[EllipticCurveData]
    ) -> List[str]:
        """Predict regime using the embedding model."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        X, ranks, _, _ = self._prepare_data(curves)
        X_scaled = self.scaler.transform(X)

        self.model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(X_scaled).to(self.device)
            ranks_t = torch.LongTensor(ranks).to(self.device)
            outputs = self.model(features, ranks_t)
            regime_idx = outputs['regime_logits'].argmax(dim=-1).cpu().numpy()

        names = ['LAMINAR', 'TRANSITION', 'TURBULENT']
        return [names[i] for i in regime_idx]

    def save(self, path: Path):
        """Save embedder to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "embedding_dim": self.embedding_dim,
            "rank_embedding_dim": self.rank_embedding_dim,
            "is_fitted": self.is_fitted,
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        if SKLEARN_AVAILABLE:
            import joblib
            joblib.dump(self.scaler, path / "scaler.pkl")

        if self.model is not None:
            torch.save(self.model.state_dict(), path / "model.pt")

        logger.info(f"Embedder saved to {path}")
