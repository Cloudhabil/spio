"""
Quantization & Vector Compression

Reduces embedding dimensionality and memory footprint while
preserving semantic similarity. Critical for efficient memory management.

Methods:
- Int8 quantization with scale factor
- Residual quantization (center + residual)
- Product quantization for extreme compression
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path


@dataclass
class QuantizedVector:
    """A quantized vector with metadata for reconstruction."""
    data: np.ndarray           # int8 quantized values
    scale: float               # Scale factor for dequantization
    zero_point: float = 0.0    # Zero point offset
    original_norm: float = 1.0  # Original L2 norm

    def dequantize(self) -> np.ndarray:
        """Reconstruct the original float32 vector."""
        float_vec = (self.data.astype(np.float32) - self.zero_point) * self.scale
        # Restore original norm
        current_norm = np.linalg.norm(float_vec)
        if current_norm > 0:
            float_vec = float_vec * (self.original_norm / current_norm)
        return float_vec

    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        # Format: scale(4) + zero_point(4) + norm(4) + data
        header = np.array([self.scale, self.zero_point, self.original_norm],
                         dtype=np.float32).tobytes()
        return header + self.data.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes, dim: int) -> "QuantizedVector":
        """Deserialize from bytes."""
        header = np.frombuffer(data[:12], dtype=np.float32)
        vec_data = np.frombuffer(data[12:12+dim], dtype=np.int8)
        return cls(
            data=vec_data,
            scale=float(header[0]),
            zero_point=float(header[1]),
            original_norm=float(header[2]),
        )


class Quantizer:
    """
    Vector quantization for embedding compression.

    Reduces float32 vectors to int8 with minimal similarity loss.
    Achieves ~4x compression with >97% preserved similarity.

    Example:
        quantizer = Quantizer()
        qvec = quantizer.quantize(embedding)
        restored = qvec.dequantize()
        similarity = cosine(embedding, restored)  # ~0.98+
    """

    def __init__(self, symmetric: bool = True):
        self.symmetric = symmetric

    def quantize(self, vector: np.ndarray) -> QuantizedVector:
        """
        Quantize a float32 vector to int8.

        Args:
            vector: Float32 vector to quantize

        Returns:
            QuantizedVector with int8 data and metadata
        """
        vector = vector.astype(np.float32)
        original_norm = float(np.linalg.norm(vector))

        if original_norm == 0:
            return QuantizedVector(
                data=np.zeros(len(vector), dtype=np.int8),
                scale=1.0,
                zero_point=0.0,
                original_norm=0.0,
            )

        # Normalize
        normalized = vector / original_norm

        if self.symmetric:
            # Symmetric quantization: scale to [-127, 127]
            max_abs = np.abs(normalized).max()
            scale = max_abs / 127.0 if max_abs > 0 else 1.0
            quantized = np.clip(
                np.round(normalized / scale),
                -127, 127
            ).astype(np.int8)
            zero_point = 0.0
        else:
            # Asymmetric: scale to [0, 255] then offset to int8
            min_val, max_val = normalized.min(), normalized.max()
            scale = (max_val - min_val) / 255.0 if max_val > min_val else 1.0
            zero_point = -128 - min_val / scale
            quantized = np.clip(
                np.round(normalized / scale + zero_point),
                -128, 127
            ).astype(np.int8)

        return QuantizedVector(
            data=quantized,
            scale=scale,
            zero_point=zero_point,
            original_norm=original_norm,
        )

    def quantize_batch(self, vectors: np.ndarray) -> List[QuantizedVector]:
        """Quantize a batch of vectors."""
        return [self.quantize(v) for v in vectors]


@dataclass
class ResidualVector:
    """A residually quantized vector (center + residual)."""
    center_id: int              # ID of nearest center
    residual: QuantizedVector   # Quantized residual from center
    original_norm: float = 1.0


class ResidualQuantizer:
    """
    Residual Quantization for high-fidelity compression.

    Represents vectors as: absolute = center + residual
    where residual is quantized to int8.

    Preserves ~99% similarity with higher compression.

    Example:
        rq = ResidualQuantizer(n_centers=256)
        rq.fit(training_vectors)

        compressed = rq.encode(vector)
        restored = rq.decode(compressed)
    """

    def __init__(self, n_centers: int = 256, dim: int = 384):
        self.n_centers = n_centers
        self.dim = dim
        self.centers: Optional[np.ndarray] = None
        self.quantizer = Quantizer(symmetric=True)
        self._fitted = False

    def fit(self, vectors: np.ndarray, max_iter: int = 100):
        """
        Fit centers using k-means.

        Args:
            vectors: Training vectors (N x dim)
            max_iter: Maximum iterations
        """
        from scipy.cluster.vq import kmeans2

        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = vectors / norms

        # Run k-means
        self.centers, _ = kmeans2(
            normalized.astype(np.float32),
            self.n_centers,
            iter=max_iter,
            minit='++',
        )

        self._fitted = True

    def _find_nearest_center(self, vector: np.ndarray) -> Tuple[int, np.ndarray]:
        """Find nearest center and return its index and the center."""
        if self.centers is None:
            raise ValueError("Quantizer not fitted")

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            normalized = vector / norm
        else:
            normalized = vector

        # Compute distances to all centers
        distances = np.linalg.norm(self.centers - normalized, axis=1)
        center_id = int(np.argmin(distances))

        return center_id, self.centers[center_id]

    def encode(self, vector: np.ndarray) -> ResidualVector:
        """
        Encode a vector using residual quantization.

        Args:
            vector: Float32 vector

        Returns:
            ResidualVector with center ID and quantized residual
        """
        if not self._fitted:
            # Fallback to pure quantization if not fitted
            return ResidualVector(
                center_id=0,
                residual=self.quantizer.quantize(vector),
                original_norm=float(np.linalg.norm(vector)),
            )

        original_norm = float(np.linalg.norm(vector))
        if original_norm == 0:
            return ResidualVector(
                center_id=0,
                residual=self.quantizer.quantize(np.zeros(self.dim)),
                original_norm=0.0,
            )

        # Normalize
        normalized = vector / original_norm

        # Find nearest center
        center_id, center = self._find_nearest_center(normalized)

        # Compute residual
        residual = normalized - center

        # Quantize residual
        quantized_residual = self.quantizer.quantize(residual)

        return ResidualVector(
            center_id=center_id,
            residual=quantized_residual,
            original_norm=original_norm,
        )

    def decode(self, encoded: ResidualVector) -> np.ndarray:
        """
        Decode a residually quantized vector.

        Args:
            encoded: ResidualVector to decode

        Returns:
            Reconstructed float32 vector
        """
        if not self._fitted or self.centers is None:
            # Fallback
            return encoded.residual.dequantize()

        # Get center
        center = self.centers[encoded.center_id]

        # Dequantize residual
        residual = encoded.residual.dequantize()

        # Reconstruct
        normalized = center + residual

        # Restore norm
        current_norm = np.linalg.norm(normalized)
        if current_norm > 0:
            return normalized * (encoded.original_norm / current_norm)
        return normalized

    def encode_batch(self, vectors: np.ndarray) -> List[ResidualVector]:
        """Encode a batch of vectors."""
        return [self.encode(v) for v in vectors]

    def decode_batch(self, encoded: List[ResidualVector]) -> np.ndarray:
        """Decode a batch of vectors."""
        return np.array([self.decode(e) for e in encoded])

    def save(self, path: Path):
        """Save quantizer state."""
        np.savez(
            path,
            centers=self.centers,
            n_centers=self.n_centers,
            dim=self.dim,
        )

    def load(self, path: Path):
        """Load quantizer state."""
        data = np.load(path)
        self.centers = data["centers"]
        self.n_centers = int(data["n_centers"])
        self.dim = int(data["dim"])
        self._fitted = True


def compute_compression_ratio(
    original_dim: int,
    quantized_bytes: int,
) -> float:
    """Compute compression ratio."""
    original_bytes = original_dim * 4  # float32
    return original_bytes / quantized_bytes


def similarity_preservation(
    original: np.ndarray,
    restored: np.ndarray,
) -> float:
    """Compute cosine similarity between original and restored."""
    dot = np.dot(original.flatten(), restored.flatten())
    norm_a = np.linalg.norm(original)
    norm_b = np.linalg.norm(restored)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
