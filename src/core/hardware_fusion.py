"""
Hardware Fusion - Direct Hardware Access

Transforms Ball Tree → NumPy tensor for direct hardware access.
Enables vectorized operations at C++ speeds.

Features:
- Tree flattening to tensor
- Memory-mapped NPY files
- Vectorized dot product search
- Zero-copy operations
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import numpy as np


@dataclass
class FusedIndex:
    """
    A fused index for fast vector search.

    Converts tree-based structures to flat tensors
    for hardware-optimized search.
    """
    embeddings: np.ndarray        # (N, D) float32 tensor
    ids: List[str]                # Item IDs
    metadata: Dict[str, Any]      # Index metadata

    created_at: float = field(default_factory=time.time)

    @property
    def size(self) -> int:
        return len(self.ids)

    @property
    def dimension(self) -> int:
        return self.embeddings.shape[1] if len(self.embeddings.shape) > 1 else 0

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Vectorized similarity search.

        Uses optimized matrix multiplication for speed.
        """
        if self.size == 0:
            return []

        # Normalize query
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        # Normalize embeddings (if not already)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = self.embeddings / norms

        # Vectorized dot product
        similarities = np.dot(normalized, query)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = [
            (self.ids[i], float(similarities[i]))
            for i in top_indices
        ]

        return results


class HardwareFusion:
    """
    Hardware Fusion Engine.

    Transforms tree-based memory structures into
    flat tensors for hardware-optimized operations.

    Example:
        fusion = HardwareFusion()

        # Add vectors
        fusion.add("id1", embedding1)
        fusion.add("id2", embedding2)

        # Build fused index
        index = fusion.build()

        # Fast search
        results = index.search(query, top_k=10)

        # Save for zero-copy loading
        fusion.save("index.npy")
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        dtype: np.dtype = np.float32,
    ):
        self.embedding_dim = embedding_dim
        self.dtype = dtype

        self.embeddings: List[np.ndarray] = []
        self.ids: List[str] = []
        self.metadata: Dict[str, Any] = {}

        self._fused: Optional[FusedIndex] = None
        self._dirty = False

    def add(
        self,
        item_id: str,
        embedding: np.ndarray,
    ):
        """Add an embedding to the fusion."""
        self.embeddings.append(embedding.astype(self.dtype))
        self.ids.append(item_id)
        self._dirty = True

    def add_batch(
        self,
        item_ids: List[str],
        embeddings: np.ndarray,
    ):
        """Add a batch of embeddings."""
        for i, item_id in enumerate(item_ids):
            self.embeddings.append(embeddings[i].astype(self.dtype))
            self.ids.append(item_id)
        self._dirty = True

    def build(self) -> FusedIndex:
        """
        Build the fused index.

        Concatenates all embeddings into a single tensor.
        """
        if not self._dirty and self._fused is not None:
            return self._fused

        if not self.embeddings:
            tensor = np.zeros((0, self.embedding_dim), dtype=self.dtype)
        else:
            tensor = np.vstack(self.embeddings)

        self._fused = FusedIndex(
            embeddings=tensor,
            ids=self.ids.copy(),
            metadata={
                "dimension": self.embedding_dim,
                "dtype": str(self.dtype),
                "size": len(self.ids),
            },
        )

        self._dirty = False
        return self._fused

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Search the fused index."""
        index = self.build()
        return index.search(query, top_k)

    def save(self, path: Path):
        """
        Save fused index to disk.

        Uses NPY format for zero-copy memory mapping.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        index = self.build()

        # Save embeddings
        np.save(str(path), index.embeddings)

        # Save metadata
        meta_path = path.with_suffix(".meta.json")
        import json
        with open(meta_path, "w") as f:
            json.dump({
                "ids": index.ids,
                "metadata": index.metadata,
            }, f)

    def load(self, path: Path):
        """
        Load fused index from disk.

        Uses memory mapping for zero-copy access.
        """
        path = Path(path)

        # Load embeddings with memory mapping
        embeddings = np.load(str(path), mmap_mode='r')

        # Load metadata
        meta_path = path.with_suffix(".meta.json")
        import json
        with open(meta_path) as f:
            meta = json.load(f)

        self._fused = FusedIndex(
            embeddings=embeddings,
            ids=meta["ids"],
            metadata=meta["metadata"],
        )

        self.embeddings = list(embeddings)
        self.ids = meta["ids"]
        self._dirty = False

    def clear(self):
        """Clear all data."""
        self.embeddings.clear()
        self.ids.clear()
        self._fused = None
        self._dirty = False

    def stats(self) -> Dict[str, Any]:
        """Get fusion statistics."""
        index = self.build()

        memory_bytes = index.embeddings.nbytes if index.embeddings.size > 0 else 0

        return {
            "size": index.size,
            "dimension": index.dimension,
            "dtype": str(self.dtype),
            "memory_mb": memory_bytes / (1024 * 1024),
            "dirty": self._dirty,
        }


class BatchProcessor:
    """
    Batch processor for hardware-optimized operations.

    Processes large batches of vectors efficiently
    using vectorized operations.
    """

    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size

    def process_similarities(
        self,
        queries: np.ndarray,
        index: FusedIndex,
        top_k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """
        Process multiple queries in batches.

        Returns results for each query.
        """
        results = []

        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            batch_results = self._process_batch(batch, index, top_k)
            results.extend(batch_results)

        return results

    def _process_batch(
        self,
        queries: np.ndarray,
        index: FusedIndex,
        top_k: int,
    ) -> List[List[Tuple[str, float]]]:
        """Process a single batch."""
        if index.size == 0:
            return [[] for _ in range(len(queries))]

        # Normalize queries
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        norms[norms == 0] = 1
        queries_normalized = queries / norms

        # Normalize index embeddings
        idx_norms = np.linalg.norm(index.embeddings, axis=1, keepdims=True)
        idx_norms[idx_norms == 0] = 1
        idx_normalized = index.embeddings / idx_norms

        # Batch matrix multiplication: (Q, D) @ (D, N) -> (Q, N)
        similarities = np.dot(queries_normalized, idx_normalized.T)

        # Get top-k for each query
        results = []
        for sim_row in similarities:
            top_indices = np.argsort(sim_row)[-top_k:][::-1]
            query_results = [
                (index.ids[i], float(sim_row[i]))
                for i in top_indices
            ]
            results.append(query_results)

        return results


def benchmark_fusion(
    n_vectors: int = 10000,
    dimension: int = 384,
    n_queries: int = 100,
    top_k: int = 10,
) -> Dict[str, float]:
    """
    Benchmark fusion search performance.

    Returns timing statistics.
    """
    # Create random data
    np.random.seed(42)
    embeddings = np.random.randn(n_vectors, dimension).astype(np.float32)
    ids = [f"id_{i}" for i in range(n_vectors)]
    queries = np.random.randn(n_queries, dimension).astype(np.float32)

    # Build index
    fusion = HardwareFusion(embedding_dim=dimension)
    fusion.add_batch(ids, embeddings)

    build_start = time.perf_counter()
    index = fusion.build()
    build_time = time.perf_counter() - build_start

    # Single query benchmark
    single_times = []
    for q in queries[:10]:
        start = time.perf_counter()
        _ = index.search(q, top_k)
        single_times.append(time.perf_counter() - start)

    # Batch benchmark
    processor = BatchProcessor(batch_size=100)
    batch_start = time.perf_counter()
    _ = processor.process_similarities(queries, index, top_k)
    batch_time = time.perf_counter() - batch_start

    return {
        "n_vectors": n_vectors,
        "dimension": dimension,
        "n_queries": n_queries,
        "build_time_ms": build_time * 1000,
        "avg_single_query_ms": np.mean(single_times) * 1000,
        "batch_query_ms": batch_time * 1000,
        "queries_per_second": n_queries / batch_time,
    }
