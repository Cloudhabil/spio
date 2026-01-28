"""
Memory Sphere System - Mitosis/Meiosis

Dynamic tree rebalancing for memory management.
Spheres split (mitosis) when over-pressurized and
merge (meiosis) when under-utilized.

Features:
- KMeans-based sphere splitting
- Neighbor-based sphere merging
- Quantized residual preservation
- Atomic operations
"""

import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import numpy as np

# Import from parent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from sovereign_pio.constants import PHI, BETA
from core.quantization import Quantizer, QuantizedVector


@dataclass
class MemoryItem:
    """A single item in a memory sphere."""
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "access_count": self.access_count,
        }


@dataclass
class Sphere:
    """
    A memory sphere containing related items.

    Spheres have a center embedding and contain items
    within a certain radius. They can split (mitosis)
    or merge (meiosis) based on pressure.
    """
    id: str
    center: np.ndarray
    items: List[MemoryItem] = field(default_factory=list)

    # Pressure thresholds
    max_items: int = 100          # Mitosis threshold
    min_items: int = 10           # Meiosis threshold

    # State
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    mitosis_count: int = 0
    meiosis_count: int = 0

    # Parent/child relationships
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    @property
    def pressure(self) -> float:
        """
        Compute sphere pressure.

        Pressure > 1.0 indicates need for mitosis.
        Pressure < 0.1 indicates candidate for meiosis.
        """
        if self.max_items == 0:
            return 0.0
        return len(self.items) / self.max_items

    @property
    def needs_mitosis(self) -> bool:
        """Check if sphere needs to split."""
        return len(self.items) >= self.max_items

    @property
    def needs_meiosis(self) -> bool:
        """Check if sphere should merge."""
        return len(self.items) < self.min_items

    def add_item(self, item: MemoryItem):
        """Add an item to the sphere."""
        self.items.append(item)
        self._update_center()
        self.last_access = time.time()

    def remove_item(self, item_id: str) -> Optional[MemoryItem]:
        """Remove an item from the sphere."""
        for i, item in enumerate(self.items):
            if item.id == item_id:
                removed = self.items.pop(i)
                self._update_center()
                return removed
        return None

    def _update_center(self):
        """Recompute center as mean of item embeddings."""
        if not self.items:
            return

        embeddings = np.array([item.embedding for item in self.items])
        self.center = np.mean(embeddings, axis=0)

    def distance_to(self, other: "Sphere") -> float:
        """Compute distance to another sphere."""
        return float(np.linalg.norm(self.center - other.center))

    def similarity_to(self, embedding: np.ndarray) -> float:
        """Compute cosine similarity to an embedding."""
        dot = np.dot(self.center, embedding)
        norm_c = np.linalg.norm(self.center)
        norm_e = np.linalg.norm(embedding)
        if norm_c == 0 or norm_e == 0:
            return 0.0
        return float(dot / (norm_c * norm_e))


class MemorySphereManager:
    """
    Memory Sphere Manager with Mitosis/Meiosis.

    Manages a collection of memory spheres with automatic
    rebalancing based on pressure.

    Example:
        manager = MemorySphereManager()

        # Add items (auto-assigns to spheres)
        manager.add("id1", "Hello world", embedding1)
        manager.add("id2", "Another memory", embedding2)

        # Rebalance if needed
        manager.rebalance()

        # Search
        results = manager.search(query_embedding, top_k=5)
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        max_spheres: int = 1000,
        mitosis_threshold: int = 100,
        meiosis_threshold: int = 10,
        persist_path: Optional[Path] = None,
    ):
        self.embedding_dim = embedding_dim
        self.max_spheres = max_spheres
        self.mitosis_threshold = mitosis_threshold
        self.meiosis_threshold = meiosis_threshold
        self.persist_path = Path(persist_path) if persist_path else None

        self.spheres: Dict[str, Sphere] = {}
        self.quantizer = Quantizer()

        # Statistics
        self.total_items = 0
        self.total_mitosis = 0
        self.total_meiosis = 0

        # ID counter
        self._next_sphere_id = 0
        self._next_item_id = 0

    def _generate_sphere_id(self) -> str:
        self._next_sphere_id += 1
        return f"sphere_{self._next_sphere_id:06d}"

    def _generate_item_id(self) -> str:
        self._next_item_id += 1
        return f"item_{self._next_item_id:08d}"

    def _find_nearest_sphere(self, embedding: np.ndarray) -> Optional[Sphere]:
        """Find the nearest sphere to an embedding."""
        if not self.spheres:
            return None

        best_sphere = None
        best_similarity = -1

        for sphere in self.spheres.values():
            sim = sphere.similarity_to(embedding)
            if sim > best_similarity:
                best_similarity = sim
                best_sphere = sphere

        return best_sphere

    def _create_sphere(self, center: np.ndarray, parent_id: Optional[str] = None) -> Sphere:
        """Create a new sphere."""
        sphere_id = self._generate_sphere_id()

        sphere = Sphere(
            id=sphere_id,
            center=center.copy(),
            max_items=self.mitosis_threshold,
            min_items=self.meiosis_threshold,
            parent_id=parent_id,
        )

        self.spheres[sphere_id] = sphere
        return sphere

    def add(
        self,
        content: str,
        embedding: np.ndarray,
        item_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add an item to the memory system.

        Automatically assigns to the nearest sphere or
        creates a new one if needed.

        Returns item ID.
        """
        item_id = item_id or self._generate_item_id()

        item = MemoryItem(
            id=item_id,
            content=content,
            embedding=embedding.astype(np.float32),
            metadata=metadata or {},
        )

        # Find or create sphere
        sphere = self._find_nearest_sphere(embedding)

        if sphere is None or sphere.similarity_to(embedding) < 0.5:
            # Create new sphere if none exists or poor fit
            if len(self.spheres) < self.max_spheres:
                sphere = self._create_sphere(embedding)
            elif sphere is None:
                # Use first available sphere
                sphere = next(iter(self.spheres.values()))

        sphere.add_item(item)
        self.total_items += 1

        # Check for mitosis
        if sphere.needs_mitosis:
            self._mitosis(sphere)

        return item_id

    def _mitosis(self, sphere: Sphere):
        """
        Split a sphere into two (mitosis).

        Uses KMeans-style splitting based on item embeddings.
        """
        if len(sphere.items) < 2:
            return

        # Get embeddings
        embeddings = np.array([item.embedding for item in sphere.items])

        # Simple 2-means clustering
        # Initialize with furthest points
        idx1 = 0
        distances = np.linalg.norm(embeddings - embeddings[idx1], axis=1)
        idx2 = int(np.argmax(distances))

        center1 = embeddings[idx1]
        center2 = embeddings[idx2]

        # Assign items to clusters
        cluster1_items = []
        cluster2_items = []

        for item in sphere.items:
            d1 = np.linalg.norm(item.embedding - center1)
            d2 = np.linalg.norm(item.embedding - center2)

            if d1 <= d2:
                cluster1_items.append(item)
            else:
                cluster2_items.append(item)

        # Update original sphere with cluster 1
        sphere.items = cluster1_items
        sphere._update_center()
        sphere.mitosis_count += 1

        # Create new sphere with cluster 2
        if cluster2_items:
            new_center = np.mean([i.embedding for i in cluster2_items], axis=0)
            new_sphere = self._create_sphere(new_center, parent_id=sphere.id)
            new_sphere.items = cluster2_items
            sphere.children_ids.append(new_sphere.id)

        self.total_mitosis += 1

    def _meiosis(self, sphere: Sphere):
        """
        Merge a sparse sphere with its nearest neighbor (meiosis).
        """
        if len(self.spheres) <= 1:
            return

        # Find nearest neighbor
        nearest = None
        nearest_distance = float('inf')

        for other in self.spheres.values():
            if other.id == sphere.id:
                continue

            dist = sphere.distance_to(other)
            if dist < nearest_distance:
                nearest_distance = dist
                nearest = other

        if nearest is None:
            return

        # Merge items into nearest
        for item in sphere.items:
            nearest.add_item(item)

        nearest.meiosis_count += 1

        # Remove empty sphere
        del self.spheres[sphere.id]

        self.total_meiosis += 1

    def rebalance(self):
        """
        Rebalance all spheres.

        Performs mitosis on over-pressurized spheres
        and meiosis on under-utilized spheres.
        """
        # Collect spheres needing action
        needs_mitosis = [s for s in self.spheres.values() if s.needs_mitosis]
        needs_meiosis = [s for s in self.spheres.values() if s.needs_meiosis]

        # Mitosis first
        for sphere in needs_mitosis:
            if sphere.id in self.spheres:  # May have been affected
                self._mitosis(sphere)

        # Then meiosis
        for sphere in needs_meiosis:
            if sphere.id in self.spheres and sphere.needs_meiosis:
                self._meiosis(sphere)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Search for items similar to query.

        Returns list of (item, similarity) tuples.
        """
        results = []

        # Search all spheres
        for sphere in self.spheres.values():
            for item in sphere.items:
                dot = np.dot(query_embedding, item.embedding)
                norm_q = np.linalg.norm(query_embedding)
                norm_i = np.linalg.norm(item.embedding)

                if norm_q > 0 and norm_i > 0:
                    similarity = float(dot / (norm_q * norm_i))
                    results.append((item, similarity))
                    item.access_count += 1

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def get_item(self, item_id: str) -> Optional[MemoryItem]:
        """Get an item by ID."""
        for sphere in self.spheres.values():
            for item in sphere.items:
                if item.id == item_id:
                    return item
        return None

    def remove_item(self, item_id: str) -> bool:
        """Remove an item by ID."""
        for sphere in self.spheres.values():
            item = sphere.remove_item(item_id)
            if item:
                self.total_items -= 1
                return True
        return False

    def stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        pressures = [s.pressure for s in self.spheres.values()]

        return {
            "total_spheres": len(self.spheres),
            "total_items": self.total_items,
            "total_mitosis": self.total_mitosis,
            "total_meiosis": self.total_meiosis,
            "avg_pressure": np.mean(pressures) if pressures else 0,
            "max_pressure": max(pressures) if pressures else 0,
            "mitosis_threshold": self.mitosis_threshold,
            "meiosis_threshold": self.meiosis_threshold,
        }
