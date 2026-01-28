"""
Causal Linker - Hidden Connection Discovery

Detects hidden connections between memory spheres.
Builds "Neutrino Strings" for metaphorical reasoning.

Features:
- Resonance detection via cosine similarity
- Causal link discovery (cosine > 0.7)
- Neutrino string construction
- Link persistence and analysis
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import numpy as np


@dataclass
class CausalLink:
    """A discovered causal connection between concepts."""
    source_id: str
    target_id: str
    resonance: float  # Cosine similarity
    discovered_at: float = field(default_factory=time.time)
    link_type: str = "resonance"  # resonance, temporal, explicit
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "resonance": self.resonance,
            "type": self.link_type,
            "discovered_at": self.discovered_at,
        }


@dataclass
class NeutrinoString:
    """
    A chain of causally linked concepts.

    Named for the way neutrinos pass through matter -
    these connections pass through the normal semantic space.
    """
    id: str
    links: List[CausalLink] = field(default_factory=list)
    total_resonance: float = 0.0
    created_at: float = field(default_factory=time.time)

    def add_link(self, link: CausalLink):
        """Add a link to the string."""
        self.links.append(link)
        self.total_resonance = sum(l.resonance for l in self.links)

    def strength(self) -> float:
        """Calculate string strength (average resonance)."""
        if not self.links:
            return 0.0
        return self.total_resonance / len(self.links)

    def path(self) -> List[str]:
        """Get the path of concept IDs."""
        if not self.links:
            return []
        path = [self.links[0].source_id]
        for link in self.links:
            path.append(link.target_id)
        return path

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "path": self.path(),
            "strength": self.strength(),
            "links": [l.to_dict() for l in self.links],
        }


class CausalLinker:
    """
    Causal Linker - Discovers Hidden Connections.

    Finds non-obvious relationships between concepts based on
    embedding resonance, even when spatially distant in memory.

    Example:
        linker = CausalLinker()

        # Add concepts with embeddings
        linker.add_concept("water", water_embedding)
        linker.add_concept("ice", ice_embedding)
        linker.add_concept("steam", steam_embedding)

        # Discover links
        links = linker.discover_links()
        for link in links:
            print(f"{link.source_id} <-> {link.target_id}: {link.resonance:.3f}")

        # Build neutrino strings
        strings = linker.build_strings()
    """

    # Threshold for causal link (high resonance)
    LINK_THRESHOLD = 0.7

    # Threshold for weak connection
    WEAK_THRESHOLD = 0.5

    def __init__(
        self,
        link_threshold: float = LINK_THRESHOLD,
        persist_path: Optional[Path] = None,
    ):
        self.link_threshold = link_threshold
        self.persist_path = Path(persist_path) if persist_path else None

        self.concepts: Dict[str, np.ndarray] = {}  # id -> embedding
        self.concept_metadata: Dict[str, Dict[str, Any]] = {}
        self.links: List[CausalLink] = []
        self.strings: List[NeutrinoString] = []

        # Statistics
        self.total_discoveries = 0
        self.string_count = 0

        if self.persist_path and self.persist_path.exists():
            self._load()

    def add_concept(
        self,
        concept_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a concept with its embedding."""
        self.concepts[concept_id] = embedding.astype(np.float32)
        self.concept_metadata[concept_id] = metadata or {}

    def remove_concept(self, concept_id: str):
        """Remove a concept."""
        self.concepts.pop(concept_id, None)
        self.concept_metadata.pop(concept_id, None)

        # Remove related links
        self.links = [
            l for l in self.links
            if l.source_id != concept_id and l.target_id != concept_id
        ]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        dot = np.dot(a, b)
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def discover_links(
        self,
        concept_ids: Optional[List[str]] = None,
    ) -> List[CausalLink]:
        """
        Discover causal links between concepts.

        Args:
            concept_ids: Specific concepts to analyze (default: all)

        Returns:
            List of discovered CausalLinks
        """
        ids = concept_ids or list(self.concepts.keys())
        new_links = []

        # Compare all pairs
        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                emb_a = self.concepts.get(id_a)
                emb_b = self.concepts.get(id_b)

                if emb_a is None or emb_b is None:
                    continue

                resonance = self._cosine_similarity(emb_a, emb_b)

                if resonance >= self.link_threshold:
                    # Check if link already exists
                    exists = any(
                        (l.source_id == id_a and l.target_id == id_b) or
                        (l.source_id == id_b and l.target_id == id_a)
                        for l in self.links
                    )

                    if not exists:
                        link = CausalLink(
                            source_id=id_a,
                            target_id=id_b,
                            resonance=resonance,
                        )
                        self.links.append(link)
                        new_links.append(link)
                        self.total_discoveries += 1

        if self.persist_path:
            self._save()

        return new_links

    def get_connections(
        self,
        concept_id: str,
        min_resonance: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Get all connections for a concept.

        Returns:
            List of (connected_id, resonance) tuples
        """
        connections = []

        for link in self.links:
            if link.resonance < min_resonance:
                continue

            if link.source_id == concept_id:
                connections.append((link.target_id, link.resonance))
            elif link.target_id == concept_id:
                connections.append((link.source_id, link.resonance))

        # Sort by resonance
        connections.sort(key=lambda x: x[1], reverse=True)
        return connections

    def build_strings(self, min_length: int = 2, max_length: int = 5) -> List[NeutrinoString]:
        """
        Build neutrino strings from discovered links.

        Finds chains of connected concepts that form
        coherent reasoning paths.
        """
        # Build adjacency list
        adjacency: Dict[str, List[Tuple[str, float]]] = {}
        for link in self.links:
            if link.source_id not in adjacency:
                adjacency[link.source_id] = []
            if link.target_id not in adjacency:
                adjacency[link.target_id] = []

            adjacency[link.source_id].append((link.target_id, link.resonance))
            adjacency[link.target_id].append((link.source_id, link.resonance))

        # DFS to find paths
        strings = []
        visited_paths = set()

        def dfs(current: str, path: List[str], resonances: List[float]):
            if len(path) >= min_length:
                # Create string
                path_key = "->".join(sorted(path))
                if path_key not in visited_paths:
                    visited_paths.add(path_key)

                    string = NeutrinoString(id=f"string_{len(strings):04d}")
                    for i in range(len(path) - 1):
                        string.add_link(CausalLink(
                            source_id=path[i],
                            target_id=path[i + 1],
                            resonance=resonances[i],
                        ))
                    strings.append(string)

            if len(path) >= max_length:
                return

            for neighbor, resonance in adjacency.get(current, []):
                if neighbor not in path:
                    dfs(neighbor, path + [neighbor], resonances + [resonance])

        # Start DFS from each concept
        for concept_id in adjacency:
            dfs(concept_id, [concept_id], [])

        self.strings = strings
        self.string_count = len(strings)

        return strings

    def find_path(
        self,
        source_id: str,
        target_id: str,
    ) -> Optional[NeutrinoString]:
        """
        Find a path between two concepts.

        Uses BFS to find shortest path through causal links.
        """
        if source_id not in self.concepts or target_id not in self.concepts:
            return None

        # Build adjacency
        adjacency: Dict[str, List[Tuple[str, float]]] = {}
        for link in self.links:
            if link.source_id not in adjacency:
                adjacency[link.source_id] = []
            if link.target_id not in adjacency:
                adjacency[link.target_id] = []

            adjacency[link.source_id].append((link.target_id, link.resonance))
            adjacency[link.target_id].append((link.source_id, link.resonance))

        # BFS
        from collections import deque

        queue = deque([(source_id, [source_id], [])])
        visited = {source_id}

        while queue:
            current, path, resonances = queue.popleft()

            if current == target_id:
                # Found path
                string = NeutrinoString(id=f"path_{source_id}_{target_id}")
                for i in range(len(path) - 1):
                    string.add_link(CausalLink(
                        source_id=path[i],
                        target_id=path[i + 1],
                        resonance=resonances[i],
                    ))
                return string

            for neighbor, resonance in adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((
                        neighbor,
                        path + [neighbor],
                        resonances + [resonance]
                    ))

        return None

    def _save(self):
        """Save links to disk."""
        if not self.persist_path:
            return

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "links": [l.to_dict() for l in self.links],
            "stats": {
                "total_discoveries": self.total_discoveries,
                "concept_count": len(self.concepts),
            },
        }

        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load links from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        with open(self.persist_path) as f:
            data = json.load(f)

        for link_data in data.get("links", []):
            link = CausalLink(
                source_id=link_data["source"],
                target_id=link_data["target"],
                resonance=link_data["resonance"],
                discovered_at=link_data.get("discovered_at", time.time()),
                link_type=link_data.get("type", "resonance"),
            )
            self.links.append(link)

        stats = data.get("stats", {})
        self.total_discoveries = stats.get("total_discoveries", len(self.links))

    def stats(self) -> Dict[str, Any]:
        """Get linker statistics."""
        return {
            "concept_count": len(self.concepts),
            "link_count": len(self.links),
            "string_count": self.string_count,
            "total_discoveries": self.total_discoveries,
            "link_threshold": self.link_threshold,
            "avg_resonance": (
                sum(l.resonance for l in self.links) / max(1, len(self.links))
            ),
        }
