"""
GPIA Memory System with Embeddings

Provides semantic memory storage and retrieval using embeddings.
Supports multiple backends: in-memory, SQLite, and vector databases.
"""

import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Protocol
from pathlib import Path
import numpy as np


@dataclass
class MemoryEntry:
    """A single memory entry with embedding."""
    key: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "key": self.key,
            "content": self.content,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        """Deserialize from dictionary."""
        embedding = None
        if data.get("embedding"):
            embedding = np.array(data["embedding"])
        return cls(
            key=data["key"],
            content=data["content"],
            embedding=embedding,
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed", time.time()),
        )


class Embedder(Protocol):
    """Protocol for embedding providers."""

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        ...

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        ...


class SimpleEmbedder:
    """
    Simple hash-based embedder for testing/fallback.

    Not semantically meaningful but provides consistent vectors.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """Generate deterministic embedding from text hash."""
        # Create deterministic seed from text
        hash_bytes = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], 'big')

        # Generate reproducible random vector
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dimension).astype(np.float32)

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [self.embed(t) for t in texts]


class OllamaEmbedder:
    """
    Embedder using Ollama's embedding models.

    Requires Ollama running with an embedding model like nomic-embed-text.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.host = host.rstrip("/")
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Get dimension from a test embedding
            test = self.embed("test")
            self._dimension = len(test)
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding using Ollama."""
        import httpx

        response = httpx.post(
            f"{self.host}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=30.0,
        )
        response.raise_for_status()

        data = response.json()
        return np.array(data["embedding"], dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        # Ollama doesn't support batch, so we do sequential
        return [self.embed(t) for t in texts]


class OpenAIEmbedder:
    """
    Embedder using OpenAI's embedding API.

    Requires OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        import os
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._dimension = 1536 if "small" in model else 3072

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI."""
        import httpx

        response = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": text},
            timeout=30.0,
        )
        response.raise_for_status()

        data = response.json()
        return np.array(data["data"][0]["embedding"], dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        import httpx

        response = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=60.0,
        )
        response.raise_for_status()

        data = response.json()
        return [np.array(d["embedding"], dtype=np.float32) for d in data["data"]]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class Memory:
    """
    GPIA Memory System.

    Provides semantic storage and retrieval with:
    - Multiple embedding backends (Simple, Ollama, OpenAI)
    - Similarity-based retrieval
    - Persistence to disk
    - Access tracking for importance weighting
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        persist_path: Optional[Path] = None,
    ):
        self.embedder = embedder or SimpleEmbedder()
        self.persist_path = Path(persist_path) if persist_path else None
        self.entries: Dict[str, MemoryEntry] = {}

        # Load existing memories if path exists
        if self.persist_path and self.persist_path.exists():
            self.load()

    def store(
        self,
        key: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        embed: bool = True,
    ) -> MemoryEntry:
        """
        Store a memory entry.

        Args:
            key: Unique identifier for the memory
            content: The content to store
            metadata: Optional metadata dict
            embed: Whether to generate embedding (default True)

        Returns:
            The created MemoryEntry
        """
        embedding = None
        if embed:
            embedding = self.embedder.embed(content)

        entry = MemoryEntry(
            key=key,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
        )

        self.entries[key] = entry

        if self.persist_path:
            self.save()

        return entry

    def retrieve(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve a memory by key."""
        entry = self.entries.get(key)
        if entry:
            entry.access_count += 1
            entry.last_accessed = time.time()
        return entry

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[tuple[MemoryEntry, float]]:
        """
        Search memories by semantic similarity.

        Args:
            query: Search query
            top_k: Maximum number of results
            threshold: Minimum similarity threshold

        Returns:
            List of (entry, similarity) tuples, sorted by similarity
        """
        if not self.entries:
            return []

        # Embed query
        query_embedding = self.embedder.embed(query)

        # Compute similarities
        results = []
        for entry in self.entries.values():
            if entry.embedding is None:
                continue

            similarity = cosine_similarity(query_embedding, entry.embedding)
            if similarity >= threshold:
                results.append((entry, similarity))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        # Update access stats
        for entry, _ in results[:top_k]:
            entry.access_count += 1
            entry.last_accessed = time.time()

        return results[:top_k]

    def delete(self, key: str) -> bool:
        """Delete a memory by key."""
        if key in self.entries:
            del self.entries[key]
            if self.persist_path:
                self.save()
            return True
        return False

    def clear(self):
        """Clear all memories."""
        self.entries.clear()
        if self.persist_path:
            self.save()

    def save(self):
        """Save memories to disk."""
        if not self.persist_path:
            return

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "entries": {k: v.to_dict() for k, v in self.entries.items()},
        }

        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load memories from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        with open(self.persist_path) as f:
            data = json.load(f)

        self.entries = {
            k: MemoryEntry.from_dict(v)
            for k, v in data.get("entries", {}).items()
        }

    def stats(self) -> dict:
        """Get memory statistics."""
        return {
            "total_entries": len(self.entries),
            "with_embeddings": sum(
                1 for e in self.entries.values() if e.embedding is not None
            ),
            "embedding_dimension": self.embedder.dimension,
            "persist_path": str(self.persist_path) if self.persist_path else None,
        }
