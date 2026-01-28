"""
GPIA - General Purpose Intelligence Architecture

The intelligence layer of Sovereign PIO.
Handles reasoning, memory, wormhole routing, and multi-model orchestration.

Components:
- Memory: Embedding-based semantic storage
- Reasoning: LLM integration with dimension routing
- Dense State: Sphere-based context retrieval
- Causal Linker: Hidden connection discovery
- Ignorance: Knowledge gap cartography
- Memory Spheres: Dynamic mitosis/meiosis rebalancing
"""

from sovereign_pio.constants import PHI, DIMENSION_NAMES, LUCAS_NUMBERS

# Memory
from .memory import (
    Memory,
    MemoryEntry,
    Embedder,
    SimpleEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    cosine_similarity,
)

# Reasoning
from .reasoning import (
    ReasoningEngine,
    ReasoningResult,
    LLMClient,
    ModelConfig,
    ModelProvider,
    MultiModelOrchestrator,
)

# Dense State
from .dense_state import (
    DenseStateMemory,
    MemorySphere,
    RetrievalResult,
    RESONANCE_THRESHOLD,
)

# Causal Linker
from .causal_linker import (
    CausalLinker,
    CausalLink,
    NeutrinoString,
)

# Ignorance Cartography
from .ignorance import (
    IgnoranceCartographer,
    IgnorancePoint,
    IgnoranceType,
    DarkSector,
    N4Boundary,
    APERTURE,
    GENESIS,
)

# Memory Spheres
from .memory_spheres import (
    MemorySphereManager,
    Sphere,
    MemoryItem,
)

__all__ = [
    # Constants
    "PHI",
    "DIMENSION_NAMES",
    "LUCAS_NUMBERS",
    # Memory
    "Memory",
    "MemoryEntry",
    "Embedder",
    "SimpleEmbedder",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "cosine_similarity",
    # Reasoning
    "ReasoningEngine",
    "ReasoningResult",
    "LLMClient",
    "ModelConfig",
    "ModelProvider",
    "MultiModelOrchestrator",
    # Dense State
    "DenseStateMemory",
    "MemorySphere",
    "RetrievalResult",
    "RESONANCE_THRESHOLD",
    # Causal Linker
    "CausalLinker",
    "CausalLink",
    "NeutrinoString",
    # Ignorance
    "IgnoranceCartographer",
    "IgnorancePoint",
    "IgnoranceType",
    "DarkSector",
    "N4Boundary",
    "APERTURE",
    "GENESIS",
    # Memory Spheres
    "MemorySphereManager",
    "Sphere",
    "MemoryItem",
    # Legacy
    "GPIAEngine",
    "WormholeRouter",
]


# Legacy aliases
class WormholeRouter:
    """Legacy alias - use ReasoningEngine instead."""
    def __init__(self):
        self.dimensions = DIMENSION_NAMES
        self.capacities = {i + 1: cap for i, cap in enumerate(LUCAS_NUMBERS)}

    def route(self, task_type: str) -> int:
        routing_map = {
            "perception": 1, "attention": 2, "security": 3, "stability": 4,
            "compression": 5, "harmony": 6, "reasoning": 7, "prediction": 8,
            "creativity": 9, "wisdom": 10, "integration": 11, "unification": 12,
        }
        return routing_map.get(task_type.lower(), 7)

    def get_capacity(self, dimension: int) -> int:
        return self.capacities.get(dimension, 1)


class GPIAEngine:
    """Legacy alias - use ReasoningEngine instead."""
    def __init__(self):
        self.memory = Memory()
        self.router = WormholeRouter()
        self.phi = PHI

    async def reason(self, query: str, context: dict = None) -> dict:
        dimension = self.router.route("reasoning")
        return {
            "query": query,
            "dimension": dimension,
            "dimension_name": DIMENSION_NAMES[dimension],
            "capacity": self.router.get_capacity(dimension),
            "result": None,
        }
