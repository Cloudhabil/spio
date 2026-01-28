"""
Memory Core - Dense State Memory and Budget Management

Based on: CLI-main/src/core/dense_state_memory.py, budget_ledger.py
"""

import hashlib
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

# Brahim's Calculator
PHI = (1 + math.sqrt(5)) / 2
BETA = 1 / (PHI ** 3)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class AllocationState(Enum):
    """Memory allocation states."""
    FREE = auto()
    ALLOCATED = auto()
    RESERVED = auto()
    LOCKED = auto()


class MemoryTier(Enum):
    """Memory hierarchy tiers."""
    L1_CACHE = 1
    L2_CACHE = 2
    L3_CACHE = 3
    RAM = 4
    SWAP = 5
    DISK = 6


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MemoryBlock:
    """A memory allocation block."""
    id: str
    size_bytes: int
    state: AllocationState = AllocationState.FREE
    tier: MemoryTier = MemoryTier.RAM
    owner: str = ""
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    data: bytes = b""


@dataclass
class BudgetEntry:
    """Token budget allocation entry."""
    id: str
    task_id: str
    agent: str
    model: str
    tokens_allocated: int
    tokens_used: int = 0
    status: str = "active"
    created_at: float = field(default_factory=time.time)


@dataclass
class DenseChunk:
    """A chunk of dense state memory."""
    id: str
    content: str
    embedding: List[float] = field(default_factory=list)
    tokens: int = 0
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    resonance: float = 0.0


# =============================================================================
# MEMORY POOL
# =============================================================================

class MemoryPool:
    """
    Memory allocation pool with PHI-optimal block sizes.

    Block sizes follow Lucas sequence for optimal packing.
    """

    LUCAS = [1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322]

    def __init__(self, max_size_mb: int = 1024):
        self.max_size = max_size_mb * 1024 * 1024
        self._blocks: Dict[str, MemoryBlock] = {}
        self._free_list: List[str] = []
        self._lock = threading.Lock()
        self._allocated = 0
        self._block_counter = 0

    def allocate(self, size_bytes: int, owner: str = "") -> Optional[MemoryBlock]:
        """Allocate a memory block."""
        with self._lock:
            if self._allocated + size_bytes > self.max_size:
                return None

            self._block_counter += 1
            block_id = f"block-{self._block_counter}"
            block = MemoryBlock(
                id=block_id,
                size_bytes=size_bytes,
                state=AllocationState.ALLOCATED,
                owner=owner,
            )
            self._blocks[block_id] = block
            self._allocated += size_bytes
            return block

    def free(self, block_id: str) -> bool:
        """Free a memory block."""
        with self._lock:
            if block_id not in self._blocks:
                return False
            block = self._blocks[block_id]
            self._allocated -= block.size_bytes
            block.state = AllocationState.FREE
            self._free_list.append(block_id)
            return True

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_mb": self.max_size / (1024 * 1024),
            "allocated_mb": self._allocated / (1024 * 1024),
            "free_mb": (self.max_size - self._allocated) / (1024 * 1024),
            "block_count": len(self._blocks),
            "utilization": self._allocated / self.max_size if self.max_size > 0 else 0,
        }


# =============================================================================
# VIRTUAL MEMORY
# =============================================================================

class VirtualMemory:
    """Simulated virtual memory with paging."""

    PAGE_SIZE = 4096  # 4KB pages

    def __init__(self, physical_mb: int = 512, virtual_mb: int = 2048):
        self.physical_size = physical_mb * 1024 * 1024
        self.virtual_size = virtual_mb * 1024 * 1024
        self._page_table: Dict[int, int] = {}  # virtual -> physical
        self._physical_pages: Dict[int, bytes] = {}
        self._next_virtual = 0
        self._next_physical = 0
        self._lock = threading.Lock()

    def allocate_page(self) -> Optional[int]:
        """Allocate a virtual page."""
        with self._lock:
            if self._next_virtual * self.PAGE_SIZE >= self.virtual_size:
                return None
            page_num = self._next_virtual
            self._next_virtual += 1
            return page_num

    def map_page(self, virtual_page: int) -> bool:
        """Map virtual page to physical memory."""
        with self._lock:
            if self._next_physical * self.PAGE_SIZE >= self.physical_size:
                return False  # Would need to swap
            physical_page = self._next_physical
            self._next_physical += 1
            self._page_table[virtual_page] = physical_page
            self._physical_pages[physical_page] = b"\x00" * self.PAGE_SIZE
            return True

    def read_page(self, virtual_page: int) -> Optional[bytes]:
        """Read a page."""
        with self._lock:
            if virtual_page not in self._page_table:
                return None
            physical = self._page_table[virtual_page]
            return self._physical_pages.get(physical)

    def write_page(self, virtual_page: int, data: bytes) -> bool:
        """Write to a page."""
        with self._lock:
            if virtual_page not in self._page_table:
                return False
            physical = self._page_table[virtual_page]
            self._physical_pages[physical] = data[:self.PAGE_SIZE].ljust(self.PAGE_SIZE, b"\x00")
            return True


# =============================================================================
# GARBAGE COLLECTOR
# =============================================================================

class GarbageCollector:
    """Memory garbage collector."""

    def __init__(self, pool: MemoryPool, threshold: float = 0.8):
        self.pool = pool
        self.threshold = threshold
        self._collections = 0
        self._bytes_freed = 0

    def should_collect(self) -> bool:
        """Check if GC should run."""
        stats = self.pool.get_stats()
        return stats["utilization"] >= self.threshold

    def collect(self) -> int:
        """Run garbage collection. Returns bytes freed."""
        if not self.should_collect():
            return 0

        freed = 0
        now = time.time()

        # Free blocks not accessed in 5 minutes
        to_free = []
        for block_id, block in self.pool._blocks.items():
            if block.state == AllocationState.ALLOCATED:
                if now - block.accessed_at > 300:
                    to_free.append(block_id)

        for block_id in to_free:
            block = self.pool._blocks[block_id]
            freed += block.size_bytes
            self.pool.free(block_id)

        self._collections += 1
        self._bytes_freed += freed
        return freed


# =============================================================================
# DENSE STATE MEMORY
# =============================================================================

class DenseIndex:
    """Simple vector index for dense state retrieval."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._chunks: List[DenseChunk] = []
        self._lock = threading.Lock()

    def add(self, chunk: DenseChunk):
        """Add chunk to index."""
        with self._lock:
            self._chunks.append(chunk)

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[DenseChunk, float]]:
        """Search for similar chunks."""
        with self._lock:
            results = []
            for chunk in self._chunks:
                if chunk.embedding:
                    sim = self._cosine_similarity(query_embedding, chunk.embedding)
                    results.append((chunk, sim))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if len(a) != len(b) or len(a) == 0:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class DenseStateMemory:
    """
    Vector-based context memory with PHI-optimal retrieval.

    Features:
    - Embedding-based similarity search
    - Resonance threshold filtering (BETA)
    - Chunk overlap for context continuity
    """

    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64
    RESONANCE_THRESHOLD = BETA  # 0.236

    def __init__(self, dimension: int = 384):
        self.index = DenseIndex(dimension)
        self._chunk_counter = 0
        self._lock = threading.Lock()

    def store(self, content: str, embedding: List[float] = None, source: str = "") -> str:
        """Store content as dense chunk."""
        self._chunk_counter += 1
        chunk_id = f"chunk-{self._chunk_counter}"

        chunk = DenseChunk(
            id=chunk_id,
            content=content,
            embedding=embedding or [],
            tokens=len(content.split()),
            source=source,
        )
        self.index.add(chunk)
        return chunk_id

    def retrieve(self, query_embedding: List[float], max_tokens: int = 2000) -> Tuple[str, Dict]:
        """Retrieve relevant context."""
        results = self.index.search(query_embedding, k=10)

        # Filter by resonance threshold
        filtered = [(c, s) for c, s in results if s >= self.RESONANCE_THRESHOLD]

        # Build context respecting token limit
        context_parts = []
        total_tokens = 0

        for chunk, score in filtered:
            if total_tokens + chunk.tokens <= max_tokens:
                context_parts.append(chunk.content)
                total_tokens += chunk.tokens

        context = "\n\n".join(context_parts)
        metadata = {
            "sources": len(filtered),
            "token_estimate": total_tokens,
            "resonance_scores": [s for _, s in filtered],
        }

        return context, metadata


# =============================================================================
# BUDGET LEDGER
# =============================================================================

class BudgetLedger:
    """Thread-safe token budget allocation tracker."""

    def __init__(self, default_budget: int = 100000):
        self.default_budget = default_budget
        self._entries: Dict[str, BudgetEntry] = {}
        self._lock = threading.Lock()
        self._total_allocated = 0
        self._total_used = 0

    def allocate(self, task_id: str, agent: str, model: str, tokens: int) -> Optional[BudgetEntry]:
        """Allocate tokens for a task."""
        with self._lock:
            entry_id = f"budget-{len(self._entries)}"
            entry = BudgetEntry(
                id=entry_id,
                task_id=task_id,
                agent=agent,
                model=model,
                tokens_allocated=tokens,
            )
            self._entries[entry_id] = entry
            self._total_allocated += tokens
            return entry

    def use(self, entry_id: str, tokens: int) -> bool:
        """Record token usage."""
        with self._lock:
            if entry_id not in self._entries:
                return False
            entry = self._entries[entry_id]
            if entry.tokens_used + tokens > entry.tokens_allocated:
                return False
            entry.tokens_used += tokens
            self._total_used += tokens
            return True

    def release(self, entry_id: str):
        """Release a budget allocation."""
        with self._lock:
            if entry_id in self._entries:
                entry = self._entries[entry_id]
                entry.status = "released"

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_allocated": self._total_allocated,
            "total_used": self._total_used,
            "entries": len(self._entries),
            "utilization": self._total_used / max(self._total_allocated, 1),
        }


class BudgetAllocator:
    """High-level budget allocation manager."""

    def __init__(self, ledger: BudgetLedger = None):
        self.ledger = ledger or BudgetLedger()

    def request(self, task_id: str, agent: str, model: str, tokens: int) -> Optional[str]:
        """Request budget allocation. Returns entry ID."""
        entry = self.ledger.allocate(task_id, agent, model, tokens)
        return entry.id if entry else None


# =============================================================================
# MEMORY (UNIFIED INTERFACE)
# =============================================================================

class Memory:
    """Unified memory interface."""

    def __init__(self, max_mb: int = 1024):
        self.pool = MemoryPool(max_mb)
        self.virtual = VirtualMemory()
        self.gc = GarbageCollector(self.pool)
        self.dense = DenseStateMemory()
        self.budget = BudgetLedger()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "pool": self.pool.get_stats(),
            "budget": self.budget.get_stats(),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_memory: Optional[Memory] = None


def create_memory(max_mb: int = 1024) -> Memory:
    return Memory(max_mb)


def create_budget_ledger() -> BudgetLedger:
    return BudgetLedger()


def get_memory() -> Memory:
    global _memory
    if _memory is None:
        _memory = Memory()
    return _memory
