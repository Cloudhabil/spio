"""
Memory Extension - Dense State Memory and Budget Management

Components:
1. DenseStateMemory - FAISS-based vector retrieval
2. BudgetLedger - Token allocation tracking
3. MemoryPool - Memory allocation management
4. VirtualMemory - Paging simulation
5. GarbageCollector - Memory cleanup

Reference: CLI-main/src/core/dense_state_memory.py
"""

from .memory_core import (
    # Enums
    AllocationState,
    MemoryTier,

    # Data classes
    MemoryBlock,
    BudgetEntry,
    DenseChunk,

    # Memory systems
    MemoryPool,
    VirtualMemory,
    GarbageCollector,

    # Dense State
    DenseStateMemory,
    DenseIndex,

    # Budget
    BudgetLedger,
    BudgetAllocator,

    # Main interface
    Memory,

    # Factories
    create_memory,
    create_budget_ledger,
    get_memory,
)

__all__ = [
    "AllocationState", "MemoryTier",
    "MemoryBlock", "BudgetEntry", "DenseChunk",
    "MemoryPool", "VirtualMemory", "GarbageCollector",
    "DenseStateMemory", "DenseIndex",
    "BudgetLedger", "BudgetAllocator",
    "Memory",
    "create_memory", "create_budget_ledger", "get_memory",
]
