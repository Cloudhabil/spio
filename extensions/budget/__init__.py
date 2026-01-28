"""
Budget Extension - Resource Allocation and Management

Ported from:
- CLI-main/src/core/budget_ledger.py
- CLI-main/src/core/dynamic_budget_orchestrator.py
- CLI-main/src/core/kernel/budget_service.py
- CLI-main/src/core/resource_analyzer.py

Implements:
- BudgetLedger: Thread-safe global token allocation tracker
- BudgetOrchestrator: Dynamic budget computation
- BudgetService: Kernel-level resource allocation
- ResourceAnalyzer: Tiered data access and context windowing
"""

from .budget_core import (
    # Ledger
    BudgetLedger, BudgetAllocation, get_budget_ledger,

    # Orchestrator
    BudgetOrchestrator, BudgetSettings, compute_budget, apply_dynamic_budget,

    # Service
    BudgetService, ResourceSnapshot, SafetyLimits, get_budget_service,

    # Analyzer
    ResourceAnalyzer,

    # Factory
    create_budget_ledger, create_budget_service,
)

__all__ = [
    # Ledger
    "BudgetLedger", "BudgetAllocation", "get_budget_ledger",
    # Orchestrator
    "BudgetOrchestrator", "BudgetSettings", "compute_budget", "apply_dynamic_budget",
    # Service
    "BudgetService", "ResourceSnapshot", "SafetyLimits", "get_budget_service",
    # Analyzer
    "ResourceAnalyzer",
    # Factory
    "create_budget_ledger", "create_budget_service",
]
