"""
Reflexes Extension - Fast Safety Mechanisms

System 1 responses for sub-50ms safety enforcement.

REFLEXES (ordered by priority):
    L1 Layer (Core):
        - engine_core (0): Kernel boot sequencing
        - biomedical_precision (5): Research accuracy
        - recency_injector (10): Memory injection

    L2 Layer (Optimization):
        - stabilizer (20): Cache-based reuse

    L4 Layer (Governance):
        - guard_l4 (90): Safety contracts
        - audit_logger (95): Audit trail

ACTIONS:
    - PASS: Allow subsequent reflexes
    - DENY: Block request
    - MODIFY_CONTEXT: Inject into context
    - REPLY: Return cached response
    - STORE_MEMORY: Save to memory
    - SEARCH_MEMORY: Query memory
"""

from .reflexes import (
    # Core types
    ReflexAction,
    ReflexResult,
    ReflexLayer,
    ReflexManifest,
    ReflexRegistry,

    # Reflexes
    EngineCore,
    RecencyInjector,
    Stabilizer,
    GuardL4,
    AuditLogger,
    BiomedicalPrecision,

    # Runner
    ReflexRunner,
)

__all__ = [
    "ReflexAction",
    "ReflexResult",
    "ReflexLayer",
    "ReflexManifest",
    "ReflexRegistry",
    "EngineCore",
    "RecencyInjector",
    "Stabilizer",
    "GuardL4",
    "AuditLogger",
    "BiomedicalPrecision",
    "ReflexRunner",
]
