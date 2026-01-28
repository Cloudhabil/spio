"""
Core Extension - Essential System Modules

130+ Core system components organized by 13 subsystems:

Subsystems:
1. Runtime (3)       - CapsuleEngine, PASSBroker, Government
2. Modes (5)         - SovereignMode, TeachingMode, GardenerMode, ModeManager
3. Sovereignty (4)   - Identity, IdentityChecker, TelemetryObserver, HeuristicsRegistry
4. Safety (5)        - SafetyCode, SafetyCodeRegistry, SafetyGeometry, CognitiveSafetyGovernor
5. Reflexes (4)      - Reflex, ReflexEngine, ReflexCorrector, ReflexPriority
6. Cognitive (4)     - MetaCortex, Affect, CognitiveState, ResonanceCalibrator
7. Budget (3)        - BudgetEntry, BudgetLedger, DynamicBudgetOrchestrator
8. Skills (3)        - SkillDefinition, SkillsLoader, SkillAssessor
9. Context (3)       - ContextPage, ContextPager, DAGPlanner
10. Infrastructure (3) - Worker, WorkerPool, GPIABridge
11. Gardening (3)    - GardenAction, FilesystemGardener, SovereignGardener
12. Substrate (2)    - SubstrateCompressor, SubstrateCracker
13. Misc (7)         - Fusion, Forager, Linker, Topology, Quorum, Enrichment

Main Interface:
- Core: Unified access to all subsystems

Reference: CLI-main/src/core/*
"""

from .core_modules import (
    # ========================================================================
    # CONSTANTS
    # ========================================================================
    PHI,
    BETA,
    GENESIS_CONSTANT,
    SUM_CONSTANT,
    CENTER,

    # ========================================================================
    # RUNTIME
    # ========================================================================
    CapsuleType,
    Capsule,
    CapsuleEngine,
    PASSBroker,
    Government,

    # ========================================================================
    # MODES
    # ========================================================================
    Mode,
    ModeBase,
    SovereignMode,
    TeachingMode,
    GardenerMode,
    ModeManager,

    # ========================================================================
    # SOVEREIGNTY
    # ========================================================================
    Identity,
    IdentityChecker,
    TelemetryObserver,
    HeuristicsRegistry,

    # ========================================================================
    # SAFETY
    # ========================================================================
    SafetyLevel,
    SafetyCode,
    SafetyCodeRegistry,
    SafetyGeometry,
    CognitiveSafetyGovernor,

    # ========================================================================
    # REFLEXES
    # ========================================================================
    ReflexPriority,
    Reflex,
    ReflexEngine,
    ReflexCorrector,

    # ========================================================================
    # COGNITIVE
    # ========================================================================
    CognitiveState,
    Affect,
    MetaCortex,
    ResonanceCalibrator,

    # ========================================================================
    # BUDGET
    # ========================================================================
    BudgetEntry,
    BudgetLedger,
    DynamicBudgetOrchestrator,

    # ========================================================================
    # SKILLS
    # ========================================================================
    SkillDefinition,
    SkillsLoader,
    SkillAssessor,

    # ========================================================================
    # CONTEXT
    # ========================================================================
    ContextPage,
    ContextPager,
    DAGPlanner,

    # ========================================================================
    # INFRASTRUCTURE
    # ========================================================================
    WorkerState,
    Worker,
    WorkerPool,
    GPIABridge,

    # ========================================================================
    # GARDENING
    # ========================================================================
    GardenAction,
    FilesystemGardener,
    SovereignGardener,

    # ========================================================================
    # SUBSTRATE
    # ========================================================================
    SubstrateCompressor,
    SubstrateCracker,

    # ========================================================================
    # MISC
    # ========================================================================
    Fusion,
    Forager,
    Linker,
    Topology,
    Quorum,
    Enrichment,

    # ========================================================================
    # MAIN INTERFACE
    # ========================================================================
    Core,

    # ========================================================================
    # FACTORY FUNCTIONS
    # ========================================================================
    create_core,
    create_capsule_engine,
    create_mode_manager,
    create_safety_governor,
)

__all__ = [
    # Constants
    "PHI", "BETA", "GENESIS_CONSTANT", "SUM_CONSTANT", "CENTER",

    # Runtime
    "CapsuleType", "Capsule", "CapsuleEngine", "PASSBroker", "Government",

    # Modes
    "Mode", "ModeBase", "SovereignMode", "TeachingMode", "GardenerMode", "ModeManager",

    # Sovereignty
    "Identity", "IdentityChecker", "TelemetryObserver", "HeuristicsRegistry",

    # Safety
    "SafetyLevel", "SafetyCode", "SafetyCodeRegistry", "SafetyGeometry", "CognitiveSafetyGovernor",

    # Reflexes
    "ReflexPriority", "Reflex", "ReflexEngine", "ReflexCorrector",

    # Cognitive
    "CognitiveState", "Affect", "MetaCortex", "ResonanceCalibrator",

    # Budget
    "BudgetEntry", "BudgetLedger", "DynamicBudgetOrchestrator",

    # Skills
    "SkillDefinition", "SkillsLoader", "SkillAssessor",

    # Context
    "ContextPage", "ContextPager", "DAGPlanner",

    # Infrastructure
    "WorkerState", "Worker", "WorkerPool", "GPIABridge",

    # Gardening
    "GardenAction", "FilesystemGardener", "SovereignGardener",

    # Substrate
    "SubstrateCompressor", "SubstrateCracker",

    # Misc
    "Fusion", "Forager", "Linker", "Topology", "Quorum", "Enrichment",

    # Main
    "Core",

    # Factories
    "create_core", "create_capsule_engine", "create_mode_manager", "create_safety_governor",
]
