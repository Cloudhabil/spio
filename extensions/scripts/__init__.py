"""
Scripts Extension - Utility Script Registry and Runner

Catalog of 100+ utility scripts organized by 12 categories:

Categories:
1. Research (12)    - BSD, Riemann, research orchestration
2. Admin (8)        - System administration, policy
3. Agents (9)       - Agent orchestration, training
4. Validation (7)   - Integrity checks, verification
5. Data (8)         - Data processing, dense state
6. Network (4)      - Network diagnostics, monitoring
7. Build (6)        - CI/CD, compilation, optimization
8. Learning (8)     - ML training, fine-tuning
9. Genesis (5)      - Initialization, birth cycles
10. Manifold (3)    - Topology, manifold operations
11. Gardener (4)    - Filesystem, maintenance
12. Misc (18)       - Various utilities

Components:
- ScriptInfo: Script metadata
- ScriptCatalog: Script registry with search
- ScriptRunner: Safe script execution
- ScriptManager: Unified management interface
- ScriptResult: Execution result

Reference: CLI-main/scripts/*
"""

from .scripts_core import (
    # ========================================================================
    # ENUMS
    # ========================================================================
    ScriptCategory,
    ScriptStatus,

    # ========================================================================
    # SCRIPT INFO
    # ========================================================================
    ScriptInfo,

    # ========================================================================
    # SCRIPT LISTS
    # ========================================================================
    RESEARCH_SCRIPTS,
    ADMIN_SCRIPTS,
    AGENTS_SCRIPTS,
    VALIDATION_SCRIPTS,
    DATA_SCRIPTS,
    NETWORK_SCRIPTS,
    BUILD_SCRIPTS,
    LEARNING_SCRIPTS,
    GENESIS_SCRIPTS,
    MANIFOLD_SCRIPTS,
    GARDENER_SCRIPTS,
    MISC_SCRIPTS,
    ALL_SCRIPTS,

    # ========================================================================
    # CATALOG
    # ========================================================================
    ScriptCatalog,

    # ========================================================================
    # RUNNER
    # ========================================================================
    ScriptResult,
    ScriptRunner,

    # ========================================================================
    # MANAGER
    # ========================================================================
    ScriptManager,

    # ========================================================================
    # FACTORY FUNCTIONS
    # ========================================================================
    create_script_catalog,
    create_script_runner,
    create_script_manager,
)

__all__ = [
    # Enums
    "ScriptCategory",
    "ScriptStatus",

    # Script Info
    "ScriptInfo",

    # Script Lists
    "RESEARCH_SCRIPTS",
    "ADMIN_SCRIPTS",
    "AGENTS_SCRIPTS",
    "VALIDATION_SCRIPTS",
    "DATA_SCRIPTS",
    "NETWORK_SCRIPTS",
    "BUILD_SCRIPTS",
    "LEARNING_SCRIPTS",
    "GENESIS_SCRIPTS",
    "MANIFOLD_SCRIPTS",
    "GARDENER_SCRIPTS",
    "MISC_SCRIPTS",
    "ALL_SCRIPTS",

    # Catalog
    "ScriptCatalog",

    # Runner
    "ScriptResult",
    "ScriptRunner",

    # Manager
    "ScriptManager",

    # Factories
    "create_script_catalog",
    "create_script_runner",
    "create_script_manager",
]
