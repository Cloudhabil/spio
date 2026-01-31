"""
IIAS Extension - Intelligent Infrastructure as a Service

Deterministic AI infrastructure framework with 125+ applications across 13 categories:

Categories:
1. Foundation (5)      - dimension_router, genesis_controller, mirror_balancer
2. Infrastructure (10) - auto_scaler, load_balancer, cost_optimizer
3. Edge (10)          - edge_ai_router, battery_manager, thermal_manager
4. AI/ML (10)         - inference_router, attention_allocator, context_manager
5. Security (10)      - threat_classifier, access_controller, anomaly_detector
6. Business (10)      - resource_allocator, task_scheduler, billing_calculator
7. Data (10)          - data_tiering, backup_scheduler, cache_invalidator
8. IoT (10)           - device_router, firmware_updater, telemetry_collector
9. Communication (10) - message_router, protocol_selector, sync_manager
10. Developer (10)    - build_optimizer, test_scheduler, feature_flagger
11. Scientific (10)   - simulation_router, experiment_scheduler, hypothesis_ranker
12. Personal (10)     - focus_manager, habit_tracker, learning_planner
13. Finance (10)      - portfolio_balancer, risk_calculator, budget_allocator

Mathematical Foundation:
- PHI = 1.618... (Golden ratio saturation)
- LUCAS = [1,3,4,7,11,18,29,47,76,123,199,322] (840 total states)
- BRAHIM_NUMBERS for 12-dimension routing
- GENESIS_CONSTANT = 2/901 for initialization

Reference: Brahim IIAS Framework
Author: Elias Oulad Brahim
"""

from .iias_core import (
    AI_ML_APPS,
    ALL_APPS,
    ALL_HANDLERS,
    BETA,
    BRAHIM_NUMBERS,
    BUSINESS_APPS,
    CENTER,
    CLOSURE_EXPONENT,
    COMMUNICATION_APPS,
    DATA_APPS,
    DEVELOPER_APPS,
    DIMENSION_NAMES,
    DIMENSIONS,
    EDGE_APPS,
    EXPLICIT_ERROR_TOLERANCE,
    FINANCE_APPS,
    FOUNDATION_APPS,
    GENESIS_CONSTANT,
    # ========================================================================
    # MAIN INTERFACE
    # ========================================================================
    IIAS,
    # ========================================================================
    # GOLDBACH CONSTANTS & FUNCTIONS
    # ========================================================================
    INDS_ALLOWED_DR,
    INDS_MIN_TYPES_PER_CLASS,
    INDS_TOTAL_TYPES,
    INFRASTRUCTURE_APPS,
    IOT_APPS,
    LUCAS,
    OMEGA,
    PERSONAL_APPS,
    # ========================================================================
    # CONSTANTS
    # ========================================================================
    PHI,
    PHI_PI_GAP,
    PRODUCTIVE_FRACTION,
    SCIENTIFIC_APPS,
    SECURITY_APPS,
    SILICON_SPECS,
    STRUCTURAL_OVERHEAD,
    SUM_CONSTANT,
    TOTAL_STATES,
    # ========================================================================
    # APP REGISTRY
    # ========================================================================
    AppRegistry,
    AppStatus,
    B,
    Badge,
    Category,
    # ========================================================================
    # DIMENSIONS
    # ========================================================================
    Dimension,
    # ========================================================================
    # DIMENSION ROUTER
    # ========================================================================
    DimensionRouter,
    # ========================================================================
    # GENESIS
    # ========================================================================
    GenesisController,
    # ========================================================================
    # APPLICATIONS
    # ========================================================================
    IIASApp,
    LucasAllocator,
    # ========================================================================
    # BALANCERS & ALLOCATORS
    # ========================================================================
    MirrorBalancer,
    PhiSaturator,
    # ========================================================================
    # ENUMS
    # ========================================================================
    SiliconLayer,
    # ========================================================================
    # SILICON SPECS
    # ========================================================================
    SiliconSpec,
    # ========================================================================
    # FACTORY FUNCTIONS
    # ========================================================================
    create_iias,
    create_registry,
    create_router,
    digital_root,
    genesis,
    inds_routing_class,
    is_within_tolerance,
    productive_capacity,
)

__all__ = [
    # Constants
    "PHI",
    "OMEGA",
    "BETA",
    "CENTER",
    "SUM_CONSTANT",
    "GENESIS_CONSTANT",
    "BRAHIM_NUMBERS",
    "B",
    "LUCAS",
    "TOTAL_STATES",

    # Enums
    "SiliconLayer",
    "AppStatus",
    "Badge",
    "Category",

    # Silicon
    "SiliconSpec",
    "SILICON_SPECS",

    # Dimensions
    "Dimension",
    "DIMENSIONS",
    "DIMENSION_NAMES",

    # Goldbach
    "INDS_ALLOWED_DR",
    "INDS_TOTAL_TYPES",
    "INDS_MIN_TYPES_PER_CLASS",
    "CLOSURE_EXPONENT",
    "PRODUCTIVE_FRACTION",
    "STRUCTURAL_OVERHEAD",
    "PHI_PI_GAP",
    "EXPLICIT_ERROR_TOLERANCE",
    "digital_root",
    "inds_routing_class",
    "productive_capacity",
    "is_within_tolerance",

    # Router
    "DimensionRouter",

    # Genesis
    "GenesisController",
    "genesis",

    # Apps
    "IIASApp",
    "FOUNDATION_APPS",
    "INFRASTRUCTURE_APPS",
    "EDGE_APPS",
    "AI_ML_APPS",
    "SECURITY_APPS",
    "BUSINESS_APPS",
    "DATA_APPS",
    "IOT_APPS",
    "COMMUNICATION_APPS",
    "DEVELOPER_APPS",
    "SCIENTIFIC_APPS",
    "PERSONAL_APPS",
    "FINANCE_APPS",
    "ALL_APPS",
    "ALL_HANDLERS",

    # Registry
    "AppRegistry",

    # Balancers
    "MirrorBalancer",
    "LucasAllocator",
    "PhiSaturator",

    # Main
    "IIAS",

    # Factories
    "create_iias",
    "create_router",
    "create_registry",
]
