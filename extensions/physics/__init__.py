"""
Physics Extension - Wormhole Physics and Einstein Field Equations

Ports from: CLI-main/src/core/wormhole_physics.py

Complete mathematical framework for traversable wormholes:
- Morris-Thorne metric tensor
- Einstein Field Equations
- Traversability conditions (NEC violation)
- Lyapunov stability analysis
- Brahim algebraic-continuous unification
"""

from .physics_core import (
    # Constants
    PHI, PHI_INV, BETA, GENESIS, LAMBDA_DECAY,
    BRAHIM_SEQUENCE, BRAHIM_SUM, BRAHIM_CENTER,

    # Enums
    WormholeType, StabilityClass, EnergyCondition,

    # Data Classes
    MetricTensor, StressEnergyTensor, ShapeFunction,
    JunctionConditions, LyapunovAnalysis, BrahimWormhole,

    # Engines
    TraversabilityEngine, EinsteinFieldEquations,
    StabilityAnalyzer, UnificationEngine,
    WormholePhysicsSystem,

    # Factory
    verify_wormhole_traversability, create_brahim_wormhole,
    create_physics_system, get_physics_system,
)

__all__ = [
    "PHI", "PHI_INV", "BETA", "GENESIS", "LAMBDA_DECAY",
    "BRAHIM_SEQUENCE", "BRAHIM_SUM", "BRAHIM_CENTER",
    "WormholeType", "StabilityClass", "EnergyCondition",
    "MetricTensor", "StressEnergyTensor", "ShapeFunction",
    "JunctionConditions", "LyapunovAnalysis", "BrahimWormhole",
    "TraversabilityEngine", "EinsteinFieldEquations",
    "StabilityAnalyzer", "UnificationEngine",
    "WormholePhysicsSystem",
    "verify_wormhole_traversability", "create_brahim_wormhole",
    "create_physics_system", "get_physics_system",
]
