"""
Wormhole Extension - Mathematical Bridge Architecture

Ports the CLI-main wormhole engine to SPIO:
- Morris-Thorne traversable wormhole geometry
- PHI-based compression and routing
- Brahim Sequence error detection
- O(1) wormhole transforms

Author: Based on CLI-main/src/core/brahim_wormhole_engine.py
"""

from .wormhole_core import (
    # Constants
    PHI, PHI_INV, ALPHA, BETA, GAMMA,
    BRAHIM_SEQUENCE, PAIR_SUM, CENTER,

    # Enums
    WormholeState, RouteType, SafetyLevel,

    # Data classes
    WormholeGeometry, TraversabilityResult, StabilityResult,
    WormholeTransformResult, RoutingResult,

    # Core classes
    BrahimWormholeEngine, WormholeRouter, WormholeTransform,

    # Factory functions
    create_engine, create_router, get_engine,
    quick_transform, verify_sequence,
)

__all__ = [
    # Constants
    "PHI", "PHI_INV", "ALPHA", "BETA", "GAMMA",
    "BRAHIM_SEQUENCE", "PAIR_SUM", "CENTER",

    # Enums
    "WormholeState", "RouteType", "SafetyLevel",

    # Data classes
    "WormholeGeometry", "TraversabilityResult", "StabilityResult",
    "WormholeTransformResult", "RoutingResult",

    # Core classes
    "BrahimWormholeEngine", "WormholeRouter", "WormholeTransform",

    # Factory functions
    "create_engine", "create_router", "get_engine",
    "quick_transform", "verify_sequence",
]
