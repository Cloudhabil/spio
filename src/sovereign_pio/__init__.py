"""
Sovereign PIO - Personal Intelligent Operator, Autonomous Edition

A unified autonomous agent platform combining:
- PIO: Personal Intelligent Operator (interface)
- GPIA: Intelligence & Reasoning
- ASIOS: Operating System Runtime
- Moltbot: Multi-Channel Gateway
"""

__version__ = "1.618.1"
__author__ = "Sovereign PIO Team"

from .constants import PHI, ALPHA, OMEGA, BETA, GAMMA
from .calculator import D, Theta, Energy, x_from_D

# Brahim API
from .brahim_api import (
    BrahimAPI,
    WormholeGeometry,
    DimensionalState,
    PhiHierarchy,
    BRAHIM_SEQUENCE,
    CENTER,
    PAIR_SUM,
    brahim_mirror,
    wormhole_transform,
    inverse_wormhole_transform,
    phi_compress,
    phi_decompress,
    optimal_route,
    dimension_transition,
    lyapunov_stability,
)

__all__ = [
    "__version__",
    # Constants
    "PHI",
    "ALPHA",
    "OMEGA",
    "BETA",
    "GAMMA",
    # Calculator
    "D",
    "Theta",
    "Energy",
    "x_from_D",
    # Brahim API
    "BrahimAPI",
    "WormholeGeometry",
    "DimensionalState",
    "PhiHierarchy",
    "BRAHIM_SEQUENCE",
    "CENTER",
    "PAIR_SUM",
    "brahim_mirror",
    "wormhole_transform",
    "inverse_wormhole_transform",
    "phi_compress",
    "phi_decompress",
    "optimal_route",
    "dimension_transition",
    "lyapunov_stability",
]
