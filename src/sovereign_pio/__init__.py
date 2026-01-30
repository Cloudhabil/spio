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

# Brahim API
from .brahim_api import (
    BRAHIM_SEQUENCE,
    CENTER,
    PAIR_SUM,
    BrahimAPI,
    DimensionalState,
    PhiHierarchy,
    WormholeGeometry,
    brahim_mirror,
    dimension_transition,
    inverse_wormhole_transform,
    lyapunov_stability,
    optimal_route,
    phi_compress,
    phi_decompress,
    wormhole_transform,
)
from .calculator import (
    D,
    D_complex,
    Energy,
    Energy_complex,
    Theta,
    Theta_complex,
    branch_spectrum,
    spectral_dimensions,
    x_from_D,
)
from .constants import ALPHA, BETA, GAMMA, OMEGA, PHI

__all__ = [
    "__version__",
    # Constants
    "PHI",
    "ALPHA",
    "OMEGA",
    "BETA",
    "GAMMA",
    # Calculator — real domain
    "D",
    "Theta",
    "Energy",
    "x_from_D",
    # Calculator — complex domain
    "D_complex",
    "Theta_complex",
    "Energy_complex",
    "branch_spectrum",
    "spectral_dimensions",
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
