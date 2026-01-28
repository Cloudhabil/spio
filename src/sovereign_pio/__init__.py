"""
Sovereign PIO - Personal Intelligent Operator, Autonomous Edition

A unified autonomous agent platform combining:
- PIO: Personal Intelligent Operator (interface)
- GPIA: Intelligence & Reasoning
- ASIOS: Operating System Runtime
- Moltbot: Multi-Channel Gateway
"""

__version__ = "1.618.0"
__author__ = "Sovereign PIO Team"

from .constants import PHI, ALPHA, OMEGA, BETA, GAMMA
from .calculator import D, Theta, Energy, x_from_D

__all__ = [
    "__version__",
    "PHI",
    "ALPHA",
    "OMEGA",
    "BETA",
    "GAMMA",
    "D",
    "Theta",
    "Energy",
    "x_from_D",
]
