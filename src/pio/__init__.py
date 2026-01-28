"""
PIO - Personal Intelligent Operator

The interface layer of Sovereign PIO.
Handles user interaction, session management, and intent routing.

Components:
- Operator: Main PIO interface
- PIO Bridge: GPIA-PIO ignorance-aware cognition
"""

from .operator import (
    PIOOperator,
    Session,
    SessionState,
    Message,
    IntentType,
    IntentDetector,
    logging_middleware,
    memory_store_middleware,
)

# GPIA-PIO Bridge
from .pio_bridge import (
    Wavelength,
    DARK_SECTOR_RATIOS,
    IgnoranceState,
    WavelengthIgnorance,
    PipelineIgnorance,
    IgnoranceCalculator,
    GPIAPIOBridge,
    IgnoranceAwareResult,
)

__all__ = [
    # Operator
    "PIOOperator",
    "Session",
    "SessionState",
    "Message",
    "IntentType",
    "IntentDetector",
    "logging_middleware",
    "memory_store_middleware",
    # PIO Bridge
    "Wavelength",
    "DARK_SECTOR_RATIOS",
    "IgnoranceState",
    "WavelengthIgnorance",
    "PipelineIgnorance",
    "IgnoranceCalculator",
    "GPIAPIOBridge",
    "IgnoranceAwareResult",
]
