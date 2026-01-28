"""
Modes Extension - Operational Mode System

Ported from: CLI-main/src/core/modes/

Implements:
- BaseMode: Abstract base for all operational modes
- SovereignMode: Primary inquiry and philosophical mode
- TeachingMode: Pedagogical tutoring mode
- GardenerMode: Autonomous file organization
- ForensicMode: Debugging and inspection
- ManifestMode: Autonomous 40-cycle construction
"""

from .modes_core import (
    # Base
    BaseMode, ModeTransition, ModeContext,

    # Modes
    SovereignMode, TeachingMode, GardenerMode,
    ForensicMode, ManifestMode,

    # Enums
    ModeType, ModeState,

    # Manager
    ModeManager,

    # Factory
    create_mode, get_mode_manager,
)

__all__ = [
    # Base
    "BaseMode", "ModeTransition", "ModeContext",
    # Modes
    "SovereignMode", "TeachingMode", "GardenerMode",
    "ForensicMode", "ManifestMode",
    # Enums
    "ModeType", "ModeState",
    # Manager
    "ModeManager",
    # Factory
    "create_mode", "get_mode_manager",
]
