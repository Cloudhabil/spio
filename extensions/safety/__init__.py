"""
Safety Extension - Hardware Protection and Threat Detection

Ports from:
- CLI-main/src/core/safety_governor.py
- CLI-main/src/core/immune_system.py

Implements:
- SafetyGovernor: VRAM/thermal/disk protection
- ImmuneSystem: Adaptive threat detection
- FaultTolerance: Graceful degradation
"""

from .safety_core import (
    # Governor
    SafetyGovernor, GPUVitals, DiskHealth,

    # Immune System
    ImmuneSystem, ThreatSignature, SecurityEvent,
    SeverityLevel, ThreatCategory,

    # Fault Tolerance
    FaultToleranceManager,

    # Factory
    create_safety_governor, create_immune_system,
    get_safety_governor, get_immune_system,
)

__all__ = [
    "SafetyGovernor", "GPUVitals", "DiskHealth",
    "ImmuneSystem", "ThreatSignature", "SecurityEvent",
    "SeverityLevel", "ThreatCategory",
    "FaultToleranceManager",
    "create_safety_governor", "create_immune_system",
    "get_safety_governor", "get_immune_system",
]
