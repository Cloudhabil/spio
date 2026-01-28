"""
Boot Extension - Genesis and Initialization Sequence

Components:
1. BootLoader - System initialization
2. Genesis - G(t) function for PIO birth
3. Preflight - Pre-boot checks
4. InitSequence - Ordered startup
5. ServiceManager - Service lifecycle

Reference: CLI-main/src/boot.py, src/iias/genesis.py
"""

from .boot_core import (
    # Enums
    BootStage,
    ServiceState,
    GenesisPhase,

    # Data classes
    BootConfig,
    PreflightResult,
    ServiceDescriptor,

    # Boot components
    BootLoader,
    Preflight,
    InitSequence,

    # Genesis
    Genesis,
    GenesisFunction,

    # Services
    ServiceManager,
    Service,

    # Main interface
    Boot,

    # Factories
    create_boot,
    run_genesis,
    get_boot,
)

__all__ = [
    "BootStage", "ServiceState", "GenesisPhase",
    "BootConfig", "PreflightResult", "ServiceDescriptor",
    "BootLoader", "Preflight", "InitSequence",
    "Genesis", "GenesisFunction",
    "ServiceManager", "Service",
    "Boot",
    "create_boot", "run_genesis", "get_boot",
]
