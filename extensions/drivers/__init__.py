"""
Drivers Extension - Hardware Abstraction Layer

Components:
1. DimensionRouter - Maps dimensions to silicon
2. SiliconSpec - Hardware specifications
3. DeviceManager - Device enumeration
4. NPUDriver - NPU abstraction
5. GPUDriver - GPU abstraction

Reference: CLI-main/src/iias/router.py, src/core/npu_utils.py
"""

from .drivers_core import (
    # Enums
    SiliconLayer,
    DeviceState,
    DimensionDomain,

    # Data classes
    SiliconSpec,
    DeviceInfo,
    BandwidthMeasurement,

    # Drivers
    NPUDriver,
    GPUDriver,
    CPUDriver,

    # Router
    DimensionRouter,
    SiliconRouter,

    # Device management
    DeviceManager,
    DeviceRegistry,

    # Main interface
    Drivers,

    # Factories
    create_drivers,
    create_dimension_router,
    get_drivers,
)

__all__ = [
    "SiliconLayer", "DeviceState", "DimensionDomain",
    "SiliconSpec", "DeviceInfo", "BandwidthMeasurement",
    "NPUDriver", "GPUDriver", "CPUDriver",
    "DimensionRouter", "SiliconRouter",
    "DeviceManager", "DeviceRegistry",
    "Drivers",
    "create_drivers", "create_dimension_router", "get_drivers",
]
