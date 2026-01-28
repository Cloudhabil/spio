"""
Drivers Core - Hardware Abstraction Layer

Based on: CLI-main/src/iias/router.py, src/core/npu_utils.py
"""

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

# Brahim's Calculator
PHI = (1 + math.sqrt(5)) / 2
BETA = 1 / (PHI ** 3)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class SiliconLayer(Enum):
    """Hardware silicon layers."""
    NPU = "npu"
    GPU = "gpu"
    CPU = "cpu"
    RAM = "ram"
    SSD = "ssd"


class DeviceState(Enum):
    """Device operational states."""
    UNKNOWN = auto()
    DETECTED = auto()
    INITIALIZED = auto()
    READY = auto()
    BUSY = auto()
    ERROR = auto()
    OFFLINE = auto()


class DimensionDomain(Enum):
    """12 dimension domains mapped to silicon."""
    PERCEPTION = 1      # D1 -> NPU
    ATTENTION = 2       # D2 -> NPU
    SECURITY = 3        # D3 -> NPU
    STABILITY = 4       # D4 -> NPU
    COMPRESSION = 5     # D5 -> CPU
    HARMONY = 6         # D6 -> CPU
    REASONING = 7       # D7 -> CPU
    PREDICTION = 8      # D8 -> CPU
    CREATIVITY = 9      # D9 -> GPU
    WISDOM = 10         # D10 -> GPU
    INTEGRATION = 11    # D11 -> GPU
    UNIFICATION = 12    # D12 -> GPU


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SiliconSpec:
    """Hardware specifications for a silicon layer."""
    layer: SiliconLayer
    bandwidth_gbps: float = 0.0
    saturation_k: float = 1.0  # PHI-based saturation constant
    capacity_gb: float = 0.0
    utilization: float = 0.0
    temperature_c: float = 0.0

    def get_effective_bandwidth(self, parallel_ops: int) -> float:
        """Calculate effective bandwidth with PHI saturation."""
        # BW(N) = BW_MAX * (1 - e^(-N/k))
        return self.bandwidth_gbps * (1 - math.exp(-parallel_ops / self.saturation_k))


@dataclass
class DeviceInfo:
    """Device information."""
    name: str
    device_type: str
    vendor: str = ""
    model: str = ""
    driver_version: str = ""
    state: DeviceState = DeviceState.UNKNOWN
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BandwidthMeasurement:
    """Bandwidth measurement result."""
    layer: SiliconLayer
    measured_gbps: float
    theoretical_gbps: float
    efficiency: float = 0.0
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# DRIVERS
# =============================================================================

class BaseDriver:
    """Base class for hardware drivers."""

    def __init__(self, layer: SiliconLayer):
        self.layer = layer
        self.state = DeviceState.UNKNOWN
        self._spec: Optional[SiliconSpec] = None
        self._lock = threading.Lock()

    def detect(self) -> bool:
        """Detect hardware presence."""
        return False

    def initialize(self) -> bool:
        """Initialize the driver."""
        if self.detect():
            self.state = DeviceState.INITIALIZED
            return True
        return False

    def get_spec(self) -> Optional[SiliconSpec]:
        """Get hardware specifications."""
        return self._spec

    def get_utilization(self) -> float:
        """Get current utilization (0.0 to 1.0)."""
        return 0.0


class NPUDriver(BaseDriver):
    """Neural Processing Unit driver."""

    # Measured specs from hardware
    BANDWIDTH_GBPS = 7.35
    SATURATION_K = PHI  # 1.618

    def __init__(self):
        super().__init__(SiliconLayer.NPU)
        self._spec = SiliconSpec(
            layer=SiliconLayer.NPU,
            bandwidth_gbps=self.BANDWIDTH_GBPS,
            saturation_k=self.SATURATION_K,
        )

    def detect(self) -> bool:
        """Detect NPU presence."""
        # Simplified detection
        try:
            # Would check for actual NPU hardware
            self.state = DeviceState.DETECTED
            return True
        except Exception:
            return False

    def get_npu_info(self) -> Dict[str, Any]:
        """Get NPU information."""
        return {
            "available": self.state != DeviceState.UNKNOWN,
            "bandwidth_gbps": self.BANDWIDTH_GBPS,
            "saturation_k": self.SATURATION_K,
            "optimal_parallel": 16,  # PHI^4 rounded
        }


class GPUDriver(BaseDriver):
    """Graphics Processing Unit driver."""

    BANDWIDTH_GBPS = 12.0
    SATURATION_K = 0.36

    def __init__(self):
        super().__init__(SiliconLayer.GPU)
        self._spec = SiliconSpec(
            layer=SiliconLayer.GPU,
            bandwidth_gbps=self.BANDWIDTH_GBPS,
            saturation_k=self.SATURATION_K,
        )
        self._vram_total_mb = 0
        self._vram_used_mb = 0

    def detect(self) -> bool:
        """Detect GPU presence."""
        try:
            # Would use nvidia-smi or similar
            self.state = DeviceState.DETECTED
            return True
        except Exception:
            return False

    def get_vram_info(self) -> Dict[str, float]:
        """Get VRAM information."""
        return {
            "total_mb": self._vram_total_mb,
            "used_mb": self._vram_used_mb,
            "free_mb": self._vram_total_mb - self._vram_used_mb,
            "utilization": self._vram_used_mb / max(self._vram_total_mb, 1),
        }


class CPUDriver(BaseDriver):
    """Central Processing Unit driver."""

    BANDWIDTH_GBPS = 26.0
    SATURATION_K = 0.90

    def __init__(self):
        super().__init__(SiliconLayer.CPU)
        self._spec = SiliconSpec(
            layer=SiliconLayer.CPU,
            bandwidth_gbps=self.BANDWIDTH_GBPS,
            saturation_k=self.SATURATION_K,
        )

    def detect(self) -> bool:
        """CPU is always present."""
        self.state = DeviceState.DETECTED
        return True

    def get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        import os
        return {
            "cores": os.cpu_count() or 1,
            "bandwidth_gbps": self.BANDWIDTH_GBPS,
        }


# =============================================================================
# DIMENSION ROUTER
# =============================================================================

class DimensionRouter:
    """
    Routes dimensions to silicon layers.

    12-dimension to silicon mapping:
    - D1-D4: NPU (Perception, Attention, Security, Stability)
    - D5-D8: CPU (Compression, Harmony, Reasoning, Prediction)
    - D9-D12: GPU (Creativity, Wisdom, Integration, Unification)
    """

    DIMENSION_TO_SILICON = {
        1: SiliconLayer.NPU, 2: SiliconLayer.NPU, 3: SiliconLayer.NPU, 4: SiliconLayer.NPU,
        5: SiliconLayer.CPU, 6: SiliconLayer.CPU, 7: SiliconLayer.CPU, 8: SiliconLayer.CPU,
        9: SiliconLayer.GPU, 10: SiliconLayer.GPU, 11: SiliconLayer.GPU, 12: SiliconLayer.GPU,
    }

    LUCAS = [1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322]

    def __init__(self):
        self.npu = NPUDriver()
        self.gpu = GPUDriver()
        self.cpu = CPUDriver()
        self._drivers = {
            SiliconLayer.NPU: self.npu,
            SiliconLayer.GPU: self.gpu,
            SiliconLayer.CPU: self.cpu,
        }

    def route(self, dimension: int) -> SiliconLayer:
        """Route dimension to silicon layer."""
        if dimension < 1 or dimension > 12:
            return SiliconLayer.CPU  # Default
        return self.DIMENSION_TO_SILICON[dimension]

    def get_states_for_dimension(self, dimension: int) -> int:
        """Get Lucas number of states for dimension."""
        if dimension < 1 or dimension > 12:
            return 0
        return self.LUCAS[dimension - 1]

    def get_driver(self, layer: SiliconLayer) -> Optional[BaseDriver]:
        """Get driver for a silicon layer."""
        return self._drivers.get(layer)

    def get_routing_cost(self, from_dim: int, to_dim: int) -> float:
        """Calculate routing cost between dimensions."""
        from_layer = self.route(from_dim)
        to_layer = self.route(to_dim)

        if from_layer == to_layer:
            return 0.0  # Same silicon

        # Cross-silicon routing has PHI-based cost
        return BETA * abs(from_dim - to_dim)


class SiliconRouter:
    """Routes tasks directly to silicon layers."""

    def __init__(self, dim_router: DimensionRouter = None):
        self.dim_router = dim_router or DimensionRouter()

    def route_task(self, task_type: str) -> SiliconLayer:
        """Route task by type to silicon."""
        # Map task types to dimensions
        TASK_DIMENSIONS = {
            "perception": 1, "attention": 2, "security": 3, "stability": 4,
            "compression": 5, "harmony": 6, "reasoning": 7, "prediction": 8,
            "creativity": 9, "wisdom": 10, "integration": 11, "unification": 12,
        }
        dimension = TASK_DIMENSIONS.get(task_type.lower(), 7)  # Default to reasoning
        return self.dim_router.route(dimension)


# =============================================================================
# DEVICE MANAGER
# =============================================================================

class DeviceRegistry:
    """Registry of available devices."""

    def __init__(self):
        self._devices: Dict[str, DeviceInfo] = {}
        self._lock = threading.Lock()

    def register(self, device_id: str, info: DeviceInfo):
        """Register a device."""
        with self._lock:
            self._devices[device_id] = info

    def unregister(self, device_id: str):
        """Unregister a device."""
        with self._lock:
            self._devices.pop(device_id, None)

    def get(self, device_id: str) -> Optional[DeviceInfo]:
        """Get device info."""
        return self._devices.get(device_id)

    def list_devices(self, device_type: str = None) -> List[DeviceInfo]:
        """List devices, optionally filtered by type."""
        devices = list(self._devices.values())
        if device_type:
            devices = [d for d in devices if d.device_type == device_type]
        return devices


class DeviceManager:
    """Manages device enumeration and initialization."""

    def __init__(self):
        self.registry = DeviceRegistry()
        self.dim_router = DimensionRouter()
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize device manager and detect devices."""
        # Detect NPU
        if self.dim_router.npu.detect():
            self.registry.register("npu-0", DeviceInfo(
                name="Neural Processing Unit",
                device_type="npu",
                state=DeviceState.DETECTED,
            ))

        # Detect GPU
        if self.dim_router.gpu.detect():
            self.registry.register("gpu-0", DeviceInfo(
                name="Graphics Processing Unit",
                device_type="gpu",
                state=DeviceState.DETECTED,
            ))

        # CPU always present
        if self.dim_router.cpu.detect():
            self.registry.register("cpu-0", DeviceInfo(
                name="Central Processing Unit",
                device_type="cpu",
                state=DeviceState.DETECTED,
            ))

        self._initialized = True
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get device manager stats."""
        return {
            "initialized": self._initialized,
            "devices": len(self.registry._devices),
            "npu_available": self.dim_router.npu.state != DeviceState.UNKNOWN,
            "gpu_available": self.dim_router.gpu.state != DeviceState.UNKNOWN,
        }


# =============================================================================
# DRIVERS (UNIFIED INTERFACE)
# =============================================================================

class Drivers:
    """Unified drivers interface."""

    def __init__(self):
        self.device_manager = DeviceManager()
        self.dim_router = DimensionRouter()
        self.silicon_router = SiliconRouter(self.dim_router)

    def initialize(self) -> bool:
        """Initialize all drivers."""
        return self.device_manager.initialize()

    def route_dimension(self, dimension: int) -> SiliconLayer:
        """Route dimension to silicon."""
        return self.dim_router.route(dimension)

    def route_task(self, task_type: str) -> SiliconLayer:
        """Route task type to silicon."""
        return self.silicon_router.route_task(task_type)

    def get_stats(self) -> Dict[str, Any]:
        return self.device_manager.get_stats()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_drivers: Optional[Drivers] = None


def create_drivers() -> Drivers:
    return Drivers()


def create_dimension_router() -> DimensionRouter:
    return DimensionRouter()


def get_drivers() -> Drivers:
    global _drivers
    if _drivers is None:
        _drivers = Drivers()
        _drivers.initialize()
    return _drivers
