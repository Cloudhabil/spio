"""
Sovereign PIO Core Utilities

Shared utilities and core functionality used across all layers.

Components:
- Constants & Calculator (PHI-based mathematics)
- Observability (structured logging, tracing, metrics)
- Skills Loader (YAML-based skill discovery)
- Quantization (vector compression)
- Reflex Engine (System 1 responses)
- Hardware Fusion (direct tensor access)
"""

from sovereign_pio.calculator import D, Energy, Theta, lucas, x_from_D
from sovereign_pio.constants import ALPHA, BETA, GAMMA, OMEGA, PHI

# Evidence Database
from .evidence_db import (
    EvidenceDB,
    OperationRecord,
)

# Hardware Fusion
from .hardware_fusion import (
    BatchProcessor,
    FusedIndex,
    HardwareFusion,
    benchmark_fusion,
)

# Measured Silicon
from .measured_silicon import (
    MEASURED,
    WORMHOLES,
    BandwidthMeasurement,
    SiliconLayer,
    WormholeConnection,
    find_wormhole,
    get_bandwidth,
    get_optimal_parallel,
    hardware_stats,
    npu_bandwidth,
    route_dimension_to_silicon,
)

# Observability
from .observability import (
    AlertLevel,
    LogLevel,
    MetricsCollector,
    SpanContext,
    StructuredLogger,
    TraceContext,
    get_current_trace,
    get_logger,
    get_metrics,
    set_current_trace,
)

# Quantization
from .quantization import (
    QuantizedVector,
    Quantizer,
    ResidualQuantizer,
    ResidualVector,
    compute_compression_ratio,
    similarity_preservation,
)

# Reflex Engine
from .reflex_engine import (
    Reflex,
    ReflexEngine,
    ReflexPriority,
    ReflexResult,
)

# Skills
from .skills_loader import (
    Permission,
    SkillCategory,
    SkillLoader,
    SkillMetadata,
    SkillSchema,
)

# 12-Wavelength Cognitive Structure
from .wavelengths import (
    PRIME_TARGET_DENSITY,
    ConvergenceResult,
    DensityArchitect,
    EnduranceLoop,
    GateResult,
    GenerativeStream,
    HephaestusAlloy,
    HomeostaticUpdater,
    MaatAligner,
    MetatronCube,
    PredictionError,
    PrimeDirective,
    PrometheusSpark,
    StochasticIngestor,
    SubstrateCrystallizer,
    SynapticBridge,
    ThetaWaveGenerator,
    TransparencyLogger,
    WavelengthGate,
    ZeroPointInitializer,
)

__all__ = [
    # Constants
    "PHI", "ALPHA", "OMEGA", "BETA", "GAMMA",
    # Calculator
    "D", "Theta", "Energy", "x_from_D", "lucas",
    # Observability
    "StructuredLogger",
    "TraceContext",
    "SpanContext",
    "MetricsCollector",
    "LogLevel",
    "AlertLevel",
    "get_logger",
    "get_metrics",
    "get_current_trace",
    "set_current_trace",
    # Skills
    "SkillLoader",
    "SkillMetadata",
    "SkillCategory",
    "Permission",
    "SkillSchema",
    # Quantization
    "Quantizer",
    "QuantizedVector",
    "ResidualQuantizer",
    "ResidualVector",
    "similarity_preservation",
    "compute_compression_ratio",
    # Reflex
    "ReflexEngine",
    "Reflex",
    "ReflexPriority",
    "ReflexResult",
    # Hardware Fusion
    "HardwareFusion",
    "FusedIndex",
    "BatchProcessor",
    "benchmark_fusion",
    # Measured Silicon
    "SiliconLayer",
    "BandwidthMeasurement",
    "WormholeConnection",
    "MEASURED",
    "WORMHOLES",
    "get_bandwidth",
    "get_optimal_parallel",
    "npu_bandwidth",
    "find_wormhole",
    "route_dimension_to_silicon",
    "hardware_stats",
    # Evidence DB
    "EvidenceDB",
    "OperationRecord",
    # 12-Wavelength
    "WavelengthGate",
    "GateResult",
    "ZeroPointInitializer",
    "PrimeDirective",
    "StochasticIngestor",
    "MaatAligner",
    "PrometheusSpark",
    "MetatronCube",
    "DensityArchitect",
    "SynapticBridge",
    "PredictionError",
    "GenerativeStream",
    "ThetaWaveGenerator",
    "HomeostaticUpdater",
    "EnduranceLoop",
    "ConvergenceResult",
    "TransparencyLogger",
    "HephaestusAlloy",
    "SubstrateCrystallizer",
    "PRIME_TARGET_DENSITY",
]
