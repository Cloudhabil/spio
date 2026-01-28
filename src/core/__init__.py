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

from sovereign_pio.constants import PHI, ALPHA, OMEGA, BETA, GAMMA
from sovereign_pio.calculator import D, Theta, Energy, x_from_D, lucas

# Observability
from .observability import (
    StructuredLogger,
    TraceContext,
    SpanContext,
    MetricsCollector,
    LogLevel,
    AlertLevel,
    get_logger,
    get_metrics,
    get_current_trace,
    set_current_trace,
)

# Skills
from .skills_loader import (
    SkillLoader,
    SkillMetadata,
    SkillCategory,
    Permission,
    SkillSchema,
)

# Quantization
from .quantization import (
    Quantizer,
    QuantizedVector,
    ResidualQuantizer,
    ResidualVector,
    similarity_preservation,
    compute_compression_ratio,
)

# Reflex Engine
from .reflex_engine import (
    ReflexEngine,
    Reflex,
    ReflexPriority,
    ReflexResult,
)

# Hardware Fusion
from .hardware_fusion import (
    HardwareFusion,
    FusedIndex,
    BatchProcessor,
    benchmark_fusion,
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
]
