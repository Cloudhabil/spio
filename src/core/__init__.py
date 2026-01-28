"""
Sovereign PIO Core Utilities

Shared utilities and core functionality used across all layers.

Components:
- Constants & Calculator (PHI-based mathematics)
- Observability (structured logging, tracing, metrics)
- Skills Loader (YAML-based skill discovery)
- Quantization (vector compression)
"""

from sovereign_pio.constants import PHI, ALPHA, OMEGA, BETA, GAMMA
from sovereign_pio.calculator import D, Theta, Energy, x_from_D, lucas

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

from .skills_loader import (
    SkillLoader,
    SkillMetadata,
    SkillCategory,
    Permission,
    SkillSchema,
)

from .quantization import (
    Quantizer,
    QuantizedVector,
    ResidualQuantizer,
    ResidualVector,
    similarity_preservation,
    compute_compression_ratio,
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
]
