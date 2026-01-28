"""
Cognitive Extension - Meta-Cortex and Cognitive Systems

Ports from: CLI-main/src/core/meta_cortex.py

Implements:
- MetaCortex: Introspection and self-model
- Thought packet observation
- Reflex improvement proposals
- Alignment tracking
"""

from .cognitive_core import (
    # Core
    MetaCortex, ThoughtPacket, SelfModel,

    # Analysis
    AlignmentTracker, ReflexProposer,

    # Factory
    create_meta_cortex, get_meta_cortex,
)

__all__ = [
    "MetaCortex", "ThoughtPacket", "SelfModel",
    "AlignmentTracker", "ReflexProposer",
    "create_meta_cortex", "get_meta_cortex",
]
