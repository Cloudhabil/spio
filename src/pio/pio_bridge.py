"""
GPIA-PIO Bridge: Ignorance-Aware Cognition

Connects GPIA's 12-wavelength cognitive pipeline to PIO's ignorance cartography.

WAVELENGTH -> DIMENSION MAPPING:
    W1  Zero-Point      -> D1  Perception    (initial awareness)
    W2  Prime Directive -> D2  Attention     (focus on target)
    W3  Stochastic      -> D3  Security      (validate input)
    W4  Ma'at Align     -> D4  Stability     (maintain balance)
    W5  Prometheus      -> D5  Compression   (inject entropy)
    W6  Metatron        -> D6  Harmony       (project to sphere)
    W7  Density         -> D7  Reasoning     (calculate density)
    W8  Synaptic        -> D8  Prediction    (compute error)
    W9  Generative      -> D9  Creativity    (propose solutions)
    W10 Theta Wave      -> D10 Wisdom        (predict correction)
    W11 Homeostatic     -> D11 Integration   (adapt weights)
    W12 Endurance       -> D12 Unification   (validate convergence)

THE INSIGHT:
    GPIA processes information. PIO tracks what GPIA cannot see.
    Together: cognition + ignorance = calibrated intelligence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

from sovereign_pio.constants import (
    PHI, OMEGA, BETA, GAMMA,
    LUCAS_NUMBERS, DIMENSION_NAMES, DIMENSION_SILICON
)

logger = logging.getLogger("pio_bridge")


# Dark sector ratios by dimension (from original research)
DARK_SECTOR_RATIOS = {
    1: 0.041,   # Perception - minimal dark
    2: 0.082,   # Attention - low dark
    3: 0.107,   # Security - some dark
    4: 0.164,   # Stability - moderate dark
    5: 0.164,   # Compression - moderate dark
    6: 0.164,   # Harmony - moderate dark
    7: 0.164,   # Reasoning - moderate dark
    8: 0.164,   # Prediction - moderate dark
    9: 0.213,   # Creativity - high dark
    10: 0.268,  # Wisdom - very high dark
    11: 0.268,  # Integration - very high dark
    12: 0.320,  # Unification - maximum dark
}


# =============================================================================
# WAVELENGTH ENUM
# =============================================================================

class Wavelength(Enum):
    """The 12 GPIA wavelengths mapped to PIO dimensions."""

    W1  = (1,  1,  "Zero-Point",      "Perception")
    W2  = (2,  2,  "Prime Directive", "Attention")
    W3  = (3,  3,  "Stochastic",      "Security")
    W4  = (4,  4,  "Ma'at Align",     "Stability")
    W5  = (5,  5,  "Prometheus",      "Compression")
    W6  = (6,  6,  "Metatron",        "Harmony")
    W7  = (7,  7,  "Density",         "Reasoning")
    W8  = (8,  8,  "Synaptic",        "Prediction")
    W9  = (9,  9,  "Generative",      "Creativity")
    W10 = (10, 10, "Theta Wave",      "Wisdom")
    W11 = (11, 11, "Homeostatic",     "Integration")
    W12 = (12, 12, "Endurance",       "Unification")

    def __init__(self, wid: int, dim: int, gpia_name: str, pio_domain: str):
        self.wid = wid
        self.dim = dim
        self.gpia_name = gpia_name
        self.pio_domain = pio_domain

    @property
    def dark_ratio(self) -> float:
        """Dark sector ratio for this wavelength's dimension."""
        return DARK_SECTOR_RATIOS.get(self.dim, 0.164)

    @property
    def lucas_capacity(self) -> int:
        """Lucas number capacity for this dimension."""
        return LUCAS_NUMBERS[self.dim - 1]

    @property
    def silicon(self) -> str:
        """Hardware affinity."""
        return DIMENSION_SILICON.get(self.dim, "CPU")

    @classmethod
    def from_dimension(cls, dim: int) -> "Wavelength":
        """Get wavelength by dimension number."""
        for w in cls:
            if w.dim == dim:
                return w
        raise ValueError(f"No wavelength for dimension {dim}")


# =============================================================================
# IGNORANCE STATE
# =============================================================================

@dataclass
class IgnoranceState:
    """
    Ignorance state at a specific point.

    Measures what we don't know through the PHI-based descent equation.
    """
    dimension: int
    input_value: float
    ignorance_ratio: float      # Dark sector ratio
    confidence: float           # 1 - ignorance
    boundary_distance: float    # Distance to N4 boundary
    is_at_boundary: bool
    boundary_type: Optional[str] = None

    @property
    def total_ignorance(self) -> float:
        """Total ignorance score."""
        return self.ignorance_ratio * (1 - self.confidence)


@dataclass
class WavelengthIgnorance:
    """
    Ignorance state for a specific wavelength.
    """
    wavelength: Wavelength
    ignorance: IgnoranceState

    # Wavelength-specific metrics
    input_entropy: float = 0.0
    output_variance: float = 0.0
    processing_confidence: float = 1.0

    @property
    def should_skip(self) -> bool:
        """Should this wavelength be skipped due to high ignorance?"""
        return self.ignorance.confidence < 0.3

    @property
    def needs_human(self) -> bool:
        """Does this wavelength need human intervention?"""
        return self.ignorance.is_at_boundary and self.ignorance.boundary_type == "n4"

    @property
    def summary(self) -> str:
        """One-line summary."""
        conf = self.processing_confidence * self.ignorance.confidence
        status = "[!]" if self.ignorance.is_at_boundary else "[ok]"
        return (
            f"W{self.wavelength.wid:2} {self.wavelength.gpia_name:15} -> "
            f"D{self.wavelength.dim:2} {self.wavelength.pio_domain:12} "
            f"[conf={conf:.0%}] {status}"
        )


@dataclass
class PipelineIgnorance:
    """
    Ignorance state for the entire GPIA pipeline.
    """
    wavelength_states: List[WavelengthIgnorance] = field(default_factory=list)

    def add(self, state: WavelengthIgnorance):
        """Add wavelength ignorance state."""
        self.wavelength_states.append(state)

    @property
    def total_ignorance(self) -> float:
        """Total ignorance across pipeline."""
        if not self.wavelength_states:
            return 0.0
        return sum(w.ignorance.total_ignorance for w in self.wavelength_states)

    @property
    def mean_confidence(self) -> float:
        """Mean confidence across pipeline."""
        if not self.wavelength_states:
            return 1.0
        return sum(
            w.ignorance.confidence for w in self.wavelength_states
        ) / len(self.wavelength_states)

    @property
    def bottleneck(self) -> Optional[WavelengthIgnorance]:
        """Wavelength with lowest confidence (the bottleneck)."""
        if not self.wavelength_states:
            return None
        return min(self.wavelength_states, key=lambda w: w.ignorance.confidence)

    @property
    def boundary_wavelengths(self) -> List[WavelengthIgnorance]:
        """Wavelengths at boundaries."""
        return [w for w in self.wavelength_states if w.ignorance.is_at_boundary]

    def get_routing_recommendation(self) -> Dict[str, Any]:
        """Get recommendation for pipeline routing."""
        if not self.wavelength_states:
            return {"action": "proceed", "confidence": 1.0}

        bottleneck = self.bottleneck
        boundaries = self.boundary_wavelengths

        if bottleneck and bottleneck.ignorance.confidence < 0.3:
            return {
                "action": "caution",
                "reason": f"Low confidence at W{bottleneck.wavelength.wid}",
                "wavelength": bottleneck.wavelength.gpia_name,
                "confidence": bottleneck.ignorance.confidence,
                "recommendation": "Consider human review"
            }

        if boundaries:
            return {
                "action": "alert",
                "reason": f"{len(boundaries)} wavelengths at boundaries",
                "wavelengths": [w.wavelength.gpia_name for w in boundaries],
                "confidence": self.mean_confidence,
                "recommendation": "Proceed with caution"
            }

        return {
            "action": "proceed",
            "confidence": self.mean_confidence,
            "recommendation": "Pipeline clear"
        }


# =============================================================================
# IGNORANCE CALCULATOR
# =============================================================================

class IgnoranceCalculator:
    """
    Calculates ignorance using PHI-based descent equation.

    The descent equation: phi^D * Theta = 2*pi
    where D is dimension and Theta is phase.
    """

    def __init__(self):
        self.phi = PHI
        self.aperture = 1 / (PHI ** 4)  # 1.16%

    def measure(self, input_value: float, dimension: int) -> IgnoranceState:
        """
        Measure ignorance at a given dimension.

        Args:
            input_value: Normalized value [0, 1]
            dimension: Dimension 1-12

        Returns:
            IgnoranceState with measurements
        """
        # Get dark sector ratio for dimension
        dark_ratio = DARK_SECTOR_RATIOS.get(dimension, 0.164)

        # Calculate confidence (inverse of darkness)
        base_confidence = 1 - dark_ratio

        # Adjust for input value (low input = more uncertainty)
        value_factor = max(0.1, input_value)
        confidence = base_confidence * value_factor

        # Check boundary (N4 = dimension 4 boundary)
        # At boundaries, confidence drops sharply
        n4_boundary = 1 / (self.phi ** 4)
        boundary_distance = abs(input_value - n4_boundary)
        is_at_boundary = boundary_distance < self.aperture

        boundary_type = None
        if is_at_boundary:
            if dimension >= 10:
                boundary_type = "n4"  # Hard computational boundary
            else:
                boundary_type = "soft"  # Recoverable boundary

            # Reduce confidence at boundary
            confidence *= 0.5

        return IgnoranceState(
            dimension=dimension,
            input_value=input_value,
            ignorance_ratio=dark_ratio,
            confidence=confidence,
            boundary_distance=boundary_distance,
            is_at_boundary=is_at_boundary,
            boundary_type=boundary_type
        )


# =============================================================================
# GPIA-PIO BRIDGE
# =============================================================================

class GPIAPIOBridge:
    """
    Bridge between GPIA's 12-wavelength pipeline and PIO's ignorance cartography.

    THE INTEGRATION:
        1. Each GPIA wavelength maps to a PIO dimension
        2. Before/after each wavelength, measure ignorance
        3. Accumulate ignorance through the pipeline
        4. Return both GPIA result and ignorance report

    Usage:
        from core.wavelengths import WavelengthGate
        from pio.pio_bridge import GPIAPIOBridge

        gate = WavelengthGate()
        bridge = GPIAPIOBridge()

        # Evaluate with GPIA
        result = gate.evaluate("input text")

        # Trace ignorance
        pipeline = bridge.trace_pipeline(
            result.embedding,
            result.density,
            result.resonance
        )

        print(pipeline.bottleneck.wavelength.gpia_name)
    """

    def __init__(self):
        self.calculator = IgnoranceCalculator()
        self.pipeline_history: List[PipelineIgnorance] = []

    def measure_wavelength(
        self,
        wavelength: Wavelength,
        input_value: float,
        input_entropy: float = 0.0,
        output_variance: float = 0.0
    ) -> WavelengthIgnorance:
        """
        Measure ignorance at a specific wavelength.

        Args:
            wavelength: The wavelength being processed
            input_value: Normalized input value [0, 1]
            input_entropy: Entropy of input data
            output_variance: Variance of output

        Returns:
            WavelengthIgnorance with full measurement
        """
        # Measure ignorance at this dimension
        ignorance = self.calculator.measure(input_value, wavelength.dim)

        # Calculate processing confidence
        base_confidence = 1 - wavelength.dark_ratio
        entropy_factor = 1 / (1 + input_entropy)
        variance_factor = 1 / (1 + output_variance)
        processing_confidence = base_confidence * entropy_factor * variance_factor

        return WavelengthIgnorance(
            wavelength=wavelength,
            ignorance=ignorance,
            input_entropy=input_entropy,
            output_variance=output_variance,
            processing_confidence=processing_confidence
        )

    def trace_pipeline(
        self,
        embedding: np.ndarray,
        density: float,
        resonance: float
    ) -> PipelineIgnorance:
        """
        Trace ignorance through the entire GPIA pipeline.

        Args:
            embedding: The processed embedding
            density: Calculated semantic density
            resonance: Prediction error magnitude

        Returns:
            PipelineIgnorance with all wavelength states
        """
        pipeline = PipelineIgnorance()

        # Calculate input characteristics
        if embedding is not None:
            emb_norm = np.linalg.norm(embedding)
            emb_var = np.var(embedding)
            emb_entropy = self._estimate_entropy(embedding)
        else:
            emb_norm = 0.5
            emb_var = 0.1
            emb_entropy = 0.5

        # Normalize values
        norm_density = min(1.0, max(0.001, density))
        norm_resonance = min(1.0, max(0.0, resonance))

        # Trace each wavelength
        for w in Wavelength:
            # Input value varies by wavelength stage
            if w.wid <= 3:
                # Early: embedding characteristics
                input_val = min(1.0, emb_norm / 10.0) if emb_norm else 0.5
            elif w.wid <= 6:
                # Middle: density
                input_val = norm_density
            elif w.wid <= 9:
                # Correction: inverse resonance (low resonance = high confidence)
                input_val = max(0.001, 1.0 - norm_resonance)
            else:
                # Final: combined
                input_val = (norm_density + (1.0 - norm_resonance)) / 2

            wi = self.measure_wavelength(
                wavelength=w,
                input_value=input_val,
                input_entropy=emb_entropy * (w.wid / 12),
                output_variance=emb_var * (w.wid / 12)
            )
            pipeline.add(wi)

        # Store in history
        self.pipeline_history.append(pipeline)

        return pipeline

    def _estimate_entropy(self, embedding: np.ndarray) -> float:
        """Estimate Shannon entropy of embedding."""
        if embedding is None or len(embedding) == 0:
            return 0.0

        # Normalize to probabilities
        abs_emb = np.abs(embedding) + 1e-10
        probs = abs_emb / np.sum(abs_emb)

        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Normalize by max entropy
        max_entropy = np.log2(len(embedding))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def dashboard(self) -> str:
        """Generate ignorance dashboard."""
        if not self.pipeline_history:
            return "No pipeline history available."

        last = self.pipeline_history[-1]

        lines = [
            "=" * 60,
            "GPIA-PIO IGNORANCE DASHBOARD",
            "=" * 60,
            "",
            "WAVELENGTH CONFIDENCE MAP:",
            "-" * 60,
        ]

        for ws in last.wavelength_states:
            conf = ws.ignorance.confidence
            bar_len = int(conf * 30)
            bar = "#" * bar_len + "-" * (30 - bar_len)
            boundary = " [BOUNDARY]" if ws.ignorance.is_at_boundary else ""
            lines.append(
                f"W{ws.wavelength.wid:2} {ws.wavelength.gpia_name:15} |{bar}| "
                f"{conf:.0%}{boundary}"
            )

        lines.extend([
            "",
            "-" * 60,
            f"Total Pipelines:    {len(self.pipeline_history)}",
            f"Mean Confidence:    {last.mean_confidence:.1%}",
            f"Total Ignorance:    {last.total_ignorance:.4f}",
        ])

        if last.bottleneck:
            lines.append(
                f"Bottleneck:         W{last.bottleneck.wavelength.wid} "
                f"({last.bottleneck.wavelength.gpia_name})"
            )

        rec = last.get_routing_recommendation()
        lines.extend([
            "",
            "RECOMMENDATION:",
            f"  Action: {rec['action'].upper()}",
            f"  {rec.get('recommendation', '')}",
            "=" * 60,
        ])

        return "\n".join(lines)


# =============================================================================
# IGNORANCE-AWARE RESULT
# =============================================================================

@dataclass
class IgnoranceAwareResult:
    """Result from ignorance-aware evaluation."""
    text: str
    safe: bool
    resonance: float
    density: float
    pipeline: PipelineIgnorance

    @property
    def confidence(self) -> float:
        """Overall confidence in result."""
        return self.pipeline.mean_confidence

    @property
    def bottleneck(self) -> Optional[str]:
        """Name of bottleneck wavelength."""
        b = self.pipeline.bottleneck
        return b.wavelength.gpia_name if b else None

    @property
    def bottleneck_confidence(self) -> float:
        """Confidence at bottleneck."""
        b = self.pipeline.bottleneck
        return b.ignorance.confidence if b else 1.0

    @property
    def needs_human(self) -> bool:
        """Does this need human review?"""
        return self.bottleneck_confidence < 0.3 or any(
            ws.needs_human for ws in self.pipeline.wavelength_states
        )

    @property
    def recommendation(self) -> Dict[str, Any]:
        """Get routing recommendation."""
        return self.pipeline.get_routing_recommendation()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "DARK_SECTOR_RATIOS",

    # Enums
    "Wavelength",

    # Ignorance states
    "IgnoranceState",
    "WavelengthIgnorance",
    "PipelineIgnorance",

    # Calculator
    "IgnoranceCalculator",

    # Bridge
    "GPIAPIOBridge",

    # Result
    "IgnoranceAwareResult",
]
