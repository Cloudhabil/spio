#!/usr/bin/env python3
"""
Brahim Onion Agent with 12-Wavelength ML Integration

Combines:
- 3-Layer Onion Architecture (Interface → Core → Symmetry)
- 12-Wavelength Active Inference Pipeline
- Brahim Mechanics Calculations

Author: Elias Oulad Brahim
DOI: 10.5281/zenodo.18356196
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

# Import wavelength pipeline
from .ml.wavelength_integration import (
    WavelengthPipeline,
    Intent,
    ConvergenceResult,
    get_pipeline,
    process_with_wavelengths,
    BRAHIM_SEQUENCE,
    SUM_CONSTANT,
    CENTER,
    PHI,
)

# Import Brahim SDK
from .agents_sdk import (
    fine_structure_constant,
    weinberg_angle,
    muon_electron_ratio,
    proton_electron_ratio,
    cosmic_fractions,
    yang_mills_mass_gap,
    mirror_operator,
    get_sequence,
    verify_mirror_symmetry,
    BRAHIM_FUNCTIONS,
)


# =============================================================================
# RESPONSE TYPES
# =============================================================================

@dataclass
class BOAResponse:
    """Response from BOA Wavelength Agent."""
    success: bool
    intent: str
    confidence: float
    resonance: float
    converged: bool
    iterations: int
    result: Dict[str, Any]
    layers_traversed: List[str]
    wavelengths_activated: List[int]
    vajra_status: str
    crystallized: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "intent": self.intent,
            "confidence": self.confidence,
            "resonance": self.resonance,
            "converged": self.converged,
            "iterations": self.iterations,
            "result": self.result,
            "layers": self.layers_traversed,
            "wavelengths": self.wavelengths_activated,
            "vajra_status": self.vajra_status,
            "crystallized": self.crystallized
        }


# =============================================================================
# BOA WAVELENGTH AGENT
# =============================================================================

class BOAWavelengthAgent:
    """
    Brahim Onion Agent with 12-Wavelength ML.

    Architecture:
        User Query
            ↓
        [12-Wavelength Pipeline]
        Wave 1-5:  Initialization + Abstract Processing
        Wave 6-9:  Optimization (Intent Classification)
        Wave 10-12: Execution + Persistence
            ↓
        [3-Layer Onion]
        Layer C: Format Response
        Layer A: Execute Calculation (via Wave 11)
        Layer B: Verify Axioms
            ↓
        Response
    """

    def __init__(self, use_riemann_target: bool = False):
        self.pipeline = WavelengthPipeline(use_riemann_target=use_riemann_target)
        self.history: List[BOAResponse] = []

    def process(self, query: str) -> BOAResponse:
        """
        Process query through 12-wavelength + 3-layer architecture.
        """
        if not query or not query.strip():
            return self._help_response()

        # Run through wavelength pipeline
        result = self.pipeline.process(query)

        # Determine which wavelengths were activated
        wavelengths = self._get_active_wavelengths(result)

        # Build response
        response = BOAResponse(
            success=True,
            intent=result["intent"],
            confidence=result["confidence"],
            resonance=result["resonance"],
            converged=result["converged"],
            iterations=result["iterations"],
            result=result["calculation"],
            layers_traversed=["C (Interface)", "A (Core)", "B (Symmetry)", "C (Output)"],
            wavelengths_activated=wavelengths,
            vajra_status=result["vajra_status"],
            crystallized=result["crystallized"]
        )

        self.history.append(response)
        return response

    def _get_active_wavelengths(self, result: Dict) -> List[int]:
        """Determine which wavelengths were activated."""
        waves = [1]  # Always: Zero Point Init

        # Phase 2 always runs
        waves.extend([2, 3, 4, 5])

        # Phase 3: Optimization
        waves.extend([6, 7, 8])
        if result["iterations"] > 1:
            waves.append(9)  # Endurance loop ran

        # Phase 4: Execution
        waves.append(10)  # Transparency log
        waves.append(11)  # Compiler channel

        if result["crystallized"] or result["vajra_status"] == "READY":
            waves.append(12)  # Crystallization

        return sorted(set(waves))

    def _help_response(self) -> BOAResponse:
        """Generate help response."""
        return BOAResponse(
            success=True,
            intent="HELP",
            confidence=1.0,
            resonance=1.0,
            converged=True,
            iterations=0,
            result={
                "message": "Brahim Onion Agent with 12-Wavelength ML",
                "capabilities": [
                    "Physics constants (fine structure, Weinberg angle, mass ratios)",
                    "Cosmology (dark matter 27%, dark energy 68%, Hubble)",
                    "Yang-Mills mass gap (1721 MeV)",
                    "Mirror operator M(x) = 214 - x",
                    "Brahim sequence retrieval",
                    "Axiom verification"
                ],
                "brahim_sequence": BRAHIM_SEQUENCE,
                "constants": {"S": SUM_CONSTANT, "C": CENTER, "phi": PHI}
            },
            layers_traversed=["C (Help)"],
            wavelengths_activated=[1, 6],
            vajra_status="N/A",
            crystallized=False
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        pipeline_stats = self.pipeline.get_stats()
        return {
            "pipeline": pipeline_stats,
            "history_count": len(self.history),
            "success_rate": sum(1 for r in self.history if r.success) / len(self.history) if self.history else 0,
            "mean_resonance": sum(r.resonance for r in self.history) / len(self.history) if self.history else 0,
            "mean_confidence": sum(r.confidence for r in self.history) / len(self.history) if self.history else 0,
        }

    # Convenience methods
    def physics(self, constant: str = "fine_structure") -> BOAResponse:
        """Calculate physics constant."""
        queries = {
            "fine_structure": "What is the fine structure constant?",
            "weinberg_angle": "What is the Weinberg angle?",
            "muon_electron": "What is the muon electron ratio?",
            "proton_electron": "What is the proton electron ratio?"
        }
        return self.process(queries.get(constant, queries["fine_structure"]))

    def cosmology(self) -> BOAResponse:
        """Calculate cosmological parameters."""
        return self.process("What is the dark matter percentage?")

    def yang_mills(self) -> BOAResponse:
        """Calculate Yang-Mills mass gap."""
        return self.process("What is the Yang-Mills mass gap?")

    def mirror(self, value: int) -> BOAResponse:
        """Apply mirror operator."""
        return self.process(f"Apply mirror to {value}")

    def sequence(self) -> BOAResponse:
        """Get Brahim sequence."""
        return self.process("What is the Brahim sequence?")

    def verify(self) -> BOAResponse:
        """Verify axioms."""
        return self.process("Verify the mirror symmetry")


# =============================================================================
# OPENAI-COMPATIBLE TOOLS
# =============================================================================

BOA_WAVELENGTH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "boa_process",
            "description": "Process any physics query through 12-wavelength ML pipeline",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query about physics"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "boa_stats",
            "description": "Get pipeline statistics including resonance, convergence, and crystallization status",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


# =============================================================================
# CLI
# =============================================================================

def main():
    """Interactive CLI for BOA Wavelength Agent."""
    print("=" * 70)
    print("BRAHIM ONION AGENT + 12-WAVELENGTH ML")
    print("=" * 70)

    agent = BOAWavelengthAgent()

    print(f"\nSubstrate initialized: {agent.pipeline.state.initialized}")
    print(f"Target density: {agent.pipeline.wave2.get_target():.6f}")
    print(f"Crystallized: {agent.pipeline.state.crystallized}")
    print("\nType 'help' for capabilities, 'stats' for statistics, 'quit' to exit.\n")

    while True:
        try:
            query = input("You: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not query:
                continue

            if query.lower() == "stats":
                stats = agent.get_stats()
                print(json.dumps(stats, indent=2, default=str))
                continue

            response = agent.process(query)

            print(f"\n[Intent: {response.intent}]")
            print(f"[Confidence: {response.confidence:.2%}]")
            print(f"[Resonance: {response.resonance:.4f}]")
            print(f"[Converged: {response.converged} in {response.iterations} iterations]")
            print(f"[Wavelengths: {response.wavelengths_activated}]")
            print(f"[Vajra: {response.vajra_status}]")
            print(f"\nResult: {json.dumps(response.result, indent=2, default=str)}")
            print()

        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
