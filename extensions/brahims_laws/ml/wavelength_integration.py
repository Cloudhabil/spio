#!/usr/bin/env python3
"""
12-Wavelength Integration for Brahim Onion Agent

Connects the Active Inference pipeline to BOA for ML-based intent classification.

Architecture:
    Phase 1: Initialization (Wave 1)
    Phase 2: Abstract Processing (Waves 2-5)
    Phase 3: Optimization (Waves 6-9)
    Phase 4: Execution (Waves 10-12)

Author: Elias Oulad Brahim
DOI: 10.5281/zenodo.18356196
"""

from __future__ import annotations

import hashlib
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from pathlib import Path

# Brahim constants (Corrected 2026-01-26)
BRAHIM_SEQUENCE = [27, 42, 60, 75, 97, 117, 139, 154, 172, 187]
BRAHIM_SEQUENCE_ORIGINAL = [27, 42, 60, 75, 97, 121, 136, 154, 172, 187]
SUM_CONSTANT = 214  # Pair sum
CENTER = 107
PHI = (1 + np.sqrt(5)) / 2
DIMENSION = 10

# Target density (1/S or Riemann critical line)
TARGET_DENSITY_BRAHIM = 1.0 / SUM_CONSTANT  # 0.00467
TARGET_DENSITY_RIEMANN = 0.00221888  # Critical line

logger = logging.getLogger("brahims_laws.wavelength")


# =============================================================================
# WAVE 1: ZERO POINT INITIALIZER
# =============================================================================

@dataclass
class SubstrateState:
    """State of the Brahim substrate."""
    weights: np.ndarray
    dimension: int = DIMENSION
    initialized: bool = False
    crystallized: bool = False
    intervention_count: int = 0


class Wave1_ZeroPointInit:
    """
    Wave 1: Initialize substrate with Brahim sequence.

    Maps the 10 Brahim numbers to initial substrate weights.
    """

    def __init__(self):
        self.state: Optional[SubstrateState] = None

    def initialize(self) -> SubstrateState:
        """Initialize substrate from Brahim sequence."""
        # Normalize Brahim sequence to weights
        b_array = np.array(BRAHIM_SEQUENCE, dtype=np.float32)
        weights = b_array / SUM_CONSTANT  # Normalize to sum ~ 1

        self.state = SubstrateState(
            weights=weights,
            dimension=DIMENSION,
            initialized=True
        )

        logger.info("Wave 1: Substrate initialized with Brahim sequence")
        return self.state

    def get_state(self) -> SubstrateState:
        if self.state is None:
            return self.initialize()
        return self.state


# =============================================================================
# WAVE 2: PRIME DIRECTIVE
# =============================================================================

class Wave2_PrimeDirective:
    """
    Wave 2: Define the truth target.

    Target density = 1/214 (Brahim) or 0.00221888 (Riemann critical line)
    """

    def __init__(self, use_riemann: bool = False):
        self.target = TARGET_DENSITY_RIEMANN if use_riemann else TARGET_DENSITY_BRAHIM
        self.threshold = 0.1  # Safe deviation threshold

    def get_target(self) -> float:
        return self.target

    def is_safe_deviation(self, deviation: float) -> bool:
        return abs(deviation) <= self.threshold


# =============================================================================
# WAVE 3: STOCHASTIC INGESTOR
# =============================================================================

class Wave3_StochasticIngest:
    """
    Wave 3: Convert user query to embedding.

    Maps text → 10-dimensional Brahim embedding space.
    """

    def __init__(self):
        self.history: List[np.ndarray] = []

    def embed(self, text: str) -> np.ndarray:
        """
        Embed text into Brahim space.

        Method: Hash text and distribute across 10 dimensions
        weighted by Brahim sequence.
        """
        if not text:
            return np.zeros(DIMENSION, dtype=np.float32)

        # Hash the text
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Convert to 10 values
        embedding = np.zeros(DIMENSION, dtype=np.float32)
        for i in range(DIMENSION):
            # Use pairs of bytes for each dimension
            byte_val = hash_bytes[i * 2] + hash_bytes[i * 2 + 1] * 256
            # Normalize and weight by Brahim number
            embedding[i] = (byte_val / 65535.0) * (BRAHIM_SEQUENCE[i] / SUM_CONSTANT)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        self.history.append(embedding)
        return embedding

    def embed_with_keywords(self, text: str, intent_keywords: Dict[str, List[str]]) -> np.ndarray:
        """
        Embed with keyword weighting for intent classification.
        """
        base_embedding = self.embed(text)
        text_lower = text.lower()

        # Keyword boost vector
        boost = np.zeros(DIMENSION, dtype=np.float32)

        for intent_idx, (intent, keywords) in enumerate(intent_keywords.items()):
            if intent_idx >= DIMENSION:
                break
            for kw in keywords:
                if kw in text_lower:
                    boost[intent_idx] += BRAHIM_SEQUENCE[intent_idx] / SUM_CONSTANT

        # Combine base + boost
        combined = base_embedding + boost * 0.5
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return combined


# =============================================================================
# WAVE 4: DENSITY ARCHITECT
# =============================================================================

class Wave4_DensityArchitect:
    """
    Wave 4: Calculate current density from embedding.
    """

    def __init__(self):
        self.history: List[float] = []

    def calculate_density(self, embedding: np.ndarray, weights: np.ndarray) -> float:
        """
        Calculate density as weighted dot product.

        density = (embedding · weights) / ||embedding|| / ||weights||
        """
        if embedding.size == 0 or weights.size == 0:
            return 0.0

        dot = np.dot(embedding, weights)
        norm_e = np.linalg.norm(embedding)
        norm_w = np.linalg.norm(weights)

        if norm_e == 0 or norm_w == 0:
            return 0.0

        density = dot / (norm_e * norm_w)
        self.history.append(density)

        return float(density)


# =============================================================================
# WAVE 5: SYNAPTIC BRIDGE
# =============================================================================

@dataclass
class PredictionError:
    """Prediction error for Active Inference."""
    current_density: float
    target_density: float
    error_delta: float
    resonance: float
    is_safe: bool
    deviation_magnitude: float


class Wave5_SynapticBridge:
    """
    Wave 5: Compare current density to target.

    The Active Inference comparator - detects prediction errors.
    """

    def __init__(self, directive: Wave2_PrimeDirective):
        self.directive = directive
        self.error_history: List[PredictionError] = []

    def compute_error(self, current_density: float) -> PredictionError:
        """Compute prediction error."""
        target = self.directive.get_target()
        error_delta = current_density - target
        deviation = abs(error_delta)
        is_safe = self.directive.is_safe_deviation(deviation)

        # Resonance = 1 / (1 + deviation)
        resonance = 1.0 / (1.0 + deviation)

        error = PredictionError(
            current_density=current_density,
            target_density=target,
            error_delta=error_delta,
            resonance=resonance,
            is_safe=is_safe,
            deviation_magnitude=deviation
        )

        self.error_history.append(error)
        return error


# =============================================================================
# WAVE 6: GENERATIVE STREAM
# =============================================================================

class Intent(Enum):
    """Intent classification for BOA."""
    PHYSICS = 0
    COSMOLOGY = 1
    YANG_MILLS = 2
    MIRROR = 3
    SEQUENCE = 4
    VERIFY = 5
    HELP = 6
    UNKNOWN = 7


@dataclass
class IntentProposal:
    """Proposed intent classification."""
    intent: Intent
    confidence: float
    scores: Dict[Intent, float]


class Wave6_GenerativeStream:
    """
    Wave 6: Generate intent classification proposal.

    Maps embedding to intent with confidence scores.
    """

    # Intent keywords mapped to Brahim indices
    INTENT_KEYWORDS = {
        Intent.PHYSICS: ["alpha", "fine structure", "weinberg", "muon", "proton", "electron", "mass ratio", "constant"],
        Intent.COSMOLOGY: ["dark matter", "dark energy", "cosmos", "universe", "hubble", "cosmology", "omega"],
        Intent.YANG_MILLS: ["yang mills", "mass gap", "qcd", "qft", "wightman", "lambda qcd", "glueball"],
        Intent.MIRROR: ["mirror", "reflect", "214", "transform", "operator", "symmetry"],
        Intent.SEQUENCE: ["sequence", "brahim numbers", "manifold", "list", "all numbers", "b1", "b10"],
        Intent.VERIFY: ["verify", "check", "axiom", "validate", "proof", "test"],
        Intent.HELP: ["help", "what can you", "capabilities", "how to", "example"],
    }

    def __init__(self):
        self.proposal_history: List[IntentProposal] = []
        # Intent weights (learnable)
        self.intent_weights = np.ones((len(Intent) - 1, DIMENSION), dtype=np.float32)
        self._init_weights()

    def _init_weights(self):
        """Initialize intent weights from Brahim sequence."""
        for i, intent in enumerate(Intent):
            if intent == Intent.UNKNOWN:
                continue
            if i < DIMENSION:
                # Weight by position in Brahim sequence
                self.intent_weights[i] = np.array(BRAHIM_SEQUENCE, dtype=np.float32) / SUM_CONSTANT
                # Emphasize the index corresponding to intent
                self.intent_weights[i, i] *= 2.0

    def propose_intent(self, text: str, embedding: np.ndarray) -> IntentProposal:
        """
        Propose intent classification.
        """
        text_lower = text.lower()
        scores = {}

        # Calculate scores for each intent
        for intent in Intent:
            if intent == Intent.UNKNOWN:
                continue

            score = 0.0
            idx = intent.value

            # Keyword matching
            keywords = self.INTENT_KEYWORDS.get(intent, [])
            keyword_hits = sum(1 for kw in keywords if kw in text_lower)
            keyword_score = keyword_hits * (BRAHIM_SEQUENCE[min(idx, 9)] / SUM_CONSTANT)

            # Embedding similarity with intent weights
            if idx < len(self.intent_weights):
                weight_vec = self.intent_weights[idx]
                embedding_score = np.dot(embedding, weight_vec) / (
                    np.linalg.norm(embedding) * np.linalg.norm(weight_vec) + 1e-9
                )
            else:
                embedding_score = 0.0

            # Combined score
            score = keyword_score * 0.7 + embedding_score * 0.3
            scores[intent] = max(0.0, score)

        # Find best intent
        if scores:
            best_intent = max(scores, key=scores.get)
            best_score = scores[best_intent]
        else:
            best_intent = Intent.UNKNOWN
            best_score = 0.0

        # Normalize confidence
        total_score = sum(scores.values())
        confidence = best_score / total_score if total_score > 0 else 0.0

        proposal = IntentProposal(
            intent=best_intent if confidence > 0.1 else Intent.UNKNOWN,
            confidence=confidence,
            scores=scores
        )

        self.proposal_history.append(proposal)
        return proposal

    def update_weights(self, correction: np.ndarray, intent_idx: int):
        """Update intent weights with correction."""
        if intent_idx < len(self.intent_weights):
            self.intent_weights[intent_idx] += correction[:DIMENSION]


# =============================================================================
# WAVE 7: THETA WAVE GENERATOR
# =============================================================================

class Wave7_ThetaWaveGen:
    """
    Wave 7: Generate correction signals.

    Creates negative feedback waves to correct prediction errors.
    """

    def __init__(self, learning_rate: float = 0.05, dampening: float = 0.95):
        self.learning_rate = learning_rate
        self.dampening = dampening
        self.wave_history: List[np.ndarray] = []

    def generate_correction(
        self,
        error: PredictionError,
        embedding: np.ndarray
    ) -> np.ndarray:
        """
        Generate theta wave correction.

        Correction = -sign(error) × direction × |error| × lr × dampening
        """
        error_delta = error.error_delta

        # Normalize embedding for direction
        direction = embedding / (np.linalg.norm(embedding) + 1e-9)

        # Correction magnitude
        magnitude = abs(error_delta) * self.learning_rate

        # Theta wave
        theta_wave = -np.sign(error_delta) * direction * magnitude * self.dampening

        self.wave_history.append(theta_wave)
        return theta_wave


# =============================================================================
# WAVE 8: HOMEOSTATIC UPDATER
# =============================================================================

class Wave8_HomeostaticUpdate:
    """
    Wave 8: Apply corrections to substrate weights.
    """

    def __init__(self):
        self.update_count = 0

    def apply_correction(
        self,
        state: SubstrateState,
        correction: np.ndarray
    ) -> SubstrateState:
        """Apply correction to substrate weights."""
        # Clip correction to prevent instability
        clipped = np.clip(correction, -0.1, 0.1)

        # Update weights
        state.weights = state.weights + clipped

        # Ensure weights stay positive and normalized
        state.weights = np.maximum(state.weights, 0.001)
        state.weights = state.weights / np.sum(state.weights) * (np.sum(BRAHIM_SEQUENCE) / SUM_CONSTANT)

        state.intervention_count += 1
        self.update_count += 1

        return state


# =============================================================================
# WAVE 9: ENDURANCE LOOP
# =============================================================================

@dataclass
class ConvergenceResult:
    """Result of convergence loop."""
    converged: bool
    iterations: int
    final_resonance: float
    final_intent: Intent
    intent_confidence: float
    error_history: List[float]


class Wave9_EnduranceLoop:
    """
    Wave 9: Iterate until convergence.

    Loops waves 5-8 until resonance exceeds threshold.
    """

    def __init__(
        self,
        max_iterations: int = 5,
        resonance_threshold: float = 0.7
    ):
        self.max_iterations = max_iterations
        self.resonance_threshold = resonance_threshold

    def converge(
        self,
        text: str,
        embedding: np.ndarray,
        state: SubstrateState,
        bridge: Wave5_SynapticBridge,
        stream: Wave6_GenerativeStream,
        theta_gen: Wave7_ThetaWaveGen,
        updater: Wave8_HomeostaticUpdate,
        density_calc: Wave4_DensityArchitect
    ) -> ConvergenceResult:
        """Run convergence loop."""
        error_history = []
        final_intent = Intent.UNKNOWN
        final_confidence = 0.0

        for iteration in range(self.max_iterations):
            # Calculate current density
            density = density_calc.calculate_density(embedding, state.weights)

            # Compute prediction error (Wave 5)
            error = bridge.compute_error(density)
            error_history.append(error.error_delta)

            # Propose intent (Wave 6)
            proposal = stream.propose_intent(text, embedding)
            final_intent = proposal.intent
            final_confidence = proposal.confidence

            # Check convergence
            if error.resonance >= self.resonance_threshold:
                return ConvergenceResult(
                    converged=True,
                    iterations=iteration + 1,
                    final_resonance=error.resonance,
                    final_intent=final_intent,
                    intent_confidence=final_confidence,
                    error_history=error_history
                )

            # Generate correction (Wave 7)
            correction = theta_gen.generate_correction(error, embedding)

            # Apply correction (Wave 8)
            state = updater.apply_correction(state, correction)

            # Update intent weights
            stream.update_weights(correction, final_intent.value)

        # Max iterations reached
        final_density = density_calc.calculate_density(embedding, state.weights)
        final_error = bridge.compute_error(final_density)

        return ConvergenceResult(
            converged=False,
            iterations=self.max_iterations,
            final_resonance=final_error.resonance,
            final_intent=final_intent,
            intent_confidence=final_confidence,
            error_history=error_history
        )


# =============================================================================
# WAVE 10: TRANSPARENCY LOGGER
# =============================================================================

@dataclass
class ResonanceEvent:
    """Logged resonance evolution event."""
    timestamp: float
    query: str
    intent: str
    resonance: float
    converged: bool
    iterations: int


class Wave10_TransparencyLog:
    """
    Wave 10: Log all resonance evolution events.
    """

    def __init__(self):
        self.events: List[ResonanceEvent] = []

    def log_event(
        self,
        query: str,
        result: ConvergenceResult
    ) -> ResonanceEvent:
        """Log a resonance evolution event."""
        import time

        event = ResonanceEvent(
            timestamp=time.time(),
            query=query[:100],  # Truncate
            intent=result.final_intent.name,
            resonance=result.final_resonance,
            converged=result.converged,
            iterations=result.iterations
        )

        self.events.append(event)
        return event

    def get_recent(self, limit: int = 10) -> List[ResonanceEvent]:
        return self.events[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        if not self.events:
            return {"count": 0, "status": "NO_DATA"}

        resonances = [e.resonance for e in self.events]
        converged_count = sum(1 for e in self.events if e.converged)

        return {
            "count": len(self.events),
            "converged_ratio": converged_count / len(self.events),
            "mean_resonance": float(np.mean(resonances)),
            "max_resonance": float(np.max(resonances)),
            "status": "ACTIVE"
        }


# =============================================================================
# WAVE 11: COMPILER CHANNEL (Execute Brahim Calculation)
# =============================================================================

class Wave11_CompilerChannel:
    """
    Wave 11: Execute Brahim calculation based on intent.

    Maps intent → calculation function → result.
    """

    def __init__(self):
        # Import Brahim SDK functions
        from ..agents_sdk import (
            fine_structure_constant,
            weinberg_angle,
            muon_electron_ratio,
            proton_electron_ratio,
            cosmic_fractions,
            yang_mills_mass_gap,
            mirror_operator,
            get_sequence,
            verify_mirror_symmetry,
        )

        self.functions = {
            Intent.PHYSICS: {
                "fine_structure": fine_structure_constant,
                "weinberg_angle": weinberg_angle,
                "muon_electron": muon_electron_ratio,
                "proton_electron": proton_electron_ratio,
            },
            Intent.COSMOLOGY: cosmic_fractions,
            Intent.YANG_MILLS: yang_mills_mass_gap,
            Intent.MIRROR: mirror_operator,
            Intent.SEQUENCE: get_sequence,
            Intent.VERIFY: verify_mirror_symmetry,
        }

    def execute(
        self,
        intent: Intent,
        text: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute calculation based on intent."""
        params = params or {}

        if intent == Intent.PHYSICS:
            # Determine which physics constant
            text_lower = text.lower()
            if "weinberg" in text_lower or "theta" in text_lower:
                func = self.functions[Intent.PHYSICS]["weinberg_angle"]
            elif "muon" in text_lower:
                func = self.functions[Intent.PHYSICS]["muon_electron"]
            elif "proton" in text_lower:
                func = self.functions[Intent.PHYSICS]["proton_electron"]
            else:
                func = self.functions[Intent.PHYSICS]["fine_structure"]
            result = func()
            return result.to_dict() if hasattr(result, 'to_dict') else result

        elif intent == Intent.COSMOLOGY:
            result = self.functions[Intent.COSMOLOGY]()
            return result.to_dict() if hasattr(result, 'to_dict') else result

        elif intent == Intent.YANG_MILLS:
            result = self.functions[Intent.YANG_MILLS]()
            return result.to_dict() if hasattr(result, 'to_dict') else result

        elif intent == Intent.MIRROR:
            # Extract number from text
            import re
            numbers = re.findall(r'\d+', text)
            value = int(numbers[0]) if numbers else CENTER
            return self.functions[Intent.MIRROR](value)

        elif intent == Intent.SEQUENCE:
            return self.functions[Intent.SEQUENCE]()

        elif intent == Intent.VERIFY:
            return self.functions[Intent.VERIFY]()

        else:
            return {
                "error": "Unknown intent",
                "intent": intent.name,
                "help": "Try asking about physics constants, cosmology, or Yang-Mills"
            }


# =============================================================================
# WAVE 12: SUBSTRATE CRYSTALLIZER
# =============================================================================

class Wave12_SubstrateCrystallize:
    """
    Wave 12: Persist learned weights.

    Vajra Seal: Requires 3 interventions before crystallization.
    """

    VAJRA_THRESHOLD = 3

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/substrate")
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def check_vajra(self, state: SubstrateState) -> str:
        """Check if Vajra seal is satisfied."""
        if state.intervention_count >= self.VAJRA_THRESHOLD:
            return "READY"
        return f"PENDING ({state.intervention_count}/{self.VAJRA_THRESHOLD})"

    def crystallize(self, state: SubstrateState) -> Dict[str, Any]:
        """Crystallize substrate if Vajra seal satisfied."""
        if state.intervention_count < self.VAJRA_THRESHOLD:
            return {
                "status": "VAJRA_PENDING",
                "interventions": state.intervention_count,
                "required": self.VAJRA_THRESHOLD
            }

        # Save weights
        weights_file = self.storage_path / "substrate_weights.npy"
        np.save(weights_file, state.weights)

        state.crystallized = True

        return {
            "status": "CRYSTALLIZED",
            "path": str(weights_file),
            "weights_shape": state.weights.shape,
            "interventions": state.intervention_count,
            "mean": float(np.mean(state.weights)),
            "std": float(np.std(state.weights))
        }

    def load_crystallized(self) -> Optional[np.ndarray]:
        """Load crystallized weights if they exist."""
        weights_file = self.storage_path / "substrate_weights.npy"
        if weights_file.exists():
            return np.load(weights_file)
        return None


# =============================================================================
# INTEGRATED PIPELINE
# =============================================================================

class WavelengthPipeline:
    """
    Complete 12-Wavelength pipeline for BOA.

    Integrates all 12 waves into a single processing unit.
    """

    def __init__(self, use_riemann_target: bool = False):
        # Phase 1
        self.wave1 = Wave1_ZeroPointInit()

        # Phase 2
        self.wave2 = Wave2_PrimeDirective(use_riemann=use_riemann_target)
        self.wave3 = Wave3_StochasticIngest()
        self.wave4 = Wave4_DensityArchitect()
        self.wave5 = Wave5_SynapticBridge(self.wave2)

        # Phase 3
        self.wave6 = Wave6_GenerativeStream()
        self.wave7 = Wave7_ThetaWaveGen()
        self.wave8 = Wave8_HomeostaticUpdate()
        self.wave9 = Wave9_EnduranceLoop()

        # Phase 4
        self.wave10 = Wave10_TransparencyLog()
        self.wave11 = Wave11_CompilerChannel()
        self.wave12 = Wave12_SubstrateCrystallize()

        # Initialize substrate
        self.state = self.wave1.initialize()

        # Try to load crystallized weights
        loaded = self.wave12.load_crystallized()
        if loaded is not None:
            self.state.weights = loaded
            self.state.crystallized = True
            logger.info("Loaded crystallized substrate weights")

    def process(self, query: str) -> Dict[str, Any]:
        """
        Process a query through all 12 waves.

        Returns complete result with intent, calculation, and metadata.
        """
        # Phase 2: Embed query
        embedding = self.wave3.embed_with_keywords(
            query,
            {intent.name.lower(): kws for intent, kws in self.wave6.INTENT_KEYWORDS.items()}
        )

        # Phase 3: Convergence loop
        convergence = self.wave9.converge(
            text=query,
            embedding=embedding,
            state=self.state,
            bridge=self.wave5,
            stream=self.wave6,
            theta_gen=self.wave7,
            updater=self.wave8,
            density_calc=self.wave4
        )

        # Phase 4: Log event
        self.wave10.log_event(query, convergence)

        # Execute calculation
        calculation = self.wave11.execute(
            intent=convergence.final_intent,
            text=query
        )

        # Check crystallization
        vajra_status = self.wave12.check_vajra(self.state)
        if vajra_status == "READY" and not self.state.crystallized:
            crystal_result = self.wave12.crystallize(self.state)
            logger.info(f"Substrate crystallized: {crystal_result}")

        return {
            "intent": convergence.final_intent.name,
            "confidence": convergence.intent_confidence,
            "resonance": convergence.final_resonance,
            "converged": convergence.converged,
            "iterations": convergence.iterations,
            "calculation": calculation,
            "vajra_status": vajra_status,
            "crystallized": self.state.crystallized
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "substrate": {
                "initialized": self.state.initialized,
                "crystallized": self.state.crystallized,
                "interventions": self.state.intervention_count,
                "weights_mean": float(np.mean(self.state.weights)),
                "weights_std": float(np.std(self.state.weights))
            },
            "transparency": self.wave10.get_stats(),
            "target_density": self.wave2.get_target(),
            "correction_count": len(self.wave7.wave_history),
            "vajra_status": self.wave12.check_vajra(self.state)
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_pipeline: Optional[WavelengthPipeline] = None


def get_pipeline() -> WavelengthPipeline:
    """Get global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = WavelengthPipeline()
    return _pipeline


def process_with_wavelengths(query: str) -> Dict[str, Any]:
    """Process query through 12-wavelength pipeline."""
    return get_pipeline().process(query)


def get_wavelength_stats() -> Dict[str, Any]:
    """Get pipeline statistics."""
    return get_pipeline().get_stats()
